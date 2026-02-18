"""
Sharded model wrapper for transformers models.
Adapted from exo's sharded_model.py implementation.

This module provides a wrapper that executes only the layers assigned to a specific shard,
enabling distributed inference across multiple nodes.
"""

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from shard import Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedModel,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

logger = logging.getLogger(__name__)


def get_layer_range_for_shard(shard: Shard, total_layers: int) -> Tuple[int, int]:
    """
    Calculate the actual layer indices for a shard.

    Args:
        shard: Shard specification
        total_layers: Total number of layers in the model

    Returns:
        Tuple of (start_layer, end_layer) indices
    """
    start_layer = max(0, shard.start_layer)
    end_layer = min(total_layers - 1, shard.end_layer)
    return start_layer, end_layer


class TransformersShard:
    """
    Sharded wrapper for transformers models.

    This wrapper takes a full model and only executes the layers assigned to this shard.
    It handles three types of shards:
    - First shard: Has embeddings + assigned layers
    - Middle shard: Has only assigned layers
    - Last shard: Has assigned layers + layer norm + LM head

    The wrapper extracts references to the needed components from the base model
    and implements a custom forward pass that only executes the assigned layers.
    """

    def __init__(self, base_model: PreTrainedModel, shard: Shard):
        """
        Initialize the sharded model wrapper.

        Args:
            base_model: The full pre-trained model
            shard: Shard specification defining which layers to execute
        """
        self.base_model = base_model
        self.shard = shard
        self.config = base_model.config

        # Determine the model architecture type
        self.model_type = self.config.model_type.lower()

        # Get total number of layers from config
        self.total_layers = self._get_total_layers()

        # Get layer range for this shard
        self.start_layer, self.end_layer = get_layer_range_for_shard(
            shard, self.total_layers
        )

        # Extract components based on model architecture
        self._extract_model_components()

        # CRITICAL FIX: Override num_hidden_layers in config to match this shard.
        # Without this, DynamicCache (and other transformers internals) pre-allocate
        # 28 slots based on config.num_hidden_layers, even though we only have 10
        # layers. This causes len(cache.key_cache) == 28 instead of 10, and
        # get_seq_length(0) returns 0 for empty padding slots → broken cache_position.
        import copy
        self.config = copy.deepcopy(base_model.config)
        shard_layer_count = self.end_layer - self.start_layer + 1
        if hasattr(self.config, "num_hidden_layers"):
            self.config.num_hidden_layers = shard_layer_count
        elif hasattr(self.config, "n_layer"):
            self.config.n_layer = shard_layer_count
        elif hasattr(self.config, "num_layers"):
            self.config.num_layers = shard_layer_count
        logger.info(
            f"Patched config: num_hidden_layers={shard_layer_count} (was {self.total_layers})"
        )

        logger.info(
            f"Initialized {self.model_type} shard with layers {self.start_layer}-{self.end_layer}/{self.total_layers}"
        )

        # Free unused layers/components from the base model to save memory
        self._free_unused_layers()

    def _free_unused_layers(self):
        """
        Free memory by removing model components not needed by this shard.

        After _extract_model_components() has extracted references to the layers,
        embeddings, norm, and lm_head that THIS shard needs, the base model still
        holds ALL layers in memory. This method replaces the base model's internal
        layer list with only our shard's layers and removes unused components
        (embeddings, norm, lm_head) so they can be garbage collected.
        """
        import gc

        # Access the inner model
        if hasattr(self.base_model, "model"):
            inner_model = self.base_model.model
        elif hasattr(self.base_model, "transformer"):
            inner_model = self.base_model.transformer
        else:
            logger.warning("Cannot determine inner model structure for memory optimization")
            return

        # Determine which attribute holds the layer list
        if hasattr(inner_model, "layers"):
            layers_attr = "layers"
        elif hasattr(inner_model, "h"):
            layers_attr = "h"
        elif hasattr(inner_model, "decoder") and hasattr(inner_model.decoder, "layers"):
            # For encoder-decoder models, we'd need special handling
            logger.warning("Encoder-decoder model: skipping layer freeing")
            return
        else:
            logger.warning("Cannot find layer list in inner model")
            return

        original_count = len(getattr(inner_model, layers_attr))

        # Replace the full layer list with only our shard's layers
        # self.layers is an nn.ModuleList containing just our subset
        setattr(inner_model, layers_attr, self.layers)

        # Free embedding weights if this is NOT the first shard
        if not self.shard.is_first_layer():
            for attr in ["embed_tokens", "wte", "word_embeddings"]:
                if hasattr(inner_model, attr) and getattr(inner_model, attr) is not None:
                    setattr(inner_model, attr, None)
                    break

        # Free final norm and LM head if this is NOT the last shard
        if not self.shard.is_last_layer():
            for attr in ["norm", "ln_f", "final_layernorm"]:
                if hasattr(inner_model, attr) and getattr(inner_model, attr) is not None:
                    setattr(inner_model, attr, None)
                    break
            if hasattr(self.base_model, "lm_head"):
                self.base_model.lm_head = None

        # Force garbage collection to reclaim freed memory
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        freed_count = original_count - len(self.layers)
        logger.info(
            f"🧹 Memory optimization: freed {freed_count} unused layers. "
            f"Keeping {len(self.layers)} layers ({self.start_layer}-{self.end_layer}) "
            f"out of {original_count} total"
        )

    def _get_total_layers(self) -> int:
        """Get the total number of layers in the model."""
        # Different models store layer count differently
        if hasattr(self.config, "num_hidden_layers"):
            return self.config.num_hidden_layers
        elif hasattr(self.config, "n_layer"):
            return self.config.n_layer
        elif hasattr(self.config, "num_layers"):
            return self.config.num_layers
        else:
            raise ValueError(
                f"Cannot determine layer count for model type {self.model_type}"
            )

    def _extract_model_components(self):
        """
        Extract the necessary components from the base model.

        This handles different model architectures (Llama, Qwen, GPT, etc.)
        by accessing the appropriate attributes.
        """
        # Get the base model (unwrap from CausalLM wrapper)
        if hasattr(self.base_model, "model"):
            # Most models: LlamaForCausalLM.model, Qwen2ForCausalLM.model, etc.
            inner_model = self.base_model.model
        elif hasattr(self.base_model, "transformer"):
            # GPT-2 style: GPT2LMHeadModel.transformer
            inner_model = self.base_model.transformer
        else:
            raise ValueError(f"Cannot find inner model for {type(self.base_model)}")

        # Extract layers
        if hasattr(inner_model, "layers"):
            # Llama, Mistral, Qwen2, etc.
            all_layers = inner_model.layers
        elif hasattr(inner_model, "h"):
            # GPT-2, GPT-Neo
            all_layers = inner_model.h
        elif hasattr(inner_model, "decoder"):
            # Some encoder-decoder models
            all_layers = inner_model.decoder.layers
        else:
            raise ValueError(f"Cannot find layers in {type(inner_model)}")

        # Extract only the layers we need
        self.layers = nn.ModuleList(
            [all_layers[i] for i in range(self.start_layer, self.end_layer + 1)]
        )

        # CRITICAL FIX: Re-index layer_idx on attention modules so the KV cache
        # uses 0-based indices within this shard.
        # Without this, a shard with layers 14-27 would create cache entries at
        # indices 14-27 (with empty padding at 0-13), causing:
        #   1. Cache reports 28 layers instead of 14
        #   2. get_seq_length(0) returns 0 (empty slot) → cache_position always starts at 0
        #   3. Position encoding corruption → gibberish output
        for new_idx, layer in enumerate(self.layers):
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                old_idx = layer.self_attn.layer_idx
                layer.self_attn.layer_idx = new_idx
                if new_idx == 0 or new_idx == len(self.layers) - 1:
                    logger.info(
                        f"  Re-indexed layer {old_idx} → cache index {new_idx}"
                    )

        # Extract rotary embeddings if present (needed for Qwen2, Llama, etc.)
        if hasattr(inner_model, "rotary_emb"):
            self.rotary_emb = inner_model.rotary_emb
        else:
            self.rotary_emb = None

        # Extract embeddings if this is the first shard
        if self.shard.is_first_layer():
            if hasattr(inner_model, "embed_tokens"):
                # Llama, Mistral, Qwen2
                self.embed_tokens = inner_model.embed_tokens
            elif hasattr(inner_model, "wte"):
                # GPT-2
                self.embed_tokens = inner_model.wte
            elif hasattr(inner_model, "word_embeddings"):
                # Some other models
                self.embed_tokens = inner_model.word_embeddings
            else:
                raise ValueError(f"Cannot find embeddings in {type(inner_model)}")
        else:
            self.embed_tokens = None

        # Extract final components if this is the last shard
        if self.shard.is_last_layer():
            # Final layer norm
            if hasattr(inner_model, "norm"):
                # Llama, Mistral, Qwen2
                self.norm = inner_model.norm
            elif hasattr(inner_model, "ln_f"):
                # GPT-2
                self.norm = inner_model.ln_f
            elif hasattr(inner_model, "final_layernorm"):
                # Some other models
                self.norm = inner_model.final_layernorm
            else:
                logger.warning(
                    f"No final norm found for {type(inner_model)}, using None"
                )
                self.norm = None

            # LM head (output projection to vocabulary)
            if hasattr(self.base_model, "lm_head"):
                self.lm_head = self.base_model.lm_head
            elif hasattr(self.base_model, "embed_out"):
                self.lm_head = self.base_model.embed_out
            else:
                raise ValueError(f"Cannot find lm_head in {type(self.base_model)}")
        else:
            self.norm = None
            self.lm_head = None

    def parameters(self):
        """Get parameters from the base model."""
        return self.base_model.parameters()

    def named_parameters(self, *args, **kwargs):
        """Get named parameters from the base model."""
        return self.base_model.named_parameters(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        """Get state dict from the base model."""
        return self.base_model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """Load state dict into the base model."""
        return self.base_model.load_state_dict(*args, **kwargs)

    def eval(self):
        """Set base model to eval mode."""
        self.base_model.eval()
        return self

    def train(self, mode=True):
        """Set base model to train mode."""
        self.base_model.train(mode)
        return self

    @property
    def device(self):
        """Get device of the base model."""
        return next(self.base_model.parameters()).device

    def __call__(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        apply_lm_head: Optional[bool] = None,  # NEW: Override for ring topology
        **kwargs,  # Catch any extra arguments
    ) -> Union[Tuple, CausalLMOutputWithPast, BaseModelOutputWithPast]:
        """
        Forward pass through the assigned shard layers.

        Args:
            input_ids: Token IDs (only for first shard)
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: KV cache from previous tokens
            inputs_embeds: Hidden states (for non-first shards)
            use_cache: Whether to return updated KV cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states
            return_dict: Whether to return a dict or tuple
            cache_position: Cache positions for each token

        Returns:
            Model outputs (logits for last shard, hidden states for others)
        """
        # Set defaults
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Validate inputs
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")

        # Get hidden states based on shard position
        if self.shard.is_first_layer():
            # First shard: convert input_ids to embeddings
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError(
                        "First shard requires either input_ids or inputs_embeds"
                    )
                logger.debug("🔍 EMBEDDING DEBUG:")
                logger.debug(f"   input_ids shape: {input_ids.shape}")
                logger.debug(
                    f"   embed_tokens weight shape: {self.embed_tokens.weight.shape}"
                )
                logger.debug(
                    f"   Expected: (vocab_size={self.config.vocab_size}, hidden_size={self.config.hidden_size})"
                )

                hidden_states = self.embed_tokens(input_ids)

                logger.debug(f"   Actual hidden_states shape: {hidden_states.shape}")
                logger.debug(
                    f"   Expected hidden_states shape: ({input_ids.shape[0]}, {input_ids.shape[1]}, {self.config.hidden_size})"
                )
            else:
                hidden_states = inputs_embeds
        else:
            # Non-first shard: expect hidden states from previous shard
            if inputs_embeds is not None:
                logger.debug("🔍 WORKER SHARD DEBUG:")
                logger.debug(f"   Received inputs_embeds shape: {inputs_embeds.shape}")
                logger.debug(
                    f"   Expected shape: (batch_size, seq_len, {self.config.hidden_size})"
                )
                logger.debug(
                    f"   This shard handles layers: {self.start_layer} to {self.end_layer}"
                )

                hidden_states = inputs_embeds
            elif input_ids is not None:
                raise ValueError(
                    "Non-first shard should receive inputs_embeds, not input_ids"
                )
            else:
                raise ValueError("Non-first shard must receive inputs_embeds")

        # Initialize past_key_values if needed
        # Note: some models (Qwen3) expect a single Cache object shared across layers
        # while others return per-layer tuples. Support both styles.
        shared_cache_object = False

        # CRITICAL FIX: For modern models (Qwen3), when use_cache=True and past_key_values=None,
        # we need to create a DynamicCache object for layers to update in-place.
        # Without this, layers won't return cache even with use_cache=True!
        if past_key_values is None and use_cache:
            from transformers import DynamicCache

            past_key_values = DynamicCache()
            shared_cache_object = True
            logger.debug("   🆕 Created new DynamicCache for use_cache=True")
        elif past_key_values is None:
            # use_cache=False, keep as None
            past_key_values = None

        # If we received a non-list/tuple (e.g., a Cache object), treat it as a shared cache
        if (
            not isinstance(past_key_values, (list, tuple))
            and past_key_values is not None
        ):
            shared_cache_object = True

        # If we have a per-layer list/tuple, ensure its length matches our layers
        if isinstance(past_key_values, (list, tuple)) and len(past_key_values) != len(
            self.layers
        ):
            # Adjust cache to match our layer count
            if len(past_key_values) > len(self.layers):
                # Take only the caches we need (from our layer range)
                past_key_values = past_key_values[self.start_layer : self.end_layer + 1]
            else:
                # Pad with None if we don't have enough
                past_key_values = list(past_key_values) + [None] * (
                    len(self.layers) - len(past_key_values)
                )

        # Build an iterable of per-layer past_key_values to zip with layers.
        # For shared cache objects (like Qwen3.Cache), use the same object for every layer.
        if shared_cache_object:
            per_layer_past = [past_key_values] * len(self.layers)
        else:
            # If past_key_values was None, create a per-layer list of None
            per_layer_past = (
                list(past_key_values)
                if isinstance(past_key_values, (list, tuple))
                else [None] * len(self.layers)
            )

        # Compute rotary position embeddings if model has rotary_emb
        # This is required for Qwen3, Qwen2, Llama, etc.
        position_embeddings = None
        if self.rotary_emb is not None:
            batch_size, seq_length = hidden_states.shape[:2]

            # CRITICAL FIX: Use cache_position for RoPE computation, NOT position_ids
            # Modern transformers (4.36+) use cache_position as the authoritative position tracker
            # and compute position_ids internally when needed. Passing both causes conflicts!
            rope_position_ids = None
            if cache_position is not None:
                # cache_position is already 1D [seq_len], need to unsqueeze to [1, seq_len]
                rope_position_ids = (
                    cache_position.unsqueeze(0)
                    if cache_position.dim() == 1
                    else cache_position
                )
                logger.debug(
                    f"✅ Using cache_position for RoPE: {cache_position.tolist()}"
                )
            elif position_ids is not None:
                # Fallback to position_ids if cache_position not available (legacy mode)
                rope_position_ids = position_ids
                logger.debug(
                    f"⚠️  Fallback: using position_ids for RoPE: {position_ids.tolist()}"
                )
            else:
                # Last resort: create position_ids from sequence length
                device = hidden_states.device
                rope_position_ids = torch.arange(
                    seq_length, dtype=torch.long, device=device
                ).unsqueeze(0)
                logger.debug(
                    f"⚠️  Created position_ids for RoPE (no cache_position or position_ids): {rope_position_ids.tolist()}"
                )

            # Compute rotary embeddings using the appropriate position tensor
            position_embeddings = self.rotary_emb(hidden_states, rope_position_ids)
            logger.debug(
                f"✅ Computed position_embeddings for positions: {rope_position_ids.tolist()}"
            )

        # Process through our shard's layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = []

        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, per_layer_past)
        ):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Prepare layer inputs
            # CRITICAL: Don't pass position_ids when using cache_position!
            # Modern transformers compute position_ids internally from cache_position.
            # Passing both causes position mismatch and gibberish output.
            layer_kwargs = {
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
                # CRITICAL: Qwen3 layer signature expects `past_key_value` (singular)
                # not `past_key_values` (plural) - see layer forward signature
                "past_key_value": past_key_value,
                "output_attentions": output_attentions,
                "use_cache": use_cache,
            }

            # Add position_embeddings if we computed them (required for Qwen3)
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings

            # Add cache_position if provided (for newer transformers versions)
            if cache_position is not None:
                layer_kwargs["cache_position"] = cache_position

            # Forward through this layer
            # Different models have different signatures, so we try to be flexible
            if layer_idx == 0:
                logger.debug(f"   Layer {layer_idx} kwargs: {list(layer_kwargs.keys())}")
                logger.debug(
                    f"   🔍 CRITICAL: use_cache value being passed: {layer_kwargs['use_cache']}"
                )
                logger.debug(
                    f"   🔍 CRITICAL: past_key_value type: {type(layer_kwargs['past_key_value'])}"
                )
                logger.debug(
                    f"   🔍 CRITICAL: past_key_value value: {layer_kwargs['past_key_value']}"
                )
                # CRITICAL: Inspect the actual layer forward signature
                import inspect

                sig = inspect.signature(layer.forward)
                logger.debug(f"   🔍 Layer forward signature: {sig}")
                logger.debug(
                    f"   🔍 Layer forward parameters: {list(sig.parameters.keys())}"
                )

            try:
                layer_outputs = layer(**layer_kwargs)
                if layer_idx == 0:
                    logger.debug(
                        f"   Layer {layer_idx} output type: {type(layer_outputs)}"
                    )
                    if isinstance(layer_outputs, tuple):
                        logger.debug(
                            f"   Layer {layer_idx} tuple length: {len(layer_outputs)}"
                        )
                        for i, item in enumerate(layer_outputs):
                            logger.debug(
                                f"   Layer {layer_idx} output[{i}] type: {type(item)}, shape: {item.shape if hasattr(item, 'shape') else 'N/A'}"
                            )
            except TypeError as e:
                # Try without cache_position if it fails
                if layer_idx == 0:
                    logger.debug(f"   Layer {layer_idx}: TypeError on first call: {e}")
                    logger.debug("   Trying without cache_position...")
                layer_kwargs.pop("cache_position", None)
                try:
                    layer_outputs = layer(**layer_kwargs)
                except TypeError as e2:
                    if layer_idx == 0:
                        logger.debug(
                            f"   Layer {layer_idx}: TypeError on second call: {e2}"
                        )
                        logger.debug(
                            "   Layer signature might not support use_cache or other parameters"
                        )
                        logger.debug("   Trying with minimal kwargs...")
                    # Last resort: try with minimal parameters
                    layer_outputs = layer(hidden_states, attention_mask=attention_mask)

            # Extract outputs
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]

                # CRITICAL FIX: Modern transformers (4.36+) don't return cache in layer outputs!
                # Instead, layers update the shared Cache object in-place via cache.update()
                # We should NOT try to extract cache from layer_outputs - it won't be there.
                # The cache object (past_key_value) is being updated in-place during layer forward.

                # Old behavior (kept for backward compatibility with older models):
                # Some older models might still return cache in tuple format
                if use_cache and not shared_cache_object:
                    # Only try to extract if we're NOT using a shared cache object
                    cache_idx = 2 if output_attentions else 1
                    if len(layer_outputs) > cache_idx:
                        next_decoder_cache.append(layer_outputs[cache_idx])
                        if layer_idx == 0:  # Log first layer only
                            cache_shape = (
                                layer_outputs[cache_idx][0].shape
                                if isinstance(layer_outputs[cache_idx], tuple)
                                else "not-tuple"
                            )
                            logger.debug(
                                f"   Layer {layer_idx}: extracted cache at index {cache_idx}, shape: {cache_shape}"
                            )
                    else:
                        next_decoder_cache.append(None)
                        if layer_idx == 0:  # Log first layer only
                            logger.debug(
                                f"   Layer {layer_idx}: Layer returned tuple length {len(layer_outputs)} (modern transformers update cache in-place)"
                            )

                if output_attentions and len(layer_outputs) > 1:
                    all_self_attns += (layer_outputs[1],)
            else:
                # Some models return objects instead of tuples
                hidden_states = (
                    layer_outputs.last_hidden_state
                    if hasattr(layer_outputs, "last_hidden_state")
                    else layer_outputs[0]
                )

                # CRITICAL FIX: Same as above - modern transformers update cache in-place
                # Only try to extract cache if NOT using shared cache object
                if use_cache and not shared_cache_object:
                    if hasattr(layer_outputs, "past_key_value"):
                        next_decoder_cache.append(layer_outputs.past_key_value)
                        if layer_idx == 0:  # Log first layer only
                            logger.debug(
                                f"   Layer {layer_idx}: extracted cache from object.past_key_value"
                            )
                    else:
                        next_decoder_cache.append(None)
                        if layer_idx == 0:  # Log first layer only
                            logger.debug(
                                f"   Layer {layer_idx}: Object output (modern transformers update cache in-place)"
                            )

                if output_attentions and hasattr(layer_outputs, "attentions"):
                    all_self_attns += (layer_outputs.attentions,)

        # Apply final processing if this is the last shard
        # CRITICAL: In ring mode, apply_lm_head overrides the default shard-based decision
        # This prevents intermediate nodes from applying LM head when forwarding tensors
        if apply_lm_head is not None:
            is_last = apply_lm_head
            logger.debug(f"🔄 Ring mode: apply_lm_head explicitly set to {apply_lm_head}")
        else:
            is_last = self.shard.is_last_layer()

        # Debug logging
        logger.debug(
            f"🔍 Shard check: start={self.shard.start_layer}, end={self.shard.end_layer}, "
            f"n_layers={self.shard.n_layers}, is_last={is_last}"
        )
        logger.debug(
            f"   Hidden states shape before final processing: {hidden_states.shape}"
        )

        if is_last:
            logger.debug("   ✅ IS LAST SHARD - Applying LM head")
            # Apply final norm if available
            if self.norm is not None:
                hidden_states = self.norm(hidden_states)
                logger.debug(f"   After norm: {hidden_states.shape}")

            # Apply lm_head to get logits
            if self.lm_head is not None:
                logger.debug(
                    f"   Applying LM head: hidden_states shape {hidden_states.shape}"
                )
                logits = self.lm_head(hidden_states)
                # Ensure logits are float32 for numerical stability
                logits = logits.float()
                logger.debug(f"   LM head output shape: {logits.shape}")
            else:
                logits = hidden_states
        else:
            # For non-last shards, output is hidden states
            logger.debug(
                f"   ❌ NOT LAST SHARD - Returning hidden states with shape {hidden_states.shape}"
            )
            logits = hidden_states

        # Add final hidden states
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Prepare cache output
        # CRITICAL FIX: For modern transformers with shared Cache objects,
        # return the cache object that was passed in (it was updated in-place)
        if use_cache:
            if shared_cache_object:
                # Modern transformers (4.36+): Cache object was updated in-place by layers
                # Return the SAME object we passed in, now containing updated key/values
                next_cache = past_key_values
            elif next_decoder_cache:
                # Old-style tuple cache (for backward compatibility)
                next_cache = tuple(next_decoder_cache)
            else:
                # use_cache=True but no cache was created/extracted
                next_cache = None
        else:
            next_cache = None

        # Debug: log cache details
        if next_cache is not None:
            if shared_cache_object:
                # For Cache objects, get sequence length from the cache itself
                try:
                    cache_seq_len = (
                        next_cache.get_seq_length(0)
                        if hasattr(next_cache, "get_seq_length")
                        else "unknown"
                    )
                    cache_num_layers = (
                        len(next_cache) if hasattr(next_cache, "__len__") else "unknown"
                    )
                    logger.debug(
                        f"   📦 Cache object returned: {cache_num_layers} layers, seq_len={cache_seq_len}"
                    )
                except Exception:
                    logger.debug(f"   📦 Cache object returned: {type(next_cache)}")
            elif len(next_cache) > 0:
                # Old-style tuple cache
                if next_cache[0] is not None:
                    cache_shape = (
                        next_cache[0][0].shape
                        if isinstance(next_cache[0], tuple)
                        else "unknown"
                    )
                    logger.debug(
                        f"   📦 Cache tuple created: {len(next_cache)} layers, first layer cache shape: {cache_shape}"
                    )
                else:
                    logger.debug(
                        f"   📦 Cache tuple created: {len(next_cache)} layers, but first layer cache is None"
                    )
        else:
            logger.debug(
                f"   📦 No cache returned (use_cache={use_cache}, shared_cache={shared_cache_object})"
            )

        # Return in requested format
        if not return_dict:
            return tuple(
                v
                for v in [logits, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )

        # Return appropriate output type
        if self.shard.is_last_layer():
            return CausalLMOutputWithPast(
                logits=logits,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )
        else:
            return BaseModelOutputWithPast(
                last_hidden_state=logits,  # Actually hidden states for non-last shards
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )


def load_sharded_model(
    model_path: str,
    shard: Shard,
    cache_dir: Optional[str] = None,
    device_map: Union[str, dict] = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    **kwargs,
) -> TransformersShard:
    """
    Load a model and wrap it in a shard.

    This loads the full model (with appropriate device mapping) and then
    wraps it to execute only the assigned layers.

    Args:
        model_path: HuggingFace model ID or path
        shard: Shard specification
        cache_dir: Directory to cache downloaded models
        device_map: Device mapping strategy
        torch_dtype: Torch dtype for model weights
        **kwargs: Additional arguments for model loading

    Returns:
        TransformersShard wrapper around the loaded model
    """
    logger.debug(f"Loading model {model_path} for shard {shard}")

    # Load config
    config = AutoConfig.from_pretrained(
        model_path, cache_dir=cache_dir, trust_remote_code=True
    )

    # Determine dtype if not specified
    if torch_dtype is None:
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

    # Load the full model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        cache_dir=cache_dir,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        **kwargs,
    )

    # Wrap in shard
    sharded_model = TransformersShard(base_model, shard)

    # Set to eval mode by default
    sharded_model.eval()

    logger.debug(f"Successfully loaded and sharded model {model_path}")
    return sharded_model
