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

        logger.info(
            f"Initialized {self.model_type} shard with layers {self.start_layer}-{self.end_layer}/{self.total_layers}"
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
                logger.info(f"🔍 EMBEDDING DEBUG:")
                logger.info(f"   input_ids shape: {input_ids.shape}")
                logger.info(f"   embed_tokens weight shape: {self.embed_tokens.weight.shape}")
                logger.info(f"   Expected: (vocab_size={self.config.vocab_size}, hidden_size={self.config.hidden_size})")
                
                hidden_states = self.embed_tokens(input_ids)
                
                logger.info(f"   Actual hidden_states shape: {hidden_states.shape}")
                logger.info(f"   Expected hidden_states shape: ({input_ids.shape[0]}, {input_ids.shape[1]}, {self.config.hidden_size})")
            else:
                hidden_states = inputs_embeds
        else:
            # Non-first shard: expect hidden states from previous shard
            if inputs_embeds is not None:
                logger.info(f"🔍 WORKER SHARD DEBUG:")
                logger.info(f"   Received inputs_embeds shape: {inputs_embeds.shape}")
                logger.info(f"   Expected shape: (batch_size, seq_len, {self.config.hidden_size})")
                logger.info(f"   This shard handles layers: {self.start_layer} to {self.end_layer}")
                
                hidden_states = inputs_embeds
            elif input_ids is not None:
                raise ValueError(
                    "Non-first shard should receive inputs_embeds, not input_ids"
                )
            else:
                raise ValueError("Non-first shard must receive inputs_embeds")

        # Initialize past_key_values if needed
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        # Ensure we have the right number of cache entries
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

        # Compute position embeddings if we have rotary embeddings
        # This is needed for Qwen2, Llama, and similar models
        position_embeddings = None

        if self.rotary_emb is not None:
            # Get sequence length
            batch_size, seq_length = hidden_states.shape[:2]

            # Create position_ids if not provided
            if position_ids is None:
                device = hidden_states.device
                position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0)
                logger.debug(f"Shard wrapper created new position_ids: {position_ids}")
            else:
                logger.debug(f"Shard wrapper received position_ids: {position_ids}")

            # Compute rotary embeddings
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Process through our shard's layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = []

        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Prepare layer inputs
            layer_kwargs = {
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_value": past_key_value,
                "output_attentions": output_attentions,
                "use_cache": use_cache,
            }

            # Add position_embeddings if we computed them
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings

            # Add cache_position if provided (for newer transformers versions)
            if cache_position is not None:
                layer_kwargs["cache_position"] = cache_position

            # Forward through this layer
            # Different models have different signatures, so we try to be flexible
            try:
                layer_outputs = layer(**layer_kwargs)
            except TypeError:
                # Try without cache_position if it fails
                layer_kwargs.pop("cache_position", None)
                layer_outputs = layer(**layer_kwargs)

            # Extract outputs
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]

                if use_cache:
                    # Cache is usually at index 1 (or 2 if attentions are output)
                    cache_idx = 2 if output_attentions else 1
                    if len(layer_outputs) > cache_idx:
                        next_decoder_cache.append(layer_outputs[cache_idx])
                    else:
                        next_decoder_cache.append(None)

                if output_attentions and len(layer_outputs) > 1:
                    all_self_attns += (layer_outputs[1],)
            else:
                # Some models return objects instead of tuples
                hidden_states = (
                    layer_outputs.last_hidden_state
                    if hasattr(layer_outputs, "last_hidden_state")
                    else layer_outputs[0]
                )

                if use_cache and hasattr(layer_outputs, "past_key_value"):
                    next_decoder_cache.append(layer_outputs.past_key_value)

                if output_attentions and hasattr(layer_outputs, "attentions"):
                    all_self_attns += (layer_outputs.attentions,)

        # Apply final processing if this is the last shard
        is_last = self.shard.is_last_layer()

        # Debug logging
        logger.info(
            f"🔍 Shard check: start={self.shard.start_layer}, end={self.shard.end_layer}, "
            f"n_layers={self.shard.n_layers}, is_last={is_last}"
        )
        logger.info(f"   Hidden states shape before final processing: {hidden_states.shape}")

        if is_last:
            logger.info(f"   ✅ IS LAST SHARD - Applying LM head")
            # Apply final norm if available
            if self.norm is not None:
                hidden_states = self.norm(hidden_states)
                logger.info(f"   After norm: {hidden_states.shape}")

            # Apply lm_head to get logits
            if self.lm_head is not None:
                logger.info(
                    f"   Applying LM head: hidden_states shape {hidden_states.shape}"
                )
                logits = self.lm_head(hidden_states)
                # Ensure logits are float32 for numerical stability
                logits = logits.float()
                logger.info(f"   LM head output shape: {logits.shape}")
            else:
                logits = hidden_states
        else:
            # For non-last shards, output is hidden states
            logger.info(
                f"   ❌ NOT LAST SHARD - Returning hidden states with shape {hidden_states.shape}"
            )
            logits = hidden_states

        # Add final hidden states
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Prepare cache output
        next_cache = (
            tuple(next_decoder_cache) if use_cache and next_decoder_cache else None
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
    logger.info(f"Loading model {model_path} for shard {shard}")

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

    logger.info(f"Successfully loaded and sharded model {model_path}")
    return sharded_model
