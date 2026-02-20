"""
Transformers-based sharded inference engine for HyperCluster.
Adapted from exo's TransformersDynamicShardInferenceEngine.
"""

import asyncio
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from inference_engine import InferenceEngine
from shard import Shard

logger = logging.getLogger(__name__)


def _get_cache_seq_length(cache_state) -> int:
    """
    Get sequence length from cache state.

    Supports both DynamicCache objects and tuple-based caches.
    Handles sharded models where early cache slots may be empty.

    CRITICAL: In transformers 5.2.0+, DynamicCache uses `.layers` (list of
    DynamicLayer objects) instead of `.key_cache`/`.value_cache`.  When a shard
    only owns layers N..M, the cache still uses the *global* layer indices, so
    layers 0..(N-1) exist in the cache but are **empty**.  The default
    `cache.get_seq_length()` checks layer 0 and returns 0, which breaks
    cache_position computation for non-first shards.

    Args:
        cache_state: Either a DynamicCache object or tuple of per-layer caches

    Returns:
        Sequence length of cached tokens (0 if no cache)
    """
    if cache_state is None:
        return 0

    # Check if it's a DynamicCache object (or similar Cache subclass)
    if hasattr(cache_state, "get_seq_length"):
        # Try default (layer 0) first
        try:
            seq_len = cache_state.get_seq_length()
            if seq_len > 0:
                return seq_len
        except Exception:
            pass

        # Layer 0 is empty — this happens with sharded models where this
        # node only owns layers N..M but the cache uses global layer indices.
        # Scan ALL layers to find one that has data.
        num_layers = len(cache_state) if hasattr(cache_state, "__len__") else 0
        for i in range(num_layers):
            try:
                seq_len = cache_state.get_seq_length(i)
                if seq_len > 0:
                    return seq_len
            except Exception:
                continue

        # Fallback for older transformers that still have key_cache attribute
        if hasattr(cache_state, "key_cache"):
            for i, key_tensor in enumerate(cache_state.key_cache):
                if key_tensor is not None and key_tensor.dim() >= 3:
                    return key_tensor.shape[2]

        return 0

    # Check if it has key_cache attribute (older DynamicCache)
    if hasattr(cache_state, "key_cache") and len(cache_state.key_cache) > 0:
        for key_tensor in cache_state.key_cache:
            if key_tensor is not None and key_tensor.dim() >= 3:
                return key_tensor.shape[2]
        return 0

    # Fallback: tuple/list of per-layer caches
    if isinstance(cache_state, (list, tuple)) and len(cache_state) > 0:
        if cache_state[0] is not None:
            if isinstance(cache_state[0], (list, tuple)) and len(cache_state[0]) > 0:
                return cache_state[0][0].shape[2]

    return 0


class TransformersShardedInferenceEngine(InferenceEngine):
    """
    Transformers-based inference engine with native shard support.

    This engine:
    - Loads only the layers assigned to its shard
    - Handles first shard (embeddings), middle shards, and last shard (LM head)
    - Manages KV cache per request for efficient generation
    - Uses thread pools for async model execution
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__()
        self.cache_dir = cache_dir or "./model_cache"
        self.model = None
        self.tokenizer = None
        self.config = None

        # Cache management - stores KV cache per request
        self.caches = OrderedDict()
        self.max_caches = 4

        # Thread pools for async execution
        self._model_thread = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="transformers"
        )
        self._tokenizer_thread = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tokenizer"
        )

        # Async lock for shard loading
        self._shard_lock = asyncio.Lock()

    async def _run_in_model_thread(self, func, *args, **kwargs):
        """Run a function in the model thread pool."""
        return await asyncio.get_running_loop().run_in_executor(
            self._model_thread, func, *args, **kwargs
        )

    async def _run_in_tokenizer_thread(self, func, *args, **kwargs):
        """Run a function in the tokenizer thread pool."""
        return await asyncio.get_running_loop().run_in_executor(
            self._tokenizer_thread, func, *args, **kwargs
        )

    async def _ensure_tokenizer(self, model_id: str):
        """Ensure the tokenizer is loaded for the given model.

        Unlike ``ensure_shard``, this does NOT touch the model weights at all.
        ``encode`` and ``decode`` only need the tokenizer, so calling
        ``ensure_shard`` from those methods is wasteful and actively harmful
        in ring-pipeline mode — it compares shard layer ranges and triggers
        a full model reload even though the tokenizer is already available.
        """
        if self.tokenizer is not None:
            return

        def _load_tokenizer():
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                use_fast=False,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer

        self.tokenizer = await self._run_in_tokenizer_thread(_load_tokenizer)

    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        """Encode text prompt to token IDs using chat template."""
        await self._ensure_tokenizer(shard.model_id)

        def _encode():
            # Use chat template for proper formatting with control tokens
            # This ensures the model receives proper start/end tokens and knows when to stop
            messages = [{"role": "user", "content": prompt}]

            # Apply chat template with add_generation_prompt=True
            # This adds the proper assistant response prompt tokens
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Now tokenize the formatted prompt
            tokens = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)

            logger.info(
                f"Encoded '{prompt[:50]}...' to {len(tokens)} tokens: {tokens[:10]}..."
            )
            logger.info(f"Formatted prompt: {formatted_prompt[:100]}...")
            return np.array(tokens, dtype=np.int64)

        return await self._run_in_tokenizer_thread(_encode)

    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        """Decode token IDs to text."""
        await self._ensure_tokenizer(shard.model_id)

        def _decode():
            # Handle both single token and arrays
            if isinstance(tokens, np.ndarray):
                if tokens.ndim == 0:
                    tokens_list = [int(tokens)]
                else:
                    tokens_list = tokens.flatten().tolist()
            else:
                tokens_list = [int(tokens)]

            # Validate token IDs are in valid range
            vocab_size = len(self.tokenizer)
            valid_tokens = [t for t in tokens_list if 0 <= t < vocab_size]
            if len(valid_tokens) != len(tokens_list):
                logger.warning(
                    f"Found {len(tokens_list) - len(valid_tokens)} invalid tokens, filtering them out"
                )
                tokens_list = valid_tokens

            if not tokens_list:
                return ""

            # Decode with skip_special_tokens to get clean output
            text = self.tokenizer.decode(tokens_list, skip_special_tokens=True)
            return text

        return await self._run_in_tokenizer_thread(_decode)

    async def sample(
        self, logits: np.ndarray, temp: float = 0.7, top_p: float = 0.9
    ) -> np.ndarray:
        """Sample next token from logits."""

        def _sample():
            # Convert to torch tensor
            logits_tensor = torch.from_numpy(logits).float()

            # Get last token logits
            if logits_tensor.dim() > 2:
                logits_tensor = logits_tensor[:, -1, :]
            elif logits_tensor.dim() == 2:
                logits_tensor = logits_tensor[-1:, :]

            # Apply temperature
            if temp > 0:
                logits_tensor = logits_tensor / temp

                # Apply top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        logits_tensor, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits_tensor[indices_to_remove] = float("-inf")

                # Sample from distribution
                probs = torch.softmax(logits_tensor, dim=-1)

                # DEBUG: Show top-5 predictions
                top_probs, top_indices = torch.topk(probs[0], k=5)
                logger.debug("🎲 Top-5 predictions:")
                for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    logger.debug(
                        f"   {i + 1}. Token {idx.item()}: {prob.item():.4f} ({prob.item() * 100:.2f}%)"
                    )

                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy sampling
                next_token = torch.argmax(logits_tensor, dim=-1, keepdim=True)

            token_id = int(next_token.item())
            logger.debug(f"Sampled token ID: {token_id} (shape: {next_token.shape})")

            return next_token.numpy().astype(
                np.int64
            )  # Changed to int64 to match encode

        return await self._run_in_model_thread(_sample)

    async def infer_tensor(
        self,
        request_id: str,
        shard: Shard,
        input_data: np.ndarray,
        inference_state: Optional[Dict] = None,
        position_ids: Optional[np.ndarray] = None,
        attention_mask: Optional[np.ndarray] = None,
        is_final: bool = False,  # NEW: Whether this completes all layers (for LM head)
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Run inference on input tensor through the assigned shard.

        CRITICAL: Like prima.cpp, we MUST receive position_ids from the coordinator.
        Without position_ids, RoPE (Rotary Position Embeddings) will fail.
        """
        await self.ensure_shard(shard)

        # Get or initialize inference state
        if inference_state is None:
            inference_state = {}

        def _infer():
            # Convert input to tensor
            if isinstance(input_data, np.ndarray):
                if input_data.dtype in [np.int64, np.int32]:
                    input_tensor = torch.from_numpy(input_data).long()
                else:
                    input_tensor = torch.from_numpy(input_data).float()
            else:
                input_tensor = torch.tensor(input_data)

            # Ensure tensor is on the right device
            device = next(self.model.parameters()).device
            input_tensor = input_tensor.to(device)

            # Get cache state
            cache_state = self.caches.get(request_id, None)
            past_length = _get_cache_seq_length(cache_state)

            # 🐛 DEBUG: Detailed cache inspection
            logger.debug("=" * 80)
            logger.debug("🔍 KV CACHE DEBUG")
            logger.debug("=" * 80)
            logger.debug(f"Request ID: {request_id}")
            logger.debug(
                f"My shard: {self.shard.start_layer}-{self.shard.end_layer} ({self.shard.end_layer - self.shard.start_layer + 1} layers)"
            )
            logger.debug(f"Total model layers: {self.shard.n_layers}")

            if cache_state is not None:
                logger.debug(f"✅ Cache EXISTS for request {request_id}")
                logger.debug(f"   Cache type: {type(cache_state).__name__}")

                # Inspect cache structure — works with both old (key_cache)
                # and new (layers) DynamicCache API in transformers 5.2.0+
                if hasattr(cache_state, "get_seq_length"):
                    # Modern Cache object (DynamicCache, StaticCache, etc.)
                    num_cached_layers = len(cache_state) if hasattr(cache_state, "__len__") else 0
                    logger.debug(f"   Number of layers in cache: {num_cached_layers}")
                    logger.debug(f"   Sequence length: {past_length}")

                    # 🚨 CRITICAL CHECK: Does cache have ALL model layers or just my shard's layers?
                    expected_layers = self.shard.end_layer - self.shard.start_layer + 1
                    if (
                        num_cached_layers != expected_layers
                        and num_cached_layers != self.shard.n_layers
                    ):
                        logger.warning("⚠️  CACHE LAYER MISMATCH!")
                        logger.warning(
                            f"   Expected {expected_layers} layers for my shard OR {self.shard.n_layers} for full model"
                        )
                        logger.warning(f"   Got {num_cached_layers} layers in cache")
                        logger.warning(
                            f"   (Note: worker shards use global layer indices, so cache may have "
                            f"{self.shard.n_layers} slots with only {expected_layers} populated)"
                        )
                elif hasattr(cache_state, "__len__"):
                    num_cached_layers = len(cache_state)
                    logger.debug(
                        f"   Number of layers in cache (tuple): {num_cached_layers}"
                    )
                    logger.debug(f"   Sequence length: {past_length}")
                else:
                    logger.debug(f"   Cache object: {cache_state}")

                logger.debug("🔄 Will REUSE cache from previous step")
            else:
                logger.debug(f"❌ NO cache found for request {request_id}")
                logger.debug("🆕 Will CREATE new cache")

            logger.debug("=" * 80)

            # Prepare inputs based on shard position and input shape
            if self.shard.is_first_layer() and input_tensor.dim() <= 2:
                # First shard: expects input_ids (2D: batch_size x seq_len)
                if input_tensor.dim() == 1:
                    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

                batch_size, seq_len = input_tensor.shape

                # Use provided position_ids and attention_mask if available
                # Otherwise create them (for backward compatibility)
                if position_ids is not None:
                    # Convert from numpy to torch
                    position_ids_tensor = (
                        torch.from_numpy(position_ids).long().to(device)
                    )
                    logger.debug(
                        f"Using provided position_ids: {position_ids_tensor.shape}"
                    )
                else:
                    # Create attention mask and position IDs (fallback)
                    # Note: attention_mask should be bool or float, not long

                    # Position IDs need to account for cached positions
                    past_length = _get_cache_seq_length(cache_state)
                    if past_length > 0:
                        position_ids_tensor = (
                            torch.arange(
                                past_length,
                                past_length + seq_len,
                                dtype=torch.long,
                                device=device,
                            )
                            .unsqueeze(0)
                            .expand(batch_size, -1)
                        )
                        logger.debug(
                            f"Created position_ids with cache: past_length={past_length}, position_ids={position_ids_tensor}"
                        )
                    else:
                        # No cache, start from 0
                        position_ids_tensor = (
                            torch.arange(seq_len, dtype=torch.long, device=device)
                            .unsqueeze(0)
                            .expand(batch_size, -1)
                        )
                        logger.debug(
                            f"Created position_ids without cache: position_ids={position_ids_tensor}"
                        )

                if attention_mask is not None:
                    # Convert from numpy to torch
                    attention_mask_tensor = (
                        torch.from_numpy(attention_mask).bool().to(device)
                    )
                    logger.debug(
                        f"Using provided attention_mask: {attention_mask_tensor.shape}"
                    )

                    # CRITICAL: When we have KV cache, attention_mask needs to span the full sequence
                    # including cached tokens. If the provided mask is only for new tokens,
                    # we need to expand it to include cached positions.
                    past_length = _get_cache_seq_length(cache_state)
                    if past_length > 0 and attention_mask_tensor.shape[1] == seq_len:
                        # attention_mask only covers new tokens, expand it
                        # to cover cached tokens too (all ones for past tokens)
                        past_mask = torch.ones(
                            batch_size, past_length, dtype=torch.bool, device=device
                        )
                        attention_mask_tensor = torch.cat(
                            [past_mask, attention_mask_tensor], dim=1
                        )
                        logger.debug(
                            f"📏 Expanded attention_mask for cache: {past_length} cached + {seq_len} new = {attention_mask_tensor.shape[1]} total"
                        )
                        # DEBUG: Check for any False values (masked positions)
                        num_masked = (~attention_mask_tensor).sum().item()
                        if num_masked > 0:
                            logger.warning(
                                f"⚠️  Attention mask has {num_masked} MASKED positions (False values)!"
                            )
                            logger.warning(
                                f"   Attention mask: {attention_mask_tensor}"
                            )
                        else:
                            logger.debug(
                                f"✅ Attention mask: all {attention_mask_tensor.shape[1]} positions UNMASKED (all True)"
                            )
                else:
                    # Create default attention mask (all ones)
                    # If we have cache, include cached positions
                    past_length = _get_cache_seq_length(cache_state)
                    if past_length > 0:
                        total_len = past_length + seq_len
                        attention_mask_tensor = torch.ones(
                            batch_size, total_len, dtype=torch.bool, device=device
                        )
                        logger.debug(
                            f"Created full attention_mask with cache: {total_len} tokens"
                        )
                    else:
                        attention_mask_tensor = torch.ones(
                            batch_size, seq_len, dtype=torch.bool, device=device
                        )

                # CRITICAL: Create cache_position for transformers models
                # This tells the model which indices in the KV cache to update
                # Without this, layers won't return cache!
                past_length = _get_cache_seq_length(cache_state)
                if past_length > 0:
                    # We have existing cache, update the next positions
                    # cache_position: indices into the cache [past_length, past_length+1, ...]
                    cache_position = torch.arange(
                        past_length,
                        past_length + seq_len,
                        dtype=torch.long,
                        device=device,
                    )
                    logger.debug(
                        f"🎯 Created cache_position: {cache_position.tolist()} (cache has {past_length} tokens)"
                    )
                else:
                    # No cache yet, start from position 0
                    cache_position = torch.arange(
                        seq_len, dtype=torch.long, device=device
                    )
                    logger.debug(
                        f"🎯 Created cache_position: {cache_position.tolist()} (new cache)"
                    )

                # CRITICAL FIX: When using cache_position (modern transformers 4.36+),
                # DO NOT pass position_ids - the model computes it internally from cache_position!
                # Passing both can cause position mismatch issues.
                # ALSO: Don't pass attention_mask - let the model create it internally
                # (matching native transformers.generate() behavior)
                inputs = {
                    "input_ids": input_tensor,
                    # attention_mask NOT included - let model compute internally
                    # position_ids NOT included - computed from cache_position internally
                    "past_key_values": cache_state,
                    "use_cache": True,
                    "cache_position": cache_position,
                    "apply_lm_head": is_final,  # Only apply LM head if this completes all layers
                }
                logger.debug(
                    f"🔧 Using cache_position={cache_position.tolist()}, letting model compute position_ids and attention_mask internally"
                )
            else:
                # Middle/last shard: expects inputs_embeds (hidden states)
                if input_tensor.dim() == 2:
                    input_tensor = input_tensor.unsqueeze(
                        0
                    )  # Add batch dimension if needed
                elif input_tensor.dim() == 1:
                    raise ValueError(
                        f"Invalid input shape for non-first shard: {input_tensor.shape}"
                    )

                batch_size, seq_len, hidden_size = input_tensor.shape

                # Validate hidden size if we have previous state
                if (
                    "hidden_size" in inference_state
                    and inference_state["hidden_size"] != hidden_size
                ):
                    raise ValueError(
                        f"Hidden size mismatch: expected {inference_state['hidden_size']}, got {hidden_size}"
                    )

                # CRITICAL FIX: When using cache_position (modern transformers 4.36+),
                # DO NOT pass position_ids or attention_mask — let the model
                # compute them internally from cache_position + past_key_values.
                # This matches the first-shard behavior (see above).
                #
                # Previously we passed the ring's attention_mask here, but during
                # autoregressive generation it has shape (1, 1) — only covering the
                # NEW token.  create_causal_mask then builds a mask that doesn't
                # attend to cached tokens, causing broken attention & garbage output.
                # Omitting it lets create_causal_mask derive the full mask from the
                # cache state, which is correct.

                # Build cache_position first (needed for inputs dict)
                past_length = _get_cache_seq_length(cache_state)
                if past_length > 0:
                    cache_position = torch.arange(
                        past_length,
                        past_length + seq_len,
                        dtype=torch.long,
                        device=device,
                    )
                else:
                    cache_position = torch.arange(
                        seq_len, dtype=torch.long, device=device
                    )

                inputs = {
                    "inputs_embeds": input_tensor,
                    # attention_mask NOT included — let model compute internally
                    # position_ids NOT included — computed from cache_position internally
                    "past_key_values": cache_state,
                    "use_cache": True,
                    "cache_position": cache_position,
                    "apply_lm_head": is_final,  # Only apply LM head if this completes all layers
                }
                logger.debug(
                    f"🔧 Non-first shard: using cache_position={cache_position.tolist()}, "
                    f"letting model compute position_ids and attention_mask internally"
                )

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Update cache - CRITICAL for autoregressive generation
                # Each shard maintains its own KV cache for its layers
                if (
                    hasattr(outputs, "past_key_values")
                    and outputs.past_key_values is not None
                ):
                    self.caches[request_id] = outputs.past_key_values

                    # 🐛 DEBUG: Detailed cache update logging
                    logger.debug("=" * 80)
                    logger.debug("💾 KV CACHE UPDATE")
                    logger.debug("=" * 80)

                    # Log cache size to verify it's growing
                    # Handle both Cache objects (modern) and tuple caches (legacy)
                    try:
                        if hasattr(outputs.past_key_values, "get_seq_length"):
                            # Modern Cache object (DynamicCache, StaticCache, etc.)
                            # Use _get_cache_seq_length for shard-safe seq length
                            cache_seq_len = _get_cache_seq_length(outputs.past_key_values)
                            cache_num_layers = len(outputs.past_key_values) if hasattr(outputs.past_key_values, "__len__") else 0

                            logger.debug(
                                f"Cache type: {type(outputs.past_key_values).__name__}"
                            )
                            logger.debug(f"Number of layers cached: {cache_num_layers}")
                            logger.debug(f"Sequence length: {cache_seq_len}")
                            logger.debug(
                                f"My shard layers: {self.shard.start_layer}-{self.shard.end_layer}"
                            )

                            # 🚨 CRITICAL: Check if cache matches shard
                            expected_shard_layers = (
                                self.shard.end_layer - self.shard.start_layer + 1
                            )
                            if cache_num_layers == expected_shard_layers:
                                logger.debug(
                                    f"✅ Cache matches shard: {cache_num_layers} layers"
                                )
                            elif cache_num_layers == self.shard.n_layers:
                                # Worker shards use global layer indices, so cache has
                                # n_layers slots but only expected_shard_layers are populated
                                if self.shard.start_layer > 0:
                                    logger.debug(
                                        f"✅ Cache uses global indices: {cache_num_layers} slots "
                                        f"({expected_shard_layers} populated for layers "
                                        f"{self.shard.start_layer}-{self.shard.end_layer})"
                                    )
                                else:
                                    logger.debug(
                                        f"✅ Cache has all model layers: {cache_num_layers}"
                                    )
                            else:
                                logger.warning(
                                    f"⚠️  UNEXPECTED cache size: {cache_num_layers} layers"
                                )
                                logger.warning(
                                    f"   Expected {expected_shard_layers} (shard) or {self.shard.n_layers} (full model)"
                                )

                            # Show per-layer cache info (using get_seq_length per layer)
                            logger.debug("Layer-by-layer cache inspection:")
                            show_layers = list(range(min(3, cache_num_layers)))
                            if cache_num_layers > 4:
                                show_layers.append(cache_num_layers - 1)
                            for i in show_layers:
                                try:
                                    layer_seq = outputs.past_key_values.get_seq_length(i)
                                    status = "populated" if layer_seq > 0 else "empty"
                                    logger.debug(
                                        f"   Layer {i}: seq_len={layer_seq} ({status})"
                                    )
                                except Exception:
                                    logger.debug(f"   Layer {i}: (error reading)")
                            if cache_num_layers > 4:
                                logger.debug(f"   ... ({cache_num_layers - len(show_layers)} layers omitted)")
                        else:
                            # Legacy tuple format
                            cache_seq_len = (
                                outputs.past_key_values[0][0].shape[2]
                                if outputs.past_key_values[0] is not None
                                else 0
                            )
                            cache_num_layers = len(outputs.past_key_values)
                            logger.debug("Cache type: tuple (legacy)")
                            logger.debug(f"Number of layers cached: {cache_num_layers}")
                            logger.debug(f"Sequence length: {cache_seq_len}")

                        logger.debug(f"✅ Cache updated for request {request_id}")
                    except Exception as e:
                        logger.warning(f"Could not inspect cache structure: {e}")
                        logger.debug(f"✅ Updated KV cache for request {request_id}")

                    logger.debug("=" * 80)
                else:
                    logger.warning("⚠️  Model did not return past_key_values!")
                    logger.warning(
                        "   Cache will NOT be updated - this will break autoregressive generation!"
                    )

                # Get output tensor
                if hasattr(outputs, "logits"):
                    output_tensor = outputs.logits
                    logger.debug(f"Extracted logits: {output_tensor.shape}")
                else:
                    # For middle shards, outputs might be hidden states
                    output_tensor = (
                        outputs.last_hidden_state
                        if hasattr(outputs, "last_hidden_state")
                        else outputs[0]
                    )
                    logger.debug(f"Extracted hidden states: {output_tensor.shape}")

                # Convert BFloat16 to float32 for numpy compatibility
                if output_tensor.dtype == torch.bfloat16:
                    output_tensor = output_tensor.float()

                # Store hidden size in inference state for validation
                if not self.shard.is_last_layer():
                    inference_state["hidden_size"] = output_tensor.shape[-1]

                return output_tensor.cpu().numpy()

        output_data = await self._run_in_model_thread(_infer)
        return output_data, inference_state

    async def ensure_shard(self, shard: Shard, force_reload: bool = False):
        """
        Ensure the correct model shard is loaded.

        Note: In ring pipeline mode, each node loads its assigned shard once.
        Subsequent calls with different shard specs (e.g., base_shard vs current_shard)
        for the SAME model_id will reuse the already-loaded shard to avoid reloading.

        Args:
            shard: The shard specification to ensure is loaded.
            force_reload: If True, force reloading even if same model_id (for re-sharding).
        """
        async with self._shard_lock:
            # Quick check if already loaded
            if self.shard == shard and not force_reload:
                return

            # For now, use HuggingFace model ID directly
            # In future, this will download from a shard downloader
            model_id = shard.model_id

            # Reload if:
            # 1. No shard loaded yet
            # 2. Different model_id
            # 3. Different layer range (shard assignment changed)
            # 4. Force reload requested (e.g., topology update changed shard assignment)
            needs_reload = (
                self.shard is None
                or self.shard.model_id != shard.model_id
                or self.shard.start_layer != shard.start_layer
                or self.shard.end_layer != shard.end_layer
                or force_reload
            )
            if needs_reload:
                logger.info(
                    f"Loading shard: {shard} "
                    f"(previous: {self.shard}, force={force_reload})"
                )
                await self._load_shard(model_id, shard)
                self.shard = shard

                # Clear caches and session when switching models/shards
                self.caches.clear()
                self.session.clear()
            else:
                # Same model and layer range but different n_layers — no reload needed
                logger.debug(
                    f"Shard metadata changed (n_layers), keeping loaded shard: "
                    f"loaded={self.shard}, requested={shard}"
                )
                self.shard = shard

    async def _load_shard(self, model_id: str, shard: Shard):
        """Load model shard and tokenizer.

        Uses selective layer loading: creates an empty model skeleton on the
        ``meta`` device (zero memory), then loads *only* the safetensors
        weights needed for this shard directly from disk.  Unneeded layers
        stay as zero-byte ``meta`` tensors and are never materialised.

        This is dramatically more memory-efficient than the old approach of
        loading the full model and then freeing unneeded layers, especially
        for large models where the full checkpoint may not fit in RAM.
        """

        def _load():
            import json
            import os

            from accelerate import init_empty_weights
            from accelerate.utils import set_module_tensor_to_device
            from safetensors import safe_open
            from transformers import AutoConfig, AutoModelForCausalLM

            logger.info(f"🔧 Loading shard {shard} for model {model_id}")

            # Load config first
            config = AutoConfig.from_pretrained(
                model_id, cache_dir=self.cache_dir, trust_remote_code=True
            )

            logger.info("📋 Model Config:")
            logger.info(f"   Model type: {config.model_type}")
            logger.info(f"   Hidden size: {config.hidden_size}")
            logger.info(f"   Num layers: {config.num_hidden_layers}")
            logger.info(f"   Vocab size: {config.vocab_size}")

            torch_dtype = self._get_torch_dtype()

            # ── Step 1: Create empty model skeleton (0 bytes) ──────────
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(
                    config, dtype=torch_dtype, trust_remote_code=True,
                )

            # ── Step 2: Resolve snapshot directory on disk ─────────────
            try:
                from huggingface_hub import snapshot_download

                snap_dir = snapshot_download(
                    model_id, cache_dir=self.cache_dir, local_files_only=True,
                )
            except Exception:
                # Fallback: try to find it manually
                safe_model_id = model_id.replace("/", "--")
                cache_root = os.path.join(self.cache_dir, f"models--{safe_model_id}")
                refs_path = os.path.join(cache_root, "refs", "main")
                if os.path.exists(refs_path):
                    with open(refs_path) as f:
                        commit_hash = f.read().strip()
                    snap_dir = os.path.join(cache_root, "snapshots", commit_hash)
                else:
                    raise FileNotFoundError(
                        f"Cannot resolve snapshot directory for {model_id} in {self.cache_dir}"
                    )

            logger.info(f"📂 Snapshot directory: {snap_dir}")

            # ── Step 3: Determine needed weight keys ───────────────────
            needed_prefixes = []
            for i in range(shard.start_layer, shard.end_layer + 1):
                needed_prefixes.append(f"model.layers.{i}.")

            if shard.is_first_layer():
                needed_prefixes.append("model.embed_tokens.")
            if shard.is_last_layer():
                needed_prefixes.append("model.norm.")
                needed_prefixes.append("lm_head.")

            # rotary_emb weights (if stored — most models compute them at
            # runtime, but check just in case)
            needed_prefixes.append("model.rotary_emb.")

            # ── Step 4: Build file→keys mapping ───────────────────────
            index_path = os.path.join(snap_dir, "model.safetensors.index.json")
            if os.path.exists(index_path):
                # Multi-file model (e.g., Llama-3.2-3B)
                with open(index_path) as f:
                    weight_map = json.load(f)["weight_map"]
                file_to_keys: dict[str, list[str]] = {}
                for key, fname in weight_map.items():
                    if any(key.startswith(p) for p in needed_prefixes):
                        file_to_keys.setdefault(fname, []).append(key)
                total_keys = len(weight_map)
            else:
                # Single-file model (e.g., Qwen3-0.6B)
                st_path = os.path.join(snap_dir, "model.safetensors")
                with safe_open(st_path, framework="pt") as f:
                    all_keys = list(f.keys())
                total_keys = len(all_keys)
                needed_keys = [
                    k for k in all_keys
                    if any(k.startswith(p) for p in needed_prefixes)
                ]
                file_to_keys = {"model.safetensors": needed_keys}

            needed_count = sum(len(v) for v in file_to_keys.values())
            skipped_count = total_keys - needed_count
            logger.info(
                f"📦 Selective loading: {needed_count}/{total_keys} tensors "
                f"(skipping {skipped_count} unneeded)"
            )

            # ── Step 5: Load only the needed tensors ───────────────────
            device = "cuda" if torch.cuda.is_available() else "cpu"
            loaded = 0
            import time as _time
            t0 = _time.time()

            for fname, keys in file_to_keys.items():
                filepath = os.path.join(snap_dir, fname)
                with safe_open(filepath, framework="pt", device="cpu") as f:
                    for key in keys:
                        tensor = f.get_tensor(key)
                        if tensor.dtype != torch_dtype:
                            tensor = tensor.to(torch_dtype)
                        set_module_tensor_to_device(model, key, device, value=tensor)
                        loaded += 1

            load_time = _time.time() - t0
            logger.info(
                f"✅ Loaded {loaded} tensors to {device} in {load_time:.2f}s"
            )

            # ── Step 6: Wrap in shard wrapper ──────────────────────────
            model = self._wrap_model_in_shard(model, shard)
            model.eval()

            logger.info(f"Successfully loaded shard {shard}")
            return model, config

        self.model, self.config = await self._run_in_model_thread(_load)

        # Load tokenizer
        def _load_tokenizer():
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                use_fast=False,
            )

            # Set pad token if not exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            return tokenizer

        self.tokenizer = await self._run_in_tokenizer_thread(_load_tokenizer)

    def _wrap_model_in_shard(self, model, shard: Shard):
        """
        Wrap model to only execute assigned layers.

        Uses the TransformersShard wrapper to extract and execute only
        the layers assigned to this shard.

        Note: With selective loading, unneeded layers are already on the
        ``meta`` device (0 bytes) so ``free_unneeded_layers()`` is
        unnecessary.  We still call it as a safety net — it handles the
        case where a layer is already ``None`` or on ``meta`` gracefully.
        """
        from sharded_model import TransformersShard

        logger.info(f"Wrapping model in shard: {shard}")
        sharded = TransformersShard(model, shard)

        # Safety net: free any layers that might still be materialised
        # (e.g. if the model was loaded via from_pretrained fallback).
        sharded.free_unneeded_layers()

        return sharded

    def _create_device_map_for_shard(self, shard: Shard) -> Union[str, Dict[str, Any]]:
        """Create device map for the specific shard."""
        # Simple auto device mapping for now
        if torch.cuda.is_available():
            return "auto"
        else:
            return "cpu"

    def _get_torch_dtype(self) -> torch.dtype:
        """Get appropriate torch dtype based on hardware."""
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            else:
                return torch.float16
        else:
            return torch.float32

    def _get_available_devices(self) -> list:
        """Get list of available compute devices."""
        devices = []

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(i)

        if not devices:
            devices.append("cpu")

        return devices

    async def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, "_model_thread"):
            self._model_thread.shutdown(wait=True)
        if hasattr(self, "_tokenizer_thread"):
            self._tokenizer_thread.shutdown(wait=True)

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, "_model_thread"):
                self._model_thread.shutdown(wait=False)
            if hasattr(self, "_tokenizer_thread"):
                self._tokenizer_thread.shutdown(wait=False)
        except Exception:
            pass
