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

    Args:
        cache_state: Either a DynamicCache object or tuple of per-layer caches

    Returns:
        Sequence length of cached tokens (0 if no cache)
    """
    if cache_state is None:
        return 0

    # Check if it's a DynamicCache object
    if hasattr(cache_state, "get_seq_length"):
        # Try default (layer 0) first
        try:
            seq_len = cache_state.get_seq_length()
            if seq_len > 0:
                return seq_len
        except Exception:
            pass

        # If layer 0 is empty (e.g. sharded model with offset layers),
        # find the first non-empty layer
        if hasattr(cache_state, "key_cache"):
            for i, key_tensor in enumerate(cache_state.key_cache):
                if key_tensor is not None and key_tensor.dim() >= 3:
                    return key_tensor.shape[2]
        return 0

    # Check if it has key_cache attribute (DynamicCache alternative method)
    if hasattr(cache_state, "key_cache") and len(cache_state.key_cache) > 0:
        # Find first non-empty cache entry
        for key_tensor in cache_state.key_cache:
            if key_tensor is not None and key_tensor.dim() >= 3:
                return key_tensor.shape[2]
        return 0

    # Fallback: tuple/list of per-layer caches
    if isinstance(cache_state, (list, tuple)) and len(cache_state) > 0:
        if cache_state[0] is not None:
            # Tuple of (key, value) tensors
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

    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        """Encode text prompt to token IDs using chat template."""
        await self.ensure_shard(shard)

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
        await self.ensure_shard(shard)

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

                # Inspect cache structure
                if hasattr(cache_state, "key_cache"):
                    # DynamicCache or similar
                    num_cached_layers = len(cache_state.key_cache)
                    logger.debug(f"   Number of layers in cache: {num_cached_layers}")
                    logger.debug(f"   Sequence length: {past_length}")

                    # Show shape of each cached layer
                    for i, key_tensor in enumerate(
                        cache_state.key_cache[: min(3, num_cached_layers)]
                    ):
                        if key_tensor is not None:
                            logger.debug(f"   Layer {i} key shape: {key_tensor.shape}")
                    if num_cached_layers > 3:
                        logger.debug(f"   ... and {num_cached_layers - 3} more layers")

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

                # CRITICAL: Non-first shards also need position_ids and attention_mask!
                # Prima.cpp passes inp_pos to ALL nodes, not just the first one.
                inputs = {
                    "inputs_embeds": input_tensor,
                    "past_key_values": cache_state,
                    "use_cache": True,
                    "apply_lm_head": is_final,  # Only apply LM head if this completes all layers
                }

                # Add position_ids if provided (CRITICAL for RoPE in middle layers)
                if position_ids is not None:
                    position_ids_tensor = (
                        torch.from_numpy(position_ids).long().to(device)
                    )
                    inputs["position_ids"] = position_ids_tensor
                    logger.debug(
                        f"Non-first shard using position_ids: {position_ids_tensor.shape}"
                    )

                # Add attention_mask if provided
                if attention_mask is not None:
                    attention_mask_tensor = (
                        torch.from_numpy(attention_mask).bool().to(device)
                    )
                    inputs["attention_mask"] = attention_mask_tensor
                    logger.debug(
                        f"Non-first shard using attention_mask: {attention_mask_tensor.shape}"
                    )

                # CRITICAL: Add cache_position for middle/last shards too
                # This is needed for the layers to properly update KV cache
                past_length = _get_cache_seq_length(cache_state)
                if past_length > 0:
                    cache_position = torch.arange(
                        past_length,
                        past_length + seq_len,
                        dtype=torch.long,
                        device=device,
                    )
                    inputs["cache_position"] = cache_position
                    logger.debug(
                        f"🎯 Non-first shard cache_position: {cache_position.tolist()}"
                    )
                else:
                    cache_position = torch.arange(
                        seq_len, dtype=torch.long, device=device
                    )
                    inputs["cache_position"] = cache_position
                    logger.debug(
                        f"🎯 Non-first shard cache_position: {cache_position.tolist()} (new cache)"
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
                            cache_seq_len = outputs.past_key_values.get_seq_length(0)
                            cache_num_layers = len(outputs.past_key_values)

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
                                logger.warning(
                                    f"⚠️  Cache has ALL model layers ({cache_num_layers}), not just shard layers ({expected_shard_layers})"
                                )
                                logger.warning(
                                    "   This might indicate the sharded model wrapper is not filtering correctly!"
                                )
                            else:
                                logger.warning(
                                    f"⚠️  UNEXPECTED cache size: {cache_num_layers} layers"
                                )
                                logger.warning(
                                    f"   Expected {expected_shard_layers} (shard) or {self.shard.n_layers} (full model)"
                                )

                            # Show individual layer cache shapes (first 3 and last 1)
                            if hasattr(outputs.past_key_values, "key_cache"):
                                logger.debug("Layer-by-layer cache inspection:")
                                for i in range(min(3, cache_num_layers)):
                                    k_shape = (
                                        outputs.past_key_values.key_cache[i].shape
                                        if outputs.past_key_values.key_cache[i]
                                        is not None
                                        else None
                                    )
                                    v_shape = (
                                        outputs.past_key_values.value_cache[i].shape
                                        if outputs.past_key_values.value_cache[i]
                                        is not None
                                        else None
                                    )
                                    logger.debug(
                                        f"   Layer {i}: K={k_shape}, V={v_shape}"
                                    )
                                if cache_num_layers > 4:
                                    i = cache_num_layers - 1
                                    k_shape = (
                                        outputs.past_key_values.key_cache[i].shape
                                        if outputs.past_key_values.key_cache[i]
                                        is not None
                                        else None
                                    )
                                    v_shape = (
                                        outputs.past_key_values.value_cache[i].shape
                                        if outputs.past_key_values.value_cache[i]
                                        is not None
                                        else None
                                    )
                                    logger.debug("   ...")
                                    logger.debug(
                                        f"   Layer {i}: K={k_shape}, V={v_shape}"
                                    )
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

    async def ensure_shard(self, shard: Shard):
        """
        Ensure the correct model shard is loaded.

        Note: In ring pipeline mode, each node loads its assigned shard once.
        Subsequent calls with different shard specs (e.g., base_shard vs current_shard)
        for the SAME model_id will reuse the already-loaded shard to avoid reloading.
        """
        async with self._shard_lock:
            # Quick check if already loaded
            if self.shard == shard:
                return

            # For now, use HuggingFace model ID directly
            # In future, this will download from a shard downloader
            model_id = shard.model_id

            # Only reload if model_id changes, NOT if just layer range changes
            # This allows ring pipeline to pass base_shard for coordination
            # while keeping the node's assigned shard loaded
            if self.shard is None or self.shard.model_id != shard.model_id:
                await self._load_shard(model_id, shard)
                self.shard = shard

                # Clear caches and session when switching models
                self.caches.clear()
                self.session.clear()
            else:
                # Same model, different layer spec - don't reload
                # This happens when ring pipeline passes base_shard
                # but node already has current_shard loaded
                logger.debug(
                    f"Shard spec changed but same model_id, keeping loaded shard: "
                    f"loaded={self.shard}, requested={shard}"
                )

    async def _load_shard(self, model_id: str, shard: Shard):
        """Load model shard and tokenizer using direct partial loading."""

        def _load():
            from shard_loader import load_shard_direct

            logger.info(f"🔧 Loading shard {shard} for model {model_id}")
            logger.info("   Using DIRECT partial loading (memory-efficient)")

            # Get device and dtype configuration
            device = self._get_target_device()
            torch_dtype = self._get_torch_dtype()

            # Use direct partial loading - only loads weights needed for this shard
            # This never allocates memory for unused layers
            model, config = load_shard_direct(
                model_id=model_id,
                shard=shard,
                cache_dir=self.cache_dir,
                device=device,
                dtype=torch_dtype,
            )

            # Wrap model in shard wrapper for forward pass handling
            model = self._wrap_model_in_shard(model, shard, pre_pruned=True)

            logger.info(f"✅ Successfully loaded shard {shard}")
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

    def _wrap_model_in_shard(self, model, shard: Shard, pre_pruned: bool = False):
        """
        Wrap model to only execute assigned layers.

        Uses the TransformersShard wrapper to extract and execute only
        the layers assigned to this shard.

        Args:
            model: The model to wrap
            shard: Shard specification
            pre_pruned: If True, the model has already been pruned to only
                       contain the shard's layers (from direct partial loading)
        """
        from sharded_model import TransformersShard

        logger.info(f"Wrapping model in shard: {shard} (pre_pruned={pre_pruned})")
        return TransformersShard(model, shard, pre_pruned=pre_pruned)

    def _get_target_device(self) -> str:
        """Get the target device string for model loading."""
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

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
