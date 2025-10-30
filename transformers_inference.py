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
        """Encode text prompt to token IDs."""
        await self.ensure_shard(shard)

        def _encode():
            # Use simple encoding - chat template with add_generation_prompt=True
            # adds trailing newlines that cause the model to generate only newlines
            tokens = self.tokenizer.encode(prompt, add_special_tokens=True)

            logger.info(
                f"Encoded '{prompt[:50]}...' to {len(tokens)} tokens: {tokens[:10]}..."
            )
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
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """Run inference on input tensor through the assigned shard."""
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

            # Prepare inputs based on shard position and input shape
            if self.shard.is_first_layer() and input_tensor.dim() <= 2:
                # First shard: expects input_ids (2D: batch_size x seq_len)
                if input_tensor.dim() == 1:
                    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

                batch_size, seq_len = input_tensor.shape

                # Create attention mask and position IDs
                # Note: attention_mask should be bool or float, not long
                attention_mask = torch.ones(
                    batch_size, seq_len, dtype=torch.bool, device=device
                )

                # Position IDs need to account for cached positions
                if cache_state is not None and len(cache_state) > 0:
                    # Get the cached sequence length from the first cache entry
                    # past_key_values is a tuple of (key, value) tuples for each layer
                    past_length = (
                        cache_state[0][0].shape[2] if cache_state[0] is not None else 0
                    )
                    position_ids = (
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
                        f"Created position_ids with cache: past_length={past_length}, position_ids={position_ids}"
                    )
                else:
                    # No cache, start from 0
                    position_ids = (
                        torch.arange(seq_len, dtype=torch.long, device=device)
                        .unsqueeze(0)
                        .expand(batch_size, -1)
                    )
                    logger.debug(
                        f"Created position_ids without cache: position_ids={position_ids}"
                    )

                inputs = {
                    "input_ids": input_tensor,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "past_key_values": cache_state,
                    "use_cache": True,
                }
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

                inputs = {
                    "inputs_embeds": input_tensor,
                    "past_key_values": cache_state,
                    "use_cache": True,
                }

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Update cache
                if (
                    hasattr(outputs, "past_key_values")
                    and outputs.past_key_values is not None
                ):
                    self.caches[request_id] = outputs.past_key_values

                # Get output tensor
                if hasattr(outputs, "logits"):
                    output_tensor = outputs.logits
                else:
                    # For middle shards, outputs might be hidden states
                    output_tensor = (
                        outputs.last_hidden_state
                        if hasattr(outputs, "last_hidden_state")
                        else outputs[0]
                    )

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
        """Load model shard and tokenizer."""

        def _load():
            from transformers import AutoConfig, AutoModelForCausalLM

            logger.info(f"Loading shard {shard} for model {model_id}")

            # Load config first
            config = AutoConfig.from_pretrained(
                model_id, cache_dir=self.cache_dir, trust_remote_code=True
            )

            # Get device and dtype configuration
            device_map = self._create_device_map_for_shard(shard)
            torch_dtype = self._get_torch_dtype()

            # Load model with appropriate configuration
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                cache_dir=self.cache_dir,
                device_map=device_map,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

            # Wrap model in shard wrapper
            model = self._wrap_model_in_shard(model, shard)

            # Set to eval mode
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
        """
        from sharded_model import TransformersShard

        logger.info(f"Wrapping model in shard: {shard}")
        return TransformersShard(model, shard)

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
