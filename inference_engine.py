"""
Inference Engine abstract base class for distributed model inference.
Adapted from exo's inference engine interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
from shard import Shard


class InferenceEngine(ABC):
    """
    Abstract base class for inference engines that support sharded model execution.

    Implementations must handle:
    - Loading and managing model shards
    - Token encoding/decoding
    - Tensor inference across shard boundaries
    - Sampling from logits
    """

    def __init__(self):
        self.shard: Optional[Shard] = None
        self.session: Dict[str, Any] = {}

    @abstractmethod
    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        """
        Encode text prompt to token IDs.

        Args:
            shard: The model shard to use for encoding
            prompt: Text prompt to encode

        Returns:
            Array of token IDs
        """
        pass

    @abstractmethod
    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        """
        Decode token IDs to text.

        Args:
            shard: The model shard to use for decoding
            tokens: Array of token IDs

        Returns:
            Decoded text string
        """
        pass

    @abstractmethod
    async def infer_tensor(
        self,
        request_id: str,
        shard: Shard,
        input_data: np.ndarray,
        inference_state: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Run inference on input tensor through the assigned shard.

        For first shard: input_data should be token IDs
        For middle/last shards: input_data should be hidden states from previous shard

        Args:
            request_id: Unique identifier for this inference request
            shard: The model shard to execute
            input_data: Input tensor (tokens or hidden states)
            inference_state: Optional state to maintain across shard boundaries

        Returns:
            Tuple of (output_tensor, updated_inference_state)
            - For first/middle shards: output is hidden states
            - For last shard: output is logits
        """
        pass

    @abstractmethod
    async def sample(
        self, logits: np.ndarray, temp: float = 0.7, top_p: float = 0.9
    ) -> np.ndarray:
        """
        Sample next token from logits.

        Args:
            logits: Model output logits
            temp: Temperature for sampling (0 = greedy)
            top_p: Nucleus sampling threshold

        Returns:
            Sampled token ID(s)
        """
        pass

    @abstractmethod
    async def ensure_shard(self, shard: Shard):
        """
        Ensure the correct model shard is loaded.
        Downloads and loads the shard if needed.

        Args:
            shard: The shard to load
        """
        pass

    async def infer_prompt(
        self,
        request_id: str,
        shard: Shard,
        prompt: str,
        inference_state: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Convenience method to encode prompt and run inference.

        Args:
            request_id: Unique identifier for this inference request
            shard: The model shard to execute
            prompt: Text prompt
            inference_state: Optional state to maintain

        Returns:
            Tuple of (output_tensor, updated_inference_state)
        """
        tokens = await self.encode(shard, prompt)
        x = tokens.reshape(1, -1)
        output_data, inference_state = await self.infer_tensor(
            request_id, shard, x, inference_state
        )
        return output_data, inference_state

    async def save_session(self, key: str, value: Any):
        """Save a value to the session."""
        self.session[key] = value

    async def clear_session(self):
        """Clear the session."""
        self.session.clear()
