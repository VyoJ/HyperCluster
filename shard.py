"""
Shard abstraction for distributed model inference.
Adapted from exo's shard implementation.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Shard:
    """
    Represents a contiguous range of model layers assigned to a node.

    Attributes:
        model_id: Identifier for the model (e.g., "llama-3.1-8B")
        start_layer: First layer index (inclusive) for this shard
        end_layer: Last layer index (inclusive) for this shard
        n_layers: Total number of layers in the complete model
    """

    model_id: str
    start_layer: int
    end_layer: int
    n_layers: int

    def __hash__(self):
        return hash((self.model_id, self.start_layer, self.end_layer, self.n_layers))

    def is_first_layer(self) -> bool:
        """Check if this shard contains the first layer (embeddings)."""
        return self.start_layer == 0

    def is_last_layer(self) -> bool:
        """Check if this shard contains the last layer (LM head)."""
        return self.end_layer == self.n_layers - 1

    def get_layer_count(self) -> int:
        """Get the number of layers in this shard."""
        return self.end_layer - self.start_layer + 1

    def to_dict(self) -> Dict:
        """Convert shard to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "n_layers": self.n_layers,
        }

    @staticmethod
    def from_dict(data: Dict) -> "Shard":
        """Create shard from dictionary."""
        return Shard(**data)

    def overlaps(self, other: "Shard") -> bool:
        """Check if this shard overlaps with another shard."""
        return shards_overlap(self, other)

    def __repr__(self) -> str:
        return f"Shard(model={self.model_id}, layers={self.start_layer}-{self.end_layer}/{self.n_layers})"


def shards_overlap(shard1: Shard, shard2: Shard) -> bool:
    """Check if two shards overlap in their layer ranges."""
    return shard1.model_id == shard2.model_id and max(
        shard1.start_layer, shard2.start_layer
    ) <= min(shard1.end_layer, shard2.end_layer)
