"""
Partitioning strategies for distributing model layers across nodes.
Adapted from exo's partitioning system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from shard import Shard
from topology import Topology


@dataclass
class Partition:
    """
    Represents a partition of the model space assigned to a node.

    The model is conceptually split into a continuous range [0, 1].
    Each partition gets a slice [start, end) which maps to actual layers.
    """

    node_id: str
    start: float  # Range: 0.0 to 1.0
    end: float  # Range: 0.0 to 1.0


class PartitioningStrategy(ABC):
    """Base class for partitioning strategies."""

    @abstractmethod
    def partition(self, topology: Topology) -> List[Partition]:
        """
        Partition the model space across nodes in the topology.

        Args:
            topology: Current network topology with node capabilities

        Returns:
            List of partitions, one per node
        """
        pass


class RingMemoryWeightedPartitioningStrategy(PartitioningStrategy):
    """
    Memory-weighted ring partitioning strategy.

    Distributes model layers proportionally to each node's available memory.
    Nodes with more memory get assigned more layers.

    This strategy:
    - Sorts nodes by memory (most to least)
    - Assigns partition sizes proportional to memory
    - Creates a ring topology for inference flow
    """

    def partition(self, topology: Topology) -> List[Partition]:
        """Partition model space based on node memory."""
        nodes = list(topology.all_nodes())

        if not nodes:
            return []

        # Sort by memory (descending), then by node_id for consistency
        nodes.sort(key=lambda x: (x[1].memory, x[0]), reverse=True)

        # Calculate total memory
        total_memory = sum(node[1].memory for node in nodes)

        if total_memory == 0:
            # Equal partitioning if no memory info
            partition_size = 1.0 / len(nodes)
            return [
                Partition(node[0], i * partition_size, (i + 1) * partition_size)
                for i, node in enumerate(nodes)
            ]

        # Create partitions proportional to memory
        partitions = []
        start = 0.0

        for node_id, capabilities in nodes:
            # Calculate this node's share
            memory_fraction = capabilities.memory / total_memory
            end = round(
                start + memory_fraction, 5
            )  # Round to avoid floating point issues

            partitions.append(Partition(node_id, start, end))
            start = end

        # Ensure the last partition extends to 1.0
        if partitions:
            partitions[-1] = Partition(
                partitions[-1].node_id, partitions[-1].start, 1.0
            )

        return partitions


class UniformPartitioningStrategy(PartitioningStrategy):
    """
    Uniform partitioning strategy.

    Distributes model layers equally across all nodes,
    regardless of their capabilities.
    """

    def partition(self, topology: Topology) -> List[Partition]:
        """Partition model space uniformly."""
        nodes = list(topology.all_nodes())

        if not nodes:
            return []

        # Sort by node_id for consistency
        nodes.sort(key=lambda x: x[0])

        partition_size = 1.0 / len(nodes)
        partitions = []

        for i, (node_id, _) in enumerate(nodes):
            start = i * partition_size
            end = (i + 1) * partition_size if i < len(nodes) - 1 else 1.0
            partitions.append(Partition(node_id, start, end))

        return partitions


def map_partitions_to_shards(
    partitions: List[Partition], num_layers: int, model_id: str
) -> List[Shard]:
    """
    Map abstract partitions to concrete shards with layer indices.

    Args:
        partitions: List of partitions in [0, 1] range
        num_layers: Total number of layers in the model
        model_id: Model identifier

    Returns:
        List of shards with concrete layer ranges
    """
    shards = []

    for i, partition in enumerate(partitions):
        # Convert fractional range to layer indices
        start_layer = int(partition.start * num_layers)
        end_layer = int(partition.end * num_layers) - 1

        # Ensure the last partition covers up to the last layer
        if i == len(partitions) - 1:
            end_layer = num_layers - 1

        # Ensure no empty shards
        if start_layer <= end_layer:
            shards.append(Shard(model_id, start_layer, end_layer, num_layers))

    # Ensure full coverage of all layers
    if shards and shards[-1].end_layer < num_layers - 1:
        shards[-1] = Shard(model_id, shards[-1].start_layer, num_layers - 1, num_layers)

    return shards
