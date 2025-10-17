"""
Example: Simple sharded inference test for HyperCluster

This script demonstrates how to set up and use sharded inference
on a single node for testing purposes.
"""

import asyncio
import logging

from device_capabilities import get_device_capabilities
from shard import Shard
from transformers_inference import TransformersShardedInferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_shard_basics():
    """Test basic shard functionality."""
    print("=" * 60)
    print("Testing Shard Basics")
    print("=" * 60)

    # Create a shard
    shard = Shard(
        model_id="Qwen/Qwen2.5-0.5B-Instruct", start_layer=0, end_layer=23, n_layers=24
    )

    print(f"Created shard: {shard}")
    print(f"Is first layer: {shard.is_first_layer()}")
    print(f"Is last layer: {shard.is_last_layer()}")
    print(f"Layer count: {shard.get_layer_count()}")
    print()


async def test_device_capabilities():
    """Test device capability detection."""
    print("=" * 60)
    print("Testing Device Capabilities")
    print("=" * 60)

    capabilities = await get_device_capabilities()
    print(f"Detected capabilities: {capabilities}")
    print()


async def test_inference_engine():
    """Test inference engine with a simple prompt."""
    print("=" * 60)
    print("Testing Inference Engine")
    print("=" * 60)

    # Create inference engine
    engine = TransformersShardedInferenceEngine(cache_dir="./model_cache")

    # Create a full-model shard (all layers)
    shard = Shard(
        model_id="Qwen/Qwen2.5-0.5B-Instruct", start_layer=0, end_layer=23, n_layers=24
    )

    print("Loading model shard...")
    await engine.ensure_shard(shard)
    print("Model loaded!")

    # Test encoding
    prompt = "What is artificial intelligence?"
    print(f"\nPrompt: {prompt}")

    tokens = await engine.encode(shard, prompt)
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")

    # Test inference
    print("\nRunning inference...")
    request_id = "test-request-1"
    output, state = await engine.infer_prompt(request_id, shard, prompt)

    print(f"Output shape: {output.shape}")

    # Sample a token
    print("\nSampling next token...")
    next_token = await engine.sample(output, temp=0.7)
    print(f"Next token: {next_token}")

    # Decode
    decoded = await engine.decode(shard, next_token)
    print(f"Decoded: '{decoded}'")

    print("\n✅ Inference engine test complete!")
    print()


async def test_partitioning():
    """Test partitioning strategy."""
    print("=" * 60)
    print("Testing Partitioning Strategy")
    print("=" * 60)

    from device_capabilities import DeviceCapabilities
    from partitioning_strategy import (
        RingMemoryWeightedPartitioningStrategy,
        map_partitions_to_shards,
    )
    from topology import Topology

    # Create mock topology
    topology = Topology()
    topology.update_node("node1", DeviceCapabilities("GPU1", "NVIDIA", 24, 82.6))
    topology.update_node("node2", DeviceCapabilities("GPU2", "NVIDIA", 16, 48.7))
    topology.update_node("node3", DeviceCapabilities("CPU1", "Intel", 8, 2.0))

    print("Topology:")
    for node_id, cap in topology.all_nodes():
        print(f"  {node_id}: {cap.memory}GB")

    # Apply partitioning
    strategy = RingMemoryWeightedPartitioningStrategy()
    partitions = strategy.partition(topology)

    print("\nPartitions:")
    for p in partitions:
        print(f"  {p.node_id}: {p.start:.3f} - {p.end:.3f}")

    # Map to shards
    shards = map_partitions_to_shards(partitions, 24, "test-model")

    print("\nShards:")
    for i, shard in enumerate(shards):
        print(f"  {i}: {shard}")

    print()


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("HyperCluster Sharded Inference Test Suite")
    print("=" * 60 + "\n")

    try:
        await test_shard_basics()
        await test_device_capabilities()
        await test_partitioning()

        # This test requires downloading a model
        print("⚠️  Next test will download a model (~1GB)")
        print("This may take a few minutes on first run...")
        response = input("Continue? (y/n): ")

        if response.lower() == "y":
            await test_inference_engine()
        else:
            print("Skipping inference test.")

        print("\n" + "=" * 60)
        print("✅ All tests complete!")
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print("\n❌ Tests failed!")


if __name__ == "__main__":
    asyncio.run(main())
