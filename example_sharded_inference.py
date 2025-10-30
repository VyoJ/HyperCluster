"""
Simple example demonstrating sharded inference.

This shows how to split a model across "virtual nodes" and run distributed inference.
In a real deployment, each shard would run on a different physical machine.
"""

import asyncio
import logging

import numpy as np
from shard import Shard
from transformers_inference import TransformersShardedInferenceEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def run_sharded_inference():
    """
    Simulate distributed inference across 3 nodes.

    In production:
    - Node 1 would run shard 0 (layers 0-7)
    - Node 2 would run shard 1 (layers 8-15)
    - Node 3 would run shard 2 (layers 16-23)

    Here we simulate all 3 nodes locally.
    """

    print("\n" + "=" * 70)
    print("DISTRIBUTED SHARDED INFERENCE DEMO")
    print("=" * 70)

    # Model configuration
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    total_layers = 24

    # Split model into 3 shards (simulating 3 nodes)
    print(f"\nModel: {model_id}")
    print(f"Total layers: {total_layers}")
    print("\nShard distribution:")
    print("  Node 1: Layers  0-7  (first shard - has embeddings)")
    print("  Node 2: Layers  8-15 (middle shard)")
    print("  Node 3: Layers 16-23 (last shard - has LM head)")

    shards = [
        Shard(model_id=model_id, start_layer=0, end_layer=7, n_layers=total_layers),
        Shard(model_id=model_id, start_layer=8, end_layer=15, n_layers=total_layers),
        Shard(model_id=model_id, start_layer=16, end_layer=23, n_layers=total_layers),
    ]

    # Create inference engines (one per "node")
    print("\nInitializing engines...")
    engines = [TransformersShardedInferenceEngine() for _ in range(3)]

    try:
        # User prompt - simple completion-style prompt
        prompt = "The three laws of robotics are"
        print(f"\n{'='*70}")
        print(f"PROMPT: {prompt}")
        print("=" * 70)

        # Step 1: Encode on first node
        print("\n[Node 1] Encoding prompt to tokens...")
        tokens = await engines[0].encode(shards[0], prompt)
        print(f"[Node 1] Encoded {len(tokens)} tokens")

        # Generate multiple tokens
        max_new_tokens = 30
        request_id = "demo-request"
        all_tokens = tokens.tolist()

        print(f"\n[Starting generation of {max_new_tokens} tokens...]")
        print("-" * 70)

        current_input = tokens

        for step in range(max_new_tokens):
            print(f"\n>> Generation step {step + 1}/{max_new_tokens}")

            # Forward through all shards
            hidden_states = current_input
            state = None

            # Shard 0: First node
            print("   [Node 1] Processing through layers 0-7...")
            print(f"   [Node 1] Input shape: {hidden_states.shape}")
            hidden_states, state = await engines[0].infer_tensor(
                request_id, shards[0], hidden_states, state
            )
            print(f"   [Node 1] Output shape: {hidden_states.shape} (hidden states)")
            print("   [Node 1] Sending to Node 2...")

            # Shard 1: Middle node
            print("   [Node 2] Processing through layers 8-15...")
            print(f"   [Node 2] Input shape: {hidden_states.shape}")
            hidden_states, state = await engines[1].infer_tensor(
                request_id, shards[1], hidden_states, state
            )
            print(f"   [Node 2] Output shape: {hidden_states.shape} (hidden states)")
            print("   [Node 2] Sending to Node 3...")

            # Shard 2: Last node
            print("   [Node 3] Processing through layers 16-23...")
            print(f"   [Node 3] Input shape: {hidden_states.shape}")
            hidden_states, state = await engines[2].infer_tensor(
                request_id, shards[2], hidden_states, state
            )
            print(f"   [Node 3] Output shape: {hidden_states.shape} (logits)")

            # Sample next token
            print("   [Node 3] Sampling next token...")
            next_token = await engines[2].sample(hidden_states, temp=0.7, top_p=0.9)
            token_id = int(next_token.flatten()[0])
            all_tokens.append(token_id)

            # Decode current sequence
            current_text = await engines[2].decode(shards[2], np.array(all_tokens))

            # Show progress
            print(f"   [Node 3] Sampled token ID: {token_id}")
            print(f"\n   Current text: {current_text}")

            # Check for end of sequence
            if token_id in [2, 151643, 151645]:  # Common EOS tokens
                print("\n   [Node 3] EOS token detected, stopping generation")
                break

            # Next iteration: send only the new token back to Node 1
            current_input = next_token
            print("   [Node 3] Sending next token to Node 1 for next iteration...")

        # Final output
        print("\n" + "=" * 70)
        print("FINAL GENERATED TEXT:")
        print("=" * 70)
        final_text = await engines[2].decode(shards[2], np.array(all_tokens))
        print(f"\n{final_text}\n")
        print("=" * 70)

        # Statistics
        print("\nGeneration statistics:")
        print(f"  Total tokens generated: {len(all_tokens) - len(tokens)}")
        print(f"  Total tokens (prompt + generated): {len(all_tokens)}")
        print("  Nodes used: 3")
        print("  Layers per node: ~8")

    finally:
        print("\nCleaning up engines...")
        for i, engine in enumerate(engines):
            await engine.cleanup()
        print("Done!")


async def compare_approaches():
    """
    Compare full model vs sharded model approaches.
    """
    print("\n" + "=" * 70)
    print("COMPARISON: Full Model vs Sharded Model")
    print("=" * 70)

    print("\nFull Model Approach:")
    print("  • All 24 layers on one device")
    print("  • Memory: ~4GB (for Qwen2.5-0.5B)")
    print("  • Latency: Low (no network)")
    print("  • Scalability: Limited by single device")
    print("  • Max model size: Limited by device memory")

    print("\nSharded Model Approach (3 nodes):")
    print("  • 8 layers per device")
    print("  • Memory: ~1.3GB per node")
    print("  • Latency: Higher (network transfer)")
    print("  • Scalability: Can add more nodes")
    print("  • Max model size: Sum of all node memories")

    print("\nNetwork Transfer per Token:")
    print("  • Hidden state size: batch × seq × hidden_dim × 4 bytes")
    print("  • For Qwen2.5-0.5B: 1 × 1 × 896 × 4 = ~3.5 KB")
    print("  • Between each node pair: ~3.5 KB")
    print("  • Total per token: ~7 KB (2 transfers)")

    print("\nWhen to use Sharded Inference:")
    print("  [*] Model too large for single device")
    print("  [*] Multiple devices available on network")
    print("  [*] Throughput more important than latency")
    print("  [*] Want to distribute computational load")


async def main():
    """Run the demo."""
    try:
        # Run the sharded inference demo
        await run_sharded_inference()

        # Show comparison
        await compare_approaches()

        print("\n" + "=" * 70)
        print("Demo complete! Check docs/SHARDED_INFERENCE_GUIDE.md for more info.")
        print("=" * 70 + "\n")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
