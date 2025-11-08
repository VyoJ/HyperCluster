#!/usr/bin/env python3
"""
Example script to test generation stats logging.

This script demonstrates how stats are automatically logged for each generation.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from stats_logger import StatsLogger


async def test_stats_logger():
    """Test the stats logger with mock data."""
    print("=" * 80)
    print("Testing Generation Stats Logger")
    print("=" * 80)
    print()

    # Initialize logger
    stats_logger = StatsLogger(log_dir="test_stats")
    print(f"✓ Stats logger initialized: {stats_logger.log_dir}")
    print()

    # Simulate a generation
    request_id = "test_request_001"
    prompt = "What is the capital of France?"

    print("Simulating generation with stats tracking...")
    print(f"Request ID: {request_id}")
    print(f"Prompt: {prompt}")
    print()

    # Start generation
    stats_logger.start_generation(
        request_id=request_id,
        prompt=prompt,
        max_tokens=10,
        temperature=0.7,
    )
    print("✓ Generation started")

    # Simulate encoding
    await asyncio.sleep(0.01)
    stats_logger.log_encoding_start(request_id)
    await asyncio.sleep(0.015)  # Simulate 15ms encoding
    stats_logger.log_encoding_end(request_id, token_count=8)
    print("✓ Encoding complete (8 tokens)")

    # Simulate inference
    stats_logger.log_inference_start(request_id)

    # Simulate token generation
    generated_tokens = [464, 3139, 310, 3444, 338, 3681, 29889]  # "The capital of France is Paris."
    
    for i, token in enumerate(generated_tokens):
        await asyncio.sleep(0.08)  # Simulate 80ms per token
        
        if i == 0:
            stats_logger.log_first_token(request_id)
            print("✓ First token generated (TTFT tracked)")
        
        # Random step time variation
        step_time = 80 + (i % 3) * 2  # 80-84ms
        stats_logger.log_generation_step(request_id, step_time)
        print(f"  Token {i+1}/{len(generated_tokens)}: {token} ({step_time:.1f}ms)")

    stats_logger.log_inference_end(request_id)
    print("✓ Inference complete")
    print()

    # End generation and write stats
    model_info = {
        "model_name": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "total_layers": 30,
        "layers_on_node": 10,
        "mode": "ring",
        "device": "cpu",
        "memory_used_mb": 512.3,
    }

    network_info = {
        "num_nodes": 3,
        "rank": 0,
        "world_size": 3,
        "avg_latency_ms": 5.2,
        "min_latency_ms": 3.1,
        "max_latency_ms": 8.4,
        "layer_window_start": 0,
        "layer_window_end": 9,
    }

    response = "The capital of France is Paris."

    stats_logger.end_generation(
        request_id=request_id,
        response=response,
        generated_token_ids=generated_tokens,
        model_info=model_info,
        network_info=network_info,
    )

    # Verify file was created
    stats_files = list(Path("test_stats").glob("gen_*.json"))
    if stats_files:
        latest_file = max(stats_files, key=lambda p: p.stat().st_mtime)
        print()
        print(f"✓ Stats file created: {latest_file}")
        print()
        
        # Load and display summary
        with open(latest_file) as f:
            data = json.load(f)
        
        print("📊 Stats Summary:")
        print(f"  Prompt tokens: {data['tokens']['prompt_tokens']}")
        print(f"  Generated tokens: {data['tokens']['generated_tokens']}")
        print(f"  Total time: {data['timing']['total_time_s']:.3f}s")
        print(f"  Tokens/sec: {data['tokens']['tokens_per_second']:.2f}")
        print(f"  TTFT: {data['tokens']['time_to_first_token_ms']:.1f}ms")
        print(f"  Avg time/token: {data['tokens']['avg_time_per_token_ms']:.1f}ms")
        if data.get('network'):
            print(f"  Nodes: {data['network']['num_nodes']}")
            print(f"  Rank: {data['network']['node_rank']}")
        print()
        print("✓ All checks passed!")
    else:
        print("❌ No stats file found!")
        return False

    return True


async def test_error_handling():
    """Test stats logging with errors."""
    print()
    print("=" * 80)
    print("Testing Error Handling")
    print("=" * 80)
    print()

    stats_logger = StatsLogger(log_dir="test_stats")
    request_id = "test_error_001"

    stats_logger.start_generation(
        request_id=request_id,
        prompt="This will fail",
    )

    # Simulate error
    stats_logger.end_generation(
        request_id=request_id,
        response="",
        generated_token_ids=[],
        model_info={"model_name": "test", "mode": "test"},
        error="Simulated error for testing",
    )

    # Check error was logged
    stats_files = list(Path("test_stats").glob("gen_*.json"))
    latest_file = max(stats_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_file) as f:
        data = json.load(f)
    
    if data.get("error") and not data.get("success"):
        print("✓ Error correctly logged in stats")
        print(f"  Error message: {data['error']}")
        return True
    else:
        print("❌ Error not properly logged")
        return False


async def main():
    """Run all tests."""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "Generation Stats Logger Test" + " " * 30 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    success = True

    # Test normal generation
    if not await test_stats_logger():
        success = False

    # Test error handling
    if not await test_error_handling():
        success = False

    print()
    print("=" * 80)
    if success:
        print("✅ All tests passed!")
        print()
        print("Stats files are in: test_stats/")
        print("You can inspect them with: cat test_stats/gen_*.json | jq")
    else:
        print("❌ Some tests failed!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    asyncio.run(main())
