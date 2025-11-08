# Generation Statistics Logging

## Overview

HyperCluster now automatically logs detailed statistics for each generation to individual JSON files. This includes performance metrics, token counts, timing breakdowns, and network information.

## Features

Each generation creates a stats file in the `stats/` directory with the following information:

### 📝 Content Tracking
- **Prompt**: The input query/prompt
- **Response**: The generated text
- **Full Text**: Combined prompt + response

### 🔢 Token Metrics
- **Prompt Tokens**: Number of input tokens
- **Generated Tokens**: Number of tokens generated
- **Total Tokens**: Combined count
- **Tokens per Second (TPS)**: Overall generation throughput
- **Token IDs**: List of all generated token IDs

### ⚡ Performance Metrics
- **Time to First Token (TTFT)**: Latency until first token is generated
- **Average Time per Token**: Mean generation time per token (excluding TTFT)
- **Step Times**: Individual timing for each token generation step

### ⏱️ Timing Breakdown
- **Total Time**: End-to-end generation time
- **Encoding Time**: Time spent tokenizing the prompt
- **Inference Time**: Time spent in model inference
- **Decoding Time**: Time spent decoding tokens to text
- **Network Wait Time**: Time spent waiting for network operations

### 🖥️ Model Information
- **Model Name**: The model being used
- **Total Layers**: Total layers in the model
- **Layers on Node**: Layers assigned to this node (in distributed mode)
- **Inference Mode**: `single`, `sharded`, or `ring`
- **Device**: CPU or CUDA
- **Memory Usage**: Memory consumption metrics (when available)

### 🌐 Network Metrics (Distributed Mode)
- **Number of Nodes**: Total nodes in the cluster
- **Node Rank**: This node's position in the ring/cluster
- **World Size**: Total number of participating nodes
- **Network Latency**: Average, min, and max network latency
- **Data Transfer**: Bytes sent and received
- **Ring Cycles**: Number of cycles through the ring (ring mode)
- **Layer Window**: Start and end layers for this node

## File Format

Stats files are saved as JSON with the naming pattern:
```
stats/gen_YYYYMMDD_HHMMSS_<request_id>.json
```

Example filename:
```
stats/gen_20251108_143052_a1b2c3d4.json
```

## Example Stats File

```json
{
  "request_id": "a1b2c3d4-5678-90ab-cdef-1234567890ab",
  "timestamp": "2025-11-08T14:30:52.123456",
  "query_id": "query_123",
  "prompt": "What is the capital of France?",
  "response": "The capital of France is Paris.",
  "full_text": "What is the capital of France? The capital of France is Paris.",
  "tokens": {
    "prompt_tokens": 8,
    "generated_tokens": 8,
    "total_tokens": 16,
    "tokens_per_second": 12.5,
    "time_to_first_token_ms": 145.3,
    "avg_time_per_token_ms": 78.2,
    "token_ids": [464, 3139, 310, 3444, 338, 3681, 29889]
  },
  "timing": {
    "total_time_s": 0.640,
    "encoding_time_ms": 12.4,
    "inference_time_ms": 620.1,
    "network_wait_time_ms": 23.5,
    "step_times_ms": [145.3, 78.1, 77.9, 79.2, 78.5, 77.8, 78.9, 78.4]
  },
  "model": {
    "model_name": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "total_layers": 30,
    "layers_on_node": 10,
    "inference_mode": "ring",
    "device": "cpu",
    "memory_used_mb": 512.3
  },
  "network": {
    "num_nodes": 3,
    "node_rank": 0,
    "world_size": 3,
    "avg_latency_ms": 5.2,
    "min_latency_ms": 3.1,
    "max_latency_ms": 8.4,
    "layer_window_start": 0,
    "layer_window_end": 9,
    "ring_cycles": 3
  },
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 50,
  "success": true
}
```

## Usage

Stats logging is **automatically enabled** for all generation modes:

### Ring Pipeline Mode
```python
# Stats are logged automatically
llm_service = LLMService(network, use_sharding=True, use_ring=True)
await llm_service.start()
# Generate text - stats will be saved to stats/
```

### Sharded Mode
```python
llm_service = LLMService(network, use_sharding=True, use_ring=False)
await llm_service.start()
# Stats automatically logged
```

### Single-Node Mode
```python
llm_service = LLMService(network, use_sharding=False)
await llm_service.start()
# Stats automatically logged
```

## Accessing Stats Programmatically

You can also access stats during generation:

```python
from stats_logger import get_stats_logger

stats_logger = get_stats_logger()

# Get stats for an active generation
current_stats = stats_logger.get_generation_stats(request_id)
if current_stats:
    print(f"Tokens generated so far: {len(current_stats['step_times'])}")
```

## Analysis Examples

### Calculate Average TPS Across Multiple Runs

```python
import json
from pathlib import Path

stats_files = Path("stats").glob("gen_*.json")
total_tps = 0
count = 0

for file in stats_files:
    with open(file) as f:
        data = json.load(f)
        if data["success"]:
            total_tps += data["tokens"]["tokens_per_second"]
            count += 1

avg_tps = total_tps / count if count > 0 else 0
print(f"Average TPS across {count} generations: {avg_tps:.2f}")
```

### Find Slowest Generation Steps

```python
import json
from pathlib import Path

# Load the most recent stats file
latest_file = max(Path("stats").glob("gen_*.json"), key=lambda p: p.stat().st_mtime)

with open(latest_file) as f:
    data = json.load(f)
    step_times = data["timing"]["step_times_ms"]
    
    slowest_step = max(enumerate(step_times), key=lambda x: x[1])
    print(f"Slowest step: #{slowest_step[0]} took {slowest_step[1]:.1f}ms")
```

### Compare Performance Across Modes

```python
import json
from pathlib import Path
from collections import defaultdict

stats_by_mode = defaultdict(list)

for file in Path("stats").glob("gen_*.json"):
    with open(file) as f:
        data = json.load(f)
        if data["success"]:
            mode = data["model"]["inference_mode"]
            tps = data["tokens"]["tokens_per_second"]
            stats_by_mode[mode].append(tps)

for mode, tps_list in stats_by_mode.items():
    avg = sum(tps_list) / len(tps_list)
    print(f"{mode}: {avg:.2f} TPS (n={len(tps_list)})")
```

## Performance Monitoring

Stats logs are useful for:

1. **Benchmarking**: Compare performance across different models, modes, and configurations
2. **Debugging**: Identify bottlenecks in encoding, inference, or network operations
3. **Optimization**: Track improvements after code changes
4. **Research**: Analyze scaling behavior with different numbers of nodes
5. **Production Monitoring**: Track performance metrics in deployed systems

## Log Rotation

The stats directory can grow over time. You may want to implement log rotation:

```bash
# Keep only last 100 stats files
cd stats
ls -t gen_*.json | tail -n +101 | xargs rm -f
```

Or archive old stats:

```bash
# Archive stats older than 7 days
find stats -name "gen_*.json" -mtime +7 -exec mv {} stats/archive/ \;
```

## Configuration

Stats logging is enabled by default. The stats directory is created automatically in the workspace root.

To customize the stats directory location, modify the initialization in `llm_service.py`:

```python
# Use custom directory
self.stats_logger = get_stats_logger(log_dir="my_custom_stats_dir")
```

## Console Output

In addition to JSON files, a summary is printed to the console after each generation:

```
================================================================================
📊 GENERATION STATISTICS
================================================================================
Request ID: a1b2c3d4-5678-90ab-cdef-1234567890ab
Model: HuggingFaceTB/SmolLM2-135M-Instruct
Mode: ring
Nodes: 3
Rank: 0

📝 Content:
  Prompt: What is the capital of France?...
  Response: The capital of France is Paris....

🔢 Tokens:
  Prompt tokens: 8
  Generated tokens: 8
  Total tokens: 16

⚡ Performance:
  Total time: 0.640s
  Tokens/sec: 12.50
  TTFT: 145.3ms
  Avg time/token: 78.2ms
  Network latency: 5.2ms
================================================================================
```

## Troubleshooting

**Issue**: Stats files not being created

**Solution**: 
- Check that the `stats/` directory exists and is writable
- Verify that generation completes successfully
- Check logs for any errors from the stats logger

**Issue**: Missing network metrics

**Solution**: Network metrics are only available in distributed modes (sharded/ring). Single-node mode will have minimal network info.

**Issue**: Token IDs are empty

**Solution**: Single-node mode doesn't currently track individual token IDs. Use ring or sharded mode for full token tracking.

## See Also

- [Ring Pipeline Documentation](RING_QUICKSTART.md)
- [Sharded Inference Guide](SHARDED_INFERENCE_GUIDE.md)
- [Development Guide](DEVELOPMENT_GUIDE.md)
