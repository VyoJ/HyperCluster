# Ring Pipeline Logging Guide

## What You'll See

The enhanced logging shows exactly what's happening at each step of the ring pipeline inference. Here's what to expect:

## 1. Ring Initialization

When you start the LLM service with `--ring`, you'll see:

```
================================================================================
🔗 RING TOPOLOGY INITIALIZED
================================================================================
📍 My Position:
   Rank: 0/3
   Role: HEAD (Coordinator)
   Node ID: a1b2c3d4e5f6g7h8...

🔄 Ring Structure:
   Previous: f7e8d9c0b1a2... (rank 2)
   Current:  a1b2c3d4e5f6g7h8... (rank 0)
   Next:     9876543210abcdef... (rank 1)

📊 My Layer Assignment:
   Layers: 0 → 15
   Count:  16 layers
   Total:  48 layers in model

🌍 Full Cluster Distribution:
👉 Rank 0 (HEAD): Layers   0- 15 (16 layers) - a1b2c3d4e5f6g7h8...
   Rank 1 (WORK-1): Layers  16- 31 (16 layers) - 9876543210abcdef...
   Rank 2 (WORK-2): Layers  32- 47 (16 layers) - f7e8d9c0b1a2...
================================================================================
```

**What this tells you:**
- Your rank in the ring (0 = head node)
- Who you receive from (previous) and send to (next)
- Which layers you're responsible for
- The full distribution across all nodes (👉 marks YOUR node)

## 2. Layer Distribution Calculation

Before the topology display, you'll see:

```
📐 Calculating layer distribution...
   Total cluster memory: 16.0 GB
   Rank 0: 4.0 GB (25.0%) → 12 layers
   Rank 1: 8.0 GB (50.0%) → 24 layers
   Rank 2: 4.0 GB (25.0%) → 12 layers
```

**What this tells you:**
- How memory is distributed
- How many layers each node gets based on their memory
- Nodes with more memory get more layers!

## 3. Starting Inference (Head Node)

When you send a query on the head node:

```
================================================================================
🚀 STARTING RING INFERENCE
================================================================================
Request ID: req-1234567890
Prompt: What is a neural network? Explain how it works...
Max tokens: 256
Model layers: 48

📝 Encoding complete:
   Input tokens: 12
   Token shape: (1, 12)
   Encode time: 45.2ms
================================================================================
```

**What this tells you:**
- The request being processed
- How many tokens the prompt encoded to
- Time taken for tokenization

## 4. Token Generation Loop (Each Step)

For each token generated, you'll see:

```
🔄 GENERATION STEP 1/256
   Tokens generated so far: 0

🔁 Ring Forward Pass
   Input shape: (1, 12)
   Total cycles needed: 1
   Total layers: 48
   🎯 Initiating ring from HEAD node...

⚙️  Processing on Rank 0
   Layers: 0 → 15 (16 layers)
   Input shape: (1, 12, 768)
   Output shape: (1, 12, 768)
   ⏱️  Compute time: 234.5ms
   Next layer: 16/48
   📤 Forwarding to Rank 1 (9876543210abcdef...)
   📤 Sending tensor: 0.14 MB
   ✅ Sent in 12.3ms
```

**What this tells you:**
- Which step of generation (1/256 means first token)
- Input tensor shape going into this step
- Which layers this rank processes (0 → 15)
- How long the computation took (234.5ms)
- Tensor size being sent (0.14 MB)
- Network send time (12.3ms)

## 5. Worker Node Processing

On worker nodes, when they receive tensors:

```
📥 RECEIVED TENSOR
   From: a1b2c3d4e5f6g7h8...
   Request: req-1234567890
   Shape: (1, 12, 768)
   Is final: False

⚙️  Processing on Rank 1
   Layers: 16 → 31 (16 layers)
   Input shape: (1, 12, 768)
   Output shape: (1, 12, 768)
   ⏱️  Compute time: 198.7ms
   Next layer: 32/48
   📤 Forwarding to Rank 2 (f7e8d9c0b1a2...)
   📤 Sending tensor: 0.14 MB
   ✅ Sent in 8.9ms
```

**What this tells you:**
- Received tensor from previous node
- Processing assigned layers (16 → 31)
- Forwarding to next node in ring

## 6. Final Layer & Sampling

When the last node completes:

```
⚙️  Processing on Rank 2
   Layers: 32 → 47 (16 layers)
   Input shape: (1, 12, 768)
   Output shape: (1, 12, 768)
   ⏱️  Compute time: 187.3ms
   Next layer: 48/48
   🏁 Final layer reached!
   📤 Worker node: Sending final result to HEAD

📥 RECEIVED TENSOR  (back at HEAD)
   From: f7e8d9c0b1a2...
   Request: req-1234567890
   Shape: (1, 12, 50257)  ← Note: This is logits!
   Is final: True

   ✅ All layers processed in 620.5ms
   ✅ Token 1 sampled: 42
   ⏱️  Step time: 633.2ms
```

**What this tells you:**
- Final layer complete
- Logits received (shape is [1, seq_len, vocab_size])
- Total time for full ring pass
- Which token was sampled

## 7. Generation Complete

After all tokens:

```
================================================================================
✨ GENERATION COMPLETE
   Total tokens: 45
   Total time: 28.49s
   Avg token latency: 633.1ms/token
================================================================================
```

**What this tells you:**
- How many tokens were generated
- Total generation time
- Average latency per token (key metric!)

## Full Example: 3-Node Ring Processing One Token

### Terminal 1 (Head Node - Rank 0)
```
🔄 GENERATION STEP 1/256
⚙️  Processing on Rank 0
   Layers: 0 → 15 (16 layers)
   ⏱️  Compute time: 234.5ms
   📤 Forwarding to Rank 1
```

### Terminal 2 (Worker - Rank 1)
```
📥 RECEIVED TENSOR
   From: a1b2c3d4... (Rank 0)
⚙️  Processing on Rank 1
   Layers: 16 → 31 (16 layers)
   ⏱️  Compute time: 198.7ms
   📤 Forwarding to Rank 2
```

### Terminal 3 (Worker - Rank 2)
```
📥 RECEIVED TENSOR
   From: 9876543210... (Rank 1)
⚙️  Processing on Rank 2
   Layers: 32 → 47 (16 layers)
   ⏱️  Compute time: 187.3ms
   📤 Sending final result to HEAD
```

### Back to Terminal 1
```
📥 RECEIVED TENSOR
   From: f7e8d9c0... (Rank 2)
   Is final: True
✅ Token 1 sampled: 42
```

**Total token latency**: 234.5 + 198.7 + 187.3 + network overhead ≈ **633ms**

## Key Metrics to Watch

### 1. Layer Distribution
```
🌍 Full Cluster Distribution:
```
Look for balanced distribution. Nodes with more memory should get more layers.

### 2. Compute Time
```
⏱️  Compute time: 234.5ms
```
This is the actual model forward pass time on each node.

### 3. Network Transfer Time
```
📤 Sending tensor: 0.14 MB
✅ Sent in 12.3ms
```
Lower is better. This is the communication overhead.

### 4. Token Latency
```
⏱️  Step time: 633.2ms
```
This is total time per token (compute + network + sampling).

## Emoji Guide

- 🔗 Ring topology setup
- 📍 Position information
- 🔄 Ring structure
- 📊 Layer assignments
- 🌍 Full cluster view
- 🚀 Inference starting
- 📝 Encoding
- 🔄 Generation step
- 🔁 Ring forward pass
- ⚙️  Processing/computing
- 📤 Sending data
- 📥 Receiving data
- ✅ Success/complete
- 🏁 Final layer
- ⏱️  Timing information
- 🎯 Target/initiate
- 🛑 Stop condition
- ✨ Complete
- ⚠️  Warning

## Debug Mode

For even more verbose logging, set:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show:
- Every layer check (`this_layer_is_mine`)
- Memory allocations
- Prefetch operations
- Internal state changes

## Performance Analysis

With these logs, you can calculate:

1. **Per-node compute time**: Look for "⏱️ Compute time"
2. **Network overhead**: "✅ Sent in" + message passing delay
3. **Ring latency**: Sum of all node compute times + network
4. **Bottleneck detection**: Which rank has longest compute time?
5. **Throughput**: tokens/second = 1000 / avg_token_latency_ms

## Example Analysis

```
Rank 0: 234.5ms compute
Rank 1: 198.7ms compute  ← Fastest!
Rank 2: 187.3ms compute
Network: ~40ms total

Total: 660ms/token
Throughput: 1.52 tokens/second
Bottleneck: Rank 0 (needs more GPU memory or faster CPU)
```

## Tips

1. **Compare ranks**: If one rank is much slower, it's a bottleneck
2. **Network vs compute**: If "Sent in" times are high, optimize network
3. **Layer balance**: Uneven distribution? Adjust memory reporting
4. **Multi-cycle**: If you see "Total cycles needed: 2", model is too big for ring

## Next Steps

After seeing the logs:
1. Note your token latency (⏱️ Step time)
2. Compare to prima.cpp (~90ms for QwQ-32B)
3. Expected: 2-3x slower (Python overhead is normal)
4. Optimize the slowest rank or add more nodes

