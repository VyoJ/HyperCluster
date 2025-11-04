# KV Cache Debug Logging Guide

## Overview

I've added comprehensive debug logging to track KV cache behavior in multi-node inference. This will help us identify exactly where and why the cache is not being properly maintained across nodes.

---

## Added Debug Sections

### 1. **transformers_inference.py - Cache Load Debug** (Line ~270)

**Location:** In `infer_tensor()` method, when loading cache

**What it logs:**
```
===============================================================================
🔍 KV CACHE DEBUG
===============================================================================
Request ID: {request_id}
My shard: {layer_start}-{layer_end} ({num_layers} layers)
Total model layers: {n_layers}

✅ Cache EXISTS for request {request_id}
   Cache type: DynamicCache
   Number of layers in cache: 28
   Sequence length: 9
   Layer 0 key shape: torch.Size([1, 16, 9, 64])
   Layer 1 key shape: torch.Size([1, 16, 9, 64])
   Layer 2 key shape: torch.Size([1, 16, 9, 64])
   ... and 25 more layers
   
🚨 CRITICAL CHECK:
   ⚠️  CACHE LAYER MISMATCH!
   Expected 18 layers for my shard OR 28 for full model
   Got 28 layers in cache

🔄 Will REUSE cache from previous step
===============================================================================
```

**Key insight:** This shows whether each node has:
- Cache for ALL 28 layers (WRONG - should only have its shard's layers)
- Cache for only its assigned layers (CORRECT)
- No cache (might be correct for step 0, wrong for step 1+)

---

### 2. **transformers_inference.py - Cache Update Debug** (Line ~540)

**Location:** After model forward pass, when updating cache

**What it logs:**
```
===============================================================================
💾 KV CACHE UPDATE
===============================================================================
Cache type: DynamicCache
Number of layers cached: 28
Sequence length: 10
My shard layers: 0-17

⚠️  Cache has ALL model layers (28), not just shard layers (18)
   This might indicate the sharded model wrapper is not filtering correctly!

Layer-by-layer cache inspection:
   Layer 0: K=torch.Size([1, 16, 10, 64]), V=torch.Size([1, 16, 10, 64])
   Layer 1: K=torch.Size([1, 16, 10, 64]), V=torch.Size([1, 16, 10, 64])
   Layer 2: K=torch.Size([1, 16, 10, 64]), V=torch.Size([1, 16, 10, 64])
   ...
   Layer 27: K=torch.Size([1, 16, 10, 64]), V=torch.Size([1, 16, 10, 64])

✅ Cache updated for request {request_id}
===============================================================================
```

**Key insight:** This reveals:
- Whether the model is returning cache for ALL layers or just the shard's layers
- If the sharded model wrapper is working correctly
- Whether cache is growing with each token (seq_len should increase)

---

### 3. **ring_pipeline.py - Pre-Step Cache Check** (Line ~320)

**Location:** At the start of each generation step (HEAD node)

**What it logs:**
```
🔍 PRE-STEP CACHE CHECK
   Request ID: c422a72b-621a-4573-8a0a-daa3d1aba12d
   ✅ Cache exists: 28 layers
   Cache seq_len: 9
```

**Key insight:** Shows HEAD node's cache state before starting each new token generation

---

### 4. **ring_pipeline.py - Worker Node Receive Debug** (Line ~790)

**Location:** When worker node receives tensor from HEAD

**What it logs:**
```
📥 RECEIVED TENSOR IN RING COORDINATOR
   From: 3842e40ae5c034e0...
   Request: c422a72b-621a-4573-8a0a-daa3d1aba12d
   Shape: (1, 1, 1024)
   Is final: False
   My rank: 1
   Has position_ids: True
   Has attention_mask: True
   
   Restored existing state (layer 18, step 1)

🔍 WORKER NODE CACHE CHECK
   ⚠️  NO CACHE found on worker node!
   This is UNEXPECTED for step > 0 in autoregressive generation!
   Worker nodes should maintain cache from previous steps!
```

**Key insight:** This is **THE SMOKING GUN**! If we see this warning, it means:
- Worker node received hidden states for a new token
- But it has NO cache from the previous step
- This means it will recompute attention from scratch (WRONG!)

---

## What to Look For

### Expected Behavior (Correct)

**Step 0 (Prompt):**
```
HEAD (Rank 0):
  - Processes layers 0-17
  - Creates cache for layers 0-17 (18 layers total)
  - Sends hidden states to Worker

Worker (Rank 1):
  - Receives hidden states
  - NO CACHE (first time) ✅
  - Processes layers 18-27
  - Creates cache for layers 18-27 (10 layers total)
  - Sends logits back to HEAD
```

**Step 1+ (Autoregressive):**
```
HEAD (Rank 0):
  - Has cache for layers 0-17, seq_len=9 ✅
  - Processes NEW token through layers 0-17
  - Updates cache to seq_len=10 ✅
  - Sends hidden states to Worker

Worker (Rank 1):
  - Receives hidden states
  - HAS cache for layers 18-27, seq_len=9 ✅
  - Processes NEW token through layers 18-27
  - Updates cache to seq_len=10 ✅
  - Sends logits back to HEAD
```

### Problem Indicators (Bugs)

**🚨 Indicator 1: Cache has ALL layers instead of shard layers**
```
⚠️  Cache has ALL model layers (28), not just shard layers (18)
```
**Meaning:** The sharded model wrapper is not properly filtering cache

**🚨 Indicator 2: Worker has no cache on step 1+**
```
⚠️  NO CACHE found on worker node!
This is UNEXPECTED for step > 0 in autoregressive generation!
```
**Meaning:** Cache is not persisting across generation steps on worker nodes

**🚨 Indicator 3: Cache not growing**
```
Step 0: seq_len=9
Step 1: seq_len=9  ← Should be 10!
```
**Meaning:** Cache is not being updated properly

---

## Running the Debug

### Single Node Test (Should work correctly)
```powershell
# Run single node
python main.py --model Qwen/Qwen3-0.6B

# Look for these patterns:
# - "Cache exists: 28 layers" (correct, single node has all layers)
# - seq_len growing: 9 → 10 → 11 → ...
```

### Multi-Node Test (Currently broken)
```powershell
# Terminal 1: Node 1
python main.py --model Qwen/Qwen3-0.6B --doc-ticket <ticket>

# Terminal 2: Node 2
python main.py --model Qwen/Qwen3-0.6B --doc-ticket <ticket>

# Look for these patterns on Worker node:
# - "NO CACHE found on worker node!" ← THE BUG
# - "Cache has ALL model layers (28), not just shard layers" ← Also a bug
```

---

## Expected Output Analysis

With these logs, we should be able to pinpoint:

1. **Does each node's cache have the correct number of layers?**
   - Head: Should have 18 layers (0-17)
   - Worker: Should have 10 layers (18-27)
   - If both have 28 layers → Bug in sharded_model.py

2. **Does worker node maintain cache across steps?**
   - Step 0: No cache (correct)
   - Step 1+: Should have cache (if missing → BUG!)

3. **Is cache growing properly?**
   - Each step should increase seq_len by 1
   - If seq_len doesn't grow → Not updating cache

---

## Next Steps

1. Run multi-node test with these logs
2. Check the debug output for the indicators above
3. Based on findings:
   - If cache has all layers → Fix sharded_model.py wrapper
   - If worker has no cache on step 1+ → Fix cache persistence
   - If cache not growing → Fix cache update logic

The logs will tell us **exactly** where the problem is! 🎯
