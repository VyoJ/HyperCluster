# Sharded Inference Quick Reference

## Core Concept

Instead of loading the **entire model** on each node, we:
1. Load the full model
2. Extract only the **layers assigned to this shard**
3. Execute forward pass through **only those layers**

## The TransformersShard Wrapper

### What it does
```python
from sharded_model import TransformersShard

# Takes full model + shard spec
wrapper = TransformersShard(full_model, shard)

# Extracts components:
wrapper.embed_tokens  # Only if shard.is_first_layer()
wrapper.layers        # Only layers in [start_layer, end_layer]
wrapper.norm          # Only if shard.is_last_layer()
wrapper.lm_head       # Only if shard.is_last_layer()
```

### Forward pass behavior
```python
# First shard (layers 0-7)
output = wrapper(input_ids=tokens)
# → embeddings → layers 0-7 → hidden_states (NO logits)

# Middle shard (layers 8-15)
output = wrapper(inputs_embeds=hidden_states)
# → layers 8-15 → hidden_states (NO logits)

# Last shard (layers 16-23)
output = wrapper(inputs_embeds=hidden_states)
# → layers 16-23 → norm → lm_head → logits
```

## Quick Examples

### Load a single shard
```python
from shard import Shard
from sharded_model import load_sharded_model

shard = Shard(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    start_layer=0,
    end_layer=7,
    n_layers=24
)

model = load_sharded_model(
    "Qwen/Qwen2.5-0.5B-Instruct",
    shard,
    device_map="auto"
)
```

### Distributed inference (3 nodes)
```python
from transformers_inference import TransformersShardedInferenceEngine

# Setup
shards = [
    Shard("model_id", 0, 7, 24),
    Shard("model_id", 8, 15, 24),
    Shard("model_id", 16, 23, 24),
]
engines = [TransformersShardedInferenceEngine() for _ in range(3)]

# Inference
tokens = await engines[0].encode(shards[0], "Hello")

hidden1, state1 = await engines[0].infer_tensor("req", shards[0], tokens)
hidden2, state2 = await engines[1].infer_tensor("req", shards[1], hidden1, state1)
logits, state3 = await engines[2].infer_tensor("req", shards[2], hidden2, state2)

next_token = await engines[2].sample(logits)
text = await engines[2].decode(shards[2], next_token)
```

## Data Flow

```
User Input: "What is AI?"
     ↓
┌────────────────────────────────────────┐
│ Node 1 (First Shard)                   │
│                                        │
│ 1. Encode: "What is AI?" → [123, 456] │
│ 2. Embed: [123, 456] → [[0.1, ...]]   │
│ 3. Layers 0-7: hidden_states_1         │
│    Shape: (1, 2, 896)                  │
└────────────────────────────────────────┘
     ↓ Send 1×2×896 = 7KB
┌────────────────────────────────────────┐
│ Node 2 (Middle Shard)                  │
│                                        │
│ 4. Receive: hidden_states_1            │
│ 5. Layers 8-15: hidden_states_2        │
│    Shape: (1, 2, 896)                  │
└────────────────────────────────────────┘
     ↓ Send 1×2×896 = 7KB
┌────────────────────────────────────────┐
│ Node 3 (Last Shard)                    │
│                                        │
│ 6. Receive: hidden_states_2            │
│ 7. Layers 16-23: hidden_states_3       │
│ 8. Norm: normalized_states             │
│ 9. LM Head: logits                     │
│    Shape: (1, 2, 151936)               │
│ 10. Sample: next_token = 42            │
│ 11. Decode: "Artificial"               │
└────────────────────────────────────────┘
     ↓
Output: "Artificial"
```

## Shard Types

### First Shard (start_layer = 0)
```python
Components:
  ✓ embed_tokens    # Token → embeddings
  ✓ layers[0:N]     # Process embeddings
  ✗ norm            # Not yet
  ✗ lm_head         # Not yet

Input:  token_ids (int64)
Output: hidden_states (float32)
```

### Middle Shard (0 < start_layer < n_layers-1)
```python
Components:
  ✗ embed_tokens    # Already embedded
  ✓ layers[N:M]     # Process hidden states
  ✗ norm            # Not yet
  ✗ lm_head         # Not yet

Input:  hidden_states (float32)
Output: hidden_states (float32)
```

### Last Shard (end_layer = n_layers-1)
```python
Components:
  ✗ embed_tokens    # Already embedded
  ✓ layers[M:24]    # Final layers
  ✓ norm            # Final normalization
  ✓ lm_head         # Project to vocabulary

Input:  hidden_states (float32)
Output: logits (float32)
```

## Common Patterns

### Pattern 1: 2-node split
```python
# 50-50 split
shards = [
    Shard(model_id, 0, 11, 24),   # First half + embeddings
    Shard(model_id, 12, 23, 24),  # Second half + lm_head
]
```

### Pattern 2: 3-node split
```python
# Even thirds
shards = [
    Shard(model_id, 0, 7, 24),    # First third
    Shard(model_id, 8, 15, 24),   # Middle third
    Shard(model_id, 16, 23, 24),  # Last third
]
```

### Pattern 3: Memory-weighted split
```python
# First/last shards larger (have embed/lm_head)
shards = [
    Shard(model_id, 0, 9, 24),    # 10 layers + embeddings
    Shard(model_id, 10, 13, 24),  # 4 layers only
    Shard(model_id, 14, 23, 24),  # 10 layers + lm_head
]
```

## Troubleshooting

### Issue: "Cannot find layers"
```python
# Model uses different attribute name
# Check model structure:
print(model.model.__dict__.keys())
# Add support in sharded_model.py _extract_model_components()
```

### Issue: "Hidden size mismatch"
```python
# Passing wrong tensor between shards
# Verify shape:
print(f"Expected: (batch, seq, hidden_size)")
print(f"Got: {tensor.shape}")
```

### Issue: "No embeddings found"
```python
# Model uses different embedding name
# Check:
print(model.model.__dict__.keys())
# Look for: embed_tokens, wte, word_embeddings
```

## Testing Commands

```bash
# Quick smoke test
python -c "
from shard import Shard
from sharded_model import load_sharded_model
s = Shard('Qwen/Qwen2.5-0.5B-Instruct', 0, 7, 24)
m = load_sharded_model('Qwen/Qwen2.5-0.5B-Instruct', s)
print('✓ Loaded successfully')
"

# Comprehensive tests
python test_comprehensive_sharding.py

# Interactive demo
python example_sharded_inference.py
```

## Key Files

```
sharded_model.py              # Core wrapper implementation
transformers_inference.py     # Inference engine using wrapper
test_comprehensive_sharding.py # Full test suite
example_sharded_inference.py  # Interactive demo

docs/
  SHARDED_INFERENCE_GUIDE.md  # Full documentation
  SHARDED_IMPLEMENTATION.md   # Implementation details
  SHARDED_QUICK_REF.md        # This file
```

## Performance Tips

### 1. Batch Processing
```python
# Better: Process multiple prompts together
tokens = encode_batch(["prompt1", "prompt2", "prompt3"])
# Network overhead amortized across batch
```

### 2. Shard Size
```python
# Too small: High network overhead
shards = [Shard(..., 0, 2, 24)]  # Only 3 layers

# Too large: Poor distribution
shards = [Shard(..., 0, 20, 24)] # 21 layers

# Good: Balanced
shards = [Shard(..., 0, 7, 24)]  # 8 layers
```

### 3. Device Placement
```python
# GPU if available
model = load_sharded_model(model_id, shard, device_map="auto")

# Specific GPU
model = load_sharded_model(model_id, shard, device_map={"": 0})

# CPU only
model = load_sharded_model(model_id, shard, device_map="cpu")
```

## Memory Calculation

```python
def estimate_memory(model_size_gb, num_layers, shard_layers):
    """
    Rough estimate of shard memory usage.
    
    Example:
      Model: Qwen2.5-0.5B = 4GB, 24 layers
      Shard: 8 layers
      Memory: ~1.3GB per node
    """
    layer_memory = model_size_gb / num_layers
    shard_memory = layer_memory * shard_layers
    
    # Add overhead for embeddings (first) or lm_head (last)
    if shard.is_first_layer() or shard.is_last_layer():
        shard_memory += layer_memory * 0.5
    
    return shard_memory

# Example: Qwen2.5-0.5B split 3 ways
total_gb = 4
total_layers = 24
layers_per_shard = 8

print(f"Memory per shard: ~{estimate_memory(total_gb, total_layers, layers_per_shard):.1f}GB")
# Output: ~1.3GB (vs 4GB for full model)
```

## See Also

- Full guide: `docs/SHARDED_INFERENCE_GUIDE.md`
- Implementation: `docs/SHARDED_IMPLEMENTATION.md`
- Tests: `test_comprehensive_sharding.py`
- Demo: `example_sharded_inference.py`
