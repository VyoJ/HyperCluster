# Transformers v5 Migration Guide for HyperCluster

## Overview

This document summarizes the Transformers v5 changes relevant to HyperCluster and the actions taken to ensure compatibility.

## Breaking Changes Applied

### 1. KV Cache API Changes (CRITICAL)

**v4 API:**
```python
# DynamicCache had key_cache and value_cache as list attributes
cache.key_cache[layer_idx]  # torch.Tensor
cache.value_cache[layer_idx]  # torch.Tensor
```

**v5 API:**
```python
# Cache now uses .layers list where each layer is a CacheLayerMixin object
cache.layers[layer_idx].keys  # torch.Tensor
cache.layers[layer_idx].values  # torch.Tensor

# Preferred: use get_seq_length() method (works for both v4 and v5)
cache.get_seq_length()
```

**Files Updated:**
- [transformers_inference.py](../transformers_inference.py) - Added compatibility helper functions:
  - `_get_cache_seq_length()` - Handles both v4 and v5 cache APIs
  - `_get_cache_num_layers()` - Gets layer count from either API
  - `_get_cache_layer_key_shape()` - Gets key tensor shape for a layer
- [ring_pipeline.py](../ring_pipeline.py) - Updated cache inspection to use new helpers

### 2. Dependency Updates

**pyproject.toml Changes:**
```toml
# Before
transformers = ">=4.41.2"
accelerate = ">=0.30.0"

# After
transformers = ">=5.0.0"
accelerate = ">=1.1.0"  # v5 requires accelerate>=1.1.0
huggingface_hub = ">=1.0.0"  # New requirement for v5
```

## Already Compatible (No Changes Needed)

### 1. `apply_chat_template` Usage

Both uses in HyperCluster pass `tokenize=False`:
- [transformers_inference.py](../transformers_inference.py#L176) - Returns string ✅
- [llm_service.py](../llm_service.py#L375) - Returns string ✅

**Note:** If `tokenize=True` (or omitted in v5), `apply_chat_template` now returns `BatchEncoding` instead of `list[int]`.

### 2. `use_auth_token` Parameter

No usage found in HyperCluster source files. v5 renamed this to `token`.

### 3. Tokenization Backend

HyperCluster uses `use_fast=False` which continues to work in v5. The new unified `TokenizersBackend` is the default for `use_fast=True`.

---

## Future Improvements (Optional)

### 1. WeightConverter API for Dynamic Weight Loading

Transformers v5 introduces `WeightConverter` API for transforming weights during checkpoint loading. This could potentially improve HyperCluster's shard loading:

**Current Approach (sharded_model.py):**
```python
# Manual layer extraction from loaded model
layers = []
for i in range(start_layer, end_layer + 1):
    layers.append(base_model.layers[i])
```

**Potential v5 Approach:**
```python
from transformers.modeling_utils import WeightConverter, ConversionOps

# Define weight patterns for specific layers
converter = WeightConverter(
    source_patterns=["model.layers.{layer_idx}.*"],
    target_patterns=["layers.{new_idx}.*"],
    operations=[
        ConversionOps.Chunk(dim=0, chunks=num_shards, chunk_idx=shard_idx)
    ]
)
```

**Benefit:** Could enable loading only the required layer weights from disk, reducing memory footprint and load time.

### 2. Model-Defined Default Cache Class

In v5, models can define their preferred cache class via `_cache_class` attribute. HyperCluster currently creates `DynamicCache` explicitly - consider checking if the model has a preferred cache class.

### 3. New Cache Types

v5 introduces several cache optimizations:
- `StaticCache` - For torch.compile() and torch.export() support
- `QuantizedCache` - For KIVI-style quantized KV caches
- `OffloadedCache` - For CPU offloading of cache layers
- `DynamicSlidingWindowLayer` - For sliding window attention models

Consider evaluating these for distributed inference scenarios.

### 4. Tokenizer Backend Update

Consider removing `use_fast=False` to use the new unified `TokenizersBackend`:
```python
# Current
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

# Consider
tokenizer = AutoTokenizer.from_pretrained(model_id)  # Uses unified backend
```

---

## v5 Features Not Yet Leveraged

1. **PyTorch 2.7 Stream Support** - Cache offloading uses `torch.Stream()` for async prefetching
2. **Torch Export Support** - `early_initialization()` for pre-allocating cache tensors
3. **Hybrid Cache Structures** - Automatic detection of sliding/chunked attention from model config
4. **Shared Layers Support** - Handling models with `num_kv_shared_layers` (e.g., Gemma3n)

---

## Testing Recommendations

1. Test with both v4-style and v5-style cache objects
2. Verify KV cache state persistence across generation steps
3. Test with models that define `_cache_class` attribute
4. Benchmark memory usage with v5 cache optimizations

---

## References

- [Transformers v5 Migration Guide](https://huggingface.co/docs/transformers/main/en/migration-guide-v5)
- [MIGRATION_GUIDE_V5.md](../../transformers/MIGRATION_GUIDE_V5.md) - Full changelog
- [cache_utils.py](../../transformers/src/transformers/cache_utils.py) - v5 cache implementation
- [core_model_loading.py](../../transformers/src/transformers/integrations/core_model_loading.py) - WeightConverter API
