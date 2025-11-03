# Tensor Synchronization Fix - November 3, 2025

## Problem Summary

The distributed inference system was experiencing **tensor transmission failures** where tensors sent from one node weren't being received properly at the destination node, causing the ring pipeline to timeout and fail.

### Root Causes Identified

1. **Missing Critical Metadata**: `position_ids` and `attention_mask` were being sent in messages but **NOT extracted or passed** to the ring coordinator, breaking RoPE embeddings and attention mechanisms.

2. **Race Condition in Tensor Sync**: Iroh's blob synchronization has latency (~5+ seconds for large tensors), but the metadata message arrives almost instantly, creating a race where:
   - Receiver gets metadata message at T+0s
   - Receiver starts looking for tensor in cache
   - **Receiver times out at T+5s** (cache empty)
   - **Tensor blob FINALLY arrives at T+6s** (too late!)

3. **Fragile Cache Matching**: Originally used tensor size as cache key, which could cause collisions if multiple tensors of same size arrived.

## Solutions Implemented

### 1. Extract and Pass Position Metadata (CRITICAL)

**File**: `node.py`, `handle_ring_tensor_message()`

```python
# Extract position_ids and attention_mask from payload
position_ids_list = payload.get("position_ids")
attention_mask_list = payload.get("attention_mask")

position_ids = np.array(position_ids_list, dtype=np.int64) if position_ids_list is not None else None
attention_mask = np.array(attention_mask_list, dtype=np.bool_) if attention_mask_list is not None else None

# Pass to ring coordinator
await llm_service.ring_coordinator.handle_incoming_tensor(
    sender_id=sender_id,
    request_id=request_id,
    tensor_data=tensor,
    shard=llm_service.current_shard,
    is_final=is_final,
    position_ids=position_ids,  # CRITICAL
    attention_mask=attention_mask  # CRITICAL
)
```

**Why This Matters**: Without position_ids, transformer models cannot compute RoPE (Rotary Position Embeddings), breaking attention completely. This is analogous to prima.cpp's `inp_pos` synchronization.

### 2. Hash-Based Tensor Identification

**File**: `ring_pipeline.py`, `_send_to_node()`

```python
# Store tensor and get its hash
tensor_hash = await doc.set_bytes(author, tensor_key, tensor_bytes)
logger.info(f"   📍 Tensor blob hash: {str(tensor_hash)[:16]}...")

# Include hash in metadata message
message = {
    "type": "ring_tensor_forward",
    "payload": {
        "tensor_hash": str(tensor_hash),  # NEW: Direct hash reference
        "tensor_shape": list(data.shape),
        # ... other metadata
    }
}
```

### 3. Deterministic Cache Lookup

**File**: `node.py`, `handle_ring_tensor_message()`

**Before** (fragile, size-based matching):
```python
# BAD: Match by size (collisions possible)
if tensor_size in self.tensor_cache:
    content_hash_str = self.tensor_cache[tensor_size]
    # Read from blobs...
```

**After** (exact hash matching):
```python
# GOOD: Wait for exact hash
expected_hash = tensor_hash_str  # From message payload
max_wait = 15.0  # Increased timeout

while time.time() - wait_start < max_wait:
    if expected_hash in self.tensor_cache:
        tensor_bytes = self.tensor_cache[expected_hash]
        del self.tensor_cache[expected_hash]
        break
    await asyncio.sleep(0.2)
```

### 4. Improved Cache Management

**File**: `node.py`

**Before**:
```python
self.tensor_cache: Dict[int, str] = {}  # size -> hash (collisions!)
```

**After**:
```python
self.tensor_cache: Dict[str, bytes] = {}  # hash -> content (no collisions)
```

Tensors are now cached directly by their content hash as soon as `CONTENT_READY` event fires:

```python
async def handle_content_ready(self, doc, content_hash):
    hash_str = str(content_hash)
    content = await self.iroh_node.blobs().read_to_bytes(content_hash)
    
    try:
        message_data = json.loads(content.decode("utf-8"))
        # Handle JSON messages...
    except (UnicodeDecodeError, json.JSONDecodeError):
        # Binary tensor data - cache by hash
        self.tensor_cache[hash_str] = content
```

## Testing

Test the fix by running two nodes:

**Node 1 (HEAD)**:
```bash
python main.py --ring
# Share the ticket shown
```

**Node 2 (WORKER)**:
```bash
python main.py --ring --join <ticket>
```

**Then on HEAD node**:
```
llm start
llm query What are 5 popular programming languages?
```

Expected behavior:
- ✅ Tensor arrives within 15s window
- ✅ Position_ids and attention_mask properly forwarded
- ✅ Ring completes successfully
- ✅ Text generation works

## Performance Impact

- **Increased timeout**: 5s → 15s (allows for network latency)
- **No performance penalty**: Cache lookup is O(1) hash-based
- **Reliability**: Eliminates race conditions and cache collisions

## Related Files Modified

1. `node.py` - Tensor cache management and reception logic
2. `ring_pipeline.py` - Tensor hash inclusion in messages
3. `docs/TENSOR_SYNC_FIX_2025_11_03.md` - This document

## References

- Based on prima.cpp's ring communication pattern
- Iroh FFI documentation: `iroh-ffi/src/doc.rs`, `iroh-ffi/src/blob.rs`
- Previous fixes: `docs/IROH_LARGE_MESSAGE_WORKAROUND.md`

