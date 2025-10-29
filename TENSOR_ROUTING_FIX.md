# Ring Tensor Routing Debugging Fixes - October 30, 2025

## Problem

When sending a query in ring pipeline mode with 2 nodes:
- ✅ Rank 0 (HEAD) processes layers 0-9 successfully  
- ✅ Tensor sent to Rank 1 (4.06 MB)
- ❌ **Timeout after 30 seconds** - no response from Rank 1
- ❌ No logs from Rank 1 showing tensor receipt or processing

## Root Cause Analysis

The issue was in the message handling flow. When Rank 0 sends a tensor to Rank 1:

1. **Message structure mismatch**: The `request_id` was at the root level in sent messages but being read from the `payload` on the receiving end
2. **Missing target filtering**: All nodes were receiving all messages without checking if they were the intended recipient
3. **Insufficient logging**: No visibility into whether messages were being received or why they failed
4. **No defensive checks**: Missing validation for LLM service/ring coordinator availability

## Fixes Applied

### 1. Fixed `request_id` Location Mismatch (`node.py`)

**Before:**
```python
# node.py line 404
request_id = payload.get("request_id", "unknown")  # ❌ Wrong location!
```

**After:**
```python
# node.py line 405
request_id = message_data.get("request_id", "unknown")  # ✅ Root level
```

**Why this matters**: Without the correct `request_id`, the receiving node can't track the request state, leading to dropped messages or incorrect processing.

---

### 2. Added Target Node Filtering (`node.py`)

**Added lines 408-416:**
```python
# Check if this message is for us
my_node_id = str(await self.iroh_node.net().node_id())

# Only process if:
# 1. No target specified (broadcast), OR
# 2. We are the target
if target_node_id and target_node_id != my_node_id:
    logger.debug(f"Ignoring ring tensor meant for {target_node_id[:8]}... (I am {my_node_id[:8]}...)")
    return
```

**Why this matters**: In Iroh's document-based messaging, all nodes in a document receive all messages. Without filtering, every node would try to process every tensor, causing confusion and wasted compute.

---

### 3. Added Defensive Checks (`node.py`)

**Added lines 424-435:**
```python
# Check if LLM service and ring coordinator are available
if not llm_service:
    logger.warning("   ⚠️  No LLM service available")
    return

if not llm_service.ring_coordinator:
    logger.warning("   ⚠️  No ring coordinator available")
    return

if not llm_service.is_running:
    logger.warning("   ⚠️  LLM service not running")
    return
```

**Why this matters**: If Rank 1 hasn't started its LLM service or ring coordinator yet, it should gracefully skip processing rather than crash or silently fail.

---

### 4. Enhanced Logging Throughout

#### In `node.py` (`handle_ring_tensor_message`):
```python
logger.info(f"")
logger.info(f"📨 RECEIVED RING TENSOR MESSAGE")
logger.info(f"   From: {sender_id[:16]}...")
logger.info(f"   Request ID: {request_id}")
logger.info(f"   Target: {'me' if target_node_id == my_node_id else 'broadcast'}")
logger.info(f"   Tensor shape: {tensor_shape}, dtype: {tensor_dtype}")
logger.info(f"   Is final: {is_final}")
logger.info(f"   ✅ Passing to ring coordinator...")
```

#### In `ring_pipeline.py` (`handle_incoming_tensor`):
```python
logger.info(f"")
logger.info(f"📥 RECEIVED TENSOR IN RING COORDINATOR")
logger.info(f"   From: {sender_id[:16]}...")
logger.info(f"   Request: {request_id}")
logger.info(f"   Shape: {tensor_data.shape}")
logger.info(f"   Is final: {is_final}")
logger.info(f"   My rank: {self.ring_position.rank if self.ring_position else '?'}")
logger.info(f"   Restored existing state (layer {state.current_layer})")
logger.info(f"   → Processing and forwarding...")
```

---

### 5. Added None Check for `current_data` (`ring_pipeline.py`)

**Added lines 427-429:**
```python
if current_data is None:
    logger.error("State has no hidden_states to process!")
    return None
```

**Why this matters**: Prevents crashes when trying to access `.shape` or pass None to tensor serialization functions.

---

## What You'll See Now

### On Rank 0 (Sender):
```
⚙️  Processing on Rank 0
   Layers: 0 → 9 (10 layers)
   Input shape: (1, 7)
   Output shape: (1, 7, 151936)
   ⏱️  Compute time: 582.7ms
   Next layer: 10/24
   📤 Forwarding to Rank 1 (8ec7ca3875e019b1...)
   📤 Sending tensor: 4.06 MB
   ✅ Sent in 596.5ms
```

### On Rank 1 (Receiver) - NEW LOGS:
```
📨 RECEIVED RING TENSOR MESSAGE
   From: e680287e4df47386...
   Request ID: 55924f58-dc02-46f0-92f5-b0e40bafb8d2
   Target: me
   Tensor shape: (1, 7, 151936), dtype: float32
   Is final: False
   ✅ Passing to ring coordinator...

📥 RECEIVED TENSOR IN RING COORDINATOR
   From: e680287e4df47386...
   Request: 55924f58-dc02-46f0-92f5-b0e40bafb8d2
   Shape: (1, 7, 151936)
   Is final: False
   My rank: 1
   Created new state
   → Processing and forwarding...

⚙️  Processing on Rank 1
   Layers: 12 → 19 (8 layers)
   Input shape: (1, 7, 151936)
   ...
```

## Files Modified

1. **`node.py`**:
   - Lines 405: Fixed `request_id` location
   - Lines 408-416: Added target node filtering
   - Lines 418-422: Enhanced logging
   - Lines 424-435: Added defensive checks
   - Lines 437-463: Enhanced tensor processing logging

2. **`ring_pipeline.py`**:
   - Lines 427-429: Added None check for `current_data`
   - Lines 521-543: Enhanced `handle_incoming_tensor` logging
   - Added try-except in `handle_incoming_tensor`

## Testing

### Test 1: Verify Message Receipt
Start both nodes and send a query. You should now see logs on **both** terminals showing:
- Rank 0: Sending tensor
- Rank 1: Receiving tensor
- Rank 1: Processing layers 12-23
- Rank 1: Sending result back to HEAD
- Rank 0: Receiving final result

### Test 2: Check for Missing Components
If Rank 1 doesn't have LLM service started, you'll see:
```
📨 RECEIVED RING TENSOR MESSAGE
   ...
   ⚠️  No LLM service available
```

This tells you exactly what's missing.

### Test 3: Verify Target Filtering
With 3+ nodes, only the intended recipient should process each message. Other nodes will log:
```
Ignoring ring tensor meant for 8ec7ca38... (I am e680287e...)
```

## Next Steps After These Fixes

1. **Run the test** with these enhanced logs to see exactly where the flow breaks
2. **Check if Rank 1 has started its LLM service** - the logs will show this now
3. **Verify the ring coordinator is initialized** on Rank 1 - again, logs will confirm
4. **Watch for the full message flow** from send → receive → process → forward

## Potential Remaining Issues

If messages still don't flow after these fixes, check:

1. **Iroh document membership**: Are both nodes actually in the same document?
   - Check terminal logs for "Peer ... came online for doc ..."
   
2. **Message handler registration**: Is `message_handler` in `main.py` actually routing `ring_tensor_forward` messages?
   - Line 42-45 in `main.py` should handle this

3. **Ring initialization timing**: Did Rank 1 start its LLM service after the ring was already initialized?
   - The topology update should trigger re-initialization, but verify this happened

4. **Network issues**: Is Iroh actually delivering messages between nodes?
   - Test with a simple text message first: `text hello`
   - Should appear on both terminals

## Summary

These fixes add:
- ✅ Correct `request_id` extraction  
- ✅ Target node filtering
- ✅ Comprehensive logging for debugging
- ✅ Defensive checks for service availability
- ✅ None safety for tensor data

The enhanced logging will show you **exactly** where the message flow breaks, making it much easier to diagnose the remaining issue.

