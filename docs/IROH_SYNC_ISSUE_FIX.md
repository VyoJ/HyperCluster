# Iroh Document Sync Issue - Large Message Delivery

## Problem Observed

The HEAD node successfully sends the ring tensor forward message (~786KB), but the WORKER node never receives it:

**HEAD logs:**
```
📤 Writing to doc...: type=ring_tensor_forward, size=786825 bytes
✅ Successfully wrote message to document
✅ Sent in 63.8ms
```

**WORKER logs:**
```
📦 Read 269 bytes from blob
📨 Content ready: type=llm_message
📦 Read 247 bytes from blob  
📨 Content ready: type=llm_message
[NO 786KB message ever received!]
```

## Root Cause

Iroh's document sync is asynchronous. When `doc.set_bytes()` returns, the message is written to the local document, but it hasn't necessarily synced to remote peers yet. The large tensor message (786KB) takes longer to sync than small messages (a few hundred bytes).

The HEAD node was:
1. Writing the large message
2. Immediately waiting for a response
3. Timing out because the worker hadn't received the message yet

## Fixes Applied

### 1. Added Sync Delay for Large Messages

In `node.py` `send_message()`:
```python
await doc.set_bytes(author, key, payload)

# For large messages, add a small delay to allow sync
if payload_size > 100000:  # > 100KB
    logger.info(f"⏳ Waiting 1s for large message to sync...")
    await asyncio.sleep(1.0)
```

This gives Iroh time to sync the large message to peers before the sender starts waiting for a response.

### 2. Increased Timeout from 30s to 60s

In `ring_pipeline.py` `_ring_forward_pass()`:
```python
timeout = 60.0  # seconds - increased for large message sync
```

More time for network latency and processing.

### 3. Enhanced Logging for Large Messages

**Sending (node.py):**
```python
if payload_size > 100000:  # > 100KB
    logger.info(f"📤 Writing LARGE message to doc {doc_id[:16]}...: type={msg_type}, size={payload_size/1024/1024:.2f}MB")
```

**Receiving (node.py):**
```python
if content_size > 100000:  # > 100KB
    logger.info(f"📨 ⚡ LARGE MESSAGE: type={msg_type}, from={sender_short}..., size={content_size/1024/1024:.2f}MB")
```

This makes it easy to spot when large messages are sent/received.

### 4. Progress Logging During Wait

In `ring_pipeline.py`:
```python
# Log progress every 5 seconds
elapsed = time.time() - start_time
if int(elapsed) % 5 == 0 and elapsed > 0:
    logger.info(f"   ⏱️  Still waiting... {elapsed:.0f}s elapsed, current_layer={state.current_layer}/{shard.n_layers}")
```

Shows if we're making progress or truly stuck.

## What to Look For in Logs

### On HEAD Node:
```
📤 Writing LARGE message to doc...: type=ring_tensor_forward, size=0.75MB  ← Should see this
⏳ Waiting 1s for large message to sync...                                  ← New delay
✅ Successfully wrote message to document
⏳ Waiting for ring completion (timeout=60s)...                             ← Longer timeout
```

### On WORKER Node (what we want to see):
```
📦 Read 786825 bytes from blob (hash=...)                                   ← Should match sent size!
📨 ⚡ LARGE MESSAGE: type=ring_tensor_forward, from=..., size=0.75MB        ← Highlights large messages
🔔 Routing ring_tensor_forward to handler
📨 RECEIVED RING TENSOR MESSAGE
✅ Passing to ring coordinator...
```

### Success Indicators:
- ✅ Worker shows "Read 786825 bytes" (matches the ~786KB sent)
- ✅ Worker shows "⚡ LARGE MESSAGE: type=ring_tensor_forward"
- ✅ Worker processes the tensor and sends result back
- ✅ HEAD receives final result within 60 seconds

### Failure Indicators:
- ❌ Worker only receives small messages (269 bytes, 247 bytes, etc.)
- ❌ Worker never shows "⚡ LARGE MESSAGE"
- ❌ "⏱️ Still waiting..." logs appear every 5 seconds
- ❌ Eventually timeout: "⚠️ Timeout waiting for ring completion!"

## Testing

Run the same test again:

**Terminal 1 (HEAD):**
```bash
python main.py start --ring
llm start Qwen/Qwen2.5-0.5B-Instruct
# Copy ticket
```

**Terminal 2 (WORKER):**
```bash
python main.py start --ring --bootstrap-ticket "<TICKET>"
llm start Qwen/Qwen2.5-0.5B-Instruct
```

**Terminal 1:**
```bash
llm query "hello"
```

Watch for:
1. HEAD: "⏳ Waiting 1s for large message to sync..." after sending
2. WORKER: "📨 ⚡ LARGE MESSAGE: type=ring_tensor_forward, size=0.75MB"
3. WORKER: Processing and sending result back
4. HEAD: Completing generation within 60 seconds

## If Still Failing

If the worker still doesn't receive the large message after these changes, the issue is likely:

1. **Network/Firewall**: Large messages blocked by firewall or network issues
2. **Iroh Bug**: Possible bug in Iroh's document sync for large blobs
3. **Memory Pressure**: System running out of memory when handling large blobs

### Workaround: Chunk Large Messages

If Iroh can't reliably sync large messages, we could:
1. Split tensors into smaller chunks (e.g., 100KB each)
2. Send multiple messages with sequence numbers
3. Reassemble on the receiving end

But let's first see if the 1-second sync delay is sufficient.

## Files Modified

- `node.py`: Added asyncio import, sync delay, enhanced logging
- `ring_pipeline.py`: Increased timeout, added progress logging

## Related Issues

- Previous fix: MULTI_NODE_FIXES_APPLIED.md (layer window initialization)
- Previous fix: MULTI_NODE_FIX.md (worker query handling)

---

**Status**: 🧪 Testing required
**Date**: October 30, 2025
**Critical change**: Added 1-second sync delay after writing large messages

