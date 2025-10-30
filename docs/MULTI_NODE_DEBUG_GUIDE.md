# Multi-Node Tensor Communication Debug Guide

## Issue
When running with multiple nodes, LLM queries timeout without generating output. Worker nodes don't show any receiving logs for tensors.

## Diagnostic Changes Made

### 1. Enhanced Logging
Added comprehensive DEBUG-level logging to trace message flow:

#### `main.py`
- **Message handler**: Logs all incoming messages with type and sender
- **Ring tensor routing**: Logs when ring_tensor_forward messages are being routed
- **Debug level**: Enabled DEBUG for `message_handler`, `node`, and `ring_pipeline` modules

#### `node.py`
- **Subscription**: Logs when subscribing to Iroh documents
- **Events**: Logs all Iroh LiveEvents (CONTENT_READY, NEIGHBOR_UP, NEIGHBOR_DOWN)
- **Content ready**: Logs blob reading and message parsing
- **Ring tensor handler**: Logs target checking and message routing decisions

#### `ring_pipeline.py`
- **Tensor sending**: Logs target node, request ID, document ID, and send success/failure

### 2. How to Test

#### Terminal 1 (HEAD Node):
```bash
cd /Users/samarth/Documents/Samarth/prima.cpp/HyperCluster-v0.1
python main.py start --ring

# Wait for node to start, then:
llm start Qwen/Qwen2.5-0.5B-Instruct

# Copy the ticket shown, e.g.:
# Created main document. Share this ticket: <TICKET>
```

#### Terminal 2 (WORKER Node):
```bash
cd /Users/samarth/Documents/Samarth/prima.cpp/HyperCluster-v0.1
python main.py start --ring --bootstrap-ticket "<TICKET_FROM_TERMINAL_1>"

# Wait for node to start, then:
llm start Qwen/Qwen2.5-0.5B-Instruct

# Wait for ring initialization (~5 seconds)
```

#### Terminal 1 (Send Query):
```bash
llm query "hello"
```

### 3. Expected Log Output

#### On HEAD Node (Terminal 1):

```
# Subscription
🔔 Subscribing to events for document <doc_id>...
✅ Subscribed to document <doc_id>...

# Neighbor detection
👋 Neighbor UP: <worker_node_id>

# Topology update
Updated topology: <worker_node_id>... - X.X GB

# Query received
📨 RECEIVED QUERY
   Query: hello...
   ✅ Processing query...
   🔁 Routing to ring pipeline

# Layer processing
⚙️  Processing on Rank 0
   Layers: 0 → X (X layers)
   Input shape: (1, Y)
   Output shape: (1, Y, Z)

# Sending to worker
📤 Forwarding to Rank 1 (<worker_node_id>...)
📤 Sending tensor: X.XX MB
📤 Target: <worker_node_id>...
📤 Request ID: <uuid>
📤 Doc ID: <doc_id>...
✅ Sent in X.Xms
```

#### On WORKER Node (Terminal 2):

```
# Subscription
🔔 Subscribing to events for document <doc_id>...
✅ Subscribed to document <doc_id>...

# Neighbor detection
👋 Neighbor UP: <head_node_id>

# Query received (should skip)
📨 RECEIVED QUERY
   ↩️  Worker node (rank 1) - skipping query, will participate in ring

# ===== THIS IS THE CRITICAL PART =====
# If working correctly, you should see:

🔔 Event received: CONTENT_READY for doc <doc_id>...
📦 Content ready event received, hash=<hash>...
📦 Read XXXXX bytes from blob
📨 Content ready: type=ring_tensor_forward, from=<head_node_id>...
📨 Calling 1 message handler(s)...
📬 Message received: type=ring_tensor_forward, sender=<head_node_id>...
🔔 Routing ring_tensor_forward to handler
   Checking target: target=<worker_node_id>..., me=<worker_node_id>...
📨 RECEIVED RING TENSOR MESSAGE
   From: <head_node_id>...
   Request ID: <uuid>
   Target: me
   ✅ Passing to ring coordinator...
📥 RECEIVED TENSOR IN RING COORDINATOR
   Shape: (1, Y, Z)
⚙️  Processing on Rank 1
   Layers: X → Y (Y layers)
```

### 4. Diagnostic Scenarios

#### Scenario A: No CONTENT_READY event on worker
**Symptom**: Worker doesn't show `🔔 Event received: CONTENT_READY`

**Possible causes**:
1. Worker not subscribed to document (check for `✅ Subscribed` log)
2. Iroh sync issue - nodes not connected
3. HEAD not successfully writing to document (check HEAD logs for "✅ Sent")

**Solution**: Verify both nodes show `👋 Neighbor UP` logs for each other

#### Scenario B: CONTENT_READY received but not ring_tensor_forward
**Symptom**: Worker shows `🔔 Event received: CONTENT_READY` but message type is not `ring_tensor_forward`

**Possible causes**:
1. Wrong message type being sent
2. Message corruption during Iroh transfer

**Solution**: Check HEAD logs for exact message being sent

#### Scenario C: ring_tensor_forward received but filtered out
**Symptom**: Worker shows message received but logs `↩️ Ignoring ring tensor meant for...`

**Possible causes**:
1. Node ID mismatch between ring topology and actual Iroh node IDs
2. Topology not synchronized between nodes

**Solution**: Compare node IDs in logs - `target=X..., me=Y...` should match

#### Scenario D: Message handler not called
**Symptom**: Worker shows CONTENT_READY but no `📬 Message received` log

**Possible causes**:
1. Message handler not registered
2. Exception in handle_content_ready
3. Message parsing failure

**Solution**: Check for error logs, verify handler registration in `run_node()`

### 5. Common Fixes

#### Fix 1: Increase Topology Sync Wait Time
If nodes don't discover each other, increase wait time in `llm_service.py`:

```python
# In _init_ring_pipeline()
await asyncio.sleep(5.0)  # Increase from 3.0 to 5.0
```

#### Fix 2: Verify Document Joining
Ensure both nodes are in the same document:

```bash
# In either terminal:
peers

# Should show the other node's ID
```

#### Fix 3: Clear State Between Runs
If retrying, restart both terminals fresh to avoid stale state.

### 6. Success Indicators

✅ Both nodes show NEIGHBOR_UP for each other
✅ Both nodes show ring topology with correct ranks
✅ HEAD shows "✅ Sent in Xms"
✅ WORKER shows "📨 RECEIVED RING TENSOR MESSAGE"
✅ WORKER shows "⚙️ Processing on Rank 1"
✅ HEAD receives final result
✅ Response is displayed

### 7. Next Steps

Once you see where the flow breaks, we can fix:

1. **If messages not reaching worker**: Iroh document sync issue
2. **If messages filtered**: Node ID mismatch in topology
3. **If processing fails**: Model loading or tensor shape issue

Run the test and share the logs from **both terminals** - especially looking for where the expected log flow breaks!

