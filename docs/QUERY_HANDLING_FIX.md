# LLM Query Handling Fix - October 30, 2025

## Problem

When running `llm query "hi"` on a single node:
- ❌ `llm services` shows empty table (even though service is running)
- ❌ Query gets sent but no response
- ❌ No logs showing query processing

## Root Causes

### Issue 1: Local Service Not Shown in `llm services`

**Problem**: The `llm services` command only shows services from `llm_nodes` dict, which is populated when receiving `llm_service_info` broadcasts from OTHER nodes. In Iroh, nodes don't receive their own broadcasts.

**Fix**: Modified the `llm services` command to include the local node's service if it's running.

**Before:**
```python
table = Table(title="LLM Services")
table.add_column("Node ID", style="cyan")
table.add_column("Model Name", style="green")
for node_id, info in llm_nodes.items():  # ❌ Only shows other nodes
    table.add_row(node_id, info.get("model_name", "N/A"))
```

**After:**
```python
table = Table(title="LLM Services")
table.add_column("Node ID", style="cyan")
table.add_column("Model Name", style="green")
table.add_column("Status", style="yellow")

# Add local service if running
if llm_service and llm_service.is_running:  # ✅ Shows local node
    my_node_id = str(await node.iroh_node.net().node_id())
    mode = "ring" if llm_service.use_ring else "sharded" if llm_service.use_sharding else "standard"
    table.add_row(
        f"{my_node_id[:16]}... (me)", 
        llm_service.model_name,
        f"✓ {mode}"
    )

# Add discovered services from other nodes
for node_id, info in llm_nodes.items():
    table.add_row(...)
```

---

### Issue 2: No Visibility Into Query Processing

**Problem**: When queries fail silently, there's no way to see where in the flow they break.

**Fix**: Added comprehensive logging throughout the query handling pipeline:

#### In `llm_service.py`:

**`handle_llm_message`** (Entry point):
```python
logger.debug(f"handle_llm_message called with type: {llm_type}")
if llm_type == LLMMessageType.QUERY.value:
    logger.debug(f"Routing to _handle_query")
```

**`_handle_query`** (Query validator and router):
```python
logger.info(f"")
logger.info(f"📨 RECEIVED QUERY")
logger.info(f"   Query ID: {query_id}")
logger.info(f"   Query: {query[:50]}...")
logger.info(f"   From: {sender_id[:16]}...")
logger.info(f"   Target: {target_node_id[:16] if target_node_id else 'broadcast'}...")
logger.info(f"   My ID: {my_node_id[:16]}...")
logger.info(f"   ✅ Processing query...")

# Route to appropriate backend
if self.use_ring and self.ring_coordinator:
    logger.info(f"   🔁 Routing to ring pipeline")
elif self.use_sharding:
    logger.info(f"   📦 Routing to sharded inference")
else:
    logger.info(f"   🔧 Routing to standard inference")
```

#### In `main.py`:

Added warning if query received but service not available:
```python
if llm_service and llm_service.is_running:
    await llm_service.handle_llm_message(message)
elif llm_type == LLMMessageType.QUERY.value:
    logging.getLogger("main").warning(
        f"Received query but LLM service not running"
    )
```

---

## What You'll See Now

### `llm services` Command:

**Before (empty):**
```
      LLM Services      
┏━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Node ID ┃ Model Name ┃
┡━━━━━━━━━╇━━━━━━━━━━━━┩
└─────────┴────────────┘
```

**After (shows local service):**
```
                    LLM Services                    
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Node ID           ┃ Model Name               ┃ Status  ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ e680287e4df47386  │ Qwen/Qwen2.5-0.5B-Instru │ ✓ ring  │
│ ... (me)          │ ct                        │         │
└───────────────────┴──────────────────────────┴─────────┘
```

### Query Processing Logs:

**When you run `llm query "hi"`, you'll now see:**

```
2025-10-30 00:xx:xx,xxx - llm_service - INFO - 
2025-10-30 00:xx:xx,xxx - llm_service - INFO - 📨 RECEIVED QUERY
2025-10-30 00:xx:xx,xxx - llm_service - INFO -    Query ID: f3b876d6-0af7-40d2-a1be-756de18369d5
2025-10-30 00:xx:xx,xxx - llm_service - INFO -    Query: hi...
2025-10-30 00:xx:xx,xxx - llm_service - INFO -    From: e680287e4df47386...
2025-10-30 00:xx:xx,xxx - llm_service - INFO -    Target: broadcast...
2025-10-30 00:xx:xx,xxx - llm_service - INFO -    My ID: e680287e4df47386...
2025-10-30 00:xx:xx,xxx - llm_service - INFO -    ✅ Processing query...
2025-10-30 00:xx:xx,xxx - llm_service - INFO -    🔁 Routing to ring pipeline
```

**If query is not for this node:**
```
2025-10-30 00:xx:xx,xxx - llm_service - INFO -    ↩️  Not for me, skipping
```

**If service not ready:**
```
2025-10-30 00:xx:xx,xxx - llm_service - WARNING -    ⚠️  Service not running or not loaded
```

---

## Files Modified

1. **`main.py`**:
   - Lines 243-266: Enhanced `llm services` command to show local service
   - Lines 79-82: Added warning for queries when service not available

2. **`llm_service.py`**:
   - Lines 155-166: Added logging to `handle_llm_message`
   - Lines 168-221: Enhanced `_handle_query` with comprehensive logging

---

## Testing

1. **Start a node with ring mode:**
   ```bash
   python main.py --ring
   ```

2. **Start LLM service:**
   ```
   llm start Qwen/Qwen2.5-0.5B-Instruct
   ```

3. **Check services (should now show your node):**
   ```
   llm services
   ```
   You should see your node listed with "(me)" and status "✓ ring"

4. **Send a query:**
   ```
   llm query "hello"
   ```
   
5. **Watch the logs** - you should see:
   - `📨 RECEIVED QUERY` 
   - Query details
   - Routing decision (🔁 ring / 📦 sharded / 🔧 standard)
   - Processing logs

---

## Debugging Next Steps

If you still don't see the query logs after this fix:

1. **Check if message is received at all:**
   - You should see a status message: `LLM Status from ...`
   - If you don't see this, the broadcast isn't working

2. **Check the routing decision:**
   - Look for "🔁 Routing to ring pipeline" or similar
   - This tells you which backend is being used

3. **Check for early returns:**
   - Look for "↩️ Not for me" - means wrong target
   - Look for "⚠️ Service not running" - means service state issue
   - Look for "⚠️ No query content" - means empty query

4. **For ring pipeline mode:**
   - If it times out, check the previous tensor routing fixes
   - Single node should process locally without network

---

## Summary

✅ **Fixed**:
- `llm services` now shows local node
- Comprehensive logging for query handling
- Easy debugging of query flow

🔍 **What to look for**:
- Query reception logs
- Routing decision logs  
- Processing mode (ring/sharded/standard)

📊 **Next**:
- Run tests with these new logs
- Identify exactly where the flow breaks
- Fix the specific issue revealed by logs

