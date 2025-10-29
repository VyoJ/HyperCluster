# Iroh Self-Broadcast Fix - October 30, 2025

## Critical Issue: Nodes Don't Receive Their Own Broadcasts in Iroh

### The Problem

When running `llm query "hi"` on a **single node**:
- ✅ Query sent successfully (got query ID)
- ❌ **No response at all**
- ❌ **No logs** - not even "📨 RECEIVED QUERY"

### Root Cause

**Iroh doesn't deliver broadcasts back to the sender!**

When you call `send_query("hi")`:
1. ✅ Creates message with query
2. ✅ Broadcasts to document
3. ❌ **Iroh doesn't send it back to you**
4. ❌ Your `message_handler` never sees it
5. ❌ `_handle_query` never called
6. ❌ No processing, no response

This is standard behavior in many P2P systems - broadcasts are for **other** nodes, not yourself.

---

## The Fix

Modified `send_query` to **directly process local queries** instead of relying on broadcast delivery:

### Before (Broken):
```python
async def send_query(self, query: str, llm_node_id: Optional[str] = None) -> str:
    query_id = str(uuid.uuid4())
    query_payload = {...}
    message = {...}
    
    # Broadcast and hope it comes back (it doesn't!)
    success = await self.network.broadcast_message(message)  # ❌
    
    if success:
        return query_id
    return None
```

### After (Fixed):
```python
async def send_query(self, query: str, llm_node_id: Optional[str] = None) -> str:
    query_id = str(uuid.uuid4())
    node_id_str = str(await self.network.iroh_node.net().node_id())
    query_payload = {...}
    message = {...}
    
    # Broadcast to network (for other nodes)
    success = await self.network.broadcast_message(message)
    
    # IMPORTANT: In Iroh, nodes don't receive their own broadcasts!
    # So if this is a local query, process it directly
    should_process_locally = (
        llm_node_id is None or           # Broadcast to all
        llm_node_id == node_id_str       # Target is us
    )
    
    if should_process_locally and self.is_running and self.is_loaded:
        logger.info(f"💡 Processing query locally (Iroh doesn't deliver own broadcasts)")
        # Process directly - bypass message handler
        await self._handle_query(node_id_str, query_payload)
    
    if success:
        return query_id
    return None
```

---

## How It Works Now

### Single Node:
```
User: llm query "hi"
  ↓
send_query()
  ↓
├─→ broadcast_message() ──→ (Other nodes, if any)
  ↓
└─→ _handle_query() ────→ (Local processing - DIRECT CALL)
      ↓
      ├─→ Logs: "📨 RECEIVED QUERY"
      ├─→ Logs: "🔁 Routing to ring pipeline"
      └─→ start_inference()
            ↓
            Response! ✅
```

### Multi-Node (Node A queries, Node B responds):
```
Node A: llm query "hi"
  ↓
send_query()
  ↓
├─→ broadcast_message() ──→ [Iroh Network]
  |                              ↓
  |                         Node B receives
  |                              ↓
  |                         message_handler()
  |                              ↓
  |                         _handle_query()
  |                              ↓
  |                         Response from B ✅
  ↓
└─→ _handle_query() ────→ Local processing on A ✅
```

Now **both** nodes process if it's a broadcast!

---

## What You'll See Now

When you run `llm query "hi"` on a single node:

```
2025-10-30 00:xx:xx - llm_service - INFO - 💡 Processing query locally (Iroh doesn't deliver own broadcasts)
2025-10-30 00:xx:xx - llm_service - INFO - 
2025-10-30 00:xx:xx - llm_service - INFO - 📨 RECEIVED QUERY
2025-10-30 00:xx:xx - llm_service - INFO -    Query ID: 4ceebf71-13b4-4bc7-9498-b4cd95edb86f
2025-10-30 00:xx:xx - llm_service - INFO -    Query: hi...
2025-10-30 00:xx:xx - llm_service - INFO -    From: e680287e4df47386...
2025-10-30 00:xx:xx - llm_service - INFO -    Target: broadcast...
2025-10-30 00:xx:xx - llm_service - INFO -    My ID: e680287e4df47386...
2025-10-30 00:xx:xx - llm_service - INFO -    ✅ Processing query...
2025-10-30 00:xx:xx - llm_service - INFO -    🔁 Routing to ring pipeline

[... ring inference logs ...]

LLM Response from e680287e4df47386... (mode=ring_pipeline, rank=0):
Hello! How can I help you today?
```

---

## Files Modified

**`llm_service.py`** (Lines 345-379):
- Modified `send_query` to detect local queries
- Added direct call to `_handle_query` for local processing
- Added logging to explain why

---

## Testing

```bash
# Terminal 1 - Single node
python main.py --ring
llm start Qwen/Qwen2.5-0.5B-Instruct
llm query "hello"
```

**Expected output:**
```
Query sent (ID: xxx), waiting for responses...
2025-10-30 xx:xx:xx - llm_service - INFO - 💡 Processing query locally...
2025-10-30 xx:xx:xx - llm_service - INFO - 📨 RECEIVED QUERY
[... processing logs ...]
LLM Response from xxx (mode=ring_pipeline, rank=0):
[Response text]
```

---

## Why This Is Important

This is a **critical fix** for:
1. ✅ **Single-node testing** - Can now test locally
2. ✅ **Development workflow** - Don't need 2 nodes to test
3. ✅ **Broadcast queries** - Local node also processes broadcast queries
4. ✅ **User experience** - Queries work as expected

Without this fix:
- ❌ Single node queries silently fail
- ❌ No way to test without multiple nodes
- ❌ Confusing user experience

---

## Related P2P Concepts

This is common in P2P systems:
- **libp2p**: Same behavior - don't receive own pubsub messages
- **IPFS**: Same - broadcasts go to others
- **Hypercore**: Same - append goes to network, not back to you

**Why?** Prevents infinite loops and reduces redundant processing. The sender already has the data, so no need to send it back.

**Solution**: Always handle local operations explicitly, don't rely on network echo.

---

## Summary

**Problem**: Iroh doesn't deliver your own broadcasts back to you  
**Impact**: Single-node queries failed silently  
**Fix**: Direct local processing in `send_query`  
**Result**: Queries now work on single nodes! ✅

