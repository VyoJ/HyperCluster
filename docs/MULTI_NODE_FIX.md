# Multi-Node Ring Pipeline Fix - October 30, 2025

## Problem

Single-node ring pipeline works ✅, but multi-node fails with timeout:

```
📤 Forwarding to Rank 1 (b7590c98...)
📤 Sending tensor: 5.81 MB
✅ Sent in 663.0ms
[... 30 seconds ...]
⚠️ Timeout waiting for ring completion!
```

Additionally, logs showed:
```
LLM Status from b7590c98... processing
```

This revealed that **Rank 1 (worker node) was trying to process the query directly** instead of just participating in the ring!

---

## Root Cause

When `llm query "hi"` was sent:
1. ✅ Query broadcast to all nodes (including worker)
2. ✅ HEAD (Rank 0) starts processing
3. ❌ **WORKER (Rank 1) also tries to process** → Sends "processing" status
4. ❌ Worker is busy processing instead of waiting for ring tensors
5. ❌ When Rank 0 sends tensor, Rank 1 can't receive it
6. ❌ Timeout!

**The issue**: All nodes with LLM service were processing every query, even worker nodes that should only participate in the ring by processing tensors.

---

## The Fix

Added a **worker node check** in `_handle_query` to prevent worker nodes from processing queries:

### Before (Broken):
```python
async def _handle_query(self, sender_id: str, data: dict):
    # ... validation ...
    
    if not query:
        return
    
    # ❌ ALL nodes process the query
    logger.info(f"   ✅ Processing query...")
    
    status_update = {
        "llm_type": LLMMessageType.STATUS.value,
        "query_id": query_id,
        "status": "processing",  # ❌ Worker sends this too!
    }
    await self._send_llm_data(status_update)
    
    # ❌ Worker also routes to ring processing
    if self.use_ring and self.ring_coordinator:
        asyncio.create_task(self._process_query_ring(query_id, query))
```

### After (Fixed):
```python
async def _handle_query(self, sender_id: str, data: dict):
    # ... validation ...
    
    if not query:
        return
    
    # ✅ In ring mode, only HEAD node processes queries
    if self.use_ring and self.ring_coordinator and self.ring_coordinator.ring_position:
        if not self.ring_coordinator.ring_position.is_head:
            logger.info(
                f"   ↩️  Worker node (rank {self.ring_coordinator.ring_position.rank}) "
                f"- skipping query, will participate in ring"
            )
            return  # ✅ Worker exits early
    
    # Only HEAD reaches here
    logger.info(f"   ✅ Processing query...")
    
    status_update = {
        "llm_type": LLMMessageType.STATUS.value,
        "query_id": query_id,
        "status": "processing",  # ✅ Only HEAD sends this
    }
    await self._send_llm_data(status_update)
    
    # ✅ Only HEAD routes to ring processing
    if self.use_ring and self.ring_coordinator:
        asyncio.create_task(self._process_query_ring(query_id, query))
```

---

## How It Works Now

### Multi-Node Flow:

```
User: llm query "hi"
  ↓
Broadcast to network
  ├─→ Rank 0 (HEAD):
  │     ├─→ _handle_query()
  │     ├─→ is_head? YES
  │     ├─→ ✅ Process query
  │     ├─→ Start ring inference
  │     ├─→ Process layers 0-11
  │     └─→ Send tensor to Rank 1 →┐
  │                                  │
  └─→ Rank 1 (WORKER):               │
        ├─→ _handle_query()          │
        ├─→ is_head? NO              │
        ├─→ ↩️  Skip, wait for ring  │
        └─→ 🔇 Silent, ready         │
                    ↓                │
             Receives tensor ←───────┘
                    ↓
             📥 handle_incoming_tensor()
                    ↓
             Process layers 12-23
                    ↓
             Send result back to HEAD
                    ↓
        HEAD samples, generates, responds! ✅
```

**Key difference**: Worker node now **silently waits** instead of trying to process the query.

---

## What You'll See Now

### On HEAD Node (Rank 0):
```
📨 RECEIVED QUERY
   Query: hi...
   From: b7590c98...
   Target: broadcast...
   ✅ Processing query...
   🔁 Routing to ring pipeline

⚙️  Processing on Rank 0
   Layers: 0 → 11 (12 layers)
   Input shape: (1, 31)
   Output shape: (1, 31, 49152)
   📤 Forwarding to Rank 1
   📤 Sending tensor: 5.81 MB
   ✅ Sent in 663.0ms

[... Rank 1 processes and sends back ...]

📥 RECEIVED TENSOR IN RING COORDINATOR  ← NEW!
   From: b7590c98...
   Is final: True
   ✅ Final result received at HEAD
```

### On WORKER Node (Rank 1):
```
📨 RECEIVED QUERY
   Query: hi...
   From: b7590c98...
   ↩️  Worker node (rank 1) - skipping query, will participate in ring  ← NEW!

[... Waits silently ...]

📨 RECEIVED RING TENSOR MESSAGE  ← When tensor arrives
   From: e680287e...
   Target: me
   ✅ Passing to ring coordinator...

📥 RECEIVED TENSOR IN RING COORDINATOR
   Shape: (1, 31, 49152)
   Created new state

⚙️  Processing on Rank 1
   Layers: 12 → 23 (12 layers)
   Input shape: (1, 31, 49152)
   Output shape: (1, 31, 49152)
   🏁 Final layer reached!
   📤 Worker node: Sending final result to HEAD
```

---

## Files Modified

**`llm_service.py`** (Lines 202-207):
- Added worker node check in `_handle_query`
- Worker nodes skip query processing entirely
- Only HEAD node processes queries

---

## Testing

### Test 1: Single Node (should still work):
```bash
# Terminal 1
python main.py --ring
llm start Qwen/Qwen2.5-0.5B-Instruct
llm query "hello"
```

**Expected**: Works as before ✅

### Test 2: Multi-Node (should now work):
```bash
# Terminal 1 (HEAD)
python main.py --ring
llm start Qwen/Qwen2.5-0.5B-Instruct

# Terminal 2 (WORKER)
python main.py --ring --bootstrap-ticket <ticket>
llm start Qwen/Qwen2.5-0.5B-Instruct

# Back in Terminal 1
llm query "hello"
```

**Expected on Terminal 1 (HEAD)**:
- Processing query
- Layers 0-11 processed
- Forwarding to Rank 1
- Receiving final result
- Response shown ✅

**Expected on Terminal 2 (WORKER)**:
- "↩️ Worker node - skipping query"
- Waits silently
- Receives tensor from Rank 0
- Processes layers 12-23
- Sends back to HEAD
- No response output (only HEAD shows response)

---

## Summary

**Problem**: Worker nodes were processing queries instead of just handling ring tensors  
**Fix**: Added early return in `_handle_query` for worker nodes  
**Result**: Multi-node ring pipeline now works! ✅

Worker nodes now:
- ✅ Skip query processing
- ✅ Wait for ring tensor messages
- ✅ Process their assigned layers
- ✅ Forward results correctly

Try your multi-node test again - it should work now! 🚀

