# Iroh Large Message Workaround - Chunking Strategy

## Problem

Even with increased sync delays (1-5 seconds), large tensor messages (~0.75MB) are not reliably syncing through Iroh documents to worker nodes. Small messages (<1KB) work fine, but large messages get stuck.

## Root Cause

Iroh's document sync appears to have issues with very large blobs (>500KB). This could be:
1. Timeout in blob transfer
2. Memory pressure during large blob sync
3. Relay limitations for large payloads
4. Network fragmentation issues

## Solution: Message Chunking

Instead of sending one 0.75MB message, split it into smaller chunks:

### Implementation Plan

```python
# In ring_pipeline.py

CHUNK_SIZE = 100 * 1024  # 100KB chunks

async def _send_to_node_chunked(
    self, target_node_id: str, data: np.ndarray, request_id: str, is_final: bool
):
    """Send tensor in chunks to avoid Iroh large message issues."""
    
    # Serialize tensor
    tensor_bytes = data.tobytes()
    total_size = len(tensor_bytes)
    num_chunks = (total_size + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    logger.info(f"   📦 Splitting {total_size/1024/1024:.2f}MB into {num_chunks} chunks")
    
    # Send metadata first
    metadata_msg = {
        "type": "ring_tensor_metadata",
        "sender_id": str(await self.network.iroh_node.net().node_id()),
        "target_node_id": target_node_id,
        "request_id": request_id,
        "payload": {
            "total_chunks": num_chunks,
            "tensor_shape": list(data.shape),
            "tensor_dtype": str(data.dtype),
            "total_size": total_size,
            "is_final": is_final,
        },
        "timestamp": time.time(),
    }
    await self.network.send_message(doc_id, metadata_msg)
    await asyncio.sleep(0.2)  # Small delay between metadata and chunks
    
    # Send chunks
    for chunk_id in range(num_chunks):
        start = chunk_id * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, total_size)
        chunk_data = tensor_bytes[start:end]
        chunk_b64 = base64.b64encode(chunk_data).decode("utf-8")
        
        chunk_msg = {
            "type": "ring_tensor_chunk",
            "sender_id": str(await self.network.iroh_node.net().node_id()),
            "target_node_id": target_node_id,
            "request_id": request_id,
            "payload": {
                "chunk_id": chunk_id,
                "total_chunks": num_chunks,
                "chunk_data": chunk_b64,
            },
            "timestamp": time.time(),
        }
        
        await self.network.send_message(doc_id, chunk_msg)
        
        logger.info(f"   📤 Sent chunk {chunk_id+1}/{num_chunks} ({len(chunk_data)/1024:.1f}KB)")
        
        # Small delay between chunks to avoid overwhelming Iroh
        if chunk_id < num_chunks - 1:
            await asyncio.sleep(0.1)
```

### Receiver Side

```python
# In node.py - add chunk reassembly

class Node:
    def __init__(self, ...):
        # ... existing init ...
        self.pending_chunks: Dict[str, Dict] = {}  # request_id -> chunk data
    
    async def handle_ring_tensor_metadata(self, message_data: Dict):
        """Handle tensor metadata message."""
        request_id = message_data.get("request_id")
        payload = message_data.get("payload", {})
        
        self.pending_chunks[request_id] = {
            "total_chunks": payload["total_chunks"],
            "tensor_shape": tuple(payload["tensor_shape"]),
            "tensor_dtype": np.dtype(payload["tensor_dtype"]),
            "total_size": payload["total_size"],
            "is_final": payload["is_final"],
            "chunks": {},  # chunk_id -> data
            "received_count": 0,
        }
        
        logger.info(f"📦 Expecting {payload['total_chunks']} chunks for request {request_id}")
    
    async def handle_ring_tensor_chunk(self, message_data: Dict, llm_service):
        """Handle incoming tensor chunk and reassemble when complete."""
        request_id = message_data.get("request_id")
        payload = message_data.get("payload", {})
        chunk_id = payload["chunk_id"]
        
        if request_id not in self.pending_chunks:
            logger.warning(f"Received chunk for unknown request: {request_id}")
            return
        
        # Store chunk
        chunk_data = base64.b64decode(payload["chunk_data"])
        self.pending_chunks[request_id]["chunks"][chunk_id] = chunk_data
        self.pending_chunks[request_id]["received_count"] += 1
        
        received = self.pending_chunks[request_id]["received_count"]
        total = self.pending_chunks[request_id]["total_chunks"]
        
        logger.info(f"📦 Received chunk {chunk_id+1}/{total} for request {request_id}")
        
        # Check if all chunks received
        if received == total:
            logger.info(f"✅ All chunks received! Reassembling tensor...")
            await self._reassemble_and_process_tensor(request_id, llm_service)
    
    async def _reassemble_and_process_tensor(self, request_id: str, llm_service):
        """Reassemble tensor from chunks and process."""
        chunk_info = self.pending_chunks[request_id]
        
        # Reassemble in order
        tensor_bytes = b""
        for chunk_id in range(chunk_info["total_chunks"]):
            tensor_bytes += chunk_info["chunks"][chunk_id]
        
        # Reconstruct tensor
        tensor = np.frombuffer(tensor_bytes, dtype=chunk_info["tensor_dtype"]).reshape(
            chunk_info["tensor_shape"]
        )
        
        logger.info(f"✅ Reassembled tensor: shape={tensor.shape}, size={len(tensor_bytes)/1024/1024:.2f}MB")
        
        # Clean up
        del self.pending_chunks[request_id]
        
        # Process as normal
        if llm_service and llm_service.ring_coordinator:
            await llm_service.ring_coordinator.handle_incoming_tensor(
                sender_id=message_data.get("sender_id"),
                request_id=request_id,
                tensor_data=tensor,
                shard=llm_service.current_shard,
                is_final=chunk_info["is_final"],
            )
```

### Message Handler Updates

```python
# In main.py message_handler

elif msg_type == "ring_tensor_metadata":
    msg_logger.info(f"📦 Routing tensor metadata to handler")
    if node:
        await node.handle_ring_tensor_metadata(message)

elif msg_type == "ring_tensor_chunk":
    msg_logger.debug(f"📦 Routing tensor chunk to handler")
    if node and llm_service:
        await node.handle_ring_tensor_chunk(message, llm_service)
```

## Benefits

1. **Reliability**: 100KB chunks sync much more reliably than 750KB monoliths
2. **Progress Visibility**: Can see chunk-by-chunk progress
3. **Graceful Degradation**: If one chunk fails, can retry just that chunk
4. **Network Friendly**: Smaller packets less likely to fragment or timeout

## Drawbacks

1. **Latency**: Multiple round-trips add ~1-2 seconds overhead
2. **Complexity**: More code to maintain
3. **Memory**: Need to buffer chunks during reassembly

## Alternative: Switch to Direct P2P

If chunking doesn't work, we could bypass Iroh documents entirely for large tensors:

```python
# Use Iroh's P2P primitives directly
# Send tensor via direct connection instead of document sync
# This would require knowing the peer's network address
```

But let's try chunking first as it's less invasive.

---

**Next Step**: Implement chunking if current fixes don't work.

