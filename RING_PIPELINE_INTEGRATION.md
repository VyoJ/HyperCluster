# Ring Pipeline Integration Guide

## Overview

This guide explains how to integrate prima.cpp's ring pipelined inference into HyperCluster using the new `ring_pipeline.py` module.

## Architecture Comparison

### Prima.cpp Ring Pipeline

```
┌─────────────────────────────────────────────────┐
│  Token Generation Loop (Auto-regressive)        │
│  ┌───────────────────────────────────────────┐  │
│  │  For each new token:                      │  │
│  │                                           │  │
│  │  Cycle 1: Process layers 0-47             │  │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  │  │
│  │  │ Rank0│─▶│ Rank1│─▶│ Rank2│─▶│ Rank3│  │  │
│  │  │ L0-15│  │L16-31│  │L32-47│  │ meta │  │  │
│  │  └──────┘  └──────┘  └──────┘  └──────┘  │  │
│  │     ▲                             │       │  │
│  │     └─────────────────────────────┘       │  │
│  │                                           │  │
│  │  Cycle 2: Process layers 48-63 (if any)  │  │
│  │  ┌──────┐  ┌──────┐                      │  │
│  │  │ Rank0│─▶│ Rank1│─▶ ... (same ring)    │  │
│  │  │L48-63│  │  -   │                      │  │
│  │  └──────┘  └──────┘                      │  │
│  │                                           │  │
│  │  Sample token from logits                │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘

Key Features:
- Ring topology: Data flows in a circle
- Multiple cycles: Activations go around multiple times
- Prefetching: Next layers load while current layers compute
- Async communication: ZeroMQ overlaps transfer with compute
```

### HyperCluster Ring Pipeline (New)

```
┌─────────────────────────────────────────────────┐
│  RingPipelineCoordinator                        │
│  ┌───────────────────────────────────────────┐  │
│  │  1. Initialize ring topology              │  │
│  │     - Assign ranks based on memory        │  │
│  │     - Calculate layer windows             │  │
│  │     - Set up ring neighbors               │  │
│  │                                           │  │
│  │  2. For each token generation:           │  │
│  │     a. Head node encodes prompt           │  │
│  │     b. Start ring forward pass            │  │
│  │        - Process my layers                │  │
│  │        - Send to next node (Iroh)         │  │
│  │        - Receive from prev node           │  │
│  │        - Repeat until all layers done     │  │
│  │     c. Head node samples next token       │  │
│  │     d. Repeat until EOS                   │  │
│  │                                           │  │
│  │  3. Prefetch worker (background)          │  │
│  │     - Preload next layers into memory     │  │
│  │     - Overlap disk I/O with compute       │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  Communication: Iroh Documents                  │
│  - Message type: "ring_tensor_forward"          │
│  - Serialization: Base64 encoded numpy arrays   │
│  - Async: Python asyncio + Iroh                 │
└─────────────────────────────────────────────────┘
```

## Key Concepts from Prima.cpp

### 1. Ring Topology Formation

**Prima.cpp** (from `common.cpp`):
```cpp
// Devices form a ring: Rank0 → Rank1 → ... → RankN → Rank0
// Each device knows:
//   - my_rank: Position in ring
//   - n_world: Total devices
//   - master_ip: Head node address
//   - next_node_ip: Next device in ring
```

**HyperCluster** (in `ring_pipeline.py`):
```python
class RingPosition:
    rank: int                 # 0 = head node
    world_size: int           # Total nodes
    prev_node_id: str         # Receive from
    next_node_id: str         # Send to
    is_head: bool             # Coordinator role
```

### 2. Layer Window Assignment

**Prima.cpp** (from `common.cpp:assign_layers_to_device`):
```cpp
// Memory-weighted assignment
for (uint32_t m = 0; m < n_world; ++m) {
    w[m] = std::round(mem_budget[m] / total_mem_budget * n_layer);
    n[m] = 0;  // GPU layers
}
```

**HyperCluster**:
```python
def _calculate_layer_windows(self, sorted_nodes, total_layers):
    """
    Assign layers proportionally to memory.
    More memory → more layers
    """
    for node, capabilities in sorted_nodes:
        memory_fraction = capabilities.memory / total_memory
        num_layers = int(total_layers * memory_fraction)
        # Assign layers [start:end]
```

### 3. Multi-Cycle Processing

**Prima.cpp** (from `llama.cpp:llama_decode_internal`):
```cpp
// If model has 64 layers but devices handle 16+16+16+16=48 per cycle:
// Cycle 1: Layers 0-47
// Cycle 2: Layers 48-63
// Data goes around ring twice

for (size_t i = 0; i < gf.size(); ++i) {
    // Process subgraph i
    // Send to next node
    // Receive from prev node
}
```

**HyperCluster**:
```python
def calculate_cycles_needed(self, total_layers):
    """
    If total_layers > sum(all_windows), need multiple cycles.
    cycles = ceil(total_layers / total_window_size)
    """
    return (total_layers + total_window - 1) // total_window
```

### 4. Tensor Communication

**Prima.cpp** (ZeroMQ):
```cpp
// Send tensors
static void llama_send_tensors(zmq::socket_t & socket, ...) {
    std::vector<zmq::message_t> send_msgs;
    send_msgs.emplace_back("sub_gf_out", ...);
    send_msgs.emplace_back(tensor_data, size);
    zmq::send_multipart(socket, send_msgs);
}

// Receive tensors
static void llama_recv_tensors(zmq::socket_t & socket, ...) {
    std::vector<zmq::message_t> recv_msgs;
    zmq::recv_multipart(socket, std::back_inserter(recv_msgs));
    // Extract tensor data
}
```

**HyperCluster** (Iroh):
```python
async def _send_to_node(self, target_node_id, data, request_id, is_final):
    """Send via Iroh document."""
    tensor_b64 = base64.b64encode(data.tobytes()).decode("utf-8")
    message = {
        "type": "ring_tensor_forward",
        "payload": {
            "tensor_data": tensor_b64,
            "tensor_shape": list(data.shape),
            "tensor_dtype": str(data.dtype),
        }
    }
    await self.network.send_message(doc_id, message)
```

### 5. Prefetching

**Prima.cpp** (from `llama.cpp:manage_graph_tensors`):
```cpp
// Use POSIX memory advice to preload pages
static void manage_graph_tensors(struct ggml_cgraph * cgraph, int advice, bool force) {
    for (int i = 0; i < ggml_graph_n_leafs(cgraph); i++) {
        struct ggml_tensor * cur = ggml_graph_leaf(cgraph, i);
        if (strstr(cur->name, "weight")) {
            // Tell OS to load these pages
            posix_madvise(cur->data, size, POSIX_MADV_WILLNEED);
            
            // Optionally force load by touching pages
            if (force) {
                volatile char * ptr = cur->data;
                for (size_t off = 0; off < len; off += page_size) {
                    (void)ptr[off];  // Touch page
                }
            }
        }
    }
}

// Called during inference
if (cparams.prefetch && n_world > 1) {
    int next_gf_id = (i + 1) % gf.size();
    manage_graph_tensors(gf[next_gf_id], POSIX_MADV_WILLNEED, force);
}
```

**HyperCluster**:
```python
async def _prefetch_worker(self):
    """Background worker to preload model weights."""
    while True:
        next_layer_id = await self.prefetch_queue.get()
        
        # For transformers models:
        # 1. Identify weight tensors for next layers
        # 2. Touch memory to bring into page cache
        # 3. Or async load from disk/network
        
        # Example with mmap:
        # import mmap
        # with open(weight_file, 'rb') as f:
        #     mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        #     mm.madvise(mmap.MADV_WILLNEED)
```

## Integration Steps

### Step 1: Update `llm_service.py`

```python
from ring_pipeline import RingPipelineCoordinator

class LLMService:
    def __init__(self, ...):
        # ... existing code ...
        
        # Add ring pipeline coordinator
        self.ring_coordinator: Optional[RingPipelineCoordinator] = None
        self.use_ring_pipeline = use_ring_pipeline  # New param
    
    async def start(self, model_id: str):
        # ... existing initialization ...
        
        if self.use_sharding and self.use_ring_pipeline:
            # Initialize ring pipeline
            self.ring_coordinator = RingPipelineCoordinator(
                inference_engine=self.inference_engine,
                network_node=self.network
            )
            
            # Set up ring topology
            topology_nodes = self.network.topology.all_nodes()
            my_node_id = str(await self.network.iroh_node.net().node_id())
            
            await self.ring_coordinator.initialize_ring(
                topology_nodes=topology_nodes,
                my_node_id=my_node_id,
                model_total_layers=self.current_shard.n_layers
            )
            
            logger.info("Ring pipeline initialized")
    
    async def _process_query_ring_pipeline(self, query: str, request_id: str):
        """Process query using ring pipeline."""
        if not self.ring_coordinator:
            return await self._process_query_sharded(query, request_id)
        
        # Only head node starts generation
        if self.ring_coordinator.ring_position.is_head:
            generated_tokens = await self.ring_coordinator.start_inference(
                request_id=request_id,
                prompt=query,
                shard=self.current_shard,
                max_tokens=self.max_generate_tokens
            )
            
            # Decode tokens
            response_text = await self.inference_engine.decode(
                self.current_shard, 
                np.array(generated_tokens)
            )
            
            return {
                "response": response_text,
                "tokens": generated_tokens,
                "node_id": str(await self.network.iroh_node.net().node_id())
            }
        else:
            # Worker nodes wait for ring messages
            return {
                "status": "waiting",
                "message": "Worker node in ring pipeline"
            }
```

### Step 2: Update Message Handler in `main.py`

```python
async def message_handler(message_data: Dict):
    """Handle incoming messages with ring pipeline support."""
    msg_type = message_data.get("type")
    
    if msg_type == "ring_tensor_forward":
        # Handle ring pipeline tensor
        sender_id = message_data.get("sender_id")
        request_id = message_data.get("payload", {}).get("request_id")
        
        # Deserialize tensor
        import base64
        tensor_b64 = message_data["payload"]["tensor_data"]
        tensor_bytes = base64.b64decode(tensor_b64)
        tensor_shape = tuple(message_data["payload"]["tensor_shape"])
        tensor_dtype = np.dtype(message_data["payload"]["tensor_dtype"])
        
        tensor = np.frombuffer(tensor_bytes, dtype=tensor_dtype).reshape(tensor_shape)
        
        # Pass to ring coordinator
        if llm_service.ring_coordinator:
            await llm_service.ring_coordinator.handle_incoming_tensor(
                sender_id=sender_id,
                request_id=request_id,
                tensor_data=tensor,
                shard=llm_service.current_shard,
                is_final=message_data["payload"].get("is_final", False)
            )
    
    elif msg_type == "topology_update":
        # Handle topology updates
        # ... existing code ...
    
    # ... other message types ...
```

### Step 3: Test with Multiple Nodes

**Terminal 1 (Head Node):**
```bash
python main.py start
# Note the ticket

# In REPL:
> llm start Qwen/Qwen2.5-0.5B-Instruct --use-ring
> llm query "Explain how transformers work"
```

**Terminal 2 (Worker Node):**
```bash
python main.py start --bootstrap-ticket <ticket>

# In REPL:
> llm start Qwen/Qwen2.5-0.5B-Instruct --use-ring
# Node automatically joins ring and processes assigned layers
```

**Terminal 3 (Worker Node):**
```bash
python main.py start --bootstrap-ticket <ticket>

# In REPL:
> llm start Qwen/Qwen2.5-0.5B-Instruct --use-ring
```

## Expected Behavior

### Ring Formation
```
Node 1 (8GB) → Assigned layers 0-31   (rank 0, head)
Node 2 (4GB) → Assigned layers 32-47  (rank 1)
Node 3 (4GB) → Assigned layers 48-63  (rank 2)

Ring: Node1 → Node2 → Node3 → Node1
```

### Inference Flow
```
1. User queries Node1 (head)
2. Node1 encodes prompt → tokens [1, 2, 3, ...]
3. Ring Forward Pass:
   a. Node1 processes layers 0-31  → hidden states H1
   b. Node1 sends H1 to Node2
   c. Node2 processes layers 32-47 → hidden states H2
   d. Node2 sends H2 to Node3
   e. Node3 processes layers 48-63 → logits L
   f. Node3 sends L back to Node1
4. Node1 samples next token from logits
5. Repeat steps 3-4 for each token
```

### Prefetching (During Inference)
```
While Node2 processes layers 32-47:
  Node1 prefetches layers 0-31 for next token
  Node3 prefetches layers 48-63 for next token

Result: Disk I/O overlaps with GPU compute
```

## Performance Optimization

### 1. Compression (Future)

**Prima.cpp**: Uses quantization (Q4K, Q6K, Q8_0)

**HyperCluster**: Add tensor compression
```python
# In _send_to_node
compressed = zlib.compress(tensor_bytes, level=1)  # Fast compression
tensor_b64 = base64.b64encode(compressed).decode("utf-8")

# In handle_incoming_tensor
compressed = base64.b64decode(tensor_b64)
tensor_bytes = zlib.decompress(compressed)
```

### 2. Batching

**Prima.cpp**: Uses batch processing

**HyperCluster**: Batch multiple requests
```python
# Process multiple requests in one forward pass
batch_tensors = [req.hidden_states for req in active_requests]
batch_output = await self.inference_engine.infer_tensor(
    request_ids=[r.id for r in active_requests],
    shard=shard,
    input_data=np.concatenate(batch_tensors, axis=0)
)
```

### 3. Pipeline Parallelism

**Prima.cpp**: Overlaps communication with computation

**HyperCluster**: Use asyncio
```python
# Start next node processing before receiving result
async def _process_and_forward_async(self, ...):
    # Process layers
    output_task = asyncio.create_task(
        self.inference_engine.infer_tensor(...)
    )
    
    # Send previous result while computing
    await asyncio.gather(
        self._send_to_node(...),  # Send previous
        output_task               # Compute current
    )
```

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('ring_pipeline').setLevel(logging.DEBUG)
```

### Trace Message Flow

```python
# Add to _send_to_node
logger.debug(
    f"SEND: {self.ring_position.rank} → {target_node_id[:8]}, "
    f"request={request_id}, shape={data.shape}"
)

# Add to handle_incoming_tensor
logger.debug(
    f"RECV: {sender_id[:8]} → {self.ring_position.rank}, "
    f"request={request_id}, shape={tensor_data.shape}"
)
```

### Monitor Latency

```python
# Measure per-node latency
start_time = time.time()
output = await self.inference_engine.infer_tensor(...)
latency = time.time() - start_time

logger.info(
    f"Layer processing: {latency*1000:.2f}ms, "
    f"layers={layers_to_process}"
)
```

## Comparison: Prima.cpp vs HyperCluster

| Feature | Prima.cpp | HyperCluster |
|---------|-----------|--------------|
| **Language** | C++ | Python |
| **Model Backend** | llama.cpp (custom) | Transformers (HuggingFace) |
| **Network** | ZeroMQ | Iroh (P2P) |
| **Serialization** | Raw bytes | Base64 JSON |
| **Memory Management** | mmap + madvise | PyTorch CUDA cache |
| **Prefetching** | POSIX madvise | AsyncIO + mmap |
| **Ring Formation** | Manual IP config | Auto via Iroh peers |
| **Layer Loading** | Lazy (mmap) | Eager (for now) |

## Next Steps

1. **Implement Layer-by-Layer Processing**
   - Currently loads full model
   - Need to extract/load only assigned layers
   - Reduces memory usage

2. **Optimize Tensor Transfer**
   - Add compression (zlib/lz4)
   - Binary protocol instead of JSON
   - Streaming for large tensors

3. **True Prefetching**
   - Implement mmap-based weight loading
   - Touch pages before needed
   - Async layer downloads

4. **Fault Tolerance**
   - Handle node failures
   - Re-route around failed nodes
   - Request retry logic

5. **Performance Metrics**
   - Latency per node
   - Network throughput
   - Memory usage
   - Tokens/second

## References

- **Prima.cpp**: `src/llama.cpp:llama_decode_internal()` (lines 17970-18367)
- **Prima.cpp Ring**: `src/llama.cpp:llama_send_tensors()` (lines 17776-17797)
- **Prima.cpp Prefetch**: `src/llama.cpp:manage_graph_tensors()` (lines 17896-17962)
- **Prima.cpp Layer Assignment**: `common/common.cpp:assign_layers_to_device()` (lines 857-1501)

