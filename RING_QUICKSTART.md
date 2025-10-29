# Ring Pipeline Quick Start

## TL;DR

This shows you exactly how to wire up the ring pipeline into your existing HyperCluster code.

## Minimal Integration (30 minutes)

### 1. Update `llm_service.py` (Add Ring Mode)

Add this to your `LLMService` class:

```python
# At the top, add import
from ring_pipeline import RingPipelineCoordinator

class LLMService:
    def __init__(self, network: "Node", use_sharding: bool = False, use_ring: bool = False):
        # ... existing init ...
        self.use_ring = use_ring
        self.ring_coordinator: Optional[RingPipelineCoordinator] = None
    
    async def start(self, model_id: str):
        """Start LLM service - MODIFIED for ring support."""
        # ... existing model initialization ...
        
        # NEW: Initialize ring if enabled
        if self.use_sharding and self.use_ring:
            logger.info("Initializing ring pipeline mode...")
            self.ring_coordinator = RingPipelineCoordinator(
                inference_engine=self.inference_engine,
                network_node=self.network
            )
            
            # Set up ring topology
            topology_nodes = self.network.topology.all_nodes()
            my_node_id = str(await self.network.iroh_node.net().node_id())
            
            if not topology_nodes:
                logger.warning("No topology nodes found, waiting for peers...")
                # Give time for peers to connect
                await asyncio.sleep(2.0)
                topology_nodes = self.network.topology.all_nodes()
            
            if topology_nodes:
                await self.ring_coordinator.initialize_ring(
                    topology_nodes=topology_nodes,
                    my_node_id=my_node_id,
                    model_total_layers=self.current_shard.n_layers
                )
                logger.info(f"Ring pipeline ready: rank={self.ring_coordinator.ring_position.rank}")
            else:
                logger.error("Cannot initialize ring: no peers found")
        
        # ... rest of existing code ...
    
    async def _handle_query(self, query: str) -> Dict:
        """Handle query - MODIFIED for ring support."""
        request_id = f"req-{int(time.time() * 1000)}"
        
        # NEW: Route to ring pipeline if enabled
        if self.use_ring and self.ring_coordinator:
            return await self._process_query_ring(query, request_id)
        elif self.use_sharding:
            return await self._process_query_sharded(query, request_id)
        else:
            return await self._process_query_single(query, request_id)
    
    async def _process_query_ring(self, query: str, request_id: str) -> Dict:
        """NEW: Process query using ring pipeline."""
        try:
            if not self.ring_coordinator.ring_position:
                return {"error": "Ring not initialized"}
            
            # Only head node starts generation
            if self.ring_coordinator.ring_position.is_head:
                logger.info(f"Head node starting ring inference for: {query[:50]}")
                
                generated_tokens = await self.ring_coordinator.start_inference(
                    request_id=request_id,
                    prompt=query,
                    shard=self.current_shard,
                    max_tokens=self.max_generate_tokens
                )
                
                # Decode tokens to text
                response_text = await self.inference_engine.decode(
                    self.current_shard,
                    np.array(generated_tokens)
                )
                
                result = {
                    "response": response_text,
                    "tokens": generated_tokens,
                    "node_id": str(await self.network.iroh_node.net().node_id()),
                    "mode": "ring_pipeline",
                    "rank": self.ring_coordinator.ring_position.rank
                }
                
                # Broadcast result
                await self.network.broadcast_message({
                    "type": "llm_response",
                    "payload": result
                })
                
                return result
            else:
                # Worker nodes participate in ring but don't initiate
                return {
                    "status": "worker",
                    "message": f"Worker node (rank {self.ring_coordinator.ring_position.rank}) waiting for ring messages",
                    "rank": self.ring_coordinator.ring_position.rank
                }
        
        except Exception as e:
            logger.error(f"Ring inference error: {e}", exc_info=True)
            return {"error": str(e)}
```

### 2. Update `node.py` Message Handler

Add this method to handle ring messages:

```python
# In Node class

async def handle_ring_tensor_message(self, message_data: Dict, llm_service):
    """Handle incoming ring tensor forward message."""
    try:
        sender_id = message_data.get("sender_id")
        request_id = message_data["payload"].get("request_id", "unknown")
        
        logger.debug(f"Handling ring tensor from {sender_id[:8]}")
        
        # Deserialize tensor
        import base64
        tensor_b64 = message_data["payload"]["tensor_data"]
        tensor_bytes = base64.b64decode(tensor_b64)
        tensor_shape = tuple(message_data["payload"]["tensor_shape"])
        tensor_dtype = np.dtype(message_data["payload"]["tensor_dtype"])
        is_final = message_data["payload"].get("is_final", False)
        
        tensor = np.frombuffer(tensor_bytes, dtype=tensor_dtype).reshape(tensor_shape)
        
        # Pass to ring coordinator
        if llm_service and llm_service.ring_coordinator:
            await llm_service.ring_coordinator.handle_incoming_tensor(
                sender_id=sender_id,
                request_id=request_id,
                tensor_data=tensor,
                shard=llm_service.current_shard,
                is_final=is_final
            )
        else:
            logger.warning("No ring coordinator to handle tensor")
    
    except Exception as e:
        logger.error(f"Error handling ring tensor: {e}", exc_info=True)
```

### 3. Update `main.py` CLI

Add ring flag to CLI:

```python
import typer

app = typer.Typer()

@app.command()
def start(
    bootstrap_ticket: Optional[str] = typer.Option(None, "--bootstrap-ticket"),
    use_ring: bool = typer.Option(False, "--ring", help="Enable ring pipeline mode")
):
    """Start HyperCluster node with optional ring pipeline."""
    async def run():
        # Start node
        node = Node()
        await node.start()
        
        # Start LLM service with ring mode
        llm_service = LLMService(node, use_sharding=True, use_ring=use_ring)
        
        # Register message handler
        async def message_handler(msg):
            msg_type = msg.get("type")
            
            # NEW: Handle ring tensor messages
            if msg_type == "ring_tensor_forward":
                await node.handle_ring_tensor_message(msg, llm_service)
            
            # ... other message handlers ...
        
        node.register_message_handler(message_handler)
        
        # ... rest of existing code ...
    
    asyncio.run(run())
```

### 4. Quick Test (Single Machine, 3 Terminals)

**Terminal 1: Head Node**
```bash
cd HyperCluster-v0.1
python main.py start --ring

# In REPL:
> llm start Qwen/Qwen2.5-0.5B-Instruct
# Copy the ticket that appears
```

**Terminal 2: Worker 1**
```bash
python main.py start --ring --bootstrap-ticket "<PASTE_TICKET>"

# In REPL:
> llm start Qwen/Qwen2.5-0.5B-Instruct
# This node auto-joins ring
```

**Terminal 3: Worker 2**
```bash
python main.py start --ring --bootstrap-ticket "<PASTE_TICKET>"

# In REPL:
> llm start Qwen/Qwen2.5-0.5B-Instruct
```

**Back to Terminal 1:**
```bash
# In REPL:
> llm query "What is a neural network?"
# Watch logs to see ring pipeline in action!
```

## Expected Log Output

### Head Node (Rank 0)
```
[INFO] Ring initialized: rank=0/3, layers=[0:21], prev=<node3>, next=<node1>
[INFO] Head node starting ring inference for: What is a neural network?
[DEBUG] Processing layers 0-21
[DEBUG] SEND: 0 → <node1>, request=req-123, shape=(1, 768)
[INFO] Received ring tensor back, sampling...
[DEBUG] Step 0: token=42
[INFO] Generated 45 tokens in 3.2s
```

### Worker Node 1 (Rank 1)
```
[INFO] Ring initialized: rank=1/3, layers=[22:42], prev=<node0>, next=<node2>
[DEBUG] RECV: <node0> → 1, request=req-123, shape=(1, 768)
[DEBUG] Processing layers 22-42
[DEBUG] SEND: 1 → <node2>, request=req-123, shape=(1, 768)
```

### Worker Node 2 (Rank 2)
```
[INFO] Ring initialized: rank=2/3, layers=[43:63], prev=<node1>, next=<node0>
[DEBUG] RECV: <node1> → 2, request=req-123, shape=(1, 768)
[DEBUG] Processing layers 43-63
[DEBUG] SEND: 2 → <node0>, request=req-123, shape=(1, 768) [FINAL]
```

## Troubleshooting

### Issue: "No topology nodes found"

**Cause**: Nodes haven't discovered each other yet

**Fix**:
```python
# In llm_service.py start(), add delay:
await asyncio.sleep(3.0)  # Give time for Iroh peer discovery
topology_nodes = self.network.topology.all_nodes()
```

### Issue: "Ring coordinator not initialized"

**Cause**: Forgot to pass `--ring` flag or topology empty

**Fix**:
```bash
# Always pass --ring flag:
python main.py start --ring

# And ensure peers are connected before starting LLM
```

### Issue: Tensors not being received

**Cause**: Message handler not registered for "ring_tensor_forward"

**Fix**: Verify in `main.py`:
```python
async def message_handler(msg):
    msg_type = msg.get("type")
    if msg_type == "ring_tensor_forward":  # Must handle this!
        await node.handle_ring_tensor_message(msg, llm_service)
```

### Issue: Worker nodes getting no messages

**Cause**: Document not shared between all nodes

**Fix**: Ensure all nodes join the same document:
```bash
# Head creates document, workers join with ticket
# Verify with: node.documents should have same doc_id
```

## Performance Tips

### 1. Use Smaller Models for Testing
```bash
# Start with tiny models to validate ring logic:
> llm start Qwen/Qwen2.5-0.5B-Instruct  # 500M params, fast
# Then scale up:
> llm start Qwen/Qwen2.5-7B-Instruct    # 7B params
```

### 2. Monitor Network Traffic
```python
# Add to _send_to_node:
size_mb = len(tensor_bytes) / 1024 / 1024
logger.info(f"Sending {size_mb:.2f} MB to {target_node_id[:8]}")
```

### 3. Profile Latency
```python
# In ring_pipeline.py _process_and_forward:
start = time.time()
output_data, _ = await self.inference_engine.infer_tensor(...)
latency_ms = (time.time() - start) * 1000
logger.info(f"Layer processing: {latency_ms:.1f}ms")
```

## Comparing to Prima.cpp

| Metric | Prima.cpp (4 devices) | HyperCluster (3 nodes) |
|--------|----------------------|------------------------|
| **Network** | ZeroMQ (TCP) | Iroh (QUIC/P2P) |
| **Serialization** | Raw bytes | Base64 JSON |
| **Token Latency** | ~90ms (QwQ-32B) | TBD (measure!) |
| **Memory Usage** | <10% pressure | TBD (measure!) |
| **Throughput** | 11 tok/s | TBD (measure!) |

**Goal**: Match prima.cpp's latency within 2-3x (accounting for Python overhead)

## Next: Real Prima.cpp Features

Once basic ring works, add these prima.cpp features:

### 1. Prefetching
```python
# Start prefetch worker
await ring_coordinator.schedule_prefetch(next_layer_id)
```

### 2. Multiple Cycles
```python
# Already implemented in calculate_cycles_needed()
# Automatically handles models with more layers than ring capacity
```

### 3. Compression
```python
# Add to _send_to_node:
import zlib
compressed = zlib.compress(tensor_bytes, level=1)
```

### 4. Layer-only Loading
```python
# Load only assigned layers instead of full model
# Requires model surgery (future work)
```

## Resources

- **ring_pipeline.py**: Core ring logic (newly created)
- **RING_PIPELINE_INTEGRATION.md**: Detailed architecture guide
- **Prima.cpp source**: Study `src/llama.cpp:17970-18367` for reference

## Getting Help

If stuck:
1. Check logs with `--log-level DEBUG`
2. Verify topology with `node.topology.all_nodes()`
3. Test without ring first (`--ring` off)
4. Ping on Discord/GitHub with log snippets

---

**Ready?** Start with the 3-terminal test above. You should see ring messages flowing within minutes! 🚀

