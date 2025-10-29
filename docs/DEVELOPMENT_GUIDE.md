# Quick Start Guide - Sharded Inference Development

This guide will help you get started with developing and testing the sharded inference system in HyperCluster.

## Setup

### 1. Install Dependencies

```bash
cd hypercluster-v0.1
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## Testing Components

### Test 1: Basic Components

```bash
python test_sharded_inference.py
```

This will test:
- Shard creation and methods
- Device capability detection
- Partitioning strategies
- (Optional) Full inference engine

**Expected output:**
```
============================================================
HyperCluster Sharded Inference Test Suite
============================================================

Testing Shard Basics
Created shard: Shard(model=Qwen/Qwen2.5-0.5B-Instruct, layers=0-23/24)
Is first layer: True
Is last layer: True
Layer count: 24

Testing Device Capabilities
Detected capabilities: DeviceCapabilities(model=..., memory=16GB, flops=5.0TFLOPS)

Testing Partitioning Strategy
Topology:
  node1: 24GB
  node2: 16GB  
  node3: 8GB
Partitions:
  node1: 0.000 - 0.500
  node2: 0.500 - 0.833
  node3: 0.833 - 1.000
Shards:
  0: Shard(model=test-model, layers=0-11/24)
  1: Shard(model=test-model, layers=12-19/24)
  2: Shard(model=test-model, layers=20-23/24)

✅ All tests complete!
```

### Test 2: Single-Node Inference

```bash
python main.py start
```

In the REPL:
```
> llm start Qwen/Qwen2.5-0.5B-Instruct
> llm query What is the capital of France?
```

**Expected behavior:**
- Model downloads (~1GB first time)
- Shard assignment logged
- Query processed
- Response generated

## Development Workflow

### Adding a New Feature

1. **Create a branch** (if using git):
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Modify code** in relevant files:
   - Inference logic: `transformers_inference.py`
   - Node behavior: `node.py`
   - Service coordination: `llm_service.py`
   - Partitioning: `partitioning_strategy.py`

3. **Test your changes**:
   ```bash
   python test_sharded_inference.py
   ```

4. **Document** in appropriate `.md` file

### Common Development Tasks

#### Task: Add a New Partitioning Strategy

1. Create class in `partitioning_strategy.py`:
```python
class MyCustomStrategy(PartitioningStrategy):
    def partition(self, topology: Topology) -> List[Partition]:
        # Your logic here
        pass
```

2. Use in `node.py`:
```python
node = Node(partitioning_strategy=MyCustomStrategy())
```

#### Task: Add Support for a New Model Architecture

1. Extend `transformers_inference.py`:
```python
def _wrap_model_in_shard(self, model, shard: Shard):
    # Add architecture-specific logic
    if "gpt" in model.config.model_type:
        return self._wrap_gpt_model(model, shard)
    # ... existing code
```

2. Test with new model:
```python
llm_service = LLMService(node, model_name="gpt2")
```

#### Task: Optimize Tensor Serialization

1. Modify `node.py` `send_tensor()`:
```python
# Add compression
import zlib
tensor_bytes = zlib.compress(tensor.tobytes())
```

2. Add decompression in message handler

#### Task: Implement Multi-Node Forwarding

**This is the key missing piece!**

1. In `llm_service.py`, complete `_forward_to_next_shard()`:

```python
async def _forward_to_next_shard(self, request_id: str, tensor: np.ndarray, inference_state: Dict):
    """Forward tensor to the next shard in the sequence."""
    # Get current and next shard info
    base_shard = Shard(
        model_id=self.model_name,
        start_layer=0,
        end_layer=23,  # TODO: Get from config
        n_layers=24
    )
    
    # Get partitions
    partitions = self.network.partitioning_strategy.partition(self.network.topology)
    shards = map_partitions_to_shards(partitions, base_shard.n_layers, base_shard.model_id)
    
    # Find current shard index
    current_idx = None
    for i, shard in enumerate(shards):
        if shard == self.current_shard:
            current_idx = i
            break
    
    if current_idx is None or current_idx >= len(shards) - 1:
        logger.error("Cannot forward: current shard not found or is last")
        return
    
    # Get next shard and node
    next_shard = shards[current_idx + 1]
    next_partition = partitions[current_idx + 1]
    next_node_id = next_partition.node_id
    
    # Get document ID (assuming first document)
    doc_id = next(iter(self.network.documents.keys()))
    
    # Send tensor
    success = await self.network.send_tensor(
        doc_id=doc_id,
        target_node_id=next_node_id,
        shard=next_shard,
        tensor=tensor,
        request_id=request_id,
        inference_state=inference_state
    )
    
    if not success:
        logger.error(f"Failed to forward tensor to {next_node_id}")
```

2. In `main.py`, add message handler for incoming tensors:

```python
async def message_handler(message: dict):
    """Handles incoming messages from the network."""
    msg_type = message.get("type")
    
    # ... existing handlers ...
    
    elif msg_type == "tensor_forward":
        await handle_tensor_forward(message)
    elif msg_type == "prompt_forward":
        await handle_prompt_forward(message)

async def handle_tensor_forward(message: dict):
    """Handle incoming tensor from previous shard."""
    payload = message.get("payload", {})
    request_id = message.get("request_id")
    
    # Deserialize tensor
    import base64
    import numpy as np
    tensor_b64 = payload.get("tensor_data")
    tensor_shape = payload.get("tensor_shape")
    tensor_dtype = payload.get("tensor_dtype")
    
    tensor_bytes = base64.b64decode(tensor_b64)
    tensor = np.frombuffer(tensor_bytes, dtype=tensor_dtype).reshape(tensor_shape)
    
    # Get shard and inference state
    shard = Shard.from_dict(payload.get("shard"))
    inference_state = payload.get("inference_state", {})
    
    # Process through our shard
    if llm_service and llm_service.inference_engine:
        output, new_state = await llm_service.inference_engine.infer_tensor(
            request_id, shard, tensor, inference_state
        )
        
        # Forward to next or finish
        if shard.is_last_layer():
            await llm_service._handle_last_shard_output(request_id, output, new_state)
        else:
            await llm_service._forward_to_next_shard(request_id, output, new_state)
```

## Debugging Tips

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Shard Assignment

```python
# In Python REPL after starting node
node_id = await node.iroh_node.net().node_id()
print(f"Node ID: {node_id}")
print(f"Topology: {node.topology}")
print(f"Current shard: {llm_service.current_shard}")
```

### Monitor Memory Usage

```python
import psutil
import torch

# Check system memory
mem = psutil.virtual_memory()
print(f"RAM: {mem.used / 1e9:.1f}GB / {mem.total / 1e9:.1f}GB")

# Check GPU memory
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
```

### Trace Inference Flow

Add logging in key methods:
```python
logger.info(f"[{request_id}] Starting inference on shard {self.current_shard}")
logger.info(f"[{request_id}] Input shape: {tensor.shape}")
logger.info(f"[{request_id}] Output shape: {output.shape}")
```

## Performance Profiling

### Measure Inference Latency

```python
import time

start = time.time()
result = await llm_service.inference_engine.infer_tensor(...)
latency = time.time() - start
print(f"Inference latency: {latency*1000:.1f}ms")
```

### Profile with PyTorch

```python
with torch.profiler.profile() as prof:
    await llm_service.inference_engine.infer_tensor(...)
print(prof.key_averages().table())
```

## Troubleshooting

### Issue: Model won't download

**Solution**: Check internet connection and HuggingFace access:
```bash
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"
```

### Issue: CUDA out of memory

**Solution**: Reduce model size or use CPU:
```python
# In transformers_inference.py
device_map = "cpu"  # Force CPU
```

### Issue: Slow inference

**Possible causes:**
- CPU mode (expected)
- Large model
- No quantization

**Solutions:**
- Use GPU if available
- Enable quantization (requires bitsandbytes)
- Use smaller model

### Issue: Shard assignment fails

**Check:**
```python
print(f"Topology nodes: {node.topology.nodes}")
print(f"Partitions: {node.partitioning_strategy.partition(node.topology)}")
```

## Next Steps

1. **Complete multi-node forwarding** (see Task above)
2. **Test with 2+ nodes**
3. **Optimize tensor transfer** (compression, batching)
4. **Add error handling** (node failures, timeouts)
5. **Benchmark performance**

## Resources

- [SHARDED_INFERENCE.md](SHARDED_INFERENCE.md) - Full documentation
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Implementation details
- [exo GitHub](https://github.com/exo-explore/exo) - Original reference
- [Iroh Docs](https://iroh.computer/) - P2P networking
- [Transformers Docs](https://huggingface.co/docs/transformers/) - Model implementation

## Getting Help

- Check logs for error messages
- Review test output
- Consult documentation
- Search exo's codebase for reference implementation

Happy coding! 🚀
