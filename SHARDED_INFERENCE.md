# HyperCluster Sharded Inference Integration

This document describes the sharded inference system integrated into HyperCluster, adapted from the exo project.

## Overview

HyperCluster now supports **distributed AI inference** where large language models are split across multiple nodes in the network. Each node processes a portion of the model (a "shard"), and tensors are passed between nodes to complete inference.

### Key Features

- **Memory-Weighted Partitioning**: Models are split based on available memory on each node
- **Automatic Shard Assignment**: Nodes automatically determine which layers to process
- **P2P Tensor Forwarding**: Hidden states are passed between nodes via Iroh
- **Flexible Topology**: Supports dynamic node join/leave
- **Dual Mode**: Can run in single-node or distributed mode

## Architecture

### Core Components

1. **Shard** (`shard.py`)
   - Represents a contiguous range of model layers
   - Tracks: model_id, start_layer, end_layer, n_layers
   - Methods: `is_first_layer()`, `is_last_layer()`, `get_layer_count()`

2. **InferenceEngine** (`inference_engine.py`)
   - Abstract base class for inference engines
   - Methods: `encode()`, `decode()`, `infer_tensor()`, `sample()`
   - Handles shard loading and KV cache management

3. **TransformersShardedInferenceEngine** (`transformers_inference.py`)
   - Concrete implementation using HuggingFace Transformers
   - Loads only assigned layers
   - Manages per-request KV cache
   - Thread pools for async execution

4. **DeviceCapabilities** (`device_capabilities.py`)
   - Detects hardware: GPU (CUDA), CPU, memory, FLOPS
   - Auto-estimates performance based on hardware

5. **Topology** (`topology.py`)
   - Tracks network topology: nodes, connections, capabilities
   - Methods: `update_node()`, `add_edge()`, `merge()`

6. **PartitioningStrategy** (`partitioning_strategy.py`)
   - Splits model across nodes
   - `RingMemoryWeightedPartitioningStrategy`: proportional to memory
   - `UniformPartitioningStrategy`: equal distribution
   - Maps partitions to concrete shards

7. **Node** (`node.py`)
   - Extended with shard management
   - Methods: `get_current_shard()`, `send_tensor()`, `send_prompt()`
   - Tracks outstanding requests and inference states

8. **LLMService** (`llm_service.py`)
   - Orchestrates distributed inference
   - Handles first-shard inference start
   - Routes tensors between nodes
   - Manages token generation and sampling

## How It Works

### Inference Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Node 1    │────▶│   Node 2    │────▶│   Node 3    │
│ (Layers 0-7)│     │ (Layers 8-15)│     │(Layers 16-23)│
│  Embedding  │     │   Hidden    │     │   LM Head   │
└─────────────┘     └─────────────┘     └─────────────┘
      ▲                                         │
      │                                         │
      └─────────────────────────────────────────┘
              Token feedback for generation
```

#### Step-by-Step:

1. **Query Initiation**: User sends prompt to any node
2. **Routing**: Query routed to first-shard node
3. **First Shard**: 
   - Encodes prompt to tokens
   - Passes through embedding + assigned layers
   - Outputs hidden states
4. **Middle Shards**:
   - Receive hidden states from previous node
   - Process through assigned layers
   - Forward to next node
5. **Last Shard**:
   - Receives hidden states
   - Applies final layers + LM head
   - Produces logits
   - Samples next token
6. **Token Feedback** (for generation):
   - Token fed back to first shard
   - Process repeats until EOS or max tokens

### Memory Management

Each node loads **only its assigned layers**:

```python
# Example with 3 nodes, 24-layer model
Node 1 (16GB): Layers 0-11   (50% of layers)
Node 2 (8GB):  Layers 12-17  (25% of layers)  
Node 3 (8GB):  Layers 18-23  (25% of layers)
```

The partitioning is **automatic** based on detected memory.

## Usage

### Starting a Sharded Inference Node

```python
from node import Node
from llm_service import LLMService

# Create node with sharding enabled
node = Node()
await node.start()

# Create document or join existing
ticket, doc_id = await node.create_document()

# Start LLM service with sharding
llm_service = LLMService(node, model_name="Qwen/Qwen2.5-0.5B-Instruct", use_sharding=True)
await llm_service.start(num_layers=24)
```

### Configuration Options

```python
# Disable sharding (single-node mode)
llm_service = LLMService(node, use_sharding=False)

# Custom partitioning strategy
from partitioning_strategy import UniformPartitioningStrategy
node = Node(partitioning_strategy=UniformPartitioningStrategy())

# Different model
llm_service = LLMService(node, model_name="meta-llama/Llama-3.2-1B-Instruct")
```

### Querying the Network

```python
# Send query (automatically routes to correct shard sequence)
query_id = await llm_service.send_query("What is quantum computing?")

# Query will be processed across all nodes
# Responses come back as "llm_message" type messages
```

## Message Protocol

### Tensor Forward Message

```json
{
  "type": "tensor_forward",
  "sender_id": "node_id_abc123",
  "target_node_id": "node_id_def456",
  "request_id": "uuid-1234",
  "payload": {
    "shard": {
      "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
      "start_layer": 8,
      "end_layer": 15,
      "n_layers": 24
    },
    "tensor_data": "base64_encoded_numpy_array",
    "tensor_shape": [1, 32, 896],
    "tensor_dtype": "float32",
    "inference_state": {
      "hidden_size": 896
    }
  }
}
```

### Prompt Forward Message

```json
{
  "type": "prompt_forward",
  "sender_id": "node_id_abc123",
  "target_node_id": "node_id_def456",
  "request_id": "uuid-1234",
  "payload": {
    "shard": {...},
    "prompt": "What is quantum computing?",
    "inference_state": {}
  }
}
```

### Topology Update Message

```json
{
  "type": "topology_update",
  "sender_id": "node_id_abc123",
  "payload": {
    "capabilities": {
      "model": "NVIDIA RTX 4090",
      "chip": "NVIDIA",
      "memory": 24,
      "flops": 82.6
    },
    "topology": {
      "nodes": {...},
      "peer_graph": {...}
    }
  }
}
```

## Current Limitations & TODOs

### Implemented ✅

- [x] Shard abstraction and management
- [x] Device capability detection
- [x] Topology tracking
- [x] Memory-weighted partitioning
- [x] Transformers-based inference engine
- [x] Single-node sharded inference
- [x] Message protocol for tensor forwarding
- [x] LLM service integration

### In Progress / Future Work 🚧

- [ ] **Multi-node tensor forwarding**: Currently logs but doesn't forward
  - Need to implement routing logic
  - Handle tensor deserialization on receiving node
  - Coordinate request IDs across nodes

- [ ] **Layer extraction**: Currently loads full model
  - Implement true layer extraction to save memory
  - Load only assigned layers from checkpoint

- [ ] **Error handling**: Robust failure recovery
  - Handle node disconnections
  - Re-route inference on failure
  - Checkpoint/resume support

- [ ] **Performance optimization**:
  - Quantization support (4-bit, 8-bit)
  - Flash attention integration
  - Pipeline parallelism

- [ ] **Model support**:
  - Currently tested with Qwen/Llama architectures
  - Extend to other model families
  - Support for multimodal models

## Testing

### Single-Node Test

```bash
cd hypercluster-v0.1
python main.py start
```

```
# In the REPL:
> llm start Qwen/Qwen2.5-0.5B-Instruct
> llm query What is AI?
```

### Multi-Node Test (Future)

```bash
# Terminal 1 - First node
python main.py start
# Save the ticket

# Terminal 2 - Second node  
python main.py start --bootstrap-ticket <ticket>

# Both nodes will:
# 1. Detect their capabilities
# 2. Build topology
# 3. Partition model
# 4. Load assigned shards
# 5. Coordinate inference
```

## Performance Considerations

### Memory Usage

- Each node loads only its shard: `~(total_model_size / num_nodes)`
- KV cache per request: `~(2 * num_layers_per_shard * hidden_size * seq_len * batch_size)`
- Model cache directory: Set via `TransformersShardedInferenceEngine(cache_dir="./models")`

### Latency

- **Single-node**: Normal inference latency
- **Multi-node**: `base_latency + (network_latency * num_shards)`
- Network latency depends on Iroh routing (typically <10ms on LAN)

### Throughput

- Increases with more nodes (parallel requests)
- Limited by slowest node in the chain
- Best with balanced capabilities

## References

- **exo**: Original sharded inference implementation
  - [GitHub](https://github.com/exo-explore/exo)
- **Iroh**: P2P networking layer
  - [Docs](https://iroh.computer/)
- **HuggingFace Transformers**: Model implementation
  - [Docs](https://huggingface.co/docs/transformers/)

## Contributing

To extend the sharded inference system:

1. **Add new partitioning strategies**: Implement `PartitioningStrategy`
2. **Add new inference engines**: Implement `InferenceEngine` (e.g., MLX, TinyGrad)
3. **Optimize tensor forwarding**: Implement compression, batching
4. **Add monitoring**: Track latency, throughput, resource usage

See the source code for detailed implementation notes.
