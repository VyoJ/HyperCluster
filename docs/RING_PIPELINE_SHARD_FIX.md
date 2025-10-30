# Ring Pipeline Shard Integration Fix

## Problem

The ring pipeline was not correctly using node-specific shards for distributed inference. Instead of each node loading and executing only its assigned layers, all nodes were attempting to use the full model or receiving incorrect shard specifications.

## Root Cause

1. **LLM Service Shard Confusion**: The `llm_service.py` created a `base_shard` (full model 0-N layers) and got node-specific shards via `get_current_shard()`, but only stored the node-specific `current_shard`. When passing shards to the ring coordinator, it was inconsistent about which shard to use.

2. **Ring Coordinator Not Creating Node Shards**: The `ring_pipeline.py` received a shard parameter but used it directly for inference, rather than creating a node-specific shard based on the `layer_window` that had already been calculated during ring initialization.

## Solution

### 1. Store Both Base and Current Shards in LLM Service

**File**: `llm_service.py`

```python
# Before:
base_shard = Shard(...)  # Local variable
self.current_shard = await self.network.get_current_shard(base_shard)

# After:
self.base_shard = Shard(...)  # Instance variable for full model spec
self.current_shard = await self.network.get_current_shard(self.base_shard)
```

**Purpose**: 
- `base_shard`: Represents the complete model (layers 0 to N-1). Used for coordination and knowing total model structure.
- `current_shard`: This specific node's assigned layer range. Used for actually loading the model shard.

### 2. Pass Base Shard to Ring Operations

**File**: `llm_service.py`

```python
# In _process_query_ring():
generated_tokens = await self.ring_coordinator.start_inference(
    request_id=query_id,
    prompt=query,
    shard=self.base_shard,  # Full model spec, not node-specific shard
    max_tokens=self.max_generate_tokens
)

# In _on_topology_change():
await self.ring_coordinator.initialize_ring(
    topology_nodes=topology_nodes,
    my_node_id=my_node_id,
    model_total_layers=self.base_shard.n_layers  # Use base_shard for full model spec
)
```

**Purpose**: The ring coordinator needs to know about ALL layers to properly coordinate the ring, not just one node's layers.

### 3. Create Node-Specific Shards in Ring Pipeline

**File**: `ring_pipeline.py`

```python
# In _process_and_forward():

# Create a shard for this node's layer window
# This ensures only the assigned layers are loaded and executed
node_shard = Shard(
    model_id=shard.model_id,
    start_layer=self.layer_window.layer_start,
    end_layer=self.layer_window.layer_end,
    n_layers=shard.n_layers
)

# Run inference on assigned layers using the sharded model
output_data, new_state = await self.inference_engine.infer_tensor(
    request_id=request_id,
    shard=node_shard,  # Use node-specific shard, not full model shard
    input_data=current_data,
    inference_state=state.metadata
)
```

**Purpose**: Each node creates a shard representing only its assigned layer range from the `layer_window` that was set during `initialize_ring()`.

## Data Flow

### Before (Broken):
```
LLM Service:
  base_shard = Shard(0, 23, 24)  [local var, lost]
  current_shard = get_current_shard(base_shard)  [e.g., Shard(0, 7, 24)]

Ring Start:
  start_inference(shard=current_shard)  [Only node 1's layers!]

Ring Process:
  infer_tensor(shard=current_shard)  [Node 2 tries to use Node 1's shard]
```

### After (Fixed):
```
LLM Service:
  self.base_shard = Shard(0, 23, 24)  [stored]
  self.current_shard = Shard(0, 7, 24)  [Node 1's assignment]

Ring Start:
  start_inference(shard=base_shard)  [Full model spec]

Ring Process (Node 1):
  layer_window = LayerWindow(0, 7)  [from initialize_ring]
  node_shard = Shard(0, 7, 24)  [created from layer_window]
  infer_tensor(shard=node_shard)  [Correct layers]

Ring Process (Node 2):
  layer_window = LayerWindow(8, 15)  [from initialize_ring]
  node_shard = Shard(8, 15, 24)  [created from layer_window]
  infer_tensor(shard=node_shard)  [Correct layers]
```

## Key Concepts

### Base Shard
- Represents the complete model
- `start_layer=0, end_layer=n_layers-1`
- Used for:
  - Ring coordination
  - Understanding total model structure
  - Calculating layer assignments
  - Encoding/decoding (needs full tokenizer)

### Current Shard (Node-Specific)
- Represents this node's assigned layers
- `start_layer` and `end_layer` from topology partitioning
- Used for:
  - Actually loading the model into memory
  - Each node loads only its assigned layers

### Layer Window
- Calculated during `initialize_ring()` based on topology
- Stored in `ring_coordinator.layer_window`
- Defines which layers this node processes
- Used to create node_shard in `_process_and_forward()`

## Testing

After these changes, distributed inference should work correctly with:
1. Each node loading only its assigned layer range
2. Tensors flowing through the ring in order
3. Each node processing only its layer window
4. Proper coordination across all model layers

## Files Changed

1. `llm_service.py`:
   - Store `self.base_shard` instead of local `base_shard`
   - Pass `base_shard` to ring operations
   - Use `current_shard` for actual model loading

2. `ring_pipeline.py`:
   - Create `node_shard` from `layer_window` in `_process_and_forward()`
   - Update docstring to clarify shard parameter usage
   - Use `node_shard` for `infer_tensor()` calls
