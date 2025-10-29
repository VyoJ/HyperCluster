# Changelog

## v0.2 - Ring Pipeline Integration (2025-01-29)

### Added
- **Ring Pipeline Architecture** inspired by prima.cpp
  - `ring_pipeline.py`: 548 lines of ring coordination logic
  - Multi-cycle inference support for large models
  - Memory-weighted layer assignment
  - Prefetching worker for overlapping I/O with computation
  
- **Documentation**
  - `RING_PIPELINE_INTEGRATION.md`: Detailed architecture guide (546 lines)
  - `RING_QUICKSTART.md`: Quick start guide (398 lines)
  - `INTEGRATION_COMPLETE.md`: Implementation summary
  
- **CLI Options**
  - `--ring` flag for enabling ring pipeline mode
  - Enhanced status display showing mode and rank

### Modified
- **llm_service.py**
  - Added `use_ring` parameter
  - Added `ring_coordinator` attribute
  - Added `_init_ring_pipeline()` method
  - Added `_process_query_ring()` method
  - Modified routing logic for ring mode

- **node.py**
  - Added `handle_ring_tensor_message()` method
  - Tensor deserialization for ring messages
  - Integration with ring coordinator

- **main.py**
  - Added `--ring` CLI flag
  - Modified `message_handler()` for ring messages
  - Added ring mode indicator
  - Enhanced LLM response display

- **README.md**
  - Added ring pipeline to features
  - Updated architecture section
  - Added new documentation links
  - Added prima.cpp credit

### Technical Details
- Ring topology with configurable ranks
- Automatic layer window calculation based on memory
- Base64 tensor serialization for Iroh transport
- Async message handling
- Support for multiple cycles per token

### Performance Targets
- Target: 2-3x prima.cpp latency (accounting for Python overhead)
- Expected QwQ-32B: ~180-270ms/token (vs prima.cpp's 90ms)
- Expected Llama-70B: ~1.3-2.0s/token (vs prima.cpp's 674ms)

### Usage
```bash
# Single node with ring
python main.py start --ring

# Multi-node ring
python main.py start --ring --bootstrap-ticket <TICKET>
```

### Known Limitations
- Prefetching not yet fully optimized (TODO: mmap-based)
- Layer loading is eager (TODO: lazy loading)
- Single request per ring at a time (TODO: batching)

---

## v0.1 - Initial Release

### Features
- Distributed AI inference with sharded execution
- P2P networking via Iroh
- Topology-aware partitioning
- Memory-weighted shard assignment
- HuggingFace Transformers backend
- Single-node and sharded modes

