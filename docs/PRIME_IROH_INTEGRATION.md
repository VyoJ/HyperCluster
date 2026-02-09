# Prime-Iroh Integration

This document describes the integration of [prime-iroh](https://github.com/PrimeIntellect-ai/prime-iroh) with HyperCluster for optimized P2P tensor transfers.

## Overview

HyperCluster now supports two communication backends for tensor transfers during distributed inference:

1. **Document-based** (default): Uses Iroh's document sync for message passing and tensor storage
2. **Prime-Iroh** (optional): Direct P2P streaming communication optimized for pipeline parallelism

## What is Prime-Iroh?

Prime-Iroh is an asynchronous P2P communication library built on top of Iroh, specifically designed for distributed AI model training and inference. It provides:

- **Direct P2P Streaming**: Bypasses document sync for large tensor data
- **Asynchronous Operations**: PyTorch-like `isend`/`irecv` API for non-blocking communication
- **Pipeline Optimization**: Designed for pipeline parallel workloads with overlapping computation and communication
- **Reliability**: Built on Iroh's networking stack with NAT traversal and automatic fallback

## When to Use Prime-Iroh

**Use Prime-Iroh when:**
- Running ring pipeline mode with large models (>1B parameters)
- You need maximum throughput for tensor transfers
- Network latency is critical for your workload
- You have multiple nodes with high-bandwidth connections

**Use Document-based when:**
- Running with small models or few nodes
- Simplicity is more important than maximum performance
- You're just getting started with HyperCluster
- Prime-iroh package is not available in your environment

## Installation

Prime-iroh is included as an optional dependency. To install it:

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install prime-iroh>=0.3.1
```

## Usage

### Starting with Prime-Iroh

To enable prime-iroh backend, add the `--prime-iroh` flag when starting your node:

```bash
# Start coordinator with ring pipeline and prime-iroh
python main.py start --ring --prime-iroh

# Join workers with prime-iroh
python main.py start --bootstrap-ticket "docaaac..." --ring --prime-iroh
```

**Important**: All nodes in the cluster should use the same communication backend (either all with `--prime-iroh` or all without).

### Fallback Behavior

If prime-iroh is not available or fails to initialize:
- The node will automatically fall back to document-based communication
- A warning message will be displayed
- The cluster will continue to function normally

## Performance Comparison

Based on the design:

| Feature | Document-based | Prime-Iroh |
|---------|---------------|------------|
| Setup Complexity | Simple | Moderate |
| Small tensors (<10MB) | Fast | Fast |
| Large tensors (>100MB) | Moderate | **Very Fast** |
| Latency overhead | ~100-200ms | ~10-50ms |
| Best for | General use, testing | Production, large models |

## Architecture

### Document-based Communication Flow

```
Node A → [Serialize tensor] → Store in Document → Sync to peers → Node B reads
```

### Prime-Iroh Communication Flow

```
Node A → [Serialize tensor] → Direct stream → Node B receives
```

Prime-iroh establishes direct connections between nodes in the ring, bypassing the document sync layer for large tensor data while still using documents for control messages and metadata.

## Configuration

The integration is controlled by:

1. **Command-line flag**: `--prime-iroh` when starting the node
2. **Automatic detection**: System checks if prime-iroh is installed
3. **Runtime fallback**: Falls back to document-based if prime-iroh fails

## Implementation Details

### Code Structure

- `prime_iroh_backend.py`: Backend implementation for prime-iroh communication
- `node.py`: Extended to support prime-iroh initialization and teardown
- `ring_pipeline.py`: Updated `_send_to_node()` to use prime-iroh when available
- `main.py`: Added command-line option and status display

### API Usage

The integration uses prime-iroh's async API:

```python
# Send tensor
send_work = prime_node.isend(tensor_data, target_peer_id)
await send_work.wait()

# Receive tensor
recv_work = prime_node.irecv()
tensor_data = await recv_work.wait()
```

## Troubleshooting

### Prime-iroh not available

**Symptom**: Warning message "Prime-iroh requested but not available"

**Solution**: Install prime-iroh package:
```bash
pip install prime-iroh>=0.3.1
```

### Connection failures with prime-iroh

**Symptom**: Nodes can't communicate when using prime-iroh

**Solution**: 
1. Ensure all nodes use `--prime-iroh` flag
2. Check network connectivity between nodes
3. Try without `--prime-iroh` to verify basic connectivity
4. Check logs for specific error messages

### Performance not improved

**Symptom**: No speed improvement with prime-iroh

**Possible causes**:
- Tensors are too small to benefit from streaming
- Network is the bottleneck, not the communication protocol
- Only one node in the cluster

## Future Enhancements

Potential improvements for the integration:

1. **Automatic peer discovery**: Use topology to automatically configure send/receive peers
2. **Hybrid mode**: Use prime-iroh for large tensors, documents for small messages
3. **Compression**: Add tensor compression for network-constrained environments
4. **Metrics**: Add performance metrics comparing both backends

## References

- [Prime-Iroh GitHub](https://github.com/PrimeIntellect-ai/prime-iroh)
- [Prime-Iroh PyPI](https://pypi.org/project/prime-iroh/)
- [Iroh Documentation](https://iroh.computer/)
- [HyperCluster Ring Pipeline](./SHARDED_INFERENCE.md)
