# HyperCluster v0.1

A distributed AI inference system built on Iroh's P2P networking layer.

## Features

- **Distributed AI Inference**: Run large language models across multiple nodes
- **Sharded Model Execution**: Automatically split models based on available memory
- **Ring Pipeline Inference**: Prima.cpp-inspired ring architecture with prefetching
- **P2P Networking**: Decentralized communication via Iroh
- **Document Synchronization**: Real-time data sync across nodes
- **Topology-Aware Partitioning**: Memory-weighted model sharding
- **Multiple Inference Modes**: Single-node, sharded, or ring pipeline execution

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running a Node

Standard mode:
```bash
python main.py start
```

Ring pipeline mode (prima.cpp-inspired):
```bash
python main.py start --ring
```

### Using Sharded Inference

See [SHARDED_INFERENCE.md](SHARDED_INFERENCE.md) for detailed documentation.

```python
# In the HyperCluster REPL:
> llm start Qwen/Qwen2.5-0.5B-Instruct
> llm query What is quantum computing?
```

## Architecture

HyperCluster integrates sharded inference capabilities adapted from the [exo](https://github.com/exo-explore/exo) project and ring pipeline architecture inspired by [prima.cpp](https://github.com/Lizonghang/prima.cpp):

- **Shard Management**: Automatic layer assignment based on device capabilities
- **Ring Pipeline**: Multi-cycle ring topology for efficient distributed inference
- **Inference Engine**: HuggingFace Transformers backend with memory optimization
- **Topology Tracking**: Real-time network topology and capability management
- **Partitioning Strategies**: Memory-weighted or uniform distribution
- **Prefetching**: Overlap disk I/O with computation for better performance

## Testing

Run the test suite:

```bash
python test_sharded_inference.py
```

## Documentation

- [SHARDED_INFERENCE.md](SHARDED_INFERENCE.md) - Detailed documentation on distributed inference
- [RING_PIPELINE_INTEGRATION.md](RING_PIPELINE_INTEGRATION.md) - Ring pipeline architecture guide
- [RING_QUICKSTART.md](RING_QUICKSTART.md) - Quick start guide for ring pipeline
- [main.py](main.py) - Main entry point and CLI interface
- [node.py](node.py) - Network node with shard management
- [llm_service.py](llm_service.py) - LLM service with distributed inference
- [ring_pipeline.py](ring_pipeline.py) - Ring pipeline coordinator

## Credits

- Sharded inference system adapted from [exo](https://github.com/exo-explore/exo)
- Ring pipeline architecture inspired by [prima.cpp](https://github.com/Lizonghang/prima.cpp)
- P2P networking powered by [Iroh](https://iroh.computer/)
- Model inference via [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
