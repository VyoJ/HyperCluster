# HyperCluster v0.1

A distributed AI inference system built on Iroh's P2P networking layer.

## Features

- **Distributed AI Inference**: Run large language models across multiple nodes
- **Sharded Model Execution**: Automatically split models based on available memory
- **P2P Networking**: Decentralized communication via Iroh
- **Document Synchronization**: Real-time data sync across nodes
- **Topology-Aware Partitioning**: Memory-weighted model sharding
- **Multiple Inference Modes**: Single-node or distributed execution

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running a Node

```bash
python main.py start
```

### Using Sharded Inference

See [SHARDED_INFERENCE.md](SHARDED_INFERENCE.md) for detailed documentation.

```python
# In the HyperCluster REPL:
> llm start Qwen/Qwen2.5-0.5B-Instruct
> llm query What is quantum computing?
```

## Architecture

HyperCluster integrates sharded inference capabilities adapted from the [exo](https://github.com/exo-explore/exo) project:

- **Shard Management**: Automatic layer assignment based on device capabilities
- **Inference Engine**: HuggingFace Transformers backend with memory optimization
- **Topology Tracking**: Real-time network topology and capability management
- **Partitioning Strategies**: Memory-weighted or uniform distribution

## Testing

Run the test suite:

```bash
python test_sharded_inference.py
```

## Documentation

- [SHARDED_INFERENCE.md](SHARDED_INFERENCE.md) - Detailed documentation on distributed inference
- [main.py](main.py) - Main entry point and CLI interface
- [node.py](node.py) - Network node with shard management
- [llm_service.py](llm_service.py) - LLM service with distributed inference

## Credits

- Sharded inference system adapted from [exo](https://github.com/exo-explore/exo)
- P2P networking powered by [Iroh](https://iroh.computer/)
- Model inference via [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
