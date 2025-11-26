# HyperCluster

**HyperCluster** is a distributed AI inference system designed to run large language models (LLMs) across a peer-to-peer (P2P) network of consumer devices. Built on top of [Iroh](https://iroh.computer/) for networking and [HuggingFace Transformers](https://huggingface.co/docs/transformers/) for inference, HyperCluster enables efficient model execution by sharding models across multiple nodes or utilizing a ring pipeline architecture.

## Key Features

*   **Distributed Inference**: Run models that are too large for a single device by splitting them across multiple nodes.
*   **P2P Networking**: Decentralized architecture using Iroh, allowing for dynamic node discovery and communication.
*   **Dynamic Sharding**: Automatically partitions model layers based on the available memory and compute capabilities of each node.
*   **Ring Pipeline Architecture**: Implements a high-performance ring topology (inspired by `prima.cpp`) for pipelined inference, optimizing throughput.
*   **Fault Tolerance**: Handles node failures and topology changes dynamically.
*   **Interactive REPL**: Built-in command-line interface for managing the cluster and running queries.

## Architecture

HyperCluster operates on a decentralized mesh where each participant runs a **Node**.

*   **Node**: The fundamental unit of the cluster. It manages network connections, discovers peers, and advertises device capabilities (Memory, FLOPs).
*   **Topology**: The cluster maintains a real-time map of all connected nodes and their capabilities to make intelligent sharding decisions.
*   **Sharding**: Models are split into "shards" (groups of layers). Each node is assigned a shard to execute.
*   **Inference Engine**:
    *   **Standard Sharded**: Sequential execution where activations are passed from one node to the next.
    *   **Ring Pipeline**: A circular topology where tokens circulate, allowing for pipelined processing and prefetching.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd HyperCluster
    ```

2.  **Install dependencies:**
    It is recommended to use [`uv`](https://docs.astral.sh/uv/)
    ```bash
    uv sync
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

## Usage

HyperCluster operates as a peer-to-peer mesh. To form a cluster, one node acts as the initial "coordinator" (creates the network), and other nodes join it.

### 1. Start the First Node (Coordinator)

Run the following command on your primary machine. This will create a new network document and generate a **Bootstrap Ticket**.

```bash
python main.py start --ring
```

**Output:**
```text
Created main document. Share this ticket:
docaaac...<LONG_TICKET_STRING>...
```
*Copy this ticket string. You will need it to connect other nodes.*

### 2. Join Other Nodes (Workers)

On your other devices (workers), run the start command with the `--bootstrap-ticket` flag using the ticket from the first node.

```bash
# Replace <TICKET> with the string copied from the first node
python main.py start --bootstrap-ticket "docaaac..."
```

*Note: Ensure all nodes use the same mode (Standard or Ring) for best compatibility.*

### 3. Verify Connectivity

Once nodes are running, you can verify they see each other using the interactive REPL.

```bash
# List connected peers
> peers

# Check your node status
> status
```

### 4. Start the LLM Service

You need to initialize the model on the cluster. You can do this from any node (usually the coordinator).

```bash
# Start with a specific model (downloads and shards automatically)
> llm start Qwen/Qwen2.5-0.5B-Instruct
```

*   The system will automatically detect connected peers.
*   It will calculate available memory on each node.
*   It will partition the model layers and assign shards to each node.

### 5. Run Inference

Once the service is started (you'll see "LLM service started" confirmation), you can run queries.

```bash
> llm query "Explain the theory of relativity in one sentence."
```

### 6. Other Commands

The REPL supports several other utility commands:

*   `llm services`: List all nodes currently participating in the LLM inference.
*   `text <message>`: Broadcast a chat message to all nodes (useful for debugging connectivity).
*   `store <key> <value>`: Save data to the shared distributed document.
*   `get <key>`: Retrieve data from the shared document.
*   `exit`: Shutdown the node.

## Documentation

Detailed documentation for specific components can be found in the `docs/` directory:

*   [**Sharded Inference Guide**](docs/Sharded_Inference.md): Deep dive into how sharding works.
*   [**Troubleshooting**](docs/Troubleshooting.md): Common issues and fixes.

---

**Acknowledgments:**
*   Networking: [Iroh](https://iroh.computer/)
*   Inference: [HuggingFace Transformers](https://huggingface.co/)
*   Inspiration: [exo](https://github.com/exo-explore/exo) (Sharding), [prima.cpp](https://github.com/Lizonghang/prima.cpp) (Ring Pipeline)
