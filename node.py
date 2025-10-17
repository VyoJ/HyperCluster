import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import iroh
import numpy as np
from device_capabilities import DeviceCapabilities, get_device_capabilities
from iroh import AddrInfoOptions, Iroh, LiveEventType, PublicKey, ShareMode
from partitioning_strategy import (
    PartitioningStrategy,
    RingMemoryWeightedPartitioningStrategy,
    map_partitions_to_shards,
)
from shard import Shard
from topology import Topology

logger = logging.getLogger(__name__)


class Node:
    def __init__(
        self,
        bootstrap_nodes: Optional[List[str]] = None,
        partitioning_strategy: Optional[PartitioningStrategy] = None,
    ):
        self.iroh_node: Optional[Iroh] = None
        self.bootstrap_nodes = bootstrap_nodes or []
        self.documents: Dict[str, iroh.Doc] = {}
        self.message_handlers: List[Callable] = []
        self.system_info = {}
        self.neighbors: Dict[str, Set[PublicKey]] = {}  # doc_id -> set of peer_ids

        # Sharded inference infrastructure
        self.topology: Topology = Topology()
        self.device_capabilities: DeviceCapabilities = DeviceCapabilities(
            model="Unknown", chip="Unknown", memory=4, flops=1.0
        )
        self.partitioning_strategy = (
            partitioning_strategy or RingMemoryWeightedPartitioningStrategy()
        )

        # Inference state tracking
        self.current_shard: Optional[Shard] = None
        self.outstanding_requests: Dict[str, str] = {}  # request_id -> status
        self.buffered_token_output: Dict[
            str, Tuple[List[int], bool]
        ] = {}  # request_id -> (tokens, is_finished)
        self.inference_states: Dict[str, Dict] = {}  # request_id -> inference_state

    async def start(self):
        """Start the Iroh node."""
        try:
            options = iroh.NodeOptions()
            options.enable_docs = True
            self.iroh_node = await Iroh.memory_with_options(options)
            node_id = await self.iroh_node.net().node_id()
            logger.info(f"Iroh node started with ID: {node_id}")

            # Detect device capabilities
            self.device_capabilities = await get_device_capabilities()
            logger.info(f"Device capabilities: {self.device_capabilities}")

            # Initialize topology with this node
            await self.update_topology()

            # If bootstrap nodes are provided, connect to them
            # Note: This now happens in main.py after node is created
            # for ticket in self.bootstrap_nodes:
            #     await self.join_document(ticket)

        except Exception as e:
            logger.error(f"Failed to start Iroh node: {e}")
            raise

    async def stop(self):
        """Stop the Iroh node."""
        if self.iroh_node:
            await self.iroh_node.node().shutdown()
            logger.info("Iroh node stopped.")

    def register_message_handler(self, handler: Callable):
        """Register a callback for incoming messages."""
        self.message_handlers.append(handler)

    def add_neighbor(self, doc_id: str, peer_id: PublicKey):
        if doc_id not in self.neighbors:
            self.neighbors[doc_id] = set()
        self.neighbors[doc_id].add(peer_id)
        logger.info(f"Peer {peer_id} came online for doc {doc_id}")

    def remove_neighbor(self, doc_id: str, peer_id: PublicKey):
        if doc_id in self.neighbors and peer_id in self.neighbors[doc_id]:
            self.neighbors[doc_id].remove(peer_id)
            logger.info(f"Peer {peer_id} went offline for doc {doc_id}")

    async def create_document(self) -> Optional[Tuple[str, str]]:
        """Create a new document and return its ticket and ID."""
        if not self.iroh_node:
            logger.error("Iroh node not started.")
            return None
        try:
            doc = await self.iroh_node.docs().create()
            ticket = await doc.share(
                ShareMode.WRITE, AddrInfoOptions.RELAY_AND_ADDRESSES
            )
            doc_id = str(doc.id())
            self.documents[doc_id] = doc
            logger.info(f"Created document with ID: {doc_id}")
            await self.subscribe_to_doc_events(doc)
            return str(ticket), doc_id
        except Exception as e:
            logger.error(f"Failed to create document: {e}")
            return None

    async def join_document(self, ticket_str: str) -> Optional[str]:
        """Join a document using a ticket and return its ID."""
        if not self.iroh_node:
            logger.error("Iroh node not started.")
            return None
        try:
            ticket = iroh.DocTicket(ticket_str)
            doc = await self.iroh_node.docs().join(ticket)
            doc_id = str(doc.id())
            self.documents[doc_id] = doc
            logger.info(f"Joined document with ID: {doc_id}")

            await self.subscribe_to_doc_events(doc)
            return doc_id
        except Exception as e:
            logger.error(f"Failed to join document: {e}")
            return None

    async def subscribe_to_doc_events(self, doc: iroh.Doc):
        """Subscribe to events for a given document."""
        doc_id_str = str(doc.id())

        class SubscribeCallback:
            def __init__(self, outer_instance, doc_id):
                self.outer = outer_instance
                self.doc_id = doc_id

            async def event(self, event):
                if event.type() == LiveEventType.CONTENT_READY:
                    hash_val = event.as_content_ready()
                    await self.outer.handle_content_ready(doc, hash_val)
                elif event.type() == LiveEventType.NEIGHBOR_UP:
                    peer_id = event.as_neighbor_up()
                    self.outer.add_neighbor(self.doc_id, peer_id)
                elif event.type() == LiveEventType.NEIGHBOR_DOWN:
                    peer_id = event.as_neighbor_down()
                    self.outer.remove_neighbor(self.doc_id, peer_id)

        callback = SubscribeCallback(self, doc_id_str)
        await doc.subscribe(callback)

    async def handle_content_ready(self, doc: iroh.Doc, content_hash: iroh.Hash):
        """Handle new content received in a document."""
        try:
            content = await self.iroh_node.blobs().read_to_bytes(content_hash)
            message_data = json.loads(content.decode("utf-8"))

            for handler in self.message_handlers:
                await handler(message_data)

        except Exception as e:
            logger.error(f"Error processing new content: {e}")

    async def send_message(self, doc_id: str, message: Dict[str, Any]):
        """Send a message by writing it to a document."""
        if doc_id not in self.documents:
            logger.error(f"Not part of document {doc_id}")
            return False

        doc = self.documents[doc_id]
        # Use the default author for simplicity
        author = await self.iroh_node.authors().default()

        try:
            key = f"message-{time.time()}".encode("utf-8")
            payload = json.dumps(message).encode("utf-8")
            await doc.set_bytes(author, key, payload)
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast a message to all joined documents."""
        success_count = 0
        for doc_id in self.documents:
            if await self.send_message(doc_id, message):
                success_count += 1
        return success_count > 0

    async def store_value(self, key: str, value: Any) -> bool:
        """Store a key-value pair in the first available document."""
        if not self.documents:
            logger.error("No documents to store value in.")
            return False

        doc_id = next(iter(self.documents))

        # For storing values, we use the key directly in the document
        author = await self.iroh_node.authors().default()
        doc = self.documents[doc_id]
        try:
            payload = json.dumps(value).encode("utf-8")
            await doc.set_bytes(author, key.encode("utf-8"), payload)
            return True
        except Exception as e:
            logger.error(f"Failed to store value: {e}")
            return False

    async def retrieve_value(self, key: str) -> Optional[Any]:
        """Retrieve a value by searching through documents."""
        if not self.documents:
            return None

        # Check the first document for the key
        doc = next(iter(self.documents.values()))
        try:
            author = await self.iroh_node.authors().default()
            query = iroh.Query.author_key_exact(author, key.encode("utf-8"))
            entry = await doc.get_one(query)
            if entry:
                content = await self.iroh_node.blobs().read_to_bytes(
                    entry.content_hash()
                )
                return json.loads(content.decode("utf-8"))
        except Exception as e:
            logger.error(f"Error retrieving value from doc {doc.id()}: {e}")
        return None

    # ===== Sharded Inference Methods =====

    async def update_topology(self):
        """Update topology with this node's capabilities and neighbors."""
        # Get this node's ID
        if not self.iroh_node:
            return

        node_id = str(await self.iroh_node.net().node_id())

        # Update this node in topology
        self.topology.update_node(node_id, self.device_capabilities)

        # Update edges based on neighbors
        for doc_id, peers in self.neighbors.items():
            for peer_id in peers:
                self.topology.add_edge(node_id, str(peer_id), f"doc:{doc_id[:8]}")

    async def get_current_shard(self, base_shard: Shard) -> Optional[Shard]:
        """
        Get the shard assigned to this node based on current topology.

        Args:
            base_shard: The base model shard (full model spec)

        Returns:
            The shard assigned to this node, or None if not assigned
        """
        if not self.iroh_node:
            return None

        node_id = str(await self.iroh_node.net().node_id())

        # Get partitions from strategy
        partitions = self.partitioning_strategy.partition(self.topology)

        # Find our partition index
        partition_index = None
        for i, partition in enumerate(partitions):
            if partition.node_id == node_id:
                partition_index = i
                break

        if partition_index is None:
            logger.warning(f"Node {node_id} not found in partitions")
            return None

        # Map partitions to shards
        shards = map_partitions_to_shards(
            partitions, base_shard.n_layers, base_shard.model_id
        )

        return shards[partition_index] if partition_index < len(shards) else None

    async def send_tensor(
        self,
        doc_id: str,
        target_node_id: str,
        shard: Shard,
        tensor: np.ndarray,
        request_id: str,
        inference_state: Optional[Dict] = None,
    ):
        """
        Send tensor data to another node for distributed inference.

        Args:
            doc_id: Document ID to use for communication
            target_node_id: Target node's ID
            shard: The shard to process
            tensor: Tensor data (as numpy array)
            request_id: Unique identifier for this request
            inference_state: Optional state to maintain across nodes
        """
        if doc_id not in self.documents:
            logger.error(f"Not part of document {doc_id}")
            return False

        # Serialize tensor
        import base64

        tensor_bytes = tensor.tobytes()
        tensor_b64 = base64.b64encode(tensor_bytes).decode("utf-8")

        message = {
            "type": "tensor_forward",
            "sender_id": str(await self.iroh_node.net().node_id()),
            "target_node_id": target_node_id,
            "request_id": request_id,
            "payload": {
                "shard": shard.to_dict(),
                "tensor_data": tensor_b64,
                "tensor_shape": list(tensor.shape),
                "tensor_dtype": str(tensor.dtype),
                "inference_state": inference_state or {},
            },
            "timestamp": time.time(),
        }

        return await self.send_message(doc_id, message)

    async def send_prompt(
        self,
        doc_id: str,
        target_node_id: str,
        shard: Shard,
        prompt: str,
        request_id: str,
        inference_state: Optional[Dict] = None,
    ):
        """
        Send prompt to another node for distributed inference.

        Args:
            doc_id: Document ID to use for communication
            target_node_id: Target node's ID
            shard: The shard to process
            prompt: Text prompt
            request_id: Unique identifier for this request
            inference_state: Optional state to maintain across nodes
        """
        if doc_id not in self.documents:
            logger.error(f"Not part of document {doc_id}")
            return False

        message = {
            "type": "prompt_forward",
            "sender_id": str(await self.iroh_node.net().node_id()),
            "target_node_id": target_node_id,
            "request_id": request_id,
            "payload": {
                "shard": shard.to_dict(),
                "prompt": prompt,
                "inference_state": inference_state or {},
            },
            "timestamp": time.time(),
        }

        return await self.send_message(doc_id, message)

    async def broadcast_topology_update(self):
        """Broadcast this node's topology information to all documents."""
        if not self.iroh_node:
            return

        node_id = str(await self.iroh_node.net().node_id())

        message = {
            "type": "topology_update",
            "sender_id": node_id,
            "payload": {
                "capabilities": self.device_capabilities.to_dict(),
                "topology": self.topology.to_json(),
            },
            "timestamp": time.time(),
        }

        await self.broadcast_message(message)
