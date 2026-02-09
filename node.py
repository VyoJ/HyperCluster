import asyncio
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
        use_prime_iroh: bool = False,
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

        # Cache for binary tensor data (hash -> content)
        self.tensor_cache: Dict[str, bytes] = {}  # hash_str -> binary_content
        
        # Prime-iroh backend (optional, for optimized tensor transfers)
        self.use_prime_iroh = use_prime_iroh
        self.prime_iroh_backend = None

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
            
            # Initialize prime-iroh backend if enabled
            if self.use_prime_iroh:
                try:
                    from prime_iroh_backend import PrimeIrohBackend, is_prime_iroh_available
                    
                    if is_prime_iroh_available():
                        self.prime_iroh_backend = PrimeIrohBackend()
                        # TODO: For ring pipeline, we should determine send_peer_id and recv_peer_id
                        # from the ring topology once it's established. For now, initialize without peers.
                        await self.prime_iroh_backend.initialize(str(node_id))
                        logger.info("✨ Prime-iroh backend enabled for optimized tensor transfers")
                    else:
                        logger.warning("Prime-iroh requested but not available, falling back to document-based communication")
                        self.use_prime_iroh = False
                except Exception as e:
                    logger.warning(f"Failed to initialize prime-iroh backend: {e}, falling back to document-based communication")
                    self.use_prime_iroh = False

        except Exception as e:
            logger.error(f"Failed to start Iroh node: {e}")
            raise

    async def stop(self):
        """Stop the Iroh node."""
        # Shutdown prime-iroh backend if active
        if self.prime_iroh_backend:
            await self.prime_iroh_backend.shutdown()
            
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

            # Use RELAY_AND_ADDRESSES to ensure connectivity
            ticket = await doc.share(
                ShareMode.WRITE, AddrInfoOptions.RELAY_AND_ADDRESSES
            )
            doc_id = str(doc.id())
            self.documents[doc_id] = doc
            logger.info(f"Created document with ID: {doc_id[:16]}...")
            logger.info("Document sharing mode: WRITE with RELAY_AND_ADDRESSES")

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
            logger.info(f"Joined document with ID: {doc_id[:16]}...")

            # Give Iroh time to establish sync with peers
            logger.info("⏳ Waiting 2s for document sync to stabilize...")
            await asyncio.sleep(2.0)

            await self.subscribe_to_doc_events(doc)
            logger.info("✅ Document joined and ready")
            return doc_id
        except Exception as e:
            logger.error(f"Failed to join document: {e}")
            return None

    async def subscribe_to_doc_events(self, doc: iroh.Doc):
        """Subscribe to events for a given document."""
        doc_id_str = str(doc.id())
        logger.info(f"🔔 Subscribing to events for document {doc_id_str[:16]}...")

        class SubscribeCallback:
            def __init__(self, outer_instance, doc_instance, doc_id):
                self.outer = outer_instance
                self.doc = doc_instance  # Store doc reference to avoid closure issues
                self.doc_id = doc_id

            async def event(self, event):
                event_type = event.type()
                logger.debug(
                    f"🔔 Event received: {event_type} for doc {self.doc_id[:16]}..."
                )

                if event_type == LiveEventType.CONTENT_READY:
                    hash_val = event.as_content_ready()
                    await self.outer.handle_content_ready(self.doc, hash_val)
                elif event_type == LiveEventType.NEIGHBOR_UP:
                    peer_id = event.as_neighbor_up()
                    logger.info(f"👋 Neighbor UP: {peer_id}")
                    self.outer.add_neighbor(self.doc_id, peer_id)
                elif event_type == LiveEventType.NEIGHBOR_DOWN:
                    peer_id = event.as_neighbor_down()
                    logger.info(f"👋 Neighbor DOWN: {peer_id}")
                    self.outer.remove_neighbor(self.doc_id, peer_id)

        callback = SubscribeCallback(self, doc, doc_id_str)
        await doc.subscribe(callback)
        logger.info(f"✅ Subscribed to document {doc_id_str[:16]}...")

    async def handle_content_ready(self, doc: iroh.Doc, content_hash: iroh.Hash):
        """Handle new content received in a document."""
        try:
            hash_str = str(content_hash)
            logger.debug(f"📦 Content ready event received, hash={hash_str[:16]}...")
            content = await self.iroh_node.blobs().read_to_bytes(content_hash)
            content_size = len(content)
            logger.info(
                f"📦 Read {content_size} bytes from blob (hash={hash_str[:16]}...)"
            )

            # Try to decode as JSON
            try:
                message_data = json.loads(content.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                # This is binary data (e.g., tensor), not a JSON message
                # Cache it by HASH (not size, to avoid collisions)
                logger.debug(
                    f"📦 Caching binary content: {content_size} bytes, hash={hash_str[:16]}..."
                )
                self.tensor_cache[hash_str] = content
                return

            # DIAGNOSTIC: Log received content with size
            msg_type = message_data.get("type", "unknown")
            sender = message_data.get("sender_id", "unknown")
            sender_short = sender[:16] if sender and len(sender) > 16 else sender

            # Highlight large messages (likely tensor forwards)
            if content_size > 100000:  # > 100KB
                logger.info(
                    f"📨 ⚡ LARGE MESSAGE: type={msg_type}, from={sender_short}..., size={content_size/1024/1024:.2f}MB"
                )
            else:
                logger.debug(
                    f"📨 Content ready: type={msg_type}, from={sender_short}..., size={content_size} bytes"
                )

            logger.debug(
                f"📨 Calling {len(self.message_handlers)} message handler(s)..."
            )
            for handler in self.message_handlers:
                await handler(message_data)

        except Exception as e:
            logger.error(f"Error processing new content: {e}", exc_info=True)

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
            payload_json = json.dumps(message)
            payload = payload_json.encode("utf-8")
            payload_size = len(payload)

            msg_type = message.get("type", "unknown")

            # Highlight large messages
            if payload_size > 100000:  # > 100KB
                logger.info(
                    f"📤 Writing LARGE message to doc {doc_id[:16]}...: type={msg_type}, size={payload_size/1024/1024:.2f}MB"
                )
            else:
                logger.debug(
                    f"📤 Writing to doc {doc_id[:16]}...: type={msg_type}, size={payload_size} bytes"
                )

            await doc.set_bytes(author, key, payload)

            logger.debug("✅ Successfully wrote message to document")

            if payload_size > 10000:  # Only for moderately large metadata
                await asyncio.sleep(0.1)

            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}", exc_info=True)
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

    async def handle_ring_tensor_message(self, message_data: Dict, llm_service):
        """
        Handle incoming ring tensor forward message.

        This fetches the tensor blob and passes it to the ring coordinator.
        Uses Iroh blobs for efficient large data transfer.
        """
        try:
            sender_id = message_data.get("sender_id")
            target_node_id = message_data.get("target_node_id")
            payload = message_data.get("payload", {})
            request_id = message_data.get("request_id", "unknown")

            # Check if this message is for us
            my_node_id = str(await self.iroh_node.net().node_id())

            logger.info(
                f"   Checking target: target={target_node_id[:16] if target_node_id else 'broadcast'}..., me={my_node_id[:16]}..."
            )

            # Only process if:
            # 1. No target specified (broadcast), OR
            # 2. We are the target
            if target_node_id and target_node_id != my_node_id:
                logger.info(
                    f"   ↩️  Ignoring ring tensor meant for {target_node_id[:16]}... (I am {my_node_id[:16]}...)"
                )
                return

            logger.info("")
            logger.info("📨 RECEIVED RING TENSOR MESSAGE")
            logger.info(f"   From: {sender_id[:16] if sender_id else 'unknown'}...")
            logger.info(f"   Request ID: {request_id}")

            # Check if LLM service and ring coordinator are available
            if not llm_service:
                logger.warning("   ⚠️  No LLM service available")
                return

            if not llm_service.ring_coordinator:
                logger.warning("   ⚠️  No ring coordinator available")
                return

            if not llm_service.is_running:
                logger.warning("   ⚠️  LLM service not running")
                return

            # Get tensor key and metadata
            tensor_key_str = payload.get("tensor_key", "")
            tensor_hash_str = payload.get("tensor_hash", "")  # The actual blob hash!

            if not tensor_key_str:
                logger.error("   ❌ No tensor key in message")
                return

            tensor_shape = tuple(payload.get("tensor_shape", []))
            tensor_dtype = np.dtype(payload.get("tensor_dtype", "float32"))
            tensor_size = payload.get("tensor_size", 0)
            is_final = payload.get("is_final", False)

            logger.info(f"   Tensor key: {tensor_key_str[:32]}...")
            logger.info(f"   Tensor hash: {tensor_hash_str[:16]}...")
            logger.info(
                f"   Tensor shape: {tensor_shape}, dtype: {tensor_dtype}, size: {tensor_size/1024/1024:.2f}MB"
            )
            logger.info(f"   Is final: {is_final}")

            # Extract position_ids and attention_mask from payload (CRITICAL!)
            position_ids_list = payload.get("position_ids")
            attention_mask_list = payload.get("attention_mask")

            position_ids = (
                np.array(position_ids_list, dtype=np.int64)
                if position_ids_list is not None
                else None
            )
            attention_mask = (
                np.array(attention_mask_list, dtype=np.bool_)
                if attention_mask_list is not None
                else None
            )

            logger.info(f"   Has position_ids: {position_ids is not None}")
            logger.info(f"   Has attention_mask: {attention_mask is not None}")
            if position_ids is not None:
                logger.info(
                    f"   Position_ids shape: {position_ids.shape}, content: {position_ids}"
                )

            # Fetch tensor from cache
            # We cached the tensor content when we saw it arrive as binary
            fetch_start = time.time()
            logger.info("   📥 Fetching tensor from cache...")

            # Fetch tensor: First check cache, then actively fetch via blobs client
            # CRITICAL: Don't rely on passive CONTENT_READY events - they can be slow!
            max_wait = 30.0  # seconds - increased timeout
            wait_start = time.time()
            tensor_bytes = None

            expected_hash = tensor_hash_str
            expected_size = tensor_size

            logger.info(f"   🎯 Looking for tensor hash: {expected_hash[:16]}...")

            # Check cache first (in case CONTENT_READY already fired)
            if expected_hash in self.tensor_cache:
                tensor_bytes = self.tensor_cache[expected_hash]
                logger.info(
                    f"   ✅ Found cached tensor immediately: {len(tensor_bytes)} bytes"
                )
                del self.tensor_cache[expected_hash]
            else:
                # Not in cache - actively fetch it using blobs client
                logger.info("   📡 Not in cache, actively fetching blob...")
                try:
                    from iroh import Hash

                    tensor_hash_obj = Hash.from_string(expected_hash)

                    # Try to fetch blob in a loop with retries
                    retry_count = 0
                    while time.time() - wait_start < max_wait:
                        try:
                            # Active fetch - this will block until blob is available
                            tensor_bytes = await self.iroh_node.blobs().read_to_bytes(
                                tensor_hash_obj
                            )
                            logger.info(
                                f"   ✅ Fetched tensor via blobs client: {len(tensor_bytes)} bytes (attempt {retry_count + 1})"
                            )
                            break
                        except Exception as read_error:
                            retry_count += 1
                            # Check cache again (maybe CONTENT_READY fired while we were trying)
                            if expected_hash in self.tensor_cache:
                                tensor_bytes = self.tensor_cache[expected_hash]
                                logger.info(
                                    f"   ✅ Found in cache during retry: {len(tensor_bytes)} bytes"
                                )
                                del self.tensor_cache[expected_hash]
                                break

                            elapsed = time.time() - wait_start
                            if int(elapsed) % 3 == 0:  # Log every 3 seconds
                                logger.debug(
                                    f"   ⏳ Fetch attempt {retry_count}, elapsed: {elapsed:.1f}s (error: {type(read_error).__name__})"
                                )
                            await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error(f"   ❌ Error setting up blob fetch: {e}")

            if tensor_bytes is None:
                logger.error(
                    f"   ❌ Timeout waiting for tensor to arrive (waited {max_wait}s)"
                )
                logger.error(f"   Expected hash: {expected_hash[:16]}...")
                logger.error(f"   Expected size: {expected_size} bytes")
                logger.error(
                    f"   Cache contents (first 5): {[(h[:16], len(c)) for h, c in list(self.tensor_cache.items())[:5]]}"
                )
                logger.error(f"   Total cache entries: {len(self.tensor_cache)}")
                return

            fetch_time = time.time() - fetch_start
            logger.info(f"   ✅ Tensor retrieved in {fetch_time*1000:.1f}ms")

            # Reconstruct tensor
            tensor = np.frombuffer(tensor_bytes, dtype=tensor_dtype).reshape(
                tensor_shape
            )
            logger.info(f"   ✅ Tensor reconstructed: shape={tensor.shape}")

            # Pass to ring coordinator WITH position_ids and attention_mask (CRITICAL!)
            logger.info("   ✅ Passing to ring coordinator...")
            await llm_service.ring_coordinator.handle_incoming_tensor(
                sender_id=sender_id,
                request_id=request_id,
                tensor_data=tensor,
                shard=llm_service.current_shard,
                is_final=is_final,
                position_ids=position_ids,  # CRITICAL: Pass position info for RoPE
                attention_mask=attention_mask,  # CRITICAL: Pass attention mask
            )

        except Exception as e:
            logger.error(f"❌ Error handling ring tensor: {e}", exc_info=True)
