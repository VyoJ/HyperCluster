import asyncio
import logging
from typing import Optional, Callable, Dict, Any, List, Tuple, Set
import iroh
from iroh import Iroh, ShareMode, AddrInfoOptions, LiveEventType, PublicKey
import json
import time

logger = logging.getLogger(__name__)

class Node:
    def __init__(self, bootstrap_nodes: Optional[List[str]] = None):
        self.iroh_node: Optional[Iroh] = None
        self.bootstrap_nodes = bootstrap_nodes or []
        self.documents: Dict[str, iroh.Doc] = {}
        self.message_handlers: List[Callable] = []
        self.system_info = {}
        self.neighbors: Dict[str, Set[PublicKey]] = {} # doc_id -> set of peer_ids

    async def start(self):
        """Start the Iroh node."""
        try:
            options = iroh.NodeOptions()
            options.enable_docs = True
            self.iroh_node = await Iroh.memory_with_options(options)
            logger.info(f"Iroh node started with ID: {await self.iroh_node.net().node_id()}")
            
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
            ticket = await doc.share(ShareMode.WRITE, AddrInfoOptions.RELAY_AND_ADDRESSES)
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
            message_data = json.loads(content.decode('utf-8'))
            
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
            key = f"message-{time.time()}".encode('utf-8')
            payload = json.dumps(message).encode('utf-8')
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
        
        message = {
            "type": "store",
            "key": key,
            "value": value
        }
        # For storing values, we can use the key directly in the document
        author = await self.iroh_node.authors().default()
        doc = self.documents[doc_id]
        try:
            payload = json.dumps(value).encode('utf-8')
            await doc.set_bytes(author, key.encode('utf-8'), payload)
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
            query = iroh.Query.author_key_exact(author, key.encode('utf-8'))
            entry = await doc.get_one(query)
            if entry:
                content = await self.iroh_node.blobs().read_to_bytes(entry.content_hash())
                return json.loads(content.decode('utf-8'))
        except Exception as e:
            logger.error(f"Error retrieving value from doc {doc.id()}: {e}")
        return None