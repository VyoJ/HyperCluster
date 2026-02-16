#!/usr/bin/env python3
"""
Test script for Gossip Protocol between 2 nodes.

This script tests the gossip-based tensor forwarding mechanism
to diagnose connectivity issues between nodes in the ring pipeline.

Usage:
    Terminal 1 (Node A - creates document):
        python test_gossip.py --mode create
    
    Terminal 2 (Node B - joins document):
        python test_gossip.py --mode join --ticket <TICKET_FROM_NODE_A>

The script will:
1. Establish gossip connection between nodes
2. Send test messages back and forth
3. Measure latency and report any failures
"""

import argparse
import asyncio
import hashlib
import json
import logging
import signal
import struct
import sys
import time
from typing import Optional

import numpy as np

try:
    import iroh
    from iroh import (
        AddrInfoOptions,
        GossipMessageCallback,
        Iroh,
        LiveEventType,
        MessageType,
        ShareMode,
    )
    # Monkey-patch iroh's event loop getter to use a stored loop
    # This fixes "no running event loop" errors from FFI callbacks
    _main_event_loop = None
    
    def _patched_get_event_loop():
        global _main_event_loop
        if _main_event_loop is not None:
            return _main_event_loop
        # Fallback to original behavior
        return asyncio.get_running_loop()
    
    # Apply the patch
    import iroh.iroh_ffi as iroh_ffi
    iroh_ffi._uniffi_get_event_loop = _patched_get_event_loop
    
except ImportError:
    print("ERROR: iroh module not found. Install with: pip install iroh")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Reduce iroh internal logging noise
logging.getLogger("iroh").setLevel(logging.WARNING)


class TestResults:
    """Track test results."""
    def __init__(self):
        self.messages_sent = 0
        self.messages_received = 0
        self.latencies = []
        self.errors = []
        self.gossip_joined = False
        self.neighbor_up = False
        self.sync_finished = False


class GossipTestCallback(GossipMessageCallback):
    """Callback for gossip messages during testing."""
    
    def __init__(self, node_id: str, results: TestResults, on_message_callback=None, loop=None):
        self.node_id = node_id
        self.results = results
        self.on_message_callback = on_message_callback
        self._pending_pings = {}  # request_id -> send_time
        self._loop = loop or asyncio.get_event_loop()
    
    async def on_message(self, msg):
        """Handle incoming gossip messages - must be async for iroh FFI."""
        try:
            msg_type = msg.type()
            
            if msg_type == MessageType.RECEIVED:
                content = msg.as_received()
                data = bytes(content.content)
                sender = str(content.delivered_from)
                
                receive_time = time.time()
                
                # Parse the test message
                try:
                    message = json.loads(data.decode("utf-8"))
                    msg_id = message.get("id", "unknown")
                    msg_type_inner = message.get("type", "unknown")
                    send_time = message.get("send_time", 0)
                    
                    if msg_type_inner == "ping":
                        logger.info(f"📥 PING received from {sender[:16]}... (id: {msg_id})")
                        self.results.messages_received += 1
                        
                        # Reply with pong
                        if self.on_message_callback:
                            await self.on_message_callback("pong", msg_id)
                            
                    elif msg_type_inner == "pong":
                        latency = (receive_time - send_time) * 1000  # ms
                        logger.info(f"📥 PONG received from {sender[:16]}... (id: {msg_id}) - latency: {latency:.1f}ms")
                        self.results.messages_received += 1
                        self.results.latencies.append(latency)
                        
                    elif msg_type_inner == "tensor_test":
                        tensor_size = message.get("tensor_size", 0)
                        latency = (receive_time - send_time) * 1000
                        logger.info(f"📥 TENSOR TEST received from {sender[:16]}... - size: {tensor_size} bytes, latency: {latency:.1f}ms")
                        self.results.messages_received += 1
                        self.results.latencies.append(latency)
                        
                except json.JSONDecodeError:
                    # Binary tensor data
                    logger.info(f"📥 Binary data received: {len(data)} bytes from {sender[:16]}...")
                    self.results.messages_received += 1
                    
            elif msg_type == MessageType.NEIGHBOR_UP:
                peer_id = str(msg.as_neighbor_up())
                logger.info(f"🔗 Gossip NEIGHBOR UP: {peer_id[:16]}...")
                self.results.neighbor_up = True
                
            elif msg_type == MessageType.NEIGHBOR_DOWN:
                peer_id = str(msg.as_neighbor_down())
                logger.info(f"🔗 Gossip NEIGHBOR DOWN: {peer_id[:16]}...")
                
            elif msg_type == MessageType.JOINED:
                nodes = msg.as_joined()
                logger.info(f"🔗 Gossip JOINED with {len(nodes)} peers: {[str(n)[:16] + '...' for n in nodes]}")
                self.results.gossip_joined = True
                
            elif msg_type == MessageType.LAGGED:
                logger.warning("⚠️  Gossip LAGGED - missed some messages")
                self.results.errors.append("Gossip lagged")
                
            elif msg_type == MessageType.ERROR:
                error = msg.as_error()
                logger.error(f"❌ Gossip ERROR: {error}")
                self.results.errors.append(f"Gossip error: {error}")
                
        except Exception as e:
            logger.error(f"Error in gossip callback: {e}", exc_info=True)
            self.results.errors.append(str(e))


class DocEventCallback:
    """Callback for document events."""
    
    def __init__(self, results: TestResults, tester_ref=None, loop=None):
        self.results = results
        self.tester_ref = tester_ref  # Reference to GossipTester instance
        self._loop = loop or asyncio.get_event_loop()
        
    async def event(self, event):
        """Handle document events - must be async for iroh FFI."""
        try:
            event_type = event.type()
            
            if event_type == LiveEventType.NEIGHBOR_UP:
                peer_id = str(event.as_neighbor_up())
                logger.info(f"👋 Document NEIGHBOR UP: {peer_id[:16]}...")
                self.results.neighbor_up = True
                # Store peer ID in tester
                if self.tester_ref:
                    self.tester_ref.peer_id = peer_id
                    logger.info(f"🎯 Peer discovered: {peer_id[:16]}...")
                    
            elif event_type == LiveEventType.NEIGHBOR_DOWN:
                peer_id = str(event.as_neighbor_down())
                logger.info(f"👋 Document NEIGHBOR DOWN: {peer_id[:16]}...")
                
            elif event_type == LiveEventType.SYNC_FINISHED:
                logger.info("✅ Document SYNC FINISHED")
                self.results.sync_finished = True
                
            elif event_type == LiveEventType.CONTENT_READY:
                hash_val = event.as_content_ready()
                logger.debug(f"📦 Document CONTENT READY: {str(hash_val)[:16]}...")
        except Exception as e:
            logger.error(f"Error in document event callback: {e}", exc_info=True)


class GossipTester:
    """Main gossip testing class."""
    
    def __init__(self):
        self.iroh_node: Optional[Iroh] = None
        self.doc = None
        self.doc_id: Optional[str] = None
        self.node_id: Optional[str] = None
        self.peer_id: Optional[str] = None
        self.gossip_sender = None
        self.results = TestResults()
        self.running = True
        
    async def start(self):
        """Start the Iroh node."""
        # Store the event loop globally for FFI callbacks
        global _main_event_loop
        _main_event_loop = asyncio.get_running_loop()
        
        options = iroh.NodeOptions()
        options.enable_docs = True
        self.iroh_node = await Iroh.memory_with_options(options)
        self.node_id = str(await self.iroh_node.net().node_id())
        logger.info(f"🚀 Node started with ID: {self.node_id}")
        logger.info(f"   (Short ID: {self.node_id[:16]}...)")
        
    async def stop(self):
        """Stop the node."""
        self.running = False
        if self.gossip_sender:
            try:
                await self.gossip_sender.cancel()
            except Exception:
                pass
        if self.iroh_node:
            await self.iroh_node.node().shutdown()
        logger.info("🛑 Node stopped")
        
    async def create_document(self) -> str:
        """Create a new document and return the ticket."""
        self.doc = await self.iroh_node.docs().create()
        self.doc_id = str(self.doc.id())
        
        ticket = await self.doc.share(ShareMode.WRITE, AddrInfoOptions.RELAY_AND_ADDRESSES)
        
        logger.info(f"📄 Document created: {self.doc_id[:16]}...")
        logger.info("")
        logger.info("=" * 60)
        logger.info("TICKET (copy this to the other node):")
        logger.info("=" * 60)
        logger.info(str(ticket))
        logger.info("=" * 60)
        logger.info("")
        
        # Subscribe to document events
        loop = asyncio.get_running_loop()
        callback = DocEventCallback(self.results, tester_ref=self, loop=loop)
        await self.doc.subscribe(callback)
        
        return str(ticket)
    
    async def join_document(self, ticket_str: str):
        """Join an existing document."""
        ticket = iroh.DocTicket(ticket_str)
        self.doc = await self.iroh_node.docs().join(ticket)
        self.doc_id = str(self.doc.id())
        
        logger.info(f"📄 Joined document: {self.doc_id[:16]}...")
        logger.info("⏳ Waiting for sync...")
        
        # Subscribe to document events  
        loop = asyncio.get_running_loop()
        callback = DocEventCallback(self.results, tester_ref=self, loop=loop)
        await self.doc.subscribe(callback)
        
        # Wait for sync
        await asyncio.sleep(3)
        
    async def setup_gossip(self, peer_ids: list):
        """Setup gossip subscription for testing."""
        if not self.doc_id:
            logger.error("No document - cannot setup gossip")
            return False
            
        # Create topic from document ID (same as node.py)
        topic_hash = hashlib.sha256(f"ring:{self.doc_id}".encode()).digest()
        
        logger.info(f"🔗 Setting up gossip...")
        logger.info(f"   Topic: {topic_hash[:8].hex()}...")
        logger.info(f"   Bootstrap peers: {[p[:16] + '...' for p in peer_ids]}")
        
        # Create callback
        loop = asyncio.get_running_loop()
        callback = GossipTestCallback(
            self.node_id,
            self.results,
            on_message_callback=self._send_reply,
            loop=loop
        )
        
        try:
            self.gossip_sender = await self.iroh_node.gossip().subscribe(
                bytearray(topic_hash),
                peer_ids,
                callback
            )
            logger.info("✅ Gossip subscription established")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to setup gossip: {e}")
            self.results.errors.append(f"Gossip setup failed: {e}")
            return False
            
    async def _send_reply(self, msg_type: str, original_id: str):
        """Send a reply message."""
        if not self.gossip_sender:
            return
            
        message = {
            "type": msg_type,
            "id": original_id,
            "send_time": time.time(),
            "from": self.node_id[:16]
        }
        payload = json.dumps(message).encode("utf-8")
        await self.gossip_sender.broadcast(bytearray(payload))
        self.results.messages_sent += 1
        
    async def send_ping(self, ping_id: int = None):
        """Send a ping message."""
        if not self.gossip_sender:
            logger.error("Gossip not initialized")
            return False
            
        if ping_id is None:
            ping_id = int(time.time() * 1000)
            
        message = {
            "type": "ping",
            "id": str(ping_id),
            "send_time": time.time(),
            "from": self.node_id[:16]
        }
        payload = json.dumps(message).encode("utf-8")
        
        logger.info(f"📤 Sending PING (id: {ping_id})")
        await self.gossip_sender.broadcast(bytearray(payload))
        self.results.messages_sent += 1
        return True
        
    async def send_tensor_test(self, size_kb: int = 100):
        """Send a test tensor to measure latency."""
        if not self.gossip_sender:
            logger.error("Gossip not initialized")
            return False
            
        # Create fake tensor data
        tensor_data = np.random.rand(size_kb * 256).astype(np.float32)  # ~1KB per 256 floats
        tensor_bytes = tensor_data.tobytes()
        
        message = {
            "type": "tensor_test",
            "id": str(int(time.time() * 1000)),
            "send_time": time.time(),
            "tensor_size": len(tensor_bytes),
            "from": self.node_id[:16]
        }
        
        # For this test, just send JSON metadata (not actual binary tensor)
        payload = json.dumps(message).encode("utf-8")
        
        logger.info(f"📤 Sending TENSOR TEST ({size_kb}KB equivalent)")
        await self.gossip_sender.broadcast(bytearray(payload))
        self.results.messages_sent += 1
        return True
        
    def print_results(self):
        """Print test results summary."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"   Messages sent:     {self.results.messages_sent}")
        logger.info(f"   Messages received: {self.results.messages_received}")
        logger.info(f"   Gossip joined:     {'✅' if self.results.gossip_joined else '❌'}")
        logger.info(f"   Neighbor up:       {'✅' if self.results.neighbor_up else '❌'}")
        logger.info(f"   Doc sync finished: {'✅' if self.results.sync_finished else '❌'}")
        
        if self.results.latencies:
            avg_latency = sum(self.results.latencies) / len(self.results.latencies)
            min_latency = min(self.results.latencies)
            max_latency = max(self.results.latencies)
            logger.info(f"   Latencies:         avg={avg_latency:.1f}ms, min={min_latency:.1f}ms, max={max_latency:.1f}ms")
        else:
            logger.info(f"   Latencies:         No data")
            
        if self.results.errors:
            logger.info(f"   Errors ({len(self.results.errors)}):")
            for err in self.results.errors:
                logger.info(f"      - {err}")
        else:
            logger.info(f"   Errors:            None")
        logger.info("=" * 60)
        
        # Overall assessment
        if self.results.messages_received > 0 and self.results.gossip_joined:
            logger.info("✅ GOSSIP IS WORKING!")
        elif self.results.gossip_joined and self.results.messages_received == 0:
            logger.info("⚠️  GOSSIP CONNECTED BUT NO MESSAGES RECEIVED")
            logger.info("   - Check if the other node is sending messages")
            logger.info("   - Verify both nodes are using the same document/topic")
        elif not self.results.gossip_joined:
            logger.info("❌ GOSSIP FAILED TO JOIN")
            logger.info("   - Check network connectivity between nodes")
            logger.info("   - Ensure relay servers are reachable")
            logger.info("   - Verify node IDs are correct")
        else:
            logger.info("❌ GOSSIP NOT WORKING")
            logger.info("   - Check logs above for error details")


async def run_creator_mode(tester: GossipTester, test_duration: int):
    """Run as the document creator."""
    await tester.start()
    await tester.create_document()
    
    logger.info("⏳ Waiting for peer to join...")
    
    # Wait for a peer to connect
    wait_start = time.time()
    while not tester.peer_id and time.time() - wait_start < 120:
        await asyncio.sleep(1)
        
    if not tester.peer_id:
        logger.error("❌ Timeout waiting for peer to join")
        tester.print_results()
        return
        
    logger.info(f"🎉 Peer connected! Setting up gossip...")
    await asyncio.sleep(2)  # Let things stabilize
    
    # Setup gossip with the peer
    if not await tester.setup_gossip([tester.peer_id]):
        tester.print_results()
        return
        
    # Wait for gossip to establish
    await asyncio.sleep(3)
    
    # Send test messages
    logger.info("")
    logger.info("📊 Starting gossip tests...")
    logger.info("")
    
    for i in range(5):
        await tester.send_ping(i + 1)
        await asyncio.sleep(2)
        
    # Send a larger message test
    await tester.send_tensor_test(100)
    await asyncio.sleep(2)
    await tester.send_tensor_test(500)
    await asyncio.sleep(2)
    
    # Wait for responses
    logger.info("⏳ Waiting for responses...")
    await asyncio.sleep(5)
    
    tester.print_results()


async def run_joiner_mode(tester: GossipTester, ticket: str, test_duration: int):
    """Run as the document joiner."""
    await tester.start()
    await tester.join_document(ticket)
    
    # Wait for peer discovery
    wait_start = time.time()
    while not tester.peer_id and time.time() - wait_start < 30:
        await asyncio.sleep(1)
        
    if not tester.peer_id:
        logger.warning("⚠️  No peer discovered via document events, checking document...")
        # The creator should be discoverable - try to get their ID from the ticket
        # For now, we'll wait for gossip to establish
        
    logger.info("🔗 Setting up gossip...")
    
    # If we have a peer ID, use it; otherwise gossip will try to discover
    peer_ids = [tester.peer_id] if tester.peer_id else []
    
    if not await tester.setup_gossip(peer_ids):
        tester.print_results()
        return
        
    # Wait for gossip to establish
    await asyncio.sleep(3)
    
    # Send test messages from this side too
    logger.info("")
    logger.info("📊 Sending test messages from joiner...")
    logger.info("")
    
    for i in range(3):
        await tester.send_ping(100 + i + 1)
        await asyncio.sleep(2)
        
    # Keep running to receive messages
    logger.info("⏳ Listening for messages...")
    remaining = test_duration
    while remaining > 0 and tester.running:
        await asyncio.sleep(1)
        remaining -= 1
        
    tester.print_results()


async def main():
    parser = argparse.ArgumentParser(description="Test Gossip Protocol between nodes")
    parser.add_argument(
        "--mode",
        choices=["create", "join"],
        required=True,
        help="'create' to create a new document, 'join' to join an existing one"
    )
    parser.add_argument(
        "--ticket",
        type=str,
        default=None,
        help="Document ticket (required for 'join' mode)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Test duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.mode == "join" and not args.ticket:
        parser.error("--ticket is required for 'join' mode")
        
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    tester = GossipTester()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        logger.info("\n🛑 Interrupted by user")
        tester.running = False
        
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        if args.mode == "create":
            await run_creator_mode(tester, args.duration)
        else:
            await run_joiner_mode(tester, args.ticket, args.duration)
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
    finally:
        await tester.stop()


if __name__ == "__main__":
    asyncio.run(main())
