"""
Direct peer-to-peer transport layer for HyperCluster.

Inspired by lattica's architecture:
- Uses iroh's QUIC connections (Endpoint.connect → Connection.open_bi → BiStream)
  for direct, low-latency binary streaming between peers.
- Implements a framed RPC protocol with length-prefixed messages.
- Maintains persistent connection pool with automatic reconnection.
- Supports both request/response and streaming RPC patterns.

This replaces the slow Doc-based tensor forwarding with direct QUIC streams,
matching lattica's performance characteristics:
  lattica:  libp2p swarm → noise+yamux → tcp/quic → Stream → framed bincode RPC
  ours:     iroh endpoint → QUIC (built-in noise) → BiStream → framed msgpack RPC
"""

import asyncio
import json
import logging
import struct
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import iroh
import numpy as np

logger = logging.getLogger(__name__)


# ─── Frame Protocol ─────────────────────────────────────────────────────────
# Mirrors lattica's StreamFrame enum with length-prefixed binary frames.
# Wire format: [4 bytes big-endian length][msgpack payload]
# This avoids JSON overhead for large tensor transfers.

class FrameType(IntEnum):
    """Frame types matching lattica's StreamFrame enum."""
    REQUEST = 1
    DATA = 2
    ERROR = 3
    CLOSE = 4
    CANCEL = 5
    PING = 6
    PONG = 7


@dataclass
class Frame:
    """A single protocol frame."""
    frame_type: FrameType
    request_id: str
    method: str = ""
    data: bytes = b""
    is_end: bool = False
    error: str = ""

    def encode(self) -> bytes:
        """Encode frame to wire format: [4-byte len][payload]."""
        # Simple binary format:
        # [1 byte type][1 byte flags][2 bytes method_len][method][16 bytes request_id_hash][data]
        flags = 0x01 if self.is_end else 0x00
        method_bytes = self.method.encode("utf-8") if self.method else b""
        rid_bytes = self.request_id.encode("utf-8")

        # Header: type(1) + flags(1) + rid_len(2) + method_len(2) = 6 bytes
        header = struct.pack(
            ">BBhh",
            int(self.frame_type),
            flags,
            len(rid_bytes),
            len(method_bytes),
        )

        if self.frame_type == FrameType.ERROR:
            payload = header + rid_bytes + method_bytes + self.error.encode("utf-8")
        else:
            payload = header + rid_bytes + method_bytes + self.data

        # Length-prefix the whole thing (like lattica's u32 length prefix)
        length_prefix = struct.pack(">I", len(payload))
        return length_prefix + payload

    @staticmethod
    def decode(raw: bytes) -> "Frame":
        """Decode a frame payload (after length prefix has been read)."""
        frame_type = FrameType(raw[0])
        flags = raw[1]
        rid_len, method_len = struct.unpack(">hh", raw[2:6])

        offset = 6
        request_id = raw[offset : offset + rid_len].decode("utf-8")
        offset += rid_len
        method = raw[offset : offset + method_len].decode("utf-8")
        offset += method_len

        is_end = bool(flags & 0x01)
        data = raw[offset:]

        error = ""
        if frame_type == FrameType.ERROR:
            error = data.decode("utf-8")
            data = b""

        return Frame(
            frame_type=frame_type,
            request_id=request_id,
            method=method,
            data=data,
            is_end=is_end,
            error=error,
        )


# ─── Stream Handle ───────────────────────────────────────────────────────────
# Mirrors lattica's StreamHandle: a persistent bidirectional connection to a peer
# with multiplexed request/response over a single QUIC BiStream.

CHUNK_SIZE = 4 * 1024 * 1024  # 4MB per frame (lattica uses 16MB, we use 4MB for Python)
ALPN_HYPERCLUSTER = b"hypercluster/rpc/1"


class StreamHandle:
    """
    A persistent connection handle to a peer, similar to lattica's StreamHandle.

    Multiplexes multiple RPC calls over a single QUIC bidirectional stream.
    Each call gets a unique request_id for routing responses.
    """

    def __init__(self, connection: iroh.Connection, peer_id: str):
        self.connection = connection
        self.peer_id = peer_id
        self.bi_stream: Optional[iroh.BiStream] = None
        self._send_stream: Optional[iroh.SendStream] = None
        self._recv_stream: Optional[iroh.RecvStream] = None

        # Pending calls: request_id → asyncio.Future or Queue
        self._pending_unary: Dict[str, asyncio.Future] = {}
        self._pending_stream: Dict[str, asyncio.Queue] = {}

        self._read_task: Optional[asyncio.Task] = None
        self._alive = True
        self._send_lock = asyncio.Lock()

    async def open(self):
        """Open a bidirectional stream to the peer."""
        try:
            self.bi_stream = await self.connection.open_bi()
            self._send_stream = self.bi_stream.send()
            self._recv_stream = self.bi_stream.recv()

            # Start background reader task (like lattica's read_task)
            self._read_task = asyncio.create_task(self._reader_loop())
            logger.debug(f"StreamHandle opened to {self.peer_id[:16]}...")
        except Exception as e:
            self._alive = False
            raise ConnectionError(f"Failed to open bi-stream to {self.peer_id[:16]}...: {e}")

    async def _reader_loop(self):
        """
        Background task that reads frames from the stream and routes them.
        Mirrors lattica's StreamHandle read task.
        """
        try:
            while self._alive:
                # Read 4-byte length prefix
                len_buf = await self._recv_stream.read_exact(4)
                if len_buf is None or len(len_buf) < 4:
                    break

                frame_len = struct.unpack(">I", bytes(len_buf))[0]
                if frame_len == 0:
                    continue
                if frame_len > 64 * 1024 * 1024:  # 64MB sanity limit
                    logger.error(f"Frame too large: {frame_len} bytes from {self.peer_id[:16]}...")
                    break

                # Read frame payload
                payload = await self._recv_stream.read_exact(frame_len)
                if payload is None or len(payload) < frame_len:
                    break

                frame = Frame.decode(bytes(payload))
                await self._handle_frame(frame)

        except Exception as e:
            if self._alive:
                logger.debug(f"Reader loop ended for {self.peer_id[:16]}...: {e}")
        finally:
            self._alive = False
            # Clean up pending calls
            for rid, fut in list(self._pending_unary.items()):
                if not fut.done():
                    fut.set_exception(ConnectionError("Stream closed"))
            self._pending_unary.clear()
            for rid, q in list(self._pending_stream.items()):
                await q.put(None)  # Signal end
            self._pending_stream.clear()

    async def _handle_frame(self, frame: Frame):
        """Route an incoming frame to the appropriate pending call."""
        if frame.frame_type == FrameType.DATA:
            # Check if it's a streaming call
            if frame.request_id in self._pending_stream:
                q = self._pending_stream[frame.request_id]
                await q.put(frame.data)
                if frame.is_end:
                    await q.put(None)  # Signal end
                    del self._pending_stream[frame.request_id]
            # Or a unary call accumulating data
            elif frame.request_id in self._pending_unary:
                fut = self._pending_unary[frame.request_id]
                if not fut.done():
                    # For unary, we accumulate into the future
                    if hasattr(fut, '_accumulated'):
                        fut._accumulated += frame.data
                    else:
                        fut._accumulated = frame.data
                    if frame.is_end:
                        fut.set_result(fut._accumulated)
                        del self._pending_unary[frame.request_id]

        elif frame.frame_type == FrameType.CLOSE:
            if frame.request_id in self._pending_stream:
                q = self._pending_stream.pop(frame.request_id)
                await q.put(None)
            elif frame.request_id in self._pending_unary:
                fut = self._pending_unary.pop(frame.request_id)
                if not fut.done():
                    data = getattr(fut, '_accumulated', b'')
                    fut.set_result(data)

        elif frame.frame_type == FrameType.ERROR:
            if frame.request_id in self._pending_stream:
                q = self._pending_stream.pop(frame.request_id)
                await q.put(None)
            elif frame.request_id in self._pending_unary:
                fut = self._pending_unary.pop(frame.request_id)
                if not fut.done():
                    fut.set_exception(RuntimeError(f"Remote error: {frame.error}"))

        elif frame.frame_type == FrameType.PONG:
            if frame.request_id in self._pending_unary:
                fut = self._pending_unary.pop(frame.request_id)
                if not fut.done():
                    fut.set_result(frame.data)

    async def _send_frame(self, frame: Frame):
        """Send a frame over the stream with lock for thread safety."""
        if not self._alive:
            raise ConnectionError(f"Stream to {self.peer_id[:16]}... is closed")

        async with self._send_lock:
            encoded = frame.encode()
            await self._send_stream.write_all(encoded)

    async def call(self, method: str, data: bytes, timeout: float = 60.0) -> bytes:
        """
        Make a unary RPC call (like lattica's call_stream).

        Sends data, waits for complete response.
        """
        request_id = str(uuid.uuid4())[:8]
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        fut._accumulated = b""
        self._pending_unary[request_id] = fut

        # Send request frame
        await self._send_frame(Frame(
            frame_type=FrameType.REQUEST,
            request_id=request_id,
            method=method,
        ))

        # Send data in chunks (like lattica's 16MB chunked writes)
        await self._send_data_chunked(request_id, data)

        # Wait for response
        try:
            result = await asyncio.wait_for(fut, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            self._pending_unary.pop(request_id, None)
            raise TimeoutError(f"RPC call {method} timed out after {timeout}s")

    async def call_stream(
        self, method: str, data: bytes, timeout: float = 60.0
    ) -> asyncio.Queue:
        """
        Make a streaming RPC call (like lattica's call_stream_iter).

        Returns an asyncio.Queue that yields response chunks.
        Queue yields None when stream ends.
        """
        request_id = str(uuid.uuid4())[:8]
        queue = asyncio.Queue(maxsize=512)
        self._pending_stream[request_id] = queue

        # Send request frame
        await self._send_frame(Frame(
            frame_type=FrameType.REQUEST,
            request_id=request_id,
            method=method,
        ))

        # Send data in chunks
        await self._send_data_chunked(request_id, data)

        return queue

    async def _send_data_chunked(self, request_id: str, data: bytes):
        """Send data in chunks, like lattica's chunked frame writes."""
        if len(data) == 0:
            await self._send_frame(Frame(
                frame_type=FrameType.DATA,
                request_id=request_id,
                data=b"",
                is_end=True,
            ))
            return

        total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE
        for i in range(total_chunks):
            start = i * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, len(data))
            is_last = i == total_chunks - 1

            await self._send_frame(Frame(
                frame_type=FrameType.DATA,
                request_id=request_id,
                data=data[start:end],
                is_end=is_last,
            ))

    async def send_fire_and_forget(self, method: str, data: bytes):
        """Send data without waiting for a response. Used for tensor forwarding."""
        request_id = str(uuid.uuid4())[:8]

        await self._send_frame(Frame(
            frame_type=FrameType.REQUEST,
            request_id=request_id,
            method=method,
        ))
        await self._send_data_chunked(request_id, data)

    async def ping(self, timeout: float = 5.0) -> float:
        """
        Measure RTT to peer using a ping/pong frame exchange.
        Returns round-trip time in milliseconds.
        """
        request_id = str(uuid.uuid4())[:8]
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self._pending_unary[request_id] = fut

        start = time.monotonic()
        await self._send_frame(Frame(
            frame_type=FrameType.PING,
            request_id=request_id,
        ))

        try:
            await asyncio.wait_for(fut, timeout=timeout)
            rtt = (time.monotonic() - start) * 1000
            return rtt
        except asyncio.TimeoutError:
            self._pending_unary.pop(request_id, None)
            return -1.0

    def is_alive(self) -> bool:
        return self._alive

    async def close(self):
        self._alive = False
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except (asyncio.CancelledError, Exception):
                pass


# ─── RPC Service (Server Side) ──────────────────────────────────────────────
# Mirrors lattica's RpcService trait — handlers registered by service name.

class RpcServiceHandler:
    """
    Base class for RPC service handlers.
    Subclass this and implement handle_request() and/or handle_stream().
    Similar to lattica's RpcService trait.
    """

    def service_name(self) -> str:
        raise NotImplementedError

    async def handle_request(
        self, method: str, data: bytes
    ) -> bytes:
        """Handle a unary request. Return response bytes."""
        raise NotImplementedError(f"Method {method} not implemented")

    async def handle_stream(
        self, method: str, data: bytes
    ) -> Optional[asyncio.Queue]:
        """Handle a streaming request. Return a queue of response chunks, or None."""
        return None


# ─── Connection Manager ─────────────────────────────────────────────────────
# Mirrors lattica's stream_handles HashMap + swarm_poll for connection lifecycle.

class ConnectionManager:
    """
    Manages peer connections and stream handles.

    Like lattica's core:
    - Maintains a pool of StreamHandle per peer
    - Auto-reconnects on failure
    - Routes incoming connections to registered services
    - Provides connect/call/stream APIs
    """

    def __init__(self, iroh_node: iroh.Iroh):
        self.iroh_node = iroh_node
        self.node_id: Optional[str] = None

        # Connection pool: peer_id_str → StreamHandle
        self._handles: Dict[str, StreamHandle] = {}
        self._handles_lock = asyncio.Lock()

        # Known peer addresses: peer_id_str → NodeAddr
        self._peer_addrs: Dict[str, iroh.NodeAddr] = {}

        # Registered RPC services: service_name → handler
        self._services: Dict[str, RpcServiceHandler] = {}

        # Incoming connection handler task
        self._accept_task: Optional[asyncio.Task] = None

        # Peer RTT cache
        self._peer_rtt: Dict[str, float] = {}

        # Callbacks
        self._on_peer_connected: Optional[Callable] = None
        self._on_peer_disconnected: Optional[Callable] = None

    async def start(self):
        """Initialize the connection manager."""
        self.node_id = str(await self.iroh_node.net().node_id())
        logger.info(f"ConnectionManager started, node_id={self.node_id[:16]}...")

    def register_service(self, service: RpcServiceHandler):
        """Register an RPC service handler (like lattica's register_service)."""
        name = service.service_name()
        self._services[name] = service
        logger.info(f"Registered RPC service: {name}")

    def set_peer_connected_callback(self, cb: Callable):
        self._on_peer_connected = cb

    def set_peer_disconnected_callback(self, cb: Callable):
        self._on_peer_disconnected = cb

    async def add_peer(self, node_addr: iroh.NodeAddr, peer_id_str: str):
        """Register a peer's address for future connections."""
        self._peer_addrs[peer_id_str] = node_addr
        # Also tell iroh about this address
        try:
            await self.iroh_node.net().add_node_addr(node_addr)
        except Exception as e:
            logger.warning(f"Failed to add node addr for {peer_id_str[:16]}...: {e}")

    async def connect(self, peer_id_str: str) -> StreamHandle:
        """
        Get or create a StreamHandle for a peer.
        Like lattica's RpcGetOrCreateStreamHandle command.
        """
        async with self._handles_lock:
            # Check existing handle
            if peer_id_str in self._handles:
                handle = self._handles[peer_id_str]
                if handle.is_alive():
                    return handle
                else:
                    # Dead handle, clean up
                    logger.debug(f"Removing dead handle for {peer_id_str[:16]}...")
                    del self._handles[peer_id_str]

            # Need to create a new connection
            logger.info(f"Opening QUIC connection to {peer_id_str[:16]}...")

            if peer_id_str not in self._peer_addrs:
                raise ConnectionError(
                    f"No address known for peer {peer_id_str[:16]}... "
                    f"Known peers: {[p[:16] for p in self._peer_addrs.keys()]}"
                )

            node_addr = self._peer_addrs[peer_id_str]

            try:
                # Connect via iroh's QUIC endpoint (like lattica's swarm.dial)
                endpoint = self.iroh_node.node().endpoint()
                connection = await endpoint.connect(node_addr, ALPN_HYPERCLUSTER)

                handle = StreamHandle(connection, peer_id_str)
                await handle.open()

                self._handles[peer_id_str] = handle

                # Measure initial RTT
                try:
                    rtt = await handle.ping(timeout=5.0)
                    if rtt >= 0:
                        self._peer_rtt[peer_id_str] = rtt
                        logger.info(f"Connected to {peer_id_str[:16]}... (RTT: {rtt:.1f}ms)")
                except Exception:
                    pass

                if self._on_peer_connected:
                    await self._on_peer_connected(peer_id_str)

                return handle

            except Exception as e:
                raise ConnectionError(f"Failed to connect to {peer_id_str[:16]}...: {e}")

    async def call(self, peer_id: str, method: str, data: bytes, timeout: float = 60.0) -> bytes:
        """
        Make a unary RPC call to a peer.
        Like lattica's Lattica::call().
        """
        handle = await self.connect(peer_id)
        return await handle.call(method, data, timeout=timeout)

    async def send_tensor(
        self,
        peer_id: str,
        tensor_data: np.ndarray,
        request_id: str,
        metadata: dict,
    ):
        """
        Send a tensor directly to a peer via QUIC stream.

        This is the key performance improvement over Doc-based forwarding.
        Like lattica's call_stream for large binary data.
        """
        handle = await self.connect(peer_id)

        # Encode metadata + tensor as a single binary payload
        # Format: [4 bytes meta_len][JSON metadata][raw tensor bytes]
        meta_json = json.dumps(metadata).encode("utf-8")
        meta_len = struct.pack(">I", len(meta_json))
        payload = meta_len + meta_json + tensor_data.tobytes()

        send_start = time.time()
        await handle.send_fire_and_forget("tensor.forward", payload)
        send_time = (time.time() - send_start) * 1000

        size_mb = len(payload) / (1024 * 1024)
        throughput = size_mb / (send_time / 1000) if send_time > 0 else 0
        logger.info(
            f"📤 Tensor sent to {peer_id[:16]}... "
            f"({size_mb:.2f}MB in {send_time:.1f}ms, {throughput:.1f}MB/s)"
        )

    async def get_peer_rtt(self, peer_id: str) -> Optional[float]:
        """Get cached RTT to a peer in milliseconds."""
        return self._peer_rtt.get(peer_id)

    async def get_connected_peers(self) -> List[str]:
        """Get list of currently connected peer IDs."""
        async with self._handles_lock:
            return [pid for pid, h in self._handles.items() if h.is_alive()]

    async def handle_incoming_connection(self, connection: iroh.Connection):
        """
        Handle an incoming QUIC connection from a peer.
        Like lattica's incoming_streams handler in swarm_poll.
        
        Reads all frames sequentially from the stream (no concurrent reads),
        accumulates data per request_id, and dispatches to services when complete.
        """
        peer_id = str(connection.remote_node_id())
        logger.info(f"📥 Incoming connection from {peer_id[:16]}...")

        try:
            bi_stream = await connection.accept_bi()
            send_stream = bi_stream.send()
            recv_stream = bi_stream.recv()

            # Track active requests: request_id → {method, data}
            active_requests: Dict[str, Dict] = {}

            # Process incoming frames in a loop (single reader, no races)
            while True:
                # Read 4-byte length prefix
                len_buf = await recv_stream.read_exact(4)
                if len_buf is None or len(len_buf) < 4:
                    break

                frame_len = struct.unpack(">I", bytes(len_buf))[0]
                if frame_len == 0:
                    continue
                if frame_len > 64 * 1024 * 1024:
                    logger.error(f"Frame too large from {peer_id[:16]}...: {frame_len}")
                    break

                payload = await recv_stream.read_exact(frame_len)
                if payload is None or len(payload) < frame_len:
                    break

                frame = Frame.decode(bytes(payload))

                if frame.frame_type == FrameType.REQUEST:
                    # Register new request
                    active_requests[frame.request_id] = {
                        "method": frame.method,
                        "data": bytearray(),
                    }

                elif frame.frame_type == FrameType.DATA:
                    rid = frame.request_id
                    if rid in active_requests:
                        active_requests[rid]["data"].extend(frame.data)
                        if frame.is_end:
                            # All data received — dispatch to service
                            req = active_requests.pop(rid)
                            asyncio.create_task(
                                self._dispatch_request(
                                    peer_id, rid, req["method"],
                                    bytes(req["data"]), send_stream
                                )
                            )

                elif frame.frame_type == FrameType.PING:
                    # Reply with pong immediately
                    pong = Frame(
                        frame_type=FrameType.PONG,
                        request_id=frame.request_id,
                    )
                    encoded = pong.encode()
                    await send_stream.write_all(encoded)

                elif frame.frame_type == FrameType.CLOSE:
                    break

        except Exception as e:
            logger.debug(f"Incoming connection from {peer_id[:16]}... ended: {e}")

    async def _dispatch_request(
        self,
        peer_id: str,
        request_id: str,
        method: str,
        data: bytes,
        send_stream: iroh.SendStream,
    ):
        """Dispatch a fully-received request to the appropriate service handler."""
        # Method format: "service.method" (like lattica)
        parts = method.split(".", 1)
        if len(parts) != 2:
            service_name = parts[0] if parts else method
            method_name = parts[1] if len(parts) > 1 else ""
        else:
            service_name, method_name = parts

        if service_name in self._services:
            service = self._services[service_name]
            try:
                result = await service.handle_request(method_name, data)

                if result is not None:
                    # Send response
                    response = Frame(
                        frame_type=FrameType.DATA,
                        request_id=request_id,
                        data=result,
                        is_end=True,
                    )
                    encoded = response.encode()
                    await send_stream.write_all(encoded)
            except Exception as e:
                logger.error(f"Service error for {method}: {e}")
                error_frame = Frame(
                    frame_type=FrameType.ERROR,
                    request_id=request_id,
                    error=str(e),
                )
                encoded = error_frame.encode()
                await send_stream.write_all(encoded)
        else:
            logger.warning(f"No service handler for '{service_name}' (method: {method})")

    async def shutdown(self):
        """Shutdown all connections."""
        if self._accept_task:
            self._accept_task.cancel()
        async with self._handles_lock:
            for handle in self._handles.values():
                await handle.close()
            self._handles.clear()
        logger.info("ConnectionManager shut down")


# ─── Iroh Protocol Registration ─────────────────────────────────────────────
# Implements iroh's ProtocolCreator/ProtocolHandler to register our ALPN so
# incoming QUIC connections are routed to ConnectionManager.handle_incoming_connection.
# This mirrors lattica's swarm event loop that dispatches incoming streams.


class HyperClusterProtocolHandler:
    """
    Iroh ProtocolHandler implementation for the hypercluster/rpc/1 ALPN.

    When a peer connects to us using our ALPN, iroh calls accept(conn).
    We route that to the ConnectionManager for RPC dispatch.
    
    Note: iroh's FFI awaits the return value of accept() and shutdown(),
    so they must be async despite the typing.Protocol definition.
    """

    def __init__(self, conn_manager: ConnectionManager):
        self._conn_manager = conn_manager
        self._tasks: List[asyncio.Task] = []

    async def accept(self, conn: iroh.Connection):
        """Called by iroh when a peer opens a connection with our ALPN."""
        task = asyncio.create_task(self._handle_accept(conn))
        self._tasks.append(task)
        # Clean up completed tasks
        self._tasks = [t for t in self._tasks if not t.done()]

    async def _handle_accept(self, conn: iroh.Connection):
        """Handle an accepted connection asynchronously."""
        try:
            await self._conn_manager.handle_incoming_connection(conn)
        except Exception as e:
            logger.debug(f"Incoming connection handler ended: {e}")

    async def shutdown(self):
        """Called by iroh when the node is shutting down."""
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()


class HyperClusterProtocolCreator:
    """
    Iroh ProtocolCreator that creates our ProtocolHandler.

    Passed to NodeOptions.protocols with ALPN_HYPERCLUSTER as the key.
    iroh calls create(endpoint) once during node startup.
    """

    def __init__(self):
        self._handler: Optional[HyperClusterProtocolHandler] = None
        self._conn_manager: Optional[ConnectionManager] = None

    def set_conn_manager(self, conn_manager: ConnectionManager):
        """
        Set the ConnectionManager after node creation.
        
        Since the ConnectionManager needs the iroh node (which is created
        with the protocol already registered), we use deferred binding:
        1. Create ProtocolCreator
        2. Pass it to NodeOptions.protocols
        3. iroh creates node, calls create(endpoint) → we store the handler
        4. After node is created, call set_conn_manager() to wire it up
        """
        self._conn_manager = conn_manager
        if self._handler:
            self._handler._conn_manager = conn_manager

    def create(self, endpoint: iroh.Endpoint):
        """Called by iroh during node startup. Returns a ProtocolHandler."""
        # Create a handler - conn_manager may be set later via set_conn_manager
        self._handler = HyperClusterProtocolHandler(self._conn_manager)
        return self._handler
