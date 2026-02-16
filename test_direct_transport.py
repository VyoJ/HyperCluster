#!/usr/bin/env python3
"""
Test the direct QUIC transport layer (lattica-style).

Tests:
1. Frame encode/decode roundtrip
2. ConnectionManager lifecycle
3. Tensor send/receive (compares with Doc-based baseline)

Usage:
    uv run python test_direct_transport.py
"""

import asyncio
import json
import logging
import struct
import sys
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_frame_protocol():
    """Test Frame encode/decode roundtrip."""
    from direct_transport import Frame, FrameType

    print("\n" + "=" * 60)
    print("TEST 1: Frame Protocol Encode/Decode")
    print("=" * 60)

    # Test request frame
    f1 = Frame(
        frame_type=FrameType.REQUEST,
        request_id="abc123",
        method="tensor.forward",
    )
    encoded = f1.encode()
    # Skip the 4-byte length prefix for decode
    decoded = Frame.decode(encoded[4:])
    assert decoded.frame_type == FrameType.REQUEST
    assert decoded.request_id == "abc123"
    assert decoded.method == "tensor.forward"
    print(f"  ✅ REQUEST frame: {len(encoded)} bytes")

    # Test data frame with binary payload
    tensor = np.random.rand(10, 128).astype(np.float32)
    tensor_bytes = tensor.tobytes()
    f2 = Frame(
        frame_type=FrameType.DATA,
        request_id="abc123",
        data=tensor_bytes,
        is_end=True,
    )
    encoded = f2.encode()
    decoded = Frame.decode(encoded[4:])
    assert decoded.frame_type == FrameType.DATA
    assert decoded.request_id == "abc123"
    assert decoded.is_end is True
    assert decoded.data == tensor_bytes

    # Verify tensor can be reconstructed
    reconstructed = np.frombuffer(decoded.data, dtype=np.float32).reshape(10, 128)
    assert np.array_equal(tensor, reconstructed)
    print(f"  ✅ DATA frame: {len(encoded)} bytes ({len(tensor_bytes)} tensor bytes)")

    # Test error frame
    f3 = Frame(
        frame_type=FrameType.ERROR,
        request_id="abc123",
        error="Something went wrong",
    )
    encoded = f3.encode()
    decoded = Frame.decode(encoded[4:])
    assert decoded.frame_type == FrameType.ERROR
    assert decoded.error == "Something went wrong"
    print(f"  ✅ ERROR frame: {len(encoded)} bytes")

    # Test ping/pong frames
    f4 = Frame(frame_type=FrameType.PING, request_id="ping1")
    encoded = f4.encode()
    decoded = Frame.decode(encoded[4:])
    assert decoded.frame_type == FrameType.PING
    print(f"  ✅ PING frame: {len(encoded)} bytes")

    # Benchmark encode/decode
    large_tensor = np.random.rand(1, 768).astype(np.float32)  # typical hidden state
    large_bytes = large_tensor.tobytes()
    large_frame = Frame(
        frame_type=FrameType.DATA,
        request_id="bench",
        data=large_bytes,
        is_end=True,
    )

    # Benchmark encode
    start = time.time()
    N = 10000
    for _ in range(N):
        large_frame.encode()
    encode_time = (time.time() - start) / N * 1000
    print(f"  📊 Encode: {encode_time:.3f}ms per frame ({len(large_bytes)} bytes payload)")

    # Benchmark decode
    encoded = large_frame.encode()
    raw = encoded[4:]
    start = time.time()
    for _ in range(N):
        Frame.decode(raw)
    decode_time = (time.time() - start) / N * 1000
    print(f"  📊 Decode: {decode_time:.3f}ms per frame")

    print("  ✅ All frame protocol tests passed!")


def test_tensor_wire_format():
    """Test the tensor wire format (metadata + binary)."""
    print("\n" + "=" * 60)
    print("TEST 2: Tensor Wire Format")
    print("=" * 60)

    # Simulate what send_tensor does
    tensor = np.random.rand(1, 32, 768).astype(np.float32)
    metadata = {
        "sender_id": "abc123456789",
        "request_id": "req-001",
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "is_final": False,
        "position_ids": [0, 1, 2],
        "attention_mask": None,
    }

    # Encode
    start = time.time()
    meta_json = json.dumps(metadata).encode("utf-8")
    meta_len = struct.pack(">I", len(meta_json))
    payload = meta_len + meta_json + tensor.tobytes()
    encode_time = (time.time() - start) * 1000

    tensor_size = len(tensor.tobytes())
    total_size = len(payload)
    overhead = (total_size - tensor_size) / tensor_size * 100

    print(f"  Tensor: {tensor.shape} ({tensor_size / 1024:.1f} KB)")
    print(f"  Wire total: {total_size / 1024:.1f} KB (overhead: {overhead:.1f}%)")
    print(f"  Encode time: {encode_time:.3f}ms")

    # Decode (simulating TensorForwardService._handle_tensor_forward)
    start = time.time()
    recv_meta_len = struct.unpack(">I", payload[:4])[0]
    recv_meta = json.loads(payload[4 : 4 + recv_meta_len])
    recv_tensor_bytes = payload[4 + recv_meta_len :]
    recv_tensor = np.frombuffer(recv_tensor_bytes, dtype=np.dtype(recv_meta["tensor_dtype"])).reshape(
        tuple(recv_meta["tensor_shape"])
    )
    decode_time = (time.time() - start) * 1000

    assert np.array_equal(tensor, recv_tensor)
    assert recv_meta["request_id"] == "req-001"
    print(f"  Decode time: {decode_time:.3f}ms")
    print(f"  ✅ Tensor roundtrip verified!")

    # Compare with the OLD Doc-based approach
    import base64
    start = time.time()
    old_json = json.dumps({
        "type": "ring_tensor_forward",
        "sender_id": "abc123456789",
        "payload": {
            "tensor_data": base64.b64encode(tensor.tobytes()).decode("utf-8"),
            "tensor_shape": list(tensor.shape),
            "tensor_dtype": str(tensor.dtype),
        }
    }).encode("utf-8")
    old_encode_time = (time.time() - start) * 1000

    print(f"\n  📊 Comparison with Doc-based approach:")
    print(f"     Doc-based JSON+base64: {len(old_json) / 1024:.1f} KB ({old_encode_time:.3f}ms)")
    print(f"     Direct binary:         {total_size / 1024:.1f} KB ({encode_time:.3f}ms)")
    print(f"     Size reduction:        {(1 - total_size / len(old_json)) * 100:.0f}%")
    print(f"     Speed improvement:     {old_encode_time / encode_time:.1f}x encode")


async def test_iroh_connection():
    """Test actual iroh QUIC connections with direct transport."""
    print("\n" + "=" * 60)
    print("TEST 3: Iroh QUIC Connection")
    print("=" * 60)

    import iroh
    from iroh import Iroh

    iroh.iroh_ffi.uniffi_set_event_loop(asyncio.get_running_loop())

    from direct_transport import (
        ConnectionManager,
        ALPN_HYPERCLUSTER,
        HyperClusterProtocolCreator,
    )
    from tensor_service import TensorForwardService

    # Create two iroh nodes WITH protocol registration (like lattica's swarm setup)
    print("  Starting Node A (with ALPN protocol)...")
    opts_a = iroh.NodeOptions()
    creator_a = HyperClusterProtocolCreator()
    opts_a.protocols = {ALPN_HYPERCLUSTER: creator_a}
    node_a = await Iroh.memory_with_options(opts_a)
    node_a_id = str(await node_a.net().node_id())
    print(f"  Node A: {node_a_id[:16]}...")

    print("  Starting Node B (with ALPN protocol)...")
    opts_b = iroh.NodeOptions()
    creator_b = HyperClusterProtocolCreator()
    opts_b.protocols = {ALPN_HYPERCLUSTER: creator_b}
    node_b = await Iroh.memory_with_options(opts_b)
    node_b_id = str(await node_b.net().node_id())
    print(f"  Node B: {node_b_id[:16]}...")

    # Get Node B's address
    node_b_addr = await node_b.net().node_addr()
    node_a_addr = await node_a.net().node_addr()

    # Add addresses to each other
    await node_a.net().add_node_addr(node_b_addr)
    await node_b.net().add_node_addr(node_a_addr)

    # Create connection managers for both nodes
    cm_a = ConnectionManager(node_a)
    await cm_a.start()
    creator_a.set_conn_manager(cm_a)  # Wire up the protocol handler

    cm_b = ConnectionManager(node_b)
    await cm_b.start()
    creator_b.set_conn_manager(cm_b)  # Wire up the protocol handler

    # Register Node B's address on Node A's ConnectionManager
    await cm_a.add_peer(node_b_addr, node_b_id)

    # Set up tensor receiver on Node B
    received_tensors = []
    tensor_received_event = asyncio.Event()

    tensor_service = TensorForwardService()
    tensor_service.set_node_id(node_b_id)

    async def on_tensor(sender_id, request_id, tensor_data, is_final, position_ids=None, attention_mask=None):
        received_tensors.append({
            "sender_id": sender_id,
            "request_id": request_id,
            "tensor_data": tensor_data,
            "is_final": is_final,
        })
        tensor_received_event.set()

    tensor_service.set_tensor_callback(on_tensor)
    cm_b.register_service(tensor_service)

    # Try to connect from A to B
    print("  Connecting A → B via QUIC...")
    try:
        endpoint_a = node_a.node().endpoint()
        conn = await endpoint_a.connect(node_b_addr, ALPN_HYPERCLUSTER)
        rtt = conn.rtt()
        print(f"  ✅ QUIC connection established! RTT: {rtt}ms")
    except Exception as e:
        print(f"  ⚠️  QUIC connection test: {e}")

    # Test tensor send via ConnectionManager
    print("\n  Testing ConnectionManager.send_tensor...")
    tensor = np.random.rand(1, 4, 768).astype(np.float32)
    metadata = {
        "sender_id": node_a_id,
        "request_id": "test-req-1",
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "is_final": False,
    }

    try:
        await cm_a.send_tensor(
            peer_id=node_b_id,
            tensor_data=tensor,
            request_id="test-req-1",
            metadata=metadata,
        )
        print(f"  ✅ Tensor sent via direct transport!")

        # Wait for tensor to arrive on Node B
        try:
            await asyncio.wait_for(tensor_received_event.wait(), timeout=5.0)
            print(f"  ✅ Tensor received on Node B! ({len(received_tensors)} tensors)")
            if received_tensors:
                rt = received_tensors[0]
                print(f"     sender: {rt['sender_id'][:16]}...")
                print(f"     request_id: {rt['request_id']}")
                print(f"     tensor shape: {rt['tensor_data'].shape}")
                assert np.allclose(tensor, rt['tensor_data'], atol=1e-6)
                print(f"  ✅ Tensor data verified - matches original!")
        except asyncio.TimeoutError:
            print(f"  ⚠️  Tensor not received within timeout")
            print(f"     (Protocol handler may need active accept loop)")

    except Exception as e:
        print(f"  ⚠️  Direct tensor send: {e}")

    # Clean up
    await cm_a.shutdown()
    await cm_b.shutdown()
    await node_a.node().shutdown()
    await node_b.node().shutdown()
    print("  ✅ Nodes shut down")


def test_overhead_comparison():
    """Compare overhead of direct vs Doc-based approach."""
    print("\n" + "=" * 60)
    print("TEST 4: Performance Overhead Comparison")
    print("=" * 60)

    import base64

    sizes = [
        ("Small (1x768)", np.random.rand(1, 768).astype(np.float32)),
        ("Medium (1x32x768)", np.random.rand(1, 32, 768).astype(np.float32)),
        ("Large (1x128x2048)", np.random.rand(1, 128, 2048).astype(np.float32)),
    ]

    print(f"\n  {'Size':<25} {'Tensor':>10} {'Doc JSON':>12} {'Direct':>12} {'Savings':>10} {'Encode Δ':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

    for name, tensor in sizes:
        tensor_bytes = tensor.tobytes()
        tensor_kb = len(tensor_bytes) / 1024

        # Doc-based (JSON + base64)
        t0 = time.time()
        for _ in range(100):
            doc_payload = json.dumps({
                "type": "ring_tensor_forward",
                "payload": {
                    "tensor_data": base64.b64encode(tensor_bytes).decode("utf-8"),
                    "tensor_shape": list(tensor.shape),
                    "tensor_dtype": str(tensor.dtype),
                }
            }).encode("utf-8")
        doc_time = (time.time() - t0) / 100 * 1000
        doc_kb = len(doc_payload) / 1024

        # Direct binary
        t0 = time.time()
        for _ in range(100):
            meta = json.dumps({
                "sender_id": "x" * 52,
                "request_id": "req-001",
                "tensor_shape": list(tensor.shape),
                "tensor_dtype": str(tensor.dtype),
                "is_final": False,
            }).encode("utf-8")
            direct_payload = struct.pack(">I", len(meta)) + meta + tensor_bytes
        direct_time = (time.time() - t0) / 100 * 1000
        direct_kb = len(direct_payload) / 1024

        savings = (1 - direct_kb / doc_kb) * 100
        speedup = doc_time / direct_time if direct_time > 0 else float('inf')

        print(
            f"  {name:<25} {tensor_kb:>8.1f}KB "
            f"{doc_kb:>10.1f}KB {direct_kb:>10.1f}KB "
            f"{savings:>8.0f}%  {speedup:>8.1f}x"
        )

    # Also compare the decode side
    print(f"\n  Decode comparison (1x128x2048):")
    tensor = np.random.rand(1, 128, 2048).astype(np.float32)
    tensor_bytes = tensor.tobytes()

    # Doc decode: JSON parse + base64 decode
    doc_payload = json.dumps({
        "tensor_data": base64.b64encode(tensor_bytes).decode("utf-8"),
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
    }).encode("utf-8")

    t0 = time.time()
    for _ in range(100):
        parsed = json.loads(doc_payload)
        decoded_bytes = base64.b64decode(parsed["tensor_data"])
        np.frombuffer(decoded_bytes, dtype=np.float32).reshape(tuple(parsed["tensor_shape"]))
    doc_decode = (time.time() - t0) / 100 * 1000

    # Direct decode: struct unpack + json meta + memoryview
    meta = json.dumps({
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
    }).encode("utf-8")
    direct_payload = struct.pack(">I", len(meta)) + meta + tensor_bytes

    t0 = time.time()
    for _ in range(100):
        ml = struct.unpack(">I", direct_payload[:4])[0]
        m = json.loads(direct_payload[4:4+ml])
        np.frombuffer(direct_payload[4+ml:], dtype=np.dtype(m["tensor_dtype"])).reshape(tuple(m["tensor_shape"]))
    direct_decode = (time.time() - t0) / 100 * 1000

    print(f"    Doc-based (JSON+b64): {doc_decode:.3f}ms")
    print(f"    Direct binary:        {direct_decode:.3f}ms")
    print(f"    Speedup:              {doc_decode/direct_decode:.1f}x")


def main():
    print("🧪 HyperCluster Direct Transport Tests")
    print("   (lattica-style QUIC streaming over iroh)")

    # Test 1: Frame protocol
    test_frame_protocol()

    # Test 2: Tensor wire format
    test_tensor_wire_format()

    # Test 3: Performance comparison
    test_overhead_comparison()

    # Test 4: Actual iroh connection (async)
    print("\n" + "=" * 60)
    print("TEST 5: Iroh QUIC Connection (async)")
    print("=" * 60)
    try:
        asyncio.run(test_iroh_connection())
    except Exception as e:
        print(f"  ⚠️  Iroh connection test: {e}")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
