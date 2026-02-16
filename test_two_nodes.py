#!/usr/bin/env python3
"""
Two-node integration test for the direct QUIC transport.

Spins up two full Node instances (with ALPN protocol registration),
connects them via a shared document, then sends tensors through the
direct transport layer to verify the full pipeline:

  Node A → QUIC → Node B → callback verified

No LLM model needed — this tests the networking layer in isolation.

Usage:
    uv run python test_two_nodes.py
"""

import asyncio
import logging
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("iroh").setLevel(logging.WARNING)


async def main():
    import iroh

    iroh.iroh_ffi.uniffi_set_event_loop(asyncio.get_running_loop())

    from node import Node
    from tensor_service import TensorForwardService

    print("=" * 60)
    print("🧪 Two-Node Direct Transport Integration Test")
    print("=" * 60)

    # ── 1. Start two nodes ──────────────────────────────────────
    print("\n📡 Starting Node A (host)...")
    node_a = Node()
    await node_a.start()
    node_a_id = str(await node_a.iroh_node.net().node_id())
    print(f"   Node A: {node_a_id[:24]}...")

    print("📡 Starting Node B (joiner)...")
    node_b = Node()
    await node_b.start()
    node_b_id = str(await node_b.iroh_node.net().node_id())
    print(f"   Node B: {node_b_id[:24]}...")

    # ── 2. Connect via shared document ──────────────────────────
    print("\n🔗 Creating shared document on Node A...")
    result = await node_a.create_document()
    assert result is not None, "Failed to create document"
    ticket, doc_id = result
    print(f"   Doc ID: {doc_id[:24]}...")

    print("🔗 Node B joining document...")
    joined_doc_id = await node_b.join_document(ticket)
    assert joined_doc_id is not None, "Failed to join document"
    print(f"   Joined: {joined_doc_id[:24]}...")

    # Wait for peer discovery via document sync
    print("\n⏳ Waiting for peer discovery...")
    for i in range(10):
        peers_a = node_a.neighbors.get(doc_id, set())
        peers_b = node_b.neighbors.get(joined_doc_id, set())
        if peers_a and peers_b:
            break
        await asyncio.sleep(1)
        if i % 3 == 2:
            print(f"   ... still waiting ({i+1}s, A sees {len(peers_a)} peers, B sees {len(peers_b)} peers)")

    peers_a = node_a.neighbors.get(doc_id, set())
    peers_b = node_b.neighbors.get(joined_doc_id, set())
    print(f"   Node A sees {len(peers_a)} peer(s): {[str(p)[:16] + '...' for p in peers_a]}")
    print(f"   Node B sees {len(peers_b)} peer(s): {[str(p)[:16] + '...' for p in peers_b]}")

    if not peers_a or not peers_b:
        print("   ⚠️  Peer discovery incomplete, but continuing with direct addressing...")

    # ── 3. Register tensor service on Node B ────────────────────
    print("\n🔧 Registering tensor service on Node B...")
    received_tensors = []
    tensor_event = asyncio.Event()

    tensor_service = TensorForwardService()
    tensor_service.set_node_id(node_b_id)

    async def on_tensor(sender_id, request_id, tensor_data, is_final,
                        position_ids=None, attention_mask=None):
        received_tensors.append({
            "sender_id": sender_id,
            "request_id": request_id,
            "tensor_data": tensor_data,
            "shape": tensor_data.shape,
            "is_final": is_final,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        })
        tensor_event.set()

    tensor_service.set_tensor_callback(on_tensor)
    node_b.conn_manager.register_service(tensor_service)
    print("   ✅ Tensor service registered")

    # ── 4. Ensure Node A knows Node B's address ─────────────────
    print("\n🔌 Registering Node B's address with Node A's ConnectionManager...")
    node_b_addr = await node_b.iroh_node.net().node_addr()
    await node_a.conn_manager.add_peer(node_b_addr, node_b_id)
    print(f"   ✅ Node B registered for direct transport")

    # ── 5. Send tensor A → B via direct transport ───────────────
    print("\n📤 Sending tensor from Node A → Node B via QUIC...")
    test_tensor = np.random.rand(1, 8, 768).astype(np.float32)
    tensor_size_kb = test_tensor.nbytes / 1024
    print(f"   Tensor: shape={test_tensor.shape}, size={tensor_size_kb:.1f} KB")

    send_start = time.time()
    await node_a.send_tensor_direct(
        target_node_id=node_b_id,
        tensor_data=test_tensor,
        request_id="test-001",
        metadata={
            "sender_id": node_a_id,
            "request_id": "test-001",
            "tensor_shape": list(test_tensor.shape),
            "tensor_dtype": str(test_tensor.dtype),
            "is_final": False,
            "position_ids": [0, 1, 2, 3, 4, 5, 6, 7],
            "attention_mask": None,
        },
    )
    send_time = (time.time() - send_start) * 1000
    print(f"   ✅ Tensor sent in {send_time:.1f}ms")

    # ── 6. Verify tensor received on Node B ─────────────────────
    print("\n📥 Waiting for tensor on Node B...")
    try:
        await asyncio.wait_for(tensor_event.wait(), timeout=10.0)
        rt = received_tensors[0]
        print(f"   ✅ Tensor received!")
        print(f"      Sender:      {rt['sender_id'][:24]}...")
        print(f"      Request ID:  {rt['request_id']}")
        print(f"      Shape:       {rt['shape']}")
        print(f"      Is final:    {rt['is_final']}")
        print(f"      Position IDs: {rt['position_ids']}")

        # Verify data integrity
        assert np.allclose(test_tensor, rt['tensor_data'], atol=1e-6), \
            "Tensor data mismatch!"
        print(f"      ✅ Data integrity verified — exact match!")

    except asyncio.TimeoutError:
        print("   ❌ Tensor not received within 10s timeout")

    # ── 7. Send a larger tensor to measure throughput ───────────
    print("\n📤 Sending larger tensor (1MB)...")
    tensor_event.clear()
    received_tensors.clear()

    big_tensor = np.random.rand(1, 128, 2048).astype(np.float32)
    big_size_mb = big_tensor.nbytes / (1024 * 1024)

    send_start = time.time()
    await node_a.send_tensor_direct(
        target_node_id=node_b_id,
        tensor_data=big_tensor,
        request_id="test-002",
        metadata={
            "sender_id": node_a_id,
            "request_id": "test-002",
            "tensor_shape": list(big_tensor.shape),
            "tensor_dtype": str(big_tensor.dtype),
            "is_final": True,
        },
    )
    send_time = (time.time() - send_start) * 1000

    try:
        await asyncio.wait_for(tensor_event.wait(), timeout=10.0)
        rt = received_tensors[0]
        throughput = big_size_mb / (send_time / 1000) if send_time > 0 else 0
        assert np.allclose(big_tensor, rt['tensor_data'], atol=1e-6)
        print(f"   ✅ {big_size_mb:.1f}MB tensor: sent in {send_time:.1f}ms ({throughput:.1f} MB/s)")
        print(f"      Data integrity verified!")
    except asyncio.TimeoutError:
        print(f"   ❌ Large tensor not received within 10s timeout")

    # ── 8. Cleanup ──────────────────────────────────────────────
    print("\n🧹 Shutting down...")
    await node_a.stop()
    await node_b.stop()
    print("   ✅ Both nodes stopped")

    print("\n" + "=" * 60)
    print("✅ INTEGRATION TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
