"""
Tensor forwarding RPC service for the direct transport layer.

This handles incoming tensor data sent via QUIC streams,
replacing the slow Doc-based CONTENT_READY polling mechanism.

Like lattica's RpcService handlers, this receives framed binary data
and dispatches it to the ring pipeline coordinator.
"""

import asyncio
import json
import logging
import struct
import time
from typing import Any, Callable, Dict, Optional

import numpy as np
from direct_transport import RpcServiceHandler

logger = logging.getLogger(__name__)


class TensorForwardService(RpcServiceHandler):
    """
    RPC service handler for direct tensor forwarding.

    Receives tensor data sent via ConnectionManager.send_tensor()
    and dispatches it to the ring pipeline coordinator.

    Wire format of incoming data:
      [4 bytes meta_len][JSON metadata][raw tensor bytes]

    This replaces the old flow of:
      Doc.set_bytes → CONTENT_READY event → blobs.read_to_bytes → retry loop
    with:
      QUIC stream → frame decode → numpy reconstruct → coordinator.handle_incoming_tensor
    """

    def __init__(self):
        self._tensor_callback: Optional[Callable] = None
        self._my_node_id: Optional[str] = None

    def service_name(self) -> str:
        return "tensor"

    def set_tensor_callback(self, callback: Callable):
        """
        Register callback for received tensors.

        Callback signature:
            async def on_tensor(sender_id, request_id, tensor_data, metadata) -> None
        """
        self._tensor_callback = callback

    def set_node_id(self, node_id: str):
        self._my_node_id = node_id

    async def handle_request(self, method: str, data: bytes) -> bytes:
        """Handle incoming tensor forward requests."""
        if method == "forward":
            await self._handle_tensor_forward(data)
            return b"ok"
        else:
            raise NotImplementedError(f"Unknown method: tensor.{method}")

    async def _handle_tensor_forward(self, raw_data: bytes):
        """
        Parse and dispatch an incoming tensor forward.

        This is the hot path — replaces the entire Doc-based tensor
        receive flow (CONTENT_READY → cache check → blob fetch → retry)
        with direct binary decode.
        """
        receive_start = time.time()

        try:
            # Parse wire format: [4 bytes meta_len][JSON metadata][raw tensor bytes]
            if len(raw_data) < 4:
                logger.error("Tensor forward data too short")
                return

            meta_len = struct.unpack(">I", raw_data[:4])[0]
            if len(raw_data) < 4 + meta_len:
                logger.error(f"Tensor forward incomplete: expected {4 + meta_len}, got {len(raw_data)}")
                return

            meta_json = raw_data[4 : 4 + meta_len]
            tensor_bytes = raw_data[4 + meta_len :]

            metadata = json.loads(meta_json.decode("utf-8"))

            # Extract tensor metadata
            tensor_shape = tuple(metadata.get("tensor_shape", []))
            tensor_dtype = np.dtype(metadata.get("tensor_dtype", "float32"))
            request_id = metadata.get("request_id", "unknown")
            sender_id = metadata.get("sender_id", "unknown")
            is_final = metadata.get("is_final", False)

            # Reconstruct tensor
            tensor_data = np.frombuffer(tensor_bytes, dtype=tensor_dtype).reshape(tensor_shape)

            # Extract position_ids and attention_mask (critical for RoPE)
            # Support compact position format: integer for single positions
            position_ids = None
            attention_mask = None
            raw_pos = metadata.get("position_ids")
            if raw_pos is not None:
                if isinstance(raw_pos, int):
                    # Compact format: single integer → [[N]]
                    position_ids = np.array([[raw_pos]], dtype=np.int64)
                else:
                    position_ids = np.array(raw_pos, dtype=np.int64)
            if metadata.get("attention_mask") is not None:
                attention_mask = np.array(metadata["attention_mask"], dtype=np.bool_)

            receive_time = (time.time() - receive_start) * 1000
            size_mb = len(tensor_bytes) / (1024 * 1024)

            logger.info(
                f"📥 Tensor received via QUIC: {size_mb:.2f}MB, "
                f"shape={tensor_shape}, decode={receive_time:.1f}ms"
            )

            # Dispatch to ring coordinator
            if self._tensor_callback:
                await self._tensor_callback(
                    sender_id=sender_id,
                    request_id=request_id,
                    tensor_data=tensor_data,
                    is_final=is_final,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )
            else:
                logger.warning("No tensor callback registered, dropping tensor")

        except Exception as e:
            logger.error(f"Error handling tensor forward: {e}", exc_info=True)
