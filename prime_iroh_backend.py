"""
Prime-Iroh Communication Backend for HyperCluster

This module provides an efficient P2P communication backend using prime-iroh
for direct streaming tensor transfers in the ring pipeline architecture.

Key Features:
- Direct peer-to-peer streaming communication
- Asynchronous send/receive operations optimized for pipeline parallelism
- Integration with HyperCluster's ring pipeline
"""

import asyncio
import logging
import numpy as np
from typing import Optional, Dict, Any
import struct

logger = logging.getLogger(__name__)


class PrimeIrohBackend:
    """
    Communication backend using prime-iroh for efficient tensor transfers.
    
    This backend provides:
    - Direct P2P streaming for large tensor data
    - Asynchronous send/receive operations
    - Integration with HyperCluster's existing topology
    """
    
    def __init__(self):
        self.prime_node = None
        self.send_peer_id: Optional[str] = None
        self.recv_peer_id: Optional[str] = None
        self.initialized = False
        self._send_lock = asyncio.Lock()
        self._recv_lock = asyncio.Lock()
    
    async def initialize(self, node_id: str, send_peer_id: Optional[str] = None, recv_peer_id: Optional[str] = None):
        """
        Initialize the prime-iroh backend.
        
        Args:
            node_id: This node's identifier
            send_peer_id: Peer ID to send data to
            recv_peer_id: Peer ID to receive data from
        
        Note: The actual prime-iroh API may differ. This implementation is based on
        the README documentation and may need adjustments when the package is available.
        """
        try:
            import prime_iroh
            
            logger.info("Initializing prime-iroh backend...")
            
            # Create prime-iroh node
            # TODO: Verify actual API - may need configuration parameters
            # such as bind addresses, secret keys, or other network parameters
            self.prime_node = prime_iroh.Node()
            
            # Configure send and receive peers if provided
            if send_peer_id:
                self.send_peer_id = send_peer_id
                logger.info(f"Send peer configured: {send_peer_id[:16]}...")
            
            if recv_peer_id:
                self.recv_peer_id = recv_peer_id
                logger.info(f"Receive peer configured: {recv_peer_id[:16]}...")
            
            self.initialized = True
            logger.info("Prime-iroh backend initialized successfully")
            
        except ImportError:
            logger.error("prime-iroh package not installed. Install with: pip install prime-iroh")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize prime-iroh backend: {e}")
            raise
    
    async def send_tensor(
        self,
        data: np.ndarray,
        target_peer_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send a tensor to a peer using prime-iroh.
        
        Args:
            data: NumPy array to send
            target_peer_id: Target peer's ID
            metadata: Optional metadata to send with tensor
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized or not self.prime_node:
            logger.error("Prime-iroh backend not initialized")
            return False
        
        async with self._send_lock:
            try:
                # Serialize tensor to bytes
                tensor_bytes = data.tobytes()
                
                # Prepare header with shape and dtype information
                shape_bytes = struct.pack(f'{len(data.shape)}i', *data.shape)
                dtype_str = str(data.dtype).encode('utf-8')
                dtype_len = struct.pack('I', len(dtype_str))
                
                # Combine header and data
                # Format: [ndim(4 bytes)][shape][dtype_len(4 bytes)][dtype][tensor_data]
                ndim = struct.pack('I', len(data.shape))
                payload = ndim + shape_bytes + dtype_len + dtype_str + tensor_bytes
                
                size_mb = len(payload) / 1024 / 1024
                logger.info(f"📤 Prime-iroh sending {size_mb:.2f} MB to {target_peer_id[:16]}...")
                
                # Use prime-iroh's isend API (mirroring torch.distributed)
                # TODO: Verify actual API against prime-iroh documentation
                # The API usage here is based on README examples and may need adjustment
                send_work = self.prime_node.isend(payload, target_peer_id)
                await send_work.wait()
                
                logger.info(f"✅ Prime-iroh send completed")
                return True
                
            except Exception as e:
                logger.error(f"Prime-iroh send failed: {e}", exc_info=True)
                return False
    
    async def receive_tensor(self, from_peer_id: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Receive a tensor from a peer using prime-iroh.
        
        Args:
            from_peer_id: Expected peer ID (optional)
            
        Returns:
            Received NumPy array, or None if failed
        """
        if not self.initialized or not self.prime_node:
            logger.error("Prime-iroh backend not initialized")
            return None
        
        async with self._recv_lock:
            try:
                logger.info(f"📥 Prime-iroh waiting to receive tensor...")
                
                # Use prime-iroh's irecv API
                # TODO: Verify if the API supports filtering by peer ID
                # This may need adjustment for ring topology where we need to
                # distinguish messages from specific peers
                recv_work = self.prime_node.irecv()
                payload = await recv_work.wait()
                
                # Parse header
                offset = 0
                ndim = struct.unpack('I', payload[offset:offset+4])[0]
                offset += 4
                
                shape = struct.unpack(f'{ndim}i', payload[offset:offset+4*ndim])
                offset += 4 * ndim
                
                dtype_len = struct.unpack('I', payload[offset:offset+4])[0]
                offset += 4
                
                dtype_str = payload[offset:offset+dtype_len].decode('utf-8')
                offset += dtype_len
                
                # Extract tensor data
                tensor_bytes = payload[offset:]
                
                # Reconstruct NumPy array
                dtype = np.dtype(dtype_str)
                data = np.frombuffer(tensor_bytes, dtype=dtype).reshape(shape)
                
                size_mb = len(tensor_bytes) / 1024 / 1024
                logger.info(f"✅ Prime-iroh received {size_mb:.2f} MB, shape: {shape}")
                
                return data
                
            except Exception as e:
                logger.error(f"Prime-iroh receive failed: {e}", exc_info=True)
                return None
    
    async def shutdown(self):
        """Shutdown the prime-iroh backend."""
        if self.prime_node:
            try:
                # Clean shutdown if supported by prime-iroh
                # Note: Actual API may differ
                logger.info("Shutting down prime-iroh backend")
                self.initialized = False
                self.prime_node = None
            except Exception as e:
                logger.error(f"Error during prime-iroh shutdown: {e}")


def is_prime_iroh_available() -> bool:
    """Check if prime-iroh is available."""
    try:
        import prime_iroh
        return True
    except ImportError:
        return False
