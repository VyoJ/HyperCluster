"""
Ring Pipeline Inference Engine - Inspired by prima.cpp

This module implements ring-based pipelined inference where:
1. Nodes form a logical ring topology
2. Activations flow around the ring multiple times (cycles)
3. Each node processes layers in windows
4. Prefetching overlaps model loading with computation
"""

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from shard import Shard

logger = logging.getLogger(__name__)


@dataclass
class RingPosition:
    """Represents a node's position in the ring."""
    
    rank: int  # 0 = head node (coordinator)
    world_size: int  # Total number of nodes
    prev_node_id: str  # Node to receive from
    next_node_id: str  # Node to send to
    is_head: bool  # True if rank == 0
    is_tail: bool  # True if rank == world_size - 1


@dataclass
class LayerWindow:
    """Represents a window of layers assigned to a node."""
    
    node_id: str
    rank: int
    layer_start: int  # First layer in window
    layer_end: int    # Last layer in window (inclusive)
    total_layers: int


@dataclass
class InferenceState:
    """Maintains state during ring pipeline inference."""
    
    request_id: str
    current_cycle: int  # Which round through the ring (0-indexed)
    total_cycles: int   # Total rounds needed
    current_layer: int  # Next layer to process
    hidden_states: Optional[np.ndarray] = None
    kv_cache: Optional[Dict] = None
    metadata: Optional[Dict] = None


class RingPipelineCoordinator:
    """
    Coordinates ring pipeline inference across nodes.
    
    Based on prima.cpp's ring communication pattern:
    - Rank 0 (head) initiates requests and collects final output
    - Data flows: Rank 0 → 1 → 2 → ... → N-1 → 0 (ring)
    - Multiple cycles needed when layers > sum(layer_windows)
    """
    
    def __init__(self, inference_engine, network_node):
        """
        Initialize ring coordinator.
        
        Args:
            inference_engine: The inference engine (TransformersShardedInferenceEngine)
            network_node: The network node (Node instance)
        """
        self.inference_engine = inference_engine
        self.network = network_node
        self.ring_position: Optional[RingPosition] = None
        self.layer_window: Optional[LayerWindow] = None
        
        # Track active inference requests
        self.active_requests: Dict[str, InferenceState] = OrderedDict()
        
        # Prefetch queue
        self.prefetch_queue = asyncio.Queue()
        self.prefetch_task: Optional[asyncio.Task] = None
    
    async def initialize_ring(
        self, 
        topology_nodes: List[Tuple[str, any]], 
        my_node_id: str,
        model_total_layers: int
    ):
        """
        Initialize ring topology and layer assignment.
        
        Args:
            topology_nodes: List of (node_id, capabilities) tuples
            my_node_id: This node's ID
            model_total_layers: Total layers in model
        """
        # Sort nodes to establish consistent ring order
        # Use capabilities (memory) to determine rank
        sorted_nodes = sorted(
            topology_nodes, 
            key=lambda x: (x[1].memory, x[0]), 
            reverse=True
        )
        
        # Find my rank
        rank = next(i for i, (nid, _) in enumerate(sorted_nodes) if nid == my_node_id)
        world_size = len(sorted_nodes)
        
        # Determine ring neighbors
        prev_rank = (rank - 1) % world_size
        next_rank = (rank + 1) % world_size
        
        prev_node_id = sorted_nodes[prev_rank][0]
        next_node_id = sorted_nodes[next_rank][0]
        
        self.ring_position = RingPosition(
            rank=rank,
            world_size=world_size,
            prev_node_id=prev_node_id,
            next_node_id=next_node_id,
            is_head=(rank == 0),
            is_tail=(rank == world_size - 1)
        )
        
        # Calculate layer windows (memory-weighted)
        layer_windows = self._calculate_layer_windows(
            sorted_nodes, model_total_layers
        )
        
        self.layer_window = layer_windows[rank]
        
        logger.info(
            f"Ring initialized: rank={rank}/{world_size}, "
            f"layers=[{self.layer_window.layer_start}:{self.layer_window.layer_end}], "
            f"prev={prev_node_id[:8]}, next={next_node_id[:8]}"
        )
        
        # Start prefetch worker
        if self.prefetch_task is None:
            self.prefetch_task = asyncio.create_task(self._prefetch_worker())
    
    def _calculate_layer_windows(
        self, 
        sorted_nodes: List[Tuple[str, any]], 
        total_layers: int
    ) -> List[LayerWindow]:
        """
        Calculate layer windows for each node based on memory.
        Similar to prima.cpp's assign_layers_to_device().
        
        Args:
            sorted_nodes: Nodes sorted by memory (descending)
            total_layers: Total model layers
            
        Returns:
            List of LayerWindow for each node
        """
        total_memory = sum(cap.memory for _, cap in sorted_nodes)
        
        windows = []
        current_layer = 0
        
        for rank, (node_id, capabilities) in enumerate(sorted_nodes):
            # Proportional allocation
            memory_fraction = capabilities.memory / total_memory
            num_layers = int(total_layers * memory_fraction)
            
            # Ensure last node gets remaining layers
            if rank == len(sorted_nodes) - 1:
                num_layers = total_layers - current_layer
            
            # Ensure at least 1 layer per node
            num_layers = max(1, num_layers)
            
            window = LayerWindow(
                node_id=node_id,
                rank=rank,
                layer_start=current_layer,
                layer_end=current_layer + num_layers - 1,
                total_layers=total_layers
            )
            windows.append(window)
            current_layer += num_layers
        
        return windows
    
    def this_layer_is_mine(self, layer_id: int) -> bool:
        """
        Check if a layer belongs to this node.
        Based on prima.cpp's this_layer_is_mine().
        """
        if not self.layer_window:
            return False
        
        return (
            self.layer_window.layer_start <= layer_id <= self.layer_window.layer_end
        )
    
    def calculate_cycles_needed(self, total_layers: int) -> int:
        """
        Calculate how many cycles through the ring are needed.
        
        In prima.cpp, this happens when model has more layers than
        can be handled in a single pass through all devices.
        """
        if not self.layer_window:
            return 1
        
        # Sum of all layer windows
        total_window = self.layer_window.layer_end + 1
        
        # Cycles = ceil(total_layers / total_window)
        cycles = (total_layers + total_window - 1) // total_window
        return cycles
    
    async def start_inference(
        self, 
        request_id: str, 
        prompt: str, 
        shard: Shard,
        max_tokens: int = 256
    ) -> List[int]:
        """
        Start ring pipeline inference (head node only).
        
        Args:
            request_id: Unique request ID
            prompt: Input prompt
            shard: Full model shard info
            max_tokens: Max tokens to generate
            
        Returns:
            List of generated token IDs
        """
        if not self.ring_position or not self.ring_position.is_head:
            raise ValueError("Only head node can start inference")
        
        logger.info(f"Starting ring inference for request {request_id}")
        
        # Encode prompt
        tokens = await self.inference_engine.encode(shard, prompt)
        input_tokens = tokens.reshape(1, -1)
        
        generated_tokens = []
        
        # Auto-regressive generation loop
        for step in range(max_tokens):
            # Process through ring pipeline
            logits = await self._ring_forward_pass(
                request_id=request_id,
                input_data=input_tokens,
                shard=shard
            )
            
            if logits is None:
                logger.warning("Ring forward pass returned None")
                break
            
            # Sample next token
            next_token = await self.inference_engine.sample(logits)
            generated_tokens.append(int(next_token[0]))
            
            # Check for EOS
            # TODO: Get proper EOS token from tokenizer
            if next_token[0] in [2, 0]:  # Common EOS tokens
                break
            
            # Prepare for next iteration
            input_tokens = next_token.reshape(1, 1)
            
            logger.debug(f"Step {step}: token={next_token[0]}")
        
        return generated_tokens
    
    async def _ring_forward_pass(
        self,
        request_id: str,
        input_data: np.ndarray,
        shard: Shard
    ) -> Optional[np.ndarray]:
        """
        Execute one forward pass through the ring.
        
        This implements prima.cpp's llama_decode_internal() ring logic:
        1. Process local layers
        2. Send to next node
        3. Receive from previous node (if not first cycle)
        4. Repeat until all layers processed
        
        Returns:
            Final logits (head node only)
        """
        total_cycles = self.calculate_cycles_needed(shard.n_layers)
        
        # Initialize state
        state = InferenceState(
            request_id=request_id,
            current_cycle=0,
            total_cycles=total_cycles,
            current_layer=0,
            hidden_states=input_data,
            metadata={"step": 0}
        )
        
        self.active_requests[request_id] = state
        
        # Head node starts the ring
        result = await self._process_and_forward(request_id, state, shard)
        
        # Wait for completion (result comes back from ring)
        timeout = 30.0  # seconds
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if state.current_layer >= shard.n_layers:
                # All layers processed
                break
            await asyncio.sleep(0.1)
        
        # Clean up
        self.active_requests.pop(request_id, None)
        
        return result
    
    async def _process_and_forward(
        self,
        request_id: str,
        state: InferenceState,
        shard: Shard
    ) -> Optional[np.ndarray]:
        """
        Process assigned layers and forward to next node.
        
        Based on prima.cpp's layer processing loop in llama_decode_internal.
        """
        # Process layers in my window
        current_data = state.hidden_states
        
        layers_to_process = []
        for layer_id in range(
            state.current_layer, 
            min(state.current_layer + 10, shard.n_layers)  # Process in chunks
        ):
            if self.this_layer_is_mine(layer_id):
                layers_to_process.append(layer_id)
        
        if layers_to_process:
            logger.debug(
                f"Processing layers {layers_to_process[0]}-{layers_to_process[-1]}"
            )
            
            # Run inference on assigned layers
            # For now, use the full model inference
            # TODO: Implement true layer-by-layer processing
            output_data, new_state = await self.inference_engine.infer_tensor(
                request_id=request_id,
                shard=shard,
                input_data=current_data,
                inference_state=state.metadata
            )
            
            current_data = output_data
            state.hidden_states = current_data
            state.current_layer = layers_to_process[-1] + 1
        
        # Check if this is the last layer
        is_final_layer = state.current_layer >= shard.n_layers
        
        if is_final_layer:
            # Return logits (head node only)
            if self.ring_position and self.ring_position.is_head:
                return current_data
            else:
                # Send back to head
                await self._send_to_node(
                    target_node_id=self._find_head_node_id(),
                    data=current_data,
                    request_id=request_id,
                    is_final=True
                )
                return None
        else:
            # Forward to next node in ring
            if not self.ring_position:
                return None
            await self._send_to_node(
                target_node_id=self.ring_position.next_node_id,
                data=current_data,
                request_id=request_id,
                is_final=False
            )
            return None
    
    async def handle_incoming_tensor(
        self,
        sender_id: str,
        request_id: str,
        tensor_data: np.ndarray,
        shard: Shard,
        is_final: bool = False
    ):
        """
        Handle incoming tensor from previous node in ring.
        
        This is called when receiving a "tensor_forward" message.
        Based on prima.cpp's llama_recv_tensors().
        """
        logger.debug(
            f"Received tensor from {sender_id[:8]} for request {request_id}"
        )
        
        # Restore or create state
        if request_id in self.active_requests:
            state = self.active_requests[request_id]
        else:
            state = InferenceState(
                request_id=request_id,
                current_cycle=0,
                total_cycles=1,
                current_layer=0,
                metadata={}
            )
            self.active_requests[request_id] = state
        
        state.hidden_states = tensor_data
        
        if is_final and self.ring_position and self.ring_position.is_head:
            # Final result received at head
            return tensor_data
        
        # Process and forward
        await self._process_and_forward(request_id, state, shard)
    
    async def _send_to_node(
        self,
        target_node_id: str,
        data: np.ndarray,
        request_id: str,
        is_final: bool
    ):
        """
        Send tensor to another node via Iroh.
        Based on prima.cpp's llama_send_tensors().
        """
        # Get document ID for communication
        if not self.network.documents:
            logger.error("No documents available for communication")
            return
        
        doc_id = next(iter(self.network.documents))
        
        # Serialize and send
        import base64
        tensor_bytes = data.tobytes()
        tensor_b64 = base64.b64encode(tensor_bytes).decode("utf-8")
        
        message = {
            "type": "ring_tensor_forward",
            "sender_id": str(await self.network.iroh_node.net().node_id()),
            "target_node_id": target_node_id,
            "request_id": request_id,
            "payload": {
                "tensor_data": tensor_b64,
                "tensor_shape": list(data.shape),
                "tensor_dtype": str(data.dtype),
                "is_final": is_final,
            },
            "timestamp": time.time(),
        }
        
        await self.network.send_message(doc_id, message)
    
    def _find_head_node_id(self) -> str:
        """Find the head node (rank 0) ID."""
        # This should be stored during initialization
        # For now, assume we can query topology
        topology_nodes = self.network.topology.all_nodes()
        if not topology_nodes:
            return self.network.topology.active_node_id or ""
        
        # Head is the node with most memory (rank 0)
        sorted_nodes = sorted(
            topology_nodes,
            key=lambda x: (x[1].memory, x[0]),
            reverse=True
        )
        return sorted_nodes[0][0]
    
    async def _prefetch_worker(self):
        """
        Background worker for prefetching model weights.
        
        Based on prima.cpp's manage_graph_tensors() with POSIX_MADV_WILLNEED.
        
        In Python, we can use:
        - mmap with MADV_WILLNEED
        - Preload model layers into cache
        - Async loading of next layers
        """
        logger.info("Prefetch worker started")
        
        while True:
            try:
                # Get next layer to prefetch
                next_layer_id = await asyncio.wait_for(
                    self.prefetch_queue.get(), 
                    timeout=1.0
                )
                
                # Prefetch logic here
                # For transformers models, this would involve:
                # 1. Identifying which weights are needed
                # 2. Touching memory pages to bring into cache
                # 3. Or pre-loading layers from disk
                
                logger.debug(f"Prefetching layer {next_layer_id}")
                
                # TODO: Implement actual prefetching
                # For now, this is a placeholder
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Prefetch worker error: {e}")
                await asyncio.sleep(1.0)
    
    async def schedule_prefetch(self, layer_id: int):
        """Schedule a layer to be prefetched."""
        if not self.prefetch_queue.full():
            await self.prefetch_queue.put(layer_id)
    
    async def shutdown(self):
        """Shutdown the ring coordinator."""
        if self.prefetch_task:
            self.prefetch_task.cancel()
            try:
                await self.prefetch_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Ring coordinator shutdown complete")

