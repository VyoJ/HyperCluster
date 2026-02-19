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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shard import Shard
from stats_logger import get_stats_logger

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
    layer_end: int  # Last layer in window (inclusive)
    total_layers: int


@dataclass
class InferenceState:
    """Maintains state during ring pipeline inference."""

    request_id: str
    current_cycle: int  # Which round through the ring (0-indexed)
    total_cycles: int  # Total rounds needed
    current_layer: int  # Next layer to process
    hidden_states: Optional[np.ndarray] = None
    kv_cache: Optional[Dict] = None
    metadata: Optional[Dict] = None
    # Critical metadata for transformers (like prima.cpp's sync_meta)
    position_ids: Optional[np.ndarray] = None  # Token positions for RoPE
    attention_mask: Optional[np.ndarray] = None  # Attention mask for tokens
    seq_len: int = 0  # Current sequence length
    final_result: Optional[np.ndarray] = None  # Final logits when ring completes
    generation_step: int = (
        0  # Which token we're generating (0=prompt, 1+=autoregressive)
    )
    last_processed_step: int = -1  # Last step this node processed (to detect new steps)


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

        # Event-based signaling for zero-latency wakeup (replaces polling)
        self.completion_events: Dict[str, asyncio.Event] = {}

        # Prefetch queue
        self.prefetch_queue = asyncio.Queue()
        self.prefetch_task: Optional[asyncio.Task] = None

    def _my_shard(self, base_shard: Shard) -> Shard:
        """Return a shard matching this node's layer window.

        The inference engine is loaded with exactly the layers in our
        ``layer_window``.  Every call to ``ensure_shard`` / ``infer_tensor``
        must receive this *node-local* shard so the engine does NOT
        reload the model.  The ``base_shard`` (full model) is only used
        for ``n_layers`` / ``model_id`` metadata.
        """
        if self.layer_window is None:
            return base_shard
        return Shard(
            model_id=base_shard.model_id,
            start_layer=self.layer_window.layer_start,
            end_layer=self.layer_window.layer_end,
            n_layers=base_shard.n_layers,
        )

    async def initialize_ring(
        self,
        topology_nodes: List[Tuple[str, Any]],
        my_node_id: str,
        model_total_layers: int,
    ):
        """
        Initialize ring topology and layer assignment.

        Args:
            topology_nodes: List of (node_id, capabilities) tuples
            my_node_id: This node's ID
            model_total_layers: Total layers in model
        """
        logger.info(
            f"🔍 Initializing ring with {len(topology_nodes)} nodes in topology"
        )
        for i, (nid, cap) in enumerate(topology_nodes):
            logger.info(f"   Node {i}: {nid[:16]}... - {cap.memory:.1f} GB")

        # Sort nodes to establish consistent ring order
        # Use capabilities (memory) to determine rank
        sorted_nodes = sorted(
            topology_nodes, key=lambda x: (x[1].memory, x[0]), reverse=True
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
            is_tail=(rank == world_size - 1),
        )

        # Calculate layer windows (memory-weighted)
        layer_windows = self._calculate_layer_windows(sorted_nodes, model_total_layers)

        self.layer_window = layer_windows[rank]

        # Detailed ring topology logging
        logger.info("=" * 80)
        logger.info("🔗 RING TOPOLOGY INITIALIZED")
        logger.info("=" * 80)
        logger.info("📍 My Position:")
        logger.info(f"   Rank: {rank}/{world_size}")
        logger.info(
            f"   Role: {'HEAD (Coordinator)' if rank == 0 else f'WORKER-{rank}'}"
        )
        logger.info(f"   Node ID: {my_node_id[:16]}...")
        logger.info("")
        logger.info("🔄 Ring Structure:")
        logger.info(
            f"   Previous: {prev_node_id[:16]}... (rank {(rank - 1) % world_size})"
        )
        logger.info(f"   Current:  {my_node_id[:16]}... (rank {rank})")
        logger.info(
            f"   Next:     {next_node_id[:16]}... (rank {(rank + 1) % world_size})"
        )
        logger.info("")
        logger.info("📊 My Layer Assignment:")
        logger.info(
            f"   Layers: {self.layer_window.layer_start} → {self.layer_window.layer_end}"
        )
        logger.info(
            f"   Count:  {self.layer_window.layer_end - self.layer_window.layer_start + 1} layers"
        )
        logger.info(f"   Total:  {model_total_layers} layers in model")
        logger.info("")
        logger.info("🌍 Full Cluster Distribution:")
        for i, window in enumerate(layer_windows):
            node_short = window.node_id[:16]
            layer_count = window.layer_end - window.layer_start + 1
            role = "HEAD" if i == 0 else f"WORK-{i}"
            marker = "👉 " if i == rank else "   "
            logger.info(
                f"{marker}Rank {i} ({role}): Layers {window.layer_start:3d}-{window.layer_end:3d} "
                f"({layer_count:2d} layers) - {node_short}..."
            )
        logger.info("=" * 80)

        # Pre-warm QUIC connections to ring neighbors
        # This eliminates the 6-20ms connection setup latency on the first tensor send
        if world_size > 1 and self.network.conn_manager and self.network.use_direct_transport:
            await self._prewarm_connections()

        # Start prefetch worker
        if self.prefetch_task is None:
            self.prefetch_task = asyncio.create_task(self._prefetch_worker())

    def _calculate_layer_windows(
        self, sorted_nodes: List[Tuple[str, Any]], total_layers: int
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
        logger.info("📐 Calculating layer distribution...")
        total_memory = sum(cap.memory for _, cap in sorted_nodes)
        logger.info(f"   Total cluster memory: {total_memory:.1f} GB")

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

            logger.info(
                f"   Rank {rank}: {capabilities.memory:.1f} GB "
                f"({memory_fraction * 100:.1f}%) → {num_layers} layers"
            )

            window = LayerWindow(
                node_id=node_id,
                rank=rank,
                layer_start=current_layer,
                layer_end=current_layer + num_layers - 1,
                total_layers=total_layers,
            )
            windows.append(window)
            current_layer += num_layers

        return windows

    async def _prewarm_connections(self):
        """
        Pre-warm QUIC connections to ring neighbors during initialization.

        This eliminates the ~6-20ms connection setup + handshake latency
        that would otherwise occur on the first tensor send of each generation.
        """
        if not self.ring_position:
            return

        neighbors = set()
        if self.ring_position.next_node_id:
            neighbors.add(self.ring_position.next_node_id)
        if self.ring_position.prev_node_id:
            neighbors.add(self.ring_position.prev_node_id)
        # Also pre-warm connection to head (for tail→head sampled token return)
        head_id = self._find_head_node_id()
        if head_id:
            neighbors.add(head_id)

        my_node_id = str(await self.network.iroh_node.net().node_id())
        neighbors.discard(my_node_id)  # Don't connect to self

        for peer_id in neighbors:
            try:
                handle = await self.network.conn_manager.connect(peer_id)
                rtt = await handle.ping(timeout=5.0)
                logger.info(
                    f"🔥 Pre-warmed QUIC connection to {peer_id[:16]}... "
                    f"(RTT: {rtt:.1f}ms)"
                )
            except Exception as e:
                logger.warning(
                    f"⚠️  Failed to pre-warm connection to {peer_id[:16]}...: {e}"
                )

    def this_layer_is_mine(self, layer_id: int) -> bool:
        """
        Check if a layer belongs to this node.
        Based on prima.cpp's this_layer_is_mine().
        """
        if not self.layer_window:
            return False

        return self.layer_window.layer_start <= layer_id <= self.layer_window.layer_end

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
        self, request_id: str, prompt: str, shard: Shard, max_tokens: int = 50
    ) -> List[int]:
        """
        Start ring pipeline inference (head node only).

        Args:
            request_id: Unique request ID
            prompt: Input prompt
            shard: Base shard representing full model (used for model_id and n_layers)
            max_tokens: Max tokens to generate

        Returns:
            List of generated token IDs
        """
        if not self.ring_position or not self.ring_position.is_head:
            raise ValueError("Only head node can start inference")

        logger.debug("=" * 80)
        logger.info("🚀 STARTING RING INFERENCE")
        logger.debug("=" * 80)
        logger.debug(f"Request ID: {request_id}")
        logger.debug(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        logger.debug(f"Max tokens: {max_tokens}")
        logger.debug(f"Model layers: {shard.n_layers}")

        # Clear any existing cache for this request
        # This ensures we start with clean state for new prompts
        logger.debug(f"Clearing any existing cache for request {request_id}")
        self.inference_engine.caches.pop(request_id, None)

        # Get stats logger
        stats_logger = get_stats_logger()

        # Encode prompt
        stats_logger.log_encoding_start(request_id)
        start_time = time.time()
        tokens = await self.inference_engine.encode(self._my_shard(shard), prompt)
        input_tokens = tokens.reshape(1, -1)
        encode_time = time.time() - start_time
        stats_logger.log_encoding_end(request_id, len(tokens))

        logger.debug("")
        logger.info("📝 Encoding complete:")
        logger.debug(f"   Input tokens: {len(tokens)}")
        logger.info(f"   Token shape: {input_tokens.shape}")
        logger.debug(f"   Encode time: {encode_time * 1000:.1f}ms")
        logger.debug("=" * 80)

        generated_tokens = []

        # Start inference timing
        stats_logger.log_inference_start(request_id)

        # Auto-regressive generation loop
        for step in range(max_tokens):
            logger.debug("")
            logger.debug(f"🔄 GENERATION STEP {step + 1}/{max_tokens}")
            logger.info(f"   Tokens generated so far: {len(generated_tokens)}")

            # 🐛 DEBUG: Check cache state before generation step
            logger.debug("")
            logger.debug("🔍 PRE-STEP CACHE CHECK")
            logger.debug(f"   Request ID: {request_id}")
            if request_id in self.inference_engine.caches:
                cache = self.inference_engine.caches[request_id]
                if hasattr(cache, "get_seq_length"):
                    # Modern Cache object (DynamicCache in transformers 5.2.0+)
                    from transformers_inference import _get_cache_seq_length
                    num_layers = len(cache) if hasattr(cache, "__len__") else 0
                    seq_len = _get_cache_seq_length(cache)
                    logger.debug(f"   ✅ Cache exists ({type(cache).__name__}): {num_layers} layers, seq_len={seq_len}")
                elif hasattr(cache, "__len__"):
                    logger.debug(f"   ✅ Cache exists (tuple): {len(cache)} layers")
                else:
                    logger.debug(f"   ✅ Cache exists: {type(cache).__name__}")
            else:
                logger.debug("   ❌ No cache found")
            logger.debug("")

            step_start = time.time()

            # Process through ring pipeline
            # CRITICAL: Position calculation for autoregressive generation
            # At the start of step N:
            # - generated_tokens has N items (from steps 0..N-1)
            # - input_tokens is the token we're ABOUT TO PROCESS
            # - For step 0 (prompt): process all prompt tokens at positions [0, 1, 2, ..., len-1]
            # - For step 1: process token sampled in step 0, which goes at position len(prompt) = 8
            # - For step 2: process token sampled in step 1, which goes at position len(prompt) + 1 = 9
            #
            # Example with 8-token prompt:
            # - Step 0: process prompt [0-7], sample token A → generated_tokens=[A]
            # - Step 1: process token A (position 8), sample token B → generated_tokens=[A,B]
            # - Step 2: process token B (position 9), sample token C → generated_tokens=[A,B,C]
            #
            # Formula: position = len(prompt) + (step - 1) for step > 0
            current_position = (
                len(tokens) + (step - 1)
                if step > 0
                else None  # For prompt pass, use positions [0, 1, 2, ..., len-1]
            )

            result = await self._ring_forward_pass(
                request_id=request_id,
                input_data=input_tokens,
                shard=shard,
                initial_position=current_position,  # Pass actual token position!
            )

            if result is None:
                logger.warning(
                    "⚠️  Ring forward pass returned None, stopping generation"
                )
                break

            # Check if result is a pre-sampled token (from tail-node sampling)
            # or full logits that need sampling here on head
            if result.dtype in (np.int64, np.int32) and result.size == 1:
                # Pre-sampled token from tail node — skip sampling
                token_id = int(result.flat[0])
                next_token = np.array([token_id], dtype=np.int64)
                logger.debug(f"   🎯 Using pre-sampled token from tail node: {token_id}")
            else:
                # Full logits — sample here on head
                next_token = await self.inference_engine.sample(
                    result, temp=0.7
                )
                token_id = int(next_token[0])

            generated_tokens.append(token_id)

            step_time = time.time() - step_start

            # Track TTFT and step times
            if step == 0:
                stats_logger.log_first_token(request_id)
            stats_logger.log_generation_step(request_id, step_time * 1000)

            logger.info(f"   ✅ Token {step + 1} sampled: {token_id}")
            logger.debug(f"   ⏱️  Step time: {step_time * 1000:.1f}ms")

            # Show decoded text every 10 tokens
            if (step + 1) % 10 == 0 or step == 0:
                try:
                    decoded_so_far = await self.inference_engine.decode(
                        shard, np.array(generated_tokens)
                    )
                    logger.debug(
                        f"   📝 Text so far: {decoded_so_far[:100]}{'...' if len(decoded_so_far) > 100 else ''}"
                    )
                except Exception as e:
                    logger.debug(f"Could not decode partial output: {e}")

            # Check for EOS - support multiple EOS tokens
            eos_token_id = self.inference_engine.tokenizer.eos_token_id

            # Some models have multiple EOS tokens (e.g., Qwen has both eos_token_id and special tokens)
            eos_tokens = {eos_token_id}
            if hasattr(self.inference_engine.tokenizer, "eos_token_ids"):
                # Handle both list and single int cases
                additional_eos = self.inference_engine.tokenizer.eos_token_ids
                if isinstance(additional_eos, (list, tuple)):
                    eos_tokens.update(additional_eos)
                else:
                    eos_tokens.add(additional_eos)

            if token_id in eos_tokens:
                logger.debug(
                    f"   🛑 EOS token ({token_id}) detected, stopping generation"
                )
                break

            # Prepare for next iteration
            input_tokens = next_token.reshape(1, 1)

        # Mark inference end
        stats_logger.log_inference_end(request_id)

        total_time = time.time() - start_time

        # Decode the generated tokens to text
        try:
            decoded_text = await self.inference_engine.decode(
                shard, np.array(generated_tokens)
            )
        except Exception as e:
            logger.error(f"Error decoding tokens: {e}")
            decoded_text = f"[Error decoding: {e}]"

        logger.debug("=" * 80)
        logger.info("✨ GENERATION COMPLETE")
        logger.info(f"   Total tokens: {len(generated_tokens)}")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.debug(
            f"   Avg token latency: {total_time / max(len(generated_tokens), 1) * 1000:.1f}ms/token"
        )
        logger.debug("=" * 80)
        logger.debug("")
        logger.info("📄 GENERATED TEXT:")
        logger.debug(f"{'─' * 80}")
        logger.debug(f"{decoded_text}")
        logger.debug(f"{'─' * 80}")
        logger.debug("")

        return generated_tokens

    async def _ring_forward_pass(
        self,
        request_id: str,
        input_data: np.ndarray,
        shard: Shard,
        initial_position: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Execute one forward pass through the ring.

        This implements prima.cpp's llama_decode_internal() ring logic:
        1. Process local layers
        2. Send to next node
        3. Receive from previous node (if not first cycle)
        4. Repeat until all layers processed

        Args:
            request_id: Unique request ID
            input_data: Input token IDs or hidden states
            shard: Model shard specification
            initial_position: Starting position for this forward pass (for autoregressive generation)

        Returns:
            Final logits (head node only)
        """
        total_cycles = self.calculate_cycles_needed(shard.n_layers)

        logger.debug("")
        logger.debug("🔁 Ring Forward Pass")
        logger.debug(f"   Input shape: {input_data.shape}")
        logger.debug(f"   Total cycles needed: {total_cycles}")
        logger.debug(f"   Total layers: {shard.n_layers}")

        # Initialize position_ids for the first node
        # This is critical - like prima.cpp's inp_pos
        batch_size = input_data.shape[0] if input_data.ndim >= 2 else 1
        seq_len = input_data.shape[1] if input_data.ndim >= 2 else input_data.shape[0]

        # CRITICAL FIX: Use initial_position if provided (for autoregressive generation)
        # Otherwise create position IDs from 0
        if initial_position is not None:
            # For autoregressive generation: single token at specific position
            # Example: if initial_position=256, position_ids = [256]
            position_ids = np.array([[initial_position]], dtype=np.int64)
            logger.debug(f"   Using provided initial_position: {initial_position}")
        else:
            # For initial prompt: sequence of positions [0, 1, 2, ..., seq_len-1]
            position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
            position_ids = np.broadcast_to(position_ids, (batch_size, seq_len))
            logger.debug(f"   Created position_ids for prompt: [0..{seq_len - 1}]")

        # Create attention mask: all ones (attend to all tokens)
        attention_mask = np.ones((batch_size, seq_len), dtype=np.bool_)

        logger.debug(f"   Initialized position_ids shape: {position_ids.shape}")
        logger.debug(f"   Position_ids content: {position_ids}")
        logger.debug(f"   Initialized attention_mask shape: {attention_mask.shape}")

        # Initialize state
        state = InferenceState(
            request_id=request_id,
            current_cycle=0,
            total_cycles=total_cycles,
            current_layer=0,
            hidden_states=input_data,
            metadata={"step": 0},
            position_ids=position_ids,  # Store position_ids
            attention_mask=attention_mask,  # Store attention_mask
            seq_len=seq_len,
            generation_step=0,  # First step (prompt processing)
            last_processed_step=-1,  # Not processed yet
        )

        self.active_requests[request_id] = state

        # Head node starts the ring
        logger.debug("   🎯 Initiating ring from HEAD node...")
        result = await self._process_and_forward(request_id, state, shard)

        # SPECIAL CASE: Single node mode - result is returned directly
        if self.ring_position and self.ring_position.world_size == 1:
            logger.debug(
                "   ✅ Single node mode: got result directly, no waiting needed"
            )
            self.active_requests.pop(request_id, None)
            return result

        # Wait for completion (result comes back from ring)
        timeout = 60.0  # seconds - increased for large message sync
        start_time = time.time()

        # Use event-based signaling for zero-latency wakeup
        event = self.completion_events.setdefault(request_id, asyncio.Event())

        logger.debug(f"   ⏳ Waiting for ring completion (timeout={timeout}s)...")
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            elapsed = time.time() - start_time
            logger.debug(f"   ✅ All layers processed in {elapsed * 1000:.1f}ms")
        except asyncio.TimeoutError:
            logger.error(
                f"   ⚠️  Timeout waiting for ring completion! State: layer={state.current_layer}/{shard.n_layers}"
            )
        finally:
            self.completion_events.pop(request_id, None)

        # Get final result from state (stored by handle_incoming_tensor)
        final_result = state.final_result if hasattr(state, "final_result") else result

        # Clean up
        self.active_requests.pop(request_id, None)

        return final_result

    async def _process_and_forward(
        self, request_id: str, state: InferenceState, shard: Shard
    ) -> Optional[np.ndarray]:
        """
        Process assigned layers and forward to next node.

        Based on prima.cpp's layer processing loop in llama_decode_internal.

        CRITICAL: Like prima.cpp, we need to:
        1. Check which layers belong to this node (this_layer_is_mine)
        2. Process ALL our layers in one batch
        3. Update state.current_layer to point to the NEXT unprocessed layer globally
        """
        # Process layers in my window
        current_data = state.hidden_states

        if current_data is None:
            logger.error("State has no hidden_states to process!")
            return None

        layers_to_process = []

        # Find all layers in our window that haven't been processed yet
        # This matches prima.cpp's approach: iterate through ALL model layers,
        # but only process the ones assigned to this node
        for layer_id in range(state.current_layer, shard.n_layers):
            if self.this_layer_is_mine(layer_id):
                layers_to_process.append(layer_id)
            elif layers_to_process:
                # We've found layers that aren't ours after processing some of ours
                # This means we've finished our contiguous block
                # (This handles the case where layer windows are not perfectly aligned)
                break

        if layers_to_process:
            logger.debug("")
            logger.debug(
                f"⚙️  Processing on Rank {self.ring_position.rank if self.ring_position else '?'}"
            )
            logger.debug(
                f"   Global layer IDs: {layers_to_process[0]} → {layers_to_process[-1]} ({len(layers_to_process)} layers)"
            )
            logger.debug(
                f"   My layer window: {self.layer_window.layer_start} → {self.layer_window.layer_end}"
            )
            logger.debug(
                f"   Current global progress: {state.current_layer}/{shard.n_layers}"
            )
            logger.debug(f"   Input shape: {current_data.shape}")

            compute_start = time.time()

            # NOTE: Do NOT create a new shard here!
            # The inference engine already has the correct shard loaded for this node
            # from initialization. We just pass the base shard spec for reference.
            # The actual layer filtering happens in the loaded model.

            # CRITICAL FIX: Determine if we should apply LM head
            # LM head should ONLY be applied by the LAST node in the ring that completes all layers
            #
            # Logic for multi-node ring (scalable to N nodes):
            # 1. Check if processing our layers will complete ALL model layers
            # 2. If yes, check if we're the last node in the ring (highest rank with layers)
            # 3. Only the last node with the highest layer range applies LM head
            #
            # Example with 3 nodes (28 layers):
            #   Node 0 (rank 0): layers 0-9   → will_be_final=False (9+1=10 < 28)
            #   Node 1 (rank 1): layers 10-18 → will_be_final=False (18+1=19 < 28)
            #   Node 2 (rank 2): layers 19-27 → will_be_final=True (27+1=28 >= 28) ✓ Apply LM head

            will_complete_all_layers = (layers_to_process[-1] + 1) >= shard.n_layers

            # Additionally check: are we the last node in the ring?
            # This is important because if a node's layer window doesn't perfectly align,
            # we need to ensure only the actual last processor applies the head
            is_last_node_in_ring = (
                self.ring_position
                and self.layer_window
                and self.layer_window.layer_end == shard.n_layers - 1
            )

            # Apply LM head ONLY if we're completing all layers AND we're the designated last node
            apply_lm_head = will_complete_all_layers and is_last_node_in_ring

            logger.debug("   🎯 LM head decision:")
            logger.debug(
                f"      - Will complete all layers: {will_complete_all_layers} (processing up to layer {layers_to_process[-1]})"
            )
            logger.debug(
                f"      - Is last node in ring: {is_last_node_in_ring} (layer_end={self.layer_window.layer_end if self.layer_window else '?'}, total={shard.n_layers})"
            )
            logger.debug(f"      - Apply LM head: {apply_lm_head}")

            # Run inference on assigned layers using the already-loaded sharded model
            output_data, new_state = await self.inference_engine.infer_tensor(
                request_id=request_id,
                shard=self._my_shard(shard),  # Node-local shard matching loaded layers
                input_data=current_data,
                inference_state=state.metadata,
                position_ids=state.position_ids,  # Pass position_ids to inference
                attention_mask=state.attention_mask,  # Pass attention_mask to inference
                is_final=apply_lm_head,  # Only apply LM head if we're the last node completing all layers
            )

            compute_time = time.time() - compute_start

            current_data = output_data
            state.hidden_states = current_data

            # Update state.current_layer to point to the next unprocessed layer
            # This is CRITICAL: we set it to one past the last layer we processed
            # Prima.cpp does this implicitly in its layer loop
            state.current_layer = layers_to_process[-1] + 1

            logger.debug(f"   Output shape: {output_data.shape}")
            logger.debug(f"   ⏱️  Compute time: {compute_time * 1000:.1f}ms")
            logger.debug(
                f"   Updated global progress: {state.current_layer}/{shard.n_layers}"
            )
            logger.debug(f"   Layers remaining: {shard.n_layers - state.current_layer}")

            # Log output type for debugging
            if output_data.shape[-1] == shard.n_layers:  # Assuming vocab size check
                logger.debug(
                    f"   📊 Output type: LOGITS (vocab_size={output_data.shape[-1]})"
                )
            else:
                logger.debug(
                    f"   📊 Output type: HIDDEN STATES (hidden_size={output_data.shape[-1]})"
                )
        else:
            logger.debug(
                f"   No layers to process in current chunk (current_layer={state.current_layer}, my_window={self.layer_window.layer_start}-{self.layer_window.layer_end})"
            )
            logger.warning(
                "   ⚠️  WARNING: Received tensor but no layers to process! This may indicate a state synchronization issue."
            )

        # Check if this is the last layer
        is_final_layer = state.current_layer >= shard.n_layers

        if is_final_layer:
            logger.debug("   🏁 Final layer reached!")

            # Determine what type of data we're sending
            data_type = "LOGITS" if current_data.shape[-1] > 10000 else "HIDDEN STATES"
            logger.debug(f"   📊 Data type: {data_type} (shape={current_data.shape})")

            # SPECIAL CASE: Single node - return logits directly
            if self.ring_position and self.ring_position.world_size == 1:
                logger.debug(
                    "   ✅ Single node mode: Returning logits directly for sampling"
                )
                return current_data

            # Return logits (head node only)
            if self.ring_position and self.ring_position.is_head:
                logger.debug("   ✅ HEAD node: Returning logits for sampling")
                return current_data
            else:
                # OPTIMIZATION: Sample token on tail node instead of sending full logits
                # This reduces transfer from ~0.58MB (logits) to ~32 bytes (token ID)
                # per autoregressive step — a ~18,000x reduction in data transfer
                logger.debug("   🎯 TAIL-NODE SAMPLING: Sampling token locally instead of sending logits")
                sample_start = time.time()
                sampled_token = await self.inference_engine.sample(
                    current_data, temp=0.7
                )
                token_id = int(sampled_token[0])
                sample_time = time.time() - sample_start
                logger.debug(f"   ✅ Sampled token {token_id} in {sample_time*1000:.1f}ms")

                # Send just the token ID back to head (tiny payload)
                await self._send_sampled_token_to_head(
                    request_id=request_id,
                    token_id=token_id,
                )
                return None
        else:
            # Forward to next node in ring
            if not self.ring_position:
                return None

            # SPECIAL CASE: Single node - don't send to network, just continue processing
            if self.ring_position.world_size == 1:
                logger.debug("   ↻ Single node mode: continuing to next layers locally")
                # Continue processing remaining layers
                return await self._process_and_forward(request_id, state, shard)

            next_rank = (self.ring_position.rank + 1) % self.ring_position.world_size
            logger.debug(
                f"   📤 Forwarding to Rank {next_rank} ({self.ring_position.next_node_id[:16]}...)"
            )

            await self._send_to_node(
                target_node_id=self.ring_position.next_node_id,
                data=current_data,
                request_id=request_id,
                is_final=False,
                position_ids=state.position_ids,
                attention_mask=state.attention_mask,
            )
            return None

    async def handle_incoming_tensor(
        self,
        sender_id: str,
        request_id: str,
        tensor_data: np.ndarray,
        shard: Shard,
        is_final: bool = False,
        position_ids: Optional[np.ndarray] = None,
        attention_mask: Optional[np.ndarray] = None,
    ):
        """
        Handle incoming tensor from previous node in ring.

        This is called when receiving a "tensor_forward" message.
        Based on prima.cpp's llama_recv_tensors().

        CRITICAL: Prima.cpp receives both hidden states AND position_ids.
        Without position_ids, RoPE embeddings fail and attention is broken.
        """
        try:
            logger.debug("")
            logger.debug("📥 RECEIVED TENSOR IN RING COORDINATOR")
            logger.debug(f"   From: {sender_id[:16]}...")
            logger.debug(f"   Request: {request_id}")
            logger.debug(f"   Shape: {tensor_data.shape}")
            logger.debug(f"   Is final: {is_final}")
            logger.debug(
                f"   My rank: {self.ring_position.rank if self.ring_position else '?'}"
            )
            logger.debug(f"   Has position_ids: {position_ids is not None}")
            logger.debug(f"   Has attention_mask: {attention_mask is not None}")

            # Restore or create state
            if request_id in self.active_requests:
                state = self.active_requests[request_id]
                logger.debug(
                    f"   Restored existing state (layer {state.current_layer}, step {state.generation_step})"
                )

                # 🐛 DEBUG: Check if we have cache for this request
                logger.debug("")
                logger.debug("🔍 WORKER NODE CACHE CHECK")
                if request_id in self.inference_engine.caches:
                    cache = self.inference_engine.caches[request_id]
                    if hasattr(cache, "get_seq_length"):
                        # Modern Cache object (DynamicCache in transformers 5.2.0+)
                        from transformers_inference import _get_cache_seq_length
                        num_layers = len(cache) if hasattr(cache, "__len__") else 0
                        seq_len = _get_cache_seq_length(cache)
                        logger.debug(
                            f"   ✅ Cache exists ({type(cache).__name__}): {num_layers} layers, seq_len={seq_len}"
                        )
                    elif hasattr(cache, "__len__"):
                        logger.debug(f"   ✅ Cache exists (tuple): {len(cache)} layers")
                    else:
                        logger.debug(f"   ✅ Cache exists: {type(cache).__name__}")
                else:
                    logger.warning("   ⚠️  NO CACHE found on worker node!")
                    logger.warning(
                        "   This is UNEXPECTED for step > 0 in autoregressive generation!"
                    )
                    logger.warning(
                        "   Worker nodes should maintain cache from previous steps!"
                    )
                logger.debug("")
            else:
                # For new state, start from the beginning of our layer window
                # This ensures we don't try to process layers that were already handled by previous nodes
                start_layer = self.layer_window.layer_start if self.layer_window else 0

                logger.debug("   Creating NEW state for request (first time seeing it)")

                # 🐛 DEBUG: Check if we should have cache
                logger.debug("")
                logger.debug("🔍 WORKER NODE - NEW REQUEST")
                if request_id in self.inference_engine.caches:
                    logger.warning(
                        "   ⚠️  UNEXPECTED: Cache exists but no active_request state!"
                    )
                    logger.warning("   This might indicate state management issue")
                else:
                    logger.debug("   ✅ No cache (expected for first time)")
                logger.debug("")

                state = InferenceState(
                    request_id=request_id,
                    current_cycle=0,
                    total_cycles=1,
                    current_layer=start_layer,  # Start from our window
                    metadata={},
                    position_ids=position_ids,  # Store position_ids from sender
                    attention_mask=attention_mask,  # Store attention_mask from sender
                    seq_len=tensor_data.shape[1]
                    if tensor_data.ndim >= 2
                    else tensor_data.shape[0],
                    generation_step=0,
                    last_processed_step=-1,
                )
                self.active_requests[request_id] = state
                logger.debug(f"   Created new state starting at layer {start_layer}")

            state.hidden_states = tensor_data
            # Update position metadata if provided (allows updates during generation)
            if position_ids is not None:
                state.position_ids = position_ids
            if attention_mask is not None:
                state.attention_mask = attention_mask

            # CRITICAL FIX: Detect if this is a new generation step
            # In autoregressive generation, each new token starts a fresh pass through the ring
            # We detect this by checking if:
            # 1. current_layer >= total layers (previous step completed all layers)
            # 2. We're receiving new data (tensor shape indicates new input)
            #
            # When detected, RESET current_layer to our window start so we process our layers again
            if state.current_layer >= shard.n_layers:
                # Previous generation step completed all layers
                # This is a NEW generation step - reset to process our layers again
                start_layer = self.layer_window.layer_start if self.layer_window else 0
                logger.debug(
                    f"   🔄 NEW GENERATION STEP DETECTED (current_layer={state.current_layer} >= {shard.n_layers})"
                )
                logger.debug(
                    f"   🔄 Resetting current_layer: {state.current_layer} → {start_layer}"
                )
                state.current_layer = start_layer
                state.generation_step += 1
                logger.debug(f"   🔄 Generation step: {state.generation_step}")
            elif (
                self.layer_window
                and state.current_layer < self.layer_window.layer_start
            ):
                # Current layer is before our window - advance to our window start
                # This handles mid-stream joins or irregular layer distributions
                logger.debug(
                    f"   ⚠️  current_layer ({state.current_layer}) < our window start ({self.layer_window.layer_start})"
                )
                logger.debug(
                    f"   ↪️  Advancing to window start: {self.layer_window.layer_start}"
                )
                state.current_layer = self.layer_window.layer_start

            if is_final and self.ring_position and self.ring_position.is_head:
                # Final result received at head
                # CRITICAL: Store result and signal completion to waiting loop
                state.final_result = tensor_data  # Store the logits or sampled token
                state.current_layer = shard.n_layers  # Mark all layers complete
                logger.debug("   ✅ Final result received at HEAD, stored in state")
                # Signal instant wakeup (no more 100ms polling delay)
                event = self.completion_events.get(request_id)
                if event:
                    event.set()
                return

            # Process and forward
            logger.debug("   → Processing and forwarding...")
            await self._process_and_forward(request_id, state, shard)

        except Exception as e:
            logger.error(f"Error in handle_incoming_tensor: {e}", exc_info=True)

    async def _send_sampled_token_to_head(
        self,
        request_id: str,
        token_id: int,
    ):
        """
        Send a sampled token ID back to the head node.

        This is the optimized return path: instead of sending full logits
        (0.58MB per step), we send just the token ID (32 bytes).
        The head node receives this as a 1-element int64 numpy array
        with is_final=True, and uses it directly for the generation loop.
        """
        head_node_id = self._find_head_node_id()
        # Encode token as a tiny numpy array so the existing tensor pathway works
        token_data = np.array([[token_id]], dtype=np.int64)

        send_start = time.time()
        logger.debug(f"   📤 Sending sampled token {token_id} to HEAD ({head_node_id[:16]}...)")

        my_node_id = str(await self.network.iroh_node.net().node_id())
        metadata = {
            "sender_id": my_node_id,
            "request_id": request_id,
            "tensor_shape": list(token_data.shape),
            "tensor_dtype": str(token_data.dtype),
            "is_final": True,
            "is_sampled_token": True,  # Signal that this is a pre-sampled token
            "token_id": token_id,
            "position_ids": None,
            "attention_mask": None,
        }

        await self.network.send_tensor_direct(
            target_node_id=head_node_id,
            tensor_data=token_data,
            request_id=request_id,
            metadata=metadata,
        )

        send_time = time.time() - send_start
        logger.debug(
            f"   ✅ Sampled token sent in {send_time * 1000:.1f}ms "
            f"(8 bytes vs ~0.58MB logits)"
        )

    async def _send_to_node(
        self,
        target_node_id: str,
        data: np.ndarray,
        request_id: str,
        is_final: bool,
        position_ids: Optional[np.ndarray] = None,
        attention_mask: Optional[np.ndarray] = None,
    ):
        """
        Send tensor to another node.

        Uses direct QUIC transport (lattica-style) for high performance:
        - Direct peer-to-peer QUIC stream (no Doc CRDT sync overhead)
        - Binary framed protocol (no JSON/base64 encoding for tensor data)
        - 4MB chunked transfer for large tensors
        - Connection pooling with auto-reconnect

        Falls back to Doc-based blob transfer if direct transport unavailable.
        """
        send_start = time.time()

        tensor_bytes = data.tobytes()
        size_mb = len(tensor_bytes) / 1024 / 1024
        logger.debug(f"   📤 Sending tensor: {size_mb:.2f} MB")
        logger.debug(f"   📤 Target: {target_node_id[:16]}...")
        logger.debug(f"   📤 Request ID: {request_id}")

        # Build metadata (like prima.cpp's sync_meta)
        my_node_id = str(await self.network.iroh_node.net().node_id())

        # OPTIMIZATION: Send position as compact integer instead of full arrays
        # Position_ids is typically [[N]] for autoregressive steps — just send N
        compact_position = None
        if position_ids is not None:
            if position_ids.size == 1:
                compact_position = int(position_ids.flat[0])
            else:
                compact_position = position_ids.tolist()

        metadata = {
            "sender_id": my_node_id,
            "request_id": request_id,
            "tensor_shape": list(data.shape),
            "tensor_dtype": str(data.dtype),
            "is_final": is_final,
            "position_ids": compact_position,
            "attention_mask": attention_mask.tolist()
            if attention_mask is not None
            else None,
        }

        # Use direct QUIC transport (lattica-style)
        await self.network.send_tensor_direct(
            target_node_id=target_node_id,
            tensor_data=data,
            request_id=request_id,
            metadata=metadata,
        )

        send_time = time.time() - send_start
        throughput = size_mb / send_time if send_time > 0 else 0
        logger.debug(
            f"   ✅ Tensor sent in {send_time * 1000:.1f}ms "
            f"({throughput:.1f} MB/s)"
        )

    def _find_head_node_id(self) -> str:
        """Find the head node (rank 0) ID."""
        # This should be stored during initialization
        # For now, assume we can query topology
        topology_nodes = self.network.topology.all_nodes()
        if not topology_nodes:
            return self.network.topology.active_node_id or ""

        # Head is the node with most memory (rank 0)
        sorted_nodes = sorted(
            topology_nodes, key=lambda x: (x[1].memory, x[0]), reverse=True
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
                    self.prefetch_queue.get(), timeout=1.0
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
