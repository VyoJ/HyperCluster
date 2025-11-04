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

        # Prefetch queue
        self.prefetch_queue = asyncio.Queue()
        self.prefetch_task: Optional[asyncio.Task] = None

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

        logger.info("=" * 80)
        logger.info("🚀 STARTING RING INFERENCE")
        logger.info("=" * 80)
        logger.info(f"Request ID: {request_id}")
        logger.info(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        logger.info(f"Max tokens: {max_tokens}")
        logger.info(f"Model layers: {shard.n_layers}")

        # Clear any existing cache for this request
        # This ensures we start with clean state for new prompts
        logger.info(f"Clearing any existing cache for request {request_id}")
        self.inference_engine.caches.pop(request_id, None)

        # Encode prompt
        start_time = time.time()
        tokens = await self.inference_engine.encode(shard, prompt)
        input_tokens = tokens.reshape(1, -1)
        encode_time = time.time() - start_time

        logger.info("")
        logger.info("📝 Encoding complete:")
        logger.info(f"   Input tokens: {len(tokens)}")
        logger.info(f"   Token shape: {input_tokens.shape}")
        logger.info(f"   Encode time: {encode_time * 1000:.1f}ms")
        logger.info("=" * 80)

        generated_tokens = []

        # Auto-regressive generation loop
        for step in range(max_tokens):
            logger.info("")
            logger.info(f"🔄 GENERATION STEP {step + 1}/{max_tokens}")
            logger.info(f"   Tokens generated so far: {len(generated_tokens)}")

            # 🐛 DEBUG: Check cache state before generation step
            logger.info("")
            logger.info("🔍 PRE-STEP CACHE CHECK")
            logger.info(f"   Request ID: {request_id}")
            if request_id in self.inference_engine.caches:
                cache = self.inference_engine.caches[request_id]
                if hasattr(cache, "key_cache"):
                    logger.info(f"   ✅ Cache exists: {len(cache.key_cache)} layers")
                    if len(cache.key_cache) > 0 and cache.key_cache[0] is not None:
                        logger.info(f"   Cache seq_len: {cache.key_cache[0].shape[2]}")
                else:
                    logger.info(f"   ✅ Cache exists (tuple): {len(cache)} layers")
            else:
                logger.info("   ❌ No cache found")
            logger.info("")

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

            logits = await self._ring_forward_pass(
                request_id=request_id,
                input_data=input_tokens,
                shard=shard,
                initial_position=current_position,  # Pass actual token position!
            )

            if logits is None:
                logger.warning(
                    "⚠️  Ring forward pass returned None, stopping generation"
                )
                break

            # Sample next token
            next_token = await self.inference_engine.sample(
                logits, temp=0.7
            )  # Use temperature sampling for better diversity
            token_id = int(next_token[0])
            generated_tokens.append(token_id)

            step_time = time.time() - step_start

            logger.info(f"   ✅ Token {step + 1} sampled: {token_id}")
            logger.info(f"   ⏱️  Step time: {step_time * 1000:.1f}ms")

            # Show decoded text every 10 tokens
            if (step + 1) % 10 == 0 or step == 0:
                try:
                    decoded_so_far = await self.inference_engine.decode(
                        shard, np.array(generated_tokens)
                    )
                    logger.info(
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
                logger.info(
                    f"   🛑 EOS token ({token_id}) detected, stopping generation"
                )
                break

            # Prepare for next iteration
            input_tokens = next_token.reshape(1, 1)

        total_time = time.time() - start_time

        # Decode the generated tokens to text
        try:
            decoded_text = await self.inference_engine.decode(
                shard, np.array(generated_tokens)
            )
        except Exception as e:
            logger.error(f"Error decoding tokens: {e}")
            decoded_text = f"[Error decoding: {e}]"

        logger.info("=" * 80)
        logger.info("✨ GENERATION COMPLETE")
        logger.info(f"   Total tokens: {len(generated_tokens)}")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(
            f"   Avg token latency: {total_time / max(len(generated_tokens), 1) * 1000:.1f}ms/token"
        )
        logger.info("=" * 80)
        logger.info("")
        logger.info("📄 GENERATED TEXT:")
        logger.info(f"{'─' * 80}")
        logger.info(f"{decoded_text}")
        logger.info(f"{'─' * 80}")
        logger.info("")

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

        logger.info("")
        logger.info("🔁 Ring Forward Pass")
        logger.info(f"   Input shape: {input_data.shape}")
        logger.info(f"   Total cycles needed: {total_cycles}")
        logger.info(f"   Total layers: {shard.n_layers}")

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
            logger.info(f"   Using provided initial_position: {initial_position}")
        else:
            # For initial prompt: sequence of positions [0, 1, 2, ..., seq_len-1]
            position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
            position_ids = np.broadcast_to(position_ids, (batch_size, seq_len))
            logger.info(f"   Created position_ids for prompt: [0..{seq_len - 1}]")

        # Create attention mask: all ones (attend to all tokens)
        attention_mask = np.ones((batch_size, seq_len), dtype=np.bool_)

        logger.info(f"   Initialized position_ids shape: {position_ids.shape}")
        logger.info(f"   Position_ids content: {position_ids}")
        logger.info(f"   Initialized attention_mask shape: {attention_mask.shape}")

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
        logger.info("   🎯 Initiating ring from HEAD node...")
        result = await self._process_and_forward(request_id, state, shard)

        # SPECIAL CASE: Single node mode - result is returned directly
        if self.ring_position and self.ring_position.world_size == 1:
            logger.info(
                "   ✅ Single node mode: got result directly, no waiting needed"
            )
            self.active_requests.pop(request_id, None)
            return result

        # Wait for completion (result comes back from ring)
        timeout = 60.0  # seconds - increased for large message sync
        start_time = time.time()

        logger.info(f"   ⏳ Waiting for ring completion (timeout={timeout}s)...")
        while time.time() - start_time < timeout:
            if state.current_layer >= shard.n_layers:
                # All layers processed
                elapsed = time.time() - start_time
                logger.info(f"   ✅ All layers processed in {elapsed * 1000:.1f}ms")
                break

            # Log progress every 5 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 5 == 0 and elapsed > 0:
                logger.info(
                    f"   ⏱️  Still waiting... {elapsed:.0f}s elapsed, current_layer={state.current_layer}/{shard.n_layers}"
                )

            await asyncio.sleep(0.5)
        else:
            logger.error(
                f"   ⚠️  Timeout waiting for ring completion! State: layer={state.current_layer}/{shard.n_layers}"
            )

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
            logger.info("")
            logger.info(
                f"⚙️  Processing on Rank {self.ring_position.rank if self.ring_position else '?'}"
            )
            logger.info(
                f"   Global layer IDs: {layers_to_process[0]} → {layers_to_process[-1]} ({len(layers_to_process)} layers)"
            )
            logger.info(
                f"   My layer window: {self.layer_window.layer_start} → {self.layer_window.layer_end}"
            )
            logger.info(
                f"   Current global progress: {state.current_layer}/{shard.n_layers}"
            )
            logger.info(f"   Input shape: {current_data.shape}")

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

            logger.info("   🎯 LM head decision:")
            logger.info(
                f"      - Will complete all layers: {will_complete_all_layers} (processing up to layer {layers_to_process[-1]})"
            )
            logger.info(
                f"      - Is last node in ring: {is_last_node_in_ring} (layer_end={self.layer_window.layer_end if self.layer_window else '?'}, total={shard.n_layers})"
            )
            logger.info(f"      - Apply LM head: {apply_lm_head}")

            # Run inference on assigned layers using the already-loaded sharded model
            output_data, new_state = await self.inference_engine.infer_tensor(
                request_id=request_id,
                shard=shard,  # Use base shard, inference engine has correct shard loaded
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

            logger.info(f"   Output shape: {output_data.shape}")
            logger.info(f"   ⏱️  Compute time: {compute_time * 1000:.1f}ms")
            logger.info(
                f"   Updated global progress: {state.current_layer}/{shard.n_layers}"
            )
            logger.info(f"   Layers remaining: {shard.n_layers - state.current_layer}")

            # Log output type for debugging
            if output_data.shape[-1] == shard.n_layers:  # Assuming vocab size check
                logger.info(
                    f"   📊 Output type: LOGITS (vocab_size={output_data.shape[-1]})"
                )
            else:
                logger.info(
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
            logger.info("   🏁 Final layer reached!")

            # Determine what type of data we're sending
            data_type = "LOGITS" if current_data.shape[-1] > 10000 else "HIDDEN STATES"
            logger.info(f"   📊 Data type: {data_type} (shape={current_data.shape})")

            # SPECIAL CASE: Single node - return logits directly
            if self.ring_position and self.ring_position.world_size == 1:
                logger.info(
                    "   ✅ Single node mode: Returning logits directly for sampling"
                )
                return current_data

            # Return logits (head node only)
            if self.ring_position and self.ring_position.is_head:
                logger.info("   ✅ HEAD node: Returning logits for sampling")
                return current_data
            else:
                # Send back to head
                logger.info(f"   📤 Worker node: Sending {data_type} back to HEAD")
                await self._send_to_node(
                    target_node_id=self._find_head_node_id(),
                    data=current_data,
                    request_id=request_id,
                    is_final=True,
                    position_ids=state.position_ids,
                    attention_mask=state.attention_mask,
                )
                return None
        else:
            # Forward to next node in ring
            if not self.ring_position:
                return None

            # SPECIAL CASE: Single node - don't send to network, just continue processing
            if self.ring_position.world_size == 1:
                logger.info("   ↻ Single node mode: continuing to next layers locally")
                # Continue processing remaining layers
                return await self._process_and_forward(request_id, state, shard)

            next_rank = (self.ring_position.rank + 1) % self.ring_position.world_size
            logger.info(
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
            logger.info("")
            logger.info("📥 RECEIVED TENSOR IN RING COORDINATOR")
            logger.info(f"   From: {sender_id[:16]}...")
            logger.info(f"   Request: {request_id}")
            logger.info(f"   Shape: {tensor_data.shape}")
            logger.info(f"   Is final: {is_final}")
            logger.info(
                f"   My rank: {self.ring_position.rank if self.ring_position else '?'}"
            )
            logger.info(f"   Has position_ids: {position_ids is not None}")
            logger.info(f"   Has attention_mask: {attention_mask is not None}")

            # Restore or create state
            if request_id in self.active_requests:
                state = self.active_requests[request_id]
                logger.info(
                    f"   Restored existing state (layer {state.current_layer}, step {state.generation_step})"
                )

                # 🐛 DEBUG: Check if we have cache for this request
                logger.info("")
                logger.info("🔍 WORKER NODE CACHE CHECK")
                if request_id in self.inference_engine.caches:
                    cache = self.inference_engine.caches[request_id]
                    if hasattr(cache, "key_cache"):
                        logger.info(
                            f"   ✅ Cache exists: {len(cache.key_cache)} layers"
                        )
                        if len(cache.key_cache) > 0 and cache.key_cache[0] is not None:
                            logger.info(
                                f"   Cache seq_len: {cache.key_cache[0].shape[2]}"
                            )
                    else:
                        logger.info(f"   ✅ Cache exists (tuple): {len(cache)} layers")
                else:
                    logger.warning("   ⚠️  NO CACHE found on worker node!")
                    logger.warning(
                        "   This is UNEXPECTED for step > 0 in autoregressive generation!"
                    )
                    logger.warning(
                        "   Worker nodes should maintain cache from previous steps!"
                    )
                logger.info("")
            else:
                # For new state, start from the beginning of our layer window
                # This ensures we don't try to process layers that were already handled by previous nodes
                start_layer = self.layer_window.layer_start if self.layer_window else 0

                logger.info("   Creating NEW state for request (first time seeing it)")

                # 🐛 DEBUG: Check if we should have cache
                logger.info("")
                logger.info("🔍 WORKER NODE - NEW REQUEST")
                if request_id in self.inference_engine.caches:
                    logger.warning(
                        "   ⚠️  UNEXPECTED: Cache exists but no active_request state!"
                    )
                    logger.warning("   This might indicate state management issue")
                else:
                    logger.info("   ✅ No cache (expected for first time)")
                logger.info("")

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
                logger.info(f"   Created new state starting at layer {start_layer}")

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
                logger.info(
                    f"   🔄 NEW GENERATION STEP DETECTED (current_layer={state.current_layer} >= {shard.n_layers})"
                )
                logger.info(
                    f"   🔄 Resetting current_layer: {state.current_layer} → {start_layer}"
                )
                state.current_layer = start_layer
                state.generation_step += 1
                logger.info(f"   🔄 Generation step: {state.generation_step}")
            elif (
                self.layer_window
                and state.current_layer < self.layer_window.layer_start
            ):
                # Current layer is before our window - advance to our window start
                # This handles mid-stream joins or irregular layer distributions
                logger.info(
                    f"   ⚠️  current_layer ({state.current_layer}) < our window start ({self.layer_window.layer_start})"
                )
                logger.info(
                    f"   ↪️  Advancing to window start: {self.layer_window.layer_start}"
                )
                state.current_layer = self.layer_window.layer_start

            if is_final and self.ring_position and self.ring_position.is_head:
                # Final result received at head
                # CRITICAL: Store result and signal completion to waiting loop
                state.final_result = tensor_data  # Store the logits
                state.current_layer = shard.n_layers  # Mark all layers complete
                logger.info("   ✅ Final result received at HEAD, stored in state")
                return

            # Process and forward
            logger.info("   → Processing and forwarding...")
            await self._process_and_forward(request_id, state, shard)

        except Exception as e:
            logger.error(f"Error in handle_incoming_tensor: {e}", exc_info=True)

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
        Send tensor to another node via Iroh blobs.
        Based on prima.cpp's llama_send_tensors().

        Prima.cpp sends:
        - sub_gf_out (hidden states)
        - inp_pos (position IDs) - CRITICAL for RoPE embeddings
        - attention_mask (implicitly via batch metadata)

        Uses Iroh's blob storage for large binary data (tensor),
        and sends only the blob hash through the document.
        """
        # Get document ID for communication
        if not self.network.documents:
            logger.error("No documents available for communication")
            return

        doc_id = next(iter(self.network.documents))

        send_start = time.time()

        # Serialize tensor to bytes
        tensor_bytes = data.tobytes()
        size_mb = len(tensor_bytes) / 1024 / 1024
        logger.info(f"   📤 Sending tensor: {size_mb:.2f} MB")
        logger.info(f"   📤 Target: {target_node_id[:16]}...")
        logger.info(f"   📤 Request ID: {request_id}")

        # Store tensor directly in document as a binary entry
        # This ensures it syncs to all peers automatically
        doc = self.network.documents[doc_id]
        author = await self.network.iroh_node.authors().default()

        # Create unique key for this tensor with timestamp and request ID
        # This helps the receiver identify the exact tensor
        timestamp_ms = int(time.time() * 1000)
        tensor_key = f"tensor-{request_id}-{timestamp_ms}".encode("utf-8")

        logger.info("   📝 Writing tensor to document as binary entry...")
        write_start = time.time()
        tensor_hash = await doc.set_bytes(author, tensor_key, tensor_bytes)
        write_time = time.time() - write_start
        logger.info(f"   ✅ Tensor written to document in {write_time * 1000:.1f}ms")
        logger.info(f"   📍 Tensor blob hash: {str(tensor_hash)[:16]}...")

        # Small delay to allow sync - give Iroh time to propagate the blob
        await asyncio.sleep(0.5)

        # Send metadata message with tensor key
        # IMPORTANT: Include position_ids and attention_mask like prima.cpp does
        message = {
            "type": "ring_tensor_forward",
            "sender_id": str(await self.network.iroh_node.net().node_id()),
            "target_node_id": target_node_id,
            "request_id": request_id,
            "payload": {
                "tensor_key": tensor_key.decode(
                    "utf-8"
                ),  # Key to fetch tensor from document
                "tensor_hash": str(tensor_hash),  # Blob hash for direct lookup
                "tensor_shape": list(data.shape),
                "tensor_dtype": str(data.dtype),
                "tensor_size": len(tensor_bytes),
                "is_final": is_final,
                # Critical metadata (like prima.cpp's inp_pos)
                "position_ids": position_ids.tolist()
                if position_ids is not None
                else None,
                "attention_mask": attention_mask.tolist()
                if attention_mask is not None
                else None,
            },
            "timestamp": time.time(),
        }

        success = await self.network.send_message(doc_id, message)

        send_time = time.time() - send_start
        if success:
            logger.info(f"   ✅ Tensor sent in {send_time * 1000:.1f}ms (total)")
        else:
            logger.error("   ❌ Failed to send message!")

        return success

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
