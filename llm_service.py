import asyncio
import logging
import os
import time
import uuid
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
from ring_pipeline import RingPipelineCoordinator
from shard import Shard
from stats_logger import get_stats_logger
from transformers_inference import TransformersShardedInferenceEngine

logger = logging.getLogger(__name__)


class LLMMessageType(Enum):
    QUERY = "query"
    RESPONSE = "response"
    SERVICE_INFO = "service_info"
    STATUS = "status"


class LLMService:
    """Manages LLM functionality in the P2P network with sharded inference support"""

    def __init__(
        self,
        network,
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        use_sharding=True,
        use_ring=False,
    ):
        """Initialize LLM service"""
        self.network = network  # This is the Node object
        self.model_name = model_name
        self.use_sharding = use_sharding
        self.use_ring = use_ring  # Enable ring pipeline mode

        # Sharded inference engine
        self.inference_engine: Optional[TransformersShardedInferenceEngine] = None

        # Model will be auto-detected based on config
        self.num_layers = None  # Will be set from model config
        self.current_shard: Optional[Shard] = None

        # Ring pipeline coordinator
        self.ring_coordinator: Optional[RingPipelineCoordinator] = None

        # Legacy single-node support
        self.model = None
        self.tokenizer = None

        self.is_loaded = False
        self.is_running = False
        self.is_loading = False
        self.last_query_time = 0
        self.query_count = 0
        self.pending_queries: Dict[str, Dict] = {}
        self.is_bitnet = "bitnet" in model_name.lower()

        # Distributed inference state
        self.max_generate_tokens = 50
        self.default_sample_temperature = 0.7

        # Stats logger for generation metrics
        self.stats_logger = get_stats_logger()

    async def start(
        self, model_name: Optional[str] = None, num_layers: Optional[int] = None
    ):
        """
        Start the LLM service by loading the model.

        Args:
            model_name: Optional model name to override default
            num_layers: Optional layer count (if None, auto-detect from model config)
        """
        if model_name:
            self.model_name = model_name
            self.is_bitnet = "bitnet" in model_name.lower()

        if self.is_loading or self.is_loaded:
            logger.warning("LLM is already loading or loaded")
            return False

        self.is_loading = True

        try:
            # Auto-detect number of layers from model config if not provided
            if num_layers is None:
                logger.info(f"Auto-detecting model configuration for {self.model_name}")
                from transformers import AutoConfig

                config = AutoConfig.from_pretrained(self.model_name)

                # Different models use different attribute names for layer count
                if hasattr(config, "num_hidden_layers"):
                    self.num_layers = config.num_hidden_layers
                elif hasattr(config, "n_layer"):
                    self.num_layers = config.n_layer
                elif hasattr(config, "num_layers"):
                    self.num_layers = config.num_layers
                else:
                    raise ValueError(
                        f"Cannot determine number of layers for {self.model_name}"
                    )

                logger.info(f"✓ Detected {self.num_layers} layers in {self.model_name}")
            else:
                self.num_layers = num_layers
                logger.info(f"Using specified layer count: {self.num_layers}")

            if self.use_sharding:
                # Initialize sharded inference engine with detected layer count
                await self._init_sharded_inference(self.num_layers)

                # Initialize ring pipeline if enabled
                if self.use_ring:
                    await self._init_ring_pipeline()
            else:
                # Legacy single-node loading
                if self.is_bitnet:
                    self._configure_bitnet_environment()

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model)

            self.is_loaded = True
            self.is_running = True
            self.is_loading = False

            await self._broadcast_service_info()

            logger.info(f"LLM service started with model: {self.model_name}")
            return True

        except Exception as e:
            self.is_loading = False
            logger.error(f"Failed to load LLM model: {e}", exc_info=True)
            return False

    def _configure_bitnet_environment(self):
        """Configure environment variables for BitNet models"""
        logger.info("Configuring environment for BitNet models")
        os.environ["PYTORCH_JIT"] = "0"
        os.environ["TORCH_INDUCTOR_DISABLE"] = "1"
        os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
        os.environ["TORCH_DYNAMO_DISABLE"] = "1"
        os.environ["PT_ENABLE_COMPILER"] = "0"
        os.environ["TORCH_COMPILE"] = "0"
        os.environ["TORCH_COMPILE_MODE"] = "reduce-overhead"
        logger.info("BitNet environment configured with compilation disabled")

    def _load_model(self):
        """Load the model and tokenizer (runs in a separate thread)"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if hasattr(torch, "_dynamo"):
                torch._dynamo.config.suppress_errors = True
                torch._dynamo.config.disable = True

            if hasattr(torch, "compile"):
                torch._original_compile = torch.compile
                torch.compile = lambda *args, **kwargs: args[0]

            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if self.is_bitnet:
                logger.info("Loading BitNet model with standard configuration")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    use_cache=True,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    async def stop(self):
        """Stop the LLM service"""
        self.is_running = False
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        logger.info("LLM service stopped")

    async def handle_llm_message(self, message: Dict[str, Any]):
        """Handle incoming LLM-related messages from the node."""
        payload = message.get("payload", {})
        llm_type = payload.get("llm_type")

        logger.debug(f"handle_llm_message called with type: {llm_type}")

        if llm_type == LLMMessageType.QUERY.value:
            logger.debug("Routing to _handle_query")
            await self._handle_query(message.get("sender_id"), payload)
        else:
            logger.debug(f"Ignoring llm_type: {llm_type}")

    async def _handle_query(self, sender_id: str, data: dict):
        """Process an LLM query from another peer"""
        my_node_id = str(await self.network.iroh_node.net().node_id())
        target_node_id = data.get("target_node_id")
        query_id = data.get("query_id")
        query = data.get("query")

        logger.info("")
        logger.info("📨 RECEIVED QUERY")
        logger.info(f"   Query ID: {query_id}")
        logger.info(f"   Query: {query[:50] if query else 'None'}...")
        logger.info(f"   From: {sender_id[:16]}...")
        logger.info(
            f"   Target: {target_node_id[:16] if target_node_id else 'broadcast'}..."
        )
        logger.info(f"   My ID: {my_node_id[:16]}...")

        if target_node_id and target_node_id != my_node_id:
            logger.info("   ↩️  Not for me, skipping")
            return

        if not self.is_running or not self.is_loaded:
            logger.warning("   ⚠️  Service not running or not loaded")
            error_response = {
                "llm_type": LLMMessageType.STATUS.value,
                "query_id": query_id,
                "status": "error",
                "message": "LLM service is not running",
            }
            await self._send_llm_data(error_response)
            return

        if not query:
            logger.warning("   ⚠️  No query content")
            return

        # In ring mode, only HEAD node processes queries
        # Worker nodes only participate by processing ring tensor messages
        if (
            self.use_ring
            and self.ring_coordinator
            and self.ring_coordinator.ring_position
        ):
            if not self.ring_coordinator.ring_position.is_head:
                logger.info(
                    f"   ↩️  Worker node (rank {self.ring_coordinator.ring_position.rank}) - skipping query, will participate in ring"
                )
                return

        logger.info("   ✅ Processing query...")

        status_update = {
            "llm_type": LLMMessageType.STATUS.value,
            "query_id": query_id,
            "status": "processing",
        }
        await self._send_llm_data(status_update)

        self.pending_queries[query_id] = {
            "sender_id": sender_id,
            "query": query,
            "timestamp": time.time(),
        }

        # Use ring pipeline, sharded, or single-node inference
        if self.use_ring and self.ring_coordinator:
            logger.info("   🔁 Routing to ring pipeline")
            asyncio.create_task(self._process_query_ring(query_id, query))
        elif self.use_sharding:
            logger.info("   📦 Routing to sharded inference")
            asyncio.create_task(self._process_query_sharded(query_id, query))
        else:
            logger.info("   🔧 Routing to standard inference")
            asyncio.create_task(self._process_query(query_id, query))

    async def _process_query(self, query_id: str, query: str):
        """Process a query using the LLM model (single-node mode)"""
        try:
            # Start stats logging
            self.stats_logger.start_generation(
                request_id=query_id,
                prompt=query,
                query_id=query_id,
                max_tokens=100,  # Default for single-node mode
                temperature=0.7,
            )
            
            self.stats_logger.log_inference_start(query_id)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self._run_inference, query)
            self.stats_logger.log_inference_end(query_id)
            
            # Note: We don't have token IDs in single-node mode easily, so we'll estimate
            # You could update _run_inference to return tokens if needed
            model_info = {
                "model_name": self.model_name,
                "mode": "single" if not self.is_bitnet else "bitnet",
                "device": "cuda" if self._has_cuda() else "cpu",
            }
            
            network_info = {
                "num_nodes": 1,
            }
            
            # End stats logging (with empty token list since we don't track in single mode)
            self.stats_logger.end_generation(
                request_id=query_id,
                response=response,
                generated_token_ids=[],  # Not tracked in single-node mode
                model_info=model_info,
                network_info=network_info,
            )
            
            result = {
                "llm_type": LLMMessageType.RESPONSE.value,
                "query_id": query_id,
                "query": query,
                "response": response,
                "model": self.model_name,
                "processing_time": time.time()
                - self.pending_queries[query_id]["timestamp"],
            }
            await self._send_llm_data(result)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            
            # Log error to stats
            self.stats_logger.end_generation(
                request_id=query_id,
                response="",
                generated_token_ids=[],
                model_info={"model_name": self.model_name, "mode": "single"},
                error=str(e),
            )
            
            error_response = {
                "llm_type": LLMMessageType.STATUS.value,
                "query_id": query_id,
                "status": "error",
                "message": f"Error processing query: {str(e)}",
            }
            await self._send_llm_data(error_response)
        finally:
            if query_id in self.pending_queries:
                del self.pending_queries[query_id]

    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _run_inference(self, query: str) -> str:
        """Run inference on the model (runs in a separate thread)"""
        try:
            import torch

            if self.is_bitnet:
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": query},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                with torch.no_grad():
                    output_sequences = self.model.generate(
                        **inputs,
                        max_new_tokens=200,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                response = self.tokenizer.decode(
                    output_sequences[0][inputs["input_ids"].shape[-1] :],
                    skip_special_tokens=True,
                )
                return response
            else:
                inputs = self.tokenizer(query, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                with torch.no_grad():
                    output_sequences = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                return self.tokenizer.decode(
                    output_sequences[0], skip_special_tokens=True
                )
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise

    async def _send_llm_data(self, llm_payload: dict):
        """Send LLM data by broadcasting it."""
        try:
            node_id_str = str(await self.network.iroh_node.net().node_id())
            message = {
                "type": "llm_message",
                "sender_id": node_id_str,
                "payload": llm_payload,
                "timestamp": time.time(),
            }
            return await self.network.broadcast_message(message)
        except Exception as e:
            logger.error(f"Error sending LLM data: {e}")
            return False

    async def _broadcast_service_info(self):
        """Broadcast information about this LLM service to the network"""
        node_id_str = str(await self.network.iroh_node.net().node_id())
        mode = (
            "sharded"
            if self.use_sharding
            else ("bitnet" if self.is_bitnet else "standard")
        )
        info_payload = {
            "model_name": self.model_name,
            "model_type": mode,
            "status": "running" if self.is_running else "stopped",
            "shard": self.current_shard.to_dict() if self.current_shard else None,
        }
        message = {
            "type": "llm_service_info",
            "sender_id": node_id_str,
            "payload": info_payload,
            "timestamp": time.time(),
        }
        return await self.network.broadcast_message(message)

    async def send_query(self, query: str, llm_node_id: Optional[str] = None) -> str:
        """Send a query by broadcasting it and return the query ID"""
        query_id = str(uuid.uuid4())
        node_id_str = str(await self.network.iroh_node.net().node_id())
        query_payload = {
            "llm_type": LLMMessageType.QUERY.value,
            "query_id": query_id,
            "query": query,
            "target_node_id": llm_node_id,
        }
        message = {
            "type": "llm_message",
            "sender_id": node_id_str,
            "payload": query_payload,
            "timestamp": time.time(),
        }

        # Broadcast to network (for other nodes)
        success = await self.network.broadcast_message(message)

        # IMPORTANT: In Iroh, nodes don't receive their own broadcasts!
        # So if this is a local query (no target or target is us), process it directly
        should_process_locally = (
            llm_node_id is None  # No specific target (broadcast to all)
            or llm_node_id == node_id_str  # Target is us
        )

        if should_process_locally and self.is_running and self.is_loaded:
            logger.info(
                "💡 Processing query locally (Iroh doesn't deliver own broadcasts)"
            )
            # Process locally
            await self._handle_query(node_id_str, query_payload)

        if success:
            return query_id
        return None

    # ===== Sharded Inference Methods =====

    async def _init_sharded_inference(self, num_layers: int):
        """Initialize sharded inference engine and determine this node's shard."""
        logger.info("Initializing sharded inference engine...")

        # Create inference engine
        self.inference_engine = TransformersShardedInferenceEngine()

        # Create base shard (full model) - this represents the complete model spec
        # needed by the ring coordinator to understand total layers
        self.base_shard = Shard(
            model_id=self.model_name,
            start_layer=0,
            end_layer=num_layers - 1,
            n_layers=num_layers,
        )

        # Update topology
        await self.network.update_topology()

        # Get this node's assigned shard (specific layer range for this node)
        self.current_shard = await self.network.get_current_shard(self.base_shard)

        if self.current_shard:
            logger.info(f"Node assigned shard: {self.current_shard}")
            # Load the shard
            await self.inference_engine.ensure_shard(self.current_shard)
        else:
            logger.warning("No shard assigned to this node")

    async def _process_query_sharded(self, query_id: str, query: str):
        """Process query using sharded inference."""
        try:
            if not self.current_shard:
                raise ValueError("No shard assigned to this node")

            # Check if this is the first shard (starts the inference)
            if self.current_shard.is_first_layer():
                await self._start_sharded_inference(query_id, query)
            else:
                # Non-first shards wait for tensor from previous shard
                logger.info(f"Node with shard {self.current_shard} waiting for input")

        except Exception as e:
            logger.error(f"Error in sharded query processing: {e}", exc_info=True)
            error_response = {
                "llm_type": LLMMessageType.STATUS.value,
                "query_id": query_id,
                "status": "error",
                "message": str(e),
            }
            await self._send_llm_data(error_response)

    async def _start_sharded_inference(self, request_id: str, prompt: str):
        """Start sharded inference from the first shard."""
        logger.info(f"Starting sharded inference for request {request_id}")

        try:
            # Start stats logging
            self.stats_logger.start_generation(
                request_id=request_id,
                prompt=prompt,
                max_tokens=self.max_generate_tokens,
                temperature=self.default_sample_temperature,
            )

            # Track request
            self.network.outstanding_requests[request_id] = "processing"
            self.network.buffered_token_output[request_id] = ([], False)

            # Encode prompt
            self.stats_logger.log_encoding_start(request_id)
            tokens = await self.inference_engine.encode(self.current_shard, prompt)
            input_tensor = tokens.reshape(1, -1)
            self.stats_logger.log_encoding_end(request_id, len(tokens))

            # Start inference timing
            self.stats_logger.log_inference_start(request_id)

            # Run first shard
            output_tensor, inference_state = await self.inference_engine.infer_tensor(
                request_id, self.current_shard, input_tensor, None
            )

            # Forward to next shard or sample if last
            if self.current_shard.is_last_layer():
                await self._handle_last_shard_output(
                    request_id, output_tensor, inference_state
                )
            else:
                await self._forward_to_next_shard(
                    request_id, output_tensor, inference_state
                )

        except Exception as e:
            logger.error(f"Error in sharded inference start: {e}", exc_info=True)
            self.network.outstanding_requests.pop(request_id, None)
            
            # Log error to stats
            self.stats_logger.end_generation(
                request_id=request_id,
                response="",
                generated_token_ids=[],
                model_info={"model_name": self.model_name, "mode": "sharded"},
                error=str(e),
            )

    async def _handle_last_shard_output(
        self, request_id: str, logits: np.ndarray, inference_state: Dict
    ):
        """Handle output from the last shard - sample and optionally continue generation."""
        # Sample next token
        step_start = time.time()
        token = await self.inference_engine.sample(
            logits, temp=self.default_sample_temperature
        )
        step_time = time.time() - step_start

        # Add to buffer
        if request_id not in self.network.buffered_token_output:
            self.network.buffered_token_output[request_id] = ([], False)

        tokens_list = self.network.buffered_token_output[request_id][0]
        tokens_list.append(int(token.item()))
        
        # Track TTFT and step times
        if len(tokens_list) == 1:
            self.stats_logger.log_first_token(request_id)
        self.stats_logger.log_generation_step(request_id, step_time * 1000)

        # Check if finished
        is_finished = (
            int(token.item()) == self.inference_engine.tokenizer.eos_token_id
            or len(tokens_list) >= self.max_generate_tokens
        )

        self.network.buffered_token_output[request_id] = (tokens_list, is_finished)

        # Decode and send intermediate result
        text = await self.inference_engine.decode(
            self.current_shard, np.array(tokens_list)
        )

        response = {
            "llm_type": LLMMessageType.RESPONSE.value,
            "query_id": request_id,
            "response": text,
            "is_finished": is_finished,
        }
        await self._send_llm_data(response)

        # Continue generation if not finished
        if not is_finished:
            # Feed token back through the network for next token
            next_input = token.reshape(1, -1)
            output_tensor, inference_state = await self.inference_engine.infer_tensor(
                request_id, self.current_shard, next_input, inference_state
            )
            await self._handle_last_shard_output(
                request_id, output_tensor, inference_state
            )
        else:
            # Mark inference end and log stats
            self.stats_logger.log_inference_end(request_id)
            
            # Collect model and network info
            model_info = {
                "model_name": self.model_name,
                "total_layers": self.num_layers,
                "layers_on_node": self.current_shard.end_layer - self.current_shard.start_layer + 1,
                "mode": "sharded",
                "device": "cpu",
            }
            
            network_info = {
                "num_nodes": len(self.network.topology.all_nodes()),
            }
            
            # End stats logging
            self.stats_logger.end_generation(
                request_id=request_id,
                response=text,
                generated_token_ids=tokens_list,
                model_info=model_info,
                network_info=network_info,
            )
            
            # Clean up
            self.network.outstanding_requests.pop(request_id, None)
            logger.info(f"Finished generation for request {request_id}")

    async def _forward_to_next_shard(
        self, request_id: str, tensor: np.ndarray, inference_state: Dict
    ):
        """Forward tensor to the next shard in the sequence."""
        # Get next shard from topology
        # This is simplified - in production you'd need proper routing
        logger.info(f"Forwarding tensor from shard {self.current_shard} to next shard")

        # For now, log that we would forward
        # In a full implementation, you'd:
        # 1. Determine next node from partitioning
        # 2. Send tensor via network.send_tensor()
        # 3. Handle response

        logger.warning(
            "Multi-node forwarding not yet fully implemented - this is a single-node test"
        )

    async def _init_ring_pipeline(self):
        """Initialize ring pipeline coordinator."""
        logger.info("Initializing ring pipeline mode...")

        self.ring_coordinator = RingPipelineCoordinator(
            inference_engine=self.inference_engine, network_node=self.network
        )

        # Broadcast topology update to let other nodes know we exist
        logger.info("Broadcasting topology update to discover peers...")
        await self.network.broadcast_topology_update()

        # Wait for topology to be populated
        logger.info("Waiting for peer discovery...")
        await asyncio.sleep(3.0)  # Give more time for topology updates

        topology_nodes = self.network.topology.all_nodes()
        logger.info(f"Found {len(topology_nodes)} nodes in topology")

        if not topology_nodes:
            logger.warning("No topology nodes found after waiting, trying again...")
            await asyncio.sleep(2.0)
            topology_nodes = self.network.topology.all_nodes()

        if topology_nodes:
            my_node_id = str(await self.network.iroh_node.net().node_id())
            await self.ring_coordinator.initialize_ring(
                topology_nodes=topology_nodes,
                my_node_id=my_node_id,
                model_total_layers=self.current_shard.n_layers,
            )
            logger.info(
                f"Ring pipeline ready: rank={self.ring_coordinator.ring_position.rank}, "
                f"layers=[{self.ring_coordinator.layer_window.layer_start}:"
                f"{self.ring_coordinator.layer_window.layer_end}]"
            )
        else:
            logger.warning(
                "Cannot initialize ring: no peers found, will run in single-node mode"
            )
            # Still initialize with just this node
            my_node_id = str(await self.network.iroh_node.net().node_id())
            topology_nodes = [(my_node_id, self.network.device_capabilities)]
            await self.ring_coordinator.initialize_ring(
                topology_nodes=topology_nodes,
                my_node_id=my_node_id,
                model_total_layers=self.current_shard.n_layers,
            )

    async def on_topology_update(self):
        """Handle topology updates - re-initialize ring if nodes join/leave."""
        if not self.use_ring or not self.ring_coordinator or not self.is_running:
            return

        logger.info("")
        logger.info("=" * 80)
        logger.info("🔄 TOPOLOGY UPDATE DETECTED - RE-INITIALIZING RING")
        logger.info("=" * 80)

        # Get updated topology
        topology_nodes = self.network.topology.all_nodes()
        logger.info(f"New topology size: {len(topology_nodes)} nodes")

        if not topology_nodes:
            logger.warning(
                "Topology update resulted in empty topology, keeping current ring"
            )
            return

        # Re-initialize ring with new topology
        my_node_id = str(await self.network.iroh_node.net().node_id())

        try:
            await self.ring_coordinator.initialize_ring(
                topology_nodes=topology_nodes,
                my_node_id=my_node_id,
                model_total_layers=self.base_shard.n_layers,  # Use base_shard for full model spec
            )
            logger.info("Ring re-initialized successfully")
        except Exception as e:
            logger.error(f"Failed to re-initialize ring: {e}", exc_info=True)

    async def _process_query_ring(self, query_id: str, query: str):
        """Process query using ring pipeline."""
        try:
            if not self.ring_coordinator or not self.ring_coordinator.ring_position:
                logger.error("Ring coordinator not initialized")
                return

            # Only head node starts generation
            if self.ring_coordinator.ring_position.is_head:
                logger.info(f"Head node starting ring inference for: {query[:50]}...")

                # Start stats logging
                self.stats_logger.start_generation(
                    request_id=query_id,
                    prompt=query,
                    query_id=query_id,
                    max_tokens=self.max_generate_tokens,
                    temperature=self.default_sample_temperature,
                )

                # Pass base_shard (full model spec) to ring coordinator
                # The ring coordinator will create node-specific shards from layer windows
                generated_tokens = await self.ring_coordinator.start_inference(
                    request_id=query_id,
                    prompt=query,
                    shard=self.base_shard,  # Full model spec, not node-specific shard
                    max_tokens=self.max_generate_tokens,
                )

                # Decode tokens to text using node's own shard
                response_text = await self.inference_engine.decode(
                    self.current_shard,  # Use current_shard for actual decoding
                    np.array(generated_tokens),
                )

                # Collect model info
                model_info = {
                    "model_name": self.model_name,
                    "total_layers": self.num_layers,
                    "layers_on_node": (
                        self.ring_coordinator.layer_window.layer_end
                        - self.ring_coordinator.layer_window.layer_start
                        + 1
                    ),
                    "mode": "ring",
                    "device": "cpu",  # Can be detected dynamically if needed
                }

                # Collect network info
                network_info = {
                    "num_nodes": self.ring_coordinator.ring_position.world_size,
                    "rank": self.ring_coordinator.ring_position.rank,
                    "world_size": self.ring_coordinator.ring_position.world_size,
                    "layer_window_start": self.ring_coordinator.layer_window.layer_start,
                    "layer_window_end": self.ring_coordinator.layer_window.layer_end,
                }

                # End stats logging
                self.stats_logger.end_generation(
                    request_id=query_id,
                    response=response_text,
                    generated_token_ids=generated_tokens,
                    model_info=model_info,
                    network_info=network_info,
                )

                result = {
                    "llm_type": LLMMessageType.RESPONSE.value,
                    "query_id": query_id,
                    "query": query,
                    "response": response_text,
                    "tokens": generated_tokens,
                    "model": self.model_name,
                    "mode": "ring_pipeline",
                    "rank": self.ring_coordinator.ring_position.rank,
                    "processing_time": time.time()
                    - self.pending_queries.get(query_id, {}).get(
                        "timestamp", time.time()
                    ),
                }

                await self._send_llm_data(result)

                if query_id in self.pending_queries:
                    del self.pending_queries[query_id]

            else:
                # Worker nodes participate in ring but don't initiate
                logger.info(
                    f"Worker node (rank {self.ring_coordinator.ring_position.rank}) "
                    f"waiting for ring messages"
                )

        except Exception as e:
            logger.error(f"Ring inference error: {e}", exc_info=True)
            
            # Log error to stats if head node
            if (
                self.ring_coordinator
                and self.ring_coordinator.ring_position
                and self.ring_coordinator.ring_position.is_head
            ):
                self.stats_logger.end_generation(
                    request_id=query_id,
                    response="",
                    generated_token_ids=[],
                    model_info={"model_name": self.model_name, "mode": "ring"},
                    error=str(e),
                )
            
            error_response = {
                "llm_type": LLMMessageType.STATUS.value,
                "query_id": query_id,
                "status": "error",
                "message": str(e),
            }
            await self._send_llm_data(error_response)
