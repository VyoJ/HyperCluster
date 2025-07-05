import asyncio
import logging
import json
import os
from enum import Enum
import time
from typing import Optional, Dict, Any
import uuid

logger = logging.getLogger(__name__)


class LLMMessageType(Enum):
    QUERY = "query"
    RESPONSE = "response"
    SERVICE_INFO = "service_info"
    STATUS = "status"


class LLMService:
    """Manages LLM functionality in the P2P network"""

    def __init__(self, network, model_name="HuggingFaceTB/SmolLM-135M"):
        """Initialize LLM service"""
        self.network = network  # This is the Node object
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.is_running = False
        self.is_loading = False
        self.last_query_time = 0
        self.query_count = 0
        self.pending_queries: Dict[str, Dict] = {}
        self.is_bitnet = "bitnet" in model_name.lower()

    async def start(self, model_name: Optional[str] = None):
        """Start the LLM service by loading the model"""
        if model_name:
            self.model_name = model_name
            self.is_bitnet = "bitnet" in model_name.lower()

        if self.is_loading or self.is_loaded:
            logger.warning("LLM is already loading or loaded")
            return False

        self.is_loading = True

        try:
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
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

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
                    self.model_name, device_map="cpu", torch_dtype=torch.float32,
                    use_cache=True, low_cpu_mem_usage=True, attn_implementation="eager"
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
        
        if llm_type == LLMMessageType.QUERY.value:
            await self._handle_query(message.get("sender_id"), payload)

    async def _handle_query(self, sender_id: str, data: dict):
        """Process an LLM query from another peer"""
        my_node_id = str(await self.network.iroh_node.net().node_id())
        target_node_id = data.get("target_node_id")
        query_id = data.get("query_id")

        if target_node_id and target_node_id != my_node_id:
            return

        if not self.is_running or not self.is_loaded:
            error_response = {"llm_type": LLMMessageType.STATUS.value, "query_id": query_id, "status": "error", "message": "LLM service is not running"}
            await self._send_llm_data(error_response)
            return

        query = data.get("query")
        if not query:
            return

        status_update = {"llm_type": LLMMessageType.STATUS.value, "query_id": query_id, "status": "processing"}
        await self._send_llm_data(status_update)

        self.pending_queries[query_id] = {"sender_id": sender_id, "query": query, "timestamp": time.time()}
        asyncio.create_task(self._process_query(query_id, query))

    async def _process_query(self, query_id: str, query: str):
        """Process a query using the LLM model"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self._run_inference, query)
            result = {
                "llm_type": LLMMessageType.RESPONSE.value, "query_id": query_id, "query": query,
                "response": response, "model": self.model_name,
                "processing_time": time.time() - self.pending_queries[query_id]["timestamp"],
            }
            await self._send_llm_data(result)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_response = {"llm_type": LLMMessageType.STATUS.value, "query_id": query_id, "status": "error", "message": f"Error processing query: {str(e)}"}
            await self._send_llm_data(error_response)
        finally:
            if query_id in self.pending_queries:
                del self.pending_queries[query_id]

    def _run_inference(self, query: str) -> str:
        """Run inference on the model (runs in a separate thread)"""
        try:
            import torch
            if self.is_bitnet:
                messages = [{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": query}]
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                with torch.no_grad():
                    output_sequences = self.model.generate(
                        **inputs, max_new_tokens=200, temperature=0.7, top_k=50,
                        top_p=0.9, do_sample=True, pad_token_id=self.tokenizer.eos_token_id
                    )
                response = self.tokenizer.decode(output_sequences[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                return response
            else:
                inputs = self.tokenizer(query, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                with torch.no_grad():
                    output_sequences = self.model.generate(
                        **inputs, max_new_tokens=100, temperature=0.7, top_k=50,
                        top_p=0.9, do_sample=True, pad_token_id=self.tokenizer.eos_token_id
                    )
                return self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise

    async def _send_llm_data(self, llm_payload: dict):
        """Send LLM data by broadcasting it."""
        try:
            node_id_str = str(await self.network.iroh_node.net().node_id())
            message = {"type": "llm_message", "sender_id": node_id_str, "payload": llm_payload, "timestamp": time.time()}
            return await self.network.broadcast_message(message)
        except Exception as e:
            logger.error(f"Error sending LLM data: {e}")
            return False

    async def _broadcast_service_info(self):
        """Broadcast information about this LLM service to the network"""
        node_id_str = str(await self.network.iroh_node.net().node_id())
        info_payload = {"model_name": self.model_name, "model_type": "bitnet" if self.is_bitnet else "standard", "status": "running" if self.is_running else "stopped"}
        message = {"type": "llm_service_info", "sender_id": node_id_str, "payload": info_payload, "timestamp": time.time()}
        return await self.network.broadcast_message(message)

    async def send_query(self, query: str, llm_node_id: Optional[str] = None) -> str:
        """Send a query by broadcasting it and return the query ID"""
        query_id = str(uuid.uuid4())
        node_id_str = str(await self.network.iroh_node.net().node_id())
        query_payload = {"llm_type": LLMMessageType.QUERY.value, "query_id": query_id, "query": query, "target_node_id": llm_node_id}
        message = {"type": "llm_message", "sender_id": node_id_str, "payload": query_payload, "timestamp": time.time()}
        success = await self.network.broadcast_message(message)
        if success:
            return query_id
        return None