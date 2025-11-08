"""
Generation Statistics Logger

Logs detailed statistics for each LLM generation including:
- Prompt and response
- Token counts
- Tokens per second (TPS)
- Time to first token (TTFT)
- Number of nodes
- Network latency
- Model information
- Generation mode (single/sharded/ring)
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NetworkMetrics:
    """Network-related metrics for distributed inference."""

    num_nodes: int = 1
    node_rank: Optional[int] = None
    world_size: Optional[int] = None
    avg_latency_ms: Optional[float] = None
    min_latency_ms: Optional[float] = None
    max_latency_ms: Optional[float] = None
    total_data_sent_bytes: int = 0
    total_data_received_bytes: int = 0
    ring_cycles: Optional[int] = None
    layer_window_start: Optional[int] = None
    layer_window_end: Optional[int] = None


@dataclass
class TokenMetrics:
    """Token-related metrics."""

    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    tokens_per_second: float
    time_to_first_token_ms: Optional[float] = None
    avg_time_per_token_ms: float = 0.0
    token_ids: List[int] = field(default_factory=list)


@dataclass
class TimingMetrics:
    """Detailed timing breakdown."""

    total_time_s: float
    encoding_time_ms: Optional[float] = None
    inference_time_ms: Optional[float] = None
    decoding_time_ms: Optional[float] = None
    sampling_time_ms: Optional[float] = None
    network_wait_time_ms: Optional[float] = None
    step_times_ms: List[float] = field(default_factory=list)


@dataclass
class ModelMetrics:
    """Model configuration and performance metrics."""

    model_name: str
    total_layers: Optional[int] = None
    layers_on_node: Optional[int] = None
    inference_mode: str = "single"  # single, sharded, ring
    dtype: Optional[str] = None
    device: str = "cpu"
    memory_used_mb: Optional[float] = None
    peak_memory_mb: Optional[float] = None


@dataclass
class GenerationStats:
    """Complete statistics for a generation request."""

    # Request metadata
    request_id: str
    timestamp: str
    query_id: Optional[str] = None

    # Content
    prompt: str = ""
    response: str = ""
    full_text: str = ""  # prompt + response

    # Sub-metrics
    tokens: Optional[TokenMetrics] = None
    timing: Optional[TimingMetrics] = None
    model: Optional[ModelMetrics] = None
    network: Optional[NetworkMetrics] = None

    # Additional metadata
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 50
    error: Optional[str] = None
    success: bool = True


class StatsLogger:
    """
    Logger for generation statistics.

    Creates individual JSON logfiles for each generation in the stats/ directory.
    """

    def __init__(self, log_dir: str = "stats"):
        """
        Initialize stats logger.

        Args:
            log_dir: Directory to store stats files (default: 'stats')
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        logger.info(f"📊 Stats logger initialized: {self.log_dir.absolute()}")

        # Track timing for current generation
        self._current_generation: Dict = {}

    def start_generation(
        self, request_id: str, prompt: str, query_id: Optional[str] = None, **kwargs
    ):
        """
        Start tracking a new generation.

        Args:
            request_id: Unique request identifier
            prompt: Input prompt
            query_id: Optional query ID for distributed queries
            **kwargs: Additional metadata (max_tokens, temperature, etc.)
        """
        self._current_generation[request_id] = {
            "start_time": time.time(),
            "prompt": prompt,
            "query_id": query_id,
            "step_times": [],
            "first_token_time": None,
            "encoding_start": None,
            "encoding_end": None,
            "inference_start": None,
            "inference_end": None,
            "network_wait_total": 0.0,
            "metadata": kwargs,
        }
        logger.debug(f"Started tracking generation: {request_id}")

    def log_encoding_start(self, request_id: str):
        """Mark encoding start time."""
        if request_id in self._current_generation:
            self._current_generation[request_id]["encoding_start"] = time.time()

    def log_encoding_end(self, request_id: str, token_count: int):
        """Mark encoding end time and store token count."""
        if request_id in self._current_generation:
            gen = self._current_generation[request_id]
            gen["encoding_end"] = time.time()
            gen["prompt_tokens"] = token_count

    def log_inference_start(self, request_id: str):
        """Mark inference start time."""
        if request_id in self._current_generation:
            self._current_generation[request_id]["inference_start"] = time.time()

    def log_inference_end(self, request_id: str):
        """Mark inference end time."""
        if request_id in self._current_generation:
            self._current_generation[request_id]["inference_end"] = time.time()

    def log_first_token(self, request_id: str):
        """Mark time when first token is generated."""
        if request_id in self._current_generation:
            gen = self._current_generation[request_id]
            if gen["first_token_time"] is None:
                gen["first_token_time"] = time.time()
                ttft = (gen["first_token_time"] - gen["start_time"]) * 1000
                logger.debug(f"TTFT for {request_id}: {ttft:.1f}ms")

    def log_generation_step(self, request_id: str, step_time_ms: float):
        """Log the time taken for a single generation step."""
        if request_id in self._current_generation:
            self._current_generation[request_id]["step_times"].append(step_time_ms)

    def log_network_wait(self, request_id: str, wait_time_ms: float):
        """Log time spent waiting for network."""
        if request_id in self._current_generation:
            self._current_generation[request_id]["network_wait_total"] += wait_time_ms

    def end_generation(
        self,
        request_id: str,
        response: str,
        generated_token_ids: List[int],
        model_info: Dict,
        network_info: Optional[Dict] = None,
        error: Optional[str] = None,
    ):
        """
        Complete generation tracking and write stats file.

        Args:
            request_id: Request identifier
            response: Generated text
            generated_token_ids: List of generated token IDs
            model_info: Dict with model metadata (name, layers, mode, etc.)
            network_info: Optional dict with network metrics
            error: Optional error message if generation failed
        """
        if request_id not in self._current_generation:
            logger.warning(f"No tracking data for request: {request_id}")
            return

        gen = self._current_generation[request_id]
        end_time = time.time()
        total_time_s = end_time - gen["start_time"]

        # Calculate token metrics
        prompt_tokens = gen.get("prompt_tokens", 0)
        generated_tokens = len(generated_token_ids)
        total_tokens = prompt_tokens + generated_tokens
        tps = generated_tokens / total_time_s if total_time_s > 0 else 0.0

        # Calculate TTFT
        ttft_ms = None
        if gen["first_token_time"]:
            ttft_ms = (gen["first_token_time"] - gen["start_time"]) * 1000

        # Calculate average time per token (excluding first token)
        avg_time_per_token_ms = 0.0
        if len(gen["step_times"]) > 1:
            avg_time_per_token_ms = sum(gen["step_times"][1:]) / len(
                gen["step_times"][1:]
            )

        # Timing breakdown
        encoding_time_ms = None
        if gen["encoding_start"] and gen["encoding_end"]:
            encoding_time_ms = (gen["encoding_end"] - gen["encoding_start"]) * 1000

        inference_time_ms = None
        if gen["inference_start"] and gen["inference_end"]:
            inference_time_ms = (gen["inference_end"] - gen["inference_start"]) * 1000

        # Create stats object
        stats = GenerationStats(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            query_id=gen.get("query_id"),
            prompt=gen["prompt"],
            response=response,
            full_text=gen["prompt"] + response,
            tokens=TokenMetrics(
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                total_tokens=total_tokens,
                tokens_per_second=tps,
                time_to_first_token_ms=ttft_ms,
                avg_time_per_token_ms=avg_time_per_token_ms,
                token_ids=generated_token_ids,
            ),
            timing=TimingMetrics(
                total_time_s=total_time_s,
                encoding_time_ms=encoding_time_ms,
                inference_time_ms=inference_time_ms,
                network_wait_time_ms=gen.get("network_wait_total"),
                step_times_ms=gen["step_times"],
            ),
            model=ModelMetrics(
                model_name=model_info.get("model_name", "unknown"),
                total_layers=model_info.get("total_layers"),
                layers_on_node=model_info.get("layers_on_node"),
                inference_mode=model_info.get("mode", "single"),
                dtype=model_info.get("dtype"),
                device=model_info.get("device", "cpu"),
                memory_used_mb=model_info.get("memory_used_mb"),
            ),
            network=self._create_network_metrics(network_info)
            if network_info
            else None,
            temperature=gen["metadata"].get("temperature", 0.7),
            top_p=gen["metadata"].get("top_p", 0.9),
            max_tokens=gen["metadata"].get("max_tokens", 50),
            error=error,
            success=error is None,
        )

        # Write to file
        self._write_stats_file(stats)

        # Clean up tracking data
        del self._current_generation[request_id]

        # Log summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 GENERATION STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Request ID: {request_id}")
        logger.info(f"Model: {stats.model.model_name}")
        logger.info(f"Mode: {stats.model.inference_mode}")
        if stats.network:
            logger.info(f"Nodes: {stats.network.num_nodes}")
            if stats.network.node_rank is not None:
                logger.info(f"Rank: {stats.network.node_rank}")
        logger.info("")
        logger.info("📝 Content:")
        logger.info(f"  Prompt: {stats.prompt[:60]}...")
        logger.info(f"  Response: {stats.response[:60]}...")
        logger.info("")
        logger.info("🔢 Tokens:")
        logger.info(f"  Prompt tokens: {stats.tokens.prompt_tokens}")
        logger.info(f"  Generated tokens: {stats.tokens.generated_tokens}")
        logger.info(f"  Total tokens: {stats.tokens.total_tokens}")
        logger.info("")
        logger.info("⚡ Performance:")
        logger.info(f"  Total time: {stats.timing.total_time_s:.3f}s")
        logger.info(f"  Tokens/sec: {stats.tokens.tokens_per_second:.2f}")
        if stats.tokens.time_to_first_token_ms:
            logger.info(f"  TTFT: {stats.tokens.time_to_first_token_ms:.1f}ms")
        logger.info(f"  Avg time/token: {stats.tokens.avg_time_per_token_ms:.1f}ms")
        if stats.network and stats.network.avg_latency_ms:
            logger.info(f"  Network latency: {stats.network.avg_latency_ms:.1f}ms")
        logger.info("=" * 80)

    def _create_network_metrics(self, network_info: Dict) -> NetworkMetrics:
        """Create NetworkMetrics from dict."""
        return NetworkMetrics(
            num_nodes=network_info.get("num_nodes", 1),
            node_rank=network_info.get("rank"),
            world_size=network_info.get("world_size"),
            avg_latency_ms=network_info.get("avg_latency_ms"),
            min_latency_ms=network_info.get("min_latency_ms"),
            max_latency_ms=network_info.get("max_latency_ms"),
            total_data_sent_bytes=network_info.get("total_data_sent_bytes", 0),
            total_data_received_bytes=network_info.get("total_data_received_bytes", 0),
            ring_cycles=network_info.get("ring_cycles"),
            layer_window_start=network_info.get("layer_window_start"),
            layer_window_end=network_info.get("layer_window_end"),
        )

    def _write_stats_file(self, stats: GenerationStats):
        """Write stats to JSON file."""
        try:
            # Create filename with timestamp and request ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gen_{timestamp}_{stats.request_id[:8]}.json"
            filepath = self.log_dir / filename

            # Convert to dict and write
            stats_dict = asdict(stats)

            with open(filepath, "w") as f:
                json.dump(stats_dict, f, indent=2)

            logger.info(f"📊 Stats saved: {filepath}")

        except Exception as e:
            logger.error(f"Failed to write stats file: {e}", exc_info=True)

    def get_generation_stats(self, request_id: str) -> Optional[Dict]:
        """Get current generation stats (for monitoring during generation)."""
        return self._current_generation.get(request_id)


# Global stats logger instance
_stats_logger: Optional[StatsLogger] = None


def get_stats_logger(log_dir: str = "stats") -> StatsLogger:
    """Get or create global stats logger instance."""
    global _stats_logger
    if _stats_logger is None:
        _stats_logger = StatsLogger(log_dir)
    return _stats_logger
