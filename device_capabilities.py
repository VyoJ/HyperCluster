"""
Device capabilities tracking for distributed inference.
Adapted from exo's device_capabilities system.
"""

import logging
import platform
from dataclasses import dataclass

import psutil

logger = logging.getLogger(__name__)


@dataclass
class DeviceCapabilities:
    """
    Represents the compute and memory capabilities of a node.

    Attributes:
        model: Device model name (e.g., "MacBookPro", "NVIDIA RTX 4090")
        chip: Chip/processor name (e.g., "M2", "AMD Ryzen 9")
        memory: Available memory in GB
        flops: Floating point operations per second (FLOPS)
        role: Node role - 'compute_provider' (loads LLM layers) or 'inference_only' (queries only)
    """

    model: str
    chip: str
    memory: int  # in GB
    flops: float  # TFLOPS
    role: str = "compute_provider"  # 'compute_provider' or 'inference_only'

    def is_compute_provider(self) -> bool:
        """Check if this node is a compute provider (loads LLM layers)."""
        return self.role == "compute_provider"

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "model": self.model,
            "chip": self.chip,
            "memory": self.memory,
            "flops": self.flops,
            "role": self.role,
        }

    @staticmethod
    def from_dict(data: dict) -> "DeviceCapabilities":
        """Create from dictionary."""
        # Handle older data that may not have 'role'
        if "role" not in data:
            data["role"] = "compute_provider"
        return DeviceCapabilities(**data)

    def __str__(self):
        return f"DeviceCapabilities(model={self.model}, chip={self.chip}, memory={self.memory}GB, flops={self.flops:.2f}TFLOPS, role={self.role})"


# Unknown/default device capabilities
UNKNOWN_DEVICE_CAPABILITIES = DeviceCapabilities(
    model="Unknown", chip="Unknown", memory=4, flops=1.0
)


async def get_device_capabilities() -> DeviceCapabilities:
    """
    Detect and return the capabilities of the current device.

    This function attempts to:
    1. Detect GPU if available (CUDA/ROCm)
    2. Fall back to CPU capabilities
    3. Estimate FLOPS based on hardware
    """
    try:
        # Try to detect CUDA GPU
        try:
            import torch

            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )

                # Estimate FLOPS based on GPU model
                flops = _estimate_gpu_flops(device_name)

                return DeviceCapabilities(
                    model=device_name,
                    chip=device_name.split()[0] if device_name else "GPU",
                    memory=int(gpu_memory_gb),
                    flops=flops,
                )
        except ImportError:
            pass

        # Fall back to CPU capabilities
        cpu_info = platform.processor() or platform.machine()
        total_memory_gb = psutil.virtual_memory().total / (1024**3)

        # Estimate CPU FLOPS (very rough approximation)
        cpu_count = psutil.cpu_count(logical=False) or 1
        cpu_freq_ghz = psutil.cpu_freq().max / 1000 if psutil.cpu_freq() else 2.0

        # Rough estimate: cores * frequency * operations per cycle
        # Modern CPUs can do ~8-16 FP32 ops per cycle with AVX
        flops_per_core = cpu_freq_ghz * 8  # GFLOPS per core
        total_flops = (cpu_count * flops_per_core) / 1000  # Convert to TFLOPS

        return DeviceCapabilities(
            model=platform.system(),
            chip=cpu_info,
            memory=int(total_memory_gb),
            flops=max(0.1, total_flops),  # At least 0.1 TFLOPS
        )

    except Exception as e:
        logger.warning(f"Failed to detect device capabilities: {e}")
        return UNKNOWN_DEVICE_CAPABILITIES


def _estimate_gpu_flops(gpu_name: str) -> float:
    """
    Estimate GPU FLOPS based on model name.
    Returns TFLOPS (FP16 performance typically).
    """
    gpu_name_lower = gpu_name.lower()

    # NVIDIA GPUs (FP16 TFLOPS)
    if "4090" in gpu_name_lower:
        return 82.6
    elif "4080" in gpu_name_lower:
        return 48.7
    elif "4070" in gpu_name_lower:
        return 29.0
    elif "3090" in gpu_name_lower:
        return 35.6
    elif "3080" in gpu_name_lower:
        return 29.8
    elif "3070" in gpu_name_lower:
        return 20.4
    elif "a100" in gpu_name_lower:
        return 312.0
    elif "v100" in gpu_name_lower:
        return 125.0
    elif "t4" in gpu_name_lower:
        return 65.0

    # AMD GPUs (approximate FP16 TFLOPS)
    elif "7900 xtx" in gpu_name_lower:
        return 61.4
    elif "7900 xt" in gpu_name_lower:
        return 51.5
    elif "6900 xt" in gpu_name_lower:
        return 46.1

    # Apple Silicon (estimated)
    elif "m3 max" in gpu_name_lower or "m3max" in gpu_name_lower:
        return 14.2
    elif "m3 pro" in gpu_name_lower or "m3pro" in gpu_name_lower:
        return 5.0
    elif "m3" in gpu_name_lower:
        return 3.6
    elif "m2 ultra" in gpu_name_lower or "m2ultra" in gpu_name_lower:
        return 27.2
    elif "m2 max" in gpu_name_lower or "m2max" in gpu_name_lower:
        return 13.6
    elif "m2 pro" in gpu_name_lower or "m2pro" in gpu_name_lower:
        return 5.3
    elif "m2" in gpu_name_lower:
        return 3.6
    elif "m1 ultra" in gpu_name_lower or "m1ultra" in gpu_name_lower:
        return 21.0
    elif "m1 max" in gpu_name_lower or "m1max" in gpu_name_lower:
        return 10.4
    elif "m1 pro" in gpu_name_lower or "m1pro" in gpu_name_lower:
        return 5.3
    elif "m1" in gpu_name_lower:
        return 2.6

    # Default estimate for unknown GPUs
    else:
        logger.warning(f"Unknown GPU model: {gpu_name}, using default FLOPS estimate")
        return 10.0  # Conservative default
