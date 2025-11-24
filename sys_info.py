import platform
import psutil
import subprocess
import sys
import json
from datetime import datetime


def get_size(bytes, suffix="B"):
    """Scale bytes to its proper format (KB, MB, GB, etc.)"""
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def get_cpu_info():
    """Get detailed CPU information"""
    try:
        import cpuinfo

        cpu_info = cpuinfo.get_cpu_info()
        cpu_details = {
            "brand": cpu_info.get("brand_raw", "Unknown"),
            "architecture": cpu_info.get("arch", platform.machine()),
            "bits": cpu_info.get("bits", "Unknown"),
            "cores_logical": psutil.cpu_count(logical=True),
            "cores_physical": psutil.cpu_count(logical=False),
            "frequency_current": (
                psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown"
            ),
            "frequency_min": psutil.cpu_freq().min if psutil.cpu_freq() else "Unknown",
            "frequency_max": psutil.cpu_freq().max if psutil.cpu_freq() else "Unknown",
            "l2_cache": get_size(cpu_info.get("l2_cache_size", 0)),
            "l3_cache": get_size(cpu_info.get("l3_cache_size", 0)),
        }
    except ImportError:
        # Fallback if py-cpuinfo not installed
        cpu_details = {
            "brand": platform.processor(),
            "architecture": platform.machine(),
            "cores_logical": psutil.cpu_count(logical=True),
            "cores_physical": psutil.cpu_count(logical=False),
            "frequency_current": (
                psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown"
            ),
            "frequency_min": psutil.cpu_freq().min if psutil.cpu_freq() else "Unknown",
            "frequency_max": psutil.cpu_freq().max if psutil.cpu_freq() else "Unknown",
        }

    return cpu_details


def get_memory_info():
    """Get RAM details"""
    svmem = psutil.virtual_memory()
    memory_details = {
        "total": get_size(svmem.total),
        "available": get_size(svmem.available),
        "used": get_size(svmem.used),
        "percentage": svmem.percent,
    }

    # Swap memory
    swap = psutil.swap_memory()
    memory_details["swap"] = {
        "total": get_size(swap.total),
        "free": get_size(swap.free),
        "used": get_size(swap.used),
        "percentage": swap.percent,
    }

    return memory_details


def get_nvidia_gpu_info():
    """Get NVIDIA GPU details using nvidia-smi"""
    try:
        nvidia_smi_output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            universal_newlines=True,
        )

        gpus = []
        for i, line in enumerate(nvidia_smi_output.strip().split("\n")):
            values = [val.strip() for val in line.split(",")]
            if len(values) >= 6:
                gpu = {
                    "index": i,
                    "name": values[0],
                    "memory_total_MB": values[1],
                    "memory_free_MB": values[2],
                    "memory_used_MB": values[3],
                    "temperature_C": values[4],
                    "utilization_percent": values[5],
                }

                # Try to get CUDA version
                try:
                    cuda_version_output = subprocess.check_output(
                        ["nvcc", "--version"], universal_newlines=True
                    )
                    for line in cuda_version_output.split("\n"):
                        if "release" in line.lower() and "V" in line:
                            gpu["cuda_version"] = (
                                line.split("V")[1].split(",")[0].strip()
                            )
                            break
                except (subprocess.SubprocessError, FileNotFoundError):
                    gpu["cuda_version"] = "Unknown (nvcc not found)"

                gpus.append(gpu)

        return gpus
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def get_amd_gpu_info():
    """Get AMD GPU details"""
    if sys.platform == "win32":
        try:
            import wmi

            w = wmi.WMI()
            gpus = []

            for i, gpu in enumerate(w.Win32_VideoController()):
                if "AMD" in gpu.Name or "Radeon" in gpu.Name:
                    gpu_info = {
                        "index": i,
                        "name": gpu.Name,
                        "driver_version": gpu.DriverVersion,
                        "adapter_ram_MB": (
                            int(int(gpu.AdapterRAM) / (1024 * 1024))
                            if gpu.AdapterRAM
                            else "Unknown"
                        ),
                    }
                    gpus.append(gpu_info)

            return gpus if gpus else None
        except ImportError:
            return "WMI module not installed. Install with 'pip install wmi' for AMD GPU detection on Windows."

    elif sys.platform == "linux":
        try:
            lspci_output = subprocess.check_output(
                "lspci | grep -i vga", shell=True, universal_newlines=True
            )
            gpus = []

            for i, line in enumerate(lspci_output.strip().split("\n")):
                if "AMD" in line or "Radeon" in line:
                    gpus.append(
                        {
                            "index": i,
                            "name": line.split(":")[-1].strip(),
                            "note": "Install ROCm tools for more details.",
                        }
                    )

            return gpus if gpus else None
        except subprocess.SubprocessError:
            return None

    return None


def get_intel_gpu_info():
    """Get Intel GPU details"""
    if sys.platform == "win32":
        try:
            import wmi

            w = wmi.WMI()
            gpus = []

            for i, gpu in enumerate(w.Win32_VideoController()):
                if "Intel" in gpu.Name:
                    gpu_info = {
                        "index": i,
                        "name": gpu.Name,
                        "driver_version": gpu.DriverVersion,
                        "adapter_ram_MB": (
                            int(int(gpu.AdapterRAM) / (1024 * 1024))
                            if gpu.AdapterRAM
                            else "Unknown"
                        ),
                    }
                    gpus.append(gpu_info)

            return gpus if gpus else None
        except ImportError:
            return "WMI module not installed. Install with 'pip install wmi' for Intel GPU detection on Windows."

    elif sys.platform == "linux":
        try:
            lspci_output = subprocess.check_output(
                "lspci | grep -i vga", shell=True, universal_newlines=True
            )
            gpus = []

            for i, line in enumerate(lspci_output.strip().split("\n")):
                if "Intel" in line:
                    gpus.append({"index": i, "name": line.split(":")[-1].strip()})

            return gpus if gpus else None
        except subprocess.SubprocessError:
            return None

    return None


def get_disk_info():
    """Get disk information"""
    disk_info = []
    for partition in psutil.disk_partitions():
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
            disk_detail = {
                "device": partition.device,
                "mountpoint": partition.mountpoint,
                "file_system_type": partition.fstype,
                "total_size": get_size(partition_usage.total),
                "used": get_size(partition_usage.used),
                "free": get_size(partition_usage.free),
                "percentage": partition_usage.percent,
            }
            disk_info.append(disk_detail)
        except (PermissionError, FileNotFoundError):
            continue

    return disk_info


def assess_llm_capability(hardware_info):
    """Basic assessment of system's capability to run LLMs"""
    assessment = {"summary": "", "details": {}}

    # Analyze RAM
    ram_gb = 0
    try:
        ram_str = hardware_info["memory"]["total"]
        if "GB" in ram_str:
            ram_gb = float(ram_str.replace("GB", ""))
        elif "MB" in ram_str:
            ram_gb = float(ram_str.replace("MB", "")) / 1024
    except Exception:
        ram_gb = 0

    assessment["details"]["ram"] = {
        "value": hardware_info["memory"]["total"],
        "assessment": (
            "Insufficient"
            if ram_gb < 8
            else (
                "Minimal" if ram_gb < 16 else ("Good" if ram_gb < 32 else "Excellent")
            )
        ),
        "recommendation": "",
    }

    if ram_gb < 8:
        assessment["details"]["ram"]["recommendation"] = (
            "Consider upgrading RAM to at least 16GB for running smaller LLMs"
        )
    elif ram_gb < 16:
        assessment["details"]["ram"]["recommendation"] = (
            "Sufficient for smaller models with quantization"
        )
    elif ram_gb < 32:
        assessment["details"]["ram"]["recommendation"] = (
            "Good for most quantized models up to ~13B parameters"
        )
    else:
        assessment["details"]["ram"]["recommendation"] = (
            "Excellent for running multiple or larger models"
        )

    # Analyze CPU
    cpu_cores = hardware_info["cpu"]["cores_logical"]
    assessment["details"]["cpu"] = {
        "value": f"{hardware_info['cpu']['brand']} ({cpu_cores} logical cores)",
        "assessment": (
            "Limited"
            if cpu_cores < 4
            else (
                "Adequate"
                if cpu_cores < 8
                else ("Good" if cpu_cores < 16 else "Excellent")
            )
        ),
        "recommendation": "",
    }

    if cpu_cores < 4:
        assessment["details"]["cpu"]["recommendation"] = (
            "CPU will be a significant bottleneck for LLMs"
        )
    elif cpu_cores < 8:
        assessment["details"]["cpu"]["recommendation"] = (
            "Can run smaller models but with limited performance"
        )
    elif cpu_cores < 16:
        assessment["details"]["cpu"]["recommendation"] = (
            "Good for most CPU-based inference"
        )
    else:
        assessment["details"]["cpu"]["recommendation"] = (
            "Excellent for CPU-based inference"
        )

    # Analyze GPU
    gpu_assessment = "None"
    gpu_recommendation = "No dedicated GPU detected. LLMs will run on CPU only, which will be significantly slower."
    gpu_value = "None detected"

    # Check NVIDIA GPU
    if "nvidia_gpus" in hardware_info and hardware_info["nvidia_gpus"]:
        gpu = hardware_info["nvidia_gpus"][0]  # First GPU if multiple exist
        gpu_value = f"{gpu['name']} with {gpu['memory_total_MB']}MB VRAM"

        # Parse VRAM size
        vram_gb = 0
        try:
            vram_gb = float(gpu["memory_total_MB"]) / 1024
        except Exception:
            vram_gb = 0

        if vram_gb < 4:
            gpu_assessment = "Limited"
            gpu_recommendation = (
                "Limited VRAM. Can only run heavily quantized smaller models."
            )
        elif vram_gb < 8:
            gpu_assessment = "Basic"
            gpu_recommendation = "Can run quantized models up to ~7B parameters."
        elif vram_gb < 12:
            gpu_assessment = "Good"
            gpu_recommendation = "Good for most quantized models up to ~13B parameters."
        elif vram_gb < 24:
            gpu_assessment = "Very Good"
            gpu_recommendation = "Can handle models up to ~30B with quantization."
        else:
            gpu_assessment = "Excellent"
            gpu_recommendation = (
                "Can handle large models including 70B+ with appropriate quantization."
            )

    # Check AMD or Intel GPU if no NVIDIA
    elif ("amd_gpus" in hardware_info and hardware_info["amd_gpus"]) or (
        "intel_gpus" in hardware_info and hardware_info["intel_gpus"]
    ):
        if "amd_gpus" in hardware_info and hardware_info["amd_gpus"]:
            gpu = hardware_info["amd_gpus"][0]
            gpu_value = f"{gpu['name']}"
            if "adapter_ram_MB" in gpu and gpu["adapter_ram_MB"] != "Unknown":
                gpu_value += f" with {gpu['adapter_ram_MB']}MB VRAM"

                try:
                    vram_gb = float(gpu["adapter_ram_MB"]) / 1024
                    if vram_gb < 4:
                        gpu_assessment = "Limited (AMD)"
                    elif vram_gb < 8:
                        gpu_assessment = "Basic (AMD)"
                    else:
                        gpu_assessment = "Good (AMD)"
                except Exception:
                    gpu_assessment = "Unknown capability (AMD)"
            else:
                gpu_assessment = "Unknown capability (AMD)"

            gpu_recommendation = "AMD GPU support for LLMs is more limited than NVIDIA. Consider using DirectML or ROCm-based tools if available."

        elif "intel_gpus" in hardware_info and hardware_info["intel_gpus"]:
            gpu = hardware_info["intel_gpus"][0]
            gpu_value = f"{gpu['name']}"
            if "adapter_ram_MB" in gpu and gpu["adapter_ram_MB"] != "Unknown":
                gpu_value += f" with {gpu['adapter_ram_MB']}MB VRAM"

            gpu_assessment = "Limited (Intel)"
            gpu_recommendation = "Intel GPU support for LLMs is limited. Consider CPU inference or an NVIDIA GPU for better performance."

    assessment["details"]["gpu"] = {
        "value": gpu_value,
        "assessment": gpu_assessment,
        "recommendation": gpu_recommendation,
    }

    # Create overall summary
    if "nvidia_gpus" in hardware_info and hardware_info["nvidia_gpus"]:
        gpu = hardware_info["nvidia_gpus"][0]
        vram_gb = (
            float(gpu["memory_total_MB"]) / 1024
            if gpu["memory_total_MB"].isdigit()
            else 0
        )

        if vram_gb >= 24 and ram_gb >= 32 and cpu_cores >= 16:
            summary = "Excellent system for running LLMs. Can handle most models including larger ones (70B+)."
        elif vram_gb >= 12 and ram_gb >= 16 and cpu_cores >= 8:
            summary = "Very good system for LLMs. Should handle models up to ~30B parameters with appropriate quantization."
        elif vram_gb >= 8 and ram_gb >= 16 and cpu_cores >= 8:
            summary = (
                "Good system for LLMs. Best suited for models up to ~13B parameters."
            )
        elif vram_gb >= 4 and ram_gb >= 8 and cpu_cores >= 4:
            summary = "Basic system for LLMs. Limited to smaller models (7B or less) with quantization."
        else:
            summary = "Limited system for LLMs. Will struggle with most models without heavy optimization."
    else:
        if ram_gb >= 32 and cpu_cores >= 16:
            summary = "Good CPU-only system. Can run quantized models but will be significantly slower than GPU-based systems."
        elif ram_gb >= 16 and cpu_cores >= 8:
            summary = "Basic CPU-only system. Limited to smaller quantized models with reduced performance."
        else:
            summary = "Limited CPU-only system. Will struggle with most LLMs."

    assessment["summary"] = summary
    return assessment


def collect_hardware_info():
    """Collect all hardware information relevant for running LLMs"""
    hardware_info = {
        "system_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "disk": get_disk_info(),
    }

    # Get GPU info
    nvidia_gpus = get_nvidia_gpu_info()
    if nvidia_gpus:
        hardware_info["nvidia_gpus"] = nvidia_gpus

    amd_gpus = get_amd_gpu_info()
    if amd_gpus and isinstance(amd_gpus, list):
        hardware_info["amd_gpus"] = amd_gpus

    intel_gpus = get_intel_gpu_info()
    if intel_gpus and isinstance(intel_gpus, list):
        hardware_info["intel_gpus"] = intel_gpus

    # Assess LLM readiness
    hardware_info["llm_assessment"] = assess_llm_capability(hardware_info)

    return hardware_info


def display_hardware_info(info):
    """Display the collected hardware information in a readable format"""
    print("\n" + "=" * 80)
    print("SYSTEM HARDWARE INFORMATION FOR LLM COMPATIBILITY")
    print("=" * 80)

    print("\n📊 OVERALL LLM ASSESSMENT:")
    print(f"   {info['llm_assessment']['summary']}")

    print("\n💻 SYSTEM INFORMATION:")
    print(
        f"   OS: {info['os']['system']} {info['os']['release']} {info['os']['version']}"
    )
    print(f"   Architecture: {info['os']['machine']}")
    print(f"   Timestamp: {info['system_time']}")

    print("\n🔍 CPU DETAILS:")
    print(f"   Model: {info['cpu']['brand']}")
    print(f"   Physical cores: {info['cpu']['cores_physical']}")
    print(f"   Logical cores: {info['cpu']['cores_logical']}")

    if (
        "frequency_current" in info["cpu"]
        and info["cpu"]["frequency_current"] != "Unknown"
    ):
        print(f"   Current frequency: {info['cpu']['frequency_current']} MHz")
    if "frequency_max" in info["cpu"] and info["cpu"]["frequency_max"] != "Unknown":
        print(f"   Max frequency: {info['cpu']['frequency_max']} MHz")

    print(f"   Assessment: {info['llm_assessment']['details']['cpu']['assessment']}")
    print(
        f"   Recommendation: {info['llm_assessment']['details']['cpu']['recommendation']}"
    )

    print("\n💾 MEMORY (RAM):")
    print(f"   Total: {info['memory']['total']}")
    print(f"   Available: {info['memory']['available']}")
    print(f"   Used: {info['memory']['used']} ({info['memory']['percentage']}%)")
    print(f"   Assessment: {info['llm_assessment']['details']['ram']['assessment']}")
    print(
        f"   Recommendation: {info['llm_assessment']['details']['ram']['recommendation']}"
    )

    print("\n🖥 GPU INFORMATION:")
    if "nvidia_gpus" in info and info["nvidia_gpus"]:
        for i, gpu in enumerate(info["nvidia_gpus"]):
            print(f"   NVIDIA GPU {i+1}: {gpu['name']}")
            print(
                f"   VRAM: {gpu['memory_total_MB']}MB total, {gpu['memory_free_MB']}MB free"
            )
            print(f"   Temperature: {gpu['temperature_C']}°C")
            print(f"   Utilization: {gpu['utilization_percent']}%")
            if "cuda_version" in gpu:
                print(f"   CUDA Version: {gpu['cuda_version']}")
    elif "amd_gpus" in info and info["amd_gpus"]:
        for i, gpu in enumerate(info["amd_gpus"]):
            print(f"   AMD GPU {i+1}: {gpu['name']}")
            if "adapter_ram_MB" in gpu and gpu["adapter_ram_MB"] != "Unknown":
                print(f"   VRAM: {gpu['adapter_ram_MB']}MB")
            if "driver_version" in gpu:
                print(f"   Driver Version: {gpu['driver_version']}")
    elif "intel_gpus" in info and info["intel_gpus"]:
        for i, gpu in enumerate(info["intel_gpus"]):
            print(f"   Intel GPU {i+1}: {gpu['name']}")
            if "adapter_ram_MB" in gpu and gpu["adapter_ram_MB"] != "Unknown":
                print(f"   VRAM: {gpu['adapter_ram_MB']}MB")
    else:
        print("   No dedicated GPU detected")

    print(f"   Assessment: {info['llm_assessment']['details']['gpu']['assessment']}")
    print(
        f"   Recommendation: {info['llm_assessment']['details']['gpu']['recommendation']}"
    )

    print("\n💿 DISK INFORMATION:")
    for i, disk in enumerate(info["disk"]):
        print(f"   Disk {i+1}: {disk['device']} mounted at {disk['mountpoint']}")
        print(
            f"      Total: {disk['total_size']}, Used: {disk['used']} ({disk['percentage']}%), Free: {disk['free']}"
        )

    print("\n" + "=" * 80)
    print("💡 Note: This assessment is based on typical requirements for current LLMs.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print("Collecting system hardware information...")
    hardware_info = collect_hardware_info()

    # Display information in readable format
    display_hardware_info(hardware_info)

    # Save to JSON file
    output_file = f"llm_hardware_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(hardware_info, f, indent=2)
    print(f"Hardware information saved to {output_file}")

    # Quick summary
    print("\nQUICK REFERENCE SUMMARY:")
    print(
        f"CPU: {hardware_info['cpu']['brand']} ({hardware_info['cpu']['cores_logical']} logical cores)"
    )
    print(f"RAM: {hardware_info['memory']['total']}")

    if "nvidia_gpus" in hardware_info and hardware_info["nvidia_gpus"]:
        gpu = hardware_info["nvidia_gpus"][0]
        print(f"GPU: {gpu['name']} with {gpu['memory_total_MB']}MB VRAM")
    elif "amd_gpus" in hardware_info and hardware_info["amd_gpus"]:
        gpu = hardware_info["amd_gpus"][0]
        print(f"GPU: {gpu['name']} (AMD)")
    elif "intel_gpus" in hardware_info and hardware_info["intel_gpus"]:
        gpu = hardware_info["intel_gpus"][0]
        print(f"GPU: {gpu['name']} (Intel)")
    else:
        print("GPU: None detected")

    print(f"LLM Capability: {hardware_info['llm_assessment']['summary']}")
