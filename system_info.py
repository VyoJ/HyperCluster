import os
import time
import tomlkit
from sys_info import collect_hardware_info

# File path for storing system info
SYSTEM_INFO_FILE = "node_system_info.toml"


def collect_and_store_system_info():
    """Collect system info and store it in a TOML file"""
    # Check if we already have a recent system info file
    if os.path.exists(SYSTEM_INFO_FILE):
        # Check file age - if less than 1 day old, use existing data
        file_age = time.time() - os.path.getmtime(SYSTEM_INFO_FILE)
        if file_age < 86400:  # 24 hours in seconds
            return load_system_info()

    # Collect new system info
    hardware_info = collect_hardware_info()

    # Extract only the relevant fields we want to share
    simplified_info = {
        "timestamp": hardware_info["system_time"],
        "os": {
            "system": hardware_info["os"]["system"],
            "version": hardware_info["os"]["version"],
            "machine": hardware_info["os"]["machine"],
        },
        "cpu": {
            "model": hardware_info["cpu"]["brand"],
            "cores_logical": hardware_info["cpu"]["cores_logical"],
            "cores_physical": hardware_info["cpu"]["cores_physical"],
        },
        "memory": {
            "total": hardware_info["memory"]["total"],
            "available": hardware_info["memory"]["available"],
        },
        "llm_capability": hardware_info["llm_assessment"]["summary"],
    }

    # Add GPU information if available
    if "nvidia_gpus" in hardware_info and hardware_info["nvidia_gpus"]:
        gpu = hardware_info["nvidia_gpus"][0]  # Take first GPU if multiple exist
        simplified_info["gpu"] = {
            "type": "NVIDIA",
            "name": gpu["name"],
            "memory": f"{gpu['memory_total_MB']}MB",
        }
    elif "amd_gpus" in hardware_info and hardware_info["amd_gpus"]:
        gpu = hardware_info["amd_gpus"][0]
        simplified_info["gpu"] = {"type": "AMD", "name": gpu["name"]}
        if "adapter_ram_MB" in gpu and gpu["adapter_ram_MB"] != "Unknown":
            simplified_info["gpu"]["memory"] = f"{gpu['adapter_ram_MB']}MB"
    elif "intel_gpus" in hardware_info and hardware_info["intel_gpus"]:
        gpu = hardware_info["intel_gpus"][0]
        simplified_info["gpu"] = {"type": "Intel", "name": gpu["name"]}
    else:
        simplified_info["gpu"] = {"type": "None", "name": "No dedicated GPU"}

    # Save to TOML file
    save_system_info(simplified_info)
    return simplified_info


def save_system_info(system_info):
    """Save system info to TOML file"""
    doc = tomlkit.document()

    # Add each section
    for section_name, section_data in system_info.items():
        if isinstance(section_data, dict):
            table = tomlkit.table()
            for key, value in section_data.items():
                table[key] = value
            doc[section_name] = table
        else:
            doc[section_name] = section_data

    with open(SYSTEM_INFO_FILE, "w") as f:
        f.write(tomlkit.dumps(doc))


def load_system_info():
    """Load system info from TOML file"""
    if not os.path.exists(SYSTEM_INFO_FILE):
        return collect_and_store_system_info()

    with open(SYSTEM_INFO_FILE, "r") as f:
        return tomlkit.parse(f.read())


if __name__ == "__main__":
    # This will collect, store, and print the system info
    system_info = collect_and_store_system_info()
    print("System information collected and stored in", SYSTEM_INFO_FILE)
    print("\n--- System Info ---")
    print(tomlkit.dumps(system_info))
    print("-------------------")
