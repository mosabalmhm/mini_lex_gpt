from __future__ import annotations

import importlib.util
import os
import platform
import sys
from pathlib import Path

def bytes_to_gib(number_of_bytes: int) -> float:
    """
    Convert bytes to gibibytes.
  
    1 GiB = 1024 ** 3 bytes
    """
    return number_of_bytes / (1024 ** 3)

def read_linux_memory() -> tuple[int | None, int | None]:
    """
    Read total and available memory from / proc/meminfo

    Returns:
        A tuple containing:
        - total memory in bytes
        - available memory in bytes
    """
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return None, None
    memory_values: dict[str, int] = {}
    for line in meminfo_path.read_text().splitlines():
        key, value = line.split(":", maxsplit=1)
        parts = value.strip().split()
    
        if not parts:
            continue
        # Linux reports the values in KiB.
        value_in_kib = int(parts[0])
        memory_values[key] = value_in_kib * 1024
    total_memory = memory_values.get("MemTotal")
    available_memory = memory_values.get("MemAvailable")
    return total_memory, available_memory

def get_linux_distribution() -> str:
    """
    Read the Linux distribution name from /etc/os-reöease
    """

    os_release_path = Path("/etc/os-release")

    if not os_release_path.exists():
        return "Unknown Linux distribution"


    values: dict[str, str] = {}


    for line in os_release_path.read_text().splitlines():

        if "=" not in line:
            continue

        key, value = line.split("=", maxsplit=1)
        values[key] = value.strip().strip('"')

    return values.get("PRETTY_NAME", "Unknown Linux distribution")



def get_cpu_model() -> str:
    """
    Read the CPU model name from / proc/cpuinfo.
    """

    cpuinfo_path = Path("/proc/cpuinfo")

    if not cpuinfo_path.exists():
        return platform.processor() or "Unknown CPU"

    for line in cpuinfo_path.read_text().splitlines():
        if line.lower().startswith("model name"):
            _, model_name = line.split(":", maxsplit=1)
            return model_name.strip()

    return platform.processor() or "Unknown CPU"

def running_inside_wsl() -> bool:
    """
    Detect whether Linux is running through Windows Subsystem for Linux.
    """

    release_text = platform.release().lower()
    return (
        "microsoft" in release_text
        or "wsl" in release_text
        or "WSL_DISTRO_NAME" in os.environ
    )

def check_pytorch() -> None:
    """
    Print pyTorch and accelerator information.
    """
    
    print("\n--- Pytorch Information ---")
    try:
        import torch
    except ImportError:
        print("PyTorch installed: No")
        print("Selected device: CPU")
        return
    print("PyTorch installed: Yes")
    print("PyTorch version:", torch.__version__)
    print("PyTorch file:", torch.
__file__)
    
    print("PyTorch CPU threads:", torch.get_num_threads())
    print(
        "PyTorch interop threads:",
        torch.get_num_interop_threads()
    )

    cuda_available = torch.cuda.is_available()
    xpu_available = (
        hasattr(torch, "xpu")
        and torch.xpu.is_available()
    )
    directml_installed = (
        importlib.util.find_spec("torch_directml") is not None
    )

    print("\nCUDA available:", cuda_available)
    print("Intel XPU available:", xpu_available)
    print("DirectML package installed:", directml_installed)

    if cuda_available:
        print("CUDA device count:", torch.cuda.device_count())
        for device_index in range(torch.cuda.device_count()):
            print(
                f"CUDA device {device_index}",
                torch.cuda.get_device_name(device_index)
            )

    if xpu_available:
        print("XPU device count:", torch.xpu.device_count())
        for device_index in range(torch.xpu.device_count()):
            print( 
                f"XPU device {device_index}:",
                torch.xpu.get_device_name(device_index)
            )

    if cuda_available:
        selected_device = "cuda"
    elif xpu_available:
        selected_device = "xpu"
    else:
        selected_device = "cpu"

    print("\nSelected training device:", selected_device)


def main() -> None:
    """
    Run all envronment checks.
    """

    total_memory, available_memory = read_linux_memory()

    print("=" * 40)
    print("MiniLexGPT Environment Check")
    print("=" * 40)

    print("\n--- Project Information ---")
    print("Current directory:", Path.cwd())
    print("Python executable:", sys.executable)
    print("Python version:", platform.python_version())

    print("\n--- Operating System ---")
    print("Linux disribution:", get_linux_distribution())
    print("Kernel:", platform.release())
    print("Running inside WSL:", running_inside_wsl())


    print("\n--- Processor ---")
    print("CPU model:", get_cpu_model())
    print("Logical processors:", os.cpu_count())


    print("\n--- Memory ---")
    if total_memory is not None:
        print(
            "Total memory:",
            f"{bytes_to_gib(total_memory):.2f} GiB"
        )
    else:
        print("Total memory: Could not be detected")
    if available_memory is not None:
        print(
            "Available memory:",
            f"{bytes_to_gib(available_memory):.2f} GiB"
        )
    else:
        print("Available memory: Could not be detected")
    check_pytorch()


    print("\n=====================================")
    print(" Environment check completed")
    print("=====================================")

if __name__ == "__main__":
    main()

