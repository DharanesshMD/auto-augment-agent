"""Reproducibility utilities for deterministic training."""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Sets seeds for: Python random, NumPy, PyTorch CPU, PyTorch CUDA.
    Also configures PyTorch for deterministic behavior.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic operations (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed for reproducible hashing
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(preference: str = "auto"):
    """Get the best available device based on preference.

    Args:
        preference: One of "auto", "cuda", "mps", "cpu".

    Returns:
        torch.device for the selected hardware.
    """
    import torch

    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def get_device_info(device) -> dict[str, Any]:
    """Get information about the selected device."""
    import torch

    info = {
        "device": str(device),
        "type": device.type,
    }

    if device.type == "cuda":
        info.update({
            "name": torch.cuda.get_device_name(device),
            "memory_total_gb": round(
                torch.cuda.get_device_properties(device).total_mem / 1e9, 2
            ),
            "memory_allocated_gb": round(torch.cuda.memory_allocated(device) / 1e9, 2),
            "cuda_version": torch.version.cuda,
        })
    elif device.type == "mps":
        info["name"] = "Apple Silicon (MPS)"

    return info


def hash_config(config: dict[str, Any]) -> str:
    """Generate a deterministic SHA256 hash of a config for unique trial identification."""
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def get_system_info() -> dict[str, str]:
    """Gather system information for reproducibility logs."""
    import platform
    import sys

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    try:
        import torch

        info["torch_version"] = torch.__version__

        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or "N/A"
            info["cudnn_version"] = str(torch.backends.cudnn.version())
            info["gpu_count"] = str(torch.cuda.device_count())
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch_version"] = "not installed"

    try:
        import transformers

        info["transformers_version"] = transformers.__version__
    except ImportError:
        pass

    return info
