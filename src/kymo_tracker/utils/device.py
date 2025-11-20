"""Utility helpers related to device selection."""

def get_default_device() -> str:
    """Return the preferred torch device available on this machine."""

    import torch

    has_mps = getattr(torch.backends, "mps", None)
    if has_mps and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def is_rocm() -> bool:
    """Check if PyTorch is running on ROCm (AMD GPU)."""
    import torch
    # ROCm builds have torch.version.hip or torch.backends.miopen
    return hasattr(torch.version, "hip") or hasattr(torch.backends, "miopen")


__all__ = ["get_default_device", "is_rocm"]
