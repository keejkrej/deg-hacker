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
    # More robust check: ROCm is only active if CUDA is not available
    # and we have ROCm-specific attributes
    if torch.cuda.is_available():
        # If CUDA is available, we're using NVIDIA, not ROCm
        return False
    # Only check for ROCm if CUDA is not available
    # ROCm builds have torch.version.hip or torch.backends.miopen
    has_hip = hasattr(torch.version, "hip") and torch.version.hip is not None
    has_miopen = hasattr(torch.backends, "miopen") and hasattr(torch.backends.miopen, "is_available")
    if has_miopen:
        has_miopen = torch.backends.miopen.is_available()
    return has_hip or has_miopen


__all__ = ["get_default_device", "is_rocm"]
