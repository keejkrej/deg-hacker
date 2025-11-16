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


__all__ = ["get_default_device"]
