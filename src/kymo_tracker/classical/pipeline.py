"""Classical tracking pipelines and utilities.

This module keeps the median/threshold peak-finding workflow in a single
place so it can be contrasted with the deep-learning inference utilities.
Each step is factored out for clarity and for easier reuse/testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.ndimage import label, median_filter
from skimage.filters import threshold_otsu

from kymo_tracker.utils.helpers import find_max_subpixel


@dataclass
class ClassicalTrackingResult:
    """Outputs of the classical median/threshold tracking pipeline."""

    filtered: np.ndarray
    binary_mask: np.ndarray
    labeled_mask: np.ndarray
    instance_masks: List[np.ndarray]
    trajectories: List[np.ndarray]
    threshold: float


# --- Filtering & segmentation helpers ---------------------------------------------------------

def median_filter_kymograph(kymograph: np.ndarray, kernel: Sequence[int] | int) -> np.ndarray:
    """Apply median filtering to suppress impulsive noise."""

    return median_filter(kymograph, size=kernel)


def compute_threshold(filtered: np.ndarray, *, mode: str = "otsu", sigma: float = 1.0) -> float:
    """Return an intensity threshold for the filtered kymograph."""

    if mode == "otsu":
        return float(threshold_otsu(filtered))

    if mode == "std":
        mean = float(filtered.mean())
        std = float(filtered.std())
        return mean + sigma * std

    raise ValueError(f"Unsupported threshold mode: {mode}")


def segment_filtered(
    filtered: np.ndarray, *, threshold: float, min_component_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Binarize the filtered kymograph and drop tiny connected components."""

    binary_mask = filtered > threshold
    labeled_mask, num_features = label(binary_mask)

    # Zero out labels that are too small so downstream masks stay sparse
    for label_idx in range(1, num_features + 1):
        component_mask = labeled_mask == label_idx
        if component_mask.sum() < min_component_size:
            labeled_mask[component_mask] = 0

    return binary_mask, labeled_mask


def extract_instance_masks(labeled_mask: np.ndarray) -> List[np.ndarray]:
    """Split labeled mask into per-instance boolean masks."""

    instance_masks: List[np.ndarray] = []
    for label_idx in range(1, int(labeled_mask.max()) + 1):
        component_mask = labeled_mask == label_idx
        if component_mask.any():
            instance_masks.append(component_mask)
    return instance_masks


# --- Full pipeline ---------------------------------------------------------------------------

def classical_median_threshold_tracking(
    kymograph: np.ndarray,
    *,
    median_kernel: Sequence[int] | int = (3, 3),
    threshold_mode: str = "otsu",
    threshold_sigma: float = 1.0,
    min_component_size: int = 8,
) -> ClassicalTrackingResult:
    """
    Recover particle tracks using median filtering, thresholding, and peak finding.

    Args:
        kymograph: Noisy input kymograph with shape (T, W).
        median_kernel: Kernel size passed to :func:`scipy.ndimage.median_filter`.
        threshold_mode: Either ``"otsu"`` or ``"std"`` (mean + sigma*std).
        threshold_sigma: Sigma multiplier used when ``threshold_mode="std"``.
        min_component_size: Minimum number of pixels to keep a connected component.

    Returns:
        ClassicalTrackingResult containing the filtered image, binary mask,
        labeled mask, per-instance masks, and per-frame peak trajectories.
    """

    if kymograph.ndim != 2:
        raise ValueError(f"kymograph must be 2D (T, W); got shape {kymograph.shape}")

    filtered = median_filter_kymograph(kymograph, median_kernel)
    threshold_value = compute_threshold(filtered, mode=threshold_mode, sigma=threshold_sigma)
    binary_mask, labeled_mask = segment_filtered(
        filtered, threshold=threshold_value, min_component_size=min_component_size
    )
    instance_masks = extract_instance_masks(labeled_mask)

    trajectories: List[np.ndarray] = []
    for component_mask in instance_masks:
        masked_kymo = np.where(component_mask, kymograph, -np.inf)
        trajectories.append(find_max_subpixel(masked_kymo))

    return ClassicalTrackingResult(
        filtered=filtered,
        binary_mask=binary_mask,
        labeled_mask=labeled_mask,
        instance_masks=instance_masks,
        trajectories=trajectories,
        threshold=threshold_value,
    )


__all__ = [
    "ClassicalTrackingResult",
    "classical_median_threshold_tracking",
    "compute_threshold",
    "extract_instance_masks",
    "median_filter_kymograph",
    "segment_filtered",
]
