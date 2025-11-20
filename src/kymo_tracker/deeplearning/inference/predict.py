"""Backward-compatible shim for deep-learning inference helpers.

The canonical import path is :mod:`kymo_tracker.deeplearning.predict`.
"""

from kymo_tracker.deeplearning.predict import (
    process_slice_independently,
    link_trajectories_across_slices,
)

__all__ = ["process_slice_independently", "link_trajectories_across_slices"]
