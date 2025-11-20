"""Compat wrapper for deep-learning inference utilities.

The canonical import path is :mod:`kymo_tracker.deeplearning`.
"""

from kymo_tracker.deeplearning import (
    process_slice_independently,
    link_trajectories_across_slices,
)

__all__ = ["process_slice_independently", "link_trajectories_across_slices"]
