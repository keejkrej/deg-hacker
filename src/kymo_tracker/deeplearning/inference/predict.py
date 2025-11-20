"""Backward-compatible shim for deep-learning inference helpers.

The canonical import path is :mod:`kymo_tracker.deeplearning.predict`.
"""

from kymo_tracker.deeplearning.predict import denoise_and_segment_chunked

__all__ = ["denoise_and_segment_chunked"]
