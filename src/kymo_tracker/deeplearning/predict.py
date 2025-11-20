"""Inference helpers for running the multi-task model."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from kymo_tracker.deeplearning.models.multitask import MultiTaskUNet


def denoise_and_segment_chunked(
    model: MultiTaskUNet,
    kymograph: np.ndarray,
    device: Optional[str] = None,
    chunk_size: int = 16,
    overlap: int = 8,
) -> Tuple[np.ndarray, dict[str, np.ndarray]]:
    """Apply the model to a kymograph with temporal chunking."""

    if device is None:
        device = next(model.parameters()).device.type

    model.eval()
    time_len, width = kymograph.shape

    if time_len <= chunk_size:
        with torch.no_grad():
            input_tensor = torch.from_numpy(kymograph).unsqueeze(0).unsqueeze(0).float().to(device)
            pred_noise, pred_centers, pred_widths = model(input_tensor)
            denoised = torch.clamp(input_tensor - pred_noise, 0.0, 1.0).squeeze().cpu().numpy()
            centers_np = pred_centers.squeeze(0).cpu().numpy().transpose(1, 0)
            widths_np = pred_widths.squeeze(0).cpu().numpy().transpose(1, 0)
            centers_px = centers_np * (width - 1)
            widths_px = widths_np * width
            track_params = {"centers": centers_px, "widths": widths_px}

            del input_tensor, pred_noise, pred_centers, pred_widths
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

        return denoised, track_params

    denoised = np.zeros((time_len, width), dtype=np.float32)
    weights = np.zeros((time_len, width), dtype=np.float32)
    temporal_weights = np.zeros((time_len, 1), dtype=np.float32)
    centers_all = None
    widths_all = None

    window = np.ones(chunk_size)
    if overlap > 0:
        fade_len = overlap // 2
        window[:fade_len] = np.linspace(0, 1, fade_len)
        window[-fade_len:] = np.linspace(1, 0, fade_len)

    with torch.no_grad():
        start = 0
        while start < time_len:
            end = min(start + chunk_size, time_len)
            chunk = kymograph[start:end]

            padded_chunk = chunk
            if chunk.shape[0] < chunk_size:
                padding = np.zeros((chunk_size - chunk.shape[0], width), dtype=chunk.dtype)
                padded_chunk = np.vstack([chunk, padding])

            chunk_tensor = torch.from_numpy(padded_chunk).unsqueeze(0).unsqueeze(0).float().to(device)
            pred_noise_chunk, pred_centers_chunk, pred_widths_chunk = model(chunk_tensor)
            denoised_chunk = torch.clamp(chunk_tensor - pred_noise_chunk, 0.0, 1.0).squeeze().cpu().numpy()
            centers_chunk = pred_centers_chunk.squeeze(0).cpu().numpy().transpose(1, 0)
            widths_chunk = pred_widths_chunk.squeeze(0).cpu().numpy().transpose(1, 0)
            centers_chunk = centers_chunk * (width - 1)
            widths_chunk = widths_chunk * width

            del chunk_tensor, pred_noise_chunk, pred_centers_chunk, pred_widths_chunk
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

            actual_len = end - start
            denoised_chunk = denoised_chunk[:actual_len]
            centers_chunk = centers_chunk[:actual_len]
            widths_chunk = widths_chunk[:actual_len]
            window_chunk = window[:actual_len]

            if centers_all is None:
                max_tracks = centers_chunk.shape[1]
                centers_all = np.zeros((time_len, max_tracks), dtype=np.float32)
                widths_all = np.zeros((time_len, max_tracks), dtype=np.float32)

            weight_chunk = window_chunk[:, np.newaxis]
            denoised[start:end] += denoised_chunk * weight_chunk
            weights[start:end] += weight_chunk
            centers_all[start:end] += centers_chunk * weight_chunk
            widths_all[start:end] += widths_chunk * weight_chunk
            temporal_weights[start:end] += weight_chunk

            del denoised_chunk, centers_chunk, widths_chunk
            start += chunk_size - overlap

    denoised = np.divide(denoised, weights, out=np.zeros_like(denoised), where=weights > 0)
    centers_all = np.divide(centers_all, temporal_weights, out=np.zeros_like(centers_all), where=temporal_weights > 0)
    widths_all = np.divide(widths_all, temporal_weights, out=np.zeros_like(widths_all), where=temporal_weights > 0)

    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return denoised, {"centers": centers_all, "widths": widths_all}


__all__ = ["denoise_and_segment_chunked"]
