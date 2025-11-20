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
            
            # Check for NaN outputs and handle gracefully
            if torch.isnan(pred_noise).any():
                import warnings
                warnings.warn(
                    "Model output contains NaN values. Model may not be properly trained. "
                    "Using input as fallback (no denoising applied).",
                    UserWarning
                )
                # If model outputs NaN, use input as fallback (no denoising)
                denoised = kymograph.copy()
                # Use default centers/widths
                max_tracks = pred_centers.shape[1] if pred_centers.ndim > 1 else 1
                centers_px = np.full((time_len, max_tracks), width / 2, dtype=np.float32)
                widths_px = np.full((time_len, max_tracks), width * 0.1, dtype=np.float32)
            else:
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
            
            actual_len = end - start
            
            # Check for NaN outputs and handle gracefully
            if torch.isnan(pred_noise_chunk).any():
                import warnings
                warnings.warn(
                    "Model output contains NaN values. Model may not be properly trained. "
                    "Using input as fallback (no denoising applied).",
                    UserWarning
                )
                # If model outputs NaN, use input as fallback (no denoising)
                denoised_chunk = chunk_tensor.squeeze().cpu().numpy()
                # Use default centers/widths
                max_tracks = pred_centers_chunk.shape[1] if pred_centers_chunk.ndim > 1 else 1
                centers_chunk = np.full((actual_len, max_tracks), width / 2, dtype=np.float32)
                widths_chunk = np.full((actual_len, max_tracks), width * 0.1, dtype=np.float32)
            else:
                denoised_chunk = torch.clamp(chunk_tensor - pred_noise_chunk, 0.0, 1.0).squeeze().cpu().numpy()
                centers_chunk = pred_centers_chunk.squeeze(0).cpu().numpy().transpose(1, 0)
                widths_chunk = pred_widths_chunk.squeeze(0).cpu().numpy().transpose(1, 0)
                centers_chunk = centers_chunk * (width - 1)
                widths_chunk = widths_chunk * width

            del chunk_tensor, pred_noise_chunk, pred_centers_chunk, pred_widths_chunk
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

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


def create_mask_from_centers_widths(
    centers: np.ndarray,
    widths: np.ndarray,
    shape: tuple[int, int],
    threshold: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a segmentation mask from predicted centers and widths.
    
    Args:
        centers: (T, N_tracks) array of center positions (in pixels)
        widths: (T, N_tracks) array of widths (in pixels)
        shape: (T, W) shape of the kymograph
        threshold: Minimum width to consider a track active
        
    Returns:
        mask: (T, W) boolean mask
        labeled_mask: (T, W) integer mask with track IDs (0 = background, 1-N = track IDs)
    """
    T, W = shape
    mask = np.zeros((T, W), dtype=bool)
    labeled_mask = np.zeros((T, W), dtype=int)
    
    n_tracks = centers.shape[1] if centers.ndim > 1 else 1
    
    for track_idx in range(n_tracks):
        if centers.ndim == 1:
            track_centers = centers
            track_widths = widths if isinstance(widths, (int, float)) else widths
        else:
            track_centers = centers[:, track_idx]
            track_widths = widths[:, track_idx]
        
        for t in range(T):
            center = track_centers[t]
            width = track_widths[t] if not isinstance(track_widths, (int, float)) else track_widths
            
            # Skip if width is too small or center is invalid
            if width < threshold or np.isnan(center) or center < 0 or center >= W:
                continue
            
            # Create corridor around center
            half_width = width / 2.0
            start_x = max(0, int(np.floor(center - half_width)))
            end_x = min(W, int(np.ceil(center + half_width)) + 1)
            
            mask[t, start_x:end_x] = True
            labeled_mask[t, start_x:end_x] = track_idx + 1
    
    return mask, labeled_mask


def extract_trajectories_from_mask(
    kymograph: np.ndarray,
    labeled_mask: np.ndarray,
    n_tracks: int,
) -> list[np.ndarray]:
    """
    Extract trajectories using argmax within masked regions.
    
    Args:
        kymograph: (T, W) raw kymograph
        labeled_mask: (T, W) integer mask with track IDs
        n_tracks: Number of tracks to extract
        
    Returns:
        trajectories: List of (T,) arrays, one per track
    """
    from kymo_tracker.utils.helpers import find_max_subpixel
    
    T, W = kymograph.shape
    trajectories = []
    
    for track_idx in range(n_tracks):
        traj = np.full(T, np.nan)
        track_mask = labeled_mask == (track_idx + 1)
        
        for t in range(T):
            if track_mask[t].any():
                # Find argmax within the masked region
                masked_row = np.where(track_mask[t], kymograph[t], -np.inf)
                max_idx = np.argmax(masked_row)
                if masked_row[max_idx] > -np.inf:
                    # Subpixel refinement
                    if 0 < max_idx < W - 1:
                        y0, y1, y2 = masked_row[max_idx-1], masked_row[max_idx], masked_row[max_idx+1]
                        denom = (y0 - 2*y1 + y2)
                        if denom != 0:
                            delta = 0.5 * (y0 - y2) / denom
                            traj[t] = max_idx + delta
                        else:
                            traj[t] = max_idx
                    else:
                        traj[t] = max_idx
        
        trajectories.append(traj)
    
    # Ensure at least one trajectory
    if not trajectories:
        trajectories.append(np.full(T, np.nan))
    
    return trajectories


__all__ = ["denoise_and_segment_chunked", "create_mask_from_centers_widths", "extract_trajectories_from_mask"]
