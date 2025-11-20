"""Inference helpers for running the multi-task model."""

from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
import torch

from kymo_tracker.deeplearning.models.multitask import MultiTaskUNet


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


def process_slice_independently(
    model: MultiTaskUNet,
    kymograph_slice: np.ndarray,
    device: Optional[str] = None,
) -> dict:
    """
    Process a single 16x512 slice independently and extract trajectories.
    
    Args:
        model: Multi-task model
        kymograph_slice: (T, W) kymograph slice (typically 16x512)
        device: Device to run model on
        
    Returns:
        Dictionary with:
        - 'denoised': (T, W) denoised slice
        - 'trajectories': List of (T,) trajectory arrays
        - 'centers': (T, N_tracks) center predictions
        - 'widths': (T, N_tracks) width predictions
        - 'mask': (T, W) boolean mask
        - 'labeled_mask': (T, W) labeled mask
    """
    if device is None:
        device = next(model.parameters()).device.type
    
    model.eval()
    T, W = kymograph_slice.shape
    
    with torch.no_grad():
        # Pad if needed to match chunk_size=16
        if T < 16:
            padding = np.zeros((16 - T, W), dtype=kymograph_slice.dtype)
            padded_slice = np.vstack([kymograph_slice, padding])
        else:
            padded_slice = kymograph_slice[:16]  # Take first 16 frames
        
        input_tensor = torch.from_numpy(padded_slice).unsqueeze(0).unsqueeze(0).float().to(device)
        pred_noise, pred_centers, pred_widths = model(input_tensor)
        
        # Check for NaN outputs
        if torch.isnan(pred_noise).any():
            import warnings
            warnings.warn(
                "Model output contains NaN values. Model may not be properly trained. "
                "Using input as fallback (no denoising applied).",
                UserWarning
            )
            denoised_slice = kymograph_slice.copy()
            max_tracks = pred_centers.shape[1] if pred_centers.ndim > 1 else 1
            centers_px = np.full((T, max_tracks), W / 2, dtype=np.float32)
            widths_px = np.full((T, max_tracks), W * 0.1, dtype=np.float32)
        else:
            denoised_chunk = torch.clamp(input_tensor - pred_noise, 0.0, 1.0).squeeze().cpu().numpy()
            denoised_slice = denoised_chunk[:T]  # Trim to actual length
            
            centers_np = pred_centers.squeeze(0).cpu().numpy().transpose(1, 0)
            widths_np = pred_widths.squeeze(0).cpu().numpy().transpose(1, 0)
            centers_px = (centers_np * (W - 1))[:T]  # Trim to actual length
            widths_px = (widths_np * W)[:T]  # Trim to actual length
        
        del input_tensor, pred_noise, pred_centers, pred_widths
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    
    # Create mask from centers/widths
    mask, labeled_mask = create_mask_from_centers_widths(
        centers_px, widths_px, kymograph_slice.shape
    )
    
    # Extract trajectories from this slice
    n_tracks = centers_px.shape[1] if centers_px.ndim > 1 else 1
    trajectories = extract_trajectories_from_mask(
        kymograph_slice, labeled_mask, n_tracks
    )
    
    # Fallback to centers if mask extraction failed
    all_nan = all(np.all(np.isnan(traj)) for traj in trajectories)
    if all_nan and centers_px.ndim > 1:
        trajectories = []
        for track_idx in range(n_tracks):
            track_centers = centers_px[:, track_idx].copy()
            if np.any(~np.isnan(track_centers)):
                trajectories.append(track_centers)
        if not trajectories:
            trajectories.append(np.full(T, np.nan))
    
    return {
        'denoised': denoised_slice,
        'trajectories': trajectories,
        'centers': centers_px,
        'widths': widths_px,
        'mask': mask,
        'labeled_mask': labeled_mask,
    }


def link_trajectories_across_slices(
    slice_trajectories_list: List[List[np.ndarray]],
    chunk_size: int = 16,
    overlap: int = 8,
    max_jump: float = 10.0,
) -> List[np.ndarray]:
    """
    Link trajectories across overlapping slices using greedy assignment.
    
    Args:
        slice_trajectories_list: List of lists, where each inner list contains trajectories
                                 for one slice. Each trajectory is (T,) array.
        chunk_size: Size of each chunk (default 16)
        overlap: Overlap between chunks (default 8)
        max_jump: Maximum allowed jump in position between slices (in pixels)
        
    Returns:
        List of linked trajectories, one per track
    """
    if not slice_trajectories_list:
        return []
    
    n_slices = len(slice_trajectories_list)
    if n_slices == 1:
        # Single slice, return trajectories as-is
        return slice_trajectories_list[0]
    
    # Determine number of tracks (max across all slices)
    n_tracks = max(len(trajs) for trajs in slice_trajectories_list)
    if n_tracks == 0:
        return []
    
    # Calculate slice boundaries
    step = chunk_size - overlap
    slice_starts = []
    slice_ends = []
    start = 0
    for i in range(n_slices):
        slice_starts.append(start)
        end = start + chunk_size
        slice_ends.append(end)
        start += step
    
    # Initialize linked trajectories
    linked_trajectories = []
    
    for track_idx in range(n_tracks):
        # Collect trajectory segments for this track across all slices
        segments = []
        for slice_idx, trajs in enumerate(slice_trajectories_list):
            if track_idx < len(trajs):
                traj = trajs[track_idx]
                start = slice_starts[slice_idx]
                end = slice_ends[slice_idx]
                # Store segment with its time range
                segments.append({
                    'trajectory': traj,
                    'start': start,
                    'end': end,
                    'slice_idx': slice_idx,
                })
        
        # Link segments together
        if not segments:
            continue
        
        # Simple linking: concatenate segments, handling overlaps by taking average
        total_length = slice_ends[-1]
        linked_traj = np.full(total_length, np.nan, dtype=np.float64)
        
        for seg in segments:
            traj = seg['trajectory']
            start = seg['start']
            end = seg['end']
            traj_len = len(traj)
            
            # Copy trajectory segment, handling overlaps
            for t in range(traj_len):
                global_t = start + t
                if global_t < total_length:
                    if np.isnan(linked_traj[global_t]):
                        linked_traj[global_t] = traj[t]
                    else:
                        # Overlap region: average the values
                        if not np.isnan(traj[t]):
                            linked_traj[global_t] = (linked_traj[global_t] + traj[t]) / 2.0
        
        linked_trajectories.append(linked_traj)
    
    # Ensure at least one trajectory
    if not linked_trajectories:
        total_length = slice_ends[-1] if slice_ends else chunk_size
        linked_trajectories.append(np.full(total_length, np.nan))
    
    return linked_trajectories


__all__ = [
    "create_mask_from_centers_widths",
    "extract_trajectories_from_mask",
    "process_slice_independently",
    "link_trajectories_across_slices",
]
