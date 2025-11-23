"""Inference helpers for running the multi-task model."""

from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
import torch
import warnings
from scipy.signal import find_peaks

from kymo_tracker.deeplearning.models.multitask import MultiTaskUNet


def extract_peaks_from_heatmap(
    heatmap: np.ndarray,
    max_peaks: int = 3,
    min_peak_value: float = 0.1,
    nms_window: int = 5,
) -> list[np.ndarray]:
    """
    Extract peaks from heatmap using argmax + non-maximum suppression.
    
    Args:
        heatmap: (T, W) heatmap array
        max_peaks: Maximum number of peaks to extract per time frame
        min_peak_value: Minimum peak value threshold
        nms_window: Window size for non-maximum suppression (in pixels)
        
    Returns:
        trajectories: List of (T,) arrays, one per track
    """
    T, W = heatmap.shape
    trajectories = []
    
    # Initialize trajectories
    for _ in range(max_peaks):
        trajectories.append(np.full(T, np.nan, dtype=np.float32))
    
    for t in range(T):
        row = heatmap[t, :]
        
        # Find local maxima
        peaks = []
        peak_values = []
        
        # Apply threshold
        thresholded = row >= min_peak_value
        
        if not thresholded.any():
            continue
        
        # Find local maxima with non-maximum suppression
        for i in range(1, W - 1):
            if not thresholded[i]:
                continue
            
            # Check if it's a local maximum
            if row[i] >= row[i-1] and row[i] >= row[i+1]:
                # Check NMS: ensure no nearby peak is higher
                is_max = True
                for j in range(max(0, i - nms_window), min(W, i + nms_window + 1)):
                    if j != i and row[j] > row[i]:
                        is_max = False
                        break
                
                if is_max:
                    # Subpixel refinement using quadratic interpolation
                    if 0 < i < W - 1:
                        y0, y1, y2 = row[i-1], row[i], row[i+1]
                        denom = (y0 - 2*y1 + y2)
                        if denom != 0:
                            delta = 0.5 * (y0 - y2) / denom
                            peak_pos = i + delta
                        else:
                            peak_pos = float(i)
                    else:
                        peak_pos = float(i)
                    
                    peaks.append(peak_pos)
                    peak_values.append(row[i])
        
        # Sort by peak value and take top max_peaks
        if peaks:
            sorted_indices = np.argsort(peak_values)[::-1]
            sorted_peaks = [peaks[i] for i in sorted_indices[:max_peaks]]
            
            for track_idx, peak_pos in enumerate(sorted_peaks):
                if track_idx < max_peaks:
                    trajectories[track_idx][t] = peak_pos
    
    # Remove trajectories that are all NaN
    trajectories = [traj for traj in trajectories if np.any(~np.isnan(traj))]
    
    # Ensure at least one trajectory
    if not trajectories:
        trajectories.append(np.full(T, np.nan))
    
    return trajectories


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


def extract_trajectories_from_heatmap(
    kymograph: np.ndarray,
    heatmap: np.ndarray,
    max_tracks: int = 3,
    peak_prominence: Optional[float] = None,
    peak_distance: Optional[float] = None,
) -> list[np.ndarray]:
    """
    Extract trajectories using heatmap-weighted raw image and peak finding.
    
    The heatmap is squared for better contrast, then applied as weights to the raw
    kymograph. For each timepoint, find_peaks is used to detect particle positions.
    
    Args:
        kymograph: (T, W) raw kymograph
        heatmap: (T, W) heatmap from model (will be squared and normalized)
        max_tracks: Maximum number of tracks to extract per timepoint
        peak_prominence: Minimum prominence for peak detection (None = auto)
        peak_distance: Minimum distance between peaks (None = auto)
        
    Returns:
        trajectories: List of (T,) arrays, one per track
    """
    T, W = kymograph.shape
    
    # Square the heatmap for better contrast
    heatmap_squared = heatmap ** 2
    
    # Normalize heatmap to [0, 1] range for each timepoint
    for t in range(T):
        row_max = heatmap_squared[t].max()
        if row_max > 0:
            heatmap_squared[t] = heatmap_squared[t] / row_max
    
    # Apply heatmap as weights to raw image
    weighted_kymograph = kymograph * heatmap_squared
    
    # Extract trajectories using peak finding for each timepoint
    trajectories = [np.full(T, np.nan) for _ in range(max_tracks)]
    
    for t in range(T):
        row = weighted_kymograph[t]
        
        # Auto-determine prominence if not provided (use percentile of row)
        if peak_prominence is None:
            prominence = np.percentile(row, 75) - np.percentile(row, 25)
            prominence = max(prominence, row.max() * 0.1)  # At least 10% of max
        else:
            prominence = peak_prominence
        
        # Auto-determine distance if not provided (assume particles are well-separated)
        if peak_distance is None:
            distance = W / (max_tracks + 1)  # Rough estimate
        else:
            distance = peak_distance
        
        # Find peaks
        peaks, properties = find_peaks(
            row,
            prominence=prominence,
            distance=distance,
        )
        
        # Sort peaks by height (descending) and take top max_tracks
        if len(peaks) > 0:
            peak_heights = row[peaks]
            sorted_indices = np.argsort(peak_heights)[::-1]
            top_peaks = peaks[sorted_indices[:max_tracks]]
            
            # Subpixel refinement for each peak
            for track_idx, peak_idx in enumerate(top_peaks):
                if 0 < peak_idx < W - 1:
                    y0, y1, y2 = row[peak_idx-1], row[peak_idx], row[peak_idx+1]
                    denom = (y0 - 2*y1 + y2)
                    if denom != 0:
                        delta = 0.5 * (y0 - y2) / denom
                        trajectories[track_idx][t] = peak_idx + delta
                    else:
                        trajectories[track_idx][t] = peak_idx
                else:
                    trajectories[track_idx][t] = peak_idx
    
    # Remove trajectories that are all NaN
    trajectories = [traj for traj in trajectories if not np.all(np.isnan(traj))]
    
    # Ensure at least one trajectory
    if not trajectories:
        trajectories.append(np.full(T, np.nan))
    
    return trajectories


def process_slice_independently(
    model: MultiTaskUNet,
    kymograph_slice: np.ndarray,
    device: Optional[str] = None,
    peak_prominence: Optional[float] = None,
    peak_distance: Optional[float] = None,
) -> dict:
    """
    Process a single 16x512 slice independently and extract trajectories from heatmap/segmentation.
    
    Uses heatmap/segmentation-weighted peak finding: the model's output is squared for contrast,
    applied as weights to the raw image, then find_peaks is used for each timepoint.
    
    Args:
        model: Multi-task model
        kymograph_slice: (T, W) kymograph slice (typically 16x512)
        device: Device to run model on
        peak_prominence: Minimum prominence for peak detection (None = auto)
        peak_distance: Minimum distance between peaks (None = auto)
        
    Returns:
        Dictionary with:
        - 'denoised': (T, W) denoised slice
        - 'trajectories': List of (T,) trajectory arrays
        - 'heatmap': (T, W) heatmap/segmentation map from model (continuous 0-1)
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
        pred_noise, pred_map = model(input_tensor)
        
        # Check for NaN outputs
        if torch.isnan(pred_noise).any() or torch.isnan(pred_map).any():
            warnings.warn(
                "Model output contains NaN values. Model may not be properly trained. "
                "Using input as fallback (no denoising applied).",
                UserWarning
            )
            denoised_slice = kymograph_slice.copy()
            map_np = np.zeros((T, W), dtype=np.float32)
        else:
            denoised_chunk = torch.clamp(input_tensor - pred_noise, 0.0, 1.0).squeeze().cpu().numpy()
            denoised_slice = denoised_chunk[:T]  # Trim to actual length
            
            # Convert to probabilities (0-1) if segmentation mode (logits)
            if model.mode == "segmentation":
                # Apply sigmoid to convert logits to probabilities
                pred_map = torch.sigmoid(pred_map)
            
            map_np = pred_map.squeeze(0).cpu().numpy()[:T]  # (T, W)
        
        del input_tensor, pred_noise, pred_map
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    
    # Extract trajectories using map-weighted peak finding (works for both heatmap and segmentation)
    trajectories = extract_trajectories_from_heatmap(
        kymograph_slice,
        map_np,
        max_tracks=model.max_tracks,
        peak_prominence=peak_prominence,
        peak_distance=peak_distance,
    )
    
    return {
        'denoised': denoised_slice,
        'trajectories': trajectories,
        'heatmap': map_np,  # Keep name 'heatmap' for backward compatibility
    }


def link_trajectories_across_slices(
    slice_trajectories_list: List[List[np.ndarray]],
    chunk_size: int = 16,
    overlap: int = 8,
    max_jump: float = 10.0,
    total_length: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Link trajectories across overlapping slices using greedy assignment.
    
    Args:
        slice_trajectories_list: List of lists, where each inner list contains trajectories
                                 for one slice. Each trajectory is (T,) array.
        chunk_size: Size of each chunk (default 16)
        overlap: Overlap between chunks (default 8)
        max_jump: Maximum allowed jump in position between slices (in pixels)
        total_length: Total length of the full kymograph. If None, calculated from slice boundaries.
        
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
    
    # Calculate actual total length from the last slice's actual end position
    # The last slice might be shorter than chunk_size
    if total_length is None:
        # Infer from the last slice's trajectory length
        if slice_trajectories_list[-1] and len(slice_trajectories_list[-1]) > 0:
            last_traj_len = len(slice_trajectories_list[-1][0])
            total_length = slice_starts[-1] + last_traj_len
        else:
            # Fallback: use slice boundary calculation but cap at reasonable value
            total_length = slice_ends[-1]
    
    # Initialize linked trajectories
    linked_trajectories = []
    
    for track_idx in range(n_tracks):
        # Collect trajectory segments for this track across all slices
        segments = []
        for slice_idx, trajs in enumerate(slice_trajectories_list):
            if track_idx < len(trajs):
                traj = trajs[track_idx]
                start = slice_starts[slice_idx]
                # Use actual trajectory length, not theoretical end
                actual_end = start + len(traj)
                # Store segment with its time range
                segments.append({
                    'trajectory': traj,
                    'start': start,
                    'end': actual_end,
                    'slice_idx': slice_idx,
                })
        
        # Link segments together
        if not segments:
            continue
        
        # Simple linking: concatenate segments, handling overlaps by taking average
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
        linked_trajectories.append(np.full(total_length, np.nan))
    
    return linked_trajectories


__all__ = [
    "extract_trajectories_from_heatmap",
    "process_slice_independently",
    "link_trajectories_across_slices",
]
