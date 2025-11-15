"""
Multi-Particle Tracking and Analysis Pipeline

This module provides:
- Multi-particle tracking using Otsu binarization and clustering
- Crossing event detection and exclusion
- Comprehensive analysis with diffusion, contrast, and noise estimation
- Parameter grid evaluation for systematic testing
"""

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label, center_of_mass
from sklearn.cluster import DBSCAN
from skimage.filters import threshold_otsu

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from train.multitask_model import load_multitask_model, _default_device, denoise_and_segment_chunked
# Import from sibling analysis module
from utils.analysis import (
    simulate_multi_particle,
    AnalysisMetrics,
    write_joint_metrics_csv,
    estimate_noise_and_contrast,
)


@dataclass
class TrackSummary:
    track_id: int
    true_radius_nm: float
    estimated_radius_nm: float
    true_diffusion: float
    estimated_diffusion: float
    position_rmse: float


def _select_peak_candidates(row, max_candidates):
    """Legacy peak selection - kept for compatibility."""
    max_candidates = min(max_candidates, len(row))
    if max_candidates <= 0:
        return np.array([], dtype=int), np.array([], dtype=row.dtype)
    partition_index = max(0, len(row) - max_candidates)
    idxs = np.argpartition(row, partition_index)[-max_candidates:]
    order = np.argsort(-row[idxs])
    idxs = idxs[order]
    scores = row[idxs]
    return idxs, scores


def _detect_particles_clustering(row, n_particles, min_separation=5.0):
    """
    Detect particles in a time slice using direct peak detection (no thresholding).
    
    Parameters:
    -----------
    row : np.ndarray
        Intensity values for one time slice (1D array) - should be denoised
    n_particles : int
        Expected number of particles
    min_separation : float
        Minimum separation between particles for clustering
    
    Returns:
    --------
    positions : np.ndarray
        Detected particle positions (centers)
    scores : np.ndarray
        Intensity scores for each detected particle
    """
    if len(row) == 0:
        return np.array([]), np.array([])
    
    # Use direct peak detection - no thresholding needed (data already has high contrast)
    # Clip at zero to ensure no negative values
    row = np.clip(row, 0, None)
    
    # Find local maxima as particle candidates
    from scipy.signal import find_peaks
    
    # Find peaks with minimum separation only (no height threshold)
    # Data already has high contrast, so just use distance
    # For masked data (mostly zeros), use very low or no prominence
    row_std = np.std(row)
    row_max = np.max(row)
    
    # Use very low prominence or none at all for masked data
    # If most pixels are zero, std will be low, so use a fraction of max instead
    prominence_threshold = min(row_std * 0.1, row_max * 0.05) if row_max > 0 else 0
    
    # Find peaks with minimum separation - very permissive prominence
    peaks, properties = find_peaks(
        row,
        distance=int(min_separation),
        prominence=prominence_threshold,  # Very low prominence for masked data
    )
    
    if len(peaks) == 0:
        return np.array([]), np.array([])
    
    # Get peak positions and scores (intensities)
    positions = peaks.astype(float)
    scores = row[peaks]
    
    # Refine positions using subpixel interpolation for better accuracy
    refined_positions = []
    for peak_idx in peaks:
        if peak_idx == 0 or peak_idx == len(row) - 1:
            refined_positions.append(float(peak_idx))
        else:
            # Parabolic interpolation for subpixel accuracy
            y0, y1, y2 = row[peak_idx - 1], row[peak_idx], row[peak_idx + 1]
            denom = (y0 - 2 * y1 + y2)
            if abs(denom) < 1e-10:
                refined_positions.append(float(peak_idx))
            else:
                delta = 0.5 * (y0 - y2) / denom
                refined_positions.append(float(peak_idx) + delta)
    
    positions = np.array(refined_positions)
    
    # If we have more detections than expected particles, use clustering to merge nearby ones
    if len(positions) > n_particles:
        # Use DBSCAN to cluster nearby detections
        positions_2d = positions.reshape(-1, 1)
        
        # eps = min_separation, min_samples = 1 (each detection is a cluster)
        clustering = DBSCAN(eps=min_separation, min_samples=1).fit(positions_2d)
        
        # Merge clusters: take center of mass of each cluster
        unique_labels = np.unique(clustering.labels_)
        merged_positions = []
        merged_scores = []
        
        for label_id in unique_labels:
            if label_id == -1:  # Noise points (shouldn't happen with min_samples=1)
                continue
            
            cluster_mask = clustering.labels_ == label_id
            cluster_positions = positions[cluster_mask]
            cluster_scores = scores[cluster_mask]
            
            # Weighted average position by intensity
            if len(cluster_positions) == 1:
                merged_positions.append(cluster_positions[0])
                merged_scores.append(cluster_scores[0])
            else:
                # Weighted by score
                weights = cluster_scores / (np.sum(cluster_scores) + 1e-10)
                weighted_pos = np.sum(cluster_positions * weights)
                max_score = np.max(cluster_scores)
                merged_positions.append(weighted_pos)
                merged_scores.append(max_score)
        
        positions = np.array(merged_positions)
        scores = np.array(merged_scores)
    
    # Sort by score (highest first)
    if len(positions) > 0:
        sort_idx = np.argsort(-scores)
        positions = positions[sort_idx]
        scores = scores[sort_idx]
    
    return positions, scores


def _predict_positions(prev, prev_prev):
    if prev is None:
        return None
    if prev_prev is None:
        return prev.copy()
    velocity = prev - prev_prev
    return prev + velocity


def _assign_candidates(
    predictions,
    candidate_positions,
    candidate_scores,
    max_jump,
    min_intensity,
    intensity_weight,
    width,
    last_positions=None,
    min_separation=3.0,  # Minimum distance to avoid overlaps
):
    """
    Assign candidate peaks to tracks using greedy assignment.
    Explicitly prevents overlaps by assigning tracks one at a time and removing used candidates.
    """
    n_tracks = len(predictions)
    if len(candidate_positions) == 0:
        # No candidates - maintain last positions or use predictions
        if last_positions is not None:
            fallback = last_positions.copy()
        else:
            fallback = predictions.copy()
        # Ensure minimum separation in fallback
        fallback = _enforce_separation(fallback, min_separation=min_separation, width=width)
        return np.clip(fallback, 0, width - 1)

    # Start with fallback positions
    assigned = predictions.copy()
    if last_positions is not None:
        assigned = last_positions.copy()
    
    # Track which candidates have been used
    used_candidates = set()
    
    # Compute cost for each track-candidate pair
    pred_matrix = predictions[:, None]
    cand_matrix = candidate_positions[None, :]
    dist = np.abs(pred_matrix - cand_matrix)
    
    # Normalize scores for intensity weighting
    if np.ptp(candidate_scores) > 0:
        norm_scores = (candidate_scores - candidate_scores.min()) / (
            np.ptp(candidate_scores) + 1e-6
        )
    else:
        norm_scores = np.zeros_like(candidate_scores)
    
    # Create list of (track_idx, candidate_idx, cost, distance, score) tuples
    assignments = []
    for r in range(n_tracks):
        for c in range(len(candidate_positions)):
            dist_val = dist[r, c]
            score = candidate_scores[c]
            
            # Skip if too far
            if dist_val > max_jump * 1.5:
                continue
            
            # Compute cost (lower is better)
            cost = (dist_val / max(max_jump, 1e-3)) ** 2
            cost -= intensity_weight * norm_scores[c]
            
            # Heavily penalize if beyond max_jump
            if dist_val > max_jump:
                cost += 1000
            
            # Check if assignment would cause overlap with already-assigned tracks
            overlap_penalty = 0
            for assigned_track_idx in range(r):
                if assigned_track_idx < len(assigned):
                    assigned_pos = assigned[assigned_track_idx]
                    cand_pos = candidate_positions[c]
                    if abs(assigned_pos - cand_pos) < min_separation:
                        # Heavy penalty for overlaps
                        overlap_penalty += 10000 * (min_separation - abs(assigned_pos - cand_pos)) / min_separation
            
            cost += overlap_penalty
            
            assignments.append((r, c, cost, dist_val, score))
    
    # Sort by cost (best matches first)
    assignments.sort(key=lambda x: x[2])
    
    # Greedy assignment: assign each track to its best available candidate
    track_assigned = [False] * n_tracks
    assigned_positions = {}  # Track actual assigned positions (candidate positions, not fallbacks)
    
    for track_idx, cand_idx, cost, dist_val, score in assignments:
        if track_assigned[track_idx]:
            continue  # This track already assigned
        
        if cand_idx in used_candidates:
            continue  # This candidate already used
        
        # Check if this assignment would cause overlap with already-assigned tracks
        cand_pos = candidate_positions[cand_idx]
        would_overlap = False
        for other_track_idx in assigned_positions:
            other_pos = assigned_positions[other_track_idx]
            if abs(other_pos - cand_pos) < min_separation:
                would_overlap = True
                break
        
        if would_overlap:
            continue  # Skip this assignment to avoid overlap
        
        # Accept assignment if within jump distance and above intensity threshold
        if dist_val <= max_jump and score >= min_intensity:
            assigned[track_idx] = cand_pos
            assigned_positions[track_idx] = cand_pos
            track_assigned[track_idx] = True
            used_candidates.add(cand_idx)
        # If close enough but low intensity, still assign (might be dim particle)
        elif dist_val <= max_jump * 1.5 and score >= min_intensity * 0.5:
            assigned[track_idx] = cand_pos
            assigned_positions[track_idx] = cand_pos
            track_assigned[track_idx] = True
            used_candidates.add(cand_idx)
    
    # For tracks that didn't get assigned, keep their predictions/last positions
    # But ensure they don't overlap with assigned tracks
    for track_idx in range(n_tracks):
        if not track_assigned[track_idx]:
            # Check if fallback position overlaps with assigned tracks
            fallback_pos = assigned[track_idx]
            for other_track_idx in assigned_positions:
                other_pos = assigned_positions[other_track_idx]
                if abs(fallback_pos - other_pos) < min_separation:
                    # Shift fallback position to avoid overlap
                    if fallback_pos < other_pos:
                        fallback_pos = other_pos - min_separation
                    else:
                        fallback_pos = other_pos + min_separation
                    fallback_pos = np.clip(fallback_pos, 0, width - 1)
            assigned[track_idx] = fallback_pos
    
    return np.clip(assigned, 0, width - 1)


def _enforce_separation(positions, min_separation, width):
    """
    Enforce minimum separation between track positions.
    Only resolves actual overlaps (< min_separation), shifting tracks apart minimally.
    """
    if len(positions) <= 1:
        return positions
    
    positions = positions.copy()
    sorted_indices = np.argsort(positions)
    
    # Iteratively resolve conflicts
    max_iterations = 10
    for iteration in range(max_iterations):
        conflicts = False
        for i in range(len(sorted_indices) - 1):
            idx1 = sorted_indices[i]
            idx2 = sorted_indices[i + 1]
            
            # Only enforce separation if tracks are actually overlapping
            if abs(positions[idx1] - positions[idx2]) < min_separation:
                conflicts = True
                # Shift tracks apart minimally (just enough to meet min_separation)
                mid = (positions[idx1] + positions[idx2]) / 2
                positions[idx1] = mid - min_separation / 2
                positions[idx2] = mid + min_separation / 2
                
                # Clamp to valid range
                positions[idx1] = np.clip(positions[idx1], 0, width - 1)
                positions[idx2] = np.clip(positions[idx2], 0, width - 1)
        
        if not conflicts:
            break
        
        # Re-sort after adjustments
        sorted_indices = np.argsort(positions)
    
    return positions


def _detect_crossing_events(tracks, crossing_threshold=5.0, padding_frames=2):
    """
    Detect when tracks cross (get too close) and mark both tracks as NaN from crossing onwards.
    Once tracks cross, their identity becomes ambiguous, so all subsequent data is excluded.
    
    Parameters:
    -----------
    tracks : np.ndarray
        Track positions, shape (n_particles, time_len)
    crossing_threshold : float
        Distance threshold below which tracks are considered crossing (default: 5.0 pixels)
    padding_frames : int
        Number of frames before crossing to also mark as NaN (default: 2)
    
    Returns:
    --------
    tracks_cleaned : np.ndarray
        Tracks with crossing events and all subsequent frames marked as NaN
    """
    tracks_cleaned = tracks.copy()
    n_particles, time_len = tracks.shape
    
    # Track which track pairs have crossed (to avoid redundant marking)
    crossed_pairs = set()
    
    # Find all crossing events
    for t in range(time_len):
        # Get valid (non-NaN) positions at this time
        valid_positions = []
        valid_indices = []
        for i in range(n_particles):
            if not np.isnan(tracks[i, t]):
                valid_positions.append(tracks[i, t])
                valid_indices.append(i)
        
        # Check all pairs for crossings
        for i_idx, i in enumerate(valid_indices):
            for j_idx, j in enumerate(valid_indices):
                if i >= j:  # Only check each pair once
                    continue
                
                # Skip if this pair already crossed earlier
                pair_key = (min(i, j), max(i, j))
                if pair_key in crossed_pairs:
                    continue
                
                pos_i = valid_positions[i_idx]
                pos_j = valid_positions[j_idx]
                distance = abs(pos_i - pos_j)
                
                if distance < crossing_threshold:
                    # Mark this pair as having crossed
                    crossed_pairs.add(pair_key)
                    
                    # Mark both tracks from padding frames before crossing to end
                    start_frame = max(0, t - padding_frames)
                    end_frame = time_len  # Mark everything after crossing
                    
                    tracks_cleaned[i, start_frame:end_frame] = np.nan
                    tracks_cleaned[j, start_frame:end_frame] = np.nan
    
    return tracks_cleaned


def track_particles(
    kymograph,
    n_particles,
    max_candidates=30,
    max_jump=15,
    smoothing=0.3,
    min_intensity=0.01,
    intensity_weight=0.2,
    detect_crossings=True,
    crossing_threshold=5.0,
    crossing_padding=2,
):
    """
    Track multiple particles through a kymograph.
    Simple, robust approach: find peaks, assign to nearest predicted positions.
    
    Parameters:
    -----------
    detect_crossings : bool
        If True, detect crossing events and mark both tracks as NaN during crossings (default: True)
    crossing_threshold : float
        Distance threshold for detecting crossings (default: 5.0 pixels)
    crossing_padding : int
        Number of frames before/after crossing to also exclude (default: 2)
    """
    time_len, width = kymograph.shape
    tracks = np.full((n_particles, time_len), np.nan)

    prev_positions = None
    prev_prev_positions = None

    for t in range(time_len):
        row = kymograph[t]
        # Use clustering-based detection instead of peak selection
        candidates, scores = _detect_particles_clustering(
            row, n_particles, 
            min_separation=max(5.0, width / (n_particles * 3))  # Adaptive separation
        )
        
        # Fallback to peak selection if clustering finds nothing
        if len(candidates) == 0:
            candidates, scores = _select_peak_candidates(row, max_candidates)
        
        if prev_positions is None:
            # Initialize from first frame: find n_particles DISTINCT, well-separated peaks
            init = np.full(n_particles, np.nan)
            
            if len(candidates) > 0:
                # Get unique candidates (remove duplicates)
                unique_candidates = []
                unique_scores = []
                seen_positions = set()
                for cand_pos, cand_score in zip(candidates, scores):
                    pos_int = int(round(cand_pos))
                    if pos_int not in seen_positions:
                        unique_candidates.append(cand_pos)
                        unique_scores.append(cand_score)
                        seen_positions.add(pos_int)
                
                # Sort by score (highest first)
                sorted_pairs = sorted(zip(unique_candidates, unique_scores), key=lambda x: x[1], reverse=True)
                
                # Greedily select well-separated candidates
                selected = []
                min_sep_init = max(10.0, width / (n_particles * 2))  # Ensure good separation
                
                for cand_pos, cand_score in sorted_pairs:
                    # Check if far enough from already selected
                    too_close = False
                    for sel_pos in selected:
                        if abs(cand_pos - sel_pos) < min_sep_init:
                            too_close = True
                            break
                    
                    if not too_close:
                        selected.append(cand_pos)
                        if len(selected) >= n_particles:
                            break
                
                # Fill remaining slots
                if len(selected) < n_particles:
                    # Use remaining best candidates even if close
                    for cand_pos, cand_score in sorted_pairs:
                        if cand_pos not in selected:
                            selected.append(cand_pos)
                            if len(selected) >= n_particles:
                                break
                
                # Sort by position for consistent track IDs
                selected = np.sort(selected[:n_particles])
                
                # Fill any remaining slots with evenly spaced positions
                if len(selected) < n_particles:
                    # Distribute remaining tracks evenly
                    remaining = n_particles - len(selected)
                    if len(selected) == 0:
                        # No candidates at all - space evenly
                        selected = np.linspace(width / (n_particles + 1), width * n_particles / (n_particles + 1), n_particles)
                    else:
                        # Fill gaps between/around selected positions
                        for i in range(remaining):
                            if len(selected) == 0:
                                new_pos = width / 2
                            elif len(selected) == 1:
                                # Distribute around single position
                                center = selected[0]
                                spacing = width / (remaining + 2)
                                new_pos = center + (i + 1 - (remaining + 1) / 2) * spacing
                            else:
                                # Find largest gap
                                gaps = np.diff(np.concatenate([[0], selected, [width - 1]]))
                                max_idx = np.argmax(gaps)
                                if max_idx == 0:
                                    new_pos = selected[0] / 2
                                elif max_idx == len(gaps) - 1:
                                    new_pos = (selected[-1] + width - 1) / 2
                                else:
                                    new_pos = (selected[max_idx - 1] + selected[max_idx]) / 2
                            selected = np.sort(np.concatenate([selected, [new_pos]]))
                    selected = selected[:n_particles]
                
                init = selected
            else:
                # No candidates found - initialize evenly spaced
                init = np.linspace(width / (n_particles + 1), width * n_particles / (n_particles + 1), n_particles)
            
            # Final check: ensure all positions are distinct and separated
            init = np.array(init)
            init = np.sort(init)  # Sort by position
            
            # Ensure we have exactly n_particles positions
            if len(init) < n_particles:
                # Fill missing positions
                while len(init) < n_particles:
                    if len(init) == 0:
                        init = np.linspace(width / (n_particles + 1), width * n_particles / (n_particles + 1), n_particles)
                        break
                    gaps = np.diff(np.concatenate([[0], init, [width - 1]]))
                    max_idx = np.argmax(gaps)
                    if max_idx == 0:
                        new = init[0] / 2
                    elif max_idx == len(gaps) - 1:
                        new = (init[-1] + width - 1) / 2
                    else:
                        new = (init[max_idx - 1] + init[max_idx]) / 2
                    init = np.sort(np.concatenate([init, [new]]))
                init = init[:n_particles]
            
            # Remove duplicates and ensure separation
            init = np.unique(init)  # Remove exact duplicates
            if len(init) < n_particles:
                # If we lost positions due to duplicates, fill them
                while len(init) < n_particles:
                    gaps = np.diff(np.concatenate([[0], init, [width - 1]]))
                    max_idx = np.argmax(gaps)
                    if max_idx == 0:
                        new = init[0] / 2
                    elif max_idx == len(gaps) - 1:
                        new = (init[-1] + width - 1) / 2
                    else:
                        new = (init[max_idx - 1] + init[max_idx]) / 2
                    init = np.sort(np.concatenate([init, [new]]))
                init = init[:n_particles]
            
            # Ensure minimum separation
            init = _enforce_separation(init, min_separation=5.0, width=width)
            
            # Final guarantee: if somehow we still don't have enough, space evenly
            if len(init) != n_particles:
                init = np.linspace(width / (n_particles + 1), width * n_particles / (n_particles + 1), n_particles)
            
            tracks[:, t] = init
        else:
            # Predict next positions based on velocity
            predictions = _predict_positions(prev_positions, prev_prev_positions)
            if predictions is None or np.any(np.isnan(predictions)):
                predictions = prev_positions.copy()
                # Fill any NaN predictions with last known position
                nan_mask = np.isnan(predictions)
                if np.any(nan_mask):
                    predictions[nan_mask] = prev_positions[nan_mask]
            
            # Assign candidates to tracks
            assigned = _assign_candidates(
                predictions,
                candidates,
                scores,
                max_jump,
                min_intensity,
                intensity_weight,
                width,
                last_positions=prev_positions,  # Use last positions as fallback
                min_separation=3.0,  # Minimum separation to avoid overlaps
            )
            
            # Apply smoothing (only if we have valid previous positions)
            if smoothing > 0 and prev_positions is not None:
                valid_mask = ~np.isnan(prev_positions)
                assigned[valid_mask] = (
                    smoothing * assigned[valid_mask] + 
                    (1 - smoothing) * prev_positions[valid_mask]
                )
            
            # Enforce separation if smoothing caused overlaps (use same min_separation as assignment)
            assigned = _enforce_separation(assigned, min_separation=3.0, width=width)
            
            tracks[:, t] = np.clip(assigned, 0, width - 1)

        prev_prev_positions = prev_positions
        prev_positions = tracks[:, t].copy()

    # Detect and mark crossing events
    if detect_crossings and n_particles > 1:
        tracks = _detect_crossing_events(tracks, crossing_threshold, crossing_padding)

    return tracks


def summarize_multi_particle_analysis(
    simulation,
    denoised_kymograph,
    estimated_tracks,
    method_label="U-Net Denoised",
    figure_subdir="multi_unet",
    noisy_kymograph=None,
):
    """
    Compute metrics and create diagnostic plot for multi-particle analysis.
    Returns a list of AnalysisMetrics (one per track).
    """
    n_particles = len(simulation.radii_nm)
    metrics_list = []
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Use percentile-based ranges to be robust to outliers (consistent with utils.py)
    vmin_noisy = np.percentile(simulation.kymograph_noisy, 1)
    vmax_noisy = np.percentile(simulation.kymograph_noisy, 99)
    vmin_denoised = np.percentile(denoised_kymograph, 1)
    vmax_denoised = np.percentile(denoised_kymograph, 99)
    
    # Top left: Noisy kymograph with true paths
    axes[0, 0].imshow(
        simulation.kymograph_noisy.T,
        aspect="auto",
        origin="lower",
        extent=[0, simulation.n_t, 0, simulation.n_x],
        vmin=vmin_noisy,
        vmax=vmax_noisy,
        cmap="gray",
    )
    axes[0, 0].set_title("Noisy Kymograph")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Position")
    # Use distinct colors for each track
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    for idx in range(n_particles):
        color = colors[idx % len(colors)]
        axes[0, 0].plot(simulation.true_paths[idx], lw=0.8, alpha=0.7, 
                       label=f"Track {idx+1}", color=color)
    
    # Top right: Denoised kymograph with estimated tracks
    axes[0, 1].imshow(
        denoised_kymograph.T,
        aspect="auto",
        origin="lower",
        extent=[0, simulation.n_t, 0, simulation.n_x],
        vmin=vmin_denoised,
        vmax=vmax_denoised,
        cmap="gray",
    )
    axes[0, 1].set_title(f"{method_label} Kymograph")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Position")
    for idx in range(n_particles):
        color = colors[idx % len(colors)]
        axes[0, 1].plot(estimated_tracks[idx], lw=1.0, alpha=0.8, 
                       label=f"Est. {idx+1}", color=color)
    
    # Bottom left: Ground truth paths
    for idx in range(n_particles):
        color = colors[idx % len(colors)]
        axes[1, 0].plot(simulation.true_paths[idx], linestyle="--", alpha=0.7, 
                       color=color, label=f"True {idx+1}")
    axes[1, 0].set_xlim(0, simulation.n_t)
    axes[1, 0].set_ylim(0, simulation.n_x)
    axes[1, 0].set_title("True Paths")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Position")
    
    # Bottom right: Estimated paths
    for idx in range(n_particles):
        color = colors[idx % len(colors)]
        axes[1, 1].plot(estimated_tracks[idx], lw=1.0, color=color, label=f"Est. {idx+1}")
    axes[1, 1].set_xlim(0, simulation.n_t)
    axes[1, 1].set_ylim(0, simulation.n_x)
    axes[1, 1].set_title(f"{method_label} Paths")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Position")
    
    # Create title with summary info
    title_parts = []
    for idx in range(n_particles):
        true_diff = simulation.diffusions[idx]
        est_diff = estimate_diffusion_msd_fit(
            estimated_tracks[idx],
            dx=simulation.x_step,
            dt=simulation.t_step,
        )
        title_parts.append(f"Track{idx+1}: D={true_diff:.3f}/{est_diff:.3f}")
    fig.suptitle(f"{method_label} | " + " | ".join(title_parts), fontsize=10)
    
    # Save figure
    figure_dir = os.path.join("figures", figure_subdir)
    os.makedirs(figure_dir, exist_ok=True)
    
    # Create filename from parameters
    radii_str = "_".join([f"{r:.1f}" for r in simulation.radii_nm])
    contrasts_str = "_".join([f"{c:.2f}" for c in simulation.contrasts])
    fig_filename = os.path.join(
        figure_dir,
        f"multi_particle_r_{radii_str}_c_{contrasts_str}_n_{simulation.noise_level:.2f}.png",
    )
    fig.savefig(fig_filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Estimate noise and contrast from noisy kymograph (if provided)
    noise_estimate_global = None
    contrast_estimate_global = None
    if noisy_kymograph is not None:
        noise_estimate_global, contrast_estimate_global = estimate_noise_and_contrast(noisy_kymograph)
    
    # Compute metrics for each track
    for idx in range(n_particles):
        true_path = simulation.true_paths[idx]
        estimated_path = estimated_tracks[idx]
        
        # Compute diffusion coefficients
        true_diffusion = simulation.diffusions[idx]
        estimated_diffusion = estimate_diffusion_msd_fit(
            estimated_path,
            dx=simulation.x_step,
            dt=simulation.t_step,
        )
        
        # Compute radii
        true_radius = simulation.radii_nm[idx]
        estimated_radius = get_particle_radius(estimated_diffusion)
        
        # Estimate contrast for this specific track from denoised kymograph
        # Extract intensity along the track
        contrast_estimate_track = None
        if len(estimated_path) > 0:
            valid_mask = ~np.isnan(estimated_path)
            if np.sum(valid_mask) > 10:  # Need enough valid points
                track_intensities = []
                for t in range(len(estimated_path)):
                    if valid_mask[t]:
                        pos = int(np.clip(estimated_path[t], 0, denoised_kymograph.shape[1] - 1))
                        track_intensities.append(denoised_kymograph[t, pos])
                
                if len(track_intensities) > 0:
                    # Contrast = peak intensity - background
                    peak_intensity = np.percentile(track_intensities, 90)
                    background = np.median(track_intensities)
                    contrast_estimate_track = max(peak_intensity - background, 0.0)
        
        # Use track-specific contrast if available, otherwise global
        contrast_estimate = contrast_estimate_track if contrast_estimate_track is not None else contrast_estimate_global
        
        metrics = AnalysisMetrics(
            method_label=method_label,
            particle_radius_nm=true_radius,
            contrast=simulation.contrasts[idx],
            noise_level=simulation.noise_level,
            diffusion_true=true_diffusion,
            diffusion_noisy=None,
            diffusion_processed=estimated_diffusion,
            radius_true=true_radius,
            radius_noisy=None,
            radius_processed=estimated_radius,
            noise_estimate=noise_estimate_global,
            contrast_estimate=contrast_estimate,
            figure_path=fig_filename,
        )
        metrics_list.append(metrics)
        
        # Count valid (non-NaN) frames
        valid_frames = np.sum(~np.isnan(estimated_path))
        total_frames = len(estimated_path)
        excluded_frames = total_frames - valid_frames
        
        # Build output string
        output_parts = [
            f"[{method_label}] Track {idx+1}:",
            f"r_true={true_radius:.2f} nm, r_est={estimated_radius:.2f} nm",
            f"D_true={true_diffusion:.3f}, D_est={estimated_diffusion:.3f} µm²/ms",
        ]
        
        if contrast_estimate is not None:
            output_parts.append(f"c_true={simulation.contrasts[idx]:.2f}, c_est={contrast_estimate:.2f}")
        
        if noise_estimate_global is not None:
            output_parts.append(f"n_true={simulation.noise_level:.2f}, n_est={noise_estimate_global:.2f}")
        
        output_parts.append(f"Valid frames: {valid_frames}/{total_frames}")
        if excluded_frames > 0:
            output_parts.append(f"(excluded {excluded_frames} due to crossings)")
        
        print(" | ".join(output_parts))
    
    print(f"[{method_label}] Figure saved to {fig_filename}")
    return metrics_list


def analyze_multi_particle(
    radii_nm,
    contrasts,
    noise_level,
    chunk_length=512,
    overlap=64,
    model_path="models/tiny_unet_denoiser.pth",
    max_candidates=30,
    max_jump=8,
):
    simulation = simulate_multi_particle(
        radii_nm=radii_nm,
        contrasts=contrasts,
        noise_level=noise_level,
    )

    device = _default_device()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    # Load multi-task model
    model = load_multitask_model(model_path, device=device)

    denoised, _ = denoise_and_segment_chunked(
        model,
        simulation.kymograph_noisy,
        device=device,
        chunk_size=chunk_length,
        overlap=overlap,
    )

    estimated_tracks = track_particles(
        denoised,
        n_particles=len(radii_nm),
        max_candidates=max_candidates,
        max_jump=max_jump,
        detect_crossings=True,  # Enable crossing detection
        crossing_threshold=5.0,  # Mark crossings when tracks < 5 pixels apart
        crossing_padding=2,  # Exclude 2 frames before/after crossing
    )

    # Generate comprehensive report (like single_particle_unet.py)
    metrics_list = summarize_multi_particle_analysis(
        simulation,
        denoised,
        estimated_tracks,
        method_label="U-Net Denoised",
        figure_subdir="multi_unet",
        noisy_kymograph=simulation.kymograph_noisy,  # Pass noisy kymograph for noise/contrast estimation
    )

    return metrics_list


def run_parameter_grid(
    particle_configs=None,
    noise_levels=[0.1, 0.3, 0.5],
    csv_path="metrics/multi_particle_unet.csv",
):
    """
    Run multi-particle analysis over a grid of parameters.
    
    Parameters:
    -----------
    particle_configs : list of tuples, optional
        List of (radii_nm, contrasts) tuples. If None, uses default examples.
        Example: [([2.5, 5.0], [0.8, 0.6]), ([5.0, 10.0, 8.0], [0.7, 0.5, 1.0])]
    noise_levels : list
        Noise levels to test
    csv_path : str
        Path to save CSV metrics
    """
    if particle_configs is None:
        # Default: test 2-particle and 3-particle scenarios
        particle_configs = [
            ([2.5, 5.0], [0.8, 0.6]),  # 2 particles
            ([5.0, 10.0, 8.0], [0.7, 0.5, 1.0]),  # 3 particles
            ([2.5, 5.0, 10.0], [0.9, 0.7, 0.6]),  # 3 particles, different config
        ]
    
    metrics_rows = []
    for radii_nm, contrasts in particle_configs:
        for n in noise_levels:
            print(f"\n[Multi U-Net] Running {len(radii_nm)} particles: r={radii_nm}, c={contrasts}, n={n}")
            metrics_rows.extend(analyze_multi_particle(
                radii_nm=radii_nm,
                contrasts=contrasts,
                noise_level=n,
            ))
    
    written_csv = write_joint_metrics_csv(metrics_rows, csv_path)
    print(f"\n[Multi U-Net] Completed {len(metrics_rows)} track analyses; aggregated metrics -> {written_csv}")


if __name__ == "__main__":
    # Run parameter grid (like single_particle_unet.py)
    run_parameter_grid()
