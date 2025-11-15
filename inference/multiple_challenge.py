"""
Multi-Particle Challenge Inference Script

Processes multi-particle kymograph files from the Hackathon folder:
- Multi-particle files: kymograph_noisy_multiple_particles_*.npy

For each file, this script:
1. Loads the noisy kymograph
2. Denoises and segments using the trained multi-task U-Net model
3. Tracks multiple particles and estimates parameters
4. Generates diagnostic plots and saves results
5. Saves metrics to CSV

Usage:
    python inference/multiple_challenge.py
    # Or with custom paths:
    python inference/multiple_challenge.py --hackathon_dir Hackathon --model_path models/multitask_unet.pth
"""

import os
import sys
import glob
import csv
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Tuple
from scipy.signal import find_peaks

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from train.multitask_model import load_multitask_model, denoise_and_segment_chunked, _default_device
from utils import (
    AnalysisMetrics,
    estimate_noise_and_contrast,
)
from utils.tracking import track_particles
from utils.helpers import estimate_diffusion_msd_fit, get_particle_radius


def write_challenge_csv(metrics_list: List[AnalysisMetrics], csv_path: str) -> str:
    """
    Write challenge CSV with only contrast and diffusion.
    
    Parameters:
    -----------
    metrics_list : List[AnalysisMetrics]
        List of metrics to write
    csv_path : str
        Path to save CSV
    
    Returns:
    --------
    csv_path : str
        Path that was written
    """
    if not metrics_list:
        return csv_path
    
    dir_name = os.path.dirname(csv_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    fieldnames = ["contrast", "diffusion"]
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in metrics_list:
            writer.writerow({
                "contrast": metrics.contrast_estimate,
                "diffusion": metrics.diffusion_processed,
            })
    return csv_path


def apply_segmentation_mask(
    denoised: np.ndarray,
    segmentation_labels: np.ndarray,
    dilation_size: Tuple[int, int] = (3, 3),
    background_weight: float = 0.3,
    gaussian_sigma: float = 1.0,
) -> np.ndarray:
    """
    Apply segmentation mask to denoised kymograph to suppress background noise.
    
    Parameters:
    -----------
    denoised : np.ndarray
        Denoised kymograph
    segmentation_labels : np.ndarray
        Binary mask from segmentation head: 0=background, 1=particle (any track)
    dilation_size : Tuple[int, int]
        Size of dilation kernel (time, space) to expand mask (default: (3, 3))
    background_weight : float
        Weight for background regions (0-1), lower = more suppression (default: 0.3)
    gaussian_sigma : float
        Sigma for Gaussian smoothing of mask edges (default: 1.0)
    
    Returns:
    --------
    denoised_masked : np.ndarray
        Denoised kymograph with background suppressed
    """
    from scipy.ndimage import binary_dilation, binary_opening, gaussian_filter
    
    # segmentation_labels is already a binary mask (0=background, 1=particle)
    # Threshold at 0.7 to be more conservative (higher = stricter particle detection)
    mask_binary = (segmentation_labels > 0.7).astype(np.float32)
    
    # Apply opening (erosion followed by dilation) to remove small noise
    opening_kernel = np.ones((3, 3), dtype=bool)  # Small kernel for opening
    mask_opened = binary_opening(mask_binary, structure=opening_kernel).astype(np.float32)
    
    # Dilate mask to make it slightly bigger
    dilation_kernel = np.ones(dilation_size, dtype=bool)
    mask_dilated = binary_dilation(mask_opened, structure=dilation_kernel).astype(np.float32)
    
    # Smooth transition at mask edges to avoid artifacts
    mask_soft = gaussian_filter(mask_dilated, sigma=gaussian_sigma)
    
    # Apply mask: binary multiplication (0/1)
    # Particle regions (mask=1): keep full denoised value
    # Background regions (mask=0): set to zero
    denoised_masked = denoised * mask_soft
    
    return denoised_masked


def estimate_n_particles(kymograph: np.ndarray, denoised: np.ndarray) -> int:
    """
    Estimate the number of particles in a kymograph.
    
    Uses peak detection on the denoised kymograph to count distinct tracks.
    
    Parameters:
    -----------
    kymograph : np.ndarray
        Noisy kymograph
    denoised : np.ndarray
        Denoised kymograph
    
    Returns:
    --------
    n_particles : int
        Estimated number of particles (1-3)
    """
    # Sample a few time slices and count distinct peaks
    time_len, width = denoised.shape
    
    # Sample middle frames (more stable than edges)
    sample_frames = [time_len // 4, time_len // 2, 3 * time_len // 4]
    peak_counts = []
    
    for t in sample_frames:
        row = denoised[t]
        # Find peaks (local maxima)
        # Normalize row for peak detection
        row_norm = (row - row.min()) / (row.max() - row.min() + 1e-8)
        
        # Find peaks with minimum height and distance
        peaks, _ = find_peaks(
            row_norm,
            height=0.3,  # At least 30% of max
            distance=max(10, width // 10),  # Minimum separation
        )
        peak_counts.append(len(peaks))
    
    # Use median count, clamp to reasonable range
    n_est = int(np.median(peak_counts))
    n_particles = max(1, min(3, n_est))  # Clamp to 1-3
    
    return n_particles


def process_multi_particle_file(
    filepath: str,
    model,
    device: str,
    n_particles: Optional[int] = None,
    output_dir: str = "challenge_results",
) -> List[AnalysisMetrics]:
    """
    Process a multi-particle kymograph file.
    
    Parameters:
    -----------
    filepath : str
        Path to the .npy file
    model : MultiTaskUNet
        Trained multi-task model (denoising + segmentation)
    device : str
        Device to run model on
    n_particles : int, optional
        Number of particles (if None, will be estimated)
    output_dir : str
        Directory to save results
    
    Returns:
    --------
    metrics_list : List[AnalysisMetrics]
        Analysis results (one per particle)
    """
    print(f"\nProcessing multi-particle file: {filepath}")
    
    # Load kymograph
    kymograph_noisy = np.load(filepath).astype(np.float32)
    
    # Challenge data format: background is gray (~0), particles are white (bright)
    # Model expects: background ~0 (dark), particles bright
    # Strategy: subtract background level, then normalize so background maps to 0
    
    # Estimate background level (use low percentile to be robust to outliers)
    background_level = np.percentile(kymograph_noisy, 10)  # 10th percentile as background
    
    # Subtract background to make it ~0
    kymograph_bg_subtracted = kymograph_noisy - background_level
    
    # Normalize to [0, 1] range: map background (now ~0) to 0, signal to 1
    # Use signal level (99th percentile) as the upper bound
    signal_level = np.percentile(kymograph_bg_subtracted, 99)  # 99th percentile as signal
    
    if signal_level > 0:
        # Normalize: background (~0) -> 0, signal -> 1
        kymograph_noisy_norm = np.clip(kymograph_bg_subtracted / signal_level, 0.0, 1.0)
    else:
        # Fallback if no signal
        kymograph_noisy_norm = np.clip(kymograph_bg_subtracted, 0.0, 1.0)
    
    # Store normalization parameters for denormalization
    kymograph_min = 0.0  # Background maps to 0
    kymograph_max = signal_level  # Signal maps to 1
    
    print(f"  Background level: {background_level:.4f}, Signal level: {signal_level:.4f}")
    print(f"  Normalized range: [{kymograph_noisy_norm.min():.4f}, {kymograph_noisy_norm.max():.4f}]")
    print(f"  Normalized bg percentile (10th): {np.percentile(kymograph_noisy_norm, 10):.4f}, signal percentile (99th): {np.percentile(kymograph_noisy_norm, 99):.4f}")
    
    # Denoise and segment using multi-task model
    denoised_norm, segmentation_labels_norm = denoise_and_segment_chunked(
        model, kymograph_noisy_norm, device=device, chunk_size=512, overlap=64
    )
    
    # Set edge regions to black (background) at all times to handle edge artifacts
    # Positions 0-11 and 500-511 (if width >= 512) are set to 0
    time_len, width = segmentation_labels_norm.shape
    segmentation_labels_norm[:, 0:12] = 0.0  # Left edge: positions 0-11
    if width >= 512:
        segmentation_labels_norm[:, 500:512] = 0.0  # Right edge: positions 500-511
    elif width > 500:
        segmentation_labels_norm[:, 500:] = 0.0  # Right edge: positions 500 to end
    
    # Apply segmentation mask to suppress background noise
    # segmentation_labels_norm is now a binary mask (0=background, 1=particle)
    from scipy.ndimage import binary_opening, binary_dilation
    
    mask_binary = (segmentation_labels_norm > 0.7).astype(np.float32)  # Threshold at 0.7 for binary (higher = stricter)
    
    # Apply opening (erosion + dilation) to remove small noise
    opening_kernel = np.ones((3, 3), dtype=bool)
    mask_opened = binary_opening(mask_binary, structure=opening_kernel).astype(np.float32)
    
    # Apply opening to segmentation_labels_norm for use in apply_segmentation_mask
    segmentation_labels_norm_opened = mask_opened.astype(np.float32)
    
    mask_dilated = (apply_segmentation_mask(
        np.ones_like(segmentation_labels_norm_opened), segmentation_labels_norm_opened, dilation_size=(3, 3)
    ) > 0.7).astype(np.float32)
    
    print(f"  Segmentation mask: {np.sum(mask_binary > 0):.0f} pixels ({100*np.mean(mask_binary):.1f}% of image)")
    print(f"  After opening: {np.sum(mask_opened > 0):.0f} pixels ({100*np.mean(mask_opened):.1f}% of image)")
    print(f"  Mask value range: [{segmentation_labels_norm.min():.3f}, {segmentation_labels_norm.max():.3f}]")
    print(f"  After dilation: {np.sum(mask_dilated > 0):.0f} pixels ({100*np.mean(mask_dilated):.1f}% of image)")
    
    # For tracking, create a version BEFORE heavy background suppression
    # Save the denoised before masking for tracking (need full intensity for Otsu/peak detection)
    denoised_norm_before_mask = denoised_norm.copy()
    
    # Apply segmentation mask to suppress background noise (for visualization/analysis)
    denoised_norm = apply_segmentation_mask(
        denoised_norm, segmentation_labels_norm_opened, dilation_size=(3, 3), background_weight=0.3
    )
    
    # Check background noise level (std of 0-80 percentile regions, outside tracks)
    # Only apply median filter if background noise is high
    background_mask = denoised_norm <= np.percentile(denoised_norm, 80)
    background_std = np.std(denoised_norm[background_mask]) if np.any(background_mask) else 0.0
    median_filter_threshold = 0.01  # Apply if background std > 0.01
    
    if background_std > median_filter_threshold:
        from scipy.signal import medfilt
        print(f"  Applying median filter (background std={background_std:.4f} > {median_filter_threshold:.4f})...")
        # Use larger kernel for sparse noise: (9, 5) = 9 time steps, 5 spatial pixels
        denoised_norm = medfilt(denoised_norm, kernel_size=(9, 5))
    else:
        print(f"  Skipping median filter (background std={background_std:.4f} <= {median_filter_threshold:.4f})")
    
    # Diagnostic: check denoised output
    noise_input, contrast_input = estimate_noise_and_contrast(kymograph_noisy_norm)
    noise_denoised, contrast_denoised = estimate_noise_and_contrast(denoised_norm)
    contrast_improvement = contrast_denoised / max(contrast_input, 1e-6)
    noise_reduction = noise_denoised / max(noise_input, 1e-6)
    print(f"  Noise: {noise_input:.4f} -> {noise_denoised:.4f} ({noise_reduction:.2f}x reduction)")
    print(f"  Contrast: {contrast_input:.4f} -> {contrast_denoised:.4f} ({contrast_improvement:.2f}x improvement)")
    
    # Denormalize for analysis (convert back to original scale)
    denoised = denoised_norm * (kymograph_max - kymograph_min) + kymograph_min + background_level
    
    # For tracking, use denoised AFTER applying segmentation mask
    # Tracking on masked denoised data (background suppressed, particles preserved)
    denoised_for_tracking = denoised_norm.copy()  # Already has segmentation mask applied, normalized [0,1]
    
    # Estimate number of particles if not provided
    # Use masked denoised for peak detection
    if n_particles is None:
        # Need to denormalize temporarily for estimate_n_particles (it expects original scale)
        denoised_for_estimation = denoised_norm * (kymograph_max - kymograph_min) + kymograph_min + background_level
        n_particles = estimate_n_particles(kymograph_noisy, denoised_for_estimation)
        print(f"  Estimated {n_particles} particles")
    
    # Debug: check denoised_for_tracking stats (should be in [0,1] range, masked)
    print(f"  Tracking input stats: min={denoised_for_tracking.min():.4f}, max={denoised_for_tracking.max():.4f}, mean={denoised_for_tracking.mean():.4f}, std={denoised_for_tracking.std():.4f}")
    print(f"  Tracking input non-zero pixels: {np.sum(denoised_for_tracking > 0.01):.0f} ({100*np.mean(denoised_for_tracking > 0.01):.1f}% of image)")
    
    # Debug: sample a few frames to see if particles are detectable
    sample_frames = [0, time_len // 4, time_len // 2, 3 * time_len // 4]
    for t in sample_frames[:3]:
        row = denoised_for_tracking[t]
        row_max = np.max(row)
        row_mean = np.mean(row)
        row_std = np.std(row)
        non_zero = np.sum(row > 0.01)
        peaks, _ = find_peaks(row, distance=10, prominence=min(row_std * 0.1, row_max * 0.05) if row_max > 0 else 0)
        print(f"  Frame {t}: max={row_max:.4f}, mean={row_mean:.4f}, std={row_std:.4f}, non-zero={non_zero}, peaks={len(peaks)} at {peaks[:5] if len(peaks) > 0 else 'none'}")
    
    # Track particles on masked denoised data [0,1] (after segmentation mask applied)
    # Masked data has background suppressed, making particle detection easier
    estimated_tracks = track_particles(
        denoised_for_tracking,
        n_particles=n_particles,
        max_candidates=30,
        max_jump=15,
        min_intensity=0.0,  # No threshold needed - masked denoised data has high contrast
        detect_crossings=True,
        crossing_threshold=5.0,
        crossing_padding=2,
    )
    
    # Debug: check what tracks were found
    print(f"  Track results:")
    for track_id in range(n_particles):
        track = estimated_tracks[track_id]
        valid_mask = ~np.isnan(track)
        if np.sum(valid_mask) > 0:
            track_min = np.min(track[valid_mask])
            track_max = np.max(track[valid_mask])
            track_mean = np.mean(track[valid_mask])
            track_std = np.std(track[valid_mask])
            print(f"    Track {track_id + 1}: valid={np.sum(valid_mask)}/{len(track)}, pos_range=[{track_min:.1f}, {track_max:.1f}], mean={track_mean:.1f}, std={track_std:.4f}")
        else:
            print(f"    Track {track_id + 1}: all NaN")
    
    # Check if faint tracks were missed: count valid tracks
    valid_tracks = np.sum([np.sum(~np.isnan(track)) > len(track) * 0.1 for track in estimated_tracks])
    
    # If we're missing tracks and denoising improved contrast, try:
    # Rerun denoising if noise is still high (no need for intensity threshold retry - masked data is high contrast)
    if valid_tracks < n_particles and contrast_improvement > 1.2:
        print(f"  Warning: Only {valid_tracks}/{n_particles} tracks detected")
        
        # If still missing tracks and noise is high, consider another denoising pass
        max_iterations = 2
        if valid_tracks < n_particles and noise_denoised > 0.05:
            print(f"    Still missing tracks, running additional denoising pass...")
            # Renormalize for another pass
            denoised_bg = np.percentile(denoised_norm, 10)
            denoised_bg_sub = denoised_norm - denoised_bg
            denoised_sig = np.percentile(denoised_bg_sub, 99)
            if denoised_sig > 0:
                denoised_norm_2 = np.clip(denoised_bg_sub / denoised_sig, 0.0, 1.0)
            else:
                denoised_norm_2 = np.clip(denoised_bg_sub, 0.0, 1.0)
            
            denoised_norm, segmentation_labels_retry = denoise_and_segment_chunked(
                model, denoised_norm_2, device=device, chunk_size=512, overlap=64
            )
            
            # Set edge regions to black (background) at all times to handle edge artifacts
            time_len_retry, width_retry = segmentation_labels_retry.shape
            segmentation_labels_retry[:, 0:12] = 0.0  # Left edge: positions 0-11
            if width_retry >= 512:
                segmentation_labels_retry[:, 500:512] = 0.0  # Right edge: positions 500-511
            elif width_retry > 500:
                segmentation_labels_retry[:, 500:] = 0.0  # Right edge: positions 500 to end
            
            # Save before masking for tracking
            denoised_norm_retry_before_mask = denoised_norm.copy()
            
            # Apply segmentation mask to suppress background noise
            denoised_norm = apply_segmentation_mask(
                denoised_norm, segmentation_labels_retry, dilation_size=(3, 3), background_weight=0.3
            )
            denoised = denoised_norm * (kymograph_max - kymograph_min) + kymograph_min + background_level
            
            # Update tracking version - use masked denoised [0,1] for retry (after segmentation mask)
            denoised_for_tracking = denoised_norm.copy()  # Already has segmentation mask applied, normalized [0,1]
            
            # Retry tracking on masked denoised (after segmentation mask applied)
            estimated_tracks = track_particles(
                denoised_for_tracking,
                n_particles=n_particles,
                max_candidates=30,
                max_jump=15,
                min_intensity=0.0,  # No threshold needed - masked denoised data has high contrast
                detect_crossings=True,
                crossing_threshold=5.0,
                crossing_padding=2,
            )
            valid_tracks = np.sum([np.sum(~np.isnan(track)) > len(track) * 0.1 for track in estimated_tracks])
            print(f"    After additional denoising: {valid_tracks}/{n_particles} tracks detected")
    
    # Estimate noise and contrast (global)
    noise_estimate_global, contrast_estimate_global = estimate_noise_and_contrast(
        kymograph_noisy
    )
    
    # Analyze each track
    metrics_list = []
    for track_id in range(n_particles):
        track = estimated_tracks[track_id]
        
        # Skip if track is all NaN
        if np.all(np.isnan(track)):
            print(f"  ⚠ Track {track_id + 1}: All NaN, skipping")
            continue
        
        # Debug: print track statistics
        valid_mask = ~np.isnan(track)
        if np.sum(valid_mask) > 0:
            track_min = np.min(track[valid_mask])
            track_max = np.max(track[valid_mask])
            track_mean = np.mean(track[valid_mask])
            track_std = np.std(track[valid_mask])
            print(f"  Track {track_id + 1} stats: min={track_min:.2f}, max={track_max:.2f}, mean={track_mean:.2f}, std={track_std:.4f}")
        
        # Estimate diffusion coefficient with error handling
        valid_mask = ~np.isnan(track)
        if np.sum(valid_mask) < 10:
            print(f"    ⚠ Too few valid points ({np.sum(valid_mask)}), skipping diffusion estimation")
            diffusion_processed = np.nan
            radius_processed = np.nan
        elif np.std(track[valid_mask]) < 0.1:
            print(f"    ⚠ Track has no variation (std={np.std(track[valid_mask]):.4f}), skipping diffusion estimation")
            print(f"    ⚠ Track values: first 10 = {track[:10]}, last 10 = {track[-10:]}")
            diffusion_processed = np.nan
            radius_processed = np.nan
        else:
            try:
                diffusion_processed = estimate_diffusion_msd_fit(track)
                if np.isnan(diffusion_processed) or diffusion_processed <= 0:
                    print(f"    ⚠ Invalid diffusion estimate ({diffusion_processed}), using NaN")
                    diffusion_processed = np.nan
                    radius_processed = np.nan
                else:
                    radius_processed = get_particle_radius(diffusion_processed)
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"    ⚠ Diffusion estimation failed ({type(e).__name__}), using NaN")
                diffusion_processed = np.nan
                radius_processed = np.nan
        
        # Estimate per-track contrast from denoised kymograph
        valid_mask = ~np.isnan(track)
        if np.sum(valid_mask) > 10:
            track_positions = track[valid_mask].astype(int)
            track_positions = np.clip(track_positions, 0, denoised.shape[1] - 1)
            
            intensities = []
            for t in range(len(track)):
                if not np.isnan(track[t]):
                    pos = int(np.clip(track[t], 0, denoised.shape[1] - 1))
                    intensities.append(denoised[t, pos])
            
            if len(intensities) > 0:
                peak_intensity = np.percentile(intensities, 90)
                background = np.median(denoised)
                contrast_estimate_track = peak_intensity - background
            else:
                contrast_estimate_track = contrast_estimate_global
        else:
            contrast_estimate_track = contrast_estimate_global
        
        # Create metrics
        # For challenge data, use estimates for contrast/noise_level (main fields)
        metrics = AnalysisMetrics(
            method_label=f"Challenge Multi Track {track_id + 1}",
            particle_radius_nm=radius_processed,  # Estimated radius
            diffusion_true=np.nan,  # True ground truth unknown
            diffusion_processed=diffusion_processed,  # Estimated diffusion
            radius_true=np.nan,  # True ground truth unknown
            radius_processed=radius_processed,  # Estimated radius
            contrast=contrast_estimate_track,  # Use estimate for main contrast field
            noise_level=noise_estimate_global,  # Use estimate for main noise_level field
            contrast_estimate=contrast_estimate_track,  # Also in estimate field
            noise_estimate=noise_estimate_global,  # Also in estimate field
            figure_path=f"{output_dir}/multi_particle_{Path(filepath).stem}_track{track_id + 1}.png",
        )
        metrics_list.append(metrics)
        
        print(f"  Track {track_id + 1}:")
        print(f"    ✓ Diffusion: {diffusion_processed:.4f} μm²/s")
        print(f"    ✓ Radius: {radius_processed:.2f} nm")
        print(f"    ✓ Contrast: {contrast_estimate_track:.3f}")
        valid_frames = np.sum(~np.isnan(track))
        print(f"    ✓ Valid frames: {valid_frames}/{len(track)}")
    
    # Create comprehensive visualization
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Use percentile-based ranges to be robust to outliers
    vmin_noisy = np.percentile(kymograph_noisy, 1)
    vmax_noisy = np.percentile(kymograph_noisy, 99)
    vmin_denoised = np.percentile(denoised, 1)
    vmax_denoised = np.percentile(denoised, 99)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    
    # Set same limits as trajectory plot
    time_len, width = kymograph_noisy.shape
    
    # Noisy input
    axes[0, 0].imshow(
        kymograph_noisy.T,
        aspect="auto",
        origin="lower",
        extent=[0, time_len, 0, width],
        vmin=vmin_noisy,
        vmax=vmax_noisy,
        cmap="gray",
    )
    axes[0, 0].set_title("Noisy Input")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Position")
    
    # Denoised with tracks (use denoised range for better contrast)
    axes[0, 1].imshow(
        denoised.T,
        aspect="auto",
        origin="lower",
        extent=[0, time_len, 0, width],
        vmin=vmin_denoised,
        vmax=vmax_denoised,
        cmap="gray",
    )
    for track_id in range(n_particles):
        track = estimated_tracks[track_id]
        valid_mask = ~np.isnan(track)
        if np.sum(valid_mask) > 0:
            color = colors[track_id % len(colors)]
            axes[0, 1].plot(
                track, color=color, lw=1.0, alpha=0.8, label=f"Track {track_id + 1}"
            )
    axes[0, 1].set_title("Denoised & Tracked")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Position")
    axes[0, 1].legend(loc='upper right', fontsize=8)
    
    # Bottom left: Segmentation mask (after opening)
    axes[1, 0].imshow(
        segmentation_labels_norm_opened.T,
        aspect="auto",
        origin="lower",
        extent=[0, time_len, 0, width],
        vmin=0.0,
        vmax=1.0,
        cmap="gray",
    )
    axes[1, 0].set_title("Segmentation Mask\n(opened, binary: 0=bg, 1=particle)")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Position")
    
    # Trajectories plot (moved to bottom right, overlay on summary)
    # Set same limits as kymograph plots (already defined above)
    for track_id in range(n_particles):
        track = estimated_tracks[track_id]
        valid_mask = ~np.isnan(track)
        if np.sum(valid_mask) > 0:
            color = colors[track_id % len(colors)]
            axes[1, 1].plot(
                track, color=color, lw=1.0, alpha=0.7, label=f"Track {track_id + 1}"
            )
    axes[1, 1].set_xlim(0, time_len)
    axes[1, 1].set_ylim(0, width)
    axes[1, 1].set_title("Estimated Trajectories")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Position")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    figure_path = f"{output_dir}/multi_particle_{Path(filepath).stem}_overview.png"
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ Overview figure saved: {figure_path}")
    
    return metrics_list


def run_multiple_challenge(
    hackathon_dir: str = "Hackathon",
    model_path: str = "models/multitask_unet.pth",
    output_dir: str = "challenge_results",
    csv_path: str = "challenge_results/multiple_particle_metrics.csv",
):
    """
    Process all multi-particle kymograph files in the Hackathon folder.
    
    Parameters:
    -----------
    hackathon_dir : str
        Directory containing kymograph files
    model_path : str
        Path to trained model
    output_dir : str
        Directory to save results and figures
    csv_path : str
        Path to save CSV metrics
    """
    print("=" * 70)
    print("MULTI-PARTICLE CHALLENGE PROCESSING")
    print("=" * 70)
    
    # Check model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model
    device = _default_device()
    print(f"\nLoading model: {model_path}")
    print(f"Device: {device}")
    model = load_multitask_model(model_path, device=device)
    model.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    def extract_multi_number(filepath: str) -> int:
        """Extract number from filename like 'kymograph_noisy_multiple_particles_1.npy' -> 1"""
        import re
        match = re.search(r'kymograph_noisy_multiple_particles_(\d+)\.npy', os.path.basename(filepath))
        return int(match.group(1)) if match else 0
    
    multi_files = sorted(
        glob.glob(os.path.join(hackathon_dir, "kymograph_noisy_multiple_particles_*.npy")),
        key=extract_multi_number
    )
    
    print(f"\nFound {len(multi_files)} multi-particle files")
    
    all_metrics = []
    
    # Process multi-particle files
    print("\n" + "=" * 70)
    print("PROCESSING MULTI-PARTICLE FILES")
    print("=" * 70)
    for filepath in multi_files:
        try:
            metrics_list = process_multi_particle_file(
                filepath, model, device, n_particles=None, output_dir=output_dir
            )
            all_metrics.extend(metrics_list)
        except Exception as e:
            print(f"  ✗ Error processing {filepath}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save CSV (only contrast_estimate and diffusion_processed)
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    written_csv = write_challenge_csv(all_metrics, csv_path)
    print(f"\n✓ Processed {len(all_metrics)} analyses")
    print(f"✓ Metrics saved to: {written_csv}")
    print(f"✓ Figures saved to: {output_dir}/")
    print("\nMulti-particle challenge processing complete!")


if __name__ == "__main__":
    run_multiple_challenge()
