"""
Single-Particle Challenge Inference Script

Processes single-particle kymograph files from the Hackathon folder:
- Single particle files: kymograph_noisy_*.npy

For each file, this script:
1. Loads the noisy kymograph
2. Denoises and segments using the trained multi-task U-Net model
3. Tracks particles and estimates parameters
4. Generates diagnostic plots and saves results
5. Saves metrics to CSV

Usage:
    python inference/single_challenge.py
    # Or with custom paths:
    python inference/single_challenge.py --hackathon_dir Hackathon --model_path models/multitask_unet.pth
"""

import os
import sys
import glob
import csv
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, List
from scipy.signal import find_peaks

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from train.multitask_model import load_multitask_model, denoise_and_segment_chunked, _default_device
from utils import (
    AnalysisMetrics,
    estimate_noise_and_contrast,
)
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


def process_single_particle_file(
    filepath: str,
    model,
    device: str,
    output_dir: str = "challenge_results",
) -> AnalysisMetrics:
    """
    Process a single-particle kymograph file.
    
    Parameters:
    -----------
    filepath : str
        Path to the .npy file
    model : MultiTaskUNet
        Trained multi-task model (denoising + segmentation)
    device : str
        Device to run model on
    output_dir : str
        Directory to save results
    
    Returns:
    --------
    metrics : AnalysisMetrics
        Analysis results
    """
    print(f"\nProcessing single-particle file: {filepath}")
    
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
    denoised, segmentation_labels = denoise_and_segment_chunked(
        model, kymograph_noisy_norm, device=device, chunk_size=512, overlap=64
    )
    
    # Set edge regions to black (background) at all times to handle edge artifacts
    # Positions 0-11 and 500-511 (if width >= 512) are set to 0
    time_len, width = segmentation_labels.shape
    segmentation_labels[:, 0:12] = 0.0  # Left edge: positions 0-11
    if width >= 512:
        segmentation_labels[:, 500:512] = 0.0  # Right edge: positions 500-511
    elif width > 500:
        segmentation_labels[:, 500:] = 0.0  # Right edge: positions 500 to end
    
    # Apply segmentation mask to suppress background noise
    # segmentation_labels is now a binary mask (0=background, 1=particle)
    from scipy.ndimage import binary_opening, binary_dilation
    
    mask_binary = (segmentation_labels > 0.7).astype(np.float32)  # Threshold at 0.7 for binary (higher = stricter)
    
    # Apply opening (erosion + dilation) to remove small noise
    opening_kernel = np.ones((3, 3), dtype=bool)
    mask_opened = binary_opening(mask_binary, structure=opening_kernel).astype(np.float32)
    
    # Apply opening to segmentation_labels for use in apply_segmentation_mask
    segmentation_labels_opened = mask_opened.astype(np.float32)
    
    mask_dilated = (apply_segmentation_mask(
        np.ones_like(segmentation_labels_opened), segmentation_labels_opened, dilation_size=(3, 3)
    ) > 0.7).astype(np.float32)
    
    denoised = apply_segmentation_mask(
        denoised, segmentation_labels_opened, dilation_size=(3, 3), background_weight=0.3
    )
    
    print(f"  Segmentation mask: {np.sum(mask_binary > 0):.0f} pixels ({100*np.mean(mask_binary):.1f}% of image)")
    print(f"  After opening: {np.sum(mask_opened > 0):.0f} pixels ({100*np.mean(mask_opened):.1f}% of image)")
    print(f"  Mask value range: [{segmentation_labels.min():.3f}, {segmentation_labels.max():.3f}]")
    print(f"  After dilation: {np.sum(mask_dilated > 0):.0f} pixels ({100*np.mean(mask_dilated):.1f}% of image)")
    
    # Check background noise level (std of 0-80 percentile regions, outside tracks)
    # Only apply median filter if background noise is high
    background_mask = denoised <= np.percentile(denoised, 80)
    background_std = np.std(denoised[background_mask]) if np.any(background_mask) else 0.0
    median_filter_threshold = 0.01  # Apply if background std > 0.01
    
    if background_std > median_filter_threshold:
        from scipy.signal import medfilt
        print(f"  Applying median filter (background std={background_std:.4f} > {median_filter_threshold:.4f})...")
        # Use larger kernel for sparse noise: (9, 5) = 9 time steps, 5 spatial pixels
        denoised = medfilt(denoised, kernel_size=(9, 5))
    else:
        print(f"  Skipping median filter (background std={background_std:.4f} <= {median_filter_threshold:.4f})")
    
    # Diagnostic: check denoised output
    noise_input, contrast_input = estimate_noise_and_contrast(kymograph_noisy_norm)
    noise_denoised, contrast_denoised = estimate_noise_and_contrast(denoised)
    contrast_improvement = contrast_denoised / max(contrast_input, 1e-6)
    noise_reduction = noise_denoised / max(noise_input, 1e-6)
    print(f"  Denoised stats: min={denoised.min():.4f}, max={denoised.max():.4f}, mean={denoised.mean():.4f}, std={denoised.std():.4f}")
    print(f"  Noise: {noise_input:.4f} -> {noise_denoised:.4f} ({noise_reduction:.2f}x reduction)")
    print(f"  Contrast: {contrast_input:.4f} -> {contrast_denoised:.4f} ({contrast_improvement:.2f}x improvement)")
    
    # Track on normalized denoised (should be in [0,1] range)
    # The model outputs normalized values, tracking should work on these
    from utils.helpers import find_max_subpixel
    
    try:
        # find_max_subpixel expects 2D array and processes each row
        estimated_path = find_max_subpixel(denoised)
    except (ValueError, RuntimeError, IndexError) as e:
        # Fallback: process row by row manually
        print(f"  ⚠ Warning: find_max_subpixel failed, using manual tracking: {e}")
        estimated_path = []
        for t in range(len(denoised)):
            row = denoised[t]
            # Check if row is valid
            if np.all(np.isnan(row)) or np.all(row == 0) or (np.max(row) == np.min(row) and not np.isnan(row).any()):
                estimated_path.append(np.nan)
            else:
                try:
                    max_idx = np.nanargmax(row)
                    # Exclude extreme positions
                    if max_idx == 0 or max_idx == len(row) - 1:
                        estimated_path.append(float(max_idx))  # Use integer position
                    else:
                        # Parabolic interpolation for subpixel accuracy
                        y0, y1, y2 = row[max_idx - 1], row[max_idx], row[max_idx + 1]
                        denom = (y0 - 2 * y1 + y2)
                        if abs(denom) < 1e-10:
                            estimated_path.append(float(max_idx))
                        else:
                            delta = 0.5 * (y0 - y2) / denom
                            estimated_path.append(float(max_idx + delta))
                except (ValueError, IndexError):
                    estimated_path.append(np.nan)
        estimated_path = np.array(estimated_path)
    
    # find_max_subpixel returns pixel coordinates (column indices), not normalized
    # So estimated_path is already in pixel coordinates [0, width-1]
    estimated_path_pixels = estimated_path.copy()
    
    # Check if path is valid (has variation and enough points)
    valid_path = ~np.isnan(estimated_path_pixels)
    if np.sum(valid_path) < 10:
        print(f"  ⚠ Warning: Too few valid points ({np.sum(valid_path)}), using NaN for diffusion")
        diffusion_processed = np.nan
        radius_processed = np.nan
    elif np.std(estimated_path_pixels[valid_path]) < 0.1:
        print(f"  ⚠ Warning: Path has no variation (std={np.std(estimated_path_pixels[valid_path]):.4f}), using NaN for diffusion")
        diffusion_processed = np.nan
        radius_processed = np.nan
    else:
        # Estimate diffusion coefficient with error handling
        try:
            diffusion_processed = estimate_diffusion_msd_fit(estimated_path_pixels)
            if np.isnan(diffusion_processed) or diffusion_processed <= 0:
                print(f"  ⚠ Warning: Invalid diffusion estimate ({diffusion_processed}), using NaN")
                diffusion_processed = np.nan
                radius_processed = np.nan
            else:
                radius_processed = get_particle_radius(diffusion_processed)
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"  ⚠ Warning: Diffusion estimation failed ({e}), using NaN")
            diffusion_processed = np.nan
            radius_processed = np.nan
    
    # Estimate noise and contrast
    noise_estimate, contrast_estimate = estimate_noise_and_contrast(
        kymograph_noisy
    )
    
    # Create metrics
    # For challenge data, we don't have ground truth, so:
    # - Use estimates for contrast/noise_level (main fields)
    # - Keep diffusion_true/radius_true as NaN (true ground truth unknown)
    # - Also populate estimate fields for consistency
    metrics = AnalysisMetrics(
        method_label="Challenge Single",
        particle_radius_nm=radius_processed,  # Estimated radius
        diffusion_true=np.nan,  # True ground truth unknown
        diffusion_processed=diffusion_processed,  # Estimated diffusion
        radius_true=np.nan,  # True ground truth unknown
        radius_processed=radius_processed,  # Estimated radius
        contrast=contrast_estimate,  # Use estimate for main contrast field
        noise_level=noise_estimate,  # Use estimate for main noise_level field
        contrast_estimate=contrast_estimate,  # Also in estimate field
        noise_estimate=noise_estimate,  # Also in estimate field
        figure_path=f"{output_dir}/single_particle_{Path(filepath).stem}.png",
    )
    
    # Create visualization - 2x2 layout for better diagnosis
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Use percentile-based ranges to be robust to outliers
    vmin_noisy = np.percentile(kymograph_noisy, 1)
    vmax_noisy = np.percentile(kymograph_noisy, 99)
    
    # Top left: Noisy input
    axes[0, 0].imshow(
        kymograph_noisy.T,
        aspect="auto",
        origin="lower",
        vmin=vmin_noisy,
        vmax=vmax_noisy,
        cmap="gray",
    )
    axes[0, 0].set_title("Noisy Input")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Position")
    
    # Top right: Denoised only (no track overlay)
    # Denormalize: denorm = norm * (max - min) + min + background
    denoised_vis = denoised * (kymograph_max - kymograph_min) + kymograph_min + background_level
    # Use denoised range for better contrast
    vmin_denoised = np.percentile(denoised_vis, 1)
    vmax_denoised = np.percentile(denoised_vis, 99)
    im = axes[0, 1].imshow(
        denoised_vis.T,
        aspect="auto",
        origin="lower",
        vmin=vmin_denoised,
        vmax=vmax_denoised,
        cmap="gray",
    )
    axes[0, 1].set_title("Denoised Output")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Position")
    
    # Bottom left: Segmentation mask (after opening)
    axes[1, 0].imshow(
        segmentation_labels_opened.T,
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        cmap="gray",
    )
    axes[1, 0].set_title("Segmentation Mask\n(opened, binary: 0=bg, 1=particle)")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Position")
    
    # Bottom right: Estimated trajectory
    # Set same limits as kymograph plots
    time_len, width = kymograph_noisy.shape
    if not np.all(np.isnan(estimated_path_pixels)):
        axes[1, 1].plot(estimated_path_pixels, color="blue", lw=1.0)
        axes[1, 1].set_xlim(0, time_len)
        axes[1, 1].set_ylim(0, width)
        axes[1, 1].set_title("Estimated Trajectory")
        axes[1, 1].set_xlabel("Time")
        axes[1, 1].set_ylabel("Position (pixels)")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].set_xlim(0, time_len)
        axes[1, 1].set_ylim(0, width)
        axes[1, 1].text(0.5, 0.5, "No valid trajectory", 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("Estimated Trajectory (No valid track)")
    
    plt.tight_layout()
    plt.savefig(metrics.figure_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ Diffusion: {diffusion_processed:.4f} μm²/s")
    print(f"  ✓ Radius: {radius_processed:.2f} nm")
    print(f"  ✓ Contrast: {contrast_estimate:.3f}")
    print(f"  ✓ Noise: {noise_estimate:.3f}")
    print(f"  ✓ Figure saved: {metrics.figure_path}")
    
    return metrics


def run_single_challenge(
    hackathon_dir: str = "Hackathon",
    model_path: str = "models/multitask_unet.pth",
    output_dir: str = "challenge_results",
    csv_path: str = "challenge_results/single_particle_metrics.csv",
):
    """
    Process all single-particle kymograph files in the Hackathon folder.
    
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
    print("SINGLE-PARTICLE CHALLENGE PROCESSING")
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
    
    # Find all single-particle files and sort by numeric order (not lexicographic)
    def extract_number(filepath: str) -> int:
        """Extract number from filename like 'kymograph_noisy_1.npy' -> 1"""
        import re
        match = re.search(r'kymograph_noisy_(\d+)\.npy', os.path.basename(filepath))
        return int(match.group(1)) if match else 0
    
    # Get single-particle files (exclude multi-particle files)
    single_files_all = glob.glob(os.path.join(hackathon_dir, "kymograph_noisy_*.npy"))
    single_files = [
        f for f in single_files_all 
        if 'multiple_particles' not in os.path.basename(f)
    ]
    single_files = sorted(single_files, key=extract_number)
    
    print(f"\nFound {len(single_files)} single-particle files")
    
    all_metrics = []
    
    # Process single-particle files
    print("\n" + "=" * 70)
    print("PROCESSING SINGLE-PARTICLE FILES")
    print("=" * 70)
    for filepath in single_files:
        try:
            metrics = process_single_particle_file(
                filepath, model, device, output_dir
            )
            all_metrics.append(metrics)
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
    print("\nSingle-particle challenge processing complete!")


if __name__ == "__main__":
    run_single_challenge()
