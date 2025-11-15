"""
Hackathon Challenge Script

Processes kymograph files from the Hackathon folder:
- Single particle files: kymograph_noisy_*.npy
- Multi-particle files: kymograph_noisy_multiple_particles_*.npy

For each file, this script:
1. Loads the noisy kymograph
2. Denoises using the trained U-Net model
3. Tracks particles and estimates parameters
4. Generates diagnostic plots and saves results
"""

import os
import glob
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional
from scipy.signal import find_peaks

from denoiser import load_model, _default_device
from single_particle_unet import denoise_kymograph_chunked
from multi_particle_unet import track_particles
from utils import (
    AnalysisMetrics,
    write_joint_metrics_csv,
    estimate_noise_and_contrast,
)
from helpers import estimate_diffusion_msd_fit, get_particle_radius


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
    model : TinyUNet
        Trained denoising model
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
    
    # Denoise
    denoised = denoise_kymograph_chunked(
        model, kymograph_noisy_norm, device=device, chunk_size=512, overlap=64
    )
    
    # Diagnostic: check denoised output
    print(f"  Denoised stats: min={denoised.min():.4f}, max={denoised.max():.4f}, mean={denoised.mean():.4f}, std={denoised.std():.4f}")
    
    # Track on normalized denoised (should be in [0,1] range)
    # The model outputs normalized values, tracking should work on these
    from helpers import find_max_subpixel
    
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
    
    # Convert path to pixel coordinates for diffusion estimation
    # estimated_path is in normalized coordinates [0, 1], convert to pixel positions
    if not np.all(np.isnan(estimated_path)):
        estimated_path_pixels = estimated_path * (kymograph_noisy.shape[1] - 1)
    else:
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
    
    # Create metrics (we don't have ground truth, so use estimates)
    metrics = AnalysisMetrics(
        method_label="Challenge Single",
        particle_radius_nm=radius_processed,  # Estimated
        diffusion_true=np.nan,  # Unknown
        diffusion_processed=diffusion_processed,
        radius_true=np.nan,  # Unknown
        radius_processed=radius_processed,
        contrast=np.nan,  # Unknown
        noise_level=np.nan,  # Unknown
        contrast_estimate=contrast_estimate,
        noise_estimate=noise_estimate,
        figure_path=f"{output_dir}/single_{Path(filepath).stem}.png",
    )
    
    # Create visualization - 2x2 layout for better diagnosis
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    vmin, vmax = kymograph_noisy.min(), kymograph_noisy.max()
    
    # Top left: Noisy input
    axes[0, 0].imshow(
        kymograph_noisy.T,
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap="gray",
    )
    axes[0, 0].set_title("Noisy Input")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Position")
    
    # Top right: Denoised only (no track overlay)
    # Denormalize: denorm = norm * (max - min) + min + background
    denoised_vis = denoised * (kymograph_max - kymograph_min) + kymograph_min + background_level
    im = axes[0, 1].imshow(
        denoised_vis.T,
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap="gray",
    )
    axes[0, 1].set_title("Denoised Output")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Position")
    
    # Bottom left: Normalized input (what goes into the model)
    axes[1, 0].imshow(
        kymograph_noisy_norm.T,
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        cmap="gray",
    )
    axes[1, 0].set_title(f"Normalized Input\n(bg={background_level:.3f}, sig={signal_level:.3f})")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Position")
    
    # Bottom right: Estimated trajectory
    if not np.all(np.isnan(estimated_path_pixels)):
        axes[1, 1].plot(estimated_path_pixels, color="blue", lw=1.0)
        axes[1, 1].set_title("Estimated Trajectory")
        axes[1, 1].set_xlabel("Time")
        axes[1, 1].set_ylabel("Position (pixels)")
        axes[1, 1].grid(True, alpha=0.3)
    else:
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
    model : TinyUNet
        Trained denoising model
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
    
    # Denoise
    denoised_norm = denoise_kymograph_chunked(
        model, kymograph_noisy_norm, device=device, chunk_size=512, overlap=64
    )
    
    # Denormalize for analysis (convert back to original scale)
    denoised = denoised_norm * (kymograph_max - kymograph_min) + kymograph_min + background_level
    
    # Estimate number of particles if not provided
    if n_particles is None:
        n_particles = estimate_n_particles(kymograph_noisy, denoised)
        print(f"  Estimated {n_particles} particles")
    
    # Track particles
    estimated_tracks = track_particles(
        denoised,
        n_particles=n_particles,
        max_candidates=30,
        max_jump=15,
        detect_crossings=True,
        crossing_threshold=5.0,
        crossing_padding=2,
    )
    
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
        
        # Estimate diffusion coefficient with error handling
        valid_mask = ~np.isnan(track)
        if np.sum(valid_mask) < 10:
            print(f"    ⚠ Too few valid points ({np.sum(valid_mask)}), skipping diffusion estimation")
            diffusion_processed = np.nan
            radius_processed = np.nan
        elif np.std(track[valid_mask]) < 0.1:
            print(f"    ⚠ Track has no variation (std={np.std(track[valid_mask]):.4f}), skipping diffusion estimation")
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
        metrics = AnalysisMetrics(
            method_label=f"Challenge Multi Track {track_id + 1}",
            particle_radius_nm=radius_processed,
            diffusion_true=np.nan,
            diffusion_processed=diffusion_processed,
            radius_true=np.nan,
            radius_processed=radius_processed,
            contrast=np.nan,
            noise_level=np.nan,
            contrast_estimate=contrast_estimate_track,
            noise_estimate=noise_estimate_global,
            figure_path=f"{output_dir}/multi_{Path(filepath).stem}_track{track_id + 1}.png",
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
    
    vmin, vmax = kymograph_noisy.min(), kymograph_noisy.max()
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    
    # Noisy input
    axes[0, 0].imshow(
        kymograph_noisy.T,
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap="gray",
    )
    axes[0, 0].set_title("Noisy Input")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Position")
    
    # Denoised with tracks
    axes[0, 1].imshow(
        denoised.T,
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
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
    
    # Trajectories plot
    for track_id in range(n_particles):
        track = estimated_tracks[track_id]
        valid_mask = ~np.isnan(track)
        if np.sum(valid_mask) > 0:
            color = colors[track_id % len(colors)]
            axes[1, 0].plot(
                track, color=color, lw=1.0, alpha=0.7, label=f"Track {track_id + 1}"
            )
    axes[1, 0].set_title("Estimated Trajectories")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Position")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics
    axes[1, 1].axis('off')
    summary_text = f"Summary ({n_particles} particles):\n\n"
    for i, metrics in enumerate(metrics_list):
        summary_text += f"Track {i + 1}:\n"
        summary_text += f"  D: {metrics.diffusion_processed:.4f} μm²/s\n"
        summary_text += f"  R: {metrics.radius_processed:.2f} nm\n"
        summary_text += f"  C: {metrics.contrast_estimate:.3f}\n\n"
    summary_text += f"Global Noise: {noise_estimate_global:.3f}\n"
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                    family='monospace')
    
    plt.tight_layout()
    figure_path = f"{output_dir}/multi_{Path(filepath).stem}_overview.png"
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ Overview figure saved: {figure_path}")
    
    return metrics_list


def run_challenge(
    hackathon_dir: str = "Hackathon",
    model_path: str = "models/tiny_unet_denoiser.pth",
    output_dir: str = "challenge_results",
    csv_path: str = "challenge_results/challenge_metrics.csv",
):
    """
    Process all kymograph files in the Hackathon folder.
    
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
    print("HACKATHON CHALLENGE PROCESSING")
    print("=" * 70)
    
    # Check model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model
    device = _default_device()
    print(f"\nLoading model: {model_path}")
    print(f"Device: {device}")
    model = load_model(model_path, device=device)
    model.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all files and sort by numeric order (not lexicographic)
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
    
    def extract_multi_number(filepath: str) -> int:
        """Extract number from filename like 'kymograph_noisy_multiple_particles_1.npy' -> 1"""
        import re
        match = re.search(r'kymograph_noisy_multiple_particles_(\d+)\.npy', os.path.basename(filepath))
        return int(match.group(1)) if match else 0
    
    multi_files = sorted(
        glob.glob(os.path.join(hackathon_dir, "kymograph_noisy_multiple_particles_*.npy")),
        key=extract_multi_number
    )
    
    print(f"\nFound {len(single_files)} single-particle files")
    print(f"Found {len(multi_files)} multi-particle files")
    
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
    
    # Save CSV
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    written_csv = write_joint_metrics_csv(all_metrics, csv_path)
    print(f"\n✓ Processed {len(all_metrics)} analyses")
    print(f"✓ Metrics saved to: {written_csv}")
    print(f"✓ Figures saved to: {output_dir}/")
    print("\nChallenge processing complete!")


if __name__ == "__main__":
    run_challenge()
