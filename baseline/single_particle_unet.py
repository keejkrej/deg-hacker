"""
Single-Particle U-Net Analysis Pipeline

This module provides:
- Single-particle denoising and tracking
- Chunked processing for large kymographs
- Parameter grid evaluation
- Comprehensive metrics and visualization
"""

from pathlib import Path

import numpy as np
import torch

from denoiser import load_model, _default_device
from utils.helpers import estimate_diffusion_msd_fit, get_particle_radius, find_max_subpixel
from utils import (
    simulate_single_particle,
    summarize_analysis,
    write_joint_metrics_csv,
)

# Default model path
MODEL_PATH = "models/tiny_unet_denoiser.pth"


def denoise_kymograph_chunked(
    model, kymograph, device=None, chunk_size=512, overlap=64
):
    """
    Denoise a kymograph by processing it in chunks.
    
    Handles kymographs larger than chunk_size by splitting along time dimension
    and reassembling with overlap handling.
    
    Parameters:
    -----------
    model : TinyUNet
        Trained denoising model
    kymograph : np.ndarray
        Noisy kymograph, shape (time, position)
    device : str, optional
        Device to use ('cuda' or 'cpu')
    chunk_size : int
        Size of chunks for processing (default: 512)
    overlap : int
        Overlap between chunks in pixels (default: 64)
    
    Returns:
    --------
    denoised : np.ndarray
        Denoised kymograph, same shape as input
    """
    if device is None:
        device = _default_device()
    
    model.eval()
    time_len, width = kymograph.shape
    
    # If kymograph fits in one chunk, process directly
    if time_len <= chunk_size:
        # Handle padding if needed
        if width < chunk_size:
            # Pad width
            padded = np.pad(
                kymograph,
                ((0, 0), (0, chunk_size - width)),
                mode="constant",
                constant_values=0,
            )
            # DDPM-style: predict noise, then subtract
            padded_tensor = torch.from_numpy(padded[None, None, :, :]).float().to(device)
            predicted_noise = model(padded_tensor)
            denoised_padded = torch.clamp(padded_tensor - predicted_noise, 0.0, 1.0).cpu().numpy()[0, 0]
            return denoised_padded[:, :width]
        else:
            # Process directly (DDPM-style: predict noise, then subtract)
            kymograph_tensor = torch.from_numpy(kymograph[None, None, :, :]).float().to(device)
            predicted_noise = model(kymograph_tensor)
            denoised = torch.clamp(kymograph_tensor - predicted_noise, 0.0, 1.0).cpu().numpy()[0, 0]
            return denoised
    
    # Process in chunks
    denoised = np.zeros_like(kymograph)
    
    # Determine if we need 2D chunking (both time and position)
    needs_2d_chunking = width > chunk_size
    
    if needs_2d_chunking:
        # 2D chunking: split both time and position
        for t_start in range(0, time_len, chunk_size - overlap):
            t_end = min(t_start + chunk_size, time_len)
            for x_start in range(0, width, chunk_size - overlap):
                x_end = min(x_start + chunk_size, width)
                
                chunk = kymograph[t_start:t_end, x_start:x_end]
                
                # Pad if necessary
                pad_t = chunk_size - (t_end - t_start)
                pad_x = chunk_size - (x_end - x_start)
                
                if pad_t > 0 or pad_x > 0:
                    chunk = np.pad(
                        chunk,
                        ((0, pad_t), (0, pad_x)),
                        mode="constant",
                        constant_values=0,
                    )
                
                # Denoise chunk (DDPM-style: predict noise, then subtract)
                with torch.no_grad():
                    chunk_tensor = (
                        torch.from_numpy(chunk[None, None, :, :]).float().to(device)
                    )
                    predicted_noise = model(chunk_tensor)
                    denoised_chunk = torch.clamp(chunk_tensor - predicted_noise, 0.0, 1.0).cpu().numpy()[0, 0]
                
                # Remove padding
                denoised_chunk = denoised_chunk[: t_end - t_start, : x_end - x_start]
                
                # Handle overlap: use weighted average
                t_slice = slice(t_start, t_end)
                x_slice = slice(x_start, x_end)
                
                if t_start > 0 or x_start > 0:
                    # Create weights for blending
                    weights = np.ones_like(denoised_chunk)
                    if t_start > 0:
                        weights[:overlap, :] *= np.linspace(0.5, 1.0, overlap)[:, None]
                    if x_start > 0:
                        weights[:, :overlap] *= np.linspace(0.5, 1.0, overlap)[None, :]
                    
                    existing = denoised[t_slice, x_slice]
                    denoised[t_slice, x_slice] = (
                        weights * denoised_chunk + (1 - weights) * existing
                    )
                else:
                    denoised[t_slice, x_slice] = denoised_chunk
    else:
        # 1D chunking: only split along time dimension
        for t_start in range(0, time_len, chunk_size - overlap):
            t_end = min(t_start + chunk_size, time_len)
            chunk = kymograph[t_start:t_end, :]
            
            # Pad if necessary
            if t_end - t_start < chunk_size:
                chunk = np.pad(
                    chunk,
                    ((0, chunk_size - (t_end - t_start)), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
            
            # Denoise chunk (DDPM-style: predict noise, then subtract)
            with torch.no_grad():
                chunk_tensor = (
                    torch.from_numpy(chunk[None, None, :, :]).float().to(device)
                )
                predicted_noise = model(chunk_tensor)
                denoised_chunk = torch.clamp(chunk_tensor - predicted_noise, 0.0, 1.0).cpu().numpy()[0, 0]
            
            # Remove padding
            denoised_chunk = denoised_chunk[: t_end - t_start, :]
            
            # Handle overlap: use weighted average
            if t_start > 0:
                weights = np.ones((t_end - t_start, width))
                weights[:overlap, :] = np.linspace(0.5, 1.0, overlap)[:, None]
                existing = denoised[t_start:t_end, :]
                denoised[t_start:t_end, :] = (
                    weights * denoised_chunk + (1 - weights) * existing
                )
            else:
                denoised[t_start:t_end, :] = denoised_chunk
    
    return denoised


def analyze_particle(p, c, n, model_path=None):
    """
    Analyze a single particle from kymograph data.
    
    Parameters:
    -----------
    p : float
        Particle radius in nm
    c : float
        Contrast
    n : float
        Noise level
    model_path : str, optional
        Path to trained model (default: models/tiny_unet_denoiser.pth)
    
    Returns:
    --------
    metrics : AnalysisMetrics
        Analysis results with diffusion, radius, and other metrics
    """
    if model_path is None:
        model_path = MODEL_PATH
    
    simulation = simulate_single_particle(p, c, n)
    device = _default_device()
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = load_model(model_path, device=device)
    
    denoised = denoise_kymograph_chunked(
        model, simulation.kymograph_noisy, device=device
    )
    
    return summarize_analysis(
        simulation,
        denoised,
        method_label="U-Net Denoised",
        figure_subdir="unet",
    )


def run_parameter_grid(
    particle_radii=[2.5, 5.0, 10.0],
    contrasts=[0.6, 0.8, 1.0],
    noise_levels=[0.1, 0.3, 0.5],
    csv_path="metrics/single_particle_unet.csv",
    model_path=None,
):
    """
    Run analysis over a grid of parameters.
    
    Parameters:
    -----------
    particle_radii : list
        Particle radii to test (nm)
    contrasts : list
        Contrast values to test
    noise_levels : list
        Noise levels to test
    csv_path : str
        Path to save CSV metrics
    model_path : str, optional
        Path to trained model
    
    Returns:
    --------
    csv_path : str
        Path to saved CSV file
    """
    if model_path is None:
        model_path = MODEL_PATH
    
    metrics_rows = []
    for p in particle_radii:
        for c in contrasts:
            for n in noise_levels:
                print(f"[U-Net] Running p={p:.1f} nm, c={c:.2f}, n={n:.2f}")
                metrics_rows.append(analyze_particle(p, c, n, model_path=model_path))
    
    written_csv = write_joint_metrics_csv(metrics_rows, csv_path)
    print(f"\n[U-Net] Completed {len(metrics_rows)} analyses; metrics -> {written_csv}")
    return written_csv


if __name__ == "__main__":
    # Run parameter grid
    run_parameter_grid()
