"""
Visualize multi-track denoising results to verify model performance.

This script loads a trained model and generates visualization plots showing
how well the model denoises multi-track kymographs (2-3 particles).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from denoiser import load_model, denoise_kymograph, _default_device
from helpers import generate_kymograph, get_diffusion_coefficient


def visualize_multi_track_denoising_results(
    model_path: str = "tiny_unet_denoiser.pth",
    n_samples: int = 3,
    length: int = 512,
    width: int = 512,
    device: str | None = None,
    save_path: str = "multi_track_denoising_results.png",
) -> None:
    """
    Generate multi-track test samples and visualize denoising results.
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model checkpoint
    n_samples : int
        Number of test samples to generate and visualize
    length : int
        Time dimension of kymograph
    width : int
        Position dimension of kymograph
    device : str, optional
        Device to run model on
    save_path : str
        Path to save the visualization
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    device = device or _default_device()
    model = load_model(model_path, device=device)
    model.eval()
    
    print(f"\nGenerating {n_samples} multi-track test samples for visualization...")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    
    # Generate test samples: mix of 2 and 3 trajectory examples
    test_configs = [
        {"n_tracks": 2, "radii": [5.0, 10.0], "contrasts": [0.7, 0.5], "noise": 0.3},
        {"n_tracks": 3, "radii": [7.5, 12.0, 8.0], "contrasts": [0.8, 0.6, 0.5], "noise": 0.4},
        {"n_tracks": 2, "radii": [3.0, 15.0], "contrasts": [0.9, 0.4], "noise": 0.2},
    ]
    
    # Ensure we have enough configs for n_samples
    while len(test_configs) < n_samples:
        test_configs.append(test_configs[len(test_configs) % len(test_configs)])
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i in range(n_samples):
        config = test_configs[i]
        radii = config["radii"]
        contrasts = config["contrasts"]
        noise = config["noise"]
        n_tracks = config["n_tracks"]
        
        # Generate multi-trajectory kymograph
        diffusions = [get_diffusion_coefficient(r) for r in radii]
        noisy, gt, true_paths = generate_kymograph(
            length=length,
            width=width,
            diffusion=diffusions,
            contrast=contrasts,
            noise_level=noise,
            peak_width=1.0,
            dt=1.0,
            dx=0.5,
        )
        
        # Denoise
        import torch
        with torch.no_grad():
            tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).float().to(device)
            predicted_noise_raw = model(tensor)
            predicted_noise = predicted_noise_raw.squeeze(0).squeeze(0).cpu().numpy()
            denoised = np.clip(noisy - predicted_noise, 0.0, 1.0)
        
        # Calculate metrics
        mse_noisy = np.mean((noisy - gt) ** 2)
        mse_denoised = np.mean((denoised - gt) ** 2)
        improvement = ((mse_noisy - mse_denoised) / mse_noisy) * 100
        
        # Additional diagnostics
        denoised_mean = np.mean(denoised)
        denoised_max = np.max(denoised)
        denoised_min = np.min(denoised)
        gt_mean = np.mean(gt)
        gt_max = np.max(gt)
        noise_mean = np.mean(noisy - gt)
        predicted_noise_mean = np.mean(predicted_noise)
        predicted_noise_std = np.std(predicted_noise)
        
        # Check what model is actually predicting
        print(f"\n  Sample {i+1} diagnostics:")
        print(f"    Noisy mean: {np.mean(noisy):.4f}, GT mean: {gt_mean:.4f}, True noise mean: {noise_mean:.4f}")
        print(f"    Predicted noise mean: {predicted_noise_mean:.4f}, std: {predicted_noise_std:.4f}")
        print(f"    Denoised mean: {denoised_mean:.4f}, max: {denoised_max:.4f}, min: {denoised_min:.4f}")
        print(f"    MSE noisy: {mse_noisy:.4f}, MSE denoised: {mse_denoised:.4f}")
        
        # Check for potential failures
        failure_indicators = []
        if improvement < 0:
            failure_indicators.append("NEGATIVE improvement")
        if abs(predicted_noise_mean) < 0.001:
            failure_indicators.append("PREDICTED NOISE near zero")
        if abs(predicted_noise_mean - noise_mean) > 0.1:
            failure_indicators.append("NOISE prediction mismatch")
        if denoised_mean < 0.01:
            failure_indicators.append("VERY LOW mean output")
        if denoised_max < 0.1:
            failure_indicators.append("VERY LOW max output")
        if abs(denoised_mean - gt_mean) > 0.3:
            failure_indicators.append("MEAN mismatch")
        
        status = "✓ OK" if len(failure_indicators) == 0 else f"⚠ FAIL: {', '.join(failure_indicators)}"
        print(f"    Status: {status}")
        
        # Plot
        vmin, vmax = 0, 1
        
        # Noisy input
        axes[i, 0].imshow(noisy.T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="gray")
        for idx in range(n_tracks):
            axes[i, 0].plot(true_paths[idx], color=colors[idx % len(colors)], 
                           lw=1.0, alpha=0.7, label=f"Track {idx+1}")
        axes[i, 0].set_title(f"Noisy Input ({n_tracks} tracks)\nMSE: {mse_noisy:.4f}, n={noise:.1f}")
        axes[i, 0].set_xlabel("Time")
        axes[i, 0].set_ylabel("Position")
        axes[i, 0].legend(loc='upper right', fontsize=8)
        
        # Denoised output
        axes[i, 1].imshow(denoised.T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="gray")
        for idx in range(n_tracks):
            axes[i, 1].plot(true_paths[idx], color=colors[idx % len(colors)], 
                           lw=1.0, alpha=0.7, linestyle='--', label=f"True {idx+1}")
        axes[i, 1].set_title(f"Denoised Output\nMSE: {mse_denoised:.4f}\nImprovement: {improvement:.1f}%")
        axes[i, 1].set_xlabel("Time")
        axes[i, 1].set_ylabel("Position")
        axes[i, 1].legend(loc='upper right', fontsize=8)
        
        # Ground truth
        axes[i, 2].imshow(gt.T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="gray")
        for idx in range(n_tracks):
            axes[i, 2].plot(true_paths[idx], color=colors[idx % len(colors)], 
                           lw=1.0, alpha=0.7, label=f"Track {idx+1}")
        axes[i, 2].set_title("Ground Truth")
        axes[i, 2].set_xlabel("Time")
        axes[i, 2].set_ylabel("Position")
        axes[i, 2].legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nMulti-track visualization saved to: {save_path}")
    plt.close()
    
    print("\nMulti-track denoising visualization complete!")


if __name__ == "__main__":
    visualize_multi_track_denoising_results()
