"""Visualization utilities for comparing different tracking approaches."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Sequence


def visualize_comparison(
    noisy_kymo: np.ndarray,
    classical_filtered: np.ndarray,
    classical_mask: np.ndarray,
    classical_trajectories: Sequence[np.ndarray],
    deeplearning_denoised: np.ndarray,
    deeplearning_mask: np.ndarray,
    deeplearning_trajectories: Sequence[np.ndarray],
    true_paths: Optional[Sequence[np.ndarray]] = None,
    output_path: str = "comparison.png",
    title: str = "Classical vs Deep Learning Comparison",
) -> None:
    """
    Create a comparison plot with 4 subplots per method.
    
    Args:
        noisy_kymo: (T, W) noisy kymograph
        classical_filtered: (T, W) median filtered result
        classical_mask: (T, W) segmentation mask from classical approach
        classical_trajectories: list of (T,) arrays, one per particle
        deeplearning_denoised: (T, W) denoised result
        deeplearning_mask: (T, W) segmentation mask from deep learning
        deeplearning_trajectories: list of (T,) arrays, one per particle
        true_paths: optional list of (T,) arrays with ground truth paths
        output_path: where to save the figure
        title: figure title
    """
    T, W = noisy_kymo.shape
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Color scheme for trajectories
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Row 1: Classical approach
    # 1. Noisy input
    ax = axes[0, 0]
    ax.imshow(noisy_kymo.T, cmap='gray', aspect='auto', origin='lower')
    ax.set_title('1. Noisy Input', fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    
    # 2. Median filtered
    ax = axes[0, 1]
    ax.imshow(classical_filtered.T, cmap='gray', aspect='auto', origin='lower')
    ax.set_title('2. Median Filtered', fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    
    # 3. Segmentation mask
    ax = axes[0, 2]
    ax.imshow(classical_mask.T, cmap='gray', aspect='auto', origin='lower')
    ax.set_title('3. Segmentation Mask', fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    
    # 4. Trajectories
    ax = axes[0, 3]
    ax.imshow(noisy_kymo.T, cmap='gray', aspect='auto', origin='lower', alpha=0.3)
    has_trajectories = False
    for i, traj in enumerate(classical_trajectories):
        if traj is not None and len(traj) > 0:
            traj = np.asarray(traj, dtype=np.float64)
            # Ensure trajectory length matches kymograph time dimension
            if len(traj) > T:
                traj = traj[:T]
            elif len(traj) < T:
                # Pad with NaN if shorter
                padded = np.full(T, np.nan, dtype=np.float64)
                padded[:len(traj)] = traj
                traj = padded
            valid = ~np.isnan(traj)
            if valid.any():
                ax.plot(np.arange(T)[valid], traj[valid], 
                       color=colors[i % len(colors)], linewidth=2, 
                       label=f'Track {i+1}', alpha=0.8)
                has_trajectories = True
    if true_paths:
        for i, true_path in enumerate(true_paths):
            if true_path is not None and len(true_path) > 0:
                true_path = np.asarray(true_path, dtype=np.float64)
                ax.plot(true_path, color=colors[i % len(colors)], 
                       linestyle='--', linewidth=1.5, alpha=0.5, 
                       label=f'True {i+1}' if i < 2 else None)
    ax.set_title('4. Predicted Trajectories', fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    if has_trajectories or true_paths:
        ax.legend(loc='upper right', fontsize=8)
    
    # Row 2: Deep learning approach
    # 1. Noisy input (same)
    ax = axes[1, 0]
    ax.imshow(noisy_kymo.T, cmap='gray', aspect='auto', origin='lower')
    ax.set_title('1. Noisy Input', fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    
    # 2. Denoised
    ax = axes[1, 1]
    ax.imshow(deeplearning_denoised.T, cmap='gray', aspect='auto', origin='lower')
    ax.set_title('2. Denoised (U-Net)', fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    
    # 3. Segmentation mask
    ax = axes[1, 2]
    ax.imshow(deeplearning_mask.T, cmap='gray', aspect='auto', origin='lower')
    ax.set_title('3. Segmentation Mask', fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    
    # 4. Trajectories
    ax = axes[1, 3]
    ax.imshow(noisy_kymo.T, cmap='gray', aspect='auto', origin='lower', alpha=0.3)
    has_trajectories = False
    for i, traj in enumerate(deeplearning_trajectories):
        if traj is not None and len(traj) > 0:
            traj = np.asarray(traj, dtype=np.float64)
            # Ensure trajectory length matches kymograph time dimension
            if len(traj) > T:
                traj = traj[:T]
            elif len(traj) < T:
                # Pad with NaN if shorter
                padded = np.full(T, np.nan, dtype=np.float64)
                padded[:len(traj)] = traj
                traj = padded
            valid = ~np.isnan(traj)
            if valid.any():
                ax.plot(np.arange(T)[valid], traj[valid], 
                       color=colors[i % len(colors)], linewidth=2, 
                       label=f'Track {i+1}', alpha=0.8)
                has_trajectories = True
    if true_paths:
        for i, true_path in enumerate(true_paths):
            if true_path is not None and len(true_path) > 0:
                true_path = np.asarray(true_path, dtype=np.float64)
                ax.plot(true_path, color=colors[i % len(colors)], 
                       linestyle='--', linewidth=1.5, alpha=0.5, 
                       label=f'True {i+1}' if i < 2 else None)
    ax.set_title('4. Predicted Trajectories', fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    if has_trajectories or true_paths:
        ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


__all__ = ["visualize_comparison"]
