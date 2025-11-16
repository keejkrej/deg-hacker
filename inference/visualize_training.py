"""
Visualize Training Set Examples

Processes and visualizes examples from the training dataset to:
1. Show model performance on training data
2. Visualize denoising and locator trajectory outputs
3. Compare ground truth vs predictions
4. Display per-particle center/width overlays for different tracks
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from train.multitask_model import (
    MultiTaskDataset,
    load_multitask_model,
    denoise_and_segment_chunked,
    _default_device,
)
from utils.helpers import get_diffusion_coefficient


def visualize_training_example(
    dataset: MultiTaskDataset,
    index: int,
    model,
    device: str,
    output_dir: str = "figures/training_visualizations",
    show_segmentation_labels: bool = True,
) -> None:
    """
    Visualize a single training example.

    Parameters:
    -----------
    dataset : MultiTaskDataset
        Training dataset
    index : int
        Index of example to visualize
    model : MultiTaskUNet
        Trained model
    device : str
        Device to run inference on
    output_dir : str
        Directory to save figures
    show_segmentation_labels : bool
        Whether to overlay ground-truth trajectory markers
    """
    # Get training example
    noisy_tensor, noise_tensor, pos_tensor, width_tensor, mask_tensor = dataset[index]
    noisy = noisy_tensor.squeeze().numpy()
    true_noise = noise_tensor.squeeze().numpy()
    gt_positions = pos_tensor.numpy()
    gt_widths = width_tensor.numpy()
    gt_mask = mask_tensor.numpy().astype(bool)

    # Get ground truth denoised
    gt_denoised = noisy - true_noise
    time_len, space_len = noisy.shape
    display_aspect = time_len / max(space_len, 1)
    gt_positions_px = gt_positions * max(space_len - 1, 1)
    gt_widths_px = gt_widths * max(space_len, 1)

    # Run inference
    model.eval()
    with torch.no_grad():
        denoised, track_params = denoise_and_segment_chunked(
            model, noisy, device=device, chunk_size=16, overlap=8
        )
    pred_centers = track_params["centers"]
    pred_widths = track_params["widths"]
    pred_centers = np.clip(pred_centers, 0.0, space_len - 1)
    pred_widths = np.clip(pred_widths, 0.0, space_len)

    # Create visualization
    os.makedirs(output_dir, exist_ok=True)
    n_tracks = gt_positions.shape[0]
    n_cols = 3  # First row always shows [noisy, GT, denoised]
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(4 * n_cols, 8),
        sharex="col",
        sharey="row",
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )

    # Color maps
    vmin_noisy = np.percentile(noisy, 1)
    vmax_noisy = np.percentile(noisy, 99)
    vmin_denoised = np.percentile(gt_denoised, 1)
    vmax_denoised = np.percentile(gt_denoised, 99)

    # Row 1: Input, Ground Truth, Denoised
    axes[0, 0].imshow(
        noisy.T,
        aspect=display_aspect,
        origin="lower",
        vmin=vmin_noisy,
        vmax=vmax_noisy,
        cmap="gray",
    )
    axes[0, 0].set_title("Noisy Input")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Position")

    axes[0, 1].imshow(
        gt_denoised.T,
        aspect=display_aspect,
        origin="lower",
        vmin=vmin_denoised,
        vmax=vmax_denoised,
        cmap="gray",
    )
    axes[0, 1].set_title("Ground Truth (Denoised)")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Position")

    axes[0, 2].imshow(
        denoised.T,
        aspect=display_aspect,
        origin="lower",
        vmin=vmin_denoised,
        vmax=vmax_denoised,
        cmap="gray",
    )
    axes[0, 2].set_title("Model Prediction (Denoised)")
    axes[0, 2].set_xlabel("Time")
    axes[0, 2].set_ylabel("Position")
    # Hide unused axes in first row if n_cols > 3
    for col in range(3, n_cols):
        axes[0, col].axis("off")

    # Row 2: ground-truth vs predicted trajectories
    time_axis = np.arange(time_len)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_tracks, 1)))

    ax_gt = axes[1, 0]
    ax_pred = axes[1, 1]

    if show_segmentation_labels:
        ax_gt.set_title("Ground Truth Trajectories")
        for track_idx in range(n_tracks):
            valid_idx = gt_mask[track_idx]
            if not valid_idx.any():
                continue
            color = colors[track_idx % len(colors)]
            ax_gt.plot(
                time_axis[valid_idx],
                gt_positions_px[track_idx][valid_idx],
                color=color,
                linewidth=1.5,
                label=f"Track {track_idx + 1}",
            )
            half_width = gt_widths_px[track_idx][valid_idx] * 0.5
            ax_gt.fill_between(
                time_axis[valid_idx],
                np.clip(gt_positions_px[track_idx][valid_idx] - half_width, 0, space_len - 1),
                np.clip(gt_positions_px[track_idx][valid_idx] + half_width, 0, space_len - 1),
                color=color,
                alpha=0.15,
            )
        ax_gt.set_xlabel("Time")
        ax_gt.set_ylabel("Position")
        if n_tracks > 0:
            ax_gt.legend(loc="upper right")
    else:
        ax_gt.axis("off")

    ax_pred.set_title("Predicted Trajectories")
    for track_idx in range(n_tracks):
        color = colors[track_idx % len(colors)]
        pred_center = pred_centers[:, track_idx]
        pred_half_width = pred_widths[:, track_idx] * 0.5
        lower = np.clip(pred_center - pred_half_width, 0, space_len - 1)
        upper = np.clip(pred_center + pred_half_width, 0, space_len - 1)
        ax_pred.fill_between(
            time_axis,
            lower,
            upper,
            color=color,
            alpha=0.2,
        )
        ax_pred.plot(time_axis, pred_center, color=color, linewidth=1.5)
    ax_pred.set_xlabel("Time")
    ax_pred.set_ylabel("Position")

    # Hide unused subplot in second row (third column)
    axes[1, 2].axis("off")

    # Save figure
    filename = f"training_example_{index:04d}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"  Saved: {filepath}")
    plt.close()

    # Print statistics
    print(f"\n  Example {index} Statistics:")
    print(f"    Input range: [{noisy.min():.3f}, {noisy.max():.3f}]")
    print(f"    Denoising MAE: {np.mean(np.abs(gt_denoised - denoised)):.4f}")
    print(f"    Denoising RMSE: {np.sqrt(np.mean((gt_denoised - denoised) ** 2)):.4f}")

    # Trajectory statistics
    gt_positions_time = gt_positions_px.transpose(1, 0)
    gt_mask_time = gt_mask.transpose(1, 0)
    denom = max(gt_mask_time.sum(), 1)
    center_mae = np.sum(np.abs(pred_centers - gt_positions_time) * gt_mask_time) / denom
    print(f"    Predicted centers shape: {pred_centers.shape}")
    print(f"    Center MAE (pixels): {center_mae:.3f}")
    active_channels = np.sum(gt_mask.max(axis=1))
    print(f"    Active GT trajectories: {active_channels}/{n_tracks}")


def visualize_training_set(
    model_path: str = "models/multitask_unet.pth",
    n_examples: int = 2,
    output_dir: str = "figures/training_visualizations",
    dataset_length: int = 512,  # Match training dimensions
    dataset_width: int = 512,  # Match training dimensions
    max_trajectories: int = 3,
    show_segmentation_labels: bool = True,
    weights_path: Optional[str] = None,
) -> None:
    """
    Visualize multiple training examples.

    Parameters:
    -----------
    model_path : str
        Path to trained model
    n_examples : int
        Number of examples to visualize
    output_dir : str
        Directory to save figures
    dataset_length : int
        Length of kymograph (time dimension)
    dataset_width : int
        Width of kymograph (space dimension)
    max_trajectories : int
        Maximum number of trajectories in dataset
    show_segmentation_labels : bool
        Whether to overlay ground-truth trajectory markers
    """
    print("=" * 70)
    print("TRAINING SET VISUALIZATION")
    print("=" * 70)

    # Determine which file to load (weights override model_path)
    load_path = weights_path or model_path
    if weights_path:
        if weights_path != model_path:
            print(f"?? Using weights override: {weights_path}")
        else:
            print(f"?? Using specified weights file: {weights_path}")

    # Check model/weights path exists
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model/weights file not found: {load_path}")

    # Load model
    device = _default_device()
    print(f"\nLoading model: {load_path}")
    print(f"Device: {device}")
    model = load_multitask_model(load_path, device=device, max_tracks=max_trajectories)
    model.eval()

    # Clean output directory so only fresh figures are kept
    if os.path.exists(output_dir):
        print(f"\nClearing previous visualizations in: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create dataset
    print(f"\nCreating training dataset...")
    print(f"  Length: {dataset_length}, Width: {dataset_width}")
    print(f"  Max trajectories: {max_trajectories}")
    dataset = MultiTaskDataset(
        length=dataset_length,
        width=dataset_width,
        max_trajectories=max_trajectories,
        multi_trajectory_prob=1.0,  # 100% multi-particle examples
        window_length=16,
    )

    # Visualize examples
    print(f"\nVisualizing {n_examples} training examples...")
    indices = np.random.choice(
        len(dataset), size=min(n_examples, len(dataset)), replace=False
    )

    for i, idx in enumerate(indices):
        print(f"\n[{i + 1}/{n_examples}] Processing example {idx}...")
        try:
            visualize_training_example(
                dataset, idx, model, device, output_dir, show_segmentation_labels
            )
        except Exception as e:
            print(f"  ✗ Error processing example {idx}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"✓ Figures saved to: {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize training set examples")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/multitask_unet.pth",
        help="Path to trained model",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional path to checkpoint/weights; only the model weights are loaded",
    )
    parser.add_argument(
        "--n_examples", type=int, default=2, help="Number of examples to visualize"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures/training_visualizations",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--length", type=int, default=512, help="Kymograph length (time dimension)"
    )
    parser.add_argument(
        "--width", type=int, default=512, help="Kymograph width (space dimension)"
    )
    parser.add_argument(
        "--max_trajectories", type=int, default=3, help="Maximum number of trajectories"
    )
    parser.add_argument(
        "--no_segmentation",
        action="store_true",
        help="Don't overlay ground-truth trajectories",
    )

    args = parser.parse_args()

    visualize_training_set(
        model_path=args.model_path,
        n_examples=args.n_examples,
        output_dir=args.output_dir,
        dataset_length=args.length,
        dataset_width=args.width,
        max_trajectories=args.max_trajectories,
        show_segmentation_labels=not args.no_segmentation,
        weights_path=args.weights,
    )
