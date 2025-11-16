"""
Visualize Training Set Examples

Processes and visualizes examples from the training dataset to:
1. Show model performance on training data
2. Visualize denoising and segmentation outputs
3. Compare ground truth vs predictions
4. Display segmentation labels for different tracks
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import colormaps
from pathlib import Path
from typing import Optional, Tuple
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
        Whether to show segmentation class labels
    """
    # Get training example
    noisy_tensor, noise_tensor, mask_tensor = dataset[index]
    noisy = noisy_tensor.squeeze().numpy()
    true_noise = noise_tensor.squeeze().numpy()
    true_mask = mask_tensor.squeeze().numpy()

    # Get ground truth denoised
    gt_denoised = noisy - true_noise

    # Run inference
    model.eval()
    with torch.no_grad():
        # For 512x512, no chunking needed (fits in one chunk)
        denoised, embeddings = denoise_and_segment_chunked(
            model, noisy, device=device, chunk_size=512, overlap=64
        )

    # Create visualization
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Color maps
    vmin_noisy = np.percentile(noisy, 1)
    vmax_noisy = np.percentile(noisy, 99)
    vmin_denoised = np.percentile(gt_denoised, 1)
    vmax_denoised = np.percentile(gt_denoised, 99)

    # Row 1: Input, Ground Truth, Denoised
    axes[0, 0].imshow(
        noisy.T,
        aspect="auto",
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
        aspect="auto",
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
        aspect="auto",
        origin="lower",
        vmin=vmin_denoised,
        vmax=vmax_denoised,
        cmap="gray",
    )
    axes[0, 2].set_title("Model Prediction (Denoised)")
    axes[0, 2].set_xlabel("Time")
    axes[0, 2].set_ylabel("Position")

    # Row 2: Embeddings visualization
    # Ground truth multi-instance mask
    max_instance_id = int(true_mask.max())
    # Use a discrete colormap for instance segmentation
    # Background (0) will be black, instances will have distinct colors

    # Create colormap: black for background, distinct colors for instances
    if max_instance_id > 0:
        # Use tab10 colormap for instances (has 10 distinct colors)
        # Get enough colors for all instances
        base_cmap = colormaps.get_cmap("tab10")
        base_colors = base_cmap(np.linspace(0, 1, 10))
        colors = ["black"]  # Background is black
        for i in range(max_instance_id):
            colors.append(base_colors[i % len(base_colors)])
        cmap = ListedColormap(colors)
        vmax = max_instance_id
    else:
        # No instances, just background
        cmap = "gray"
        vmax = 1

    im1 = axes[1, 0].imshow(
        true_mask.T,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )
    axes[1, 0].set_title("Ground Truth Multi-Instance Mask")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Position")
    if max_instance_id > 0:
        plt.colorbar(
            im1,
            ax=axes[1, 0],
            label="Instance ID (0=bg, 1..N=instances)",
            ticks=range(max_instance_id + 1),
        )
    else:
        plt.colorbar(im1, ax=axes[1, 0], label="Mask (0=bg)")

    # Visualize 2D embeddings as color
    embedding_dim = embeddings.shape[2]

    if embedding_dim == 2:
        # 2D embeddings: convert to color using HSV
        emb_x = embeddings[:, :, 0]
        emb_y = embeddings[:, :, 1]

        # Angle -> Hue, Magnitude -> Saturation
        angle = np.arctan2(emb_y, emb_x)
        magnitude = np.sqrt(emb_x**2 + emb_y**2)

        hue = (angle + np.pi) / (2 * np.pi)  # [0, 1]

        mag_min, mag_max = np.percentile(magnitude, [1, 99])
        if mag_max > mag_min:
            saturation = np.clip((magnitude - mag_min) / (mag_max - mag_min), 0, 1)
        else:
            saturation = np.ones_like(magnitude) * 0.5

        from matplotlib.colors import hsv_to_rgb

        hsv = np.stack([hue, saturation, np.ones_like(hue)], axis=2)
        emb_rgb = hsv_to_rgb(hsv)  # Shape: (time, width, 3)

        # Transpose spatial dimensions only: (time, width, 3) -> (width, time, 3)
        emb_rgb_display = np.transpose(emb_rgb, (1, 0, 2))
        im2 = axes[1, 1].imshow(emb_rgb_display, aspect="auto", origin="lower")
        axes[1, 1].set_title("Embeddings (2D → Color)")
    elif embedding_dim >= 3:
        # Use first 3 dimensions as RGB
        emb_rgb = np.zeros((embeddings.shape[0], embeddings.shape[1], 3))
        for i in range(3):
            emb_dim = embeddings[:, :, i]
            vmin, vmax = np.percentile(emb_dim, [1, 99])
            if vmax > vmin:
                emb_rgb[:, :, i] = np.clip((emb_dim - vmin) / (vmax - vmin), 0, 1)
            else:
                emb_rgb[:, :, i] = 0.5
        # Transpose spatial dimensions only: (time, width, 3) -> (width, time, 3)
        emb_rgb_display = np.transpose(emb_rgb, (1, 0, 2))
        im2 = axes[1, 1].imshow(emb_rgb_display, aspect="auto", origin="lower")
        axes[1, 1].set_title(
            f"Embeddings (First 3 dims as RGB)\nTotal dims: {embedding_dim}"
        )
        # No colorbar for RGB images
    else:
        # Single dimension: show as grayscale
        emb_vis = embeddings[:, :, 0]
        vmin, vmax = np.percentile(emb_vis, [1, 99])
        im2 = axes[1, 1].imshow(
            emb_vis.T,
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        axes[1, 1].set_title(f"Embeddings (Dim 0)\nTotal dims: {embedding_dim}")
        plt.colorbar(im2, ax=axes[1, 1])  # Colorbar only for scalar images
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Position")

    # Show embedding magnitude or PCA visualization
    if show_segmentation_labels:
        # Compute embedding magnitude (L2 norm)
        emb_magnitude = np.linalg.norm(embeddings, axis=2)
        vmin, vmax = np.percentile(emb_magnitude, [1, 99])
        im3 = axes[1, 2].imshow(
            emb_magnitude.T,
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="hot",
        )
        axes[1, 2].set_title("Embedding Magnitude\n(L2 norm)")
        axes[1, 2].set_xlabel("Time")
        axes[1, 2].set_ylabel("Position")
        plt.colorbar(im3, ax=axes[1, 2], label="Magnitude")
    else:
        # Show denoising error
        denoising_error = np.abs(gt_denoised - denoised)
        im3 = axes[1, 2].imshow(
            denoising_error.T, aspect="auto", origin="lower", cmap="hot", vmin=0
        )
        axes[1, 2].set_title("Denoising Error\n(Ground Truth - Prediction)")
        axes[1, 2].set_xlabel("Time")
        axes[1, 2].set_ylabel("Position")
        plt.colorbar(im3, ax=axes[1, 2], label="Absolute Error")

    plt.tight_layout()

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

    # Embedding statistics
    print(f"    Embedding shape: {embeddings.shape}")
    print(f"    Embedding dims: {embeddings.shape[2]}")
    print(f"    Embedding range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
    print(f"    Embedding mean: {embeddings.mean():.3f}, std: {embeddings.std():.3f}")

    # Ground truth mask statistics
    unique_instances = np.unique(true_mask)
    n_instances = len(unique_instances[unique_instances > 0])  # Exclude background (0)
    print(f"    GT instances: {n_instances}")
    if n_instances > 0:
        instance_ids = unique_instances[unique_instances > 0]
        for inst_id in instance_ids:
            coverage = np.mean(true_mask == inst_id) * 100
            print(f"      Instance {int(inst_id)}: {coverage:.1f}% coverage")
    bg_coverage = np.mean(true_mask == 0) * 100
    print(f"    GT background coverage: {bg_coverage:.1f}%")


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
        Whether to show segmentation class labels
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
        help="Don't show segmentation labels (show denoising error instead)",
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
