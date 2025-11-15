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
from pathlib import Path
from typing import Optional, Tuple

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
        denoised, segmentation_labels = denoise_and_segment_chunked(
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
    axes[0, 0].imshow(noisy.T, aspect="auto", origin="lower", 
                      vmin=vmin_noisy, vmax=vmax_noisy, cmap="gray")
    axes[0, 0].set_title("Noisy Input")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Position")
    
    axes[0, 1].imshow(gt_denoised.T, aspect="auto", origin="lower",
                      vmin=vmin_denoised, vmax=vmax_denoised, cmap="gray")
    axes[0, 1].set_title("Ground Truth (Denoised)")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Position")
    
    axes[0, 2].imshow(denoised.T, aspect="auto", origin="lower",
                      vmin=vmin_denoised, vmax=vmax_denoised, cmap="gray")
    axes[0, 2].set_title("Model Prediction (Denoised)")
    axes[0, 2].set_xlabel("Time")
    axes[0, 2].set_ylabel("Position")
    
    # Row 2: Segmentation masks (binary)
    # Ground truth binary mask
    im1 = axes[1, 0].imshow(true_mask.T, aspect="auto", origin="lower",
                           cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1, 0].set_title("Ground Truth Binary Mask")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Position")
    plt.colorbar(im1, ax=axes[1, 0], label="Mask (0=bg, 1=particle)")
    
    # Predicted binary mask
    im2 = axes[1, 1].imshow(segmentation_labels.T, aspect="auto", origin="lower",
                           cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1, 1].set_title("Predicted Binary Mask")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Position")
    plt.colorbar(im2, ax=axes[1, 1], label="Mask (0=bg, 1=particle)")
    
    # Error/difference
    if show_segmentation_labels:
        # Show binary mask error
        error_mask = np.abs(true_mask - segmentation_labels)
        im3 = axes[1, 2].imshow(error_mask.T, aspect="auto", origin="lower",
                               cmap="Reds", vmin=0, vmax=1)
        axes[1, 2].set_title("Segmentation Error\n(Red = Mismatch)")
        axes[1, 2].set_xlabel("Time")
        axes[1, 2].set_ylabel("Position")
        plt.colorbar(im3, ax=axes[1, 2], label="Absolute Error")
    else:
        # Show denoising error
        denoising_error = np.abs(gt_denoised - denoised)
        im3 = axes[1, 2].imshow(denoising_error.T, aspect="auto", origin="lower",
                               cmap="hot", vmin=0)
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
    print(f"    Denoising RMSE: {np.sqrt(np.mean((gt_denoised - denoised)**2)):.4f}")
    
    # Segmentation statistics (binary)
    # Threshold predicted mask at 0.5 for binary comparison
    pred_binary = (segmentation_labels > 0.5).astype(np.float32)
    true_binary = (true_mask > 0.5).astype(np.float32)
    
    correct = (true_binary == pred_binary).sum()
    total = true_mask.size
    accuracy = correct / total
    print(f"    Segmentation accuracy: {accuracy:.3f} ({correct}/{total})")
    
    # Binary classification metrics
    true_positives = ((true_binary == 1) & (pred_binary == 1)).sum()
    false_positives = ((true_binary == 0) & (pred_binary == 1)).sum()
    false_negatives = ((true_binary == 1) & (pred_binary == 0)).sum()
    true_negatives = ((true_binary == 0) & (pred_binary == 0)).sum()
    
    if (true_positives + false_positives) > 0:
        precision = true_positives / (true_positives + false_positives)
        print(f"    Precision: {precision:.3f}")
    
    if (true_positives + false_negatives) > 0:
        recall = true_positives / (true_positives + false_negatives)
        print(f"    Recall: {recall:.3f}")
    
    print(f"    Mask coverage: {100*np.mean(true_binary):.1f}% (GT), {100*np.mean(pred_binary):.1f}% (Pred)")


def visualize_training_set(
    model_path: str = "models/multitask_unet.pth",
    n_examples: int = 10,
    output_dir: str = "figures/training_visualizations",
    dataset_length: int = 4000,
    dataset_width: int = 256,
    max_trajectories: int = 3,
    show_segmentation_labels: bool = True,
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
    
    # Check model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model
    device = _default_device()
    print(f"\nLoading model: {model_path}")
    print(f"Device: {device}")
    model = load_multitask_model(model_path, device=device, max_tracks=max_trajectories)
    model.eval()
    
    # Create dataset
    print(f"\nCreating training dataset...")
    print(f"  Length: {dataset_length}, Width: {dataset_width}")
    print(f"  Max trajectories: {max_trajectories}")
    dataset = MultiTaskDataset(
        length=dataset_length,
        width=dataset_width,
        max_trajectories=max_trajectories,
    )
    
    # Visualize examples
    print(f"\nVisualizing {n_examples} training examples...")
    indices = np.random.choice(len(dataset), size=min(n_examples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        print(f"\n[{i+1}/{n_examples}] Processing example {idx}...")
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
    parser.add_argument("--model_path", type=str, default="models/multitask_unet.pth",
                       help="Path to trained model")
    parser.add_argument("--n_examples", type=int, default=10,
                       help="Number of examples to visualize")
    parser.add_argument("--output_dir", type=str, default="figures/training_visualizations",
                       help="Output directory for figures")
    parser.add_argument("--length", type=int, default=4000,
                       help="Kymograph length (time dimension)")
    parser.add_argument("--width", type=int, default=256,
                       help="Kymograph width (space dimension)")
    parser.add_argument("--max_trajectories", type=int, default=3,
                       help="Maximum number of trajectories")
    parser.add_argument("--no_segmentation", action="store_true",
                       help="Don't show segmentation labels (show denoising error instead)")
    
    args = parser.parse_args()
    
    visualize_training_set(
        model_path=args.model_path,
        n_examples=args.n_examples,
        output_dir=args.output_dir,
        dataset_length=args.length,
        dataset_width=args.width,
        max_trajectories=args.max_trajectories,
        show_segmentation_labels=not args.no_segmentation,
    )
