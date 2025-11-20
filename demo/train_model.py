"""Stage 2: Train the deep learning model."""

from pathlib import Path
import sys

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import torch
import warnings

# Disable cuDNN backend (maps to MIOpen on ROCm) to force PyTorch native BatchNorm
# This avoids MIOpen kernel compilation issues on certain GPU architectures
# Only disable on ROCm (AMD GPUs), not on NVIDIA CUDA where cuDNN is beneficial
if (hasattr(torch.version, "hip") or hasattr(torch.backends, "miopen")) and hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.enabled = False
    warnings.warn(
        "cuDNN disabled (ROCm detected). Using PyTorch native BatchNorm to avoid MIOpen compilation issues. "
        "Performance may be slower than optimized MIOpen kernels.",
        UserWarning,
        stacklevel=2
    )

from kymo_tracker.data.multitask_dataset import MultiTaskDataset
from kymo_tracker.deeplearning.training.multitask import (
    MultiTaskConfig,
    train_multitask_model,
    save_multitask_model,
)
from kymo_tracker.deeplearning.training.config import (
    DEFAULT_EPOCHS,
    DEFAULT_SAMPLES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MASK_PEAK_WIDTH_SAMPLES,
    DEFAULT_WINDOW_LENGTH,
    DEFAULT_WIDTH,
    DEFAULT_RADII_NM,
    DEFAULT_CONTRAST,
    DEFAULT_NOISE_LEVEL,
    DEFAULT_MULTI_TRAJECTORY_PROB,
    DEFAULT_MAX_TRAJECTORIES,
)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train the deep learning model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/demo_model.pth",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for training checkpoints",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="Number of training samples",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip training if model already exists",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("STAGE 2: Training Model")
    print("=" * 70)
    
    model_path = Path(args.model_path)
    
    # Check if model exists
    if model_path.exists() and args.skip_if_exists:
        print(f"Model already exists at {model_path}, skipping training...")
        return
    
    print(f"Model will be saved to: {model_path}")
    print(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    print(f"Training samples: {args.n_samples}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    
    # Create dataset
    print("\nCreating training dataset...")
    dataset = MultiTaskDataset(
        n_samples=args.n_samples,
        window_length=DEFAULT_WINDOW_LENGTH,
        length=DEFAULT_WIDTH,
        width=DEFAULT_WIDTH,
        radii_nm=DEFAULT_RADII_NM,
        contrast=DEFAULT_CONTRAST,
        noise_level=DEFAULT_NOISE_LEVEL,
        multi_trajectory_prob=DEFAULT_MULTI_TRAJECTORY_PROB,
        max_trajectories=DEFAULT_MAX_TRAJECTORIES,
        mask_peak_width_samples=DEFAULT_MASK_PEAK_WIDTH_SAMPLES,
    )
    print(f"Dataset created with {len(dataset)} samples")
    
    # Create config
    config = MultiTaskConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=1.5e-3,
        checkpoint_dir=args.checkpoint_dir,
        auto_resume=False,  # Don't resume for demo
    )
    
    # Train model
    print("\nStarting training...")
    model = train_multitask_model(config, dataset)
    
    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_multitask_model(model, str(model_path))
    print(f"\nModel saved to {model_path}")
    
    print("\n" + "=" * 70)
    print("Stage 2 complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
