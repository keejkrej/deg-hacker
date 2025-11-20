"""Stage 2: Train the deep learning model."""

from pathlib import Path
import sys

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from kymo_tracker.data.multitask_dataset import MultiTaskDataset
from kymo_tracker.deeplearning.training.multitask import (
    MultiTaskConfig,
    train_multitask_model,
    save_multitask_model,
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
        default=8192,
        help="Number of training samples",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
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
        window_length=16,
        length=512,
        width=512,
        radii_nm=(3.0, 70.0),
        contrast=(0.5, 1.1),
        noise_level=(0.08, 0.8),
        multi_trajectory_prob=1.0,
        max_trajectories=3,
        mask_peak_width_samples=10.0,
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
