# Training Guide

## Quick Start

To train the multi-task U-Net model (denoising + segmentation):

```bash
python train/multitask_model.py
```

This will:
1. Create a synthetic dataset with 1024 samples
2. Train the model for 12 epochs
3. Save the model to `models/multitask_unet.pth`

## Customizing Training

You can modify the training script directly or create your own script:

```python
from train.multitask_model import (
    MultiTaskDataset,
    MultiTaskConfig,
    train_multitask_model,
    save_multitask_model,
)

# Create dataset
dataset = MultiTaskDataset(
    n_samples=2048,          # Number of training samples
    length=4000,             # Kymograph length (time dimension)
    width=256,               # Kymograph width (space dimension)
    multi_trajectory_prob=0.3,  # Probability of multi-particle examples
    max_trajectories=3,      # Maximum number of tracks
    mask_peak_width_samples=2.0,  # Segmentation mask width
)

# Training configuration
config = MultiTaskConfig(
    epochs=20,               # Number of training epochs
    batch_size=8,            # Batch size (adjust for GPU memory)
    learning_rate=1e-3,       # Learning rate
    denoise_loss_weight=1.0,  # Weight for denoising loss
    segment_loss_weight=1.0, # Weight for segmentation loss
    denoise_loss="l2",        # "l2" or "l1" for denoising
    segment_loss="ce",        # "ce" (CrossEntropy) or "dice" for segmentation
    use_gradient_clipping=True,
    max_grad_norm=1.0,
    use_lr_scheduler=True,
)

# Train
model = train_multitask_model(config, dataset)

# Save
save_multitask_model(model, "models/multitask_unet.pth")
```

## Resuming Training

Training automatically resumes from the latest checkpoint by default! Just set `checkpoint_dir`:

```python
config = MultiTaskConfig(
    epochs=20,
    checkpoint_dir="models/checkpoints",  # Auto-resumes from latest checkpoint
    # ... other config options
)

model = train_multitask_model(config, dataset)
```

The system will:
- **Auto-detect** the latest `checkpoint_epoch_*.pth` file (highest epoch number)
- Fall back to `best_model.pth` if no epoch checkpoints found
- Load model weights, optimizer, and scheduler state
- Resume training from the saved epoch

**Manual checkpoint selection:**
```python
config = MultiTaskConfig(
    epochs=20,
    resume_from="models/checkpoints/checkpoint_epoch_10.pth",  # Specific checkpoint
    checkpoint_dir="models/checkpoints",
    # ... other config options
)
```

**Disable auto-resume:**
```python
config = MultiTaskConfig(
    epochs=20,
    auto_resume=False,  # Train from scratch even if checkpoints exist
    checkpoint_dir="models/checkpoints",
    # ... other config options
)
```

**Note**: Checkpoints saved after this update include full training state (model, optimizer, scheduler, epoch, best_loss). Older checkpoints (model weights only) can still be loaded for inference but won't resume training state.

## Configuration Options

### Dataset Parameters (`MultiTaskDataset`)
- `n_samples`: Number of training examples to generate
- `length`: Kymograph time dimension (default: 512)
- `width`: Kymograph space dimension (default: 512)
- `radii_nm`: Particle radius range in nm (default: (3.0, 15.0))
- `contrast`: Contrast range (default: (0.5, 1.0))
- `noise_level`: Noise level range (default: (0.1, 0.5))
- `multi_trajectory_prob`: Probability of multi-particle examples (default: 0.3)
- `max_trajectories`: Maximum number of tracks (default: 3)
- `mask_peak_width_samples`: Segmentation mask width in pixels (default: 2.0)

### Training Parameters (`MultiTaskConfig`)
- `epochs`: Number of training epochs
- `batch_size`: Batch size (reduce if GPU memory is limited)
- `learning_rate`: Initial learning rate
- `denoise_loss_weight`: Weight for denoising loss component
- `segment_loss_weight`: Weight for segmentation loss component
- `denoise_loss`: "l2" (MSE) or "l1" (MAE)
- `segment_loss`: "ce" (CrossEntropy) or "dice" (Dice loss)
- `use_gradient_clipping`: Enable gradient clipping (default: True)
- `max_grad_norm`: Maximum gradient norm for clipping (default: 1.0)
- `use_lr_scheduler`: Enable learning rate scheduling (default: True)
- `resume_from`: Path to checkpoint file to resume training from (default: None, auto-detect if auto_resume=True)
- `resume_epoch`: Epoch number to resume from (if None, inferred from checkpoint)
- `auto_resume`: Automatically resume from latest checkpoint if available (default: True)

## Model Architecture

The `MultiTaskUNet` model:
- **Base channels**: 48 (configurable via `base_channels` parameter)
- **Output heads**:
  1. Denoising head: Predicts noise to subtract (DDPM-style)
  2. Segmentation head: Predicts multi-class labels (background + tracks)

## Output

The trained model will be saved to:
- `models/multitask_unet.pth`

Training progress is printed to console showing:
- Per-epoch losses (denoising, segmentation, total)
- Training time per epoch

## Tips

1. **GPU Memory**: If you run out of GPU memory, reduce `batch_size` or `length`/`width`
2. **Training Time**: Larger datasets (`n_samples`) and longer kymographs take more time
3. **Multi-particle**: Increase `multi_trajectory_prob` to train more on multi-particle cases
4. **Loss Balancing**: Adjust `denoise_loss_weight` and `segment_loss_weight` to balance tasks
