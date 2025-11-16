# Training Guide

## Quick Start (CLI)

```bash
python src/main.py train --samples 4096 --epochs 4 --checkpoint-dir checkpoints
```

This command synthesizes a dataset, trains the multi-task model, stores checkpoints under `checkpoints/`, and saves the final weights to `artifacts/multitask_unet.pth`.

## Programmatic Usage

```python
from kymo_tracker.data.multitask_dataset import MultiTaskDataset
from kymo_tracker.training.multitask import (
    MultiTaskConfig,
    train_multitask_model,
    save_multitask_model,
)

dataset = MultiTaskDataset(
    n_samples=2048,
    window_length=16,
    width=512,
    length=512,
    multi_trajectory_prob=0.5,
)
config = MultiTaskConfig(
    epochs=12,
    batch_size=16,
    learning_rate=1e-3,
    locator_loss_weight=2.0,
    checkpoint_dir="checkpoints",
)
model = train_multitask_model(config, dataset)
save_multitask_model(model, "artifacts/multitask_unet.pth")
```

## Resuming Training

`MultiTaskConfig` supports `resume_from`, `init_weights`, and `auto_resume`. For example:

```python
config = MultiTaskConfig(
    epochs=20,
    checkpoint_dir="checkpoints",
    auto_resume=True,
)
```

The trainer automatically searches `checkpoint_dir` for the latest `checkpoint_epoch_*.pth` and resumes training if present.

## Configuration Notes

- `locator_center_weight` / `locator_width_weight` balance the two regression terms inside the locator loss.
- Set `auto_balance_losses=False` if you prefer manual weighting between denoising and locator losses.
- Checkpoints are no longer stored under `models/`; default to `checkpoints/` to keep artifacts separate from code.
