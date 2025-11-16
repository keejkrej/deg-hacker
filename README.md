# Kymo-Tracker: Multi-Particle Kymograph Denoising and Tracking

A modular deep learning toolkit for denoising and tracking multiple particles in kymograph data. The project ships with a DDPM-style denoiser, an attention-based locator that predicts per-track center/width trajectories, and classical tracking utilities for physics-aligned analysis.

## Features

- **Modular package**: `kymo_tracker/` contains models, datasets, training, inference, and utilities ready for reuse.
- **U-Net Denoising**: Lightweight DDPM-style U-Net predicts noise residuals on 512(space) × 16(time) strips.
- **Temporal Locator**: CNN + 1D ViT head regresses per-track center/width envelopes instead of dense heatmaps.
- **Classical Tracking**: Otsu binarization, connected components, DBSCAN clustering, and greedy assignment with overlap prevention.
- **Comprehensive Analysis**: Utilities for diffusion/contrast/noise estimation, parameter sweeps, and visualizations.
- **Typer CLI**: `main.py` exposes `train` and `infer` commands for repeatable experiments.

## Installation

```bash
git clone <repository-url>
cd kymo-tracker
conda create -n kymo python=3.13
conda activate kymo
pip install -r requirements.txt
```

## CLI Quick Start

Train the multi-task model on synthetic data (checkpoints saved to `checkpoints/`, final weights under `artifacts/`):

```bash
python src/main.py train --samples 4096 --epochs 4 --checkpoint-dir checkpoints
```

Run inference on a saved kymograph (`.npy` file shaped `[time, width]`):

```bash
python src/main.py infer artifacts/multitask_unet.pth data/sample_kymo.npy --output-dir runs/demo
```

Outputs include `denoised.npy`, `centers.npy`, and `widths.npy` for downstream analysis.

## Using the Python API

```python
from kymo_tracker.data.multitask_dataset import MultiTaskDataset
from kymo_tracker.training.multitask import MultiTaskConfig, train_multitask_model
from kymo_tracker.utils.tracking import analyze_multi_particle

# Train programmatically
dataset = MultiTaskDataset(n_samples=1024, window_length=16)
config = MultiTaskConfig(epochs=6, batch_size=16, checkpoint_dir="checkpoints")
model = train_multitask_model(config, dataset)

# Classical analysis utilities remain available
metrics = analyze_multi_particle(
    radii_nm=[5.0, 10.0],
    contrasts=[0.7, 0.5],
    noise_level=0.3,
)
```

## Project Structure

```
kymo-tracker/
├── src/
│   ├── kymo_tracker/
│   │   ├── data/           # Datasets and target builders
│   │   ├── inference/      # Prediction utilities + visualizers
│   │   ├── models/         # Neural network definitions
│   │   ├── training/       # Training loops + configs
│   │   └── utils/          # Analysis, tracking, helper functions
│   └── main.py             # Typer CLI (train / infer)
├── baseline/               # Classical baselines
├── tests/                  # Comprehensive test suite
├── requirements.txt
└── pyproject.toml
```

## Key Algorithms

### Denoiser
- **Architecture**: Tiny U-Net with three resolution levels and optional dropout.
- **Objective**: Predict additive noise (DDPM-style) to recover denoised strips.
- **Chunking**: `denoise_and_segment_chunked` blends overlapping windows for arbitrarily long kymographs.

### Locator
- **Tokens**: Spatial encoder averages each column into a token; ViT layers reason over all 16 frames.
- **Outputs**: Each track channel predicts `(center, width)` per frame (normalized), converted to pixel corridors.
- **Classical Post-processing**: Within each corridor, traditional peak/CoM finder keeps trajectories faithful to the raw signal.

### Tracking & Analysis
- **Detection**: Otsu thresholding + connected components + DBSCAN merging.
- **Assignment**: Greedy mapping with explicit overlap prevention and crossing detection.
- **Metrics**: Diffusion, radius, contrast, and noise estimations plus CSV/figure exports.

## Testing

```bash
pytest tests/
```

## Notes

- Default checkpoints now live under `checkpoints/`; trained weights under `artifacts/` by convention.
- Core utilities reside in `kymo_tracker/utils`; please avoid modifying helper internals in `helpers.py` unless necessary.
- Activate the provided conda environment (`conda activate kymo`) before running CLI commands.

## Citation

If this project helps your research, please cite it appropriately.
