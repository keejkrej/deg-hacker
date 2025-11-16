# DEG-Hacker: Multi-Particle Kymograph Denoising and Tracking

A deep learning pipeline for denoising and tracking multiple particles in kymograph data using U-Net architecture.

## Features

- **U-Net Denoising**: 2D U-Net model trained to predict noise (DDPM-style) for kymograph denoising
- **Multi-Particle Tracking**: Robust tracking of 2-3 particles simultaneously
- **Crossing Detection**: Automatic detection and exclusion of track crossing events
- **Parameter Estimation**: Estimates diffusion coefficient, particle radius, contrast, and noise level
- **Comprehensive Analysis**: Generates metrics, visualizations, and CSV reports

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd deg-hacker
```

2. Create and activate conda environment:
```bash
conda create -n kymo python=3.13
conda activate kymo
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Single Particle Analysis

```python
from single_particle_unet import analyze_particle

# Analyze a single particle
metrics = analyze_particle(p=5.0, c=0.7, n=0.3)
```

### Multi-Particle Analysis

```python
from utils.tracking import analyze_multi_particle

# Analyze 2 particles
metrics = analyze_multi_particle(
    radii_nm=[5.0, 10.0],
    contrasts=[0.7, 0.5],
    noise_level=0.3
)
```

### Training a New Model

```python
from denoiser import train_denoiser, TrainingConfig, SyntheticKymographDataset

dataset = SyntheticKymographDataset(n_samples=1000, length=512, width=512)
config = TrainingConfig(
    epochs=12,
    batch_size=10,
    lr=1e-3,
    use_residual_connection=True,
    use_lr_scheduler=True,
)
model = train_denoiser(config, dataset)
```

## Project Structure

```
deg-hacker/
├── denoiser.py              # U-Net model definition and training
├── single_particle_unet.py     # Single-particle analysis pipeline
├── multi_particle_unet.py   # Multi-particle tracking and analysis
├── helpers.py               # Utility functions (do not modify)
├── utils.py                 # Analysis utilities and data structures
├── tests/                   # Comprehensive test suite
├── models/                  # Trained model checkpoints
├── figures/                 # Generated analysis figures
├── metrics/                 # CSV metrics output
└── requirements.txt         # Python dependencies
```

## Key Algorithms

### Denoising
- **Architecture**: Tiny U-Net with 3 resolution levels, batch normalization
- **Training**: DDPM-style noise prediction (predicts noise to subtract)
- **Loss**: L2 loss on predicted vs. true noise
- **Processing**: Chunked processing for large kymographs (>512x512)

### Tracking
- **Detection**: Otsu binarization + connected components + DBSCAN clustering
- **Assignment**: Greedy assignment with explicit overlap prevention
- **Crossing Detection**: Automatic detection and exclusion of crossing events
- **Separation**: Minimum separation enforcement to prevent track overlaps

### Parameter Estimation
- **Diffusion**: MSD (Mean Squared Displacement) fitting
- **Radius**: Stokes-Einstein equation from diffusion coefficient
- **Contrast**: Per-track intensity analysis from denoised kymograph
- **Noise**: Global estimation from noisy kymograph residuals

## Usage Examples

### Run Parameter Grid Analysis

```python
from utils.tracking import run_parameter_grid

# Test multiple configurations
run_parameter_grid(
    particle_configs=[
        ([2.5, 5.0], [0.8, 0.6]),  # 2 particles
        ([5.0, 10.0, 8.0], [0.7, 0.5, 1.0]),  # 3 particles
    ],
    noise_levels=[0.1, 0.3, 0.5],
    csv_path="metrics/multi_particle_unet.csv"
)
```

### Visualize Denoising Results

After training, denoising results are automatically visualized. You can also call:

```python
from denoiser import visualize_denoising_results, load_model

model = load_model("models/tiny_unet_denoiser.pth")
visualize_denoising_results(model, n_samples=3)
```

Multi-particle analysis automatically generates diagnostic plots in `figures/multi_unet/`.

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_denoiser.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Configuration

### Training Configuration

Key parameters in `TrainingConfig`:
- `epochs`: Number of training epochs (default: 12)
- `batch_size`: Batch size (default: 10, adjust for VRAM)
- `base_channels`: U-Net base channels (default: 56, affects model capacity)
- `use_residual_connection`: Enable residual connections (default: True)
- `use_lr_scheduler`: Enable learning rate scheduling (default: True)

### Tracking Configuration

Key parameters in `track_particles()`:
- `max_jump`: Maximum allowed jump between frames (default: 15 pixels)
- `detect_crossings`: Enable crossing detection (default: True)
- `crossing_threshold`: Distance threshold for crossings (default: 5.0 pixels)
- `crossing_padding`: Frames to exclude around crossings (default: 2)

## Output

### Metrics CSV
Analysis results are saved to `metrics/` directory with columns:
- `method_label`: Processing method
- `particle_radius_nm`: True particle radius
- `diffusion_true`, `diffusion_processed`: True and estimated diffusion
- `radius_true`, `radius_processed`: True and estimated radius
- `contrast`, `noise_level`: True values
- `contrast_estimate`, `noise_estimate`: Estimated values
- `figure_path`: Path to diagnostic figure

### Figures
Diagnostic figures saved to `figures/` showing:
- Noisy and denoised kymographs
- True and estimated particle tracks
- Comparison plots

## Notes

- **Do not modify `helpers.py`**: Core utility functions are maintained separately
- **Use conda `kymo` environment**: `conda activate kymo` before running
- **Model checkpoints**: Trained models saved in `models/` directory
- **Crossing events**: Tracks are automatically excluded after crossing to prevent ambiguous data

## Citation

If you use this code, please cite appropriately.

## License

[Add your license here]
