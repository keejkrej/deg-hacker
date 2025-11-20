# Demo: Classical vs Deep Learning Comparison

This demo script automatically trains a model (if needed) and generates 5 comparison plots showing classical and deep learning approaches side-by-side.

## Quick Start

Simply run:

```bash
./demo/run.sh
```

Or manually:

```bash
uv run python demo/run_demo.py
```

## What It Does

1. **Trains a model** (if `artifacts/demo_model.pth` doesn't exist)
   - Uses lightweight training: 1024 samples, 3 epochs
   - Suitable for systems with limited RAM (40GB) and VRAM (24GB)

2. **Generates 5 test cases**:
   - Case 1: Single particle, low noise
   - Case 2: Single particle, high noise
   - Case 3: Two particles, moderate noise
   - Case 4: Three particles, moderate noise
   - Case 5: Two particles, high noise

3. **Runs both pipelines**:
   - Classical: Median filter → Thresholding → Segmentation → Trajectory extraction
   - Deep Learning: U-Net denoising → Locator → Mask creation → Trajectory extraction

4. **Creates comparison plots**:
   Each plot shows 4 subplots per method:
   - Noisy input kymograph
   - Denoised/filtered result
   - Segmentation mask
   - Predicted trajectories (with ground truth overlay)

## Output

All plots are saved to `demo/results/`:
- `comparison_case_01_single_particle_(low_noise).png`
- `comparison_case_02_single_particle_(high_noise).png`
- `comparison_case_03_two_particles_(moderate_noise).png`
- `comparison_case_04_three_particles_(moderate_noise).png`
- `comparison_case_05_two_particles_(high_noise).png`

## Files

- `run.sh` - Shell script wrapper (run all stages or individual stages)
- `generate_data.py` - Stage 1: Generate synthetic test cases
- `train_model.py` - Stage 2: Train the deep learning model
- `run_classical.py` - Stage 3: Run classical inference pipeline
- `run_deeplearning.py` - Stage 4: Run deep learning inference pipeline
- `visualize.py` - Stage 5: Create comparison visualization plots
- `run_demo.py` - Legacy all-in-one script (deprecated, use `run.sh` instead)

## Usage

### Run All Stages

```bash
./demo/run.sh
```

This will:
1. Generate 5 test cases
2. Train the model (skips if already exists)
3. Run classical inference
4. Run deep learning inference
5. Create comparison plots

### Run Individual Stages

You can run specific stages by passing stage numbers or names:

```bash
# Generate data only
./demo/run.sh 1

# Train model only
./demo/run.sh 2

# Run inference stages only
./demo/run.sh 3 4

# Create visualizations only (requires previous stages)
./demo/run.sh 5

# Use stage names
./demo/run.sh generate_data train_model
```

### Custom Paths

You can customize directories and paths:

```bash
./demo/run.sh --data-dir my_data --model-path my_model.pth --output-dir my_results
```

### Help

```bash
./demo/run.sh --help
```

## Stage Details

### Stage 1: Generate Data (`generate_data.py`)
- Creates 5 synthetic test cases with different scenarios
- Saves kymographs as `.npy` files
- Saves metadata as `cases_metadata.json`

### Stage 2: Train Model (`train_model.py`)
- Trains the multi-task U-Net model
- Uses lightweight settings for demo (1024 samples, 3 epochs)
- Saves model to `artifacts/demo_model.pth` by default
- Supports `--skip-if-exists` to avoid retraining

### Stage 3: Classical Pipeline (`run_classical.py`)
- Runs median filter + Otsu thresholding
- Extracts trajectories using classical methods
- Saves results to `demo/results/classical/`

### Stage 4: Deep Learning Pipeline (`run_deeplearning.py`)
- Runs U-Net denoising + temporal locator
- Extracts trajectories using mask-based argmax
- Saves results to `demo/results/deeplearning/`

### Stage 5: Visualize (`visualize.py`)
- Creates 2×4 comparison plots
- Compares classical vs deep learning approaches
- Saves plots to `demo/results/`

## Integrated Functions

The demo uses functions integrated into the main `kymo_tracker` package:

- `kymo_tracker.deeplearning.predict.create_mask_from_centers_widths()` - Creates segmentation masks from locator predictions
- `kymo_tracker.deeplearning.predict.extract_trajectories_from_mask()` - Extracts trajectories from masks
- `kymo_tracker.utils.visualization.visualize_comparison()` - Creates comparison plots
