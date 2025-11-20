# Dimensions Used in Kymo-Tracker Project

This document lists all dimension specifications used throughout the project.

## Kymograph Dimensions (Time × Space)

### Demo/Production
- **Demo test cases**: `512 × 512` (time × space pixels)
  - Generated in `demo/generate_data.py` and `demo/run_demo.py`
  - Sliced into `512 × 16` windows for neural network processing

### Training
- **Training windows**: `16 × 512` (time × space pixels)
  - Generated directly in `MultiTaskDataset`
  - Input shape: `(1, 16, 512)` tensors
  - No cropping needed during training

### Default/Helper Functions
- **Default in `generate_kymograph()`**: `16 × 512` (time × space)
  - Used as default parameters in `src/kymo_tracker/utils/helpers.py`
  - Matches training window size

### Analysis Functions
- **Analysis functions**: `16 × 512` (time × space)
  - Used in `src/kymo_tracker/utils/analysis.py`
  - Matches training window size

### Tests
- **Test datasets**: Various sizes
  - `100 × 128` (small tests)
  - `256 × 256` (medium tests)
  - `128 × 128` (integration tests)

## Neural Network Processing

### Chunk/Window Sizes
- **Chunk size**: `16` time frames
  - Used in `denoise_and_segment_chunked()` for inference
  - Default: `chunk_size=16`
- **Window length**: `16` time frames
  - Training window size: `window_length=16`
- **Overlap**: `8` time frames
  - Overlap between chunks during inference: `overlap=8`
  - Ensures smooth trajectory linking

### Model Architecture Dimensions
- **Token dimension**: `128`
  - `locator_token_dim=128` in TemporalLocator
- **Max tokens**: `512`
  - `max_tokens=512` in TemporalLocator
- **Max trajectories**: `3`
  - Maximum number of particles tracked simultaneously
  - `max_tracks=3` in MultiTaskUNet

### Convolutional Layers
- **Conv kernel size**: `3 × 3`
  - Standard convolutional layers
- **MaxPool kernel**: `(1, 2)` stride `(1, 2)`
  - Spatial downsampling only (not temporal)

## Batch Sizes

- **Demo training**: `16`
  - `batch_size=16` in `demo/train_model.py`
- **Main CLI default**: `32`
  - `batch_size=32` in `src/main.py`
- **Tests**: `2`, `4`, `8`
  - Various batch sizes for testing

## Filtering Dimensions

### Median Filter Kernels
- **Demo classical**: `(11, 11)`
  - `median_kernel=(11, 11)` in `demo/run_classical.py`
  - Better for high noise cases
- **Previous default**: `(5, 1)`
  - Was used before, changed to `(11, 11)`
- **Noise estimation**: `(5, 3)`
  - `kernel_size=(5, 3)` in `estimate_noise_and_contrast()`

### Component Filtering
- **Min component size (demo)**: `50` pixels
  - `min_component_size=50` in classical pipeline
- **Min component size (tracking)**: `20` pixels
  - Used in `filter_small_clusters()`
- **Previous default**: `8` pixels
  - Was used before, increased to filter noise

### Other Filter Sizes
- **Smoothing kernel**: `3`
  - `kernel_size=3` for instance mask smoothing
- **Max hole size**: `10` pixels
  - `max_hole_size=10` for hole filling

## Dataset Parameters

### Training Dataset
- **Number of samples (demo)**: `1024`
  - Reduced for faster demo training
- **Number of samples (production)**: `4096` (default in CLI)
- **Spatial width**: `512` pixels
- **Temporal window**: `16` frames
- **Peak width**: `2.0` samples
  - `mask_peak_width_samples=2.0`

## Visualization Dimensions

### Figure Sizes
- **Comparison plots**: `(16, 8)` inches
  - `figsize=(16, 8)` in `visualize_comparison()`
- **Analysis plots**: `(12, 10)` inches
  - Used in tracking visualization
- **Small plots**: `(8, 7)` inches
  - Used in analysis functions
- **Training visualization**: `(4 * n_cols, 8)` inches
  - Dynamic width based on number of columns

## Spatial Sampling

- **dx (spatial step)**: `0.5` micrometers
  - Default spatial sampling resolution
- **dt (temporal step)**: `1.0` milliseconds
  - Default temporal sampling resolution

## Summary Table

| Dimension Type | Value | Usage |
|---------------|-------|-------|
| Demo kymograph | 512 × 512 | Full test cases |
| Training window | 16 × 512 | Neural network input |
| Default kymograph | 16 × 512 | Helper defaults |
| Analysis kymograph | 16 × 512 | Analysis functions |
| Chunk size | 16 | Inference processing |
| Overlap | 8 | Chunk overlap |
| Spatial width | 512 | Most common |
| Temporal length | 512 | Demo |
| Temporal length | 16 | Training windows, defaults, analysis |
| Batch size (demo) | 16 | Demo training |
| Batch size (default) | 32 | Production |
| Max trajectories | 3 | Simultaneous tracking |
| Token dim | 128 | Model architecture |
| Max tokens | 512 | Model architecture |
| Median kernel | (11, 11) | Classical filtering |
| Min component | 50 | Noise filtering |
