## Notes for Codex agents

- Do not modify `helpers.py`. If functionality is needed, add it elsewhere.
- Use conda `kymo` environment for Python: `conda activate kymo`
- Use `uv` for Python environment management: `uv sync` to install dependencies, `uv run` to execute commands

## Project Structure

```
kymo-tracker/
├── demo/                         # Demo scripts and test cases
│   ├── generate_data.py         # Stage 1: Generate synthetic test cases (512×512)
│   ├── train_model.py           # Stage 2: Train deep learning model
│   ├── run_classical.py         # Stage 3: Classical inference pipeline
│   ├── run_deeplearning.py      # Stage 4: Deep learning inference pipeline
│   ├── visualize.py             # Stage 5: Create comparison plots
│   ├── run.sh                   # Main demo script (runs stages 1-5)
│   ├── data/                    # Generated test case data (gitignored)
│   └── results/                 # Inference results and plots (gitignored)
├── src/
│   ├── kymo_tracker/           # Main package
│   │   ├── classical/          # Classical median/threshold pipeline
│   │   │   └── pipeline.py    # Median filter + thresholding implementation
│   │   ├── data/              # Dataset generation
│   │   │   └── multitask_dataset.py  # MultiTaskDataset (generates 16×512 windows)
│   │   ├── deeplearning/      # Deep learning modules
│   │   │   ├── models/        # Neural network architectures
│   │   │   │   └── multitask.py  # MultiTaskUNet (denoiser + locator)
│   │   │   ├── training/      # Training utilities
│   │   │   │   ├── multitask.py  # Training loop and configuration
│   │   │   │   └── config.py     # Shared training configuration constants
│   │   │   ├── inference/      # Inference utilities
│   │   │   │   └── visualize_training.py  # Training visualization
│   │   │   └── predict.py     # Per-slice processing and trajectory linking
│   │   └── utils/             # Analysis and utility functions
│   │       ├── helpers.py     # MSD fitting, particle size estimation, kymograph generation
│   │       ├── device.py      # Device detection utilities
│   │       └── visualization.py  # Comparison plotting utilities
│   └── main.py                # Typer CLI (train / infer)
├── demo.png                   # Demo visualization image
├── pyproject.toml            # Project configuration and dependencies
└── uv.lock                   # Dependency lock file
```

## Multi-Particle Tracking

The multi-particle tracking system uses:
- **Per-slice processing**: Each `16×512` slice is processed independently
- **Trajectory extraction**: Trajectories are extracted from masks using subpixel peak finding
- **Trajectory linking**: Trajectories from overlapping slices are linked using greedy assignment

Key features:
- Handles 2-3 particles simultaneously (configurable via `max_tracks=3`)
- Processes slices independently, then links trajectories at the end
- Uses intensity-weighted peak finding for accurate position estimation
- Falls back to center predictions if mask extraction fails

## Recommended Pipeline Breakdown

- **Stage 1 – Denoiser**: DDPM-style U-Net trained on `16×512` (time × space) windows
  boosts SNR but does not replace the raw kymograph used for physics-based metrics.
- **Stage 2 – Locator**: Lightweight attention-based regressor consumes the denoised
  slice to predict per-track center/width trajectories over the 16-frame window, keeping
  each particle's ID distinct without dense heatmaps.
- **Stage 3 – Trajectory extraction**: Within each predicted corridor (center ± width), run
  the traditional peak finder on the original (or denoised) signal to extract
  trajectories. This keeps the final tracks faithful to the physical signal even
  though detection is learned.
- **Stage 4 – Trajectory linking**: Link trajectories across overlapping slices using
  greedy assignment with overlap averaging.

## Tooling Notes

- Neural networks, datasets, and training utilities live under `src/kymo_tracker/`.
- Use the Typer CLI (`python src/main.py train` / `python src/main.py infer`) for training and inference.
- Use `demo/run.sh` for running the complete demo pipeline (stages 1-5).
- Default checkpoints are stored in `checkpoints/` directory.
- Trained models are saved to `artifacts/` directory.
