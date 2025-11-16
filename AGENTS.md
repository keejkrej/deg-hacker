## Notes for Codex agents

- Do not modify `helpers.py`. If functionality is needed, add it elsewhere.
- Use conda `kymo` environment for Python: `conda activate kymo`

## Multi-Particle Tracking

The multi-particle tracking system (`multi_particle_unet.py`) uses:
- **Otsu binarization** for particle detection (optimal for denoised data)
- **Connected components** to identify distinct particle blobs
- **DBSCAN clustering** to merge nearby detections
- **Greedy assignment** with explicit overlap prevention for robust tracking

Key features:
- Handles 2-3 particles simultaneously
- Prevents track overlaps through explicit separation enforcement
- Uses intensity-weighted center-of-mass for accurate position estimation
- Falls back to peak selection if clustering finds no particles

## Recommended Pipeline Breakdown

- **Stage 1 – Denoiser**: DDPM-style U-Net trained on 512(space) × 16(time) windows
  boosts SNR but does not replace the raw kymograph used for physics-based metrics.
- **Stage 2 – Locator**: Lightweight attention-based regressor consumes the denoised
  slice to predict per-track center/width trajectories over the 16-frame window, keeping
  each particle's ID distinct without dense heatmaps.
- **Stage 3 – Classical maxima**: Within each predicted corridor (center ± width), run
  the traditional peak/CoM finder on the original (or denoised) signal to extract
  trajectories. This keeps the final tracks faithful to the physical signal even
  though detection is learned.
