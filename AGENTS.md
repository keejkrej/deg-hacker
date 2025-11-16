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
- **Stage 2 – Locator**: Lightweight detector consumes the denoised slice to produce
  per-column particle likelihoods and soft masks; each particle keeps a distinct ID.
- **Stage 3 – Classical maxima**: Within each mask, run the traditional peak/CoM
  finder on the original (or denoised) signal to extract trajectories. This ensures
  the final tracks remain faithful to the physical signal even though detection is
  learned.
