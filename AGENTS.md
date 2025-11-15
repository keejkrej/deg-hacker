## Notes for Codex agents

- Do not modify `helpers.py`. If functionality is needed, add it elsewhere.
- Use conda `ml` environment for Python: `conda activate ml`

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
