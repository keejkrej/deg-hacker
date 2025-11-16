"""
Single-Particle Median Filter Analysis

Baseline method using median filtering for comparison with U-Net denoising.
"""

from scipy.signal import medfilt

from kymo_tracker.utils import (
    estimate_noise_and_contrast,
    simulate_single_particle,
    summarize_analysis,
    write_joint_metrics_csv,
)


DEFAULT_PARTICLE_RADII = [2.5, 5.0, 10.0]
DEFAULT_CONTRASTS = [0.6, 0.8, 1.0]
DEFAULT_NOISE_LEVELS = [0.1, 0.3, 0.5]


def analyze_particle(p, c, n):
    """
    Analyze particle diffusion from kymograph data.

    Parameters:
    -----------
    p : float
        Particle size in nm
    c : float
        Contrast
    n : float
        Noise level
    """
    simulation = simulate_single_particle(p, c, n)

    kymograph_noise_estimate, kymograph_contrast_estimate = estimate_noise_and_contrast(
        simulation.kymograph_noisy
    )
    kymograph_noisy_filtered = medfilt(
        simulation.kymograph_noisy, kernel_size=(5, 3)
    )
    return summarize_analysis(
        simulation,
        kymograph_noisy_filtered,
        method_label="Median Filter",
        figure_subdir="medfilt",
        noise_contrast_est=(kymograph_noise_estimate, kymograph_contrast_estimate),
    )


def run_parameter_grid(
    particle_radii=DEFAULT_PARTICLE_RADII,
    contrasts=DEFAULT_CONTRASTS,
    noise_levels=DEFAULT_NOISE_LEVELS,
    csv_path="metrics/single_particle_medfilt.csv",
):
    metrics_rows = []
    for p in particle_radii:
        for c in contrasts:
            for n in noise_levels:
                print(f"\n[Median Filter] Running p={p}, c={c}, n={n}")
                metrics_rows.append(analyze_particle(p, c, n))
    written_csv = write_joint_metrics_csv(metrics_rows, csv_path)
    print(
        f"[Median Filter] Completed {len(metrics_rows)} runs; aggregated metrics -> {written_csv}"
    )


if __name__ == "__main__":
    run_parameter_grid()
