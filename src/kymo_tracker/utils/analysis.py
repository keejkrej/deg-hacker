"""
Analysis Utilities and Data Structures

This module provides:
- Simulation data structures (SimulationData, MultiSimulationData)
- Analysis metrics (AnalysisMetrics)
- Noise and contrast estimation
- CSV export utilities
"""

import csv
from dataclasses import dataclass, asdict
import os
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from kymo_tracker.utils.helpers import (
    find_max_subpixel,
    get_diffusion_coefficient,
    get_particle_radius,
    estimate_diffusion_msd_fit,
    generate_kymograph,
)


@dataclass
class SimulationData:
    """Container for a single simulated particle scenario."""

    p: float
    c: float
    n: float
    diffusion: float
    x_step: float
    t_step: float
    n_t: int
    n_x: int
    kymograph_noisy: np.ndarray
    kymograph_gt: np.ndarray
    true_path: np.ndarray


@dataclass
class AnalysisMetrics:
    method_label: str
    particle_radius_nm: float
    contrast: float
    noise_level: float
    diffusion_true: float
    diffusion_processed: float
    radius_true: float
    radius_processed: float
    # Optional fields (must come after required fields)
    diffusion_noisy: Optional[float] = None  # Optional: not always computed
    radius_noisy: Optional[float] = None  # Optional: not always computed
    noise_estimate: Optional[float] = None
    contrast_estimate: Optional[float] = None
    figure_path: Optional[str] = None


@dataclass
class MultiSimulationData:
    radii_nm: Sequence[float]
    contrasts: Sequence[float]
    noise_level: float
    diffusions: Sequence[float]
    x_step: float
    t_step: float
    n_t: int
    n_x: int
    kymograph_noisy: np.ndarray
    kymograph_gt: np.ndarray
    true_paths: np.ndarray


METRIC_FIELDNAMES = [
    "method_label",
    "particle_radius_nm",
    "contrast",
    "noise_level",
    "diffusion_true",
    "diffusion_noisy",
    "diffusion_processed",
    "radius_true",
    "radius_noisy",
    "radius_processed",
    "noise_estimate",
    "contrast_estimate",
    "figure_path",
]


def simulate_single_particle(
    p, c, n, x_step=0.5, t_step=1.0, n_t=4000, n_x=256, peak_width=1
):
    """Generate a noisy/ground-truth kymograph and metadata for a single particle."""
    diffusion = get_diffusion_coefficient(p)
    kymograph_noisy, kymograph_gt, true_path = generate_kymograph(
        length=n_t,
        width=n_x,
        diffusion=diffusion,
        contrast=c,
        noise_level=n,
        peak_width=peak_width,
        dx=x_step,
        dt=t_step,
    )
    return SimulationData(
        p=p,
        c=c,
        n=n,
        diffusion=diffusion,
        x_step=x_step,
        t_step=t_step,
        n_t=n_t,
        n_x=n_x,
        kymograph_noisy=kymograph_noisy,
        kymograph_gt=kymograph_gt,
        true_path=true_path,
    )


def simulate_multi_particle(
    radii_nm,
    contrasts,
    noise_level,
    x_step=0.5,
    t_step=1.0,
    n_t=4000,
    n_x=256,
    peak_width=1,
):
    """Simulate a multi-particle kymograph with independent radii/contrasts."""
    radii_list = list(radii_nm)
    contrasts_list = list(contrasts)

    if len(radii_list) != len(contrasts_list):
        raise ValueError("radii_nm and contrasts must have the same length")

    diffusions = [get_diffusion_coefficient(r) for r in radii_list]
    kymograph_noisy, kymograph_gt, true_paths = generate_kymograph(
        length=n_t,
        width=n_x,
        diffusion=diffusions,
        contrast=contrasts_list,
        noise_level=noise_level,
        peak_width=peak_width,
        dx=x_step,
        dt=t_step,
    )

    return MultiSimulationData(
        radii_nm=radii_list,
        contrasts=contrasts_list,
        noise_level=noise_level,
        diffusions=diffusions,
        x_step=x_step,
        t_step=t_step,
        n_t=n_t,
        n_x=n_x,
        kymograph_noisy=kymograph_noisy,
        kymograph_gt=kymograph_gt,
        true_paths=true_paths,
    )


def estimate_noise_and_contrast(kymograph_noisy, kernel_size=(5, 3)):
    """
    Estimate the noise floor and contrast straight from the noisy kymograph.

    A light spatial median filter acts as a crude baseline; the residual carries
    mostly the additive noise so its standard deviation is a proxy for the noise
    level used in the simulator. The contrast is approximated as the distance
    between the bright ridges (row maxima of the filtered kymograph) and the
    global background level.
    """
    from scipy.signal import medfilt

    filtered = medfilt(kymograph_noisy, kernel_size=kernel_size)
    residual = kymograph_noisy - filtered
    noise_level = np.std(residual)

    row_maxima = np.max(filtered, axis=1)
    # Use upper quantile to avoid being biased low when the particle is dim.
    signal_level = np.percentile(row_maxima, 90)
    background_level = np.median(filtered)
    contrast = max(signal_level - background_level, 0.0)
    return noise_level, contrast


def write_joint_metrics_csv(
    metrics_list: Sequence[AnalysisMetrics], csv_path: str
) -> str:
    """
    Write a metrics CSV containing all provided AnalysisMetrics rows.

    Returns the path that was written. If the list is empty, nothing is written.
    """
    if not metrics_list:
        return csv_path

    dir_name = os.path.dirname(csv_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        for metrics in metrics_list:
            writer.writerow(asdict(metrics))
    return csv_path


def summarize_analysis(
    simulation,
    processed_kymograph,
    method_label,
    figure_subdir,
    noise_contrast_est=None,
):
    """
    Compute the quantitative metrics and persist the diagnostic plot.

    Parameters:
        simulation (SimulationData): bundle produced by simulate_single_particle.
        processed_kymograph (np.ndarray): denoised/filtered version of the noisy data.
        method_label (str): label describing the processing method.
        figure_subdir (str): figures/<subdir> destination.
        noise_contrast_est (tuple, optional): (noise_est, contrast_est) for logging.
    """
    processed_path = find_max_subpixel(processed_kymograph)
    true_path = simulation.true_path

    estimated_diffusion_true = estimate_diffusion_msd_fit(
        true_path.T, dx=simulation.x_step, dt=simulation.t_step
    )
    estimated_diffusion_processed = estimate_diffusion_msd_fit(
        processed_path, dx=simulation.x_step, dt=simulation.t_step
    )

    estimated_radius_true = get_particle_radius(estimated_diffusion_true)
    estimated_radius_processed = get_particle_radius(estimated_diffusion_processed)

    if noise_contrast_est is not None:
        noise_est, contrast_est = noise_contrast_est
    else:
        noise_est, contrast_est = None, None

    fig, ax = plt.subplots(2, 2, figsize=(8, 7))
    fig.suptitle(
        f"{method_label}: p={simulation.p:.1f} nm, c={simulation.c:.2f}, "
        f"n={simulation.n:.2f} | D true/{method_label} = "
        f"{estimated_diffusion_true:.3f}/{estimated_diffusion_processed:.3f} µm²/ms"
    )
    # Use percentile-based ranges to be robust to outliers
    vmin_noisy = np.percentile(simulation.kymograph_noisy, 1)
    vmax_noisy = np.percentile(simulation.kymograph_noisy, 99)
    vmin_processed = np.percentile(processed_kymograph, 1)
    vmax_processed = np.percentile(processed_kymograph, 99)
    
    # Top row: Noisy kymograph and true path
    ax[0, 0].imshow(
        simulation.kymograph_noisy.T,
        aspect="auto",
        origin="lower",
        extent=[0, simulation.n_t, 0, simulation.n_x],
        vmin=vmin_noisy,
        vmax=vmax_noisy,
        cmap="gray",
    )
    ax[0, 0].set_title("Noisy Kymograph")
    ax[0, 0].set_xlabel("Time")
    ax[0, 0].set_ylabel("Position")
    
    ax[0, 1].plot(true_path.T)
    ax[0, 1].set_xlim(0, simulation.n_t)
    ax[0, 1].set_ylim(0, simulation.n_x)
    ax[0, 1].set_title("True Path")
    ax[0, 1].set_xlabel("Time")
    ax[0, 1].set_ylabel("Position")
    
    # Bottom row: Cleaned kymograph and processed path
    ax[1, 0].imshow(
        processed_kymograph.T,
        aspect="auto",
        origin="lower",
        extent=[0, simulation.n_t, 0, simulation.n_x],
        vmin=vmin_processed,
        vmax=vmax_processed,
        cmap="gray",
    )
    ax[1, 0].set_title(f"{method_label} Kymograph")
    ax[1, 0].set_xlabel("Time")
    ax[1, 0].set_ylabel("Position")
    
    ax[1, 1].plot(processed_path)
    ax[1, 1].set_xlim(0, simulation.n_t)
    ax[1, 1].set_ylim(0, simulation.n_x)
    ax[1, 1].set_title(f"{method_label} Path")
    ax[1, 1].set_xlabel("Time")
    ax[1, 1].set_ylabel("Position")

    figure_dir = os.path.join("figures", figure_subdir)
    os.makedirs(figure_dir, exist_ok=True)
    fig_filename = os.path.join(
        figure_dir,
        f"single_particle_p_{simulation.p}_c_{simulation.c}_n_{simulation.n}.png",
    )
    fig.savefig(fig_filename)
    plt.close(fig)

    # Note: diffusion_noisy and radius_noisy are not computed anymore
    # since we removed the "Dumbest Approach" plot
    metrics = AnalysisMetrics(
        method_label=method_label,
        particle_radius_nm=simulation.p,
        contrast=simulation.c,
        noise_level=simulation.n,
        diffusion_true=estimated_diffusion_true,
        diffusion_noisy=None,  # Not computed - removed bare estimated_path plot
        diffusion_processed=estimated_diffusion_processed,
        radius_true=estimated_radius_true,
        radius_noisy=None,  # Not computed - removed bare estimated_path plot
        radius_processed=estimated_radius_processed,
        noise_estimate=noise_est,
        contrast_estimate=contrast_est,
        figure_path=fig_filename,
    )
    print(f"[{method_label}] figure saved to {fig_filename}")
    return metrics
