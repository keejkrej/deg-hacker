import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from denoiser import load_model, denoise_kymograph, _default_device
from helpers import estimate_diffusion_msd_fit, get_particle_radius
from one_particle_unet import denoise_kymograph_chunked
from utils import simulate_multi_particle


@dataclass
class TrackSummary:
    track_id: int
    true_radius_nm: float
    estimated_radius_nm: float
    true_diffusion: float
    estimated_diffusion: float
    position_rmse: float


def _select_peak_candidates(row, max_candidates):
    max_candidates = min(max_candidates, len(row))
    if max_candidates <= 0:
        return np.array([], dtype=int), np.array([], dtype=row.dtype)
    partition_index = max(0, len(row) - max_candidates)
    idxs = np.argpartition(row, partition_index)[-max_candidates:]
    order = np.argsort(-row[idxs])
    idxs = idxs[order]
    scores = row[idxs]
    return idxs, scores


def _predict_positions(prev, prev_prev):
    if prev is None:
        return None
    if prev_prev is None:
        return prev.copy()
    velocity = prev - prev_prev
    return prev + velocity


def _assign_candidates(
    predictions,
    candidate_positions,
    candidate_scores,
    max_jump,
    min_intensity,
    intensity_weight,
    width,
):
    n_tracks = len(predictions)
    if len(candidate_positions) == 0:
        return np.clip(predictions, 0, width - 1)

    pred_matrix = predictions[:, None]
    cand_matrix = candidate_positions[None, :]
    dist = np.abs(pred_matrix - cand_matrix)
    cost = (dist / max(max_jump, 1e-3)) ** 2
    if candidate_scores.ptp() > 0:
        norm_scores = (candidate_scores - candidate_scores.min()) / (
            candidate_scores.ptp() + 1e-6
        )
    else:
        norm_scores = np.zeros_like(candidate_scores)
    cost -= intensity_weight * norm_scores
    cost[dist > max_jump] = 1e6 + dist[dist > max_jump]

    row_ind, col_ind = linear_sum_assignment(cost)
    assigned = predictions.copy()
    for r, c in zip(row_ind, col_ind):
        pos = candidate_positions[c]
        score = candidate_scores[c]
        if dist[r, c] <= max_jump and score >= min_intensity:
            assigned[r] = pos

    return np.clip(assigned, 0, width - 1)


def track_particles(
    kymograph,
    n_particles,
    max_candidates=30,
    max_jump=8,
    smoothing=0.4,
    min_intensity=0.02,
    intensity_weight=0.3,
):
    time_len, width = kymograph.shape
    tracks = np.full((n_particles, time_len), np.nan)

    prev_positions = None
    prev_prev_positions = None

    for t in range(time_len):
        row = kymograph[t]
        candidates, scores = _select_peak_candidates(row, max_candidates)
        if prev_positions is None:
            init = np.sort(candidates[:n_particles])
            if len(init) == 0:
                init = np.linspace(0, width - 1, n_particles)
            elif len(init) < n_particles:
                init = np.pad(init, (0, n_particles - len(init)), "edge")
            tracks[:, t] = init[:n_particles]
        else:
            predictions = _predict_positions(prev_positions, prev_prev_positions)
            if predictions is None:
                predictions = prev_positions
            assigned = _assign_candidates(
                predictions,
                candidates,
                scores,
                max_jump,
                min_intensity,
                intensity_weight,
                width,
            )
            if smoothing > 0:
                assigned = smoothing * assigned + (1 - smoothing) * prev_positions
            tracks[:, t] = np.clip(assigned, 0, width - 1)

        prev_prev_positions = prev_positions
        prev_positions = tracks[:, t]

    return tracks


def summarize_tracks(simulation, estimated_tracks, figure_path="figures/multi_unet"):
    os.makedirs(figure_path, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axes[0].imshow(
        simulation.kymograph_noisy.T,
        aspect="auto",
        origin="lower",
        extent=[0, simulation.n_t, 0, simulation.n_x],
        vmin=0,
        vmax=0.5,
    )
    axes[0].set_title("Noisy kymograph")
    axes[1].imshow(
        simulation.kymograph_gt.T,
        aspect="auto",
        origin="lower",
        extent=[0, simulation.n_t, 0, simulation.n_x],
        vmin=0,
        vmax=0.5,
    )
    axes[1].set_title("Ground truth vs estimates")

    for idx in range(len(simulation.radii_nm)):
        axes[0].plot(simulation.true_paths[idx], lw=0.8)
        axes[1].plot(simulation.true_paths[idx], lw=0.7, linestyle="--")
        axes[1].plot(estimated_tracks[idx], lw=1.0)

    for ax in axes:
        ax.set_xlim(0, simulation.n_t)
        ax.set_ylim(0, simulation.n_x)
        ax.set_ylabel("Position (px)")
        ax.set_xlabel("Frame")

    figfile = os.path.join(figure_path, "multi_particle_tracks.png")
    fig.savefig(figfile, dpi=150)
    plt.close(fig)
    print(f"[Multi U-Net] Figure saved to {figfile}")


def analyze_multi_particle(
    radii_nm,
    contrasts,
    noise_level,
    chunk_length=512,
    overlap=64,
    model_path="tiny_unet_denoiser.pth",
    max_candidates=30,
    max_jump=8,
):
    simulation = simulate_multi_particle(
        radii_nm=radii_nm,
        contrasts=contrasts,
        noise_level=noise_level,
    )

    device = _default_device()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = load_model(model_path, device=device)

    denoised = denoise_kymograph_chunked(
        model,
        simulation.kymograph_noisy,
        device=device,
        chunk_length=chunk_length,
        overlap=overlap,
    )

    estimated_tracks = track_particles(
        denoised,
        n_particles=len(radii_nm),
        max_candidates=max_candidates,
        max_jump=max_jump,
    )

    summaries = []
    for idx in range(len(radii_nm)):
        estimated_diffusion = estimate_diffusion_msd_fit(
            estimated_tracks[idx],
            dx=simulation.x_step,
            dt=simulation.t_step,
        )
        true_diffusion = simulation.diffusions[idx]
        estimated_radius = get_particle_radius(estimated_diffusion)
        rmse = np.sqrt(
            np.nanmean((estimated_tracks[idx] - simulation.true_paths[idx]) ** 2)
        )
        summaries.append(
            TrackSummary(
                track_id=idx,
                true_radius_nm=simulation.radii_nm[idx],
                estimated_radius_nm=estimated_radius,
                true_diffusion=true_diffusion,
                estimated_diffusion=estimated_diffusion,
                position_rmse=rmse,
            )
        )

    summarize_tracks(simulation, estimated_tracks)
    for summary in summaries:
        print(
            "Track {track_id}: r_true={true_radius_nm:.2f} nm, r_est={estimated_radius_nm:.2f} nm, "
            "D_true={true_diffusion:.3f}, D_est={estimated_diffusion:.3f} µm²/ms, RMSE={position_rmse:.2f} px".format(
                **summary.__dict__
            )
        )

    return summaries


if __name__ == "__main__":
    # Example setup with three particles of varying radii/contrast.
    analyze_multi_particle(
        radii_nm=[2.5, 5.0, 8.0],
        contrasts=[0.8, 0.6, 1.0],
        noise_level=0.3,
    )
