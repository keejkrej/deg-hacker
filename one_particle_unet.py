from helpers import (
    find_max_subpixel,
    get_diffusion_coefficient,
    get_particle_radius,
    estimate_diffusion_msd_fit,
    generate_kymograph,
)

from denoiser import load_model, denoise_kymograph, _default_device
import matplotlib.pyplot as plt
import os
import numpy as np


def denoise_kymograph_chunked(
    model, kymograph, device=None, chunk_length=512, overlap=64
):
    """
    Denoise a kymograph by processing it in chunks along the time dimension.

    Parameters:
    -----------
    model : TinyUNet
        Trained U-Net model
    kymograph : np.ndarray
        Input kymograph of shape (time, position)
    device : str, optional
        Device to run model on
    chunk_length : int
        Length of each chunk in time dimension (default: 512, matches training)
    overlap : int
        Overlap between chunks to avoid boundary artifacts (default: 64)

    Returns:
    --------
    np.ndarray
        Denoised kymograph of same shape as input
    """
    device = device or _default_device()
    model = model.to(device)
    model.eval()

    time_len, pos_len = kymograph.shape

    # If kymograph fits in one chunk, process directly
    if time_len <= chunk_length:
        return denoise_kymograph(model, kymograph, device=device)

    # Process in overlapping chunks
    stride = chunk_length - overlap
    denoised_chunks = []

    for start_idx in range(0, time_len, stride):
        end_idx = min(start_idx + chunk_length, time_len)
        chunk = kymograph[start_idx:end_idx, :]

        # Pad if necessary to reach chunk_length
        if chunk.shape[0] < chunk_length:
            padding = np.zeros((chunk_length - chunk.shape[0], chunk.shape[1]))
            chunk = np.vstack([chunk, padding])

        chunk_denoised = denoise_kymograph(model, chunk, device=device)

        # Remove padding if we added it
        if end_idx - start_idx < chunk_length:
            chunk_denoised = chunk_denoised[: end_idx - start_idx, :]

        denoised_chunks.append((start_idx, end_idx, chunk_denoised))

    # Combine chunks with overlap handling (averaging in overlap regions)
    denoised = np.zeros_like(kymograph)
    counts = np.zeros(time_len)

    for start_idx, end_idx, chunk_denoised in denoised_chunks:
        actual_len = end_idx - start_idx
        denoised[start_idx:end_idx, :] += chunk_denoised[:actual_len, :]
        counts[start_idx:end_idx] += 1

    # Average in overlap regions
    denoised = denoised / np.maximum(counts[:, np.newaxis], 1)

    return denoised


def analyze_particle(p, c, n):
    """
    Analyze particle diffusion from kymograph data using U-Net denoising.

    Parameters:
    -----------
    p : float
        Particle size in nm
    c : float
        Contrast
    n : float
        Noise level
    """
    d = get_diffusion_coefficient(p)
    x_step = 0.5
    t_step = 1.0
    n_t = 4000
    n_x = 256
    kymograph_noisy, kymograph_gt, true_path = generate_kymograph(
        length=n_t,
        width=n_x,
        diffusion=d,
        contrast=c,
        noise_level=n,
        peak_width=1,
        dx=x_step,
        dt=t_step,
    )

    # Load trained U-Net model
    device = _default_device()
    model_path = "tiny_unet_denoiser.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_model(model_path, device=device)

    # Denoise using chunked processing
    kymograph_denoised = denoise_kymograph_chunked(
        model, kymograph_noisy, device=device, chunk_length=512, overlap=64
    )

    estimated_path = find_max_subpixel(kymograph_noisy)
    estimated_path_denoised = find_max_subpixel(kymograph_denoised)

    estimated_diffusion_true = estimate_diffusion_msd_fit(
        true_path.T, dx=x_step, dt=t_step
    )
    estimated_diffusion = estimate_diffusion_msd_fit(
        estimated_path, dx=x_step, dt=t_step
    )
    estimated_diffusion_denoised = estimate_diffusion_msd_fit(
        estimated_path_denoised, dx=x_step, dt=t_step
    )
    estimated_radius_true = get_particle_radius(estimated_diffusion_true)
    estimated_radius = get_particle_radius(estimated_diffusion)
    estimated_radius_denoised = get_particle_radius(estimated_diffusion_denoised)

    print(f"True Diffusion Coefficient: {d:.4f} µm²/ms")
    print(
        f"Estimated Diffusion Coefficient True Path: {estimated_diffusion_true:.4f} µm²/ms"
    )
    print(f"Estimated Diffusion Coefficient: {estimated_diffusion:.4f} µm²/ms")
    print(
        f"Estimated Diffusion Coefficient Denoised (U-Net): {estimated_diffusion_denoised:.4f} µm²/ms"
    )
    print(f"True Particle Radius: {p:.3f} nm")
    print(f"Estimated Particle Radius True Path: {estimated_radius_true:.3f} nm")
    print(f"Estimated Particle Radius: {estimated_radius:.3f} nm")
    print(
        f"Estimated Particle Radius Denoised (U-Net): {estimated_radius_denoised:.3f} nm"
    )

    fig, ax = plt.subplots(2, 2, figsize=(8, 7))
    fig.suptitle(
        f"p={p} nm, c={c}, n={n}, D={d:.4f} µm²/ms \n D_estimated_true={estimated_diffusion_true:.4f} µm²/ms, D_estimated={estimated_diffusion:.4f} µm²/ms, D_estimated_unet={estimated_diffusion_denoised:.4f} µm²/ms"
    )
    ax[0, 0].imshow(
        kymograph_noisy.T,
        aspect="auto",
        origin="lower",
        extent=[0, n_t, 0, n_x],
        vmin=0,
        vmax=0.5,
    )
    ax[0, 1].plot(true_path.T)
    ax[0, 1].set_xlim(0, n_t)
    ax[0, 1].set_ylim(0, n_x)
    ax[1, 0].plot(estimated_path)
    ax[1, 0].set_xlim(0, n_t)
    ax[1, 0].set_ylim(0, n_x)
    ax[1, 1].plot(estimated_path_denoised)
    ax[1, 1].set_xlim(0, n_t)
    ax[1, 1].set_ylim(0, n_x)
    ax[0, 0].set_title("Noisy Kymograph")
    ax[0, 1].set_title("True Path")
    ax[1, 0].set_title("Dumbest Approach")
    ax[1, 1].set_title("U-Net Denoised")

    fig_filename = f"figures/unet/one_particle_p_{p}_c_{c}_n_{n}.png"
    fig.savefig(fig_filename)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    os.makedirs("figures/unet", exist_ok=True)
    # for p in [2.5, 5, 10]:
    #     for c in [0.6, 0.8, 1.0]:
    #         for n in [0.1, 0.3, 0.5]:
    #             analyze_particle(p, c, n)
    analyze_particle(p=5, c=0.6, n=0.3)
