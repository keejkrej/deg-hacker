from denoiser import load_model, denoise_kymograph, _default_device
import os
import numpy as np

from utils import simulate_single_particle, summarize_analysis, write_joint_metrics_csv


DEFAULT_PARTICLE_RADII = [2.5, 5.0, 10.0]
DEFAULT_CONTRASTS = [0.6, 0.8, 1.0]
DEFAULT_NOISE_LEVELS = [0.1, 0.3, 0.5]


def denoise_kymograph_chunked(
    model, kymograph, device=None, chunk_size=512, overlap=64
):
    """
    Denoise a kymograph by processing it in chunks along both dimensions if needed.
    Handles kymographs of any size, including smaller than chunk_size (e.g., 256x256).

    Parameters:
    -----------
    model : TinyUNet
        Trained U-Net model
    kymograph : np.ndarray
        Input kymograph of shape (time, position)
    device : str, optional
        Device to run model on
    chunk_size : int
        Size of each chunk in both dimensions (default: 512, matches training)
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

    # If kymograph fits in one chunk (both dimensions <= chunk_size), process directly
    # For smaller inputs (e.g., 256x256), pad to chunk_size, process, then crop back
    if time_len <= chunk_size and pos_len <= chunk_size:
        if time_len < chunk_size or pos_len < chunk_size:
            # Pad smaller inputs to chunk_size for optimal processing
            padded = np.zeros((chunk_size, chunk_size), dtype=kymograph.dtype)
            padded[:time_len, :pos_len] = kymograph
            denoised_padded = denoise_kymograph(model, padded, device=device)
            return denoised_padded[:time_len, :pos_len]
        else:
            # Exact size, process directly
            return denoise_kymograph(model, kymograph, device=device)

    # Need chunking - determine which dimensions need chunking
    chunk_time = time_len > chunk_size
    chunk_pos = pos_len > chunk_size

    if not chunk_time and not chunk_pos:
        # Shouldn't reach here, but handle just in case
        return denoise_kymograph(model, kymograph, device=device)

    # Process in overlapping chunks
    stride = chunk_size - overlap
    denoised_chunks = []

    # Generate chunk coordinates
    time_starts = list(range(0, time_len, stride))
    pos_starts = list(range(0, pos_len, stride)) if chunk_pos else [0]

    for t_start in time_starts:
        for p_start in pos_starts:
            t_end = min(t_start + chunk_size, time_len)
            p_end = min(p_start + chunk_size, pos_len)
            
            chunk = kymograph[t_start:t_end, p_start:p_end]
            
            # Pad chunk to chunk_size x chunk_size if needed
            padded_chunk = np.zeros((chunk_size, chunk_size), dtype=chunk.dtype)
            chunk_t_len, chunk_p_len = chunk.shape
            padded_chunk[:chunk_t_len, :chunk_p_len] = chunk
            
            # Denoise the padded chunk
            chunk_denoised_padded = denoise_kymograph(model, padded_chunk, device=device)
            
            # Crop back to original chunk size
            chunk_denoised = chunk_denoised_padded[:chunk_t_len, :chunk_p_len]
            
            denoised_chunks.append((t_start, t_end, p_start, p_end, chunk_denoised))

    # Combine chunks with overlap handling (averaging in overlap regions)
    denoised = np.zeros_like(kymograph)
    counts = np.zeros_like(kymograph, dtype=np.float32)

    for t_start, t_end, p_start, p_end, chunk_denoised in denoised_chunks:
        denoised[t_start:t_end, p_start:p_end] += chunk_denoised
        counts[t_start:t_end, p_start:p_end] += 1.0

    # Average in overlap regions (avoid division by zero)
    denoised = denoised / np.maximum(counts, 1.0)

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
    simulation = simulate_single_particle(p, c, n)

    # Load trained U-Net model
    device = _default_device()
    model_path = "tiny_unet_denoiser.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_model(model_path, device=device)

    # Denoise using chunked processing (handles any size, including 256x256)
    kymograph_denoised = denoise_kymograph_chunked(
        model,
        simulation.kymograph_noisy,
        device=device,
        chunk_size=512,
        overlap=64,
    )

    return summarize_analysis(
        simulation,
        kymograph_denoised,
        method_label="U-Net Denoised",
        figure_subdir="unet",
    )


def run_parameter_grid(
    particle_radii=DEFAULT_PARTICLE_RADII,
    contrasts=DEFAULT_CONTRASTS,
    noise_levels=DEFAULT_NOISE_LEVELS,
    csv_path="metrics/one_particle_unet.csv",
):
    metrics_rows = []
    for p in particle_radii:
        for c in contrasts:
            for n in noise_levels:
                print(f"\n[U-Net] Running p={p}, c={c}, n={n}")
                metrics_rows.append(analyze_particle(p, c, n))
    written_csv = write_joint_metrics_csv(metrics_rows, csv_path)
    print(f"[U-Net] Completed {len(metrics_rows)} runs; aggregated metrics -> {written_csv}")


if __name__ == "__main__":
    run_parameter_grid()
