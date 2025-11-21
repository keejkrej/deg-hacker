"""Typer CLI exposing kymo-tracker training and inference commands."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import typer

from kymo_tracker.data.multitask_dataset import MultiTaskDataset
from kymo_tracker.deeplearning.training.multitask import (
    MultiTaskConfig,
    train_multitask_model,
    save_multitask_model,
    load_multitask_model,
)
from kymo_tracker.deeplearning.training.config import (
    DEFAULT_EPOCHS,
    DEFAULT_SAMPLES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MASK_PEAK_WIDTH_SAMPLES,
    DEFAULT_WINDOW_LENGTH,
    DEFAULT_WIDTH,
    DEFAULT_RADII_NM,
    DEFAULT_CONTRAST,
    DEFAULT_NOISE_LEVEL,
    DEFAULT_MULTI_TRAJECTORY_PROB,
    DEFAULT_MAX_TRAJECTORIES,
)
from kymo_tracker.deeplearning.predict import (
    process_slice_independently,
    link_trajectories_across_slices,
)
from kymo_tracker.utils.device import get_default_device

app = typer.Typer(add_completion=False)


@app.command()
def train(
    samples: int = typer.Option(16384, help="Number of synthetic samples to generate."),
    epochs: int = typer.Option(6, help="Number of training epochs."),
    batch_size: int = typer.Option(32, help="Batch size for training."),
    checkpoint_dir: Path = typer.Option(Path("checkpoints"), help="Directory for checkpoints."),
    save_model_path: Path = typer.Option(
        Path("artifacts/multitask_unet.pth"),
        help="Where to store the final trained weights.",
    ),
    window_length: int = typer.Option(16, help="Temporal window length for each sample."),
) -> None:
    """Train the multi-task denoising + locator model on synthetic data."""

    dataset = MultiTaskDataset(
        n_samples=samples,
        length=DEFAULT_WIDTH,
        width=DEFAULT_WIDTH,
        radii_nm=DEFAULT_RADII_NM,
        contrast=DEFAULT_CONTRAST,
        noise_level=DEFAULT_NOISE_LEVEL,
        multi_trajectory_prob=DEFAULT_MULTI_TRAJECTORY_PROB,
        max_trajectories=DEFAULT_MAX_TRAJECTORIES,
        mask_peak_width_samples=DEFAULT_MASK_PEAK_WIDTH_SAMPLES,
        window_length=window_length,
    )
    typer.echo(f"Dataset created with {len(dataset)} samples of shape 512x{window_length}.")

    config = MultiTaskConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=1.5e-3,
        checkpoint_dir=str(checkpoint_dir),
        auto_resume=True,
    )

    model = train_multitask_model(config, dataset)

    if save_model_path:
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        save_multitask_model(model, str(save_model_path))


@app.command()
def infer(
    model_path: Path = typer.Argument(..., help="Path to trained model weights."),
    input_path: Path = typer.Argument(..., help="Path to .npy file containing a kymograph."),
    output_dir: Path = typer.Option(Path("runs/inference"), help="Directory to store predictions."),
    chunk_size: int = typer.Option(16, help="Temporal chunk size for inference."),
    overlap: int = typer.Option(8, help="Temporal overlap between chunks."),
) -> None:
    """Run inference on a kymograph saved as a NumPy array."""

    kymograph = np.load(input_path)
    if kymograph.ndim != 2:
        raise typer.BadParameter("Input array must be 2D (time, width)")

    device = get_default_device()
    model = load_multitask_model(str(model_path), device=device)
    
    # Process each slice independently
    T, W = kymograph.shape
    slice_results = []
    start = 0
    while start < T:
        end = min(start + chunk_size, T)
        slice_data = kymograph[start:end]
        slice_result = process_slice_independently(model, slice_data, device=device)
        slice_results.append(slice_result)
        start += chunk_size - overlap
    
    # Link trajectories
    slice_trajectories_list = [result['trajectories'] for result in slice_results]
    linked_trajectories = link_trajectories_across_slices(
        slice_trajectories_list,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    
    # Reconstruct denoised kymograph (blend slices)
    denoised = np.zeros((T, W), dtype=np.float32)
    weights = np.zeros((T, W), dtype=np.float32)
    window = np.ones(chunk_size)
    if overlap > 0:
        fade_len = overlap // 2
        window[:fade_len] = np.linspace(0, 1, fade_len)
        window[-fade_len:] = np.linspace(1, 0, fade_len)
    
    start = 0
    for result in slice_results:
        end = min(start + chunk_size, T)
        actual_len = end - start
        weight_chunk = window[:actual_len, np.newaxis]
        denoised[start:end] += result['denoised'] * weight_chunk
        weights[start:end] += weight_chunk
        start += chunk_size - overlap
    
    denoised = np.divide(denoised, weights, out=np.zeros_like(denoised), where=weights > 0)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "denoised.npy", denoised)
    np.save(output_dir / "trajectories.npy", np.array(linked_trajectories, dtype=object))

    typer.echo(
        f"Saved denoised kymograph and trajectories to {output_dir.resolve()}"
    )


if __name__ == "__main__":
    app()
