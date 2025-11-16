"""Typer CLI exposing kymo-tracker training and inference commands."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import typer

from kymo_tracker.data.multitask_dataset import MultiTaskDataset
from kymo_tracker.training.multitask import (
    MultiTaskConfig,
    train_multitask_model,
    save_multitask_model,
    load_multitask_model,
)
from kymo_tracker.inference.predict import denoise_and_segment_chunked

app = typer.Typer(add_completion=False)


@app.command()
def train(
    samples: int = typer.Option(4096, help="Number of synthetic samples to generate."),
    epochs: int = typer.Option(4, help="Number of training epochs."),
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
        length=512,
        width=512,
        radii_nm=(3.0, 70.0),
        contrast=(0.5, 1.1),
        noise_level=(0.08, 0.8),
        multi_trajectory_prob=1.0,
        max_trajectories=3,
        mask_peak_width_samples=2.0,
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

    model = load_multitask_model(str(model_path))
    denoised, track_params = denoise_and_segment_chunked(
        model,
        kymograph,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "denoised.npy", denoised)
    np.save(output_dir / "centers.npy", track_params["centers"])
    np.save(output_dir / "widths.npy", track_params["widths"])

    typer.echo(
        f"Saved denoised kymograph and trajectories to {output_dir.resolve()}"
    )


if __name__ == "__main__":
    app()
