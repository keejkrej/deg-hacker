"""Training utilities for the kymo-tracker multi-task model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from kymo_tracker.data.multitask_dataset import MultiTaskDataset
from kymo_tracker.deeplearning.models.multitask import MultiTaskUNet
from kymo_tracker.utils.device import get_default_device


def masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute mean absolute error on valid positions only."""

    # Mask out NaN values in target
    valid_target = ~torch.isnan(target)
    combined_mask = mask * valid_target.float()
    
    # Compute difference, replacing NaN with 0 before masking
    diff = pred - target
    diff = torch.where(torch.isnan(diff), torch.zeros_like(diff), diff)
    
    masked_diff = torch.abs(diff) * combined_mask
    denom = combined_mask.sum().clamp_min(eps)
    return masked_diff.sum() / denom


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task model training."""

    epochs: int = 12
    batch_size: int = 8
    learning_rate: float = 1e-3
    denoise_loss_weight: float = 1.0
    locator_loss_weight: float = 2.0
    denoise_loss: str = "l2"
    use_gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    weight_decay: float = 1e-4
    dropout: float = 0.1
    encoder_dropout: float = 0.1
    decoder_dropout: float = 0.1
    use_lr_scheduler: bool = True
    label_smoothing: float = 0.0
    device: str = get_default_device()
    checkpoint_dir: Optional[str] = None
    save_best: bool = True
    checkpoint_every: int = 1
    auto_balance_losses: bool = True
    balance_min_scale: float = 0.1
    balance_max_scale: float = 10.0
    resume_from: Optional[str] = None
    resume_epoch: Optional[int] = None
    auto_resume: bool = False
    init_weights: Optional[str] = None


def _build_model(config: MultiTaskConfig, max_tracks: int = 3) -> MultiTaskUNet:
    return MultiTaskUNet(
        base_channels=48,
        use_bn=True,
        max_tracks=max_tracks,
        dropout=config.dropout,
        encoder_dropout=config.encoder_dropout,
        decoder_dropout=config.decoder_dropout,
    ).to(config.device)


def _resolve_checkpoint_path(config: MultiTaskConfig) -> Optional[str]:
    if config.resume_from:
        return config.resume_from
    if not config.auto_resume or not config.checkpoint_dir:
        return None
    ckpt_dir = Path(config.checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"))
    if checkpoints:
        return str(checkpoints[-1])
    best_model = ckpt_dir / "best_model.pth"
    if best_model.exists():
        return str(best_model)
    return None


def train_multitask_model(config: MultiTaskConfig, dataset: MultiTaskDataset) -> MultiTaskUNet:
    """Train the multi-task denoising + locator model."""

    model = _build_model(config, max_tracks=dataset.max_trajectories)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    if config.denoise_loss == "l2":
        denoise_criterion = nn.MSELoss()
    elif config.denoise_loss == "l1":
        denoise_criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unknown denoise loss: {config.denoise_loss}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    resume_checkpoint = _resolve_checkpoint_path(config)
    start_epoch = 0
    best_loss = float("inf")

    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=config.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", {}))
            start_epoch = checkpoint.get("epoch", 0)
            best_loss = checkpoint.get("best_loss", best_loss)
        else:
            model.load_state_dict(checkpoint)

    elif config.init_weights and Path(config.init_weights).exists():
        state_dict = torch.load(config.init_weights, map_location=config.device, weights_only=True)
        model.load_state_dict(state_dict)

    scheduler = None
    if config.use_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
        )

    if config.checkpoint_dir:
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_start = time.time()
        epoch_denoise = 0.0
        epoch_locator = 0.0
        epoch_total = 0.0
        batch_count = 0

        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch + 1}/{config.epochs}",
            unit="batch",
        )

        for batch_idx, batch in pbar:
            noisy, true_noise, target_heatmap = [
                tensor.to(config.device) for tensor in batch
            ]

            optimizer.zero_grad()
            pred_noise, pred_centers, pred_widths, pred_heatmap = model(noisy, return_heatmap=True)
            denoise_loss = denoise_criterion(pred_noise, true_noise)
            # L2 loss for heatmap prediction
            # pred_heatmap is (batch, time, space), need to add channel dim to match target_heatmap (batch, 1, time, space)
            pred_heatmap = pred_heatmap.unsqueeze(1)  # (batch, 1, time, space)
            locator_loss = nn.functional.mse_loss(pred_heatmap, target_heatmap)

            adaptive_denoise_weight = config.denoise_loss_weight
            if config.auto_balance_losses:
                with torch.no_grad():
                    ratio = (locator_loss.detach() + 1e-6) / (denoise_loss.detach() + 1e-6)
                    ratio = torch.clamp(
                        ratio,
                        min=config.balance_min_scale,
                        max=config.balance_max_scale,
                    )
                    adaptive_denoise_weight *= ratio.item()

            total_loss = adaptive_denoise_weight * denoise_loss + config.locator_loss_weight * locator_loss
            total_loss.backward()

            if config.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()

            epoch_denoise += denoise_loss.item()
            epoch_locator += locator_loss.item()
            epoch_total += total_loss.item()
            batch_count += 1

            pbar.set_postfix(
                denoise=f"{denoise_loss.item():.4f}",
                locator=f"{locator_loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                total=f"{total_loss.item():.4f}",
            )

        pbar.close()

        avg_denoise = epoch_denoise / max(batch_count, 1)
        avg_locator = epoch_locator / max(batch_count, 1)
        avg_total = epoch_total / max(batch_count, 1)
        print(
            f"Epoch {epoch + 1}/{config.epochs} completed: Total={avg_total:.6f}, "
            f"Denoise={avg_denoise:.6f}, Locator={avg_locator:.6f}, "
            f"Time={time.time() - epoch_start:.2f}s",
        )

        if config.checkpoint_dir and (epoch + 1) % config.checkpoint_every == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
                "config": config,
            }
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f"checkpoint_epoch_{epoch + 1}.pth",
            )
            torch.save(checkpoint, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

        if config.save_best and avg_total < best_loss:
            best_loss = avg_total
            if config.checkpoint_dir:
                best_path = os.path.join(config.checkpoint_dir, "best_model.pth")
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "config": config,
                }
                torch.save(checkpoint, best_path)
                print(f"  New best model saved to {best_path}")

        if scheduler is not None:
            scheduler.step(avg_total)

    model.eval()
    return model


def save_multitask_model(model: MultiTaskUNet, path: str) -> None:
    """Persist the trained model weights."""

    torch.save(model.state_dict(), path)
    print(f"Multi-task model saved to {path}")


def load_multitask_model(
    path: str,
    device: Optional[str] = None,
    max_tracks: int = 3,
) -> MultiTaskUNet:
    """Load a trained multi-task model."""
    
    device = device or get_default_device()
    model = MultiTaskUNet(max_tracks=max_tracks).to(device)

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


__all__ = [
    "MultiTaskConfig",
    "train_multitask_model",
    "save_multitask_model",
    "load_multitask_model",
]
