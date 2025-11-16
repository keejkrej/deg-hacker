"""Train a multi-task network that denoises and regresses particle positions.

This single model performs both tasks:
1. Denoising: Predicts noise to subtract (DDPM-style)
2. Localization: Predicts per-particle center/width trajectories for each frame

Benefits:
- Shared encoder learns common features
- Efficient joint training while keeping final tracking classical
- Temporal self-attention reasons over the 16-frame window instead of dense heatmaps
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import time
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add parent directory to path for imports (only needed if not installed as package)
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import simulate_single_particle, simulate_multi_particle
from utils.helpers import generate_kymograph, get_diffusion_coefficient


class ConvBlock(nn.Module):
    """Convolutional block with two conv layers, batch norm, ReLU, and optional dropout."""
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _default_device() -> str:
    """Get default device for PyTorch."""
    import torch
    has_mps = getattr(torch.backends, "mps", None)
    if has_mps and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


class DenoiseUNet(nn.Module):
    """UNet that only predicts denoising residuals."""

    def __init__(self, base_channels: int = 48, use_bn: bool = True,
                 dropout: float = 0.0, encoder_dropout: float = 0.0,
                 decoder_dropout: float = 0.0) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, base_channels, use_bn=use_bn, dropout=encoder_dropout)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, use_bn=use_bn, dropout=encoder_dropout)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, use_bn=use_bn, dropout=encoder_dropout)
        self.down = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8, use_bn=use_bn, dropout=dropout)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=(1, 2), stride=(1, 2))
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4, use_bn=use_bn, dropout=decoder_dropout)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=(1, 2), stride=(1, 2))
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2, use_bn=use_bn, dropout=decoder_dropout)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=(1, 2), stride=(1, 2))
        self.dec1 = ConvBlock(base_channels * 2, base_channels, use_bn=use_bn, dropout=decoder_dropout)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)
        nn.init.xavier_uniform_(self.head.weight, gain=0.1)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))
        b = self.bottleneck(self.down(e3))

        d3 = self.up3(b)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = torch.nn.functional.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = torch.nn.functional.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = torch.nn.functional.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.head(d1)


class TemporalLocator(nn.Module):
    """Lightweight temporal locator with 1D ViT-style attention."""

    def __init__(
        self,
        in_channels: int = 1,
        spatial_channels: int = 96,
        token_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        max_tracks: int = 3,
        max_tokens: int = 512,
    ) -> None:
        super().__init__()
        self.max_tracks = max_tracks
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, spatial_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(spatial_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_channels, spatial_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(spatial_channels),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(spatial_channels, token_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(token_dim, max_tracks * 2)
        nn.init.xavier_uniform_(self.head.weight, gain=0.1)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, 1, T, W]
        features = self.spatial_encoder(x)  # [B, C, T, W]
        pooled = features.mean(dim=-1)  # Average over spatial axis -> [B, C, T]
        tokens = pooled.permute(0, 2, 1)  # [B, T, C]
        tokens = self.proj(tokens)
        seq_len = tokens.size(1)
        if seq_len > self.pos_embed.size(1):
            raise ValueError(
                f"TemporalLocator received sequence of length {seq_len} but "
                f"positional embedding supports up to {self.pos_embed.size(1)}"
            )
        tokens = tokens + self.pos_embed[:, :seq_len]
        encoded = self.transformer(tokens)
        preds = self.head(encoded)  # [B, T, max_tracks*2]
        preds = preds.view(encoded.size(0), seq_len, self.max_tracks, 2)
        preds = preds.permute(0, 2, 3, 1)  # [B, max_tracks, 2, T]
        center_logits = preds[:, :, 0]
        width_logits = preds[:, :, 1]
        centers = torch.sigmoid(center_logits)  # normalized position [0,1]
        widths = torch.nn.functional.softplus(width_logits) + 1e-3  # positive width fraction
        return centers, widths


class MultiTaskUNet(nn.Module):
    """Wrapper that couples denoising U-Net with a temporal locator."""

    def __init__(
        self,
        base_channels: int = 48,
        use_bn: bool = True,
        max_tracks: int = 3,
        dropout: float = 0.0,
        encoder_dropout: float = 0.0,
        decoder_dropout: float = 0.0,
        locator_token_dim: int = 128,
        locator_layers: int = 2,
        locator_heads: int = 4,
    ) -> None:
        super().__init__()
        self.max_tracks = max_tracks
        self.denoiser = DenoiseUNet(
            base_channels=base_channels,
            use_bn=use_bn,
            dropout=dropout,
            encoder_dropout=encoder_dropout,
            decoder_dropout=decoder_dropout,
        )
        self.locator = TemporalLocator(
            in_channels=1,
            spatial_channels=base_channels * 2,
            token_dim=locator_token_dim,
            num_layers=locator_layers,
            num_heads=locator_heads,
            max_tracks=max_tracks,
            max_tokens=512,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted_noise = self.denoiser(x)
        denoised = torch.clamp(x - predicted_noise, 0.0, 1.0)
        centers, widths = self.locator(denoised.detach())
        return predicted_noise, centers, widths


def create_position_targets(
    paths: np.ndarray,
    length: int,
    width: int,
    max_tracks: int,
    default_width_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create normalized center/width targets and validity mask."""

    if paths.ndim == 1:
        paths = paths.reshape(1, -1)
    n_particles, path_length = paths.shape
    positions = np.zeros((max_tracks, length), dtype=np.float32)
    widths = np.zeros((max_tracks, length), dtype=np.float32)
    valid = np.zeros((max_tracks, length), dtype=np.float32)
    spatial_denominator = max(width - 1, 1)
    width_denominator = max(width, 1)
    n_channels = min(max_tracks, n_particles)
    for i in range(n_channels):
        for t in range(min(length, path_length)):
            pos = paths[i, t]
            if np.isnan(pos):
                continue
            clamped = np.clip(pos, 0.0, width - 1.0)
            positions[i, t] = clamped / spatial_denominator
            widths[i, t] = default_width_px / width_denominator
            valid[i, t] = 1.0
    return positions, widths, valid


def masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute mean absolute error on valid positions only."""

    masked_diff = torch.abs(pred - target) * mask
    denom = mask.sum().clamp_min(eps)
    return masked_diff.sum() / denom


class MultiTaskDataset(Dataset):
    """Dataset for multi-task training: noisy kymograph -> (noise, center/width targets).

    When ``window_length`` is provided the simulator directly generates sequences of that
    temporal length (e.g., 16 frames) so no post-generation cropping is required.
    """
    
    def __init__(
        self,
        length: int = 512,
        width: int = 512,
        radii_nm: Tuple[float, float] = (3.0, 15.0),
        contrast: Tuple[float, float] = (0.5, 1.0),
        noise_level: Tuple[float, float] = (0.1, 0.5),
        seed: Optional[int] = None,
        peak_width: float = 1.0,
        dt: float = 1.0,
        dx: float = 0.5,
        n_samples: int = 1024,
        multi_trajectory_prob: float = 0.3,
        max_trajectories: int = 3,
        mask_peak_width_samples: float = 2.0,
        window_length: Optional[int] = None,
    ) -> None:
        # If a window length is specified, treat it as the effective generation length
        # so trajectories are synthesized directly at the desired temporal extent.
        if window_length is not None:
            if window_length <= 0:
                raise ValueError("window_length must be positive")
            self.length = int(window_length)
        else:
            self.length = length
        self.width = width
        self.radii_nm = radii_nm
        self.contrast = contrast
        self.noise_level = noise_level
        self.peak_width = peak_width
        self.dt = dt
        self.dx = dx
        self.n_samples = n_samples
        self.multi_trajectory_prob = multi_trajectory_prob
        self.max_trajectories = max_trajectories
        self.mask_peak_width_samples = mask_peak_width_samples
        self.window_length = int(window_length) if window_length is not None else None
        self.rng = np.random.default_rng(seed)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def sample_parameters(self, n_particles: int = 1) -> Tuple[list[float], list[float], float]:
        """Sample parameters for n_particles trajectories."""
        radii = [float(self.rng.uniform(*self.radii_nm)) for _ in range(n_particles)]
        contrasts = [float(self.rng.uniform(*self.contrast)) for _ in range(n_particles)]
        noise = float(self.rng.uniform(*self.noise_level))
        return radii, contrasts, noise
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Decide number of trajectories
        if self.rng.random() < self.multi_trajectory_prob:
            n_particles = self.rng.integers(2, self.max_trajectories + 1)
        else:
            n_particles = 1
        
        radii, contrasts, noise = self.sample_parameters(n_particles)
        
        # Generate kymograph
        if n_particles == 1:
            simulation = simulate_single_particle(
                p=radii[0],
                c=contrasts[0],
                n=noise,
                x_step=self.dx,
                t_step=self.dt,
                n_t=self.length,
                n_x=self.width,
                peak_width=self.peak_width,
            )
            noisy = simulation.kymograph_noisy
            gt = simulation.kymograph_gt
            paths = simulation.true_path
            if paths.ndim == 1:
                paths = paths.reshape(1, -1)
        else:
            diffusions = [get_diffusion_coefficient(r) for r in radii]
            noisy, gt, paths = generate_kymograph(
                length=self.length,
                width=self.width,
                diffusion=diffusions,
                contrast=contrasts,
                noise_level=noise,
                peak_width=self.peak_width,
                dt=self.dt,
                dx=self.dx,
            )
        
        # Compute true noise (for denoising task)
        true_noise = noisy - gt

        # Create normalized center/width targets
        peak_width_samples = self.peak_width / self.dx
        default_width = max(peak_width_samples, self.mask_peak_width_samples)
        positions, widths, valid_mask = create_position_targets(
            paths,
            length=self.length,
            width=self.width,
            max_tracks=self.max_trajectories,
            default_width_px=default_width,
        )

        # Convert to tensors
        noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).float()
        noise_tensor = torch.from_numpy(true_noise).unsqueeze(0).float()
        position_tensor = torch.from_numpy(positions).float()
        width_tensor = torch.from_numpy(widths).float()
        mask_tensor = torch.from_numpy(valid_mask).float()
        
        return noisy_tensor, noise_tensor, position_tensor, width_tensor, mask_tensor


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task model training."""
    epochs: int = 12
    batch_size: int = 8
    learning_rate: float = 1e-3
    denoise_loss_weight: float = 1.0  # Weight for denoising loss
    locator_loss_weight: float = 2.0  # Weight for combined locator loss
    locator_center_weight: float = 1.0  # Relative weight for center regression
    locator_width_weight: float = 0.5  # Relative weight for width regression
    denoise_loss: str = "l2"  # "l2" or "l1"
    use_gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    weight_decay: float = 1e-4  # L2 regularization (weight decay) - prevents overfitting
    dropout: float = 0.1  # Dropout rate for bottleneck (0 = disabled, typical: 0.1-0.3)
    encoder_dropout: float = 0.1  # Dropout rate for encoder layers (0 = disabled, typical: 0.1-0.3, commonly used in U-Nets/DDPM)
    decoder_dropout: float = 0.1  # Dropout rate for decoder layers (0 = disabled, typical: 0.1-0.3)
    use_lr_scheduler: bool = True
    label_smoothing: float = 0.0  # Label smoothing for segmentation (0 = disabled, typical: 0.05-0.1)
    device: str = _default_device()
    checkpoint_dir: Optional[str] = None  # Directory to save checkpoints (None = don't save)
    save_best: bool = True  # Save best model based on total loss
    checkpoint_every: int = 1  # Save checkpoint every N epochs (1 = every epoch)
    auto_balance_losses: bool = True  # Automatically rebalance denoise vs locator losses
    balance_min_scale: float = 0.1  # Clamp for adaptive denoise weight scaling
    balance_max_scale: float = 10.0
    resume_from: Optional[str] = None  # Path to checkpoint to resume training from (None = auto-detect latest)
    resume_epoch: Optional[int] = None  # Epoch number to resume from (if None, inferred from checkpoint)
    auto_resume: bool = False  # Automatically resume from latest checkpoint if available (opt-in)
    init_weights: Optional[str] = None  # Initialize model weights from this path (no optimizer state)


def dice_loss_binary(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Binary Dice loss for segmentation.
    
    Parameters:
    -----------
    pred : torch.Tensor
        Binary logits [B, 1, H, W]
    target : torch.Tensor
        Binary labels [B, H, W] with values in [0, 1]
    """
    # Convert logits to probabilities using sigmoid
    pred_probs = torch.sigmoid(pred)  # [B, 1, H, W]
    
    # Flatten
    pred_flat = pred_probs.view(-1)
    target_flat = target.view(-1)
    
    # Compute Dice
    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    # Return Dice loss (1 - dice)
    return 1.0 - dice


def spatial_smoothness_loss(pred_logits: torch.Tensor) -> torch.Tensor:
    """Spatial smoothness loss to encourage spatially consistent segmentation.
    
    Computes the L2 norm of gradients in time and space dimensions.
    Lower values = smoother segmentation.
    
    Parameters:
    -----------
    pred_logits : torch.Tensor
        Binary logits [B, 1, H, W] or [B, H, W]
    
    Returns:
    --------
    loss : torch.Tensor
        Smoothness loss (scalar)
    """
    if pred_logits.dim() == 4:
        pred = pred_logits.squeeze(1)  # [B, H, W]
    else:
        pred = pred_logits  # [B, H, W]
    
    # Compute gradients in time (vertical) and space (horizontal) dimensions
    # Time gradient: difference along dimension 1 (H)
    time_grad = pred[:, 1:, :] - pred[:, :-1, :]  # [B, H-1, W]
    # Space gradient: difference along dimension 2 (W)
    space_grad = pred[:, :, 1:] - pred[:, :, :-1]  # [B, H, W-1]
    
    # L2 norm of gradients (encourages smoothness)
    smoothness = (time_grad ** 2).mean() + (space_grad ** 2).mean()
    
    return smoothness


def total_variation_loss(pred_logits: torch.Tensor, range_size: int = 1) -> torch.Tensor:
    """Total Variation (TV) loss to enforce flatness of segmentation mask over specified range.
    
    TV loss uses L1 norm of gradients, which encourages piecewise constant regions.
    This is better than L2 for enforcing flatness because it penalizes small variations
    more uniformly and encourages sharp boundaries between flat regions.
    
    Parameters:
    -----------
    pred_logits : torch.Tensor
        Binary logits [B, 1, H, W] or [B, H, W]
    range_size : int
        Range over which to compute gradients (1 = immediate neighbors, 2 = 2 pixels apart, etc.)
        Larger values enforce flatness over longer distances
    
    Returns:
    --------
    loss : torch.Tensor
        TV loss (scalar)
    """
    if pred_logits.dim() == 4:
        pred = pred_logits.squeeze(1)  # [B, H, W]
    else:
        pred = pred_logits  # [B, H, W]
    
    # Convert logits to probabilities for TV computation
    pred_probs = torch.sigmoid(pred)
    
    # Compute gradients over specified range
    # Time gradient: difference over range_size pixels along dimension 1 (H)
    time_grad = pred_probs[:, range_size:, :] - pred_probs[:, :-range_size, :]  # [B, H-range_size, W]
    # Space gradient: difference over range_size pixels along dimension 2 (W)
    space_grad = pred_probs[:, :, range_size:] - pred_probs[:, :, :-range_size]  # [B, H, W-range_size]
    
    # Normalize by range_size to make loss scale-independent
    time_grad = time_grad / range_size
    space_grad = space_grad / range_size
    
    # L1 norm of gradients (Total Variation) - encourages flat regions over longer range
    tv_loss = torch.abs(time_grad).mean() + torch.abs(space_grad).mean()
    
    return tv_loss


def laplacian_flatness_loss(pred_logits: torch.Tensor, range_size: int = 1) -> torch.Tensor:
    """Laplacian-based loss to enforce flatness (second-order smoothness) over specified range.
    
    Penalizes second derivatives (Laplacian), which enforces even flatter regions
    than first-order gradients. This encourages regions to be locally constant.
    
    Parameters:
    -----------
    pred_logits : torch.Tensor
        Binary logits [B, 1, H, W] or [B, H, W]
    range_size : int
        Range over which to compute second derivatives (1 = immediate neighbors, 2 = 2 pixels apart)
        Larger values enforce flatness over longer distances
    
    Returns:
    --------
    loss : torch.Tensor
        Laplacian flatness loss (scalar)
    """
    if pred_logits.dim() == 4:
        pred = pred_logits.squeeze(1)  # [B, H, W]
    else:
        pred = pred_logits  # [B, H, W]
    
    # Convert logits to probabilities
    pred_probs = torch.sigmoid(pred)
    
    # Compute second derivatives (Laplacian approximation) over specified range
    # Time dimension: second derivative over range_size
    time_second = (pred_probs[:, 2*range_size:, :] - 
                   2 * pred_probs[:, range_size:-range_size, :] + 
                   pred_probs[:, :-2*range_size, :])
    # Space dimension: second derivative over range_size
    space_second = (pred_probs[:, :, 2*range_size:] - 
                    2 * pred_probs[:, :, range_size:-range_size] + 
                    pred_probs[:, :, :-2*range_size])
    
    # Normalize by range_size^2 to make loss scale-independent
    time_second = time_second / (range_size ** 2)
    space_second = space_second / (range_size ** 2)
    
    # L1 norm of second derivatives (encourages locally flat regions over longer range)
    laplacian_loss = torch.abs(time_second).mean() + torch.abs(space_second).mean()
    
    return laplacian_loss


def focal_loss_binary(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal Loss for binary segmentation.
    
    Focal loss addresses class imbalance by down-weighting easy examples
    and focusing on hard examples.
    
    Parameters:
    -----------
    pred_logits : torch.Tensor
        Binary logits [B, 1, H, W] or [B, H, W]
    target : torch.Tensor
        Binary labels [B, H, W] with values in [0, 1]
    alpha : float
        Weighting factor for rare class (default: 0.25)
    gamma : float
        Focusing parameter (default: 2.0, higher = more focus on hard examples)
    
    Returns:
    --------
    loss : torch.Tensor
        Focal loss (scalar)
    """
    if pred_logits.dim() == 4:
        pred_logits = pred_logits.squeeze(1)  # [B, H, W]
    
    # Compute BCE loss
    bce_loss = nn.functional.binary_cross_entropy_with_logits(
        pred_logits, target, reduction='none'
    )
    
    # Convert logits to probabilities
    pt = torch.exp(-bce_loss)  # pt = p if target=1, else 1-p
    
    # Compute focal weight: (1 - pt)^gamma
    focal_weight = (1 - pt) ** gamma
    
    # Apply alpha weighting: alpha for positive class, (1-alpha) for negative
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    
    # Focal loss
    focal_loss = alpha_t * focal_weight * bce_loss
    
    return focal_loss.mean()


def _find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint file in the checkpoint directory.
    
    Looks for files matching 'checkpoint_epoch_*.pth' and returns the one with highest epoch number.
    Falls back to 'best_model.pth' if no epoch checkpoints found.
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    import re
    
    # Find all checkpoint files
    checkpoint_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pth'):
            # Try to extract epoch number from filename
            match = re.search(r'checkpoint_epoch_(\d+)\.pth', filename)
            if match:
                epoch_num = int(match.group(1))
                checkpoint_files.append((epoch_num, os.path.join(checkpoint_dir, filename)))
    
    if checkpoint_files:
        # Return checkpoint with highest epoch number
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)
        return checkpoint_files[0][1]
    
    # Fallback to best_model.pth if it exists
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        return best_model_path
    
    return None


def train_multitask_model(
    config: MultiTaskConfig,
    dataset: MultiTaskDataset,
) -> MultiTaskUNet:
    """Train the multi-task model.
    
    If config.resume_from is set, loads checkpoint and resumes training from that point.
    If config.auto_resume is True and checkpoint_dir is set, automatically finds and resumes
    from the latest checkpoint.
    """
    
    # Create model
    model = MultiTaskUNet(
        base_channels=48, 
        use_bn=True, 
        max_tracks=dataset.max_trajectories,
        dropout=config.dropout,
        encoder_dropout=config.encoder_dropout,
        decoder_dropout=config.decoder_dropout,
    ).to(config.device)
    
    # Resume from checkpoint if specified or auto-detect
    start_epoch = 0
    best_loss = float('inf')
    checkpoint_path = config.resume_from
    weights_path = None
    resume_state_source = None
    
    if config.resume_from and config.init_weights:
        print("?? Warning: Both resume_from and init_weights set; using resume_from for full resume.")
    
    # Auto-detect latest checkpoint if enabled and no explicit path given
    if checkpoint_path is None:
        if config.init_weights:
            weights_path = config.init_weights
        elif config.auto_resume and config.checkpoint_dir:
            checkpoint_path = _find_latest_checkpoint(config.checkpoint_dir)
            if checkpoint_path:
                print(f"\n?? Auto-detected latest checkpoint: {checkpoint_path}")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n?? Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
        resume_state_source = checkpoint_path
        
        # Handle both old format (just state_dict) and new format (full checkpoint)
        if 'model_state_dict' in checkpoint:
            # New format: full checkpoint with training state
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            if config.resume_epoch is not None:
                start_epoch = config.resume_epoch
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"  Resuming from epoch {start_epoch + 1}")
            print(f"  Best loss so far: {best_loss:.6f}")
        else:
            # Old format: just model weights (for inference checkpoints)
            try:
                model.load_state_dict(checkpoint)
                print(f"  ? Loaded model weights (old checkpoint format)")
                print(f"  Starting from epoch 0 (no training state in checkpoint)")
                start_epoch = 0
                best_loss = float('inf')
            except Exception as e:
                print(f"  ? Error loading checkpoint: {e}")
                print(f"  Starting training from scratch")
                start_epoch = 0
                best_loss = float('inf')
    elif checkpoint_path:
        print(f"? Warning: Resume checkpoint not found: {checkpoint_path}")
        print("  Starting training from scratch")
    elif weights_path:
        if os.path.exists(weights_path):
            print(f"\n?? Initializing model weights from: {weights_path}")
            try:
                state_dict = torch.load(weights_path, map_location=config.device, weights_only=True)
                model.load_state_dict(state_dict)
            except Exception:
                checkpoint = torch.load(weights_path, map_location=config.device, weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            print("  Starting from epoch 1 with provided weights (no optimizer state)")
        else:
            print(f"? Warning: Initial weights file not found: {weights_path}")
            print("  Starting training from scratch")
    
    # Loss functions
    if config.denoise_loss == "l2":
        denoise_criterion = nn.MSELoss()
    elif config.denoise_loss == "l1":
        denoise_criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unknown denoise loss: {config.denoise_loss}")
    
    print(
        f"  Locator loss weights => total: {config.locator_loss_weight}, "
        f"center: {config.locator_center_weight}, width: {config.locator_width_weight}"
    )
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Resume optimizer state if resuming (only for new checkpoint format)
    checkpoint_path_for_optimizer = resume_state_source
    
    if checkpoint_path_for_optimizer and os.path.exists(checkpoint_path_for_optimizer):
        # Load checkpoint with weights_only=False to handle MultiTaskConfig
        checkpoint = torch.load(checkpoint_path_for_optimizer, map_location=config.device, weights_only=False)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"  Resumed optimizer state")
    
    # Learning rate scheduler
    scheduler = None
    if config.use_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5
        )
        # Resume scheduler state if resuming (only for new checkpoint format)
        checkpoint_path_for_scheduler = resume_state_source
        
        if checkpoint_path_for_scheduler and os.path.exists(checkpoint_path_for_scheduler):
            checkpoint = torch.load(checkpoint_path_for_scheduler, map_location=config.device, weights_only=False)
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"  Resumed scheduler state")
    
    # Data loader
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    
    print(f"\nTraining multi-task model on {config.device}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Denoise loss: {config.denoise_loss} (weight: {config.denoise_loss_weight})")
    print(
        f"  Locator loss weight: {config.locator_loss_weight}"
        f" (center={config.locator_center_weight}, width={config.locator_width_weight})"
    )
    print(f"  Weight decay (L2): {config.weight_decay}")
    if config.encoder_dropout > 0:
        print(f"  Dropout (encoder): {config.encoder_dropout}")
    if config.dropout > 0:
        print(f"  Dropout (bottleneck): {config.dropout}")
    if config.decoder_dropout > 0:
        print(f"  Dropout (decoder): {config.decoder_dropout}")
    if config.label_smoothing > 0:
        print(f"  Label smoothing: {config.label_smoothing}")
    if config.auto_balance_losses:
        print(
            f"  Auto loss balancing enabled (scale clamp: "
            f"{config.balance_min_scale}-{config.balance_max_scale})"
        )
    if config.checkpoint_dir:
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        print(f"  Checkpoint directory: {config.checkpoint_dir}")
    
    model.train()
    
    for epoch in range(start_epoch, config.epochs):
        epoch_start_time = time.time()
        epoch_denoise_loss = 0.0
        epoch_locator_loss = 0.0
        epoch_center_loss = 0.0
        epoch_width_loss = 0.0
        epoch_total_loss = 0.0
        batch_count = 0
        
        # Progress bar for batches in this epoch
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch+1}/{config.epochs}",
            unit="batch"
        )
        
        for batch_idx, (noisy, true_noise, target_positions, target_widths, valid_mask) in pbar:
            noisy = noisy.to(config.device)
            true_noise = true_noise.to(config.device)
            target_positions = target_positions.to(config.device)
            target_widths = target_widths.to(config.device)
            valid_mask = valid_mask.to(config.device)
            
            optimizer.zero_grad()
            
            # Forward pass: get denoising and trajectory parameters
            pred_noise, pred_centers, pred_widths = model(noisy)
            
            # Compute losses
            denoise_loss = denoise_criterion(pred_noise, true_noise)
            
            # Locator loss (center + width regression)
            center_loss = masked_l1_loss(pred_centers, target_positions, valid_mask)
            width_loss = masked_l1_loss(pred_widths, target_widths, valid_mask)
            locator_loss_val = (
                config.locator_center_weight * center_loss +
                config.locator_width_weight * width_loss
            )
            
            # Adaptive weighting to keep locator loss from lagging behind
            adaptive_denoise_weight = config.denoise_loss_weight
            if config.auto_balance_losses:
                with torch.no_grad():
                    ratio = (locator_loss_val.detach() + 1e-6) / (denoise_loss.detach() + 1e-6)
                    ratio = torch.clamp(
                        ratio,
                        min=config.balance_min_scale,
                        max=config.balance_max_scale,
                    )
                    adaptive_denoise_weight = adaptive_denoise_weight * ratio.item()
            total_loss = (
                adaptive_denoise_weight * denoise_loss +
                config.locator_loss_weight * locator_loss_val
            )
            
            # Backward pass
            total_loss.backward()
            
            # Optional gradient diagnostics for locator head
            if batch_idx == 0 and epoch % 2 == 0:
                head_grad = model.locator.head.weight.grad
                if head_grad is not None:
                    grad_norm = head_grad.norm().item()
                    print(f"\n  [Debug] Locator head grad norm: {grad_norm:.6f}")
            
            # Gradient clipping
            if config.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            
            epoch_denoise_loss += denoise_loss.item()
            epoch_locator_loss += locator_loss_val.item()
            epoch_center_loss += center_loss.item()
            epoch_width_loss += width_loss.item()
            epoch_total_loss += total_loss.item()
            batch_count += 1
            
            # Update progress bar with current losses
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'denoise': f'{denoise_loss.item():.4f}',
                'locator': f'{locator_loss_val.item():.4f}',
                'center': f'{center_loss.item():.4f}',
                'width': f'{width_loss.item():.4f}',
                'lr': f'{current_lr:.2e}',
                'total': f'{total_loss.item():.4f}'
            })
        
        pbar.close()
        
        avg_denoise_loss = epoch_denoise_loss / batch_count
        avg_locator_loss = epoch_locator_loss / batch_count
        avg_center_loss = epoch_center_loss / batch_count
        avg_width_loss = epoch_width_loss / batch_count
        avg_total_loss = epoch_total_loss / batch_count
        epoch_time = time.time() - epoch_start_time
   
        print(f"Epoch {epoch + 1}/{config.epochs} completed: "
              f"Total={avg_total_loss:.6f}, "
              f"Denoise={avg_denoise_loss:.6f}, "
              f"Locator={avg_locator_loss:.6f} (center={avg_center_loss:.6f}, width={avg_width_loss:.6f}), "
              f"Time={epoch_time:.2f}s")
        
        # Save checkpoint if configured
        if config.checkpoint_dir and (epoch + 1) % config.checkpoint_every == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            # Save full checkpoint with training state for resuming
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'config': config,
            }
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(checkpoint, checkpoint_path)
            print(f"  ?? Checkpoint saved: {checkpoint_path}")
            
        # Save best model if configured
        if config.save_best and avg_total_loss < best_loss:
            best_loss = avg_total_loss
            if config.checkpoint_dir:
                best_path = os.path.join(config.checkpoint_dir, "best_model.pth")
                # Save best model with full checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'config': config,
                }
                if scheduler is not None:
                    checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                torch.save(checkpoint, best_path)
                print(f"  ? New best model saved (loss: {best_loss:.6f})")
                
        
        if scheduler is not None:
            scheduler.step(avg_total_loss)
    
    model.eval()
    return model


def save_multitask_model(model: MultiTaskUNet, path: str) -> None:
    """Save multi-task model."""
    torch.save(model.state_dict(), path)
    print(f"Multi-task model saved to {path}")


def load_multitask_model(path: str, device: str = None, max_tracks: int = 3) -> MultiTaskUNet:
    """Load multi-task model.
    
    Handles both formats:
    - Checkpoint format: dict with 'model_state_dict' key
    - Model format: direct state_dict
    
    Note: Dropout is set to 0.0 for inference (disabled during eval mode anyway).
    """
    if device is None:
        device = _default_device()
    
    # Create model with dropout=0.0 for inference (dropout disabled in eval mode anyway)
    model = MultiTaskUNet(
        base_channels=48, 
        use_bn=True, 
        max_tracks=max_tracks,
        dropout=0.0,
        encoder_dropout=0.0,
        decoder_dropout=0.0,
    ).to(device)
    
    # Try loading with weights_only first (for weights-only files)
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception:
        # If that fails, try loading full checkpoint (may contain config, optimizer, etc.)
        # Import the module to make classes available for pickle
        import sys
        import importlib
        # Ensure the module is imported so pickle can find MultiTaskConfig
        if 'train.multitask_model' not in sys.modules:
            import train.multitask_model
        
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except AttributeError as e:
            # If MultiTaskConfig can't be found, try to fix the module path
            if 'MultiTaskConfig' in str(e):
                # Create a dummy config class in the current namespace for pickle
                import types
                # Get the actual module
                mtm_module = sys.modules.get('train.multitask_model')
                if mtm_module:
                    # Temporarily add to __main__ namespace
                    import __main__
                    if not hasattr(__main__, 'MultiTaskConfig'):
                        __main__.MultiTaskConfig = getattr(mtm_module, 'MultiTaskConfig')
                checkpoint = torch.load(path, map_location=device, weights_only=False)
            else:
                raise
        
        # Handle both checkpoint format and direct state_dict format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Checkpoint format: extract model_state_dict
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Direct state_dict format
            model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def denoise_and_segment_chunked(
    model: MultiTaskUNet,
    kymograph: np.ndarray,
    device: str = None,
    chunk_size: int = 16,
    overlap: int = 8,
) -> Tuple[np.ndarray, dict[str, np.ndarray]]:
    """Apply multi-task model to kymograph with chunking.
    
    Returns denoised kymograph and per-particle center/width predictions.
    
    Returns:
    --------
    denoised : np.ndarray
        Denoised kymograph, shape (time, width), dtype float32, values in [0, 1]
    track_params : dict
        ``{"centers": array, "widths": array}`` with shapes (time, max_tracks)
    """
    if device is None:
        device = _default_device()
    
    model.eval()
    time_len, width = kymograph.shape
    
    # If kymograph fits in one chunk, process directly
    if time_len <= chunk_size:
        with torch.no_grad():
            input_tensor = torch.from_numpy(kymograph).unsqueeze(0).unsqueeze(0).float().to(device)
            pred_noise, pred_centers, pred_widths = model(input_tensor)
            denoised = torch.clamp(input_tensor - pred_noise, 0.0, 1.0).squeeze().cpu().numpy()
            centers_np = pred_centers.squeeze(0).cpu().numpy().transpose(1, 0)
            widths_np = pred_widths.squeeze(0).cpu().numpy().transpose(1, 0)
            centers_px = centers_np * (width - 1)
            widths_px = widths_np * width
            track_params = {
                "centers": centers_px,
                "widths": widths_px,
            }
            
            # Clear GPU memory
            del input_tensor, pred_noise, pred_centers, pred_widths
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            
        return denoised, track_params
    
    # Process in chunks with overlap
    denoised = np.zeros((time_len, width), dtype=np.float32)
    weights = np.zeros((time_len, width), dtype=np.float32)
    temporal_weights = np.zeros((time_len, 1), dtype=np.float32)
    centers_all = None
    widths_all = None
    
    # Create window function for blending
    window = np.ones(chunk_size)
    if overlap > 0:
        fade_len = overlap // 2
        window[:fade_len] = np.linspace(0, 1, fade_len)
        window[-fade_len:] = np.linspace(1, 0, fade_len)
    
    with torch.no_grad():
        start = 0
        while start < time_len:
            end = min(start + chunk_size, time_len)
            chunk = kymograph[start:end]
            
            # Pad if needed
            padded_chunk = chunk
            if chunk.shape[0] < chunk_size:
                padding = np.zeros((chunk_size - chunk.shape[0], width), dtype=chunk.dtype)
                padded_chunk = np.vstack([chunk, padding])
            
            # Process chunk
            chunk_tensor = torch.from_numpy(padded_chunk).unsqueeze(0).unsqueeze(0).float().to(device)
            pred_noise_chunk, pred_centers_chunk, pred_widths_chunk = model(chunk_tensor)
            denoised_chunk = torch.clamp(chunk_tensor - pred_noise_chunk, 0.0, 1.0).squeeze().cpu().numpy()
            centers_chunk = pred_centers_chunk.squeeze(0).cpu().numpy().transpose(1, 0)
            widths_chunk = pred_widths_chunk.squeeze(0).cpu().numpy().transpose(1, 0)
            centers_chunk = centers_chunk * (width - 1)
            widths_chunk = widths_chunk * width
            
            # Clear GPU memory immediately
            del chunk_tensor, pred_noise_chunk, pred_centers_chunk, pred_widths_chunk
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            
            # Extract actual size (remove padding)
            actual_len = end - start
            denoised_chunk = denoised_chunk[:actual_len]
            centers_chunk = centers_chunk[:actual_len]
            widths_chunk = widths_chunk[:actual_len]
            window_chunk = window[:actual_len]
            
            if centers_all is None:
                max_tracks = centers_chunk.shape[1]
                centers_all = np.zeros((time_len, max_tracks), dtype=np.float32)
                widths_all = np.zeros((time_len, max_tracks), dtype=np.float32)
            
            # Blend denoised output and locator predictions with window
            weight_chunk = window_chunk[:, np.newaxis]
            denoised[start:end] += denoised_chunk * weight_chunk
            weights[start:end] += weight_chunk
            centers_all[start:end] += centers_chunk * weight_chunk
            widths_all[start:end] += widths_chunk * weight_chunk
            temporal_weights[start:end] += weight_chunk
            
            # Clear chunk data
            del denoised_chunk, centers_chunk, widths_chunk
            
            # Move to next chunk
            start += chunk_size - overlap
    
    # Normalize denoised output and locator predictions by weights
    denoised = np.divide(denoised, weights, out=np.zeros_like(denoised), where=weights > 0)
    centers_all = np.divide(
        centers_all,
        temporal_weights,
        out=np.zeros_like(centers_all),
        where=temporal_weights > 0,
    )
    widths_all = np.divide(
        widths_all,
        temporal_weights,
        out=np.zeros_like(widths_all),
        where=temporal_weights > 0,
    )
    del weights, temporal_weights
    
    # Clear GPU cache
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    
    return denoised, {"centers": centers_all, "widths": widths_all}


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Train the multi-task UNet for multi-particle tracking.")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to checkpoint/weights to initialize from (only model weights are loaded).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint file to fully resume training state from.",
    )
    parser.add_argument(
        "--auto-resume",
        nargs="?",
        const="",
        default=None,
        help="Automatically resume from latest checkpoint; optionally provide a directory (default uses models/checkpoints).",
    )
    args = parser.parse_args()
    
    # Create dataset
    print("Creating multi-task dataset...")
    dataset = MultiTaskDataset(
        n_samples=4096,
        length=512,
        width=512,
        radii_nm=(3.0, 70.0),  # Extended to match test data (test shows up to ~68 nm)
        contrast=(0.5, 1.1),    # Extended to match test data (test shows up to ~1.03)
        noise_level=(0.08, 0.8),  # Extended upper bound to include high noise levels (>0.5)
        multi_trajectory_prob=1.0,  # 100% multi-particle examples
        max_trajectories=3,
        mask_peak_width_samples=2.0,
        window_length=16,
    )
    print(f"  Total samples: {len(dataset)}")
    sample_noisy, _, _, _, _ = dataset[0]
    print(
        f"  Sample tensor shape: channels={sample_noisy.shape[0]}, "
        f"time={sample_noisy.shape[1]}, space={sample_noisy.shape[2]}"
    )
    
    checkpoint_dir = "models/checkpoints"
    auto_resume_flag = False
    if args.auto_resume is not None:
        auto_resume_flag = True
        if args.auto_resume:
            checkpoint_dir = args.auto_resume

    # Training config
    config = MultiTaskConfig(
        epochs=2,
        batch_size=32,
        learning_rate=1.5e-3,  # Slightly reduced from 2e-3 for more stable training
        denoise_loss_weight=1.0,
        locator_loss_weight=2.0,
        denoise_loss="l2",
        weight_decay=1e-4,  # L2 regularization to prevent overfitting
        dropout=0.1,  # Dropout in bottleneck
        encoder_dropout=0.1,  # Dropout in encoder layers (commonly used in U-Nets/DDPM)
        decoder_dropout=0.1,  # Dropout in decoder layers (helps prevent overfitting)
        use_gradient_clipping=True,
        max_grad_norm=1.0,
        use_lr_scheduler=True,
        checkpoint_dir=checkpoint_dir,  # Save checkpoints after each epoch
        save_best=True,  # Save best model based on loss
        checkpoint_every=1,  # Save checkpoint every epoch
        auto_resume=auto_resume_flag,  # Automatically resume from latest checkpoint if available
        resume_from=args.checkpoint,
        init_weights=args.weights,
    )
    
    # Train
    model = train_multitask_model(config, dataset)
    
    # Save final model
    os.makedirs("models", exist_ok=True)
    model_path = "models/multitask_unet.pth"
    save_multitask_model(model, model_path)
    print(f"\nFinal model saved to: {model_path}")
