"""Train a multi-task U-Net that outputs both denoised kymograph and segmentation mask.

This single model performs both tasks:
1. Denoising: Predicts noise to subtract (DDPM-style)
2. Segmentation: Outputs probability map of particle locations

Benefits:
- Shared encoder learns common features
- More efficient than two separate models
- Better feature learning through multi-task learning
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
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label

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


class MultiTaskUNet(nn.Module):
    """U-Net with two output heads: denoising and embedding-based instance segmentation.
    
    Architecture:
    - Shared encoder
    - Separate decoders for denoising and segmentation
    - Two output heads:
      1. Denoising head: predicts noise (DDPM-style)
      2. Embedding head: predicts instance embeddings for clustering-based instance segmentation
    """
    
    def __init__(self, base_channels: int = 48, use_bn: bool = True, max_tracks: int = 3, 
                 dropout: float = 0.0, encoder_dropout: float = 0.0, decoder_dropout: float = 0.0,
                 embedding_dim: int = 8) -> None:
        super().__init__()
        self.max_tracks = max_tracks
        self.embedding_dim = embedding_dim
        # Embedding-based instance segmentation: embeddings are clustered to get instances
        
        # Shared encoder (dropout helps with generalization, commonly used in U-Nets/DDPM)
        self.enc1 = ConvBlock(1, base_channels, use_bn=use_bn, dropout=encoder_dropout)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, use_bn=use_bn, dropout=encoder_dropout)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, use_bn=use_bn, dropout=encoder_dropout)
        
        self.down = nn.MaxPool2d(2)
        
        # Bottleneck dropout
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8, use_bn=use_bn, dropout=dropout)
        
        # Separate decoder for denoising (dropout in decoder helps prevent overfitting)
        self.denoise_up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.denoise_dec3 = ConvBlock(base_channels * 8, base_channels * 4, use_bn=use_bn, dropout=decoder_dropout)
        
        self.denoise_up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.denoise_dec2 = ConvBlock(base_channels * 4, base_channels * 2, use_bn=use_bn, dropout=decoder_dropout)
        
        self.denoise_up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.denoise_dec1 = ConvBlock(base_channels * 2, base_channels, use_bn=use_bn, dropout=decoder_dropout)
        
        # Separate decoder for segmentation (dropout in decoder helps prevent overfitting)
        # Simple approach: use denoised decoder features directly
        self.segment_up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.segment_dec3 = ConvBlock(base_channels * 8, base_channels * 4, use_bn=use_bn, dropout=decoder_dropout)
        
        self.segment_up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.segment_dec2 = ConvBlock(base_channels * 4, base_channels * 2, use_bn=use_bn, dropout=decoder_dropout)
        
        self.segment_up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.segment_dec1 = ConvBlock(base_channels * 2, base_channels, use_bn=use_bn, dropout=decoder_dropout)
        
        # Two output heads
        # Head 1: Denoising (predicts noise, no activation)
        self.denoise_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        nn.init.xavier_uniform_(self.denoise_head.weight, gain=0.1)
        nn.init.constant_(self.denoise_head.bias, 0.0)
        
        # Head 2: Embeddings (predicts instance embeddings for clustering-based segmentation)
        # Output: embedding_dim channels (continuous vectors for clustering)
        self.embedding_head = nn.Conv2d(base_channels, embedding_dim, kernel_size=1)
        # Initialize embeddings to small random values
        nn.init.xavier_uniform_(self.embedding_head.weight, gain=0.1)
        nn.init.constant_(self.embedding_head.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returns (predicted_noise, embeddings)."""
        # Shared encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))
        
        b = self.bottleneck(self.down(e3))
        
        # Denoising decoder path
        denoise_d3 = self.denoise_up3(b)
        if denoise_d3.shape[2:] != e3.shape[2:]:
            denoise_d3 = torch.nn.functional.interpolate(denoise_d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        denoise_d3 = torch.cat([denoise_d3, e3], dim=1)
        denoise_d3 = self.denoise_dec3(denoise_d3)
        
        denoise_d2 = self.denoise_up2(denoise_d3)
        if denoise_d2.shape[2:] != e2.shape[2:]:
            denoise_d2 = torch.nn.functional.interpolate(denoise_d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        denoise_d2 = torch.cat([denoise_d2, e2], dim=1)
        denoise_d2 = self.denoise_dec2(denoise_d2)
        
        denoise_d1 = self.denoise_up1(denoise_d2)
        if denoise_d1.shape[2:] != e1.shape[2:]:
            denoise_d1 = torch.nn.functional.interpolate(denoise_d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        denoise_d1 = torch.cat([denoise_d1, e1], dim=1)
        denoise_d1 = self.denoise_dec1(denoise_d1)
        
        # Segmentation decoder path - use denoised decoder features directly
        # This makes segmentation leverage the smoother denoised features
        segment_d3 = self.segment_up3(b)
        if segment_d3.shape[2:] != e3.shape[2:]:
            segment_d3 = torch.nn.functional.interpolate(segment_d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        # Use denoised decoder features instead of encoder features for smoother segmentation
        segment_d3 = torch.cat([segment_d3, denoise_d3], dim=1)
        segment_d3 = self.segment_dec3(segment_d3)
        
        segment_d2 = self.segment_up2(segment_d3)
        if segment_d2.shape[2:] != e2.shape[2:]:
            segment_d2 = torch.nn.functional.interpolate(segment_d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        # Use denoised decoder features
        segment_d2 = torch.cat([segment_d2, denoise_d2], dim=1)
        segment_d2 = self.segment_dec2(segment_d2)
        
        segment_d1 = self.segment_up1(segment_d2)
        if segment_d1.shape[2:] != e1.shape[2:]:
            segment_d1 = torch.nn.functional.interpolate(segment_d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        # Use denoised decoder features
        segment_d1 = torch.cat([segment_d1, denoise_d1], dim=1)
        segment_d1 = self.segment_dec1(segment_d1)
        
        # Two output heads
        predicted_noise = self.denoise_head(denoise_d1)  # No activation (can be positive/negative)
        embeddings = self.embedding_head(segment_d1)  # Embeddings: [B, embedding_dim, H, W]
        
        return predicted_noise, embeddings


def create_instance_mask(paths: np.ndarray, shape: Tuple[int, int], 
                         peak_width_samples: float = 2.0, max_tracks: int = 3) -> np.ndarray:
    """Create instance segmentation mask from particle paths.
    
    Returns:
    --------
    mask : np.ndarray
        Instance labels: 0=background, 1..N=instance IDs (one per particle)
        Shape: (length, width), dtype: int32
    """
    length, width = shape
    mask = np.zeros((length, width), dtype=np.int32)  # Background = 0
    
    # Handle single particle case
    if paths.ndim == 1:
        paths = paths.reshape(1, -1)
    
    n_particles, path_length = paths.shape
    xs = np.arange(width, dtype=np.float32)
    
    for t in range(min(length, path_length)):
        for i in range(n_particles):
            pos = paths[i, t]
            if not np.isnan(pos):
                # Create Gaussian around particle position
                gaussian = np.exp(-0.5 * ((xs - pos) / peak_width_samples) ** 2)
                # Assign to instance mask: instance ID = i + 1 (0 is background)
                mask_indices = gaussian > 0.1  # Threshold for assignment
                # For overlapping particles, use maximum (later particle overwrites)
                # This creates distinct instance IDs
                mask[t, mask_indices] = np.where(
                    gaussian[mask_indices] > 0.1,
                    i + 1,  # Instance ID (1-indexed)
                    mask[t, mask_indices]
                )
    
    return mask


class MultiTaskDataset(Dataset):
    """Dataset for multi-task training: noisy kymograph -> (noise, segmentation_mask)."""
    
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
    ) -> None:
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
        
        # Create instance mask (for embedding-based segmentation task)
        peak_width_samples = self.peak_width / self.dx
        mask = create_instance_mask(
            paths,
            shape=(self.length, self.width),
            peak_width_samples=max(peak_width_samples, self.mask_peak_width_samples),
            max_tracks=self.max_trajectories
        )
        
        # Convert to tensors
        noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).float()
        noise_tensor = torch.from_numpy(true_noise).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).long()  # Long tensor for instance mask (0=bg, 1..N=instances)
        
        return noisy_tensor, noise_tensor, mask_tensor


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task model training."""
    epochs: int = 12
    batch_size: int = 8
    learning_rate: float = 1e-3
    denoise_loss_weight: float = 1.0  # Weight for denoising loss
    segment_loss_weight: float = 2.0  # Weight for segmentation loss (increased to prioritize segmentation)
    denoise_loss: str = "l2"  # "l2" or "l1"
    segment_loss: str = "bce"  # "bce" (BinaryCrossEntropy), "dice" (Dice for binary), or "focal" (Focal Loss)
    focal_alpha: float = 0.25  # Alpha parameter for focal loss (class weighting)
    focal_gamma: float = 2.0  # Gamma parameter for focal loss (focusing parameter)
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
    segment_class_weights: Optional[Tuple[float, ...]] = None  # Deprecated: kept for compatibility, use pos_weight instead
    segment_pos_weight: Optional[float] = None  # Positive class weight for binary segmentation (None = auto, default 3.0)
    segment_smoothness_weight: float = 0.0  # Weight for spatial smoothness loss (0 = disabled)
    segment_flatness_loss: str = "tv"  # "none", "tv" (Total Variation - BEST for flatness), "laplacian", or "smoothness" (L2 gradients)
    segment_flatness_weight: float = 0.2  # Weight for flatness loss (reduced to not compete with segmentation loss)
    segment_flatness_range: int = 1  # Range over which to enforce flatness (1 = immediate neighbors, 2-3 = longer range)
    segment_entropy_weight: float = 0.5  # Weight for entropy regularization (penalizes uncertain predictions, standard approach)
    # Embedding loss parameters
    embedding_loss_weight: float = 2.0  # Weight for embedding loss (replaces segmentation loss)
    embedding_var_weight: float = 1.0  # Weight for variance loss (pull same-instance together)
    embedding_sep_weight: float = 1.5  # Weight for separation loss (push different instances apart) - ENHANCED
    embedding_bg_weight: float = 0.5  # Weight for background loss
    embedding_smoothness_weight: float = 0.1  # Weight for smoothness loss (reduces noise)
    embedding_reg_weight: float = 0.01  # Weight for L2 regularization on embeddings
    embedding_separation_margin: float = 2.0  # Minimum distance between instance embeddings - ENHANCED
    embedding_dim: int = 2  # Dimension of embedding vectors (2D like CellPose)
    resume_from: Optional[str] = None  # Path to checkpoint to resume training from (None = auto-detect latest)
    resume_epoch: Optional[int] = None  # Epoch number to resume from (if None, inferred from checkpoint)
    auto_resume: bool = False  # Automatically resume from latest checkpoint if available (opt-in)
    init_weights: Optional[str] = None  # Initialize model weights from this path (no optimizer state)


def compute_instance_iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """Compute IoU between two binary instance masks."""
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def extract_instances_from_mask(mask: np.ndarray, class_id: int) -> list[np.ndarray]:
    """Extract individual instance masks for a given class from a multi-class mask.
    
    Returns list of binary masks, one per connected component.
    """
    binary = (mask == class_id).astype(np.uint8)
    labeled, num_features = label(binary)
    instances = []
    for i in range(1, num_features + 1):
        instances.append((labeled == i).astype(np.float32))
    return instances


def hungarian_matching_loss(
    pred_logits: torch.Tensor,
    target_mask: torch.Tensor,
    n_classes: int,
    base_criterion: nn.Module,
) -> torch.Tensor:
    """
    Compute instance segmentation loss using Hungarian matching.
    
    For each sample in the batch:
    1. Extract predicted instances (one per class > 0)
    2. Extract ground truth instances (one per class > 0)
    3. Compute IoU matrix between all predicted and GT instances
    4. Use Hungarian matching to find optimal assignment
    5. Compute loss only on matched pairs
    
    Parameters:
    -----------
    pred_logits : torch.Tensor
        [B, C, H, W] predicted logits
    target_mask : torch.Tensor
        [B, H, W] ground truth class labels
    n_classes : int
        Number of classes (including background)
    base_criterion : nn.Module
        Base loss criterion (e.g., CrossEntropyLoss)
    
    Returns:
    --------
    loss : torch.Tensor
        Scalar loss value
    """
    B, C, H, W = pred_logits.shape
    device = pred_logits.device
    
    # Convert to numpy for instance extraction
    pred_probs = torch.softmax(pred_logits, dim=1)  # [B, C, H, W]
    pred_labels = torch.argmax(pred_logits, dim=1)  # [B, H, W]
    
    total_loss = 0.0
    n_valid_samples = 0
    
    for b in range(B):
        pred_mask_np = pred_labels[b].cpu().numpy()  # [H, W]
        target_mask_np = target_mask[b].cpu().numpy()  # [H, W]
        
        # Extract GT instances (one per class > 0)
        gt_instances = []
        gt_classes = []
        for class_id in range(1, n_classes):  # Skip background (0)
            instances = extract_instances_from_mask(target_mask_np, class_id)
            for inst in instances:
                gt_instances.append(inst)
                gt_classes.append(class_id)
        
        # Extract predicted instances (one per class > 0)
        pred_instances = []
        pred_classes = []
        for class_id in range(1, n_classes):  # Skip background (0)
            instances = extract_instances_from_mask(pred_mask_np, class_id)
            for inst in instances:
                pred_instances.append(inst)
                pred_classes.append(class_id)
        
        # If no instances, use standard loss
        if len(gt_instances) == 0 and len(pred_instances) == 0:
            # Both empty - just compute standard loss for this sample
            sample_loss = base_criterion(
                pred_logits[b:b+1],  # [1, C, H, W]
                target_mask[b:b+1]   # [1, H, W]
            )
            total_loss += sample_loss
            n_valid_samples += 1
            continue
        
        # Build cost matrix: -IoU (negative because we want to maximize IoU)
        n_pred = len(pred_instances)
        n_gt = len(gt_instances)
        
        if n_pred == 0 or n_gt == 0:
            # Mismatch: penalize all unmatched instances
            # Use standard loss but with high weight for unmatched
            sample_loss = base_criterion(
                pred_logits[b:b+1],
                target_mask[b:b+1]
            )
            # Penalty for unmatched instances
            if n_pred > 0:
                sample_loss = sample_loss * (1.0 + 0.5 * n_pred)
            if n_gt > 0:
                sample_loss = sample_loss * (1.0 + 0.5 * n_gt)
            total_loss += sample_loss
            n_valid_samples += 1
            continue
        
        # Compute IoU matrix
        cost_matrix = np.zeros((n_pred, n_gt))
        for i, pred_inst in enumerate(pred_instances):
            for j, gt_inst in enumerate(gt_instances):
                iou = compute_instance_iou(pred_inst, gt_inst)
                cost_matrix[i, j] = -iou  # Negative because Hungarian minimizes
        
        # Hungarian matching
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        
        # Create matched target mask: remap GT classes to match predicted classes
        matched_target = target_mask_np.copy()
        
        # For matched pairs, remap GT class to predicted class
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            pred_class = pred_classes[pred_idx]
            gt_class = gt_classes[gt_idx]
            gt_mask = (target_mask_np == gt_class)
            matched_target[gt_mask] = pred_class
        
        # For unmatched GT instances, keep original class (will be penalized)
        # For unmatched pred instances, they'll be penalized naturally
        
        # Convert back to tensor and compute loss
        matched_target_tensor = torch.from_numpy(matched_target).long().to(device)
        
        sample_loss = base_criterion(
            pred_logits[b:b+1],  # [1, C, H, W]
            matched_target_tensor.unsqueeze(0)  # [1, H, W]
        )
        
        total_loss += sample_loss
        n_valid_samples += 1
    
    if n_valid_samples == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return total_loss / n_valid_samples


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


def compute_embedding_smoothness(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute smoothness loss on embeddings.
    
    Encourages spatial coherence, reduces noise in embedding space.
    Computes L2 norm of gradients in time and space dimensions.
    
    Parameters:
    -----------
    embeddings : torch.Tensor
        Embeddings [B, D, H, W] where D is embedding dimension
    
    Returns:
    --------
    loss : torch.Tensor
        Smoothness loss (scalar)
    """
    B, D, H, W = embeddings.shape
    
    # Compute gradients in time (vertical) and space (horizontal) dimensions
    # Time gradient: difference along dimension 2 (H)
    time_grad = embeddings[:, :, 1:, :] - embeddings[:, :, :-1, :]  # [B, D, H-1, W]
    # Space gradient: difference along dimension 3 (W)
    space_grad = embeddings[:, :, :, 1:] - embeddings[:, :, :, :-1]  # [B, D, H, W-1]
    
    # L2 norm of gradients (encourages smoothness)
    smoothness = (time_grad ** 2).mean() + (space_grad ** 2).mean()
    
    return smoothness


def compute_variance_loss(pred_embeddings: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Compute variance loss: pull same-instance pixels together.
    
    For each instance in target_mask, compute variance of embeddings.
    Lower variance = embeddings are closer together (good).
    
    Parameters:
    -----------
    pred_embeddings : torch.Tensor
        Predicted embeddings [B, D, H, W]
    target_mask : torch.Tensor
        Instance mask [B, H, W] with values 0=background, 1..N=instances
    
    Returns:
    --------
    loss : torch.Tensor
        Variance loss (scalar)
    """
    B, D, H, W = pred_embeddings.shape
    device = pred_embeddings.device
    
    # Reshape embeddings to [B, H*W, D]
    embeddings_flat = pred_embeddings.permute(0, 2, 3, 1).reshape(B, H * W, D)
    target_flat = target_mask.reshape(B, H * W)
    
    total_variance = 0.0
    n_instances = 0
    
    for b in range(B):
        unique_labels = torch.unique(target_flat[b])
        
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
            
            # Get embeddings for this instance
            instance_mask = (target_flat[b] == label)  # [H*W]
            if instance_mask.sum() < 2:  # Need at least 2 pixels
                continue
            
            instance_embeddings = embeddings_flat[b, instance_mask]  # [N, D]
            
            # Compute mean embedding for this instance
            mean_embedding = instance_embeddings.mean(dim=0)  # [D]
            
            # Compute variance (L2 distance from mean)
            variance = ((instance_embeddings - mean_embedding.unsqueeze(0)) ** 2).mean()
            total_variance += variance
            n_instances += 1
    
    if n_instances == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return total_variance / n_instances


def compute_separation_loss(pred_embeddings: torch.Tensor, target_mask: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Compute enhanced separation loss: push different instances apart.
    
    Uses a stronger penalty function that:
    1. Applies squared penalty for stronger gradients when instances are close
    2. Uses exponential penalty for very close instances (distance < margin/2)
    3. Ensures minimum distance >= margin between all instance pairs
    
    Parameters:
    -----------
    pred_embeddings : torch.Tensor
        Predicted embeddings [B, D, H, W]
    target_mask : torch.Tensor
        Instance mask [B, H, W] with values 0=background, 1..N=instances
    margin : float
        Minimum distance between instance embeddings (default: 1.0)
    
    Returns:
    --------
    loss : torch.Tensor
        Separation loss (scalar)
    """
    B, D, H, W = pred_embeddings.shape
    device = pred_embeddings.device
    
    # Reshape embeddings to [B, H*W, D]
    embeddings_flat = pred_embeddings.permute(0, 2, 3, 1).reshape(B, H * W, D)
    target_flat = target_mask.reshape(B, H * W)
    
    total_separation_loss = 0.0
    n_pairs = 0
    
    for b in range(B):
        unique_labels = torch.unique(target_flat[b])
        instance_labels = unique_labels[unique_labels != 0]  # Remove background
        
        if len(instance_labels) < 2:
            continue
        
        # Compute mean embedding for each instance
        instance_means = []
        for label in instance_labels:
            instance_mask = (target_flat[b] == label)
            if instance_mask.sum() == 0:
                continue
            
            instance_embeddings = embeddings_flat[b, instance_mask]  # [N, D]
            mean_embedding = instance_embeddings.mean(dim=0)  # [D]
            instance_means.append(mean_embedding)
        
        if len(instance_means) < 2:
            continue
        
        # Compute pairwise distances
        for i in range(len(instance_means)):
            for j in range(i + 1, len(instance_means)):
                mean_i = instance_means[i]  # [D]
                mean_j = instance_means[j]  # [D]
                
                # L2 distance between means
                distance = torch.norm(mean_i - mean_j)
                
                # Enhanced separation loss with stronger penalty
                if distance < margin:
                    # When distance < margin, apply penalty
                    gap = margin - distance
                    
                    # For very close instances (distance < margin/2), use exponential penalty
                    # This creates very strong gradients when instances are too close
                    if distance < margin / 2:
                        # Exponential penalty: exp(2 * gap / margin) - 1
                        # This gives much stronger penalty for very close instances
                        separation_loss = torch.exp(2.0 * gap / margin) - 1.0
                    else:
                        # For moderately close instances, use squared penalty
                        # This gives stronger gradients than linear hinge loss
                        separation_loss = gap ** 2
                else:
                    # Instances are far enough apart, no penalty
                    separation_loss = torch.tensor(0.0, device=device)
                
                total_separation_loss += separation_loss
                n_pairs += 1
    
    if n_pairs == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return total_separation_loss / n_pairs


def compute_background_loss(pred_embeddings: torch.Tensor, target_mask: torch.Tensor, 
                            background_embedding: torch.Tensor = None) -> torch.Tensor:
    """Compute background loss: pull background pixels to a fixed embedding.
    
    Parameters:
    -----------
    pred_embeddings : torch.Tensor
        Predicted embeddings [B, D, H, W]
    target_mask : torch.Tensor
        Instance mask [B, H, W] with values 0=background, 1..N=instances
    background_embedding : torch.Tensor, optional
        Target embedding for background [D]. If None, uses zero vector.
    
    Returns:
    --------
    loss : torch.Tensor
        Background loss (scalar)
    """
    B, D, H, W = pred_embeddings.shape
    device = pred_embeddings.device
    
    if background_embedding is None:
        background_embedding = torch.zeros(D, device=device)
    else:
        background_embedding = background_embedding.to(device)
    
    # Reshape embeddings to [B, H*W, D]
    embeddings_flat = pred_embeddings.permute(0, 2, 3, 1).reshape(B, H * W, D)
    target_flat = target_mask.reshape(B, H * W)
    
    total_bg_loss = pred_embeddings.new_tensor(0.0)
    total_bg_pixels = 0
    
    for b in range(B):
        bg_mask = (target_flat[b] == 0)  # Background pixels
        if bg_mask.sum() == 0:
            continue
        
        bg_embeddings = embeddings_flat[b, bg_mask]  # [N, D]
        n_pixels = bg_mask.sum().item()
        
        # L2 distance from background embedding
        bg_loss = ((bg_embeddings - background_embedding.unsqueeze(0)) ** 2).mean()
        total_bg_loss += bg_loss * n_pixels
        total_bg_pixels += n_pixels
    
    if total_bg_pixels == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return total_bg_loss / total_bg_pixels


def compute_instance_norm_loss(pred_embeddings: torch.Tensor, target_mask: torch.Tensor,
                               target_norm: float = 1.0) -> torch.Tensor:
    """Encourage foreground embeddings to maintain approximately unit norm."""
    B, D, H, W = pred_embeddings.shape
    device = pred_embeddings.device
    
    embeddings_flat = pred_embeddings.permute(0, 2, 3, 1).reshape(B, H * W, D)
    target_flat = target_mask.reshape(B, H * W)
    
    total_loss = pred_embeddings.new_tensor(0.0)
    total_pixels = 0
    
    for b in range(B):
        instance_mask = target_flat[b] > 0
        if instance_mask.sum() == 0:
            continue
        inst_embeddings = embeddings_flat[b, instance_mask]
        norms = torch.norm(inst_embeddings, dim=1)
        loss = ((norms - target_norm) ** 2).mean()
        n_pixels = instance_mask.sum().item()
        total_loss += loss * n_pixels
        total_pixels += n_pixels
    
    if total_pixels == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return total_loss / total_pixels




def embedding_loss(pred_embeddings: torch.Tensor, target_mask: torch.Tensor,
                   var_weight: float = 1.0, sep_weight: float = 0.5, bg_weight: float = 0.5,
                   smoothness_weight: float = 0.1, reg_weight: float = 0.01,
                   separation_margin: float = 1.0,
                   norm_weight: float = 0.2) -> torch.Tensor:
    """
    Enhanced embedding loss with smoothness regularization for noise robustness.
    
    Parameters:
    -----------
    pred_embeddings : torch.Tensor
        Predicted embeddings [B, D, H, W]
    target_mask : torch.Tensor
        Instance mask [B, H, W] with values 0=background, 1..N=instances
    var_weight : float
        Weight for variance loss (pull same-instance together)
    sep_weight : float
        Weight for separation loss (push different instances apart)
    bg_weight : float
        Weight for background loss
    smoothness_weight : float
        Weight for smoothness loss (reduces noise)
    reg_weight : float
        Weight for L2 regularization
    separation_margin : float
        Minimum distance between instance embeddings
    
    Returns:
    --------
    loss : torch.Tensor
        Total embedding loss (scalar)
    """
    # 1. Variance loss (pull same-instance together)
    var_loss = compute_variance_loss(pred_embeddings, target_mask)
    
    # 2. Separation loss (push different instances apart)
    sep_loss = compute_separation_loss(pred_embeddings, target_mask, margin=separation_margin)
    
    # 3. Background loss
    bg_loss = compute_background_loss(pred_embeddings, target_mask)
    
    # 4. Instance norm consistency (push embeddings away from background norm)
    norm_loss = compute_instance_norm_loss(pred_embeddings, target_mask)
    
    # 5. Smoothness loss (reduces noise)
    smoothness_loss = compute_embedding_smoothness(pred_embeddings)
    
    # 6. L2 regularization
    reg_loss = (pred_embeddings ** 2).mean()
    
    total_loss = (var_weight * var_loss + 
                  sep_weight * sep_loss + 
                  bg_weight * bg_loss +
                  norm_weight * norm_loss +
                  smoothness_weight * smoothness_loss +
                  reg_weight * reg_loss)
    
    return total_loss


def entropy_regularization_loss(pred_logits: torch.Tensor) -> torch.Tensor:
    """Entropy regularization loss (standard approach for penalizing uncertainty).
    
    Penalizes high entropy (uncertain predictions) in the probability distribution.
    This encourages confident predictions without specifically targeting 0.5.
    
    Entropy for binary: H(p) = -p*log(p) - (1-p)*log(1-p)
    - Maximum entropy at p=0.5 (most uncertain)
    - Minimum entropy at p=0 or p=1 (most certain)
    
    Parameters:
    -----------
    pred_logits : torch.Tensor
        Binary logits [B, 1, H, W] or [B, H, W]
    
    Returns:
    --------
    loss : torch.Tensor
        Entropy regularization loss (scalar)
    """
    if pred_logits.dim() == 4:
        pred_logits = pred_logits.squeeze(1)  # [B, H, W]
    
    # Convert logits to probabilities
    pred_probs = torch.sigmoid(pred_logits)  # [B, H, W], values in [0, 1]
    
    # Compute binary entropy: -p*log(p) - (1-p)*log(1-p)
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    entropy = -(pred_probs * torch.log(pred_probs + eps) + 
                (1 - pred_probs) * torch.log(1 - pred_probs + eps))
    
    # Return mean entropy (higher entropy = more uncertain = higher penalty)
    return entropy.mean()


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
        embedding_dim=config.embedding_dim
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
    
    # Embedding loss (replaces segmentation loss)
    print(f"  Using embedding-based segmentation")
    print(f"  Embedding dimension: {config.embedding_dim}")
    print(f"  Embedding loss weights: var={config.embedding_var_weight}, "
          f"sep={config.embedding_sep_weight}, bg={config.embedding_bg_weight}, "
          f"smooth={config.embedding_smoothness_weight}, reg={config.embedding_reg_weight}")
    
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
    print(f"  Embedding loss (weight: {config.embedding_loss_weight})")
    print(f"  Weight decay (L2): {config.weight_decay}")
    if config.encoder_dropout > 0:
        print(f"  Dropout (encoder): {config.encoder_dropout}")
    if config.dropout > 0:
        print(f"  Dropout (bottleneck): {config.dropout}")
    if config.decoder_dropout > 0:
        print(f"  Dropout (decoder): {config.decoder_dropout}")
    if config.label_smoothing > 0:
        print(f"  Label smoothing: {config.label_smoothing}")
    print(f"  Dataset size: {len(dataset)}")
    if config.checkpoint_dir:
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        print(f"  Checkpoint directory: {config.checkpoint_dir}")
    
    model.train()
    
    for epoch in range(start_epoch, config.epochs):
        epoch_start_time = time.time()
        epoch_denoise_loss = 0.0
        epoch_segment_loss = 0.0
        epoch_total_loss = 0.0
        batch_count = 0
        
        # Progress bar for batches in this epoch
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch+1}/{config.epochs}",
            unit="batch"
        )
        
        for batch_idx, (noisy, true_noise, true_mask) in pbar:
            noisy = noisy.to(config.device)
            true_noise = true_noise.to(config.device)
            true_mask = true_mask.to(config.device)  # Instance mask: [B, H, W] with values 0=bg, 1..N=instances
            
            optimizer.zero_grad()
            
            # Forward pass: get denoising and embeddings
            pred_noise, pred_embeddings = model(noisy)  # embeddings: [B, D, H, W]
            
            # Compute losses
            denoise_loss = denoise_criterion(pred_noise, true_noise)
            
            # Embedding loss (replaces segmentation loss)
            embedding_loss_val = embedding_loss(
                pred_embeddings,
                true_mask,
                var_weight=config.embedding_var_weight,
                sep_weight=config.embedding_sep_weight,
                bg_weight=config.embedding_bg_weight,
                smoothness_weight=config.embedding_smoothness_weight,
                reg_weight=config.embedding_reg_weight,
                separation_margin=config.embedding_separation_margin
            )
            
            # Combined loss
            total_loss = (
                config.denoise_loss_weight * denoise_loss +
                config.embedding_loss_weight * embedding_loss_val
            )
            
            # Backward pass
            total_loss.backward()
            
            # Monitor gradients for embedding head (diagnostic)
            if batch_idx == 0 and epoch % 2 == 0:  # Print every 2 epochs, first batch
                emb_head_grad = model.embedding_head.weight.grad
                if emb_head_grad is not None:
                    grad_norm = emb_head_grad.norm().item()
                    print(f"\n  [Debug] Embedding head grad norm: {grad_norm:.6f}")
            
            # Gradient clipping
            if config.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            
            epoch_denoise_loss += denoise_loss.item()
            epoch_segment_loss += embedding_loss_val.item()
            epoch_total_loss += total_loss.item()
            batch_count += 1
            
            # Update progress bar with current losses
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'denoise': f'{denoise_loss.item():.4f}',
                'embed': f'{embedding_loss_val.item():.4f}',
                'lr': f'{current_lr:.2e}',
                'total': f'{total_loss.item():.4f}'
            })
        
        pbar.close()
        
        avg_denoise_loss = epoch_denoise_loss / batch_count
        avg_segment_loss = epoch_segment_loss / batch_count
        avg_total_loss = epoch_total_loss / batch_count
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch + 1}/{config.epochs} completed: "
              f"Total={avg_total_loss:.6f}, "
              f"Denoise={avg_denoise_loss:.6f}, "
              f"Embedding={avg_segment_loss:.6f}, "
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


def load_multitask_model(path: str, device: str = None, max_tracks: int = 3, embedding_dim: int = 2) -> MultiTaskUNet:
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
        embedding_dim=embedding_dim
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
    chunk_size: int = 512,
    overlap: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply multi-task model to kymograph with chunking.
    
    Returns denoised kymograph and embeddings (no clustering).
    
    Returns:
    --------
    denoised : np.ndarray
        Denoised kymograph, shape (time, width), dtype float32, values in [0, 1]
    embeddings : np.ndarray
        Embeddings, shape (time, width, embedding_dim), dtype float32
    """
    if device is None:
        device = _default_device()
    
    model.eval()
    time_len, width = kymograph.shape
    
    # If kymograph fits in one chunk, process directly
    if time_len <= chunk_size:
        with torch.no_grad():
            input_tensor = torch.from_numpy(kymograph).unsqueeze(0).unsqueeze(0).float().to(device)
            pred_noise, pred_embeddings = model(input_tensor)
            denoised = torch.clamp(input_tensor - pred_noise, 0.0, 1.0).squeeze().cpu().numpy()
            
            # Convert embeddings: [B, D, T, W] -> [T, W, D]
            embeddings_np = pred_embeddings.squeeze().permute(1, 2, 0).cpu().numpy()
            
            # Clear GPU memory
            del input_tensor, pred_noise, pred_embeddings
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            
        return denoised, embeddings_np
    
    # Process in chunks with overlap
    denoised = np.zeros((time_len, width), dtype=np.float32)
    weights = np.zeros((time_len, width), dtype=np.float32)
    
    # Get embedding dimension from first chunk
    embedding_dim = None
    embeddings_all = None
    
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
            pred_noise_chunk, pred_embeddings_chunk = model(chunk_tensor)
            denoised_chunk = torch.clamp(chunk_tensor - pred_noise_chunk, 0.0, 1.0).squeeze().cpu().numpy()
            
            # Get embeddings: [B, D, T, W] -> [T, W, D]
            embeddings_chunk = pred_embeddings_chunk.squeeze().permute(1, 2, 0).cpu().numpy()
            
            # Clear GPU memory immediately
            del chunk_tensor, pred_noise_chunk, pred_embeddings_chunk
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            
            # Extract actual size (remove padding)
            actual_len = end - start
            denoised_chunk = denoised_chunk[:actual_len]
            embeddings_chunk = embeddings_chunk[:actual_len]
            window_chunk = window[:actual_len]
            
            # Initialize embeddings array
            if embeddings_all is None:
                embedding_dim = embeddings_chunk.shape[2]
                embeddings_all = np.zeros((time_len, width, embedding_dim), dtype=np.float32)
            
            # Blend denoised and embeddings with window
            weight_chunk = window_chunk[:, np.newaxis]
            denoised[start:end] += denoised_chunk * weight_chunk
            embeddings_all[start:end] += embeddings_chunk * weight_chunk[:, :, np.newaxis]
            weights[start:end] += weight_chunk
            
            # Clear chunk data
            del denoised_chunk, embeddings_chunk
            
            # Move to next chunk
            start += chunk_size - overlap
    
    # Normalize denoised and embeddings by weights
    denoised = np.divide(denoised, weights, out=np.zeros_like(denoised), where=weights > 0)
    weights_expanded = weights[:, :, np.newaxis]
    embeddings_all = np.divide(embeddings_all, weights_expanded, 
                              out=np.zeros_like(embeddings_all), where=weights_expanded > 0)
    del weights, weights_expanded
    
    # Clear GPU cache
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    
    return denoised, embeddings_all


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
        n_samples=1024,
        length=512,
        width=512,
        radii_nm=(3.0, 70.0),  # Extended to match test data (test shows up to ~68 nm)
        contrast=(0.5, 1.1),    # Extended to match test data (test shows up to ~1.03)
        noise_level=(0.08, 0.8),  # Extended upper bound to include high noise levels (>0.5)
        multi_trajectory_prob=1.0,  # 100% multi-particle examples
        max_trajectories=3,
        mask_peak_width_samples=2.0,
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
        batch_size=8,
        learning_rate=1.5e-3,  # Slightly reduced from 2e-3 for more stable training
        denoise_loss_weight=1.0,
        embedding_loss_weight=2.0,  # Weight for embedding loss
        denoise_loss="l2",
        # Embedding loss parameters
        embedding_var_weight=1.0,  # Pull same-instance together
        embedding_sep_weight=1.5,  # Push different instances apart - ENHANCED (was 0.5)
        embedding_bg_weight=0.5,  # Background loss
        embedding_smoothness_weight=0.1,  # Smoothness regularization (reduces noise)
        embedding_reg_weight=0.01,  # L2 regularization
        embedding_separation_margin=2.0,  # Minimum distance between instances - ENHANCED (was 1.0)
        embedding_dim=2,  # Embedding dimension (2D like CellPose)
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
