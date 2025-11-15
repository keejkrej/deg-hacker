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
    """Convolutional block with two conv layers, batch norm, and ReLU."""
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
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
    """U-Net with two output heads: denoising and segmentation.
    
    Architecture:
    - Shared encoder-decoder backbone
    - Two output heads:
      1. Denoising head: predicts noise (DDPM-style)
      2. Segmentation head: predicts multi-class labels (n_tracks + background)
    """
    
    def __init__(self, base_channels: int = 48, use_bn: bool = True, max_tracks: int = 3) -> None:
        super().__init__()
        self.max_tracks = max_tracks
        self.n_classes = max_tracks + 1  # max_tracks + background (class 0)
        
        # Shared encoder
        self.enc1 = ConvBlock(1, base_channels, use_bn=use_bn)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, use_bn=use_bn)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, use_bn=use_bn)
        
        self.down = nn.MaxPool2d(2)
        
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8, use_bn=use_bn)
        
        # Shared decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4, use_bn=use_bn)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2, use_bn=use_bn)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels, use_bn=use_bn)
        
        # Two output heads
        # Head 1: Denoising (predicts noise, no activation)
        self.denoise_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        nn.init.xavier_uniform_(self.denoise_head.weight, gain=0.1)
        nn.init.constant_(self.denoise_head.bias, 0.0)
        
        # Head 2: Segmentation (predicts class logits, no activation - use CrossEntropyLoss)
        # Output: n_classes channels (background=0, track1=1, track2=2, track3=3)
        self.segment_head = nn.Conv2d(base_channels, self.n_classes, kernel_size=1)
        nn.init.xavier_uniform_(self.segment_head.weight, gain=1.0)
        nn.init.constant_(self.segment_head.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returns (predicted_noise, segmentation_mask)."""
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
        
        # Two output heads
        predicted_noise = self.denoise_head(d1)  # No activation (can be positive/negative)
        segmentation_logits = self.segment_head(d1)  # Class logits: [B, n_classes, H, W]
        
        return predicted_noise, segmentation_logits


def create_segmentation_mask(paths: np.ndarray, shape: Tuple[int, int], 
                             peak_width_samples: float = 2.0, max_tracks: int = 3) -> np.ndarray:
    """Create multi-class segmentation mask from particle paths.
    
    Returns:
    --------
    mask : np.ndarray
        Class labels: 0=background, 1=track1, 2=track2, 3=track3
        Shape: (length, width), dtype: int64
    """
    length, width = shape
    mask = np.zeros((length, width), dtype=np.int64)  # Background = 0
    
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
                # Assign class label: track i gets class i+1 (class 0 is background)
                track_class = i + 1
                # Only assign if Gaussian is strong enough and not already assigned to a higher priority track
                # Priority: earlier tracks (lower index) have priority
                mask_indices = gaussian > 0.1  # Threshold for assignment
                # Only assign where mask is still background (0) or where this track's Gaussian is stronger
                for x_idx in np.where(mask_indices)[0]:
                    if mask[t, x_idx] == 0 or gaussian[x_idx] > 0.5:  # Override if strong enough
                        mask[t, x_idx] = track_class
    
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
        
        # Create segmentation mask (for segmentation task) - multi-class labels
        peak_width_samples = self.peak_width / self.dx
        mask = create_segmentation_mask(
            paths,
            shape=(self.length, self.width),
            peak_width_samples=max(peak_width_samples, self.mask_peak_width_samples),
            max_tracks=self.max_trajectories
        )
        
        # Convert to tensors
        noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).float()
        noise_tensor = torch.from_numpy(true_noise).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).long()  # Long tensor for class labels
        
        return noisy_tensor, noise_tensor, mask_tensor


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task model training."""
    epochs: int = 12
    batch_size: int = 8
    learning_rate: float = 1e-3
    denoise_loss_weight: float = 1.0  # Weight for denoising loss
    segment_loss_weight: float = 1.0  # Weight for segmentation loss
    denoise_loss: str = "l2"  # "l2" or "l1"
    segment_loss: str = "ce"  # "ce" (CrossEntropy with Hungarian matching) or "dice" (Dice for multi-class)
    use_gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    use_lr_scheduler: bool = True
    device: str = _default_device()
    checkpoint_dir: Optional[str] = None  # Directory to save checkpoints (None = don't save)
    save_best: bool = True  # Save best model based on total loss
    checkpoint_every: int = 1  # Save checkpoint every N epochs (1 = every epoch)
    segment_class_weights: Optional[Tuple[float, ...]] = None  # Class weights for segmentation (None = auto)


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


def dice_loss_multiclass(pred: torch.Tensor, target: torch.Tensor, n_classes: int, smooth: float = 1e-6) -> torch.Tensor:
    """Multi-class Dice loss for segmentation.
    
    Parameters:
    -----------
    pred : torch.Tensor
        Class logits [B, n_classes, H, W]
    target : torch.Tensor
        Class labels [B, H, W] with values in [0, n_classes-1]
    n_classes : int
        Number of classes
    """
    # Convert logits to probabilities
    pred_probs = torch.softmax(pred, dim=1)  # [B, n_classes, H, W]
    
    # One-hot encode target
    target_one_hot = torch.zeros_like(pred_probs)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)  # [B, n_classes, H, W]
    
    # Compute Dice for each class and average
    dice_scores = []
    for c in range(n_classes):
        pred_c = pred_probs[:, c].view(-1)
        target_c = target_one_hot[:, c].view(-1)
        
        intersection = (pred_c * target_c).sum()
        dice = (2.0 * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
        dice_scores.append(dice)
    
    # Return average Dice loss (1 - dice)
    mean_dice = torch.stack(dice_scores).mean()
    return 1.0 - mean_dice


def train_multitask_model(
    config: MultiTaskConfig,
    dataset: MultiTaskDataset,
) -> MultiTaskUNet:
    """Train the multi-task model."""
    
    # Create model
    model = MultiTaskUNet(base_channels=48, use_bn=True, max_tracks=dataset.max_trajectories).to(config.device)
    
    # Loss functions
    if config.denoise_loss == "l2":
        denoise_criterion = nn.MSELoss()
    elif config.denoise_loss == "l1":
        denoise_criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unknown denoise loss: {config.denoise_loss}")
    
    if config.segment_loss == "ce":
        # Use weighted CrossEntropyLoss to handle class imbalance
        # Background (class 0) is much more common, so we need to weight foreground classes higher
        if config.segment_class_weights is not None:
            class_weights = torch.tensor(config.segment_class_weights, device=config.device, dtype=torch.float32)
        else:
            # Auto-compute: down-weight background, up-weight tracks
            # Background typically ~60-80% of pixels, tracks ~5-15% each
            class_weights = torch.ones(model.n_classes, device=config.device, dtype=torch.float32)
            class_weights[0] = 0.1  # Down-weight background (most common)
            class_weights[1:] = 3.0  # Up-weight track classes (rare but important)
        print(f"  Segmentation class weights: {class_weights.cpu().numpy()}")
        base_criterion = nn.CrossEntropyLoss(weight=class_weights)
        # Wrap with Hungarian matching for instance segmentation
        segment_criterion = lambda pred, target: hungarian_matching_loss(
            pred, target, model.n_classes, base_criterion
        )
    elif config.segment_loss == "dice":
        segment_criterion = lambda pred, target: dice_loss_multiclass(pred, target, model.n_classes)
    else:
        raise ValueError(f"Unknown segment loss: {config.segment_loss}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Learning rate scheduler
    scheduler = None
    if config.use_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
    
    # Data loader
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    
    print(f"\nTraining multi-task model on {config.device}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Denoise loss: {config.denoise_loss} (weight: {config.denoise_loss_weight})")
    print(f"  Segment loss: {config.segment_loss} (weight: {config.segment_loss_weight})")
    print(f"  Dataset size: {len(dataset)}")
    if config.checkpoint_dir:
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        print(f"  Checkpoint directory: {config.checkpoint_dir}")
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(config.epochs):
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
            true_mask = true_mask.to(config.device)
            
            optimizer.zero_grad()
            
            # Forward pass: get both outputs
            pred_noise, pred_mask_logits = model(noisy)
            
            # Compute losses
            denoise_loss = denoise_criterion(pred_noise, true_noise)
            if config.segment_loss == "ce":
                # CrossEntropyLoss expects: pred [B, C, H, W], target [B, H, W] with class indices
                segment_loss = segment_criterion(pred_mask_logits, true_mask.squeeze(1))
            elif config.segment_loss == "dice":
                segment_loss = segment_criterion(pred_mask_logits, true_mask.squeeze(1))
            
            # Combined loss
            total_loss = (
                config.denoise_loss_weight * denoise_loss +
                config.segment_loss_weight * segment_loss
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if config.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            
            epoch_denoise_loss += denoise_loss.item()
            epoch_segment_loss += segment_loss.item()
            epoch_total_loss += total_loss.item()
            batch_count += 1
            
            # Update progress bar with current losses
            pbar.set_postfix({
                'denoise': f'{denoise_loss.item():.4f}',
                'segment': f'{segment_loss.item():.4f}',
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
              f"Segment={avg_segment_loss:.6f}, "
              f"Time={epoch_time:.2f}s")
        
        # Save checkpoint if configured
        if config.checkpoint_dir and (epoch + 1) % config.checkpoint_every == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            save_multitask_model(model, checkpoint_path)
        
        # Save best model if configured
        if config.save_best and avg_total_loss < best_loss:
            best_loss = avg_total_loss
            if config.checkpoint_dir:
                best_path = os.path.join(config.checkpoint_dir, "best_model.pth")
                save_multitask_model(model, best_path)
                print(f"  ✓ New best model saved (loss: {best_loss:.6f})")
        
        if scheduler is not None:
            scheduler.step(avg_total_loss)
    
    model.eval()
    return model


def save_multitask_model(model: MultiTaskUNet, path: str) -> None:
    """Save multi-task model."""
    torch.save(model.state_dict(), path)
    print(f"Multi-task model saved to {path}")


def load_multitask_model(path: str, device: str = None, max_tracks: int = 3) -> MultiTaskUNet:
    """Load multi-task model."""
    if device is None:
        device = _default_device()
    
    model = MultiTaskUNet(base_channels=48, use_bn=True, max_tracks=max_tracks).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
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
    
    Returns:
    --------
    denoised : np.ndarray
        Denoised kymograph, shape (time, width)
    segmentation_labels : np.ndarray
        Class labels, shape (time, width), values in [0, n_classes-1]
        where 0=background, 1=track1, 2=track2, 3=track3
    """
    if device is None:
        device = _default_device()
    
    model.eval()
    time_len, width = kymograph.shape
    
    # If kymograph fits in one chunk, process directly
    if time_len <= chunk_size:
        with torch.no_grad():
            input_tensor = torch.from_numpy(kymograph).unsqueeze(0).unsqueeze(0).float().to(device)
            pred_noise, pred_mask_logits = model(input_tensor)
            denoised = torch.clamp(input_tensor - pred_noise, 0.0, 1.0).squeeze().cpu().numpy()
            # Convert logits to class predictions
            pred_classes = torch.argmax(pred_mask_logits, dim=1).squeeze().cpu().numpy().astype(np.int64)
        return denoised, pred_classes
    
    # Process in chunks with overlap
    denoised = np.zeros((time_len, width), dtype=np.float32)
    mask = np.zeros((time_len, width), dtype=np.int64)  # Class labels
    weights = np.zeros((time_len, width), dtype=np.float32)
    
    # For majority voting: accumulate class votes
    from collections import defaultdict
    class_votes = defaultdict(lambda: np.zeros((time_len, width), dtype=np.float32))
    
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
            if chunk.shape[0] < chunk_size:
                padding = np.zeros((chunk_size - chunk.shape[0], width), dtype=chunk.dtype)
                chunk = np.vstack([chunk, padding])
            
            # Process chunk
            chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).float().to(device)
            pred_noise_chunk, pred_mask_logits_chunk = model(chunk_tensor)
            denoised_chunk = torch.clamp(chunk_tensor - pred_noise_chunk, 0.0, 1.0).squeeze().cpu().numpy()
            # Convert logits to class probabilities for voting
            pred_probs_chunk = torch.softmax(pred_mask_logits_chunk, dim=1).squeeze().cpu().numpy()  # [n_classes, H, W]
            
            # Extract actual size (remove padding)
            actual_len = end - start
            denoised_chunk = denoised_chunk[:actual_len]
            pred_probs_chunk = pred_probs_chunk[:, :actual_len, :]  # [n_classes, actual_len, width]
            window_chunk = window[:actual_len]
            
            # Blend denoised with window
            weight_chunk = window_chunk[:, np.newaxis]
            denoised[start:end] += denoised_chunk * weight_chunk
            weights[start:end] += weight_chunk
            
            # Accumulate class probabilities for voting
            for c in range(pred_probs_chunk.shape[0]):
                class_votes[c][start:end] += pred_probs_chunk[c] * weight_chunk
            
            # Move to next chunk
            start += chunk_size - overlap
    
    # Normalize denoised by weights
    denoised = np.divide(denoised, weights, out=np.zeros_like(denoised), where=weights > 0)
    
    # Majority voting: assign class with highest accumulated probability
    for t in range(time_len):
        for w in range(width):
            if weights[t, w] > 0:
                class_scores = [class_votes[c][t, w] / weights[t, w] for c in range(len(class_votes))]
                mask[t, w] = np.argmax(class_scores)
    
    return denoised, mask


if __name__ == "__main__":
    import os
    
    # Create dataset
    print("Creating multi-task dataset...")
    dataset = MultiTaskDataset(
        n_samples=1024,
        length=512,
        width=512,
        multi_trajectory_prob=0.3,
        max_trajectories=3,
        mask_peak_width_samples=2.0,
    )
    
    # Training config
    config = MultiTaskConfig(
        epochs=12,
        batch_size=8,
        learning_rate=1e-3,
        denoise_loss_weight=1.0,
        segment_loss_weight=1.0,
        denoise_loss="l2",
        segment_loss="ce",  # CrossEntropy for multi-class segmentation
        use_gradient_clipping=True,
        max_grad_norm=1.0,
        use_lr_scheduler=True,
        checkpoint_dir="models/checkpoints",  # Save checkpoints after each epoch
        save_best=True,  # Save best model based on loss
        checkpoint_every=1,  # Save checkpoint every epoch
    )
    
    # Train
    model = train_multitask_model(config, dataset)
    
    # Save final model
    os.makedirs("models", exist_ok=True)
    model_path = "models/multitask_unet.pth"
    save_multitask_model(model, model_path)
    print(f"\nFinal model saved to: {model_path}")
