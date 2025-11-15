"""Train a tiny U-Net denoiser for kymograph trajectories.

Uses DDPM-style noise prediction: the model predicts the noise ε that was added
to the clean image, then denoises by subtracting: clean = noisy - predicted_noise.

This approach prevents the model from collapsing to zero outputs (a common issue
with sparse ground truth data) and is more stable than directly predicting clean images.

The network is a small 2D U-Net operating on (time, position) images. Training can
finish within a few minutes on CPU for the default settings.

Training dimensions:
- length (time): 512 (can be chunked for longer sequences)
- width (position): 512 (must match validation data dimension)

Validation data is 20000x512 (time x position). The model processes longer
sequences by chunking along the time dimension.
"""

from dataclasses import dataclass
from typing import Tuple
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from utils import simulate_single_particle
from helpers import generate_kymograph, get_diffusion_coefficient


class SyntheticKymographDataset(Dataset):
    """On-the-fly dataset of synthetic noisy/clean kymographs.
    
    Supports single and multiple trajectories (up to 3 particles).
    Mixes single and multi-trajectory examples during training.
    """

    def __init__(
        self,
        length: int = 512,  # Time dimension (can be chunked)
        width: int = 512,  # Position dimension (must match validation: 512)
        radii_nm: Tuple[float, float] = (3.0, 15.0),
        contrast: Tuple[float, float] = (0.5, 1.0),
        noise_level: Tuple[float, float] = (0.1, 0.5),
        seed: int | None = None,
        peak_width: float = 1.0,
        dt: float = 1.0,
        dx: float = 0.5,
        n_samples: int = 1024,
        multi_trajectory_prob: float = 0.3,  # Probability of generating multi-trajectory sample
        max_trajectories: int = 3,  # Maximum number of trajectories (1-3)
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
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:  # type: ignore[override]
        return self.n_samples

    def sample_parameters(self, n_particles: int = 1) -> Tuple[list[float], list[float], float]:
        """Sample parameters for n_particles trajectories."""
        radii = [float(self.rng.uniform(*self.radii_nm)) for _ in range(n_particles)]
        contrasts = [float(self.rng.uniform(*self.contrast)) for _ in range(n_particles)]
        noise = float(self.rng.uniform(*self.noise_level))
        return radii, contrasts, noise

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        # Decide number of trajectories for this sample
        if self.rng.random() < self.multi_trajectory_prob:
            # Multi-trajectory: 2 or 3 particles
            n_particles = self.rng.integers(2, self.max_trajectories + 1)
        else:
            # Single trajectory
            n_particles = 1
        
        radii, contrasts, noise = self.sample_parameters(n_particles)
        
        # Generate kymograph with multiple trajectories if needed
        if n_particles == 1:
            # Use simulate_single_particle for single trajectory (faster)
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
        else:
            # Generate multi-trajectory kymograph directly
            diffusions = [get_diffusion_coefficient(r) for r in radii]
            noisy, gt, _ = generate_kymograph(
                length=self.length,
                width=self.width,
                diffusion=diffusions,
                contrast=contrasts,
                noise_level=noise,
                peak_width=self.peak_width,
                dt=self.dt,
                dx=self.dx,
            )
        
        noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).float()
        gt_tensor = torch.from_numpy(gt).unsqueeze(0).float()
        return noisy_tensor, gt_tensor


class ConvBlock(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class TinyUNet(nn.Module):
    """U-Net with three resolution levels, batch normalization, and improved capacity."""

    def __init__(self, base_channels: int = 48, use_residual: bool = False, use_bn: bool = True) -> None:
        super().__init__()
        self.use_residual = use_residual
        
        self.enc1 = ConvBlock(1, base_channels, use_bn=use_bn)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, use_bn=use_bn)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, use_bn=use_bn)

        self.down = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8, use_bn=use_bn)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4, use_bn=use_bn)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2, use_bn=use_bn)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels, use_bn=use_bn)

        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=1)
        
        # Initialize output layer to encourage non-zero outputs
        nn.init.xavier_uniform_(self.out_conv.weight, gain=0.1)
        nn.init.constant_(self.out_conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))

        b = self.bottleneck(self.down(e3))

        d3 = self.up3(b)
        # Handle potential dimension mismatch (shouldn't happen for 512x512, but safe)
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

        output = self.out_conv(d1)
        
        # DDPM-style: predict noise directly (no activation, noise can be positive/negative)
        # This prevents the model from collapsing to zero outputs
        # Add residual connection if enabled
        if self.use_residual:
            output = output + x
        
        return output


def _default_device() -> str:
    has_mps = getattr(torch.backends, "mps", None)
    if has_mps and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 8
    lr: float = 1e-3
    loss: str = "l2"  # "l2", "l1", "weighted_l2", "focal", "combined"
    device: str = _default_device()
    use_gradient_clipping: bool = True  # Clip gradients to prevent instability
    max_grad_norm: float = 1.0  # Maximum gradient norm for clipping
    use_residual_connection: bool = False  # Add residual connection from input
    use_lr_scheduler: bool = False  # Use learning rate scheduler
    # For weighted_l2 loss
    weight_nonzero: float = 10.0  # Weight multiplier for non-zero ground truth pixels


class WeightedMSELoss(nn.Module):
    """MSE loss with higher weight for non-zero ground truth pixels."""
    def __init__(self, weight_nonzero: float = 10.0):
        super().__init__()
        self.weight_nonzero = weight_nonzero
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Create weight mask: higher weight for non-zero pixels
        weights = torch.ones_like(target)
        weights[target > 1e-6] = self.weight_nonzero
        return torch.mean(weights * (pred - target) ** 2)


class CombinedLoss(nn.Module):
    """Combined L1 + L2 loss to prevent zero outputs."""
    def __init__(self, l1_weight: float = 0.5, l2_weight: float = 0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1_weight * self.l1_loss(pred, target) + self.l2_weight * self.l2_loss(pred, target)


class FocalLoss(nn.Module):
    """Focal loss variant for sparse targets - focuses on hard examples."""
    def __init__(self, alpha: float = 2.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # For sparse targets, focus more on non-zero regions
        mse = (pred - target) ** 2
        # Higher weight for pixels where target is non-zero
        weights = torch.ones_like(target)
        weights[target > 1e-6] = self.alpha
        # Focus on hard examples (large errors)
        focal_weight = weights * (mse + 1e-6) ** (self.gamma / 2)
        return torch.mean(focal_weight * mse)


def get_loss_fn(name: str, config: TrainingConfig | None = None) -> nn.Module:
    """Get loss function for noise prediction (DDPM-style)."""
    name = name.lower()
    if name == "l2":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    if name == "weighted_l2":
        weight = config.weight_nonzero if config else 10.0
        return WeightedMSELoss(weight_nonzero=weight)
    if name == "combined":
        return CombinedLoss(l1_weight=0.5, l2_weight=0.5)
    if name == "focal":
        return FocalLoss(alpha=2.0, gamma=2.0)
    raise ValueError(f"Unsupported loss '{name}'. Options: l2, l1, weighted_l2, combined, focal")


def train_denoiser(config: TrainingConfig, dataset: SyntheticKymographDataset) -> TinyUNet:
    model = TinyUNet(
        use_residual=config.use_residual_connection
    ).to(config.device)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    # Loss is computed on predicted noise vs true noise
    criterion = get_loss_fn(config.loss, config)
    
    # Learning rate scheduler
    scheduler = None
    if config.use_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )

    print(f"Starting training...")
    print(f"  Device: {config.device}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Loss function: {config.loss}")
    print(f"  Training configuration:")
    print(f"    - Loss function: {config.loss} (on predicted noise vs true noise)")
    print(f"    - Gradient clipping: {config.use_gradient_clipping} (max_norm={config.max_grad_norm})")
    print(f"    - Residual connection: {config.use_residual_connection}")
    print(f"    - LR scheduler: {config.use_lr_scheduler}")
    print(f"  Dataset size: {len(dataset)} samples")
    print(f"  Batches per epoch: {len(dataloader)}")
    print("-" * 60)

    model.train()
    total_start_time = time.time()
    
    # Track losses for plotting
    epoch_losses = []
    
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        batch_count = 0
        
        # Track diagnostics for zero output investigation
        zero_output_detections = 0
        output_stats = []
        gradient_norms = []
        
        for batch_idx, (noisy, clean) in enumerate(dataloader):
            noisy = noisy.to(config.device)
            clean = clean.to(config.device)
            optimizer.zero_grad()
            
            # DDPM-style: Model predicts the noise ε that was added to the clean image
            # Loss: ||predicted_noise - true_noise||²
            # Denoised: clean = noisy - predicted_noise
            predicted_noise = model(noisy)
            denoised = torch.clamp(noisy - predicted_noise, 0.0, 1.0)
            
            # Compute true noise: ε = noisy - clean
            true_noise = noisy - clean
            loss = criterion(predicted_noise, true_noise)
            
            # Check denoised output (not predicted noise)
            output_to_check = denoised
            
            # Check for zero outputs
            denoised_mean = output_to_check.mean().item()
            denoised_max = output_to_check.max().item()
            denoised_min = output_to_check.min().item()
            denoised_std = output_to_check.std().item()
            
            # Detect if output is all zeros (or very close to zero)
            is_zero_output = denoised_max < 1e-6
            if is_zero_output:
                zero_output_detections += 1
            
            output_stats.append({
                'mean': denoised_mean,
                'max': denoised_max,
                'min': denoised_min,
                'std': denoised_std,
                'is_zero': is_zero_output
            })
            
            
            loss.backward()
            
            # Compute gradient norms to detect vanishing gradients
            total_grad_norm = 0.0
            param_count = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.data.norm(2).item()
                    total_grad_norm += param_grad_norm ** 2
                    param_count += 1
            if param_count > 0:
                total_grad_norm = total_grad_norm ** 0.5
                gradient_norms.append(total_grad_norm)
            
            # Gradient clipping to prevent instability
            if config.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            
            batch_loss = loss.item()
            epoch_loss += batch_loss * noisy.size(0)
            batch_count += 1
        
        epoch_loss /= len(dataloader.dataset)
        epoch_time = time.time() - epoch_start_time
        epoch_losses.append(epoch_loss)
        
        # Update learning rate scheduler if enabled
        if scheduler is not None:
            scheduler.step(epoch_loss)
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = config.lr
        
        # Standard textbook-style print message
        print(f"Epoch [{epoch + 1}/{config.epochs}] | Loss: {epoch_loss:.6f} | Time: {epoch_time:.2f}s | LR: {current_lr:.2e}")
        
        # Compute statistics for diagnostics (less verbose, only warnings)
        avg_output_mean = np.mean([s['mean'] for s in output_stats])
        avg_output_max = np.mean([s['max'] for s in output_stats])
        avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0.0
        
        # Only print warnings if issues detected
        zero_pct = (zero_output_detections / batch_count) * 100
        if zero_output_detections > 0:
            print(f"  ⚠️  WARNING: Zero outputs detected: {zero_output_detections}/{batch_count} ({zero_pct:.1f}%)")
        if gradient_norms and avg_grad_norm < 1e-6:
            print(f"  ⚠️  WARNING: Vanishing gradients detected (grad_norm: {avg_grad_norm:.6f})")
        
        # Print overall progress every few epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == config.epochs:
            elapsed_time = time.time() - total_start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            remaining_epochs = config.epochs - (epoch + 1)
            if remaining_epochs > 0:
                estimated_remaining = avg_time_per_epoch * remaining_epochs
                print(f"  Progress: {elapsed_time:.1f}s elapsed | ~{estimated_remaining:.1f}s remaining")

    total_time = time.time() - total_start_time
    print("\n" + "=" * 60)
    print(f"Training completed!")
    print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"  Average time per epoch: {total_time/config.epochs:.2f}s")
    print(f"  Final loss: {epoch_losses[-1]:.6f}")
    print(f"  Best loss: {min(epoch_losses):.6f} (epoch {epoch_losses.index(min(epoch_losses)) + 1})")
    print("=" * 60)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    loss_plot_path = "training_loss.png"
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    print(f"\nLoss plot saved to: {loss_plot_path}")
    plt.close()

    return model


def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_model(path: str, device: str | None = None) -> TinyUNet:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def denoise_kymograph(model: TinyUNet, kymograph: np.ndarray, device: str | None = None) -> np.ndarray:
    """
    Denoise a kymograph by predicting and subtracting noise (DDPM-style).
    
    Works for both single and multiple trajectories. The model denoises the entire
    kymograph regardless of how many trajectories are present.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(kymograph).unsqueeze(0).unsqueeze(0).float().to(device)
        predicted_noise = model(tensor)
        denoised = torch.clamp(tensor - predicted_noise, 0.0, 1.0)
        return denoised.squeeze(0).squeeze(0).cpu().numpy()


def extract_trajectories(
    kymograph: np.ndarray,
    max_trajectories: int = 3,
    min_peak_height: float = 0.1,
    max_jump_distance: float = None,
) -> list[np.ndarray]:
    """
    Extract individual trajectories from a denoised kymograph.
    
    Uses peak finding and tracking to identify up to max_trajectories particles.
    This is a simple implementation - for production use, consider more sophisticated
    tracking algorithms.
    
    Parameters:
    -----------
    kymograph : np.ndarray
        Denoised kymograph of shape (time, position)
    max_trajectories : int
        Maximum number of trajectories to extract (default: 3)
    min_peak_height : float
        Minimum peak height threshold for detection (default: 0.1)
    max_jump_distance : float, optional
        Maximum allowed jump distance between frames (default: 30% of width)
    
    Returns:
    --------
    list[np.ndarray]
        List of trajectory arrays, each of shape (time,). Values are NaN where
        no peak was detected for that trajectory.
    """
    time_len, pos_len = kymograph.shape
    
    if max_jump_distance is None:
        max_jump_distance = pos_len * 0.3
    
    # Initialize trajectories
    trajectories = [np.full(time_len, np.nan) for _ in range(max_trajectories)]
    
    # Track active trajectories
    active_trajectories = []  # List of (traj_idx, last_position, last_time)
    
    for t in range(time_len):
        row = kymograph[t, :]
        
        # Find local maxima (peaks)
        peaks = []
        for i in range(1, pos_len - 1):
            if row[i] > row[i-1] and row[i] > row[i+1] and row[i] > min_peak_height:
                peaks.append(i)
        
        # Sort peaks by intensity (descending)
        peaks.sort(key=lambda i: row[i], reverse=True)
        
        # Match peaks to existing trajectories
        used_peaks = set()
        
        # Try to match each active trajectory to a peak
        for traj_info in active_trajectories[:]:
            traj_idx, last_pos, last_time = traj_info
            
            # Find closest unmatched peak
            best_peak = None
            best_dist = np.inf
            
            for peak_pos in peaks:
                if peak_pos not in used_peaks:
                    dist = abs(peak_pos - last_pos)
                    if dist < best_dist and dist < max_jump_distance:
                        best_dist = dist
                        best_peak = peak_pos
            
            if best_peak is not None:
                # Update trajectory
                trajectories[traj_idx][t] = best_peak
                traj_info[1] = best_peak  # Update last position
                traj_info[2] = t  # Update last time
                used_peaks.add(best_peak)
            else:
                # No matching peak found - trajectory may have ended
                # Keep it active for a few more frames in case it reappears
                if t - last_time > 5:  # Remove if inactive for 5 frames
                    active_trajectories.remove(traj_info)
        
        # Assign remaining peaks to new trajectories
        for peak_pos in peaks:
            if peak_pos not in used_peaks and len(active_trajectories) < max_trajectories:
                # Find an unused trajectory slot
                for traj_idx in range(max_trajectories):
                    if traj_idx not in [ti[0] for ti in active_trajectories]:
                        trajectories[traj_idx][t] = peak_pos
                        active_trajectories.append([traj_idx, peak_pos, t])
                        break
    
    return trajectories


def visualize_denoising_results(
    model: TinyUNet,
    n_samples: int = 3,
    length: int = 512,
    width: int = 512,
    device: str | None = None,
    save_path: str = "denoising_results.png",
) -> None:
    """
    Generate test samples and visualize denoising results.
    
    Parameters:
    -----------
    model : TinyUNet
        Trained denoising model
    n_samples : int
        Number of test samples to generate and visualize
    length : int
        Time dimension of kymograph
    width : int
        Position dimension of kymograph
    device : str, optional
        Device to run model on
    save_path : str
        Path to save the visualization
    """
    device = device or _default_device()
    model = model.to(device)
    model.eval()
    
    print(f"\nGenerating {n_samples} test samples for visualization...")
    
    # Generate test samples: mix of single and multi-trajectory examples
    test_configs = [
        {"type": "single", "radius": 5.0, "contrast": 0.7, "noise": 0.3},
        {"type": "multi", "radii": [5.0, 10.0], "contrasts": [0.7, 0.5], "noise": 0.3},
        {"type": "multi", "radii": [7.5, 12.0, 8.0], "contrasts": [0.8, 0.6, 0.5], "noise": 0.4},
    ]
    
    # Ensure we have enough configs for n_samples
    while len(test_configs) < n_samples:
        test_configs.append(test_configs[len(test_configs) % len(test_configs)])
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        config = test_configs[i]
        
        if config["type"] == "single":
            # Single trajectory
            simulation = simulate_single_particle(
                p=config["radius"],
                c=config["contrast"],
                n=config["noise"],
                x_step=0.5,
                t_step=1.0,
                n_t=length,
                n_x=width,
                peak_width=1.0,
            )
            noisy = simulation.kymograph_noisy
            gt = simulation.kymograph_gt
            title_info = f"r={config['radius']:.1f}nm, c={config['contrast']:.1f}, n={config['noise']:.1f}"
        else:
            # Multi-trajectory
            radii = config["radii"]
            contrasts = config["contrasts"]
            noise = config["noise"]
            diffusions = [get_diffusion_coefficient(r) for r in radii]
            noisy, gt, true_paths = generate_kymograph(
                length=length,
                width=width,
                diffusion=diffusions,
                contrast=contrasts,
                noise_level=noise,
                peak_width=1.0,
                dt=1.0,
                dx=0.5,
            )
            n_traj = len(radii)
            title_info = f"{n_traj} trajectories, n={noise:.1f}"
        
        # Denoise
        denoised = denoise_kymograph(model, noisy, device=device)
        
        # Calculate metrics
        mse_noisy = np.mean((noisy - gt) ** 2)
        mse_denoised = np.mean((denoised - gt) ** 2)
        improvement = ((mse_noisy - mse_denoised) / mse_noisy) * 100
        
        # Plot
        vmin, vmax = 0, 1
        
        axes[i, 0].imshow(noisy.T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="gray")
        axes[i, 0].set_title(f"Noisy Input\nMSE: {mse_noisy:.4f}\n{title_info}")
        axes[i, 0].set_xlabel("Time")
        axes[i, 0].set_ylabel("Position")
        
        axes[i, 1].imshow(denoised.T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="gray")
        axes[i, 1].set_title(f"Denoised Output\nMSE: {mse_denoised:.4f}\nImprovement: {improvement:.1f}%")
        axes[i, 1].set_xlabel("Time")
        axes[i, 1].set_ylabel("Position")
        
        axes[i, 2].imshow(gt.T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="gray")
        axes[i, 2].set_title("Ground Truth")
        axes[i, 2].set_xlabel("Time")
        axes[i, 2].set_ylabel("Position")
        
        # Overlay true paths on ground truth for multi-trajectory
        if config["type"] == "multi":
            for path in true_paths:
                axes[i, 2].plot(path, color='red', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved to: {save_path}")
    plt.close()
    
    print("\nDenoising visualization complete!")


if __name__ == "__main__":
    # Training dimensions: length=512 (time, chunkable), width=512 (position, matches validation)
    # Include multi-trajectory examples (30% probability, up to 3 trajectories)
    dataset = SyntheticKymographDataset(
        n_samples=1024,
        length=512,
        width=512,
        multi_trajectory_prob=0.3,  # 30% multi-trajectory examples
        max_trajectories=3,  # Up to 3 trajectories
    )
    # DDPM-style: Model always predicts noise to subtract
    # Increased epochs for multi-trajectory training (more complex patterns to learn)
    config = TrainingConfig(
        epochs=12,  # Increased from 5 to handle multi-trajectory complexity
        batch_size=8,
        loss="l2",  # L2 loss on predicted noise vs true noise
        use_gradient_clipping=True,
        max_grad_norm=1.0,
        use_residual_connection=True,  # Helps with gradient flow and training stability
        use_lr_scheduler=True,  # Enable LR scheduler for better convergence
    )
    model = train_denoiser(config, dataset)
    save_model(model, "tiny_unet_denoiser.pth")
    print(f"\nModel saved to: tiny_unet_denoiser.pth")
    
    # Visualize denoising results
    print("\n" + "=" * 60)
    print("Visualizing denoising results...")
    print("=" * 60)
    visualize_denoising_results(model, n_samples=3, device=config.device)
