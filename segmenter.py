"""Train a segmentation model to identify particle tracks in denoised kymographs.

The model takes a denoised kymograph as input and outputs a binary mask or probability
map indicating where particles are located. This can improve tracking accuracy by
providing a cleaner signal for tracking algorithms.

Architecture: Tiny U-Net (same as denoiser) but outputs sigmoid-activated probability map.
Loss: Binary Cross-Entropy (BCE) or Dice loss for segmentation.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import time
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from utils import simulate_single_particle, simulate_multi_particle
from helpers import generate_kymograph, get_diffusion_coefficient
from denoiser import TinyUNet, ConvBlock, _default_device, load_model


class SegmentationUNet(nn.Module):
    """U-Net for segmentation: outputs probability map of particle locations.
    
    Same architecture as TinyUNet but with sigmoid activation for binary segmentation.
    """
    
    def __init__(self, base_channels: int = 48, use_bn: bool = True) -> None:
        super().__init__()
        
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
        
        # Initialize output layer
        nn.init.xavier_uniform_(self.out_conv.weight, gain=1.0)
        nn.init.constant_(self.out_conv.bias, 0.0)
    
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
        
        output = self.out_conv(d1)
        
        # Sigmoid activation for probability map [0, 1]
        output = torch.sigmoid(output)
        
        return output


def create_segmentation_mask(paths: np.ndarray, shape: Tuple[int, int], 
                             peak_width_samples: float = 2.0) -> np.ndarray:
    """Create a binary/soft segmentation mask from particle paths.
    
    Parameters:
    -----------
    paths : np.ndarray
        Particle paths, shape (n_particles, length) or (length,) for single particle
    shape : Tuple[int, int]
        Output shape (length, width)
    peak_width_samples : float
        Width of Gaussian around each particle position (in pixels)
    
    Returns:
    --------
    mask : np.ndarray
        Segmentation mask, shape (length, width), values in [0, 1]
    """
    length, width = shape
    mask = np.zeros((length, width), dtype=np.float32)
    
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
                mask[t] = np.maximum(mask[t], gaussian)
    
    # Normalize to [0, 1] if needed (already should be, but clip to be safe)
    mask = np.clip(mask, 0.0, 1.0)
    
    return mask


class SegmentationDataset(Dataset):
    """Dataset for segmentation: denoised kymograph -> segmentation mask."""
    
    def __init__(
        self,
        denoiser_model: nn.Module,
        denoiser_device: str,
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
        self.denoiser_model = denoiser_model
        self.denoiser_device = denoiser_device
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
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
            paths = simulation.true_path
            # Convert single path to 2D array
            if paths.ndim == 1:
                paths = paths.reshape(1, -1)
        else:
            diffusions = [get_diffusion_coefficient(r) for r in radii]
            noisy, _, paths = generate_kymograph(
                length=self.length,
                width=self.width,
                diffusion=diffusions,
                contrast=contrasts,
                noise_level=noise,
                peak_width=self.peak_width,
                dt=self.dt,
                dx=self.dx,
            )
        
        # Denoise using the denoiser model
        noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).float().to(self.denoiser_device)
        with torch.no_grad():
            predicted_noise = self.denoiser_model(noisy_tensor)
            denoised_tensor = torch.clamp(noisy_tensor - predicted_noise, 0.0, 1.0)
        denoised = denoised_tensor.squeeze().cpu().numpy()
        
        # Create segmentation mask from paths
        peak_width_samples = self.peak_width / self.dx
        mask = create_segmentation_mask(
            paths, 
            shape=(self.length, self.width),
            peak_width_samples=max(peak_width_samples, self.mask_peak_width_samples)
        )
        
        # Convert to tensors
        denoised_tensor = torch.from_numpy(denoised).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        return denoised_tensor, mask_tensor


@dataclass
class SegmentationConfig:
    """Configuration for segmentation model training."""
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    loss: str = "bce"  # "bce" or "dice"
    use_gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    use_lr_scheduler: bool = True
    use_transfer_learning: bool = True  # Initialize with denoiser weights
    device: str = _default_device()
    denoiser_model_path: str = "models/tiny_unet_denoiser.pth"


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Dice loss for segmentation."""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return 1.0 - dice


def train_segmenter(
    config: SegmentationConfig,
    dataset: SegmentationDataset,
    use_transfer_learning: bool = True,
) -> SegmentationUNet:
    """Train the segmentation model.
    
    Parameters:
    -----------
    use_transfer_learning : bool
        If True, initialize segmentation model with denoiser weights (except output layer).
        This leverages learned features from denoising task.
    """
    
    # Load denoiser model
    print(f"Loading denoiser model from {config.denoiser_model_path}...")
    denoiser_model = load_model(config.denoiser_model_path, device=config.device, base_channels=56, use_residual=True)
    denoiser_model.eval()
    
    # Update dataset with denoiser model
    dataset.denoiser_model = denoiser_model
    dataset.denoiser_device = config.device
    
    # Create segmentation model
    # Use same base_channels as denoiser for transfer learning compatibility
    # Note: denoiser uses base_channels=56, but we'll use 48 for segmentation (can still transfer)
    model = SegmentationUNet(base_channels=48, use_bn=True).to(config.device)
    
    # Transfer learning: copy weights from denoiser (except output layer)
    if use_transfer_learning:
        print("Applying transfer learning: initializing with denoiser weights...")
        denoiser_state = denoiser_model.state_dict()
        segmenter_state = model.state_dict()
        
        # Copy matching layers (encoder, decoder, bottleneck)
        # Note: base_channels differ (56 vs 48), so we can only copy layers with matching shapes
        copied_layers = 0
        skipped_layers = []
        for name, param in denoiser_state.items():
            # Skip output layer (out_conv) - it's different (noise prediction vs probability)
            if name == "out_conv.weight" or name == "out_conv.bias":
                continue
            
            # Copy if layer exists in segmentation model and shapes match
            if name in segmenter_state:
                if segmenter_state[name].shape == param.shape:
                    segmenter_state[name] = param.clone()
                    copied_layers += 1
                else:
                    skipped_layers.append(f"{name}: {segmenter_state[name].shape} vs {param.shape}")
        
        model.load_state_dict(segmenter_state)
        print(f"  Copied {copied_layers} layers from denoiser")
        if skipped_layers:
            print(f"  Skipped {len(skipped_layers)} layers due to shape mismatch (base_channels differ)")
        print("  Output layer (out_conv) randomly initialized for segmentation task")
    
    # Loss function
    if config.loss == "bce":
        criterion = nn.BCELoss()
    elif config.loss == "dice":
        criterion = dice_loss
    else:
        raise ValueError(f"Unknown loss: {config.loss}")
    
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
    
    print(f"\nTraining segmentation model on {config.device}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Loss: {config.loss}")
    print(f"  Transfer learning: {use_transfer_learning}")
    print(f"  Dataset size: {len(dataset)}")
    
    model.train()
    
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, (denoised, mask) in enumerate(dataloader):
            denoised = denoised.to(config.device)
            mask = mask.to(config.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_mask = model(denoised)
            
            # Compute loss
            if config.loss == "bce":
                loss = criterion(pred_mask, mask)
            else:  # dice
                loss = criterion(pred_mask, mask)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if config.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch + 1}/{config.epochs}: Loss={avg_loss:.6f}, Time={epoch_time:.2f}s")
        
        if scheduler is not None:
            scheduler.step(avg_loss)
    
    model.eval()
    return model


def save_segmentation_model(model: SegmentationUNet, path: str) -> None:
    """Save segmentation model."""
    torch.save(model.state_dict(), path)
    print(f"Segmentation model saved to {path}")


def load_segmentation_model(path: str, device: str = None) -> SegmentationUNet:
    """Load segmentation model."""
    if device is None:
        device = _default_device()
    
    model = SegmentationUNet(base_channels=48, use_bn=True).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


if __name__ == "__main__":
    import os
    
    # Check if denoiser model exists
    denoiser_path = "models/tiny_unet_denoiser.pth"
    if not os.path.exists(denoiser_path):
        print(f"Error: Denoiser model not found at {denoiser_path}")
        print("Please train the denoiser first using denoiser.py")
        exit(1)
    
    # Create dataset
    print("Creating segmentation dataset...")
    # Load denoiser temporarily to pass to dataset
    denoiser_model = load_model(denoiser_path, device=_default_device(), base_channels=56, use_residual=True)
    denoiser_model.eval()
    
    dataset = SegmentationDataset(
        denoiser_model=denoiser_model,
        denoiser_device=_default_device(),
        n_samples=1024,
        length=512,
        width=512,
        multi_trajectory_prob=0.3,
        max_trajectories=3,
        mask_peak_width_samples=2.0,  # Width of segmentation mask around tracks
    )
    
    # Training config
    config = SegmentationConfig(
        epochs=10,
        batch_size=8,
        learning_rate=1e-3,
        loss="bce",  # Binary cross-entropy
        use_gradient_clipping=True,
        max_grad_norm=1.0,
        use_lr_scheduler=True,
        denoiser_model_path=denoiser_path,
    )
    
    # Train
    model = train_segmenter(config, dataset, use_transfer_learning=config.use_transfer_learning)
    
    # Save
    os.makedirs("models", exist_ok=True)
    model_path = "models/tiny_unet_segmenter.pth"
    save_segmentation_model(model, model_path)
    print(f"\nSegmentation model saved to: {model_path}")
