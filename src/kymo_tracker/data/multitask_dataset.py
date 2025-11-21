"""Dataset for multi-task model training."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Union, Tuple, Optional

from kymo_tracker.utils.helpers import generate_kymograph, get_diffusion_coefficient


class MultiTaskDataset(Dataset):
    """Dataset for training multi-task denoising + heatmap locator model.
    
    Generates synthetic kymograph windows with multiple particles and returns:
    - noisy: noisy input kymograph (1, height, width)
    - true_noise: the noise that was added (1, height, width)
    - target_heatmap: target heatmap with Gaussians at peak locations (1, height, width)
    """

    def __init__(
        self,
        n_samples: int,
        length: int = 512,
        width: int = 512,
        window_length: int = 16,
        radii_nm: Union[float, Tuple[float, float]] = (3.0, 70.0),
        contrast: Union[float, Tuple[float, float]] = (0.5, 1.1),
        noise_level: Union[float, Tuple[float, float]] = (0.08, 0.8),
        multi_trajectory_prob: float = 1.0,
        max_trajectories: int = 3,
        mask_peak_width_samples: float = 10.0,
        particle_peak_width_samples: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the dataset.
        
        Args:
            n_samples: Number of samples to generate
            length: Unused (kept for backward compatibility)
            width: Spatial dimension (width) of kymograph in pixels
            window_length: Temporal window length (time frames) for each sample
            radii_nm: Particle radius range in nanometers (single value or tuple)
            contrast: Contrast range (single value or tuple)
            noise_level: Noise level range (single value or tuple)
            multi_trajectory_prob: Probability of generating multiple trajectories
            max_trajectories: Maximum number of trajectories per sample
            mask_peak_width_samples: Target width for locator training (in pixels)
            particle_peak_width_samples: Actual particle width for generation (in pixels).
                If None, uses a fixed value (2.0) to keep denoiser training consistent.
            seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.width = width
        self.window_length = window_length
        self.max_trajectories = max_trajectories
        self.mask_peak_width_samples = mask_peak_width_samples
        # Use fixed particle width for denoiser training, separate from locator target
        self.particle_peak_width_samples = particle_peak_width_samples if particle_peak_width_samples is not None else 2.0
        # Gaussian sigma for heatmap generation (in pixels)
        # Use mask_peak_width_samples as a proxy for Gaussian width
        self.heatmap_sigma = mask_peak_width_samples / 2.355  # Convert FWHM to sigma
        
        # Normalize ranges
        if isinstance(radii_nm, (int, float)):
            self.radii_range = (radii_nm, radii_nm)
        else:
            self.radii_range = radii_nm
            
        if isinstance(contrast, (int, float)):
            self.contrast_range = (contrast, contrast)
        else:
            self.contrast_range = contrast
            
        if isinstance(noise_level, (int, float)):
            self.noise_range = (noise_level, noise_level)
        else:
            self.noise_range = noise_level
            
        self.multi_trajectory_prob = multi_trajectory_prob
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def __len__(self) -> int:
        return self.n_samples

    def _create_heatmap(self, paths: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Create a heatmap with Gaussians centered on peak locations.
        
        Args:
            paths: (n_trajectories, window_length) array of peak positions in pixels
            shape: (window_length, width) shape of the output heatmap
            
        Returns:
            heatmap: (window_length, width) array with Gaussians at peak locations
        """
        window_length, width = shape
        heatmap = np.zeros((window_length, width), dtype=np.float32)
        
        # Create spatial coordinate array
        x = np.arange(width, dtype=np.float32)
        
        for t in range(window_length):
            for traj_idx in range(paths.shape[0]):
                center_px = paths[traj_idx, t]
                if np.isnan(center_px) or center_px < 0 or center_px >= width:
                    continue
                
                # Create Gaussian centered at center_px
                gaussian = np.exp(-0.5 * ((x - center_px) / self.heatmap_sigma) ** 2)
                # Add to heatmap (use max to handle overlapping peaks)
                heatmap[t, :] = np.maximum(heatmap[t, :], gaussian)
        
        return heatmap

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Generate a single training sample.
        
        Returns:
            Tuple of (noisy, true_noise, target_heatmap)
        """
        # Determine number of trajectories
        if np.random.rand() < self.multi_trajectory_prob:
            n_trajectories = np.random.randint(1, self.max_trajectories + 1)
        else:
            n_trajectories = 1
        
        # Sample parameters
        radii = [
            np.random.uniform(*self.radii_range)
            for _ in range(n_trajectories)
        ]
        contrasts = [
            np.random.uniform(*self.contrast_range)
            for _ in range(n_trajectories)
        ]
        noise_level = np.random.uniform(*self.noise_range)
        
        # Generate diffusion coefficients
        diffusions = [get_diffusion_coefficient(r) for r in radii]
        
        # Generate kymograph directly at window size (16 time frames, 512 spatial pixels)
        # Use fixed particle width for denoiser training (separate from locator target width)
        noisy_window, gt_window, paths_window = generate_kymograph(
            length=self.window_length,  # Generate 16 time frames directly
            width=self.width,  # 512 spatial pixels
            diffusion=diffusions if len(diffusions) > 1 else diffusions[0],
            contrast=contrasts if len(contrasts) > 1 else contrasts[0],
            noise_level=noise_level,
            peak_width=self.particle_peak_width_samples * 0.5,  # Convert samples to micrometers
            dt=1.0,
            dx=0.5,
            seed=None,  # Use random seed
        )
        
        # paths_window is already (n_trajectories, window_length)
        
        # Compute true noise
        true_noise_window = noisy_window - gt_window
        
        # Create heatmap target with Gaussians at peak locations
        target_heatmap = self._create_heatmap(paths_window, (self.window_length, self.width))
        
        # Convert to tensors and add channel dimension
        noisy_tensor = torch.from_numpy(noisy_window).float().unsqueeze(0)  # (1, window_length, width)
        true_noise_tensor = torch.from_numpy(true_noise_window).float().unsqueeze(0)  # (1, window_length, width)
        target_heatmap_tensor = torch.from_numpy(target_heatmap).float().unsqueeze(0)  # (1, window_length, width)
        
        return (
            noisy_tensor,
            true_noise_tensor,
            target_heatmap_tensor,
        )
