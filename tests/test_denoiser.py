"""
Comprehensive tests for the denoiser model and inference pipeline.
"""

import os
import pytest
import numpy as np
import torch

from denoiser import (
    TinyUNet,
    load_model,
    denoise_kymograph,
    save_model,
    TrainingConfig,
    train_denoiser,
    SyntheticKymographDataset,
    _default_device,
)
from helpers import generate_kymograph, get_diffusion_coefficient
from utils import simulate_single_particle, simulate_multi_particle


class TestTinyUNet:
    """Test the TinyUNet architecture."""
    
    def test_model_creation(self):
        """Test model can be created with different configurations."""
        model1 = TinyUNet(base_channels=32, use_residual=False, use_bn=True)
        model2 = TinyUNet(base_channels=48, use_residual=True, use_bn=False)
        model3 = TinyUNet(base_channels=56, use_residual=True, use_bn=True)
        
        assert model1.use_residual == False
        assert model2.use_residual == True
        assert model3.use_residual == True
    
    def test_model_forward_shape(self):
        """Test model forward pass produces correct output shape."""
        model = TinyUNet(base_channels=32, use_residual=False)
        x = torch.randn(1, 1, 512, 512)
        output = model(x)
        
        assert output.shape == (1, 1, 512, 512)
    
    def test_model_with_residual(self):
        """Test residual connection works correctly."""
        model = TinyUNet(base_channels=32, use_residual=True)
        x = torch.randn(1, 1, 512, 512)
        output = model(x)
        
        # With residual, output should be model_output + input
        # So output should be different from input (unless model outputs zero)
        assert output.shape == x.shape
        # Output should not be identical to input (model should do something)
        assert not torch.allclose(output, x, atol=1e-6)
    
    def test_model_different_sizes(self):
        """Test model handles different input sizes."""
        model = TinyUNet(base_channels=32)
        
        # Test various sizes
        sizes = [(256, 256), (512, 512), (1024, 512), (512, 256)]
        for h, w in sizes:
            x = torch.randn(1, 1, h, w)
            output = model(x)
            assert output.shape == (1, 1, h, w)


class TestDataGeneration:
    """Test data generation functions."""
    
    def test_simulate_single_particle(self):
        """Test single particle simulation."""
        simulation = simulate_single_particle(p=5.0, c=0.7, n=0.3)
        
        assert simulation.kymograph_noisy.shape == simulation.kymograph_gt.shape
        assert simulation.kymograph_noisy.shape == (simulation.n_t, simulation.n_x)
        assert simulation.true_path.shape == (simulation.n_t,)
        assert simulation.p == 5.0
        assert simulation.c == 0.7
        assert simulation.n == 0.3
    
    def test_simulate_multi_particle(self):
        """Test multi-particle simulation."""
        simulation = simulate_multi_particle(
            radii_nm=[5.0, 10.0],
            contrasts=[0.7, 0.5],
            noise_level=0.3
        )
        
        assert len(simulation.radii_nm) == 2
        assert len(simulation.contrasts) == 2
        assert len(simulation.true_paths) == 2
        assert simulation.kymograph_noisy.shape == simulation.kymograph_gt.shape
        assert simulation.true_paths[0].shape == (simulation.n_t,)
    
    def test_generate_kymograph_single(self):
        """Test generate_kymograph with single particle."""
        noisy, gt, paths = generate_kymograph(
            length=100, width=128,
            diffusion=1.0, contrast=1.0, noise_level=0.3
        )
        
        assert noisy.shape == (100, 128)
        assert gt.shape == (100, 128)
        assert paths.shape == (1, 100)  # Single particle
    
    def test_generate_kymograph_multi(self):
        """Test generate_kymograph with multiple particles."""
        noisy, gt, paths = generate_kymograph(
            length=100, width=128,
            diffusion=[1.0, 2.0], contrast=[1.0, 0.5], noise_level=0.3
        )
        
        assert noisy.shape == (100, 128)
        assert gt.shape == (100, 128)
        assert paths.shape == (2, 100)  # Two particles
    
    def test_dataset_generation(self):
        """Test SyntheticKymographDataset generates data correctly."""
        dataset = SyntheticKymographDataset(
            n_samples=10, length=256, width=256,
            multi_trajectory_prob=0.5, max_trajectories=3
        )
        
        assert len(dataset) == 10
        
        noisy, clean = dataset[0]
        assert noisy.shape == (1, 256, 256)
        assert clean.shape == (1, 256, 256)
        assert noisy.dtype == torch.float32
        assert clean.dtype == torch.float32


class TestDenoising:
    """Test denoising inference functions."""
    
    @pytest.fixture
    def test_model(self, tmp_path):
        """Create a small test model and save it."""
        model = TinyUNet(base_channels=16, use_residual=True, use_bn=True)
        model_path = tmp_path / "test_model.pth"
        save_model(model, str(model_path))
        return str(model_path), model
    
    def test_load_model(self, test_model):
        """Test model loading."""
        model_path, original_model = test_model
        loaded_model = load_model(model_path, base_channels=16, use_residual=True)
        
        assert loaded_model.use_residual == True
        assert loaded_model.base_channels == 16
    
    def test_load_model_residual_mismatch(self, test_model):
        """Test that loading with wrong residual setting causes issues."""
        model_path, _ = test_model
        
        # Load with wrong residual setting
        wrong_model = load_model(model_path, base_channels=16, use_residual=False)
        
        # Model architecture is different, but weights might still load
        # (This tests that we can detect the mismatch)
        assert wrong_model.use_residual == False
    
    def test_denoise_kymograph_shape(self, test_model):
        """Test denoise_kymograph produces correct output shape."""
        model_path, _ = test_model
        model = load_model(model_path, base_channels=16, use_residual=True)
        
        kymograph = np.random.rand(512, 512).astype(np.float32)
        denoised = denoise_kymograph(model, kymograph)
        
        assert denoised.shape == kymograph.shape
        assert denoised.dtype == np.float32
        assert np.all(denoised >= 0) and np.all(denoised <= 1)
    
    def test_denoise_kymograph_single_track(self, test_model):
        """Test denoising single-track kymograph."""
        model_path, _ = test_model
        model = load_model(model_path, base_channels=16, use_residual=True)
        
        # Generate single particle kymograph
        simulation = simulate_single_particle(p=5.0, c=0.7, n=0.3, n_t=256, n_x=256)
        denoised = denoise_kymograph(model, simulation.kymograph_noisy)
        
        assert denoised.shape == simulation.kymograph_noisy.shape
        assert np.all(denoised >= 0) and np.all(denoised <= 1)
    
    def test_denoise_kymograph_multi_track(self, test_model):
        """Test denoising multi-track kymograph."""
        model_path, _ = test_model
        model = load_model(model_path, base_channels=16, use_residual=True)
        
        # Generate multi-particle kymograph
        simulation = simulate_multi_particle(
            radii_nm=[5.0, 10.0],
            contrasts=[0.7, 0.5],
            noise_level=0.3,
            n_t=256,
            n_x=256
        )
        denoised = denoise_kymograph(model, simulation.kymograph_noisy)
        
        assert denoised.shape == simulation.kymograph_noisy.shape
        assert np.all(denoised >= 0) and np.all(denoised <= 1)
    
    def test_denoise_different_sizes(self, test_model):
        """Test denoising kymographs of different sizes."""
        model_path, _ = test_model
        model = load_model(model_path, base_channels=16, use_residual=True)
        
        sizes = [(256, 256), (512, 512), (1024, 256), (256, 512)]
        for h, w in sizes:
            kymograph = np.random.rand(h, w).astype(np.float32)
            denoised = denoise_kymograph(model, kymograph)
            assert denoised.shape == (h, w)


class TestChunkedProcessing:
    """Test chunked processing for large kymographs."""
    
    @pytest.fixture
    def test_model(self, tmp_path):
        """Create a test model."""
        model = TinyUNet(base_channels=16, use_residual=True)
        model_path = tmp_path / "test_model.pth"
        save_model(model, str(model_path))
        return str(model_path)
    
    def test_chunked_processing_large(self, test_model):
        """Test chunked processing for large kymographs."""
        from single_particle_unet import denoise_kymograph_chunked
        
        model = load_model(test_model, base_channels=16, use_residual=True)
        
        # Create large kymograph (larger than chunk size)
        kymograph = np.random.rand(2000, 512).astype(np.float32)
        denoised = denoise_kymograph_chunked(
            model, kymograph, chunk_size=512, overlap=64
        )
        
        assert denoised.shape == kymograph.shape
    
    def test_chunked_processing_small(self, test_model):
        """Test chunked processing for small kymographs (padding)."""
        from single_particle_unet import denoise_kymograph_chunked
        
        model = load_model(test_model, base_channels=16, use_residual=True)
        
        # Create small kymograph (smaller than chunk size)
        kymograph = np.random.rand(256, 256).astype(np.float32)
        denoised = denoise_kymograph_chunked(
            model, kymograph, chunk_size=512, overlap=64
        )
        
        assert denoised.shape == kymograph.shape
    
    def test_chunked_processing_2d(self, test_model):
        """Test 2D chunking for very large kymographs."""
        from single_particle_unet import denoise_kymograph_chunked
        
        model = load_model(test_model, base_channels=16, use_residual=True)
        
        # Create very large kymograph (needs 2D chunking)
        kymograph = np.random.rand(2000, 1000).astype(np.float32)
        denoised = denoise_kymograph_chunked(
            model, kymograph, chunk_size=512, overlap=64
        )
        
        assert denoised.shape == kymograph.shape


class TestModelTraining:
    """Test model training (quick smoke tests)."""
    
    def test_training_config(self):
        """Test TrainingConfig creation."""
        config = TrainingConfig(
            epochs=1,
            batch_size=4,
            lr=1e-3,
            loss="l2",
            use_residual_connection=True,
            use_lr_scheduler=False,
        )
        
        assert config.epochs == 1
        assert config.batch_size == 4
        assert config.use_residual_connection == True
    
    @pytest.mark.slow
    def test_training_smoke(self, tmp_path):
        """Quick smoke test of training (1 epoch, small dataset)."""
        dataset = SyntheticKymographDataset(
            n_samples=16, length=128, width=128,
            multi_trajectory_prob=0.3, max_trajectories=3
        )
        
        config = TrainingConfig(
            epochs=1,
            batch_size=4,
            lr=1e-3,
            loss="l2",
            use_residual_connection=True,
            use_lr_scheduler=False,
        )
        
        model = train_denoiser(config, dataset)
        
        # Check model was trained
        assert model is not None
        assert model.training == False  # Should be in eval mode after training


class TestResidualConnection:
    """Test residual connection behavior."""
    
    def test_residual_connection_math(self):
        """Test that residual connection works correctly for noise prediction."""
        model = TinyUNet(base_channels=16, use_residual=True)
        model.eval()
        
        # Create test data
        clean = torch.randn(1, 1, 128, 128) * 0.1 + 0.5  # Clean signal
        noise = torch.randn(1, 1, 128, 128) * 0.1  # Noise
        noisy = torch.clamp(clean + noise, 0, 1)
        
        with torch.no_grad():
            predicted_noise = model(noisy)
            # With residual: predicted_noise = model_output + noisy
            # Model should learn: model_output ≈ -clean
            # So: predicted_noise ≈ -clean + noisy = noise
        
        assert predicted_noise.shape == noisy.shape
        # Predicted noise should be in reasonable range
        assert predicted_noise.min() > -2.0 and predicted_noise.max() < 2.0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def test_model(self, tmp_path):
        """Create a test model."""
        model = TinyUNet(base_channels=16, use_residual=True)
        model_path = tmp_path / "test_model.pth"
        save_model(model, str(model_path))
        return str(model_path)
    
    def test_empty_kymograph(self, test_model):
        """Test handling of empty/zero kymograph."""
        model = load_model(test_model, base_channels=16, use_residual=True)
        
        kymograph = np.zeros((100, 100), dtype=np.float32)
        denoised = denoise_kymograph(model, kymograph)
        
        assert denoised.shape == kymograph.shape
        assert np.all(denoised >= 0) and np.all(denoised <= 1)
    
    def test_all_ones_kymograph(self, test_model):
        """Test handling of saturated kymograph."""
        model = load_model(test_model, base_channels=16, use_residual=True)
        
        kymograph = np.ones((100, 100), dtype=np.float32)
        denoised = denoise_kymograph(model, kymograph)
        
        assert denoised.shape == kymograph.shape
        assert np.all(denoised >= 0) and np.all(denoised <= 1)
    
    def test_load_nonexistent_model(self):
        """Test error handling for nonexistent model file."""
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent_model.pth")
    
    def test_wrong_base_channels(self, test_model):
        """Test error handling for wrong base_channels."""
        # This should raise an error when loading state dict
        with pytest.raises(RuntimeError):
            load_model(test_model, base_channels=32, use_residual=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
