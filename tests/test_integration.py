"""
Integration tests for the full pipeline.
"""

import pytest
import numpy as np
import os
import tempfile

from denoiser import load_model, denoise_kymograph, save_model, TinyUNet
from single_particle_unet import denoise_kymograph_chunked, analyze_particle
from utils import (
    analyze_multi_particle,
    track_particles,
    simulate_single_particle,
    simulate_multi_particle,
)


class TestSingleParticlePipeline:
    """Test full single-particle analysis pipeline."""
    
    @pytest.fixture
    def test_model(self, tmp_path):
        """Create a small trained model."""
        from denoiser import train_denoiser, TrainingConfig, SyntheticKymographDataset
        
        dataset = SyntheticKymographDataset(
            n_samples=8, length=128, width=128,
            multi_trajectory_prob=0.0  # Single particle only
        )
        config = TrainingConfig(
            epochs=1, batch_size=2, lr=1e-3,
            use_residual_connection=True, use_lr_scheduler=False
        )
        model = train_denoiser(config, dataset)
        
        model_path = tmp_path / "test_model.pth"
        save_model(model, str(model_path))
        return str(model_path)
    
    def test_analyze_particle_pipeline(self, test_model):
        """Test full single-particle analysis."""
        # Temporarily override model path
        import single_particle_unet
        original_path = getattr(single_particle_unet, 'MODEL_PATH', None)
        
        try:
            # This would normally load from default path, but we'll test the function directly
            simulation = simulate_single_particle(p=5.0, c=0.7, n=0.3, n_t=256, n_x=256)
            model = load_model(test_model, base_channels=56, use_residual=True)
            
            from single_particle_unet import denoise_kymograph_chunked
            denoised = denoise_kymograph_chunked(
                model, simulation.kymograph_noisy, chunk_size=128, overlap=32
            )
            
            assert denoised.shape == simulation.kymograph_noisy.shape
            assert np.all(denoised >= 0) and np.all(denoised <= 1)
        finally:
            if original_path:
                setattr(single_particle_unet, 'MODEL_PATH', original_path)


class TestMultiParticlePipeline:
    """Test full multi-particle analysis pipeline."""
    
    @pytest.fixture
    def test_model(self, tmp_path):
        """Create a small trained model."""
        from denoiser import train_denoiser, TrainingConfig, SyntheticKymographDataset
        
        dataset = SyntheticKymographDataset(
            n_samples=8, length=128, width=128,
            multi_trajectory_prob=0.5, max_trajectories=3
        )
        config = TrainingConfig(
            epochs=1, batch_size=2, lr=1e-3,
            use_residual_connection=True, use_lr_scheduler=False
        )
        model = train_denoiser(config, dataset)
        
        model_path = tmp_path / "test_model.pth"
        save_model(model, str(model_path))
        return str(model_path)
    
    def test_analyze_multi_particle_pipeline(self, test_model):
        """Test full multi-particle analysis."""
        simulation = simulate_multi_particle(
            radii_nm=[5.0, 10.0],
            contrasts=[0.7, 0.5],
            noise_level=0.3,
            n_t=256,
            n_x=256
        )
        
        model = load_model(test_model, base_channels=56, use_residual=True)
        
        # Denoise
        from single_particle_unet import denoise_kymograph_chunked
        denoised = denoise_kymograph_chunked(
            model, simulation.kymograph_noisy, chunk_size=128, overlap=32
        )
        
        # Track particles
        tracks = track_particles(denoised, n_particles=2, max_jump=15)
        
        assert denoised.shape == simulation.kymograph_noisy.shape
        assert tracks.shape == (2, 256)
        assert np.sum(~np.isnan(tracks)) > 100  # Most tracks should be valid


class TestModelConsistency:
    """Test model consistency across different use cases."""
    
    @pytest.fixture
    def test_model(self, tmp_path):
        """Create a test model."""
        model = TinyUNet(base_channels=16, use_residual=True)
        model_path = tmp_path / "test_model.pth"
        save_model(model, str(model_path))
        return str(model_path)
    
    def test_model_consistency_single_vs_multi(self, test_model):
        """Test model works consistently for single and multi-particle."""
        model = load_model(test_model, base_channels=16, use_residual=True)
        
        # Single particle
        sim_single = simulate_single_particle(p=5.0, c=0.7, n=0.3, n_t=128, n_x=128)
        denoised_single = denoise_kymograph(model, sim_single.kymograph_noisy)
        
        # Multi particle
        sim_multi = simulate_multi_particle(
            radii_nm=[5.0], contrasts=[0.7], noise_level=0.3, n_t=128, n_x=128
        )
        denoised_multi = denoise_kymograph(model, sim_multi.kymograph_noisy)
        
        # Both should work
        assert denoised_single.shape == sim_single.kymograph_noisy.shape
        assert denoised_multi.shape == sim_multi.kymograph_noisy.shape
    
    def test_model_deterministic(self, test_model):
        """Test model produces consistent results (deterministic)."""
        model = load_model(test_model, base_channels=16, use_residual=True)
        model.eval()
        
        kymograph = np.random.rand(128, 128).astype(np.float32)
        
        # Run twice
        denoised1 = denoise_kymograph(model, kymograph)
        denoised2 = denoise_kymograph(model, kymograph)
        
        # Should be identical (deterministic)
        assert np.allclose(denoised1, denoised2, atol=1e-6)


class TestEdgeCasesIntegration:
    """Test edge cases in full pipeline."""
    
    @pytest.fixture
    def test_model(self, tmp_path):
        """Create a test model."""
        model = TinyUNet(base_channels=16, use_residual=True)
        model_path = tmp_path / "test_model.pth"
        save_model(model, str(model_path))
        return str(model_path)
    
    def test_very_noisy_input(self, test_model):
        """Test handling of very noisy input."""
        model = load_model(test_model, base_channels=16, use_residual=True)
        
        # Create very noisy kymograph
        clean = np.random.rand(128, 128) * 0.1
        noise = np.random.randn(128, 128) * 0.5  # Very high noise
        noisy = np.clip(clean + noise, 0, 1)
        
        denoised = denoise_kymograph(model, noisy)
        
        assert denoised.shape == noisy.shape
        assert np.all(denoised >= 0) and np.all(denoised <= 1)
    
    def test_sparse_signal(self, test_model):
        """Test handling of very sparse signal."""
        model = load_model(test_model, base_channels=16, use_residual=True)
        
        # Create sparse kymograph (mostly zeros)
        kymograph = np.zeros((128, 128), dtype=np.float32)
        kymograph[50:60, 50:60] = 0.5  # Small bright patch
        
        denoised = denoise_kymograph(model, kymograph)
        
        assert denoised.shape == kymograph.shape
        assert np.all(denoised >= 0) and np.all(denoised <= 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
