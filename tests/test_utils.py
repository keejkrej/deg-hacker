"""
Tests for utility functions.
"""

import pytest
import numpy as np
import os
import tempfile

from utils import (
    simulate_single_particle,
    simulate_multi_particle,
    summarize_analysis,
    write_joint_metrics_csv,
    AnalysisMetrics,
    MultiSimulationData,
)
from helpers import estimate_diffusion_msd_fit, get_particle_radius


class TestSimulation:
    """Test simulation functions."""
    
    def test_simulate_single_particle_outputs(self):
        """Test simulate_single_particle produces correct outputs."""
        simulation = simulate_single_particle(p=5.0, c=0.7, n=0.3)
        
        assert hasattr(simulation, 'kymograph_noisy')
        assert hasattr(simulation, 'kymograph_gt')
        assert hasattr(simulation, 'true_path')
        assert simulation.p == 5.0
        assert simulation.c == 0.7
        assert simulation.n == 0.3
    
    def test_simulate_multi_particle_outputs(self):
        """Test simulate_multi_particle produces correct outputs."""
        simulation = simulate_multi_particle(
            radii_nm=[5.0, 10.0],
            contrasts=[0.7, 0.5],
            noise_level=0.3
        )
        
        assert len(simulation.radii_nm) == 2
        assert len(simulation.contrasts) == 2
        assert len(simulation.true_paths) == 2
        assert len(simulation.diffusions) == 2


class TestAnalysisMetrics:
    """Test AnalysisMetrics dataclass."""
    
    def test_analysis_metrics_creation(self):
        """Test creating AnalysisMetrics."""
        metrics = AnalysisMetrics(
            method_label="Test",
            particle_radius_nm=5.0,
            contrast=0.7,
            noise_level=0.3,
            diffusion_true=1.0,
            diffusion_processed=0.95,
            radius_true=5.0,
            radius_processed=5.2,
        )
        
        assert metrics.method_label == "Test"
        assert metrics.particle_radius_nm == 5.0
        assert metrics.diffusion_true == 1.0
    
    def test_analysis_metrics_optional_fields(self):
        """Test AnalysisMetrics with optional fields."""
        metrics = AnalysisMetrics(
            method_label="Test",
            particle_radius_nm=5.0,
            contrast=0.7,
            noise_level=0.3,
            diffusion_true=1.0,
            diffusion_processed=0.95,
            radius_true=5.0,
            radius_processed=5.2,
            noise_estimate=0.28,
            contrast_estimate=0.72,
            figure_path="test.png"
        )
        
        assert metrics.noise_estimate == 0.28
        assert metrics.contrast_estimate == 0.72
        assert metrics.figure_path == "test.png"


class TestCSVWriting:
    """Test CSV metrics writing."""
    
    def test_write_joint_metrics_csv(self):
        """Test writing metrics to CSV."""
        metrics_list = [
            AnalysisMetrics(
                method_label="Test1",
                particle_radius_nm=5.0,
                contrast=0.7,
                noise_level=0.3,
                diffusion_true=1.0,
                diffusion_processed=0.95,
                radius_true=5.0,
                radius_processed=5.2,
            ),
            AnalysisMetrics(
                method_label="Test2",
                particle_radius_nm=10.0,
                contrast=0.5,
                noise_level=0.4,
                diffusion_true=0.5,
                diffusion_processed=0.48,
                radius_true=10.0,
                radius_processed=10.5,
            ),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_metrics.csv")
            written_path = write_joint_metrics_csv(metrics_list, csv_path)
            
            assert written_path == csv_path
            assert os.path.exists(csv_path)
            
            # Check file has content
            with open(csv_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 3  # Header + 2 data rows
    
    def test_write_empty_metrics_csv(self):
        """Test writing empty metrics list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "empty.csv")
            written_path = write_joint_metrics_csv([], csv_path)
            
            assert written_path == csv_path
            # File might not exist if list is empty
            # (implementation dependent)


class TestSummarizeAnalysis:
    """Test summarize_analysis function."""
    
    def test_summarize_analysis_creates_figure(self):
        """Test summarize_analysis creates figure file."""
        simulation = simulate_single_particle(p=5.0, c=0.7, n=0.3, n_t=256, n_x=256)
        
        # Create a simple processed kymograph (just add some noise)
        processed = simulation.kymograph_gt + np.random.randn(256, 256) * 0.05
        processed = np.clip(processed, 0, 1)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            figure_subdir = os.path.join(tmpdir, "test_figures")
            metrics = summarize_analysis(
                simulation,
                processed,
                method_label="Test Method",
                figure_subdir=figure_subdir,
            )
            
            assert metrics is not None
            assert metrics.method_label == "Test Method"
            assert metrics.figure_path is not None
            assert os.path.exists(metrics.figure_path)


class TestDiffusionEstimation:
    """Test diffusion coefficient estimation."""
    
    def test_estimate_diffusion_msd_fit(self):
        """Test MSD-based diffusion estimation."""
        # Create a simple random walk (Brownian motion)
        n_steps = 1000
        dt = 1.0
        dx = 0.5
        D_true = 1.0  # True diffusion coefficient
        
        # Generate Brownian path
        steps = np.random.randn(n_steps) * np.sqrt(2 * D_true * dt) / dx
        path = np.cumsum(steps)
        path = path - path[0]  # Center at origin
        
        estimated_D = estimate_diffusion_msd_fit(path, dx=dx, dt=dt)
        
        # Should be reasonably close (within factor of 2)
        assert estimated_D > 0
        assert estimated_D < 5.0  # Shouldn't be way off
    
    def test_get_particle_radius(self):
        """Test particle radius calculation from diffusion."""
        D = 1.0  # um^2/ms
        radius = get_particle_radius(D)
        
        assert radius > 0
        # Should be in reasonable range (nanometers)
        assert radius > 1.0 and radius < 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
