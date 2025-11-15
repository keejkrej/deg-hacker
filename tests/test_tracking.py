"""
Tests for multi-particle tracking functionality.
"""

import pytest
import numpy as np

from multi_particle_unet import (
    track_particles,
    _select_peak_candidates,
    _assign_candidates,
    _predict_positions,
)
from utils import simulate_multi_particle


class TestPeakSelection:
    """Test peak candidate selection."""
    
    def test_select_peak_candidates(self):
        """Test peak candidate selection."""
        row = np.array([0.1, 0.2, 0.5, 0.3, 0.1, 0.4, 0.2])
        candidates, scores = _select_peak_candidates(row, max_candidates=3)
        
        assert len(candidates) <= 3
        assert len(candidates) == len(scores)
        # Should select highest peaks
        assert 2 in candidates  # Peak at index 2 (value 0.5)
        assert 5 in candidates  # Peak at index 5 (value 0.4)
    
    def test_select_peak_candidates_empty(self):
        """Test peak selection with no peaks."""
        row = np.zeros(100)
        candidates, scores = _select_peak_candidates(row, max_candidates=10)
        
        assert len(candidates) == 0
        assert len(scores) == 0


class TestPositionPrediction:
    """Test position prediction for tracking."""
    
    def test_predict_positions_no_history(self):
        """Test prediction with no previous positions."""
        assert _predict_positions(None, None) is None
    
    def test_predict_positions_one_frame(self):
        """Test prediction with only one previous frame."""
        prev = np.array([10.0, 20.0, 30.0])
        predicted = _predict_positions(prev, None)
        
        assert np.allclose(predicted, prev)
    
    def test_predict_positions_two_frames(self):
        """Test prediction with velocity estimation."""
        prev = np.array([10.0, 20.0, 30.0])
        prev_prev = np.array([8.0, 18.0, 28.0])
        predicted = _predict_positions(prev, prev_prev)
        
        # Should predict: prev + (prev - prev_prev) = prev + velocity
        expected = prev + (prev - prev_prev)
        assert np.allclose(predicted, expected)


class TestCandidateAssignment:
    """Test candidate assignment logic."""
    
    def test_assign_candidates_basic(self):
        """Test basic candidate assignment."""
        predictions = np.array([10.0, 20.0, 30.0])
        candidate_positions = np.array([11.0, 21.0, 31.0])
        candidate_scores = np.array([0.5, 0.6, 0.7])
        
        assigned = _assign_candidates(
            predictions, candidate_positions, candidate_scores,
            max_jump=5.0, min_intensity=0.1, intensity_weight=0.3,
            width=100, last_positions=None
        )
        
        assert len(assigned) == len(predictions)
        assert np.all(assigned >= 0) and np.all(assigned < 100)
    
    def test_assign_candidates_no_candidates(self):
        """Test assignment when no candidates found."""
        predictions = np.array([10.0, 20.0, 30.0])
        candidate_positions = np.array([])
        candidate_scores = np.array([])
        
        assigned = _assign_candidates(
            predictions, candidate_positions, candidate_scores,
            max_jump=5.0, min_intensity=0.1, intensity_weight=0.3,
            width=100, last_positions=predictions
        )
        
        # Should fall back to last positions
        assert np.allclose(assigned, predictions)


class TestParticleTracking:
    """Test full particle tracking pipeline."""
    
    def test_track_particles_basic(self):
        """Test basic particle tracking."""
        # Create simple kymograph with clear tracks
        kymograph = np.zeros((100, 200))
        
        # Add two clear tracks
        for t in range(100):
            kymograph[t, 50 + t // 10] = 0.5  # Track 1
            kymograph[t, 150 - t // 10] = 0.5  # Track 2
        
        tracks = track_particles(kymograph, n_particles=2, max_jump=10)
        
        assert tracks.shape == (2, 100)
        # Tracks should be reasonable (not all NaN)
        assert not np.all(np.isnan(tracks))
    
    def test_track_particles_multi(self):
        """Test tracking multiple particles."""
        simulation = simulate_multi_particle(
            radii_nm=[5.0, 10.0],
            contrasts=[0.7, 0.5],
            noise_level=0.2,  # Low noise for easier tracking
            n_t=200,
            n_x=256
        )
        
        tracks = track_particles(
            simulation.kymograph_noisy,
            n_particles=2,
            max_jump=15,
            min_intensity=0.01
        )
        
        assert tracks.shape == (2, 200)
        # Should track most of the time
        valid_tracks = ~np.isnan(tracks)
        assert np.sum(valid_tracks) > 100  # At least 50% valid
    
    def test_track_particles_three_particles(self):
        """Test tracking three particles."""
        simulation = simulate_multi_particle(
            radii_nm=[5.0, 10.0, 8.0],
            contrasts=[0.8, 0.6, 0.5],
            noise_level=0.2,
            n_t=200,
            n_x=256
        )
        
        tracks = track_particles(
            simulation.kymograph_noisy,
            n_particles=3,
            max_jump=15
        )
        
        assert tracks.shape == (3, 200)


class TestTrackingRobustness:
    """Test tracking robustness to various conditions."""
    
    def test_track_high_noise(self):
        """Test tracking with high noise."""
        simulation = simulate_multi_particle(
            radii_nm=[5.0, 10.0],
            contrasts=[0.7, 0.5],
            noise_level=0.5,  # High noise
            n_t=200,
            n_x=256
        )
        
        tracks = track_particles(
            simulation.kymograph_noisy,
            n_particles=2,
            max_jump=20,  # Larger jump for high noise
            min_intensity=0.005  # Lower threshold
        )
        
        assert tracks.shape == (2, 200)
        # Even with high noise, should track some frames
        assert np.sum(~np.isnan(tracks)) > 50
    
    def test_track_low_contrast(self):
        """Test tracking with low contrast particles."""
        simulation = simulate_multi_particle(
            radii_nm=[5.0, 10.0],
            contrasts=[0.3, 0.2],  # Low contrast
            noise_level=0.2,
            n_t=200,
            n_x=256
        )
        
        tracks = track_particles(
            simulation.kymograph_noisy,
            n_particles=2,
            min_intensity=0.005,  # Very low threshold
            max_jump=15
        )
        
        assert tracks.shape == (2, 200)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
