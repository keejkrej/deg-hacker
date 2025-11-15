"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
import torch


@pytest.fixture(scope="session")
def device():
    """Get default device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    yield
    # Reset seed after test


@pytest.fixture
def small_kymograph():
    """Create a small test kymograph."""
    return np.random.rand(128, 128).astype(np.float32)


@pytest.fixture
def medium_kymograph():
    """Create a medium-sized test kymograph."""
    return np.random.rand(512, 512).astype(np.float32)
