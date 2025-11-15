# Test Suite

Comprehensive tests for the deg-hacker denoising pipeline.

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_denoiser.py

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run only fast tests (skip slow training tests)
pytest tests/ -m "not slow"
```

## Test Structure

- **test_denoiser.py**: Tests for model architecture, training, and inference
- **test_tracking.py**: Tests for multi-particle tracking algorithms
- **test_utils.py**: Tests for utility functions and data structures
- **test_integration.py**: End-to-end pipeline tests

## Test Categories

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test full pipeline workflows
- **Edge case tests**: Test error handling and boundary conditions

## Fixtures

- `test_model`: Creates a small trained model for testing
- `device`: Returns available device (CUDA or CPU)
- `small_kymograph`: Creates 128x128 test kymograph
- `medium_kymograph`: Creates 512x512 test kymograph
