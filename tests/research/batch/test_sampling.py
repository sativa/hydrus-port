import numpy as np
import pytest

from hydrus_research.batch.sampling import (
    lhs, grid, uniform_random,
)


def test_lhs_returns_right_shape():
    bounds = np.array([[0.001, 1.0], [1.05, 5.0], [0.01, 100.0]])
    samples = lhs(bounds, n=32, seed=42)
    assert samples.shape == (32, 3)
    # Every column should span its bounds approximately
    for j in range(3):
        assert samples[:, j].min() >= bounds[j, 0]
        assert samples[:, j].max() <= bounds[j, 1]


def test_lhs_reproducible_with_seed():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    a = lhs(bounds, n=8, seed=42)
    b = lhs(bounds, n=8, seed=42)
    np.testing.assert_array_equal(a, b)


def test_grid_full_factorial():
    bounds = np.array([[0.0, 1.0], [10.0, 20.0]])
    samples = grid(bounds, points_per_axis=[3, 4])
    assert samples.shape == (3 * 4, 2)
    # First column should have 3 unique values, second should have 4
    assert len(np.unique(samples[:, 0])) == 3
    assert len(np.unique(samples[:, 1])) == 4


def test_uniform_random():
    bounds = np.array([[0.0, 1.0], [-5.0, 5.0]])
    samples = uniform_random(bounds, n=100, seed=7)
    assert samples.shape == (100, 2)
    assert (samples[:, 0] >= 0).all() and (samples[:, 0] <= 1).all()
    assert (samples[:, 1] >= -5).all() and (samples[:, 1] <= 5).all()
