"""Tests for barycentric interpolation helpers."""
import numpy as np
import pytest
from hydrus_research.simulator._interp import barycentric_2d, barycentric_3d


def test_barycentric_2d_unit_triangle():
    """Triangle (0,0), (1,0), (0,1); point (0.25, 0.25) — interpolate
    f(v) = [1, 2, 3] linearly."""
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    fvals = np.array([1.0, 2.0, 3.0])
    pt = np.array([0.25, 0.25])
    val = barycentric_2d(verts, fvals, pt)
    # Bary coords: (0.5, 0.25, 0.25); val = 0.5*1 + 0.25*2 + 0.25*3 = 1.75
    assert val == pytest.approx(1.75, abs=1e-6)


def test_barycentric_3d_unit_tet():
    verts = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=float)
    fvals = np.array([1.0, 2.0, 3.0, 4.0])
    pt = np.array([0.25, 0.25, 0.25])
    val = barycentric_3d(verts, fvals, pt)
    # Bary coords: (0.25, 0.25, 0.25, 0.25); val = mean = 2.5
    assert val == pytest.approx(2.5, abs=1e-6)
