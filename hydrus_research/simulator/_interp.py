"""Barycentric interpolation helpers for 2D triangles + 3D tetrahedra."""
from __future__ import annotations
import numpy as np


def barycentric_2d(triangle: np.ndarray, fvals: np.ndarray,
                   point: np.ndarray) -> float:
    """triangle: (3, 2) vertex coords; fvals: (3,) values at vertices;
    point: (2,) interpolation location. Returns f(point) via bary coords.

    No bounds check: caller is responsible for ensuring point is inside
    the triangle (or accepting linear extrapolation)."""
    A = np.empty((3, 3), dtype=float)
    A[:, :2] = triangle
    A[:, 2] = 1.0
    rhs = np.array([point[0], point[1], 1.0])
    try:
        weights = np.linalg.solve(A.T, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(A.T, rhs, rcond=None)[0]
    return float(weights @ fvals)


def barycentric_3d(tetra: np.ndarray, fvals: np.ndarray,
                   point: np.ndarray) -> float:
    """tetra: (4, 3) vertex coords; fvals: (4,) values; point: (3,)."""
    A = np.empty((4, 4), dtype=float)
    A[:, :3] = tetra
    A[:, 3] = 1.0
    rhs = np.array([point[0], point[1], point[2], 1.0])
    try:
        weights = np.linalg.solve(A.T, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(A.T, rhs, rcond=None)[0]
    return float(weights @ fvals)
