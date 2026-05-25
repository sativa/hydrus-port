"""Sampling helpers for the BatchRunner CLI.

These produce a `thetas: (N, D)` array given a bounds array (D, 2). The
runner itself takes thetas directly — these helpers exist so the CLI can
spell out a sweep as `--n 32 --sampler lhs`."""
from __future__ import annotations
import numpy as np


def lhs(bounds: np.ndarray, n: int, seed: int | None = None) -> np.ndarray:
    """Latin Hypercube sampling via scipy.stats.qmc.

    bounds: shape (D, 2) — (lo, hi) per parameter (in user coords).
    Returns: shape (n, D) samples uniformly distributed within each [lo, hi]."""
    from scipy.stats import qmc
    bounds = np.asarray(bounds, dtype=float)
    D = bounds.shape[0]
    sampler = qmc.LatinHypercube(d=D, seed=seed)
    u = sampler.random(n)                                # (n, D) in [0, 1)
    return qmc.scale(u, bounds[:, 0], bounds[:, 1])


def grid(bounds: np.ndarray, points_per_axis: list[int]) -> np.ndarray:
    """Full-factorial grid. Returns (prod(points), D) samples."""
    bounds = np.asarray(bounds, dtype=float)
    D = bounds.shape[0]
    if len(points_per_axis) != D:
        raise ValueError(f"points_per_axis length {len(points_per_axis)} != D {D}")
    axes = [np.linspace(bounds[j, 0], bounds[j, 1], points_per_axis[j]) for j in range(D)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([m.ravel() for m in mesh])


def uniform_random(bounds: np.ndarray, n: int, seed: int | None = None) -> np.ndarray:
    """Plain uniform random sampling."""
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    return rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, bounds.shape[0]))
