"""Surrogate evaluation metrics."""
from __future__ import annotations
import numpy as np


def nse(sim, obs):
    sim = np.asarray(sim, dtype=float).ravel()
    obs = np.asarray(obs, dtype=float).ravel()
    num = np.sum((sim - obs) ** 2)
    den = np.sum((obs - obs.mean()) ** 2)
    return float("nan") if den == 0 else float(1.0 - num / den)


def rmse(sim, obs):
    sim = np.asarray(sim, dtype=float).ravel()
    obs = np.asarray(obs, dtype=float).ravel()
    return float(np.sqrt(np.mean((sim - obs) ** 2)))


def coverage(means, stds, obs, z=1.96):
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    obs = np.asarray(obs, dtype=float)
    return float(np.mean((obs >= means - z*stds) & (obs <= means + z*stds)))
