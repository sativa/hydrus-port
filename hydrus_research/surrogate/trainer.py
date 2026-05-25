"""Train surrogate models from M3 BatchResult artifacts."""
from __future__ import annotations
import numpy as np

from .gp_sklearn import SklearnGPSurrogate
from .pck import PCKSurrogate


def _filter_converged(br):
    keep = br.converged
    if not keep.any():
        raise ValueError("BatchResult has zero converged rows; cannot train")
    return br.thetas[keep], br.ys[keep]


def train_gp(batch_result, kernel="matern52"):
    thetas, ys = _filter_converged(batch_result)
    surr = SklearnGPSurrogate(kernel=kernel)
    surr.fit(thetas, ys)
    return surr


def train_pck(batch_result, pce_degree=3):
    thetas, ys = _filter_converged(batch_result)
    surr = PCKSurrogate(pce_degree=pce_degree)
    surr.fit(thetas, ys)
    return surr
