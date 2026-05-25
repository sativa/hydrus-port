"""scikit-learn Gaussian Process surrogate (Matérn 5/2 kernel default).

Persistence uses joblib — the canonical sklearn idiom. The serialized
artefact is our own file written immediately before loading; never load
joblib files from untrusted sources."""
from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

from .base import SurrogateModel


class SklearnGPSurrogate(SurrogateModel):
    def __init__(self, kernel: str = "matern52", alpha: float = 1e-6):
        self.kernel_name = kernel
        self.alpha = alpha
        self._gprs: list[GaussianProcessRegressor] | None = None
        self._D: int | None = None
        self._M: int | None = None

    def _make_kernel(self):
        if self.kernel_name == "matern52":
            return ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) \
                   + WhiteKernel(noise_level=1e-5)
        if self.kernel_name == "matern32":
            return ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) \
                   + WhiteKernel(noise_level=1e-5)
        raise ValueError(f"unknown kernel {self.kernel_name!r}")

    def fit(self, thetas, ys):
        thetas = np.asarray(thetas, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if ys.ndim == 1:
            ys = ys.reshape(-1, 1)
        self._D = thetas.shape[1]
        self._M = ys.shape[1]
        self._gprs = []
        for j in range(self._M):
            gpr = GaussianProcessRegressor(
                kernel=self._make_kernel(), alpha=self.alpha,
                normalize_y=True, n_restarts_optimizer=2, random_state=42,
            )
            gpr.fit(thetas, ys[:, j])
            self._gprs.append(gpr)

    def predict(self, theta):
        if self._gprs is None:
            raise RuntimeError("must call fit() first")
        theta = np.asarray(theta, dtype=float).reshape(1, -1)
        means = np.empty(self._M, dtype=float)
        stds = np.empty(self._M, dtype=float)
        for j, gpr in enumerate(self._gprs):
            m, s = gpr.predict(theta, return_std=True)
            means[j] = float(m[0])
            stds[j] = float(s[0])
        return means, stds

    def save(self, path):
        joblib.dump({
            "kernel_name": self.kernel_name, "alpha": self.alpha,
            "gprs": self._gprs, "D": self._D, "M": self._M,
        }, Path(path))

    @classmethod
    def load(cls, path):
        d = joblib.load(Path(path))
        obj = cls(kernel=d["kernel_name"], alpha=d["alpha"])
        obj._gprs = d["gprs"]
        obj._D = d["D"]
        obj._M = d["M"]
        return obj
