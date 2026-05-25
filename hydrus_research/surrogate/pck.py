"""PC-Kriging via smt KPLS — sklearn-style API for hydrology response surfaces."""
from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np

from .base import SurrogateModel


class PCKSurrogate(SurrogateModel):
    def __init__(self, pce_degree: int = 3):
        self.pce_degree = pce_degree
        self._kpls = None
        self._theta_min = None
        self._theta_range = None
        self._M: int | None = None

    def fit(self, thetas, ys):
        from smt.surrogate_models import KPLS
        thetas = np.asarray(thetas, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if ys.ndim == 1:
            ys = ys.reshape(-1, 1)
        self._M = ys.shape[1]
        self._theta_min = thetas.min(axis=0)
        self._theta_range = np.maximum(thetas.max(axis=0) - self._theta_min, 1e-9)
        theta_n = (thetas - self._theta_min) / self._theta_range
        self._kpls = []
        for j in range(self._M):
            sm = KPLS(print_global=False)
            sm.set_training_values(theta_n, ys[:, j])
            sm.train()
            self._kpls.append(sm)

    def predict(self, theta):
        if self._kpls is None:
            raise RuntimeError("must call fit() first")
        theta = np.asarray(theta, dtype=float).reshape(1, -1)
        theta_n = (theta - self._theta_min) / self._theta_range
        means = np.empty(self._M, dtype=float)
        stds = np.empty(self._M, dtype=float)
        for j, sm in enumerate(self._kpls):
            means[j] = float(sm.predict_values(theta_n).flatten()[0])
            try:
                stds[j] = float(np.sqrt(sm.predict_variances(theta_n).flatten()[0]))
            except Exception:
                stds[j] = float("nan")
        return means, stds

    def save(self, path):
        joblib.dump({
            "pce_degree": self.pce_degree, "kpls": self._kpls, "M": self._M,
            "theta_min": self._theta_min, "theta_range": self._theta_range,
        }, Path(path))

    @classmethod
    def load(cls, path):
        d = joblib.load(Path(path))
        obj = cls(pce_degree=d["pce_degree"])
        obj._kpls = d["kpls"]; obj._M = d["M"]
        obj._theta_min = d["theta_min"]; obj._theta_range = d["theta_range"]
        return obj
