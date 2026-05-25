"""Public surrogate API."""
import numpy as np
from .metrics import nse, rmse, coverage


def evaluate(surrogate, batch_result) -> dict:
    thetas = batch_result.thetas
    obs = batch_result.ys
    M = obs.shape[1]
    means = np.empty_like(obs)
    stds = np.empty_like(obs)
    for i, t in enumerate(thetas):
        m, s = surrogate.predict(t)
        means[i] = m; stds[i] = s
    return {
        "NSE":      [nse(means[:, j], obs[:, j]) for j in range(M)],
        "RMSE":     [rmse(means[:, j], obs[:, j]) for j in range(M)],
        "coverage": [coverage(means[:, j], stds[:, j], obs[:, j]) for j in range(M)],
    }
