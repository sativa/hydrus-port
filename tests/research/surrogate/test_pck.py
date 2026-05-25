import numpy as np
import pytest

smt = pytest.importorskip("smt", reason="smt not installed; in [research-3d] extras")
from hydrus_research.surrogate.pck import PCKSurrogate


def f(theta):
    return np.array([np.sin(theta[0]) + np.cos(theta[1])])


def test_pck_fit_predict():
    rng = np.random.default_rng(13)
    thetas = rng.uniform(-np.pi, np.pi, size=(32, 2))
    ys = np.array([f(t) for t in thetas])
    surr = PCKSurrogate(pce_degree=2)
    surr.fit(thetas, ys)
    mean, std = surr.predict(np.array([0.5, 1.0]))
    assert mean.shape == (1,)
    assert abs(float(mean[0]) - float(f(np.array([0.5, 1.0]))[0])) < 0.3
    assert std[0] >= 0
