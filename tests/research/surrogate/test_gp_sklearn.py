import numpy as np
import pytest
from hydrus_research.surrogate.gp_sklearn import SklearnGPSurrogate


def f(theta):
    return np.array([np.sin(theta[0]) + np.cos(theta[1])])


def test_gp_fit_predict():
    rng = np.random.default_rng(7)
    thetas = rng.uniform(-np.pi, np.pi, size=(32, 2))
    ys = np.array([f(t) for t in thetas])
    surr = SklearnGPSurrogate()
    surr.fit(thetas, ys)
    theta_test = np.array([0.5, 1.0])
    mean, std = surr.predict(theta_test)
    assert mean.shape == (1,) and std.shape == (1,)
    assert abs(float(mean[0]) - float(f(theta_test)[0])) < 0.1
    assert std[0] >= 0


def test_gp_save_load(tmp_path):
    rng = np.random.default_rng(11)
    thetas = rng.uniform(0, 1, size=(16, 2))
    ys = np.array([f(t) for t in thetas])
    surr = SklearnGPSurrogate()
    surr.fit(thetas, ys)
    p = tmp_path / "gp.joblib"
    surr.save(p)
    surr2 = SklearnGPSurrogate.load(p)
    m1, _ = surr.predict(np.array([0.5, 0.5]))
    m2, _ = surr2.predict(np.array([0.5, 0.5]))
    np.testing.assert_allclose(m1, m2)
