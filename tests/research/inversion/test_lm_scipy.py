import numpy as np
import pytest
from hydrus_research.inversion import fit_lm, InversionResult
from hydrus_research.parameters import ParameterSpec, ParameterMap
from hydrus_research.observations import ObservationSpec, ObservationSet


def quadratic(theta):
    a, b = theta
    return np.array([a + b, a * b, b ** 2])


def test_lm_recovers_synthetic_theta_within_5pct():
    pm = ParameterMap([
        ParameterSpec(name="a", target="a", bounds=(-5.0, 5.0)),
        ParameterSpec(name="b", target="b", bounds=(-5.0, 5.0)),
    ])
    theta_true = np.array([1.5, 2.0])
    y_obs = quadratic(theta_true)
    obs = ObservationSet(
        specs=[ObservationSpec(name=f"o{i}", kind="theta",
                               location={"z_cm": 0.0}, time_day=0.0)
               for i in range(3)],
        values=y_obs,
        sigmas=np.array([0.01, 0.01, 0.01]),
    )
    result = fit_lm(forward=quadratic, param_map=pm, obs=obs,
                    x0=pm.to_vector({"a": 0.5, "b": 0.5}),
                    max_nfev=200)
    assert isinstance(result, InversionResult)
    assert result.backend == "lm_scipy"
    np.testing.assert_allclose(result.best_params["a"], 1.5, atol=0.1)
    np.testing.assert_allclose(result.best_params["b"], 2.0, atol=0.1)
    assert result.n_forward_calls > 0
    assert result.wall_s > 0


def test_lm_returns_ci_from_hessian():
    pm = ParameterMap([ParameterSpec(name="a", target="a", bounds=(-5, 5))])
    obs = ObservationSet(
        specs=[ObservationSpec(name="o", kind="theta",
                               location={"z_cm": 0.0}, time_day=0.0)],
        values=np.array([4.0]), sigmas=np.array([0.01]),
    )
    result = fit_lm(forward=lambda t: np.array([t[0] ** 2]),
                    param_map=pm, obs=obs,
                    x0=pm.to_vector({"a": 1.0}), max_nfev=100)
    assert result.parameter_ci_lo
    assert result.parameter_ci_hi
    assert result.parameter_ci_lo["a"] < result.best_params["a"]
    assert result.parameter_ci_hi["a"] > result.best_params["a"]
