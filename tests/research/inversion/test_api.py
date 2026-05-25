import pytest
import numpy as np
from hydrus_research.inversion import fit, InversionResult
from hydrus_research.parameters import ParameterSpec, ParameterMap
from hydrus_research.observations import ObservationSpec, ObservationSet


def quadratic(theta):
    return np.array([theta[0] + theta[1], theta[0] * theta[1]])


def test_fit_auto_picks_lm_for_small_d():
    pm = ParameterMap([
        ParameterSpec(name="a", target="a", bounds=(-5, 5)),
        ParameterSpec(name="b", target="b", bounds=(-5, 5)),
    ])
    obs = ObservationSet(
        specs=[ObservationSpec(name=f"o{i}", kind="theta",
                               location={"z_cm": 0.0}, time_day=0.0)
               for i in range(2)],
        values=np.array([3.5, 3.0]), sigmas=np.array([0.01, 0.01]),
    )
    r = fit(forward=quadratic, param_map=pm, obs=obs,
            scenario_dir=None, backend="auto", simulator_dimension=1)
    assert r.backend == "lm_scipy"


def test_fit_explicit_backend_lm():
    pm = ParameterMap([ParameterSpec(name="a", target="a", bounds=(-5, 5))])
    obs = ObservationSet(
        specs=[ObservationSpec(name="o", kind="theta",
                               location={"z_cm": 0.0}, time_day=0.0)],
        values=np.array([4.0]), sigmas=np.array([0.01]),
    )
    r = fit(forward=lambda t: np.array([t[0] ** 2]),
            param_map=pm, obs=obs, scenario_dir=None, backend="lm")
    assert r.backend == "lm_scipy"


def test_fit_unknown_backend_raises():
    pm = ParameterMap([ParameterSpec(name="a", target="a", bounds=(-5, 5))])
    obs = ObservationSet(
        specs=[ObservationSpec(name="o", kind="theta",
                               location={"z_cm": 0.0}, time_day=0.0)],
        values=np.array([1.0]), sigmas=np.array([0.01]),
    )
    with pytest.raises(ValueError):
        fit(forward=lambda t: np.array([t[0]]),
            param_map=pm, obs=obs, scenario_dir=None, backend="quantum")
