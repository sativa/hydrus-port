import pytest
import numpy as np

pymc = pytest.importorskip("pymc", reason="pymc not installed; in [research-uq] extras")
from hydrus_research.inversion import InversionResult
from hydrus_research.inversion.pymc_bayes import fit_pymc
from hydrus_research.parameters import ParameterSpec, ParameterMap
from hydrus_research.observations import ObservationSpec, ObservationSet


def quadratic(theta):
    a = theta[0]
    return np.array([a * a])


def test_fit_pymc_recovers_a_true_within_2sigma():
    pm_map = ParameterMap([
        ParameterSpec(name="a", target="a", bounds=(0.1, 5.0),
                      prior_mean=1.0, prior_std=2.0),
    ])
    a_true = 2.5
    y_obs = quadratic(np.array([a_true]))
    obs = ObservationSet(
        specs=[ObservationSpec(name="o", kind="theta",
                               location={"z_cm": 0.0}, time_day=0.0)],
        values=y_obs, sigmas=np.array([0.5]),
    )
    result = fit_pymc(forward=quadratic, param_map=pm_map, obs=obs,
                      draws=500, tune=500, chains=2, seed=42)
    assert isinstance(result, InversionResult)
    assert result.backend == "pymc_nuts"
    assert result.posterior_ensemble is not None
    # Posterior mean should be within 2σ of a_true
    posterior_a = np.array(result.posterior_ensemble)[:, 0]
    mean = posterior_a.mean()
    std = posterior_a.std()
    assert abs(mean - a_true) < 2 * std + 0.5     # generous tolerance


def test_fit_pymc_returns_diagnostics():
    pm_map = ParameterMap([ParameterSpec(name="a", target="a", bounds=(0.1, 5.0))])
    obs = ObservationSet(
        specs=[ObservationSpec(name="o", kind="theta",
                               location={"z_cm": 0.0}, time_day=0.0)],
        values=np.array([4.0]), sigmas=np.array([0.5]),
    )
    result = fit_pymc(forward=quadratic, param_map=pm_map, obs=obs,
                      draws=200, tune=200, chains=2, seed=7)
    assert "r_hat" in result.diagnostics
    assert "ess_bulk" in result.diagnostics
