import numpy as np
import pytest
from dataclasses import replace

from hydrus_research.simulator import (
    Forcing, InitialState, SimResult, Simulator, make_forward,
)
from hydrus_research.parameters import ParameterSpec, ParameterMap
from hydrus_research.observations import ObservationSpec


class _FakeSimulator(Simulator):
    """Pretends to be a 1-D solver. theta_at(z, t) = alpha * z + n * t.
    Reads alpha/n straight out of the patched scenario dict."""
    name = "fake"
    dimension = 1

    def run(self, scenario, forcing, ic):
        alpha = scenario["x"]
        n = scenario["y"]
        z = np.linspace(0.0, 100.0, 11)
        t = np.array([0.0, 1.0, 2.0, 5.0])
        theta = np.outer(t * n, np.ones_like(z)) + np.outer(np.ones_like(t), z * alpha)
        return SimResult(
            times=t, z=z, theta=theta, h=np.zeros_like(theta),
            c=None, fluxes={}, mass_balance={},
            final_state=InitialState(z_cm=z, theta=theta[-1], h_cm=None,
                                     c_mg_per_L=None, t_celsius=None),
            meta={"solver": "fake", "wall_s": 0.0},
        )

    def observable_at(self, result, spec):
        z_target = spec.location["z_cm"]
        t_target = spec.time_day
        theta_at_z = np.array([np.interp(z_target, result.z, row) for row in result.theta])
        return float(np.interp(t_target, result.times, theta_at_z))


def test_make_forward_returns_callable_of_right_shape():
    sim = _FakeSimulator()
    pm = ParameterMap([
        ParameterSpec(name="alpha", target="x", bounds=(0.0, 1.0)),
        ParameterSpec(name="n",     target="y", bounds=(0.0, 5.0)),
    ])
    template = {"x": 0.0, "y": 0.0}     # FakeSimulator reads scenario["x"], scenario["y"]
    obs_specs = [
        ObservationSpec(name="a", kind="theta", location={"z_cm": 50.0}, time_day=1.0),
        ObservationSpec(name="b", kind="theta", location={"z_cm": 100.0}, time_day=2.0),
    ]
    forward = make_forward(sim, pm,
                           template_scenario=template,
                           forcing=None, ic=None,
                           obs_specs=obs_specs)
    theta = pm.to_vector({"alpha": 0.1, "n": 1.5})
    y = forward(theta)
    assert y.shape == (2,)
    # at (z=50, t=1): 0.1*50 + 1.5*1 = 6.5
    assert y[0] == pytest.approx(6.5, rel=1e-6)
    # at (z=100, t=2): 0.1*100 + 1.5*2 = 13.0
    assert y[1] == pytest.approx(13.0, rel=1e-6)
