import numpy as np
import pytest
from hydrus_research.simulator import Forcing, InitialState, SimResult, Event


def test_event_is_frozen_dataclass():
    e = Event(time_day=1.0, depth_cm=0.0, amount=10.0,
              method="drip", solute_concs_mg_l={"NO3": 50.0})
    assert e.time_day == 1.0
    with pytest.raises(Exception):
        e.time_day = 2.0  # frozen


def test_forcing_minimum_construction():
    f = Forcing(
        times_days=np.array([0.0, 1.0, 2.0]),
        precip_cm_per_day=np.zeros(3),
        pet_cm_per_day=np.full(3, 0.5),
        lai=np.full(3, 2.0),
        root_depth_cm=np.full(3, 30.0),
        root_density_fn=lambda z, t: np.exp(-z / 30.0),
        irrigation_events=[],
        fert_events=[],
        n_source_terms=lambda z, t, theta, c: (0.0, 0.0),
        air_temp_c=None,
    )
    assert f.times_days.shape == (3,)
    assert f.air_temp_c is None
    assert f.root_density_fn(np.array([0.0, 30.0]), 0.0)[0] == 1.0


def test_initial_state_holds_profile():
    ic = InitialState(
        z_cm=np.linspace(0, 100, 11),
        theta=None,
        h_cm=np.full(11, -100.0),
        c_mg_per_L=None,
        t_celsius=None,
    )
    assert ic.h_cm[0] == -100.0
    assert ic.theta is None


def test_sim_result_holds_arrays_and_meta():
    z = np.linspace(0, 100, 5)
    t = np.array([0.0, 1.0])
    theta = np.zeros((2, 5))
    sr = SimResult(
        times=t, z=z, theta=theta, h=np.zeros_like(theta),
        c=None, fluxes={}, mass_balance={"total": 0.0},
        final_state=InitialState(z_cm=z, theta=theta[-1], h_cm=None,
                                 c_mg_per_L=None, t_celsius=None),
        meta={"solver": "test", "wall_s": 0.0},
    )
    assert sr.theta.shape == (2, 5)
    assert sr.meta["solver"] == "test"
