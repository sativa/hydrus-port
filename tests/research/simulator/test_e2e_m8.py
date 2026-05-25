"""M8 acceptance: Morris sweep on EX1 (2D) via the new Swms2DSimulator.

This test runs ~16 real SWMS_2D solver invocations; expect 3-10 min total.
"""
import numpy as np
import pytest
from pathlib import Path
from hydrus_research.simulator import make_forward
from hydrus_research.simulator.swms2d_adapter import Swms2DSimulator
from hydrus_research.parameters import ParameterSpec, ParameterMap
from hydrus_research.observations import ObservationSpec
from hydrus_research.sensitivity import morris_screen
from hydrus_port.adapters.swms2d import load as load_2d


@pytest.mark.slow
def test_morris_on_ex1_2d():
    template = load_2d(Path("tests/fixtures/EX1/inputs")).to_dict()
    a0 = template["materials"][0]["alpha"]
    ks0 = template["materials"][0]["Ks"]
    pm = ParameterMap([
        ParameterSpec(name="alpha", target="materials[0].alpha",
                      bounds=(a0 * 0.5, a0 * 2.0), transform="log"),
        ParameterSpec(name="Ks", target="materials[0].Ks",
                      bounds=(ks0 * 0.1, ks0 * 5.0),
                      transform="log"),
    ])
    # EX1 time axis is in seconds (0-5400); use 1800.0 s as the target time
    obs = [ObservationSpec(name="theta_node5_t1800", kind="theta",
                           location={"node": 5}, time_day=1800.0)]
    sim = Swms2DSimulator()
    forward = make_forward(sim, pm, template_scenario=template,
                           forcing=None, ic=None, obs_specs=obs)
    r = morris_screen(forward, pm, ["theta_node5_t1800"],
                      n_trajectories=4, num_levels=4, seed=42, n_workers=1)
    assert r.method == "morris"
    # mu_star: list[list[float]] (M observables x D params)
    mu_star = np.array(r.indices["mu_star"][0])  # first (only) observable
    assert (mu_star >= 0).all()
    # At least one of (alpha, Ks) should affect the observable visibly
    assert mu_star.max() > 1e-6
