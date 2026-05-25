"""M4 acceptance: Morris on infiltr_v1 with 3 params + sanity ranking."""
import numpy as np
import pytest
from pathlib import Path

from hydrus_research.sensitivity import morris_screen
from hydrus_research.parameters import ParameterSpec, ParameterMap
from hydrus_research.observations import ObservationSpec
from hydrus_research.simulator import make_forward
from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
from hydrus_port.adapters.hydrus1d import load as load_h1d_canonical


def test_morris_on_infiltr_v1():
    template = load_h1d_canonical(Path("tests/fixtures/infiltr_v1/inputs")).to_dict()
    a0 = template["materials"][0]["alpha"]
    n0 = template["materials"][0]["n"]
    Ks0 = template["materials"][0]["Ks"]
    pm = ParameterMap([
        ParameterSpec(name="alpha", target="materials[0].alpha",
                      bounds=(a0 * 0.5, a0 * 2.0), transform="log"),
        ParameterSpec(name="n", target="materials[0].n",
                      bounds=(max(1.05, n0 * 0.8), n0 * 1.5), transform="linear"),
        ParameterSpec(name="Ks", target="materials[0].Ks",
                      bounds=(Ks0 * 0.2, Ks0 * 5.0), transform="log"),
    ])
    obs = [ObservationSpec(name="theta_z30_d1", kind="theta",
                           location={"z_cm": -30.0}, time_day=1.0)]
    sim = Hydrus1DSimulator()
    forward = make_forward(sim, pm, template_scenario=template,
                           forcing=None, ic=None, obs_specs=obs)
    r = morris_screen(forward, pm, ["theta_z30_d1"],
                      n_trajectories=4, num_levels=4, seed=42, n_workers=2)
    # 4 trajectories × (3+1) = 16 forward calls; ~2 min on a laptop
    assert r.method == "morris"
    assert len(r.indices["mu_star"][0]) == 3
    # All mu_star ≥ 0; at least one parameter has visible effect
    mu_star = np.array(r.indices["mu_star"][0])
    assert (mu_star >= 0).all()
    assert mu_star.max() > 1e-6, f"all mu_star ~ 0; sweep had no effect: {mu_star}"
