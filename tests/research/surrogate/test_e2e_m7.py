"""M7 acceptance: LHS=8 on infiltr_v1 → train GP → evaluate."""
import numpy as np
import pytest
from pathlib import Path

from hydrus_research.batch import BatchRunner
from hydrus_research.batch.sampling import lhs
from hydrus_research.parameters import ParameterSpec, ParameterMap
from hydrus_research.observations import ObservationSpec
from hydrus_research.simulator import make_forward
from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
from hydrus_research.surrogate import train_gp, evaluate
from hydrus_port.adapters.hydrus1d import load as load_h1d_canonical


def test_train_gp_on_infiltr_v1():
    template = load_h1d_canonical(Path("tests/fixtures/infiltr_v1/inputs")).to_dict()
    a0 = template["materials"][0]["alpha"]
    pm = ParameterMap([
        ParameterSpec(name="alpha", target="materials[0].alpha",
                      bounds=(a0 * 0.5, a0 * 2.0), transform="log"),
    ])
    obs = [ObservationSpec(name="theta_z30_d1", kind="theta",
                           location={"z_cm": -30.0}, time_day=1.0)]
    forward = make_forward(Hydrus1DSimulator(), pm,
                           template_scenario=template,
                           forcing=None, ic=None, obs_specs=obs)
    runner = BatchRunner(forward=forward, param_names=["alpha"],
                         obs_names=["theta_z30_d1"], n_workers=2, show_progress=False)
    train_br = runner.run(lhs(pm.bounds_array(), n=8, seed=42))
    test_br = runner.run(lhs(pm.bounds_array(), n=4, seed=43))
    surr = train_gp(train_br)
    metrics = evaluate(surr, test_br)
    assert metrics["NSE"][0] > -0.5  # very loose — just confirms it trains
