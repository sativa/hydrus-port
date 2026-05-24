import pytest
from pathlib import Path

from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
from hydrus_port.adapters.hydrus1d import load as load_h1d_canonical


def _infiltr_v1_inputs() -> Path:
    return Path("tests/fixtures/infiltr_v1/inputs")


def _infiltr_v1_template_dict() -> dict:
    sc = load_h1d_canonical(_infiltr_v1_inputs())
    return sc.to_dict()


def test_hydrus1d_adapter_construction():
    sim = Hydrus1DSimulator(work_root=Path("/tmp"))
    assert sim.name == "hydrus1d"
    assert sim.dimension == 1


import numpy as np


def test_hydrus1d_run_returns_sim_result_on_infiltr_v1():
    template = _infiltr_v1_template_dict()
    sim = Hydrus1DSimulator(work_root=Path("/tmp/hres_m0"))
    # Run the unpatched template (no parameter sweep needed for this test).
    result = sim.run(template, forcing=None, ic=None)
    assert result.theta.ndim == 2
    assert result.times.ndim == 1
    assert result.z.ndim == 1
    assert result.theta.shape[0] == result.times.shape[0]
    assert result.theta.shape[1] == result.z.shape[0]
    assert "wall_s" in result.meta
    assert result.meta["solver"] == "hydrus1d"
    assert isinstance(result.mass_balance, dict)
