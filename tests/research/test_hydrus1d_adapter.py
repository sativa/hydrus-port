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
