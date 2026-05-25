"""Tests for Richards3DSimulator adapter."""
import numpy as np
import pytest
from pathlib import Path
from hydrus_research.simulator.richards3d_adapter import Richards3DSimulator


def test_richards3d_construction():
    sim = Richards3DSimulator(work_root=Path("/tmp/r3d_test"))
    assert sim.name == "richards3d"
    assert sim.dimension == 3


def test_richards3d_run_demo():
    """Use the built-in box demo via validate_richards3d.run_case."""
    sim = Richards3DSimulator(work_root=Path("/tmp/r3d_demo"))
    scenario = {"geometry": {"kind": "3d", "demo": True}, "meta": {"name": "box"}}
    r = sim.run(scenario, forcing=None, ic=None)
    assert r.theta.ndim == 2      # (NT, Nnode) — NT=1 for final-snapshot only
    assert r.meta["solver"] == "richards3d"
    assert r.theta.shape[1] > 0
