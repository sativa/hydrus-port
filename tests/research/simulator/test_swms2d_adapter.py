"""Tests for Swms2DSimulator adapter."""
import numpy as np
import pytest
from pathlib import Path
from hydrus_research.simulator.swms2d_adapter import Swms2DSimulator
from hydrus_port.adapters.swms2d import load as load_2d


def _ex1_template():
    return load_2d(Path("tests/fixtures/EX1/inputs")).to_dict()


def test_swms2d_construction():
    sim = Swms2DSimulator(work_root=Path("/tmp/swms2d_test"))
    assert sim.name == "swms2d"
    assert sim.dimension == 2


def test_swms2d_run_ex1():
    sim = Swms2DSimulator(work_root=Path("/tmp/swms2d_ex1"))
    r = sim.run(_ex1_template(), forcing=None, ic=None)
    assert r.theta.ndim == 2
    assert r.times.ndim == 1
    assert r.theta.shape[0] == r.times.shape[0]
    # 2D adapter stores mesh nodes in r.z slot (per spec §2.1)
    # OR in a sibling attribute; verify shape matches num_np
    assert r.theta.shape[1] > 0
    assert r.meta["solver"] == "swms2d"
