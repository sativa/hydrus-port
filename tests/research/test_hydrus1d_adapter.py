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

from hydrus_research.observations import ObservationSpec


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


def test_hydrus1d_observable_at_theta_interp():
    template = _infiltr_v1_template_dict()
    sim = Hydrus1DSimulator(work_root=Path("/tmp/hres_m0"))
    result = sim.run(template, forcing=None, ic=None)
    # Pick a depth midway and a time present in the run
    z_target = float(result.z[len(result.z) // 2])
    t_target = float(result.times[len(result.times) // 2])
    spec = ObservationSpec(name="theta_mid", kind="theta",
                           location={"z_cm": z_target}, time_day=t_target)
    v = sim.observable_at(result, spec)
    # Should equal the exact array value (no interp needed at a sample point)
    expected = result.theta[len(result.times) // 2, len(result.z) // 2]
    assert v == pytest.approx(expected, rel=1e-6)


def test_hydrus1d_observable_at_h_interp_between_nodes():
    template = _infiltr_v1_template_dict()
    sim = Hydrus1DSimulator(work_root=Path("/tmp/hres_m0"))
    result = sim.run(template, forcing=None, ic=None)
    # Halfway between first two depth nodes, midway in time
    z_target = 0.5 * (float(result.z[0]) + float(result.z[1]))
    t_target = float(result.times[len(result.times) // 2])
    spec = ObservationSpec(name="h_between", kind="h",
                           location={"z_cm": z_target}, time_day=t_target)
    v = sim.observable_at(result, spec)
    # Expected: bilinear interp along z then along t (here t is exact)
    # Note: result.z is descending (0 to -185), so we must reverse for np.interp
    h_at_t = result.h[len(result.times) // 2]
    z_for_interp = result.z[::-1] if result.z[0] > result.z[-1] else result.z
    h_for_interp = h_at_t[::-1] if result.z[0] > result.z[-1] else h_at_t
    expected = float(np.interp(z_target, z_for_interp, h_for_interp))
    assert v == pytest.approx(expected, rel=1e-6)
