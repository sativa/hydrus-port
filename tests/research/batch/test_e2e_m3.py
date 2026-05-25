"""M3 acceptance: LHS=8 on infiltr_v1, parquet round-trip, sanity bounds."""
import numpy as np
import pytest
from pathlib import Path

from hydrus_research.batch import BatchRunner, BatchResult
from hydrus_research.batch.sampling import lhs
from hydrus_research.parameters import ParameterSpec, ParameterMap
from hydrus_research.observations import ObservationSpec
from hydrus_research.simulator import make_forward
from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
from hydrus_port.adapters.hydrus1d import load as load_h1d_canonical


def test_m3_lhs8_on_infiltr_v1(tmp_path):
    template = load_h1d_canonical(Path("tests/fixtures/infiltr_v1/inputs")).to_dict()
    a0 = template["materials"][0]["alpha"]

    pm = ParameterMap([
        ParameterSpec(name="alpha", target="materials[0].alpha",
                      bounds=(a0 * 0.3, a0 * 3.0), transform="log"),
    ])
    # Use multiple observations at different times to capture parameter sensitivity
    obs = [
        ObservationSpec(name=f"theta_z30_d{t}", kind="theta",
                        location={"z_cm": -30.0}, time_day=t)
        for t in [0.5, 1.0, 2.0]
    ]
    sim = Hydrus1DSimulator()
    forward = make_forward(sim, pm,
                           template_scenario=template,
                           forcing=None, ic=None, obs_specs=obs)

    thetas = lhs(pm.bounds_array(), n=8, seed=42)
    runner = BatchRunner(forward=forward,
                         param_names=["alpha"],
                         obs_names=[f"theta_z30_d{t}" for t in [0.5, 1.0, 2.0]],
                         n_workers=2, show_progress=False)
    result = runner.run(thetas)

    # Shape + convergence
    assert result.N == 8
    assert result.D == 1
    assert result.M == 3
    assert result.n_converged >= 6        # allow up to 2 outlier failures
    # Physical range on converged rows
    valid = result.ys[result.converged]
    assert ((valid >= 0) & (valid <= 1)).all()
    # Variability — different alphas should produce different ys
    assert valid.flatten().std() > 1e-3

    # Parquet round-trip
    out = tmp_path / "m3_sweep.parquet"
    result.to_parquet(out)
    assert out.exists()
    result2 = BatchResult.from_parquet(out)
    np.testing.assert_array_equal(result.thetas, result2.thetas)
    np.testing.assert_array_equal(result.ys, result2.ys)
