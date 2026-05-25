import shutil
import pytest
import numpy as np

pyemu = pytest.importorskip("pyemu", reason="pyemu not installed")
from hydrus_research.inversion import fit_pyemu, InversionResult
from hydrus_research.parameters import ParameterSpec, ParameterMap
from hydrus_research.observations import ObservationSpec, ObservationSet


def test_pyemu_fit_raises_when_binary_missing():
    """If pestpp-ies isn't on PATH, the wrapper must raise a clear actionable
    error — never silently fabricate a result."""
    binary = shutil.which("pestpp-ies") or shutil.which("PESTPP-IES")
    if binary:
        pytest.skip("pestpp-ies IS available; can't test the missing-binary path")

    pm = ParameterMap([ParameterSpec(name="alpha", target="materials[0].alpha",
                                     bounds=(0.005, 0.05), transform="log")])
    obs = ObservationSet(
        specs=[ObservationSpec(name="theta_z30_d1", kind="theta",
                               location={"z_cm": -30.0}, time_day=1.0)],
        values=np.array([0.31]), sigmas=np.array([0.02]),
    )
    with pytest.raises(RuntimeError, match="pestpp-ies"):
        fit_pyemu(scenario_dir="tests/fixtures/infiltr_v1/inputs",
                  param_map=pm, obs=obs, method="ies",
                  n_real=4, n_iter=1)


@pytest.mark.skipif(
    not (shutil.which("pestpp-ies") or shutil.which("PESTPP-IES")),
    reason="pestpp-ies binary not on PATH",
)
def test_pyemu_ies_runs_synthetic_recovery():
    """Real IES smoke (slow; opt-in). Skipped if pestpp-ies missing."""
    pm = ParameterMap([ParameterSpec(name="alpha", target="materials[0].alpha",
                                     bounds=(0.005, 0.05), transform="log")])
    obs = ObservationSet(
        specs=[ObservationSpec(name="theta_z30_d1", kind="theta",
                               location={"z_cm": -30.0}, time_day=1.0)],
        values=np.array([0.31]), sigmas=np.array([0.02]),
    )
    result = fit_pyemu(scenario_dir="tests/fixtures/infiltr_v1/inputs",
                      param_map=pm, obs=obs,
                      method="ies", n_real=4, n_iter=1)
    assert isinstance(result, InversionResult)
    assert result.backend == "pyemu_ies"
    assert result.n_forward_calls > 0
