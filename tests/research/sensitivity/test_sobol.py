import numpy as np
import pytest
from hydrus_research.sensitivity import sobol_decompose, SensitivityResult


def ishigami(theta: np.ndarray) -> np.ndarray:
    x1, x2, x3 = theta
    return np.array([np.sin(x1) + 7.0 * np.sin(x2) ** 2
                     + 0.1 * x3 ** 4 * np.sin(x1)])


class _IshigamiParamMap:
    @property
    def names(self): return ["x1", "x2", "x3"]
    def bounds_array(self):
        return np.array([[-np.pi, np.pi]] * 3)


def test_sobol_returns_sensitivity_result():
    r = sobol_decompose(forward=ishigami, param_map=_IshigamiParamMap(),
                        obs_names=["f"], n_base=512, seed=42)
    assert isinstance(r, SensitivityResult)
    assert r.method == "sobol"
    for key in ("S1", "ST", "S1_conf", "ST_conf"):
        assert key in r.indices


def test_sobol_ishigami_indices_within_5pct():
    """Documented analytic Ishigami: S1 ≈ [0.314, 0.443, 0.0],
    ST ≈ [0.557, 0.443, 0.244]. With n_base=1024 we should be within ~5%."""
    r = sobol_decompose(forward=ishigami, param_map=_IshigamiParamMap(),
                        obs_names=["f"], n_base=1024, seed=42)
    S1 = np.array(r.indices["S1"][0])
    ST = np.array(r.indices["ST"][0])
    np.testing.assert_allclose(S1, [0.314, 0.443, 0.0], atol=0.05)
    np.testing.assert_allclose(ST, [0.557, 0.443, 0.244], atol=0.05)
