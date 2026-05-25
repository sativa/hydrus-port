import numpy as np
import pytest
from hydrus_research.sensitivity import fast_indices, SensitivityResult


def ishigami(theta: np.ndarray) -> np.ndarray:
    x1, x2, x3 = theta
    return np.array([np.sin(x1) + 7.0 * np.sin(x2) ** 2
                     + 0.1 * x3 ** 4 * np.sin(x1)])


class _IshigamiParamMap:
    @property
    def names(self): return ["x1", "x2", "x3"]
    def bounds_array(self):
        return np.array([[-np.pi, np.pi]] * 3)


def test_fast_returns_sensitivity_result():
    r = fast_indices(forward=ishigami, param_map=_IshigamiParamMap(),
                     obs_names=["f"], n=512, seed=42)
    assert isinstance(r, SensitivityResult)
    assert r.method == "fast"
    assert "S1" in r.indices and "ST" in r.indices


def test_fast_ranks_x2_x1_x3_correctly():
    """FAST on Ishigami: x2 strongest direct effect (sin² interaction);
    S1 ranking should be x2 > x1 > x3."""
    r = fast_indices(forward=ishigami, param_map=_IshigamiParamMap(),
                     obs_names=["f"], n=1024, seed=42)
    S1 = np.array(r.indices["S1"][0])
    assert S1[1] > S1[0] > 0, f"unexpected ranking; S1 = {S1}"
