import numpy as np
import pytest
from hydrus_research.sensitivity import pawn_kde, SensitivityResult


def ishigami(theta: np.ndarray) -> np.ndarray:
    x1, x2, x3 = theta
    return np.array([np.sin(x1) + 7.0 * np.sin(x2) ** 2
                     + 0.1 * x3 ** 4 * np.sin(x1)])


class _IshigamiParamMap:
    @property
    def names(self): return ["x1", "x2", "x3"]
    def bounds_array(self):
        return np.array([[-np.pi, np.pi]] * 3)


def test_pawn_returns_sensitivity_result():
    r = pawn_kde(forward=ishigami, param_map=_IshigamiParamMap(),
                 obs_names=["f"], n=2000, s=10, seed=42)
    assert isinstance(r, SensitivityResult)
    assert r.method == "pawn"
    # PAWN returns minimum, mean, median, maximum, CV per param
    for key in ("minimum", "mean", "median", "maximum", "CV"):
        assert key in r.indices
        assert len(r.indices[key][0]) == 3
