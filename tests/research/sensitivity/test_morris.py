import numpy as np
import pytest
from hydrus_research.sensitivity import morris_screen, SensitivityResult


def ishigami(theta: np.ndarray) -> np.ndarray:
    """SALib's canonical 3-param test function: f = sin(x1) + 7sin²(x2) + 0.1·x3⁴·sin(x1)."""
    x1, x2, x3 = theta
    return np.array([np.sin(x1) + 7.0 * np.sin(x2) ** 2
                     + 0.1 * x3 ** 4 * np.sin(x1)])


class _IshigamiParamMap:
    """Minimal duck-typed ParameterMap for SALib's bounds_array() call."""
    def __init__(self):
        self.specs = []                     # not used by Morris
    @property
    def names(self): return ["x1", "x2", "x3"]
    def bounds_array(self):
        return np.array([[-np.pi, np.pi]] * 3)


def test_morris_returns_sensitivity_result():
    r = morris_screen(forward=ishigami,
                      param_map=_IshigamiParamMap(),
                      obs_names=["f"],
                      n_trajectories=20,
                      num_levels=4,
                      seed=42)
    assert isinstance(r, SensitivityResult)
    assert r.method == "morris"
    assert r.param_names == ["x1", "x2", "x3"]
    assert r.obs_names == ["f"]
    for key in ("mu", "mu_star", "sigma", "mu_star_conf"):
        assert key in r.indices
        # Each index is per observable, length D=3
        assert len(r.indices[key]) == 1            # 1 observable
        assert len(r.indices[key][0]) == 3         # 3 params


def test_morris_ishigami_ranking():
    """On Ishigami: x2 has strongest direct effect (sin²), x1 has direct
    effect + interaction, x3 only in interaction term. Check characteristic
    signature: x2 high mu_star, x3 low mu but high sigma."""
    r = morris_screen(forward=ishigami,
                      param_map=_IshigamiParamMap(),
                      obs_names=["f"],
                      n_trajectories=40, num_levels=4, seed=42)
    mu_star = np.array(r.indices["mu_star"][0])
    sigma = np.array(r.indices["sigma"][0])
    # x2 (index 1) should have high mu_star (direct effect)
    assert mu_star[1] > mu_star[2], "x2 should rank above x3 by mu_star"
    # x3 (index 2) should have high sigma relative to its mu (interaction signature)
    assert sigma[2] > mu_star[2], "x3 should show interaction signature (σ > μ*)"
