import numpy as np
import pytest
from hydrus_research.uq import propagate_ptf_uncertainty, UQResult


class _FakePTF:
    """Stand-in for PTFResult; duck-typed."""
    def __init__(self):
        self.theta_r = 0.05
        self.theta_s = 0.43
        self.alpha = 0.036
        self.n = 1.56
        self.Ks = 24.96
        # Diagonal covariance (5×5)
        self.covariance = np.diag([1e-4, 1e-4, 1e-6, 1e-2, 1.0]).tolist()


def _forward(theta):
    """Toy: returns [sum, product] of the 5 VG params."""
    return np.array([theta.sum(), float(np.prod(theta[:3]))])


def test_propagate_ptf_returns_uq_result():
    ptf = _FakePTF()
    r = propagate_ptf_uncertainty(forward=_forward, ptf=ptf,
                                  param_names=["theta_r","theta_s","alpha","n","Ks"],
                                  obs_names=["sum", "product"],
                                  n=32, seed=42)
    assert isinstance(r, UQResult)
    assert r.method == "ptf_mc"
    assert len(r.ys) == 32
    assert len(r.ys[0]) == 2
    # Quantile bands populated
    for q in ("p2.5", "p50", "p97.5"):
        assert q in r.quantiles
        assert len(r.quantiles[q]) == 2


def test_propagate_ptf_reproducible_with_seed():
    ptf = _FakePTF()
    r1 = propagate_ptf_uncertainty(forward=_forward, ptf=ptf,
                                   param_names=["a","b","c","d","e"],
                                   obs_names=["s","p"], n=8, seed=1)
    r2 = propagate_ptf_uncertainty(forward=_forward, ptf=ptf,
                                   param_names=["a","b","c","d","e"],
                                   obs_names=["s","p"], n=8, seed=1)
    assert r1.ys == r2.ys
