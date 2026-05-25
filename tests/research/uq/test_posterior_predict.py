import pytest
import numpy as np
from hydrus_research.uq import predict_with_posterior, UQResult


class _FakeInversionResult:
    backend = "lm_scipy"
    posterior_ensemble = [[0.1, 1.5], [0.2, 1.6], [0.15, 1.55]]
    posterior_param_names = ["alpha", "n"]


def test_predict_with_posterior_runs_forward_per_member():
    def fwd(theta): return np.array([theta[0] + theta[1]])
    r = predict_with_posterior(forward=fwd,
                               inv_result=_FakeInversionResult(),
                               obs_names=["sum"])
    assert isinstance(r, UQResult)
    assert r.method == "posterior_predict"
    assert len(r.ys) == 3
    assert r.ys[0] == [pytest.approx(1.6)]


def test_predict_with_posterior_raises_when_no_ensemble():
    class _NoEnsemble: posterior_ensemble = None
    with pytest.raises(ValueError):
        predict_with_posterior(forward=lambda t: t,
                               inv_result=_NoEnsemble(),
                               obs_names=[])
