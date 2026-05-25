import pytest
from hydrus_research.uq import UQResult


def test_uq_result_minimum():
    r = UQResult(method="ptf_mc", param_names=["alpha", "n"],
                 obs_names=["theta_z30"], ys=[[0.31], [0.32], [0.29]],
                 n_samples=3)
    assert len(r.ys) == 3
    assert r.weights is None


def test_uq_result_rejects_unknown_method():
    with pytest.raises(Exception):
        UQResult(method="banana", param_names=[], obs_names=[],
                 ys=[], n_samples=0)
