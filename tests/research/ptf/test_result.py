import pytest
import numpy as np
from hydrus_research.ptf import PTFResult


def test_ptf_result_minimum():
    r = PTFResult(theta_r=0.05, theta_s=0.43, alpha=0.036, n=1.56, Ks=4.31,
                  method="carsel_parrish")
    assert r.L == 0.5         # default Mualem tortuosity
    assert r.covariance is None


def test_ptf_result_with_covariance():
    cov = np.eye(5).tolist()
    r = PTFResult(theta_r=0.05, theta_s=0.43, alpha=0.036, n=1.56, Ks=4.31,
                  method="rosetta3_h2", covariance=cov)
    assert len(r.covariance) == 5
    assert len(r.covariance[0]) == 5


def test_ptf_result_method_must_be_known():
    with pytest.raises(Exception):
        PTFResult(theta_r=0.05, theta_s=0.43, alpha=0.036, n=1.56, Ks=4.31,
                  method="banana")
