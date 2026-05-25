import pytest
from hydrus_research.sensitivity import SensitivityResult


def test_sensitivity_result_construction():
    r = SensitivityResult(
        method="sobol",
        param_names=["alpha", "n", "Ks"],
        obs_names=["theta_z30_d1"],
        indices={"S1": [[0.31, 0.44, 0.0]],
                 "ST": [[0.56, 0.44, 0.24]],
                 "S1_conf": [[0.05, 0.05, 0.01]]},
        sample_size=1024,
        forward_cost_s=12.5,
    )
    assert r.method == "sobol"
    assert len(r.indices["S1"][0]) == 3


def test_sensitivity_result_rejects_unknown_method():
    with pytest.raises(Exception):
        SensitivityResult(method="banana", param_names=[], obs_names=[],
                          indices={}, sample_size=0, forward_cost_s=0)
