import pytest
from hydrus_research.inversion import InversionResult


def test_inversion_result_lm_minimum():
    r = InversionResult(
        backend="lm_scipy",
        best_params={"alpha": 0.036, "n": 1.56},
        parameter_ci_lo={"alpha": 0.030, "n": 1.50},
        parameter_ci_hi={"alpha": 0.042, "n": 1.62},
        n_forward_calls=42,
        wall_s=12.3,
    )
    assert r.backend == "lm_scipy"
    assert r.posterior_ensemble is None


def test_inversion_result_ies_with_posterior():
    r = InversionResult(
        backend="pyemu_ies",
        best_params={"alpha": 0.036, "n": 1.56},
        posterior_ensemble=[[0.034, 1.55], [0.036, 1.56], [0.038, 1.57]],
        posterior_param_names=["alpha", "n"],
        n_forward_calls=300,
        wall_s=120.0,
    )
    assert len(r.posterior_ensemble) == 3


def test_inversion_result_rejects_unknown_backend():
    with pytest.raises(Exception):
        InversionResult(backend="grpc", best_params={}, n_forward_calls=0, wall_s=0)
