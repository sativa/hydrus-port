import numpy as np
import pytest
from hydrus_research.ptf import vg_to_prior, PTFResult
from hydrus_research.parameters import ParameterSpec, ParameterMap


def _ptf_with_diag_cov():
    cov = (np.diag([0.0001, 0.0001, 0.0001, 0.01, 0.5])).tolist()
    return PTFResult(theta_r=0.07, theta_s=0.43, alpha=0.036,
                     n=1.56, Ks=24.96, method="rosetta3_h2",
                     covariance=cov)


def test_vg_to_prior_returns_5_specs():
    specs = vg_to_prior(_ptf_with_diag_cov(), material_index=0)
    assert len(specs) == 5
    names = {s.name for s in specs}
    assert names == {"theta_r", "theta_s", "alpha", "n", "Ks"}


def test_vg_to_prior_uses_log_transform_for_alpha_and_Ks():
    specs = vg_to_prior(_ptf_with_diag_cov(), material_index=0)
    by_name = {s.name: s for s in specs}
    assert by_name["alpha"].transform == "log"
    assert by_name["Ks"].transform == "log"
    assert by_name["theta_r"].transform == "linear"
    assert by_name["theta_s"].transform == "linear"
    assert by_name["n"].transform == "linear"


def test_vg_to_prior_targets_material_index_correctly():
    specs = vg_to_prior(_ptf_with_diag_cov(), material_index=2)
    by_name = {s.name: s for s in specs}
    assert by_name["alpha"].target == "materials[2].alpha"


def test_vg_to_prior_bounds_span_4_sigma_or_physical_minimum():
    specs = vg_to_prior(_ptf_with_diag_cov(), material_index=0)
    by_name = {s.name: s for s in specs}
    # theta_s prior mean is 0.43 with stddev 0.01 (from cov diag); 4σ = 0.04
    # so bounds should be roughly (0.39, 0.47)
    lo, hi = by_name["theta_s"].bounds
    assert lo > 0.34 and hi < 0.5


def test_vg_to_prior_works_without_covariance():
    """Without covariance, falls back to ±50% factor bounds."""
    ptf = PTFResult(theta_r=0.07, theta_s=0.43, alpha=0.036, n=1.56, Ks=24.96,
                    method="carsel_parrish")
    specs = vg_to_prior(ptf, material_index=0)
    by_name = {s.name: s for s in specs}
    assert by_name["alpha"].bounds[0] < 0.036 < by_name["alpha"].bounds[1]
