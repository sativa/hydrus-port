"""End-to-end runner test against the in-repo HYDRUS-1D adapter."""
from datetime import date
import numpy as np
import pytest

from hydrus_research.agronomy.runner import run_agronomy
from hydrus_research.agronomy.types import AgronomyRequest, IrrigEvent, FertEvent


@pytest.mark.slow
def test_runner_returns_finite_theta_and_water_balance(tmp_path):
    req = AgronomyRequest(
        crop_id="maize", soil_id="loam", weather_id="n_china_avg",
        horizon_days=30, start_year=2026,
        irrigation=[IrrigEvent(date=date(2026, 5, 10), depth_mm=20.0)],
        fertilizer=[],
    )
    result = run_agronomy(req, work_dir=tmp_path)

    z = np.asarray(result.z_cm)
    theta = np.asarray(result.theta_zt)

    assert z[0] < z[-1]                       # ascending (surface first)
    assert theta.shape == (len(result.t_days), len(z))
    assert np.all(np.isfinite(theta))
    assert np.all((theta >= 0) & (theta <= 0.6))   # physical bounds for loam
    # Water balance is populated.
    assert result.water_balance.rain_mm >= 0
    assert result.water_balance.irrig_mm == pytest.approx(20.0, abs=0.5)


@pytest.mark.slow
def test_runner_with_fert_produces_nonzero_n_profile(tmp_path):
    """Irrigation + fertilizer run must yield non-zero N-NO₃ near the surface after fert day."""
    req = AgronomyRequest(
        crop_id="maize", soil_id="loam", weather_id="n_china_avg",
        horizon_days=20, start_year=2026,
        irrigation=[IrrigEvent(date=date(2026, 5, 10), depth_mm=20.0)],
        fertilizer=[FertEvent(date=date(2026, 5, 12), kg_n_ha=60.0)],
    )
    result = run_agronomy(req, work_dir=tmp_path)

    z = np.asarray(result.z_cm)
    theta = np.asarray(result.theta_zt)
    n_zt = np.asarray(result.n_zt)

    # Basic shape sanity.
    assert n_zt.shape == theta.shape, (
        f"n_zt shape {n_zt.shape} must match theta shape {theta.shape}"
    )

    # At least one non-zero concentration must exist (fertilizer was applied).
    assert n_zt.max() > 0.0, "N-NO₃ field is all zeros — solute transport not active"

    # After the fert event (day 12 = t_day 10), near-surface nodes (z < 10 cm)
    # must have a measurable concentration (> 0.1 mg/L).
    fert_t_day = 10.0  # fert date minus sow date offset
    t_days = np.asarray(result.t_days)
    # Find time indices after the fertilizer application.
    post_fert_idx = np.where(t_days >= fert_t_day)[0]
    shallow_idx = np.where(z <= 10.0)[0]
    assert post_fert_idx.size > 0, "No time steps after fertilizer application"
    assert shallow_idx.size > 0, "No shallow (z ≤ 10 cm) nodes found"

    max_shallow_n = n_zt[np.ix_(post_fert_idx, shallow_idx)].max()
    assert max_shallow_n > 0.1, (
        f"Peak near-surface N-NO₃ after fert = {max_shallow_n:.4f} mg/L; expected > 0.1 mg/L"
    )

    # Percolation should be positive (irrigation drives downward drainage).
    assert result.water_balance.percolation_mm > 0, (
        "Percolation should be > 0 with 20 mm irrigation"
    )

    # N budget: applied should equal 60 kg N/ha.
    assert result.n_budget.applied_kg_ha == pytest.approx(60.0, abs=0.1)
