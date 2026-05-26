"""End-to-end runner test against the in-repo HYDRUS-1D adapter."""
from datetime import date
import numpy as np
import pytest

from hydrus_research.agronomy.runner import run_agronomy
from hydrus_research.agronomy.types import AgronomyRequest, IrrigEvent


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
