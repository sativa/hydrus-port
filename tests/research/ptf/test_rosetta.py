import pytest


rosetta = pytest.importorskip("rosetta",
                              reason="rosetta-soil package not installed")
from hydrus_research.ptf.rosetta import rosetta3_predict


def test_h1_sand_silt_clay_only():
    """H1 model uses sand/silt/clay only (no BD)."""
    r = rosetta3_predict(sand_pct=45, silt_pct=35, clay_pct=20)
    assert 0.0 < r.theta_r < 0.2
    assert 0.2 < r.theta_s < 0.6
    assert r.alpha > 0
    assert r.n > 1
    assert r.Ks > 0
    assert r.method == "rosetta3_h1"


def test_h2_with_bd_uses_h2_model():
    r = rosetta3_predict(sand_pct=45, silt_pct=35, clay_pct=20,
                         bulk_density_g_cm3=1.4)
    assert r.method == "rosetta3_h2"
    assert r.covariance is not None
    assert len(r.covariance) == 5


def test_h3_with_theta33():
    r = rosetta3_predict(sand_pct=45, silt_pct=35, clay_pct=20,
                         bulk_density_g_cm3=1.4, theta_33=0.31)
    assert r.method == "rosetta3_h3"


def test_h4_with_theta33_and_theta1500():
    r = rosetta3_predict(sand_pct=45, silt_pct=35, clay_pct=20,
                         bulk_density_g_cm3=1.4, theta_33=0.31, theta_1500=0.12)
    assert r.method == "rosetta3_h4"


def test_invalid_texture_sum_raises():
    with pytest.raises(ValueError):
        rosetta3_predict(sand_pct=50, silt_pct=50, clay_pct=10)    # sums to 110
