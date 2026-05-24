import pytest
from hydrus_research.ptf.wosten_hypres import wosten_predict


def test_returns_5_param_ptf_result():
    r = wosten_predict(sand_pct=45, silt_pct=35, clay_pct=20,
                       bulk_density_g_cm3=1.4, organic_matter_pct=1.5,
                       topsoil=True)
    assert r.method == "wosten"
    assert 0.0 < r.theta_r < 0.2
    assert 0.3 < r.theta_s < 0.7
    assert 0.001 < r.alpha < 1.0
    assert 1.0 < r.n < 3.0
    assert r.Ks > 0


def test_topsoil_subsoil_differ():
    a = wosten_predict(sand_pct=45, silt_pct=35, clay_pct=20,
                       bulk_density_g_cm3=1.4, organic_matter_pct=1.5,
                       topsoil=True)
    b = wosten_predict(sand_pct=45, silt_pct=35, clay_pct=20,
                       bulk_density_g_cm3=1.4, organic_matter_pct=1.5,
                       topsoil=False)
    # Topsoil and subsoil PTFs differ in at least one parameter
    differ = any(abs(getattr(a, p) - getattr(b, p)) > 1e-6
                 for p in ("theta_s", "alpha", "n", "Ks"))
    assert differ


def test_rejects_bad_texture():
    with pytest.raises(ValueError):
        wosten_predict(sand_pct=120, silt_pct=0, clay_pct=0,
                       bulk_density_g_cm3=1.4, organic_matter_pct=1.5,
                       topsoil=True)
