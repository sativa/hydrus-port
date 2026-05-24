import pytest
from hydrus_research.ptf import texture_to_vg


def test_auto_picks_h2_when_bd_given():
    """rosetta3_auto = pick the highest-hierarchy ROSETTA model the inputs support."""
    pytest.importorskip("rosetta")
    r = texture_to_vg(sand_pct=45, silt_pct=35, clay_pct=20,
                      bulk_density_g_cm3=1.4)
    assert r.method == "rosetta3_h2"


def test_carsel_method_explicit():
    """Caller can force Carsel-Parrish by passing a USDA class instead of texture %."""
    from hydrus_research.ptf.api import texture_to_vg as f
    # Carsel mode is reached via usda_class_to_vg, not texture_to_vg.
    # texture_to_vg with method='carsel_parrish' should fall through to the
    # nearest USDA class center.
    r = f(sand_pct=40, silt_pct=40, clay_pct=20, method="carsel_parrish")
    assert r.method == "carsel_parrish"


def test_wosten_method_explicit():
    r = texture_to_vg(sand_pct=45, silt_pct=35, clay_pct=20,
                      bulk_density_g_cm3=1.4, organic_matter_pct=1.5,
                      method="wosten")
    assert r.method == "wosten"


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        texture_to_vg(sand_pct=45, silt_pct=35, clay_pct=20, method="other_thing")
