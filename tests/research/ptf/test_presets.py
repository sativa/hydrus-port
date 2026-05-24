import pytest
from hydrus_research.ptf import usda_class_to_vg
from hydrus_research.ptf.presets import USDA_TEXTURE_CENTERS


def test_class_to_vg_loam():
    r = usda_class_to_vg("loam")
    # Same as Carsel-Parrish since presets aliases the lookup
    assert r.method == "carsel_parrish"
    assert 0.02 < r.alpha < 0.05


def test_usda_texture_centers_has_12_entries():
    assert len(USDA_TEXTURE_CENTERS) == 12
    sand_center = USDA_TEXTURE_CENTERS["sand"]
    assert sand_center["sand_pct"] > 85
