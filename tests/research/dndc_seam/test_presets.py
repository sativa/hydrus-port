import pytest
from hydrus_research.dndc_seam.presets import (
    list_crop_presets, load_crop_preset,
)
from hydrus_research.dndc_seam.schema import FeddesParams, RootGrowth


def test_list_presets_contains_core_crops():
    names = list_crop_presets()
    for c in ("maize", "wheat", "rice", "soybean", "cotton",
              "tomato", "potato", "grass", "alfalfa"):
        assert c in names
    assert len(names) >= 15


def test_load_maize_preset():
    feddes, root, desc = load_crop_preset("maize")
    assert isinstance(feddes, FeddesParams)
    assert isinstance(root, RootGrowth)
    assert feddes.h4 == -8000
    assert root.z_max_cm == 100
    assert "Maize" in desc or "maize" in desc.lower()


def test_load_unknown_preset_raises():
    with pytest.raises(KeyError):
        load_crop_preset("not_a_crop")


def test_bare_soil_preset_no_transpiration():
    feddes, root, _ = load_crop_preset("bare_soil")
    assert feddes.pet_high_cm_d == 0.0
    assert root.z_max_cm < 1.0
