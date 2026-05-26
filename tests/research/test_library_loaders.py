"""Smoke tests for hydrus_research.library loaders."""
from __future__ import annotations

from hydrus_research.library.crops import load_crops, get_crop
from hydrus_research.library.soils import load_soils, get_soil
from hydrus_research.library.weather import (
    load_weather_meta, load_weather_series,
)


def test_crops_lib_has_10_known_ids():
    crops = load_crops()
    ids = {c.id for c in crops}
    assert ids >= {"maize", "wheat_winter", "rice", "cotton", "tomato",
                   "grape", "apple", "tea", "rapeseed", "soybean"}


def test_crop_lookup_returns_pydantic_model():
    maize = get_crop("maize")
    assert maize.name_zh == "玉米"
    assert maize.feddes.P3 == -8000
    assert maize.root.max_depth_cm == 120
    assert maize.season.sow_doy == 121


def test_soils_lib_has_12_known_ids():
    soils = load_soils()
    ids = {s.id for s in soils}
    assert ids >= {"sand", "loamy_sand", "sandy_loam", "loam", "silt",
                   "silt_loam", "sandy_clay_loam", "clay_loam",
                   "silty_clay_loam", "clay", "sand_over_clay",
                   "topsoil_subsoil_bedrock"}


def test_soil_layered_preset_has_multiple_layers():
    s = get_soil("sand_over_clay")
    assert len(s.layers) == 2
    assert s.layers[0].depth_cm == 40
    assert s.layers[0].vg.alpha == 0.145
    assert s.layers[1].vg.alpha == 0.008


def test_weather_meta_lists_6_profiles():
    meta = load_weather_meta()
    ids = {m["id"] for m in meta}
    assert ids >= {"n_china_avg", "n_china_wet", "n_china_dry",
                   "c_china_meiyu", "s_china_double", "nw_china_irrig"}


def test_weather_series_returns_365_day_arrays():
    s = load_weather_series("n_china_avg")
    assert len(s["doy"]) == 365
    assert len(s["P_mm"]) == 365
    assert len(s["PET_mm"]) == 365
    assert all(p >= 0 for p in s["P_mm"])
    assert all(e > 0 for e in s["PET_mm"])


def test_loaders_return_fresh_lists_so_mutation_does_not_leak():
    a = load_crops(); a.append("poison")
    b = load_crops()
    assert len(b) == 10 and "poison" not in b

    s = load_weather_series("n_china_avg"); s["P_mm"].append(999.0)
    s2 = load_weather_series("n_china_avg")
    assert 999.0 not in s2["P_mm"]
