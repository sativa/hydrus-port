from datetime import date
from hydrus_research.agronomy.types import (
    AgronomyRequest, IrrigEvent, FertEvent,
)
from hydrus_research.agronomy.scenario_builder import build_scenario
from hydrus_research.library.crops import get_crop
from hydrus_research.library.soils import get_soil
from hydrus_research.library.weather import load_weather_series


def test_build_scenario_roundtrips_through_canonical():
    crop = get_crop("maize")
    soil = get_soil("loam")
    weather = load_weather_series("n_china_avg")

    req = AgronomyRequest(
        crop_id="maize", soil_id="loam", weather_id="n_china_avg",
        horizon_days=150, start_year=2026,
        irrigation=[IrrigEvent(date=date(2026, 6, 1), depth_mm=30.0)],
        fertilizer=[FertEvent(date=date(2026, 5, 15), kg_n_ha=60.0)],
    )

    sc = build_scenario(crop, soil, weather, req)

    # Smoke checks: scenario has the right dimension, season, and a profile.
    d = sc.to_dict()
    assert d["dimension"] == "1d"
    assert d["sim"]["t_max"] >= 150
    # Profile depth equals sum of soil layer depths.
    expected_depth = sum(L.depth_cm for L in soil.layers)
    assert abs(d["profile"]["depth_cm"] - expected_depth) < 1e-6
    # Feddes block carries the crop's P3.
    assert d["sink"]["feddes"]["P3"] == crop.feddes.P3


def test_event_date_maps_to_t_day():
    from hydrus_research.agronomy.scenario_builder import event_to_t_day
    sow = date(2026, 5, 1)
    assert event_to_t_day(date(2026, 5, 1), sow) == 0
    assert event_to_t_day(date(2026, 6, 1), sow) == 31
