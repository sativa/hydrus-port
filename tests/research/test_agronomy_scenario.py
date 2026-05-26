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


def test_canonical_scenario_writes_valid_hydrus1d_input(tmp_path):
    """Catches Critical issues 1-3 — the canonical Scenario must be adapter-writable."""
    from hydrus_research.agronomy.scenario_builder import build_scenario
    from hydrus_research.library.crops import get_crop
    from hydrus_research.library.soils import get_soil
    from hydrus_research.library.weather import load_weather_series
    from hydrus_research.agronomy.types import AgronomyRequest, IrrigEvent, FertEvent
    from hydrus_port.adapters.hydrus1d import save as save_h1d
    from datetime import date

    req = AgronomyRequest(
        crop_id="maize", soil_id="loam", weather_id="n_china_avg",
        horizon_days=14, start_year=2026,
        irrigation=[IrrigEvent(date=date(2026, 5, 10), depth_mm=20.0)],
        fertilizer=[FertEvent(date=date(2026, 5, 12), kg_n_ha=60.0)],
    )
    sc = build_scenario(
        get_crop("maize"), get_soil("loam"),
        load_weather_series("n_china_avg"), req,
    )
    # Adapter should be able to write the canonical Scenario without crashing.
    save_h1d(sc.scenario, str(tmp_path))
    # And produce the canonical files HYDRUS-1D expects.
    files = {p.name for p in tmp_path.iterdir()}
    assert "Profile.dat" in files
    assert "Selector.in" in files

    # z convention check: Profile.dat must have z=0 at surface and negative below.
    profile = (tmp_path / "Profile.dat").read_text().splitlines()
    z_values = []
    for line in profile:
        parts = line.split()
        if len(parts) >= 2:
            try:
                z_values.append(float(parts[1]))
            except ValueError:
                continue
    assert z_values, "no numeric z values found in Profile.dat"
    # All z values must be ≤ 0; first valid one is surface=0; last is ≤ -50 (loam profile 200 cm).
    assert max(z_values) == 0.0, f"surface z should be 0, got max={max(z_values)}"
    assert min(z_values) <= -50.0, f"deepest z should be deeply negative, got min={min(z_values)}"


def test_to_dict_returns_deep_copy_so_nested_mutation_does_not_leak():
    from hydrus_research.agronomy.scenario_builder import build_scenario
    from hydrus_research.library.crops import get_crop
    from hydrus_research.library.soils import get_soil
    from hydrus_research.library.weather import load_weather_series
    from hydrus_research.agronomy.types import AgronomyRequest
    sc = build_scenario(
        get_crop("maize"), get_soil("loam"),
        load_weather_series("n_china_avg"),
        AgronomyRequest(crop_id="maize", soil_id="loam",
                        weather_id="n_china_avg", horizon_days=30),
    )
    d1 = sc.to_dict()
    d1["sink"]["feddes"]["P3"] = 999.0
    d2 = sc.to_dict()
    assert d2["sink"]["feddes"]["P3"] == -8000, \
        "to_dict() must return a deep copy"
