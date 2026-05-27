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

    # ATMOSPH.IN must be written when the request has atmospheric BC.
    assert "ATMOSPH.IN" in files, "ATMOSPH.IN missing — atmospheric BC not written"
    atm_text = (tmp_path / "ATMOSPH.IN").read_text()
    assert "tAtm" in atm_text, "ATMOSPH.IN header must contain 'tAtm'"
    # Count data rows (lines with numeric first token = one row per day in horizon).
    atm_data_rows = [
        ln for ln in atm_text.splitlines()
        if ln.strip() and ln.strip().split()[0].lstrip("-").replace(".", "").isdigit()
    ]
    assert len(atm_data_rows) >= 14, (
        f"ATMOSPH.IN should have ≥ 14 data rows for horizon_days=14, got {len(atm_data_rows)}"
    )


def test_scenario_with_fert_creates_chem_block(tmp_path):
    """BLOCK F must appear in Selector.in; cTop in ATMOSPH.IN; Profile.dat has Conc column."""
    from hydrus_research.agronomy.scenario_builder import build_scenario
    from hydrus_research.library.crops import get_crop
    from hydrus_research.library.soils import get_soil
    from hydrus_research.library.weather import load_weather_series
    from hydrus_research.agronomy.types import AgronomyRequest, IrrigEvent, FertEvent
    from hydrus_port.adapters.hydrus1d import save as save_h1d
    from datetime import date

    req = AgronomyRequest(
        crop_id="maize", soil_id="loam", weather_id="n_china_avg",
        horizon_days=20, start_year=2026,
        irrigation=[IrrigEvent(date=date(2026, 5, 10), depth_mm=20.0)],
        fertilizer=[FertEvent(date=date(2026, 5, 12), kg_n_ha=60.0)],
    )
    sc = build_scenario(
        get_crop("maize"), get_soil("loam"),
        load_weather_series("n_china_avg"), req,
    )
    save_h1d(sc.scenario, str(tmp_path))

    # 1. Selector.in must contain BLOCK F (solute transport section).
    selector_text = (tmp_path / "Selector.in").read_text()
    assert "BLOCK F" in selector_text, "BLOCK F missing from Selector.in"
    assert "Bulk.d" in selector_text, "BLOCK F chem_params header missing from Selector.in"

    # 2. ATMOSPH.IN must exist and have a non-zero cTop on the fertilizer day.
    assert (tmp_path / "ATMOSPH.IN").exists(), "ATMOSPH.IN missing"
    atm_text = (tmp_path / "ATMOSPH.IN").read_text()
    # With NS=1 solute, rows have 13 columns: tAtm ... Ampl cTop_1 cBot_1.
    # cTop_1 is the second-to-last column (index -2); cBot_1 is the last.
    ctop_values = []
    for line in atm_text.splitlines():
        parts = line.split()
        if len(parts) >= 13 and parts[0].lstrip("-").replace(".", "").isdigit():
            try:
                ctop_values.append(float(parts[-2]))  # cTop_1 is second-to-last
            except ValueError:
                continue
    assert any(v > 0 for v in ctop_values), (
        f"ATMOSPH.IN must have at least one non-zero cTop_1 (fertilizer day), got: {ctop_values[:5]}"
    )

    # 3. Profile.dat data rows must have extra columns (Temp + Conc) appended.
    # Profile.dat structure: 2-line BC header, then header line, then node rows.
    # Node rows: Node x h Mat Lay Beta Axz Bxz Dxz [Temp Conc1 ...]  (≥ 9 standard cols).
    profile_text = (tmp_path / "Profile.dat").read_text()
    data_rows = []
    for line in profile_text.splitlines():
        parts = line.split()
        # Skip short BC-code lines (4 cols) and the text header line
        if len(parts) >= 9 and parts[0].lstrip("-").isdigit():
            data_rows.append(parts)
    assert data_rows, "no node data rows (≥9 cols) found in Profile.dat"
    # Standard Profile.dat row has 9 columns; with lChem=True adds Temp + Conc = 11 columns.
    assert len(data_rows[0]) >= 10, (
        f"Profile.dat row should have ≥ 10 columns with Temp+Conc, got {len(data_rows[0])}: {data_rows[0]}"
    )


def test_result_parser_returns_ascending_z_and_finite_theta(tmp_path):
    """Synthesize a NOD_INF.OUT-like file and check the parser."""
    from hydrus_research.agronomy.result_parser import parse_nod_inf

    p = tmp_path / "NOD_INF.OUT"
    # 2 time blocks, 3 depths (HYDRUS-1D descending z; surface=0, deep negative)
    p.write_text(
        "Time:    0.000\n"
        "Node    z       h       theta\n"
        "1       0.000  -100.0   0.30\n"
        "2     -50.000  -150.0   0.25\n"
        "3    -100.000  -200.0   0.20\n"
        "Time:    1.000\n"
        "Node    z       h       theta\n"
        "1       0.000   -90.0   0.32\n"
        "2     -50.000  -140.0   0.27\n"
        "3    -100.000  -190.0   0.22\n"
    )
    z, t, theta = parse_nod_inf(p)
    # ascending z (0,50,100)
    assert z[0] < z[1] < z[2]
    assert list(z) == [0.0, 50.0, 100.0]
    assert len(t) == 2
    assert theta.shape == (2, 3)
    # theta[0,0] was 0.30 at surface
    assert abs(theta[0, 0] - 0.30) < 1e-6


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


def test_parse_nod_inf_skips_empty_time_blocks(tmp_path):
    """Empty Time block (no data rows) used to crash with IndexError."""
    from hydrus_research.agronomy.result_parser import parse_nod_inf
    p = tmp_path / "NOD_INF.OUT"
    p.write_text(
        "Time:    0.000\n"
        "Node    z       h       theta\n"
        "1       0.000  -100.0   0.30\n"
        "2     -50.000  -150.0   0.25\n"
        "Time:    0.500\n"
        "Node    z       h       theta\n"
        "Time:    1.000\n"
        "Node    z       h       theta\n"
        "1       0.000   -90.0   0.32\n"
        "2     -50.000  -140.0   0.27\n"
    )
    z, t, theta = parse_nod_inf(p)
    # Empty block silently skipped; 2 valid blocks remain.
    assert len(t) == 2
    assert theta.shape == (2, 2)


def test_parse_balance_real_format_extracts_storage_and_percolation(tmp_path):
    from hydrus_research.agronomy.result_parser import parse_balance
    p = tmp_path / "BALANCE.OUT"
    p.write_text(
        "----------------------------------------------------------\n"
        " Time       [T]        0.0000\n"
        "----------------------------------------------------------\n"
        " W-volume [L]        0.20000E+02  0.20000E+02\n"
        " Top Flux [L/T]      0.10000E+01\n"
        " Bot Flux [L/T]     -0.50000E+00\n"
        "----------------------------------------------------------\n"
        "\n"
        "----------------------------------------------------------\n"
        " Time       [T]        1.0000\n"
        "----------------------------------------------------------\n"
        " W-volume [L]        0.21500E+02  0.21500E+02\n"
        " Top Flux [L/T]      0.10000E+01\n"
        " Bot Flux [L/T]     -0.50000E+00\n"
        "----------------------------------------------------------\n"
    )
    b = parse_balance(p)
    # storage change: (21.5 - 20.0) cm × 10 = 15 mm
    assert abs(b["storage_change_mm"] - 15.0) < 1e-6
    # rain (positive Top Flux): 1 cm/day × 1 day = 1 cm = 10 mm
    assert abs(b["rain_mm"] - 10.0) < 1e-6
    # percolation (|negative Bot Flux|): 0.5 cm/day × 1 day = 0.5 cm = 5 mm
    assert abs(b["percolation_mm"] - 5.0) < 1e-6
    # no ET in this synthetic (Top Flux always positive)
    assert b["et_mm"] == 0.0


def test_parse_balance_on_real_fixture():
    """Smoke against a real BALANCE.OUT from the in-repo fixtures."""
    from hydrus_research.agronomy.result_parser import parse_balance
    from pathlib import Path
    p = Path("tests/fixtures/soil_sand_drain/reference_out/BALANCE.OUT")
    if not p.exists():
        import pytest
        pytest.skip("fixture not present")
    b = parse_balance(p)
    # On this drainage scenario, water leaves the profile (storage_change < 0)
    # and bottom flux drains (percolation > 0).
    assert b["storage_change_mm"] < 0
    assert b["percolation_mm"] > 0
