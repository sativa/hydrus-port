import pytest
try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("FastAPI not installed", allow_module_level=True)
from hydrus_port_server.app import create_app


def _client():
    return TestClient(create_app())


def test_lib_crops_returns_known_ids():
    r = _client().get("/research/agronomy/lib/crops")
    assert r.status_code == 200
    ids = {c["id"] for c in r.json()["crops"]}
    assert "maize" in ids and "wheat_winter" in ids


def test_lib_weather_meta_and_series():
    c = _client()
    r = c.get("/research/agronomy/lib/weather")
    assert r.status_code == 200
    ids = {w["id"] for w in r.json()["weather"]}
    assert "n_china_avg" in ids

    r2 = c.get("/research/agronomy/lib/weather/n_china_avg")
    assert r2.status_code == 200
    assert len(r2.json()["doy"]) == 365


def test_agronomy_run_smoke():
    payload = {
        "crop_id": "maize", "soil_id": "loam", "weather_id": "n_china_avg",
        "horizon_days": 14, "start_year": 2026,
        "irrigation": [{"date": "2026-05-10", "depth_mm": 20}],
        "fertilizer": [],
    }
    r = _client().post("/research/agronomy/run", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "theta_zt" in body
    assert body["water_balance"]["irrig_mm"] == 20
