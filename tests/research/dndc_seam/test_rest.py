import json
import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:                  # gui extra not installed
    pytest.skip("FastAPI not installed; install hydrus-port[gui]",
                allow_module_level=True)

from hydrus_port_server.app import build_app


@pytest.fixture(scope="module")
def client():
    return TestClient(build_app())


def test_dndc_validate_ok(client):
    payload = {
        "atm": {"dates": ["2026-05-01"], "precip_cm": [0.0], "pet_cm": [0.4]},
        "et":  {"mode": "lai_beer", "lai": [2.0], "extinction_k": 0.6},
        "root": {"z_max_cm": 50, "growth_curve": "logistic", "days_to_zmax": 30,
                 "density_profile": "linear_decline"},
        "feddes": {"h1": -15, "h2": -30, "h3_high": -325, "h3_low": -600, "h4": -8000,
                   "pet_high_cm_d": 0.5, "pet_low_cm_d": 0.1},
        "n_transform": {"mode": "constant_rates", "k_nitrification_d": 0.1},
        "plant_n_uptake": {"mode": "passive_with_water"},
        "state": {"z_grid_cm": [0.0]},
    }
    r = client.post("/research/dndc/validate", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True


def test_dndc_validate_fails_on_bad_data(client):
    payload = {"atm": {"dates": ["2026-05-01"], "precip_cm": []}}    # length mismatch
    r = client.post("/research/dndc/validate", json=payload)
    assert r.status_code == 422


def test_dndc_crop_presets_lists_15plus(client):
    r = client.get("/research/dndc/crop-presets")
    assert r.status_code == 200
    body = r.json()
    assert "maize" in body and "wheat" in body and "bare_soil" in body
    assert len(body) >= 15
