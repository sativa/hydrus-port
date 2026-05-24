import pytest
try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("FastAPI not installed", allow_module_level=True)

from hydrus_port_server.app import build_app


@pytest.fixture(scope="module")
def client():
    return TestClient(build_app())


def test_ptf_usda_classes(client):
    r = client.get("/research/ptf/usda-classes")
    assert r.status_code == 200
    body = r.json()
    assert "loam" in body and "clay" in body and "sand" in body
    assert len(body) == 12
    assert "sand_pct" in body["loam"]


def test_ptf_predict_carsel_parrish(client):
    payload = {"sand_pct": 40, "silt_pct": 40, "clay_pct": 20,
               "method": "carsel_parrish"}
    r = client.post("/research/ptf/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["method"] == "carsel_parrish"
    assert 0.02 < body["alpha"] < 0.05    # loam range


def test_ptf_predict_invalid_texture(client):
    payload = {"sand_pct": 50, "silt_pct": 50, "clay_pct": 50,
               "method": "carsel_parrish"}   # sums to 150
    r = client.post("/research/ptf/predict", json=payload)
    assert r.status_code in (400, 422)
