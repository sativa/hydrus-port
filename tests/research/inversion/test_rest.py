import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("FastAPI not installed", allow_module_level=True)

from hydrus_port_server.app import build_app


@pytest.fixture(scope="module")
def client():
    return TestClient(build_app())


def test_lm_endpoint_synthetic(client):
    payload = {
        "scenario_dir": "tests/fixtures/infiltr_v1/inputs",
        "params": [
            {"name": "alpha", "target": "materials[0].alpha",
             "bounds": [0.005, 0.05], "transform": "log"},
        ],
        "obs_inline": {
            "specs": [{"name": "theta_z30_d1", "kind": "theta",
                       "location": {"z_cm": -30.0}, "time_day": 1.0}],
            "values": [0.31],
            "sigmas": [0.02],
        },
        "max_nfev": 10,                  # tight cap for CI
    }
    r = client.post("/research/inversion/lm", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["backend"] == "lm_scipy"
    assert "alpha" in body["best_params"]


def test_unknown_backend_404(client):
    r = client.post("/research/inversion/quantum", json={})
    assert r.status_code in (404, 422)
