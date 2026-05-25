import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("FastAPI not installed", allow_module_level=True)

from hydrus_port_server.app import build_app


@pytest.fixture(scope="module")
def client():
    return TestClient(build_app())


def _minimal_payload():
    return {
        "scenario_dir": "tests/fixtures/infiltr_v1/inputs",
        "params": [
            {"name": "alpha", "target": "materials[0].alpha",
             "bounds": [0.005, 0.05], "transform": "log"},
        ],
        "obs": [
            {"name": "theta_z30_d1", "kind": "theta",
             "location": {"z_cm": -30.0}, "time_day": 1.0},
        ],
        "n": 8,                             # tiny — just to confirm the route wiring
        "workers": 1,
        "seed": 42,
    }


def test_morris_endpoint_returns_indices(client):
    r = client.post("/research/sensitivity/morris", json=_minimal_payload())
    # 8 trajectories * (1+1) = 16 forward calls; ~2 minutes on a laptop —
    # the test is slow but verifies the wiring end-to-end.
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["method"] == "morris"
    assert "mu_star" in body["indices"]


def test_unknown_method_404(client):
    r = client.post("/research/sensitivity/cubism", json=_minimal_payload())
    assert r.status_code == 404
