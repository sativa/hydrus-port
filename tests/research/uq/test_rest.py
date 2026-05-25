import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("FastAPI not installed", allow_module_level=True)

from hydrus_port_server.app import build_app


@pytest.fixture(scope="module")
def client():
    return TestClient(build_app())


def test_glue_endpoint_smoke(client):
    """POST /research/uq/glue with a tiny synthetic BatchResult payload."""
    payload = {
        "thetas": [[0.1], [0.2], [0.3]],
        "ys": [[0.31], [0.32], [0.40]],
        "param_names": ["alpha"],
        "obs_names": ["theta_z30"],
        "obs_values": [0.31],
        "obs_sigmas": [0.05],
        "likelihood_cutoff": 0.1,
    }
    r = client.post("/research/uq/glue", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["method"] == "glue"


def test_unknown_uq_method_404(client):
    r = client.post("/research/uq/quantum", json={})
    assert r.status_code in (404, 422)
