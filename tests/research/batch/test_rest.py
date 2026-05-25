"""Tests for /research/batch/{start,status,result} REST endpoints (M3.7)."""
import time
import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("FastAPI not installed", allow_module_level=True)

from hydrus_port_server.app import build_app


@pytest.fixture(scope="module")
def client():
    return TestClient(build_app())


def test_batch_start_returns_job_id(client):
    payload = {
        "scenario_dir": "tests/fixtures/infiltr_v1/inputs",
        "params": [{"name": "alpha", "target": "materials[0].alpha",
                    "bounds": [0.005, 0.05], "transform": "log"}],
        "obs": [{"name": "theta_z30_d1", "kind": "theta",
                 "location": {"z_cm": -30.0}, "time_day": 1.0}],
        "n": 2,
        "sampler": "lhs",
        "workers": 1,
        "seed": 42,
    }
    r = client.post("/research/batch/start", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body


def test_batch_status_progresses_to_done(client):
    payload = {
        "scenario_dir": "tests/fixtures/infiltr_v1/inputs",
        "params": [{"name": "alpha", "target": "materials[0].alpha",
                    "bounds": [0.005, 0.05], "transform": "log"}],
        "obs": [{"name": "theta", "kind": "theta",
                 "location": {"z_cm": -30.0}, "time_day": 1.0}],
        "n": 2, "sampler": "lhs", "workers": 1, "seed": 7,
    }
    r = client.post("/research/batch/start", json=payload)
    job_id = r.json()["job_id"]
    # Poll for completion (test fixture has 2 runs * ~10s each = up to 30s)
    for _ in range(60):
        s = client.get(f"/research/batch/{job_id}/status")
        assert s.status_code == 200
        if s.json()["state"] == "done":
            break
        time.sleep(1)
    else:
        pytest.fail("batch job did not complete within 60 seconds")

    res = client.get(f"/research/batch/{job_id}/result")
    assert res.status_code == 200
    # res.content is parquet bytes; verify it round-trips
    import io, tempfile, pathlib
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        f.write(res.content)
        f.flush()
        from hydrus_research.batch import BatchResult
        br = BatchResult.from_parquet(pathlib.Path(f.name))
    assert br.N == 2
