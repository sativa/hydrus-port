import pytest
try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("FastAPI not installed", allow_module_level=True)
from hydrus_port_server.app import build_app


def test_surrogate_train_route_smoke(tmp_path):
    # Pre-build a tiny parquet
    import numpy as np
    from hydrus_research.batch import BatchResult
    br = BatchResult(thetas=np.random.uniform(0, 1, size=(8, 2)),
                     ys=np.random.uniform(0, 1, size=(8, 1)),
                     wall_s=np.zeros(8), converged=np.ones(8, dtype=bool),
                     param_names=["a","b"], obs_names=["o"], meta={})
    p = tmp_path / "br.parquet"
    br.to_parquet(p)
    client = TestClient(build_app())
    r = client.post("/research/surrogate/train",
                    json={"batch_parquet": str(p), "type": "gp"})
    assert r.status_code == 200
    body = r.json()
    assert "model_id" in body
