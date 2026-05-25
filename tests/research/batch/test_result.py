import numpy as np
import pytest
from pathlib import Path

from hydrus_research.batch import BatchResult


def _make_result(N=4, D=3, M=2):
    rng = np.random.default_rng(42)
    return BatchResult(
        thetas=rng.uniform(size=(N, D)),
        ys=rng.uniform(size=(N, M)),
        wall_s=rng.uniform(0.1, 1.0, size=N),
        converged=np.ones(N, dtype=bool),
        param_names=["alpha", "n", "Ks"][:D],
        obs_names=[f"obs_{i}" for i in range(M)],
        meta={"simulator": "test_fake", "n_workers": 1},
    )


def test_batch_result_shape_invariants():
    r = _make_result(N=5, D=3, M=2)
    assert r.N == 5
    assert r.D == 3
    assert r.M == 2
    assert r.thetas.shape == (5, 3)
    assert r.ys.shape == (5, 2)
    assert r.wall_s.shape == (5,)


def test_batch_result_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        BatchResult(
            thetas=np.zeros((4, 3)),
            ys=np.zeros((3, 2)),                # mismatched N
            wall_s=np.zeros(4),
            converged=np.ones(4, dtype=bool),
            param_names=["a", "b", "c"],
            obs_names=["x", "y"],
            meta={},
        )


def test_batch_result_parquet_round_trip(tmp_path):
    r = _make_result(N=6, D=2, M=3)
    p = tmp_path / "sweep.parquet"
    r.to_parquet(p)
    assert p.exists()
    r2 = BatchResult.from_parquet(p)
    np.testing.assert_array_equal(r.thetas, r2.thetas)
    np.testing.assert_array_equal(r.ys, r2.ys)
    np.testing.assert_array_equal(r.wall_s, r2.wall_s)
    np.testing.assert_array_equal(r.converged, r2.converged)
    assert r2.param_names == r.param_names
    assert r2.obs_names == r.obs_names
    assert r2.meta["simulator"] == "test_fake"


def test_batch_result_handles_failed_runs():
    """When a forward call fails, converged=False and ys row is NaN."""
    r = BatchResult(
        thetas=np.array([[0.1, 1.5, 5.0], [0.2, 2.0, 10.0]]),
        ys=np.array([[0.31, 0.28], [np.nan, np.nan]]),
        wall_s=np.array([0.5, 0.0]),
        converged=np.array([True, False]),
        param_names=["alpha", "n", "Ks"],
        obs_names=["theta_z10", "theta_z20"],
        meta={},
    )
    # Convenience selectors
    assert r.n_converged == 1
    assert r.n_failed == 1
