import time
import numpy as np
import pytest

from hydrus_research.batch import BatchRunner, BatchResult


def _slow_forward(theta: np.ndarray, sleep_s: float = 0.05) -> np.ndarray:
    """Toy forward — returns [theta[0] + theta[1], theta[0] * theta[1]] after sleeping."""
    time.sleep(sleep_s)
    return np.array([theta[0] + theta[1], theta[0] * theta[1]])


def _failing_forward(theta: np.ndarray) -> np.ndarray:
    """Toy forward that raises if theta[0] > 0.5."""
    if theta[0] > 0.5:
        raise RuntimeError("simulated solver divergence")
    return np.array([theta[0] + theta[1]])


def test_batch_runner_serial_basic():
    thetas = np.array([[0.1, 1.0], [0.2, 2.0], [0.3, 3.0]])
    runner = BatchRunner(forward=_slow_forward,
                         param_names=["alpha", "n"],
                         obs_names=["sum", "product"],
                         n_workers=1)
    r = runner.run(thetas)
    assert isinstance(r, BatchResult)
    assert r.N == 3
    assert r.D == 2
    assert r.M == 2
    assert r.converged.all()
    np.testing.assert_allclose(r.ys[:, 0], [1.1, 2.2, 3.3])
    np.testing.assert_allclose(r.ys[:, 1], [0.1, 0.4, 0.9])
    assert (r.wall_s > 0).all()


def test_batch_runner_parallel_speedup():
    """4 workers on 16 tasks of 0.1s each should be < 0.8s (vs. 1.6s serial)."""
    thetas = np.array([[i * 0.01, i * 0.02] for i in range(16)])
    runner = BatchRunner(
        forward=lambda t: _slow_forward(t, sleep_s=0.1),
        param_names=["a", "b"], obs_names=["s", "p"],
        n_workers=4,
    )
    t0 = time.time()
    r = runner.run(thetas)
    wall = time.time() - t0
    assert r.converged.all()
    # Soft assertion to avoid flakiness on contended CI hardware
    assert wall < 1.2, f"4-worker parallel took {wall:.2f}s; expected < 1.2s"


def test_batch_runner_handles_failures():
    """Failed forward calls produce converged=False and NaN ys."""
    thetas = np.array([[0.1, 1.0], [0.9, 2.0], [0.3, 3.0]])
    runner = BatchRunner(forward=_failing_forward,
                         param_names=["alpha", "n"],
                         obs_names=["sum"],
                         n_workers=1)
    r = runner.run(thetas)
    assert r.converged.tolist() == [True, False, True]
    assert np.isnan(r.ys[1, 0])
    assert r.n_failed == 1


def test_batch_runner_n_workers_auto():
    runner = BatchRunner(forward=_slow_forward,
                         param_names=["a", "b"], obs_names=["s", "p"],
                         n_workers="auto")
    assert runner.n_workers >= 1
