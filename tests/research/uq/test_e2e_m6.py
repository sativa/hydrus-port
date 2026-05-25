"""M6 acceptance: GLUE on a tiny synthetic batch."""
import numpy as np
from hydrus_research.batch import BatchResult
from hydrus_research.uq import glue_filter


def test_glue_on_synthetic_batch():
    rng = np.random.default_rng(42)
    thetas = rng.uniform(0, 1, size=(20, 3))
    # ys mimic θ(z=-30, t=1) — clustered around 0.30 with random noise
    ys = 0.30 + rng.normal(0, 0.05, size=(20, 1))
    br = BatchResult(thetas=thetas, ys=ys,
                     wall_s=np.zeros(20), converged=np.ones(20, dtype=bool),
                     param_names=["a","b","c"], obs_names=["theta"], meta={})
    r = glue_filter(br, obs_values=np.array([0.30]),
                    obs_sigmas=np.array([0.05]),
                    likelihood_cutoff=0.3)
    assert r.n_samples > 0
    assert r.n_samples <= 20
    assert abs(sum(r.weights) - 1.0) < 1e-6
    assert r.quantiles["p50"][0] == pytest.approx(0.30, abs=0.1)


import pytest  # noqa
