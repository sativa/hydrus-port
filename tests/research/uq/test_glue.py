import numpy as np
import pytest
from hydrus_research.uq import glue_filter, UQResult
from hydrus_research.batch import BatchResult


def _toy_batch():
    rng = np.random.default_rng(7)
    thetas = rng.uniform(0, 1, size=(16, 2))
    ys = rng.uniform(0.2, 0.4, size=(16, 3))
    return BatchResult(thetas=thetas, ys=ys,
                       wall_s=np.ones(16), converged=np.ones(16, dtype=bool),
                       param_names=["a", "b"], obs_names=["o1", "o2", "o3"],
                       meta={})


def test_glue_filter_returns_uq_result():
    br = _toy_batch()
    obs = np.array([0.31, 0.30, 0.33])
    sigma = np.array([0.05, 0.05, 0.05])
    r = glue_filter(batch_result=br, obs_values=obs, obs_sigmas=sigma,
                    likelihood_cutoff=0.5)
    assert isinstance(r, UQResult)
    assert r.method == "glue"
    assert r.weights is not None
    assert len(r.weights) <= 16     # at most original N (filtered behavioral)


def test_glue_filter_uses_gaussian_likelihood():
    br = _toy_batch()
    obs = br.ys[5]                     # exactly equal to one row
    sigma = np.full(3, 0.01)
    r = glue_filter(batch_result=br, obs_values=obs, obs_sigmas=sigma,
                    likelihood_cutoff=0.0)
    assert len(r.weights) == 16        # no filtering at cutoff=0
    # Row 5 should have highest weight (exact match → max likelihood)
    assert np.argmax(r.weights) == 5
