import numpy as np
from hydrus_research.surrogate.metrics import nse, rmse, coverage


def test_nse_perfect():
    assert nse([1, 2, 3], [1, 2, 3]) == 1.0


def test_rmse_zero():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0


def test_coverage_one_when_all_inside():
    assert coverage([1, 2, 3], [10, 10, 10], [1, 2, 3]) == 1.0
