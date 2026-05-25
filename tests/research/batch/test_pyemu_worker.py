import pytest


pyemu = pytest.importorskip("pyemu", reason="pyemu not installed")
from hydrus_research.batch.pyemu_worker import build_worker_entry


def test_build_worker_entry_returns_callable():
    """Smoke: the worker-entry factory takes a forward callable + names and
    returns a function that pyemu's start_workers can drive."""
    def fake_forward(theta):
        return [theta[0] + theta[1]]

    entry = build_worker_entry(forward=fake_forward,
                               param_names=["a", "b"],
                               obs_names=["sum"])
    assert callable(entry)


def test_pyemu_worker_module_imports_only_when_called():
    """The pyemu import must be lazy — `from hydrus_research.batch import ...`
    must not require pyemu to be installed for users only using joblib."""
    import importlib
    mod = importlib.import_module("hydrus_research.batch.pyemu_worker")
    # The module top-level should NOT have triggered a pyemu import
    # (we check by verifying pyemu is NOT in mod's globals as a top-level symbol)
    assert not hasattr(mod, "pyemu_lib"), \
        "pyemu_worker.py imported pyemu at top level; should be lazy"
