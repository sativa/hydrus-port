import numpy as np
import pytest
from hydrus_research.surrogate import SurrogateSimulator, SurrogateModel


def test_surrogate_simulator_is_simulator():
    from hydrus_research.simulator.base import Simulator
    class _Dummy(SurrogateModel):
        def fit(self, t, y): pass
        def predict(self, t): return (np.zeros(1), np.zeros(1))
        def save(self, p): pass
        @classmethod
        def load(cls, p): return cls()
    s = SurrogateSimulator(_Dummy(), param_names=["a"], obs_names=["o"])
    assert isinstance(s, Simulator)
    assert s.dimension == -1
