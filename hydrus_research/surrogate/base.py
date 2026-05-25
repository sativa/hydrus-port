"""SurrogateSimulator — drop-in M0 Simulator backed by a trained model."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from ..simulator.base import Simulator, SimResult, InitialState


class SurrogateModel(ABC):
    """Common interface for all surrogate backends."""
    @abstractmethod
    def fit(self, thetas: np.ndarray, ys: np.ndarray) -> None: ...
    @abstractmethod
    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, stddev) — both shape (M,) where M = n_obs."""
    @abstractmethod
    def save(self, path) -> None: ...
    @classmethod
    @abstractmethod
    def load(cls, path) -> "SurrogateModel": ...


class SurrogateSimulator(Simulator):
    """Wraps a trained SurrogateModel as an M0 Simulator."""
    name: str = "surrogate"
    dimension: int = -1

    def __init__(self, model: SurrogateModel,
                 param_names: list[str], obs_names: list[str]):
        self.model = model
        self.param_names = list(param_names)
        self.obs_names = list(obs_names)

    def run(self, scenario, forcing, ic):
        raise NotImplementedError(
            "SurrogateSimulator.run is not used directly; consumers wire "
            "`forward = lambda theta: surrogate.model.predict(theta)[0]` "
            "into M4/M5/M6 workflows."
        )

    def observable_at(self, result, spec):
        raise NotImplementedError(
            "SurrogateSimulator doesn't produce a full SimResult; observables "
            "are returned directly by model.predict(theta) in obs_names order."
        )
