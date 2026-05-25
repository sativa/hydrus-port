"""scikit-learn Gaussian Process surrogate (stub — replaced in Task 2)."""
from .base import SurrogateModel


class SklearnGPSurrogate(SurrogateModel):
    def fit(self, thetas, ys): raise NotImplementedError
    def predict(self, theta): raise NotImplementedError
    def save(self, path): raise NotImplementedError
    @classmethod
    def load(cls, path): raise NotImplementedError
