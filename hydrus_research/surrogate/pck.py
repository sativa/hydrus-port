"""PC-Kriging surrogate (stub — replaced in Task 3)."""
from .base import SurrogateModel


class PCKSurrogate(SurrogateModel):
    def fit(self, thetas, ys): raise NotImplementedError
    def predict(self, theta): raise NotImplementedError
    def save(self, path): raise NotImplementedError
    @classmethod
    def load(cls, path): raise NotImplementedError
