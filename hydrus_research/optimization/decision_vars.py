"""Encode irrigation/fert event schedules as flat θ vectors for optimizers.

Convention: each event contributes (amount, day_offset) to the vector;
events are ordered by their slot in the schedule. Optimizers see a
flat 2N-vector for an N-event schedule."""
from __future__ import annotations
import numpy as np


def encode_schedule(events: list[dict]) -> np.ndarray:
    """events: list of {"amount": float, "day": float}; returns flat 2N array."""
    out = np.empty(2 * len(events), dtype=float)
    for i, e in enumerate(events):
        out[2*i] = float(e["amount"])
        out[2*i + 1] = float(e["day"])
    return out


def decode_schedule(theta: np.ndarray) -> list[dict]:
    if theta.shape[0] % 2 != 0:
        raise ValueError("theta length must be even (amount, day pairs)")
    n = theta.shape[0] // 2
    return [{"amount": float(theta[2*i]), "day": float(theta[2*i + 1])}
            for i in range(n)]
