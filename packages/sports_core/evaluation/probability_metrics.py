"""Probability scoring helpers."""

from __future__ import annotations

from collections.abc import Iterable
import math


def brier_score(probabilities: Iterable[float], outcomes: Iterable[float]) -> float:
    pairs = [(float(p), float(y)) for p, y in zip(probabilities, outcomes)]
    if not pairs:
        return 0.0
    return sum((p - y) ** 2 for p, y in pairs) / float(len(pairs))


def log_loss(probabilities: Iterable[float], outcomes: Iterable[float], eps: float = 1e-15) -> float:
    pairs = [(float(p), float(y)) for p, y in zip(probabilities, outcomes)]
    if not pairs:
        return 0.0
    clipped = [(min(1.0 - eps, max(eps, p)), y) for p, y in pairs]
    return -sum(y * math.log(p) + (1.0 - y) * math.log(1.0 - p) for p, y in clipped) / float(len(clipped))
