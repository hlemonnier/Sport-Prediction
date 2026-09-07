"""Input contracts for metrics: absent evidence must never score as perfect."""

from __future__ import annotations

from collections.abc import Iterable
import math


def binary_pairs(probabilities: Iterable[float], outcomes: Iterable[float]) -> list[tuple[float, float]]:
    """Materialize both inputs before matching, refusing silent truncation."""

    try:
        predicted = [float(value) for value in probabilities]
        observed = [float(value) for value in outcomes]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("probabilities and outcomes must be numeric") from exc
    if not predicted or not observed:
        raise ValueError("metrics require nonempty predictions and observed outcomes")
    if len(predicted) != len(observed):
        raise ValueError("predictions and outcomes must have equal lengths")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in predicted):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if any(not math.isfinite(value) or value not in (0.0, 1.0) for value in observed):
        raise ValueError("binary outcomes must be observed zeros or ones")
    return list(zip(predicted, observed))
