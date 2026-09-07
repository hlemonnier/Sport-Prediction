"""Probability calibration diagnostics."""

from __future__ import annotations

from collections.abc import Iterable

from .validation import binary_pairs


def expected_calibration_error(
    probabilities: Iterable[float],
    outcomes: Iterable[float],
    *,
    bins: int = 10,
) -> float:
    pairs = binary_pairs(probabilities, outcomes)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    bucket_count = bins
    total = len(pairs)
    error = 0.0
    for idx in range(bucket_count):
        lower = idx / bucket_count
        upper = (idx + 1) / bucket_count
        bucket = [pair for pair in pairs if lower <= pair[0] < upper or (idx == bucket_count - 1 and pair[0] == 1.0)]
        if not bucket:
            continue
        confidence = sum(pair[0] for pair in bucket) / len(bucket)
        accuracy = sum(pair[1] for pair in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(confidence - accuracy)
    return error
