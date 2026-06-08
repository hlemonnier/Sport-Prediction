"""Ranking metrics shared by F1 and future ordered-outcome models."""

from __future__ import annotations

from collections.abc import Sequence


def mean_absolute_rank_error(predicted_order: Sequence[str], actual_order: Sequence[str]) -> float:
    actual_rank = {item: idx for idx, item in enumerate(actual_order)}
    errors = [
        abs(idx - actual_rank[item])
        for idx, item in enumerate(predicted_order)
        if item in actual_rank
    ]
    if not errors:
        return 0.0
    return sum(errors) / float(len(errors))
