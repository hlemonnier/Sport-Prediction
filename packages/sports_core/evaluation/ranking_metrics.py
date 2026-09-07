"""Ranking metrics shared by F1 and future ordered-outcome models."""

from __future__ import annotations

from collections.abc import Sequence


def mean_absolute_rank_error(predicted_order: Sequence[str], actual_order: Sequence[str]) -> float:
    if not predicted_order or not actual_order:
        raise ValueError("rank error requires nonempty complete orders")
    if len(set(predicted_order)) != len(predicted_order) or len(set(actual_order)) != len(actual_order):
        raise ValueError("orders must not contain duplicate participants")
    if set(predicted_order) != set(actual_order):
        raise ValueError("rank error requires the same complete participant field")
    actual_rank = {item: idx for idx, item in enumerate(actual_order)}
    errors = [
        abs(idx - actual_rank[item])
        for idx, item in enumerate(predicted_order)
    ]
    return sum(errors) / float(len(errors))
