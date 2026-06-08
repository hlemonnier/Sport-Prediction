"""Pre-qualifying evaluation entrypoints."""

from __future__ import annotations

from packages.f1.orchestration.backtest import evaluate_prediction_rows


def evaluate_pre_quali_predictions(*args: object, **kwargs: object) -> object:
    """Evaluate predicted qualifying order against actual session results."""

    return evaluate_prediction_rows(*args, **kwargs)


__all__ = ["evaluate_pre_quali_predictions", "evaluate_prediction_rows"]
