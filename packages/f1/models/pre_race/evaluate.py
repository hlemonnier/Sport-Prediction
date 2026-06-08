"""Pre-race evaluation entrypoints."""

from __future__ import annotations

from packages.f1.orchestration.backtest import evaluate_prediction_rows


def evaluate_pre_race_predictions(*args: object, **kwargs: object) -> object:
    """Evaluate predicted race order against classified race results."""

    return evaluate_prediction_rows(*args, **kwargs)


__all__ = ["evaluate_pre_race_predictions", "evaluate_prediction_rows"]
