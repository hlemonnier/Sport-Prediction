"""Pre-race prediction entrypoints."""

from __future__ import annotations

from dataclasses import replace

from packages.f1.data.schemas import PredictionConfig, PredictionResult
from packages.f1.orchestration.prediction import run_prediction as run_weekend_prediction


def run_pre_race_prediction(config: PredictionConfig) -> PredictionResult:
    """Run the race prediction flow from official or predicted qualifying context."""

    mode = str(config.mode or "").lower()
    if mode != "race":
        config = replace(config, mode="race")
    return run_weekend_prediction(config)


run_prediction = run_pre_race_prediction

__all__ = ["run_pre_race_prediction", "run_prediction"]
