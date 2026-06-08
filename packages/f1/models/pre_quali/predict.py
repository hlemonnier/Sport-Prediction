"""Pre-qualifying prediction entrypoints."""

from __future__ import annotations

from dataclasses import replace

from packages.f1.data.schemas import PredictionConfig, PredictionResult
from packages.f1.orchestration.prediction import run_prediction as run_weekend_prediction


def run_pre_quali_prediction(config: PredictionConfig) -> PredictionResult:
    """Run the qualifying-focused prediction flow."""

    mode = str(config.mode or "").lower()
    if mode != "qualifying":
        config = replace(config, mode="qualifying")
    return run_weekend_prediction(config)


run_prediction = run_pre_quali_prediction

__all__ = ["run_pre_quali_prediction", "run_prediction"]
