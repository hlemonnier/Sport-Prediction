"""Pre-qualifying prediction entrypoints."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from packages.f1.data.schemas import PredictionConfig, PredictionResult
from packages.f1.models.pre_quali.classification import QualifyingStageProbabilityModel
from packages.f1.models.pre_quali.pairwise import (
    PairwiseQualifyingRanker,
    QualifyingRankingForecast,
)
from packages.f1.orchestration.prediction import run_prediction as run_weekend_prediction
from packages.f1.models.ultimate_lap_time.achievable import (
    AchievableBestLapModel,
    SharedQualifyingForecast,
)


def run_pre_quali_prediction(config: PredictionConfig) -> PredictionResult:
    """Run the qualifying-focused prediction flow."""

    mode = str(config.mode or "").lower()
    if mode != "qualifying":
        config = replace(config, mode="qualifying")
    return run_weekend_prediction(config)


run_prediction = run_pre_quali_prediction


def predict_pre_quali_pairwise_event(
    model: PairwiseQualifyingRanker,
    frame: pd.DataFrame,
    *,
    samples: int = 2000,
    temperature: float = 1.0,
    seed: int = 42,
) -> QualifyingRankingForecast:
    """Rank one complete event field with legal joint outputs."""

    return model.predict_event(
        frame,
        samples=samples,
        temperature=temperature,
        seed=seed,
    )


def predict_pre_quali_stage_probabilities(
    model: QualifyingStageProbabilityModel,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Predict valid-lap and conditional Q2/Q3 probabilities for one field."""

    return model.predict_event(frame)


def predict_shared_qualifying_event(
    model: AchievableBestLapModel,
    frame: pd.DataFrame,
    *,
    samples: int = 5_000,
    seed: int = 0,
) -> SharedQualifyingForecast:
    """Generate Best-Lap marginals and official order from one fitted model."""

    return model.predict_qualifying(frame, samples=samples, seed=seed)


__all__ = [
    "predict_pre_quali_pairwise_event",
    "predict_pre_quali_stage_probabilities",
    "predict_shared_qualifying_event",
    "run_pre_quali_prediction",
    "run_prediction",
]
