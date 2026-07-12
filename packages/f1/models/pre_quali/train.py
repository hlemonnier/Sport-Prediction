"""Pre-qualifying training entrypoints."""

from __future__ import annotations

import pandas as pd

from packages.f1.models.pre_quali.classification import (
    QualifyingStageProbabilityModel,
    StageProbabilityConfig,
    fit_qualifying_stage_probability_model,
)
from packages.f1.models.pre_quali.pairwise import (
    PairwiseQualifyingRanker,
    PairwiseRankerConfig,
    fit_pairwise_qualifying_ranker,
)
from packages.f1.models.training import train_model


def train_pre_quali_model(*args: object, **kwargs: object) -> object:
    """Train the shared rank model for qualifying targets."""

    return train_model(*args, **kwargs)


def train_pre_quali_pairwise_ranker(
    history: pd.DataFrame,
    *,
    config: PairwiseRankerConfig,
    target_event_key: int | None = None,
) -> PairwiseQualifyingRanker:
    """Train the event-pure Qualifying rank challenger."""

    return fit_pairwise_qualifying_ranker(
        history,
        config=config,
        target_event_key=target_event_key,
    )


def train_pre_quali_stage_model(
    history: pd.DataFrame,
    *,
    config: StageProbabilityConfig,
    target_event_key: int | None = None,
) -> QualifyingStageProbabilityModel:
    """Train separate valid-lap, Q2, and Q3 conditional probability models."""

    return fit_qualifying_stage_probability_model(
        history,
        config=config,
        target_event_key=target_event_key,
    )


__all__ = [
    "train_pre_quali_model",
    "train_pre_quali_pairwise_ranker",
    "train_pre_quali_stage_model",
    "train_model",
]
