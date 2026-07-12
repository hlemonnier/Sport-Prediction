"""Pre-race training entrypoints."""

from __future__ import annotations

from packages.f1.models.training import train_model
from packages.f1.models.pre_race.joint import SurvivalAwareRaceModel


def train_pre_race_model(*args: object, **kwargs: object) -> object:
    """Train the shared rank model for race-order targets."""

    return train_model(*args, **kwargs)


def train_survival_aware_race_model(
    history: object,
    **kwargs: object,
) -> SurvivalAwareRaceModel:
    """Fit the reason-coded terminal and conditional-order factors."""

    model = SurvivalAwareRaceModel()
    return model.fit(history, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "train_model",
    "train_pre_race_model",
    "train_survival_aware_race_model",
]
