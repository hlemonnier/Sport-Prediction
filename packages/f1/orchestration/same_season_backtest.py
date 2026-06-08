"""F1 same-season/regime-aware backtest helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from packages.f1.orchestration.backtest import evaluate_prediction_rows


@dataclass(frozen=True)
class SameSeasonBacktestSpec:
    target_season: int
    target_rounds: tuple[int, ...]
    train_only_prior_rounds: bool = True


def same_season_train_rounds(target_round: int) -> list[int]:
    """Return same-season rounds allowed before a target round."""

    return list(range(1, max(1, int(target_round))))


def evaluate_same_season_rows(*args: object, **kwargs: object) -> object:
    """Evaluate same-season prediction rows with the common F1 backtest scorer."""

    return evaluate_prediction_rows(*args, **kwargs)


def build_same_season_specs(target_season: int, rounds: Iterable[int]) -> list[SameSeasonBacktestSpec]:
    return [
        SameSeasonBacktestSpec(target_season=int(target_season), target_rounds=(int(round_number),))
        for round_number in rounds
    ]


__all__ = [
    "SameSeasonBacktestSpec",
    "build_same_season_specs",
    "evaluate_same_season_rows",
    "same_season_train_rounds",
]
