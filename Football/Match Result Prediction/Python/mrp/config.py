"""Configuration objects for match result prediction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionConfig:
    league: str
    season: int
    round_number: int
    mode: str
    data_source: str | None = None
    train_seasons: list[int] | None = None
    cache_dir: str | None = None
