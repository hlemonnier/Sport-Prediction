"""Shared data contracts used across sports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True)
class VenueLocation:
    name: str
    latitude: float
    longitude: float
    timezone: str = "auto"


@dataclass(frozen=True)
class SportEventKey:
    sport: str
    competition: str | None
    season: int | None
    round_number: int | None
    event_id: str
    starts_at: datetime | None = None


@dataclass(frozen=True)
class ModelPrediction:
    model_name: str
    event_key: SportEventKey
    rows: tuple[Mapping[str, Any], ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
