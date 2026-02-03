"""Configuration objects for predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PredictionConfig:
    source: str
    mode: str
    year: int
    round_number: int
    train_seasons: List[int]
    include_standings: bool
    cache_dir: Optional[str]
    meeting_name: Optional[str]
    country_name: Optional[str]


@dataclass
class PredictionResult:
    version: str
    table: "object"  # pandas DataFrame
    notes: List[str]
