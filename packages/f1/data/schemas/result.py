"""F1 prediction result schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class PredictionResult:
    version: str
    table: "object"  # pandas DataFrame
    notes: List[str]
    model_name: str
    model_family: str
    device_used: Optional[str]
    dl_available: bool
    candidate_leaderboard: List[dict[str, Any]]
    extras: dict[str, Any] = field(default_factory=dict)
