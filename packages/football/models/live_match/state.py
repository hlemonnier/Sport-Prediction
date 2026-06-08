"""Football live-match state contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class LiveMatchState:
    match_id: str
    minute: int
    score_home: int = 0
    score_away: int = 0
    events: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
