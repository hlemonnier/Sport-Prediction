"""Football player schema placeholder."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlayerRecord:
    player_id: str
    player_name: str
    team_id: str | None = None
    position: str | None = None
