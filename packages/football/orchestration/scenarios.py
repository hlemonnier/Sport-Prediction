"""Football fixture scenario contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class FootballScenario:
    name: str
    fixture_overrides: Mapping[str, Any] = field(default_factory=dict)
    weather: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = field(default_factory=tuple)
