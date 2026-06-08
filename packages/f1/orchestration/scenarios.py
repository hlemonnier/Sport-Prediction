"""F1 weekend scenario contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class F1Scenario:
    name: str
    predicted_grid: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    weather: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = field(default_factory=tuple)
