"""Shared backtest result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class BacktestResult:
    model_name: str
    rows: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, float]
    notes: tuple[str, ...] = field(default_factory=tuple)
