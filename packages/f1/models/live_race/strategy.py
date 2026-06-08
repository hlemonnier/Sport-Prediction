"""Extension hooks for future telemetry and strategy modules.

These interfaces are intentionally lightweight in Horizon B v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class TelemetryFeatureAdapter(Protocol):
    """Optional telemetry adapter for future B2 models."""

    def build_lap_features(self, laps: pd.DataFrame) -> pd.DataFrame:
        ...


class StrategyPolicyAdapter(Protocol):
    """Optional strategy adapter for future B3 policy evaluation."""

    def evaluate_actions(self, state_frame: pd.DataFrame) -> pd.DataFrame:
        ...


@dataclass
class NoopTelemetryFeatureAdapter:
    """No-op implementation used in Horizon B v1."""

    def build_lap_features(self, laps: pd.DataFrame) -> pd.DataFrame:
        if laps.empty:
            return pd.DataFrame(index=laps.index)
        return pd.DataFrame(index=laps.index)


@dataclass
class NoopStrategyPolicyAdapter:
    """No-op implementation used in Horizon B v1."""

    def evaluate_actions(self, state_frame: pd.DataFrame) -> pd.DataFrame:
        if state_frame.empty:
            return pd.DataFrame(index=state_frame.index)
        return pd.DataFrame(index=state_frame.index)
