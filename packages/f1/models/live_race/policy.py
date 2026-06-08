"""Policy adapters for live race strategy evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

import pandas as pd

from packages.f1.models.live_race.action_space import (
    ACTION_PIT_NEXT_LAP,
    ACTION_PIT_NOW,
    ACTION_STAY_OUT,
    StrategyAction,
    normalize_compound,
)
from packages.f1.models.live_race.environment import StrategyState
from packages.f1.models.live_race.planner import DeterministicStrategyPlanner, PlannerResult
from packages.f1.models.live_race.strategy import BaselineStrategyPolicyAdapter, StrategyPolicyAdapter


class LiveStrategyPolicy(Protocol):
    def select_action(self, state: StrategyState) -> StrategyAction:
        ...


@dataclass
class DeterministicBaselinePolicy:
    """Wrap the existing deterministic strategy adapter in the action contract."""

    adapter: StrategyPolicyAdapter = field(default_factory=BaselineStrategyPolicyAdapter)

    def select_action(self, state: StrategyState) -> StrategyAction:
        row = pd.DataFrame([_state_to_adapter_row(state)])
        scored = self.adapter.evaluate_actions(row)
        if scored.empty:
            return StrategyAction(ACTION_STAY_OUT)
        return action_from_strategy_row(scored.iloc[0])


@dataclass
class PlannerPolicy:
    planner: DeterministicStrategyPlanner

    def select_action(self, state: StrategyState) -> StrategyAction:
        return self.plan(state).action

    def plan(self, state: StrategyState) -> PlannerResult:
        return self.planner.plan(state)


def action_from_strategy_row(row: Mapping[str, object] | pd.Series) -> StrategyAction:
    recommended = str(row.get("recommended_action", ACTION_STAY_OUT) or ACTION_STAY_OUT)
    next_compound = normalize_compound(row.get("next_compound", None))
    if recommended == ACTION_STAY_OUT:
        return StrategyAction(ACTION_STAY_OUT)
    if recommended == ACTION_PIT_NEXT_LAP:
        compound = next_compound if next_compound != "UNKNOWN" else "MEDIUM"
        return StrategyAction(ACTION_PIT_NEXT_LAP, compound=compound)
    if recommended == ACTION_PIT_NOW:
        compound = next_compound if next_compound != "UNKNOWN" else "MEDIUM"
        return StrategyAction(ACTION_PIT_NOW, compound=compound)
    return StrategyAction.from_key(recommended)


def _state_to_adapter_row(state: StrategyState) -> dict[str, object]:
    return {
        "driver_id": state.driver_id,
        "compound": state.compound,
        "tyre_age": state.tyre_age,
        "stint_id": state.stint_id,
        "deg_rate_mean": state.deg_rate_mean,
        "pace_penalty_mean": state.pace_penalty_mean,
        "track_status": state.track_status,
        "lap_last": state.lap_number,
        "race_total_laps": state.total_laps,
        "remaining_laps": state.remaining_laps,
        "pit_loss_estimate_seconds": state.pit_loss_estimate_seconds,
        "next_lap_mean": state.next_lap_mean,
        "is_red": state.is_red,
        "is_sc_vsc": state.is_sc_vsc,
        "is_yellow": state.is_yellow,
        "is_greenish": state.is_greenish,
    }


__all__ = [
    "DeterministicBaselinePolicy",
    "LiveStrategyPolicy",
    "PlannerPolicy",
    "action_from_strategy_row",
]
