"""MPC-style live strategy replanning wrapper.

The heavy calibrated race simulator belongs to a later roadmap phase.  This
wrapper uses the deterministic DP planner as the inner optimizer and replans at
each supplied live state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from packages.f1.models.live_race.environment import StrategyState
from packages.f1.models.live_race.planner import DeterministicStrategyPlanner, PlannerResult


@dataclass
class MPCReplanStep:
    state: StrategyState
    result: PlannerResult

    @property
    def action_key(self) -> str:
        return self.result.action.key


@dataclass
class MPCStrategyPlanner:
    """Replan over a sequence of live states using the deterministic planner."""

    planner: DeterministicStrategyPlanner = field(default_factory=DeterministicStrategyPlanner)

    def replan(self, state: StrategyState | Mapping[str, object]) -> PlannerResult:
        strategy_state = state if isinstance(state, StrategyState) else StrategyState.from_mapping(state)
        return self.planner.plan(strategy_state)

    def replan_many(self, states: Iterable[StrategyState | Mapping[str, object]]) -> tuple[MPCReplanStep, ...]:
        steps: list[MPCReplanStep] = []
        for state in states:
            strategy_state = state if isinstance(state, StrategyState) else StrategyState.from_mapping(state)
            steps.append(MPCReplanStep(state=strategy_state, result=self.planner.plan(strategy_state)))
        return tuple(steps)


__all__ = ["MPCReplanStep", "MPCStrategyPlanner"]
