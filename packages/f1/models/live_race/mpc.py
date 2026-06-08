"""MPC-style live strategy replanning wrappers.

The legacy wrapper can replan with any planner-like object and defaults to the
deterministic DP baseline. Phase 6 adds a simulator-backed wrapper that replans
with scenario-scored legal action sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Protocol

from packages.f1.models.live_race.environment import StrategyState
from packages.f1.models.live_race.planner import DeterministicStrategyPlanner, PlannerResult, SimulatorMPCPlanner


class PlannerLike(Protocol):
    def plan(self, state: StrategyState) -> PlannerResult:
        ...


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

    planner: PlannerLike = field(default_factory=DeterministicStrategyPlanner)

    def replan(self, state: StrategyState | Mapping[str, object]) -> PlannerResult:
        strategy_state = state if isinstance(state, StrategyState) else StrategyState.from_mapping(state)
        return self.planner.plan(strategy_state)

    def replan_many(self, states: Iterable[StrategyState | Mapping[str, object]]) -> tuple[MPCReplanStep, ...]:
        steps: list[MPCReplanStep] = []
        for state in states:
            strategy_state = state if isinstance(state, StrategyState) else StrategyState.from_mapping(state)
            steps.append(MPCReplanStep(state=strategy_state, result=self.planner.plan(strategy_state)))
        return tuple(steps)


@dataclass
class SimulatorMPCStrategyPlanner:
    """Replan over live states with the Phase 6 simulator-backed MPC planner."""

    planner: SimulatorMPCPlanner = field(default_factory=SimulatorMPCPlanner)

    def replan(self, state: StrategyState | Mapping[str, object]) -> PlannerResult:
        strategy_state = state if isinstance(state, StrategyState) else StrategyState.from_mapping(state)
        return self.planner.plan(strategy_state)

    def replan_many(self, states: Iterable[StrategyState | Mapping[str, object]]) -> tuple[MPCReplanStep, ...]:
        steps: list[MPCReplanStep] = []
        for state in states:
            strategy_state = state if isinstance(state, StrategyState) else StrategyState.from_mapping(state)
            steps.append(MPCReplanStep(state=strategy_state, result=self.planner.plan(strategy_state)))
        return tuple(steps)


__all__ = ["MPCReplanStep", "MPCStrategyPlanner", "PlannerLike", "SimulatorMPCStrategyPlanner"]
