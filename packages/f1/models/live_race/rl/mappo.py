"""MAPPO-style centralized-training/decentralized-execution policy.

This is intentionally pragmatic: centralized training evaluates deterministic
staggered schedules in the multi-agent simulator, then exports a lightweight
masked policy that can select actions from a single car's public observation.
It gives Phase 8 the MAPPO shape without pretending the repo has a full neural
RL stack or enough online data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

from packages.f1.models.live_race.action_space import ACTION_PIT_NOW, ACTION_STAY_OUT, StrategyAction
from packages.f1.models.live_race.environment import StrategyState
from packages.f1.models.live_race.rl.multi_agent_env import (
    MultiAgentLiveRaceEnv,
    MultiAgentRaceState,
    build_traffic_heavy_scenario,
)


@dataclass(frozen=True)
class MAPPOConfig:
    """Configuration for the dependency-light MAPPO-style learner."""

    model_id: str = "mappo_style_staggered_strategy_v1"
    pit_compound: str = "HARD"
    candidate_pit_laps: tuple[int, ...] = ()
    max_stagger_laps: int = 3
    rollout_horizon_laps: Optional[int] = None
    max_same_lap_pit_fraction: float = 0.34
    recent_pit_fraction_cap: float = 0.34
    overdue_lap_grace: int = 1
    tyre_age_panic_threshold: int = 28
    pit_target_lap_fallback: int = 6
    schedule_regularization_seconds: float = 0.10


@dataclass
class MAPPOStylePolicy:
    """Masked policy trained with centralized schedule scoring."""

    pit_schedule_by_driver: dict[str, int]
    config: MAPPOConfig = field(default_factory=MAPPOConfig)
    action_space: tuple[StrategyAction, ...] = field(default_factory=tuple)
    training_diagnostics: dict[str, object] = field(default_factory=dict)
    model_id: str = "mappo_style_staggered_strategy_v1"

    @property
    def diagnostics(self) -> dict[str, object]:
        return {
            **self.training_diagnostics,
            "policy_family": "mappo_style_tabular_ctde",
            "centralized_training": True,
            "decentralized_execution": True,
            "legal_mask_aware": True,
            "neural_stack_required": False,
        }

    def select_action(self, state: StrategyState) -> StrategyAction:
        """Select from one car's public observation only."""

        if "HARD" in set(state.used_compounds) or (state.remaining_laps or 0) <= 2:
            return StrategyAction(ACTION_STAY_OUT)
        target_lap = int(
            self.pit_schedule_by_driver.get(
                str(state.driver_id),
                int(state.metadata.get("preferred_pit_lap", self.config.pit_target_lap_fallback)),
            )
        )
        recent_pit_fraction = _finite(state.metadata.get("multi_agent_recent_pit_fraction"), 0.0)
        is_overdue = int(state.lap_number) >= target_lap + int(self.config.overdue_lap_grace)
        panic = int(state.tyre_age) >= int(self.config.tyre_age_panic_threshold)
        if int(state.lap_number) >= target_lap and (
            recent_pit_fraction <= float(self.config.recent_pit_fraction_cap) or is_overdue or panic
        ):
            return StrategyAction(ACTION_PIT_NOW, compound=self.config.pit_compound)
        return StrategyAction(ACTION_STAY_OUT)

    def select_actions(self, state: MultiAgentRaceState) -> tuple[StrategyAction, ...]:
        """Convenience batch execution with a public same-lap cap."""

        proposed = [self.select_action(car) for car in state.cars]
        pit_indices = [idx for idx, action in enumerate(proposed) if action.action_type == ACTION_PIT_NOW]
        cap = max(1, int(np.floor(float(state.car_count) * float(self.config.max_same_lap_pit_fraction))))
        if len(pit_indices) <= cap:
            return tuple(proposed)
        scored = sorted(
            pit_indices,
            key=lambda idx: (
                int(state.cars[idx].lap_number) - int(self.pit_schedule_by_driver.get(str(state.cars[idx].driver_id), 0)),
                int(state.cars[idx].tyre_age),
                -idx,
            ),
            reverse=True,
        )
        allowed = set(scored[:cap])
        return tuple(action if idx in allowed or action.action_type != ACTION_PIT_NOW else StrategyAction(ACTION_STAY_OUT) for idx, action in enumerate(proposed))

    def plan(self, state: StrategyState | MultiAgentRaceState) -> "MAPPOPlan":
        """Return a plan-style payload for planner/simulator adapters."""

        if isinstance(state, MultiAgentRaceState):
            actions = self.select_actions(state)
            scores = self.centralized_action_scores(state)
            return MAPPOPlan(
                action=actions[0],
                actions=actions,
                value=float(sum(max(driver_scores.values()) for driver_scores in scores.values())),
                diagnostics={
                    "planner": self.config.model_id,
                    "centralized_training": True,
                    "decentralized_execution": True,
                    "centralized_scores": scores,
                },
            )
        action = self.select_action(state)
        return MAPPOPlan(
            action=action,
            actions=(action,),
            value=0.0,
            diagnostics={
                "planner": self.config.model_id,
                "centralized_training": True,
                "decentralized_execution": True,
            },
        )

    def centralized_action_scores(self, state: MultiAgentRaceState) -> dict[str, dict[str, float]]:
        """Expose centralized action scores used for diagnostics and audits."""

        scheduled_counts: dict[int, int] = {}
        for lap in self.pit_schedule_by_driver.values():
            scheduled_counts[int(lap)] = int(scheduled_counts.get(int(lap), 0) + 1)
        scores: dict[str, dict[str, float]] = {}
        for car in state.cars:
            target_lap = int(self.pit_schedule_by_driver.get(str(car.driver_id), self.config.pit_target_lap_fallback))
            tyre_pressure = float(car.tyre_age) * _finite(car.deg_rate_mean, 0.04)
            traffic_pressure = _finite(car.metadata.get("multi_agent_traffic_pressure"), 0.0)
            schedule_bonus = 3.0 if int(car.lap_number) >= target_lap else -1.0 * float(target_lap - int(car.lap_number))
            sync_risk = float(scheduled_counts.get(int(car.lap_number), 0))
            pit_score = (2.2 * tyre_pressure) + (0.8 * traffic_pressure) + schedule_bonus - (1.4 * sync_risk)
            stay_score = -(1.5 * tyre_pressure) - (0.6 * traffic_pressure)
            if "HARD" in set(car.used_compounds) or (car.remaining_laps or 0) <= 2:
                pit_score = -1e6
            scores[str(car.driver_id)] = {
                StrategyAction(ACTION_STAY_OUT).key: float(stay_score),
                StrategyAction(ACTION_PIT_NOW, compound=self.config.pit_compound).key: float(pit_score),
            }
        return scores


@dataclass(frozen=True)
class MAPPOPlan:
    action: StrategyAction
    actions: tuple[StrategyAction, ...]
    value: float
    diagnostics: Mapping[str, object] = field(default_factory=dict)


def fit_mappo_style_policy(
    env: MultiAgentLiveRaceEnv | MultiAgentRaceState | None = None,
    start_states: MultiAgentRaceState | Sequence[MultiAgentRaceState] | None = None,
    *,
    config: MAPPOConfig | None = None,
) -> MAPPOStylePolicy:
    """Fit a deterministic centralized schedule critic and return a policy."""

    if isinstance(env, MultiAgentRaceState):
        start_states = env
        env = None
    race_env = env if isinstance(env, MultiAgentLiveRaceEnv) else MultiAgentLiveRaceEnv()
    cfg = config or MAPPOConfig()
    states = _normalise_start_states(start_states)
    if not states:
        states = (build_traffic_heavy_scenario(seed=int(race_env.config.seed)),)

    candidates: list[dict[str, int]] = []
    for state in states:
        for base_lap in _candidate_laps(state, cfg):
            for width in range(1, max(1, int(cfg.max_stagger_laps)) + 1):
                for rotation in range(width):
                    schedule = {
                        str(car.driver_id): int(base_lap + ((idx + rotation) % width))
                        for idx, car in enumerate(state.cars)
                    }
                    candidates.append(schedule)
    unique_candidates = _unique_schedules(candidates)
    if not unique_candidates:
        unique_candidates = [{str(car.driver_id): int(car.lap_number + 1) for car in states[0].cars}]

    scored: list[tuple[float, dict[str, int], dict[str, object]]] = []
    for schedule in unique_candidates:
        policy = _FixedSchedulePolicy(schedule_by_driver=schedule, pit_compound=cfg.pit_compound)
        rollouts = [
            race_env.rollout(state, policy=policy, max_laps=cfg.rollout_horizon_laps, policy_name="centralized_candidate")
            for state in states
        ]
        team_time = float(np.mean([rollout.team_time_seconds for rollout in rollouts]))
        sync_penalty = float(np.mean([rollout.coordination_penalty_seconds for rollout in rollouts]))
        max_sync = float(np.mean([rollout.max_same_lap_pit_count for rollout in rollouts]))
        late_penalty = _schedule_regularization(schedule, states, cfg)
        value = -team_time - sync_penalty - late_penalty
        scored.append(
            (
                float(value),
                schedule,
                {
                    "mean_team_time_seconds": float(team_time),
                    "mean_coordination_penalty_seconds": float(sync_penalty),
                    "mean_max_same_lap_pit_count": float(max_sync),
                    "schedule_regularization_seconds": float(late_penalty),
                },
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    best_value, best_schedule, best_metrics = scored[0]
    all_values = np.asarray([item[0] for item in scored], dtype=float)
    return MAPPOStylePolicy(
        pit_schedule_by_driver=dict(best_schedule),
        config=cfg,
        action_space=tuple(race_env.action_space),
        model_id=cfg.model_id,
        training_diagnostics={
            "algorithm": "centralized_schedule_search_with_decentralized_masked_policy",
            "centralized_training": True,
            "decentralized_execution": True,
            "candidate_schedules": int(len(scored)),
            "training_scenarios": int(len(states)),
            "best_centralized_value": float(best_value),
            "best_schedule_by_driver": dict(best_schedule),
            "best_candidate_metrics": best_metrics,
            "value_min": float(np.min(all_values)) if all_values.size else None,
            "value_max": float(np.max(all_values)) if all_values.size else None,
            "value_mean": float(np.mean(all_values)) if all_values.size else None,
            "centralized_critic_features": (
                "team_time_seconds",
                "coordination_penalty_seconds",
                "max_same_lap_pit_count",
                "schedule_regularization_seconds",
            ),
        },
    )


@dataclass
class _FixedSchedulePolicy:
    schedule_by_driver: Mapping[str, int]
    pit_compound: str = "HARD"
    model_id: str = "fixed_centralized_schedule_candidate"

    def select_action(self, state: StrategyState) -> StrategyAction:
        if "HARD" in set(state.used_compounds) or (state.remaining_laps or 0) <= 2:
            return StrategyAction(ACTION_STAY_OUT)
        target = int(self.schedule_by_driver.get(str(state.driver_id), int(state.lap_number) + 1))
        if int(state.lap_number) >= target:
            return StrategyAction(ACTION_PIT_NOW, compound=self.pit_compound)
        return StrategyAction(ACTION_STAY_OUT)


def _normalise_start_states(
    start_states: MultiAgentRaceState | Sequence[MultiAgentRaceState] | None,
) -> tuple[MultiAgentRaceState, ...]:
    if start_states is None:
        return ()
    if isinstance(start_states, MultiAgentRaceState):
        return (start_states,)
    return tuple(start_states)


def _candidate_laps(state: MultiAgentRaceState, config: MAPPOConfig) -> tuple[int, ...]:
    if config.candidate_pit_laps:
        return tuple(int(lap) for lap in config.candidate_pit_laps)
    first = int(state.lap_number)
    max_remaining = min(int(first + 4), int(max(car.total_laps or first + 4 for car in state.cars)) - 2)
    return tuple(range(first, max(first, max_remaining) + 1))


def _unique_schedules(candidates: Sequence[Mapping[str, int]]) -> list[dict[str, int]]:
    seen: set[tuple[tuple[str, int], ...]] = set()
    unique: list[dict[str, int]] = []
    for candidate in candidates:
        key = tuple(sorted((str(driver_id), int(lap)) for driver_id, lap in candidate.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append({driver_id: lap for driver_id, lap in key})
    return unique


def _schedule_regularization(
    schedule: Mapping[str, int],
    states: Sequence[MultiAgentRaceState],
    config: MAPPOConfig,
) -> float:
    penalty = 0.0
    for state in states:
        for car in state.cars:
            target = int(schedule.get(str(car.driver_id), int(car.lap_number)))
            penalty += abs(float(target - int(car.lap_number))) * float(config.schedule_regularization_seconds)
    return float(penalty / max(1, sum(state.car_count for state in states)))


def _finite(value: object, default: float) -> float:
    try:
        if value is None:
            return float(default)
        numeric = float(value)
        if not np.isfinite(numeric):
            return float(default)
        return numeric
    except Exception:
        return float(default)


__all__ = ["MAPPOConfig", "MAPPOPlan", "MAPPOStylePolicy", "fit_mappo_style_policy"]
