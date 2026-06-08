"""Self-play helpers for Phase 8 multi-agent live-race strategy."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Optional, Sequence

import numpy as np

from packages.f1.models.live_race.action_space import ACTION_PIT_NOW, ACTION_STAY_OUT, StrategyAction
from packages.f1.models.live_race.environment import StrategyState
from packages.f1.models.live_race.rl.mappo import MAPPOConfig, MAPPOStylePolicy, fit_mappo_style_policy
from packages.f1.models.live_race.rl.multi_agent_env import (
    MultiAgentLiveRaceEnv,
    MultiAgentRaceConfig,
    MultiAgentRaceState,
    MultiAgentRolloutResult,
    build_traffic_heavy_scenario,
)


@dataclass(frozen=True)
class SelfPlayEvaluationConfig:
    seeds: tuple[int, ...] = (7, 11, 19)
    car_count: int = 6
    max_laps: Optional[int] = None
    min_delta_vs_single_agent_seconds: float = 1.0


@dataclass
class StayOutPolicy:
    model_id: str = "stay_out_baseline"

    def select_action(self, state: StrategyState) -> StrategyAction:
        return StrategyAction(ACTION_STAY_OUT)


@dataclass
class SingleAgentTyreThresholdPolicy:
    """Local greedy baseline that ignores strategic peer coordination."""

    tyre_age_threshold: int = 18
    pit_compound: str = "HARD"
    model_id: str = "single_agent_tyre_threshold_baseline"

    def select_action(self, state: StrategyState) -> StrategyAction:
        if "HARD" in set(state.used_compounds) or (state.remaining_laps or 0) <= 2:
            return StrategyAction(ACTION_STAY_OUT)
        if int(state.tyre_age) >= int(self.tyre_age_threshold):
            return StrategyAction(ACTION_PIT_NOW, compound=self.pit_compound)
        return StrategyAction(ACTION_STAY_OUT)


@dataclass(frozen=True)
class SelfPlayComparisonResult:
    """Evaluation result comparing multi-agent policy to simpler baselines."""

    metrics: Mapping[str, object]
    rollouts: Mapping[str, tuple[MultiAgentRolloutResult, ...]]
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        return {
            "metrics": dict(self.metrics),
            "rollouts": {
                name: [rollout.metrics() for rollout in policy_rollouts]
                for name, policy_rollouts in self.rollouts.items()
            },
            "diagnostics": dict(self.diagnostics),
        }


def evaluate_phase8_self_play(
    *,
    env: MultiAgentLiveRaceEnv | None = None,
    start_state: MultiAgentRaceState | None = None,
    multi_agent_policy: MAPPOStylePolicy | None = None,
    single_agent_policy: SingleAgentTyreThresholdPolicy | None = None,
    stay_out_policy: StayOutPolicy | None = None,
    config: SelfPlayEvaluationConfig | None = None,
    mappo_config: MAPPOConfig | None = None,
) -> SelfPlayComparisonResult:
    """Train if needed, then compare policies under identical seeded scenarios."""

    cfg = config or SelfPlayEvaluationConfig()
    race_env = env or MultiAgentLiveRaceEnv(config=MultiAgentRaceConfig(seed=int(cfg.seeds[0])))
    initial = start_state or build_traffic_heavy_scenario(car_count=int(cfg.car_count), seed=int(cfg.seeds[0]))
    mappo_policy = multi_agent_policy or fit_mappo_style_policy(race_env, initial, config=mappo_config)
    single_policy = single_agent_policy or SingleAgentTyreThresholdPolicy()
    stay_policy = stay_out_policy or StayOutPolicy()

    policies: dict[str, object] = {
        "multi_agent": mappo_policy,
        "single_agent": single_policy,
        "stay_out": stay_policy,
    }
    rollouts: dict[str, list[MultiAgentRolloutResult]] = {name: [] for name in policies}
    replay_stable = True
    replay_fingerprints: dict[str, list[str]] = {name: [] for name in policies}

    for seed in cfg.seeds:
        seeded_env = MultiAgentLiveRaceEnv(config=replace(race_env.config, seed=int(seed)), action_space=race_env.action_space)
        seeded_state = _with_seed(initial, seed=int(seed))
        for name, policy in policies.items():
            rollout = seeded_env.rollout(seeded_state, policy=policy, max_laps=cfg.max_laps, policy_name=name)
            repeat = seeded_env.rollout(seeded_state, policy=policy, max_laps=cfg.max_laps, policy_name=name)
            replay_stable = bool(replay_stable and rollout.fingerprint() == repeat.fingerprint())
            replay_fingerprints[name].append(rollout.fingerprint())
            rollouts[name].append(rollout)

    summary = {name: _summarise_rollouts(policy_rollouts) for name, policy_rollouts in rollouts.items()}
    multi_time = float(summary["multi_agent"]["mean_team_time_seconds"])
    single_time = float(summary["single_agent"]["mean_team_time_seconds"])
    stay_time = float(summary["stay_out"]["mean_team_time_seconds"])
    delta_vs_single = float(single_time - multi_time)
    delta_vs_stay = float(stay_time - multi_time)
    sync_threshold = race_env.sync_pit_threshold(initial.car_count)

    metrics = {
        "available": True,
        "scenario": "traffic_heavy_synthetic",
        "seeded_replay_stable": bool(replay_stable),
        "multi_agent_beats_single_agent": bool(delta_vs_single > float(cfg.min_delta_vs_single_agent_seconds)),
        "multi_agent_delta_vs_single_agent_seconds": float(delta_vs_single),
        "multi_agent_delta_vs_stay_out_seconds": float(delta_vs_stay),
        "sync_pit_threshold": int(sync_threshold),
        "multi_agent_sync_guard_pass": bool(summary["multi_agent"]["max_same_lap_pit_count"] <= sync_threshold),
        "summary_by_policy": summary,
        "promotion_gate_pass": bool(
            replay_stable
            and delta_vs_single > float(cfg.min_delta_vs_single_agent_seconds)
            and summary["multi_agent"]["max_same_lap_pit_count"] <= sync_threshold
        ),
    }
    return SelfPlayComparisonResult(
        metrics=metrics,
        rollouts={name: tuple(policy_rollouts) for name, policy_rollouts in rollouts.items()},
        diagnostics={
            "mappo_policy": mappo_policy.diagnostics,
            "baseline_policy": single_policy.model_id,
            "stay_out_policy": stay_policy.model_id,
            "replay_fingerprints": replay_fingerprints,
        },
    )


def compare_multi_agent_to_single_agent(
    *,
    env: MultiAgentLiveRaceEnv | None = None,
    start_state: MultiAgentRaceState | None = None,
    config: SelfPlayEvaluationConfig | None = None,
    mappo_config: MAPPOConfig | None = None,
) -> SelfPlayComparisonResult:
    return evaluate_phase8_self_play(
        env=env,
        start_state=start_state,
        config=config,
        mappo_config=mappo_config,
    )


def _summarise_rollouts(rollouts: Sequence[MultiAgentRolloutResult]) -> dict[str, object]:
    if not rollouts:
        return {
            "mean_team_time_seconds": None,
            "mean_team_reward": None,
            "mean_synchronized_pit_rate": None,
            "max_same_lap_pit_count": 0,
            "mean_coordination_penalty_seconds": None,
            "illegal_action_count": 0,
        }
    return {
        "mean_team_time_seconds": float(np.mean([rollout.team_time_seconds for rollout in rollouts])),
        "mean_team_reward": float(np.mean([rollout.team_reward for rollout in rollouts])),
        "mean_synchronized_pit_rate": float(np.mean([rollout.synchronized_pit_rate for rollout in rollouts])),
        "max_same_lap_pit_count": int(max(rollout.max_same_lap_pit_count for rollout in rollouts)),
        "mean_coordination_penalty_seconds": float(np.mean([rollout.coordination_penalty_seconds for rollout in rollouts])),
        "illegal_action_count": int(sum(rollout.illegal_action_count for rollout in rollouts)),
        "pit_lap_histograms": [rollout.pit_lap_histogram for rollout in rollouts],
    }


def _with_seed(state: MultiAgentRaceState, *, seed: int) -> MultiAgentRaceState:
    cars = []
    for car in state.cars:
        metadata = {
            **dict(car.metadata or {}),
            "multi_agent_seed": int(seed),
            "multi_agent_scenario_id": f"{state.scenario_id}_seed_{seed}",
        }
        cars.append(replace(car, metadata=metadata))
    return replace(
        state,
        cars=tuple(cars),
        seed=int(seed),
        scenario_id=f"{state.scenario_id}_seed_{seed}",
        pit_counts_by_lap={},
    )


__all__ = [
    "SelfPlayComparisonResult",
    "SelfPlayEvaluationConfig",
    "SingleAgentTyreThresholdPolicy",
    "StayOutPolicy",
    "compare_multi_agent_to_single_agent",
    "evaluate_phase8_self_play",
]
