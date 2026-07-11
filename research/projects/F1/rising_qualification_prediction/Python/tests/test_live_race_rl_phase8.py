from __future__ import annotations

from dataclasses import replace

from packages.f1.models.live_race.action_space import (
    ACTION_PIT_NOW,
    ACTION_STAY_OUT,
    StrategyAction,
    build_legal_action_mask,
)
from packages.f1.models.live_race.rl.schedule_search import (
    CentralizedSchedulePolicy,
    CentralizedScheduleSearchConfig,
    fit_centralized_schedule_search,
)
from packages.f1.models.live_race.rl.multi_agent_env import (
    MultiAgentLiveRaceEnv,
    build_traffic_heavy_scenario,
)
from packages.f1.models.live_race.rl.self_play import (
    SelfPlayEvaluationConfig,
    SingleAgentTyreThresholdPolicy,
    evaluate_phase8_self_play,
)


def test_multi_agent_env_is_seed_deterministic_and_penalizes_synchronized_pits() -> None:
    env = MultiAgentLiveRaceEnv()
    state = build_traffic_heavy_scenario(car_count=6, seed=123)
    all_pit = tuple(StrategyAction(ACTION_PIT_NOW, compound="HARD") for _ in state.cars)

    left = env.step(state, all_pit)
    right = env.step(state, all_pit)

    assert left.fingerprint() == right.fingerprint()
    assert left.diagnostics["car_count"] == 6
    assert left.diagnostics["pit_count"] == 6
    assert left.diagnostics["sync_pit_threshold"] == 2
    assert left.diagnostics["synchronized_pit_pattern"] is True
    assert left.diagnostics["coordination_penalty_seconds"] > 0.0
    assert all(mask.is_legal(action) for mask, action in zip(left.legal_action_masks, all_pit))


def test_schedule_search_returns_decentralized_policy_with_honest_diagnostics() -> None:
    env = MultiAgentLiveRaceEnv()
    state = env.reset(build_traffic_heavy_scenario(car_count=6, seed=7))
    policy = fit_centralized_schedule_search(
        env,
        state,
        config=CentralizedScheduleSearchConfig(candidate_pit_laps=(5, 6), max_stagger_laps=3),
    )

    assert isinstance(policy, CentralizedSchedulePolicy)
    assert policy.diagnostics["centralized_training"] is False
    assert policy.diagnostics["centralized_search"] is True
    assert policy.diagnostics["reinforcement_learning"] is False
    assert policy.diagnostics["policy_gradient"] is False
    assert "mappo" not in policy.model_id.lower()
    assert "mappo" not in str(policy.diagnostics).lower()
    assert policy.diagnostics["decentralized_execution"] is True
    assert policy.diagnostics["candidate_schedules"] >= 3
    assert set(policy.pit_schedule_by_driver) == set(state.driver_ids)

    scores = policy.centralized_action_scores(state)
    assert set(scores) == set(state.driver_ids)
    assert all("pit_now:HARD:conservative" in driver_scores for driver_scores in scores.values())
    assert all(policy.select_action(car).action_type in {"stay_out", "pit_now"} for car in state.cars)


def test_schedule_policy_masks_illegal_decentralized_pit_actions() -> None:
    env = MultiAgentLiveRaceEnv()
    state = env.reset(build_traffic_heavy_scenario(car_count=6, seed=7))
    policy = CentralizedSchedulePolicy(
        pit_schedule_by_driver={driver_id: int(state.lap_number) for driver_id in state.driver_ids},
        config=CentralizedScheduleSearchConfig(candidate_pit_laps=(5,), max_stagger_laps=1),
        action_space=env.action_space,
        action_mask_config=env.config.action_mask,
    )

    hard_unavailable = replace(
        state.cars[0],
        metadata={**state.cars[0].metadata, "available_compounds": ("MEDIUM",)},
    )
    action = policy.select_action(hard_unavailable)
    mask = build_legal_action_mask(hard_unavailable, action_space=env.action_space, config=env.config.action_mask)
    scores = policy.centralized_action_scores(replace(state, cars=(hard_unavailable, *state.cars[1:])))

    assert action.action_type == ACTION_STAY_OUT
    assert mask.is_legal(action)
    assert scores[hard_unavailable.driver_id]["pit_now:HARD:conservative"] == -1e6


def test_phase8_self_play_beats_single_agent_baseline_and_avoids_sync_pattern() -> None:
    env = MultiAgentLiveRaceEnv()
    state = build_traffic_heavy_scenario(car_count=6, seed=7)
    policy = fit_centralized_schedule_search(
        env,
        state,
        config=CentralizedScheduleSearchConfig(candidate_pit_laps=(5, 6), max_stagger_laps=3),
    )

    result = evaluate_phase8_self_play(
        env=env,
        start_state=state,
        multi_agent_policy=policy,
        single_agent_policy=SingleAgentTyreThresholdPolicy(tyre_age_threshold=18),
        config=SelfPlayEvaluationConfig(seeds=(7, 11), car_count=6, min_delta_vs_single_agent_seconds=1.0),
    )

    assert result.metrics["seeded_replay_stable"] is True
    assert result.metrics["multi_agent_beats_single_agent"] is True
    assert result.metrics["multi_agent_delta_vs_single_agent_seconds"] > 1.0
    assert result.metrics["multi_agent_sync_guard_pass"] is True
    assert result.metrics["summary_by_policy"]["multi_agent"]["max_same_lap_pit_count"] <= 2
    assert result.metrics["summary_by_policy"]["single_agent"]["max_same_lap_pit_count"] > 2
    assert result.metrics["summary_by_policy"]["multi_agent"]["illegal_action_count"] == 0
