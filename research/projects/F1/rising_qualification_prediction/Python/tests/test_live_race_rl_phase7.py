from __future__ import annotations

from dataclasses import replace

import pytest

from packages.f1.models.live_race.action_space import (
    ACTION_PIT_NOW,
    ACTION_STAY_OUT,
    ActionMaskConfig,
    StrategyAction,
    build_action_space,
    build_legal_action_mask,
)
from packages.f1.models.live_race.environment import StrategyReward, StrategyState, StrategyTransition
from packages.f1.models.live_race.replay_buffer import ReplayBufferRecord
from packages.f1.models.live_race.rl.behavior_cloning import (
    BehaviorCloningConfig,
    TrivialLegalActionBaseline,
    evaluate_behavior_cloning,
    fit_behavior_cloning,
)
from packages.f1.models.live_race.rl.offline import (
    ConservativeOfflineRLConfig,
    evaluate_offline_rl_policy,
    fit_conservative_offline_q,
)
from packages.f1.models.live_race.rl.replay_buffer import (
    DEFAULT_STATE_FEATURE_NAMES,
    StrategyActionIndex,
    build_rl_replay_dataset,
    state_to_feature_vector,
)


ACTION_INDEX = StrategyActionIndex.from_action_space(
    build_action_space(compounds=("MEDIUM", "HARD"), modes=("conservative",), include_pit_next_lap=False)
)


def _state(
    *,
    lap: int,
    tyre_age: int,
    compound: str = "MEDIUM",
    used: tuple[str, ...] = ("MEDIUM", "HARD"),
) -> StrategyState:
    return StrategyState(
        event_key=202601,
        driver_id="44",
        lap_number=lap,
        total_laps=10,
        remaining_laps=max(0, 10 - lap),
        stint_id=1,
        compound=compound,
        tyre_age=tyre_age,
        used_compounds=used,
        race_time_seconds=90.0 * lap,
        track_status="1",
        is_greenish=True,
        deg_rate_mean=0.06,
        next_lap_mean=90.0,
        metadata={
            "available_compounds": ("MEDIUM", "HARD"),
            "pit_lane_open": True,
            "ignored_future_columns": (),
        },
    )


def _transition(*, lap: int, tyre_age: int, action: StrategyAction, reward: float) -> StrategyTransition:
    state = _state(lap=lap, tyre_age=tyre_age)
    if action.action_type == ACTION_PIT_NOW:
        next_state = replace(
            _state(lap=lap + 1, tyre_age=0, compound=action.compound or "HARD"),
            stint_id=2,
            used_compounds=("MEDIUM", "HARD"),
        )
    else:
        next_state = _state(lap=lap + 1, tyre_age=tyre_age + 1)
    legal_mask = build_legal_action_mask(state, action_space=ACTION_INDEX.actions)
    return StrategyTransition(
        state_t=state,
        action_t=action,
        reward_t=StrategyReward(value=reward, components={"synthetic_return": reward}),
        state_t1=next_state,
        done=False,
        legal_action_mask=legal_mask,
        metadata={"source": "synthetic_phase7_test"},
    )


def _records() -> list[ReplayBufferRecord]:
    stay = StrategyAction(ACTION_STAY_OUT)
    pit_hard = StrategyAction(ACTION_PIT_NOW, compound="HARD")
    transitions = [
        _transition(lap=1, tyre_age=1, action=stay, reward=4.0),
        _transition(lap=2, tyre_age=1, action=stay, reward=4.0),
        _transition(lap=3, tyre_age=1, action=stay, reward=4.0),
        _transition(lap=4, tyre_age=1, action=stay, reward=4.0),
        _transition(lap=5, tyre_age=5, action=pit_hard, reward=10.0),
        _transition(lap=6, tyre_age=5, action=pit_hard, reward=10.0),
        _transition(lap=7, tyre_age=5, action=pit_hard, reward=10.0),
    ]
    return [
        ReplayBufferRecord.from_transition(
            transition,
            source="synthetic",
            split_key="train",
            metadata={"record_number": idx},
        )
        for idx, transition in enumerate(transitions)
    ]


def test_rl_replay_dataset_preserves_masks_actions_rewards_and_metadata() -> None:
    dataset = build_rl_replay_dataset(_records(), action_index=ACTION_INDEX, strict=True)

    assert dataset.rows == 7
    assert dataset.states.shape[0] == 7
    assert dataset.states.shape[1] == len(dataset.feature_names)
    assert dataset.legal_action_masks.shape == (7, ACTION_INDEX.size)
    assert dataset.actions.tolist().count(ACTION_INDEX.index_for(StrategyAction(ACTION_PIT_NOW, compound="HARD"))) == 3
    assert dataset.rewards.max() == 10.0
    assert dataset.examples[0].source == "synthetic"
    assert dataset.examples[0].split_key == "train"
    assert dataset.examples[0].ood is False
    assert dataset.diagnostics()["ood_rows"] == 0


def test_unknown_race_horizon_is_explicit_and_keeps_causal_lap_number() -> None:
    unknown = replace(_state(lap=37, tyre_age=12), total_laps=None, remaining_laps=None)
    remaining_only = replace(
        _state(lap=37, tyre_age=12),
        total_laps=None,
        remaining_laps=15,
    )
    known = _state(lap=5, tyre_age=5)
    unknown_features = dict(
        zip(DEFAULT_STATE_FEATURE_NAMES, state_to_feature_vector(unknown))
    )
    known_features = dict(
        zip(DEFAULT_STATE_FEATURE_NAMES, state_to_feature_vector(known))
    )
    remaining_only_features = dict(
        zip(DEFAULT_STATE_FEATURE_NAMES, state_to_feature_vector(remaining_only))
    )

    assert unknown_features["race_horizon_known"] == 0.0
    assert unknown_features["lap_fraction"] == 0.0
    assert unknown_features["remaining_fraction"] == 0.0
    assert unknown_features["lap_number_per_100"] == pytest.approx(0.37)
    assert remaining_only_features["race_horizon_known"] == 1.0
    assert remaining_only_features["lap_fraction"] == pytest.approx(37.0 / 52.0)
    assert remaining_only_features["remaining_fraction"] == pytest.approx(15.0 / 52.0)
    assert known_features["race_horizon_known"] == 1.0
    assert known_features["lap_fraction"] == pytest.approx(0.5)
    assert known_features["remaining_fraction"] == pytest.approx(0.5)
    assert known_features["lap_number_per_100"] == pytest.approx(0.05)


def test_rl_action_index_reuses_the_environment_action_mask_contract() -> None:
    action_space = build_action_space(
        compounds=("MEDIUM", "HARD"),
        modes=("conservative",),
        include_pit_next_lap=False,
    )
    action_index = StrategyActionIndex.from_action_space(
        action_space,
        action_mask_config=ActionMaskConfig(
            default_available_compounds=("MEDIUM", "HARD"),
        ),
    )
    state = _state(lap=5, tyre_age=5)
    state = replace(
        state,
        metadata={
            key: value
            for key, value in state.metadata.items()
            if key != "available_compounds"
        },
    )

    legal = action_index.legal_mask_for_state(state)
    pit_hard = action_index.index_for(StrategyAction(ACTION_PIT_NOW, compound="HARD"))

    assert bool(legal[pit_hard]) is True


def test_behavior_cloning_is_masked_and_beats_trivial_action_baseline_on_replay() -> None:
    dataset = build_rl_replay_dataset(_records(), action_index=ACTION_INDEX, strict=True)
    policy = fit_behavior_cloning(dataset, config=BehaviorCloningConfig(bucket_precision=2))
    result = evaluate_behavior_cloning(policy, dataset)

    assert result["diagnostics"]["not_promoted_strategy_optimizer"] is True
    assert result["diagnostics"]["promotion_gate_pass"] is False
    assert (
        result["metrics"]["action_selection_accuracy"]
        > result["trivial_baseline"]["metrics"]["action_selection_accuracy"]
    )
    assert result["metrics"]["pit_decision_f1"] > result["trivial_baseline"]["metrics"]["pit_decision_f1"]
    assert result["metrics"]["illegal_prediction_rate"] == 0.0

    hard_unavailable = _state(lap=5, tyre_age=5)
    hard_unavailable = replace(
        hard_unavailable,
        metadata={**hard_unavailable.metadata, "available_compounds": ("MEDIUM",)},
    )
    assert policy.select_action(hard_unavailable).action_type == ACTION_STAY_OUT


def test_conservative_offline_q_is_bounded_masked_and_fails_closed_without_simulator() -> None:
    dataset = build_rl_replay_dataset(_records(), action_index=ACTION_INDEX, strict=True)
    bc_policy = fit_behavior_cloning(dataset)
    offline_policy = fit_conservative_offline_q(
        dataset,
        behavior_policy=bc_policy,
        config=ConservativeOfflineRLConfig(
            discount=0.0,
            iterations=25,
            ood_action_penalty=20.0,
            reward_clip=(-20.0, 20.0),
            value_clip=(-50.0, 50.0),
        ),
    )

    assert offline_policy.select_action(_state(lap=1, tyre_age=1)).action_type == ACTION_STAY_OUT
    assert offline_policy.select_action(_state(lap=5, tyre_age=5)).action_type == ACTION_PIT_NOW
    hard_unavailable = _state(lap=5, tyre_age=5)
    hard_unavailable = replace(
        hard_unavailable,
        metadata={**hard_unavailable.metadata, "available_compounds": ("MEDIUM",)},
    )
    assert offline_policy.select_action(hard_unavailable).action_type == ACTION_STAY_OUT
    assert offline_policy.training_diagnostics["value_min"] >= -50.0
    assert offline_policy.training_diagnostics["value_max"] <= 50.0

    unavailable = evaluate_offline_rl_policy(offline_policy)
    assert unavailable.available is False
    assert unavailable.metrics["promotion_gate_pass"] is False
    assert unavailable.metrics["reason"] == "locked_simulator_required"


def test_conservative_offline_q_requires_gamma_one_for_aggregated_smdp_reward() -> None:
    stay = StrategyAction(ACTION_STAY_OUT)
    action_index = StrategyActionIndex.from_action_space((stay,))
    transition = _transition(lap=1, tyre_age=1, action=stay, reward=10.0)
    transition = replace(
        transition,
        metadata={**transition.metadata, "elapsed_laps": 3},
    )
    dataset = build_rl_replay_dataset(
        [ReplayBufferRecord.from_transition(transition)],
        action_index=action_index,
        strict=True,
    )

    with pytest.raises(ValueError, match="discount must equal 1.0"):
        fit_conservative_offline_q(
            dataset,
            config=ConservativeOfflineRLConfig(discount=0.5),
        )

    policy = fit_conservative_offline_q(
        dataset,
        config=ConservativeOfflineRLConfig(
            discount=1.0,
            iterations=1,
            learning_rate=1.0,
            conservative_penalty=0.0,
            ood_action_penalty=0.0,
            reward_clip=(-100.0, 100.0),
            value_clip=(-100.0, 100.0),
        ),
    )

    assert policy.value(transition.state_t, stay) == 20.0
    assert policy.training_diagnostics["discount_semantics"] == (
        "undiscounted_composite_interval_reward_gamma_one_for_aggregated_smdp_transitions"
    )
    assert policy.training_diagnostics["aggregated_multi_lap_reward_rows"] == 1
    assert policy.training_diagnostics["elapsed_laps_max"] == 3


def test_offline_rl_evaluation_exposes_locked_simulator_comparisons() -> None:
    dataset = build_rl_replay_dataset(_records(), action_index=ACTION_INDEX, strict=True)
    bc_policy = fit_behavior_cloning(dataset)
    offline_policy = fit_conservative_offline_q(
        dataset,
        behavior_policy=bc_policy,
        config=ConservativeOfflineRLConfig(discount=0.0, iterations=10, ood_action_penalty=20.0),
    )
    baseline = TrivialLegalActionBaseline.fit(dataset)

    class LockedSyntheticSimulator:
        def evaluate_policy(self, policy: object) -> dict[str, object]:
            action = policy.select_action(_state(lap=5, tyre_age=5))  # type: ignore[attr-defined]
            mean_return = 12.0 if action.action_type == ACTION_PIT_NOW else 3.0
            return {
                "metrics": {
                    "mean_return": mean_return,
                    "promotion_gate_pass": True,
                },
                "selected_action": action.key,
            }

    result = evaluate_offline_rl_policy(
        offline_policy,
        simulator=LockedSyntheticSimulator(),
        behavior_cloning_policy=bc_policy,
        dp_mpc_policy=baseline,
    )

    assert result.available is True
    assert result.metrics["evaluation_setting"] == "locked_simulator"
    assert result.metrics["historical_accuracy_used_for_promotion"] is False
    assert set(result.comparison_payloads) == {"offline_rl", "behavior_cloning", "dp_mpc"}
    assert result.metrics["comparison_deltas"]["policy_value_delta_vs_dp_mpc"] == 9.0
    assert result.metrics["promotion_gate_pass"] is False
    assert "evaluation_not_locked" in result.metrics["locked_evaluation_issues"]


def test_offline_rl_promotion_gate_requires_beating_locked_simulator_comparators() -> None:
    dataset = build_rl_replay_dataset(_records(), action_index=ACTION_INDEX, strict=True)
    bc_policy = fit_behavior_cloning(dataset)
    offline_policy = fit_conservative_offline_q(
        dataset,
        behavior_policy=bc_policy,
        config=ConservativeOfflineRLConfig(discount=0.0, iterations=10, ood_action_penalty=20.0),
    )
    baseline = TrivialLegalActionBaseline.fit(dataset)

    class LockedLosingSimulator:
        def evaluate_policy(self, policy: object) -> dict[str, object]:
            if policy is offline_policy:
                mean_return = 5.0
            elif policy is bc_policy:
                mean_return = 6.0
            else:
                mean_return = 7.0
            return {
                "metrics": {
                    "mean_return": mean_return,
                    "promotion_gate_pass": True,
                }
            }

    result = evaluate_offline_rl_policy(
        offline_policy,
        simulator=LockedLosingSimulator(),
        behavior_cloning_policy=bc_policy,
        dp_mpc_policy=baseline,
    )

    assert result.available is True
    assert result.metrics["comparison_gate_pass"] is False
    assert result.metrics["promotion_gate_pass"] is False
    assert result.metrics["comparison_deltas"]["policy_value_delta_vs_behavior_cloning"] == -1.0
    assert result.metrics["comparison_deltas"]["policy_value_delta_vs_dp_mpc"] == -2.0


def test_offline_rl_gate_requires_and_accepts_complete_locked_evidence_contract() -> None:
    dataset = build_rl_replay_dataset(_records(), action_index=ACTION_INDEX, strict=True)
    bc_policy = fit_behavior_cloning(dataset)
    offline_policy = fit_conservative_offline_q(
        dataset,
        behavior_policy=bc_policy,
        config=ConservativeOfflineRLConfig(discount=0.0, iterations=10, ood_action_penalty=20.0),
    )
    baseline = TrivialLegalActionBaseline.fit(dataset)

    class CompleteLockedEvaluator:
        def evaluate_policy(self, policy: object) -> dict[str, object]:
            mean_return = 12.0 if policy is offline_policy else 4.0 if policy is bc_policy else 3.0
            return {
                "metrics": {
                    "mean_return": mean_return,
                    "promotion_gate_pass": True,
                    "evaluation_locked": True,
                    "simulator_artifact_hash": "a" * 64,
                    "calibration_report_hash": "b" * 64,
                    "evaluation_split_id": "heldout-race-prefixes-v1",
                    "causal_event_time_replay": True,
                    "illegal_action_rate": 0.0,
                    "policy_error_count": 0,
                    "support_coverage": 0.95,
                }
            }

    result = evaluate_offline_rl_policy(
        offline_policy,
        simulator=CompleteLockedEvaluator(),
        behavior_cloning_policy=bc_policy,
        dp_mpc_policy=baseline,
    )

    assert result.metrics["locked_evaluation_issues"] == []
    assert result.metrics["comparison_gate_pass"] is True
    assert result.metrics["promotion_gate_pass"] is True
