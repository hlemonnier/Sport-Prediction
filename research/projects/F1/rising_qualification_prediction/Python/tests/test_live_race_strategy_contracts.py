from __future__ import annotations

import pandas as pd

from packages.f1.models.live_race.action_space import (
    ACTION_PIT_NEXT_LAP,
    ACTION_PIT_NOW,
    ACTION_STAY_OUT,
    ActionMaskConfig,
    StrategyAction,
    build_action_space,
    build_legal_action_mask,
)
from packages.f1.models.live_race.environment import (
    RewardConfig,
    StrategyState,
    build_replay_transitions,
    compute_transition_reward,
)
from packages.f1.models.live_race.evaluate_policy import evaluate_strategy_policy
from packages.f1.models.live_race.mpc import MPCStrategyPlanner
from packages.f1.models.live_race.planner import DeterministicStrategyPlanner, PlannerConfig
from packages.f1.models.live_race.policy import DeterministicBaselinePolicy, PlannerPolicy
from packages.f1.models.live_race.replay_buffer import replay_buffer_from_transitions


def _state(**overrides: object) -> StrategyState:
    base = {
        "event_key": 202601,
        "driver_id": "44",
        "lap_number": 20,
        "total_laps": 58,
        "remaining_laps": 38,
        "stint_id": 1,
        "compound": "MEDIUM",
        "tyre_age": 10,
        "used_compounds": ("MEDIUM",),
        "race_time_seconds": 1800.0,
        "track_status": "1",
        "is_greenish": True,
        "pace_penalty_mean": 0.0,
        "deg_rate_mean": 0.04,
        "next_lap_mean": 90.0,
        "metadata": {"available_compounds": ("SOFT", "MEDIUM", "HARD")},
    }
    base.update(overrides)
    return StrategyState(**base)


def _replay_rows(extra_lap: bool = False) -> pd.DataFrame:
    rows = [
        {
            "event_key": 202601,
            "driver_id": "44",
            "lap_number": 1,
            "total_laps": 6,
            "remaining_laps": 5,
            "stint_id": 1,
            "compound": "MEDIUM",
            "tyre_age": 0,
            "used_compounds": "MEDIUM",
            "available_compounds": "MEDIUM,HARD",
            "track_status": "1",
            "lap_time_seconds": 90.0,
            "race_time_seconds": 90.0,
            "timestamp": 1.0,
            "observed_action": "stay_out",
            "final_position": 7,
        },
        {
            "event_key": 202601,
            "driver_id": "44",
            "lap_number": 2,
            "total_laps": 6,
            "remaining_laps": 4,
            "stint_id": 1,
            "compound": "MEDIUM",
            "tyre_age": 1,
            "used_compounds": "MEDIUM",
            "available_compounds": "MEDIUM,HARD",
            "track_status": "1",
            "lap_time_seconds": 91.0,
            "race_time_seconds": 181.0,
            "timestamp": 2.0,
            "observed_action": "pit_now",
            "next_compound": "HARD",
            "final_position": 7,
        },
        {
            "event_key": 202601,
            "driver_id": "44",
            "lap_number": 3,
            "total_laps": 6,
            "remaining_laps": 3,
            "stint_id": 2,
            "compound": "HARD",
            "tyre_age": 0,
            "used_compounds": "MEDIUM,HARD",
            "available_compounds": "MEDIUM,HARD",
            "track_status": "1",
            "lap_time_seconds": 112.0,
            "race_time_seconds": 293.0,
            "timestamp": 3.0,
            "observed_action": "stay_out",
            "final_position": 7,
        },
    ]
    if extra_lap:
        rows.append(
            {
                "event_key": 202601,
                "driver_id": "44",
                "lap_number": 4,
                "total_laps": 6,
                "remaining_laps": 2,
                "stint_id": 2,
                "compound": "HARD",
                "tyre_age": 1,
                "used_compounds": "MEDIUM,HARD",
                "available_compounds": "MEDIUM,HARD",
                "track_status": "1",
                "lap_time_seconds": 91.5,
                "race_time_seconds": 384.5,
                "timestamp": 4.0,
                "observed_action": "stay_out",
                "final_position": 4,
            }
        )
    return pd.DataFrame(rows)


def test_action_mask_blocks_impossible_pit_and_compound_actions() -> None:
    action_space = build_action_space(compounds=("SOFT", "MEDIUM", "HARD"), include_pit_next_lap=True)

    red_state = _state(track_status="5", is_red=True, is_greenish=False)
    red_mask = build_legal_action_mask(red_state, action_space=action_space)
    assert red_mask.is_legal(StrategyAction(ACTION_STAY_OUT, mode="conservative"))
    assert not red_mask.is_legal(StrategyAction(ACTION_STAY_OUT, mode="aggressive"))
    assert not red_mask.is_legal(StrategyAction(ACTION_PIT_NOW, compound="HARD"))

    limited_state = _state(metadata={"available_compounds": ("MEDIUM", "HARD")})
    limited_mask = build_legal_action_mask(limited_state, action_space=action_space)
    assert not limited_mask.is_legal(StrategyAction(ACTION_PIT_NOW, compound="SOFT"))
    assert limited_mask.is_legal(StrategyAction(ACTION_PIT_NOW, compound="HARD"))

    late_state = _state(remaining_laps=2)
    late_mask = build_legal_action_mask(late_state, action_space=action_space, config=ActionMaskConfig(min_laps_after_stop=2))
    assert not late_mask.is_legal(StrategyAction(ACTION_PIT_NOW, compound="HARD"))
    assert not late_mask.is_legal(StrategyAction(ACTION_PIT_NEXT_LAP, compound="HARD"))


def test_replay_transitions_are_no_leakage_prefix_invariant_and_bufferable() -> None:
    truncated = build_replay_transitions(_replay_rows(extra_lap=False))
    full = build_replay_transitions(_replay_rows(extra_lap=True))

    assert len(truncated) == 2
    assert len(full) == 3
    assert truncated[0].state_t.metadata["available_through_lap"] == 1
    assert "final_position" in truncated[0].state_t.metadata["ignored_future_columns"]
    assert truncated[0].state_t.fingerprint() == full[0].state_t.fingerprint()

    buffer = replay_buffer_from_transitions(truncated)
    assert len(buffer) == 2
    assert buffer.prefix_invariant_with(replay_buffer_from_transitions(full), cutoff_lap=2)
    frame = buffer.to_frame()
    assert set(frame["action_type"]) == {ACTION_STAY_OUT, ACTION_PIT_NOW}
    assert frame["is_action_legal"].all()
    assert buffer.sample(1, seed=7)[0].record_id in set(frame["record_id"])


def test_reward_function_penalizes_pit_actions_and_illegal_masks() -> None:
    state_t = _state(lap_number=10, race_time_seconds=900.0)
    state_t1 = _state(lap_number=11, race_time_seconds=991.0, tyre_age=11)
    stay = StrategyAction(ACTION_STAY_OUT)
    pit = StrategyAction(ACTION_PIT_NOW, compound="HARD")
    mask = build_legal_action_mask(state_t)

    stay_reward = compute_transition_reward(state_t, stay, state_t1, legal_action_mask=mask)
    pit_reward = compute_transition_reward(
        state_t,
        pit,
        state_t1,
        legal_action_mask=mask,
        config=RewardConfig(pit_action_penalty=5.0),
    )

    assert stay_reward.components["race_time_delta_seconds"] == 91.0
    assert pit_reward.value < stay_reward.value
    assert pit_reward.components["pit_action_penalty"] == 5.0


def test_dp_planner_changes_behavior_for_deg_sc_and_low_overtake_tracks() -> None:
    planner = DeterministicStrategyPlanner(config=PlannerConfig(horizon_laps=8, strategy_score_weight=0.0))

    high_deg = _state(
        compound="SOFT",
        tyre_age=21,
        used_compounds=("SOFT",),
        deg_rate_mean=0.18,
        circuit_tyre_degradation=0.95,
        circuit_overtaking_difficulty=0.35,
    )
    high_deg_action = planner.plan(high_deg).action
    assert high_deg_action.action_type in {ACTION_PIT_NOW, ACTION_PIT_NEXT_LAP}

    low_deg = _state(
        compound="HARD",
        tyre_age=4,
        used_compounds=("MEDIUM", "HARD"),
        deg_rate_mean=0.015,
        circuit_tyre_degradation=0.25,
        circuit_overtaking_difficulty=0.70,
    )
    assert planner.plan(low_deg).action.action_type == ACTION_STAY_OUT

    sc_window = _state(
        compound="SOFT",
        tyre_age=18,
        used_compounds=("SOFT",),
        deg_rate_mean=0.08,
        track_status="4",
        is_sc_vsc=True,
        is_greenish=False,
        circuit_tyre_degradation=0.80,
        circuit_overtaking_difficulty=0.40,
    )
    assert planner.plan(sc_window).action.action_type in {ACTION_PIT_NOW, ACTION_PIT_NEXT_LAP}

    monaco_style = _state(
        compound="MEDIUM",
        tyre_age=9,
        used_compounds=("SOFT", "MEDIUM"),
        deg_rate_mean=0.02,
        circuit_tyre_degradation=0.30,
        circuit_overtaking_difficulty=1.0,
    )
    assert planner.plan(monaco_style).action.action_type == ACTION_STAY_OUT


def test_policy_evaluator_reports_masks_regret_distribution_and_invariance() -> None:
    truncated = build_replay_transitions(_replay_rows(extra_lap=False))
    full = build_replay_transitions(_replay_rows(extra_lap=True))
    planner = DeterministicStrategyPlanner(config=PlannerConfig(horizon_laps=4, strategy_score_weight=0.0))

    result = evaluate_strategy_policy(
        truncated,
        policy=planner,
        oracle_planner=planner,
        comparison_transitions=full,
    )
    metrics = result.metrics

    assert metrics["available"] is True
    assert metrics["illegal_action_rate"] == 0.0
    assert metrics["transition_consistency"]["ok"] is True
    assert metrics["no_leakage_replay_invariance"]["prefix_invariant"] is True
    assert metrics["regret_vs_oracle"]["available"] is True
    assert metrics["regret_vs_oracle"]["mean"] is not None
    assert float(metrics["regret_vs_oracle"]["mean"]) <= 1e-9
    assert metrics["action_distribution"]["by_type"]
    assert metrics["promotion_gate_pass"] is True


def test_mpc_and_policy_wrappers_return_contract_actions() -> None:
    planner = DeterministicStrategyPlanner(config=PlannerConfig(horizon_laps=4, strategy_score_weight=0.0))
    state = _state(compound="SOFT", tyre_age=19, deg_rate_mean=0.12, circuit_tyre_degradation=0.85)

    mpc_result = MPCStrategyPlanner(planner=planner).replan(state)
    planner_policy_action = PlannerPolicy(planner=planner).select_action(state)
    baseline_action = DeterministicBaselinePolicy().select_action(state)

    assert planner_policy_action == mpc_result.action
    assert baseline_action.action_type in {ACTION_STAY_OUT, ACTION_PIT_NOW, ACTION_PIT_NEXT_LAP}
    assert isinstance(mpc_result.action, StrategyAction)
