from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

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
    LEAKAGE_CONTRACT_VERSION,
    TRANSITION_FINGERPRINT_VERSION,
    RewardConfig,
    StrategyState,
    build_replay_transitions,
    compute_transition_reward,
)
from packages.f1.models.live_race.evaluate_policy import evaluate_strategy_policy
from packages.f1.models.live_race.mpc import MPCStrategyPlanner
from packages.f1.models.live_race.planner import DeterministicStrategyPlanner, PlannerConfig
from packages.f1.models.live_race.policy import DeterministicBaselinePolicy, PlannerPolicy
from packages.f1.models.live_race.replay_buffer import (
    ReplayBufferRecord,
    replay_buffer_from_transitions,
)
from packages.f1.models.live_race.rl.behavior_cloning import (
    evaluate_behavior_cloning,
    fit_behavior_cloning,
)
from packages.f1.models.live_race.rl.offline import fit_conservative_offline_q
from packages.f1.models.live_race.rl.replay_buffer import build_rl_replay_dataset
from packages.f1.models.live_race.sources import _build_stint_id


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
        "metadata": {
            "available_compounds": ("SOFT", "MEDIUM", "HARD"),
            "pit_lane_open": True,
        },
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
            "pit_lane_open": True,
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
            "pit_lane_open": True,
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
            "pit_lane_open": True,
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
                "pit_lane_open": True,
                "track_status": "1",
                "lap_time_seconds": 91.5,
                "race_time_seconds": 384.5,
                "timestamp": 4.0,
                "observed_action": "stay_out",
                "final_position": 4,
            }
        )
    frame = pd.DataFrame(rows)
    frame["position"] = 7
    frame["is_box_lap"] = False
    frame["forced_pit_commitment_known"] = True
    frame["forced_pit_next_compound"] = None
    return frame


def _physical_pit_rows(*, extra_lap: bool = False) -> pd.DataFrame:
    rows = [
        {
            "event_key": 202601,
            "driver_id": "44",
            "lap_number": 1,
            "total_laps": 6,
            "remaining_laps": 5,
            "stint_id": 1,
            "compound": "MEDIUM",
            "tyre_age": 4,
            "used_compounds": "MEDIUM",
            "race_time_seconds": 90.0,
            "is_pit_in_lap": False,
            "is_pit_out_lap": False,
            "pit_lane_open_known": False,
            "compound_inventory_known": False,
            "behavior_action_support_known": False,
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
            "tyre_age": 5,
            "used_compounds": "MEDIUM",
            "race_time_seconds": 202.0,
            "is_pit_in_lap": True,
            "is_pit_out_lap": False,
            "pit_lane_open_known": False,
            "compound_inventory_known": False,
            "behavior_action_support_known": False,
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
            "race_time_seconds": 310.0,
            "is_pit_in_lap": False,
            "is_pit_out_lap": True,
            "pit_lane_open_known": False,
            "compound_inventory_known": False,
            "behavior_action_support_known": False,
            "final_position": 7,
        },
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
            "race_time_seconds": 401.0,
            "is_pit_in_lap": False,
            "is_pit_out_lap": False,
            "pit_lane_open_known": False,
            "compound_inventory_known": False,
            "behavior_action_support_known": False,
            "final_position": 7,
        },
        {
            "event_key": 202601,
            "driver_id": "44",
            "lap_number": 5,
            "total_laps": 6,
            "remaining_laps": 1,
            "stint_id": 2,
            "compound": "HARD",
            "tyre_age": 2,
            "used_compounds": "MEDIUM,HARD",
            "race_time_seconds": 492.5,
            "is_pit_in_lap": False,
            "is_pit_out_lap": False,
            "pit_lane_open_known": False,
            "compound_inventory_known": False,
            "behavior_action_support_known": False,
            "final_position": 7,
        },
    ]
    if extra_lap:
        rows.append(
            {
                **rows[-1],
                "lap_number": 6,
                "remaining_laps": 0,
                "tyre_age": 3,
                "race_time_seconds": 584.5,
                "final_position": 3,
            }
        )
    frame = pd.DataFrame(rows)
    frame["position"] = 7
    frame["track_status"] = "1"
    frame["is_box_lap"] = False
    frame["forced_pit_commitment_known"] = True
    frame["forced_pit_next_compound"] = None
    return frame


def test_action_mask_blocks_impossible_pit_and_compound_actions() -> None:
    action_space = build_action_space(compounds=("SOFT", "MEDIUM", "HARD"), include_pit_next_lap=True)

    red_state = _state(track_status="5", is_red=True, is_greenish=False)
    red_mask = build_legal_action_mask(red_state, action_space=action_space)
    assert red_mask.is_legal(StrategyAction(ACTION_STAY_OUT, mode="conservative"))
    assert not red_mask.is_legal(StrategyAction(ACTION_STAY_OUT, mode="aggressive"))
    assert not red_mask.is_legal(StrategyAction(ACTION_PIT_NOW, compound="HARD"))

    limited_state = _state(
        metadata={"available_compounds": ("MEDIUM", "HARD"), "pit_lane_open": True}
    )
    limited_mask = build_legal_action_mask(limited_state, action_space=action_space)
    assert not limited_mask.is_legal(StrategyAction(ACTION_PIT_NOW, compound="SOFT"))
    assert limited_mask.is_legal(StrategyAction(ACTION_PIT_NOW, compound="HARD"))

    late_state = _state(remaining_laps=2)
    late_mask = build_legal_action_mask(late_state, action_space=action_space, config=ActionMaskConfig(min_laps_after_stop=2))
    assert not late_mask.is_legal(StrategyAction(ACTION_PIT_NOW, compound="HARD"))
    assert not late_mask.is_legal(StrategyAction(ACTION_PIT_NEXT_LAP, compound="HARD"))


def test_mandatory_change_forces_last_feasible_pit_or_marks_mask_infeasible() -> None:
    action_space = build_action_space(
        compounds=("MEDIUM", "HARD"),
        include_pit_next_lap=True,
    )
    last_feasible = _state(
        remaining_laps=3,
        compound="MEDIUM",
        used_compounds=("MEDIUM",),
        metadata={
            "available_compounds": ("MEDIUM", "HARD"),
            "pit_lane_open": True,
            "mandatory_compound_change_required": True,
        },
    )
    mask = build_legal_action_mask(last_feasible, action_space=action_space)

    assert mask.constraint_feasible is True
    assert mask.operational_fallback_applied is False
    assert not mask.is_legal(StrategyAction(ACTION_STAY_OUT))
    assert not mask.is_legal(StrategyAction(ACTION_PIT_NOW, compound="MEDIUM"))
    assert mask.is_legal(StrategyAction(ACTION_PIT_NOW, compound="HARD"))
    assert all(
        not mask.is_legal(action)
        for action in action_space
        if action.action_type == ACTION_PIT_NEXT_LAP
    )

    already_too_late = replace(last_feasible, remaining_laps=2)
    infeasible = build_legal_action_mask(
        already_too_late,
        action_space=action_space,
    )
    fallback = StrategyAction(ACTION_STAY_OUT)
    assert infeasible.constraint_feasible is False
    assert infeasible.operational_fallback_applied is True
    assert infeasible.legal_actions == ()
    assert infeasible.legal_count == 0
    assert not infeasible.is_legal(fallback)
    assert infeasible.is_selectable(fallback)
    assert infeasible.to_payload()["legal_action_keys"] == []
    assert not any(infeasible.to_payload()["constraint_legal_mask"])
    assert infeasible.to_payload()["selectable_action_keys"] == [fallback.key]


def test_invalid_mandatory_change_override_blocks_full_mask_certification() -> None:
    rows = _physical_pit_rows()
    rows["action_mode"] = "conservative"
    rows["pit_lane_open"] = True
    rows["pit_lane_open_known"] = True
    rows["available_compounds"] = "MEDIUM,HARD"
    rows["compound_inventory_known"] = True
    rows["mandatory_compound_change_required"] = "unknown"
    transition = next(
        item
        for item in build_replay_transitions(rows)
        if item.action_t.is_pit_action
    )

    current_evidence = transition.metadata["legal_action_mask_input_evidence"]
    next_evidence = transition.metadata[
        "next_legal_action_mask_input_evidence"
    ]
    assert "mandatory_compound_change_override" in current_evidence["blockers"]
    assert "mandatory_compound_change_override" in next_evidence["blockers"]
    dataset = build_rl_replay_dataset([transition])
    assert dataset.offline_q_examples() == ()
    assert "legal_action_mask_input_unknown:mandatory_compound_change_override" in (
        dataset.examples[0].ood_reasons
    )
    with pytest.raises(ValueError, match="full_legal_action_mask_not_certified"):
        build_rl_replay_dataset([transition], strict=True)


def test_operational_fallback_masks_are_never_offline_q_or_ope_eligible() -> None:
    rows = _replay_rows()
    rows["lap_number"] = [8, 9, 10]
    rows["total_laps"] = 10
    rows["remaining_laps"] = [2, 1, 0]
    rows["stint_id"] = 1
    rows["compound"] = "MEDIUM"
    rows["used_compounds"] = "MEDIUM"
    rows["observed_action"] = "stay_out"
    rows["action_mode"] = "conservative"
    rows["pit_lane_open_known"] = True
    rows["compound_inventory_known"] = True
    rows["behavior_action_support_known"] = True
    rows["behavior_action_probability"] = 0.5
    rows["mandatory_compound_change_required"] = True
    rows["mandatory_compound_change_override_known"] = True
    transition = build_replay_transitions(rows)[0]

    assert transition.done is False
    assert transition.legal_action_mask.constraint_feasible is False
    assert transition.legal_action_mask.operational_fallback_applied is True
    assert transition.metadata["full_legal_action_mask_certified"] is True
    assert transition.metadata["next_full_legal_action_mask_certified"] is True
    assert transition.metadata["policy_training_eligible"] is False
    assert "legal_action_mask_constraint_infeasible" in transition.metadata[
        "policy_training_blockers"
    ]
    assert "next_legal_action_mask_constraint_infeasible" in transition.metadata[
        "policy_training_blockers"
    ]
    dataset = build_rl_replay_dataset([transition])
    reasons = dataset.examples[0].ood_reasons
    assert "legal_action_mask_constraint_infeasible" in reasons
    assert "legal_action_mask_operational_fallback_not_training_eligible" in reasons
    assert "next_legal_action_mask_constraint_infeasible" in reasons
    assert dataset.behavior_cloning_examples() == ()
    assert dataset.offline_q_examples() == ()
    assert dataset.propensity_ope_examples() == ()
    assert not dataset.action_index.legal_mask_for_state(
        transition.state_t
    ).any()
    with pytest.raises(ValueError, match="no constraint-legal"):
        fit_behavior_cloning(dataset).select_action(transition.state_t)
    with pytest.raises(ValueError, match="no constraint-legal"):
        fit_conservative_offline_q(dataset).select_action(transition.state_t)
    with pytest.raises(
        ValueError,
        match="legal_action_mask_constraint_infeasible",
    ):
        build_rl_replay_dataset([transition], strict=True)


def test_strategy_state_preserves_pit_lane_and_box_legality_inputs() -> None:
    state = StrategyState.from_mapping(
        {
            "event_key": 202601,
            "driver_id": "44",
            "lap_number": 20,
            "total_laps": 58,
            "remaining_laps": 38,
            "stint_id": 1,
            "compound": "MEDIUM",
            "tyre_age": 10,
            "used_compounds": "MEDIUM",
            "available_compounds": "HARD",
            "pit_lane_open": False,
            "is_box_lap": True,
        }
    )
    mask = build_legal_action_mask(state)

    assert state.metadata["pit_lane_open"] is False
    assert state.metadata["is_box_lap"] is True
    assert all(not action.is_pit_action for action in mask.legal_actions)


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


def test_physical_stop_is_one_causal_semi_markov_transition() -> None:
    transitions = build_replay_transitions(_physical_pit_rows())

    pit_transitions = [transition for transition in transitions if transition.action_t.is_pit_action]
    assert len(pit_transitions) == 1
    pit = pit_transitions[0]
    assert (pit.state_t.lap_number, pit.state_t1.lap_number) == (1, 4)
    assert pit.action_t.action_type == ACTION_PIT_NOW
    assert pit.action_t.compound == "HARD"
    assert pit.metadata["transition_kind"] == "pit_stop_semi_markov"
    assert pit.metadata["elapsed_laps"] == 3
    assert pit.metadata["pit_in_lap"] == 2
    assert pit.metadata["pit_out_observed"] is True
    assert pit.metadata["end_is_first_post_pit_running_lap"] is True
    assert pit.metadata["action_legality_status"] == "unknown"
    assert pit.metadata["behavior_action_support_status"] == "unknown"
    assert pit.metadata["policy_training_eligible"] is False
    assert pit.metadata["propensity_ope_eligible"] is False
    assert pit.metadata["policy_learning_eligible"] is False
    assert pit.reward_t.components["illegal_action_penalty"] == 0.0
    assert pit.reward_t.value == -311.0
    assert pit.state_t.metadata["available_through_lap"] == 1
    assert pit.state_t.metadata["leakage_contract_version"] == LEAKAGE_CONTRACT_VERSION

    dataset = build_rl_replay_dataset(transitions)
    pit_example = next(example for example in dataset.examples if example.action_key == pit.action_t.key)
    assert pit_example.elapsed_laps == 3
    assert pit_example.ood is True
    assert "observed_action_legality_unknown" in pit_example.ood_reasons
    assert "behavior_action_support_unknown" in (
        pit_example.propensity_ope_ineligible_reasons
    )
    assert "observed_action_illegal_under_mask" not in pit_example.ood_reasons
    with pytest.raises(ValueError, match="not learnable"):
        build_rl_replay_dataset(transitions, strict=True)


def test_semi_markov_pit_state_does_not_leak_post_stop_targets() -> None:
    base = _physical_pit_rows()
    changed = _physical_pit_rows(extra_lap=True)
    changed.loc[changed["lap_number"] >= 2, "final_position"] = 1
    changed.loc[changed["lap_number"] >= 2, "next_actual_lap_time_seconds"] = 10.0

    reference = next(
        transition for transition in build_replay_transitions(base) if transition.action_t.is_pit_action
    )
    candidate = next(
        transition for transition in build_replay_transitions(changed) if transition.action_t.is_pit_action
    )

    assert reference.state_t.fingerprint() == candidate.state_t.fingerprint()
    assert reference.fingerprint() == candidate.fingerprint()
    assert "final_position" in reference.state_t.metadata["ignored_future_columns"]
    assert reference.state_t.metadata["available_through_lap"] == 1


def test_zero_behavior_propensity_blocks_ope_but_not_policy_training() -> None:
    rows = _physical_pit_rows()
    rows["action_mode"] = "conservative"
    rows["pit_lane_open"] = True
    rows["pit_lane_open_known"] = True
    rows["available_compounds"] = "MEDIUM,HARD"
    rows["compound_inventory_known"] = True
    rows["behavior_action_probability"] = 0.0

    pit = next(
        transition
        for transition in build_replay_transitions(rows)
        if transition.action_t.is_pit_action
    )
    dataset = build_rl_replay_dataset([pit])

    assert pit.metadata["action_legality_status"] == "known_legal"
    assert pit.metadata["behavior_action_support_status"] == "zero_support"
    assert pit.metadata["policy_training_eligible"] is True
    assert pit.metadata["policy_learning_eligible"] is True
    assert pit.metadata["propensity_ope_eligible"] is False
    assert dataset.examples[0].ood is False
    assert "behavior_action_zero_support" in (
        dataset.examples[0].propensity_ope_ineligible_reasons
    )
    assert len(dataset.learning_examples()) == 1
    assert dataset.propensity_ope_examples() == ()

    supported_rows = rows.copy()
    supported_rows["behavior_action_support_known"] = True
    supported_rows["behavior_action_probability"] = 0.2
    supported_pit = next(
        transition
        for transition in build_replay_transitions(supported_rows)
        if transition.action_t.is_pit_action
    )
    assert supported_pit.metadata["policy_training_eligible"] is True
    assert supported_pit.metadata["propensity_ope_eligible"] is True
    assert supported_pit.metadata["policy_learning_eligible"] is True
    supported_dataset = build_rl_replay_dataset([supported_pit])
    assert len(supported_dataset.learning_examples()) == 1
    assert len(supported_dataset.propensity_ope_examples()) == 1
    assert supported_pit.fingerprint() != pit.fingerprint()


def test_offline_q_requires_observed_weighted_reward_components() -> None:
    rows = _physical_pit_rows()
    rows["action_mode"] = "conservative"
    rows["pit_lane_open"] = True
    rows["pit_lane_open_known"] = True
    rows["available_compounds"] = "MEDIUM,HARD"
    rows["compound_inventory_known"] = True
    rows.loc[rows["lap_number"].eq(4), "race_time_seconds"] = np.nan

    incomplete = next(
        transition
        for transition in build_replay_transitions(rows)
        if transition.action_t.is_pit_action
    )
    incomplete_dataset = build_rl_replay_dataset([incomplete])
    assert incomplete.metadata["reward_observation_status"] == "incomplete"
    assert "elapsed_race_time_endpoints_unobserved" in incomplete.metadata[
        "reward_observation_blockers"
    ]
    assert incomplete.metadata["policy_training_eligible"] is False
    assert "reward_observation_not_certified" in incomplete_dataset.examples[0].ood_reasons
    assert incomplete_dataset.offline_q_examples() == ()

    no_position = _physical_pit_rows().drop(columns=["position"])
    no_position["action_mode"] = "conservative"
    no_position["pit_lane_open"] = True
    no_position["pit_lane_open_known"] = True
    no_position["available_compounds"] = "MEDIUM,HARD"
    no_position["compound_inventory_known"] = True
    time_only = next(
        transition
        for transition in build_replay_transitions(
            no_position,
            reward_config=RewardConfig(position_gain_weight=0.0),
        )
        if transition.action_t.is_pit_action
    )
    assert time_only.metadata["reward_observation_status"] == (
        "observed_required_components"
    )
    assert time_only.metadata["policy_training_eligible"] is True


def test_propensity_ope_carries_numeric_probability_and_fingerprints_it() -> None:
    rows = _physical_pit_rows()
    rows["action_mode"] = "conservative"
    rows["pit_lane_open"] = True
    rows["pit_lane_open_known"] = True
    rows["available_compounds"] = "MEDIUM,HARD"
    rows["compound_inventory_known"] = True
    rows["behavior_action_support_known"] = True
    rows["behavior_action_probability"] = 0.2
    low_probability = next(
        transition
        for transition in build_replay_transitions(rows)
        if transition.action_t.is_pit_action
    )
    rows["behavior_action_probability"] = 0.8
    high_probability = next(
        transition
        for transition in build_replay_transitions(rows)
        if transition.action_t.is_pit_action
    )

    assert low_probability.fingerprint() != high_probability.fingerprint()
    low_dataset = build_rl_replay_dataset([low_probability])
    high_dataset = build_rl_replay_dataset([high_probability])
    assert low_dataset.examples[0].behavior_action_probability == pytest.approx(0.2)
    assert high_dataset.examples[0].behavior_action_probability == pytest.approx(0.8)
    assert low_dataset.behavior_action_probabilities.tolist() == pytest.approx([0.2])
    assert len(low_dataset.propensity_ope_examples()) == 1

    missing_numeric = replace(
        low_probability,
        metadata={
            **low_probability.metadata,
            "behavior_action_probability": None,
            "behavior_action_support_status": "known_positive",
            "behavior_action_support_known": True,
            "propensity_ope_eligible": True,
        },
    )
    missing_dataset = build_rl_replay_dataset([missing_numeric])
    assert missing_dataset.propensity_ope_examples() == ()
    assert "behavior_action_probability_missing" in (
        missing_dataset.examples[0].propensity_ope_ineligible_reasons
    )


def test_training_rejects_noncausal_gap_and_pit_without_physical_boundary() -> None:
    gap_rows = pd.DataFrame(
        [
            {
                "event_key": 202601,
                "driver_id": "44",
                "lap_number": 1,
                "total_laps": 10,
                "remaining_laps": 9,
                "stint_id": 1,
                "compound": "MEDIUM",
                "available_compounds": "MEDIUM,HARD",
                "pit_lane_open": True,
                "observed_action": "stay_out",
            },
            {
                "event_key": 202601,
                "driver_id": "44",
                "lap_number": 4,
                "total_laps": 10,
                "remaining_laps": 6,
                "stint_id": 1,
                "compound": "MEDIUM",
                "available_compounds": "MEDIUM,HARD",
                "pit_lane_open": True,
                "observed_action": "stay_out",
            },
        ]
    )
    gap_transition = build_replay_transitions(gap_rows)[0]
    gap_dataset = build_rl_replay_dataset([gap_transition])
    assert gap_transition.metadata["causal_transition_boundary_status"] == "invalid"
    assert "causal_transition_boundary_invalid" in gap_dataset.examples[0].ood_reasons
    with pytest.raises(ValueError, match="not learnable"):
        build_rl_replay_dataset([gap_transition], strict=True)

    duplicate_lap_rows = gap_rows.copy()
    duplicate_lap_rows.loc[1, "lap_number"] = 1
    duplicate_lap_rows.loc[0, "timestamp"] = 1.0
    duplicate_lap_rows.loc[1, "timestamp"] = 2.0
    duplicate_transition = build_replay_transitions(duplicate_lap_rows)[0]
    duplicate_dataset = build_rl_replay_dataset([duplicate_transition])
    assert duplicate_transition.metadata["raw_lap_delta"] == 0
    assert (
        duplicate_transition.metadata["causal_transition_boundary_status"]
        == "invalid"
    )
    assert "causal_transition_boundary_blocker:adjacent_transition_is_not_one_lap" in (
        duplicate_dataset.examples[0].ood_reasons
    )

    unsupported_pit = next(
        transition
        for transition in build_replay_transitions(_replay_rows())
        if transition.action_t.is_pit_action
    )
    pit_dataset = build_rl_replay_dataset([unsupported_pit])
    assert unsupported_pit.metadata["causal_transition_boundary_status"] == "invalid"
    assert "causal_transition_boundary_blocker:pit_action_missing_physical_pit_boundary_evidence" in (
        pit_dataset.examples[0].ood_reasons
    )


def test_explicit_negative_eligibility_never_fails_open_without_blockers() -> None:
    transition = build_replay_transitions(_replay_rows())[0]
    declared_training_ineligible = replace(
        transition,
        metadata={
            **transition.metadata,
            "policy_training_eligible": False,
            "policy_training_blockers": (),
            "policy_learning_eligible": False,
            "policy_learning_blockers": (),
        },
    )
    with pytest.raises(ValueError, match="policy_training_declared_ineligible"):
        build_rl_replay_dataset([declared_training_ineligible], strict=True)

    propensity_rows = _replay_rows().iloc[:2].copy()
    propensity_rows["action_mode"] = "conservative"
    propensity_rows["pit_lane_open_known"] = True
    propensity_rows["compound_inventory_known"] = True
    propensity_rows["behavior_action_support_known"] = True
    propensity_rows["behavior_action_probability"] = 0.4
    supported = build_replay_transitions(propensity_rows)[0]
    declared_ope_ineligible = replace(
        supported,
        metadata={
            **supported.metadata,
            "propensity_ope_eligible": False,
            "propensity_ope_blockers": (),
        },
    )
    dataset = build_rl_replay_dataset([declared_ope_ineligible], strict=True)
    assert len(dataset.learning_examples()) == 1
    assert dataset.propensity_ope_examples() == ()
    assert "propensity_ope_declared_ineligible" in (
        dataset.examples[0].propensity_ope_ineligible_reasons
    )

    synthetic_record = ReplayBufferRecord.from_transition(
        replace(
            transition,
            metadata={
                key: value
                for key, value in transition.metadata.items()
                if key
                not in {
                    "behavior_action_probability",
                    "propensity_ope_eligible",
                }
            },
        ),
        source="synthetic_self_play",
    )
    assert not replay_buffer_from_transitions(
        [synthetic_record.transition], source="synthetic_self_play"
    ).to_frame()["propensity_ope_eligible"].iloc[0]


@pytest.mark.parametrize("noncanonical_false", [np.bool_(False), 0, "false"])
def test_noncanonical_false_eligibility_values_fail_closed(
    noncanonical_false: object,
) -> None:
    rows = _replay_rows().iloc[:2].copy()
    rows["action_mode"] = "conservative"
    rows["pit_lane_open_known"] = True
    rows["compound_inventory_known"] = True
    rows["behavior_action_support_known"] = True
    rows["behavior_action_probability"] = 0.4
    transition = build_replay_transitions(rows)[0]

    training_tamper = replace(
        transition,
        metadata={
            **transition.metadata,
            "policy_training_eligible": noncanonical_false,
            "policy_training_blockers": (),
        },
    )
    with pytest.raises(ValueError, match="policy_training_declared_ineligible"):
        build_rl_replay_dataset([training_tamper], strict=True)

    ope_tamper = replace(
        transition,
        metadata={
            **transition.metadata,
            "propensity_ope_eligible": noncanonical_false,
            "propensity_ope_blockers": (),
        },
    )
    dataset = build_rl_replay_dataset([ope_tamper], strict=True)
    assert len(dataset.learning_examples()) == 1
    assert dataset.propensity_ope_examples() == ()
    assert "propensity_ope_declared_ineligible" in (
        dataset.examples[0].propensity_ope_ineligible_reasons
    )


def test_declared_known_legal_action_must_match_computed_legal_mask() -> None:
    pit = next(
        transition
        for transition in build_replay_transitions(_physical_pit_rows())
        if transition.action_t.is_pit_action
    )
    assert not pit.is_action_legal()
    tampered = replace(
        pit,
        metadata={
            **pit.metadata,
            "action_legality_status": "known_legal",
            "action_legality_known": True,
            "action_legality_unknown_reasons": (),
            "policy_training_eligible": True,
            "policy_training_blockers": (),
            "policy_learning_eligible": True,
            "policy_learning_blockers": (),
        },
    )

    with pytest.raises(
        ValueError,
        match="observed_action_legality_mask_conflict",
    ):
        build_rl_replay_dataset([tampered], strict=True)


def test_incomplete_pit_evidence_fails_closed_without_invented_action_labels() -> None:
    unknown_compound = _physical_pit_rows()
    unknown_compound.loc[unknown_compound["lap_number"] >= 3, "compound"] = "UNKNOWN"
    unknown_compound.loc[unknown_compound["lap_number"] >= 3, "stint_id"] = 2

    unmatched_pit_out = _physical_pit_rows()
    unmatched_pit_out["is_pit_in_lap"] = False

    pit_in_only = _physical_pit_rows()
    pit_in_only["is_pit_out_lap"] = False

    assert not any(
        transition.action_t.is_pit_action
        for transition in build_replay_transitions(unknown_compound)
    )
    assert not any(
        transition.action_t.is_pit_action
        for transition in build_replay_transitions(unmatched_pit_out)
    )
    assert not any(
        transition.action_t.is_pit_action
        for transition in build_replay_transitions(pit_in_only)
    )


def test_replay_is_event_disjoint_and_versions_changed_record_semantics() -> None:
    first = _replay_rows()
    second = _replay_rows()
    second["event_key"] = 202602
    combined = pd.concat([first, second], ignore_index=True)

    transitions = build_replay_transitions(combined)

    assert len(transitions) == 4
    assert all(
        transition.state_t.event_key == transition.state_t1.event_key
        for transition in transitions
    )
    record_payload = ReplayBufferRecord.from_transition(transitions[0]).to_payload()
    assert record_payload["record_schema_version"] == (
        "live_strategy_replay_record_v7_full_current_next_mask_input_and_feasibility_evidence"
    )
    assert record_payload["transition_fingerprint_version"] == TRANSITION_FINGERPRINT_VERSION
    dataset = build_rl_replay_dataset(transitions)
    assert dataset.metadata["dataset_builder"] == (
        "live_strategy_rl_replay_v7_full_current_next_mask_input_and_feasibility_evidence"
    )
    assert dataset.metadata["bellman_discount_semantics"] == (
        "aggregated_multi_lap_rewards_require_gamma_one_unless_per_lap_rewards_exist"
    )


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


def test_missing_mode_is_partial_bc_label_and_never_offline_q_evidence() -> None:
    dataset = build_rl_replay_dataset(
        build_replay_transitions(_physical_pit_rows())
    )
    pit = next(
        example
        for example in dataset.examples
        if example.action_key.startswith("pit_now:HARD")
    )
    compatible = {
        dataset.action_index.action_for(int(idx)).key
        for idx in np.flatnonzero(pit.behavior_cloning_action_mask)
    }

    assert pit.behavior_cloning_label_kind == "coarsened_missing_mode"
    assert compatible == {
        "pit_now:HARD:conservative",
        "pit_now:HARD:aggressive",
    }
    assert pit in dataset.behavior_cloning_examples()
    assert dataset.offline_q_examples() == ()
    assert dataset.propensity_ope_examples() == ()
    support = dataset.action_support_diagnostics()
    assert support["behavior_cloning"]["pace_mode_evidence_rows"] == 0
    assert support["offline_q"]["rows"] == 0
    evaluation = evaluate_behavior_cloning(fit_behavior_cloning(dataset), dataset)
    assert evaluation["metrics"]["illegal_prediction_rate"] is None
    assert evaluation["metrics"]["legal_mask_certified_rows"] == 0
    assert evaluation["metrics"]["legal_mask_unknown_rows"] == len(
        dataset.behavior_cloning_examples()
    )


def test_reward_value_is_bound_into_v7_transition_fingerprint() -> None:
    transition = build_replay_transitions(_replay_rows())[0]
    changed_reward = replace(
        transition,
        reward_t=replace(
            transition.reward_t,
            value=transition.reward_t.value - 1.0,
        ),
    )

    assert transition.fingerprint() != changed_reward.fingerprint()


def test_nonterminal_offline_q_requires_certified_next_state_mask() -> None:
    rows = _physical_pit_rows()
    rows["action_mode"] = "conservative"
    rows["pit_lane_open"] = True
    rows["pit_lane_open_known"] = True
    rows["available_compounds"] = "MEDIUM,HARD"
    rows["compound_inventory_known"] = True
    transition = next(
        item
        for item in build_replay_transitions(rows)
        if item.action_t.is_pit_action
    )

    assert transition.done is False
    assert transition.metadata[
        "next_full_legal_action_mask_certified"
    ] is True
    assert transition.metadata["next_legal_action_mask"]
    assert transition.metadata["next_legal_action_mask_fingerprint"]
    assert len(build_rl_replay_dataset([transition]).offline_q_examples()) == 1

    uncertified_next_state = replace(
        transition.state_t1,
        metadata={
            **transition.state_t1.metadata,
            "pit_lane_open_known": False,
            "compound_inventory_known": False,
        },
    )
    uncertified = replace(
        transition,
        state_t1=uncertified_next_state,
    )
    dataset = build_rl_replay_dataset([uncertified])
    assert "next_full_legal_action_mask_not_certified" in (
        dataset.examples[0].ood_reasons
    )
    assert dataset.offline_q_examples() == ()
    with pytest.raises(
        ValueError,
        match="next_full_legal_action_mask_not_certified",
    ):
        build_rl_replay_dataset([uncertified], strict=True)

    tampered_payload = replace(
        transition,
        metadata={
            **transition.metadata,
            "next_legal_action_mask": {
                **transition.metadata["next_legal_action_mask"],
                "mask": [False]
                * len(transition.metadata["next_legal_action_mask"]["mask"]),
            },
        },
    )
    assert tampered_payload.fingerprint() != transition.fingerprint()
    with pytest.raises(
        ValueError,
        match="next_legal_action_mask_evidence_mismatch",
    ):
        build_rl_replay_dataset([tampered_payload], strict=True)

    tampered_current_mask = transition.legal_action_mask.mask.copy()
    tampered_current_mask[-1] = not bool(tampered_current_mask[-1])
    tampered_current = replace(
        transition,
        legal_action_mask=replace(
            transition.legal_action_mask,
            mask=tampered_current_mask,
        ),
    )
    with pytest.raises(
        ValueError,
        match="legal_action_mask_evidence_mismatch",
    ):
        build_rl_replay_dataset([tampered_current], strict=True)


@pytest.mark.parametrize(
    ("dropped_columns", "expected_blocker"),
    [
        (("track_status",), "red_flag_status"),
        (("used_compounds",), "used_compound_history"),
        (("total_laps", "remaining_laps"), "race_horizon"),
        (("is_box_lap",), "box_lap_status"),
        (
            ("forced_pit_commitment_known", "forced_pit_next_compound"),
            "forced_pit_commitment",
        ),
    ],
)
def test_offline_q_fails_closed_when_any_next_mask_input_is_unobserved(
    dropped_columns: tuple[str, ...],
    expected_blocker: str,
) -> None:
    rows = _physical_pit_rows().drop(columns=list(dropped_columns))
    rows["action_mode"] = "conservative"
    rows["pit_lane_open"] = True
    rows["pit_lane_open_known"] = True
    rows["available_compounds"] = "MEDIUM,HARD"
    rows["compound_inventory_known"] = True
    transition = next(
        item
        for item in build_replay_transitions(rows)
        if item.action_t.is_pit_action
    )

    evidence = transition.metadata["next_legal_action_mask_input_evidence"]
    assert evidence["certified"] is False
    assert expected_blocker in evidence["blockers"]
    dataset = build_rl_replay_dataset([transition])
    assert dataset.offline_q_examples() == ()
    assert "next_full_legal_action_mask_not_certified" in (
        dataset.examples[0].ood_reasons
    )
    assert (
        f"next_legal_action_mask_input_unknown:{expected_blocker}"
        in dataset.examples[0].ood_reasons
    )
    with pytest.raises(
        ValueError,
        match="next_full_legal_action_mask_not_certified",
    ):
        build_rl_replay_dataset([transition], strict=True)


def test_certified_forced_pit_commitment_is_bound_into_next_mask() -> None:
    rows = _replay_rows()
    rows["action_mode"] = "conservative"
    rows["pit_lane_open_known"] = True
    rows["compound_inventory_known"] = True
    rows.loc[rows.index[1], "forced_pit_next_compound"] = "HARD"
    transition = build_replay_transitions(rows)[0]

    assert transition.done is False
    assert transition.metadata[
        "next_full_legal_action_mask_certified"
    ] is True
    next_legal_keys = set(
        transition.metadata["next_legal_action_mask"]["legal_action_keys"]
    )
    assert next_legal_keys == {
        "pit_now:HARD:conservative",
        "pit_now:HARD:aggressive",
    }
    assert len(build_rl_replay_dataset([transition]).offline_q_examples()) == 1


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


def test_stint_fallback_resets_per_driver_in_interleaved_lap_table() -> None:
    frame = pd.DataFrame(
        {
            "driver_id": ["A", "B", "A", "B", "A", "B"],
            "lap_number": [1, 1, 2, 2, 3, 3],
            "is_pit_in_lap": [False, False, True, False, False, False],
            "is_pit_out_lap": [False, False, False, False, False, False],
        }
    )

    stint = _build_stint_id(frame)

    assert stint.loc[[0, 2, 4]].tolist() == [1, 2, 2]
    assert stint.loc[[1, 3, 5]].tolist() == [1, 1, 1]


def test_legacy_lap_replay_without_policy_evidence_fails_learning_closed() -> None:
    transition = build_replay_transitions(_replay_rows(extra_lap=False))[0]
    stripped_metadata = {
        key: value
        for key, value in transition.metadata.items()
        if key
        not in {
            "action_legality_status",
            "behavior_action_support_status",
            "policy_training_eligible",
            "policy_training_blockers",
            "propensity_ope_eligible",
            "propensity_ope_blockers",
            "policy_learning_eligible",
            "policy_learning_blockers",
        }
    }
    legacy = replace(transition, metadata=stripped_metadata)
    record = ReplayBufferRecord.from_transition(legacy, source="lap_replay")

    dataset = build_rl_replay_dataset([record])
    reasons = set(dataset.examples[0].ood_reasons)
    propensity_reasons = set(
        dataset.examples[0].propensity_ope_ineligible_reasons
    )

    assert "observed_action_legality_evidence_missing" in reasons
    assert "policy_training_eligibility_evidence_missing" in reasons
    assert "behavior_action_support_evidence_missing" in propensity_reasons
    assert "propensity_ope_eligibility_evidence_missing" in propensity_reasons
    assert dataset.learning_examples() == ()
    assert not replay_buffer_from_transitions([legacy]).to_frame()[
        "policy_learning_eligible"
    ].iloc[0]
