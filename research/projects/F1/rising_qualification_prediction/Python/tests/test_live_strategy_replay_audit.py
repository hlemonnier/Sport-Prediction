from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

import run_live_strategy_replay_audit as replay_audit


def test_pit_lane_status_uses_only_strictly_prior_entry_messages() -> None:
    frame = pd.DataFrame(
        {
            "driver_id": ["1", "1", "1", "1"],
            "lap_number": [10, 11, 12, 13],
        }
    )
    messages = pd.DataFrame(
        {
            "Time": [
                "2026-01-01 12:00:00",
                "2026-01-01 12:00:01",
                "2026-01-01 12:01:00",
                "2026-01-01 12:02:00",
            ],
            "Lap": [9, 10, 11, 12],
            "Message": [
                "PIT EXIT CLOSED",
                "PIT LANE ENTRY CLOSED",
                "PIT LANE ENTRY OPEN",
                "PIT LANE ENTRY CLOSED",
            ],
        }
    )

    observed, diagnostics = replay_audit.apply_causal_pit_lane_status(frame, messages)

    # Pit-exit status is a different rule, so lap 10 remains unknown.  Every
    # entry signal becomes visible one completed lap later, never on its own lap.
    assert observed["pit_lane_open_known"].tolist() == [False, True, True, True]
    assert observed["pit_lane_open"].tolist() == [False, False, True, False]
    assert observed["pit_lane_status_source_lap"].tolist()[1:] == [10.0, 11.0, 12.0]
    assert diagnostics["pit_lane_entry_signal_rows"] == 3
    assert diagnostics["ignored_pit_exit_messages"] == 1
    assert diagnostics["minimum_source_lap_lag"] == 1
    assert diagnostics["strictly_prior_source_lap_contract"] is True


def test_frozen_input_discovery_binds_nine_laps_and_nine_control_files() -> None:
    root = replay_audit._repo_root()
    pairs = replay_audit._input_pairs(
        root,
        weekends_dir=root / "data/f1/raw/weekends",
    )

    assert [round_number for round_number, _, _ in pairs] == list(range(1, 10))
    assert len({laps for _, laps, _ in pairs}) == 9
    assert len({messages for _, _, messages in pairs}) == 9
    assert all(laps.name.endswith("_race_laps.csv") for _, laps, _ in pairs)
    assert all(
        messages.name.endswith("_race_race_control_messages.csv")
        for _, _, messages in pairs
    )


def test_replay_audit_exports_constraint_legality_contract_version() -> None:
    assert replay_audit.LEGAL_ACTION_MASK_SCHEMA_VERSION == (
        "live_strategy_legal_action_mask_v3_"
        "sporting_deadlines_derived_from_action_timing"
    )


def test_locked_count_gate_fails_closed_on_any_replay_drift() -> None:
    expected = replay_audit.LOCKED_COUNTS
    summary = {
        "rounds": expected["rounds"],
        "transitions": expected["transitions"],
        "behavior_cloning_rows": expected["behavior_cloning_rows"],
        "offline_q_rows": expected["offline_q_rows"],
        "propensity_ope_rows": expected["propensity_ope_rows"],
        "states_with_multiple_used_compounds": expected[
            "states_with_multiple_used_compounds"
        ],
        "action_family_counts": deepcopy(expected["action_family_counts"]),
        "behavior_cloning_action_support": {
            "compatible_action_key_count": expected["compatible_action_key_count"],
            "exact_action_key_count": expected["exact_action_key_count"],
            "supported_action_family_count": expected[
                "supported_action_family_count"
            ],
        },
        "reward": {
            "elapsed_time_positive_rows": expected["elapsed_time_positive_rows"],
            "reward_components_fully_observed_rows": expected[
                "reward_components_fully_observed_rows"
            ],
            "reward_components_incomplete_rows": expected[
                "reward_components_incomplete_rows"
            ],
            "nonzero_position_gain_rows": expected["nonzero_position_gain_rows"],
        },
    }
    replay_audit._assert_locked_counts(summary)

    changed = deepcopy(summary)
    changed["offline_q_rows"] = 1
    with pytest.raises(RuntimeError, match="refusing to publish"):
        replay_audit._assert_locked_counts(changed)
