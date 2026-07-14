from __future__ import annotations

import pandas as pd
import pytest

from packages.f1.models.live_race.predict import (
    _blend_next_lap_point_forecast,
    _prior_observations_for_baseline,
)
from packages.f1.models.live_race.environment import build_replay_transitions
from packages.f1.models.live_race.sources import _standardize_laps


def test_missing_clock_time_is_not_relabelled_as_global_timestamp() -> None:
    raw = pd.DataFrame(
        {
            "Driver": ["AAA", "AAA"],
            "LapNumber": [1, 2],
            "LapTime": [90.0, 91.0],
        }
    )

    result = _standardize_laps(raw, event_key=202601, source_used="test", session_name="Race")

    assert result["timestamp"].isna().all()
    assert not result["timestamp_known"].any()
    assert result["timestamp_source"].eq("unavailable").all()


def test_source_preserves_distinct_pit_in_and_pit_out_signals() -> None:
    raw = pd.DataFrame(
        {
            "Driver": ["AAA"] * 5,
            "LapNumber": [1, 2, 3, 4, 5],
            "LapTime": [90.0, 112.0, 108.0, 91.0, 91.5],
            "Compound": ["MEDIUM", "MEDIUM", "HARD", "HARD", "HARD"],
            "is_pit_in_lap": [False, True, False, False, False],
            "is_pit_out_lap": [False, False, True, False, False],
        }
    )

    result = _standardize_laps(raw, event_key=202601, source_used="test", session_name="Race")

    assert result.loc[result["is_pit_in_lap"], "lap_number"].tolist() == [2]
    assert result.loc[result["is_pit_out_lap"], "lap_number"].tolist() == [3]
    assert result.loc[result["is_box_lap"], "lap_number"].tolist() == [2, 3]
    assert result["pit_in_signal_known"].all()
    assert result["pit_out_signal_known"].all()
    assert not result["pit_lane_open_known"].any()
    assert not result["compound_inventory_known"].any()
    assert not result["behavior_action_support_known"].any()


def test_legality_and_behavior_support_knownness_is_row_wise_and_positive() -> None:
    raw = pd.DataFrame(
        {
            "Driver": ["AAA"] * 3,
            "LapNumber": [1, 2, 3],
            "LapTime": [90.0, 91.0, 92.0],
            "pit_lane_open": [True, None, False],
            "available_compounds": ["MEDIUM,HARD", None, ""],
            "behavior_action_probability": [0.25, None, 0.0],
        }
    )

    result = _standardize_laps(
        raw,
        event_key=202601,
        source_used="test",
        session_name="Race",
    )

    assert result["pit_lane_open_known"].tolist() == [True, False, True]
    assert result["compound_inventory_known"].tolist() == [True, False, False]
    assert result["behavior_action_support_known"].tolist() == [True, False, False]


def test_completed_lap_running_position_is_preserved_for_state_and_reward() -> None:
    raw = pd.DataFrame(
        {
            "Driver": ["AAA", "AAA"],
            "LapNumber": [1, 2],
            "LapTime": [90.0, 91.0],
            "Position": [5.0, 3.0],
        }
    )

    result = _standardize_laps(
        raw,
        event_key=202601,
        source_used="test",
        session_name="Race",
    )

    assert result["position"].tolist() == [5.0, 3.0]
    transitions = build_replay_transitions(result)
    assert len(transitions) == 1
    assert transitions[0].state_t.position == 5.0
    assert transitions[0].state_t1.position == 3.0
    assert transitions[0].reward_t.components["position_gain"] == 2.0
    assert transitions[0].reward_t.value == pytest.approx(-87.0)


def test_source_preserves_physical_tyre_life_and_expanding_compound_history() -> None:
    raw = pd.DataFrame(
        {
            "Driver": ["AAA", "BBB", "AAA", "BBB", "AAA"],
            "LapNumber": [1, 1, 2, 2, 3],
            "LapTime": [90.0, 91.0, None, 92.0, 93.0],
            "Compound": ["MEDIUM", "SOFT", "MEDIUM", "SOFT", "HARD"],
            "Stint": [1, 1, 1, 1, 2],
            "TyreLife": [7.0, 1.0, 8.0, 2.0, 4.0],
            "IsAccurate": [True, True, False, True, True],
            "is_pit_in_lap": [False, False, True, False, False],
        }
    )

    result = _standardize_laps(
        raw,
        event_key=202601,
        source_used="test",
        session_name="Race",
    )
    aaa = result.loc[result["driver_id"].eq("AAA")].sort_values("lap_number")
    bbb = result.loc[result["driver_id"].eq("BBB")].sort_values("lap_number")

    # The used set began this stint at age seven; neither an inaccurate lap nor
    # a box lap makes the physical tyre younger.
    assert aaa["tyre_age"].tolist() == [7, 8, 4]
    assert aaa["used_compounds"].tolist() == [
        ("MEDIUM",),
        ("MEDIUM",),
        ("MEDIUM", "HARD"),
    ]
    assert bbb["used_compounds"].tolist() == [("SOFT",), ("SOFT",)]


def test_source_tyre_age_fallback_counts_every_completed_stint_lap() -> None:
    raw = pd.DataFrame(
        {
            "Driver": ["AAA"] * 4,
            "LapNumber": [1, 2, 3, 4],
            "LapTime": [90.0, None, 120.0, 91.0],
            "Compound": ["MEDIUM"] * 4,
            "Stint": [1] * 4,
            "IsAccurate": [True, False, False, True],
            "is_pit_in_lap": [False, False, True, False],
        }
    )

    result = _standardize_laps(
        raw,
        event_key=202601,
        source_used="test",
        session_name="Race",
    ).sort_values("lap_number")

    assert result["tyre_age"].tolist() == [1, 2, 3, 4]


def test_observed_completed_lap_time_preserves_elapsed_reward_when_lap_time_missing() -> None:
    raw = pd.DataFrame(
        {
            "Driver": ["AAA", "AAA"],
            "LapNumber": [1, 2],
            "LapTime": [90.0, None],
            "Time": [90.0, 205.0],
            "Position": [5, 4],
        }
    )

    result = _standardize_laps(
        raw,
        event_key=202601,
        source_used="test",
        session_name="Race",
    ).sort_values("lap_number")
    transition = build_replay_transitions(result)[0]

    assert result["race_time_seconds"].tolist() == [90.0, 205.0]
    assert transition.reward_t.components["race_time_delta_seconds"] == 115.0
    assert transition.reward_t.components["position_gain"] == 1.0
    assert transition.reward_t.value == pytest.approx(-113.0)
    assert transition.metadata["reward_observation_status"] == (
        "observed_required_components"
    )


def test_cross_driver_baseline_uses_global_event_time_not_lap_number() -> None:
    observations = pd.DataFrame(
        {
            "driver_id": ["current", "late_lapped", "already_known"],
            "lap_number": [10, 9, 9],
            "timestamp": [100.0, 110.0, 90.0],
            "timestamp_known": [True, True, True],
            "lap_time_seconds": [90.0, 92.0, 91.0],
        }
    )

    prior, mode = _prior_observations_for_baseline(
        observations,
        lap_number=10,
        timestamp=100.0,
        timestamp_known=True,
    )

    assert mode == "global_event_time"
    assert prior["driver_id"].tolist() == ["already_known"]


def test_unknown_event_time_fallback_is_explicitly_non_global() -> None:
    observations = pd.DataFrame(
        {
            "driver_id": ["prior_lap", "same_lap"],
            "lap_number": [4, 5],
            "timestamp": [float("nan"), float("nan")],
            "timestamp_known": [False, False],
        }
    )

    prior, mode = _prior_observations_for_baseline(
        observations,
        lap_number=5,
        timestamp=float("nan"),
        timestamp_known=False,
    )

    assert mode == "lap_number_fallback_not_global_time"
    assert prior["driver_id"].tolist() == ["prior_lap"]


def test_next_lap_blend_is_causal_and_falls_back_when_history_is_missing() -> None:
    assert _blend_next_lap_point_forecast(90.0, 92.0, ssm_weight=0.45) == pytest.approx(91.1)
    assert _blend_next_lap_point_forecast(
        90.0,
        float("nan"),
        ssm_weight=0.45,
    ) == 90.0
