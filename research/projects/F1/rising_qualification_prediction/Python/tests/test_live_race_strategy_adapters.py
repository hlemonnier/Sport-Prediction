from __future__ import annotations

import pandas as pd

from packages.f1.models.live_race.strategy import (
    BaselineStrategyPolicyAdapter,
    BaselineTelemetryFeatureAdapter,
    NoopStrategyPolicyAdapter,
    NoopTelemetryFeatureAdapter,
)


def _sample_laps() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "driver_id": "1",
                "lap_number": 1,
                "stint_id": 1,
                "compound": "MEDIUM",
                "tyre_age": 0,
                "lap_time_seconds": 90.0,
                "race_time_seconds": 90.0,
                "gap_to_leader_seconds": 0.0,
                "track_status": "1",
                "is_box_lap": False,
                "is_accurate": True,
                "timestamp": 1.0,
            },
            {
                "driver_id": "2",
                "lap_number": 1,
                "stint_id": 1,
                "compound": "HARD",
                "tyre_age": 0,
                "lap_time_seconds": 90.4,
                "race_time_seconds": 90.4,
                "gap_to_leader_seconds": 0.4,
                "track_status": "1",
                "is_box_lap": False,
                "is_accurate": True,
                "timestamp": 1.0,
            },
            {
                "driver_id": "1",
                "lap_number": 2,
                "stint_id": 1,
                "compound": "MEDIUM",
                "tyre_age": 1,
                "lap_time_seconds": 90.9,
                "race_time_seconds": 180.9,
                "gap_to_leader_seconds": 0.0,
                "track_status": "1",
                "is_box_lap": False,
                "is_accurate": True,
                "timestamp": 2.0,
            },
            {
                "driver_id": "2",
                "lap_number": 2,
                "stint_id": 1,
                "compound": "HARD",
                "tyre_age": 1,
                "lap_time_seconds": 90.7,
                "race_time_seconds": 181.1,
                "gap_to_leader_seconds": 0.2,
                "track_status": "2",
                "is_box_lap": False,
                "is_accurate": True,
                "timestamp": 2.0,
            },
            {
                "driver_id": "1",
                "lap_number": 3,
                "stint_id": 1,
                "compound": "MEDIUM",
                "tyre_age": 2,
                "lap_time_seconds": 91.8,
                "race_time_seconds": 272.7,
                "gap_to_leader_seconds": 0.0,
                "track_status": "4",
                "is_box_lap": False,
                "is_accurate": True,
                "timestamp": 3.0,
            },
        ]
    )


def test_telemetry_adapter_builds_live_lap_features() -> None:
    features = BaselineTelemetryFeatureAdapter().build_lap_features(_sample_laps())

    assert list(features.index) == [0, 1, 2, 3, 4]
    assert "tyre_life_used_ratio" in features.columns
    assert "rolling_clean_pace_delta_3" in features.columns
    assert "estimated_deg_slope_5" in features.columns
    assert features.loc[0, "compound_normalized"] == "MEDIUM"
    assert features.loc[1, "compound_normalized"] == "HARD"
    assert bool(features.loc[3, "is_yellow"]) is True
    assert bool(features.loc[4, "is_sc_vsc"]) is True
    assert float(features.loc[2, "tyre_life_used_ratio"]) > float(features.loc[0, "tyre_life_used_ratio"])
    assert pd.notna(features.loc[2, "rolling_clean_pace_delta_3"])


def test_strategy_policy_scores_pit_actions_from_live_state() -> None:
    state = pd.DataFrame(
        [
            {
                "driver_id": "early",
                "compound": "HARD",
                "tyre_age": 4,
                "stint_id": 2,
                "deg_rate_mean": 0.02,
                "pace_penalty_mean": 0.0,
                "track_status": "1",
                "lap_last": 10,
                "race_total_laps": 58,
            },
            {
                "driver_id": "urgent",
                "compound": "SOFT",
                "tyre_age": 19,
                "stint_id": 1,
                "deg_rate_mean": 0.08,
                "pace_penalty_mean": 1.5,
                "track_status": "4",
                "lap_last": 28,
                "race_total_laps": 58,
            },
        ]
    )

    actions = BaselineStrategyPolicyAdapter().evaluate_actions(state)

    assert actions.loc[0, "recommended_action"] == "stay_out"
    assert actions.loc[1, "recommended_action"] == "pit_now"
    assert float(actions.loc[1, "score_pit_now"]) > float(actions.loc[1, "score_stay_out"])
    assert float(actions.loc[1, "pit_loss_estimate_seconds"]) < 21.0
    assert actions.loc[1, "next_compound"] == "HARD"
    assert actions.loc[1, "policy_version"] == "deterministic_baseline_v1"


def test_live_strategy_adapters_preserve_empty_input_shape() -> None:
    empty = pd.DataFrame()

    features = BaselineTelemetryFeatureAdapter().build_lap_features(empty)
    actions = BaselineStrategyPolicyAdapter().evaluate_actions(empty)

    assert features.empty
    assert actions.empty
    assert "track_risk_score" in features.columns
    assert "recommended_action" in actions.columns


def test_default_strategy_adapters_are_deterministic_baselines() -> None:
    laps = _sample_laps()
    state = pd.DataFrame(
        [
            {
                "driver_id": "1",
                "compound": "MEDIUM",
                "tyre_age": 15,
                "stint_id": 1,
                "deg_rate_mean": 0.055,
                "pace_penalty_mean": 0.8,
                "track_status": "1",
                "lap_last": 34,
                "race_total_laps": 58,
            }
        ]
    )

    telemetry = NoopTelemetryFeatureAdapter()
    strategy = NoopStrategyPolicyAdapter()

    pd.testing.assert_frame_equal(
        telemetry.build_lap_features(laps),
        telemetry.build_lap_features(laps),
    )
    pd.testing.assert_frame_equal(
        strategy.evaluate_actions(state),
        strategy.evaluate_actions(state),
    )
