from __future__ import annotations

import pandas as pd

from packages.f1.models.live_race.action_space import StrategyAction
from packages.f1.models.live_race.environment import build_replay_transitions
from packages.f1.models.live_race.evaluate_policy import evaluate_strategy_policy
from packages.f1.models.live_race.planner import DeterministicStrategyPlanner, PlannerConfig


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
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
                "race_time_seconds": 293.0,
                "timestamp": 3.0,
                "observed_action": "stay_out",
                "final_position": 7,
            },
        ]
    )


def test_policy_evaluator_reports_replay_metrics_without_planner() -> None:
    transitions = build_replay_transitions(_rows())

    result = evaluate_strategy_policy(transitions)

    assert result.metrics["available"] is True
    assert result.metrics["rows"] == 2
    assert result.metrics["illegal_action_rate"] == 0.0
    assert result.metrics["transition_consistency"]["ok"] is True
    assert result.metrics["no_leakage_replay_invariance"]["metadata_available_through_lap_ok"] is True
    assert result.metrics["no_leakage_replay_invariance"]["available"] is False
    assert "replay_prefix_invariance" in result.metrics["missing_metrics"]
    assert "candidate_policy" in result.metrics["missing_metrics"]
    assert "counterfactual_policy_value" in result.metrics["missing_metrics"]
    assert "regret_vs_oracle" in result.metrics["missing_metrics"]
    assert result.metrics["promotion_gate_pass"] is False


def test_policy_exceptions_cannot_fall_back_to_replay_and_pass() -> None:
    transitions = build_replay_transitions(_rows())
    planner = DeterministicStrategyPlanner(
        config=PlannerConfig(horizon_laps=2, strategy_score_weight=0.0)
    )

    def broken_policy(_state: object) -> StrategyAction:
        raise RuntimeError("policy failed")

    result = evaluate_strategy_policy(
        transitions,
        policy=broken_policy,
        oracle_planner=planner,
        comparison_transitions=transitions,
    )

    assert result.metrics["policy_error_count"] == len(transitions)
    assert "policy_execution_errors" in result.metrics["missing_metrics"]
    assert result.metrics["promotion_gate_pass"] is False


def test_policy_evaluator_fails_closed_on_empty_replay() -> None:
    result = evaluate_strategy_policy([])

    assert result.metrics["available"] is False
    assert "policy_value" in result.metrics["missing_metrics"]
    assert "illegal_action_rate" in result.metrics["missing_metrics"]
    assert result.metrics["promotion_gate_pass"] is False
