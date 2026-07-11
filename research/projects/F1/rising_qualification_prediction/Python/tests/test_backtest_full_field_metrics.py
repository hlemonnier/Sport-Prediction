from __future__ import annotations

import math

import pandas as pd
import pytest

from packages.f1.orchestration.backtest import evaluate_prediction_rows


def _actual_field(size: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "driver_id": [str(index) for index in range(1, size + 1)],
            "position": list(range(1, size + 1)),
        }
    )


def _perfect_prediction_field(size: int = 10) -> list[dict[str, object]]:
    return [
        {
            "driver_id": str(rank),
            "rank": rank,
            "proba_win": 1.0 if rank == 1 else 0.0,
            "proba_top3": 1.0 if rank <= 3 else 0.0,
            "proba_top10": 1.0 if rank <= 10 else 0.0,
            "pos_p10": float(rank) - 0.5,
            "pos_p90": float(rank) + 0.5,
        }
        for rank in range(1, size + 1)
    ]


def test_full_field_metrics_cover_ranking_probability_and_position_intervals() -> None:
    result = evaluate_prediction_rows(_perfect_prediction_field(), _actual_field(), "position")

    assert result["exact_position_accuracy"] == pytest.approx(1.0)
    assert result["top5_hit"] == pytest.approx(1.0)
    assert result["kendall_tau_b"] == pytest.approx(1.0)
    for label in ("win", "top3", "top10"):
        assert result[f"{label}_log_loss"] == pytest.approx(0.0, abs=1e-12)
        assert result[f"{label}_brier"] == pytest.approx(0.0)
        assert result[f"{label}_ece"] == pytest.approx(0.0)
    assert result["proba_win_total"] == pytest.approx(1.0)
    assert result["proba_top3_total"] == pytest.approx(3.0)
    assert result["proba_top10_total"] == pytest.approx(10.0)
    assert result["position_interval_coverage"] == pytest.approx(1.0)
    assert result["position_interval_mean_width"] == pytest.approx(1.0)


def test_kendall_tau_b_accounts_for_tied_predicted_positions() -> None:
    predicted = [
        {"driver_id": "a", "rank": 1},
        {"driver_id": "b", "rank": 2},
        {"driver_id": "c", "rank": 2},
        {"driver_id": "d", "rank": 4},
    ]
    actual = pd.DataFrame(
        {
            "driver_id": ["a", "b", "c", "d"],
            "position": [1, 3, 2, 4],
        }
    )

    result = evaluate_prediction_rows(predicted, actual, "position")

    assert result["exact_position_accuracy"] == pytest.approx(0.75)
    assert result["top5_hit"] == pytest.approx(1.0)
    assert result["kendall_tau_b"] == pytest.approx(5.0 / math.sqrt(30.0))


def test_new_metrics_fail_closed_for_incomplete_field() -> None:
    predicted = _perfect_prediction_field()[:-1]

    result = evaluate_prediction_rows(predicted, _actual_field(), "position")

    assert result["metric_available"] is False
    for key in (
        "exact_position_accuracy",
        "top5_hit",
        "kendall_tau_b",
        "win_log_loss",
        "win_brier",
        "win_ece",
        "proba_win_total",
        "position_interval_coverage",
        "position_interval_mean_width",
    ):
        assert result[key] is None


def test_complete_field_rejects_unexpected_extra_prediction() -> None:
    predicted = _perfect_prediction_field()
    predicted.append({"driver_id": "reserve", "rank": 11})

    result = evaluate_prediction_rows(predicted, _actual_field(), "position")

    assert result["field_coverage"] == pytest.approx(1.0)
    assert result["complete_field"] is False
    assert result["metric_available"] is False
    assert result["unexpected_prediction_count"] == 1
    assert result["missing_actual_count"] == 0
    assert result["evaluation_reason"] == "field_roster_mismatch"


def test_probability_and_interval_families_fail_closed_independently() -> None:
    predicted = _perfect_prediction_field()
    predicted[4]["proba_top3"] = 1.1
    predicted[7]["pos_p10"] = 9.0
    predicted[7]["pos_p90"] = 7.0

    result = evaluate_prediction_rows(predicted, _actual_field(), "position")

    assert result["metric_available"] is True
    assert result["win_log_loss"] == pytest.approx(0.0, abs=1e-12)
    assert result["proba_win_total"] == pytest.approx(1.0)
    assert result["top3_log_loss"] is None
    assert result["top3_brier"] is None
    assert result["top3_ece"] is None
    assert result["proba_top3_total"] is None
    assert result["top10_log_loss"] == pytest.approx(0.0, abs=1e-12)
    assert result["position_interval_coverage"] is None
    assert result["position_interval_mean_width"] is None
