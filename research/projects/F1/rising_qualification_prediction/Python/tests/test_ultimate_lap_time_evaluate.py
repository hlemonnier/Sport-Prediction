from __future__ import annotations

import json

import pandas as pd
import pytest

import packages.f1.models.ultimate_lap_time.evaluate as ultimate_evaluate
from packages.f1.models.ultimate_lap_time.evaluate import (
    evaluate_ultimate_lap_time_predictions,
    pinball_loss,
    write_ultimate_lap_time_evaluation_report,
)


def test_ultimate_lap_evaluation_metrics_and_report_shape(tmp_path) -> None:
    actual = pd.DataFrame(
        {
            "event_key": ["bahrain", "bahrain", "bahrain", "jeddah", "jeddah", "jeddah"],
            "session": ["Q", "Q", "Q", "Q", "Q", "Q"],
            "circuit_id": ["bahrain", "bahrain", "bahrain", "jeddah", "jeddah", "jeddah"],
            "driver_id": ["VER", "LEC", "NOR", "VER", "LEC", "NOR"],
            "lap_time_seconds": [89.8, 90.1, 90.4, 88.9, 89.2, 89.5],
        }
    )
    predictions = pd.DataFrame(
        {
            "lap_p05": [89.6, 89.9, 90.2, 88.7, 89.0, 89.3],
            "lap_p50": [89.9, 90.0, 90.5, 88.8, 89.4, 89.6],
            "lap_p90": [90.2, 90.3, 90.8, 89.2, 89.7, 89.9],
        }
    )

    result = evaluate_ultimate_lap_time_predictions(actual, predictions, model_name="unit_test_model")

    assert result.model_name == "unit_test_model"
    assert result.row_count == 6
    assert result.metrics["p50_mae"] == pytest.approx(0.1166666667)
    assert result.metrics["p50_rmse"] > 0.0
    assert result.metrics["interval_coverage"] == pytest.approx(1.0)
    assert result.metrics["fastest_lap_winner_hit_rate"] == pytest.approx(1.0)
    assert result.metrics["top3_fastest_lap_accuracy"] == pytest.approx(1.0)
    assert result.leakage_issues == ()
    assert result.promotion_gate_passed
    assert result.calibration_curve

    output_path = write_ultimate_lap_time_evaluation_report(result, tmp_path / "eval.json")
    payload = json.loads(output_path.read_text())
    assert payload["promotion_gate_passed"] is True
    assert payload["metrics"]["p05_pinball"] == pytest.approx(result.metrics["p05_pinball"])


def test_pinball_loss_matches_quantile_definition() -> None:
    assert pinball_loss([10.0], [9.0], 0.90) == pytest.approx(0.9)
    assert pinball_loss([10.0], [11.0], 0.90) == pytest.approx(0.1)


def test_evaluation_flags_exact_target_prediction_as_leakage() -> None:
    frame = pd.DataFrame(
        {
            "event_key": ["e"] * 4,
            "session": ["Q"] * 4,
            "lap_time_seconds": [90.0, 91.0, 92.0, 93.0],
            "lap_p05": [89.9, 90.9, 91.9, 92.9],
            "lap_p50": [90.0, 91.0, 92.0, 93.0],
            "lap_p90": [90.1, 91.1, 92.1, 93.1],
        }
    )

    result = evaluate_ultimate_lap_time_predictions(frame)

    assert result.leakage_issues == ("predicted_p50 exactly matches actual lap time on every finite row",)
    assert not result.promotion_gate_passed


def test_default_report_path_is_repo_rooted(monkeypatch, tmp_path) -> None:
    result = ultimate_evaluate.UltimateLapTimeEvaluationResult(
        model_name="unit_test_model",
        row_count=1,
        metrics={metric: 1.0 for metric in ultimate_evaluate.REQUIRED_METRICS},
        calibration_curve=[],
        leakage_issues=(),
    )
    monkeypatch.setattr(ultimate_evaluate, "find_repo_root", lambda _: tmp_path)

    output_path = write_ultimate_lap_time_evaluation_report(result)

    assert output_path == tmp_path / ultimate_evaluate.DEFAULT_REPORT_RELATIVE_PATH
    assert output_path.exists()
