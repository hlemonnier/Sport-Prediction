from __future__ import annotations

import json

import pandas as pd
import pytest

import packages.f1.models.ultimate_lap_time.evaluate as ultimate_evaluate
from packages.f1.models.ultimate_lap_time.model import aggregate_ideal_lap_holdout_targets
from packages.f1.models.ultimate_lap_time.evaluate import (
    DETERMINISTIC_BASELINE_MODEL_NAME,
    evaluate_ultimate_lap_time_baseline_backtest,
    evaluate_ultimate_lap_time_predictions,
    pinball_loss,
    write_ultimate_lap_time_baseline_backtest_report,
    write_ultimate_lap_time_evaluation_report,
)
from packages.f1.models.ultimate_lap_time.schemas import (
    ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT,
    IDEAL_LAP_TARGET_CONTRACT,
)


def _baseline_laps(event_key: str, offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_key": [event_key] * 4,
            "session": ["Q"] * 4,
            "circuit_id": ["bahrain"] * 4,
            "driver_id": ["VER", "LEC", "NOR", "PIA"],
            "team_id": ["red_bull", "ferrari", "mclaren", "mclaren"],
            "lap_time_seconds": [89.8 + offset, 90.1 + offset, 90.4 + offset, 90.6 + offset],
            "sector1_seconds": [29.8 + offset / 3.0, 30.0 + offset / 3.0, 30.2 + offset / 3.0, 30.3 + offset / 3.0],
            "sector2_seconds": [30.0, 30.1, 30.2, 30.3],
            "sector3_seconds": [30.0, 30.0, 30.0, 30.0],
            "is_accurate": [True] * 4,
            "is_box_lap": [False] * 4,
            "track_status": ["1"] * 4,
        }
    )


def test_ultimate_lap_evaluation_metrics_and_report_shape(tmp_path) -> None:
    actual = pd.DataFrame(
        {
            "event_key": ["bahrain", "bahrain", "bahrain", "jeddah", "jeddah", "jeddah"],
            "session": ["Q", "Q", "Q", "Q", "Q", "Q"],
            "circuit_id": ["bahrain", "bahrain", "bahrain", "jeddah", "jeddah", "jeddah"],
            "driver_id": ["VER", "LEC", "NOR", "VER", "LEC", "NOR"],
            "ideal_lap_time_seconds": [89.8, 90.1, 90.4, 88.9, 89.2, 89.5],
            "target_contract": [IDEAL_LAP_TARGET_CONTRACT] * 6,
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
    assert result.evaluation_contract_passed
    assert not result.promotion_gate_passed
    assert result.calibration_curve

    output_path = write_ultimate_lap_time_evaluation_report(result, tmp_path / "eval.json")
    payload = json.loads(output_path.read_text())
    assert payload["evaluation_contract_passed"] is True
    assert payload["promotion_gate_passed"] is False
    assert payload["metrics"]["p05_pinball"] == pytest.approx(result.metrics["p05_pinball"])


def test_pinball_loss_matches_quantile_definition() -> None:
    assert pinball_loss([10.0], [9.0], 0.90) == pytest.approx(0.9)
    assert pinball_loss([10.0], [11.0], 0.90) == pytest.approx(0.1)


def test_evaluation_flags_exact_target_prediction_as_leakage() -> None:
    frame = pd.DataFrame(
        {
            "event_key": ["e"] * 4,
            "session": ["Q"] * 4,
            "ideal_lap_time_seconds": [90.0, 91.0, 92.0, 93.0],
            "target_contract": [IDEAL_LAP_TARGET_CONTRACT] * 4,
            "lap_p05": [89.9, 90.9, 91.9, 92.9],
            "lap_p50": [90.0, 91.0, 92.0, 93.0],
            "lap_p90": [90.1, 91.1, 92.1, 93.1],
        }
    )

    result = evaluate_ultimate_lap_time_predictions(frame)

    assert result.leakage_issues == ("predicted_p50 exactly matches actual lap time on every finite row",)
    assert not result.promotion_gate_passed


def test_evaluation_rejects_arbitrary_raw_lap_rows() -> None:
    raw_laps = _baseline_laps("raw-holdout")
    predictions = pd.DataFrame(
        {
            "lap_p05": [89.0] * len(raw_laps),
            "lap_p50": [90.0] * len(raw_laps),
            "lap_p90": [91.0] * len(raw_laps),
        }
    )

    with pytest.raises(ValueError, match="aggregate raw holdout laps first"):
        evaluate_ultimate_lap_time_predictions(raw_laps, predictions)


def test_realized_scalar_targets_are_valid_for_quantile_evaluation() -> None:
    actual = pd.DataFrame(
        {
            "event_key": ["bahrain"] * 3,
            "session": ["Q"] * 3,
            "driver_id": ["VER", "LEC", "NOR"],
            "achievable_session_end_lap_time_seconds": [89.8, 90.1, 90.4],
            "target_contract": [ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT] * 3,
            "p05_target": [89.8, 90.1, 90.4],
            "p50_target": [89.8, 90.1, 90.4],
            "p90_target": [89.8, 90.1, 90.4],
        }
    )
    predictions = pd.DataFrame(
        {
            "lap_p05": [89.5, 89.8, 90.1],
            "lap_p50": [89.8, 90.1, 90.4],
            "lap_p90": [90.2, 90.5, 90.8],
        }
    )

    result = evaluate_ultimate_lap_time_predictions(actual, predictions)

    assert result.target_diagnostics["degenerate_quantile_target_rows"] == 3
    assert result.target_diagnostics["promotion_grade_validation_passed"]
    assert result.promotion_grade_validation_passed
    assert "quantile_targets_are_degenerate" not in result.to_dict()["promotion_blockers"]

    with pytest.raises(ValueError, match="prediction target contract does not match"):
        evaluate_ultimate_lap_time_predictions(
            actual,
            predictions.assign(target_contract=IDEAL_LAP_TARGET_CONTRACT),
        )
    with pytest.raises(ValueError, match="prediction target semantics do not match"):
        evaluate_ultimate_lap_time_predictions(
            actual,
            predictions.assign(
                target_contract=ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT,
                target_semantics="theoretical_sector_floor",
            ),
        )


def test_holdout_target_is_explicit_sector_minimum_aggregation() -> None:
    laps = pd.DataFrame(
        {
            "event_key": ["e1", "e1"],
            "session": ["Q", "Q"],
            "circuit_id": ["bahrain", "bahrain"],
            "driver_id": ["VER", "VER"],
            "team_id": ["red_bull", "red_bull"],
            "lap_time_seconds": [90.0, 90.2],
            "sector1_seconds": [29.0, 30.0],
            "sector2_seconds": [31.0, 30.0],
            "sector3_seconds": [30.0, 30.2],
        }
    )

    targets = aggregate_ideal_lap_holdout_targets(laps)

    assert len(targets) == 1
    assert targets.loc[0, "ideal_lap_time_seconds"] == pytest.approx(89.0)
    assert targets.loc[0, "target_contract"] == IDEAL_LAP_TARGET_CONTRACT
    assert targets.loc[0, "clean_lap_count"] == 2
    assert "lap_time_seconds" not in targets.columns


def test_holdout_sector_lower_bound_never_mixes_incompatible_compounds() -> None:
    laps = pd.DataFrame(
        {
            "event_key": ["e1", "e1"],
            "session": ["Q", "Q"],
            "circuit_id": ["silverstone", "silverstone"],
            "driver_id": ["LEC", "LEC"],
            "team_id": ["ferrari", "ferrari"],
            "compound": ["SOFT", "HARD"],
            "lap_time_seconds": [90.0, 90.2],
            "sector1_seconds": [29.0, 31.0],
            "sector2_seconds": [31.0, 29.0],
            "sector3_seconds": [30.0, 30.2],
        }
    )

    targets = aggregate_ideal_lap_holdout_targets(laps)

    # Unconstrained cherry-picking would report 88.0s (29 + 29 + 30), a lap
    # that no compatible tyre/session state supports.
    assert targets.loc[0, "ideal_lap_time_seconds"] == pytest.approx(90.0)
    assert targets.loc[0, "ideal_lap_construction"] == "compatible_sector_lower_bound_v2"
    assert targets.loc[0, "sector_compatibility_columns"] == ["session", "compound"]
    assert targets.loc[0, "sector_compatibility_candidate_count"] == 2


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


def test_relative_report_path_is_repo_rooted_from_research_cwd(monkeypatch, tmp_path) -> None:
    research_cwd = tmp_path / "research/projects/F1/rising_qualification_prediction/Python"
    research_cwd.mkdir(parents=True)
    monkeypatch.chdir(research_cwd)
    monkeypatch.setattr(ultimate_evaluate, "find_repo_root", lambda _: tmp_path)
    result = ultimate_evaluate.UltimateLapTimeEvaluationResult(
        model_name="unit_test_model",
        row_count=1,
        metrics={metric: 1.0 for metric in ultimate_evaluate.REQUIRED_METRICS},
        calibration_curve=[],
        leakage_issues=(),
    )

    output_path = write_ultimate_lap_time_evaluation_report(
        result,
        "artifacts/reports/f1/ultimate_lap_time/custom_eval.json",
    )

    assert output_path == tmp_path / "artifacts/reports/f1/ultimate_lap_time/custom_eval.json"
    assert not (research_cwd / "artifacts").exists()


def test_ultimate_baseline_backtest_report_uses_artifacts_backtests(monkeypatch, tmp_path) -> None:
    research_cwd = tmp_path / "research/projects/F1/rising_qualification_prediction/Python"
    research_cwd.mkdir(parents=True)
    monkeypatch.chdir(research_cwd)
    monkeypatch.setattr(ultimate_evaluate, "find_repo_root", lambda _: tmp_path)

    result = evaluate_ultimate_lap_time_baseline_backtest(
        _baseline_laps("2026-bahrain-train"),
        _baseline_laps("2026-bahrain-holdout", offset=0.2),
    )
    output_path = write_ultimate_lap_time_baseline_backtest_report(
        result,
        "artifacts/backtests/f1/ultimate_lap_time/baseline.json",
    )
    payload = json.loads(output_path.read_text())

    assert output_path == tmp_path / "artifacts/backtests/f1/ultimate_lap_time/baseline.json"
    assert payload["artifact_type"] == "ultimate_lap_time_baseline_backtest"
    assert payload["model_name"] == DETERMINISTIC_BASELINE_MODEL_NAME
    assert payload["evaluation"]["model_name"] == DETERMINISTIC_BASELINE_MODEL_NAME
    assert payload["evaluation"]["missing_metrics"] == []
    assert payload["evaluation"]["target_contract"] == IDEAL_LAP_TARGET_CONTRACT
    assert payload["training_summary"]["holdout_raw_lap_rows"] == 4
    assert payload["training_summary"]["holdout_ideal_target_rows"] == 4
    assert not (research_cwd / "artifacts").exists()
