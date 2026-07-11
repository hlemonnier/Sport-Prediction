"""Evaluation harness for Ultimate Lap-Time models."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from packages.f1.models.ultimate_lap_time.model import (
    IDEAL_LAP_TARGET_COLUMN,
    UltimateLapTimeConfig,
    aggregate_ideal_lap_holdout_targets,
    fit_ultimate_lap_time_model,
)
from packages.f1.models.ultimate_lap_time.schemas import (
    IDEAL_LAP_TARGET_CONTRACT,
    UltimateLapTelemetryExample,
)
from packages.sports_core.paths import find_repo_root


DETERMINISTIC_BASELINE_MODEL_NAME = "ultimate_lap_time_deterministic_baseline_v1"
DEFAULT_REPORT_RELATIVE_PATH = Path("artifacts/reports/f1/ultimate_lap_time_evaluation.json")
DEFAULT_BASELINE_BACKTEST_RELATIVE_PATH = Path(
    "artifacts/backtests/f1/ultimate_lap_time_deterministic_baseline_v1.json"
)
DEFAULT_REPORT_PATH = find_repo_root(__file__) / DEFAULT_REPORT_RELATIVE_PATH
DEFAULT_BASELINE_BACKTEST_PATH = find_repo_root(__file__) / DEFAULT_BASELINE_BACKTEST_RELATIVE_PATH
ACTUAL_COLUMNS: tuple[str, ...] = (IDEAL_LAP_TARGET_COLUMN,)
TARGET_CONTRACT_COLUMN = "target_contract"
P05_COLUMNS: tuple[str, ...] = ("lap_p05", "p05_prediction", "predicted_p05", "p05", "pace_floor_seconds")
P50_COLUMNS: tuple[str, ...] = (
    "lap_p50",
    "p50_prediction",
    "predicted_p50",
    "p50",
    "ultimate_lap_time_seconds",
)
P90_COLUMNS: tuple[str, ...] = ("lap_p90", "p90_prediction", "predicted_p90", "p90", "pace_ceiling_seconds")
DEFAULT_GROUP_COLUMNS: tuple[str, ...] = ("event_key", "session")
CALIBRATION_GROUP_COLUMNS: tuple[str, ...] = (
    "circuit_id",
    "session",
    "weather",
    "weather_condition",
    "track_status",
)
REQUIRED_METRICS: tuple[str, ...] = (
    "p50_mae",
    "p50_rmse",
    "p05_pinball",
    "p50_pinball",
    "p90_pinball",
    "interval_coverage",
    "fastest_lap_winner_hit_rate",
    "top3_fastest_lap_accuracy",
)


def _resolve_artifact_path(path: str | Path | None, default_relative_path: Path) -> Path:
    if path is None:
        return find_repo_root(__file__) / default_relative_path
    output_path = Path(path).expanduser()
    if output_path.is_absolute():
        return output_path
    if not output_path.parts or output_path.parts[0] != "artifacts":
        raise ValueError("relative artifact paths must live under artifacts/")
    return find_repo_root(__file__) / output_path


def _as_dataframe(data: pd.DataFrame | Sequence[Mapping[str, Any]] | Sequence[UltimateLapTelemetryExample]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    rows: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, UltimateLapTelemetryExample):
            rows.append(item.as_flat_record())
        else:
            rows.append(dict(item))
    return pd.DataFrame(rows)


def _find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        match = lowered.get(candidate.lower())
        if match is not None:
            return match
    return None


def _numeric_series(frame: pd.DataFrame, candidates: Sequence[str], *, required: bool) -> pd.Series:
    column = _find_column(frame, candidates)
    if column is None:
        if required:
            raise ValueError(f"missing required column from candidates {tuple(candidates)}")
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def pinball_loss(y_true: Sequence[float], y_pred: Sequence[float], quantile: float) -> float:
    """Mean quantile pinball loss."""

    q = float(quantile)
    if not 0.0 < q < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(pred)
    if not mask.any():
        return float("nan")
    error = actual[mask] - pred[mask]
    return float(np.mean(np.maximum(q * error, (q - 1.0) * error)))


def _rank_spearman(actual: pd.Series, predicted: pd.Series) -> float:
    mask = np.isfinite(actual.to_numpy(dtype=float)) & np.isfinite(predicted.to_numpy(dtype=float))
    if int(mask.sum()) < 2:
        return float("nan")
    actual_rank = actual[mask].rank(method="average", ascending=True)
    predicted_rank = predicted[mask].rank(method="average", ascending=True)
    corr = actual_rank.corr(predicted_rank, method="pearson")
    return float(corr) if corr is not None and np.isfinite(corr) else float("nan")


def _ranking_metrics(frame: pd.DataFrame, group_columns: Sequence[str]) -> dict[str, float]:
    groups = [column for column in group_columns if column in frame.columns]
    if not groups:
        grouped = [("all", frame)]
    else:
        grouped = list(frame.groupby(groups, dropna=False, sort=False))

    fastest_hits: list[float] = []
    top3_scores: list[float] = []
    spearman_scores: list[float] = []
    for _, group in grouped:
        valid = group[np.isfinite(group["actual_lap_time"]) & np.isfinite(group["predicted_p50"])]
        if len(valid) < 2:
            continue
        actual_fastest_idx = valid["actual_lap_time"].idxmin()
        predicted_fastest_idx = valid["predicted_p50"].idxmin()
        fastest_hits.append(float(actual_fastest_idx == predicted_fastest_idx))
        k = int(min(3, len(valid)))
        actual_top = set(valid.nsmallest(k, "actual_lap_time").index)
        predicted_top = set(valid.nsmallest(k, "predicted_p50").index)
        top3_scores.append(float(len(actual_top & predicted_top) / k))
        spearman_scores.append(_rank_spearman(valid["actual_lap_time"], valid["predicted_p50"]))

    return {
        "fastest_lap_winner_hit_rate": float(np.nanmean(fastest_hits)) if fastest_hits else float("nan"),
        "top3_fastest_lap_accuracy": float(np.nanmean(top3_scores)) if top3_scores else float("nan"),
        "ranking_spearman": float(np.nanmean(spearman_scores)) if spearman_scores else float("nan"),
    }


def _calibration_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    group_columns = [column for column in CALIBRATION_GROUP_COLUMNS if column in frame.columns]
    if not group_columns:
        group_columns = ["__all__"]
        frame = frame.assign(__all__="all")

    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_columns, dropna=False, sort=True):
        valid = group[
            np.isfinite(group["actual_lap_time"])
            & np.isfinite(group["predicted_p05"])
            & np.isfinite(group["predicted_p50"])
            & np.isfinite(group["predicted_p90"])
        ]
        if valid.empty:
            continue
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {column: value for column, value in zip(group_columns, key_tuple)}
        error = valid["predicted_p50"] - valid["actual_lap_time"]
        row.update(
            {
                "row_count": int(len(valid)),
                "mean_error_seconds": float(error.mean()),
                "mae_seconds": float(error.abs().mean()),
                "coverage": float(
                    ((valid["predicted_p05"] <= valid["actual_lap_time"]) & (valid["actual_lap_time"] <= valid["predicted_p90"])).mean()
                ),
                "mean_interval_width_seconds": float((valid["predicted_p90"] - valid["predicted_p05"]).mean()),
            }
        )
        rows.append(row)
    return rows


def leakage_issues_for_evaluation(frame: pd.DataFrame) -> tuple[str, ...]:
    """Detect basic leakage and payload integrity issues in an evaluation frame."""

    issues: list[str] = []
    if "split_key" in frame.columns and "split_name" in frame.columns:
        split_counts = frame.dropna(subset=["split_key", "split_name"]).groupby("split_key")["split_name"].nunique()
        leaking_keys = split_counts[split_counts > 1]
        if not leaking_keys.empty:
            issues.append(f"{int(len(leaking_keys))} split_key values appear in multiple splits")

    if {"actual_lap_time", "predicted_p50"}.issubset(frame.columns):
        actual = frame["actual_lap_time"].to_numpy(dtype=float)
        p50 = frame["predicted_p50"].to_numpy(dtype=float)
        mask = np.isfinite(actual) & np.isfinite(p50)
        if int(mask.sum()) >= 4 and np.allclose(actual[mask], p50[mask], atol=1e-12, rtol=0.0):
            issues.append("predicted_p50 exactly matches actual lap time on every finite row")

    if {"predicted_p05", "predicted_p50", "predicted_p90"}.issubset(frame.columns):
        bad_quantiles = frame[
            np.isfinite(frame["predicted_p05"])
            & np.isfinite(frame["predicted_p50"])
            & np.isfinite(frame["predicted_p90"])
            & ~((frame["predicted_p05"] <= frame["predicted_p50"]) & (frame["predicted_p50"] <= frame["predicted_p90"]))
        ]
        if not bad_quantiles.empty:
            issues.append(f"{int(len(bad_quantiles))} rows have non-monotonic predicted quantiles")
    return tuple(issues)


@dataclass(frozen=True)
class UltimateLapTimeEvaluationResult:
    """Stable evaluation payload for reports and promotion gates."""

    model_name: str
    row_count: int
    metrics: dict[str, float]
    calibration_curve: list[dict[str, Any]]
    leakage_issues: tuple[str, ...]
    required_metrics: tuple[str, ...] = REQUIRED_METRICS
    target_contract: str = IDEAL_LAP_TARGET_CONTRACT

    @property
    def missing_metrics(self) -> tuple[str, ...]:
        missing: list[str] = []
        for metric in self.required_metrics:
            value = self.metrics.get(metric)
            if value is None or not np.isfinite(float(value)):
                missing.append(metric)
        return tuple(missing)

    @property
    def evaluation_contract_passed(self) -> bool:
        return not self.missing_metrics and not self.leakage_issues

    @property
    def promotion_gate_passed(self) -> bool:
        """A standalone evaluation can never authorize production promotion."""

        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "row_count": self.row_count,
            "metrics": {key: float(value) if value is not None and np.isfinite(value) else None for key, value in self.metrics.items()},
            "calibration_curve": self.calibration_curve,
            "leakage_issues": list(self.leakage_issues),
            "required_metrics": list(self.required_metrics),
            "missing_metrics": list(self.missing_metrics),
            "evaluation_contract_passed": self.evaluation_contract_passed,
            "promotion_gate_passed": self.promotion_gate_passed,
            "promotion_blockers": ["registry_baseline_comparison_and_artifact_evidence_required"],
            "target_contract": self.target_contract,
        }


@dataclass(frozen=True)
class UltimateLapTimeBaselineBacktestResult:
    """Comparable deterministic-baseline backtest payload for Phase 0 locking."""

    model_name: str
    training_summary: dict[str, Any]
    evaluation: UltimateLapTimeEvaluationResult
    artifact_relative_path: str = str(DEFAULT_BASELINE_BACKTEST_RELATIVE_PATH)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": "ultimate_lap_time_baseline_backtest",
            "model_name": self.model_name,
            "baseline_model": self.model_name,
            "artifact_relative_path": self.artifact_relative_path,
            "training_summary": self.training_summary,
            "evaluation": self.evaluation.to_dict(),
        }


def normalize_evaluation_frame(
    actual: pd.DataFrame | Sequence[Mapping[str, Any]] | Sequence[UltimateLapTelemetryExample],
    predictions: pd.DataFrame | Sequence[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Combine actual and prediction payloads into the canonical evaluation shape."""

    actual_frame = _as_dataframe(actual)
    if actual_frame.empty:
        return pd.DataFrame(
            columns=["actual_lap_time", "predicted_p05", "predicted_p50", "predicted_p90"]
        )
    contract_column = _find_column(actual_frame, (TARGET_CONTRACT_COLUMN, "target_kind"))
    if contract_column is None:
        raise ValueError(
            "ultimate lap-time evaluation requires an explicit theoretical ideal-lap target contract; "
            "aggregate raw holdout laps first"
        )
    contracts = actual_frame[contract_column].astype(str).str.strip().str.lower()
    invalid_contracts = sorted(set(contracts[contracts != IDEAL_LAP_TARGET_CONTRACT].tolist()))
    if invalid_contracts:
        raise ValueError(
            "ultimate lap-time evaluation only accepts theoretical ideal-lap targets; "
            f"invalid contracts={invalid_contracts}"
        )
    actual_lap = _numeric_series(actual_frame, ACTUAL_COLUMNS, required=True)

    if predictions is None:
        pred_frame = actual_frame
    else:
        pred_frame = _as_dataframe(predictions)
        if len(pred_frame) != len(actual_frame):
            raise ValueError("predictions length must match actual length")
        pred_frame = pred_frame.set_index(actual_frame.index)

    frame = actual_frame.copy()
    frame["actual_lap_time"] = actual_lap
    frame["predicted_p05"] = _numeric_series(pred_frame, P05_COLUMNS, required=True)
    frame["predicted_p50"] = _numeric_series(pred_frame, P50_COLUMNS, required=True)
    frame["predicted_p90"] = _numeric_series(pred_frame, P90_COLUMNS, required=True)
    return frame


def evaluate_ultimate_lap_time_predictions(
    actual: pd.DataFrame | Sequence[Mapping[str, Any]] | Sequence[UltimateLapTelemetryExample],
    predictions: pd.DataFrame | Sequence[Mapping[str, Any]] | None = None,
    *,
    model_name: str = "ultimate_lap_time_model",
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
) -> UltimateLapTimeEvaluationResult:
    """Evaluate Ultimate Lap-Time quantile predictions."""

    frame = normalize_evaluation_frame(actual, predictions)
    if frame.empty:
        metrics = {metric: float("nan") for metric in REQUIRED_METRICS}
        return UltimateLapTimeEvaluationResult(
            model_name=model_name,
            row_count=0,
            metrics=metrics,
            calibration_curve=[],
            leakage_issues=("empty evaluation payload",),
        )

    valid = frame[
        np.isfinite(frame["actual_lap_time"])
        & np.isfinite(frame["predicted_p05"])
        & np.isfinite(frame["predicted_p50"])
        & np.isfinite(frame["predicted_p90"])
    ].copy()
    if valid.empty:
        metrics = {metric: float("nan") for metric in REQUIRED_METRICS}
        return UltimateLapTimeEvaluationResult(
            model_name=model_name,
            row_count=int(len(frame)),
            metrics=metrics,
            calibration_curve=[],
            leakage_issues=("no finite actual/prediction rows",),
        )

    error = valid["predicted_p50"] - valid["actual_lap_time"]
    metrics = {
        "p50_mae": float(error.abs().mean()),
        "p50_rmse": float(math.sqrt(float(np.mean(np.square(error))))),
        "p05_pinball": pinball_loss(valid["actual_lap_time"], valid["predicted_p05"], 0.05),
        "p50_pinball": pinball_loss(valid["actual_lap_time"], valid["predicted_p50"], 0.50),
        "p90_pinball": pinball_loss(valid["actual_lap_time"], valid["predicted_p90"], 0.90),
        "interval_coverage": float(
            ((valid["predicted_p05"] <= valid["actual_lap_time"]) & (valid["actual_lap_time"] <= valid["predicted_p90"])).mean()
        ),
        "mean_interval_width_seconds": float((valid["predicted_p90"] - valid["predicted_p05"]).mean()),
        "mean_bias_seconds": float(error.mean()),
    }
    metrics.update(_ranking_metrics(valid, group_columns))
    return UltimateLapTimeEvaluationResult(
        model_name=model_name,
        row_count=int(len(valid)),
        metrics=metrics,
        calibration_curve=_calibration_rows(valid),
        leakage_issues=leakage_issues_for_evaluation(valid),
    )


def write_ultimate_lap_time_evaluation_report(
    result: UltimateLapTimeEvaluationResult,
    path: str | Path | None = None,
) -> Path:
    """Write an evaluation payload under artifacts/reports/f1 by default."""

    output_path = _resolve_artifact_path(path, DEFAULT_REPORT_RELATIVE_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def evaluate_ultimate_lap_time_baseline_backtest(
    train_laps: pd.DataFrame,
    evaluation_laps: pd.DataFrame,
    *,
    config: UltimateLapTimeConfig | None = None,
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
) -> UltimateLapTimeBaselineBacktestResult:
    """Fit and evaluate the locked deterministic baseline on explicit holdout rows."""

    if not isinstance(train_laps, pd.DataFrame) or not isinstance(evaluation_laps, pd.DataFrame):
        raise TypeError("train_laps and evaluation_laps must be pandas DataFrames")
    if train_laps.empty:
        raise ValueError("train_laps must contain at least one timing row")
    if evaluation_laps.empty:
        raise ValueError("evaluation_laps must contain at least one timing row")

    model = fit_ultimate_lap_time_model(train_laps, config=config)
    ideal_holdout = aggregate_ideal_lap_holdout_targets(
        evaluation_laps,
        config=config,
    )
    predictions = model.predict_details(ideal_holdout)
    evaluation = evaluate_ultimate_lap_time_predictions(
        ideal_holdout,
        predictions,
        model_name=DETERMINISTIC_BASELINE_MODEL_NAME,
        group_columns=group_columns,
    )
    return UltimateLapTimeBaselineBacktestResult(
        model_name=DETERMINISTIC_BASELINE_MODEL_NAME,
        training_summary={
            **asdict(model.training_summary),
            "holdout_raw_lap_rows": int(len(evaluation_laps)),
            "holdout_ideal_target_rows": int(len(ideal_holdout)),
            "holdout_target_contract": IDEAL_LAP_TARGET_CONTRACT,
        },
        evaluation=evaluation,
    )


def write_ultimate_lap_time_baseline_backtest_report(
    result: UltimateLapTimeBaselineBacktestResult,
    path: str | Path | None = None,
) -> Path:
    """Write the deterministic baseline backtest report under artifacts/backtests/f1."""

    output_path = _resolve_artifact_path(path, DEFAULT_BASELINE_BACKTEST_RELATIVE_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return output_path


__all__ = [
    "DEFAULT_BASELINE_BACKTEST_PATH",
    "DEFAULT_BASELINE_BACKTEST_RELATIVE_PATH",
    "DEFAULT_REPORT_PATH",
    "DEFAULT_REPORT_RELATIVE_PATH",
    "DETERMINISTIC_BASELINE_MODEL_NAME",
    "REQUIRED_METRICS",
    "UltimateLapTimeBaselineBacktestResult",
    "UltimateLapTimeEvaluationResult",
    "evaluate_ultimate_lap_time_baseline_backtest",
    "evaluate_ultimate_lap_time_predictions",
    "leakage_issues_for_evaluation",
    "normalize_evaluation_frame",
    "pinball_loss",
    "write_ultimate_lap_time_baseline_backtest_report",
    "write_ultimate_lap_time_evaluation_report",
]
