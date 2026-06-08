"""Evaluation harness for Ultimate Lap-Time models."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from packages.f1.models.ultimate_lap_time.schemas import UltimateLapTelemetryExample
from packages.sports_core.paths import find_repo_root


DEFAULT_REPORT_RELATIVE_PATH = Path("artifacts/reports/f1/ultimate_lap_time_evaluation.json")
DEFAULT_REPORT_PATH = find_repo_root(__file__) / DEFAULT_REPORT_RELATIVE_PATH
ACTUAL_COLUMNS: tuple[str, ...] = ("lap_time_seconds", "lap_duration", "LapTime", "lap_time", "duration")
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

    @property
    def missing_metrics(self) -> tuple[str, ...]:
        missing: list[str] = []
        for metric in self.required_metrics:
            value = self.metrics.get(metric)
            if value is None or not np.isfinite(float(value)):
                missing.append(metric)
        return tuple(missing)

    @property
    def promotion_gate_passed(self) -> bool:
        return not self.missing_metrics and not self.leakage_issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "row_count": self.row_count,
            "metrics": {key: float(value) if value is not None and np.isfinite(value) else None for key, value in self.metrics.items()},
            "calibration_curve": self.calibration_curve,
            "leakage_issues": list(self.leakage_issues),
            "required_metrics": list(self.required_metrics),
            "missing_metrics": list(self.missing_metrics),
            "promotion_gate_passed": self.promotion_gate_passed,
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

    output_path = Path(path) if path is not None else find_repo_root(__file__) / DEFAULT_REPORT_RELATIVE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return output_path


__all__ = [
    "DEFAULT_REPORT_PATH",
    "DEFAULT_REPORT_RELATIVE_PATH",
    "REQUIRED_METRICS",
    "UltimateLapTimeEvaluationResult",
    "evaluate_ultimate_lap_time_predictions",
    "leakage_issues_for_evaluation",
    "normalize_evaluation_frame",
    "pinball_loss",
    "write_ultimate_lap_time_evaluation_report",
]
