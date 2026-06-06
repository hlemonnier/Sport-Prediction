"""Shared evaluation helpers for offline/local prediction checks."""

from __future__ import annotations

import math
from typing import Any, Optional

import pandas as pd

from .identity import driver_key_column as _driver_key_column
from .identity import resolve_driver_matches as _resolve_driver_matches


def evaluate_prediction_rows(
    predicted_rows: list[dict[str, Any]],
    actual_results: pd.DataFrame,
    actual_position_col: str,
    *,
    include_podium_and_winner: bool = False,
    include_match_rows: bool = False,
    min_field_coverage: float = 0.95,
    require_complete_field: bool = True,
    missing_driver_penalty_scale: float = 1.0,
) -> dict[str, Any]:
    if not predicted_rows:
        return {"available": False, "reason": "prediction_rows_unavailable"}
    if actual_results is None or actual_results.empty:
        return {"available": False, "reason": "actual_results_unavailable"}

    pred = pd.DataFrame(predicted_rows).copy()
    if pred.empty or _driver_key_column(pred) is None:
        return {"available": False, "reason": "prediction_driver_key_unavailable"}
    if "rank" in pred.columns:
        pred["pred_rank"] = pd.to_numeric(pred["rank"], errors="coerce")
    else:
        pred["pred_rank"] = pd.Series(range(1, len(pred) + 1), index=pred.index, dtype=float)
    pred = pred.dropna(subset=["pred_rank"])
    pred["pred_rank"] = pred["pred_rank"].astype(float)
    if pred.empty:
        return {"available": False, "reason": "prediction_rank_unavailable"}

    actual = actual_results.copy()
    if actual_position_col not in actual.columns:
        return {"available": False, "reason": "actual_position_unavailable"}
    if _driver_key_column(actual) is None:
        return {"available": False, "reason": "actual_driver_key_unavailable"}
    actual["actual_rank"] = pd.to_numeric(actual[actual_position_col], errors="coerce")
    actual = actual[actual["actual_rank"].notna()]
    if actual.empty:
        return {"available": False, "reason": "actual_clean_unavailable"}

    pred_unique = pred.sort_values("pred_rank", kind="mergesort")
    actual_unique = actual.sort_values("actual_rank", kind="mergesort")
    matches, diagnostics = _resolve_driver_matches(pred_unique, actual_unique)
    if matches.empty:
        return {
            "available": False,
            "reason": "no_common_drivers",
            "rows_predicted": int(len(pred_unique)),
            "rows_actual": int(len(actual_unique)),
            "rows_common": 0,
            "field_coverage": 0.0,
            **diagnostics,
        }
    merged = (
        matches.merge(pred_unique[["pred_rank"]], left_on="pred_index", right_index=True, how="left")
        .merge(actual_unique[["actual_rank"]], left_on="actual_index", right_index=True, how="left")
        .dropna(subset=["pred_rank", "actual_rank"])
    )

    errors = (merged["pred_rank"] - merged["actual_rank"]) if not merged.empty else pd.Series(dtype=float)
    abs_errors = errors.abs()
    mae = float(abs_errors.mean()) if not merged.empty else None
    rmse = float(math.sqrt(float((errors**2).mean()))) if not merged.empty else None
    spearman = None
    if len(merged) >= 2:
        value = merged[["pred_rank", "actual_rank"]].corr(method="spearman").iloc[0, 1]
        if pd.notna(value):
            spearman = float(value)
    rows_actual = int(len(actual_unique))
    rows_common = int(len(merged))
    field_coverage = float(rows_common / rows_actual) if rows_actual else 0.0
    coverage_passed = bool(field_coverage >= float(min_field_coverage))
    complete_field = bool(rows_common >= rows_actual)
    metric_available = bool(mae is not None and coverage_passed and (complete_field or not require_complete_field))
    missing_count = max(0, rows_actual - rows_common)
    penalty = float(missing_driver_penalty_scale) * float(missing_count) / max(1.0, float(rows_actual)) * float(rows_actual)
    penalized_mae = float(mae + penalty) if mae is not None else None

    match_by_pred = dict(zip(merged["pred_index"], merged["actual_index"]))
    predicted_top10 = {
        match_by_pred[idx]
        for idx in pred_unique.sort_values("pred_rank", kind="mergesort").head(10).index
        if idx in match_by_pred
    }
    actual_top10 = set(actual_unique[actual_unique["actual_rank"] <= 10].index.tolist())
    top10_hit = None
    if actual_top10:
        top10_hit = float(len(predicted_top10.intersection(actual_top10)) / float(min(10, len(actual_top10))))

    predicted_top3 = {
        match_by_pred[idx]
        for idx in pred_unique.sort_values("pred_rank", kind="mergesort").head(3).index
        if idx in match_by_pred
    }
    actual_top3 = set(actual_unique[actual_unique["actual_rank"] <= 3].index.tolist())
    top3_hit = None
    if actual_top3:
        top3_hit = float(len(predicted_top3.intersection(actual_top3)) / float(min(3, len(actual_top3))))

    pred_top10_indices = set(pred_unique.sort_values("pred_rank", kind="mergesort").head(10).index.tolist())
    pred_top10 = merged[merged["pred_index"].isin(pred_top10_indices)].copy()
    pred_top10_mae = float((pred_top10["pred_rank"] - pred_top10["actual_rank"]).abs().mean()) if not pred_top10.empty else None

    payload: dict[str, Any] = {
        "available": True,
        "rows_predicted": int(len(pred_unique)),
        "rows_actual": rows_actual,
        "rows_common": rows_common,
        "field_coverage": field_coverage,
        "coverage_threshold": float(min_field_coverage),
        "coverage_passed": coverage_passed,
        "complete_field": complete_field,
        "metric_available": metric_available,
        "mae_valid": metric_available,
        "mae": mae if metric_available else None,
        "field_mae": mae if metric_available else None,
        "mae_on_common": mae,
        "rmse": rmse if metric_available else None,
        "rmse_on_common": rmse,
        "spearman": spearman if metric_available else None,
        "spearman_on_common": spearman,
        "field_mae_penalized": penalized_mae,
        "evaluation_reason": "ok" if metric_available else "field_coverage_failed",
        "top3_hit": top3_hit if metric_available else None,
        "top3_hit_on_common": top3_hit,
        "top10_hit": top10_hit if metric_available else None,
        "top10_hit_on_common": top10_hit,
        "pred_top10_mae": pred_top10_mae if metric_available else None,
        "pred_top10_mae_on_common": pred_top10_mae,
        **diagnostics,
    }

    if include_podium_and_winner:
        payload["podium_hit_count"] = float(len(predicted_top3.intersection(actual_top3))) if actual_top3 else None

        winner_pred_key = [
            match_by_pred[idx]
            for idx in pred_unique.sort_values("pred_rank", kind="mergesort").head(1).index
            if idx in match_by_pred
        ]
        winner_actual_key = actual_unique.sort_values("actual_rank", kind="mergesort").head(1).index.tolist()
        payload["winner_hit"] = None
        if winner_pred_key and winner_actual_key:
            payload["winner_hit"] = bool(winner_pred_key[0] == winner_actual_key[0])

    if include_match_rows:
        match_debug = merged[["pred_index", "actual_index", "matched_alias", "pred_rank", "actual_rank"]].copy()
        def _json_scalar(value: Any) -> Any:
            if hasattr(value, "item"):
                try:
                    return value.item()
                except Exception:
                    return value
            return value

        payload["match_rows"] = [
            {
                "pred_index": _json_scalar(row["pred_index"]),
                "actual_index": _json_scalar(row["actual_index"]),
                "matched_alias": row["matched_alias"],
                "pred_rank": float(row["pred_rank"]),
                "actual_rank": float(row["actual_rank"]),
            }
            for _, row in match_debug.iterrows()
        ]

    return payload
