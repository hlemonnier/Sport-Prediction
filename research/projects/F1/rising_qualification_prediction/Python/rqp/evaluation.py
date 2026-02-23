"""Shared evaluation helpers for offline/local prediction checks."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def _normalize_name_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    return " ".join(text.split())


def _actual_name_column(frame: pd.DataFrame) -> Optional[str]:
    for col in ["driver_name", "driver_id", "Abbreviation", "Driver"]:
        if col in frame.columns:
            return col
    return None


def evaluate_prediction_rows(
    predicted_rows: list[dict[str, Any]],
    actual_results: pd.DataFrame,
    actual_position_col: str,
    *,
    include_podium_and_winner: bool = False,
) -> dict[str, Any]:
    if not predicted_rows:
        return {"available": False, "reason": "prediction_rows_unavailable"}
    if actual_results is None or actual_results.empty:
        return {"available": False, "reason": "actual_results_unavailable"}

    pred = pd.DataFrame(predicted_rows).copy()
    if pred.empty or "driver_name" not in pred.columns:
        return {"available": False, "reason": "prediction_driver_name_unavailable"}
    pred["driver_key"] = pred["driver_name"].map(_normalize_name_key)
    pred = pred[pred["driver_key"] != ""]
    if pred.empty:
        return {"available": False, "reason": "prediction_driver_key_unavailable"}
    if "rank" in pred.columns:
        pred["pred_rank"] = pd.to_numeric(pred["rank"], errors="coerce")
    else:
        pred["pred_rank"] = pd.Series(range(1, len(pred) + 1), index=pred.index, dtype=float)
    pred = pred.dropna(subset=["pred_rank"])
    pred["pred_rank"] = pred["pred_rank"].astype(float)

    actual = actual_results.copy()
    if actual_position_col not in actual.columns:
        return {"available": False, "reason": "actual_position_unavailable"}
    name_col = _actual_name_column(actual)
    if name_col is None:
        return {"available": False, "reason": "actual_driver_name_unavailable"}
    actual["driver_key"] = actual[name_col].map(_normalize_name_key)
    actual["actual_rank"] = pd.to_numeric(actual[actual_position_col], errors="coerce")
    actual = actual[(actual["driver_key"] != "") & actual["actual_rank"].notna()]
    if actual.empty:
        return {"available": False, "reason": "actual_clean_unavailable"}

    pred_unique = pred.sort_values("pred_rank", kind="mergesort").drop_duplicates(subset=["driver_key"], keep="first")
    actual_unique = actual.sort_values("actual_rank", kind="mergesort").drop_duplicates(
        subset=["driver_key"],
        keep="first",
    )
    merged = pred_unique.merge(actual_unique[["driver_key", "actual_rank"]], on="driver_key", how="inner")
    merged = merged.dropna(subset=["pred_rank", "actual_rank"])

    mae = float((merged["pred_rank"] - merged["actual_rank"]).abs().mean()) if not merged.empty else None
    predicted_top10 = set(pred_unique.sort_values("pred_rank").head(10)["driver_key"].tolist())
    actual_top10 = set(actual_unique[actual_unique["actual_rank"] <= 10]["driver_key"].tolist())
    top10_hit = None
    if actual_top10:
        top10_hit = float(len(predicted_top10.intersection(actual_top10)) / float(min(10, len(actual_top10))))

    payload: dict[str, Any] = {
        "available": True,
        "rows_predicted": int(len(pred_unique)),
        "rows_actual": int(len(actual_unique)),
        "rows_common": int(len(merged)),
        "mae_on_common": mae,
        "top10_hit": top10_hit,
    }

    if include_podium_and_winner:
        predicted_top3 = set(pred_unique.sort_values("pred_rank").head(3)["driver_key"].tolist())
        actual_top3 = set(actual_unique[actual_unique["actual_rank"] <= 3]["driver_key"].tolist())
        payload["podium_hit_count"] = float(len(predicted_top3.intersection(actual_top3))) if actual_top3 else None

        winner_pred_key = pred_unique.sort_values("pred_rank").head(1)["driver_key"].tolist()
        winner_actual_key = actual_unique.sort_values("actual_rank").head(1)["driver_key"].tolist()
        payload["winner_hit"] = None
        if winner_pred_key and winner_actual_key:
            payload["winner_hit"] = bool(winner_pred_key[0] == winner_actual_key[0])

    return payload
