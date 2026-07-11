"""Shared evaluation helpers for offline/local prediction checks."""

from __future__ import annotations

import math
from typing import Any, Optional

import pandas as pd

from packages.f1.data.schemas.driver import driver_key_column as _driver_key_column
from packages.f1.data.schemas.driver import resolve_driver_matches as _resolve_driver_matches


_PROBABILITY_METRICS: tuple[tuple[str, str, int], ...] = (
    ("win", "proba_win", 1),
    ("top3", "proba_top3", 3),
    ("top10", "proba_top10", 10),
)
_ECE_BINS = 10


def _finite_numeric_series(values: pd.Series) -> Optional[pd.Series]:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if numeric.empty or numeric.isna().any():
        return None
    if not all(math.isfinite(float(value)) for value in numeric.tolist()):
        return None
    return numeric


def _kendall_tau_b(predicted: pd.Series, actual: pd.Series) -> Optional[float]:
    """Compute Kendall's tau-b without requiring SciPy."""

    predicted_values = _finite_numeric_series(predicted)
    actual_values = _finite_numeric_series(actual)
    if predicted_values is None or actual_values is None or len(predicted_values) != len(actual_values):
        return None
    if len(predicted_values) < 2:
        return None

    concordant = 0
    discordant = 0
    tied_predicted_only = 0
    tied_actual_only = 0
    pairs = list(zip(predicted_values.tolist(), actual_values.tolist()))
    for left_index, (left_predicted, left_actual) in enumerate(pairs[:-1]):
        for right_predicted, right_actual in pairs[left_index + 1 :]:
            predicted_delta = left_predicted - right_predicted
            actual_delta = left_actual - right_actual
            if predicted_delta == 0.0 and actual_delta == 0.0:
                continue
            if predicted_delta == 0.0:
                tied_predicted_only += 1
            elif actual_delta == 0.0:
                tied_actual_only += 1
            elif predicted_delta * actual_delta > 0.0:
                concordant += 1
            else:
                discordant += 1

    comparable = concordant + discordant
    denominator = math.sqrt(
        float(comparable + tied_predicted_only) * float(comparable + tied_actual_only),
    )
    if denominator <= 0.0:
        return None
    return float((concordant - discordant) / denominator)


def _binary_probability_metrics(
    actual_ranks: pd.Series,
    probabilities: pd.Series,
    *,
    cutoff: int,
    bins: int = _ECE_BINS,
) -> Optional[dict[str, float]]:
    ranks = _finite_numeric_series(actual_ranks)
    values = _finite_numeric_series(probabilities)
    if ranks is None or values is None or len(ranks) != len(values):
        return None
    if ((values < 0.0) | (values > 1.0)).any():
        return None

    labels = [1.0 if float(rank) <= float(cutoff) else 0.0 for rank in ranks.tolist()]
    predicted = [float(value) for value in values.tolist()]
    sample_size = len(labels)
    if sample_size == 0:
        return None

    epsilon = 1e-15
    log_loss = -sum(
        label * math.log(min(1.0 - epsilon, max(epsilon, probability)))
        + (1.0 - label) * math.log(min(1.0 - epsilon, max(epsilon, 1.0 - probability)))
        for label, probability in zip(labels, predicted)
    ) / float(sample_size)
    brier = sum((probability - label) ** 2 for label, probability in zip(labels, predicted)) / float(sample_size)

    bin_count = max(2, int(bins))
    counts = [0] * bin_count
    probability_sums = [0.0] * bin_count
    label_sums = [0.0] * bin_count
    for label, probability in zip(labels, predicted):
        bin_index = min(int(probability * bin_count), bin_count - 1)
        counts[bin_index] += 1
        probability_sums[bin_index] += probability
        label_sums[bin_index] += label
    ece = 0.0
    for count, probability_sum, label_sum in zip(counts, probability_sums, label_sums):
        if count == 0:
            continue
        average_probability = probability_sum / float(count)
        observed_frequency = label_sum / float(count)
        ece += (float(count) / float(sample_size)) * abs(observed_frequency - average_probability)

    return {
        "log_loss": float(log_loss),
        "brier": float(brier),
        "ece": float(ece),
        "total": float(sum(predicted)),
    }


def _matched_prediction_values(
    predicted: pd.DataFrame,
    matched: pd.DataFrame,
    column: str,
) -> Optional[pd.Series]:
    if column not in predicted.columns:
        return None
    values_by_index = pd.to_numeric(predicted[column], errors="coerce")
    return _finite_numeric_series(matched["pred_index"].map(values_by_index))


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
    unexpected_prediction_count = max(0, int(len(pred_unique)) - rows_common)
    missing_actual_count = max(0, rows_actual - rows_common)
    complete_field = bool(
        rows_common == rows_actual
        and rows_common == int(len(pred_unique))
    )
    metric_available = bool(mae is not None and coverage_passed and (complete_field or not require_complete_field))
    mismatch_count = missing_actual_count + unexpected_prediction_count
    penalty = float(missing_driver_penalty_scale) * float(mismatch_count)
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

    predicted_top5 = {
        match_by_pred[idx]
        for idx in pred_unique.sort_values("pred_rank", kind="mergesort").head(5).index
        if idx in match_by_pred
    }
    actual_top5 = set(actual_unique[actual_unique["actual_rank"] <= 5].index.tolist())
    top5_hit = None
    if actual_top5:
        top5_hit = float(len(predicted_top5.intersection(actual_top5)) / float(min(5, len(actual_top5))))

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

    # The additional metrics describe a complete event distribution, so they
    # fail closed unless prediction and result fields match one-to-one. This is
    # deliberately stricter than diagnostics computed on the common subset.
    full_field_metric_available = bool(
        metric_available
        and complete_field
        and rows_common == len(pred_unique)
        and rows_common == rows_actual
    )
    exact_position_accuracy = None
    kendall_tau_b = None
    if full_field_metric_available:
        exact_position_accuracy = float((merged["pred_rank"] == merged["actual_rank"]).mean())
        kendall_tau_b = _kendall_tau_b(merged["pred_rank"], merged["actual_rank"])

    probability_values: dict[str, Optional[float]] = {}
    for label, probability_column, cutoff in _PROBABILITY_METRICS:
        metrics = None
        if full_field_metric_available:
            probabilities = _matched_prediction_values(pred_unique, merged, probability_column)
            if probabilities is not None:
                metrics = _binary_probability_metrics(merged["actual_rank"], probabilities, cutoff=cutoff)
        probability_values[f"{label}_log_loss"] = metrics["log_loss"] if metrics is not None else None
        probability_values[f"{label}_brier"] = metrics["brier"] if metrics is not None else None
        probability_values[f"{label}_ece"] = metrics["ece"] if metrics is not None else None
        probability_values[f"{probability_column}_total"] = metrics["total"] if metrics is not None else None

    position_interval_coverage = None
    position_interval_mean_width = None
    if full_field_metric_available:
        lower = _matched_prediction_values(pred_unique, merged, "pos_p10")
        upper = _matched_prediction_values(pred_unique, merged, "pos_p90")
        if lower is not None and upper is not None and (lower <= upper).all():
            actual_ranks = merged["actual_rank"].astype(float)
            position_interval_coverage = float(((lower <= actual_ranks) & (actual_ranks <= upper)).mean())
            position_interval_mean_width = float((upper - lower).mean())

    payload: dict[str, Any] = {
        "available": True,
        "rows_predicted": int(len(pred_unique)),
        "rows_actual": rows_actual,
        "rows_common": rows_common,
        "field_coverage": field_coverage,
        "coverage_threshold": float(min_field_coverage),
        "coverage_passed": coverage_passed,
        "complete_field": complete_field,
        "missing_actual_count": missing_actual_count,
        "unexpected_prediction_count": unexpected_prediction_count,
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
        "evaluation_reason": (
            "ok"
            if metric_available
            else "field_roster_mismatch"
            if coverage_passed and require_complete_field and not complete_field
            else "field_coverage_failed"
        ),
        "exact_position_accuracy": exact_position_accuracy,
        "top3_hit": top3_hit if metric_available else None,
        "top3_hit_on_common": top3_hit,
        "top5_hit": top5_hit if full_field_metric_available else None,
        "top10_hit": top10_hit if metric_available else None,
        "top10_hit_on_common": top10_hit,
        "pred_top10_mae": pred_top10_mae if metric_available else None,
        "pred_top10_mae_on_common": pred_top10_mae,
        "kendall_tau_b": kendall_tau_b,
        "position_interval_coverage": position_interval_coverage,
        "position_interval_mean_width": position_interval_mean_width,
        **probability_values,
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
