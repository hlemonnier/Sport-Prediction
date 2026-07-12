"""Evaluation helpers for live state-space replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:  # pragma: no cover - optional dependency
    ARIMA = None


@dataclass
class ReplayMetrics:
    mae: Optional[float]
    rmse: Optional[float]
    nll_like: Optional[float]
    rows: int


def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_std: Optional[np.ndarray]) -> ReplayMetrics:
    if y_true.size == 0 or y_pred.size == 0 or y_true.size != y_pred.size:
        return ReplayMetrics(mae=None, rmse=None, nll_like=None, rows=0)

    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    if y_std is None or y_std.size != y_true.size:
        nll_like = None
    else:
        sigma = np.clip(y_std, 1e-3, None)
        nll = 0.5 * np.log(2.0 * np.pi * (sigma**2)) + 0.5 * ((err / sigma) ** 2)
        nll_like = float(np.mean(nll))

    return ReplayMetrics(mae=mae, rmse=rmse, nll_like=nll_like, rows=int(y_true.size))


def _collect_eval_rows(trace: pd.DataFrame, warmup_laps: int = 3) -> pd.DataFrame:
    if trace.empty:
        return pd.DataFrame()
    frame = trace.copy()
    frame["eval_included"] = frame.get("eval_included", False).astype(bool)
    frame["lap_time_seconds"] = pd.to_numeric(frame.get("lap_time_seconds"), errors="coerce")
    frame["one_step_pred_mean"] = pd.to_numeric(frame.get("one_step_pred_mean"), errors="coerce")
    frame["one_step_pred_std"] = pd.to_numeric(frame.get("one_step_pred_std"), errors="coerce")
    frame["assim_laps_driver"] = pd.to_numeric(frame.get("assim_laps_driver"), errors="coerce").fillna(0)

    frame = frame[
        frame["eval_included"]
        & frame["lap_time_seconds"].notna()
        & frame["one_step_pred_mean"].notna()
        & (frame["assim_laps_driver"] >= int(warmup_laps))
    ].copy()
    return frame


def _collect_baseline_history_rows(trace: pd.DataFrame) -> pd.DataFrame:
    """Return every causal clean observation available to live baselines.

    Warm-up rows are valid history even though they are not scored.  Building a
    baseline only from scored rows incorrectly removes the immediately prior
    lap for each driver and makes comparator populations differ.
    """

    if trace.empty:
        return pd.DataFrame()
    frame = trace.copy()
    frame["eval_included"] = frame.get("eval_included", False).astype(bool)
    frame["lap_time_seconds"] = pd.to_numeric(frame.get("lap_time_seconds"), errors="coerce")
    return frame[frame["eval_included"] & frame["lap_time_seconds"].notna()].copy()


def _naive_baseline_predictions(frame: pd.DataFrame) -> pd.Series:
    out = pd.Series(index=frame.index, dtype=float)
    for _, idx in frame.groupby("driver_id", sort=False).groups.items():
        subset = frame.loc[idx].sort_values("lap_number", kind="mergesort")
        pred = subset["lap_time_seconds"].shift(1)
        out.loc[subset.index] = pred
    return out


def _arima_baseline_predictions(frame: pd.DataFrame) -> tuple[pd.Series, bool]:
    out = pd.Series(index=frame.index, dtype=float)
    if ARIMA is None:
        return out, False

    available = False
    for _, idx in frame.groupby("driver_id", sort=False).groups.items():
        subset = frame.loc[idx].sort_values("lap_number", kind="mergesort")
        y = pd.to_numeric(subset["lap_time_seconds"], errors="coerce")
        if len(y) <= 8:
            continue
        # Expanding-origin forecasts only.  The former implementation fit once
        # on the full evaluation series and reported in-sample predictions.
        for position in range(8, len(y)):
            history = y.iloc[:position].dropna().to_numpy(dtype=float)
            if len(history) < 8:
                continue
            try:
                fitted = ARIMA(
                    history,
                    order=(2, 1, 2),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()
                forecast = np.asarray(fitted.forecast(steps=1), dtype=float)
                if forecast.size and np.isfinite(forecast[0]):
                    out.loc[subset.index[position]] = float(forecast[0])
                    available = True
            except Exception:
                continue

    return out, available


def evaluate_live_replay(trace: pd.DataFrame, warmup_laps: int = 3) -> dict[str, object]:
    eval_rows = _collect_eval_rows(trace, warmup_laps=warmup_laps)
    if eval_rows.empty:
        return {
            "available": False,
            "reason": "insufficient_eval_rows",
            "rows": 0,
        }

    y_true = eval_rows["lap_time_seconds"].to_numpy(dtype=float)
    y_pred = eval_rows["one_step_pred_mean"].to_numpy(dtype=float)
    y_std = eval_rows["one_step_pred_std"].to_numpy(dtype=float)

    model_metrics = _safe_metrics(y_true, y_pred, y_std)

    baseline_history = _collect_baseline_history_rows(trace)
    naive_pred = _naive_baseline_predictions(baseline_history).reindex(eval_rows.index)
    naive_valid = naive_pred.notna()
    model_on_naive_metrics = _safe_metrics(
        eval_rows.loc[naive_valid, "lap_time_seconds"].to_numpy(dtype=float),
        eval_rows.loc[naive_valid, "one_step_pred_mean"].to_numpy(dtype=float),
        eval_rows.loc[naive_valid, "one_step_pred_std"].to_numpy(dtype=float),
    )
    naive_metrics = _safe_metrics(
        eval_rows.loc[naive_valid, "lap_time_seconds"].to_numpy(dtype=float),
        naive_pred.loc[naive_valid].to_numpy(dtype=float),
        None,
    )

    arima_pred, arima_available = _arima_baseline_predictions(baseline_history)
    arima_pred = arima_pred.reindex(eval_rows.index)
    arima_valid = arima_pred.notna()
    model_on_arima_metrics = _safe_metrics(
        eval_rows.loc[arima_valid, "lap_time_seconds"].to_numpy(dtype=float),
        eval_rows.loc[arima_valid, "one_step_pred_mean"].to_numpy(dtype=float),
        eval_rows.loc[arima_valid, "one_step_pred_std"].to_numpy(dtype=float),
    )
    arima_metrics = _safe_metrics(
        eval_rows.loc[arima_valid, "lap_time_seconds"].to_numpy(dtype=float),
        arima_pred.loc[arima_valid].to_numpy(dtype=float),
        None,
    )

    payload: dict[str, object] = {
        "available": True,
        "rows": int(len(eval_rows)),
        "model": {
            "mae": model_metrics.mae,
            "rmse": model_metrics.rmse,
            "nll_like": model_metrics.nll_like,
            "rows": model_metrics.rows,
            "population": "all_model_valid_rows",
        },
        "model_on_naive_rows": {
            "mae": model_on_naive_metrics.mae,
            "rmse": model_on_naive_metrics.rmse,
            "nll_like": model_on_naive_metrics.nll_like,
            "rows": model_on_naive_metrics.rows,
            "population": "matched_model_and_naive_rows",
        },
        "naive_last_lap": {
            "mae": naive_metrics.mae,
            "rmse": naive_metrics.rmse,
            "rows": naive_metrics.rows,
            "population": "matched_model_and_naive_rows",
        },
        "arima_212": {
            "available": bool(arima_available and arima_metrics.rows > 0),
            "mae": arima_metrics.mae,
            "rmse": arima_metrics.rmse,
            "rows": arima_metrics.rows,
            "population": "matched_model_and_causal_expanding_arima_rows",
            "fit_contract": "expanding_origin_prior_rows_only",
        },
        "model_on_arima_rows": {
            "mae": model_on_arima_metrics.mae,
            "rmse": model_on_arima_metrics.rmse,
            "nll_like": model_on_arima_metrics.nll_like,
            "rows": model_on_arima_metrics.rows,
            "population": "matched_model_and_causal_expanding_arima_rows",
        },
        "baseline_arima_unavailable": not bool(arima_available and arima_metrics.rows > 0),
    }

    if model_on_naive_metrics.mae is not None and naive_metrics.mae is not None:
        payload["mae_gain_vs_naive"] = float(naive_metrics.mae - model_on_naive_metrics.mae)
    if model_on_naive_metrics.rmse is not None and naive_metrics.rmse is not None:
        payload["rmse_gain_vs_naive"] = float(naive_metrics.rmse - model_on_naive_metrics.rmse)
    if model_on_arima_metrics.mae is not None and arima_metrics.mae is not None:
        payload["mae_gain_vs_arima"] = float(arima_metrics.mae - model_on_arima_metrics.mae)
    if model_on_arima_metrics.rmse is not None and arima_metrics.rmse is not None:
        payload["rmse_gain_vs_arima"] = float(arima_metrics.rmse - model_on_arima_metrics.rmse)

    return payload
