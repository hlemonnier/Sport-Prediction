"""Frozen nonlinear next-eligible-lap point forecasts used by both F1 runtimes.

This module never trains, downloads or unpickles a model. The bundled numeric
trees are hash pinned. The 2022–23 fit passed retrospective 2024–26 transfer;
neither intervals, order probabilities nor strategy value are certified by it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from os import environ
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .next_lap_features import KEYS, REQUIRED, observed_features, driver_key

MODEL_ID = "frontier_live_hgb_l15_i150_20260907"
MODEL_SHA256 = "d8bd93f1b04ecf290eb821f02397253b77248cf3c1943b05e1c540d4942e7e36"
MODEL_PATH = Path(__file__).with_name("assets") / f"{MODEL_ID}.json"
TARGET = "same_driver_next_eligible_clean_completed_lap_may_skip_numbered_laps"
EVIDENCE_STATUS = "retrospective_multi_season_transfer_passed_not_prospective"
SECONDS_COLUMNS = [
    "Time",
    "LapTime",
    "PitInTime",
    "PitOutTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
]
NUMERIC_COLUMNS = SECONDS_COLUMNS + [
    "LapNumber",
    "Stint",
    "TyreLife",
    "Position",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
]


@lru_cache(maxsize=1)
def load_model() -> dict[str, Any]:
    raw = MODEL_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MODEL_SHA256:
        raise ValueError("bundled next-lap model failed its SHA256 integrity check")
    model = json.loads(raw)
    if model["format"] != "numeric_hgb_regression_v1" or model["model_id"] != MODEL_ID:
        raise ValueError("unsupported next-lap model format or identity")
    # All node arrays stay private and read-only after loading.
    arrays = [np.asarray(tree, dtype=np.float64) for tree in model["trees"]]
    for array in arrays:
        array.setflags(write=False)
    model["tree_arrays"] = arrays
    return model


def predict_correction(
    features: pd.DataFrame, model: dict[str, Any] | None = None
) -> np.ndarray:
    """Evaluate raw numeric splits in the same order as the fitted HGB model."""
    model = load_model() if model is None else model
    x = features[model["features"]].to_numpy(dtype=np.float64)
    prediction = np.full(len(x), model["baseline"], dtype=np.float64)
    for tree in model["tree_arrays"]:
        nodes = np.zeros(len(x), dtype=np.intp)
        active = np.arange(len(x))
        while len(active):
            current = tree[nodes[active]]
            leaf = current[:, 5].astype(bool)
            prediction[active[leaf]] += current[leaf, 6]
            active = active[~leaf]
            current = current[~leaf]
            values = x[active, current[:, 0].astype(np.intp)]
            left = np.where(
                np.isnan(values), current[:, 4].astype(bool), values <= current[:, 1]
            )
            nodes[active] = np.where(left, current[:, 2], current[:, 3]).astype(np.intp)
    return np.clip(
        prediction, -model["correction_clip_seconds"], model["correction_clip_seconds"]
    )


@dataclass
class ForecastBatch:
    frame: pd.DataFrame
    status: str
    reason: str | None = None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "model_id": MODEL_ID,
            "model_sha256": MODEL_SHA256,
            "status": self.status,
            "reason": self.reason,
            "issuances": len(self.frame),
            "target": TARGET,
            "units": "seconds",
            "point_statistic": "conditional_median_estimate",
            "evidence_status": EVIDENCE_STATUS,
            "interval_calibrated": False,
        }


def forecast_laps(
    observations: pd.DataFrame, event_key: int, *, enabled: bool = True
) -> ForecastBatch:
    """Forecast every causal issuance; reject absent columns instead of filling them.

    Row-level missing measurements present in the source are encoded with the
    frozen missing indicators. An absent source column is a different contract
    and triggers baseline fallback. No future target is required to issue.
    """
    if not enabled:
        return ForecastBatch(pd.DataFrame(), "disabled", "configured_baseline")
    try:
        features = observed_features(observations, event_key)
        if features.empty:
            return ForecastBatch(
                pd.DataFrame(), "warming_up", "three_eligible_observations_required"
            )
        result = features[KEYS + ["forecast_naive_seconds"]].copy()
        result["forecast_seconds"] = result.forecast_naive_seconds + predict_correction(
            features
        )
        if (
            not np.isfinite(result.forecast_seconds).all()
            or not result.forecast_seconds.gt(0).all()
        ):
            raise ValueError("next-lap inference produced an invalid duration")
        return ForecastBatch(result, "available")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        return ForecastBatch(pd.DataFrame(), "fallback", f"{type(exc).__name__}: {exc}")


def snapshot_lap_forecasts(
    snapshot: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """API adapter for observed, completed lap history in the platform snapshot.

    The history is supplied by the reducer, never recovered from future files
    or rebuilt using the latest driver's tyre/track state. ``asOf`` is in the
    same session-clock seconds as Time. Unknown/future timestamps fail closed.
    """
    rows = snapshot.get("lapObservations", [])
    info = snapshot.get("sessionInfo") or snapshot.get("session_info") or {}
    session_type = (
        str(info.get("session_type") or info.get("session_name") or "").strip().lower()
    )
    setting = environ.get("F1_LIVE_NEXT_LAP_POINT_MODEL", "frontier_hgb")
    if setting not in {"frontier_hgb", "baseline"}:
        raise ValueError(
            "F1_LIVE_NEXT_LAP_POINT_MODEL must be frontier_hgb or baseline"
        )
    if setting == "baseline":
        batch = ForecastBatch(pd.DataFrame(), "disabled", "configured_baseline")
    elif session_type not in {"race", "r", "grand prix"}:
        batch = ForecastBatch(
            pd.DataFrame(), "fallback", "race_session_identity_required"
        )
    elif not isinstance(rows, list) or not rows:
        batch = ForecastBatch(
            pd.DataFrame(), "fallback", "completed_lap_history_unavailable"
        )
    else:
        try:
            if any(
                not isinstance(row, dict) or set(REQUIRED) - set(row) for row in rows
            ):
                raise ValueError("completed-lap history has missing observed columns")
            raw = pd.DataFrame(rows)
            for column in NUMERIC_COLUMNS:
                raw[column] = pd.to_numeric(raw[column], errors="raise").astype(float)
            as_of = float(snapshot["lapObservationsAsOfTimeSeconds"])
            if (
                not np.isfinite(as_of)
                or not np.isfinite(raw.Time).all()
                or raw.Time.gt(as_of).any()
            ):
                raise ValueError(
                    "lap history contains unknown or future completion timestamps"
                )
            # Session identity is metadata only; no numerical feature uses it.
            batch = forecast_laps(raw, 0)
        except (ValueError, TypeError, KeyError) as exc:
            batch = ForecastBatch(
                pd.DataFrame(), "fallback", f"{type(exc).__name__}: {exc}"
            )
    forecasts = {}
    if not batch.frame.empty:
        latest = (
            batch.frame.sort_values("issued_at_timestamp")
            .groupby("driver_id", sort=False)
            .tail(1)
        )
        for row in latest.to_dict("records"):
            forecasts[driver_key(row["driver_id"])] = {
                "seconds": float(row["forecast_seconds"]),
                "baseline_seconds": float(row["forecast_naive_seconds"]),
                "model_id": MODEL_ID,
                "model_sha256": MODEL_SHA256,
                "issued_after_lap": int(row["issued_after_lap_number"]),
                "issued_at_session_seconds": float(row["issued_at_timestamp"]),
                "target": TARGET,
                "status": "available",
                "interval_seconds": None,
                "evidence_status": EVIDENCE_STATUS,
            }
    return forecasts, batch.diagnostics()


def lap_time_payload(
    forecast: dict[str, Any] | None,
    diagnostics: dict[str, Any],
    *,
    last_lap: float | None,
    eligible: bool,
) -> dict[str, Any]:
    """Keep a lap-time point target separate from the API's order heuristics."""
    if not eligible:
        return {
            "seconds": None,
            "status": "unavailable",
            "reason": "driver_not_running",
            "target": TARGET,
            "interval_seconds": None,
            "model_id": None,
        }
    if forecast is not None:
        return dict(forecast)
    usable = last_lap is not None and np.isfinite(last_lap) and last_lap > 0
    return {
        "seconds": float(last_lap) if usable else None,
        "baseline_seconds": float(last_lap) if usable else None,
        "model_id": "last_observed_lap_snapshot_baseline_v1" if usable else None,
        "status": "fallback" if usable else "unavailable",
        "reason": diagnostics.get("reason") or "three_eligible_observations_required",
        "target": TARGET,
        "interval_seconds": None,
        "evidence_status": "snapshot_baseline_clean_lap_eligibility_unverified",
    }
