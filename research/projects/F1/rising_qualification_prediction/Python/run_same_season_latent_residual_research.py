#!/usr/bin/env python3
"""Bounded same-season research for pre-Qualifying pace residuals.

This script is deliberately separate from the production Qualifying and
Best-Lap runners.  It answers one narrow question: after observing only prior
2026 weekends, can a small residual model improve both the session-best lap
estimate and the rank induced by that estimate?

The chronology is enforced in code.  For every event, all candidate forecasts
are generated and hashed before the target loader is called.  Candidate and
hyperparameter selection uses complete event blocks before a mechanically
frozen audit block.  By default, every forecast and fitted-state manifest for
that block is frozen before the first audit target is read.  Because the 2026
outcomes were inspected during development, this remains a post-development
replay diagnostic rather than independent prospective promotion evidence.  An
explicitly named prequential diagnostic remains available for comparison.
"""

from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

try:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import HuberRegressor, Ridge
except Exception as error:  # pragma: no cover - exercised by runtime doctoring
    raise RuntimeError("scikit-learn is required for this research harness") from error

from packages.f1.models.pre_quali.pairwise import (
    PairwiseRankerConfig,
    fit_pairwise_qualifying_ranker,
)
from packages.f1.models.ultimate_lap_time.achievable import ACTUAL_LAP_COLUMN
from run_qualifying_pairwise_challenger_backtest import (
    _event_inference_frame,
    _load_target_after_frozen_forecast,
    _root,
    _round_number,
)


SCHEMA_VERSION = "f1_same_season_latent_residual_research_v3"
MODEL_FAMILY = "same_season_event_weighted_latent_residual_research"
BASELINE_SECONDS_COLUMN = "research_rehearsal_seconds"
QUALIFYING_TARGET_COLUMN = "qualy_position"
PAIRWISE_MODEL_ID = "current_pairwise_ranker_c0_5_move3_v1"
PAIRWISE_MINIMUM_TRAINING_EVENTS = 4
RESIDUAL_MINIMUM_TRAINING_EVENTS = 3
DEFAULT_SEED = 20260713

# This allowlist is intentionally compact.  It contains no outcome, driver,
# team, circuit, or event identity.  Every value is available at inference
# time and receives an explicit training-median imputation plus missing flag.
RESIDUAL_FEATURE_ALLOWLIST: tuple[str, ...] = (
    "field_relative_anchor_seconds",
    "teammate_relative_anchor_seconds",
    "anchor_uncertainty_seconds",
    "valid_minus_potential_seconds",
    "best_two_spread_seconds",
    "deleted_potential_lap_count",
    "best_lap_session_progress",
    "evidence_coverage_rate",
)

# The existing pairwise model also receives an explicit, causal subset.  Its
# implementation uses training medians for missing values; that difference is
# recorded in the output rather than silently pretending the designs match.
PAIRWISE_FEATURE_ALLOWLIST: tuple[str, ...] = RESIDUAL_FEATURE_ALLOWLIST

FORBIDDEN_CURRENT_EVENT_TARGET_COLUMNS = frozenset(
    {
        ACTUAL_LAP_COLUMN,
        QUALIFYING_TARGET_COLUMN,
        "actual_best_lap_seconds",
        "actual_qualifying_position",
        "qualifying_best_lap_time_seconds",
        "qualifying_position",
        "q1_time",
        "q2_time",
        "q3_time",
        "qualifying_q1_lap_time_seconds",
        "qualifying_q2_lap_time_seconds",
        "qualifying_q3_lap_time_seconds",
        "has_valid_qualifying_lap",
        "reached_q2",
        "reached_q3",
    }
)


@dataclass(frozen=True)
class ResidualCandidate:
    model_id: str
    family: str
    alpha: float | None = None
    epsilon: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "family": self.family,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
        }


# Predeclared before any event is read.  The grid is intentionally small: two
# ridge penalties and two Huber robustness settings, all on the same features.
RESIDUAL_CANDIDATE_GRID: tuple[ResidualCandidate, ...] = (
    ResidualCandidate("raw_rehearsal_v1", "raw_rehearsal"),
    ResidualCandidate("expanding_source_shift_v1", "source_shift"),
    ResidualCandidate("event_weighted_ridge_alpha_2_v1", "ridge", alpha=2.0),
    ResidualCandidate("event_weighted_ridge_alpha_10_v1", "ridge", alpha=10.0),
    ResidualCandidate(
        "event_weighted_huber_epsilon_1_35_v1",
        "huber",
        alpha=0.01,
        epsilon=1.35,
    ),
    ResidualCandidate(
        "event_weighted_huber_epsilon_1_75_v1",
        "huber",
        alpha=0.01,
        epsilon=1.75,
    ),
)


@dataclass(frozen=True)
class ResearchEvent:
    """A target-free event snapshot plus a deferred evaluation target."""

    event_key: int
    inference: pd.DataFrame
    event_info: Mapping[str, object]
    pre_target_input_paths: tuple[Path, ...] = ()
    target_path: Path | None = None


@dataclass(frozen=True)
class NumericDesign:
    feature_columns: tuple[str, ...]
    feature_names: tuple[str, ...]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        raw = _numeric_feature_matrix(frame, self.feature_columns)
        missing = (~np.isfinite(raw)).astype(float)
        filled = np.where(np.isfinite(raw), raw, self.medians[np.newaxis, :])
        expanded = np.column_stack([filled, missing])
        return (expanded - self.means[np.newaxis, :]) / self.scales[np.newaxis, :]


@dataclass(frozen=True)
class FrozenAuditEventForecast:
    event: ResearchEvent
    inference: pd.DataFrame
    candidate_forecasts: pd.DataFrame
    failures: tuple[dict[str, object], ...]
    candidate_manifests: Mapping[str, Mapping[str, object]]
    artifact: Mapping[str, object]


TargetLoader = Callable[
    [ResearchEvent, Mapping[str, object]],
    tuple[pd.DataFrame, Mapping[str, object]],
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return value


def _payload_sha256(payload: Mapping[str, object] | Sequence[object]) -> str:
    serialized = json.dumps(
        _json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    if frame.empty:
        return hashlib.sha256(b"").hexdigest()
    ordered_columns = sorted(map(str, frame.columns))
    normalized = frame.loc[:, ordered_columns].copy()
    sort_columns = [
        column
        for column in ("event_key", "driver_id", "model_id")
        if column in normalized.columns
    ]
    if sort_columns:
        normalized = normalized.sort_values(sort_columns, kind="mergesort")
    text = normalized.to_csv(index=False, na_rep="<NA>", float_format="%.12g")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_rank(values: Sequence[float], drivers: Sequence[str]) -> np.ndarray:
    """Return a legal rank using provider row order, never driver identity."""

    numeric = np.asarray(values, dtype=float)
    if len(drivers) != len(numeric):
        raise ValueError("rank values and driver rows must be aligned")
    finite_or_inf = np.where(np.isfinite(numeric), numeric, np.inf)
    order = np.lexsort((np.arange(len(numeric)), finite_or_inf))
    ranks = np.empty(len(numeric), dtype=int)
    ranks[order] = np.arange(1, len(numeric) + 1)
    return ranks


def _normalize_inference(event: ResearchEvent, *, expected_year: int) -> pd.DataFrame:
    frame = event.inference.copy()
    forbidden = sorted(FORBIDDEN_CURRENT_EVENT_TARGET_COLUMNS.intersection(frame.columns))
    if forbidden:
        raise ValueError(
            f"event {event.event_key} inference contains evaluation targets: {forbidden}"
        )
    if int(event.event_key) // 100 != int(expected_year):
        raise ValueError(
            f"same-season research forbids event {event.event_key} outside {expected_year}"
        )
    required = {"event_key", "driver_id", "rehearsal_source"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"event {event.event_key} inference is missing columns: {missing}")
    event_keys = pd.to_numeric(frame["event_key"], errors="coerce")
    if (
        frame.empty
        or event_keys.isna().any()
        or set(event_keys.astype(int).tolist()) != {int(event.event_key)}
    ):
        raise ValueError("inference must contain one non-empty matching event block")
    drivers = frame["driver_id"].astype(str).str.strip()
    if drivers.eq("").any() or drivers.duplicated().any():
        raise ValueError("inference requires unique non-empty driver ids")
    frame["driver_id"] = drivers

    valid_clean = pd.to_numeric(frame.get("valid_clean_best_seconds"), errors="coerce")
    quality_anchor = pd.to_numeric(
        frame.get("quality_aware_anchor_seconds"), errors="coerce"
    )
    if not isinstance(valid_clean, pd.Series):
        valid_clean = pd.Series(np.nan, index=frame.index, dtype=float)
    if not isinstance(quality_anchor, pd.Series):
        quality_anchor = pd.Series(np.nan, index=frame.index, dtype=float)
    baseline = valid_clean.where(valid_clean.notna(), quality_anchor)
    if baseline.isna().any():
        missing_drivers = frame.loc[baseline.isna(), "driver_id"].astype(str).tolist()
        raise ValueError(
            f"event {event.event_key} lacks a raw rehearsal time for {missing_drivers}"
        )
    frame[BASELINE_SECONDS_COLUMN] = baseline.astype(float)
    frame["latest_qualifying_rehearsal_source"] = frame.get(
        "latest_qualifying_rehearsal_source", frame["rehearsal_source"]
    )
    frame["latest_qualifying_rehearsal_rank"] = _stable_rank(
        frame[BASELINE_SECONDS_COLUMN], frame["driver_id"]
    )
    for column in set(RESIDUAL_FEATURE_ALLOWLIST).union(PAIRWISE_FEATURE_ALLOWLIST):
        if column not in frame.columns:
            frame[column] = np.nan
    return frame.reset_index(drop=True)


def _event_balanced_row_weights(frame: pd.DataFrame) -> np.ndarray:
    events = pd.to_numeric(frame["event_key"], errors="coerce")
    if events.isna().any():
        raise ValueError("event-balanced weights require finite event keys")
    counts = events.map(events.value_counts()).to_numpy(dtype=float)
    if np.any(counts <= 0.0):
        raise ValueError("event-balanced weights encountered an empty event")
    return 1.0 / counts


def _history_event_keys(frame: pd.DataFrame) -> list[int]:
    if frame.empty or "event_key" not in frame.columns:
        return []
    return sorted(
        pd.to_numeric(frame["event_key"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )


def _weighted_quantile(
    values: Sequence[float], weights: Sequence[float], probability: float
) -> float:
    numeric = np.asarray(values, dtype=float)
    weight_values = np.asarray(weights, dtype=float)
    mask = np.isfinite(numeric) & np.isfinite(weight_values) & (weight_values > 0.0)
    if not mask.any():
        return float("nan")
    numeric = numeric[mask]
    weight_values = weight_values[mask]
    order = np.argsort(numeric, kind="mergesort")
    numeric = numeric[order]
    cumulative = np.cumsum(weight_values[order])
    threshold = float(np.clip(probability, 0.0, 1.0)) * float(cumulative[-1])
    index = min(int(np.searchsorted(cumulative, threshold, side="left")), len(numeric) - 1)
    return float(numeric[index])


def _source_shift_map(history: pd.DataFrame) -> tuple[dict[str, float], float]:
    if history.empty:
        return {}, 0.0
    actual = pd.to_numeric(history[ACTUAL_LAP_COLUMN], errors="coerce")
    baseline = pd.to_numeric(history[BASELINE_SECONDS_COLUMN], errors="coerce")
    usable_mask = actual.notna() & baseline.notna()
    usable = history.loc[usable_mask].reset_index(drop=True)
    if usable.empty:
        return {}, 0.0
    residual = (
        pd.to_numeric(usable[ACTUAL_LAP_COLUMN], errors="coerce")
        - pd.to_numeric(usable[BASELINE_SECONDS_COLUMN], errors="coerce")
    )
    weights = _event_balanced_row_weights(usable)
    global_shift = _weighted_quantile(residual, weights, 0.50)
    global_shift = global_shift if np.isfinite(global_shift) else 0.0
    source_shifts: dict[str, float] = {}
    sources = usable["rehearsal_source"].fillna("unknown").astype(str)
    for source in sorted(sources.unique().tolist()):
        mask = sources.eq(source).to_numpy()
        shift = _weighted_quantile(residual.to_numpy()[mask], weights[mask], 0.50)
        source_shifts[source] = float(shift if np.isfinite(shift) else global_shift)
    return source_shifts, float(global_shift)


def _source_shift_values(
    frame: pd.DataFrame,
    source_shifts: Mapping[str, float],
    global_shift: float,
) -> np.ndarray:
    return (
        frame["rehearsal_source"]
        .fillna("unknown")
        .astype(str)
        .map(lambda value: float(source_shifts.get(value, global_shift)))
        .to_numpy(dtype=float)
    )


def _numeric_feature_matrix(
    frame: pd.DataFrame, feature_columns: Sequence[str]
) -> np.ndarray:
    return np.column_stack(
        [
            pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            for column in feature_columns
        ]
    )


def _fit_numeric_design(
    history: pd.DataFrame,
    weights: np.ndarray,
    feature_columns: tuple[str, ...] = RESIDUAL_FEATURE_ALLOWLIST,
) -> tuple[NumericDesign, np.ndarray]:
    raw = _numeric_feature_matrix(history, feature_columns)
    medians = np.asarray(
        [
            _weighted_quantile(raw[:, column], weights, 0.50)
            for column in range(raw.shape[1])
        ],
        dtype=float,
    )
    medians = np.where(np.isfinite(medians), medians, 0.0)
    missing = (~np.isfinite(raw)).astype(float)
    filled = np.where(np.isfinite(raw), raw, medians[np.newaxis, :])
    expanded = np.column_stack([filled, missing])
    normalized_weights = weights / float(weights.sum())
    means = np.sum(expanded * normalized_weights[:, np.newaxis], axis=0)
    centered = expanded - means[np.newaxis, :]
    scales = np.sqrt(
        np.sum(np.square(centered) * normalized_weights[:, np.newaxis], axis=0)
    )
    scales = np.where(np.isfinite(scales) & (scales > 1e-9), scales, 1.0)
    names = tuple(feature_columns) + tuple(
        f"{column}__missing" for column in feature_columns
    )
    design = NumericDesign(
        feature_columns=tuple(feature_columns),
        feature_names=names,
        medians=medians,
        means=means,
        scales=scales,
    )
    return design, centered / scales[np.newaxis, :]


def _fit_residual_candidate(
    candidate: ResidualCandidate,
    history: pd.DataFrame,
    inference: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, object]]:
    baseline = pd.to_numeric(
        inference[BASELINE_SECONDS_COLUMN], errors="coerce"
    ).to_numpy(dtype=float)
    source_shifts, global_shift = _source_shift_map(history)
    inference_shift = _source_shift_values(
        inference, source_shifts, global_shift
    )
    if candidate.family == "raw_rehearsal":
        manifest: dict[str, object] = {
            "family": candidate.family,
            "training_event_keys": [],
            "source_shifts_seconds": {},
        }
        manifest["fitted_state_sha256"] = _payload_sha256(
            {"candidate": candidate.as_dict(), "state": "identity"}
        )
        return baseline, manifest
    if candidate.family == "source_shift":
        manifest = {
            "family": candidate.family,
            "training_event_keys": _history_event_keys(history),
            "source_shifts_seconds": source_shifts,
            "global_shift_seconds": global_shift,
        }
        manifest["fitted_state_sha256"] = _payload_sha256(
            {
                "candidate": candidate.as_dict(),
                "training_data_sha256": _frame_sha256(history),
                "source_shifts_seconds": source_shifts,
                "global_shift_seconds": global_shift,
            }
        )
        return baseline + inference_shift, manifest
    if candidate.family not in {"ridge", "huber"}:
        raise ValueError(f"unsupported residual candidate family: {candidate.family}")

    event_keys = _history_event_keys(history)
    if len(event_keys) < RESIDUAL_MINIMUM_TRAINING_EVENTS:
        raise ValueError(
            "minimum_training_events_not_met:"
            f"requires={RESIDUAL_MINIMUM_TRAINING_EVENTS},got={len(event_keys)}"
        )
    actual = pd.to_numeric(history[ACTUAL_LAP_COLUMN], errors="coerce")
    history_baseline = pd.to_numeric(
        history[BASELINE_SECONDS_COLUMN], errors="coerce"
    )
    history_shift = _source_shift_values(history, source_shifts, global_shift)
    target = actual.to_numpy(dtype=float) - (
        history_baseline.to_numpy(dtype=float) + history_shift
    )
    finite_target = np.isfinite(target) & np.isfinite(
        history_baseline.to_numpy(dtype=float)
    )
    training = history.loc[finite_target].reset_index(drop=True)
    target = target[finite_target]
    if training.empty:
        raise ValueError("no_finite_best_lap_residual_targets")
    weights = _event_balanced_row_weights(training)
    design, transformed = _fit_numeric_design(training, weights)

    if candidate.family == "ridge":
        estimator: Any = Ridge(
            alpha=float(candidate.alpha),
            fit_intercept=True,
            solver="auto",
        )
    else:
        estimator = HuberRegressor(
            alpha=float(candidate.alpha),
            epsilon=float(candidate.epsilon),
            fit_intercept=True,
            max_iter=200,
            tol=1e-6,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        estimator.fit(transformed, target, sample_weight=weights)
    predicted_residual = np.asarray(
        estimator.predict(design.transform(inference)), dtype=float
    )
    coefficient = np.asarray(estimator.coef_, dtype=float)
    manifest = {
        "family": candidate.family,
        "alpha": candidate.alpha,
        "epsilon": candidate.epsilon,
        "training_event_keys": event_keys,
        "training_rows": int(len(training)),
        "event_weight_sums": {
            str(event_key): float(
                weights[
                    pd.to_numeric(training["event_key"], errors="coerce")
                    .astype(int)
                    .eq(event_key)
                    .to_numpy()
                ].sum()
            )
            for event_key in event_keys
        },
        "feature_names": list(design.feature_names),
        "missingness_policy": "training_event_weighted_median_plus_explicit_indicator",
        "nonzero_coefficients": int(np.count_nonzero(np.abs(coefficient) > 1e-12)),
        "coefficient_l2_norm": float(np.linalg.norm(coefficient)),
        "source_shifts_seconds": source_shifts,
        "global_shift_seconds": global_shift,
    }
    manifest["fitted_state_sha256"] = _payload_sha256(
        {
            "candidate": candidate.as_dict(),
            "training_data_sha256": _frame_sha256(training),
            "feature_names": design.feature_names,
            "medians": design.medians,
            "means": design.means,
            "scales": design.scales,
            "coefficient": coefficient,
            "intercept": np.atleast_1d(np.asarray(estimator.intercept_)),
            "source_shifts_seconds": source_shifts,
            "global_shift_seconds": global_shift,
        }
    )
    return baseline + inference_shift + predicted_residual, manifest


def _pairwise_prediction(
    history: pd.DataFrame,
    inference: pd.DataFrame,
    *,
    target_event_key: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    event_keys = _history_event_keys(history)
    if len(event_keys) < PAIRWISE_MINIMUM_TRAINING_EVENTS:
        raise ValueError(
            "minimum_training_events_not_met:"
            f"requires={PAIRWISE_MINIMUM_TRAINING_EVENTS},got={len(event_keys)}"
        )
    config = PairwiseRankerConfig(
        feature_columns=PAIRWISE_FEATURE_ALLOWLIST,
        minimum_training_events=PAIRWISE_MINIMUM_TRAINING_EVENTS,
        max_movement=3,
        regularization_c=0.5,
        random_state=int(seed),
    )
    model = fit_pairwise_qualifying_ranker(
        history,
        config=config,
        target_event_key=int(target_event_key),
    )
    forecast = model.predict_event(
        inference,
        samples=1,
        seed=int(seed) + int(target_event_key),
    ).point_order.set_index("driver_id")
    positions = inference["driver_id"].astype(str).map(
        forecast["predicted_qualifying_position"]
    )
    if positions.isna().any():
        raise ValueError("pairwise forecast omitted inference drivers")
    manifest: dict[str, object] = {
        "family": "current_pairwise_ranker",
        "training_event_keys": event_keys,
        "minimum_training_events": PAIRWISE_MINIMUM_TRAINING_EVENTS,
        "regularization_c": 0.5,
        "max_movement": 3,
        "feature_columns": list(PAIRWISE_FEATURE_ALLOWLIST),
        "missingness_policy": "current_pairwise_training_median_imputation",
    }
    manifest["fitted_state_sha256"] = _payload_sha256(
        {
            "training_data_sha256": _frame_sha256(history),
            "config": vars(config),
            "training_event_keys": model.training_event_keys,
            "training_pair_counts": model.training_pair_counts,
            "feature_names": model.design.feature_names,
            "medians": model.design.medians,
            "means": model.design.means,
            "scales": model.design.scales,
            "coefficient": np.asarray(model.estimator.coef_),
            "intercept": np.atleast_1d(np.asarray(model.estimator.intercept_)),
        }
    )
    return positions.to_numpy(dtype=int), manifest


def _forecast_event_candidates(
    history: pd.DataFrame,
    inference: pd.DataFrame,
    *,
    event_key: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    manifests: dict[str, dict[str, object]] = {}
    drivers = inference["driver_id"].astype(str).to_numpy()
    baseline = pd.to_numeric(
        inference[BASELINE_SECONDS_COLUMN], errors="coerce"
    ).to_numpy(dtype=float)

    for candidate in RESIDUAL_CANDIDATE_GRID:
        try:
            predicted_lap, manifest = _fit_residual_candidate(
                candidate, history, inference
            )
        except (RuntimeError, ValueError, FloatingPointError) as error:
            failed.append(
                {
                    "event_key": int(event_key),
                    "model_id": candidate.model_id,
                    "stage": "forecast",
                    "reason": str(error),
                }
            )
            continue
        manifests[candidate.model_id] = manifest
        predicted_rank = _stable_rank(predicted_lap, drivers)
        for index, driver in enumerate(drivers):
            rows.append(
                {
                    "event_key": int(event_key),
                    "driver_id": str(driver),
                    "model_id": candidate.model_id,
                    "model_family": candidate.family,
                    "predicted_best_lap_seconds": float(predicted_lap[index]),
                    "predicted_qualifying_position": int(predicted_rank[index]),
                    "raw_rehearsal_seconds": float(baseline[index]),
                    "raw_rehearsal_rank": int(
                        inference.loc[index, "latest_qualifying_rehearsal_rank"]
                    ),
                }
            )

    try:
        pairwise_rank, pairwise_manifest = _pairwise_prediction(
            history,
            inference,
            target_event_key=int(event_key),
            seed=int(seed),
        )
    except (RuntimeError, ValueError, FloatingPointError) as error:
        failed.append(
            {
                "event_key": int(event_key),
                "model_id": PAIRWISE_MODEL_ID,
                "stage": "forecast",
                "reason": str(error),
            }
        )
    else:
        manifests[PAIRWISE_MODEL_ID] = pairwise_manifest
        for index, driver in enumerate(drivers):
            rows.append(
                {
                    "event_key": int(event_key),
                    "driver_id": str(driver),
                    "model_id": PAIRWISE_MODEL_ID,
                    "model_family": "current_pairwise_ranker",
                    "predicted_best_lap_seconds": float("nan"),
                    "predicted_qualifying_position": int(pairwise_rank[index]),
                    "raw_rehearsal_seconds": float(baseline[index]),
                    "raw_rehearsal_rank": int(
                        inference.loc[index, "latest_qualifying_rehearsal_rank"]
                    ),
                }
            )
    return pd.DataFrame(rows), failed, manifests


def _freeze_forecast_artifact(
    *,
    event_key: int,
    history: pd.DataFrame,
    inference: pd.DataFrame,
    candidate_forecasts: pd.DataFrame,
    candidate_manifests: Mapping[str, Mapping[str, object]],
    selected_models: Mapping[str, object] | None,
) -> dict[str, object]:
    target_leaks = sorted(
        FORBIDDEN_CURRENT_EVENT_TARGET_COLUMNS.intersection(inference.columns)
    )
    if target_leaks:
        raise RuntimeError(f"cannot freeze forecast with target columns: {target_leaks}")
    history_keys = _history_event_keys(history)
    if history_keys and max(history_keys) >= int(event_key):
        raise RuntimeError("forecast history reaches or crosses target event")
    payload: dict[str, object] = {
        "schema_version": "f1_same_season_residual_frozen_event_forecast_v2",
        "event_key": int(event_key),
        "training_event_keys": history_keys,
        "training_data_sha256": _frame_sha256(history),
        "inference_data_sha256": _frame_sha256(inference),
        "candidate_forecasts_sha256": _frame_sha256(candidate_forecasts),
        "candidate_model_ids": sorted(
            candidate_forecasts.get("model_id", pd.Series(dtype=str))
            .astype(str)
            .unique()
            .tolist()
        ),
        "candidate_manifests": candidate_manifests,
        "selected_models_frozen_before_target": selected_models,
        "target_columns_present_at_freeze": [],
    }
    payload["artifact_sha256"] = _payload_sha256(payload)
    return payload


def _freeze_complete_audit_forecast_block(
    events: Sequence[ResearchEvent],
    *,
    history: pd.DataFrame,
    selected_models: Mapping[str, object],
    year: int,
    seed: int,
) -> tuple[dict[int, FrozenAuditEventForecast], dict[str, object]]:
    """Materialize the complete audit block before any audit target is read."""

    frozen_history = history.copy(deep=True)
    training_event_keys = _history_event_keys(frozen_history)
    training_data_sha256 = _frame_sha256(frozen_history)
    staged: dict[int, FrozenAuditEventForecast] = {}
    base_artifact_rows: list[dict[str, object]] = []
    for event in sorted(events, key=lambda value: int(value.event_key)):
        event_key = int(event.event_key)
        inference = _normalize_inference(event, expected_year=int(year))
        forecasts, failures, manifests = _forecast_event_candidates(
            frozen_history,
            inference,
            event_key=event_key,
            seed=int(seed),
        )
        artifact = _freeze_forecast_artifact(
            event_key=event_key,
            history=frozen_history,
            inference=inference,
            candidate_forecasts=forecasts,
            candidate_manifests=manifests,
            selected_models=selected_models,
        )
        staged[event_key] = FrozenAuditEventForecast(
            event=event,
            inference=inference,
            candidate_forecasts=forecasts,
            failures=tuple(failures),
            candidate_manifests=manifests,
            artifact=artifact,
        )
        base_artifact_rows.append(
            {
                "event_key": event_key,
                "artifact_sha256": artifact["artifact_sha256"],
                "candidate_forecasts_sha256": artifact[
                    "candidate_forecasts_sha256"
                ],
            }
        )

    selected_ids = sorted(
        {
            str(selected_models["best_lap_selected_model_id"]),
            str(selected_models["qualifying_selected_model_id"]),
        }
    )
    selected_fit_manifest_sha256: dict[str, str] = {}
    for model_id in selected_ids:
        state_hashes: set[str] = set()
        for event_key, frozen in staged.items():
            manifest = frozen.candidate_manifests.get(model_id)
            if manifest is None:
                raise RuntimeError(
                    f"selected model {model_id} did not forecast audit event {event_key}"
                )
            fitted_state_sha256 = str(manifest.get("fitted_state_sha256") or "")
            if re.fullmatch(r"[0-9a-f]{64}", fitted_state_sha256) is None:
                raise RuntimeError(
                    f"selected model {model_id} lacks a fitted-state digest"
                )
            state_hashes.add(fitted_state_sha256)
        if len(state_hashes) != 1:
            raise RuntimeError(
                f"selected model {model_id} fitted state changed within audit block"
            )
        selected_fit_manifest_sha256[model_id] = next(iter(state_hashes))

    block_hash_payload: dict[str, object] = {
        "schema_version": "f1_same_season_frozen_audit_forecast_block_v1",
        "audit_event_keys": sorted(staged),
        "training_event_keys": training_event_keys,
        "training_data_sha256": training_data_sha256,
        "selection_sha256": selected_models["selection_sha256"],
        "selected_fitted_state_manifest_sha256": selected_fit_manifest_sha256,
        "base_event_forecasts": base_artifact_rows,
        "all_forecasts_materialized_before_first_audit_target": True,
        "formal_promotion_evidence": False,
        "prospective_development_evidence": False,
        "evidence_role": "postdevelopment_replay_diagnostic",
    }
    block_sha256 = _payload_sha256(block_hash_payload)
    frozen_output: dict[int, FrozenAuditEventForecast] = {}
    final_artifact_hashes: dict[str, str] = {}
    for event_key, frozen in staged.items():
        artifact = dict(frozen.artifact)
        artifact.pop("artifact_sha256", None)
        artifact.update(
            {
                "audit_forecast_block_sha256": block_sha256,
                "audit_forecast_block_event_keys": sorted(staged),
                "audit_fitted_training_data_sha256": training_data_sha256,
                "all_audit_forecasts_frozen_before_first_audit_target": True,
            }
        )
        artifact["artifact_sha256"] = _payload_sha256(artifact)
        final_artifact_hashes[str(event_key)] = str(artifact["artifact_sha256"])
        frozen_output[event_key] = FrozenAuditEventForecast(
            event=frozen.event,
            inference=frozen.inference,
            candidate_forecasts=frozen.candidate_forecasts,
            failures=frozen.failures,
            candidate_manifests=frozen.candidate_manifests,
            artifact=artifact,
        )
    metadata = {
        **block_hash_payload,
        "audit_forecast_block_sha256": block_sha256,
        "final_event_forecast_artifact_sha256": final_artifact_hashes,
    }
    return frozen_output, metadata


def _attach_target(
    forecasts: pd.DataFrame,
    target: pd.DataFrame,
    *,
    event_key: int,
) -> pd.DataFrame:
    required = {"driver_id", ACTUAL_LAP_COLUMN, QUALIFYING_TARGET_COLUMN}
    missing = sorted(required.difference(target.columns))
    if missing:
        raise ValueError(f"event {event_key} target is missing columns: {missing}")
    target_rows = target.loc[:, sorted(required)].copy()
    target_rows["driver_id"] = target_rows["driver_id"].astype(str).str.strip()
    if target_rows["driver_id"].duplicated().any():
        raise ValueError(f"event {event_key} target has duplicate drivers")
    scored = forecasts.merge(target_rows, on="driver_id", how="left", validate="many_to_one")
    if scored[QUALIFYING_TARGET_COLUMN].isna().any():
        missing_drivers = scored.loc[
            scored[QUALIFYING_TARGET_COLUMN].isna(), "driver_id"
        ].unique().tolist()
        raise ValueError(f"event {event_key} target omits drivers: {missing_drivers}")
    scored["best_lap_error_seconds"] = (
        pd.to_numeric(scored["predicted_best_lap_seconds"], errors="coerce")
        - pd.to_numeric(scored[ACTUAL_LAP_COLUMN], errors="coerce")
    )
    scored["qualifying_position_error"] = (
        pd.to_numeric(scored["predicted_qualifying_position"], errors="coerce")
        - pd.to_numeric(scored[QUALIFYING_TARGET_COLUMN], errors="coerce")
    )
    return scored


def _event_candidate_metrics(scored: pd.DataFrame) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for (event_key, model_id), group in scored.groupby(
        ["event_key", "model_id"], sort=True
    ):
        actual_position = pd.to_numeric(
            group[QUALIFYING_TARGET_COLUMN], errors="coerce"
        )
        predicted_position = pd.to_numeric(
            group["predicted_qualifying_position"], errors="coerce"
        )
        position_mask = actual_position.notna() & predicted_position.notna()
        actual_lap = pd.to_numeric(group[ACTUAL_LAP_COLUMN], errors="coerce")
        predicted_lap = pd.to_numeric(
            group["predicted_best_lap_seconds"], errors="coerce"
        )
        lap_mask = actual_lap.notna() & predicted_lap.notna()
        field_size = int(position_mask.sum())
        top3_size = min(3, field_size)
        top10_size = min(10, field_size)
        actual_top3 = set(
            group.loc[position_mask]
            .nsmallest(top3_size, QUALIFYING_TARGET_COLUMN)["driver_id"]
            .astype(str)
        )
        predicted_top3 = set(
            group.loc[position_mask]
            .nsmallest(top3_size, "predicted_qualifying_position")["driver_id"]
            .astype(str)
        )
        actual_top10 = set(
            group.loc[position_mask]
            .nsmallest(top10_size, QUALIFYING_TARGET_COLUMN)["driver_id"]
            .astype(str)
        )
        predicted_top10 = set(
            group.loc[position_mask]
            .nsmallest(top10_size, "predicted_qualifying_position")["driver_id"]
            .astype(str)
        )
        output.append(
            {
                "event_key": int(event_key),
                "model_id": str(model_id),
                "rows": int(len(group)),
                "best_lap_rows": int(lap_mask.sum()),
                "best_lap_mae_seconds": float(
                    (predicted_lap.loc[lap_mask] - actual_lap.loc[lap_mask])
                    .abs()
                    .mean()
                )
                if lap_mask.any()
                else float("nan"),
                "best_lap_signed_bias_seconds": float(
                    (predicted_lap.loc[lap_mask] - actual_lap.loc[lap_mask]).mean()
                )
                if lap_mask.any()
                else float("nan"),
                "qualifying_rows": field_size,
                "qualifying_mae_positions": float(
                    (
                        predicted_position.loc[position_mask]
                        - actual_position.loc[position_mask]
                    )
                    .abs()
                    .mean()
                )
                if position_mask.any()
                else float("nan"),
                "qualifying_kendall_tau_b": float(
                    predicted_position.loc[position_mask].corr(
                        actual_position.loc[position_mask], method="kendall"
                    )
                )
                if field_size >= 2
                else float("nan"),
                "pole_hit": bool(
                    field_size
                    and str(
                        group.loc[position_mask]
                        .nsmallest(1, "predicted_qualifying_position")
                        .iloc[0]["driver_id"]
                    )
                    == str(
                        group.loc[position_mask]
                        .nsmallest(1, QUALIFYING_TARGET_COLUMN)
                        .iloc[0]["driver_id"]
                    )
                ),
                "top3_overlap": float(len(actual_top3 & predicted_top3) / top3_size)
                if top3_size
                else float("nan"),
                "top10_overlap": float(len(actual_top10 & predicted_top10) / top10_size)
                if top10_size
                else float("nan"),
            }
        )
    return output


def _selection_event_keys(
    event_keys: Sequence[int], *, audit_event_count: int
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    ordered = tuple(sorted({int(value) for value in event_keys}))
    if audit_event_count < 1:
        raise ValueError("audit_event_count must be positive")
    if len(ordered) < PAIRWISE_MINIMUM_TRAINING_EVENTS + audit_event_count + 2:
        raise ValueError(
            "same-season nested research requires at least "
            f"{PAIRWISE_MINIMUM_TRAINING_EVENTS + audit_event_count + 2} events"
        )
    audit = ordered[-int(audit_event_count) :]
    pre_audit = ordered[: -int(audit_event_count)]
    # Event at zero-based index 4 is the first with four complete training
    # events, so all candidates share the exact same selection blocks.
    selection = pre_audit[PAIRWISE_MINIMUM_TRAINING_EVENTS:]
    development = pre_audit[:PAIRWISE_MINIMUM_TRAINING_EVENTS]
    if len(selection) < 2:
        raise ValueError("nested model selection requires at least two event blocks")
    return development, selection, audit


def _select_models(
    metric_rows: Sequence[Mapping[str, object]],
    *,
    selection_event_keys: Sequence[int],
) -> dict[str, object]:
    metrics = pd.DataFrame(metric_rows)
    selection_keys = tuple(int(value) for value in selection_event_keys)
    selection = metrics.loc[
        pd.to_numeric(metrics["event_key"], errors="coerce").isin(selection_keys)
    ].copy()
    candidate_order = [candidate.model_id for candidate in RESIDUAL_CANDIDATE_GRID]
    qualifying_order = [*candidate_order, PAIRWISE_MODEL_ID]

    def trace_for(
        mode: str,
        candidates: Sequence[str],
        metric_column: str,
        tie_column: str | None,
    ) -> tuple[str, list[dict[str, object]]]:
        trace: list[dict[str, object]] = []
        eligible_rows: list[tuple[float, float, int, str]] = []
        for order, model_id in enumerate(candidates):
            rows = selection.loc[selection["model_id"].eq(model_id)].sort_values(
                "event_key", kind="mergesort"
            )
            observed_keys = tuple(
                pd.to_numeric(rows["event_key"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
            metric = pd.to_numeric(
                rows[metric_column]
                if metric_column in rows.columns
                else pd.Series(np.nan, index=rows.index, dtype=float),
                errors="coerce",
            )
            complete = observed_keys == selection_keys and metric.notna().all()
            mean_metric = float(metric.mean()) if complete else float("nan")
            tie_value = (
                float(
                    pd.to_numeric(
                        rows[tie_column]
                        if tie_column in rows.columns
                        else pd.Series(np.nan, index=rows.index, dtype=float),
                        errors="coerce",
                    ).mean()
                )
                if complete and tie_column is not None
                else 0.0
            )
            trace.append(
                {
                    "mode": mode,
                    "model_id": model_id,
                    "eligible": bool(complete),
                    "selection_event_keys": list(observed_keys),
                    "event_metric_values": metric.tolist(),
                    "mean_event_metric": mean_metric,
                    "mean_tie_break_metric": tie_value,
                    "exclusion_reason": None
                    if complete
                    else "missing_or_nonfinite_common_selection_event",
                }
            )
            if complete:
                # Kendall is a higher-is-better deterministic tie break.  Best
                # Lap has no secondary metric, so predeclared order resolves ties.
                eligible_rows.append(
                    (mean_metric, -tie_value if tie_column else 0.0, order, model_id)
                )
        if not eligible_rows:
            raise RuntimeError(f"no eligible {mode} candidate on common selection events")
        selected = min(eligible_rows)[-1]
        for row in trace:
            row["selected"] = bool(row["model_id"] == selected)
        return selected, trace

    best_model, best_trace = trace_for(
        "best_estimated_lap",
        candidate_order,
        "best_lap_mae_seconds",
        None,
    )
    qualifying_model, qualifying_trace = trace_for(
        "qualifying_prediction",
        qualifying_order,
        "qualifying_mae_positions",
        "qualifying_kendall_tau_b",
    )
    payload: dict[str, object] = {
        "status": "frozen_before_first_audit_target_read",
        "selection_event_keys": list(selection_keys),
        "event_block_is_unit": True,
        "row_level_pooling_for_selection": False,
        "best_lap_selected_model_id": best_model,
        "qualifying_selected_model_id": qualifying_model,
        "best_lap_trace": best_trace,
        "qualifying_trace": qualifying_trace,
        "audit_outcomes_used": False,
    }
    payload["selection_sha256"] = _payload_sha256(payload)
    return payload


def _aggregate_metrics(
    metrics: pd.DataFrame,
    *,
    event_keys: Sequence[int],
    model_id: str,
) -> dict[str, object]:
    rows = metrics.loc[
        metrics["model_id"].eq(model_id)
        & pd.to_numeric(metrics["event_key"], errors="coerce").isin(event_keys)
    ].copy()
    return {
        "model_id": model_id,
        "event_keys": sorted(
            pd.to_numeric(rows.get("event_key"), errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        ),
        "events": int(len(rows)),
        "best_lap_event_mean_mae_seconds": float(
            pd.to_numeric(rows.get("best_lap_mae_seconds"), errors="coerce").mean()
        ),
        "best_lap_event_mean_signed_bias_seconds": float(
            pd.to_numeric(
                rows.get("best_lap_signed_bias_seconds"), errors="coerce"
            ).mean()
        ),
        "qualifying_event_mean_mae_positions": float(
            pd.to_numeric(
                rows.get("qualifying_mae_positions"), errors="coerce"
            ).mean()
        ),
        "qualifying_event_mean_kendall_tau_b": float(
            pd.to_numeric(
                rows.get("qualifying_kendall_tau_b"), errors="coerce"
            ).mean()
        ),
        "pole_hit_rate": float(
            pd.to_numeric(rows.get("pole_hit"), errors="coerce").mean()
        ),
        "top3_overlap": float(
            pd.to_numeric(rows.get("top3_overlap"), errors="coerce").mean()
        ),
        "top10_overlap": float(
            pd.to_numeric(rows.get("top10_overlap"), errors="coerce").mean()
        ),
    }


def _manifest(paths: Sequence[Path], *, root: Path) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for path in sorted({value.expanduser().resolve() for value in paths}):
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            display = str(path.relative_to(root))
        except ValueError:
            display = str(path)
        output.append(
            {
                "path": display,
                "sha256": _file_sha256(path),
                "bytes": int(path.stat().st_size),
            }
        )
    return output


def run_event_stream(
    events: Sequence[ResearchEvent],
    *,
    target_loader: TargetLoader,
    year: int = 2026,
    audit_event_count: int = 1,
    seed: int = DEFAULT_SEED,
    code_paths: Sequence[Path] | None = None,
    prequential_audit_diagnostic: bool = False,
) -> dict[str, object]:
    """Run same-season model selection and a frozen audit block by default."""

    ordered_events = tuple(sorted(events, key=lambda value: int(value.event_key)))
    event_keys = tuple(int(event.event_key) for event in ordered_events)
    if len(set(event_keys)) != len(event_keys):
        raise ValueError("research events must have unique event keys")
    if any(event_key // 100 != int(year) for event_key in event_keys):
        raise ValueError("same-season research received a prior/future-season event")
    development_keys, selection_keys, audit_keys = _selection_event_keys(
        event_keys, audit_event_count=int(audit_event_count)
    )

    history_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    event_metric_rows: list[dict[str, object]] = []
    failed_candidates: list[dict[str, object]] = []
    forecast_artifacts: list[dict[str, object]] = []
    target_infos: dict[int, Mapping[str, object]] = {}
    frozen_selection: dict[str, object] | None = None
    frozen_audit_forecasts: dict[int, FrozenAuditEventForecast] = {}
    audit_forecast_block_metadata: dict[str, object] | None = None
    audit_events = tuple(
        event for event in ordered_events if int(event.event_key) in set(audit_keys)
    )

    for event in ordered_events:
        event_key = int(event.event_key)
        history = (
            pd.concat(history_parts, ignore_index=True, sort=False)
            if history_parts
            else pd.DataFrame()
        )
        if event_key in audit_keys and frozen_selection is None:
            frozen_selection = _select_models(
                event_metric_rows,
                selection_event_keys=selection_keys,
            )
            if not prequential_audit_diagnostic:
                (
                    frozen_audit_forecasts,
                    audit_forecast_block_metadata,
                ) = _freeze_complete_audit_forecast_block(
                    audit_events,
                    history=history,
                    selected_models=frozen_selection,
                    year=int(year),
                    seed=int(seed),
                )

        if event_key in frozen_audit_forecasts:
            frozen_event = frozen_audit_forecasts[event_key]
            inference = frozen_event.inference
            candidate_forecasts = frozen_event.candidate_forecasts
            event_failures = list(frozen_event.failures)
            artifact = dict(frozen_event.artifact)
        else:
            inference = _normalize_inference(event, expected_year=int(year))
            candidate_forecasts, event_failures, candidate_manifests = (
                _forecast_event_candidates(
                    history,
                    inference,
                    event_key=event_key,
                    seed=int(seed),
                )
            )
            artifact = _freeze_forecast_artifact(
                event_key=event_key,
                history=history,
                inference=inference,
                candidate_forecasts=candidate_forecasts,
                candidate_manifests=candidate_manifests,
                selected_models=(
                    frozen_selection if event_key in audit_keys else None
                ),
            )
        failed_candidates.extend(event_failures)
        forecast_artifacts.append(artifact)

        # This is the only target read in the event loop, and it occurs after
        # the complete multi-candidate forecast has an immutable digest.
        target, target_info = target_loader(event, artifact)
        target_infos[event_key] = target_info
        scored = _attach_target(candidate_forecasts, target, event_key=event_key)
        scored["partition"] = (
            "audit"
            if event_key in audit_keys
            else "selection"
            if event_key in selection_keys
            else "development"
        )
        prediction_parts.append(scored)
        event_metrics = _event_candidate_metrics(scored)
        event_metric_rows.extend(event_metrics)

        labelled_history = inference.merge(
            target,
            on="driver_id",
            how="left",
            validate="one_to_one",
        )
        if labelled_history[QUALIFYING_TARGET_COLUMN].isna().any():
            raise ValueError(f"event {event_key} target does not cover inference field")
        history_parts.append(labelled_history)

    if frozen_selection is None:
        raise RuntimeError("audit boundary was never reached")
    if prequential_audit_diagnostic:
        audit_forecast_block_metadata = {
            "schema_version": "f1_same_season_prequential_audit_diagnostic_v1",
            "audit_event_keys": list(audit_keys),
            "mode": "opt_in_prequential_expanding_audit_diagnostic",
            "all_forecasts_materialized_before_first_audit_target": False,
            "audit_outcomes_may_update_later_audit_fits": True,
            "formal_promotion_evidence": False,
        }
    elif audit_forecast_block_metadata is None:
        raise RuntimeError("frozen audit forecast block was not materialized")
    predictions = pd.concat(prediction_parts, ignore_index=True, sort=False)
    metrics = pd.DataFrame(event_metric_rows)
    best_selected = str(frozen_selection["best_lap_selected_model_id"])
    qualifying_selected = str(
        frozen_selection["qualifying_selected_model_id"]
    )
    shadow_ids = [
        *[candidate.model_id for candidate in RESIDUAL_CANDIDATE_GRID],
        PAIRWISE_MODEL_ID,
    ]
    audit_shadow = [
        _aggregate_metrics(metrics, event_keys=audit_keys, model_id=model_id)
        for model_id in shadow_ids
    ]

    root = _root()
    default_code_paths = (
        Path(__file__).resolve(),
        root / "packages/f1/models/pre_quali/pairwise.py",
        root / "packages/f1/features/qualifying_lap.py",
        root
        / "research/projects/F1/rising_qualification_prediction/Python/"
        "run_qualifying_pairwise_challenger_backtest.py",
    )
    input_paths = [
        path for event in ordered_events for path in event.pre_target_input_paths
    ]
    target_paths = [
        event.target_path for event in ordered_events if event.target_path is not None
    ]
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "mode": "same_season_latent_residual_research",
        "model_family": MODEL_FAMILY,
        "protocol": {
            "year": int(year),
            "training": "strictly_earlier_same_season_complete_events",
            "prior_season_training_rows": 0,
            "target_read_boundary": "all_candidate_forecasts_hashed_before_target_loader",
            "event_block_is_training_weight_and_selection_unit": True,
            "workers": 1,
            "seed": int(seed),
            "development_event_keys": list(development_keys),
            "selection_event_keys": list(selection_keys),
            "audit_event_keys": list(audit_keys),
            "final_untouched_audit_event_keys": (
                [] if prequential_audit_diagnostic else list(audit_keys)
            ),
            "prequential_diagnostic_event_keys": (
                list(audit_keys) if prequential_audit_diagnostic else []
            ),
            "audit_training_mode": (
                "opt_in_prequential_expanding_diagnostic"
                if prequential_audit_diagnostic
                else "within_run_frozen_block_postdevelopment_replay"
            ),
            "audit_model_ids_frozen_before_first_audit_target": True,
            "audit_fitted_state_frozen_before_first_audit_target": bool(
                not prequential_audit_diagnostic
            ),
            "all_audit_forecasts_frozen_before_first_audit_target": bool(
                not prequential_audit_diagnostic
            ),
            "audit_outcomes_used_for_hyperparameter_selection": False,
            "audit_outcomes_used_for_later_audit_fits": bool(
                prequential_audit_diagnostic
            ),
            "audit_evidence_eligible_for_formal_promotion": False,
            "prospective_development_evidence": False,
            "evidence_role": "postdevelopment_replay_diagnostic",
            "within_run_target_isolation": bool(not prequential_audit_diagnostic),
            "audit_training_note": (
                "explicit prequential diagnostic; earlier audit outcomes may update "
                "later fits and the resulting metrics are not promotion evidence"
                if prequential_audit_diagnostic
                else "complete audit forecasts and fitted-state manifests were frozen "
                "from pre-audit history before the first audit target read within this "
                "replay; prior development inspection makes the block ineligible for "
                "formal prospective promotion"
            ),
        },
        "feature_contract": {
            "residual_feature_allowlist": list(RESIDUAL_FEATURE_ALLOWLIST),
            "residual_missingness": (
                "training-event-weighted median imputation plus one explicit missing flag "
                "per feature; fit state frozen before inference"
            ),
            "pairwise_feature_allowlist": list(PAIRWISE_FEATURE_ALLOWLIST),
            "pairwise_missingness": "current implementation training-median imputation",
            "identity_or_outcome_features_allowed": False,
        },
        "candidate_grid": [
            candidate.as_dict() for candidate in RESIDUAL_CANDIDATE_GRID
        ]
        + [
            {
                "model_id": PAIRWISE_MODEL_ID,
                "family": "current_pairwise_ranker",
                "regularization_c": 0.5,
                "max_movement": 3,
                "minimum_training_events": PAIRWISE_MINIMUM_TRAINING_EVENTS,
            }
        ],
        "selection": frozen_selection,
        "audit_forecast_block": audit_forecast_block_metadata,
        "audit_metrics": {
            "best_lap_selected": _aggregate_metrics(
                metrics, event_keys=audit_keys, model_id=best_selected
            ),
            "best_lap_raw_rehearsal": _aggregate_metrics(
                metrics, event_keys=audit_keys, model_id="raw_rehearsal_v1"
            ),
            "qualifying_selected": _aggregate_metrics(
                metrics, event_keys=audit_keys, model_id=qualifying_selected
            ),
            "qualifying_raw_rehearsal": _aggregate_metrics(
                metrics, event_keys=audit_keys, model_id="raw_rehearsal_v1"
            ),
        },
        "audit_shadow_metrics_not_used_for_selection": audit_shadow,
        "per_round_metrics": metrics.to_dict(orient="records"),
        "per_round_prediction_vs_reality": predictions.to_dict(orient="records"),
        "failed_candidates": failed_candidates,
        "frozen_event_forecasts": forecast_artifacts,
        "target_metadata_read_after_freeze": target_infos,
        "input_manifest": {
            "pre_target_inputs": _manifest(input_paths, root=root),
            "evaluation_targets_read_after_forecast_freeze": _manifest(
                target_paths, root=root
            ),
        },
        "code_manifest": _manifest(
            default_code_paths if code_paths is None else code_paths,
            root=root,
        ),
    }
    payload["input_manifest_sha256"] = _payload_sha256(payload["input_manifest"])
    payload["code_manifest_sha256"] = _payload_sha256(payload["code_manifest"])
    payload["result_sha256"] = _payload_sha256(
        {
            "schema_version": payload["schema_version"],
            "protocol": payload["protocol"],
            "feature_contract": payload["feature_contract"],
            "candidate_grid": payload["candidate_grid"],
            "selection": payload["selection"],
            "audit_forecast_block": payload["audit_forecast_block"],
            "audit_metrics": payload["audit_metrics"],
            "per_round_metrics": payload["per_round_metrics"],
            "per_round_prediction_vs_reality": payload[
                "per_round_prediction_vs_reality"
            ],
            "failed_candidates": payload["failed_candidates"],
            "frozen_event_forecasts": payload["frozen_event_forecasts"],
            "input_manifest_sha256": payload["input_manifest_sha256"],
            "code_manifest_sha256": payload["code_manifest_sha256"],
        }
    )
    return _json_ready(payload)  # type: ignore[return-value]


def _load_local_events(
    *,
    weekends_dir: Path,
    year: int,
    rounds: Sequence[int] | None,
) -> tuple[ResearchEvent, ...]:
    root = _root()
    requested = None if rounds is None else {int(value) for value in rounds}
    events: list[ResearchEvent] = []
    for event_dir in sorted(
        (weekends_dir / str(year)).glob("round_*"), key=_round_number
    ):
        round_number = _round_number(event_dir)
        if requested is not None and round_number not in requested:
            continue
        inference, info, input_paths, target_path = _event_inference_frame(
            root, event_dir
        )
        pre_target_paths = tuple(
            path.resolve()
            for path in input_paths
            if path.resolve() != target_path.resolve()
        )
        events.append(
            ResearchEvent(
                event_key=int(info["event_key"]),
                inference=inference,
                event_info=info,
                pre_target_input_paths=pre_target_paths,
                target_path=target_path.resolve(),
            )
        )
    if not events:
        raise ValueError("no local same-season Qualifying events found")
    return tuple(events)


def _local_target_loader(
    event: ResearchEvent,
    frozen_forecast_artifact: Mapping[str, object],
) -> tuple[pd.DataFrame, Mapping[str, object]]:
    if event.target_path is None:
        raise ValueError(f"event {event.event_key} has no deferred target path")
    return _load_target_after_frozen_forecast(
        event.target_path,
        expected_event_key=int(event.event_key),
        frozen_forecast_artifact=dict(frozen_forecast_artifact),
    )


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weekends-dir",
        type=Path,
        default=_root() / "data/f1/raw/weekends",
    )
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--rounds", default="auto")
    parser.add_argument("--audit-events", type=int, default=1)
    parser.add_argument(
        "--prequential-audit-diagnostic",
        action="store_true",
        help=(
            "opt in to expanding audit fits; diagnostic only and ineligible for "
            "formal promotion evidence"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            _root()
            / "artifacts/backtests/f1/research/"
            "same_season_latent_residual_v2.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rounds = (
        None
        if str(args.rounds).strip().lower() == "auto"
        else _csv_ints(str(args.rounds))
    )
    events = _load_local_events(
        weekends_dir=args.weekends_dir.expanduser().resolve(),
        year=int(args.year),
        rounds=rounds,
    )
    payload = run_event_stream(
        events,
        target_loader=_local_target_loader,
        year=int(args.year),
        audit_event_count=int(args.audit_events),
        seed=int(args.seed),
        prequential_audit_diagnostic=bool(args.prequential_audit_diagnostic),
    )
    output = args.output.expanduser()
    if not output.is_absolute():
        output = _root() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "selection": payload["selection"],
                "audit_metrics": payload["audit_metrics"],
                "failed_candidates": len(payload["failed_candidates"]),
                "result_sha256": payload["result_sha256"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
