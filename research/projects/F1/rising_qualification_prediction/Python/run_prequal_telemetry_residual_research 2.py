#!/usr/bin/env python3
"""Chronological low-capacity benchmark for pre-Qualifying telemetry bags.

The supervised unit is one driver-event bag, never one telemetry tensor.  For
each target event, every shift estimate, scaler, hyper-parameter decision, and
estimator is fit using complete earlier events only.  A robust prior-event
shift is conditioned on whether the rehearsal was Practice 3 or Sprint
Qualifying.  Telemetry then predicts only the remaining driver-relative
correction from target-free, event-relative features; raw speed/RPM never reach
the model and cannot act as circuit identifiers.

This is deliberately a research benchmark, not a promotion path.  It tests
both compact aggregate telemetry and a regularized distance-ordered temporal
summary.  Sample capacity is reported from event-disjoint folds and effective
degrees of freedom; it is not inferred from an arbitrary event-count cutoff.
"""

from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

from packages.f1.data.providers.telemetry_cache import (
    NORMALIZED_TELEMETRY_CHANNELS,
    sha256_file,
)
from packages.f1.data.providers.telemetry_supervised import (
    LAP_TIME_CENSORED_STATUS,
    LAP_TIME_OBSERVED_STATUS,
    SUPERVISED_TELEMETRY_SCHEMA_VERSION,
    canonical_sha256,
    validate_prequal_telemetry_supervised_manifest,
)
from packages.sports_core.paths import find_repo_root

try:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import HuberRegressor, Ridge
except ImportError as exc:  # pragma: no cover - environment contract
    raise RuntimeError(
        "scikit-learn is required for the telemetry residual research benchmark"
    ) from exc


SCHEMA_VERSION = "f1_prequal_telemetry_residual_research_v4"
TARGET_NAME = "source_shift_adjusted_driver_relative_qualifying_residual_seconds"


def _implementation_manifest(repo_root: Path) -> list[dict[str, Any]]:
    paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("repo_bootstrap.py").resolve(),
        (repo_root / "packages/f1/data/providers/telemetry_cache.py").resolve(),
        (repo_root / "packages/f1/data/providers/telemetry_supervised.py").resolve(),
        (repo_root / "packages/sports_core/paths.py").resolve(),
    )
    return [
        {
            "path": path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for path in paths
    ]

LAP_FEATURE_NAMES: tuple[str, ...] = (
    "speed_q10",
    "speed_q50",
    "speed_q90",
    "speed_iqr",
    "speed_abs_gradient_q90",
    "rpm_q10",
    "rpm_q50",
    "rpm_q90",
    "rpm_iqr",
    "gear_q50",
    "gear_shift_rate",
    "throttle_q10",
    "throttle_q50",
    "throttle_q90",
    "full_throttle_fraction",
    "throttle_abs_gradient_q90",
    "brake_fraction",
    "brake_transition_rate",
    "drs_fraction",
)

DISPERSION_LAP_FEATURES: tuple[str, ...] = (
    "speed_q50",
    "throttle_q50",
    "brake_fraction",
    "drs_fraction",
)

EVENT_RELATIVE_RAW_FEATURE_NAMES: tuple[str, ...] = (
    "lap_median__speed_q10",
    "lap_median__speed_q50",
    "lap_median__speed_q90",
    "lap_median__speed_abs_gradient_q90",
    "lap_median__rpm_q50",
    "lap_median__gear_shift_rate",
    "lap_median__throttle_q10",
    "lap_median__throttle_q50",
    "lap_median__throttle_q90",
    "lap_median__full_throttle_fraction",
    "lap_median__brake_fraction",
    "lap_median__brake_transition_rate",
    "lap_median__drs_fraction",
    "lap_mad__speed_q50",
    "lap_mad__throttle_q50",
    "rehearsal_lap_range_seconds",
    "rehearsal_reference_seconds",
    "tensor_count",
)

EVENT_RANK_RAW_FEATURE_NAMES: tuple[str, ...] = (
    "lap_median__speed_q50",
    "lap_median__throttle_q50",
    "lap_median__brake_fraction",
    "rehearsal_reference_seconds",
)

FEATURE_NAMES: tuple[str, ...] = (
    *(f"event_relative_z__{name}" for name in EVENT_RELATIVE_RAW_FEATURE_NAMES),
    *(f"event_rank__{name}" for name in EVENT_RANK_RAW_FEATURE_NAMES),
)

# Eight fixed distance segments retain coarse corner-to-corner ordering while
# keeping the learned part of the diagnostic linear and strongly regularized.
# These summaries are not a TCN: no convolution kernel is learned.  They are a
# cheap falsification test for whether ordered telemetry contains incremental
# signal before spending the nine independent events on a neural search.
TEMPORAL_SEGMENT_COUNT = 8
TEMPORAL_STATISTICS: tuple[str, ...] = ("mean", "mean_abs_gradient")
TEMPORAL_RAW_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"temporal_{statistic}__{channel}__segment_{segment:02d}"
    for channel in NORMALIZED_TELEMETRY_CHANNELS
    for statistic in TEMPORAL_STATISTICS
    for segment in range(TEMPORAL_SEGMENT_COUNT)
)
TEMPORAL_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"event_relative_z__{name}" for name in TEMPORAL_RAW_FEATURE_NAMES
)

RIDGE_CANDIDATES: tuple[dict[str, object], ...] = tuple(
    {
        "candidate_id": f"ridge_alpha_{alpha:g}",
        "family": "ridge",
        "alpha": float(alpha),
    }
    for alpha in (100.0, 1_000.0, 10_000.0, 100_000.0)
)

HUBER_CANDIDATES: tuple[dict[str, object], ...] = tuple(
    {
        "candidate_id": f"huber_epsilon_{epsilon:g}_alpha_{alpha:g}",
        "family": "huber",
        "epsilon": float(epsilon),
        "alpha": float(alpha),
    }
    for epsilon in (1.2, 1.35)
    for alpha in (1.0, 10.0, 100.0, 1_000.0)
)

ZERO_CORRECTION_CANDIDATE: dict[str, object] = {
    "candidate_id": "zero_telemetry_correction",
    "family": "zero",
}

TEMPORAL_RIDGE_CANDIDATES: tuple[dict[str, object], ...] = tuple(
    {
        "candidate_id": f"temporal_summary_ridge_alpha_{alpha:g}",
        "family": "ridge",
        "alpha": float(alpha),
    }
    for alpha in (1_000.0, 10_000.0, 100_000.0, 1_000_000.0)
)


class TelemetryResidualResearchError(ValueError):
    """Raised when evidence or chronology is insufficient for the benchmark."""


def _without(mapping: Mapping[str, Any], key: str) -> dict[str, Any]:
    payload = dict(mapping)
    payload.pop(key, None)
    return payload


def _assert_hash(mapping: Mapping[str, Any], field: str, *, label: str) -> None:
    declared = str(mapping.get(field) or "").strip().lower()
    actual = canonical_sha256(_without(mapping, field))
    if declared != actual:
        raise TelemetryResidualResearchError(
            f"{label} hash mismatch: declared={declared or '<missing>'}, actual={actual}"
        )


def _resolve(path_value: object, *, root: Path) -> Path:
    text = str(path_value or "").strip()
    if not text:
        raise TelemetryResidualResearchError("telemetry tensor path is missing")
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), float(probability)))


def _transition_rate(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return 0.0
    return float(np.mean(np.diff(array) != 0.0))


def _tensor_features(path: Path, *, expected_sha256: str) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"telemetry tensor does not exist: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != str(expected_sha256).strip().lower():
        raise TelemetryResidualResearchError(
            f"telemetry tensor hash mismatch for {path}: "
            f"declared={expected_sha256}, actual={actual_sha256}"
        )
    try:
        with np.load(path, allow_pickle=False) as payload:
            values = np.asarray(payload["values"], dtype=float)
            channels = tuple(str(value) for value in payload["channel_names"].tolist())
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise TelemetryResidualResearchError(
            f"invalid telemetry tensor {path}: {exc}"
        ) from exc
    if channels != NORMALIZED_TELEMETRY_CHANNELS:
        raise TelemetryResidualResearchError(
            f"unexpected telemetry channels in {path}: {channels}"
        )
    if (
        values.ndim != 2
        or values.shape[0] != len(channels)
        or values.shape[1] < TEMPORAL_SEGMENT_COUNT
    ):
        raise TelemetryResidualResearchError(
            "telemetry tensor must be channels x at least "
            f"{TEMPORAL_SEGMENT_COUNT} distance bins: {path}"
        )
    if not np.isfinite(values).all():
        raise TelemetryResidualResearchError(f"telemetry tensor is non-finite: {path}")

    channel = {name: values[index] for index, name in enumerate(channels)}
    speed = channel["Speed"]
    rpm = channel["RPM"]
    gear = channel["nGear"]
    throttle = channel["Throttle"]
    brake = channel["Brake"] > 0.5
    drs = channel["DRS"] > 0.5
    features = {
        "speed_q10": _quantile(speed, 0.10),
        "speed_q50": _quantile(speed, 0.50),
        "speed_q90": _quantile(speed, 0.90),
        "speed_iqr": _quantile(speed, 0.75) - _quantile(speed, 0.25),
        "speed_abs_gradient_q90": _quantile(np.abs(np.diff(speed)), 0.90),
        "rpm_q10": _quantile(rpm, 0.10),
        "rpm_q50": _quantile(rpm, 0.50),
        "rpm_q90": _quantile(rpm, 0.90),
        "rpm_iqr": _quantile(rpm, 0.75) - _quantile(rpm, 0.25),
        "gear_q50": _quantile(gear, 0.50),
        "gear_shift_rate": _transition_rate(gear),
        "throttle_q10": _quantile(throttle, 0.10),
        "throttle_q50": _quantile(throttle, 0.50),
        "throttle_q90": _quantile(throttle, 0.90),
        "full_throttle_fraction": float(np.mean(throttle >= 95.0)),
        "throttle_abs_gradient_q90": _quantile(np.abs(np.diff(throttle)), 0.90),
        "brake_fraction": float(np.mean(brake)),
        "brake_transition_rate": _transition_rate(brake.astype(float)),
        "drs_fraction": float(np.mean(drs)),
    }
    for channel_name in NORMALIZED_TELEMETRY_CHANNELS:
        channel_values = np.asarray(channel[channel_name], dtype=float)
        for segment, segment_values in enumerate(
            np.array_split(channel_values, TEMPORAL_SEGMENT_COUNT)
        ):
            features[
                f"temporal_mean__{channel_name}__segment_{segment:02d}"
            ] = float(np.mean(segment_values))
            segment_gradient = np.abs(np.diff(segment_values))
            features[
                f"temporal_mean_abs_gradient__{channel_name}__segment_{segment:02d}"
            ] = float(np.mean(segment_gradient)) if len(segment_gradient) else 0.0
    return features


def _median_absolute_deviation(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    center = float(np.median(array))
    return float(np.median(np.abs(array - center)))


def _aggregate_bag(bag: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    _assert_hash(bag, "bag_sha256", label="driver-event bag")
    feature = bag.get("feature")
    target = bag.get("target")
    if not isinstance(feature, Mapping) or not isinstance(target, Mapping):
        raise TelemetryResidualResearchError("bag requires feature and target objects")
    _assert_hash(feature, "feature_bag_sha256", label="feature bag")
    _assert_hash(target, "target_sha256", label="target")
    if target.get("inference_eligible") is not False:
        raise TelemetryResidualResearchError(
            "training targets must be explicitly ineligible for inference"
        )
    if str(bag.get("row_unit") or "") != "driver_event_bag":
        raise TelemetryResidualResearchError("supervised row unit must be driver_event_bag")
    rehearsal_source = str(feature.get("rehearsal_source") or "").strip()
    if not rehearsal_source:
        raise TelemetryResidualResearchError("feature bag rehearsal_source is required")

    tensors = feature.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise TelemetryResidualResearchError("feature bag has no telemetry tensors")
    if int(feature.get("tensor_count", -1)) != len(tensors):
        raise TelemetryResidualResearchError("feature tensor_count does not match tensors")

    lap_features: list[dict[str, float]] = []
    rehearsal_times: list[float] = []
    coverages: list[float] = []
    for tensor in tensors:
        if not isinstance(tensor, Mapping):
            raise TelemetryResidualResearchError("telemetry tensor evidence must be an object")
        path = _resolve(tensor.get("path"), root=root)
        lap_features.append(
            _tensor_features(path, expected_sha256=str(tensor.get("sha256") or ""))
        )
        try:
            rehearsal = float(tensor.get("rehearsal_lap_time_seconds"))
            coverage = float(tensor.get("distance_coverage"))
        except (TypeError, ValueError) as exc:
            raise TelemetryResidualResearchError(
                "telemetry tensors require numeric rehearsal time and coverage"
            ) from exc
        if not math.isfinite(rehearsal) or rehearsal <= 0.0:
            raise TelemetryResidualResearchError("rehearsal lap time must be positive")
        if not math.isfinite(coverage) or coverage <= 0.0:
            raise TelemetryResidualResearchError("distance coverage must be positive")
        rehearsal_times.append(rehearsal)
        coverages.append(coverage)

    try:
        event_key = int(bag["event_key"])
        year = int(bag["year"])
        round_number = int(bag["round"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TelemetryResidualResearchError("bag identity or target is invalid") from exc
    has_legal_lap = target.get("has_legal_qualifying_lap")
    lap_time_observed = target.get("lap_time_observed")
    target_status = str(target.get("lap_time_target_status") or "").strip()
    if not isinstance(has_legal_lap, bool) or not isinstance(lap_time_observed, bool):
        raise TelemetryResidualResearchError(
            "target requires Boolean has_legal_qualifying_lap and lap_time_observed"
        )
    if lap_time_observed != has_legal_lap:
        raise TelemetryResidualResearchError(
            "lap_time_observed must match has_legal_qualifying_lap"
        )
    if lap_time_observed:
        if target_status != LAP_TIME_OBSERVED_STATUS:
            raise TelemetryResidualResearchError(
                "observed lap target has inconsistent lap_time_target_status"
            )
        try:
            actual_lap = float(target["lap_time_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TelemetryResidualResearchError(
                "observed Qualifying target lap must be numeric"
            ) from exc
        if not math.isfinite(actual_lap) or actual_lap <= 0.0:
            raise TelemetryResidualResearchError("Qualifying target lap must be positive")
    else:
        if target_status != LAP_TIME_CENSORED_STATUS:
            raise TelemetryResidualResearchError(
                "censored lap target has inconsistent lap_time_target_status"
            )
        if target.get("lap_time_seconds") is not None:
            raise TelemetryResidualResearchError(
                "censored lap-time target must have lap_time_seconds=null"
            )
        actual_lap = float("nan")
    driver_id = str(bag.get("driver_id") or "").strip().upper()
    event_name = str(bag.get("event_name") or "").strip()
    if not driver_id or not event_name:
        raise TelemetryResidualResearchError("bag driver_id and event_name are required")
    if event_key != year * 100 + round_number:
        raise TelemetryResidualResearchError("event_key must equal year * 100 + round")
    row: dict[str, Any] = {
        "event_key": event_key,
        "year": year,
        "round": round_number,
        "event_name": event_name,
        "driver_id": driver_id,
        "rehearsal_source": rehearsal_source,
        "bag_sha256": str(bag["bag_sha256"]),
        "tensor_count_raw": len(tensors),
        "rehearsal_reference_seconds": float(min(rehearsal_times)),
        "actual_lap_time_seconds": actual_lap,
        "has_legal_qualifying_lap": has_legal_lap,
        "lap_time_observed": lap_time_observed,
        "lap_time_target_status": target_status,
    }
    row["target_residual_seconds"] = float(
        actual_lap - row["rehearsal_reference_seconds"]
        if lap_time_observed
        else float("nan")
    )
    for name in LAP_FEATURE_NAMES:
        row[f"lap_median__{name}"] = float(
            np.median([lap[name] for lap in lap_features])
        )
    for name in DISPERSION_LAP_FEATURES:
        row[f"lap_mad__{name}"] = _median_absolute_deviation(
            lap[name] for lap in lap_features
        )
    for name in TEMPORAL_RAW_FEATURE_NAMES:
        row[name] = float(np.median([lap[name] for lap in lap_features]))
    row["tensor_count"] = float(len(tensors))
    row["distance_coverage_median"] = float(np.median(coverages))
    row["rehearsal_lap_range_seconds"] = float(max(rehearsal_times) - min(rehearsal_times))
    return row


def aggregate_supervised_manifest(
    manifest: Mapping[str, Any],
    *,
    root: Path,
) -> pd.DataFrame:
    """Validate and aggregate immutable bags into one row per driver-event."""

    if str(manifest.get("schema_version") or "") != SUPERVISED_TELEMETRY_SCHEMA_VERSION:
        raise TelemetryResidualResearchError("unsupported supervised telemetry schema")
    if manifest.get("targets_inference_eligible") is not False:
        raise TelemetryResidualResearchError("manifest targets must be inference-ineligible")
    if manifest.get("tensor_rows_are_independent_supervised_rows") is not False:
        raise TelemetryResidualResearchError("tensor rows cannot be supervised rows")
    if str(manifest.get("independent_evaluation_unit") or "") != "event":
        raise TelemetryResidualResearchError("independent evaluation unit must be event")
    bags = manifest.get("bags")
    if not isinstance(bags, list) or not bags:
        raise TelemetryResidualResearchError("supervised manifest has no bags")
    declared_bag_set = str(manifest.get("bag_set_sha256") or "").strip().lower()
    actual_bag_set = canonical_sha256([bag.get("bag_sha256") for bag in bags])
    if declared_bag_set != actual_bag_set:
        raise TelemetryResidualResearchError("supervised bag-set hash mismatch")

    frame = pd.DataFrame([_aggregate_bag(bag, root=root) for bag in bags])
    if frame.duplicated(["event_key", "driver_id"]).any():
        raise TelemetryResidualResearchError("duplicate driver-event supervised row")
    event_identity = frame.groupby("event_key")[
        ["year", "round", "event_name", "rehearsal_source"]
    ].nunique()
    if (event_identity > 1).any().any():
        raise TelemetryResidualResearchError("event identity is inconsistent across bags")

    def event_percentile(series: pd.Series) -> pd.Series:
        if len(series) == 1:
            return pd.Series(0.5, index=series.index, dtype=float)
        return (series.rank(method="average") - 1.0) / float(len(series) - 1)

    for raw_name in EVENT_RELATIVE_RAW_FEATURE_NAMES:
        event_median = frame.groupby("event_key")[raw_name].transform("median")
        absolute_deviation = (frame[raw_name] - event_median).abs()
        event_mad = absolute_deviation.groupby(frame["event_key"]).transform("median")
        q75 = frame.groupby("event_key")[raw_name].transform(
            lambda series: float(series.quantile(0.75))
        )
        q25 = frame.groupby("event_key")[raw_name].transform(
            lambda series: float(series.quantile(0.25))
        )
        robust_scale = 1.4826 * event_mad
        iqr_scale = (q75 - q25) / 1.349
        # Float32 tensors can create micro-differences when a common speed/RPM
        # offset is subtracted.  Do not amplify those numerical crumbs into a
        # unit-scale pseudo-signal merely because their MAD is nonzero.
        scale_floor = (event_median.abs() * 1e-5).clip(lower=1e-6)
        robust_scale = robust_scale.where(robust_scale > scale_floor, iqr_scale)
        robust_scale = robust_scale.where(robust_scale > scale_floor, 1.0)
        frame[f"event_relative_z__{raw_name}"] = (
            frame[raw_name] - event_median
        ) / robust_scale
    for raw_name in EVENT_RANK_RAW_FEATURE_NAMES:
        frame[f"event_rank__{raw_name}"] = frame.groupby(
            "event_key", group_keys=False
        )[raw_name].apply(event_percentile)
    temporal_columns: dict[str, pd.Series] = {}
    for raw_name in TEMPORAL_RAW_FEATURE_NAMES:
        event_median = frame.groupby("event_key")[raw_name].transform("median")
        absolute_deviation = (frame[raw_name] - event_median).abs()
        event_mad = absolute_deviation.groupby(frame["event_key"]).transform("median")
        q75 = frame.groupby("event_key")[raw_name].transform(
            lambda series: float(series.quantile(0.75))
        )
        q25 = frame.groupby("event_key")[raw_name].transform(
            lambda series: float(series.quantile(0.25))
        )
        robust_scale = 1.4826 * event_mad
        iqr_scale = (q75 - q25) / 1.349
        scale_floor = (event_median.abs() * 1e-5).clip(lower=1e-6)
        robust_scale = robust_scale.where(robust_scale > scale_floor, iqr_scale)
        robust_scale = robust_scale.where(robust_scale > scale_floor, 1.0)
        temporal_columns[f"event_relative_z__{raw_name}"] = (
            frame[raw_name] - event_median
        ) / robust_scale
    frame = pd.concat([frame, pd.DataFrame(temporal_columns, index=frame.index)], axis=1)
    frame = frame.sort_values(["event_key", "driver_id"], kind="mergesort").reset_index(
        drop=True
    )
    numeric = frame.loc[:, FEATURE_NAMES].to_numpy(dtype=float)
    if numeric.shape[1] != len(FEATURE_NAMES) or not np.isfinite(numeric).all():
        raise TelemetryResidualResearchError("aggregated telemetry features are non-finite")
    temporal = frame.loc[:, TEMPORAL_FEATURE_NAMES].to_numpy(dtype=float)
    if temporal.shape[1] != len(TEMPORAL_FEATURE_NAMES) or not np.isfinite(temporal).all():
        raise TelemetryResidualResearchError(
            "event-relative temporal telemetry features are non-finite"
        )
    return frame


def _event_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("event_key")["driver_id"].transform("count").to_numpy(dtype=float)
    return 1.0 / counts


def _source_shift_predictions(
    train: pd.DataFrame,
    score: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate one robust rehearsal-to-Q shift per source from prior events only."""

    train_events = sorted(int(value) for value in train["event_key"].unique())
    score_events = sorted(int(value) for value in score["event_key"].unique())
    if not train_events or not score_events or max(train_events) >= min(score_events):
        raise TelemetryResidualResearchError(
            "source shift requires complete training events strictly before scored events"
        )
    source_per_event = train.groupby("event_key")["rehearsal_source"].nunique()
    if (source_per_event != 1).any():
        raise TelemetryResidualResearchError("each training event requires one rehearsal source")
    event_shifts = (
        train.groupby(["event_key", "rehearsal_source"], as_index=False)[
            "target_residual_seconds"
        ]
        .median()
        .rename(columns={"target_residual_seconds": "event_median_shift_seconds"})
    )
    global_shift = float(np.median(event_shifts["event_median_shift_seconds"]))
    source_rows: dict[str, dict[str, Any]] = {}
    source_shifts: dict[str, float] = {}
    for source, rows in event_shifts.groupby("rehearsal_source", sort=True):
        source_text = str(source)
        source_shift = float(np.median(rows["event_median_shift_seconds"]))
        source_shifts[source_text] = source_shift
        source_rows[source_text] = {
            "prior_event_keys": sorted(int(value) for value in rows["event_key"]),
            "prior_event_count": int(len(rows)),
            "median_shift_seconds": source_shift,
        }
    score_sources = score["rehearsal_source"].astype(str)
    predictions = score_sources.map(source_shifts).fillna(global_shift).to_numpy(dtype=float)
    fallback_sources = sorted(
        source for source in score_sources.unique().tolist() if source not in source_shifts
    )
    audit = {
        "fit_event_keys": train_events,
        "scored_event_keys": score_events,
        "global_event_balanced_median_shift_seconds": global_shift,
        "sources": source_rows,
        "scored_sources": sorted(score_sources.unique().tolist()),
        "fallback_to_global_sources": fallback_sources,
        "target_event_used": False,
    }
    return predictions, audit


def _raw_shift_prediction(train: pd.DataFrame) -> float:
    event_medians = train.groupby("event_key")["target_residual_seconds"].median()
    return float(np.median(event_medians.to_numpy(dtype=float)))


def _driver_relative_training_target(
    train: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Remove event shifts and robustly cap unmodelable timing anomalies."""

    raw = train["target_residual_seconds"].astype(float)
    event_center = raw.groupby(train["event_key"]).transform("median")
    centered = raw - event_center
    event_mads = centered.abs().groupby(train["event_key"]).median().to_numpy(dtype=float)
    event_balanced_mad = float(np.median(event_mads))
    clip_abs_seconds = float(max(0.25, 4.0 * 1.4826 * event_balanced_mad))
    unclipped = centered.to_numpy(dtype=float)
    clipped = np.clip(unclipped, -clip_abs_seconds, clip_abs_seconds)
    return clipped, {
        "event_balanced_mad_seconds": event_balanced_mad,
        "clip_abs_seconds": clip_abs_seconds,
        "clipped_training_row_count": int(np.sum(clipped != unclipped)),
    }


def _weighted_scaler(
    x: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = weights / float(np.sum(weights))
    mean = np.sum(x * normalized[:, None], axis=0)
    variance = np.sum(((x - mean) ** 2) * normalized[:, None], axis=0)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    return mean, scale


def _fit_predict(
    train: pd.DataFrame,
    score: pd.DataFrame,
    candidate: Mapping[str, object],
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_events = sorted(int(value) for value in train["event_key"].unique())
    score_events = sorted(int(value) for value in score["event_key"].unique())
    if not train_events or not score_events or max(train_events) >= min(score_events):
        raise TelemetryResidualResearchError(
            "model fit requires non-empty complete events strictly before scored events"
        )
    selected_features = tuple(str(value) for value in feature_names)
    if not selected_features:
        raise TelemetryResidualResearchError("model feature set cannot be empty")
    x_train = train.loc[:, selected_features].to_numpy(dtype=float)
    x_score = score.loc[:, selected_features].to_numpy(dtype=float)
    y_train, target_clip_audit = _driver_relative_training_target(train)
    weights = _event_equal_weights(train)

    family = str(candidate["family"])
    if family == "zero":
        prediction = np.zeros(len(score), dtype=float)
        mean = np.zeros(len(selected_features), dtype=float)
        scale = np.ones(len(selected_features), dtype=float)
        scaler_used = False
        effective_degrees_of_freedom = 0.0
    else:
        mean, scale = _weighted_scaler(x_train, weights)
        x_train_scaled = (x_train - mean) / scale
        x_score_scaled = (x_score - mean) / scale
        scaler_used = True
    if family == "ridge":
        estimator = Ridge(
            alpha=float(candidate["alpha"]),
            fit_intercept=True,
            solver="svd",
        )
    elif family == "huber":
        estimator = HuberRegressor(
            alpha=float(candidate["alpha"]),
            epsilon=float(candidate["epsilon"]),
            fit_intercept=True,
            max_iter=1000,
            tol=1e-6,
        )
    elif family != "zero":
        raise TelemetryResidualResearchError(f"unsupported model family: {family}")
    if family != "zero":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            estimator.fit(x_train_scaled, y_train, sample_weight=weights)
        raw_prediction = np.asarray(estimator.predict(x_score_scaled), dtype=float)
        prediction = np.clip(
            raw_prediction,
            -float(target_clip_audit["clip_abs_seconds"]),
            float(target_clip_audit["clip_abs_seconds"]),
        )
        if family == "ridge":
            weighted_design = x_train_scaled * np.sqrt(weights)[:, None]
            singular_values = np.linalg.svd(weighted_design, compute_uv=False)
            squared = singular_values**2
            effective_degrees_of_freedom = float(
                np.sum(squared / (squared + float(candidate["alpha"])))
            )
        else:
            effective_degrees_of_freedom = None
    predictions_clipped_count = (
        0
        if family == "zero"
        else int(np.sum(prediction != raw_prediction))
    )
    if prediction.shape != (len(score),) or not np.isfinite(prediction).all():
        raise TelemetryResidualResearchError(f"{family} produced invalid predictions")
    event_weight_sums = {
        str(event_key): float(weights[train["event_key"].to_numpy() == event_key].sum())
        for event_key in train_events
    }
    audit = {
        "candidate": dict(candidate),
        "training_event_keys": train_events,
        "scored_event_keys": score_events,
        "training_event_count": len(train_events),
        "training_driver_event_count": int(len(train)),
        "scored_driver_event_count": int(len(score)),
        "event_weight_sums": event_weight_sums,
        "scaler_fit_event_keys": train_events,
        "feature_count": len(selected_features),
        "feature_schema_sha256": canonical_sha256(list(selected_features)),
        "learned_coefficient_count_including_intercept": (
            0 if family == "zero" else len(selected_features) + 1
        ),
        "regularized_effective_degrees_of_freedom_excluding_intercept": (
            effective_degrees_of_freedom
        ),
        "training_target": (
            "source_adjusted_residual_then_within_training_event_median_centered"
        ),
        "training_target_sha256": canonical_sha256(y_train.tolist()),
        "training_target_clip_audit": target_clip_audit,
        "scored_prediction_clipped_count": predictions_clipped_count,
        "scaler_used": scaler_used,
        "scaler_sha256": canonical_sha256(
            {"mean": mean.tolist(), "scale": scale.tolist()}
        ),
        "pca_used": False,
    }
    return prediction, audit


def _candidate_selection(
    train: pd.DataFrame,
    *,
    candidates: Sequence[Mapping[str, object]],
    minimum_inner_train_events: int,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> tuple[dict[str, object], dict[str, Any]]:
    event_keys = sorted(int(value) for value in train["event_key"].unique())
    if len(event_keys) <= int(minimum_inner_train_events):
        raise TelemetryResidualResearchError(
            "not enough prior events for chronological hyper-parameter selection"
        )
    inner_targets = event_keys[int(minimum_inner_train_events) :]
    candidate_rows: list[dict[str, Any]] = []
    for raw_candidate in candidates:
        candidate = dict(raw_candidate)
        fold_maes: list[dict[str, object]] = []
        failed: str | None = None
        for target_event in inner_targets:
            inner_train = train.loc[train["event_key"] < target_event].copy()
            inner_score = train.loc[train["event_key"] == target_event].copy()
            try:
                correction_prediction, _ = _fit_predict(
                    inner_train,
                    inner_score,
                    candidate,
                    feature_names=feature_names,
                )
                source_shift, _ = _source_shift_predictions(inner_train, inner_score)
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                failed = f"{type(exc).__name__}: {exc}"
                break
            actual = inner_score["actual_lap_time_seconds"].to_numpy(dtype=float)
            reference = inner_score["rehearsal_reference_seconds"].to_numpy(dtype=float)
            lap_prediction = reference + source_shift + correction_prediction
            fold_maes.append(
                {
                    "target_event_key": int(target_event),
                    "prior_event_keys": sorted(
                        int(value) for value in inner_train["event_key"].unique()
                    ),
                    "mae_seconds": float(np.mean(np.abs(lap_prediction - actual))),
                }
            )
        candidate_rows.append(
            {
                "candidate": candidate,
                "valid": failed is None and bool(fold_maes),
                "failure": failed,
                "inner_event_mae_seconds": (
                    float(np.mean([float(row["mae_seconds"]) for row in fold_maes]))
                    if failed is None and fold_maes
                    else None
                ),
                "folds": fold_maes,
            }
        )
    valid = [row for row in candidate_rows if row["valid"]]
    if not valid:
        raise TelemetryResidualResearchError("every model candidate failed inner selection")
    zero_rows = [
        row
        for row in valid
        if row["candidate"]["candidate_id"] == "zero_telemetry_correction"
    ]
    if len(zero_rows) != 1:
        raise TelemetryResidualResearchError(
            "selection grid requires exactly one valid zero-telemetry candidate"
        )
    zero_row = zero_rows[0]
    zero_score = float(zero_row["inner_event_mae_seconds"])
    zero_fold_maes = {
        int(fold["target_event_key"]): float(fold["mae_seconds"])
        for fold in zero_row["folds"]
    }
    required_wins = int(math.ceil(0.60 * len(inner_targets)))
    eligible: list[dict[str, Any]] = []
    for row in valid:
        candidate_id = str(row["candidate"]["candidate_id"])
        reasons: list[str] = []
        if candidate_id != "zero_telemetry_correction":
            if len(inner_targets) < 3:
                reasons.append("fewer_than_three_independent_inner_test_events")
            if float(row["inner_event_mae_seconds"]) > zero_score * 0.98:
                reasons.append("less_than_two_percent_mean_mae_gain_vs_zero")
            fold_maes = {
                int(fold["target_event_key"]): float(fold["mae_seconds"])
                for fold in row["folds"]
            }
            wins = sum(
                fold_maes[event_key] < zero_fold_maes[event_key]
                for event_key in inner_targets
            )
            if wins < required_wins:
                reasons.append("insufficient_inner_event_wins_vs_zero")
            if any(
                fold_maes[event_key] > zero_fold_maes[event_key] * 1.25 + 0.01
                for event_key in inner_targets
            ):
                reasons.append("inner_event_degradation_guard_failed")
        row["eligible_under_zero_shrinkage_guard"] = not reasons
        row["zero_shrinkage_guard_reasons"] = reasons
        if not reasons:
            eligible.append(row)
    if not eligible:
        raise TelemetryResidualResearchError("zero-shrinkage selection guard failed closed")
    selected_row = min(
        eligible,
        key=lambda row: (
            float(row["inner_event_mae_seconds"]),
            str(row["candidate"]["candidate_id"]),
        ),
    )
    selected = dict(selected_row["candidate"])
    audit = {
        "selection_unit": "complete_chronological_event",
        "selection_metric": "mean_inner_event_mae_seconds",
        "minimum_inner_train_events": int(minimum_inner_train_events),
        "selection_source_event_keys": event_keys,
        "feature_count": len(tuple(feature_names)),
        "feature_schema_sha256": canonical_sha256(list(feature_names)),
        "inner_target_event_keys": inner_targets,
        "selected_candidate_id": selected["candidate_id"],
        "zero_shrinkage_guard": {
            "zero_candidate_id": "zero_telemetry_correction",
            "zero_inner_event_mae_seconds": zero_score,
            "minimum_independent_inner_test_events_for_nonzero": 3,
            "required_relative_mean_mae_gain_vs_zero": 0.02,
            "required_event_win_fraction_vs_zero": 0.60,
            "maximum_single_event_relative_degradation_vs_zero": 0.25,
        },
        "candidates": candidate_rows,
    }
    return selected, audit


def _prediction_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {
        "mae_seconds": float(np.mean(np.abs(error))),
        "median_absolute_error_seconds": float(np.median(np.abs(error))),
        "bias_seconds": float(np.mean(error)),
    }


def _summary(
    folds: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    actual = np.asarray([row["actual_lap_time_seconds"] for row in predictions], dtype=float)
    summary: dict[str, Any] = {
        "scored_event_count": int(len(folds)),
        "scored_driver_event_count": int(len(predictions)),
    }
    model_names = (
        "raw_baseline",
        "source_shift_baseline",
        "ridge_or_zero",
        "huber_or_zero",
        "temporal_ridge_or_zero",
    )
    for name in model_names:
        prediction = np.asarray(
            [row[f"{name}_predicted_lap_time_seconds"] for row in predictions],
            dtype=float,
        )
        row_metrics = _prediction_metrics(actual, prediction)
        event_maes = [float(fold["metrics"][name]["mae_seconds"]) for fold in folds]
        summary[name] = {
            **row_metrics,
            "event_balanced_mae_seconds": float(np.mean(event_maes)),
            "event_mae_standard_deviation_seconds": float(np.std(event_maes)),
            "events_beating_source_shift_baseline": (
                None
                if name in ("raw_baseline", "source_shift_baseline")
                else int(
                    sum(
                        float(fold["metrics"][name]["mae_seconds"])
                        < float(
                            fold["metrics"]["source_shift_baseline"]["mae_seconds"]
                        )
                        for fold in folds
                    )
                )
            ),
        }
    raw_baseline_mae = float(summary["raw_baseline"]["event_balanced_mae_seconds"])
    source_baseline_mae = float(
        summary["source_shift_baseline"]["event_balanced_mae_seconds"]
    )
    summary["source_shift_baseline"]["delta_vs_raw_baseline_seconds"] = (
        source_baseline_mae - raw_baseline_mae
    )
    summary["source_shift_baseline"]["relative_improvement_vs_raw_baseline"] = (
        (raw_baseline_mae - source_baseline_mae) / raw_baseline_mae
        if raw_baseline_mae > 0.0
        else None
    )
    for name in ("ridge_or_zero", "huber_or_zero", "temporal_ridge_or_zero"):
        model_mae = float(summary[name]["event_balanced_mae_seconds"])
        summary[name]["delta_vs_source_shift_baseline_seconds"] = (
            model_mae - source_baseline_mae
        )
        summary[name]["relative_improvement_vs_source_shift_baseline"] = (
            (source_baseline_mae - model_mae) / source_baseline_mae
            if source_baseline_mae > 0.0
            else None
        )
    return summary


def _temporal_capacity_diagnostic(
    folds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe observed capacity without pretending event count is a law.

    The chronological trace changes both training size and target circuit, so
    its slope is reported as a diagnostic, not interpreted as a controlled
    learning curve or a promotion test.
    """

    trace: list[dict[str, Any]] = []
    for fold in folds:
        fit = fold["temporal_ridge_fit_audit"]
        selection = fold["temporal_ridge_selection"]
        temporal_mae = float(fold["metrics"]["temporal_ridge_or_zero"]["mae_seconds"])
        baseline_mae = float(fold["metrics"]["source_shift_baseline"]["mae_seconds"])
        trace.append(
            {
                "target_event_key": int(fold["target_event_key"]),
                "training_event_count": int(len(fold["prior_event_keys"])),
                "training_driver_event_count": int(fold["training_driver_event_count"]),
                "selected_candidate_id": str(selection["selected_candidate_id"]),
                "regularized_effective_degrees_of_freedom_excluding_intercept": fit[
                    "regularized_effective_degrees_of_freedom_excluding_intercept"
                ],
                "source_shift_baseline_mae_seconds": baseline_mae,
                "temporal_ridge_or_zero_mae_seconds": temporal_mae,
                "delta_vs_source_shift_baseline_seconds": temporal_mae - baseline_mae,
            }
        )
    event_counts = np.asarray(
        [row["training_event_count"] for row in trace], dtype=float
    )
    deltas = np.asarray(
        [row["delta_vs_source_shift_baseline_seconds"] for row in trace], dtype=float
    )
    slope = (
        float(np.polyfit(event_counts, deltas, deg=1)[0])
        if len(trace) >= 2 and float(np.ptp(event_counts)) > 0.0
        else None
    )
    selected_nonzero = sum(
        row["selected_candidate_id"] != "zero_telemetry_correction" for row in trace
    )
    return {
        "diagnostic_model": "event_relative_fixed_distance_summary_ridge_or_zero",
        "sequence_order_preserved": True,
        "learned_convolution_used": False,
        "true_tcn_evaluated": False,
        "true_tcn_runtime": {
            "minimum_supported_python": "3.9",
            "python_version_supported": sys.version_info >= (3, 9),
            "torch_dependency_available": importlib.util.find_spec("torch") is not None,
            "research_evaluable_in_current_runtime": bool(
                sys.version_info >= (3, 9)
                and importlib.util.find_spec("torch") is not None
            ),
        },
        "representation_feature_count": len(TEMPORAL_FEATURE_NAMES),
        "independent_sample_unit": "complete_event",
        "driver_event_rows_are_not_counted_as_independent_events": True,
        "fixed_minimum_event_count_claim_used": False,
        "chronological_capacity_trace": trace,
        "trace_caveat": (
            "training size and target event change together; the slope is descriptive "
            "and cannot identify a causal learning-curve plateau"
        ),
        "delta_slope_seconds_per_added_training_event": slope,
        "nonzero_temporal_candidate_selected_fold_count": int(selected_nonzero),
        "scored_fold_count": int(len(trace)),
        "events_beating_source_shift_baseline": int(np.sum(deltas < 0.0)),
        "promotion_eligible": False,
        "promotion_blockers": [
            "sequence_diagnostic_is_model_development_evidence",
            "no_future_locked_event_after_sequence_model_development",
            "true_tcn_not_evaluated_by_this_sklearn_diagnostic",
        ],
    }


def run_expanding_window_benchmark(
    frame: pd.DataFrame,
    *,
    minimum_train_events: int = 3,
    minimum_inner_train_events: int = 2,
) -> dict[str, Any]:
    """Run nested, event-blocked expanding-window Ridge and Huber forecasts.

    Event-relative telemetry features are computed over every telemetry bag.
    Continuous lap-time models then explicitly condition on observed legal-lap
    targets.  Censored bags remain part of the manifest and are counted in the
    audit instead of disappearing upstream.
    """

    if int(minimum_train_events) <= int(minimum_inner_train_events):
        raise TelemetryResidualResearchError(
            "minimum_train_events must exceed minimum_inner_train_events"
        )
    required_target_columns = {
        "has_legal_qualifying_lap",
        "lap_time_observed",
        "lap_time_target_status",
    }
    missing_target_columns = sorted(required_target_columns.difference(frame.columns))
    if missing_target_columns:
        raise TelemetryResidualResearchError(
            f"aggregated manifest is missing target-state columns: {missing_target_columns}"
        )
    full_frame = frame.copy()
    observed_mask = full_frame["lap_time_observed"].eq(True)  # noqa: E712
    if not (
        full_frame.loc[observed_mask, "actual_lap_time_seconds"].map(np.isfinite).all()
        and full_frame.loc[~observed_mask, "actual_lap_time_seconds"].isna().all()
    ):
        raise TelemetryResidualResearchError(
            "observed/censored lap-time states do not match numeric target availability"
        )
    model_frame = full_frame.loc[observed_mask].copy()
    event_keys = sorted(int(value) for value in full_frame["event_key"].unique())
    observed_event_keys = sorted(int(value) for value in model_frame["event_key"].unique())
    if len(observed_event_keys) <= int(minimum_train_events):
        raise TelemetryResidualResearchError(
            f"need more than {minimum_train_events} independent events with observed legal "
            f"laps; found {len(observed_event_keys)}"
        )
    target_counts_by_event = {
        int(event_key): {
            "driver_event_bag_count": int(len(rows)),
            "observed_lap_time_target_count": int(rows["lap_time_observed"].sum()),
            "censored_lap_time_target_count": int((~rows["lap_time_observed"]).sum()),
        }
        for event_key, rows in full_frame.groupby("event_key", sort=True)
    }
    folds: list[dict[str, Any]] = []
    flat_predictions: list[dict[str, Any]] = []
    for target_event in observed_event_keys[int(minimum_train_events) :]:
        train = model_frame.loc[model_frame["event_key"] < target_event].copy()
        score = model_frame.loc[model_frame["event_key"] == target_event].copy()
        train_events = sorted(int(value) for value in train["event_key"].unique())
        expected_train_events = [value for value in observed_event_keys if value < target_event]
        if train_events != expected_train_events or len(train_events) < minimum_train_events:
            raise TelemetryResidualResearchError(
                "outer fold does not contain every and only complete prior events"
            )
        ridge_candidate, ridge_selection = _candidate_selection(
            train,
            candidates=(*RIDGE_CANDIDATES, ZERO_CORRECTION_CANDIDATE),
            minimum_inner_train_events=minimum_inner_train_events,
        )
        huber_candidate, huber_selection = _candidate_selection(
            train,
            candidates=(*HUBER_CANDIDATES, ZERO_CORRECTION_CANDIDATE),
            minimum_inner_train_events=minimum_inner_train_events,
        )
        temporal_candidate, temporal_selection = _candidate_selection(
            train,
            candidates=(*TEMPORAL_RIDGE_CANDIDATES, ZERO_CORRECTION_CANDIDATE),
            minimum_inner_train_events=minimum_inner_train_events,
            feature_names=TEMPORAL_FEATURE_NAMES,
        )
        ridge_correction, ridge_fit = _fit_predict(train, score, ridge_candidate)
        huber_correction, huber_fit = _fit_predict(train, score, huber_candidate)
        temporal_correction, temporal_fit = _fit_predict(
            train,
            score,
            temporal_candidate,
            feature_names=TEMPORAL_FEATURE_NAMES,
        )
        raw_shift = _raw_shift_prediction(train)
        source_shift, source_shift_audit = _source_shift_predictions(train, score)
        reference = score["rehearsal_reference_seconds"].to_numpy(dtype=float)
        actual = score["actual_lap_time_seconds"].to_numpy(dtype=float)
        model_laps = {
            "raw_baseline": reference + raw_shift,
            "source_shift_baseline": reference + source_shift,
            "ridge_or_zero": reference + source_shift + ridge_correction,
            "huber_or_zero": reference + source_shift + huber_correction,
            "temporal_ridge_or_zero": reference + source_shift + temporal_correction,
        }
        prediction_rows: list[dict[str, Any]] = []
        for index, (_, source) in enumerate(score.iterrows()):
            row: dict[str, Any] = {
                "event_key": int(source["event_key"]),
                "round": int(source["round"]),
                "event_name": str(source["event_name"]),
                "driver_id": str(source["driver_id"]),
                "rehearsal_source": str(source["rehearsal_source"]),
                "bag_sha256": str(source["bag_sha256"]),
                "rehearsal_reference_seconds": float(reference[index]),
                "actual_lap_time_seconds": float(actual[index]),
                "actual_residual_seconds": float(source["target_residual_seconds"]),
                "raw_baseline_predicted_residual_seconds": float(raw_shift),
                "source_shift_predicted_residual_seconds": float(source_shift[index]),
                "ridge_or_zero_predicted_correction_seconds": float(
                    ridge_correction[index]
                ),
                "huber_or_zero_predicted_correction_seconds": float(
                    huber_correction[index]
                ),
                "temporal_ridge_or_zero_predicted_correction_seconds": float(
                    temporal_correction[index]
                ),
                "raw_baseline_predicted_lap_time_seconds": float(
                    model_laps["raw_baseline"][index]
                ),
                "source_shift_baseline_predicted_lap_time_seconds": float(
                    model_laps["source_shift_baseline"][index]
                ),
                "ridge_or_zero_predicted_lap_time_seconds": float(
                    model_laps["ridge_or_zero"][index]
                ),
                "huber_or_zero_predicted_lap_time_seconds": float(
                    model_laps["huber_or_zero"][index]
                ),
                "temporal_ridge_or_zero_predicted_lap_time_seconds": float(
                    model_laps["temporal_ridge_or_zero"][index]
                ),
            }
            row["prediction_sha256"] = canonical_sha256(row)
            prediction_rows.append(row)
        metrics = {
            name: _prediction_metrics(actual, prediction)
            for name, prediction in model_laps.items()
        }
        fold = {
            "target_event_key": int(target_event),
            "round": int(score["round"].iloc[0]),
            "event_name": str(score["event_name"].iloc[0]),
            "prior_event_keys": train_events,
            "training_driver_event_count": int(len(train)),
            "training_censored_driver_event_count": int(
                len(
                    full_frame.loc[
                        (full_frame["event_key"] < target_event)
                        & (~full_frame["lap_time_observed"])
                    ]
                )
            ),
            "training_tensor_count": int(train["tensor_count_raw"].sum()),
            "scored_driver_event_count": int(len(score)),
            "target_event_driver_event_bag_count": int(
                target_counts_by_event[target_event]["driver_event_bag_count"]
            ),
            "target_event_observed_lap_time_target_count": int(
                target_counts_by_event[target_event]["observed_lap_time_target_count"]
            ),
            "target_event_censored_lap_time_target_count": int(
                target_counts_by_event[target_event]["censored_lap_time_target_count"]
            ),
            "scored_tensor_count": int(score["tensor_count_raw"].sum()),
            "rehearsal_source": str(score["rehearsal_source"].iloc[0]),
            "raw_training_event_median_residual_seconds": float(raw_shift),
            "source_shift_audit": source_shift_audit,
            "ridge_selection": ridge_selection,
            "huber_selection": huber_selection,
            "temporal_ridge_selection": temporal_selection,
            "ridge_fit_audit": ridge_fit,
            "huber_fit_audit": huber_fit,
            "temporal_ridge_fit_audit": temporal_fit,
            "metrics": metrics,
            "predictions": prediction_rows,
            "prediction_set_sha256": canonical_sha256(
                [row["prediction_sha256"] for row in prediction_rows]
            ),
        }
        fold["fold_sha256"] = canonical_sha256(fold)
        folds.append(fold)
        flat_predictions.extend(prediction_rows)

    result = {
        "event_keys": event_keys,
        "lap_time_observed_event_keys": observed_event_keys,
        "warmup_event_keys": observed_event_keys[: int(minimum_train_events)],
        "minimum_train_events": int(minimum_train_events),
        "minimum_inner_train_events": int(minimum_inner_train_events),
        "supervised_driver_event_count": int(len(full_frame)),
        "lap_time_model_driver_event_count": int(len(model_frame)),
        "censored_lap_time_driver_event_count": int((~observed_mask).sum()),
        "lap_time_target_counts_by_event": {
            str(event_key): counts for event_key, counts in target_counts_by_event.items()
        },
        "lap_time_benchmark_conditioning": (
            "continuous residual benchmark filters explicitly to "
            "lap_time_observed=true after target-free event-relative feature construction"
        ),
        "validated_tensor_count": int(full_frame["tensor_count_raw"].sum()),
        "lap_time_model_tensor_count": int(model_frame["tensor_count_raw"].sum()),
        "feature_names": list(FEATURE_NAMES),
        "feature_schema_sha256": canonical_sha256(list(FEATURE_NAMES)),
        "temporal_feature_names": list(TEMPORAL_FEATURE_NAMES),
        "temporal_feature_schema_sha256": canonical_sha256(
            list(TEMPORAL_FEATURE_NAMES)
        ),
        "candidate_grid_sha256": canonical_sha256(
            {
                "ridge_or_zero": [*RIDGE_CANDIDATES, ZERO_CORRECTION_CANDIDATE],
                "huber_or_zero": [*HUBER_CANDIDATES, ZERO_CORRECTION_CANDIDATE],
                "temporal_ridge_or_zero": [
                    *TEMPORAL_RIDGE_CANDIDATES,
                    ZERO_CORRECTION_CANDIDATE,
                ],
            }
        ),
        "folds": folds,
        "predictions": flat_predictions,
        "prediction_set_sha256": canonical_sha256(
            [row["prediction_sha256"] for row in flat_predictions]
        ),
        "summary": _summary(folds, flat_predictions),
        "temporal_capacity_diagnostic": _temporal_capacity_diagnostic(folds),
    }
    return result


def build_research_artifact(
    *,
    manifest_path: Path,
    root: Path,
    minimum_train_events: int = 3,
    minimum_inner_train_events: int = 2,
    generated_at: str | None = None,
) -> dict[str, Any]:
    repo_root = root.expanduser().resolve()
    implementation_manifest = _implementation_manifest(repo_root)
    source = manifest_path.expanduser().resolve()
    try:
        source_relative_path = source.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise TelemetryResidualResearchError(
            "supervised manifest must be stored inside the repository"
        ) from exc
    try:
        source_file_sha256 = sha256_file(source)
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TelemetryResidualResearchError(
            f"invalid supervised manifest {source}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise TelemetryResidualResearchError("supervised manifest must be a JSON object")
    if sha256_file(source) != source_file_sha256:
        raise TelemetryResidualResearchError(
            "supervised manifest changed while it was being loaded"
        )
    source_validation = validate_prequal_telemetry_supervised_manifest(
        manifest, root=repo_root
    )
    frame = aggregate_supervised_manifest(manifest, root=repo_root)
    benchmark = run_expanding_window_benchmark(
        frame,
        minimum_train_events=minimum_train_events,
        minimum_inner_train_events=minimum_inner_train_events,
    )
    if sha256_file(source) != source_file_sha256:
        raise TelemetryResidualResearchError(
            "supervised manifest changed during residual evaluation"
        )
    if _implementation_manifest(repo_root) != implementation_manifest:
        raise TelemetryResidualResearchError(
            "residual implementation changed during evaluation"
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "research_only_not_promotion_eligible",
        "implementation_manifest": implementation_manifest,
        "implementation_manifest_sha256": canonical_sha256(
            implementation_manifest
        ),
        "source_manifest": {
            "path": source_relative_path,
            "sha256": source_file_sha256,
            "schema_version": source_validation["schema_version"],
            "bag_set_sha256": source_validation["bag_set_sha256"],
            "feature_input_manifest_sha256": source_validation[
                "feature_input_manifest_sha256"
            ],
            "target_input_manifest_sha256": source_validation[
                "target_input_manifest_sha256"
            ],
        },
        "target": {
            "name": TARGET_NAME,
            "prediction_reconstruction": (
                "fastest_prequalifying_rehearsal_lap_seconds + "
                "train_only_source_shift_seconds + predicted_driver_correction_seconds"
            ),
            "source_shift": (
                "event-balanced median rehearsal-to-Q residual among strictly prior "
                "events with the same rehearsal_source; global prior-event median fallback"
            ),
            "model_training_target": (
                "source-adjusted residual centered by its training event median, "
                "which is algebraically the driver residual minus that training event median"
            ),
            "truth": (
                "best legal Grand Prix Qualifying lap conditional on "
                "has_legal_qualifying_lap=true"
            ),
            "censored_bags": (
                "bags with no legal Qualifying lap remain in source evidence with "
                "lap_time_observed=false and are counted but excluded from this continuous "
                "residual fit and score"
            ),
        },
        "validation_contract": {
            "supervised_row_unit": "driver_event_bag",
            "independent_evaluation_unit": "complete_event",
            "outer_split": "strict_expanding_window",
            "inner_selection_split": "strict_expanding_window_within_prior_events_only",
            "random_row_cross_validation_used": False,
            "tensor_level_split_used": False,
            "same_event_rows_cross_split": False,
            "target_event_used_for_training": False,
            "target_event_used_for_scaler": False,
            "target_event_used_for_hyperparameter_selection": False,
            "continuous_lap_time_model_conditions_on_observed_legal_lap": True,
            "censored_lap_time_bags_dropped_from_source_manifest": False,
            "event_equal_training_weights": True,
            "absolute_speed_or_rpm_model_features_used": False,
            "event_relative_feature_normalization_uses_targets": False,
            "event_relative_feature_normalization_available_prequalifying": True,
            "zero_telemetry_correction_is_selection_candidate": True,
            "pca_used": False,
            "deep_learning_used": False,
            "distance_ordered_temporal_summary_used": True,
            "temporal_summary_normalized_with_target_values": False,
            "learned_convolution_used": False,
            "fixed_event_count_as_capacity_claim_used": False,
            "source_and_implementation_rechecked_after_evaluation": True,
        },
        **benchmark,
    }
    payload["artifact_payload_sha256"] = canonical_sha256(
        _without(payload, "artifact_payload_sha256")
    )
    return payload


def _parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run event-blocked Ridge/Huber research on telemetry bags."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "data/f1/derived/prequal_telemetry_supervised_2026.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--minimum-train-events", type=int, default=3)
    parser.add_argument("--minimum-inner-train-events", type=int, default=2)
    parser.add_argument("--generated-at", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    root = find_repo_root()
    args = _parser(root).parse_args(argv)
    payload = build_research_artifact(
        manifest_path=args.manifest,
        root=root,
        minimum_train_events=args.minimum_train_events,
        minimum_inner_train_events=args.minimum_inner_train_events,
        generated_at=args.generated_at,
    )
    output = args.output or (
        root
        / "artifacts/backtests/f1/telemetry/prequal_telemetry_residual_research_2026.json"
    )
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": {
                    "path": str(destination),
                    "sha256": sha256_file(destination),
                    "size_bytes": int(destination.stat().st_size),
                },
                "artifact_payload_sha256": payload["artifact_payload_sha256"],
                "source_manifest": payload["source_manifest"],
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
