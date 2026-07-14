#!/usr/bin/env python3
"""Immutable diagnostic audit for full-field Qualifying probabilities.

The runner reads an already-produced Qualifying backtest.  It fits a frozen
temperature/Sinkhorn calibrator only on the artifact's explicitly declared
calibration partition, then compares raw and calibrated probabilities only on
the disjoint declared audit partition.  Outputs use exclusive-create semantics
and bind every result to the exact source bytes and event-level forecast hashes.
"""

from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from packages.f1.models.pre_quali.probability_calibration import (
    QualifyingPositionOutcome,
    QualifyingPositionProbabilityMatrix,
    audit_qualifying_position_probabilities,
    fit_qualifying_probability_calibrator,
)


AUDIT_SCHEMA_VERSION = "f1_qualifying_probability_temperature_sinkhorn_audit_v3"
SUPPORTED_SOURCE_SCHEMAS = frozenset(
    {
        "f1_shared_qualifying_latent_event_block_v3",
        "f1_shared_qualifying_latent_event_block_v4",
        "f1_shared_qualifying_latent_event_block_v5",
        "f1_shared_qualifying_latent_event_block_v6",
    }
)
SUPPORTED_FORECAST_SCHEMAS = frozenset(
    {
        "f1_shared_qualifying_forecast_artifact_v3",
        "f1_shared_qualifying_forecast_artifact_v4",
        "f1_shared_qualifying_forecast_artifact_v5",
    }
)
MIN_CALIBRATION_EVENTS_FOR_FIT = 2
MIN_CALIBRATION_EVENTS_FOR_PROMOTION_CONSIDERATION = 4
JEFFREYS_PSEUDOCOUNT = 0.5
JEFFREYS_SOURCE_MODEL_SUFFIX = "finite_sample_jeffreys_alpha_0_5_v1"
PROBABILITY_STOCHASTIC_TOLERANCE = 1e-7
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
POSITION_COLUMN_PATTERN = re.compile(r"^p_position_(\d+)$")
MARGINAL_BASE_COLUMNS = (
    "driver_id",
    "fastest_driver_probability",
    "fastest_lap_top3_probability",
    "valid_lap_probability_sampled",
    "reaches_q2_probability_sampled",
    "reaches_q3_probability_sampled",
    "expected_qualifying_position",
    "pole_probability",
    "top3_probability",
)
MARGINAL_STATUS_COLUMNS = (
    "probability_calibration_status",
    "position_marginals_calibrated",
)
PROPER_SCORE_FIELDS = (
    "multiclass_log_loss",
    "normalized_multiclass_brier",
    "top1_ece",
    "top3_ece",
    "top10_ece",
)
IMPLEMENTATION_RELATIVE_PATHS = (
    Path(
        "research/projects/F1/rising_qualification_prediction/Python/"
        "run_qualifying_probability_calibration_audit.py"
    ),
    Path(
        "research/projects/F1/rising_qualification_prediction/Python/"
        "repo_bootstrap.py"
    ),
    Path("packages/f1/models/pre_quali/probability_calibration.py"),
)


def _root() -> Path:
    return Path(__file__).resolve().parents[5]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _repository_relative_path(
    path: Path,
    *,
    repository_root: Path,
    field_name: str,
) -> str:
    resolved = path.expanduser().resolve()
    root = repository_root.expanduser().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"{field_name} must be stored inside the repository") from exc


def _implementation_manifest(repository_root: Path) -> list[dict[str, object]]:
    root = repository_root.expanduser().resolve()
    manifest: list[dict[str, object]] = []
    for relative_path in IMPLEMENTATION_RELATIVE_PATHS:
        path = (root / relative_path).resolve()
        observed_relative_path = _repository_relative_path(
            path,
            repository_root=root,
            field_name="implementation file",
        )
        if observed_relative_path != relative_path.as_posix():
            raise ValueError(
                "implementation file resolved to an unexpected repository path: "
                f"expected {relative_path.as_posix()!r}, observed {observed_relative_path!r}"
            )
        manifest.append(
            {
                "path": observed_relative_path,
                "sha256": _sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    return manifest


def _assert_evidence_unchanged(
    *,
    source_path: Path,
    source_sha256: str,
    implementation_manifest: Sequence[Mapping[str, object]],
    repository_root: Path,
) -> None:
    try:
        observed_source_sha256 = _sha256_file(source_path)
    except OSError as exc:
        raise RuntimeError("source artifact became unavailable during evaluation") from exc
    if observed_source_sha256 != source_sha256:
        raise RuntimeError("source artifact changed during probability calibration audit")
    try:
        observed_implementation_manifest = _implementation_manifest(repository_root)
    except OSError as exc:
        raise RuntimeError(
            "probability calibration implementation became unavailable during evaluation"
        ) from exc
    if list(implementation_manifest) != observed_implementation_manifest:
        raise RuntimeError(
            "probability calibration implementation changed during evaluation"
        )


def _validate_sha256(value: object, *, field_name: str) -> str:
    text = str(value).strip().lower() if value is not None else ""
    if SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return text


def _duplicate_rejecting_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"source JSON contains duplicate key {key!r}")
        output[key] = value
    return output


def _read_source(
    path: Path,
    *,
    expected_sha256: str | None,
) -> tuple[dict[str, Any], str]:
    source_bytes = path.read_bytes()
    source_sha256 = _sha256_bytes(source_bytes)
    if expected_sha256 is not None:
        pinned = _validate_sha256(expected_sha256, field_name="expected_source_sha256")
        if source_sha256 != pinned:
            raise ValueError(
                f"source artifact SHA mismatch: expected {pinned}, observed {source_sha256}"
            )
    try:
        payload = json.loads(
            source_bytes.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("source artifact is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("source artifact root must be a JSON object")
    return payload, source_sha256


def _frame_digest(frame: pd.DataFrame) -> str:
    """Reproduce the shared-forecast canonical dataframe digest."""

    canonical = frame.copy()
    sort_columns = [column for column in ("event_key", "driver_id") if column in canonical]
    if sort_columns:
        canonical = canonical.sort_values(sort_columns, kind="mergesort")
    canonical = canonical.reindex(sorted(canonical.columns), axis=1).reset_index(drop=True)
    payload = canonical.to_json(
        orient="split",
        date_format="iso",
        double_precision=15,
        index=False,
    )
    return _sha256_bytes(payload.encode("utf-8"))


def _integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{field_name} must be an integer")
    return int(numeric)


def _finite_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be finite numeric data")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be finite numeric data") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite numeric data")
    return numeric


def _jeffreys_smooth_permutation_marginals(
    probabilities: np.ndarray,
    *,
    sample_count: int,
    event_key: int | str = "support_smoothing",
) -> np.ndarray:
    """Give finite Monte Carlo marginals fixed, strictly positive support.

    Each full-order draw is a permutation, so every row and every column has
    exactly ``sample_count`` observations.  Adding the same Jeffreys count to
    every cell and dividing by ``sample_count + field_size / 2`` therefore
    preserves both stochastic constraints.  The pseudo-count is fixed here,
    rather than exposed as an audit-tunable hyperparameter.
    """

    key = str(event_key)
    if isinstance(sample_count, bool) or not isinstance(sample_count, (int, np.integer)):
        raise ValueError(f"event {key}: joint_sample_count must be an integer")
    count = int(sample_count)
    if count <= 0:
        raise ValueError(f"event {key}: joint_sample_count must be positive")
    try:
        raw = np.array(probabilities, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"event {key}: probabilities must be numeric") from exc
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or raw.shape[0] < 2:
        raise ValueError(
            f"event {key}: probabilities must be a square full-field matrix"
        )
    if not np.isfinite(raw).all():
        raise ValueError(f"event {key}: probabilities must all be finite")
    if (raw < 0.0).any() or (raw > 1.0).any():
        raise ValueError(f"event {key}: probabilities must lie in [0, 1]")
    if not np.allclose(
        raw.sum(axis=1),
        1.0,
        atol=PROBABILITY_STOCHASTIC_TOLERANCE,
        rtol=0.0,
    ):
        raise ValueError(f"event {key}: every driver row must sum to one")
    if not np.allclose(
        raw.sum(axis=0),
        1.0,
        atol=PROBABILITY_STOCHASTIC_TOLERANCE,
        rtol=0.0,
    ):
        raise ValueError(f"event {key}: every position column must sum to one")

    field_size = int(raw.shape[0])
    denominator = float(count) + JEFFREYS_PSEUDOCOUNT * field_size
    smoothed = (raw * float(count) + JEFFREYS_PSEUDOCOUNT) / denominator
    if not (smoothed > 0.0).all():
        raise RuntimeError(f"event {key}: Jeffreys smoothing did not create full support")
    if not np.allclose(
        smoothed.sum(axis=1),
        1.0,
        atol=PROBABILITY_STOCHASTIC_TOLERANCE,
        rtol=0.0,
    ) or not np.allclose(
        smoothed.sum(axis=0),
        1.0,
        atol=PROBABILITY_STOCHASTIC_TOLERANCE,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"event {key}: Jeffreys smoothing broke permutation marginal constraints"
        )
    smoothed.setflags(write=False)
    return smoothed


def _event_keys(values: Sequence[object], *, field_name: str) -> tuple[int, ...]:
    keys = tuple(_integer(value, field_name=field_name) for value in values)
    if not keys:
        raise ValueError(f"{field_name} must declare at least one event")
    if len(set(keys)) != len(keys):
        raise ValueError(f"{field_name} must contain unique event keys")
    return tuple(sorted(keys))


def _source_partitions(payload: Mapping[str, object]) -> Mapping[str, object]:
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("source artifact protocol must be an object")
    partitions = protocol.get("event_partitions")
    if not isinstance(partitions, Mapping):
        raise ValueError("source artifact must declare protocol.event_partitions")
    return partitions


def _declared_partition(
    payload: Mapping[str, object],
    *,
    role: str,
    override: Sequence[int] | None,
) -> tuple[tuple[int, ...], str]:
    if override is not None:
        return _event_keys(override, field_name=f"{role}_event_keys"), "cli_explicit_event_keys"
    partitions = _source_partitions(payload)
    values = partitions.get(role)
    if not isinstance(values, list):
        raise ValueError(f"source artifact partition role {role!r} is missing or not a list")
    return _event_keys(values, field_name=f"protocol.event_partitions.{role}"), (
        f"source_protocol_partition_role:{role}"
    )


def _validate_source_schema(payload: Mapping[str, object]) -> str:
    schema = str(payload.get("schema_version", "")).strip()
    if schema not in SUPPORTED_SOURCE_SCHEMAS:
        raise ValueError(
            f"unsupported Qualifying backtest schema {schema!r}; "
            f"supported={sorted(SUPPORTED_SOURCE_SCHEMAS)}"
        )
    if payload.get("mode") != "qualifying_prediction":
        raise ValueError("source artifact mode must be qualifying_prediction")
    if payload.get("target") != "official_grand_prix_qualifying_classification":
        raise ValueError("source artifact has an incompatible target contract")
    if not isinstance(payload.get("predictions"), list):
        raise ValueError("source artifact predictions must be a list")
    if not isinstance(payload.get("shared_forecast_artifacts"), list):
        raise ValueError("source artifact shared_forecast_artifacts must be a list")
    _source_partitions(payload)
    return schema


def _validate_forecast_artifact(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("shared forecast artifact entries must be objects")
    schema = str(value.get("schema_version", "")).strip()
    if schema not in SUPPORTED_FORECAST_SCHEMAS:
        raise ValueError(f"unsupported shared forecast artifact schema {schema!r}")
    event_key = _integer(value.get("event_key"), field_name="shared_forecast.event_key")

    artifact_sha256 = _validate_sha256(
        value.get("artifact_sha256"),
        field_name=f"shared_forecast[{event_key}].artifact_sha256",
    )
    without_hash = dict(value)
    without_hash.pop("artifact_sha256", None)
    if _canonical_sha256(without_hash) != artifact_sha256:
        raise ValueError(f"shared forecast artifact {event_key} has an invalid artifact_sha256")

    for key, item in value.items():
        if key.endswith("sha256"):
            _validate_sha256(item, field_name=f"shared_forecast[{event_key}].{key}")
    model_manifest = value.get("model_manifest")
    if not isinstance(model_manifest, dict):
        raise ValueError(f"shared forecast artifact {event_key} has no model_manifest")
    expected_model_manifest_sha = _validate_sha256(
        value.get("model_manifest_sha256"),
        field_name=f"shared_forecast[{event_key}].model_manifest_sha256",
    )
    if _canonical_sha256(model_manifest) != expected_model_manifest_sha:
        raise ValueError(f"shared forecast artifact {event_key} has an invalid model manifest hash")
    training_manifest = value.get("training_partition_manifest")
    if not isinstance(training_manifest, dict):
        raise ValueError(f"shared forecast artifact {event_key} has no training partition manifest")
    expected_training_sha = _validate_sha256(
        model_manifest.get("training_partition_sha256"),
        field_name=f"shared_forecast[{event_key}].training_partition_sha256",
    )
    if _canonical_sha256(training_manifest) != expected_training_sha:
        raise ValueError(f"shared forecast artifact {event_key} has an invalid training manifest hash")

    raw_drivers = value.get("driver_ids")
    if not isinstance(raw_drivers, list):
        raise ValueError(f"shared forecast artifact {event_key} driver_ids must be a list")
    drivers = tuple(str(item).strip() for item in raw_drivers)
    if not drivers or any(not driver for driver in drivers) or len(set(drivers)) != len(drivers):
        raise ValueError(f"shared forecast artifact {event_key} driver_ids are invalid")
    return value


def _forecast_artifact_index(payload: Mapping[str, object]) -> Mapping[int, dict[str, Any]]:
    index: dict[int, dict[str, Any]] = {}
    raw_artifacts = payload.get("shared_forecast_artifacts")
    assert isinstance(raw_artifacts, list)
    for raw in raw_artifacts:
        artifact = _validate_forecast_artifact(raw)
        event_key = _integer(artifact["event_key"], field_name="shared_forecast.event_key")
        if event_key in index:
            raise ValueError(f"duplicate shared forecast artifact for event {event_key}")
        index[event_key] = artifact
    return index


def _prediction_rows_by_event(payload: Mapping[str, object]) -> Mapping[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    raw_predictions = payload.get("predictions")
    assert isinstance(raw_predictions, list)
    for raw in raw_predictions:
        if not isinstance(raw, dict):
            raise ValueError("prediction rows must be objects")
        event_key = _integer(raw.get("event_key"), field_name="prediction.event_key")
        grouped.setdefault(event_key, []).append(raw)
    return grouped


def _event_inputs(
    *,
    event_key: int,
    rows: Sequence[dict[str, Any]],
    forecast_artifact: Mapping[str, Any],
) -> tuple[QualifyingPositionProbabilityMatrix, QualifyingPositionOutcome, dict[str, object]]:
    if not rows:
        raise ValueError(f"event {event_key} has no prediction rows")
    field_sizes = {
        _integer(row.get("field_size"), field_name=f"event {event_key} field_size")
        for row in rows
    }
    if len(field_sizes) != 1:
        raise ValueError(f"event {event_key} has inconsistent field_size values")
    field_size = next(iter(field_sizes))
    if field_size < 2 or len(rows) != field_size:
        raise ValueError(
            f"event {event_key} has {len(rows)} rows but declares field_size={field_size}"
        )

    sorted_rows = sorted(rows, key=lambda row: str(row.get("driver_id", "")).strip())
    drivers = tuple(str(row.get("driver_id", "")).strip() for row in sorted_rows)
    if any(not driver for driver in drivers) or len(set(drivers)) != field_size:
        raise ValueError(f"event {event_key} driver field is incomplete or duplicated")
    forecast_drivers = tuple(str(value).strip() for value in forecast_artifact["driver_ids"])
    if set(drivers) != set(forecast_drivers):
        raise ValueError(f"event {event_key} prediction field differs from forecast manifest")

    expected_forecast_sha = _validate_sha256(
        forecast_artifact.get("artifact_sha256"),
        field_name=f"shared_forecast[{event_key}].artifact_sha256",
    )
    row_forecast_hashes = {
        _validate_sha256(
            row.get("shared_forecast_artifact_sha256"),
            field_name=f"event {event_key} prediction shared_forecast_artifact_sha256",
        )
        for row in rows
    }
    if row_forecast_hashes != {expected_forecast_sha}:
        raise ValueError(f"event {event_key} prediction rows are not bound to their forecast artifact")

    source_models = {str(row.get("qualifying_model", "")).strip() for row in rows}
    if len(source_models) != 1 or not next(iter(source_models)):
        raise ValueError(f"event {event_key} has inconsistent qualifying_model values")
    source_model_id = next(iter(source_models))

    required_probability_columns = tuple(
        f"p_position_{position}" for position in range(1, field_size + 1)
    )
    for column in required_probability_columns:
        if any(column not in row for row in rows):
            raise ValueError(f"event {event_key} is missing probability column {column}")
    all_position_columns = sorted(
        {
            key
            for row in rows
            for key in row
            if POSITION_COLUMN_PATTERN.fullmatch(key) is not None
        },
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )
    for column in all_position_columns:
        position = int(column.rsplit("_", 1)[1])
        if position <= field_size:
            continue
        if any(row.get(column) is not None and np.isfinite(float(row[column])) for row in rows):
            raise ValueError(f"event {event_key} contains probabilities beyond its field size")

    probabilities = np.array(
        [
            [
                _finite_float(row[column], field_name=f"event {event_key} {column}")
                for column in required_probability_columns
            ]
            for row in sorted_rows
        ],
        dtype=float,
    )
    positions = tuple(range(1, field_size + 1))

    actual_positions = tuple(
        _integer(
            row.get("actual_qualifying_position"),
            field_name=f"event {event_key} actual_qualifying_position",
        )
        for row in sorted_rows
    )
    if tuple(sorted(actual_positions)) != positions:
        raise ValueError(f"event {event_key} actual positions are not a complete permutation")
    outcome_payload = {
        "event_key": event_key,
        "driver_ids": list(drivers),
        "actual_positions": list(actual_positions),
        "target": "official_grand_prix_qualifying_classification",
    }
    outcome = QualifyingPositionOutcome(
        event_key=event_key,
        driver_ids=drivers,
        actual_positions=actual_positions,
        outcome_evidence_id=f"qualifying_outcome_sha256:{_canonical_sha256(outcome_payload)}",
    )

    marginal_columns = [
        *MARGINAL_BASE_COLUMNS,
        *required_probability_columns,
        *MARGINAL_STATUS_COLUMNS,
    ]
    missing_marginal_columns = sorted(
        column for column in marginal_columns if any(column not in row for row in rows)
    )
    if missing_marginal_columns:
        raise ValueError(
            f"event {event_key} is missing hashed marginal columns {missing_marginal_columns}"
        )
    reconstructed = pd.DataFrame(rows)[marginal_columns]
    observed_marginal_sha = _frame_digest(reconstructed)
    expected_marginal_sha = _validate_sha256(
        forecast_artifact.get("qualifying_position_marginals_sha256"),
        field_name=f"shared_forecast[{event_key}].qualifying_position_marginals_sha256",
    )
    if observed_marginal_sha != expected_marginal_sha:
        raise ValueError(f"event {event_key} probability marginals do not match their stored hash")

    if any(row.get("position_marginals_calibrated") is not False for row in rows):
        raise ValueError(f"event {event_key} input marginals must be explicitly uncalibrated")
    status_values = {str(row.get("probability_calibration_status", "")).strip() for row in rows}
    if len(status_values) != 1 or not next(iter(status_values)):
        raise ValueError(f"event {event_key} has inconsistent probability calibration status")
    pole = np.array(
        [_finite_float(row.get("pole_probability"), field_name="pole_probability") for row in sorted_rows]
    )
    top3 = np.array(
        [_finite_float(row.get("top3_probability"), field_name="top3_probability") for row in sorted_rows]
    )
    expected_position = np.array(
        [
            _finite_float(row.get("expected_qualifying_position"), field_name="expected_qualifying_position")
            for row in sorted_rows
        ]
    )
    if not np.allclose(pole, probabilities[:, 0], atol=1e-9, rtol=0.0):
        raise ValueError(f"event {event_key} pole_probability is inconsistent with position marginals")
    if not np.allclose(top3, probabilities[:, : min(3, field_size)].sum(axis=1), atol=1e-9, rtol=0.0):
        raise ValueError(f"event {event_key} top3_probability is inconsistent with position marginals")
    if not np.allclose(
        expected_position,
        probabilities @ np.arange(1, field_size + 1, dtype=float),
        atol=1e-9,
        rtol=0.0,
    ):
        raise ValueError(f"event {event_key} expected position is inconsistent with position marginals")

    joint_sample_count = _integer(
        forecast_artifact.get("joint_sample_count"),
        field_name=f"shared_forecast[{event_key}].joint_sample_count",
    )
    smoothed_probabilities = _jeffreys_smooth_permutation_marginals(
        probabilities,
        sample_count=joint_sample_count,
        event_key=event_key,
    )
    calibration_input_source_model_id = (
        f"{source_model_id}+{JEFFREYS_SOURCE_MODEL_SUFFIX}"
    )
    matrix = QualifyingPositionProbabilityMatrix(
        event_key=event_key,
        driver_ids=drivers,
        position_ids=positions,
        probabilities=smoothed_probabilities,
        source_model_id=calibration_input_source_model_id,
        prediction_evidence_id=expected_forecast_sha,
    )

    metadata = {
        "event_key": event_key,
        "field_size": field_size,
        "source_model_id": source_model_id,
        "calibration_input_source_model_id": calibration_input_source_model_id,
        "prediction_evidence_id": expected_forecast_sha,
        "outcome_evidence_id": outcome.outcome_evidence_id,
        "qualifying_position_marginals_sha256": expected_marginal_sha,
        "finite_sample_support_correction": {
            "method": "symmetric_jeffreys_cell_pseudocount",
            "pseudo_count_per_driver_position_cell": JEFFREYS_PSEUDOCOUNT,
            "joint_sample_count": joint_sample_count,
            "field_size": field_size,
            "normalization_denominator": (
                float(joint_sample_count) + JEFFREYS_PSEUDOCOUNT * field_size
            ),
            "raw_zero_probability_cell_count": int(np.count_nonzero(probabilities == 0.0)),
            "smoothed_minimum_probability": float(smoothed_probabilities.min()),
            "strictly_positive_support": True,
            "preserves_row_and_column_stochasticity": True,
        },
    }
    return matrix, outcome, metadata


def _metric_payload(metrics: object) -> dict[str, object]:
    payload = asdict(metrics)  # type: ignore[arg-type]
    payload["event_keys"] = list(payload["event_keys"])
    return payload


def _metric_delta(calibrated: Mapping[str, object], raw: Mapping[str, object]) -> dict[str, float]:
    return {
        field: float(calibrated[field]) - float(raw[field])
        for field in PROPER_SCORE_FIELDS
    }


def run(
    *,
    source_artifact: Path,
    expected_source_sha256: str | None = None,
    calibration_event_keys: Sequence[int] | None = None,
    audit_event_keys: Sequence[int] | None = None,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Build a deterministic diagnostic payload without writing it."""

    root = (repository_root or _root()).expanduser().resolve()
    source_path = source_artifact.expanduser().resolve()
    source_relative_path = _repository_relative_path(
        source_path,
        repository_root=root,
        field_name="source artifact",
    )
    implementation_manifest = _implementation_manifest(root)
    source, source_sha256 = _read_source(
        source_path,
        expected_sha256=expected_source_sha256,
    )
    source_schema = _validate_source_schema(source)
    calibration_keys, calibration_key_source = _declared_partition(
        source,
        role="calibration",
        override=calibration_event_keys,
    )
    audit_keys, audit_key_source = _declared_partition(
        source,
        role="audit",
        override=audit_event_keys,
    )
    overlap = sorted(set(calibration_keys).intersection(audit_keys))
    if overlap:
        raise ValueError(f"calibration and audit event keys overlap: {overlap}")
    if max(calibration_keys) >= min(audit_keys):
        raise ValueError("calibration events must be strictly earlier than every audit event")

    forecast_index = _forecast_artifact_index(source)
    prediction_index = _prediction_rows_by_event(source)
    selected_keys = tuple(sorted(set(calibration_keys).union(audit_keys)))
    matrices: list[QualifyingPositionProbabilityMatrix] = []
    outcomes: list[QualifyingPositionOutcome] = []
    event_metadata: dict[int, dict[str, object]] = {}
    for event_key in selected_keys:
        if event_key not in forecast_index:
            raise ValueError(f"event {event_key} has no shared forecast artifact")
        if event_key not in prediction_index:
            raise ValueError(f"event {event_key} has no prediction rows")
        matrix, outcome, metadata = _event_inputs(
            event_key=event_key,
            rows=prediction_index[event_key],
            forecast_artifact=forecast_index[event_key],
        )
        matrices.append(matrix)
        outcomes.append(outcome)
        event_metadata[event_key] = metadata

    # This is the only fit call.  Audit matrices and outcomes are not even
    # passed to it, in addition to the calibrator enforcing declared keys.
    calibration_key_set = set(calibration_keys)
    calibration_matrices = [
        matrix for matrix in matrices if int(matrix.event_key) in calibration_key_set
    ]
    calibration_outcomes = [
        outcome for outcome in outcomes if int(outcome.event_key) in calibration_key_set
    ]
    calibrator = fit_qualifying_probability_calibrator(
        calibration_matrices,
        calibration_outcomes,
        calibration_event_keys=calibration_keys,
    )
    raw_aggregate_metrics = audit_qualifying_position_probabilities(
        matrices,
        outcomes,
        audit_event_keys=audit_keys,
    )
    calibrated_aggregate_metrics = audit_qualifying_position_probabilities(
        matrices,
        outcomes,
        audit_event_keys=audit_keys,
        calibrator=calibrator,
    )
    raw_aggregate = _metric_payload(raw_aggregate_metrics)
    calibrated_aggregate = _metric_payload(calibrated_aggregate_metrics)

    per_event: list[dict[str, object]] = []
    for event_key in audit_keys:
        raw = _metric_payload(
            audit_qualifying_position_probabilities(
                matrices,
                outcomes,
                audit_event_keys=[event_key],
            )
        )
        calibrated = _metric_payload(
            audit_qualifying_position_probabilities(
                matrices,
                outcomes,
                audit_event_keys=[event_key],
                calibrator=calibrator,
            )
        )
        per_event.append(
            {
                **event_metadata[event_key],
                "jeffreys_smoothed_uncalibrated": raw,
                "calibrated": calibrated,
                "delta_calibrated_minus_jeffreys_smoothed_uncalibrated": (
                    _metric_delta(calibrated, raw)
                ),
            }
        )

    calibration_count = len(calibration_keys)
    promotion_status = (
        "diagnostic_only_insufficient_independent_calibration_events"
        if calibration_count < MIN_CALIBRATION_EVENTS_FOR_PROMOTION_CONSIDERATION
        else "diagnostic_only_requires_separate_locked_promotion_audit"
    )
    _assert_evidence_unchanged(
        source_path=source_path,
        source_sha256=source_sha256,
        implementation_manifest=implementation_manifest,
        repository_root=root,
    )
    result: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "mode": "qualifying_position_probability_calibration_audit",
        "promotion_status": promotion_status,
        "promoted": False,
        "source_artifact": {
            "path": source_relative_path,
            "sha256": source_sha256,
            "schema_version": source_schema,
        },
        "implementation_manifest": implementation_manifest,
        "implementation_manifest_sha256": _canonical_sha256(
            implementation_manifest
        ),
        "protocol": {
            "calibration_event_keys": list(calibration_keys),
            "calibration_event_key_source": calibration_key_source,
            "audit_event_keys": list(audit_keys),
            "audit_event_key_source": audit_key_source,
            "partitions_disjoint": True,
            "calibration_strictly_precedes_audit": True,
            "audit_outcomes_used_in_fit": False,
            "fit_objective": "event_balanced_multiclass_negative_log_likelihood",
            "finite_sample_support_correction": {
                "method": "symmetric_jeffreys_cell_pseudocount",
                "pseudo_count_per_driver_position_cell": JEFFREYS_PSEUDOCOUNT,
                "status": "fixed_untuned_preprocessing",
                "applied_before_temperature_and_sinkhorn": True,
                "requires_declared_joint_sample_count": True,
                "preserves_permutation_row_and_column_stochasticity": True,
                "calibration_source_model_suffix": JEFFREYS_SOURCE_MODEL_SUFFIX,
            },
            "transform": (
                "fixed_jeffreys_support_then_log_temperature_scaling_"
                "then_sinkhorn_projection"
            ),
        },
        "calibration": {
            "independent_event_count": calibration_count,
            "minimum_events_for_fit": MIN_CALIBRATION_EVENTS_FOR_FIT,
            "minimum_events_for_promotion_consideration": (
                MIN_CALIBRATION_EVENTS_FOR_PROMOTION_CONSIDERATION
            ),
            "model_card": calibrator.model_card.to_dict(),
            "event_provenance": [event_metadata[key] for key in calibration_keys],
        },
        "aggregate_audit": {
            "jeffreys_smoothed_uncalibrated": raw_aggregate,
            "calibrated": calibrated_aggregate,
            "delta_calibrated_minus_jeffreys_smoothed_uncalibrated": (
                _metric_delta(
                    calibrated_aggregate,
                    raw_aggregate,
                )
            ),
        },
        "per_event_audit": per_event,
        "artifact_contract": {
            "write_mode": "exclusive_create",
            "result_hash_excludes_only_result_sha256": True,
            "source_path_scope": "repository_relative_only",
            "source_read_mode": (
                "single_immutable_byte_snapshot_with_post_evaluation_digest_check"
            ),
            "source_verified_unchanged_after_evaluation": True,
            "implementation_manifest_verified_unchanged_after_evaluation": True,
        },
    }
    result["result_sha256"] = _canonical_sha256(result)
    return result


def write_exclusive(payload: Mapping[str, object], output: Path) -> Path:
    """Write one valid-JSON audit artifact without overwriting prior evidence."""

    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    )
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.write("\n")
    return destination


def _csv_event_keys(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            _root()
            / "artifacts/backtests/f1/qualifying/shared_latent_same_season_v1.json"
        ),
        help=(
            "canonical same-season Qualifying artifact containing raw p_position_* "
            "rows; cross-season diagnostics require an explicit --input"
        ),
    )
    parser.add_argument(
        "--source-sha256",
        default=None,
        help="optional immutable source pin; a mismatch fails closed",
    )
    parser.add_argument(
        "--calibration-events",
        type=_csv_event_keys,
        default=None,
        help="explicit event keys; default is protocol.event_partitions.calibration",
    )
    parser.add_argument(
        "--audit-events",
        type=_csv_event_keys,
        default=None,
        help="explicit event keys; default is protocol.event_partitions.audit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            _root()
            / "artifacts/backtests/f1/qualifying/"
            "shared_latent_same_season_temperature_sinkhorn_probability_audit.json"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(
        source_artifact=args.input,
        expected_source_sha256=args.source_sha256,
        calibration_event_keys=args.calibration_events,
        audit_event_keys=args.audit_events,
    )
    destination = write_exclusive(payload, args.output)
    print(
        json.dumps(
            {
                "output": str(destination),
                "source_sha256": payload["source_artifact"]["sha256"],
                "result_sha256": payload["result_sha256"],
                "promotion_status": payload["promotion_status"],
                "selected_temperature": payload["calibration"]["model_card"][
                    "selected_temperature"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Suggested commit name: feat(f1-quali): add immutable position-probability calibration audit
