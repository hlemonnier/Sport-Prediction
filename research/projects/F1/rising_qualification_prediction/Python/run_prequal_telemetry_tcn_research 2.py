#!/usr/bin/env python3
"""Run a real, bounded TCN on pre-Qualifying telemetry driver-event bags.

This runner deliberately does not route the supervised manifest through the
production six-output Ultimate Lap-Time trainer.  That trainer requires an
achievable-session-end lap plus sector labels; the manifest contains a best
legal Qualifying lap and one to three correlated rehearsal tensors per
driver-event.  Here exactly one tensor is causally aligned to the bag's fastest
rehearsal-lap reference: a unique ``push_lap_rank=1`` tensor is preferred and
must agree with the minimum rehearsal time, otherwise a unique minimum-time
tensor is required.  The other tensors remain validated correlated evidence;
they never become independent rows.  A very small dilated TCN then predicts
only a driver-relative correction on top of a train-only rehearsal-source
shift.

Every outer target is a complete later event.  The immediately preceding
event is an inner validation block used for early stopping and for the locked
TCN-versus-zero decision.  The network is then refit for the frozen epoch count
using every prior event and scored once on the untouched target event.  This
is an honest low-capacity research evaluation, never promotion evidence.
"""

from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from packages.f1.data.providers.telemetry_cache import (
    NORMALIZED_TELEMETRY_CHANNELS,
    sha256_file,
)
from packages.f1.data.providers.telemetry_supervised import (
    canonical_sha256,
    validate_prequal_telemetry_supervised_manifest,
)
from packages.f1.models.ultimate_lap_time.deep import (
    DistanceTelemetryResidualTCN,
    DistanceTelemetryResidualTCNConfig,
    seed_torch,
    torch,
    torch_available,
    trainable_parameter_count,
)
from packages.sports_core.paths import find_repo_root
from run_prequal_telemetry_residual_research import (
    TelemetryResidualResearchError,
    _driver_relative_training_target,
    _event_equal_weights,
    _prediction_metrics,
    _raw_shift_prediction,
    _source_shift_predictions,
    aggregate_supervised_manifest,
)


SCHEMA_VERSION = "f1_prequal_telemetry_true_tcn_research_v2"
TENSOR_ADAPTER_CONTRACT = "fastest_rehearsal_reference_tensor_v1"
ANCHOR_LAP_TIME_TOLERANCE_SECONDS = 0.001
STATIC_FEATURE_NAMES: tuple[str, ...] = (
    "event_relative_rehearsal_reference_z",
    "event_relative_rehearsal_reference_rank",
)
TELEMETRY_INPUT_OBSERVED = "observed_anchor_aligned"
TELEMETRY_INPUT_ZERO_ABLATION = "zero_telemetry_static_anchor_sham"
TELEMETRY_INPUT_MODES: tuple[str, ...] = (
    TELEMETRY_INPUT_OBSERVED,
    TELEMETRY_INPUT_ZERO_ABLATION,
)
PAIRED_EVENT_BOOTSTRAP_DRAWS = 20_000
PAIRED_EVENT_BOOTSTRAP_SEED = 20260714
PROFILE_DESIGN_STAGE_ORIGINAL_CONTROL = "original_pre_sensitivity_control"
PROFILE_DESIGN_STAGE_POSTDEVELOPMENT = (
    "postdevelopment_outer_results_informed_before_durable_matrix"
)
PROFILE_DESIGN_STAGE_POST_SENSITIVITY = (
    "post_sensitivity_outer_results_informed_followup"
)
PROFILE_DESIGN_STAGE_UNSPECIFIED = (
    "standalone_design_history_unspecified_fail_closed"
)
PROFILE_DESIGN_STAGE_OUTER_TARGET_INFORMED: dict[str, bool] = {
    PROFILE_DESIGN_STAGE_ORIGINAL_CONTROL: False,
    PROFILE_DESIGN_STAGE_POSTDEVELOPMENT: True,
    PROFILE_DESIGN_STAGE_POST_SENSITIVITY: True,
    PROFILE_DESIGN_STAGE_UNSPECIFIED: True,
}
PROFILE_DESIGN_PROVENANCE_FIELDS: tuple[str, ...] = (
    "design_stage",
    "prior_outer_results_informed_profile_design",
    "same_outer_evaluation_targets_seen_before_profile_freeze",
    "hyperparameters_tuned_on_outer_targets",
    "durable_matrix_results_used_to_select_profile",
    "promotion_eligible_from_profile_design",
)


@dataclass(frozen=True)
class TCNResearchConfig:
    minimum_train_events: int = 4
    maximum_epochs: int = 60
    early_stopping_patience: int = 8
    early_stopping_min_delta_seconds: float = 0.001
    learning_rate: float = 3e-3
    weight_decay: float = 1e-2
    huber_beta_seconds: float = 0.25
    seed: int = 20260714
    hidden_channels: int = 4
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2)
    dropout: float = 0.0
    head_hidden_dim: int = 4
    max_abs_correction_seconds: float = 1.5
    minimum_inner_relative_gain_to_select_tcn: float = 0.02
    telemetry_input_mode: str = TELEMETRY_INPUT_OBSERVED


@dataclass(frozen=True)
class TCNBagDataset:
    frame: pd.DataFrame
    telemetry: np.ndarray
    static_features: np.ndarray
    channel_names: tuple[str, ...]
    distance_bins: int
    validated_tensor_count: int
    feature_set_sha256: str

    def __post_init__(self) -> None:
        rows = len(self.frame)
        if self.telemetry.shape != (
            rows,
            len(self.channel_names),
            int(self.distance_bins),
        ):
            raise ValueError("telemetry array is not aligned to driver-event rows")
        if self.static_features.shape != (rows, len(STATIC_FEATURE_NAMES)):
            raise ValueError("static feature array is not aligned to driver-event rows")
        if self.frame.duplicated(["event_key", "driver_id"]).any():
            raise ValueError("TCN dataset must contain one row per driver-event bag")
        if not np.isfinite(self.telemetry).all():
            raise ValueError("TCN telemetry contains non-finite values")
        if not np.isfinite(self.static_features).all():
            raise ValueError("TCN static features contain non-finite values")


def normalize_profile_design_provenance(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return an explicit, internally consistent profile-design contract.

    The expanding-window folds still keep each *current execution* target out
    of fitting, early stopping, and the locked inner selector.  That property
    does not make a post-development profile promotion evidence when the same
    outer evaluation targets were already inspected while choosing its
    hyperparameters.  Unknown standalone history therefore fails closed as
    outer-target-informed instead of silently claiming an a-priori design.
    """

    if value is None:
        payload: dict[str, Any] = {
            "design_stage": PROFILE_DESIGN_STAGE_UNSPECIFIED,
            "prior_outer_results_informed_profile_design": True,
            "same_outer_evaluation_targets_seen_before_profile_freeze": True,
            "hyperparameters_tuned_on_outer_targets": True,
            "durable_matrix_results_used_to_select_profile": False,
            "promotion_eligible_from_profile_design": False,
        }
    else:
        payload = dict(value)
    if set(payload) != set(PROFILE_DESIGN_PROVENANCE_FIELDS):
        raise TelemetryResidualResearchError(
            "profile design provenance fields must match the frozen contract"
        )
    stage = str(payload.get("design_stage") or "")
    if stage not in PROFILE_DESIGN_STAGE_OUTER_TARGET_INFORMED:
        raise TelemetryResidualResearchError(
            "profile design provenance uses an unsupported design stage"
        )
    informed = PROFILE_DESIGN_STAGE_OUTER_TARGET_INFORMED[stage]
    boolean_fields = PROFILE_DESIGN_PROVENANCE_FIELDS[1:]
    if any(not isinstance(payload.get(field), bool) for field in boolean_fields):
        raise TelemetryResidualResearchError(
            "profile design provenance flags must be JSON booleans"
        )
    consistency_fields = (
        "prior_outer_results_informed_profile_design",
        "same_outer_evaluation_targets_seen_before_profile_freeze",
        "hyperparameters_tuned_on_outer_targets",
    )
    if any(payload[field] is not informed for field in consistency_fields):
        raise TelemetryResidualResearchError(
            "profile design provenance contradicts its outer-target history"
        )
    if payload["durable_matrix_results_used_to_select_profile"] is not False:
        raise TelemetryResidualResearchError(
            "durable sensitivity-matrix results cannot select a profile"
        )
    if payload["promotion_eligible_from_profile_design"] is not False:
        raise TelemetryResidualResearchError(
            "TCN profile design must remain promotion-ineligible"
        )
    return payload


def _resolve_tensor_path(value: object, *, root: Path) -> Path:
    text = str(value or "").strip()
    if not text:
        raise TelemetryResidualResearchError("telemetry tensor path is missing")
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_bag_tensor(
    bag: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load all correlated evidence and select the fastest-reference tensor.

    Selection is independent of manifest order and path naming.  A unique
    rank-one tensor is authoritative only when its rehearsal time agrees with
    the minimum time to millisecond precision.  Without rank one, the minimum
    itself must be unique at that precision.  This fails closed rather than
    silently training on an arbitrary tensor.
    """

    feature = bag.get("feature")
    if not isinstance(feature, Mapping):
        raise TelemetryResidualResearchError("telemetry bag feature must be an object")
    tensors = feature.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise TelemetryResidualResearchError("telemetry bag requires at least one tensor")
    if len(tensors) > 3:
        raise TelemetryResidualResearchError(
            "telemetry bag cannot contain more than three correlated tensors"
        )
    loaded: list[dict[str, Any]] = []
    expected_shape: tuple[int, int] | None = None
    for evidence in tensors:
        if not isinstance(evidence, Mapping):
            raise TelemetryResidualResearchError("tensor evidence must be an object")
        try:
            lap_number = int(evidence["lap_number"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TelemetryResidualResearchError(
                "tensor evidence requires a positive integer lap_number"
            ) from exc
        if isinstance(evidence.get("lap_number"), bool) or lap_number <= 0:
            raise TelemetryResidualResearchError(
                "tensor evidence requires a positive integer lap_number"
            )
        raw_rank = evidence.get("push_lap_rank")
        if raw_rank is None:
            push_lap_rank = None
        else:
            try:
                push_lap_rank = int(raw_rank)
            except (TypeError, ValueError) as exc:
                raise TelemetryResidualResearchError(
                    "push_lap_rank must be a positive integer or null"
                ) from exc
            if isinstance(raw_rank, bool) or push_lap_rank <= 0:
                raise TelemetryResidualResearchError(
                    "push_lap_rank must be a positive integer or null"
                )
        raw_rehearsal_time = evidence.get("rehearsal_lap_time_seconds")
        if isinstance(raw_rehearsal_time, bool):
            raise TelemetryResidualResearchError(
                "tensor evidence requires a positive rehearsal_lap_time_seconds"
            )
        try:
            rehearsal_time = float(raw_rehearsal_time)
        except (TypeError, ValueError) as exc:
            raise TelemetryResidualResearchError(
                "tensor evidence requires a positive rehearsal_lap_time_seconds"
            ) from exc
        if not math.isfinite(rehearsal_time) or rehearsal_time <= 0.0:
            raise TelemetryResidualResearchError(
                "tensor evidence requires a positive rehearsal_lap_time_seconds"
            )
        path = _resolve_tensor_path(evidence.get("path"), root=root)
        expected_digest = str(evidence.get("sha256") or "").strip().lower()
        actual_digest = sha256_file(path)
        if not expected_digest or expected_digest != actual_digest:
            raise TelemetryResidualResearchError(
                f"telemetry tensor hash mismatch for {path}"
            )
        with np.load(path, allow_pickle=False) as payload:
            array = np.asarray(payload["values"], dtype=np.float32)
            channels = tuple(str(value) for value in payload["channel_names"].tolist())
        if channels != tuple(NORMALIZED_TELEMETRY_CHANNELS):
            raise TelemetryResidualResearchError(
                f"unexpected telemetry channel order for {path}: {channels}"
            )
        if array.ndim != 2 or array.shape[0] != len(channels):
            raise TelemetryResidualResearchError(
                f"telemetry tensor has invalid shape for {path}: {array.shape}"
            )
        if not np.isfinite(array).all():
            raise TelemetryResidualResearchError(
                f"telemetry tensor contains non-finite values: {path}"
            )
        if expected_shape is None:
            expected_shape = tuple(int(value) for value in array.shape)
        elif tuple(array.shape) != expected_shape:
            raise TelemetryResidualResearchError(
                "all tensors inside a driver-event bag must share one shape"
            )
        loaded.append(
            {
                "array": array,
                "lap_number": lap_number,
                "push_lap_rank": push_lap_rank,
                "rehearsal_lap_time_seconds": rehearsal_time,
                "sha256": actual_digest,
            }
        )

    minimum_time = min(
        float(item["rehearsal_lap_time_seconds"]) for item in loaded
    )
    rank_one = [item for item in loaded if item["push_lap_rank"] == 1]
    if len(rank_one) > 1:
        raise TelemetryResidualResearchError(
            "telemetry bag has ambiguous duplicate push_lap_rank=1 tensors"
        )
    if rank_one:
        selected = rank_one[0]
        if not math.isclose(
            float(selected["rehearsal_lap_time_seconds"]),
            minimum_time,
            rel_tol=0.0,
            abs_tol=ANCHOR_LAP_TIME_TOLERANCE_SECONDS,
        ):
            raise TelemetryResidualResearchError(
                "push_lap_rank=1 tensor time is inconsistent with the fastest "
                "rehearsal reference"
            )
        selection_method = "unique_push_lap_rank_1_verified_against_minimum_time"
    else:
        minimum_candidates = [
            item
            for item in loaded
            if math.isclose(
                float(item["rehearsal_lap_time_seconds"]),
                minimum_time,
                rel_tol=0.0,
                abs_tol=ANCHOR_LAP_TIME_TOLERANCE_SECONDS,
            )
        ]
        if len(minimum_candidates) != 1:
            raise TelemetryResidualResearchError(
                "telemetry bag has no rank-one anchor and its minimum rehearsal "
                "time is ambiguous"
            )
        selected = minimum_candidates[0]
        selection_method = "unique_minimum_time_without_push_lap_rank_1"

    correlated_evidence = [
        {
            "lap_number": int(item["lap_number"]),
            "push_lap_rank": item["push_lap_rank"],
            "rehearsal_lap_time_seconds": float(
                item["rehearsal_lap_time_seconds"]
            ),
            "sha256": str(item["sha256"]),
        }
        for item in sorted(
            loaded,
            key=lambda item: (
                float(item["rehearsal_lap_time_seconds"]),
                int(item["push_lap_rank"] or 10**9),
                int(item["lap_number"]),
                str(item["sha256"]),
            ),
        )
    ]
    selection_audit = {
        "adapter_contract": TENSOR_ADAPTER_CONTRACT,
        "selection_method": selection_method,
        "anchor_time_tolerance_seconds": ANCHOR_LAP_TIME_TOLERANCE_SECONDS,
        "correlated_tensor_count": int(len(loaded)),
        "correlated_tensors": correlated_evidence,
        "selected_tensor_lap_number": int(selected["lap_number"]),
        "selected_tensor_push_lap_rank": selected["push_lap_rank"],
        "selected_tensor_rehearsal_lap_time_seconds": float(
            selected["rehearsal_lap_time_seconds"]
        ),
        "selected_tensor_sha256": str(selected["sha256"]),
    }
    # One driver-event remains the supervised unit.  The unselected rehearsal
    # laps are validated and audited, but never aggregated or sampled as rows.
    return np.asarray(selected["array"], dtype=np.float32), selection_audit


def _event_relative_tensor_features(
    raw: np.ndarray,
    event_keys: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove circuit-common shape using only the pre-target event roster."""

    values = np.asarray(raw, dtype=np.float32)
    keys = np.asarray(event_keys)
    output = np.empty_like(values)
    scale_rows: dict[str, list[float]] = {}
    for event_key in sorted(int(value) for value in np.unique(keys)):
        mask = keys == event_key
        event = values[mask]
        center = np.median(event, axis=0)
        centered = event - center[None, :, :]
        scales = 1.4826 * np.median(np.abs(centered), axis=(0, 2))
        fallback = np.std(centered, axis=(0, 2), dtype=np.float64)
        scales = np.where(scales > 1e-4, scales, fallback)
        scales = np.where(scales > 1e-4, scales, 1.0).astype(np.float32)
        output[mask] = np.clip(centered / scales[None, :, None], -6.0, 6.0)
        scale_rows[str(event_key)] = [float(value) for value in scales]
    return output, {
        "method": "per_event_per_distance_bin_driver_median_then_per_channel_robust_scale",
        "target_values_used": False,
        "target_event_feature_roster_used": True,
        "available_before_qualifying": True,
        "clip": [-6.0, 6.0],
        "per_event_channel_scales": scale_rows,
    }


def _event_relative_static_features(frame: pd.DataFrame) -> np.ndarray:
    output = np.empty((len(frame), len(STATIC_FEATURE_NAMES)), dtype=np.float32)
    for event_key, rows in frame.groupby("event_key", sort=True):
        indices = rows.index.to_numpy(dtype=int)
        reference = rows["rehearsal_reference_seconds"].to_numpy(dtype=float)
        median = float(np.median(reference))
        scale = float(1.4826 * np.median(np.abs(reference - median)))
        if not math.isfinite(scale) or scale <= 1e-6:
            scale = float(np.std(reference))
        if not math.isfinite(scale) or scale <= 1e-6:
            scale = 1.0
        z_score = np.clip((reference - median) / scale, -6.0, 6.0)
        if len(reference) == 1:
            rank = np.full(1, 0.5, dtype=float)
        else:
            ordinal = (
                pd.Series(reference).rank(method="average").to_numpy(dtype=float)
            )
            rank = (ordinal - 1.0) / float(len(reference) - 1)
        output[indices, 0] = z_score.astype(np.float32)
        output[indices, 1] = (2.0 * rank - 1.0).astype(np.float32)
    return output


def build_tcn_bag_dataset(
    manifest: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[TCNBagDataset, dict[str, Any]]:
    """Validate and adapt the manifest without inventing quantile/sector labels."""

    root = root.expanduser().resolve()
    # The existing aggregate adapter validates the manifest, nested hashes,
    # evidence timestamps, tensor digests and one-bag-per-driver-event contract.
    full_frame = aggregate_supervised_manifest(manifest, root=root)
    bags = manifest.get("bags")
    if not isinstance(bags, list):
        raise TelemetryResidualResearchError("supervised manifest bags must be a list")
    raw_by_identity: dict[tuple[int, str], np.ndarray] = {}
    tensor_counts: dict[tuple[int, str], int] = {}
    tensor_selection_by_identity: dict[tuple[int, str], dict[str, Any]] = {}
    common_shape: tuple[int, int] | None = None
    for raw_bag in bags:
        if not isinstance(raw_bag, Mapping):
            raise TelemetryResidualResearchError("supervised bag must be an object")
        identity = (
            int(raw_bag["event_key"]),
            str(raw_bag["driver_id"]).strip().upper(),
        )
        tensor, selection_audit = _load_bag_tensor(raw_bag, root=root)
        if identity in raw_by_identity:
            raise TelemetryResidualResearchError(
                f"duplicate telemetry driver-event bag: {identity}"
            )
        if common_shape is None:
            common_shape = tuple(int(value) for value in tensor.shape)
        elif tuple(tensor.shape) != common_shape:
            raise TelemetryResidualResearchError(
                "all driver-event bag tensors must share one canonical shape"
            )
        raw_by_identity[identity] = tensor
        tensor_counts[identity] = int(selection_audit["correlated_tensor_count"])
        tensor_selection_by_identity[identity] = selection_audit
    if common_shape is None:
        raise TelemetryResidualResearchError("supervised manifest contains no tensors")

    # Build every feature using the complete pre-target telemetry roster first.
    # Filtering the continuous outcome after normalization prevents a future
    # no-legal-lap status from changing another driver's input tensor.
    ordered = full_frame.sort_values(
        ["event_key", "driver_id"], kind="mergesort"
    ).reset_index(drop=True)
    all_identities = [
        (int(row.event_key), str(row.driver_id))
        for row in ordered.itertuples(index=False)
    ]
    missing = [
        identity for identity in all_identities if identity not in raw_by_identity
    ]
    if missing:
        raise TelemetryResidualResearchError(
            f"supervised rows are missing telemetry bags: {missing[:5]}"
        )
    tensor_selection_records: list[dict[str, Any]] = []
    for identity, row in zip(all_identities, ordered.itertuples(index=False)):
        selection = tensor_selection_by_identity[identity]
        selected_time = float(
            selection["selected_tensor_rehearsal_lap_time_seconds"]
        )
        rehearsal_reference = float(row.rehearsal_reference_seconds)
        alignment_error = abs(selected_time - rehearsal_reference)
        if not math.isclose(
            selected_time,
            rehearsal_reference,
            rel_tol=0.0,
            abs_tol=ANCHOR_LAP_TIME_TOLERANCE_SECONDS,
        ):
            raise TelemetryResidualResearchError(
                f"selected telemetry tensor for {identity} does not align with "
                "the fastest rehearsal reference"
            )
        tensor_selection_records.append(
            {
                "event_key": int(identity[0]),
                "driver_id": str(identity[1]),
                "selection_method": str(selection["selection_method"]),
                "anchor_time_tolerance_seconds": float(
                    selection["anchor_time_tolerance_seconds"]
                ),
                "correlated_tensor_count": int(
                    selection["correlated_tensor_count"]
                ),
                "correlated_tensors": selection["correlated_tensors"],
                "selected_tensor_lap_number": int(
                    selection["selected_tensor_lap_number"]
                ),
                "selected_tensor_push_lap_rank": selection[
                    "selected_tensor_push_lap_rank"
                ],
                "selected_tensor_rehearsal_lap_time_seconds": selected_time,
                "selected_tensor_sha256": str(
                    selection["selected_tensor_sha256"]
                ),
                "rehearsal_reference_seconds": rehearsal_reference,
                "reference_alignment_error_seconds": float(alignment_error),
            }
        )
    tensor_selection_sha256 = canonical_sha256(tensor_selection_records)
    raw = np.stack(
        [raw_by_identity[identity] for identity in all_identities], axis=0
    )
    event_keys = ordered["event_key"].to_numpy(dtype=int)
    all_telemetry, normalization_audit = _event_relative_tensor_features(
        raw, event_keys
    )
    all_static = _event_relative_static_features(ordered)
    observed_mask = ordered["lap_time_observed"].to_numpy(dtype=bool)
    observed = ordered.loc[observed_mask].reset_index(drop=True)
    telemetry = all_telemetry[observed_mask]
    static = all_static[observed_mask]
    identities = [
        identity
        for identity, is_observed in zip(
            all_identities, observed_mask.tolist()
        )
        if is_observed
    ]
    feature_set_sha256 = canonical_sha256(
        {
            "tensor_adapter_contract": TENSOR_ADAPTER_CONTRACT,
            "anchor_lap_time_tolerance_seconds": (
                ANCHOR_LAP_TIME_TOLERANCE_SECONDS
            ),
            "driver_event_tensor_selection_sha256": tensor_selection_sha256,
            "bag_sha256": observed["bag_sha256"].astype(str).tolist(),
            "tensor_counts": [tensor_counts[identity] for identity in identities],
            "telemetry_sha256": hashlib.sha256(telemetry.tobytes()).hexdigest(),
            "static_sha256": hashlib.sha256(static.tobytes()).hexdigest(),
        }
    )
    dataset = TCNBagDataset(
        frame=observed,
        telemetry=telemetry,
        static_features=static,
        channel_names=tuple(NORMALIZED_TELEMETRY_CHANNELS),
        distance_bins=int(common_shape[1]),
        validated_tensor_count=int(sum(tensor_counts.values())),
        feature_set_sha256=feature_set_sha256,
    )
    audit = {
        "supervised_row_unit": "driver_event_bag",
        "tensor_rows_used_as_independent_examples": False,
        "bag_tensor_adapter_contract": TENSOR_ADAPTER_CONTRACT,
        "bag_tensor_selection_method": (
            "prefer_unique_push_lap_rank_1_verified_against_minimum_time_else_"
            "require_unique_minimum_time"
        ),
        "bag_tensor_anchor_time_tolerance_seconds": (
            ANCHOR_LAP_TIME_TOLERANCE_SECONDS
        ),
        "bag_tensor_elementwise_aggregation_used": False,
        "unselected_correlated_tensors_used_as_model_inputs": False,
        "all_correlated_tensor_files_hash_validated": True,
        "selected_tensors_aligned_to_rehearsal_references": True,
        "driver_event_tensor_selection_sha256": tensor_selection_sha256,
        "driver_event_tensor_selection_records": tensor_selection_records,
        "driver_event_bag_count": int(len(full_frame)),
        "observed_lap_time_driver_event_bag_count": int(len(observed)),
        "censored_lap_time_driver_event_bag_count": int(
            (~full_frame["lap_time_observed"]).sum()
        ),
        "validated_tensor_count": int(sum(tensor_counts.values())),
        "channel_names": list(dataset.channel_names),
        "distance_bins": int(dataset.distance_bins),
        "static_feature_names": list(STATIC_FEATURE_NAMES),
        "telemetry_normalization": normalization_audit,
        "censored_feature_bags_included_before_outcome_filter": True,
        "feature_fields_from_target_object": [],
        "target_fields_used_only_after_feature_construction": [
            "lap_time_observed",
            "lap_time_seconds",
        ],
        "classification_position_used": False,
        "stage_reached_used": False,
        "qualifying_sector_labels_used": False,
        "fake_quantile_labels_created": False,
        "production_six_output_deep_adapter_used": False,
        "feature_set_sha256": feature_set_sha256,
    }
    return dataset, audit


def _validate_research_config(config: TCNResearchConfig) -> None:
    if int(config.minimum_train_events) < 3:
        raise ValueError("minimum_train_events must be at least three")
    if int(config.maximum_epochs) <= 0:
        raise ValueError("maximum_epochs must be positive")
    if int(config.early_stopping_patience) <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if (
        not math.isfinite(float(config.early_stopping_min_delta_seconds))
        or float(config.early_stopping_min_delta_seconds) < 0.0
    ):
        raise ValueError("early_stopping_min_delta_seconds must be finite and nonnegative")
    if not math.isfinite(float(config.learning_rate)) or float(config.learning_rate) <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(float(config.weight_decay)) or float(config.weight_decay) < 0.0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if (
        not math.isfinite(float(config.huber_beta_seconds))
        or float(config.huber_beta_seconds) <= 0.0
    ):
        raise ValueError("huber_beta_seconds must be finite and positive")
    if int(config.hidden_channels) <= 0 or int(config.head_hidden_dim) <= 0:
        raise ValueError("hidden channel dimensions must be positive")
    if int(config.kernel_size) <= 0 or int(config.kernel_size) % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    dilations = tuple(int(value) for value in config.dilations)
    if not dilations or any(value <= 0 for value in dilations):
        raise ValueError("dilations must be a non-empty sequence of positive integers")
    if dilations != tuple(sorted(set(dilations))):
        raise ValueError("dilations must be strictly increasing and unique")
    if not math.isfinite(float(config.dropout)) or not 0.0 <= float(config.dropout) < 1.0:
        raise ValueError("dropout must be finite and in [0, 1)")
    if (
        not math.isfinite(float(config.max_abs_correction_seconds))
        or float(config.max_abs_correction_seconds) <= 0.0
    ):
        raise ValueError("max_abs_correction_seconds must be finite and positive")
    if (
        not math.isfinite(float(config.minimum_inner_relative_gain_to_select_tcn))
        or not 0.0
        <= float(config.minimum_inner_relative_gain_to_select_tcn)
        < 1.0
    ):
        raise ValueError(
            "minimum_inner_relative_gain_to_select_tcn must be in [0, 1)"
        )
    if str(config.telemetry_input_mode) not in TELEMETRY_INPUT_MODES:
        raise ValueError(
            f"telemetry_input_mode must be one of {TELEMETRY_INPUT_MODES}"
        )


def _architecture_config(
    dataset: TCNBagDataset,
    config: TCNResearchConfig,
) -> DistanceTelemetryResidualTCNConfig:
    return DistanceTelemetryResidualTCNConfig(
        input_channels=len(dataset.channel_names),
        distance_bins=int(dataset.distance_bins),
        static_feature_dim=len(STATIC_FEATURE_NAMES),
        hidden_channels=int(config.hidden_channels),
        kernel_size=int(config.kernel_size),
        dilations=tuple(int(value) for value in config.dilations),
        dropout=float(config.dropout),
        head_hidden_dim=int(config.head_hidden_dim),
        max_abs_correction_seconds=float(config.max_abs_correction_seconds),
    )


def _new_network(
    dataset: TCNBagDataset,
    config: TCNResearchConfig,
    *,
    seed: int,
) -> Any:
    seed_torch(int(seed))
    network = DistanceTelemetryResidualTCN(_architecture_config(dataset, config))
    # Start near, but not exactly at, the zero-correction baseline.  An exactly
    # zero final weight blocks the first gradient from reaching the TCN feature
    # extractor; a tiny fixed-seed weight lets even an early-stopped checkpoint
    # represent a genuinely trained convolution rather than only a learned
    # output bias.
    final_layer = network.head[-1]
    torch.nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
    torch.nn.init.zeros_(final_layer.bias)
    return network.to(torch.device("cpu"))


def _state_sha256(network: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(network.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(value.detach().cpu()).tobytes())
    return digest.hexdigest()


def _parameter_change_l2(
    before: Mapping[str, Any],
    network: Any,
    *,
    excluded_prefixes: Sequence[str] = (),
) -> float:
    squared = 0.0
    for name, value in network.state_dict().items():
        if any(name.startswith(prefix) for prefix in excluded_prefixes):
            continue
        delta = value.detach().cpu().to(torch.float64) - before[name].detach().cpu().to(
            torch.float64
        )
        squared += float(torch.sum(delta * delta).item())
    return float(math.sqrt(squared))


def _indices_for_events(frame: pd.DataFrame, event_keys: Sequence[int]) -> np.ndarray:
    keys = {int(value) for value in event_keys}
    return np.flatnonzero(frame["event_key"].isin(keys).to_numpy())


def _training_tensors(
    dataset: TCNBagDataset,
    indices: np.ndarray,
    config: TCNResearchConfig,
) -> tuple[Any, Any, Any, Any, dict[str, float | int]]:
    rows = dataset.frame.iloc[indices]
    target, target_audit = _driver_relative_training_target(rows)
    bound = float(config.max_abs_correction_seconds)
    unclipped_target = target.copy()
    target = np.clip(target, -bound, bound)
    target_audit = {
        **target_audit,
        "model_output_bound_seconds": bound,
        "pre_model_bound_max_abs_target_seconds": float(
            np.max(np.abs(unclipped_target)) if len(unclipped_target) else 0.0
        ),
        "rows_clipped_to_model_output_bound": int(
            np.sum(np.abs(unclipped_target) > bound)
        ),
        "training_target_matches_model_output_support": True,
    }
    weights = _event_equal_weights(rows)
    weights = weights / float(np.sum(weights))
    telemetry_values = dataset.telemetry[indices]
    if config.telemetry_input_mode == TELEMETRY_INPUT_ZERO_ABLATION:
        telemetry_values = np.zeros_like(telemetry_values)
    return (
        torch.from_numpy(telemetry_values).to(torch.float32),
        torch.from_numpy(dataset.static_features[indices]).to(torch.float32),
        torch.from_numpy(target.astype(np.float32)),
        torch.from_numpy(weights.astype(np.float32)),
        target_audit,
    )


def _optimization_step(
    network: Any,
    optimizer: Any,
    telemetry: Any,
    static: Any,
    target: Any,
    weights: Any,
    *,
    huber_beta_seconds: float,
) -> float:
    network.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = network(telemetry, static)
    row_loss = torch.nn.functional.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
        beta=float(huber_beta_seconds),
    )
    loss = torch.sum(row_loss * weights)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.detach().cpu().item())


def _predict_correction(
    network: Any,
    dataset: TCNBagDataset,
    indices: np.ndarray,
    config: TCNResearchConfig,
) -> np.ndarray:
    telemetry_values = dataset.telemetry[indices]
    if config.telemetry_input_mode == TELEMETRY_INPUT_ZERO_ABLATION:
        telemetry_values = np.zeros_like(telemetry_values)
    network.eval()
    with torch.no_grad():
        prediction = network(
            torch.from_numpy(telemetry_values).to(torch.float32),
            torch.from_numpy(dataset.static_features[indices]).to(torch.float32),
        )
    values = prediction.detach().cpu().numpy().astype(float)
    if values.shape != (len(indices),) or not np.isfinite(values).all():
        raise TelemetryResidualResearchError("TCN produced invalid corrections")
    return values


def _inner_early_stopping(
    dataset: TCNBagDataset,
    *,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    config: TCNResearchConfig,
    seed: int,
) -> dict[str, Any]:
    train = dataset.frame.iloc[train_indices].copy()
    validation = dataset.frame.iloc[validation_indices].copy()
    train_events = sorted(int(value) for value in train["event_key"].unique())
    validation_events = sorted(int(value) for value in validation["event_key"].unique())
    if len(validation_events) != 1 or max(train_events) >= validation_events[0]:
        raise TelemetryResidualResearchError(
            "inner validation must be one complete event strictly after training"
        )
    shift, shift_audit = _source_shift_predictions(train, validation)
    reference = validation["rehearsal_reference_seconds"].to_numpy(dtype=float)
    actual = validation["actual_lap_time_seconds"].to_numpy(dtype=float)
    baseline_lap = reference + shift
    baseline_mae = float(np.mean(np.abs(baseline_lap - actual)))
    train_telemetry, train_static, train_target, weights, target_audit = (
        _training_tensors(dataset, train_indices, config)
    )
    network = _new_network(dataset, config, seed=seed)
    initial_state = copy.deepcopy(network.state_dict())
    parameter_count = trainable_parameter_count(network)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    best_epoch = 0
    best_mae = float("inf")
    best_state: dict[str, Any] | None = None
    best_correction: np.ndarray | None = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(config.maximum_epochs) + 1):
        loss = _optimization_step(
            network,
            optimizer,
            train_telemetry,
            train_static,
            train_target,
            weights,
            huber_beta_seconds=float(config.huber_beta_seconds),
        )
        correction = _predict_correction(
            network, dataset, validation_indices, config
        )
        validation_mae = float(
            np.mean(np.abs(baseline_lap + correction - actual))
        )
        history.append(
            {
                "epoch": int(epoch),
                "event_equal_train_huber_loss": float(loss),
                "validation_mae_seconds": validation_mae,
            }
        )
        if validation_mae < best_mae - float(
            config.early_stopping_min_delta_seconds
        ):
            best_epoch = int(epoch)
            best_mae = validation_mae
            best_state = copy.deepcopy(network.state_dict())
            best_correction = correction.copy()
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(config.early_stopping_patience):
                break
    if best_state is None or best_correction is None or best_epoch <= 0:
        raise TelemetryResidualResearchError(
            "inner TCN training never produced a finite learned checkpoint"
        )
    network.load_state_dict(best_state)
    required_gain = float(config.minimum_inner_relative_gain_to_select_tcn)
    selected = (
        "tcn_driver_correction"
        if best_mae <= baseline_mae * (1.0 - required_gain)
        else "zero_telemetry_correction"
    )
    return {
        "inner_train_event_keys": train_events,
        "inner_validation_event_key": validation_events[0],
        "inner_train_driver_event_count": int(len(train)),
        "inner_validation_driver_event_count": int(len(validation)),
        "source_shift_audit": shift_audit,
        "source_shift_baseline_validation_mae_seconds": baseline_mae,
        "best_tcn_validation_mae_seconds": float(best_mae),
        "best_epoch": int(best_epoch),
        "epochs_executed": int(len(history)),
        "early_stopped": bool(len(history) < int(config.maximum_epochs)),
        "selected_candidate_id": selected,
        "required_relative_gain_to_select_tcn": required_gain,
        "observed_relative_gain_vs_source_shift": float(
            (baseline_mae - best_mae) / baseline_mae
            if baseline_mae > 0.0
            else 0.0
        ),
        "parameter_count": int(parameter_count),
        "initialization_seed": int(seed),
        "training_target_audit": target_audit,
        "best_state_sha256": _state_sha256(network),
        "tcn_feature_extractor_parameter_change_l2": _parameter_change_l2(
            initial_state,
            network,
            excluded_prefixes=("head.2",),
        ),
        "history": history,
        "target_event_used": False,
    }


def _refit_prior_events(
    dataset: TCNBagDataset,
    *,
    train_indices: np.ndarray,
    epochs: int,
    config: TCNResearchConfig,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    telemetry, static, target, weights, target_audit = _training_tensors(
        dataset, train_indices, config
    )
    network = _new_network(dataset, config, seed=seed)
    initial_state = copy.deepcopy(network.state_dict())
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    final_loss = float("nan")
    for _ in range(int(epochs)):
        final_loss = _optimization_step(
            network,
            optimizer,
            telemetry,
            static,
            target,
            weights,
            huber_beta_seconds=float(config.huber_beta_seconds),
        )
    rows = dataset.frame.iloc[train_indices]
    weights_np = _event_equal_weights(rows)
    event_weight_sums = {
        str(int(event_key)): float(
            weights_np[rows["event_key"].to_numpy(dtype=int) == int(event_key)].sum()
        )
        for event_key in sorted(rows["event_key"].unique())
    }
    return network, {
        "training_event_keys": sorted(
            int(value) for value in rows["event_key"].unique()
        ),
        "training_driver_event_count": int(len(rows)),
        "epochs": int(epochs),
        "event_weight_sums": event_weight_sums,
        "final_event_equal_train_huber_loss": final_loss,
        "training_target_audit": target_audit,
        "state_sha256": _state_sha256(network),
        "parameter_count": trainable_parameter_count(network),
        "initialization_seed": int(seed),
        "tcn_feature_extractor_parameter_change_l2": _parameter_change_l2(
            initial_state,
            network,
            excluded_prefixes=("head.2",),
        ),
    }


def _benchmark_summary(
    folds: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    actual = np.asarray(
        [row["actual_lap_time_seconds"] for row in predictions], dtype=float
    )
    summary: dict[str, Any] = {
        "scored_event_count": int(len(folds)),
        "scored_driver_event_count": int(len(predictions)),
    }
    for name in (
        "raw_baseline",
        "source_shift_baseline",
        "tcn_driver_correction",
        "locked_selected_policy",
    ):
        predicted = np.asarray(
            [row[f"{name}_predicted_lap_time_seconds"] for row in predictions],
            dtype=float,
        )
        metrics = _prediction_metrics(actual, predicted)
        event_maes = [float(fold["metrics"][name]["mae_seconds"]) for fold in folds]
        summary[name] = {
            **metrics,
            "event_balanced_mae_seconds": float(np.mean(event_maes)),
            "event_mae_standard_deviation_seconds": float(np.std(event_maes)),
            "events_beating_source_shift_baseline": (
                None
                if name == "source_shift_baseline"
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
    baseline = float(
        summary["source_shift_baseline"]["event_balanced_mae_seconds"]
    )
    for name in ("tcn_driver_correction", "locked_selected_policy"):
        candidate = float(summary[name]["event_balanced_mae_seconds"])
        summary[name]["delta_vs_source_shift_baseline_seconds"] = candidate - baseline
        summary[name]["relative_improvement_vs_source_shift_baseline"] = float(
            (baseline - candidate) / baseline if baseline > 0.0 else 0.0
        )
    paired_deltas = np.asarray(
        [
            float(fold["metrics"]["tcn_driver_correction"]["mae_seconds"])
            - float(fold["metrics"]["source_shift_baseline"]["mae_seconds"])
            for fold in folds
        ],
        dtype=float,
    )
    nonzero = paired_deltas[~np.isclose(paired_deltas, 0.0, atol=1e-15)]
    wins = int(np.sum(nonzero < 0.0))
    losses = int(np.sum(nonzero > 0.0))
    sign_n = int(len(nonzero))
    if sign_n:
        tail = sum(
            math.comb(sign_n, index)
            for index in range(min(wins, losses) + 1)
        ) / float(2**sign_n)
        sign_p = min(1.0, 2.0 * tail)
    else:
        sign_p = 1.0
    rng = np.random.default_rng(PAIRED_EVENT_BOOTSTRAP_SEED)
    if len(paired_deltas):
        draw_indices = rng.integers(
            0,
            len(paired_deltas),
            size=(PAIRED_EVENT_BOOTSTRAP_DRAWS, len(paired_deltas)),
        )
        bootstrap_means = np.mean(paired_deltas[draw_indices], axis=1)
        bootstrap_ci = np.quantile(bootstrap_means, [0.025, 0.975])
        probability_improves = float(np.mean(bootstrap_means < 0.0))
    else:
        bootstrap_ci = np.asarray([float("nan"), float("nan")])
        probability_improves = float("nan")
    summary["paired_event_tcn_vs_source_shift"] = {
        "delta_definition": "tcn_event_mae_minus_source_shift_event_mae",
        "event_keys": [int(fold["target_event_key"]) for fold in folds],
        "event_mae_deltas_seconds": paired_deltas.tolist(),
        "mean_delta_seconds": float(np.mean(paired_deltas)),
        "median_delta_seconds": float(np.median(paired_deltas)),
        "tcn_event_wins": wins,
        "tcn_event_losses": losses,
        "exact_two_sided_sign_test_p_value": float(sign_p),
        "paired_event_bootstrap": {
            "draws": PAIRED_EVENT_BOOTSTRAP_DRAWS,
            "seed": PAIRED_EVENT_BOOTSTRAP_SEED,
            "mean_delta_ci95_seconds": [
                float(bootstrap_ci[0]),
                float(bootstrap_ci[1]),
            ],
            "probability_mean_delta_below_zero": probability_improves,
            "descriptive_only_not_used_for_selection": True,
        },
    }
    return summary


def run_expanding_tcn_benchmark(
    dataset: TCNBagDataset,
    *,
    config: TCNResearchConfig | None = None,
) -> dict[str, Any]:
    cfg = config or TCNResearchConfig()
    _validate_research_config(cfg)
    if not torch_available():
        raise RuntimeError("PyTorch is required for the true TCN benchmark")
    torch.set_num_threads(1)
    events = sorted(int(value) for value in dataset.frame["event_key"].unique())
    if len(events) <= int(cfg.minimum_train_events):
        raise TelemetryResidualResearchError(
            f"need more than {cfg.minimum_train_events} complete events; found {len(events)}"
        )
    folds: list[dict[str, Any]] = []
    flat_predictions: list[dict[str, Any]] = []
    for target_event in events[int(cfg.minimum_train_events) :]:
        prior_events = [value for value in events if value < target_event]
        if len(prior_events) < int(cfg.minimum_train_events):
            raise TelemetryResidualResearchError("outer fold lacks required prior events")
        inner_validation_event = prior_events[-1]
        inner_train_events = prior_events[:-1]
        if len(inner_train_events) < 2:
            raise TelemetryResidualResearchError(
                "nested TCN split requires at least two inner training events"
            )
        inner = _inner_early_stopping(
            dataset,
            train_indices=_indices_for_events(dataset.frame, inner_train_events),
            validation_indices=_indices_for_events(
                dataset.frame, [inner_validation_event]
            ),
            config=cfg,
            seed=int(cfg.seed) + int(target_event) * 2,
        )
        outer_train_indices = _indices_for_events(dataset.frame, prior_events)
        target_indices = _indices_for_events(dataset.frame, [target_event])
        network, fit_audit = _refit_prior_events(
            dataset,
            train_indices=outer_train_indices,
            epochs=int(inner["best_epoch"]),
            config=cfg,
            # The frozen epoch count was selected for this exact deterministic
            # initialization. Reusing it with a different seed silently changes
            # the selected training procedure.
            seed=int(cfg.seed) + int(target_event) * 2,
        )
        train = dataset.frame.iloc[outer_train_indices].copy()
        score = dataset.frame.iloc[target_indices].copy()
        shift, shift_audit = _source_shift_predictions(train, score)
        raw_shift = _raw_shift_prediction(train)
        correction = _predict_correction(
            network, dataset, target_indices, cfg
        )
        reference = score["rehearsal_reference_seconds"].to_numpy(dtype=float)
        actual = score["actual_lap_time_seconds"].to_numpy(dtype=float)
        source_lap = reference + shift
        tcn_lap = source_lap + correction
        selected_tcn = inner["selected_candidate_id"] == "tcn_driver_correction"
        selected_lap = tcn_lap if selected_tcn else source_lap
        model_laps = {
            "raw_baseline": reference + float(raw_shift),
            "source_shift_baseline": source_lap,
            "tcn_driver_correction": tcn_lap,
            "locked_selected_policy": selected_lap,
        }
        prediction_rows: list[dict[str, Any]] = []
        for position, (_, row) in enumerate(score.iterrows()):
            prediction: dict[str, Any] = {
                "event_key": int(target_event),
                "round": int(row["round"]),
                "event_name": str(row["event_name"]),
                "driver_id": str(row["driver_id"]),
                "rehearsal_source": str(row["rehearsal_source"]),
                "bag_sha256": str(row["bag_sha256"]),
                "rehearsal_reference_seconds": float(reference[position]),
                "actual_lap_time_seconds": float(actual[position]),
                "source_shift_predicted_residual_seconds": float(shift[position]),
                "tcn_predicted_driver_correction_seconds": float(correction[position]),
                "locked_selected_candidate_id": str(inner["selected_candidate_id"]),
            }
            for model_name, values in model_laps.items():
                prediction[f"{model_name}_predicted_lap_time_seconds"] = float(
                    values[position]
                )
            prediction["prediction_sha256"] = canonical_sha256(prediction)
            prediction_rows.append(prediction)
        metrics = {
            name: _prediction_metrics(actual, values)
            for name, values in model_laps.items()
        }
        fold: dict[str, Any] = {
            "target_event_key": int(target_event),
            "round": int(score["round"].iloc[0]),
            "event_name": str(score["event_name"].iloc[0]),
            "prior_event_keys": prior_events,
            "inner_validation_event_key": int(inner_validation_event),
            "target_event_used_for_training_or_selection": False,
            "training_driver_event_count": int(len(train)),
            "scored_driver_event_count": int(len(score)),
            "training_tensor_count": int(train["tensor_count_raw"].sum()),
            "scored_tensor_count": int(score["tensor_count_raw"].sum()),
            "inner_selection_and_early_stopping": inner,
            "outer_refit": fit_audit,
            "source_shift_audit": shift_audit,
            "metrics": metrics,
            "predictions": prediction_rows,
            "prediction_set_sha256": canonical_sha256(
                [row["prediction_sha256"] for row in prediction_rows]
            ),
        }
        fold["fold_sha256"] = canonical_sha256(fold)
        folds.append(fold)
        flat_predictions.extend(prediction_rows)

    architecture = _architecture_config(dataset, cfg)
    probe = _new_network(dataset, cfg, seed=int(cfg.seed))
    parameter_count = trainable_parameter_count(probe)
    result: dict[str, Any] = {
        "event_keys": events,
        "warmup_event_keys": events[: int(cfg.minimum_train_events)],
        "scored_event_keys": events[int(cfg.minimum_train_events) :],
        "minimum_train_events": int(cfg.minimum_train_events),
        "folds": folds,
        "predictions": flat_predictions,
        "prediction_set_sha256": canonical_sha256(
            [row["prediction_sha256"] for row in flat_predictions]
        ),
        "summary": _benchmark_summary(folds, flat_predictions),
        "capacity": {
            "independent_event_count": int(len(events)),
            "driver_event_bag_count": int(len(dataset.frame)),
            "validated_correlated_tensor_count": int(dataset.validated_tensor_count),
            "correlated_tensor_count_treated_as_sample_size": False,
            "trainable_scalar_parameter_count": int(parameter_count),
            "statistical_effective_degrees_of_freedom_claimed": False,
            "capacity_proxy": "exact_trainable_scalar_parameter_count",
            "trainable_parameters_per_independent_event": float(
                parameter_count / len(events)
            ),
            "outer_training_driver_event_counts": [
                int(fold["training_driver_event_count"]) for fold in folds
            ],
            "frozen_epoch_counts": [
                int(fold["outer_refit"]["epochs"]) for fold in folds
            ],
        },
        "architecture": asdict(architecture),
        "training_config": asdict(cfg),
        "model_input_ablation": {
            "telemetry_input_mode": str(cfg.telemetry_input_mode),
            "observed_telemetry_values_passed_to_model": bool(
                cfg.telemetry_input_mode == TELEMETRY_INPUT_OBSERVED
            ),
            "telemetry_zeroed_after_target_free_normalization": bool(
                cfg.telemetry_input_mode == TELEMETRY_INPUT_ZERO_ABLATION
            ),
            "static_anchor_features_retained": True,
            "architecture_and_trainable_parameter_count_unchanged": True,
            "telemetry_model_input_sha256": hashlib.sha256(
                (
                    dataset.telemetry
                    if cfg.telemetry_input_mode == TELEMETRY_INPUT_OBSERVED
                    else np.zeros_like(dataset.telemetry)
                ).tobytes()
            ).hexdigest(),
        },
        "runtime": {
            "torch_version": str(torch.__version__),
            "device": "cpu",
            "deterministic_algorithms_requested": True,
            "torch_thread_count": int(torch.get_num_threads()),
        },
    }
    return result


def _tcn_implementation_manifest(repo_root: Path) -> list[dict[str, Any]]:
    implementation_paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name(
            "run_prequal_telemetry_residual_research.py"
        ).resolve(),
        Path(__file__).with_name(
            "run_prequal_telemetry_tcn_sensitivity.py"
        ).resolve(),
        repo_root / "packages/f1/data/providers/telemetry_supervised.py",
        repo_root / "packages/f1/data/providers/telemetry_cache.py",
        repo_root / "packages/f1/models/ultimate_lap_time/deep.py",
    )
    return [
        {
            "path": path.resolve().relative_to(repo_root).as_posix(),
            "sha256": sha256_file(path.resolve()),
            "size_bytes": int(path.resolve().stat().st_size),
        }
        for path in implementation_paths
    ]


def build_research_artifact(
    *,
    manifest_path: Path,
    root: Path,
    config: TCNResearchConfig | None = None,
    generated_at: str | None = None,
    profile_design_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = root.expanduser().resolve()
    design_provenance = normalize_profile_design_provenance(
        profile_design_provenance
    )
    implementation_manifest = _tcn_implementation_manifest(repo_root)
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
        raise TelemetryResidualResearchError("supervised manifest must be an object")
    if sha256_file(source) != source_file_sha256:
        raise TelemetryResidualResearchError(
            "supervised manifest changed while it was being loaded"
        )
    source_validation = validate_prequal_telemetry_supervised_manifest(
        manifest, root=repo_root
    )
    dataset, adapter_audit = build_tcn_bag_dataset(manifest, root=repo_root)
    benchmark = run_expanding_tcn_benchmark(dataset, config=config)
    source_validation_after = validate_prequal_telemetry_supervised_manifest(
        manifest, root=repo_root
    )
    if source_validation_after != source_validation:
        raise TelemetryResidualResearchError(
            "supervised nested input evidence changed during TCN evaluation"
        )
    if sha256_file(source) != source_file_sha256:
        raise TelemetryResidualResearchError(
            "supervised manifest changed during TCN evaluation"
        )
    if _tcn_implementation_manifest(repo_root) != implementation_manifest:
        raise TelemetryResidualResearchError(
            "TCN implementation files changed during evaluation"
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "research_evaluated_not_promotion_eligible",
        "promotion_eligible": False,
        "deployment_changed": False,
        "reason_not_promotion_eligible": (
            f"{source_validation['audit']['event_count']} same-season event groups "
            "permit a bounded chronological TCN "
            "falsification run, but not stable architecture selection, calibration, "
            "or a separate promotion-grade season/regime holdout"
        ),
        "profile_design_provenance": design_provenance,
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
        "implementation_manifest": implementation_manifest,
        "implementation_manifest_sha256": canonical_sha256(
            implementation_manifest
        ),
        "estimand": {
            "truth": "best_legal_grand_prix_qualifying_lap_seconds",
            "prediction": (
                "fastest_rehearsal_lap_seconds + prior_event_source_shift_seconds + "
                "bounded_tcn_driver_relative_correction_seconds"
            ),
            "continuous_fit_condition": "lap_time_observed=true",
            "censored_bags_dropped_from_manifest": False,
        },
        "validation_contract": {
            "outer_split": "strict_complete_event_expanding_window",
            "inner_split": "last_prior_complete_event_for_early_stopping_and_locked_selection",
            "outer_target_use_scope": (
                "current_execution_fold_only; historical profile design is declared "
                "separately"
            ),
            "random_row_split_used": False,
            "tensor_level_split_used": False,
            "same_event_rows_cross_split": False,
            "outer_target_used_for_training": False,
            "outer_target_used_for_early_stopping": False,
            "outer_target_used_for_model_selection": False,
            "current_run_fold_hyperparameter_search_performed": False,
            "current_run_fold_architecture_search_performed": False,
            "prior_outer_results_informed_profile_design": design_provenance[
                "prior_outer_results_informed_profile_design"
            ],
            "same_outer_evaluation_targets_seen_before_profile_freeze": (
                design_provenance[
                    "same_outer_evaluation_targets_seen_before_profile_freeze"
                ]
            ),
            "hyperparameters_tuned_on_outer_targets": design_provenance[
                "hyperparameters_tuned_on_outer_targets"
            ],
            "fixed_small_architecture": True,
            "current_run_zero_correction_selector_uses_only_prior_events": True,
            "event_equal_training_weights": True,
        },
        "adapter_audit": adapter_audit,
        **benchmark,
    }
    payload["artifact_payload_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_payload_sha256"}
    )
    return payload


def _parser(root: Path) -> argparse.ArgumentParser:
    defaults = TCNResearchConfig()
    parser = argparse.ArgumentParser(
        description="Run a bounded event-blocked true TCN telemetry benchmark."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "data/f1/derived/prequal_telemetry_supervised_2026.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--minimum-train-events", type=int, default=defaults.minimum_train_events
    )
    parser.add_argument(
        "--maximum-epochs", type=int, default=defaults.maximum_epochs
    )
    parser.add_argument(
        "--patience", type=int, default=defaults.early_stopping_patience
    )
    parser.add_argument(
        "--early-stopping-min-delta-seconds",
        type=float,
        default=defaults.early_stopping_min_delta_seconds,
    )
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument(
        "--huber-beta-seconds", type=float, default=defaults.huber_beta_seconds
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--hidden-channels", type=int, default=defaults.hidden_channels
    )
    parser.add_argument("--kernel-size", type=int, default=defaults.kernel_size)
    parser.add_argument(
        "--dilations",
        default=",".join(str(value) for value in defaults.dilations),
        help="Comma-separated positive dilation factors.",
    )
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument(
        "--head-hidden-dim", type=int, default=defaults.head_hidden_dim
    )
    parser.add_argument(
        "--max-abs-correction-seconds",
        type=float,
        default=defaults.max_abs_correction_seconds,
    )
    parser.add_argument(
        "--minimum-inner-relative-gain",
        type=float,
        default=defaults.minimum_inner_relative_gain_to_select_tcn,
    )
    parser.add_argument(
        "--telemetry-input-mode",
        choices=TELEMETRY_INPUT_MODES,
        default=defaults.telemetry_input_mode,
    )
    parser.add_argument("--generated-at", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    root = find_repo_root()
    args = _parser(root).parse_args(argv)
    try:
        dilations = tuple(
            int(value.strip())
            for value in str(args.dilations).split(",")
            if value.strip()
        )
    except ValueError as exc:
        raise SystemExit("--dilations must be comma-separated integers") from exc
    config = TCNResearchConfig(
        minimum_train_events=int(args.minimum_train_events),
        maximum_epochs=int(args.maximum_epochs),
        early_stopping_patience=int(args.patience),
        early_stopping_min_delta_seconds=float(
            args.early_stopping_min_delta_seconds
        ),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        huber_beta_seconds=float(args.huber_beta_seconds),
        seed=int(args.seed),
        hidden_channels=int(args.hidden_channels),
        kernel_size=int(args.kernel_size),
        dilations=dilations,
        dropout=float(args.dropout),
        head_hidden_dim=int(args.head_hidden_dim),
        max_abs_correction_seconds=float(args.max_abs_correction_seconds),
        minimum_inner_relative_gain_to_select_tcn=float(
            args.minimum_inner_relative_gain
        ),
        telemetry_input_mode=str(args.telemetry_input_mode),
    )
    payload = build_research_artifact(
        manifest_path=args.manifest,
        root=root,
        config=config,
        generated_at=args.generated_at,
    )
    output = args.output or (
        root
        / "artifacts/backtests/f1/telemetry/prequal_telemetry_true_tcn_research_v2_2026.json"
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
                "status": payload["status"],
                "capacity": payload["capacity"],
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
