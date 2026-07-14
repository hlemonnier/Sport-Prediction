#!/usr/bin/env python3
"""Chronological 2026 backtest for achievable qualifying best-lap estimates."""

from __future__ import annotations
import repo_bootstrap  # noqa: F401

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from packages.f1.domain.weekend import qualifying_elimination_rule
from packages.f1.data.providers.telemetry_cache import (
    audit_telemetry_cache_manifests,
)
from packages.f1.data.providers.telemetry_supervised import (
    SUPERVISED_TELEMETRY_SCHEMA_VERSION,
    validate_prequal_telemetry_supervised_manifest,
)
from packages.f1.features.qualifying_lap import (
    build_quality_aware_rehearsal_features,
    finite_lap_seconds,
)
from packages.f1.models.ultimate_lap_time.achievable import (
    ACTUAL_LAP_COLUMN,
    LATENT_POTENTIAL_ANCHOR_COLUMN,
    Q1_LAP_COLUMN,
    Q2_LAP_COLUMN,
    Q2_VALID_LAP_COLUMN,
    Q3_LAP_COLUMN,
    Q3_VALID_LAP_COLUMN,
    SHARED_QUALIFYING_ENABLE_ROBUST_RESIDUAL,
    SHARED_QUALIFYING_SAMPLE_COUNT,
    SHARED_QUALIFYING_SAMPLE_SEED_BASE,
    build_shared_qualifying_event_forecast,
    calibrate_achievable_best_lap_model,
    fit_achievable_best_lap_model,
    shared_qualifying_forecast_artifact,
    shared_point_predictor_sha256,
    shared_structural_point_predictor_sha256,
)
from packages.f1.models.ultimate_lap_time.schemas import (
    ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT,
    TARGET_CONTRACT_SEMANTICS,
)
from packages.f1.orchestration.non_live_validation import validate_event_partitions
from packages.sports_core.paths import find_repo_root


SCHEMA_VERSION = "f1_best_estimated_lap_shared_latent_v11"
MODEL_NAME = "shared_qualifying_latent_lap_huber_v4"
QUALITY_LOCATION_MODEL_NAME = "shared_qualifying_latent_lap_location_v4"
BASELINE_MODEL_NAME = "achievable_best_lap_rehearsal_shift_v1"
ROUND_PATTERN = re.compile(r"^round_(\d{2})_")
MIN_SUPPORTED_WEEKEND_YEAR = 2024
TELEMETRY_MANIFEST_SCHEMA_VERSION = "f1_prequal_telemetry_cache_v2"
TELEMETRY_READINESS_SCHEMA_VERSION = "f1_prequal_telemetry_cache_audit_v3"
TCN_RESEARCH_EVIDENCE_SCHEMA_VERSION = "f1_prequal_telemetry_true_tcn_research_v2"
TCN_SENSITIVITY_MATRIX_SCHEMA_VERSION = (
    "f1_prequal_telemetry_tcn_sensitivity_matrix_v1"
)
TCN_SENSITIVITY_MATRIX_STATUS = (
    "postdevelopment_descriptive_matrix_complete_not_selection_evidence"
)
TCN_REFERENCE_PROFILE_ID = "d1_optimizer_primary"
TCN_REFERENCE_EVIDENCE_ROLE = (
    "posthoc_exploratory_profile_predeclared_before_durable_matrix_run"
)
TCN_OUTER_TARGET_INFORMED_PROFILE_IDS = [
    "d1_optimizer_primary",
    "d2_lower_capacity_broad_receptive_field",
    "d3_primary_seed_stability",
    "d1_zero_telemetry_static_anchor_sham",
    "d4_posthoc_lr_1e4",
    "d4_posthoc_lr_1e4_zero_telemetry_sham",
]
TCN_SUPERVISED_SOURCE_SCHEMA_VERSION = SUPERVISED_TELEMETRY_SCHEMA_VERSION
MINIMUM_DEEP_TELEMETRY_DRIVERS_PER_EVENT = 18
TELEMETRY_DIAGNOSTIC_MINIMUM_TRAIN_EVENTS = 3
TELEMETRY_DIAGNOSTIC_MINIMUM_EVENTS = TELEMETRY_DIAGNOSTIC_MINIMUM_TRAIN_EVENTS + 1
MINIMUM_INTERVAL_PROMOTION_EVENTS = 4
MINIMUM_INTERVAL_AUDIT_EVENTS = 3
INTERVAL_NOMINAL_COVERAGE = 0.85
INTERVAL_EVENT_BALANCED_COVERAGE_TOLERANCE = 0.05
MINIMUM_PER_EVENT_INTERVAL_COVERAGE = 0.70
MINIMUM_PER_STRATUM_INTERVAL_COVERAGE = 0.75
MAXIMUM_EVENT_BALANCED_WIDTH_RATIO = 1.10
MAXIMUM_PER_EVENT_WIDTH_RATIO = 1.25
MAXIMUM_PER_STRATUM_WIDTH_RATIO = 1.15
REQUIRED_WEEKEND_STRATA = ("standard", "sprint")


def _repo_root() -> Path:
    return find_repo_root(__file__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty(root: Path) -> bool | None:
    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _implementation_paths(root: Path) -> list[Path]:
    return sorted(
        {
            Path(__file__).resolve(),
            (root / "packages/sports_core/paths.py").resolve(),
            (root / "packages/f1/data/providers/telemetry_cache.py").resolve(),
            (root / "packages/f1/data/providers/telemetry_supervised.py").resolve(),
            (root / "packages/f1/features/qualifying_lap.py").resolve(),
            Path(__file__).with_name(
                "run_prequal_telemetry_residual_research.py"
            ).resolve(),
            Path(__file__).with_name(
                "run_prequal_telemetry_tcn_research.py"
            ).resolve(),
            *(path.resolve() for path in (root / "packages/f1/models/ultimate_lap_time").rglob("*.py")),
        }
    )


def _hash_manifest(paths: Sequence[Path], *, root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): _sha256(path) for path in sorted(set(paths))}


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _default_tcn_evidence_path(root: Path, *, year: int) -> Path:
    return (
        root
        / "artifacts/backtests/f1/telemetry"
        / f"prequal_telemetry_true_tcn_research_v2_{int(year)}.json"
    ).resolve()


def _valid_sha256(value: object) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", str(value or "").strip().lower()) is not None


def _validated_tcn_profile_design_provenance(
    value: object,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("TCN profile design provenance is missing")
    required_fields = {
        "design_stage",
        "prior_outer_results_informed_profile_design",
        "same_outer_evaluation_targets_seen_before_profile_freeze",
        "hyperparameters_tuned_on_outer_targets",
        "durable_matrix_results_used_to_select_profile",
        "promotion_eligible_from_profile_design",
    }
    if set(value) != required_fields:
        raise ValueError("TCN profile design provenance fields are invalid")
    provenance = dict(value)
    informed_flags = (
        provenance["prior_outer_results_informed_profile_design"],
        provenance["same_outer_evaluation_targets_seen_before_profile_freeze"],
        provenance["hyperparameters_tuned_on_outer_targets"],
    )
    if any(not isinstance(flag, bool) for flag in informed_flags):
        raise ValueError("TCN profile design provenance flags are invalid")
    if len(set(informed_flags)) != 1:
        raise ValueError(
            "TCN profile design provenance contradicts its outer-target history"
        )
    if provenance["durable_matrix_results_used_to_select_profile"] is not False:
        raise ValueError("TCN durable matrix cannot select a profile")
    if provenance["promotion_eligible_from_profile_design"] is not False:
        raise ValueError("TCN profile design cannot claim promotion eligibility")
    if not str(provenance["design_stage"] or "").strip():
        raise ValueError("TCN profile design stage is missing")
    return provenance


def _bound_repo_path(value: object, *, root: Path, label: str) -> Path:
    text = str(value or "").strip()
    declared = Path(text).expanduser()
    if not text or declared.is_absolute():
        raise ValueError(f"{label} must be a repository-relative path")
    resolved = (root / declared).resolve()
    try:
        canonical = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository") from exc
    if declared.as_posix() != canonical:
        raise ValueError(f"{label} is not canonical repository-relative")
    return resolved


def _load_tcn_sensitivity_matrix_binding(
    payload: Mapping[str, Any],
    *,
    source: Path,
    root: Path,
) -> tuple[dict[str, Any] | None, list[Path]]:
    sensitivity = payload.get("sensitivity_profile")
    if sensitivity is None:
        return None, []
    if not isinstance(sensitivity, Mapping):
        raise ValueError("TCN sensitivity profile binding must be an object")
    design_provenance = _validated_tcn_profile_design_provenance(
        payload.get("profile_design_provenance")
    )
    if design_provenance[
        "prior_outer_results_informed_profile_design"
    ] is not True:
        raise ValueError(
            "TCN sensitivity reference must disclose outer-target-informed design"
        )
    required_flags = {
        "profile_id": TCN_REFERENCE_PROFILE_ID,
        "evidence_role": TCN_REFERENCE_EVIDENCE_ROLE,
        "reference_profile_fixed_before_durable_matrix_execution": True,
        "prior_outer_results_informed_profile_design": True,
        "durable_matrix_results_used_to_select_reference_profile": False,
        "promotion_allowed_from_matrix": False,
        "completed_matrix_required_for_downstream_consumption": True,
    }
    if any(sensitivity.get(key) != value for key, value in required_flags.items()):
        raise ValueError("TCN sensitivity profile binding is not fail-closed")
    if sensitivity.get("profile_design_provenance") != design_provenance:
        raise ValueError(
            "TCN sensitivity binding contradicts profile design provenance"
        )

    plan_path = _bound_repo_path(
        sensitivity.get("plan_path"), root=root, label="TCN sensitivity plan path"
    )
    matrix_path = _bound_repo_path(
        sensitivity.get("matrix_output_path"),
        root=root,
        label="TCN sensitivity matrix path",
    )
    declared_plan_sha = str(sensitivity.get("plan_sha256") or "")
    if not _valid_sha256(declared_plan_sha) or _sha256(plan_path) != declared_plan_sha:
        raise ValueError("TCN sensitivity plan hash mismatch")
    try:
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("completed TCN sensitivity matrix is unavailable") from exc
    if not isinstance(matrix, dict):
        raise ValueError("TCN sensitivity matrix must be an object")
    if matrix.get("schema_version") != TCN_SENSITIVITY_MATRIX_SCHEMA_VERSION:
        raise ValueError("unsupported TCN sensitivity matrix schema")
    if matrix.get("status") != TCN_SENSITIVITY_MATRIX_STATUS:
        raise ValueError("TCN sensitivity matrix is not a completed commit marker")
    if matrix.get("promotion_eligible") is not False or matrix.get(
        "deployment_changed"
    ) is not False:
        raise ValueError("TCN sensitivity matrix is not fail-closed")
    declared_matrix_hash = str(matrix.get("artifact_payload_sha256") or "")
    actual_matrix_hash = _canonical_sha256(
        {
            key: value
            for key, value in matrix.items()
            if key != "artifact_payload_sha256"
        }
    )
    if not _valid_sha256(declared_matrix_hash) or declared_matrix_hash != actual_matrix_hash:
        raise ValueError("TCN sensitivity matrix payload hash mismatch")

    plan = matrix.get("plan")
    if not isinstance(plan, Mapping):
        raise ValueError("TCN sensitivity matrix plan binding is missing")
    if (
        str(plan.get("path") or "") != plan_path.relative_to(root).as_posix()
        or str(plan.get("sha256") or "") != declared_plan_sha
    ):
        raise ValueError("TCN sensitivity matrix binds a different plan")
    decision = plan.get("decision_policy")
    required_decision = {
        "reference_profile_id": TCN_REFERENCE_PROFILE_ID,
        "reference_profile_fixed_before_durable_matrix_execution": True,
        "profile_design_provenance_declared_per_profile": True,
        "durable_matrix_results_used_to_select_reference_profile": False,
        "promotion_allowed_from_this_matrix": False,
        "posthoc_profiles_never_selection_evidence": True,
    }
    if not isinstance(decision, Mapping) or any(
        decision.get(key) != value for key, value in required_decision.items()
    ):
        raise ValueError("TCN sensitivity matrix decision policy is not honest")
    if decision.get("outer_target_informed_profile_ids") != (
        TCN_OUTER_TARGET_INFORMED_PROFILE_IDS
    ):
        raise ValueError("TCN sensitivity matrix outer-target profile set is invalid")
    execution = matrix.get("execution")
    if not isinstance(execution, Mapping) or any(
        execution.get(key) != value
        for key, value in {
            "profile_design_provenance_declared_per_profile": True,
            "durable_matrix_results_used_to_select_reference_profile": False,
            "reference_profile_id": TCN_REFERENCE_PROFILE_ID,
            "reference_profile_fixed_before_durable_matrix_execution": True,
            "matrix_selects_winner": False,
        }.items()
    ):
        raise ValueError("TCN sensitivity matrix execution semantics are invalid")
    if execution.get("outer_target_informed_profile_ids") != (
        TCN_OUTER_TARGET_INFORMED_PROFILE_IDS
    ):
        raise ValueError("TCN sensitivity execution profile set is invalid")

    implementation = matrix.get("implementation_manifest")
    if not isinstance(implementation, list) or not implementation:
        raise ValueError("TCN sensitivity matrix implementation manifest is missing")
    if matrix.get("implementation_manifest_sha256") != _canonical_sha256(
        implementation
    ):
        raise ValueError("TCN sensitivity matrix implementation hash mismatch")
    implementation_paths: list[Path] = []
    observed_paths: set[str] = set()
    for row in implementation:
        if not isinstance(row, Mapping):
            raise ValueError("TCN sensitivity implementation row is invalid")
        implementation_path = _bound_repo_path(
            row.get("path"), root=root, label="TCN sensitivity implementation path"
        )
        relative = implementation_path.relative_to(root).as_posix()
        if relative in observed_paths:
            raise ValueError("TCN sensitivity implementation paths are duplicated")
        observed_paths.add(relative)
        implementation_paths.append(implementation_path)
        if int(row.get("size_bytes") or -1) != implementation_path.stat().st_size:
            raise ValueError("TCN sensitivity implementation size mismatch")
        if str(row.get("sha256") or "") != _sha256(implementation_path):
            raise ValueError("TCN sensitivity implementation file hash mismatch")
    required_matrix_paths = {
        plan_path.relative_to(root).as_posix(),
        (
            "research/projects/F1/rising_qualification_prediction/Python/"
            "run_prequal_telemetry_tcn_research.py"
        ),
        (
            "research/projects/F1/rising_qualification_prediction/Python/"
            "run_prequal_telemetry_tcn_sensitivity.py"
        ),
    }
    if not required_matrix_paths.issubset(observed_paths):
        raise ValueError("TCN sensitivity matrix omits required implementation paths")

    source_manifest = matrix.get("source_manifest")
    payload_source = payload.get("source_manifest")
    if not isinstance(source_manifest, Mapping) or not isinstance(
        payload_source, Mapping
    ):
        raise ValueError("TCN sensitivity source binding is missing")
    if (
        source_manifest.get("path") != payload_source.get("path")
        or source_manifest.get("sha256") != payload_source.get("sha256")
    ):
        raise ValueError("TCN sensitivity matrix binds a different source manifest")

    profiles = matrix.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError("TCN sensitivity matrix profiles are missing")
    matching = [
        row
        for row in profiles
        if isinstance(row, Mapping)
        and row.get("profile_id") == TCN_REFERENCE_PROFILE_ID
    ]
    if len(matching) != 1:
        raise ValueError("TCN sensitivity matrix reference profile is ambiguous")
    record = matching[0]
    seed_records = [
        row
        for row in profiles
        if isinstance(row, Mapping)
        and row.get("profile_id") == "d3_primary_seed_stability"
    ]
    if len(seed_records) != 1:
        raise ValueError("TCN sensitivity matrix seed control is ambiguous")
    if record.get("evidence_role") != TCN_REFERENCE_EVIDENCE_ROLE:
        raise ValueError("TCN sensitivity reference evidence role is invalid")
    if record.get("output_path") != source.relative_to(root).as_posix():
        raise ValueError("TCN sensitivity matrix points to another profile artifact")
    if record.get("output_sha256") != _sha256(source):
        raise ValueError("TCN sensitivity matrix profile file hash mismatch")
    if int(record.get("output_size_bytes") or -1) != source.stat().st_size:
        raise ValueError("TCN sensitivity matrix profile size mismatch")
    for field in (
        "artifact_payload_sha256",
        "profile_design_provenance",
        "training_config",
        "architecture",
        "capacity",
        "model_input_ablation",
        "summary",
    ):
        if record.get(field) != payload.get(field):
            raise ValueError(f"TCN sensitivity matrix profile {field} mismatch")

    comparisons = matrix.get("comparisons")
    if not isinstance(comparisons, Mapping):
        raise ValueError("TCN sensitivity matrix comparisons are missing")
    sham = comparisons.get("reference_vs_equal_architecture_zero_telemetry_sham")
    seed = comparisons.get("reference_vs_fixed_seed_repeat")
    if not isinstance(sham, Mapping) or not isinstance(seed, Mapping):
        raise ValueError("TCN sensitivity robustness controls are missing")
    seed_summary = seed_records[0].get("summary")
    if not isinstance(seed_summary, Mapping):
        raise ValueError("TCN sensitivity seed-control summary is missing")
    try:
        seed_source_mae = float(
            seed_summary["source_shift_baseline"]["event_balanced_mae_seconds"]
        )
        seed_tcn_mae = float(
            seed_summary["tcn_driver_correction"]["event_balanced_mae_seconds"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("TCN sensitivity seed-control metrics are malformed") from exc
    diagnostics = {
        "status": str(matrix["status"]),
        "matrix_file": {
            "path": matrix_path.relative_to(root).as_posix(),
            "sha256": _sha256(matrix_path),
            "artifact_payload_sha256": declared_matrix_hash,
        },
        "plan_file": {
            "path": plan_path.relative_to(root).as_posix(),
            "sha256": declared_plan_sha,
        },
        "reference_profile_id": TCN_REFERENCE_PROFILE_ID,
        "postdevelopment_descriptive_only": True,
        "telemetry_minus_sham_mean_event_mae_seconds": float(
            sham["mean_delta_seconds"]
        ),
        "fixed_seed_mean_absolute_event_delta_seconds": float(
            seed["mean_absolute_event_delta_seconds"]
        ),
        "fixed_seed_source_shift_event_balanced_mae_seconds": seed_source_mae,
        "fixed_seed_tcn_event_balanced_mae_seconds": seed_tcn_mae,
        "fixed_seed_repeat_improves_source_shift": bool(
            seed_tcn_mae < seed_source_mae
        ),
    }
    return diagnostics, [matrix_path, plan_path, *implementation_paths]


def _load_tcn_research_evidence(
    path: Path,
    *,
    root: Path,
    year: int,
    telemetry_audit: Mapping[str, Any],
    validated_input_files: list[Path] | None = None,
    require_completed_sensitivity_matrix: bool = True,
) -> dict[str, Any]:
    """Validate a completed TCN result and return its decision-safe summary."""

    source = path.expanduser().resolve()
    try:
        relative_path = str(source.relative_to(root))
    except ValueError as exc:
        raise ValueError("TCN evidence must be stored inside the repository") from exc
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid TCN evidence {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("TCN evidence must be a JSON object")
    cache_integrity_blockers = [
        str(value)
        for value in telemetry_audit.get("blockers", [])
        if str(value)
        not in {
            "insufficient_complete_events_for_requested_protocol",
            "insufficient_independent_prequalifying_telemetry_events",
        }
    ]
    if cache_integrity_blockers:
        raise ValueError(
            "TCN evidence cannot bind to a telemetry cache with integrity blockers: "
            f"{cache_integrity_blockers}"
        )
    if str(payload.get("schema_version") or "") != TCN_RESEARCH_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported TCN research evidence schema")
    declared_payload_hash = str(payload.get("artifact_payload_sha256") or "")
    actual_payload_hash = _canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "artifact_payload_sha256"
        }
    )
    if not _valid_sha256(declared_payload_hash) or declared_payload_hash != actual_payload_hash:
        raise ValueError("TCN evidence artifact payload hash mismatch")
    design_provenance = _validated_tcn_profile_design_provenance(
        payload.get("profile_design_provenance")
    )
    sensitivity_diagnostics, sensitivity_input_paths = (
        _load_tcn_sensitivity_matrix_binding(
            payload,
            source=source,
            root=root.expanduser().resolve(),
        )
    )
    if require_completed_sensitivity_matrix and sensitivity_diagnostics is None:
        raise ValueError(
            "TCN readiness evidence requires a completed sensitivity matrix binding"
        )
    implementation_manifest = payload.get("implementation_manifest")
    if not isinstance(implementation_manifest, list) or not implementation_manifest:
        raise ValueError("TCN evidence implementation manifest is missing")
    if str(payload.get("implementation_manifest_sha256") or "") != _canonical_sha256(
        implementation_manifest
    ):
        raise ValueError("TCN evidence implementation manifest hash mismatch")
    required_implementation_paths = {
        "packages/f1/data/providers/telemetry_cache.py",
        "packages/f1/data/providers/telemetry_supervised.py",
        "packages/f1/models/ultimate_lap_time/deep.py",
        (
            "research/projects/F1/rising_qualification_prediction/Python/"
            "run_prequal_telemetry_residual_research.py"
        ),
        (
            "research/projects/F1/rising_qualification_prediction/Python/"
            "run_prequal_telemetry_tcn_research.py"
        ),
    }
    observed_implementation_paths: set[str] = set()
    implementation_evidence_paths: list[Path] = []
    for row in implementation_manifest:
        if not isinstance(row, Mapping):
            raise ValueError("TCN implementation evidence row must be an object")
        path_text = str(row.get("path") or "").strip()
        declared_path = Path(path_text).expanduser()
        if not path_text or declared_path.is_absolute():
            raise ValueError("TCN implementation path must be repository-relative")
        resolved_path = (root / declared_path).resolve()
        try:
            canonical_path = resolved_path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("TCN implementation path escapes the repository") from exc
        if canonical_path != declared_path.as_posix():
            raise ValueError("TCN implementation path is not canonical")
        if canonical_path in observed_implementation_paths:
            raise ValueError("TCN implementation manifest contains duplicate paths")
        observed_implementation_paths.add(canonical_path)
        implementation_evidence_paths.append(resolved_path)
        if int(row.get("size_bytes") or -1) != int(resolved_path.stat().st_size):
            raise ValueError("TCN implementation file size mismatch")
        if str(row.get("sha256") or "") != _sha256(resolved_path):
            raise ValueError("TCN implementation file hash mismatch")
    if not required_implementation_paths.issubset(observed_implementation_paths):
        raise ValueError("TCN implementation manifest omits required code paths")
    if payload.get("promotion_eligible") is not False:
        raise ValueError("TCN evidence must remain explicitly promotion-ineligible")
    if payload.get("deployment_changed") is not False:
        raise ValueError("TCN research evidence cannot claim a deployment change")
    if str(payload.get("status") or "") != "research_evaluated_not_promotion_eligible":
        raise ValueError("TCN evidence has an unsupported evaluation status")

    source_manifest = payload.get("source_manifest")
    if not isinstance(source_manifest, Mapping):
        raise ValueError("TCN evidence source_manifest is missing")
    if (
        str(source_manifest.get("schema_version") or "")
        != TCN_SUPERVISED_SOURCE_SCHEMA_VERSION
    ):
        raise ValueError("TCN evidence uses an unsupported supervised source schema")
    for field in (
        "sha256",
        "bag_set_sha256",
        "feature_input_manifest_sha256",
        "target_input_manifest_sha256",
    ):
        if not _valid_sha256(source_manifest.get(field)):
            raise ValueError(f"TCN source manifest {field} is invalid")
    source_manifest_path_text = str(source_manifest.get("path") or "").strip()
    if not source_manifest_path_text:
        raise ValueError("TCN source manifest path is missing")
    declared_source_path = Path(source_manifest_path_text).expanduser()
    if declared_source_path.is_absolute():
        raise ValueError("TCN source manifest path must be repository-relative")
    repo_root = root.expanduser().resolve()
    resolved_source_manifest = (repo_root / declared_source_path).resolve()
    try:
        canonical_source_path = resolved_source_manifest.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise ValueError("TCN source manifest path escapes the repository") from exc
    if declared_source_path.as_posix() != canonical_source_path:
        raise ValueError("TCN source manifest path is not canonical repository-relative")
    try:
        actual_source_sha = _sha256(resolved_source_manifest)
        supervised_manifest = json.loads(
            resolved_source_manifest.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"invalid TCN source manifest {resolved_source_manifest}: {exc}"
        ) from exc
    if not isinstance(supervised_manifest, dict):
        raise ValueError("TCN source manifest must be a JSON object")
    if _sha256(resolved_source_manifest) != actual_source_sha:
        raise ValueError("TCN source manifest changed while it was being loaded")
    if str(source_manifest["sha256"]) != actual_source_sha:
        raise ValueError("TCN source manifest external SHA-256 mismatch")
    try:
        source_validation = validate_prequal_telemetry_supervised_manifest(
            supervised_manifest, root=repo_root
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"TCN source manifest provenance validation failed: {exc}") from exc
    if _sha256(resolved_source_manifest) != actual_source_sha:
        raise ValueError("TCN source manifest changed during provenance validation")
    source_hash_fields = (
        "schema_version",
        "bag_set_sha256",
        "feature_input_manifest_sha256",
        "target_input_manifest_sha256",
    )
    mismatched_source_fields = [
        field
        for field in source_hash_fields
        if str(source_manifest.get(field)) != str(source_validation.get(field))
    ]
    if mismatched_source_fields:
        raise ValueError(
            "TCN source manifest binding mismatch: "
            f"{mismatched_source_fields}"
        )
    nested_source_paths: list[Path] = []
    for field in ("feature_input_files", "target_input_files"):
        rows = supervised_manifest.get(field)
        if not isinstance(rows, list):
            raise ValueError(f"TCN source manifest {field} is missing")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"TCN source manifest {field} row is invalid")
            nested_path = (repo_root / str(row.get("path") or "")).resolve()
            try:
                nested_path.relative_to(repo_root)
            except ValueError as exc:
                raise ValueError("TCN nested source path escapes repository") from exc
            nested_source_paths.append(nested_path)
    source_audit = source_validation["audit"]
    expected_source_counts = {
        "event_count": int(telemetry_audit.get("event_count") or 0),
        "driver_event_bag_count": int(
            telemetry_audit.get("driver_event_count") or 0
        ),
        "validated_tensor_count": int(
            telemetry_audit.get("validated_tensor_count") or 0
        ),
    }
    if any(
        int(source_audit.get(field) or -1) != expected
        for field, expected in expected_source_counts.items()
    ):
        raise ValueError(
            "TCN source manifest counts do not match the revalidated telemetry cache"
        )

    validation = payload.get("validation_contract")
    if not isinstance(validation, Mapping):
        raise ValueError("TCN validation contract is missing")
    required_validation_flags = {
        "random_row_split_used": False,
        "tensor_level_split_used": False,
        "same_event_rows_cross_split": False,
        "outer_target_used_for_training": False,
        "outer_target_used_for_early_stopping": False,
        "outer_target_used_for_model_selection": False,
        "current_run_fold_hyperparameter_search_performed": False,
        "current_run_fold_architecture_search_performed": False,
        "fixed_small_architecture": True,
        "current_run_zero_correction_selector_uses_only_prior_events": True,
        "event_equal_training_weights": True,
    }
    mismatched_flags = [
        field
        for field, expected in required_validation_flags.items()
        if validation.get(field) is not expected
    ]
    if mismatched_flags:
        raise ValueError(
            f"TCN evidence violates its event protocol: {mismatched_flags}"
        )
    for field in (
        "prior_outer_results_informed_profile_design",
        "same_outer_evaluation_targets_seen_before_profile_freeze",
        "hyperparameters_tuned_on_outer_targets",
    ):
        if validation.get(field) is not design_provenance[field]:
            raise ValueError(
                "TCN validation contract contradicts profile design provenance"
            )
    if str(validation.get("outer_target_use_scope") or "") != (
        "current_execution_fold_only; historical profile design is declared "
        "separately"
    ):
        raise ValueError("TCN outer-target use scope is ambiguous")
    if str(validation.get("outer_split") or "") != "strict_complete_event_expanding_window":
        raise ValueError("TCN evidence outer split is not a complete-event expanding window")

    try:
        event_keys = [int(value) for value in payload["event_keys"]]
        warmup_event_keys = [int(value) for value in payload["warmup_event_keys"]]
        scored_event_keys = [int(value) for value in payload["scored_event_keys"]]
        minimum_train_events = int(payload["minimum_train_events"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("TCN evidence event protocol is malformed") from exc
    if event_keys != sorted(set(event_keys)) or any(
        event_key // 100 != int(year) for event_key in event_keys
    ):
        raise ValueError("TCN evidence event keys are not a unique target-year chronology")
    live_event_keys = [
        int(value) for value in telemetry_audit.get("complete_event_keys", [])
    ]
    if event_keys != live_event_keys:
        raise ValueError("TCN evidence event keys do not match the revalidated telemetry cache")
    if event_keys != [int(value) for value in source_validation["event_keys"]]:
        raise ValueError("TCN evidence event keys do not match its source manifest")
    if minimum_train_events < TELEMETRY_DIAGNOSTIC_MINIMUM_TRAIN_EVENTS:
        raise ValueError("TCN evidence warmup is below the bounded protocol minimum")
    if warmup_event_keys != event_keys[:minimum_train_events]:
        raise ValueError("TCN evidence warmup events are not the chronological prefix")
    if scored_event_keys != event_keys[minimum_train_events:]:
        raise ValueError("TCN evidence scored events are not the untouched suffix")

    capacity = payload.get("capacity")
    if not isinstance(capacity, Mapping):
        raise ValueError("TCN evidence capacity audit is missing")
    expected_counts = {
        "independent_event_count": int(telemetry_audit.get("event_count") or 0),
        "driver_event_bag_count": int(telemetry_audit.get("driver_event_count") or 0),
        "validated_correlated_tensor_count": int(
            telemetry_audit.get("validated_tensor_count") or 0
        ),
    }
    if any(int(capacity.get(field) or -1) != expected for field, expected in expected_counts.items()):
        raise ValueError("TCN evidence sample counts do not match the live telemetry cache")
    if capacity.get("correlated_tensor_count_treated_as_sample_size") is not False:
        raise ValueError("TCN evidence incorrectly treats correlated tensors as samples")
    if capacity.get("statistical_effective_degrees_of_freedom_claimed") is not False:
        raise ValueError("TCN evidence makes an unsupported effective-capacity claim")

    folds = payload.get("folds")
    predictions = payload.get("predictions")
    if not isinstance(folds, list) or not isinstance(predictions, list):
        raise ValueError("TCN evidence folds or predictions are missing")
    if len(folds) != len(scored_event_keys):
        raise ValueError("TCN evidence fold count does not match scored events")
    flattened_prediction_hashes: list[str] = []
    selected_candidate_ids: list[str] = []
    recomputed_event_maes: dict[str, list[float]] = {
        "source_shift_baseline": [],
        "tcn_driver_correction": [],
        "locked_selected_policy": [],
    }
    for fold, target_event in zip(folds, scored_event_keys):
        if not isinstance(fold, Mapping):
            raise ValueError("TCN evidence fold must be an object")
        if int(fold.get("target_event_key") or -1) != target_event:
            raise ValueError("TCN fold target does not match scored chronology")
        expected_prior = [value for value in event_keys if value < target_event]
        if [int(value) for value in fold.get("prior_event_keys", [])] != expected_prior:
            raise ValueError("TCN fold does not use every and only prior events")
        if fold.get("target_event_used_for_training_or_selection") is not False:
            raise ValueError("TCN fold used its outer target for training or selection")
        inner = fold.get("inner_selection_and_early_stopping")
        if not isinstance(inner, Mapping):
            raise ValueError("TCN fold inner selection audit is missing")
        if int(inner.get("inner_validation_event_key") or -1) != expected_prior[-1]:
            raise ValueError("TCN inner validation is not the last prior complete event")
        if [int(value) for value in inner.get("inner_train_event_keys", [])] != expected_prior[:-1]:
            raise ValueError("TCN inner training events are not strictly before validation")
        if inner.get("target_event_used") is not False:
            raise ValueError("TCN inner selector claims target-event access")
        selected_candidate_id = str(inner.get("selected_candidate_id") or "")
        if selected_candidate_id not in {
            "zero_telemetry_correction",
            "tcn_driver_correction",
        }:
            raise ValueError("TCN inner selector chose an unsupported candidate")
        selected_candidate_ids.append(selected_candidate_id)
        fold_predictions = fold.get("predictions")
        if not isinstance(fold_predictions, list):
            raise ValueError("TCN fold predictions are missing")
        fold_hashes: list[str] = []
        for row in fold_predictions:
            if not isinstance(row, Mapping):
                raise ValueError("TCN prediction row must be an object")
            declared_row_hash = str(row.get("prediction_sha256") or "")
            actual_row_hash = _canonical_sha256(
                {key: value for key, value in row.items() if key != "prediction_sha256"}
            )
            if declared_row_hash != actual_row_hash:
                raise ValueError("TCN prediction row hash mismatch")
            try:
                source_prediction = float(
                    row["source_shift_baseline_predicted_lap_time_seconds"]
                )
                tcn_prediction = float(
                    row["tcn_driver_correction_predicted_lap_time_seconds"]
                )
                selected_prediction = float(
                    row["locked_selected_policy_predicted_lap_time_seconds"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("TCN selected-policy prediction is malformed") from exc
            expected_selected_prediction = (
                tcn_prediction
                if selected_candidate_id == "tcn_driver_correction"
                else source_prediction
            )
            if not np.isclose(
                selected_prediction,
                expected_selected_prediction,
                atol=1e-12,
                rtol=0.0,
            ):
                raise ValueError(
                    "TCN selected-policy prediction does not match its locked selector"
                )
            if str(row.get("locked_selected_candidate_id") or "") != (
                selected_candidate_id
            ):
                raise ValueError(
                    "TCN prediction row selector does not match its fold selector"
                )
            fold_hashes.append(declared_row_hash)
        try:
            actual_laps = np.asarray(
                [float(row["actual_lap_time_seconds"]) for row in fold_predictions],
                dtype=float,
            )
            for model_name in recomputed_event_maes:
                predicted_laps = np.asarray(
                    [
                        float(row[f"{model_name}_predicted_lap_time_seconds"])
                        for row in fold_predictions
                    ],
                    dtype=float,
                )
                event_mae = float(np.mean(np.abs(predicted_laps - actual_laps)))
                declared_event_mae = float(
                    fold["metrics"][model_name]["mae_seconds"]
                )
                if not np.isclose(
                    event_mae, declared_event_mae, atol=1e-12, rtol=0.0
                ):
                    raise ValueError(
                        f"TCN fold {model_name} metric does not match predictions"
                    )
                recomputed_event_maes[model_name].append(event_mae)
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("TCN fold"):
                raise
            raise ValueError("TCN fold prediction metrics are malformed") from exc
        if str(fold.get("prediction_set_sha256") or "") != _canonical_sha256(fold_hashes):
            raise ValueError("TCN fold prediction-set hash mismatch")
        declared_fold_hash = str(fold.get("fold_sha256") or "")
        actual_fold_hash = _canonical_sha256(
            {key: value for key, value in fold.items() if key != "fold_sha256"}
        )
        if declared_fold_hash != actual_fold_hash:
            raise ValueError("TCN fold hash mismatch")
        flattened_prediction_hashes.extend(fold_hashes)
    if [str(row.get("prediction_sha256") or "") for row in predictions] != flattened_prediction_hashes:
        raise ValueError("TCN flat predictions do not match the chronological folds")
    if str(payload.get("prediction_set_sha256") or "") != _canonical_sha256(
        flattened_prediction_hashes
    ):
        raise ValueError("TCN global prediction-set hash mismatch")

    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("TCN evidence summary is missing")
    source_summary = summary.get("source_shift_baseline")
    tcn_summary = summary.get("tcn_driver_correction")
    locked_summary = summary.get("locked_selected_policy")
    if not all(isinstance(value, Mapping) for value in (source_summary, tcn_summary, locked_summary)):
        raise ValueError("TCN evidence model summaries are missing")
    source_mae = float(source_summary["event_balanced_mae_seconds"])
    tcn_mae = float(tcn_summary["event_balanced_mae_seconds"])
    locked_mae = float(locked_summary["event_balanced_mae_seconds"])
    delta = float(tcn_summary["delta_vs_source_shift_baseline_seconds"])
    relative_improvement = float(
        tcn_summary["relative_improvement_vs_source_shift_baseline"]
    )
    expected_delta = tcn_mae - source_mae
    expected_relative = (source_mae - tcn_mae) / source_mae
    if not all(np.isfinite(value) for value in (source_mae, tcn_mae, locked_mae, delta, relative_improvement)):
        raise ValueError("TCN evidence contains non-finite headline metrics")
    if not np.isclose(delta, expected_delta, atol=1e-12, rtol=0.0):
        raise ValueError("TCN evidence delta does not match its MAEs")
    if not np.isclose(relative_improvement, expected_relative, atol=1e-12, rtol=0.0):
        raise ValueError("TCN evidence relative improvement does not match its MAEs")
    recomputed_headline = {
        model_name: float(np.mean(values))
        for model_name, values in recomputed_event_maes.items()
    }
    declared_headline = {
        "source_shift_baseline": source_mae,
        "tcn_driver_correction": tcn_mae,
        "locked_selected_policy": locked_mae,
    }
    if any(
        not np.isclose(
            recomputed_headline[model_name], declared, atol=1e-12, rtol=0.0
        )
        for model_name, declared in declared_headline.items()
    ):
        raise ValueError("TCN headline MAEs do not match complete-event predictions")
    if len(recomputed_event_maes["tcn_driver_correction"]) != len(
        recomputed_event_maes["source_shift_baseline"]
    ):
        raise ValueError("TCN and source event metrics are not aligned")
    recomputed_tcn_wins = int(
        sum(
            tcn_value < source_value
            for tcn_value, source_value in zip(
                recomputed_event_maes["tcn_driver_correction"],
                recomputed_event_maes["source_shift_baseline"],
            )
        )
    )
    if int(tcn_summary["events_beating_source_shift_baseline"]) != recomputed_tcn_wins:
        raise ValueError("TCN event-win count does not match fold predictions")
    selected_zero_count = sum(
        candidate == "zero_telemetry_correction"
        for candidate in selected_candidate_ids
    )
    selected_tcn_count = len(scored_event_keys) - selected_zero_count
    tcn_improves = bool(tcn_mae < source_mae)
    selected_policy_improves = bool(locked_mae < source_mae)
    reference_gain_seconds = source_mae - tcn_mae
    parameter_stable: bool | None = None
    if sensitivity_diagnostics is not None:
        parameter_stable = bool(
            sensitivity_diagnostics["fixed_seed_repeat_improves_source_shift"]
            and reference_gain_seconds
            > sensitivity_diagnostics[
                "fixed_seed_mean_absolute_event_delta_seconds"
            ]
        )
        evaluation_status = (
            "evaluated_postdevelopment_descriptive_gain_not_promotion_eligible"
            if tcn_improves and parameter_stable
            else "evaluated_parameter_sensitive_inconclusive"
        )
    else:
        evaluation_status = (
            "evaluated_improving_not_promotion_eligible"
            if tcn_improves
            else "evaluated_rejected_no_incremental_gain"
        )
    promotion_blockers = [
        "no_independent_post_development_season_or_regime_holdout",
        "insufficient_independent_events_for_promotion_grade_tcn_evidence",
    ]
    if design_provenance[
        "prior_outer_results_informed_profile_design"
    ]:
        promotion_blockers.insert(
            0,
            "profile_hyperparameters_tuned_after_same_outer_targets_were_observed",
        )
    if not tcn_improves:
        promotion_blockers.insert(
            0, "tcn_failed_incremental_mae_gate_vs_source_shift"
        )
    if sensitivity_diagnostics is not None:
        promotion_blockers.insert(
            0, "postdevelopment_profile_matrix_is_descriptive_not_selection_evidence"
        )
        if not sensitivity_diagnostics[
            "fixed_seed_repeat_improves_source_shift"
        ]:
            promotion_blockers.insert(
                1, "fixed_seed_repeat_failed_incremental_gain"
            )
        if reference_gain_seconds <= sensitivity_diagnostics[
            "fixed_seed_mean_absolute_event_delta_seconds"
        ]:
            promotion_blockers.insert(
                1, "reference_gain_smaller_than_fixed_seed_variation"
            )
        promotion_blockers.append(
            "telemetry_vs_equal_architecture_sham_is_descriptive_not_confirmatory"
        )
    if selected_zero_count == len(scored_event_keys):
        promotion_blockers.insert(
            1,
            "locked_inner_selector_retained_zero_telemetry_correction_in_all_folds",
        )
    if validated_input_files is not None:
        validated_input_files.extend(
            [
                source,
                resolved_source_manifest,
                *implementation_evidence_paths,
                *nested_source_paths,
                *sensitivity_input_paths,
            ]
        )
    return {
        "status": evaluation_status,
        "evidence_file": {
            "path": relative_path,
            "sha256": _sha256(source),
            "artifact_payload_sha256": declared_payload_hash,
        },
        "source_manifest": {
            "path": canonical_source_path,
            "schema_version": str(source_manifest["schema_version"]),
            "sha256": str(source_manifest["sha256"]),
            "bag_set_sha256": str(source_manifest["bag_set_sha256"]),
            "feature_input_manifest_sha256": str(
                source_manifest["feature_input_manifest_sha256"]
            ),
            "target_input_manifest_sha256": str(
                source_manifest["target_input_manifest_sha256"]
            ),
        },
        "event_protocol": {
            "event_keys": event_keys,
            "warmup_event_keys": warmup_event_keys,
            "scored_event_keys": scored_event_keys,
            "scored_event_count": len(scored_event_keys),
            "strict_complete_event_expanding_window": True,
            "target_event_used_for_training_or_selection": False,
        },
        "capacity": {
            "independent_event_count": int(capacity["independent_event_count"]),
            "driver_event_bag_count": int(capacity["driver_event_bag_count"]),
            "validated_correlated_tensor_count": int(
                capacity["validated_correlated_tensor_count"]
            ),
            "trainable_scalar_parameter_count": int(
                capacity["trainable_scalar_parameter_count"]
            ),
        },
        "profile_design_provenance": design_provenance,
        "result": {
            "source_shift_event_balanced_mae_seconds": source_mae,
            "tcn_event_balanced_mae_seconds": tcn_mae,
            "tcn_delta_vs_source_shift_seconds": delta,
            "tcn_relative_improvement_vs_source_shift": relative_improvement,
            "tcn_events_beating_source_shift": int(
                tcn_summary["events_beating_source_shift_baseline"]
            ),
            "scored_event_count": len(scored_event_keys),
            "locked_selected_policy_event_balanced_mae_seconds": locked_mae,
            "locked_zero_correction_fold_count": int(selected_zero_count),
            "locked_tcn_selected_fold_count": int(selected_tcn_count),
            "tcn_improves_source_shift": tcn_improves,
            "locked_selected_policy_improves_source_shift": (
                selected_policy_improves
            ),
            "parameter_stable_across_fixed_seed_repeat": parameter_stable,
        },
        "sensitivity_matrix": sensitivity_diagnostics,
        "promotion_ready": False,
        "promotion_blockers": promotion_blockers,
    }


def _telemetry_readiness_audit(
    *,
    root: Path,
    year: int,
    minimum_independent_events: int = TELEMETRY_DIAGNOSTIC_MINIMUM_EVENTS,
    minimum_drivers_per_event: int = MINIMUM_DEEP_TELEMETRY_DRIVERS_PER_EVENT,
) -> tuple[dict[str, Any], list[Path]]:
    """Revalidate telemetry for a concrete event-disjoint research protocol."""

    telemetry_root = root / "data/f1/telemetry/pre_qualifying" / str(int(year))
    manifest_paths = sorted(telemetry_root.glob("round_*/telemetry_manifest.json"))
    manifest_evidence: list[dict[str, Any]] = []
    records: list[Mapping[str, Any]] = []
    tensor_paths: list[Path] = []
    for manifest_path in manifest_paths:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(payload.get("schema_version") or "") != TELEMETRY_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported telemetry manifest schema: {manifest_path}")
        event_key = int(payload["event_key"])
        manifest_year = int(payload.get("year", event_key // 100))
        if manifest_year != int(year) or event_key // 100 != int(year):
            raise ValueError(
                f"telemetry manifest {manifest_path} does not belong to {int(year)}"
            )
        feature_records = payload.get("feature_records", [])
        if not isinstance(feature_records, list) or not all(
            isinstance(record, Mapping) for record in feature_records
        ):
            raise ValueError(f"telemetry manifest has invalid feature records: {manifest_path}")
        manifest_evidence.append(
            {
                "path": str(manifest_path.relative_to(root)),
                "sha256": _sha256(manifest_path),
                "schema_version": TELEMETRY_MANIFEST_SCHEMA_VERSION,
                "event_key": event_key,
                "qualifying_start_utc": str(
                    payload.get("qualifying_start_utc") or ""
                ),
                "feature_record_count": len(feature_records),
                "rejected_feature_record_count": len(
                    payload.get("rejected_feature_records", [])
                ),
            }
        )
        records.extend(feature_records)
        for record in feature_records:
            path_text = str(record.get("telemetry_path") or "").strip()
            if not path_text:
                continue
            path = Path(path_text)
            resolved = (path if path.is_absolute() else root / path).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"telemetry input must be inside the repository: {resolved}"
                ) from exc
            tensor_paths.append(resolved)

    audit = audit_telemetry_cache_manifests(
        records,
        root=root,
        minimum_independent_events=int(minimum_independent_events),
        minimum_drivers_per_event=int(minimum_drivers_per_event),
    )
    # Missing tensors are represented by the provider's blocker/count, but a
    # nonexistent file cannot have a content digest in the input manifest.
    existing_tensor_paths = [path for path in tensor_paths if path.is_file()]
    accessed_paths = sorted(set([*manifest_paths, *existing_tensor_paths]))
    accessed_manifest = _hash_manifest(accessed_paths, root=root)
    return (
        {
            "schema_version": TELEMETRY_READINESS_SCHEMA_VERSION,
            "year": int(year),
            "source": "live_local_manifest_and_tensor_revalidation",
            "manifests": manifest_evidence,
            "manifest_set_sha256": _canonical_sha256(manifest_evidence),
            "accessed_manifest_file_count": len(manifest_paths),
            "accessed_tensor_file_count": len(set(existing_tensor_paths)),
            "accessed_input_file_count": len(accessed_paths),
            "accessed_input_manifest_sha256": _canonical_sha256(accessed_manifest),
            "audit": audit.to_payload(),
        },
        accessed_paths,
    )


def _deep_model_readiness_decision(
    telemetry_audit: Mapping[str, Any],
    *,
    tcn_runtime_available: bool | None = None,
    tcn_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    telemetry_blockers = [str(value) for value in telemetry_audit.get("blockers", [])]
    sample_blockers = {
        "insufficient_complete_events_for_requested_protocol",
        # Read old cache artifacts without preserving their old semantics.
        "insufficient_independent_prequalifying_telemetry_events",
    }
    integrity_blockers = [
        blocker
        for blocker in telemetry_blockers
        if blocker not in sample_blockers
    ]
    independent_events = int(telemetry_audit.get("event_count") or 0)
    protocol_minimum = int(
        telemetry_audit.get("minimum_independent_events")
        or TELEMETRY_DIAGNOSTIC_MINIMUM_EVENTS
    )
    protocol_ready = bool(
        telemetry_audit.get(
            "ready_for_requested_event_protocol",
            telemetry_audit.get("ready_for_deep_model", False),
        )
    )
    bounded_research_ready = bool(
        not integrity_blockers
        and independent_events >= protocol_minimum
        and independent_events > TELEMETRY_DIAGNOSTIC_MINIMUM_TRAIN_EVENTS
    )
    if tcn_runtime_available is None:
        # The bounded TCN runner and its focused suite execute on the project's
        # supported Python 3.9 runtime.  The previous 3.10 gate was historical
        # scaffolding, not a requirement of the implemented model.
        tcn_runtime_available = bool(
            sys.version_info >= (3, 9) and importlib.util.find_spec("torch") is not None
        )
    tcn_research_ready = bool(bounded_research_ready and tcn_runtime_available)
    blockers = list(integrity_blockers)
    tcn_evaluated = tcn_evidence is not None
    if tcn_evaluated:
        blockers.extend(
            str(value) for value in tcn_evidence.get("promotion_blockers", [])
        )
    elif bounded_research_ready:
        blockers.append(
            "true_tcn_not_yet_evaluated_under_event_disjoint_protocol"
            if tcn_runtime_available
            else "true_tcn_runtime_dependency_unavailable"
        )
        blockers.append("no_future_locked_event_after_sequence_model_development")
    else:
        blockers.append("insufficient_complete_events_for_event_disjoint_diagnostic")
    return {
        "deep_model_evaluation_status": (
            str(tcn_evidence["status"])
            if tcn_evaluated
            else (
                "bounded_sequence_and_tcn_research_evaluable_now_not_promotion_ready"
                if tcn_research_ready
                else (
                    "regularized_sequence_research_evaluable_tcn_runtime_unavailable_"
                    "not_promotion_ready"
                    if bounded_research_ready
                    else "cache_not_ready_for_event_disjoint_sequence_diagnostic"
                )
            )
        ),
        "telemetry_model_tiers": {
            "regularized_temporal_summary": {
                "ready": bounded_research_ready,
                "minimum_independent_events": protocol_minimum,
                "minimum_drivers_per_event": int(
                    telemetry_audit.get("minimum_drivers_per_event")
                    or MINIMUM_DEEP_TELEMETRY_DRIVERS_PER_EVENT
                ),
                "partition_rationale": (
                    f"{TELEMETRY_DIAGNOSTIC_MINIMUM_TRAIN_EVENTS} initial fit events "
                    "plus at least one strictly later held-out event"
                ),
                "evaluation_owner": "prequal_telemetry_residual_research_harness",
                "status": (
                    "ready_for_event_blocked_sequence_research"
                    if bounded_research_ready
                    else "insufficient_or_invalid_cache_for_sequence_research"
                ),
            },
            "bounded_supervised_tcn_research": {
                "ready": bool(tcn_evaluated or tcn_research_ready),
                "data_protocol_ready": bounded_research_ready,
                "runtime_dependency_available": bool(tcn_runtime_available),
                "evaluated": bool(tcn_evaluated),
                "fixed_capacity_event_threshold": None,
                "event_requirement_semantics": (
                    "split_feasibility_only; capacity must be inferred from event-disjoint "
                    "learning traces and regularized effective degrees of freedom"
                ),
                "minimum_drivers_per_event": int(
                    telemetry_audit.get("minimum_drivers_per_event")
                    or MINIMUM_DEEP_TELEMETRY_DRIVERS_PER_EVENT
                ),
                "promotion_ready": False,
                "status": (
                    str(tcn_evidence["status"])
                    if tcn_evaluated
                    else (
                        "research_evaluable_not_yet_run"
                        if tcn_research_ready
                        else (
                            "data_ready_runtime_dependency_unavailable"
                            if bounded_research_ready
                            else "cache_not_ready_for_bounded_tcn_research"
                        )
                    )
                ),
                "evaluation_evidence": (
                    dict(tcn_evidence) if tcn_evaluated else None
                ),
            },
        },
        "cache_integrity_ready": not integrity_blockers,
        "requested_event_protocol_ready": protocol_ready,
        "fixed_twenty_event_capacity_gate_used": False,
        "deep_model_telemetry_blockers": telemetry_blockers,
        "deep_model_blockers": blockers,
        "tcn_evidence": dict(tcn_evidence) if tcn_evaluated else None,
    }


def _assert_manifest_unchanged(
    before: dict[str, str],
    *,
    root: Path,
    label: str,
) -> None:
    after = {name: _sha256(root / name) for name in before}
    changed = sorted(name for name, digest in before.items() if after[name] != digest)
    if changed:
        raise RuntimeError(f"{label} changed during evaluation: {changed}")


def _lap_seconds(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    unresolved = numeric.isna()
    if unresolved.any():
        timedeltas = pd.to_timedelta(values.where(unresolved), errors="coerce").dt.total_seconds()
        numeric = numeric.fillna(timedeltas)
    return pd.to_numeric(numeric, errors="coerce").astype(float)


def _bool_series(values: pd.Series, *, default: bool) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(default).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapped = normalized.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "false": False,
            "0": False,
            "no": False,
            "nan": default,
            "none": default,
            "": default,
        }
    )
    return mapped.fillna(default).astype(bool)


def _clean_driver_best_laps(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    required = {"Driver", "LapTime"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing lap columns: {missing}")
    # A driver with exactly one finite clean lap is valid evidence.  Do not
    # apply a variance/MAD/sample-count gate here.
    seconds = finite_lap_seconds(frame["LapTime"])
    valid = seconds.between(40.0, 180.0) & np.isfinite(seconds)
    if "Deleted" in frame.columns:
        valid &= ~_bool_series(frame["Deleted"], default=False)
    if "IsAccurate" in frame.columns:
        valid &= _bool_series(frame["IsAccurate"], default=False)
    for column in ("PitInTime", "PitOutTime"):
        if column in frame.columns:
            valid &= frame[column].isna()
    clean = pd.DataFrame(
        {
            "driver_id": frame["Driver"].astype(str).str.strip(),
            "lap_time_seconds": seconds,
        },
        index=frame.index,
    ).loc[valid]
    clean = clean[clean["driver_id"] != ""]
    return clean.groupby("driver_id", sort=False)["lap_time_seconds"].min()


def _qualifying_results_path(qualifying_laps_path: Path) -> Path:
    candidate = qualifying_laps_path.with_name(
        qualifying_laps_path.name.replace("_laps.csv", "_results.csv")
    )
    if not candidate.exists():
        raise FileNotFoundError(f"missing official stage labels: {candidate}")
    return candidate


def _session_results_path(laps_path: Path) -> Path:
    candidate = laps_path.with_name(laps_path.name.replace("_laps.csv", "_results.csv"))
    if not candidate.exists():
        raise FileNotFoundError(f"missing pre-session roster snapshot: {candidate}")
    return candidate


def _official_qualifying_positions(frame: pd.DataFrame) -> pd.Series:
    """Return a complete official order without using segment-time presence."""

    for column in ("Position", "ClassifiedPosition"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            break
    else:
        values = pd.Series(np.nan, index=frame.index, dtype=float)
    used = {int(value) for value in values.dropna() if float(value) > 0.0}
    next_position = 1
    completed = values.copy()
    for index in completed.index[completed.isna() | completed.le(0)]:
        while next_position in used:
            next_position += 1
        completed.loc[index] = next_position
        used.add(next_position)
    order = np.lexsort((np.arange(len(completed)), completed.to_numpy(dtype=float)))
    ranks = np.empty(len(completed), dtype=int)
    ranks[order] = np.arange(1, len(completed) + 1)
    return pd.Series(ranks, index=frame.index, dtype=int)


def _qualifying_stage_labels(qualifying_laps_path: Path) -> pd.DataFrame:
    """Load labels for history/evaluation only, never the inference roster."""

    path = _qualifying_results_path(qualifying_laps_path)
    frame = pd.read_csv(path)
    driver_column = next(
        (
            column
            for column in ("Abbreviation", "Driver", "DriverId", "DriverNumber")
            if column in frame.columns
        ),
        None,
    )
    if driver_column is None:
        raise ValueError(f"{path} has no driver identifier")

    def has_time(column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(False, index=frame.index)
        seconds = _lap_seconds(frame[column])
        return seconds.between(40.0, 180.0) & np.isfinite(seconds)

    q1 = has_time("Q1")
    q2 = has_time("Q2")
    q3 = has_time("Q3")
    stage_seconds = {
        column: (
            _lap_seconds(frame[column])
            if column in frame.columns
            else pd.Series(np.nan, index=frame.index)
        )
        for column in ("Q1", "Q2", "Q3")
    }
    official_positions = _official_qualifying_positions(frame)
    field_size = int(frame[driver_column].astype(str).str.strip().nunique())
    if field_size >= 12:
        try:
            elimination = qualifying_elimination_rule(field_size)
        except ValueError:
            # Some historical result snapshots have an odd field after a
            # withdrawal or exclusion.  Preserve the regulation's ten-car Q3
            # and split the remaining eliminations as evenly as possible; do
            # not infer advancement from whether a segment time happens to be
            # present.
            q3_slots = min(10, field_size)
            q2_slots = q3_slots + math.ceil((field_size - q3_slots) / 2.0)
        else:
            q2_slots = int(elimination.period_2_cars)
            q3_slots = int(elimination.period_3_cars)
    else:
        q2_slots = field_size
        q3_slots = field_size
    status = frame.get(
        "QualifyingStatus",
        frame.get("Status", pd.Series("", index=frame.index)),
    ).fillna("").astype(str).str.strip().str.lower()
    explicit_q3 = status.str.contains(
        r"(?:^|[^a-z0-9])q3(?:$|[^a-z0-9])|period[_ ]?3|segment[_ ]?3",
        regex=True,
    )
    explicit_q2 = explicit_q3 | status.str.contains(
        r"(?:^|[^a-z0-9])q2(?:$|[^a-z0-9])|period[_ ]?2|segment[_ ]?2",
        regex=True,
    )
    reached_q2 = official_positions.le(q2_slots) | explicit_q2
    reached_q3 = official_positions.le(q3_slots) | explicit_q3
    reached_q2 = reached_q2 | reached_q3
    labels = pd.DataFrame(
        {
            "driver_id": frame[driver_column].astype(str).str.strip(),
            "has_valid_qualifying_lap": (q1 | q2 | q3).astype(int),
            "reached_q2": reached_q2.astype(int),
            "reached_q3": reached_q3.astype(int),
            Q2_VALID_LAP_COLUMN: q2.astype(int),
            Q3_VALID_LAP_COLUMN: q3.astype(int),
            Q1_LAP_COLUMN: stage_seconds["Q1"],
            Q2_LAP_COLUMN: stage_seconds["Q2"],
            Q3_LAP_COLUMN: stage_seconds["Q3"],
            ACTUAL_LAP_COLUMN: pd.concat(
                list(stage_seconds.values()), axis=1
            ).min(axis=1, skipna=True),
        }
    )
    return labels.drop_duplicates("driver_id", keep="first").set_index("driver_id")


def _official_driver_best_laps(qualifying_laps_path: Path) -> pd.Series:
    """Use the same official Q1/Q2/Q3 target consumed by Qualifying mode."""

    return pd.to_numeric(
        _qualifying_stage_labels(qualifying_laps_path)[ACTUAL_LAP_COLUMN],
        errors="coerce",
    ).dropna()


def _load_target_after_frozen_forecasts(
    qualifying_laps_path: Path,
    *,
    expected_event_key: int,
    frozen_forecast_artifact: Mapping[str, object],
    root: Path,
) -> tuple[pd.Series, dict[str, str]]:
    """Open and hash evaluation truth only after event forecasts are frozen."""

    artifact_hash = str(frozen_forecast_artifact.get("artifact_sha256") or "")
    artifact_event_key = frozen_forecast_artifact.get("event_key")
    if re.fullmatch(r"[0-9a-f]{64}", artifact_hash) is None:
        raise RuntimeError("Best Lap target read requires a frozen forecast artifact")
    try:
        matching_event = int(artifact_event_key) == int(expected_event_key)
    except (TypeError, ValueError):
        matching_event = False
    if not matching_event:
        raise RuntimeError("Best Lap target read artifact belongs to another event")
    actual = _official_driver_best_laps(qualifying_laps_path)
    target_manifest = _hash_manifest(
        [qualifying_laps_path, _qualifying_results_path(qualifying_laps_path)],
        root=root,
    )
    return actual, target_manifest


def _label_quality_history(
    features: pd.DataFrame,
    *,
    actual: pd.Series,
    qualifying_laps_path: Path,
    history_weight: float,
    weak_transfer_prior: bool,
) -> pd.DataFrame:
    labelled = features.copy()
    stages = _qualifying_stage_labels(qualifying_laps_path)
    labelled[ACTUAL_LAP_COLUMN] = labelled["driver_id"].map(
        stages[ACTUAL_LAP_COLUMN]
    )
    labelled[ACTUAL_LAP_COLUMN] = labelled[ACTUAL_LAP_COLUMN].fillna(
        labelled["driver_id"].map(actual)
    )
    for column in (
        "has_valid_qualifying_lap",
        "reached_q2",
        "reached_q3",
        Q2_VALID_LAP_COLUMN,
        Q3_VALID_LAP_COLUMN,
    ):
        labelled[column] = pd.to_numeric(
            labelled["driver_id"].map(stages[column]), errors="coerce"
        )
    for column in (Q1_LAP_COLUMN, Q2_LAP_COLUMN, Q3_LAP_COLUMN):
        labelled[column] = pd.to_numeric(
            labelled["driver_id"].map(stages[column]), errors="coerce"
        )
    labelled["rehearsal_lap_time_seconds"] = labelled["valid_clean_best_seconds"]
    labelled["history_weight"] = float(history_weight)
    labelled["weak_transfer_prior"] = bool(weak_transfer_prior)
    return labelled


def _quality_aware_rehearsal(
    path: Path,
    *,
    event_key: int,
    source: str,
    include_earlier_evidence: bool = True,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "Driver" not in frame.columns:
        raise ValueError(f"{path} is missing Driver")
    earlier_parts: list[pd.DataFrame] = []
    # The target-aligned rehearsal owns the active pre-Q roster. Earlier
    # practice laps remain useful fallback pace evidence, but their result
    # tables contain reserve drivers who may already have surrendered the seat.
    roster_source = pd.read_csv(_session_results_path(path))
    for earlier_path in (
        _earlier_evidence_paths(path, source=source) if include_earlier_evidence else []
    ):
        earlier = pd.read_csv(earlier_path)
        if "Driver" not in earlier.columns:
            continue
        earlier["rehearsal_source"] = _source_from_filename(earlier_path)
        earlier_parts.append(earlier)
    earlier_laps = pd.concat(earlier_parts, ignore_index=True) if earlier_parts else None
    roster_parts: list[pd.DataFrame] = []
    for roster_source in (roster_source,):
        roster_driver = next(
            (
                column
                for column in ("Abbreviation", "Driver", "DriverId", "DriverNumber")
                if column in roster_source.columns
            ),
            None,
        )
        if roster_driver is None:
            continue
        roster_team = next(
            (
                column
                for column in ("TeamName", "Team", "team_id")
                if column in roster_source.columns
            ),
            None,
        )
        roster_parts.append(
            pd.DataFrame(
                {
                    "driver_id": roster_source[roster_driver].astype(str).str.strip(),
                    "team_id": (
                        roster_source[roster_team].astype(str).str.strip()
                        if roster_team is not None
                        else "unknown_team"
                    ),
                }
            )
        )
    if not roster_parts:
        raise ValueError(f"{path} has no pre-Q roster driver identifier")
    roster = pd.concat(roster_parts, ignore_index=True).drop_duplicates(
        "driver_id", keep="first"
    )
    features = build_quality_aware_rehearsal_features(
        frame,
        entrants=roster,
        earlier_laps=earlier_laps,
        official_session_timing=True,
    )
    features["event_key"] = int(event_key)
    features["rehearsal_source"] = str(source)
    return features


def _earlier_evidence_paths(rehearsal_path: Path, *, source: str) -> list[Path]:
    if source == "practice_3":
        patterns = ("*_practice_1_laps.csv", "*_practice_2_laps.csv")
    elif source == "sprint_qualifying":
        patterns = ("*_practice_1_laps.csv",)
    else:
        patterns = ()
    return sorted(
        {
            candidate
            for pattern in patterns
            for candidate in rehearsal_path.parent.glob(pattern)
            if candidate != rehearsal_path
        }
    )


def _source_from_filename(path: Path) -> str:
    name = path.stem.lower()
    for source in ("practice_1", "practice_2", "practice_3", "sprint_qualifying"):
        if source in name:
            return source
    return "earlier_session"


def _round_number(directory: Path) -> int:
    match = ROUND_PATTERN.match(directory.name)
    if match is None:
        raise ValueError(f"not an F1 round directory: {directory}")
    return int(match.group(1))


def _discover_rounds(weekends_dir: Path, year: int) -> list[Path]:
    directories = [path for path in (weekends_dir / str(year)).glob("round_*") if path.is_dir()]
    return sorted(directories, key=_round_number)


def _target_aligned_files(round_dir: Path) -> tuple[str, Path, Path]:
    sprint_qualifying = sorted(round_dir.glob("*_sprint_qualifying_laps.csv"))
    practice_3 = sorted(round_dir.glob("*_practice_3_laps.csv"))
    qualifying = sorted(
        path
        for path in round_dir.glob("*_qualifying_laps.csv")
        if "sprint_qualifying" not in path.name
    )
    if not qualifying:
        raise FileNotFoundError(f"{round_dir} has no Grand Prix qualifying laps")
    if sprint_qualifying:
        return "sprint_qualifying", sprint_qualifying[0], qualifying[0]
    if practice_3:
        return "practice_3", practice_3[0], qualifying[0]
    raise FileNotFoundError(
        f"{round_dir} has neither Sprint Qualifying nor FP3 as a target-aligned rehearsal"
    )


def _build_weak_transfer_history(
    weekends_dir: Path,
    *,
    target_year: int,
) -> tuple[list[pd.DataFrame], list[Path], list[dict[str, Any]]]:
    """Build regulation-aware weak priors from invariant rehearsal-to-Q effects.

    Absolute lap time is never transferred: every label is the residual from
    that event's own causal rehearsal anchor. Older seasons receive explicit
    recency weights, and their team/driver effects are disabled by the model's
    ``weak_transfer_prior`` contract.
    """

    parts: list[pd.DataFrame] = []
    files: list[Path] = []
    summary: list[dict[str, Any]] = []
    for season in range(max(2022, int(target_year) - 4), int(target_year)):
        weight = float(max(0.03, 0.30 / max(1, target_year - season)))
        used_events = 0
        skipped_events = 0
        rows = 0
        for round_dir in _discover_rounds(weekends_dir, season):
            try:
                source, rehearsal_path, qualifying_path = _target_aligned_files(round_dir)
            except FileNotFoundError:
                # 2022-23 Sprint weekends have no pre-GP-Qualifying FP3/SQ
                # source. They are excluded instead of using post-cutoff data.
                skipped_events += 1
                continue
            event_key = (int(season) * 100) + _round_number(round_dir)
            quality = _quality_aware_rehearsal(
                rehearsal_path,
                event_key=event_key,
                source=source,
                include_earlier_evidence=True,
            )
            actual = _official_driver_best_laps(qualifying_path)
            parts.append(
                _label_quality_history(
                    quality,
                    actual=actual,
                    qualifying_laps_path=qualifying_path,
                    history_weight=weight,
                    weak_transfer_prior=True,
                )
            )
            files.extend(
                [
                    rehearsal_path,
                    _session_results_path(rehearsal_path),
                    qualifying_path,
                    _qualifying_results_path(qualifying_path),
                    *_earlier_evidence_paths(rehearsal_path, source=source),
                    *(
                        _session_results_path(value)
                        for value in _earlier_evidence_paths(
                            rehearsal_path, source=source
                        )
                    ),
                ]
            )
            used_events += 1
            rows += len(quality)
        summary.append(
            {
                "season": int(season),
                "event_weight": weight,
                "used_event_blocks": used_events,
                "skipped_no_causal_target_aligned_source": skipped_events,
                "entrant_rows": rows,
                "transferred_effects": [
                    "source_session_shift",
                    "invariant_quality_feature_residuals",
                    "valid_lap_hurdle_prior",
                ],
                "blocked_effects": ["absolute_lap_time", "team_effect", "driver_effect"],
            }
        )
    return parts, files, summary


def _event_metrics(
    frame: pd.DataFrame,
    *,
    baseline_column: str = "baseline_lap_p50",
    raw_rehearsal_column: str = "raw_rehearsal_lap_p50",
) -> dict[str, Any]:
    error = frame["lap_p50"] - frame[ACTUAL_LAP_COLUMN]
    baseline_error = frame[baseline_column] - frame[ACTUAL_LAP_COLUMN]
    raw_error = frame[raw_rehearsal_column] - frame[ACTUAL_LAP_COLUMN]
    predicted_rank = frame["lap_p50"].rank(method="average", ascending=True)
    baseline_rank = frame[baseline_column].rank(method="average", ascending=True)
    actual_rank = frame[ACTUAL_LAP_COLUMN].rank(method="average", ascending=True)
    interval_valid = frame["lap_p05"].notna() & frame["lap_p90"].notna()
    coverage = (
        (
            frame.loc[interval_valid, ACTUAL_LAP_COLUMN].ge(frame.loc[interval_valid, "lap_p05"])
            & frame.loc[interval_valid, ACTUAL_LAP_COLUMN].le(frame.loc[interval_valid, "lap_p90"])
        ).mean()
        if interval_valid.any()
        else float("nan")
    )
    predicted_top3 = set(frame.nsmallest(min(3, len(frame)), "lap_p50")["driver_id"])
    baseline_top3 = set(frame.nsmallest(min(3, len(frame)), baseline_column)["driver_id"])
    actual_top3 = set(frame.nsmallest(min(3, len(frame)), ACTUAL_LAP_COLUMN)["driver_id"])
    return {
        "rows": int(len(frame)),
        "p50_mae_seconds": float(error.abs().mean()),
        "raw_rehearsal_mae_seconds": float(raw_error.abs().mean()),
        "baseline_p50_mae_seconds": float(baseline_error.abs().mean()),
        "challenger_minus_baseline_mae_seconds": float(
            error.abs().mean() - baseline_error.abs().mean()
        ),
        "p50_rmse_seconds": float(np.sqrt(np.mean(np.square(error)))),
        "mean_error_seconds": float(error.mean()),
        "spearman": float(predicted_rank.corr(actual_rank, method="pearson")),
        "baseline_spearman": float(baseline_rank.corr(actual_rank, method="pearson")),
        "fastest_driver_hit": bool(frame.loc[frame["lap_p50"].idxmin(), "driver_id"] == frame.loc[frame[ACTUAL_LAP_COLUMN].idxmin(), "driver_id"]),
        "baseline_fastest_driver_hit": bool(
            frame.loc[frame[baseline_column].idxmin(), "driver_id"]
            == frame.loc[frame[ACTUAL_LAP_COLUMN].idxmin(), "driver_id"]
        ),
        "top3_overlap_rate": float(len(predicted_top3 & actual_top3) / max(1, min(3, len(frame)))),
        "baseline_top3_overlap_rate": float(
            len(baseline_top3 & actual_top3) / max(1, min(3, len(frame)))
        ),
        "interval_rows": int(interval_valid.sum()),
        "interval_coverage": float(coverage),
        "interval_mean_width_seconds": float(
            (frame.loc[interval_valid, "lap_p90"] - frame.loc[interval_valid, "lap_p05"]).mean()
        )
        if interval_valid.any()
        else float("nan"),
    }


def _aggregate_point_mae(event_metrics: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Map each point comparator to its own event-balanced MAE."""

    if not event_metrics:
        raise ValueError("point MAE aggregation requires at least one event")
    metric_fields = {
        "conditional_event_mean_p50_mae_seconds": "p50_mae_seconds",
        "conditional_event_mean_raw_rehearsal_mae_seconds": (
            "raw_rehearsal_mae_seconds"
        ),
        "conditional_event_mean_baseline_p50_mae_seconds": (
            "baseline_p50_mae_seconds"
        ),
    }
    aggregate: dict[str, float] = {}
    for output_field, metric_field in metric_fields.items():
        values = np.asarray(
            [float(item[metric_field]) for item in event_metrics],
            dtype=float,
        )
        if not np.isfinite(values).all():
            raise ValueError(f"event metric {metric_field} must be finite")
        aggregate[output_field] = float(values.mean())
    return aggregate


def _paired_bootstrap(
    challenger: Sequence[float],
    baseline: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    challenger_values = np.asarray(challenger, dtype=float)
    baseline_values = np.asarray(baseline, dtype=float)
    delta = challenger_values - baseline_values
    rng = np.random.default_rng(int(seed))
    draws = delta[rng.integers(0, len(delta), size=(int(samples), len(delta)))].mean(axis=1)
    return {
        "mean_delta_seconds": float(delta.mean()),
        "ci95_seconds": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "bootstrap_probability_of_improvement": float(np.mean(draws < 0.0)),
        "samples": int(samples),
        "seed": int(seed),
    }


def _event_stability(event_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = np.asarray(
        [float(item["challenger_minus_baseline_mae_seconds"]) for item in event_metrics],
        dtype=float,
    )
    improvements = np.maximum(-deltas, 0.0)
    total_gain = float(improvements.sum())
    leave_one_out = [
        float(np.delete(deltas, index).mean()) if len(deltas) > 1 else float("nan")
        for index in range(len(deltas))
    ]
    return {
        "event_deltas_seconds": deltas.tolist(),
        "leave_one_event_out_mean_deltas_seconds": leave_one_out,
        "leave_one_event_out_directionally_stable": bool(
            leave_one_out and all(value < 0.0 for value in leave_one_out if np.isfinite(value))
        ),
        "largest_event_share_of_positive_gain": (
            float(improvements.max() / total_gain) if total_gain > 0.0 else float("nan")
        ),
        "single_event_gain_concentration_gate_passed": bool(
            total_gain > 0.0 and float(improvements.max() / total_gain) <= 0.50
        ),
    }


def _weekend_stratum(rehearsal_source: object) -> str:
    source = str(rehearsal_source).strip().lower()
    if source == "sprint_qualifying":
        return "sprint"
    if source in {"practice_3", "fp3"}:
        return "standard"
    return "unknown"


def _interval_block_summary(
    frame: pd.DataFrame,
    *,
    lower_column: str,
    upper_column: str,
    status_column: str,
) -> dict[str, Any]:
    """Summarize interval evidence with complete events as the primary unit.

    Row-weighted coverage is retained as a diagnostic, but promotion gates use
    the event-balanced summaries below so a large or easy field cannot mask a
    weak weekend.  Weekend-format summaries are themselves event balanced.
    """

    required = {
        "event_key",
        "rehearsal_source",
        ACTUAL_LAP_COLUMN,
        lower_column,
        upper_column,
        status_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"interval block summary missing columns: {missing}")

    working = frame.copy()
    event_keys = pd.to_numeric(working["event_key"], errors="coerce")
    if event_keys.isna().any():
        raise ValueError("interval block summary requires finite event keys")
    working["_event_key"] = event_keys.astype(int)
    working["_weekend_stratum"] = working["rehearsal_source"].map(
        _weekend_stratum
    )
    lower = pd.to_numeric(working[lower_column], errors="coerce")
    upper = pd.to_numeric(working[upper_column], errors="coerce")
    actual = pd.to_numeric(working[ACTUAL_LAP_COLUMN], errors="coerce")
    width = upper - lower
    valid = (
        lower.notna()
        & upper.notna()
        & actual.notna()
        & width.ge(0.0)
        & working[status_column].eq("calibrated_disjoint_event_partition")
    )
    working["_interval_valid"] = valid
    working["_interval_covered"] = valid & actual.ge(lower) & actual.le(upper)
    working["_interval_width_seconds"] = width.where(valid)

    by_event: dict[str, dict[str, Any]] = {}
    for event_key, event in working.groupby("_event_key", sort=True):
        event_valid = event["_interval_valid"]
        validated_rows = int(event_valid.sum())
        covered_rows = int(event.loc[event_valid, "_interval_covered"].sum())
        strata = sorted(set(event["_weekend_stratum"].astype(str)))
        stratum = strata[0] if len(strata) == 1 else "mixed"
        by_event[str(int(event_key))] = {
            "event_key": int(event_key),
            "weekend_stratum": stratum,
            "target_rows": int(len(event)),
            "validated_interval_rows": validated_rows,
            "validated_interval_row_rate": (
                float(validated_rows / len(event)) if len(event) else 0.0
            ),
            "covered_interval_rows": covered_rows,
            "coverage": (
                float(covered_rows / validated_rows)
                if validated_rows
                else float("nan")
            ),
            "mean_width_seconds": (
                float(event.loc[event_valid, "_interval_width_seconds"].mean())
                if validated_rows
                else float("nan")
            ),
        }

    event_values = list(by_event.values())
    valid_events = [
        item for item in event_values if int(item["validated_interval_rows"]) > 0
    ]
    by_weekend_stratum: dict[str, dict[str, Any]] = {}
    for stratum in sorted({str(item["weekend_stratum"]) for item in event_values}):
        stratum_events = [
            item for item in event_values if str(item["weekend_stratum"]) == stratum
        ]
        valid_stratum_events = [
            item
            for item in stratum_events
            if int(item["validated_interval_rows"]) > 0
        ]
        validated_rows = sum(
            int(item["validated_interval_rows"]) for item in stratum_events
        )
        covered_rows = sum(
            int(item["covered_interval_rows"]) for item in stratum_events
        )
        target_rows = sum(int(item["target_rows"]) for item in stratum_events)
        coverages = [float(item["coverage"]) for item in valid_stratum_events]
        widths = [float(item["mean_width_seconds"]) for item in valid_stratum_events]
        by_weekend_stratum[stratum] = {
            "event_keys": [int(item["event_key"]) for item in stratum_events],
            "event_count": int(len(stratum_events)),
            "events_with_validated_intervals": int(len(valid_stratum_events)),
            "target_rows": int(target_rows),
            "validated_interval_rows": int(validated_rows),
            "validated_interval_row_rate": (
                float(validated_rows / target_rows) if target_rows else 0.0
            ),
            "pooled_coverage": (
                float(covered_rows / validated_rows)
                if validated_rows
                else float("nan")
            ),
            "event_balanced_coverage": (
                float(np.mean(coverages)) if coverages else float("nan")
            ),
            "minimum_event_coverage": (
                float(np.min(coverages)) if coverages else float("nan")
            ),
            "event_balanced_mean_width_seconds": (
                float(np.mean(widths)) if widths else float("nan")
            ),
        }

    validated_rows = int(working["_interval_valid"].sum())
    covered_rows = int(working.loc[working["_interval_valid"], "_interval_covered"].sum())
    coverages = [float(item["coverage"]) for item in valid_events]
    widths = [float(item["mean_width_seconds"]) for item in valid_events]
    return {
        "audit_event_count": int(len(event_values)),
        "events_with_validated_intervals": int(len(valid_events)),
        "all_audit_events_have_validated_interval_rows": bool(
            event_values and len(valid_events) == len(event_values)
        ),
        "target_rows": int(len(working)),
        "validated_interval_rows": validated_rows,
        "validated_interval_row_rate": (
            float(validated_rows / len(working)) if len(working) else 0.0
        ),
        "covered_interval_rows": covered_rows,
        "pooled_coverage": (
            float(covered_rows / validated_rows) if validated_rows else float("nan")
        ),
        "row_weighted_mean_width_seconds": (
            float(working.loc[working["_interval_valid"], "_interval_width_seconds"].mean())
            if validated_rows
            else float("nan")
        ),
        "event_balanced_coverage": (
            float(np.mean(coverages)) if coverages else float("nan")
        ),
        "minimum_event_coverage": (
            float(np.min(coverages)) if coverages else float("nan")
        ),
        "event_balanced_mean_width_seconds": (
            float(np.mean(widths)) if widths else float("nan")
        ),
        "by_event": by_event,
        "by_weekend_stratum": by_weekend_stratum,
    }


def _interval_width_comparison(
    candidate: Mapping[str, Any],
    comparator: Mapping[str, Any],
    *,
    candidate_label: str,
    comparator_label: str,
) -> dict[str, Any]:
    """Compare interval sharpness only when the products are genuinely distinct."""

    distinct_products = str(candidate_label) != str(comparator_label)
    candidate_events = dict(candidate.get("by_event", {}))
    comparator_events = dict(comparator.get("by_event", {}))
    event_ratios: dict[str, float] = {}
    if distinct_products:
        for event_key in sorted(set(candidate_events) & set(comparator_events)):
            candidate_width = float(candidate_events[event_key]["mean_width_seconds"])
            comparator_width = float(comparator_events[event_key]["mean_width_seconds"])
            if (
                np.isfinite(candidate_width)
                and np.isfinite(comparator_width)
                and comparator_width > 0.0
            ):
                event_ratios[event_key] = float(candidate_width / comparator_width)

    candidate_strata = dict(candidate.get("by_weekend_stratum", {}))
    comparator_strata = dict(comparator.get("by_weekend_stratum", {}))
    stratum_ratios: dict[str, float] = {}
    if distinct_products:
        for stratum in sorted(set(candidate_strata) & set(comparator_strata)):
            candidate_width = float(
                candidate_strata[stratum]["event_balanced_mean_width_seconds"]
            )
            comparator_width = float(
                comparator_strata[stratum]["event_balanced_mean_width_seconds"]
            )
            if (
                np.isfinite(candidate_width)
                and np.isfinite(comparator_width)
                and comparator_width > 0.0
            ):
                stratum_ratios[stratum] = float(candidate_width / comparator_width)

    candidate_event_balanced_width = float(
        candidate.get("event_balanced_mean_width_seconds", float("nan"))
    )
    comparator_event_balanced_width = float(
        comparator.get("event_balanced_mean_width_seconds", float("nan"))
    )
    event_balanced_ratio = (
        float(candidate_event_balanced_width / comparator_event_balanced_width)
        if distinct_products
        and np.isfinite(candidate_event_balanced_width)
        and np.isfinite(comparator_event_balanced_width)
        and comparator_event_balanced_width > 0.0
        else float("nan")
    )
    candidate_event_keys = set(candidate_events)
    comparator_event_keys = set(comparator_events)
    all_events_comparable = bool(
        distinct_products
        and candidate_event_keys
        and candidate_event_keys == comparator_event_keys == set(event_ratios)
    )
    all_required_strata_comparable = bool(
        distinct_products
        and set(REQUIRED_WEEKEND_STRATA).issubset(stratum_ratios)
    )
    reference_available = bool(
        distinct_products
        and np.isfinite(event_balanced_ratio)
        and all_events_comparable
        and all_required_strata_comparable
    )
    return {
        "candidate_label": str(candidate_label),
        "comparator_label": str(comparator_label),
        "distinct_interval_products": distinct_products,
        "reference_available": reference_available,
        "unavailable_reason": (
            None
            if reference_available
            else (
                "candidate_and_comparator_are_same_interval_product"
                if not distinct_products
                else "incomplete_event_or_weekend_stratum_width_reference"
            )
        ),
        "event_balanced_width_ratio": event_balanced_ratio,
        "by_event_width_ratio": event_ratios,
        "by_weekend_stratum_width_ratio": stratum_ratios,
        "all_audit_events_comparable": all_events_comparable,
        "all_required_weekend_strata_comparable": all_required_strata_comparable,
    }


def _best_lap_point_promotion_gates(
    *,
    relative_mae_gain: float,
    paired_retained: Mapping[str, Any],
    observed_target_coverage: float,
    all_weekend_strata_improve: bool,
    stability: Mapping[str, Any],
) -> dict[str, bool]:
    """Promotion gates for the per-driver seconds target only.

    Fastest-driver and top-three hits are useful cross-mode diagnostics, but
    they are Qualifying ranking objectives rather than proper losses for a
    per-driver lap-time forecast.  They therefore must not veto this mode.
    """

    return {
        "mae_improves_at_least_five_percent": bool(relative_mae_gain >= 0.05),
        "event_bootstrap_upper_bound_below_zero": bool(
            paired_retained["ci95_seconds"][1] < 0.0
        ),
        "bootstrap_probability_of_improvement_at_least_095": bool(
            paired_retained["bootstrap_probability_of_improvement"] >= 0.95
        ),
        "observed_target_coverage_is_100_percent": bool(
            observed_target_coverage >= 1.0
        ),
        "all_weekend_strata_improve": bool(all_weekend_strata_improve),
        "leave_one_event_out_directionally_stable": bool(
            stability["leave_one_event_out_directionally_stable"]
        ),
        "no_single_event_supplies_more_than_half_gain": bool(
            stability["single_event_gain_concentration_gate_passed"]
        ),
    }


def _best_lap_interval_promotion_gates(
    *,
    retained_interval_summary: Mapping[str, Any],
    retained_interval_calibration_event_count: int,
    interval_width_comparison: Mapping[str, Any],
) -> dict[str, bool]:
    """Apply fail-closed event-block, format, and sharpness interval gates."""

    pooled_coverage = float(
        retained_interval_summary.get("pooled_coverage", float("nan"))
    )
    event_balanced_coverage = float(
        retained_interval_summary.get("event_balanced_coverage", float("nan"))
    )
    minimum_event_coverage = float(
        retained_interval_summary.get("minimum_event_coverage", float("nan"))
    )
    strata = dict(retained_interval_summary.get("by_weekend_stratum", {}))
    required_weekend_strata_present = bool(
        set(REQUIRED_WEEKEND_STRATA).issubset(strata)
    )
    every_required_stratum_coverage_passes = bool(
        required_weekend_strata_present
        and all(
            np.isfinite(float(strata[stratum]["event_balanced_coverage"]))
            and float(strata[stratum]["event_balanced_coverage"])
            >= float(MINIMUM_PER_STRATUM_INTERVAL_COVERAGE)
            for stratum in REQUIRED_WEEKEND_STRATA
        )
    )
    event_balanced_width_ratio = float(
        interval_width_comparison.get("event_balanced_width_ratio", float("nan"))
    )
    event_width_ratios = dict(
        interval_width_comparison.get("by_event_width_ratio", {})
    )
    stratum_width_ratios = dict(
        interval_width_comparison.get("by_weekend_stratum_width_ratio", {})
    )
    return {
        "independent_interval_calibration_events_at_least_four": bool(
            int(retained_interval_calibration_event_count)
            >= int(MINIMUM_INTERVAL_PROMOTION_EVENTS)
        ),
        "independent_interval_audit_events_at_least_three": bool(
            int(retained_interval_summary.get("audit_event_count") or 0)
            >= int(MINIMUM_INTERVAL_AUDIT_EVENTS)
        ),
        "pooled_interval_coverage_within_five_points_of_85_percent": bool(
            np.isfinite(pooled_coverage)
            and abs(pooled_coverage - INTERVAL_NOMINAL_COVERAGE)
            <= INTERVAL_EVENT_BALANCED_COVERAGE_TOLERANCE
        ),
        "event_balanced_interval_coverage_within_five_points_of_85_percent": bool(
            np.isfinite(event_balanced_coverage)
            and abs(event_balanced_coverage - INTERVAL_NOMINAL_COVERAGE)
            <= INTERVAL_EVENT_BALANCED_COVERAGE_TOLERANCE
        ),
        "every_audit_event_interval_coverage_at_least_70_percent": bool(
            retained_interval_summary.get(
                "all_audit_events_have_validated_interval_rows"
            )
            and np.isfinite(minimum_event_coverage)
            and minimum_event_coverage >= MINIMUM_PER_EVENT_INTERVAL_COVERAGE
        ),
        "standard_and_sprint_interval_evidence_present": (
            required_weekend_strata_present
        ),
        "every_weekend_stratum_interval_coverage_at_least_75_percent": (
            every_required_stratum_coverage_passes
        ),
        "validated_interval_coverage_is_100_percent": bool(
            float(
                retained_interval_summary.get("validated_interval_row_rate") or 0.0
            )
            >= 1.0
        ),
        "non_tautological_interval_width_reference_available": bool(
            interval_width_comparison.get("reference_available")
        ),
        "event_balanced_interval_width_inflation_at_most_ten_percent": bool(
            interval_width_comparison.get("reference_available")
            and np.isfinite(event_balanced_width_ratio)
            and event_balanced_width_ratio <= MAXIMUM_EVENT_BALANCED_WIDTH_RATIO
        ),
        "every_audit_event_interval_width_inflation_at_most_25_percent": bool(
            interval_width_comparison.get("all_audit_events_comparable")
            and event_width_ratios
            and all(
                float(ratio) <= MAXIMUM_PER_EVENT_WIDTH_RATIO
                for ratio in event_width_ratios.values()
            )
        ),
        "every_weekend_stratum_interval_width_inflation_at_most_15_percent": bool(
            interval_width_comparison.get(
                "all_required_weekend_strata_comparable"
            )
            and all(
                stratum in stratum_width_ratios
                and float(stratum_width_ratios[stratum])
                <= MAXIMUM_PER_STRATUM_WIDTH_RATIO
                for stratum in REQUIRED_WEEKEND_STRATA
            )
        ),
    }


def _locked_best_lap_partitions(
    prior_parts: Sequence[pd.DataFrame],
    *,
    target_event_keys: Sequence[int],
    target_year: int,
    allow_empty_development: bool = False,
) -> dict[str, tuple[int, ...]]:
    prior_keys = tuple(
        sorted(
            {
                int(value)
                for part in prior_parts
                for value in pd.to_numeric(part.get("event_key"), errors="coerce").dropna()
            }
        )
    )
    target_keys = tuple(sorted({int(value) for value in target_event_keys}))
    if len(target_keys) < 8:
        raise ValueError(
            "Best Lap requires at least eight target-season events: two frozen point-fit, "
            "four held-out calibration, and at least two audit events"
        )
    partitions = {
        "development": prior_keys,
        "selection": target_keys[:2],
        "calibration": target_keys[2:6],
        "audit": target_keys[6:],
    }
    issues = validate_event_partitions(
        **{name: [str(value) for value in values] for name, values in partitions.items()}
    )
    if allow_empty_development:
        issues = tuple(
            issue for issue in issues if issue != "development_events_missing"
        )
    if issues:
        raise ValueError(f"invalid Best Lap event partitions: {list(issues)}")
    return partitions


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _build_shared_event_forecast(
    history: pd.DataFrame,
    inference: pd.DataFrame,
    *,
    target_event_key: int,
    interval_calibration_predictions: pd.DataFrame | None = None,
):
    """Best-Lap runner adapter to the cross-mode frozen forecast path."""

    return build_shared_qualifying_event_forecast(
        history,
        inference,
        target_event_key=int(target_event_key),
        interval_calibration_predictions=interval_calibration_predictions,
    )


def run_backtest(
    *,
    weekends_dir: Path,
    year: int,
    rounds: Sequence[int] | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
    use_weak_transfer_priors: bool = False,
    tcn_evidence_path: Path | None = None,
) -> dict[str, Any]:
    if int(year) < MIN_SUPPORTED_WEEKEND_YEAR:
        raise ValueError(
            "best-lap rehearsal backtest supports 2024+ weekend chronology only; "
            "pre-2024 Sprint Qualifying/Sprint Shootout occurs after or does not exist "
            "before Grand Prix qualifying"
        )
    root = _repo_root()
    implementation_manifest = _hash_manifest(_implementation_paths(root), root=root)
    selected = [
        path
        for path in _discover_rounds(weekends_dir, year)
        if rounds is None or _round_number(path) in set(int(value) for value in rounds)
    ]
    if not selected:
        raise ValueError("no completed local rounds selected")
    prior_parts, prior_files, prior_summary = (
        _build_weak_transfer_history(weekends_dir, target_year=int(year))
        if use_weak_transfer_priors
        else ([], [], [])
    )
    target_event_keys = tuple(
        int(year) * 100 + _round_number(path) for path in selected
    )
    partitions = _locked_best_lap_partitions(
        prior_parts,
        target_event_keys=target_event_keys,
        target_year=int(year),
        allow_empty_development=not bool(use_weak_transfer_priors),
    )
    enable_selected_residual = SHARED_QUALIFYING_ENABLE_ROBUST_RESIDUAL
    residual_selector = {
        "status": "architecture_locked_without_cross_season_selection",
        "selected_enable_robust_residual": enable_selected_residual,
        "shared_cross_mode_selected_enable_robust_residual": enable_selected_residual,
        "selection_data_seasons": [int(year)],
        "prior_season_files_loaded": 0,
        "reason": (
            "the production point architecture is fixed before the same-season audit; "
            "alternative residual learners are evaluated only in the separate nested "
            "same-season research harness"
        ),
        "reproducible_selector_executed": False,
    }
    telemetry_readiness, telemetry_input_files = _telemetry_readiness_audit(
        root=root,
        year=int(year),
    )
    telemetry_audit = telemetry_readiness["audit"]
    resolved_tcn_evidence_path = (
        tcn_evidence_path.expanduser().resolve()
        if tcn_evidence_path is not None
        else _default_tcn_evidence_path(root, year=int(year))
    )
    if tcn_evidence_path is not None and not resolved_tcn_evidence_path.is_file():
        raise FileNotFoundError(
            f"explicit TCN evidence does not exist: {resolved_tcn_evidence_path}"
        )
    tcn_evidence_input_files: list[Path] = []
    tcn_evidence = (
        _load_tcn_research_evidence(
            resolved_tcn_evidence_path,
            root=root,
            year=int(year),
            telemetry_audit=telemetry_audit,
            validated_input_files=tcn_evidence_input_files,
        )
        if resolved_tcn_evidence_path.is_file()
        else None
    )
    deep_model_decision = _deep_model_readiness_decision(
        telemetry_audit,
        tcn_evidence=tcn_evidence,
    )
    inference_input_files: list[Path] = [
        *prior_files,
        *telemetry_input_files,
        *tcn_evidence_input_files,
    ]
    evaluation_target_files: list[Path] = []
    for round_dir in selected:
        source, rehearsal_path, qualifying_path = _target_aligned_files(round_dir)
        inference_input_files.extend(
            [
                rehearsal_path,
                _session_results_path(rehearsal_path),
                *_earlier_evidence_paths(rehearsal_path, source=source),
                *(
                    _session_results_path(value)
                    for value in _earlier_evidence_paths(
                        rehearsal_path, source=source
                    )
                ),
            ]
        )
        evaluation_target_files.extend(
            [qualifying_path, _qualifying_results_path(qualifying_path)]
        )
    selected_input_files = [*inference_input_files, *evaluation_target_files]
    inference_input_manifest_before = _hash_manifest(
        inference_input_files,
        root=root,
    )
    evaluation_target_manifest_after_forecast: dict[str, str] = {}

    baseline_history_parts: list[pd.DataFrame] = [
        part[
            [
                "event_key",
                "driver_id",
                "rehearsal_source",
                "rehearsal_lap_time_seconds",
                ACTUAL_LAP_COLUMN,
                "history_weight",
                "weak_transfer_prior",
            ]
        ].copy()
        for part in prior_parts
    ]
    challenger_history_parts: list[pd.DataFrame] = list(prior_parts)
    event_payloads: list[dict[str, Any]] = []
    all_scored: list[pd.DataFrame] = []
    all_challenger_scored: list[pd.DataFrame] = []
    shared_forecast_artifacts: list[dict[str, object]] = []
    baseline_interval_calibration_rows: list[pd.DataFrame] = []
    location_interval_calibration_rows: list[pd.DataFrame] = []
    robust_interval_calibration_rows: list[pd.DataFrame] = []
    frozen_structural_point_predictor_sha256: str | None = None
    frozen_complete_predictor_sha256: str | None = None
    input_files: list[Path] = [
        *prior_files,
        *telemetry_input_files,
        *tcn_evidence_input_files,
    ]

    for round_dir in selected:
        round_number = _round_number(round_dir)
        event_key = (int(year) * 100) + int(round_number)
        source, rehearsal_path, qualifying_path = _target_aligned_files(round_dir)
        input_files.extend(
            [
                rehearsal_path,
                _session_results_path(rehearsal_path),
                qualifying_path,
                _qualifying_results_path(qualifying_path),
                *_earlier_evidence_paths(rehearsal_path, source=source),
                *(
                    _session_results_path(value)
                    for value in _earlier_evidence_paths(
                        rehearsal_path, source=source
                    )
                ),
            ]
        )
        rehearsal = _clean_driver_best_laps(rehearsal_path)
        quality_features = _quality_aware_rehearsal(
            rehearsal_path, event_key=event_key, source=source
        )
        inference_ids = pd.Index(quality_features["driver_id"].astype(str))
        baseline_inference_ids = rehearsal.index
        if inference_ids.empty or baseline_inference_ids.empty:
            raise ValueError(f"round {round_number} has no causal rehearsal entrants")

        baseline_history = (
            pd.concat(baseline_history_parts, ignore_index=True)
            if baseline_history_parts
            else pd.DataFrame()
        )
        challenger_history = (
            pd.concat(challenger_history_parts, ignore_index=True)
            if challenger_history_parts
            else pd.DataFrame()
        )
        calibration_keys_available = (
            tuple(partitions["calibration"])
            if event_key in set(partitions["audit"])
            else ()
        )
        baseline_model = fit_achievable_best_lap_model(
            baseline_history,
            target_event_key=event_key,
            enable_robust_residual=False,
            calibration_event_keys=(),
        )
        location_model = fit_achievable_best_lap_model(
            challenger_history,
            target_event_key=event_key,
            enable_robust_residual=False,
            calibration_event_keys=(),
            model_name="shared_qualifying_latent_lap_v4",
        )
        robust_model = fit_achievable_best_lap_model(
            challenger_history,
            target_event_key=event_key,
            enable_robust_residual=True,
            calibration_event_keys=(),
        )
        if event_key in set(partitions["audit"]):
            if not (
                baseline_interval_calibration_rows
                and location_interval_calibration_rows
                and robust_interval_calibration_rows
            ):
                raise RuntimeError("audit reached before held-out interval calibration completed")
            baseline_model = calibrate_achievable_best_lap_model(
                baseline_model,
                pd.concat(baseline_interval_calibration_rows, ignore_index=True),
            )
            location_model = calibrate_achievable_best_lap_model(
                location_model,
                pd.concat(location_interval_calibration_rows, ignore_index=True),
            )
            robust_model = calibrate_achievable_best_lap_model(
                robust_model,
                pd.concat(robust_interval_calibration_rows, ignore_index=True),
            )
        held_out_interval_rows = (
            pd.concat(location_interval_calibration_rows, ignore_index=True)
            if event_key in set(partitions["audit"])
            else None
        )
        challenger_model, shared_forecast, artifact = _build_shared_event_forecast(
            challenger_history,
            quality_features,
            target_event_key=event_key,
            interval_calibration_predictions=held_out_interval_rows,
        )
        if shared_point_predictor_sha256(challenger_model) != shared_point_predictor_sha256(
            location_model
        ):
            raise RuntimeError("Best Lap diverged from common shared point-model builder")
        current_point_hash = shared_point_predictor_sha256(challenger_model)
        current_structural_hash = shared_structural_point_predictor_sha256(
            challenger_model
        )
        if event_key >= min(partitions["calibration"]):
            if frozen_structural_point_predictor_sha256 is None:
                frozen_structural_point_predictor_sha256 = current_structural_hash
            elif current_structural_hash != frozen_structural_point_predictor_sha256:
                raise RuntimeError(
                    "Best Lap structural point fit changed after point-fit freeze"
                )
        if event_key in set(partitions["audit"]):
            if frozen_complete_predictor_sha256 is None:
                frozen_complete_predictor_sha256 = current_point_hash
            elif current_point_hash != frozen_complete_predictor_sha256:
                raise RuntimeError(
                    "complete Best Lap point-plus-interval predictor changed after freeze"
                )
        baseline_inference = pd.DataFrame(
            {
                "event_key": event_key,
                "driver_id": baseline_inference_ids.astype(str),
                "rehearsal_source": source,
                "rehearsal_lap_time_seconds": rehearsal.loc[
                    baseline_inference_ids
                ].to_numpy(dtype=float),
            }
        )
        joint_seed = SHARED_QUALIFYING_SAMPLE_SEED_BASE + int(event_key)
        # The legacy baseline is a per-driver rehearsal-shift estimator, not a
        # full classification simulator. Drivers without a valid rehearsal lap
        # are intentionally outside its scored population, which can be odd;
        # applying FIA elimination sampling to that subset is mathematically
        # invalid. Use its analytic lap forecast directly.
        baseline_predictions = baseline_model.predict(baseline_inference)
        baseline_indexed = baseline_predictions.set_index("driver_id")
        baseline_by_driver = baseline_indexed["lap_p50"]
        raw_rehearsal_by_driver = baseline_inference.set_index("driver_id")[
            "rehearsal_lap_time_seconds"
        ]
        shared_forecast_artifacts.append(artifact)
        predictions = shared_forecast.lap_predictions.copy()
        predictions["shared_forecast_artifact_sha256"] = artifact["artifact_sha256"]
        predictions["shared_joint_samples_sha256"] = artifact["joint_samples_sha256"]
        robust_diagnostic = robust_model.predict_qualifying(
            quality_features,
            samples=5_000,
            seed=joint_seed,
            allow_diagnostic_stage_fallback=True,
        ).lap_predictions.set_index("driver_id")
        location_diagnostic = location_model.predict_qualifying(
            quality_features,
            samples=5_000,
            seed=joint_seed,
            allow_diagnostic_stage_fallback=True,
        ).lap_predictions.set_index("driver_id")
        actual, target_manifest = _load_target_after_frozen_forecasts(
            qualifying_path,
            expected_event_key=event_key,
            frozen_forecast_artifact=artifact,
            root=root,
        )
        evaluation_target_manifest_after_forecast.update(target_manifest)
        common = baseline_inference_ids.intersection(actual.index)
        if common.empty:
            raise ValueError(
                f"round {round_number} has no matched valid rehearsal/qualifying laps"
            )
        evaluated = predictions.copy()
        evaluated["quality_location_lap_p50"] = evaluated["driver_id"].map(
            location_diagnostic["lap_p50"]
        )
        evaluated["robust_residual_lap_p50"] = evaluated["driver_id"].map(
            robust_diagnostic["lap_p50"]
        )
        evaluated["selected_residual_enabled"] = enable_selected_residual
        evaluated["baseline_lap_p50"] = evaluated["driver_id"].map(baseline_by_driver)
        evaluated["raw_rehearsal_lap_p50"] = evaluated["driver_id"].map(
            raw_rehearsal_by_driver
        )
        evaluated["baseline_lap_p05"] = evaluated["driver_id"].map(
            baseline_indexed["lap_p05"]
        )
        evaluated["baseline_lap_p90"] = evaluated["driver_id"].map(
            baseline_indexed["lap_p90"]
        )
        evaluated["baseline_interval_status"] = evaluated["driver_id"].map(
            baseline_indexed["interval_status"]
        )
        evaluated[ACTUAL_LAP_COLUMN] = evaluated["driver_id"].map(actual)
        evaluated["target_observed"] = evaluated[ACTUAL_LAP_COLUMN].notna()
        evaluated["baseline_available"] = evaluated["baseline_lap_p50"].notna()
        challenger_scored = evaluated.loc[
            evaluated["target_observed"] & evaluated["lap_p50"].notna()
        ].copy()
        scored = challenger_scored.loc[challenger_scored["baseline_available"]].copy()
        scored["round"] = int(round_number)
        scored["absolute_error_seconds"] = (
            scored["lap_p50"] - scored[ACTUAL_LAP_COLUMN]
        ).abs()
        scored["baseline_absolute_error_seconds"] = (
            scored["baseline_lap_p50"] - scored[ACTUAL_LAP_COLUMN]
        ).abs()
        scored["raw_rehearsal_absolute_error_seconds"] = (
            scored["raw_rehearsal_lap_p50"] - scored[ACTUAL_LAP_COLUMN]
        ).abs()
        challenger_scored["round"] = int(round_number)
        challenger_scored["absolute_error_seconds"] = (
            challenger_scored["lap_p50"] - challenger_scored[ACTUAL_LAP_COLUMN]
        ).abs()
        metrics = _event_metrics(scored)
        observed_target_coverage = float(len(challenger_scored) / len(actual)) if len(actual) else 0.0
        comparison_rows = evaluated.assign(
            challenger_absolute_error_seconds=(
                evaluated["lap_p50"] - evaluated[ACTUAL_LAP_COLUMN]
            ).abs(),
            baseline_absolute_error_seconds=(
                evaluated["baseline_lap_p50"] - evaluated[ACTUAL_LAP_COLUMN]
            ).abs(),
            raw_rehearsal_absolute_error_seconds=(
                evaluated["raw_rehearsal_lap_p50"]
                - evaluated[ACTUAL_LAP_COLUMN]
            ).abs(),
        )
        if event_key in set(partitions["calibration"]):
            def interval_rows(values: pd.Series) -> pd.DataFrame:
                rows = pd.DataFrame(
                    {
                        "event_key": event_key,
                        "driver_id": inference_ids.astype(str),
                        "rehearsal_source": source,
                        "lap_p50": inference_ids.astype(str).map(values),
                        ACTUAL_LAP_COLUMN: inference_ids.astype(str).map(actual),
                    }
                )
                return rows.dropna(subset=["lap_p50", ACTUAL_LAP_COLUMN])

            baseline_interval_calibration_rows.append(
                interval_rows(baseline_indexed["lap_p50"])
            )
            location_interval_calibration_rows.append(
                interval_rows(location_diagnostic["lap_p50"])
            )
            robust_interval_calibration_rows.append(
                interval_rows(robust_diagnostic["lap_p50"])
            )
        event_payloads.append(
            {
                "round": int(round_number),
                "event_key": int(event_key),
                "event_directory": str(round_dir.relative_to(_repo_root())),
                "rehearsal_source": source,
                "rehearsal_file": str(rehearsal_path.relative_to(_repo_root())),
                "target_file": str(qualifying_path.relative_to(_repo_root())),
                "rehearsal_driver_count": int(len(rehearsal)),
                "quality_aware_inference_driver_count": int(len(inference_ids)),
                "target_driver_count": int(len(actual)),
                "matched_driver_count": int(len(common)),
                "causal_inference_driver_count": int(len(inference_ids)),
                "evaluation_union_driver_count": int(len(inference_ids.union(actual.index))),
                "target_observed_given_inference_rate": float(
                    evaluated["target_observed"].mean()
                ),
                "inference_coverage_of_observed_target_rate": observed_target_coverage,
                "end_to_end_scored_union_rate": float(
                    len(challenger_scored) / len(inference_ids.union(actual.index))
                ),
                "missing_target_drivers": sorted(set(inference_ids) - set(actual.index)),
                "missing_rehearsal_drivers": sorted(set(actual.index) - set(inference_ids)),
                "baseline_training_event_keys": list(baseline_model.training_event_keys),
                "challenger_training_event_keys": list(challenger_model.training_event_keys),
                "event_partition_role": (
                    "point_fit"
                    if event_key in set(partitions["selection"])
                    else "calibration"
                    if event_key in set(partitions["calibration"])
                    else "audit"
                ),
                "interval_calibration_event_keys": list(calibration_keys_available),
                "shared_forecast_artifact": artifact,
                "residual_selector": residual_selector,
                "metrics": metrics,
                "prediction_vs_reality": comparison_rows.to_dict(orient="records"),
                "predictions": comparison_rows.to_dict(orient="records"),
            }
        )
        all_scored.append(scored)
        all_challenger_scored.append(challenger_scored)
        # Only the two declared point-fit events enter the structural fit.
        # Four later events calibrate both the held-out median correction and
        # interval residuals. Audit outcomes are never reused; the complete
        # point-plus-interval predictor is frozen before the audit block.
        if event_key in set(partitions["selection"]):
            baseline_history_parts.append(
                pd.DataFrame(
                    {
                        "event_key": event_key,
                        "driver_id": common.astype(str),
                        "rehearsal_source": source,
                        "rehearsal_lap_time_seconds": rehearsal.loc[common].to_numpy(dtype=float),
                        ACTUAL_LAP_COLUMN: actual.loc[common].to_numpy(dtype=float),
                        "history_weight": 1.0,
                        "weak_transfer_prior": False,
                    }
                )
            )
            challenger_history_parts.append(
                _label_quality_history(
                    quality_features,
                    actual=actual,
                    qualifying_laps_path=qualifying_path,
                    history_weight=1.0,
                    weak_transfer_prior=False,
                )
            )

    joined = pd.concat(all_scored, ignore_index=True)
    joined_challenger = pd.concat(all_challenger_scored, ignore_index=True)
    audit_key_set = set(partitions["audit"])
    audit_payloads = [
        payload for payload in event_payloads if int(payload["event_key"]) in audit_key_set
    ]
    event_metrics = [payload["metrics"] for payload in audit_payloads]
    joined = joined.loc[pd.to_numeric(joined["event_key"], errors="coerce").isin(audit_key_set)]
    joined_challenger = joined_challenger.loc[
        pd.to_numeric(joined_challenger["event_key"], errors="coerce").isin(audit_key_set)
    ]
    paired_retained = _paired_bootstrap(
        [float(item["p50_mae_seconds"]) for item in event_metrics],
        [float(item["baseline_p50_mae_seconds"]) for item in event_metrics],
        samples=int(bootstrap_samples),
        seed=int(bootstrap_seed),
    )
    paired_raw = _paired_bootstrap(
        [float(item["p50_mae_seconds"]) for item in event_metrics],
        [float(item["raw_rehearsal_mae_seconds"]) for item in event_metrics],
        samples=int(bootstrap_samples),
        seed=int(bootstrap_seed),
    )
    stability = _event_stability(event_metrics)
    point_mae_aggregate = _aggregate_point_mae(event_metrics)
    challenger_event_mae = point_mae_aggregate[
        "conditional_event_mean_p50_mae_seconds"
    ]
    baseline_event_mae = point_mae_aggregate[
        "conditional_event_mean_baseline_p50_mae_seconds"
    ]
    relative_mae_gain = (
        float((baseline_event_mae - challenger_event_mae) / baseline_event_mae)
        if baseline_event_mae > 0.0
        else float("nan")
    )
    target_rows = sum(item["target_driver_count"] for item in audit_payloads)
    observed_target_coverage = float(len(joined_challenger) / target_rows)
    challenger_interval_summary = _interval_block_summary(
        joined_challenger,
        lower_column="lap_p05",
        upper_column="lap_p90",
        status_column="interval_status",
    )
    baseline_interval_summary = _interval_block_summary(
        joined_challenger,
        lower_column="baseline_lap_p05",
        upper_column="baseline_lap_p90",
        status_column="baseline_interval_status",
    )
    interval_coverage = float(challenger_interval_summary["pooled_coverage"])
    interval_width = float(
        challenger_interval_summary["row_weighted_mean_width_seconds"]
    )
    validated_interval_rate = float(
        challenger_interval_summary["validated_interval_row_rate"]
    )
    baseline_interval_coverage = float(
        baseline_interval_summary["pooled_coverage"]
    )
    baseline_interval_width = float(
        baseline_interval_summary["row_weighted_mean_width_seconds"]
    )
    baseline_validated_interval_rate = float(
        baseline_interval_summary["validated_interval_row_rate"]
    )
    fastest_non_worse = float(np.mean([item["fastest_driver_hit"] for item in event_metrics])) >= float(
        np.mean([item["baseline_fastest_driver_hit"] for item in event_metrics])
    )
    top3_non_worse = float(np.mean([item["top3_overlap_rate"] for item in event_metrics])) >= float(
        np.mean([item["baseline_top3_overlap_rate"] for item in event_metrics])
    )
    weekend_stratum_deltas: dict[str, list[float]] = {"standard": [], "sprint": []}
    for payload in audit_payloads:
        stratum = (
            "sprint"
            if str(payload["rehearsal_source"]) == "sprint_qualifying"
            else "standard"
        )
        weekend_stratum_deltas[stratum].append(
            float(payload["metrics"]["challenger_minus_baseline_mae_seconds"])
        )
    weekend_stratum_mean_deltas = {
        name: float(np.mean(values))
        for name, values in weekend_stratum_deltas.items()
        if values
    }
    all_weekend_strata_improve = bool(
        {"standard", "sprint"}.issubset(weekend_stratum_mean_deltas)
        and all(
            weekend_stratum_mean_deltas[name] < 0.0
            for name in ("standard", "sprint")
        )
    )
    point_promotion_gates = _best_lap_point_promotion_gates(
        relative_mae_gain=relative_mae_gain,
        paired_retained=paired_retained,
        observed_target_coverage=observed_target_coverage,
        all_weekend_strata_improve=all_weekend_strata_improve,
        stability=stability,
    )
    point_retained = bool(all(point_promotion_gates.values()))
    # Interval evidence is a separate forecasting product.  If the challenger
    # point head loses, evaluate the calibrated interval around the retained
    # baseline point instead of coupling interval promotion to a rejected head.
    retained_interval_source = "challenger" if point_retained else "retained_baseline"
    retained_interval_summary = (
        challenger_interval_summary if point_retained else baseline_interval_summary
    )
    retained_interval_coverage = (
        interval_coverage if point_retained else baseline_interval_coverage
    )
    retained_interval_width = interval_width if point_retained else baseline_interval_width
    retained_validated_interval_rate = (
        validated_interval_rate
        if point_retained
        else baseline_validated_interval_rate
    )
    baseline_interval_calibration_event_count = len(
        {
            int(value)
            for frame in baseline_interval_calibration_rows
            for value in pd.to_numeric(frame["event_key"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        }
    )
    challenger_interval_calibration_event_count = len(
        {
            int(value)
            for frame in location_interval_calibration_rows
            for value in pd.to_numeric(frame["event_key"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        }
    )
    retained_interval_calibration_event_count = (
        challenger_interval_calibration_event_count
        if point_retained
        else baseline_interval_calibration_event_count
    )
    interval_width_comparison = _interval_width_comparison(
        retained_interval_summary,
        baseline_interval_summary,
        candidate_label=retained_interval_source,
        comparator_label="retained_baseline",
    )
    retained_event_balanced_coverage = float(
        retained_interval_summary["event_balanced_coverage"]
    )
    interval_promotion_gates = _best_lap_interval_promotion_gates(
        retained_interval_summary=retained_interval_summary,
        retained_interval_calibration_event_count=(
            retained_interval_calibration_event_count
        ),
        interval_width_comparison=interval_width_comparison,
    )
    promotion_gates = {**point_promotion_gates, **interval_promotion_gates}
    intervals_promoted = bool(all(interval_promotion_gates.values()))
    if set(input_files) != set(selected_input_files):
        raise RuntimeError("Best Lap accessed an unexpected input-file set")
    input_manifest_before = dict(
        sorted(
            {
                **inference_input_manifest_before,
                **evaluation_target_manifest_after_forecast,
            }.items()
        )
    )
    if set(input_manifest_before) != {
        str(path.relative_to(root)) for path in set(selected_input_files)
    }:
        raise RuntimeError("Best Lap did not hash every target after its forecast freeze")
    _assert_manifest_unchanged(
        implementation_manifest,
        root=root,
        label="Best Lap implementation",
    )
    _assert_manifest_unchanged(
        input_manifest_before,
        root=root,
        label="Best Lap input data",
    )
    return _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _utc_now(),
            "mode": "best_estimated_lap_time",
            "model": MODEL_NAME if enable_selected_residual else QUALITY_LOCATION_MODEL_NAME,
            "robust_residual_diagnostic_model": MODEL_NAME,
            "baseline_model": BASELINE_MODEL_NAME,
            "model_family": (
                "shared_latent_lap_nested_driver_hurdles_huber_empirical_residual_intervals"
                if enable_selected_residual
                else "shared_latent_lap_nested_driver_hurdles_location_empirical_residual_intervals"
            ),
            "target_contract": ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT,
            "target_semantics": TARGET_CONTRACT_SEMANTICS[
                ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT
            ],
            "target_definition": "fastest valid session-end Grand Prix qualifying lap per driver",
            "theoretical_sector_floor_is_target": False,
            "invocation": {
                "year": int(year),
                "rounds": [_round_number(path) for path in selected],
                "bootstrap_samples": int(bootstrap_samples),
                "bootstrap_seed": int(bootstrap_seed),
                "use_weak_transfer_priors": bool(use_weak_transfer_priors),
            },
            "provenance": {
                "repo_root": str(root),
                "git_head": _git_head(root),
                "git_dirty": _git_dirty(root),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "source": "locked_local_weekend_csv",
            },
            "protocol": {
                "event_specific_pace_training": "first_two_target_season_point_fit_events_only",
                "older_season_use": (
                    "research_only_weak_invariant_transition_priors"
                    if use_weak_transfer_priors
                    else "none_same_season_only"
                ),
                "training_window": (
                    "structural_point_fit_frozen_after_events_1_2; events_3_6_held_out_"
                    "median_and_interval_calibration; complete_predictor_frozen_before_"
                    "events_7_plus_current_replay_diagnostic; current_replay_does_not_"
                    "fit_or_select_on_events_7_plus"
                ),
                "within_run_audit_outcomes_reused_for_fit_or_selection": False,
                "prospective_development_evidence": False,
                "evidence_role": (
                    "postdevelopment_replay_diagnostic_after_prior_R7_R9_inspection"
                ),
                "round_order": "ascending_sequential",
                "standard_weekend_rehearsal": "FP3",
                "sprint_weekend_rehearsal": "Sprint Qualifying",
                "supported_weekend_era": "2024_plus",
                "target_outcome_available_to_inference": False,
                "validation_unit": "complete_chronological_event_block",
                "event_partitions": {
                    name: list(values) for name, values in partitions.items()
                },
                "partition_validation_issues": [],
                "interval_quantile_semantics": {
                    "lap_p05": 0.05,
                    "lap_p50": 0.50,
                    "lap_p90": 0.90,
                    "nominal_mass": INTERVAL_NOMINAL_COVERAGE,
                },
                "interval_promotion_thresholds": {
                    "minimum_calibration_events": int(
                        MINIMUM_INTERVAL_PROMOTION_EVENTS
                    ),
                    "minimum_audit_events": int(MINIMUM_INTERVAL_AUDIT_EVENTS),
                    "event_balanced_coverage_range": [
                        INTERVAL_NOMINAL_COVERAGE
                        - INTERVAL_EVENT_BALANCED_COVERAGE_TOLERANCE,
                        INTERVAL_NOMINAL_COVERAGE
                        + INTERVAL_EVENT_BALANCED_COVERAGE_TOLERANCE,
                    ],
                    "minimum_per_event_coverage": (
                        MINIMUM_PER_EVENT_INTERVAL_COVERAGE
                    ),
                    "minimum_per_weekend_stratum_coverage": (
                        MINIMUM_PER_STRATUM_INTERVAL_COVERAGE
                    ),
                    "required_weekend_strata": list(REQUIRED_WEEKEND_STRATA),
                    "maximum_event_balanced_width_ratio": (
                        MAXIMUM_EVENT_BALANCED_WIDTH_RATIO
                    ),
                    "maximum_per_event_width_ratio": (
                        MAXIMUM_PER_EVENT_WIDTH_RATIO
                    ),
                    "maximum_per_weekend_stratum_width_ratio": (
                        MAXIMUM_PER_STRATUM_WIDTH_RATIO
                    ),
                    "finite_sample_rationale": (
                        "70_percent is an event-level floor near the nominal 85_percent "
                        "minus two binomial standard errors for a roughly 20-driver field; "
                        "75_percent is the format-level floor after event balancing"
                    ),
                },
                "weak_transfer_prior_summary": prior_summary,
                "robust_residual_selector": residual_selector,
                "frozen_structural_point_predictor_sha256": (
                    frozen_structural_point_predictor_sha256
                ),
                "frozen_complete_predictor_sha256": frozen_complete_predictor_sha256,
                "held_out_interval_calibration_uses_final_predictor_residuals": True,
                "audit_weekend_stratum_mean_deltas_seconds": weekend_stratum_mean_deltas,
                "baseline_and_challenger_scored_on_same_rows": True,
                "deleted_laps_are_potential_only": True,
                "one_heavy_training_job_at_a_time": True,
            },
            "aggregate": {
                "rounds": len(audit_payloads),
                "all_prediction_rounds": len(event_payloads),
                "rows": int(len(joined_challenger)),
                "paired_baseline_rows": int(len(joined)),
                "causal_inference_rows": int(
                    sum(item["causal_inference_driver_count"] for item in audit_payloads)
                ),
                "observed_target_rows": int(
                    sum(item["target_driver_count"] for item in audit_payloads)
                ),
                "evaluation_union_rows": int(
                    sum(item["evaluation_union_driver_count"] for item in audit_payloads)
                ),
                **point_mae_aggregate,
                "conditional_row_weighted_p50_mae_seconds": float(
                    joined_challenger["absolute_error_seconds"].mean()
                ),
                "relative_event_mean_mae_gain": relative_mae_gain,
                "conditional_event_mean_spearman": float(
                    np.mean([item["spearman"] for item in event_metrics])
                ),
                "conditional_fastest_driver_hit_rate": float(
                    np.mean([item["fastest_driver_hit"] for item in event_metrics])
                ),
                "baseline_fastest_driver_hit_rate": float(
                    np.mean([item["baseline_fastest_driver_hit"] for item in event_metrics])
                ),
                "conditional_top3_overlap_rate": float(
                    np.mean([item["top3_overlap_rate"] for item in event_metrics])
                ),
                "baseline_top3_overlap_rate": float(
                    np.mean([item["baseline_top3_overlap_rate"] for item in event_metrics])
                ),
                "target_observed_given_inference_rate": float(
                    len(joined_challenger)
                    / sum(item["causal_inference_driver_count"] for item in audit_payloads)
                ),
                "inference_coverage_of_observed_target_rate": observed_target_coverage,
                "end_to_end_scored_union_rate": float(
                    len(joined_challenger)
                    / sum(item["evaluation_union_driver_count"] for item in audit_payloads)
                ),
                "interval_rows": int(
                    challenger_interval_summary["validated_interval_rows"]
                ),
                "validated_interval_row_rate": validated_interval_rate,
                "interval_coverage": interval_coverage,
                "interval_mean_width_seconds": interval_width,
                "interval_event_balanced_coverage": float(
                    challenger_interval_summary["event_balanced_coverage"]
                ),
                "interval_event_balanced_mean_width_seconds": float(
                    challenger_interval_summary[
                        "event_balanced_mean_width_seconds"
                    ]
                ),
                "baseline_interval_coverage": baseline_interval_coverage,
                "baseline_interval_mean_width_seconds": baseline_interval_width,
                "baseline_interval_event_balanced_coverage": float(
                    baseline_interval_summary["event_balanced_coverage"]
                ),
                "baseline_interval_event_balanced_mean_width_seconds": float(
                    baseline_interval_summary[
                        "event_balanced_mean_width_seconds"
                    ]
                ),
                "retained_interval_source": retained_interval_source,
                "retained_interval_calibration_event_count": (
                    retained_interval_calibration_event_count
                ),
                "retained_interval_coverage": retained_interval_coverage,
                "retained_interval_mean_width_seconds": retained_interval_width,
                "retained_interval_event_balanced_coverage": (
                    retained_event_balanced_coverage
                ),
                "retained_interval_event_balanced_mean_width_seconds": float(
                    retained_interval_summary[
                        "event_balanced_mean_width_seconds"
                    ]
                ),
            },
            "paired_event_bootstrap_vs_retained_baseline_conditional_matched_population": (
                paired_retained
            ),
            "paired_event_bootstrap_vs_raw_rehearsal_conditional_matched_population": (
                paired_raw
            ),
            "event_stability": stability,
            "ranking_diagnostics_not_point_promotion_gates": {
                "fastest_driver_non_worse": bool(fastest_non_worse),
                "top3_non_worse": bool(top3_non_worse),
                "reason": (
                    "fastest-driver and top-three ranking are Qualifying-mode "
                    "objectives, not proper losses for the per-driver seconds target"
                ),
            },
            "interval_evidence": {
                "challenger": challenger_interval_summary,
                "retained_baseline": baseline_interval_summary,
                "retained_product": retained_interval_summary,
                "retained_product_source": retained_interval_source,
                "width_comparison_vs_retained_baseline": (
                    interval_width_comparison
                ),
            },
            "promotion_gates": promotion_gates,
            "point_promotion_gates": point_promotion_gates,
            "interval_promotion_gates": interval_promotion_gates,
            "deep_model_telemetry_readiness": telemetry_readiness,
            "decision": {
                "conditional_point_estimate_retained": point_retained,
                "full_mode_point_promoted": point_retained,
                "probabilistic_intervals_promoted": intervals_promoted,
                "deep_model_promoted": False,
                **deep_model_decision,
                "reason": (
                    "quality-aware selected challenger cleared every declared promotion gate"
                    if point_retained
                    else "quality-aware selected challenger retained as diagnostic because one or more declared gates failed"
                ),
                "failed_promotion_gates": [
                    name for name, passed in promotion_gates.items() if not passed
                ],
                "point_blockers": [
                    name
                    for name, passed in point_promotion_gates.items()
                    if not passed
                ],
                "full_mode_blockers": [
                    name for name, passed in promotion_gates.items() if not passed
                ],
                "interval_blockers": [
                    name
                    for name, passed in interval_promotion_gates.items()
                    if not passed
                ],
            },
            "events": event_payloads,
            "shared_forecast_artifacts": shared_forecast_artifacts,
            "input_manifest": input_manifest_before,
            "implementation_manifest": implementation_manifest,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weekends-dir",
        default=str(_repo_root() / "data/f1/raw/weekends"),
    )
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--rounds", default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=200_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260711)
    parser.add_argument(
        "--tcn-evidence",
        default="auto",
        help=(
            "completed true-TCN research JSON; 'auto' discovers the canonical "
            "year-specific artifact and otherwise leaves the model unevaluated"
        ),
    )
    transfer = parser.add_mutually_exclusive_group()
    transfer.add_argument(
        "--use-weak-transfer-priors",
        action="store_true",
        help=(
            "research-only ablation: enable recency-weighted older-season "
            "transition priors; production evidence defaults to same-season only"
        ),
    )
    transfer.add_argument(
        "--no-weak-transfer-priors",
        action="store_false",
        dest="use_weak_transfer_priors",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(use_weak_transfer_priors=False)
    parser.add_argument(
        "--output",
        default=str(
            _repo_root()
            / "artifacts/backtests/f1/best_estimated_lap/2026_walk_forward_quality_aware_huber_v2.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rounds = None
    if str(args.rounds).strip().lower() != "auto":
        rounds = [int(value.strip()) for value in str(args.rounds).split(",") if value.strip()]
    tcn_evidence_path = (
        None
        if str(args.tcn_evidence).strip().lower() == "auto"
        else Path(args.tcn_evidence).expanduser().resolve()
    )
    payload = run_backtest(
        weekends_dir=Path(args.weekends_dir).expanduser().resolve(),
        year=int(args.year),
        rounds=rounds,
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
        use_weak_transfer_priors=bool(args.use_weak_transfer_priors),
        tcn_evidence_path=tcn_evidence_path,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(output), "aggregate": payload["aggregate"], "decision": payload["decision"]}, indent=2))


if __name__ == "__main__":
    main()
