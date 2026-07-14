#!/usr/bin/env python3
"""Execute the fixed post-development 2026 TCN diagnostic matrix.

Prior outer-fold observations informed the profile design.  The checked-in plan
fixes execution order and an equal-architecture sham before this durable run,
but the matrix does not select a winner and cannot promote a model.  Runs are
deliberately sequential for the 16 GB workstation.
"""

from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from packages.f1.data.providers.telemetry_cache import sha256_file
from packages.f1.data.providers.telemetry_supervised import canonical_sha256
from packages.sports_core.paths import find_repo_root
from run_prequal_telemetry_tcn_research import (
    SCHEMA_VERSION as TCN_SCHEMA_VERSION,
    TELEMETRY_INPUT_OBSERVED,
    TELEMETRY_INPUT_ZERO_ABLATION,
    TCNResearchConfig,
    _validate_research_config,
    build_research_artifact,
    normalize_profile_design_provenance,
)


PLAN_SCHEMA_VERSION = "f1_prequal_telemetry_tcn_sensitivity_plan_v1"
MATRIX_SCHEMA_VERSION = "f1_prequal_telemetry_tcn_sensitivity_matrix_v1"
EXPECTED_PROFILE_IDS: tuple[str, ...] = (
    "d0_control",
    "d1_optimizer_primary",
    "d2_lower_capacity_broad_receptive_field",
    "d3_primary_seed_stability",
    "d1_zero_telemetry_static_anchor_sham",
    "d4_posthoc_lr_1e4",
    "d4_posthoc_lr_1e4_zero_telemetry_sham",
)
OUTER_TARGET_INFORMED_PROFILE_IDS: tuple[str, ...] = EXPECTED_PROFILE_IDS[1:]
REFERENCE_PROFILE_ID = "d1_optimizer_primary"
SHAM_PROFILE_ID = "d1_zero_telemetry_static_anchor_sham"
SEED_STABILITY_PROFILE_ID = "d3_primary_seed_stability"
ARCHITECTURE_PROFILE_ID = "d2_lower_capacity_broad_receptive_field"
POSTHOC_PROFILE_ID = "d4_posthoc_lr_1e4"
POSTHOC_SHAM_PROFILE_ID = "d4_posthoc_lr_1e4_zero_telemetry_sham"


class TCNSensitivityError(ValueError):
    pass


def _resolve_repo_path(value: object, *, root: Path, label: str) -> Path:
    text = str(value or "").strip()
    path = Path(text).expanduser()
    if not text or path.is_absolute():
        raise TCNSensitivityError(f"{label} must be a repository-relative path")
    resolved = (root / path).resolve()
    try:
        canonical = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise TCNSensitivityError(f"{label} escapes the repository") from exc
    if canonical != path.as_posix():
        raise TCNSensitivityError(f"{label} must be canonical repository-relative")
    return resolved


def _config_from_mapping(payload: Mapping[str, Any]) -> TCNResearchConfig:
    expected = {field.name for field in fields(TCNResearchConfig)}
    if set(payload) != expected:
        raise TCNSensitivityError(
            "profile config fields do not exactly match TCNResearchConfig"
        )
    values = dict(payload)
    dilations = values.get("dilations")
    if not isinstance(dilations, list):
        raise TCNSensitivityError("profile dilations must be a JSON list")
    values["dilations"] = tuple(int(value) for value in dilations)
    config = TCNResearchConfig(**values)
    _validate_research_config(config)
    return config


def _only_config_difference(
    left: TCNResearchConfig,
    right: TCNResearchConfig,
    *,
    allowed_fields: set[str],
) -> bool:
    left_values = asdict(left)
    right_values = asdict(right)
    return all(
        left_values[field] == right_values[field]
        for field in left_values
        if field not in allowed_fields
    ) and all(
        left_values[field] != right_values[field]
        for field in allowed_fields
    )


def _load_plan(path: Path, *, root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TCNSensitivityError(f"invalid sensitivity plan {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TCNSensitivityError("sensitivity plan must be an object")
    if payload.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise TCNSensitivityError("unsupported sensitivity plan schema")
    if int(payload.get("year") or 0) != 2026:
        raise TCNSensitivityError("sensitivity plan must target 2026")
    decision = payload.get("decision_policy")
    if not isinstance(decision, Mapping):
        raise TCNSensitivityError("sensitivity decision policy is missing")
    required_decision = {
        "reference_profile_id": REFERENCE_PROFILE_ID,
        "reference_profile_fixed_before_durable_matrix_execution": True,
        "profile_design_provenance_declared_per_profile": True,
        "durable_matrix_results_used_to_select_reference_profile": False,
        "matrix_interpretation": "descriptive_parameter_and_input_ablation_only",
        "promotion_allowed_from_this_matrix": False,
        "posthoc_profiles_never_selection_evidence": True,
    }
    if any(decision.get(key) != value for key, value in required_decision.items()):
        raise TCNSensitivityError("sensitivity decision policy is not fail-closed")
    if decision.get("outer_target_informed_profile_ids") != list(
        OUTER_TARGET_INFORMED_PROFILE_IDS
    ):
        raise TCNSensitivityError(
            "sensitivity decision policy has an invalid outer-target profile set"
        )
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, list):
        raise TCNSensitivityError("sensitivity profiles are missing")
    profile_ids = [str(profile.get("profile_id") or "") for profile in raw_profiles]
    if tuple(profile_ids) != EXPECTED_PROFILE_IDS:
        raise TCNSensitivityError("sensitivity profile order/identity changed")
    profiles: list[dict[str, Any]] = []
    output_paths: set[Path] = set()
    for raw in raw_profiles:
        if not isinstance(raw, Mapping):
            raise TCNSensitivityError("sensitivity profile must be an object")
        output_path = _resolve_repo_path(
            raw.get("output_path"), root=root, label="profile output_path"
        )
        if output_path in output_paths:
            raise TCNSensitivityError("sensitivity output paths must be unique")
        output_paths.add(output_path)
        config_payload = raw.get("config")
        if not isinstance(config_payload, Mapping):
            raise TCNSensitivityError("sensitivity profile config is missing")
        provenance_payload = raw.get("profile_design_provenance")
        if not isinstance(provenance_payload, Mapping):
            raise TCNSensitivityError(
                "sensitivity profile design provenance is missing"
            )
        try:
            design_provenance = normalize_profile_design_provenance(
                provenance_payload
            )
        except ValueError as exc:
            raise TCNSensitivityError(
                f"invalid sensitivity profile design provenance: {exc}"
            ) from exc
        profiles.append(
            {
                "profile_id": str(raw["profile_id"]),
                "purpose": str(raw.get("purpose") or ""),
                "evidence_role": str(raw.get("evidence_role") or ""),
                "profile_design_provenance": design_provenance,
                "output_path": output_path,
                "config": _config_from_mapping(config_payload),
            }
        )

    by_id = {profile["profile_id"]: profile for profile in profiles}
    reference = by_id[REFERENCE_PROFILE_ID]["config"]
    sham = by_id[SHAM_PROFILE_ID]["config"]
    if not _only_config_difference(
        reference,
        sham,
        allowed_fields={"telemetry_input_mode"},
    ):
        raise TCNSensitivityError(
            "zero-telemetry sham must equal reference except telemetry_input_mode"
        )
    if reference.telemetry_input_mode != TELEMETRY_INPUT_OBSERVED:
        raise TCNSensitivityError("reference profile must use observed telemetry")
    if sham.telemetry_input_mode != TELEMETRY_INPUT_ZERO_ABLATION:
        raise TCNSensitivityError("sham profile must zero telemetry")
    seed_check = by_id[SEED_STABILITY_PROFILE_ID]["config"]
    if not _only_config_difference(reference, seed_check, allowed_fields={"seed"}):
        raise TCNSensitivityError(
            "seed-stability profile must equal reference except seed"
        )
    architecture = by_id[ARCHITECTURE_PROFILE_ID]["config"]
    if not _only_config_difference(
        reference,
        architecture,
        allowed_fields={
            "hidden_channels",
            "kernel_size",
            "dilations",
            "head_hidden_dim",
        },
    ):
        raise TCNSensitivityError(
            "architecture profile must equal reference outside architecture fields"
        )
    posthoc = by_id[POSTHOC_PROFILE_ID]["config"]
    posthoc_sham = by_id[POSTHOC_SHAM_PROFILE_ID]["config"]
    if not _only_config_difference(
        posthoc,
        posthoc_sham,
        allowed_fields={"telemetry_input_mode"},
    ):
        raise TCNSensitivityError(
            "posthoc zero-telemetry sham must equal posthoc profile except input mode"
        )
    for profile in profiles:
        profile_id = str(profile["profile_id"])
        if profile_id == "d0_control":
            expected_role = "original_optimization_control_not_selection_evidence"
        elif profile_id.startswith("d4_"):
            expected_role = "posthoc_hypothesis_after_outer_sensitivity_observed"
        else:
            expected_role = (
                "posthoc_exploratory_profile_predeclared_before_durable_matrix_run"
            )
        if profile["evidence_role"] != expected_role:
            raise TCNSensitivityError("sensitivity evidence role is incorrect")
        provenance = profile["profile_design_provenance"]
        expected_outer_informed = profile_id != "d0_control"
        if (
            provenance["prior_outer_results_informed_profile_design"]
            is not expected_outer_informed
            or provenance["hyperparameters_tuned_on_outer_targets"]
            is not expected_outer_informed
        ):
            raise TCNSensitivityError(
                "sensitivity profile design history is inconsistent with its role"
            )
    return payload, profiles


def _artifact_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_artifact(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(_artifact_bytes(payload))
    temporary.replace(path)


def _event_mae_by_key(payload: Mapping[str, Any]) -> dict[int, float]:
    return {
        int(fold["target_event_key"]): float(
            fold["metrics"]["tcn_driver_correction"]["mae_seconds"]
        )
        for fold in payload["folds"]
    }


def _profile_record(
    profile: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    path = Path(profile["output_path"])
    serialized = _artifact_bytes(artifact)
    return {
        "profile_id": str(profile["profile_id"]),
        "purpose": str(profile["purpose"]),
        "evidence_role": str(profile["evidence_role"]),
        "profile_design_provenance": artifact[
            "profile_design_provenance"
        ],
        "output_path": path.relative_to(root).as_posix(),
        "output_sha256": hashlib.sha256(serialized).hexdigest(),
        "output_size_bytes": len(serialized),
        "artifact_schema_version": str(artifact["schema_version"]),
        "artifact_payload_sha256": str(artifact["artifact_payload_sha256"]),
        "training_config": artifact["training_config"],
        "architecture": artifact["architecture"],
        "capacity": artifact["capacity"],
        "model_input_ablation": artifact["model_input_ablation"],
        "summary": artifact["summary"],
        "per_event_tcn_mae_seconds": {
            str(key): value for key, value in _event_mae_by_key(artifact).items()
        },
    }


def _matrix_implementation_manifest(
    *,
    root: Path,
    plan_path: Path,
) -> list[dict[str, Any]]:
    paths = (
        Path(__file__).resolve(),
        plan_path.resolve(),
        Path(__file__).with_name(
            "run_prequal_telemetry_tcn_research.py"
        ).resolve(),
    )
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for path in paths
    ]


def run_matrix(
    *,
    plan_path: Path,
    root: Path,
    generated_at: str | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = root.expanduser().resolve()
    plan_source = plan_path.expanduser().resolve()
    try:
        plan_relative = plan_source.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise TCNSensitivityError("sensitivity plan must be inside repository") from exc
    plan_sha256 = sha256_file(plan_source)
    plan, profiles = _load_plan(plan_source, root=repo_root)
    implementation_manifest = _matrix_implementation_manifest(
        root=repo_root,
        plan_path=plan_source,
    )
    source_manifest = _resolve_repo_path(
        plan.get("source_manifest"), root=repo_root, label="source_manifest"
    )
    matrix_output = (
        output_path.expanduser().resolve()
        if output_path is not None
        else _resolve_repo_path(
            plan.get("matrix_output"), root=repo_root, label="matrix_output"
        )
    )
    try:
        matrix_output.relative_to(repo_root)
    except ValueError as exc:
        raise TCNSensitivityError("matrix output must be inside repository") from exc
    timestamp = generated_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )

    artifacts: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        artifact = build_research_artifact(
            manifest_path=source_manifest,
            root=repo_root,
            config=profile["config"],
            generated_at=timestamp,
            profile_design_provenance=profile[
                "profile_design_provenance"
            ],
        )
        design_provenance = profile["profile_design_provenance"]
        validation_contract = artifact.get("validation_contract")
        if artifact.get("profile_design_provenance") != design_provenance:
            raise TCNSensitivityError(
                "TCN artifact did not preserve profile design provenance"
            )
        if not isinstance(validation_contract, Mapping) or any(
            validation_contract.get(field) is not design_provenance[field]
            for field in (
                "prior_outer_results_informed_profile_design",
                "same_outer_evaluation_targets_seen_before_profile_freeze",
                "hyperparameters_tuned_on_outer_targets",
            )
        ):
            raise TCNSensitivityError(
                "TCN artifact validation contract contradicts profile design provenance"
            )
        artifact["sensitivity_profile"] = {
            "profile_id": str(profile["profile_id"]),
            "evidence_role": str(profile["evidence_role"]),
            "profile_design_provenance": design_provenance,
            "plan_path": plan_relative,
            "plan_sha256": plan_sha256,
            "matrix_output_path": matrix_output.relative_to(repo_root).as_posix(),
            "reference_profile_fixed_before_durable_matrix_execution": True,
            "prior_outer_results_informed_profile_design": design_provenance[
                "prior_outer_results_informed_profile_design"
            ],
            "durable_matrix_results_used_to_select_reference_profile": False,
            "promotion_allowed_from_matrix": False,
            "completed_matrix_required_for_downstream_consumption": True,
        }
        artifact["artifact_payload_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in artifact.items()
                if key != "artifact_payload_sha256"
            }
        )
        artifacts[str(profile["profile_id"])] = artifact

    reference = artifacts[REFERENCE_PROFILE_ID]
    sham = artifacts[SHAM_PROFILE_ID]
    seed_check = artifacts[SEED_STABILITY_PROFILE_ID]
    posthoc = artifacts[POSTHOC_PROFILE_ID]
    posthoc_sham = artifacts[POSTHOC_SHAM_PROFILE_ID]
    if reference["architecture"] != sham["architecture"]:
        raise TCNSensitivityError("reference and sham architectures differ")
    if (
        reference["capacity"]["trainable_scalar_parameter_count"]
        != sham["capacity"]["trainable_scalar_parameter_count"]
    ):
        raise TCNSensitivityError("reference and sham parameter counts differ")
    if reference["scored_event_keys"] != sham["scored_event_keys"]:
        raise TCNSensitivityError("reference and sham scored events differ")
    reference_seeds = [
        int(fold["inner_selection_and_early_stopping"]["initialization_seed"])
        for fold in reference["folds"]
    ]
    sham_seeds = [
        int(fold["inner_selection_and_early_stopping"]["initialization_seed"])
        for fold in sham["folds"]
    ]
    if reference_seeds != sham_seeds:
        raise TCNSensitivityError("reference and sham fold seeds differ")

    reference_event = _event_mae_by_key(reference)
    sham_event = _event_mae_by_key(sham)
    seed_event = _event_mae_by_key(seed_check)
    posthoc_event = _event_mae_by_key(posthoc)
    posthoc_sham_event = _event_mae_by_key(posthoc_sham)
    event_keys = sorted(reference_event)
    if (
        event_keys != sorted(sham_event)
        or event_keys != sorted(seed_event)
        or event_keys != sorted(posthoc_event)
        or event_keys != sorted(posthoc_sham_event)
    ):
        raise TCNSensitivityError("comparison profiles do not share event keys")
    reference_minus_sham = [
        reference_event[event_key] - sham_event[event_key]
        for event_key in event_keys
    ]
    reference_minus_seed = [
        reference_event[event_key] - seed_event[event_key]
        for event_key in event_keys
    ]
    posthoc_minus_sham = [
        posthoc_event[event_key] - posthoc_sham_event[event_key]
        for event_key in event_keys
    ]
    if sha256_file(plan_source) != plan_sha256:
        raise TCNSensitivityError("sensitivity plan changed during matrix execution")
    if _matrix_implementation_manifest(
        root=repo_root,
        plan_path=plan_source,
    ) != implementation_manifest:
        raise TCNSensitivityError(
            "sensitivity implementation changed during matrix execution"
        )
    matrix: dict[str, Any] = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "generated_at": timestamp,
        "status": "postdevelopment_descriptive_matrix_complete_not_selection_evidence",
        "promotion_eligible": False,
        "deployment_changed": False,
        "tcn_artifact_schema_version": TCN_SCHEMA_VERSION,
        "plan": {
            "path": plan_relative,
            "sha256": plan_sha256,
            "schema_version": PLAN_SCHEMA_VERSION,
            "decision_policy": plan["decision_policy"],
        },
        "source_manifest": {
            "path": source_manifest.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(source_manifest),
        },
        "implementation_manifest": implementation_manifest,
        "implementation_manifest_sha256": canonical_sha256(
            implementation_manifest
        ),
        "execution": {
            "sequential_single_training_job": True,
            "profile_order": list(EXPECTED_PROFILE_IDS),
            "profile_design_provenance_declared_per_profile": True,
            "outer_target_informed_profile_ids": list(
                OUTER_TARGET_INFORMED_PROFILE_IDS
            ),
            "current_run_outer_fold_targets_used_for_fit_or_inner_selection": False,
            "durable_matrix_results_used_to_select_reference_profile": False,
            "reference_profile_id": REFERENCE_PROFILE_ID,
            "reference_profile_fixed_before_durable_matrix_execution": True,
            "matrix_selects_winner": False,
        },
        "profiles": [
            _profile_record(profile, artifacts[str(profile["profile_id"])], root=repo_root)
            for profile in profiles
        ],
        "comparisons": {
            "reference_vs_equal_architecture_zero_telemetry_sham": {
                "delta_definition": "reference_tcn_event_mae_minus_sham_event_mae",
                "event_keys": event_keys,
                "event_mae_deltas_seconds": reference_minus_sham,
                "mean_delta_seconds": float(np.mean(reference_minus_sham)),
                "median_delta_seconds": float(np.median(reference_minus_sham)),
                "reference_event_wins": int(
                    np.sum(np.asarray(reference_minus_sham) < 0.0)
                ),
                "sham_event_wins": int(
                    np.sum(np.asarray(reference_minus_sham) > 0.0)
                ),
                "outer_results_descriptive_only": True,
            },
            "reference_vs_fixed_seed_repeat": {
                "delta_definition": "reference_seed_event_mae_minus_repeat_seed_event_mae",
                "event_keys": event_keys,
                "event_mae_deltas_seconds": reference_minus_seed,
                "mean_absolute_event_delta_seconds": float(
                    np.mean(np.abs(reference_minus_seed))
                ),
                "outer_results_descriptive_only": True,
            },
            "posthoc_lr_1e4_vs_equal_architecture_zero_telemetry_sham": {
                "evidence_role": (
                    "posthoc_hypothesis_after_outer_sensitivity_observed"
                ),
                "delta_definition": "posthoc_tcn_event_mae_minus_posthoc_sham_event_mae",
                "event_keys": event_keys,
                "event_mae_deltas_seconds": posthoc_minus_sham,
                "mean_delta_seconds": float(np.mean(posthoc_minus_sham)),
                "posthoc_event_wins": int(
                    np.sum(np.asarray(posthoc_minus_sham) < 0.0)
                ),
                "never_selection_or_promotion_evidence": True,
            },
        },
    }
    matrix["artifact_payload_sha256"] = canonical_sha256(matrix)
    # Profiles are built fully in memory.  Recheck immutable inputs, publish
    # every profile atomically, and publish the matrix last.  Downstream
    # consumers require the completed matrix to bind the exact profile hash,
    # so a crash cannot make a partial profile consumable.
    if sha256_file(plan_source) != plan_sha256:
        raise TCNSensitivityError("sensitivity plan changed before publication")
    if _matrix_implementation_manifest(
        root=repo_root,
        plan_path=plan_source,
    ) != implementation_manifest:
        raise TCNSensitivityError(
            "sensitivity implementation changed before publication"
        )
    for profile in profiles:
        _write_artifact(
            Path(profile["output_path"]),
            artifacts[str(profile["profile_id"])],
        )
    _write_artifact(matrix_output, matrix)
    return matrix


def _parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed post-development sequential 2026 TCN diagnostic matrix."
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=root / "configs/f1/prequal_telemetry_tcn_sensitivity_2026.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--generated-at", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    root = find_repo_root()
    args = _parser(root).parse_args(argv)
    matrix = run_matrix(
        plan_path=args.plan,
        root=root,
        generated_at=args.generated_at,
        output_path=args.output,
    )
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (root / "artifacts/backtests/f1/telemetry/prequal_telemetry_tcn_sensitivity_matrix_v1_2026.json").resolve()
    )
    print(
        json.dumps(
            {
                "artifact": {
                    "path": str(output),
                    "sha256": sha256_file(output),
                    "size_bytes": int(output.stat().st_size),
                },
                "artifact_payload_sha256": matrix["artifact_payload_sha256"],
                "status": matrix["status"],
                "comparisons": matrix["comparisons"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
