from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import pytest

import run_best_estimated_lap_2026_backtest as best_runner
from packages.f1.data.providers.telemetry_cache import sha256_file
from packages.f1.data.providers.telemetry_supervised import (
    build_prequal_telemetry_supervised_manifest,
)
from test_prequal_telemetry_supervised import _write_fixture


def _rehash(payload: dict[str, object]) -> None:
    payload["artifact_payload_sha256"] = best_runner._canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "artifact_payload_sha256"
        }
    )


def _evidence_payload() -> dict[str, object]:
    prediction_rows: list[dict[str, object]] = [
        {
            "event_key": 202605,
            "driver_id": "AAA",
            "actual_lap_time_seconds": 80.0,
            "source_shift_baseline_predicted_lap_time_seconds": 81.0,
            "tcn_driver_correction_predicted_lap_time_seconds": 81.2,
            "locked_selected_policy_predicted_lap_time_seconds": 81.0,
            "locked_selected_candidate_id": "zero_telemetry_correction",
        },
        {
            "event_key": 202605,
            "driver_id": "BBB",
            "actual_lap_time_seconds": 82.0,
            "source_shift_baseline_predicted_lap_time_seconds": 81.0,
            "tcn_driver_correction_predicted_lap_time_seconds": 80.8,
            "locked_selected_policy_predicted_lap_time_seconds": 81.0,
            "locked_selected_candidate_id": "zero_telemetry_correction",
        },
    ]
    for row in prediction_rows:
        row["prediction_sha256"] = best_runner._canonical_sha256(row)
    prediction_hashes = [str(row["prediction_sha256"]) for row in prediction_rows]
    fold: dict[str, object] = {
        "target_event_key": 202605,
        "prior_event_keys": [202601, 202602, 202603, 202604],
        "target_event_used_for_training_or_selection": False,
        "inner_selection_and_early_stopping": {
            "inner_train_event_keys": [202601, 202602, 202603],
            "inner_validation_event_key": 202604,
            "selected_candidate_id": "zero_telemetry_correction",
            "target_event_used": False,
        },
        "metrics": {
            "source_shift_baseline": {"mae_seconds": 1.0},
            "tcn_driver_correction": {"mae_seconds": 1.2},
            "locked_selected_policy": {"mae_seconds": 1.0},
        },
        "predictions": prediction_rows,
        "prediction_set_sha256": best_runner._canonical_sha256(prediction_hashes),
    }
    fold["fold_sha256"] = best_runner._canonical_sha256(fold)
    payload: dict[str, object] = {
        "schema_version": best_runner.TCN_RESEARCH_EVIDENCE_SCHEMA_VERSION,
        "status": "research_evaluated_not_promotion_eligible",
        "promotion_eligible": False,
        "deployment_changed": False,
        "profile_design_provenance": {
            "design_stage": "original_pre_sensitivity_control",
            "prior_outer_results_informed_profile_design": False,
            "same_outer_evaluation_targets_seen_before_profile_freeze": False,
            "hyperparameters_tuned_on_outer_targets": False,
            "durable_matrix_results_used_to_select_profile": False,
            "promotion_eligible_from_profile_design": False,
        },
        "source_manifest": {
        },
        "validation_contract": {
            "outer_split": "strict_complete_event_expanding_window",
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
            "prior_outer_results_informed_profile_design": False,
            "same_outer_evaluation_targets_seen_before_profile_freeze": False,
            "hyperparameters_tuned_on_outer_targets": False,
            "fixed_small_architecture": True,
            "current_run_zero_correction_selector_uses_only_prior_events": True,
            "event_equal_training_weights": True,
        },
        "event_keys": [202601, 202602, 202603, 202604, 202605],
        "warmup_event_keys": [202601, 202602, 202603, 202604],
        "scored_event_keys": [202605],
        "minimum_train_events": 4,
        "capacity": {
            "independent_event_count": 5,
            "driver_event_bag_count": 10,
            "validated_correlated_tensor_count": 15,
            "correlated_tensor_count_treated_as_sample_size": False,
            "statistical_effective_degrees_of_freedom_claimed": False,
            "trainable_scalar_parameter_count": 274,
        },
        "folds": [fold],
        "predictions": prediction_rows,
        "prediction_set_sha256": best_runner._canonical_sha256(prediction_hashes),
        "summary": {
            "scored_event_count": 1,
            "scored_driver_event_count": 2,
            "source_shift_baseline": {"event_balanced_mae_seconds": 1.0},
            "tcn_driver_correction": {
                "event_balanced_mae_seconds": 1.2,
                "delta_vs_source_shift_baseline_seconds": 0.2,
                "relative_improvement_vs_source_shift_baseline": -0.2,
                "events_beating_source_shift_baseline": 0,
            },
            "locked_selected_policy": {"event_balanced_mae_seconds": 1.0},
        },
    }
    _rehash(payload)
    return payload


def _telemetry_audit() -> dict[str, object]:
    return {
        "event_count": 5,
        "driver_event_count": 10,
        "validated_tensor_count": 15,
        "complete_event_keys": [
            "202601",
            "202602",
            "202603",
            "202604",
            "202605",
        ],
        "minimum_independent_events": 4,
        "minimum_drivers_per_event": 1,
        "ready_for_requested_event_protocol": True,
        "blockers": [],
    }


def _write_supervised_source(root: Path) -> tuple[Path, dict[str, object]]:
    paths = _write_fixture(root)
    source_telemetry = paths["telemetry_manifest"].parent
    source_weekend = paths["metadata"].parent
    for round_number in range(2, 6):
        slug = f"round_{round_number:02d}_test_grand_prix_{round_number}"
        telemetry_event = source_telemetry.parent / slug
        weekend_event = source_weekend.parent / slug
        shutil.copytree(source_telemetry, telemetry_event)
        shutil.copytree(source_weekend, weekend_event)

        telemetry_manifest_path = telemetry_event / "telemetry_manifest.json"
        telemetry_manifest = json.loads(
            telemetry_manifest_path.read_text(encoding="utf-8")
        )
        telemetry_manifest["round"] = round_number
        telemetry_manifest["event_key"] = 202600 + round_number
        telemetry_manifest["event_name"] = f"Test Grand Prix {round_number}"
        for record in telemetry_manifest["feature_records"]:
            record["event_key"] = 202600 + round_number
            tensor_path = telemetry_event / "features" / Path(
                str(record["telemetry_path"])
            ).name
            record["telemetry_path"] = str(tensor_path.relative_to(root))
            record["telemetry_sha256"] = sha256_file(tensor_path)
        telemetry_manifest_path.write_text(
            json.dumps(telemetry_manifest, indent=2), encoding="utf-8"
        )

        metadata_path = weekend_event / "weekend_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["round_number"] = round_number
        metadata["event_name"] = f"Test Grand Prix {round_number}"
        session = metadata["sessions"][0]
        laps_path = weekend_event / "04_qualifying_laps.csv"
        results_path = weekend_event / "04_qualifying_results.csv"
        session["laps_path"] = str(laps_path.relative_to(root))
        session["results_path"] = str(results_path.relative_to(root))
        session["files"]["laps"]["sha256"] = sha256_file(laps_path)
        session["files"]["results"]["sha256"] = sha256_file(results_path)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    manifest = build_prequal_telemetry_supervised_manifest(
        root=root,
        telemetry_root=paths["telemetry_root"],
        weekends_root=paths["weekends_root"],
        year=2026,
        generated_at="2026-01-02T00:00:00Z",
    )
    path = root / "data/f1/derived/prequal_telemetry_supervised_2026.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    binding = {
        "path": str(path.relative_to(root)),
        "schema_version": manifest["schema_version"],
        "sha256": sha256_file(path),
        "bag_set_sha256": manifest["bag_set_sha256"],
        "feature_input_manifest_sha256": manifest[
            "feature_input_manifest_sha256"
        ],
        "target_input_manifest_sha256": manifest[
            "target_input_manifest_sha256"
        ],
    }
    return path, binding


def _write_evidence(root: Path, payload: dict[str, object]) -> Path:
    original_hash_valid = payload.get("artifact_payload_sha256") == best_runner._canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "artifact_payload_sha256"
        }
    )
    _, binding = _write_supervised_source(root)
    overrides = payload.get("source_manifest")
    assert isinstance(overrides, dict)
    payload["source_manifest"] = {**binding, **overrides}
    implementation_paths = [
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
    ]
    implementation_manifest: list[dict[str, object]] = []
    for relative_path in implementation_paths:
        implementation_path = root / relative_path
        implementation_path.parent.mkdir(parents=True, exist_ok=True)
        implementation_path.write_text(
            f"# fixture for {relative_path}\n",
            encoding="utf-8",
        )
        implementation_manifest.append(
            {
                "path": relative_path,
                "sha256": sha256_file(implementation_path),
                "size_bytes": implementation_path.stat().st_size,
            }
        )
    payload["implementation_manifest"] = implementation_manifest
    payload["implementation_manifest_sha256"] = best_runner._canonical_sha256(
        implementation_manifest
    )
    if original_hash_valid:
        _rehash(payload)
    path = root / "artifacts/backtests/f1/telemetry/tcn.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _attach_sensitivity_binding(root: Path, payload: dict[str, object]) -> Path:
    plan_path = root / "configs/f1/tcn_sensitivity.json"
    matrix_path = root / "artifacts/backtests/f1/telemetry/matrix.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan = {
        "decision_policy": {
            "reference_profile_id": best_runner.TCN_REFERENCE_PROFILE_ID,
            "reference_profile_fixed_before_durable_matrix_execution": True,
            "profile_design_provenance_declared_per_profile": True,
            "outer_target_informed_profile_ids": (
                best_runner.TCN_OUTER_TARGET_INFORMED_PROFILE_IDS
            ),
            "durable_matrix_results_used_to_select_reference_profile": False,
            "promotion_allowed_from_this_matrix": False,
            "posthoc_profiles_never_selection_evidence": True,
        }
    }
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    design_provenance = {
        "design_stage": (
            "postdevelopment_outer_results_informed_before_durable_matrix"
        ),
        "prior_outer_results_informed_profile_design": True,
        "same_outer_evaluation_targets_seen_before_profile_freeze": True,
        "hyperparameters_tuned_on_outer_targets": True,
        "durable_matrix_results_used_to_select_profile": False,
        "promotion_eligible_from_profile_design": False,
    }
    payload["profile_design_provenance"] = design_provenance
    validation = payload["validation_contract"]
    assert isinstance(validation, dict)
    for field in (
        "prior_outer_results_informed_profile_design",
        "same_outer_evaluation_targets_seen_before_profile_freeze",
        "hyperparameters_tuned_on_outer_targets",
    ):
        validation[field] = design_provenance[field]
    payload["sensitivity_profile"] = {
        "profile_id": best_runner.TCN_REFERENCE_PROFILE_ID,
        "evidence_role": best_runner.TCN_REFERENCE_EVIDENCE_ROLE,
        "profile_design_provenance": design_provenance,
        "plan_path": plan_path.relative_to(root).as_posix(),
        "plan_sha256": sha256_file(plan_path),
        "matrix_output_path": matrix_path.relative_to(root).as_posix(),
        "reference_profile_fixed_before_durable_matrix_execution": True,
        "prior_outer_results_informed_profile_design": True,
        "durable_matrix_results_used_to_select_reference_profile": False,
        "promotion_allowed_from_matrix": False,
        "completed_matrix_required_for_downstream_consumption": True,
    }
    _rehash(payload)
    return matrix_path


def _write_sensitivity_matrix(
    root: Path,
    evidence_path: Path,
    matrix_path: Path,
) -> dict[str, object]:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    sensitivity = evidence["sensitivity_profile"]
    plan_path = root / sensitivity["plan_path"]
    implementation_paths = [
        plan_path,
        root
        / "research/projects/F1/rising_qualification_prediction/Python/"
        "run_prequal_telemetry_tcn_research.py",
        root
        / "research/projects/F1/rising_qualification_prediction/Python/"
        "run_prequal_telemetry_tcn_sensitivity.py",
    ]
    for implementation_path in implementation_paths:
        implementation_path.parent.mkdir(parents=True, exist_ok=True)
        if not implementation_path.exists():
            implementation_path.write_text("# matrix fixture\n", encoding="utf-8")
    implementation = [
        {
            "path": implementation_path.relative_to(root).as_posix(),
            "sha256": sha256_file(implementation_path),
            "size_bytes": implementation_path.stat().st_size,
        }
        for implementation_path in implementation_paths
    ]
    seed_summary = copy.deepcopy(evidence["summary"])
    seed_summary["tcn_driver_correction"]["event_balanced_mae_seconds"] = 1.3
    matrix: dict[str, object] = {
        "schema_version": best_runner.TCN_SENSITIVITY_MATRIX_SCHEMA_VERSION,
        "status": best_runner.TCN_SENSITIVITY_MATRIX_STATUS,
        "promotion_eligible": False,
        "deployment_changed": False,
        "plan": {
            "path": plan_path.relative_to(root).as_posix(),
            "sha256": sha256_file(plan_path),
            "decision_policy": json.loads(plan_path.read_text(encoding="utf-8"))[
                "decision_policy"
            ],
        },
        "source_manifest": {
            "path": evidence["source_manifest"]["path"],
            "sha256": evidence["source_manifest"]["sha256"],
        },
        "implementation_manifest": implementation,
        "implementation_manifest_sha256": best_runner._canonical_sha256(
            implementation
        ),
        "execution": {
            "profile_design_provenance_declared_per_profile": True,
            "outer_target_informed_profile_ids": (
                best_runner.TCN_OUTER_TARGET_INFORMED_PROFILE_IDS
            ),
            "durable_matrix_results_used_to_select_reference_profile": False,
            "reference_profile_id": best_runner.TCN_REFERENCE_PROFILE_ID,
            "reference_profile_fixed_before_durable_matrix_execution": True,
            "matrix_selects_winner": False,
        },
        "profiles": [
            {
                "profile_id": best_runner.TCN_REFERENCE_PROFILE_ID,
                "evidence_role": best_runner.TCN_REFERENCE_EVIDENCE_ROLE,
                "profile_design_provenance": evidence[
                    "profile_design_provenance"
                ],
                "output_path": evidence_path.relative_to(root).as_posix(),
                "output_sha256": sha256_file(evidence_path),
                "output_size_bytes": evidence_path.stat().st_size,
                "artifact_payload_sha256": evidence["artifact_payload_sha256"],
                "training_config": evidence.get("training_config"),
                "architecture": evidence.get("architecture"),
                "capacity": evidence["capacity"],
                "model_input_ablation": evidence.get("model_input_ablation"),
                "summary": evidence["summary"],
            },
            {
                "profile_id": "d3_primary_seed_stability",
                "summary": seed_summary,
            },
        ],
        "comparisons": {
            "reference_vs_equal_architecture_zero_telemetry_sham": {
                "mean_delta_seconds": 0.01,
            },
            "reference_vs_fixed_seed_repeat": {
                "mean_absolute_event_delta_seconds": 0.3,
            },
        },
    }
    matrix["artifact_payload_sha256"] = best_runner._canonical_sha256(matrix)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text(json.dumps(matrix, sort_keys=True), encoding="utf-8")
    return matrix


def test_valid_tcn_evidence_reports_rejected_metrics_and_binds_file_hash(
    tmp_path: Path,
) -> None:
    path = _write_evidence(tmp_path, _evidence_payload())
    validated_inputs: list[Path] = []

    evidence = best_runner._load_tcn_research_evidence(
        path,
        root=tmp_path,
        year=2026,
        telemetry_audit=_telemetry_audit(),
        validated_input_files=validated_inputs,
        require_completed_sensitivity_matrix=False,
    )
    decision = best_runner._deep_model_readiness_decision(
        _telemetry_audit(),
        tcn_runtime_available=False,
        tcn_evidence=evidence,
    )

    assert evidence["status"] == "evaluated_rejected_no_incremental_gain"
    assert evidence["result"] == {
        "source_shift_event_balanced_mae_seconds": 1.0,
        "tcn_event_balanced_mae_seconds": 1.2,
        "tcn_delta_vs_source_shift_seconds": pytest.approx(0.2),
        "tcn_relative_improvement_vs_source_shift": pytest.approx(-0.2),
        "tcn_events_beating_source_shift": 0,
        "scored_event_count": 1,
        "locked_selected_policy_event_balanced_mae_seconds": 1.0,
        "locked_zero_correction_fold_count": 1,
        "locked_tcn_selected_fold_count": 0,
        "tcn_improves_source_shift": False,
        "locked_selected_policy_improves_source_shift": False,
        "parameter_stable_across_fixed_seed_repeat": None,
    }
    assert decision["deep_model_evaluation_status"] == (
        "evaluated_rejected_no_incremental_gain"
    )
    assert "true_tcn_not_yet_evaluated_under_event_disjoint_protocol" not in decision[
        "deep_model_blockers"
    ]
    assert "tcn_failed_incremental_mae_gate_vs_source_shift" in decision[
        "deep_model_blockers"
    ]
    manifest = best_runner._hash_manifest([path], root=tmp_path)
    assert manifest[evidence["evidence_file"]["path"]] == evidence["evidence_file"][
        "sha256"
    ]
    relative_inputs = {
        input_path.relative_to(tmp_path).as_posix()
        for input_path in validated_inputs
    }
    assert evidence["evidence_file"]["path"] in relative_inputs
    assert evidence["source_manifest"]["path"] in relative_inputs
    assert {
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
    }.issubset(relative_inputs)
    assert len(relative_inputs) > 7


def test_tcn_readiness_requires_completed_sensitivity_matrix_by_default(
    tmp_path: Path,
) -> None:
    path = _write_evidence(tmp_path, _evidence_payload())

    with pytest.raises(ValueError, match="requires a completed sensitivity matrix"):
        best_runner._load_tcn_research_evidence(
            path,
            root=tmp_path,
            year=2026,
            telemetry_audit=_telemetry_audit(),
        )


def test_completed_sensitivity_matrix_binds_profile_and_seed_sham_controls(
    tmp_path: Path,
) -> None:
    payload = _evidence_payload()
    matrix_path = _attach_sensitivity_binding(tmp_path, payload)
    path = _write_evidence(tmp_path, payload)
    _write_sensitivity_matrix(tmp_path, path, matrix_path)
    validated_inputs: list[Path] = []

    evidence = best_runner._load_tcn_research_evidence(
        path,
        root=tmp_path,
        year=2026,
        telemetry_audit=_telemetry_audit(),
        validated_input_files=validated_inputs,
    )

    assert evidence["status"] == "evaluated_parameter_sensitive_inconclusive"
    assert evidence["sensitivity_matrix"]["postdevelopment_descriptive_only"] is True
    assert evidence["result"]["parameter_stable_across_fixed_seed_repeat"] is False
    assert "fixed_seed_repeat_failed_incremental_gain" in evidence[
        "promotion_blockers"
    ]
    assert matrix_path in validated_inputs
    assert tmp_path / payload["sensitivity_profile"]["plan_path"] in validated_inputs


def test_sensitivity_matrix_rejects_profile_file_hash_mismatch(
    tmp_path: Path,
) -> None:
    payload = _evidence_payload()
    matrix_path = _attach_sensitivity_binding(tmp_path, payload)
    path = _write_evidence(tmp_path, payload)
    matrix = _write_sensitivity_matrix(tmp_path, path, matrix_path)
    matrix["profiles"][0]["output_sha256"] = "0" * 64
    matrix["artifact_payload_sha256"] = best_runner._canonical_sha256(
        {
            key: value
            for key, value in matrix.items()
            if key != "artifact_payload_sha256"
        }
    )
    matrix_path.write_text(json.dumps(matrix, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="profile file hash mismatch"):
        best_runner._load_tcn_research_evidence(
            path,
            root=tmp_path,
            year=2026,
            telemetry_audit=_telemetry_audit(),
        )


def test_tcn_evidence_rejects_payload_hash_mutation(tmp_path: Path) -> None:
    payload = _evidence_payload()
    payload["status"] = "tampered"
    path = _write_evidence(tmp_path, payload)

    with pytest.raises(ValueError, match="payload hash mismatch"):
        best_runner._load_tcn_research_evidence(
            path,
            root=tmp_path,
            year=2026,
            telemetry_audit=_telemetry_audit(),
            require_completed_sensitivity_matrix=False,
        )


def test_tcn_evidence_rejects_rehashed_metrics_not_supported_by_predictions(
    tmp_path: Path,
) -> None:
    payload = _evidence_payload()
    summary = payload["summary"]
    assert isinstance(summary, dict)
    tcn = summary["tcn_driver_correction"]
    assert isinstance(tcn, dict)
    tcn["event_balanced_mae_seconds"] = 1.3
    tcn["delta_vs_source_shift_baseline_seconds"] = 0.3
    tcn["relative_improvement_vs_source_shift_baseline"] = -0.3
    _rehash(payload)
    path = _write_evidence(tmp_path, payload)

    with pytest.raises(ValueError, match="headline MAEs"):
        best_runner._load_tcn_research_evidence(
            path,
            root=tmp_path,
            year=2026,
            telemetry_audit=_telemetry_audit(),
            require_completed_sensitivity_matrix=False,
        )


def test_tcn_evidence_rejects_unbound_source_manifest_schema(tmp_path: Path) -> None:
    payload = _evidence_payload()
    source_manifest = payload["source_manifest"]
    assert isinstance(source_manifest, dict)
    source_manifest["schema_version"] = "unknown"
    _rehash(payload)
    path = _write_evidence(tmp_path, payload)

    with pytest.raises(ValueError, match="supervised source schema"):
        best_runner._load_tcn_research_evidence(
            path,
            root=tmp_path,
            year=2026,
            telemetry_audit=_telemetry_audit(),
            require_completed_sensitivity_matrix=False,
        )


def test_tcn_evidence_rejects_missing_exact_source_manifest(tmp_path: Path) -> None:
    path = _write_evidence(tmp_path, _evidence_payload())
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_manifest = payload["source_manifest"]
    source_path = tmp_path / str(source_manifest["path"])
    source_path.unlink()

    with pytest.raises(ValueError, match="invalid TCN source manifest"):
        best_runner._load_tcn_research_evidence(
            path,
            root=tmp_path,
            year=2026,
            telemetry_audit=_telemetry_audit(),
            require_completed_sensitivity_matrix=False,
        )


def test_tcn_evidence_rejects_tampered_exact_source_manifest(tmp_path: Path) -> None:
    path = _write_evidence(tmp_path, _evidence_payload())
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_path = tmp_path / str(payload["source_manifest"]["path"])
    supervised = json.loads(source_path.read_text(encoding="utf-8"))
    supervised["generated_at"] = "2026-01-03T00:00:00Z"
    source_path.write_text(json.dumps(supervised, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="external SHA-256 mismatch"):
        best_runner._load_tcn_research_evidence(
            path,
            root=tmp_path,
            year=2026,
            telemetry_audit=_telemetry_audit(),
            require_completed_sensitivity_matrix=False,
        )


def test_tcn_evidence_rejects_tampered_nested_source_input(tmp_path: Path) -> None:
    path = _write_evidence(tmp_path, _evidence_payload())
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_path = tmp_path / str(payload["source_manifest"]["path"])
    supervised = json.loads(source_path.read_text(encoding="utf-8"))
    nested_path = tmp_path / str(supervised["target_input_files"][0]["path"])
    nested_path.write_bytes(nested_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="provenance validation failed"):
        best_runner._load_tcn_research_evidence(
            path,
            root=tmp_path,
            year=2026,
            telemetry_audit=_telemetry_audit(),
            require_completed_sensitivity_matrix=False,
        )


def test_tcn_evidence_rejects_absolute_source_manifest_path(tmp_path: Path) -> None:
    path = _write_evidence(tmp_path, _evidence_payload())
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_manifest = payload["source_manifest"]
    source_manifest["path"] = str(
        (tmp_path / str(source_manifest["path"])).resolve()
    )
    _rehash(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="must be repository-relative"):
        best_runner._load_tcn_research_evidence(
            path,
            root=tmp_path,
            year=2026,
            telemetry_audit=_telemetry_audit(),
            require_completed_sensitivity_matrix=False,
        )


def test_tcn_evidence_accepts_improving_locked_tcn_without_promoting(
    tmp_path: Path,
) -> None:
    payload = _evidence_payload()
    folds = payload["folds"]
    assert isinstance(folds, list)
    fold = folds[0]
    assert isinstance(fold, dict)
    inner = fold["inner_selection_and_early_stopping"]
    assert isinstance(inner, dict)
    inner["selected_candidate_id"] = "tcn_driver_correction"
    predictions = fold["predictions"]
    assert isinstance(predictions, list)
    assert len(predictions) == 2
    for row, predicted in zip(predictions, (80.2, 81.8)):
        row["tcn_driver_correction_predicted_lap_time_seconds"] = predicted
        row["locked_selected_policy_predicted_lap_time_seconds"] = predicted
        row["locked_selected_candidate_id"] = "tcn_driver_correction"
        row["prediction_sha256"] = best_runner._canonical_sha256(
            {key: value for key, value in row.items() if key != "prediction_sha256"}
        )
    prediction_hashes = [str(row["prediction_sha256"]) for row in predictions]
    fold["prediction_set_sha256"] = best_runner._canonical_sha256(prediction_hashes)
    fold_metrics = fold["metrics"]
    assert isinstance(fold_metrics, dict)
    fold_metrics["tcn_driver_correction"]["mae_seconds"] = 0.2
    fold_metrics["locked_selected_policy"]["mae_seconds"] = 0.2
    fold["fold_sha256"] = best_runner._canonical_sha256(
        {key: value for key, value in fold.items() if key != "fold_sha256"}
    )
    payload["predictions"] = predictions
    payload["prediction_set_sha256"] = best_runner._canonical_sha256(
        prediction_hashes
    )
    summary = payload["summary"]
    assert isinstance(summary, dict)
    summary["tcn_driver_correction"].update(
        {
            "event_balanced_mae_seconds": 0.2,
            "delta_vs_source_shift_baseline_seconds": -0.8,
            "relative_improvement_vs_source_shift_baseline": 0.8,
            "events_beating_source_shift_baseline": 1,
        }
    )
    summary["locked_selected_policy"]["event_balanced_mae_seconds"] = 0.2
    _rehash(payload)
    path = _write_evidence(tmp_path, payload)

    evidence = best_runner._load_tcn_research_evidence(
        path,
        root=tmp_path,
        year=2026,
        telemetry_audit=_telemetry_audit(),
        require_completed_sensitivity_matrix=False,
    )

    assert evidence["status"] == "evaluated_improving_not_promotion_eligible"
    assert evidence["promotion_ready"] is False
    assert evidence["result"]["locked_tcn_selected_fold_count"] == 1
    assert evidence["result"]["tcn_improves_source_shift"] is True
    assert "tcn_failed_incremental_mae_gate_vs_source_shift" not in evidence[
        "promotion_blockers"
    ]
