from __future__ import annotations

import hashlib
import json
from pathlib import Path

from packages.f1.orchestration.evidence import (
    BASELINE_LADDER_SCHEMA_VERSION,
    HORIZON_BENCHMARK_SCHEMA_VERSION,
    audit_evidence_path,
    f1_runtime_manifest,
)


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(path: Path) -> dict[str, object]:
    files = [{"path": path.name, "sha256": _digest(path)}]
    aggregate = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "manifest_version": "test_v1",
        "aggregate_sha256": aggregate,
        "file_count": 1,
        "files": files,
    }


def test_stale_baseline_artifact_fails_closed(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "old.json",
        {"workflow": "f1_baseline_ladder", "summary": {"available": True}},
    )

    audit = audit_evidence_path(path, root=tmp_path)

    assert audit.valid is False
    assert "stale_or_missing_schema_version" in audit.reasons
    assert "promotion_gate_missing" in audit.reasons


def test_current_baseline_manifest_detects_implementation_drift(tmp_path: Path) -> None:
    implementation = tmp_path / "runner.py"
    implementation.write_text("print('v1')", encoding="utf-8")
    input_data = tmp_path / "weekend.csv"
    input_data.write_text("driver,position\na,1\n", encoding="utf-8")
    artifact = _write(
        tmp_path / "baseline.json",
        {
            "schema_version": BASELINE_LADDER_SCHEMA_VERSION,
            "workflow": "f1_baseline_ladder",
            "validation_contract": {"headline_metrics_require_complete_field": True},
            "implementation": _manifest(implementation),
            "runtime": f1_runtime_manifest(),
            "input_data": _manifest(input_data),
            "summary": {"promotion_gate": {"passed": False, "reasons": ["no_edge"]}},
        },
    )
    assert audit_evidence_path(artifact, root=tmp_path).valid is True

    implementation.write_text("print('v2')", encoding="utf-8")
    audit = audit_evidence_path(artifact, root=tmp_path)
    assert audit.valid is False
    assert "implementation_hash_mismatch" in audit.reasons


def test_current_baseline_manifest_detects_input_data_drift(tmp_path: Path) -> None:
    implementation = tmp_path / "runner.py"
    implementation.write_text("print('stable')", encoding="utf-8")
    input_data = tmp_path / "weekend.csv"
    input_data.write_text("driver,position\na,1\n", encoding="utf-8")
    artifact = _write(
        tmp_path / "baseline.json",
        {
            "schema_version": BASELINE_LADDER_SCHEMA_VERSION,
            "workflow": "f1_baseline_ladder",
            "validation_contract": {"headline_metrics_require_complete_field": True},
            "implementation": _manifest(implementation),
            "runtime": f1_runtime_manifest(),
            "input_data": _manifest(input_data),
            "summary": {"promotion_gate": {"passed": False, "reasons": ["no_edge"]}},
        },
    )
    assert audit_evidence_path(artifact, root=tmp_path).valid is True

    input_data.write_text("driver,position\na,2\n", encoding="utf-8")
    audit = audit_evidence_path(artifact, root=tmp_path)
    assert audit.valid is False
    assert "input_data_hash_mismatch" in audit.reasons


def test_current_baseline_manifest_detects_runtime_drift(tmp_path: Path) -> None:
    implementation = tmp_path / "runner.py"
    implementation.write_text("stable", encoding="utf-8")
    input_data = tmp_path / "weekend.csv"
    input_data.write_text("stable", encoding="utf-8")
    runtime = f1_runtime_manifest()
    runtime["python_version"] = "0.0.0"
    artifact = _write(
        tmp_path / "baseline.json",
        {
            "schema_version": BASELINE_LADDER_SCHEMA_VERSION,
            "workflow": "f1_baseline_ladder",
            "validation_contract": {"headline_metrics_require_complete_field": True},
            "implementation": _manifest(implementation),
            "runtime": runtime,
            "input_data": _manifest(input_data),
            "summary": {"promotion_gate": {"passed": False, "reasons": ["no_edge"]}},
        },
    )

    audit = audit_evidence_path(artifact, root=tmp_path)

    assert audit.valid is False
    assert "runtime_python_version_mismatch" in audit.reasons


def test_horizon_manifest_requires_complete_hashed_same_population_inputs(tmp_path: Path) -> None:
    inputs = {}
    for name in ("horizon_a", "horizon_b", "trace"):
        path = tmp_path / f"{name}.json"
        path.write_text(name, encoding="utf-8")
        inputs[f"{name}_path"] = path.name
        inputs[f"{name}_sha256"] = _digest(path)
    implementation = tmp_path / "horizon_runner.py"
    implementation.write_text("print('horizon')", encoding="utf-8")
    input_data = tmp_path / "race_results.csv"
    input_data.write_text("driver,position\na,1\nb,2\n", encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}", encoding="utf-8")
    artifact = _write(
        tmp_path / "horizon.json",
        {
            "schema_version": HORIZON_BENCHMARK_SCHEMA_VERSION,
            "validation_status": "valid_complete_field_locked_calibration",
            "implementation": _manifest(implementation),
            "runtime": f1_runtime_manifest(),
            "input_data": _manifest(input_data),
            "population_contract": {
                "horizon_a_and_b_use_same_actual_field": True,
                "requested_rounds_complete": True,
                "all_requested_cutoffs_complete": True,
                "issues_empty": True,
                "expected_cutoff_count_per_round": 2,
            },
            "rounds_requested": [1],
            "rounds_with_output": [1],
            "round_cutoff_counts": {"1": 2},
            "issues": [],
            "live_calibration": {
                "artifact": {"path": calibration.name, "sha256": _digest(calibration)},
                "prior_calibration_ready": True,
                "uses_hand_tuned_priors": False,
                "calibration_mode": "locked_replay",
            },
            "by_cutoff": [],
            "input_artifacts": [
                {
                    **inputs,
                    "round": 1,
                    "complete_field": True,
                    "horizon_a_rows": 2,
                    "actual_rows": 2,
                    "horizon_b_rows_by_cutoff": [2, 2],
                    "horizon_b_matched_by_cutoff": [2, 2],
                    "cutoff_count": 2,
                    "expected_cutoff_count": 2,
                    "horizon_b_complete_field": True,
                    "same_actual_field": True,
                }
            ],
            "artifacts": {"summary": "artifacts/summary.json"},
        },
    )

    assert audit_evidence_path(artifact, root=tmp_path).valid is True

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["input_artifacts"][0]["complete_field"] = False
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    audit = audit_evidence_path(artifact, root=tmp_path)
    assert audit.valid is False
    assert "incomplete_horizon_a_population" in audit.reasons


def test_horizon_manifest_rejects_partial_requested_cutoff_coverage(tmp_path: Path) -> None:
    inputs = {}
    for name in ("horizon_a", "horizon_b", "trace"):
        source = tmp_path / f"{name}.json"
        source.write_text(name, encoding="utf-8")
        inputs[f"{name}_path"] = source.name
        inputs[f"{name}_sha256"] = _digest(source)
    implementation = tmp_path / "runner.py"
    implementation.write_text("stable", encoding="utf-8")
    input_data = tmp_path / "race.csv"
    input_data.write_text("stable", encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}", encoding="utf-8")
    artifact = _write(
        tmp_path / "partial.json",
        {
            "schema_version": HORIZON_BENCHMARK_SCHEMA_VERSION,
            "validation_status": "valid_complete_field_locked_calibration",
            "implementation": _manifest(implementation),
            "runtime": f1_runtime_manifest(),
            "input_data": _manifest(input_data),
            "population_contract": {
                "horizon_a_and_b_use_same_actual_field": True,
                "requested_rounds_complete": True,
                "all_requested_cutoffs_complete": False,
                "issues_empty": False,
                "expected_cutoff_count_per_round": 2,
            },
            "rounds_requested": [1],
            "rounds_with_output": [1],
            "round_cutoff_counts": {"1": 1},
            "issues": ["missing cutoff"],
            "live_calibration": {
                "artifact": {"path": calibration.name, "sha256": _digest(calibration)},
                "prior_calibration_ready": True,
                "uses_hand_tuned_priors": False,
                "calibration_mode": "locked_replay",
            },
            "by_cutoff": [],
            "input_artifacts": [
                {
                    **inputs,
                    "round": 1,
                    "complete_field": True,
                    "horizon_a_rows": 2,
                    "actual_rows": 2,
                    "horizon_b_rows_by_cutoff": [2],
                    "horizon_b_matched_by_cutoff": [2],
                    "cutoff_count": 1,
                    "horizon_b_complete_field": False,
                    "same_actual_field": False,
                }
            ],
        },
    )

    audit = audit_evidence_path(artifact, root=tmp_path)

    assert audit.valid is False
    assert "requested_cutoff_coverage_incomplete" in audit.reasons
    assert "round_cutoff_count_mismatch" in audit.reasons
    assert "horizon_b_field_or_cutoff_mismatch" in audit.reasons
