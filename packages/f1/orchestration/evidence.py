"""Fail-closed validation for F1 benchmark evidence artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping, Optional


BASELINE_LADDER_SCHEMA_VERSION = "f1_baseline_ladder_complete_field_v2"
HORIZON_BENCHMARK_SCHEMA_VERSION = "f1_horizon_a_vs_b_complete_field_calibration_v3"


def f1_runtime_manifest() -> dict[str, object]:
    packages: dict[str, Optional[str]] = {}
    for distribution in ("numpy", "pandas", "scikit-learn"):
        try:
            packages[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "packages": packages,
    }


@dataclass(frozen=True)
class EvidenceAudit:
    path: str
    artifact_type: str
    valid: bool
    reasons: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "artifact_type": self.artifact_type,
            "valid": self.valid,
            "reasons": list(self.reasons),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_reference(reference: object, root: Path) -> Optional[Path]:
    if not isinstance(reference, str) or not reference.strip():
        return None
    candidate = Path(reference).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _absolute_references(value: object, prefix: str = "") -> list[str]:
    issues: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            issues.extend(_absolute_references(nested, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for idx, nested in enumerate(value):
            issues.extend(_absolute_references(nested, f"{prefix}[{idx}]"))
    elif isinstance(value, str) and value.startswith("/"):
        issues.append(prefix or "root")
    return issues


def _manifest_digest(files: list[dict[str, str]]) -> str:
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _normalized_int_list(value: object) -> Optional[list[int]]:
    if not isinstance(value, list) or not value:
        return None
    try:
        return sorted({int(item) for item in value})
    except (TypeError, ValueError):
        return None


def _audit_file_manifest(
    manifest: object,
    *,
    root: Path,
    label: str,
) -> list[str]:
    if not isinstance(manifest, Mapping):
        return [f"{label}_manifest_missing"]
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        return [f"{label}_files_missing"]
    reasons: list[str] = []
    canonical_files: list[dict[str, str]] = []
    for item in raw_files:
        if not isinstance(item, Mapping):
            reasons.append(f"{label}_file_entry_invalid")
            continue
        reference = item.get("path")
        expected = item.get("sha256")
        if isinstance(reference, str) and Path(reference).expanduser().is_absolute():
            reasons.append(f"{label}_path_not_portable")
        resolved = _resolve_reference(reference, root)
        if resolved is None or not resolved.exists() or not resolved.is_file():
            reasons.append(f"{label}_file_missing")
            continue
        if not isinstance(expected, str) or expected != _sha256(resolved):
            reasons.append(f"{label}_hash_mismatch")
        if isinstance(reference, str) and isinstance(expected, str):
            canonical_files.append({"path": reference, "sha256": expected})
    expected_aggregate = manifest.get("aggregate_sha256")
    if not isinstance(expected_aggregate, str) or expected_aggregate != _manifest_digest(canonical_files):
        reasons.append(f"{label}_aggregate_hash_mismatch")
    declared_count = manifest.get("file_count")
    if not isinstance(declared_count, int) or declared_count != len(raw_files):
        reasons.append(f"{label}_file_count_mismatch")
    return reasons


def _audit_runtime_manifest(runtime: object) -> list[str]:
    if not isinstance(runtime, Mapping):
        return ["runtime_manifest_missing"]
    current = f1_runtime_manifest()
    reasons: list[str] = []
    for key in ("python_version", "python_implementation", "python_executable_major_minor"):
        if runtime.get(key) != current.get(key):
            reasons.append(f"runtime_{key}_mismatch")
    if runtime.get("packages") != current.get("packages"):
        reasons.append("runtime_package_versions_mismatch")
    return reasons


def audit_baseline_ladder(payload: Mapping[str, Any], *, path: Path, root: Path) -> EvidenceAudit:
    reasons: list[str] = []
    if payload.get("schema_version") != BASELINE_LADDER_SCHEMA_VERSION:
        reasons.append("stale_or_missing_schema_version")
    contract = payload.get("validation_contract")
    if not isinstance(contract, Mapping) or not bool(contract.get("headline_metrics_require_complete_field")):
        reasons.append("complete_field_contract_missing")
    reasons.extend(_audit_file_manifest(payload.get("implementation"), root=root, label="implementation"))
    reasons.extend(_audit_file_manifest(payload.get("input_data"), root=root, label="input_data"))
    reasons.extend(_audit_runtime_manifest(payload.get("runtime")))
    if _absolute_references(payload.get("implementation", {})):
        reasons.append("implementation_paths_not_portable")
    if _absolute_references(payload.get("input_data", {})):
        reasons.append("input_data_paths_not_portable")
    summary = payload.get("summary")
    mode_summaries: Iterable[object]
    if isinstance(summary, Mapping) and isinstance(summary.get("mode_summaries"), Mapping):
        mode_summaries = summary["mode_summaries"].values()
    else:
        mode_summaries = (summary,)
    for mode_summary in mode_summaries:
        if not isinstance(mode_summary, Mapping) or not isinstance(mode_summary.get("promotion_gate"), Mapping):
            reasons.append("promotion_gate_missing")
            break
    if _absolute_references(payload.get("artifacts", {})):
        reasons.append("absolute_artifact_paths_not_portable")
    return EvidenceAudit(str(path), "baseline_ladder", not reasons, tuple(dict.fromkeys(reasons)))


def audit_horizon_benchmark(payload: Mapping[str, Any], *, path: Path, root: Path) -> EvidenceAudit:
    reasons: list[str] = []
    if payload.get("schema_version") != HORIZON_BENCHMARK_SCHEMA_VERSION:
        reasons.append("stale_or_missing_schema_version")
    if payload.get("validation_status") != "valid_complete_field_locked_calibration":
        reasons.append("complete_field_validation_missing")

    reasons.extend(_audit_file_manifest(payload.get("implementation"), root=root, label="implementation"))
    reasons.extend(_audit_file_manifest(payload.get("input_data"), root=root, label="input_data"))
    reasons.extend(_audit_runtime_manifest(payload.get("runtime")))
    if _absolute_references(payload.get("implementation", {})):
        reasons.append("implementation_paths_not_portable")
    if _absolute_references(payload.get("input_data", {})):
        reasons.append("input_data_paths_not_portable")

    contract = payload.get("population_contract")
    if not isinstance(contract, Mapping) or not bool(contract.get("horizon_a_and_b_use_same_actual_field")):
        reasons.append("same_population_contract_missing")
    if not isinstance(contract, Mapping) or not bool(contract.get("requested_rounds_complete")):
        reasons.append("requested_round_coverage_incomplete")
    if not isinstance(contract, Mapping) or not bool(contract.get("all_requested_cutoffs_complete")):
        reasons.append("requested_cutoff_coverage_incomplete")
    if not isinstance(contract, Mapping) or not bool(contract.get("issues_empty")):
        reasons.append("benchmark_issues_present")
    expected_cutoff_count = int(contract.get("expected_cutoff_count_per_round", 0)) if isinstance(contract, Mapping) else 0
    if expected_cutoff_count <= 0:
        reasons.append("expected_cutoff_count_missing")

    rounds_requested_raw = payload.get("rounds_requested")
    rounds_output_raw = payload.get("rounds_with_output")
    rounds_requested = _normalized_int_list(rounds_requested_raw) or []
    rounds_output = _normalized_int_list(rounds_output_raw) or []
    if not rounds_requested or rounds_requested != rounds_output:
        reasons.append("requested_rounds_do_not_match_output")
    raw_counts = payload.get("round_cutoff_counts")
    if not isinstance(raw_counts, Mapping):
        reasons.append("round_cutoff_counts_missing")
    else:
        for round_number in rounds_requested:
            try:
                count = int(raw_counts.get(str(round_number), raw_counts.get(round_number, 0)))
            except Exception:
                count = 0
            if count != expected_cutoff_count:
                reasons.append("round_cutoff_count_mismatch")
                break
    issues = payload.get("issues")
    if not isinstance(issues, list) or issues:
        reasons.append("benchmark_issues_not_empty")

    calibration = payload.get("live_calibration")
    if not isinstance(calibration, Mapping):
        reasons.append("live_calibration_manifest_missing")
    else:
        if calibration.get("calibration_mode") != "locked_replay":
            reasons.append("locked_replay_calibration_required")
        if not bool(calibration.get("prior_calibration_ready")):
            reasons.append("prior_calibration_not_ready")
        if bool(calibration.get("uses_hand_tuned_priors")):
            reasons.append("hand_tuned_priors_not_allowed")
        artifact = calibration.get("artifact")
        if not isinstance(artifact, Mapping):
            reasons.append("calibration_artifact_manifest_missing")
        else:
            reference = artifact.get("path")
            if isinstance(reference, str) and Path(reference).expanduser().is_absolute():
                reasons.append("calibration_artifact_path_not_portable")
            calibration_path = _resolve_reference(reference, root)
            expected_hash = artifact.get("sha256")
            if calibration_path is None or not calibration_path.exists():
                reasons.append("calibration_artifact_missing")
            elif not isinstance(expected_hash, str) or expected_hash != _sha256(calibration_path):
                reasons.append("calibration_artifact_hash_mismatch")

    inputs = payload.get("input_artifacts")
    if not isinstance(inputs, list) or not inputs:
        reasons.append("input_artifact_manifest_missing")
    else:
        manifest_rounds: list[int] = []
        for item in inputs:
            if not isinstance(item, Mapping) or not bool(item.get("complete_field")):
                reasons.append("incomplete_horizon_a_population")
                continue
            try:
                manifest_rounds.append(int(item.get("round")))
                horizon_a_rows = int(item.get("horizon_a_rows", 0))
                actual_rows = int(item.get("actual_rows", 0))
                cutoff_count = int(item.get("cutoff_count", 0))
            except Exception:
                reasons.append("input_population_counts_invalid")
                continue
            if actual_rows <= 0 or horizon_a_rows != actual_rows:
                reasons.append("horizon_a_field_count_mismatch")
            b_rows = item.get("horizon_b_rows_by_cutoff")
            b_matched = item.get("horizon_b_matched_by_cutoff")
            try:
                b_counts_valid = bool(
                    isinstance(b_rows, list)
                    and isinstance(b_matched, list)
                    and len(b_rows) == expected_cutoff_count
                    and len(b_matched) == expected_cutoff_count
                    and all(int(value) == actual_rows for value in b_rows)
                    and all(int(value) == actual_rows for value in b_matched)
                )
            except Exception:
                b_counts_valid = False
            if (
                not bool(item.get("horizon_b_complete_field"))
                or not bool(item.get("same_actual_field"))
                or cutoff_count != expected_cutoff_count
                or not b_counts_valid
            ):
                reasons.append("horizon_b_field_or_cutoff_mismatch")
            for prefix in ("horizon_a", "horizon_b", "trace"):
                reference = item.get(f"{prefix}_path")
                if isinstance(reference, str) and Path(reference).expanduser().is_absolute():
                    reasons.append(f"{prefix}_input_path_not_portable")
                input_path = _resolve_reference(reference, root)
                expected = item.get(f"{prefix}_sha256")
                if input_path is None or not input_path.exists():
                    reasons.append(f"{prefix}_input_missing")
                elif not isinstance(expected, str) or expected != _sha256(input_path):
                    reasons.append(f"{prefix}_input_hash_mismatch")
        if sorted(set(manifest_rounds)) != rounds_requested or len(inputs) != len(rounds_requested):
            reasons.append("input_manifest_round_coverage_mismatch")
    if _absolute_references(payload.get("artifacts", {})):
        reasons.append("absolute_artifact_paths_not_portable")
    return EvidenceAudit(str(path), "horizon_a_vs_b", not reasons, tuple(dict.fromkeys(reasons)))


def audit_evidence_path(path: str | Path, *, root: str | Path) -> EvidenceAudit:
    artifact_path = Path(path).expanduser().resolve()
    project_root = Path(root).expanduser().resolve()
    if not artifact_path.exists():
        return EvidenceAudit(str(artifact_path), "unknown", False, ("artifact_missing",))
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception:
        return EvidenceAudit(str(artifact_path), "unknown", False, ("invalid_json",))
    if not isinstance(payload, Mapping):
        return EvidenceAudit(str(artifact_path), "unknown", False, ("payload_not_object",))
    if payload.get("workflow") == "f1_baseline_ladder":
        return audit_baseline_ladder(payload, path=artifact_path, root=project_root)
    if "by_cutoff" in payload or "crossover_summary" in payload:
        return audit_horizon_benchmark(payload, path=artifact_path, root=project_root)
    return EvidenceAudit(str(artifact_path), "unknown", False, ("unsupported_artifact_type",))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed audit for F1 benchmark evidence artifacts.")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--root", default=str(Path.cwd()))
    args = parser.parse_args(argv)
    audits = [audit_evidence_path(path, root=args.root) for path in args.paths]
    print(json.dumps({"valid": all(audit.valid for audit in audits), "audits": [audit.to_payload() for audit in audits]}, indent=2))
    return 0 if all(audit.valid for audit in audits) else 1


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())
