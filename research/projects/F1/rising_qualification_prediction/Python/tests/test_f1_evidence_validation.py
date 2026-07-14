from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.f1.orchestration.evidence_validation import (
    canonical_json_sha256,
    sha256_file,
    validate_artifact_provenance,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_recursive_artifact_provenance_supports_all_manifest_shapes(tmp_path: Path) -> None:
    implementation = tmp_path / "packages/model.py"
    direct_input = tmp_path / "data/input.csv"
    tensor = tmp_path / "data/features/lap.npz"
    implementation.parent.mkdir(parents=True)
    direct_input.parent.mkdir(parents=True)
    tensor.parent.mkdir(parents=True)
    implementation.write_text("MODEL = 1\n", encoding="utf-8")
    direct_input.write_text("lap,time\n1,90\n", encoding="utf-8")
    tensor.write_bytes(b"tensor")

    feature_files = [
        {
            "prefix_path": "data/features/lap.npz",
            "prefix_sha256": sha256_file(tensor),
        }
    ]
    nested_payload = {
        "schema_version": "nested_v1",
        "feature_input_files": feature_files,
        "feature_input_manifest_sha256": canonical_json_sha256(feature_files),
    }
    nested = tmp_path / "data/nested.json"
    _write_json(nested, nested_payload)

    implementation_manifest = {
        "packages/model.py": sha256_file(implementation),
    }
    input_manifest = [
        {"path": "data/input.csv", "sha256": sha256_file(direct_input)},
        {"data/nested.json": {"sha256": sha256_file(nested)}},
    ]
    artifact_payload = {
        "schema_version": "artifact_v1",
        "implementation_manifest": implementation_manifest,
        "implementation_manifest_sha256": canonical_json_sha256(
            implementation_manifest
        ),
        "input_manifest": input_manifest,
        "input_manifest_sha256": canonical_json_sha256(input_manifest),
    }
    artifact = tmp_path / "artifacts/result.json"
    _write_json(artifact, artifact_payload)

    result = validate_artifact_provenance(
        root=tmp_path,
        artifact_path=artifact,
        expected_sha256=sha256_file(artifact),
        expected_schema_version="artifact_v1",
    )

    assert result["implementation_manifest_direct_entries"] == 1
    assert result["input_manifest_direct_entries"] == 2
    assert result["transitive_file_count"] == 4
    assert result["nested_json_count"] == 1
    assert result["aggregate_manifest_digest_count"] == 3
    assert result["provenance_closure_status"] == "pass"


def test_recursive_artifact_provenance_fails_on_drift_and_conflicts(tmp_path: Path) -> None:
    implementation = tmp_path / "packages/model.py"
    source = tmp_path / "data/input.csv"
    implementation.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    implementation.write_text("MODEL = 1\n", encoding="utf-8")
    source.write_text("lap,time\n1,90\n", encoding="utf-8")
    payload = {
        "schema_version": "artifact_v1",
        "implementation_manifest": {
            "packages/model.py": sha256_file(implementation),
        },
        "input_manifest": {
            "data/input.csv": sha256_file(source),
        },
    }
    artifact = tmp_path / "artifacts/result.json"
    _write_json(artifact, payload)
    artifact_sha = sha256_file(artifact)

    source.write_text("lap,time\n1,91\n", encoding="utf-8")
    with pytest.raises(ValueError, match="referenced file digest mismatch"):
        validate_artifact_provenance(
            root=tmp_path,
            artifact_path=artifact,
            expected_sha256=artifact_sha,
            expected_schema_version="artifact_v1",
        )

    payload["input_manifest"] = [
        {"path": "data/input.csv", "sha256": "0" * 64},
        {"data/input.csv": "1" * 64},
    ]
    _write_json(artifact, payload)
    with pytest.raises(ValueError, match="conflicting SHA-256 declarations"):
        validate_artifact_provenance(
            root=tmp_path,
            artifact_path=artifact,
            expected_sha256=sha256_file(artifact),
            expected_schema_version="artifact_v1",
        )


@pytest.mark.parametrize(
    "malformed_reference",
    [
        {"path": "data/missing.csv"},
        {"path": "data/missing.csv", "sha256": "not-a-sha"},
        {"prefix_path": "data/missing.csv"},
        {"prefix_path": "data/missing.csv", "prefix_sha256": "not-a-sha"},
        {"data/missing.csv": "not-a-sha"},
        {"data/missing.csv": {}},
        {"telemetry_path": "data/missing.csv"},
        {"telemetry_path": "data/missing.csv", "telemetry_sha256": "not-a-sha"},
    ],
    ids=(
        "path-missing-digest",
        "path-malformed-digest",
        "prefix-path-missing-digest",
        "prefix-path-malformed-digest",
        "path-key-map-malformed-digest",
        "path-key-map-missing-digest",
        "generic-path-missing-digest",
        "generic-path-malformed-digest",
    ),
)
def test_recursive_artifact_provenance_rejects_partial_manifest_entries(
    tmp_path: Path,
    malformed_reference: object,
) -> None:
    implementation = tmp_path / "packages/model.py"
    source = tmp_path / "data/input.csv"
    implementation.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    implementation.write_text("MODEL = 1\n", encoding="utf-8")
    source.write_text("lap,time\n1,90\n", encoding="utf-8")
    payload = {
        "schema_version": "artifact_v1",
        "implementation_manifest": {
            "packages/model.py": sha256_file(implementation),
        },
        "input_manifest": [
            {"path": "data/input.csv", "sha256": sha256_file(source)},
            malformed_reference,
        ],
    }
    artifact = tmp_path / "artifacts/result.json"
    _write_json(artifact, payload)

    with pytest.raises(ValueError, match="missing or invalid"):
        validate_artifact_provenance(
            root=tmp_path,
            artifact_path=artifact,
            expected_sha256=sha256_file(artifact),
            expected_schema_version="artifact_v1",
        )


def test_recursive_artifact_provenance_ignores_ordinary_unpaired_paths(
    tmp_path: Path,
) -> None:
    implementation = tmp_path / "packages/model.py"
    source = tmp_path / "data/input.csv"
    implementation.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    implementation.write_text("MODEL = 1\n", encoding="utf-8")
    source.write_text("lap,time\n1,90\n", encoding="utf-8")
    payload = {
        "schema_version": "artifact_v1",
        "implementation_manifest": {
            "packages/model.py": sha256_file(implementation),
        },
        "input_manifest": {
            "data/input.csv": sha256_file(source),
        },
        "runtime_diagnostics": [
            {"path": "runtime/search-location", "exists": False},
            {"path": "/opt/local/lib", "exists": False},
            {"cache_path": "runtime/cache", "exists": False},
        ],
    }
    artifact = tmp_path / "artifacts/result.json"
    _write_json(artifact, payload)

    result = validate_artifact_provenance(
        root=tmp_path,
        artifact_path=artifact,
        expected_sha256=sha256_file(artifact),
        expected_schema_version="artifact_v1",
    )

    assert result["transitive_file_count"] == 2
    assert result["provenance_closure_status"] == "pass"
