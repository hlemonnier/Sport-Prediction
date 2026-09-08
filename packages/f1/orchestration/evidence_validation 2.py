"""Fail-closed provenance validation for frozen F1 research artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMPLEMENTATION_FIELDS = ("implementation_manifest", "code_manifest")
_INPUT_FIELDS = (
    "input_manifest",
    "data_input_manifest",
    "source_manifest",
    "source_artifact",
    "feature_input_files",
    "target_input_files",
)
_AGGREGATE_FIELD_BY_DIGEST = {
    "implementation_manifest_sha256": "implementation_manifest",
    "code_manifest_sha256": "code_manifest",
    "input_manifest_sha256": "input_manifest",
    "data_input_manifest_sha256": "input_manifest",
    "feature_input_manifest_sha256": "feature_input_files",
    "target_input_manifest_sha256": "target_input_files",
    "source_manifest_sha256": "source_manifest",
    "configuration_manifest_sha256": "configuration_manifest",
    "protocol_sha256": "protocol",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _valid_sha(value: object) -> bool:
    return bool(isinstance(value, str) and _SHA256.fullmatch(value))


def _safe_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    path = PurePosixPath(value.replace("\\", "/"))
    return bool(
        not path.is_absolute()
        and ".." not in path.parts
        and path.parts
    )


def _path_like(value: object) -> bool:
    """Heuristically identify a path used as a mapping key.

    Explicit ``path``/``*_path`` fields are unambiguous and use the broader
    ``_safe_relative_path`` check.  Mapping keys require at least one directory
    component so ordinary labels such as ``model.py`` are not reclassified as
    provenance records outside a declared manifest.
    """

    if not _safe_relative_path(value):
        return False
    path = PurePosixPath(str(value).replace("\\", "/"))
    return len(path.parts) >= 2


def _validated_reference_pair(
    value: Mapping[str, object],
    *,
    path_key: str,
    digest_key: str,
    strict_manifest: bool,
) -> tuple[str, str] | None:
    """Return one explicit path/digest pair or reject a malformed reference.

    Outside a declared manifest, an unpaired ``*_path`` field can be ordinary
    runtime metadata and is ignored.  Once its digest partner is present, or
    while traversing a declared manifest, it is a provenance claim and must be
    complete, repository-relative, and SHA-256 bound.
    """

    if path_key not in value:
        return None
    if digest_key not in value and not strict_manifest:
        return None

    raw_path = value.get(path_key)
    if not _safe_relative_path(raw_path):
        raise ValueError(
            f"manifest reference {path_key} must be a safe repository-relative path"
        )
    digest = value.get(digest_key)
    if not _valid_sha(digest):
        raise ValueError(
            f"manifest reference {raw_path} has missing or invalid {digest_key}"
        )
    return str(raw_path), str(digest)


def _iter_manifest_references(
    value: object,
    *,
    strict_manifest: bool = False,
) -> Iterable[tuple[str, str]]:
    """Yield every supported path/digest representation from a manifest.

    ``strict_manifest`` is used for fields whose schema explicitly declares a
    file manifest.  It prevents a malformed entry from disappearing merely
    because a second entry in the same manifest is valid.
    """

    if isinstance(value, list):
        for item in value:
            yield from _iter_manifest_references(
                item,
                strict_manifest=strict_manifest,
            )
        return
    if not isinstance(value, Mapping):
        return

    path_reference = _validated_reference_pair(
        value,
        path_key="path",
        digest_key="sha256",
        strict_manifest=strict_manifest,
    )
    if path_reference is not None:
        yield path_reference
    prefix_reference = _validated_reference_pair(
        value,
        path_key="prefix_path",
        digest_key="prefix_sha256",
        strict_manifest=strict_manifest,
    )
    if prefix_reference is not None:
        yield prefix_reference

    for key, item in value.items():
        if not _path_like(key):
            continue
        if _valid_sha(item):
            yield str(key), str(item)
            continue
        if isinstance(item, Mapping) and "sha256" in item:
            digest = item.get("sha256")
            if not _valid_sha(digest):
                raise ValueError(
                    f"manifest reference {key} has missing or invalid sha256"
                )
            yield str(key), str(digest)
            continue
        if strict_manifest:
            raise ValueError(
                f"manifest reference {key} has missing or invalid sha256"
            )

    # Generic paired fields such as output_path/output_sha256 or
    # telemetry_path/telemetry_sha256 occur inside nested dataset manifests.
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or key in {"path", "prefix_path"}
            or not key.endswith("_path")
        ):
            continue
        paired = f"{key[:-5]}_sha256"
        reference = _validated_reference_pair(
            value,
            path_key=key,
            digest_key=paired,
            strict_manifest=strict_manifest,
        )
        if reference is not None:
            yield reference

    for item in value.values():
        if isinstance(item, (Mapping, list)):
            yield from _iter_manifest_references(
                item,
                strict_manifest=strict_manifest,
            )


def _unique_references(
    value: object,
    *,
    strict_manifest: bool = False,
) -> dict[str, str]:
    references: dict[str, str] = {}
    for raw_path, digest in _iter_manifest_references(
        value,
        strict_manifest=strict_manifest,
    ):
        normalized = PurePosixPath(raw_path.replace("\\", "/")).as_posix()
        previous = references.get(normalized)
        if previous is not None and previous != digest:
            raise ValueError(
                f"conflicting SHA-256 declarations for {normalized}: {previous} != {digest}"
            )
        references[normalized] = digest
    return references


def _resolve_regular_file(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"artifact reference escapes repository root: {relative_path}")
    resolved = (root / Path(*pure.parts)).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"artifact reference resolves outside repository root: {relative_path}")
    if not resolved.is_file():
        raise FileNotFoundError(f"artifact reference is not a regular file: {relative_path}")
    return resolved


def _declared_aggregate_digests(payload: Mapping[str, object]) -> dict[str, str]:
    declared: dict[str, str] = {}
    for key, value in payload.items():
        if key in _AGGREGATE_FIELD_BY_DIGEST and _valid_sha(value):
            declared[key] = str(value)
    nested = payload.get("manifest_hashes")
    if isinstance(nested, Mapping):
        for key, value in nested.items():
            if key in _AGGREGATE_FIELD_BY_DIGEST and _valid_sha(value):
                previous = declared.get(str(key))
                if previous is not None and previous != value:
                    raise ValueError(f"conflicting aggregate digest declaration for {key}")
                declared[str(key)] = str(value)
    return declared


def _validate_aggregate_digests(payload: Mapping[str, object]) -> int:
    checked = 0
    for digest_key, expected in _declared_aggregate_digests(payload).items():
        field_name = _AGGREGATE_FIELD_BY_DIGEST[digest_key]
        if field_name not in payload:
            raise ValueError(
                f"{digest_key} is declared but {field_name} is absent"
            )
        observed = canonical_json_sha256(payload[field_name])
        if observed != expected:
            raise ValueError(
                f"aggregate digest mismatch for {field_name}: {observed} != {expected}"
            )
        checked += 1
    return checked


@dataclass
class _ClosureState:
    root: Path
    references: dict[str, str] = field(default_factory=dict)
    roles: dict[str, set[str]] = field(default_factory=dict)
    visited_json: set[str] = field(default_factory=set)
    aggregate_digest_count: int = 0

    def add_reference(self, relative_path: str, digest: str, *, role: str) -> None:
        normalized = PurePosixPath(relative_path.replace("\\", "/")).as_posix()
        previous = self.references.get(normalized)
        if previous is not None and previous != digest:
            raise ValueError(
                f"conflicting SHA-256 declarations for {normalized}: {previous} != {digest}"
            )
        path = _resolve_regular_file(self.root, normalized)
        observed = sha256_file(path)
        if observed != digest:
            raise ValueError(
                f"referenced file digest mismatch for {normalized}: {observed} != {digest}"
            )
        self.references[normalized] = digest
        self.roles.setdefault(normalized, set()).add(role)
        if path.suffix.lower() == ".json" and normalized not in self.visited_json:
            self.visited_json.add(normalized)
            nested = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(nested, Mapping):
                raise ValueError(f"referenced JSON root must be a mapping: {normalized}")
            self.scan_payload(nested, origin=normalized)

    def scan_payload(self, payload: Mapping[str, object], *, origin: str) -> None:
        self.aggregate_digest_count += _validate_aggregate_digests(payload)
        claimed_fields: set[str] = set()
        for field_name in _IMPLEMENTATION_FIELDS:
            if field_name not in payload:
                continue
            claimed_fields.add(field_name)
            for path, digest in _unique_references(
                payload[field_name],
                strict_manifest=True,
            ).items():
                self.add_reference(path, digest, role="implementation")
        for field_name in _INPUT_FIELDS:
            if field_name not in payload:
                continue
            claimed_fields.add(field_name)
            for path, digest in _unique_references(
                payload[field_name],
                strict_manifest=True,
            ).items():
                self.add_reference(path, digest, role="input")

        # Dataset JSON often nests ordinary ``files`` maps rather than naming
        # them input_manifest. Scan remaining structures as transitive inputs.
        remainder = {
            key: value
            for key, value in payload.items()
            if key not in claimed_fields
        }
        for path, digest in _unique_references(remainder).items():
            self.add_reference(path, digest, role="input")


def validate_artifact_provenance(
    *,
    root: Path,
    artifact_path: Path,
    expected_sha256: str,
    expected_schema_version: str,
) -> dict[str, object]:
    """Validate a frozen artifact plus its recursive code/data closure."""

    root = root.resolve()
    artifact_path = artifact_path.resolve()
    if artifact_path != root and root not in artifact_path.parents:
        raise ValueError("registered artifact is outside repository root")
    if not artifact_path.is_file():
        raise FileNotFoundError(f"registered artifact is not a file: {artifact_path}")
    observed_sha256 = sha256_file(artifact_path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"registered artifact digest mismatch: {observed_sha256} != {expected_sha256}"
        )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("registered artifact root must be a mapping")
    observed_schema = payload.get("schema_version")
    if observed_schema != expected_schema_version:
        raise ValueError(
            f"registered artifact schema mismatch: {observed_schema!r} != {expected_schema_version!r}"
        )

    direct_implementation: dict[str, str] = {}
    for field_name in _IMPLEMENTATION_FIELDS:
        if field_name in payload:
            direct_implementation.update(
                _unique_references(
                    payload[field_name],
                    strict_manifest=True,
                )
            )
    direct_input: dict[str, str] = {}
    for field_name in _INPUT_FIELDS:
        if field_name in payload:
            direct_input.update(
                _unique_references(
                    payload[field_name],
                    strict_manifest=True,
                )
            )
    if not direct_implementation:
        raise ValueError("registered artifact has no implementation/code manifest entries")
    if not direct_input:
        raise ValueError("registered artifact has no input/source manifest entries")

    state = _ClosureState(root=root)
    state.scan_payload(payload, origin=str(artifact_path.relative_to(root)))
    implementation_files = {
        path for path, roles in state.roles.items() if "implementation" in roles
    }
    input_files = {path for path, roles in state.roles.items() if "input" in roles}
    return {
        "artifact_sha256_valid": True,
        "schema_version_valid": True,
        "implementation_manifest_direct_entries": int(len(direct_implementation)),
        "input_manifest_direct_entries": int(len(direct_input)),
        "implementation_manifest_files": int(len(implementation_files)),
        "input_manifest_files": int(len(input_files)),
        "transitive_file_count": int(len(state.references)),
        "nested_json_count": int(len(state.visited_json)),
        "aggregate_manifest_digest_count": int(state.aggregate_digest_count),
        "implementation_manifest_valid": True,
        "input_manifest_valid": True,
        "aggregate_manifest_digests_valid": True,
        "provenance_closure_status": "pass",
    }


__all__ = [
    "canonical_json_sha256",
    "sha256_file",
    "validate_artifact_provenance",
]
