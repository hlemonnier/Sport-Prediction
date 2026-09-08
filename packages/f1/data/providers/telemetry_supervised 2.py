"""Build labelled pre-Qualifying telemetry bags without row pseudo-replication.

The feature cache stores up to a few rehearsal laps per driver.  The supervised
unit is nevertheless one driver-event: every bag owns one Qualifying target and
one or more independently hashed telemetry tensors.  Target files are separate,
post-Qualifying, and explicitly ineligible for inference.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from packages.f1.data.providers.telemetry_cache import (
    NORMALIZED_TELEMETRY_CHANNELS,
    sha256_file,
    validate_cached_telemetry_tensor,
)


SUPERVISED_TELEMETRY_SCHEMA_VERSION = "f1_prequal_telemetry_supervised_bags_v2"
TELEMETRY_SOURCE_MANIFEST_SCHEMA_VERSION = "f1_prequal_telemetry_cache_v2"
TARGET_CONTRACT = "grand_prix_qualifying_valid_lap_with_optional_best_legal_lap_v2"
LAP_TIME_OBSERVED_STATUS = "observed_best_legal_lap"
LAP_TIME_CENSORED_STATUS = "censored_no_legal_lap"
FEATURE_CAPTURE_SEMANTICS = "manifest_timestamped_prequalifying_telemetry"
TARGET_CAPTURE_SEMANTICS = "separate_postqualifying_training_truth"
SUPERVISED_TARGET_SEMANTICS: dict[str, str] = {
    "valid_lap_label": (
        "has_legal_qualifying_lap is a Boolean outcome for every telemetry bag"
    ),
    "lap_time_label": (
        "lap_time_seconds is observed only when has_legal_qualifying_lap is true"
    ),
    "censoring": (
        "lap_time_target_status=censored_no_legal_lap keeps the bag but forbids "
        "using it as a continuous lap-time regression target"
    ),
    "advancement": (
        "reaches_q2/reaches_q3 is the union of recorded later-session times and "
        "capacity-limited official classification-position advancement for "
        "explicit 20/22-car fields"
    ),
}

# The current Grand Prix Qualifying format always retains ten cars for Q3.
# A 20-car field eliminates five in Q1, while the 22-car 2026 field eliminates
# six.  Keeping this table explicit prevents an unsupported field size from
# being silently forced through an assumed arithmetic rule.
QUALIFYING_ADVANCEMENT_SLOTS: dict[int, tuple[int, int]] = {
    20: (15, 10),
    22: (16, 10),
}


class TelemetrySupervisedDatasetError(ValueError):
    """Raised when feature and target evidence cannot form a causal bag."""


def canonical_sha256(payload: Any) -> str:
    """Hash JSON-like evidence using one stable representation."""

    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _portable(path: Path, root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(root.expanduser().resolve()))
    except ValueError:
        return str(resolved)


def _resolve_path(value: object, *, root: Path) -> Path:
    text = str(value or "").strip()
    if not text:
        raise TelemetrySupervisedDatasetError("evidence path is missing")
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _file_evidence(path: Path, *, root: Path, role: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{role} file does not exist: {path}")
    return {
        "path": _portable(path, root),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "role": str(role),
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TelemetrySupervisedDatasetError(f"invalid JSON evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TelemetrySupervisedDatasetError(f"JSON evidence must be an object: {path}")
    return payload


def _without(mapping: Mapping[str, Any], field: str) -> dict[str, Any]:
    payload = dict(mapping)
    payload.pop(field, None)
    return payload


def _assert_canonical_hash(
    mapping: Mapping[str, Any],
    field: str,
    *,
    label: str,
) -> None:
    declared = str(mapping.get(field) or "").strip().lower()
    actual = canonical_sha256(_without(mapping, field))
    if declared != actual:
        raise TelemetrySupervisedDatasetError(
            f"{label} canonical hash mismatch: "
            f"declared={declared or '<missing>'}, actual={actual}"
        )


def _repository_relative_file(
    value: object,
    *,
    root: Path,
    label: str,
) -> tuple[str, Path]:
    """Resolve one canonical repository-relative evidence path, fail closed."""

    text = str(value or "").strip()
    if not text:
        raise TelemetrySupervisedDatasetError(f"{label} path is missing")
    declared = Path(text).expanduser()
    if declared.is_absolute():
        raise TelemetrySupervisedDatasetError(
            f"{label} path must be repository-relative: {text}"
        )
    repo_root = root.expanduser().resolve()
    resolved = (repo_root / declared).resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError as exc:
        raise TelemetrySupervisedDatasetError(
            f"{label} path escapes the repository: {text}"
        ) from exc
    canonical = relative.as_posix()
    if declared.as_posix() != canonical:
        raise TelemetrySupervisedDatasetError(
            f"{label} path is not canonical repository-relative evidence: {text}"
        )
    if not resolved.is_file():
        raise TelemetrySupervisedDatasetError(
            f"{label} file does not exist: {resolved}"
        )
    return canonical, resolved


def _validate_input_file_manifest(
    manifest: Mapping[str, Any],
    *,
    files_field: str,
    hash_field: str,
    allowed_roles: frozenset[str],
    root: Path,
) -> tuple[list[dict[str, object]], dict[str, tuple[dict[str, object], Path]]]:
    raw_rows = manifest.get(files_field)
    if not isinstance(raw_rows, list) or not raw_rows:
        raise TelemetrySupervisedDatasetError(f"{files_field} must be a non-empty list")
    rows: list[dict[str, object]] = []
    by_path: dict[str, tuple[dict[str, object], Path]] = {}
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise TelemetrySupervisedDatasetError(
                f"{files_field}[{index}] must be an object"
            )
        row = dict(raw)
        role = str(row.get("role") or "").strip()
        if role not in allowed_roles:
            raise TelemetrySupervisedDatasetError(
                f"{files_field}[{index}] has unsupported role {role!r}"
            )
        path, resolved = _repository_relative_file(
            row.get("path"), root=root, label=f"{files_field}[{index}]"
        )
        if path in by_path:
            raise TelemetrySupervisedDatasetError(
                f"{files_field} contains duplicate path {path}"
            )
        declared_sha = str(row.get("sha256") or "").strip().lower()
        if len(declared_sha) != 64 or any(
            character not in "0123456789abcdef" for character in declared_sha
        ):
            raise TelemetrySupervisedDatasetError(
                f"{files_field}[{index}] has invalid SHA-256"
            )
        try:
            declared_size = int(row["size_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TelemetrySupervisedDatasetError(
                f"{files_field}[{index}] has invalid size_bytes"
            ) from exc
        if isinstance(row.get("size_bytes"), bool) or declared_size < 0:
            raise TelemetrySupervisedDatasetError(
                f"{files_field}[{index}] has invalid size_bytes"
            )
        actual_sha = sha256_file(resolved)
        actual_size = int(resolved.stat().st_size)
        if declared_sha != actual_sha:
            raise TelemetrySupervisedDatasetError(
                f"{files_field} SHA-256 mismatch for {path}: "
                f"declared={declared_sha}, actual={actual_sha}"
            )
        if declared_size != actual_size:
            raise TelemetrySupervisedDatasetError(
                f"{files_field} size mismatch for {path}: "
                f"declared={declared_size}, actual={actual_size}"
            )
        normalized = {
            "path": path,
            "sha256": declared_sha,
            "size_bytes": declared_size,
            "role": role,
        }
        if row != normalized:
            raise TelemetrySupervisedDatasetError(
                f"{files_field}[{index}] is not canonical file evidence"
            )
        rows.append(normalized)
        by_path[path] = (normalized, resolved)
    if [str(row["path"]) for row in rows] != sorted(str(row["path"]) for row in rows):
        raise TelemetrySupervisedDatasetError(f"{files_field} must be sorted by path")
    declared_manifest_hash = str(manifest.get(hash_field) or "").strip().lower()
    actual_manifest_hash = canonical_sha256(rows)
    if declared_manifest_hash != actual_manifest_hash:
        raise TelemetrySupervisedDatasetError(
            f"{hash_field} mismatch: declared={declared_manifest_hash or '<missing>'}, "
            f"actual={actual_manifest_hash}"
        )
    return rows, by_path


def _lap_seconds(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    unresolved = numeric.isna()
    if unresolved.any():
        numeric = numeric.fillna(
            pd.to_timedelta(values.where(unresolved), errors="coerce").dt.total_seconds()
        )
    return pd.to_numeric(numeric, errors="coerce").astype(float)


def _bool_series(values: pd.Series, *, default: bool) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
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


def _optional_positive(value: object) -> float | None:
    if isinstance(value, (pd.Timedelta, np.timedelta64)):
        value = pd.to_timedelta(value).total_seconds()
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        duration = pd.to_timedelta(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(duration):
            return None
        numeric = float(duration.total_seconds())
    return numeric if np.isfinite(numeric) and numeric > 0.0 else None


def _optional_int(value: object) -> int | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return int(numeric) if np.isfinite(numeric) else None


def _has_session_time(value: object) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(str(value).strip())


def _utc_timestamp(value: object, *, label: str) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise TelemetrySupervisedDatasetError(f"{label} must be a valid UTC timestamp")
    return pd.Timestamp(timestamp)


def _driver_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    return next((name for name in candidates if name in frame.columns), None)


def _best_legal_qualifying_targets(laps: pd.DataFrame) -> dict[str, dict[str, object]]:
    driver_column = _driver_column(laps, ("Driver", "Abbreviation", "driver_id"))
    if driver_column is None or "LapTime" not in laps.columns or "Deleted" not in laps.columns:
        raise TelemetrySupervisedDatasetError(
            "Qualifying laps require Driver/Abbreviation, LapTime, and Deleted columns"
        )
    frame = laps.copy()
    frame["_driver_id"] = frame[driver_column].fillna("").astype(str).str.strip().str.upper()
    frame["_lap_time_seconds"] = _lap_seconds(frame["LapTime"])
    eligible = (
        frame["_driver_id"].ne("")
        & frame["_lap_time_seconds"].map(np.isfinite)
        & frame["_lap_time_seconds"].gt(0.0)
    )
    eligible &= ~_bool_series(frame["Deleted"], default=True)
    legal = frame.loc[eligible].copy()
    if legal.empty:
        return {}
    legal["_lap_number_sort"] = pd.to_numeric(
        legal.get("LapNumber", pd.Series(index=legal.index, dtype=float)), errors="coerce"
    ).fillna(float("inf"))
    best = (
        legal.sort_values(
            ["_driver_id", "_lap_time_seconds", "_lap_number_sort"],
            kind="mergesort",
        )
        .drop_duplicates("_driver_id", keep="first")
    )
    targets: dict[str, dict[str, object]] = {}
    for _, row in best.iterrows():
        driver_id = str(row["_driver_id"])
        targets[driver_id] = {
            "lap_time_seconds": float(row["_lap_time_seconds"]),
            "lap_number": _optional_int(row.get("LapNumber")),
            "sector1_seconds": _optional_positive(row.get("Sector1Time")),
            "sector2_seconds": _optional_positive(row.get("Sector2Time")),
            "sector3_seconds": _optional_positive(row.get("Sector3Time")),
        }
    return targets


def _qualifying_advancement_slots(field_size: int) -> tuple[int, int]:
    try:
        return QUALIFYING_ADVANCEMENT_SLOTS[int(field_size)]
    except KeyError as exc:
        supported = ", ".join(str(value) for value in sorted(QUALIFYING_ADVANCEMENT_SLOTS))
        raise TelemetrySupervisedDatasetError(
            "cannot infer Qualifying advancement from classification position for "
            f"unsupported field size {field_size}; supported field sizes: {supported}"
        ) from exc


def _stage_labels(results: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    if results is None or results.empty:
        return {}
    driver_column = _driver_column(
        results,
        ("Abbreviation", "Driver", "driver_id", "DriverId"),
    )
    if driver_column is None:
        return {}
    normalized_drivers = results[driver_column].fillna("").astype(str).str.strip().str.upper()
    field_size = int(normalized_drivers.loc[normalized_drivers.ne("")].nunique())
    classification_positions = pd.Series(np.nan, index=results.index, dtype=float)
    for candidate in ("Position", "ClassifiedPosition"):
        if candidate not in results.columns:
            continue
        parsed = pd.to_numeric(results[candidate], errors="coerce")
        if parsed.map(np.isfinite).any():
            classification_positions = parsed
            break
    has_numeric_classification = bool(classification_positions.map(np.isfinite).any())
    q2_slots: int | None = None
    q3_slots: int | None = None
    if has_numeric_classification:
        q2_slots, q3_slots = _qualifying_advancement_slots(field_size)

    raw_labels: list[dict[str, object]] = []
    seen_drivers: set[str] = set()
    for provider_order, (_, row) in enumerate(results.iterrows()):
        raw_driver_id = row.get(driver_column)
        driver_id = "" if pd.isna(raw_driver_id) else str(raw_driver_id).strip().upper()
        if not driver_id:
            continue
        if driver_id in seen_drivers:
            raise TelemetrySupervisedDatasetError(
                f"duplicate Qualifying classification for driver {driver_id}"
            )
        seen_drivers.add(driver_id)
        has_q1 = _has_session_time(row.get("Q1"))
        has_q2 = _has_session_time(row.get("Q2"))
        has_q3 = _has_session_time(row.get("Q3"))
        position = _optional_int(classification_positions.loc[row.name])
        raw_labels.append(
            {
                "driver_id": driver_id,
                "has_q1_time": has_q1,
                "has_q2_time": has_q2,
                "has_q3_time": has_q3,
                "classification_position": position,
                "provider_order": int(provider_order),
            }
        )

    # Official positions can recover a driver who advanced but set no time in
    # the next segment.  They may also be shifted by a post-session penalty or
    # disqualification, so position inference fills only the unused stage
    # capacity after recorded Q2/Q3 times are counted.  It never creates a 17th
    # Q2 participant or an 11th Q3 participant.
    q2_position_drivers: set[str] = set()
    q3_position_drivers: set[str] = set()
    if q2_slots is not None and q3_slots is not None:
        timed_q2_count = sum(
            bool(row["has_q2_time"] or row["has_q3_time"]) for row in raw_labels
        )
        timed_q3_count = sum(bool(row["has_q3_time"]) for row in raw_labels)

        def position_candidates(*, slots: int, timed_fields: tuple[str, ...]) -> list[str]:
            candidates = [
                row
                for row in raw_labels
                if not any(bool(row[field]) for field in timed_fields)
                and row["classification_position"] is not None
                and 1 <= int(row["classification_position"]) <= slots
            ]
            candidates.sort(
                key=lambda row: (
                    int(row["classification_position"]),
                    int(row["provider_order"]),
                )
            )
            return [str(row["driver_id"]) for row in candidates]

        q2_missing = max(0, q2_slots - timed_q2_count)
        q3_missing = max(0, q3_slots - timed_q3_count)
        q2_position_drivers.update(
            position_candidates(
                slots=q2_slots,
                timed_fields=("has_q2_time", "has_q3_time"),
            )[:q2_missing]
        )
        q3_position_drivers.update(
            position_candidates(slots=q3_slots, timed_fields=("has_q3_time",))[
                :q3_missing
            ]
        )

    labels: dict[str, dict[str, object]] = {}
    for raw in raw_labels:
        driver_id = str(raw["driver_id"])
        has_q1 = bool(raw["has_q1_time"])
        has_q2 = bool(raw["has_q2_time"])
        has_q3 = bool(raw["has_q3_time"])
        position = _optional_int(raw["classification_position"])
        reaches_q2_from_position = driver_id in q2_position_drivers
        reaches_q3_from_position = driver_id in q3_position_drivers
        reaches_q3 = bool(has_q3 or reaches_q3_from_position)
        reaches_q2 = bool(
            has_q2 or has_q3 or reaches_q2_from_position or reaches_q3_from_position
        )
        labels[driver_id] = {
            "has_q1_time": bool(has_q1),
            "has_q2_time": bool(has_q2),
            "has_q3_time": bool(has_q3),
            "reaches_q2": reaches_q2,
            "reaches_q3": reaches_q3,
            "stage_reached": 3 if reaches_q3 else (2 if reaches_q2 else 1),
            "classification_position": position,
            "classification_field_size": field_size,
            "q2_advancement_slot_count": q2_slots,
            "q3_advancement_slot_count": q3_slots,
            "reaches_q2_from_position": reaches_q2_from_position,
            "reaches_q3_from_position": reaches_q3_from_position,
        }
    return labels


def _qualifying_session(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    sessions = metadata.get("sessions", [])
    if not isinstance(sessions, list):
        raise TelemetrySupervisedDatasetError("weekend metadata sessions must be a list")
    matching = [
        session
        for session in sessions
        if isinstance(session, Mapping)
        and str(session.get("session_name") or "").strip().lower() == "qualifying"
    ]
    if len(matching) != 1:
        raise TelemetrySupervisedDatasetError(
            "weekend metadata must contain exactly one Grand Prix Qualifying "
            f"session; found {len(matching)}"
        )
    return matching[0]


def _declared_file_digest(session: Mapping[str, Any], role: str) -> str | None:
    files = session.get("files")
    if not isinstance(files, Mapping):
        return None
    evidence = files.get(role)
    if not isinstance(evidence, Mapping):
        return None
    text = str(evidence.get("sha256") or "").strip().lower()
    return text or None


def _assert_declared_digest(
    session: Mapping[str, Any],
    *,
    role: str,
    evidence: Mapping[str, object],
) -> None:
    declared = _declared_file_digest(session, role)
    if declared is not None and declared != evidence["sha256"]:
        raise TelemetrySupervisedDatasetError(
            f"Qualifying {role} SHA-256 does not match weekend metadata"
        )


def _round_directory(weekends_year_root: Path, round_number: int) -> Path:
    matches = sorted(weekends_year_root.glob(f"round_{int(round_number):02d}_*"))
    if len(matches) != 1:
        raise TelemetrySupervisedDatasetError(
            f"expected exactly one weekend directory for round {round_number}, found {len(matches)}"
        )
    return matches[0]


def _target_sources(
    weekend_dir: Path,
    *,
    root: Path,
    expected_year: int,
    expected_round: int,
    expected_event_name: str,
) -> tuple[
    Mapping[str, Any],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
    list[dict[str, object]],
]:
    metadata_path = weekend_dir / "weekend_metadata.json"
    metadata = _load_json(metadata_path)
    try:
        metadata_year = int(metadata["year"])
        metadata_round = int(metadata["round_number"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TelemetrySupervisedDatasetError(
            f"weekend metadata has invalid event identity: {metadata_path}"
        ) from exc
    metadata_event_name = str(metadata.get("event_name") or "").strip()
    if (metadata_year, metadata_round) != (int(expected_year), int(expected_round)):
        raise TelemetrySupervisedDatasetError(
            "weekend target identity does not match the telemetry event"
        )
    if not metadata_event_name or metadata_event_name.casefold() != expected_event_name.casefold():
        raise TelemetrySupervisedDatasetError(
            "weekend target name does not match the telemetry event"
        )
    session = _qualifying_session(metadata)
    qualifying_start = _utc_timestamp(
        session.get("scheduled_start_utc"),
        label="Qualifying scheduled_start_utc",
    )
    target_available = _utc_timestamp(
        session.get("available_at") or session.get("captured_at"),
        label="Qualifying target availability",
    )
    if target_available < qualifying_start:
        raise TelemetrySupervisedDatasetError(
            "Qualifying target evidence claims availability before the session starts"
        )
    laps_path = _resolve_path(session.get("laps_path"), root=root)
    laps_evidence = _file_evidence(laps_path, root=root, role="qualifying_laps_target")
    _assert_declared_digest(session, role="laps", evidence=laps_evidence)
    laps = pd.read_csv(laps_path)
    targets = _best_legal_qualifying_targets(laps)

    results_path_text = str(session.get("results_path") or "").strip()
    results_evidence: dict[str, object] | None = None
    results: pd.DataFrame | None = None
    if results_path_text:
        results_path = _resolve_path(results_path_text, root=root)
        results_evidence = _file_evidence(
            results_path,
            root=root,
            role="qualifying_results_stage_labels",
        )
        _assert_declared_digest(session, role="results", evidence=results_evidence)
        results = pd.read_csv(results_path)
    labels = _stage_labels(results)
    evidence = [
        _file_evidence(metadata_path, root=root, role="weekend_target_metadata"),
        laps_evidence,
    ]
    if results_evidence is not None:
        evidence.append(results_evidence)
    return session, targets, labels, evidence


def _feature_tensor_payload(
    record: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    validation = validate_cached_telemetry_tensor(record, root=root)
    tensor_path = _resolve_path(record.get("telemetry_path"), root=root)
    evidence = _file_evidence(tensor_path, root=root, role="prequal_telemetry_tensor")
    if evidence["sha256"] != validation["sha256"]:
        raise TelemetrySupervisedDatasetError(
            "validated telemetry digest changed during dataset build"
        )
    payload = {
        "path": evidence["path"],
        "sha256": evidence["sha256"],
        "lap_number": int(record["lap_number"]),
        "push_lap_rank": _optional_int(record.get("push_lap_rank")),
        "rehearsal_lap_time_seconds": _optional_positive(record.get("lap_time_seconds")),
        "feature_as_of": validation["feature_as_of"],
        "shape": validation["shape"],
        "channels": validation["channels"],
        "distance_bins": validation["distance_bins"],
        "expected_lap_distance_m": validation["expected_lap_distance_m"],
        "distance_coverage": validation["distance_coverage"],
    }
    return payload, evidence


def _deduplicate_file_evidence(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_path: dict[str, dict[str, object]] = {}
    for row in rows:
        path = str(row["path"])
        normalized = dict(row)
        existing = by_path.get(path)
        if existing is not None and existing != normalized:
            raise TelemetrySupervisedDatasetError(f"conflicting file evidence for {path}")
        by_path[path] = normalized
    return [by_path[path] for path in sorted(by_path)]


def build_prequal_telemetry_supervised_manifest(
    *,
    root: Path,
    telemetry_root: Path,
    weekends_root: Path,
    year: int = 2026,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Join validated rehearsal tensors to one post-Q outcome per driver-event.

    A bag is retained even when the driver recorded no legal Qualifying lap.
    Such a bag owns a valid-lap classification target and an explicitly
    censored lap-time target; it is never discarded merely because the
    continuous target is unobserved.
    """

    repo_root = root.expanduser().resolve()
    telemetry_year_root = telemetry_root.expanduser().resolve() / str(int(year))
    weekends_year_root = weekends_root.expanduser().resolve() / str(int(year))
    manifest_paths = sorted(telemetry_year_root.glob("round_*/telemetry_manifest.json"))
    if not manifest_paths:
        raise TelemetrySupervisedDatasetError(
            f"no telemetry manifests found for {year} under {telemetry_year_root}"
        )

    bags: list[dict[str, object]] = []
    feature_evidence: list[dict[str, object]] = []
    target_evidence: list[dict[str, object]] = []
    seen_events: set[int] = set()
    seen_driver_events: set[tuple[int, str]] = set()
    seen_tensor_records: set[tuple[int, str, int]] = set()
    seen_tensor_paths: set[str] = set()
    target_only_driver_events = 0

    for manifest_path in manifest_paths:
        manifest = _load_json(manifest_path)
        try:
            event_key = int(manifest["event_key"])
            manifest_year = int(manifest.get("year", event_key // 100))
            round_number = int(manifest.get("round", event_key % 100))
        except (KeyError, TypeError, ValueError) as exc:
            raise TelemetrySupervisedDatasetError(
                f"telemetry manifest has invalid event identity: {manifest_path}"
            ) from exc
        if manifest_year != int(year) or event_key // 100 != int(year):
            raise TelemetrySupervisedDatasetError(
                f"telemetry event {event_key} does not belong to requested year {year}"
            )
        if round_number != event_key % 100:
            raise TelemetrySupervisedDatasetError(
                f"telemetry round {round_number} does not match event key {event_key}"
            )
        if event_key in seen_events:
            raise TelemetrySupervisedDatasetError(
                f"duplicate telemetry manifest for event {event_key}"
            )
        seen_events.add(event_key)

        manifest_evidence = _file_evidence(
            manifest_path,
            root=repo_root,
            role="prequal_telemetry_manifest",
        )
        feature_evidence.append(manifest_evidence)
        feature_records = manifest.get("feature_records", [])
        if not isinstance(feature_records, list) or not feature_records:
            raise TelemetrySupervisedDatasetError(
                f"telemetry manifest {manifest_path} has no feature records"
            )

        event_name = str(manifest.get("event_name") or "").strip()
        if not event_name:
            raise TelemetrySupervisedDatasetError(
                f"telemetry manifest has no event_name: {manifest_path}"
            )
        manifest_qualifying_start = _utc_timestamp(
            manifest.get("qualifying_start_utc"),
            label=f"telemetry event {event_key} Qualifying cutoff",
        )
        weekend_dir = _round_directory(weekends_year_root, round_number)
        qualifying_session, targets, labels, event_target_evidence = _target_sources(
            weekend_dir,
            root=repo_root,
            expected_year=year,
            expected_round=round_number,
            expected_event_name=event_name,
        )
        qualifying_start = _utc_timestamp(
            qualifying_session.get("scheduled_start_utc"),
            label="Qualifying scheduled_start_utc",
        )
        if manifest_qualifying_start != qualifying_start:
            raise TelemetrySupervisedDatasetError(
                f"telemetry cutoff does not match the Qualifying schedule for event {event_key}"
            )
        target_evidence.extend(event_target_evidence)
        records_by_driver: dict[str, list[Mapping[str, Any]]] = {}
        for raw_record in feature_records:
            if not isinstance(raw_record, Mapping):
                raise TelemetrySupervisedDatasetError("telemetry feature record must be an object")
            try:
                record_event = int(raw_record.get("event_key"))
                lap_number = int(raw_record.get("lap_number"))
            except (TypeError, ValueError) as exc:
                raise TelemetrySupervisedDatasetError(
                    "telemetry feature record requires integer event_key and lap_number"
                ) from exc
            driver_id = str(raw_record.get("driver_id") or "").strip().upper()
            if record_event != event_key or not driver_id:
                raise TelemetrySupervisedDatasetError(
                    f"telemetry feature identity does not match event manifest {event_key}"
                )
            record_qualifying_start = _utc_timestamp(
                raw_record.get("qualifying_start_utc"),
                label=f"telemetry record {event_key}/{driver_id}/{lap_number} cutoff",
            )
            if record_qualifying_start != manifest_qualifying_start:
                raise TelemetrySupervisedDatasetError(
                    f"telemetry record cutoff does not match event {event_key}"
                )
            record_key = (event_key, driver_id, lap_number)
            path_text = str(raw_record.get("telemetry_path") or "").strip()
            path_key = (
                str(_resolve_path(path_text, root=repo_root)) if path_text else ""
            )
            if record_key in seen_tensor_records:
                raise TelemetrySupervisedDatasetError(
                    f"duplicate telemetry record {record_key}"
                )
            if not path_key or path_key in seen_tensor_paths:
                raise TelemetrySupervisedDatasetError(
                    f"duplicate or missing telemetry path for {record_key}"
                )
            seen_tensor_records.add(record_key)
            seen_tensor_paths.add(path_key)
            records_by_driver.setdefault(driver_id, []).append(raw_record)

        target_only_driver_events += len(set(targets).difference(records_by_driver))

        target_source_rows = _deduplicate_file_evidence(event_target_evidence)
        target_source_sha = canonical_sha256(target_source_rows)
        for driver_id in sorted(records_by_driver):
            driver_event_key = (event_key, driver_id)
            if driver_event_key in seen_driver_events:
                raise TelemetrySupervisedDatasetError(
                    f"duplicate supervised driver-event bag {driver_event_key}"
                )
            seen_driver_events.add(driver_event_key)
            tensors: list[dict[str, object]] = []
            for record in sorted(
                records_by_driver[driver_id],
                key=lambda item: (
                    _optional_int(item.get("push_lap_rank")) or 10**9,
                    int(item["lap_number"]),
                ),
            ):
                tensor, tensor_evidence = _feature_tensor_payload(record, root=repo_root)
                tensors.append(tensor)
                feature_evidence.append(tensor_evidence)

            feature_payload = {
                "rehearsal_source": str(manifest.get("rehearsal_source") or ""),
                "qualifying_start_utc": str(manifest.get("qualifying_start_utc") or ""),
                "telemetry_manifest_path": manifest_evidence["path"],
                "telemetry_manifest_sha256": manifest_evidence["sha256"],
                "tensor_count": int(len(tensors)),
                "tensors": tensors,
            }
            feature_payload["feature_bag_sha256"] = canonical_sha256(feature_payload)
            legal_lap = targets.get(driver_id)
            has_legal_lap = legal_lap is not None
            lap_target = legal_lap or {
                "lap_time_seconds": None,
                "lap_number": None,
                "sector1_seconds": None,
                "sector2_seconds": None,
                "sector3_seconds": None,
            }
            target_payload = {
                "target_contract": TARGET_CONTRACT,
                "has_legal_qualifying_lap": has_legal_lap,
                "lap_time_observed": has_legal_lap,
                "lap_time_target_status": (
                    LAP_TIME_OBSERVED_STATUS if has_legal_lap else LAP_TIME_CENSORED_STATUS
                ),
                **lap_target,
                **labels.get(
                    driver_id,
                    {
                        "has_q1_time": None,
                        "has_q2_time": None,
                        "has_q3_time": None,
                        "reaches_q2": None,
                        "reaches_q3": None,
                        "stage_reached": None,
                        "classification_position": None,
                        "classification_field_size": None,
                        "q2_advancement_slot_count": None,
                        "q3_advancement_slot_count": None,
                        "reaches_q2_from_position": None,
                        "reaches_q3_from_position": None,
                    },
                ),
                "target_session": "Qualifying",
                "target_available_at": str(
                    qualifying_session.get("available_at")
                    or qualifying_session.get("captured_at")
                    or ""
                ),
                "target_source_manifest_sha256": target_source_sha,
                "inference_eligible": False,
            }
            target_payload["target_sha256"] = canonical_sha256(target_payload)
            bag = {
                "event_key": int(event_key),
                "year": int(year),
                "round": int(round_number),
                "event_name": event_name,
                "driver_id": driver_id,
                "row_unit": "driver_event_bag",
                "feature": feature_payload,
                "target": target_payload,
            }
            bag["bag_sha256"] = canonical_sha256(bag)
            bags.append(bag)

    bags.sort(key=lambda row: (int(row["event_key"]), str(row["driver_id"])))
    feature_files = _deduplicate_file_evidence(feature_evidence)
    target_files = _deduplicate_file_evidence(target_evidence)
    event_counts: dict[int, int] = {}
    for bag in bags:
        event_key = int(bag["event_key"])
        event_counts[event_key] = int(event_counts.get(event_key, 0) + 1)
    tensor_count = sum(int(bag["feature"]["tensor_count"]) for bag in bags)  # type: ignore[index]
    stage_label_count = sum(
        bag["target"]["stage_reached"] is not None  # type: ignore[index]
        for bag in bags
    )
    complete_sector_count = sum(
        all(
            bag["target"][name] is not None  # type: ignore[index]
            for name in ("sector1_seconds", "sector2_seconds", "sector3_seconds")
        )
        for bag in bags
    )
    observed_lap_time_count = sum(
        bag["target"]["lap_time_observed"] is True  # type: ignore[index]
        for bag in bags
    )
    censored_lap_time_count = int(len(bags) - observed_lap_time_count)
    payload: dict[str, Any] = {
        "schema_version": SUPERVISED_TELEMETRY_SCHEMA_VERSION,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "year": int(year),
        "target_contract": TARGET_CONTRACT,
        "targets_inference_eligible": False,
        "supervised_row_unit": "driver_event_bag",
        "independent_evaluation_unit": "event",
        "tensor_rows_are_independent_supervised_rows": False,
        "feature_capture_semantics": FEATURE_CAPTURE_SEMANTICS,
        "target_capture_semantics": TARGET_CAPTURE_SEMANTICS,
        "target_semantics": dict(SUPERVISED_TARGET_SEMANTICS),
        "feature_input_files": feature_files,
        "feature_input_manifest_sha256": canonical_sha256(feature_files),
        "target_input_files": target_files,
        "target_input_manifest_sha256": canonical_sha256(target_files),
        "bags": bags,
        "bag_set_sha256": canonical_sha256([bag["bag_sha256"] for bag in bags]),
        "audit": {
            "event_count": int(len(event_counts)),
            "driver_event_bag_count": int(len(bags)),
            "validated_tensor_count": int(tensor_count),
            "event_driver_counts": {
                str(event_key): int(count) for event_key, count in sorted(event_counts.items())
            },
            "stage_label_bag_count": int(stage_label_count),
            "observed_lap_time_target_bag_count": int(observed_lap_time_count),
            "censored_lap_time_target_bag_count": int(censored_lap_time_count),
            "complete_sector_target_bag_count": int(complete_sector_count),
            "target_only_driver_event_count": int(target_only_driver_events),
            "duplicate_driver_event_count": 0,
            "target_objects_per_bag": 1,
            "target_objects_per_tensor": 0,
            "inference_eligible_target_count": 0,
        },
    }
    return payload


def validate_prequal_telemetry_supervised_manifest(
    manifest: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    """Revalidate a supervised telemetry manifest and all nested provenance.

    This is intentionally stronger than trusting the manifest's own hashes. It
    opens every repository-relative source file, verifies byte size and digest,
    reconstructs every feature/target/bag object with the same causal builder,
    and then recomputes the complete audit.  A research runner can therefore
    bind to one exact source manifest instead of accepting a self-consistent but
    detached JSON payload.
    """

    if not isinstance(manifest, Mapping):
        raise TelemetrySupervisedDatasetError("supervised manifest must be an object")
    required_semantics: dict[str, object] = {
        "schema_version": SUPERVISED_TELEMETRY_SCHEMA_VERSION,
        "target_contract": TARGET_CONTRACT,
        "targets_inference_eligible": False,
        "supervised_row_unit": "driver_event_bag",
        "independent_evaluation_unit": "event",
        "tensor_rows_are_independent_supervised_rows": False,
        "feature_capture_semantics": FEATURE_CAPTURE_SEMANTICS,
        "target_capture_semantics": TARGET_CAPTURE_SEMANTICS,
        "target_semantics": SUPERVISED_TARGET_SEMANTICS,
    }
    mismatched = [
        field
        for field, expected in required_semantics.items()
        if manifest.get(field) != expected
    ]
    if mismatched:
        raise TelemetrySupervisedDatasetError(
            f"supervised manifest schema/semantics mismatch: {mismatched}"
        )
    try:
        year = int(manifest["year"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TelemetrySupervisedDatasetError("supervised manifest year is invalid") from exc
    if year <= 0 or isinstance(manifest.get("year"), bool):
        raise TelemetrySupervisedDatasetError("supervised manifest year is invalid")
    _utc_timestamp(manifest.get("generated_at"), label="supervised generated_at")
    repo_root = root.expanduser().resolve()

    feature_rows, feature_by_path = _validate_input_file_manifest(
        manifest,
        files_field="feature_input_files",
        hash_field="feature_input_manifest_sha256",
        allowed_roles=frozenset(
            {"prequal_telemetry_manifest", "prequal_telemetry_tensor"}
        ),
        root=repo_root,
    )
    target_rows, target_by_path = _validate_input_file_manifest(
        manifest,
        files_field="target_input_files",
        hash_field="target_input_manifest_sha256",
        allowed_roles=frozenset(
            {
                "weekend_target_metadata",
                "qualifying_laps_target",
                "qualifying_results_stage_labels",
            }
        ),
        root=repo_root,
    )

    telemetry_manifest_rows = [
        (row, resolved)
        for row, resolved in feature_by_path.values()
        if row["role"] == "prequal_telemetry_manifest"
    ]
    target_metadata_rows = [
        (row, resolved)
        for row, resolved in target_by_path.values()
        if row["role"] == "weekend_target_metadata"
    ]
    if not telemetry_manifest_rows or not target_metadata_rows:
        raise TelemetrySupervisedDatasetError(
            "supervised manifest requires telemetry and target metadata manifests"
        )

    event_sources: dict[int, dict[str, Any]] = {}
    reconstructed_feature_evidence: list[dict[str, object]] = []
    seen_tensor_paths: set[str] = set()
    for manifest_evidence, manifest_path in telemetry_manifest_rows:
        telemetry_manifest = _load_json(manifest_path)
        if (
            str(telemetry_manifest.get("schema_version") or "")
            != TELEMETRY_SOURCE_MANIFEST_SCHEMA_VERSION
        ):
            raise TelemetrySupervisedDatasetError(
                f"unsupported telemetry source manifest schema: {manifest_path}"
            )
        try:
            event_key = int(telemetry_manifest["event_key"])
            manifest_year = int(telemetry_manifest.get("year", event_key // 100))
            round_number = int(
                telemetry_manifest.get("round", event_key % 100)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TelemetrySupervisedDatasetError(
                f"telemetry manifest has invalid event identity: {manifest_path}"
            ) from exc
        if (
            manifest_year != year
            or event_key != year * 100 + round_number
            or event_key in event_sources
        ):
            raise TelemetrySupervisedDatasetError(
                f"telemetry manifest has duplicate or inconsistent event {event_key}"
            )
        event_name = str(telemetry_manifest.get("event_name") or "").strip()
        rehearsal_source = str(
            telemetry_manifest.get("rehearsal_source") or ""
        ).strip()
        if not event_name or not rehearsal_source:
            raise TelemetrySupervisedDatasetError(
                f"telemetry manifest event metadata is incomplete: {manifest_path}"
            )
        qualifying_start = _utc_timestamp(
            telemetry_manifest.get("qualifying_start_utc"),
            label=f"telemetry event {event_key} Qualifying cutoff",
        )
        feature_records = telemetry_manifest.get("feature_records")
        if not isinstance(feature_records, list) or not feature_records:
            raise TelemetrySupervisedDatasetError(
                f"telemetry manifest {manifest_path} has no feature records"
            )
        records_by_driver: dict[str, list[Mapping[str, Any]]] = {}
        seen_record_keys: set[tuple[str, int]] = set()
        reconstructed_feature_evidence.append(dict(manifest_evidence))
        for raw_record in feature_records:
            if not isinstance(raw_record, Mapping):
                raise TelemetrySupervisedDatasetError(
                    "telemetry feature record must be an object"
                )
            try:
                record_event = int(raw_record.get("event_key"))
                lap_number = int(raw_record.get("lap_number"))
            except (TypeError, ValueError) as exc:
                raise TelemetrySupervisedDatasetError(
                    "telemetry feature record identity is invalid"
                ) from exc
            driver_id = str(raw_record.get("driver_id") or "").strip().upper()
            if record_event != event_key or not driver_id:
                raise TelemetrySupervisedDatasetError(
                    f"telemetry feature record does not belong to event {event_key}"
                )
            if (driver_id, lap_number) in seen_record_keys:
                raise TelemetrySupervisedDatasetError(
                    f"duplicate telemetry feature record {(event_key, driver_id, lap_number)}"
                )
            seen_record_keys.add((driver_id, lap_number))
            record_cutoff = _utc_timestamp(
                raw_record.get("qualifying_start_utc"),
                label=f"telemetry record {event_key}/{driver_id}/{lap_number} cutoff",
            )
            if record_cutoff != qualifying_start:
                raise TelemetrySupervisedDatasetError(
                    f"telemetry record cutoff does not match event {event_key}"
                )
            path, _ = _repository_relative_file(
                raw_record.get("telemetry_path"),
                root=repo_root,
                label=f"telemetry record {event_key}/{driver_id}/{lap_number}",
            )
            if path in seen_tensor_paths:
                raise TelemetrySupervisedDatasetError(
                    f"telemetry tensor path is reused across records: {path}"
                )
            seen_tensor_paths.add(path)
            listed = feature_by_path.get(path)
            if listed is None or listed[0]["role"] != "prequal_telemetry_tensor":
                raise TelemetrySupervisedDatasetError(
                    f"telemetry record path is absent from feature_input_files: {path}"
                )
            if str(raw_record.get("telemetry_sha256") or "").strip().lower() != str(
                listed[0]["sha256"]
            ):
                raise TelemetrySupervisedDatasetError(
                    f"telemetry record digest disagrees with feature evidence: {path}"
                )
            records_by_driver.setdefault(driver_id, []).append(raw_record)
        event_sources[event_key] = {
            "round": round_number,
            "event_name": event_name,
            "rehearsal_source": rehearsal_source,
            "qualifying_start": qualifying_start,
            "manifest_evidence": dict(manifest_evidence),
            "records_by_driver": records_by_driver,
        }

    target_sources: dict[int, dict[str, Any]] = {}
    reconstructed_target_evidence: list[dict[str, object]] = []
    for metadata_evidence, metadata_path in target_metadata_rows:
        metadata = _load_json(metadata_path)
        try:
            metadata_year = int(metadata["year"])
            round_number = int(metadata["round_number"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TelemetrySupervisedDatasetError(
                f"weekend target metadata has invalid identity: {metadata_path}"
            ) from exc
        event_key = metadata_year * 100 + round_number
        if metadata_year != year or event_key in target_sources:
            raise TelemetrySupervisedDatasetError(
                f"target metadata has duplicate or inconsistent event {event_key}"
            )
        event_name = str(metadata.get("event_name") or "").strip()
        feature_source = event_sources.get(event_key)
        if feature_source is None:
            raise TelemetrySupervisedDatasetError(
                f"target metadata has no matching telemetry event {event_key}"
            )
        if event_name.casefold() != str(feature_source["event_name"]).casefold():
            raise TelemetrySupervisedDatasetError(
                f"target and telemetry event names disagree for {event_key}"
            )
        session = _qualifying_session(metadata)
        for role, field in (("qualifying_laps_target", "laps_path"),):
            nested_path, _ = _repository_relative_file(
                session.get(field),
                root=repo_root,
                label=f"target metadata event {event_key} {field}",
            )
            listed = target_by_path.get(nested_path)
            if listed is None or listed[0]["role"] != role:
                raise TelemetrySupervisedDatasetError(
                    f"target metadata {field} is absent from target_input_files"
                )
        results_text = str(session.get("results_path") or "").strip()
        if results_text:
            results_path, _ = _repository_relative_file(
                results_text,
                root=repo_root,
                label=f"target metadata event {event_key} results_path",
            )
            listed = target_by_path.get(results_path)
            if (
                listed is None
                or listed[0]["role"] != "qualifying_results_stage_labels"
            ):
                raise TelemetrySupervisedDatasetError(
                    "target metadata results_path is absent from target_input_files"
                )
        (
            qualifying_session,
            targets,
            labels,
            event_target_evidence,
        ) = _target_sources(
            metadata_path.parent,
            root=repo_root,
            expected_year=year,
            expected_round=round_number,
            expected_event_name=str(feature_source["event_name"]),
        )
        canonical_event_evidence = _deduplicate_file_evidence(event_target_evidence)
        if dict(metadata_evidence) not in canonical_event_evidence:
            raise TelemetrySupervisedDatasetError(
                f"target metadata evidence is not canonical for event {event_key}"
            )
        reconstructed_target_evidence.extend(canonical_event_evidence)
        target_sources[event_key] = {
            "event_name": event_name,
            "session": qualifying_session,
            "targets": targets,
            "labels": labels,
            "source_sha256": canonical_sha256(canonical_event_evidence),
        }

    if set(event_sources) != set(target_sources):
        raise TelemetrySupervisedDatasetError(
            "telemetry and target event sets do not match"
        )

    raw_bags = manifest.get("bags")
    if not isinstance(raw_bags, list) or not raw_bags:
        raise TelemetrySupervisedDatasetError("supervised manifest has no bags")
    by_identity: dict[tuple[int, str], Mapping[str, Any]] = {}
    for raw_bag in raw_bags:
        if not isinstance(raw_bag, Mapping):
            raise TelemetrySupervisedDatasetError("supervised bag must be an object")
        _assert_canonical_hash(raw_bag, "bag_sha256", label="driver-event bag")
        feature = raw_bag.get("feature")
        target = raw_bag.get("target")
        if not isinstance(feature, Mapping) or not isinstance(target, Mapping):
            raise TelemetrySupervisedDatasetError(
                "supervised bag requires feature and target objects"
            )
        _assert_canonical_hash(feature, "feature_bag_sha256", label="feature bag")
        _assert_canonical_hash(target, "target_sha256", label="target")
        try:
            event_key = int(raw_bag["event_key"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TelemetrySupervisedDatasetError("supervised bag event_key is invalid") from exc
        driver_id = str(raw_bag.get("driver_id") or "").strip().upper()
        identity = (event_key, driver_id)
        if not driver_id or identity in by_identity:
            raise TelemetrySupervisedDatasetError(
                f"duplicate or invalid driver-event supervised bag {identity}"
            )
        by_identity[identity] = raw_bag

    expected_bags: list[dict[str, Any]] = []
    expected_feature_evidence = list(reconstructed_feature_evidence)
    target_only_driver_events = 0
    for event_key in sorted(event_sources):
        source = event_sources[event_key]
        target_source = target_sources[event_key]
        records_by_driver = source["records_by_driver"]
        targets = target_source["targets"]
        labels = target_source["labels"]
        target_only_driver_events += len(set(targets).difference(records_by_driver))
        session = target_source["session"]
        for driver_id in sorted(records_by_driver):
            tensors: list[dict[str, object]] = []
            for record in sorted(
                records_by_driver[driver_id],
                key=lambda item: (
                    _optional_int(item.get("push_lap_rank")) or 10**9,
                    int(item["lap_number"]),
                ),
            ):
                tensor, tensor_evidence = _feature_tensor_payload(
                    record, root=repo_root
                )
                tensors.append(tensor)
                expected_feature_evidence.append(tensor_evidence)
            feature_payload: dict[str, Any] = {
                "rehearsal_source": source["rehearsal_source"],
                "qualifying_start_utc": str(
                    _load_json(
                        feature_by_path[
                            str(source["manifest_evidence"]["path"])
                        ][1]
                    ).get("qualifying_start_utc")
                    or ""
                ),
                "telemetry_manifest_path": source["manifest_evidence"]["path"],
                "telemetry_manifest_sha256": source["manifest_evidence"]["sha256"],
                "tensor_count": len(tensors),
                "tensors": tensors,
            }
            feature_payload["feature_bag_sha256"] = canonical_sha256(feature_payload)
            legal_lap = targets.get(driver_id)
            has_legal_lap = legal_lap is not None
            lap_target = legal_lap or {
                "lap_time_seconds": None,
                "lap_number": None,
                "sector1_seconds": None,
                "sector2_seconds": None,
                "sector3_seconds": None,
            }
            target_payload: dict[str, Any] = {
                "target_contract": TARGET_CONTRACT,
                "has_legal_qualifying_lap": has_legal_lap,
                "lap_time_observed": has_legal_lap,
                "lap_time_target_status": (
                    LAP_TIME_OBSERVED_STATUS
                    if has_legal_lap
                    else LAP_TIME_CENSORED_STATUS
                ),
                **lap_target,
                **labels.get(
                    driver_id,
                    {
                        "has_q1_time": None,
                        "has_q2_time": None,
                        "has_q3_time": None,
                        "reaches_q2": None,
                        "reaches_q3": None,
                        "stage_reached": None,
                        "classification_position": None,
                        "classification_field_size": None,
                        "q2_advancement_slot_count": None,
                        "q3_advancement_slot_count": None,
                        "reaches_q2_from_position": None,
                        "reaches_q3_from_position": None,
                    },
                ),
                "target_session": "Qualifying",
                "target_available_at": str(
                    session.get("available_at") or session.get("captured_at") or ""
                ),
                "target_source_manifest_sha256": target_source["source_sha256"],
                "inference_eligible": False,
            }
            target_payload["target_sha256"] = canonical_sha256(target_payload)
            expected_bag: dict[str, Any] = {
                "event_key": event_key,
                "year": year,
                "round": source["round"],
                "event_name": source["event_name"],
                "driver_id": driver_id,
                "row_unit": "driver_event_bag",
                "feature": feature_payload,
                "target": target_payload,
            }
            expected_bag["bag_sha256"] = canonical_sha256(expected_bag)
            actual_bag = by_identity.get((event_key, driver_id))
            if actual_bag is None:
                raise TelemetrySupervisedDatasetError(
                    f"missing supervised driver-event bag {(event_key, driver_id)}"
                )
            if dict(actual_bag) != expected_bag:
                raise TelemetrySupervisedDatasetError(
                    f"supervised driver-event bag disagrees with nested evidence: "
                    f"{(event_key, driver_id)}"
                )
            expected_bags.append(expected_bag)
    if len(by_identity) != len(expected_bags):
        extras = sorted(set(by_identity).difference(
            (int(bag["event_key"]), str(bag["driver_id"])) for bag in expected_bags
        ))
        raise TelemetrySupervisedDatasetError(
            f"supervised manifest contains bags without source telemetry: {extras[:5]}"
        )
    if [dict(bag) for bag in raw_bags] != expected_bags:
        raise TelemetrySupervisedDatasetError(
            "supervised bags are not in canonical event/driver order"
        )

    canonical_feature_evidence = _deduplicate_file_evidence(expected_feature_evidence)
    canonical_target_evidence = _deduplicate_file_evidence(
        reconstructed_target_evidence
    )
    if feature_rows != canonical_feature_evidence:
        raise TelemetrySupervisedDatasetError(
            "feature_input_files do not exactly cover nested feature evidence"
        )
    if target_rows != canonical_target_evidence:
        raise TelemetrySupervisedDatasetError(
            "target_input_files do not exactly cover nested target evidence"
        )
    declared_bag_set = str(manifest.get("bag_set_sha256") or "").strip().lower()
    actual_bag_set = canonical_sha256(
        [str(bag["bag_sha256"]) for bag in expected_bags]
    )
    if declared_bag_set != actual_bag_set:
        raise TelemetrySupervisedDatasetError(
            "supervised bag-set hash mismatch"
        )

    event_counts: dict[int, int] = {}
    for bag in expected_bags:
        event_key = int(bag["event_key"])
        event_counts[event_key] = event_counts.get(event_key, 0) + 1
    observed_count = sum(
        bag["target"]["lap_time_observed"] is True for bag in expected_bags
    )
    expected_audit = {
        "event_count": len(event_counts),
        "driver_event_bag_count": len(expected_bags),
        "validated_tensor_count": sum(
            int(bag["feature"]["tensor_count"]) for bag in expected_bags
        ),
        "event_driver_counts": {
            str(event_key): count for event_key, count in sorted(event_counts.items())
        },
        "stage_label_bag_count": sum(
            bag["target"]["stage_reached"] is not None for bag in expected_bags
        ),
        "observed_lap_time_target_bag_count": observed_count,
        "censored_lap_time_target_bag_count": len(expected_bags) - observed_count,
        "complete_sector_target_bag_count": sum(
            all(
                bag["target"][field] is not None
                for field in ("sector1_seconds", "sector2_seconds", "sector3_seconds")
            )
            for bag in expected_bags
        ),
        "target_only_driver_event_count": target_only_driver_events,
        "duplicate_driver_event_count": 0,
        "target_objects_per_bag": 1,
        "target_objects_per_tensor": 0,
        "inference_eligible_target_count": 0,
    }
    if manifest.get("audit") != expected_audit:
        raise TelemetrySupervisedDatasetError(
            "supervised manifest audit counts do not match nested evidence"
        )
    return {
        "schema_version": SUPERVISED_TELEMETRY_SCHEMA_VERSION,
        "year": year,
        "target_contract": TARGET_CONTRACT,
        "feature_input_manifest_sha256": str(
            manifest["feature_input_manifest_sha256"]
        ),
        "target_input_manifest_sha256": str(
            manifest["target_input_manifest_sha256"]
        ),
        "bag_set_sha256": actual_bag_set,
        "event_keys": sorted(event_counts),
        "audit": expected_audit,
    }


__all__ = [
    "SUPERVISED_TELEMETRY_SCHEMA_VERSION",
    "TELEMETRY_SOURCE_MANIFEST_SCHEMA_VERSION",
    "TARGET_CONTRACT",
    "LAP_TIME_OBSERVED_STATUS",
    "LAP_TIME_CENSORED_STATUS",
    "FEATURE_CAPTURE_SEMANTICS",
    "TARGET_CAPTURE_SEMANTICS",
    "SUPERVISED_TARGET_SEMANTICS",
    "QUALIFYING_ADVANCEMENT_SLOTS",
    "TelemetrySupervisedDatasetError",
    "build_prequal_telemetry_supervised_manifest",
    "validate_prequal_telemetry_supervised_manifest",
    "canonical_sha256",
]
