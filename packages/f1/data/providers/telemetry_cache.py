"""Causal pre-Qualifying telemetry selection and cache-readiness auditing.

The deep Ultimate-Lap model is deliberately blocked until this cache contains
enough independent events.  Feature telemetry is captured from a completed
FP/Sprint-Qualifying rehearsal before Grand Prix Qualifying starts; target
Qualifying outcomes live in a separate training-only artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REQUIRED_TELEMETRY_CHANNELS: tuple[str, ...] = (
    "Distance",
    "Speed",
    "RPM",
    "nGear",
    "Throttle",
    "Brake",
    "DRS",
)
NORMALIZED_TELEMETRY_CHANNELS: tuple[str, ...] = REQUIRED_TELEMETRY_CHANNELS[1:]
TELEMETRY_TENSOR_SCHEMA_VERSION = "f1_prequal_distance_tensor_v1"
DEFAULT_DISTANCE_BINS = 200
DEFAULT_MINIMUM_DISTANCE_COVERAGE = 0.95


class TelemetryHashMismatchError(ValueError):
    """Raised when a cached tensor no longer matches its manifest digest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(array: np.ndarray, *, name: str) -> Any:
    value = np.asarray(array)
    if value.ndim != 0:
        raise ValueError(f"cached telemetry {name} must be a scalar")
    return value.item()


def validate_cached_telemetry_tensor(
    record: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    """Verify one normalized NPZ against its point-in-time manifest record.

    The cache is model evidence rather than an opaque file-existence flag.  A
    tensor therefore has to prove its digest, schema, channels, dimensions,
    distance grid, coverage, and embedded sample timestamps before it can
    contribute to the independent-event promotion gate.
    """

    path_text = str(record.get("telemetry_path") or "").strip()
    if not path_text:
        raise FileNotFoundError("telemetry_path is missing")
    path = Path(path_text)
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file():
        raise FileNotFoundError(f"telemetry tensor does not exist: {resolved}")

    declared_digest = str(record.get("telemetry_sha256") or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", declared_digest) is None:
        raise ValueError("telemetry_sha256 must be a lowercase SHA-256 digest")
    actual_digest = sha256_file(resolved)
    if actual_digest != declared_digest:
        raise TelemetryHashMismatchError(
            f"telemetry SHA-256 mismatch for {resolved}: "
            f"declared={declared_digest}, actual={actual_digest}"
        )

    required_arrays = {
        "schema_version",
        "values",
        "channel_names",
        "distance_grid_m",
        "sample_timestamp_ns",
        "feature_as_of_ns",
        "expected_lap_distance_m",
        "distance_coverage",
    }
    try:
        with np.load(resolved, allow_pickle=False) as payload:
            missing = sorted(required_arrays.difference(payload.files))
            if missing:
                raise ValueError(f"cached telemetry arrays are missing: {missing}")
            schema_version = str(_scalar(payload["schema_version"], name="schema_version"))
            values = np.asarray(payload["values"])
            channel_array = np.asarray(payload["channel_names"])
            distance_grid = np.asarray(payload["distance_grid_m"])
            sample_timestamps = np.asarray(payload["sample_timestamp_ns"])
            feature_as_of_ns = int(
                _scalar(payload["feature_as_of_ns"], name="feature_as_of_ns")
            )
            expected_lap_distance = float(
                _scalar(
                    payload["expected_lap_distance_m"],
                    name="expected_lap_distance_m",
                )
            )
            distance_coverage = float(
                _scalar(payload["distance_coverage"], name="distance_coverage")
            )
    except TelemetryHashMismatchError:
        raise
    except (OSError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"cached telemetry NPZ is invalid: {exc}") from exc

    if schema_version != TELEMETRY_TENSOR_SCHEMA_VERSION:
        raise ValueError(f"unsupported cached telemetry schema: {schema_version!r}")
    if str(record.get("tensor_schema_version") or "") != schema_version:
        raise ValueError("tensor schema does not match the manifest record")
    if record.get("distance_normalized") is not True:
        raise ValueError("manifest does not declare a distance-normalized tensor")

    if channel_array.ndim != 1:
        raise ValueError("cached telemetry channel_names must be one-dimensional")
    channels = tuple(str(value) for value in channel_array.tolist())
    if channels != NORMALIZED_TELEMETRY_CHANNELS:
        raise ValueError(
            f"cached telemetry channels must equal {NORMALIZED_TELEMETRY_CHANNELS}, "
            f"received {channels}"
        )
    declared_channels = tuple(str(value) for value in record.get("channels", ()))
    if declared_channels != channels:
        raise ValueError("cached telemetry channels do not match the manifest record")

    try:
        declared_shape = tuple(int(value) for value in record.get("telemetry_shape", ()))
        declared_bins = int(record.get("distance_bins"))
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest telemetry_shape and distance_bins must be integers") from exc
    if len(declared_shape) != 2 or declared_shape != values.shape:
        raise ValueError(
            f"cached telemetry shape {values.shape} does not match declared {declared_shape}"
        )
    if declared_shape != (len(channels), declared_bins) or declared_bins <= 1:
        raise ValueError("cached telemetry shape is not channels x distance_bins")
    if values.dtype.kind not in "fiu" or not np.isfinite(values.astype(float)).all():
        raise ValueError("cached telemetry values must be finite numeric data")

    if not np.isfinite(expected_lap_distance) or expected_lap_distance <= 0.0:
        raise ValueError("cached telemetry expected lap distance must be positive and finite")
    try:
        declared_expected_distance = float(record.get("expected_lap_distance_m"))
        declared_coverage = float(record.get("distance_coverage"))
        minimum_coverage = float(
            record.get("minimum_distance_coverage", DEFAULT_MINIMUM_DISTANCE_COVERAGE)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest distance metadata must be numeric") from exc
    if not np.isclose(declared_expected_distance, expected_lap_distance, rtol=0.0, atol=1e-6):
        raise ValueError("cached expected lap distance does not match the manifest record")
    if not np.isclose(declared_coverage, distance_coverage, rtol=0.0, atol=1e-6):
        raise ValueError("cached distance coverage does not match the manifest record")
    if not 0.5 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_distance_coverage must be between 0.5 and 1.0")
    if not np.isfinite(distance_coverage) or not minimum_coverage <= distance_coverage <= 1.05:
        raise ValueError("cached telemetry does not meet declared lap-distance coverage")

    if distance_grid.shape != (declared_bins,):
        raise ValueError("cached telemetry distance grid has the wrong shape")
    if distance_grid.dtype.kind not in "fiu" or not np.isfinite(distance_grid.astype(float)).all():
        raise ValueError("cached telemetry distance grid must be finite numeric data")
    expected_grid = np.linspace(0.0, expected_lap_distance, num=declared_bins)
    if not np.allclose(distance_grid.astype(float), expected_grid, rtol=1e-6, atol=1e-3):
        raise ValueError("cached telemetry distance grid is not normalized to the declared lap")
    if not np.all(np.diff(distance_grid.astype(float)) > 0.0):
        raise ValueError("cached telemetry distance grid must be strictly increasing")

    if sample_timestamps.shape != (declared_bins,) or sample_timestamps.dtype.kind not in "iu":
        raise ValueError(
            "cached telemetry sample timestamps must be integer ns at every distance bin"
        )
    sample_timestamps = sample_timestamps.astype(np.int64, copy=False)
    if not np.all(np.diff(sample_timestamps) >= 0):
        raise ValueError("cached telemetry sample timestamps must be nondecreasing")
    if int(sample_timestamps.max()) != feature_as_of_ns:
        raise ValueError("cached telemetry feature_as_of is not its latest sample timestamp")

    manifest_as_of = pd.to_datetime(record.get("feature_as_of"), utc=True, errors="coerce")
    cutoff = pd.to_datetime(record.get("qualifying_start_utc"), utc=True, errors="coerce")
    if pd.isna(manifest_as_of) or pd.isna(cutoff):
        raise ValueError("manifest requires valid feature_as_of and Qualifying cutoff timestamps")
    if int(manifest_as_of.value) != feature_as_of_ns:
        raise ValueError("embedded feature_as_of does not match the manifest record")
    if feature_as_of_ns >= int(cutoff.value):
        raise ValueError("cached telemetry crosses the Qualifying start cutoff")

    return {
        "path": str(resolved),
        "sha256": actual_digest,
        "shape": list(values.shape),
        "channels": list(channels),
        "distance_bins": declared_bins,
        "expected_lap_distance_m": expected_lap_distance,
        "distance_coverage": distance_coverage,
        "feature_as_of": manifest_as_of.isoformat().replace("+00:00", "Z"),
    }


def _seconds(values: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(values):
        return values.dt.total_seconds()
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return numeric
    return pd.to_timedelta(values, errors="coerce").dt.total_seconds()


def _bool(values: pd.Series, *, default: bool) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default).astype(bool)
    normalized = values.astype(str).str.strip().str.lower().map(
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
    return normalized.fillna(default).astype(bool)


def select_representative_push_laps(
    laps: pd.DataFrame,
    *,
    maximum_laps_per_driver: int = 3,
) -> pd.DataFrame:
    """Select each entrant's fastest causal clean push-lap evidence.

    Selection never uses a Qualifying target.  Deleted, inaccurate, pit, and
    explicitly non-green laps are retained in the raw weekend cache but are not
    eligible for the promotion-grade telemetry tensor.
    """

    if not isinstance(laps, pd.DataFrame):
        raise TypeError("laps must be a pandas DataFrame")
    if int(maximum_laps_per_driver) < 1:
        raise ValueError("maximum_laps_per_driver must be positive")
    if laps.empty:
        return pd.DataFrame()
    driver_column = "Driver" if "Driver" in laps.columns else "driver_id"
    lap_column = "LapNumber" if "LapNumber" in laps.columns else "lap_number"
    time_column = "LapTime" if "LapTime" in laps.columns else "lap_time_seconds"
    required = {driver_column, lap_column, time_column}
    missing = sorted(required.difference(laps.columns))
    if missing:
        raise ValueError(f"rehearsal laps are missing columns: {missing}")

    frame = laps.copy()
    frame["driver_id"] = frame[driver_column].fillna("").astype(str).str.strip()
    frame["lap_number"] = pd.to_numeric(frame[lap_column], errors="coerce")
    frame["lap_time_seconds"] = _seconds(frame[time_column])
    eligible = (
        frame["driver_id"].ne("")
        & frame["lap_number"].notna()
        & frame["lap_time_seconds"].map(math.isfinite)
        & frame["lap_time_seconds"].gt(0.0)
    )
    if "Deleted" in frame.columns:
        eligible &= ~_bool(frame["Deleted"], default=False)
    if "IsAccurate" in frame.columns:
        eligible &= _bool(frame["IsAccurate"], default=False)
    for pit_column in ("PitOutTime", "PitInTime"):
        if pit_column in frame.columns:
            eligible &= frame[pit_column].isna()
    if "TrackStatus" in frame.columns:
        status = frame["TrackStatus"].fillna("").astype(str).str.strip()
        eligible &= status.isin(("", "1"))

    selected = frame.loc[eligible].sort_values(
        ["driver_id", "lap_time_seconds", "lap_number"], kind="mergesort"
    )
    selected = selected.groupby("driver_id", sort=False, as_index=False).head(
        int(maximum_laps_per_driver)
    ).copy()
    selected["push_lap_rank"] = (
        selected.groupby("driver_id", sort=False).cumcount() + 1
    )
    return selected


def validate_telemetry_frame(
    telemetry: pd.DataFrame,
    *,
    qualifying_start_utc: str | datetime,
) -> dict[str, Any]:
    """Validate channels, physical distance, finiteness, and point-in-time cutoff."""

    if not isinstance(telemetry, pd.DataFrame) or telemetry.empty:
        raise ValueError("telemetry must be a non-empty DataFrame")
    missing = sorted(set(REQUIRED_TELEMETRY_CHANNELS).difference(telemetry.columns))
    if missing:
        raise ValueError(f"telemetry is missing required channels: {missing}")
    numeric = telemetry.loc[:, list(REQUIRED_TELEMETRY_CHANNELS)].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("telemetry channels must all be finite")
    distance = numeric["Distance"].to_numpy(dtype=float)
    if len(distance) < 2 or float(np.nanmax(distance) - np.nanmin(distance)) <= 0.0:
        raise ValueError("telemetry must span a positive physical distance")

    cutoff = pd.Timestamp(qualifying_start_utc)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    timestamp_column = next(
        (name for name in ("Date", "date", "timestamp") if name in telemetry.columns), None
    )
    if timestamp_column is None:
        raise ValueError("telemetry requires an absolute timestamp column")
    timestamps = pd.to_datetime(telemetry[timestamp_column], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("every telemetry row requires a valid absolute timestamp")
    maximum = timestamps.max()
    if maximum >= cutoff:
        raise ValueError("pre-Qualifying telemetry crosses the Qualifying start cutoff")
    return {
        "rows": int(len(telemetry)),
        "distance_min_m": float(np.min(distance)),
        "distance_max_m": float(np.max(distance)),
        "feature_as_of": maximum.isoformat().replace("+00:00", "Z"),
        "qualifying_start_utc": cutoff.isoformat().replace("+00:00", "Z"),
        "channels": list(REQUIRED_TELEMETRY_CHANNELS),
    }


@dataclass(frozen=True)
class TelemetryCacheAudit:
    record_count: int
    validated_tensor_count: int
    event_count: int
    driver_event_count: int
    complete_event_keys: tuple[str, ...]
    validated_cache_sha256: str
    qualifying_cutoff_violations: int
    missing_file_count: int
    hash_mismatch_count: int
    invalid_tensor_count: int
    minimum_independent_events: int
    minimum_drivers_per_event: int
    cache_integrity_ready: bool
    ready_for_requested_event_protocol: bool
    ready_for_deep_model: bool
    blockers: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "record_count": int(self.record_count),
            "validated_tensor_count": int(self.validated_tensor_count),
            "event_count": int(self.event_count),
            "driver_event_count": int(self.driver_event_count),
            "complete_event_keys": list(self.complete_event_keys),
            "validated_cache_sha256": self.validated_cache_sha256,
            "qualifying_cutoff_violations": int(self.qualifying_cutoff_violations),
            "missing_file_count": int(self.missing_file_count),
            "hash_mismatch_count": int(self.hash_mismatch_count),
            "invalid_tensor_count": int(self.invalid_tensor_count),
            "minimum_independent_events": int(self.minimum_independent_events),
            "minimum_drivers_per_event": int(self.minimum_drivers_per_event),
            "cache_integrity_ready": bool(self.cache_integrity_ready),
            "ready_for_requested_event_protocol": bool(
                self.ready_for_requested_event_protocol
            ),
            "complete_event_shortfall_for_requested_protocol": int(
                max(0, self.minimum_independent_events - self.event_count)
            ),
            "event_threshold_semantics": (
                "caller_supplied_protocol_requirement_not_model_capacity_claim"
            ),
            # Backward-compatible alias.  Capacity and promotion readiness must
            # be decided from event-disjoint model evidence, not this cache flag.
            "ready_for_deep_model": bool(self.ready_for_deep_model),
            "blockers": list(self.blockers),
        }


def audit_telemetry_cache_manifests(
    manifests: Iterable[Mapping[str, Any]],
    *,
    root: Path,
    minimum_independent_events: int = 1,
    minimum_drivers_per_event: int = 18,
) -> TelemetryCacheAudit:
    """Validate cache integrity and a caller-defined evaluation protocol.

    ``minimum_independent_events`` is not a neural-network sample-complexity
    claim.  It describes the number of complete events required by the caller's
    concrete split (for example, three fit events plus one held-out event).
    Model capacity must be judged from event-disjoint learning evidence.
    """

    if int(minimum_independent_events) < 1:
        raise ValueError("minimum_independent_events must be positive")
    if int(minimum_drivers_per_event) < 1:
        raise ValueError("minimum_drivers_per_event must be positive")

    rows = list(manifests)
    event_drivers: dict[str, set[str]] = {}
    cutoff_violations = 0
    missing_files = 0
    hash_mismatches = 0
    invalid_tensors = 0
    validated_records: list[dict[str, Any]] = []
    seen_record_keys: set[tuple[str, str, int]] = set()
    seen_paths: set[str] = set()
    for manifest in rows:
        event_key = str(manifest.get("event_key") or "").strip()
        driver_id = str(manifest.get("driver_id") or "").strip()
        feature_as_of = pd.to_datetime(manifest.get("feature_as_of"), utc=True, errors="coerce")
        cutoff = pd.to_datetime(
            manifest.get("qualifying_start_utc"), utc=True, errors="coerce"
        )
        if not event_key or not driver_id:
            invalid_tensors += 1
            continue
        try:
            lap_number = int(manifest.get("lap_number"))
        except (TypeError, ValueError):
            invalid_tensors += 1
            continue
        path_key = str(manifest.get("telemetry_path") or "").strip()
        record_key = (event_key, driver_id, lap_number)
        if lap_number < 1 or record_key in seen_record_keys or path_key in seen_paths:
            invalid_tensors += 1
            continue
        seen_record_keys.add(record_key)
        seen_paths.add(path_key)
        if pd.isna(feature_as_of) or pd.isna(cutoff) or feature_as_of >= cutoff:
            cutoff_violations += 1
        try:
            validation = validate_cached_telemetry_tensor(manifest, root=root)
        except FileNotFoundError:
            missing_files += 1
            continue
        except TelemetryHashMismatchError:
            hash_mismatches += 1
            continue
        except (OSError, TypeError, ValueError, OverflowError):
            invalid_tensors += 1
            continue
        event_drivers.setdefault(event_key, set()).add(driver_id)
        validated_records.append(
            {
                "event_key": event_key,
                "driver_id": driver_id,
                "lap_number": lap_number,
                "feature_as_of": validation["feature_as_of"],
                "qualifying_start_utc": str(manifest.get("qualifying_start_utc")),
                "telemetry_path": str(manifest.get("telemetry_path")),
                "telemetry_sha256": validation["sha256"],
                "shape": validation["shape"],
                "channels": validation["channels"],
                "expected_lap_distance_m": validation["expected_lap_distance_m"],
                "distance_coverage": validation["distance_coverage"],
            }
        )

    complete_event_keys = tuple(
        sorted(
            event_key
            for event_key, drivers in event_drivers.items()
            if len(drivers) >= int(minimum_drivers_per_event)
        )
    )
    complete_events = len(complete_event_keys)
    canonical_records = sorted(
        validated_records,
        key=lambda record: (
            record["event_key"],
            record["driver_id"],
            record["lap_number"],
            record["telemetry_path"],
        ),
    )
    validated_cache_sha256 = hashlib.sha256(
        json.dumps(
            canonical_records,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    integrity_blockers: list[str] = []
    protocol_blockers: list[str] = []
    if complete_events < int(minimum_independent_events):
        protocol_blockers.append("insufficient_complete_events_for_requested_protocol")
    if cutoff_violations:
        integrity_blockers.append("telemetry_cutoff_violations")
    if missing_files:
        integrity_blockers.append("telemetry_files_missing")
    if hash_mismatches:
        integrity_blockers.append("telemetry_hash_mismatches")
    if invalid_tensors:
        integrity_blockers.append("telemetry_tensor_content_or_shape_invalid")
    blockers = [*protocol_blockers, *integrity_blockers]
    cache_integrity_ready = not integrity_blockers
    protocol_ready = not blockers
    return TelemetryCacheAudit(
        record_count=int(len(rows)),
        validated_tensor_count=int(len(validated_records)),
        event_count=int(complete_events),
        driver_event_count=int(sum(len(value) for value in event_drivers.values())),
        complete_event_keys=complete_event_keys,
        validated_cache_sha256=validated_cache_sha256,
        qualifying_cutoff_violations=int(cutoff_violations),
        missing_file_count=int(missing_files),
        hash_mismatch_count=int(hash_mismatches),
        invalid_tensor_count=int(invalid_tensors),
        minimum_independent_events=int(minimum_independent_events),
        minimum_drivers_per_event=int(minimum_drivers_per_event),
        cache_integrity_ready=cache_integrity_ready,
        ready_for_requested_event_protocol=protocol_ready,
        ready_for_deep_model=protocol_ready,
        blockers=tuple(blockers),
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "DEFAULT_DISTANCE_BINS",
    "DEFAULT_MINIMUM_DISTANCE_COVERAGE",
    "NORMALIZED_TELEMETRY_CHANNELS",
    "REQUIRED_TELEMETRY_CHANNELS",
    "TELEMETRY_TENSOR_SCHEMA_VERSION",
    "TelemetryCacheAudit",
    "TelemetryHashMismatchError",
    "audit_telemetry_cache_manifests",
    "select_representative_push_laps",
    "sha256_file",
    "utc_now",
    "validate_cached_telemetry_tensor",
    "validate_telemetry_frame",
]


# Suggested commit name: feat(f1-telemetry): add causal pre-quali cache contracts
