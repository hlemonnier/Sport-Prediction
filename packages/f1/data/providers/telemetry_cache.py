"""Causal pre-Qualifying telemetry selection and cache-readiness auditing.

The deep Ultimate-Lap model is deliberately blocked until this cache contains
enough independent events.  Feature telemetry is captured from a completed
FP/Sprint-Qualifying rehearsal before Grand Prix Qualifying starts; target
Qualifying outcomes live in a separate training-only artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
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
    event_count: int
    driver_event_count: int
    qualifying_cutoff_violations: int
    missing_file_count: int
    minimum_independent_events: int
    ready_for_deep_model: bool
    blockers: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "event_count": int(self.event_count),
            "driver_event_count": int(self.driver_event_count),
            "qualifying_cutoff_violations": int(self.qualifying_cutoff_violations),
            "missing_file_count": int(self.missing_file_count),
            "minimum_independent_events": int(self.minimum_independent_events),
            "ready_for_deep_model": bool(self.ready_for_deep_model),
            "blockers": list(self.blockers),
        }


def audit_telemetry_cache_manifests(
    manifests: Iterable[Mapping[str, Any]],
    *,
    root: Path,
    minimum_independent_events: int = 20,
    minimum_drivers_per_event: int = 18,
) -> TelemetryCacheAudit:
    """Fail closed until enough timestamped entrant/event evidence exists."""

    rows = list(manifests)
    event_drivers: dict[str, set[str]] = {}
    cutoff_violations = 0
    missing_files = 0
    for manifest in rows:
        event_key = str(manifest.get("event_key") or "")
        driver_id = str(manifest.get("driver_id") or "")
        feature_as_of = pd.to_datetime(manifest.get("feature_as_of"), utc=True, errors="coerce")
        cutoff = pd.to_datetime(
            manifest.get("qualifying_start_utc"), utc=True, errors="coerce"
        )
        if not event_key or not driver_id:
            continue
        event_drivers.setdefault(event_key, set()).add(driver_id)
        if pd.isna(feature_as_of) or pd.isna(cutoff) or feature_as_of >= cutoff:
            cutoff_violations += 1
        path = Path(str(manifest.get("telemetry_path") or ""))
        resolved = path if path.is_absolute() else root / path
        if not path.as_posix() or not resolved.is_file():
            missing_files += 1

    complete_events = sum(
        1 for drivers in event_drivers.values() if len(drivers) >= int(minimum_drivers_per_event)
    )
    blockers: list[str] = []
    if complete_events < int(minimum_independent_events):
        blockers.append("insufficient_independent_prequalifying_telemetry_events")
    if cutoff_violations:
        blockers.append("telemetry_cutoff_violations")
    if missing_files:
        blockers.append("telemetry_files_missing")
    return TelemetryCacheAudit(
        event_count=int(complete_events),
        driver_event_count=int(sum(len(value) for value in event_drivers.values())),
        qualifying_cutoff_violations=int(cutoff_violations),
        missing_file_count=int(missing_files),
        minimum_independent_events=int(minimum_independent_events),
        ready_for_deep_model=not blockers,
        blockers=tuple(blockers),
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "REQUIRED_TELEMETRY_CHANNELS",
    "TelemetryCacheAudit",
    "audit_telemetry_cache_manifests",
    "select_representative_push_laps",
    "utc_now",
    "validate_telemetry_frame",
]


# Suggested commit name: feat(f1-telemetry): add causal pre-quali cache contracts
