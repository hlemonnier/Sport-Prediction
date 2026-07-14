#!/usr/bin/env python3
"""Capture causal pre-Qualifying telemetry and audit TCN data readiness."""

from __future__ import annotations
import repo_bootstrap  # noqa: F401

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from packages.f1.data.providers.telemetry_cache import (
    DEFAULT_DISTANCE_BINS,
    DEFAULT_MINIMUM_DISTANCE_COVERAGE,
    NORMALIZED_TELEMETRY_CHANNELS,
    TELEMETRY_TENSOR_SCHEMA_VERSION,
    audit_telemetry_cache_manifests,
    select_representative_push_laps,
    sha256_file,
    utc_now,
    validate_telemetry_frame,
)
from packages.f1.models.ultimate_lap_time.datasets import (
    build_distance_normalized_telemetry,
)
from packages.sports_core.paths import find_repo_root


SCHEMA_VERSION = "f1_prequal_telemetry_cache_v2"
REHEARSAL_PRIORITY = (
    "Sprint Qualifying",
    "Sprint Shootout",
    "Practice 3",
    "Practice 2",
    "Practice 1",
)


def _fastf1() -> Any:
    try:
        import fastf1
    except Exception as exc:  # pragma: no cover - optional provider
        raise RuntimeError("FastF1 is required to capture telemetry") from exc
    return fastf1


def _timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        return parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def _event_sessions(event: pd.Series) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    for order in range(1, 6):
        name = event.get(f"Session{order}")
        start = event.get(f"Session{order}DateUtc")
        if pd.isna(name) or pd.isna(start):
            continue
        sessions.append({"order": order, "name": str(name), "start": _timestamp(start)})
    return sessions


def _rehearsal_contract(event: pd.Series) -> tuple[str, pd.Timestamp]:
    sessions = _event_sessions(event)
    qualifying = next((item for item in sessions if item["name"] == "Qualifying"), None)
    if qualifying is None:
        raise ValueError("event schedule has no Grand Prix Qualifying session")
    eligible = {item["name"]: item for item in sessions if item["start"] < qualifying["start"]}
    for name in REHEARSAL_PRIORITY:
        if name in eligible:
            return name, qualifying["start"]
    raise ValueError("event has no completed rehearsal scheduled before Qualifying")


def _portable(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "event"


def _absolute_telemetry(lap: Any) -> pd.DataFrame:
    telemetry = lap.get_telemetry().copy()
    if "Date" not in telemetry.columns:
        lap_date = pd.to_datetime(lap.get("Date"), utc=True, errors="coerce")
        if pd.isna(lap_date):
            raise ValueError("lap telemetry has no absolute timestamp anchor")
        if "Time" not in telemetry.columns:
            raise ValueError("lap telemetry has neither Date nor relative Time")
        relative = pd.to_timedelta(telemetry["Time"], errors="coerce")
        telemetry["Date"] = lap_date - relative.max() + relative
    return telemetry


def _write_lap_tensor(
    telemetry: pd.DataFrame,
    *,
    path: Path,
    expected_lap_distance_m: float,
    distance_bins: int,
    minimum_distance_coverage: float,
) -> dict[str, Any]:
    """Persist a canonical channels x distance-bins tensor with time evidence."""

    tensor = build_distance_normalized_telemetry(
        telemetry,
        distance_bins=int(distance_bins),
        channel_names=NORMALIZED_TELEMETRY_CHANNELS,
        distance_column="Distance",
        expected_lap_distance=float(expected_lap_distance_m),
        minimum_distance_coverage=float(minimum_distance_coverage),
    )
    distance = pd.to_numeric(telemetry["Distance"], errors="coerce").to_numpy(dtype=float)
    timestamps = pd.to_datetime(telemetry["Date"], utc=True, errors="coerce")
    timestamp_ns = timestamps.astype("int64").to_numpy(dtype=np.int64)
    finite = np.isfinite(distance) & timestamps.notna().to_numpy()
    distance = distance[finite]
    timestamp_ns = timestamp_ns[finite]
    order = np.argsort(distance, kind="mergesort")
    distance = distance[order]
    timestamp_ns = timestamp_ns[order]
    unique_distance, unique_indices = np.unique(distance, return_index=True)
    timestamp_ns = timestamp_ns[unique_indices]
    target_distance = np.linspace(
        0.0, float(expected_lap_distance_m), num=int(distance_bins), dtype=float
    )
    timestamp_origin_ns = int(timestamp_ns.min())
    normalized_timestamp_ns = timestamp_origin_ns + np.rint(
        np.interp(
            target_distance,
            unique_distance,
            (timestamp_ns - timestamp_origin_ns).astype(float),
        )
    ).astype(np.int64)
    normalized_timestamp_ns = np.minimum(normalized_timestamp_ns, int(timestamp_ns.max()))
    normalized_timestamp_ns = np.maximum.accumulate(normalized_timestamp_ns)
    normalized_timestamp_ns[-1] = int(timestamp_ns.max())

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(TELEMETRY_TENSOR_SCHEMA_VERSION),
        values=tensor.values.astype(np.float32, copy=False),
        channel_names=np.asarray(tensor.channel_names),
        distance_grid_m=target_distance.astype(np.float32),
        sample_timestamp_ns=normalized_timestamp_ns,
        feature_as_of_ns=np.asarray(int(timestamp_ns.max()), dtype=np.int64),
        expected_lap_distance_m=np.asarray(
            float(tensor.expected_lap_distance), dtype=np.float64
        ),
        distance_coverage=np.asarray(float(tensor.distance_coverage), dtype=np.float64),
        raw_distance_start_m=np.asarray(float(tensor.raw_distance_start), dtype=np.float64),
        raw_distance_end_m=np.asarray(float(tensor.raw_distance_end), dtype=np.float64),
    )
    return {
        "tensor_schema_version": TELEMETRY_TENSOR_SCHEMA_VERSION,
        "distance_normalized": True,
        "telemetry_shape": list(tensor.shape),
        "distance_bins": int(tensor.distance_bins),
        "channels": list(tensor.channel_names),
        "expected_lap_distance_m": float(tensor.expected_lap_distance),
        "distance_coverage": float(tensor.distance_coverage),
        "minimum_distance_coverage": float(minimum_distance_coverage),
    }


def _training_targets(session: Any, *, event_key: int) -> pd.DataFrame:
    laps = session.laps.copy()
    if laps.empty:
        return pd.DataFrame()
    valid = laps.copy()
    if "Deleted" in valid.columns:
        deleted = valid["Deleted"].fillna(False).astype(bool)
        valid = valid.loc[~deleted]
    valid["lap_time_seconds"] = pd.to_timedelta(valid["LapTime"], errors="coerce").dt.total_seconds()
    valid = valid.loc[np.isfinite(valid["lap_time_seconds"])]
    best = (
        valid.sort_values(["Driver", "lap_time_seconds"], kind="mergesort")
        .groupby("Driver", sort=False, as_index=False)
        .first()
    )
    best = best.rename(columns={"Driver": "driver_id"})
    best["event_key"] = int(event_key)
    result = best[["event_key", "driver_id", "lap_time_seconds"]].copy()
    classifications = getattr(session, "results", pd.DataFrame())
    if isinstance(classifications, pd.DataFrame) and not classifications.empty:
        info = classifications.reset_index(drop=True).copy()
        driver_column = "Abbreviation" if "Abbreviation" in info.columns else "DriverId"
        if driver_column in info.columns:
            stage = pd.DataFrame({"driver_id": info[driver_column].astype(str)})
            for name in ("Q1", "Q2", "Q3"):
                stage[f"has_{name.lower()}_time"] = info.get(name, pd.Series(index=info.index)).notna()
            result = result.merge(stage, on="driver_id", how="left", validate="one_to_one")
    result["target_available_after_qualifying"] = True
    return result


def capture_round(
    *,
    year: int,
    round_number: int,
    output_root: Path,
    cache_dir: Path,
    maximum_laps_per_driver: int,
    distance_bins: int,
    minimum_distance_coverage: float,
    include_training_targets: bool,
) -> Path:
    fastf1 = _fastf1()
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    event = fastf1.get_event(int(year), int(round_number))
    rehearsal_name, qualifying_start = _rehearsal_contract(event)
    event_key = int(year) * 100 + int(round_number)
    event_name = str(event.get("EventName") or f"Round {round_number}")
    event_dir = output_root / str(year) / f"round_{round_number:02d}_{_slug(event_name)}"
    rehearsal = fastf1.get_session(int(year), int(round_number), rehearsal_name)
    rehearsal.load(laps=True, telemetry=True, weather=False, messages=True)
    selected = select_representative_push_laps(
        rehearsal.laps, maximum_laps_per_driver=int(maximum_laps_per_driver)
    )
    if selected.empty:
        raise ValueError("no promotion-grade pre-Qualifying push laps were found")

    pending: list[tuple[pd.Series, pd.DataFrame, dict[str, Any]]] = []
    rejected_records: list[dict[str, Any]] = []
    for row_index, row in selected.iterrows():
        driver_id = str(row["driver_id"])
        lap_number = int(row["lap_number"])
        try:
            lap = rehearsal.laps.loc[row_index]
            telemetry = _absolute_telemetry(lap)
            validation = validate_telemetry_frame(
                telemetry, qualifying_start_utc=qualifying_start.isoformat()
            )
        except ValueError as exc:
            rejected_records.append(
                {
                    "driver_id": driver_id,
                    "lap_number": lap_number,
                    "reason": str(exc),
                }
            )
            continue
        pending.append((row, telemetry, validation))

    if not pending:
        raise ValueError("no selected push lap has valid timestamped physical telemetry")

    # The session median is robust to an occasional FastF1 distance trace that
    # runs into the following lap.  Individual tensors still have to clear the
    # explicit coverage and 5% overrun checks in the shared dataset builder.
    expected_lap_distance_m = float(
        np.median([validation["distance_max_m"] for _, _, validation in pending])
    )
    records: list[dict[str, Any]] = []
    for row, telemetry, validation in pending:
        driver_id = str(row["driver_id"])
        lap_number = int(row["lap_number"])
        tensor_path = event_dir / "features" / f"{driver_id}_lap_{lap_number:03d}.npz"
        try:
            tensor_metadata = _write_lap_tensor(
                telemetry,
                path=tensor_path,
                expected_lap_distance_m=expected_lap_distance_m,
                distance_bins=int(distance_bins),
                minimum_distance_coverage=float(minimum_distance_coverage),
            )
        except ValueError as exc:
            rejected_records.append(
                {
                    "driver_id": driver_id,
                    "lap_number": lap_number,
                    "reason": str(exc),
                }
            )
            continue
        records.append(
            {
                "event_key": event_key,
                "driver_id": driver_id,
                "lap_number": lap_number,
                "push_lap_rank": int(row["push_lap_rank"]),
                "lap_time_seconds": float(row["lap_time_seconds"]),
                "rehearsal_source": rehearsal_name,
                "feature_as_of": validation["feature_as_of"],
                "qualifying_start_utc": validation["qualifying_start_utc"],
                "telemetry_path": _portable(tensor_path, find_repo_root()),
                "telemetry_sha256": _sha256(tensor_path),
                "raw_telemetry_rows": validation["rows"],
                "raw_distance_min_m": validation["distance_min_m"],
                "raw_distance_max_m": validation["distance_max_m"],
                **tensor_metadata,
            }
        )

    target_evidence: dict[str, Any] | None = None
    if include_training_targets:
        qualifying = fastf1.get_session(int(year), int(round_number), "Qualifying")
        qualifying.load(laps=True, telemetry=False, weather=False, messages=False)
        targets = _training_targets(qualifying, event_key=event_key)
        target_path = event_dir / "training_targets_after_qualifying.csv"
        event_dir.mkdir(parents=True, exist_ok=True)
        targets.to_csv(target_path, index=False)
        target_evidence = {
            "path": _portable(target_path, find_repo_root()),
            "sha256": _sha256(target_path),
            "rows": int(len(targets)),
            "inference_eligible": False,
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "event_key": event_key,
        "year": int(year),
        "round": int(round_number),
        "event_name": event_name,
        "rehearsal_source": rehearsal_name,
        "qualifying_start_utc": qualifying_start.isoformat().replace("+00:00", "Z"),
        "captured_at": utc_now(),
        "tensor_contract": {
            "tensor_schema_version": TELEMETRY_TENSOR_SCHEMA_VERSION,
            "distance_normalized": True,
            "channels": list(NORMALIZED_TELEMETRY_CHANNELS),
            "distance_bins": int(distance_bins),
            "expected_lap_distance_m": expected_lap_distance_m,
            "minimum_distance_coverage": float(minimum_distance_coverage),
        },
        "feature_records": records,
        "rejected_feature_records": rejected_records,
        "training_target_evidence": target_evidence,
        "causality": {
            "feature_files_available_before_qualifying": True,
            "target_file_separate_and_inference_ineligible": True,
            "target_session_used_for_roster": False,
        },
    }
    manifest_path = event_dir / "telemetry_manifest.json"
    event_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _load_records(
    output_root: Path,
    *,
    year: int | None = None,
) -> Iterable[dict[str, Any]]:
    search_root = output_root / str(int(year)) if year is not None else output_root
    pattern = "round_*/telemetry_manifest.json" if year is not None else "*/round_*/telemetry_manifest.json"
    for path in sorted(search_root.glob(pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest_year = int(payload.get("year", int(payload["event_key"]) // 100))
        if year is not None and manifest_year != int(year):
            raise ValueError(
                f"telemetry manifest {path} declares year {manifest_year}, expected {year}"
            )
        yield from payload.get("feature_records", [])


def _manifest_evidence(
    output_root: Path,
    *,
    year: int,
    root: Path,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    search_root = output_root / str(int(year))
    for path in sorted(search_root.glob("round_*/telemetry_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest_year = int(payload.get("year", int(payload["event_key"]) // 100))
        if manifest_year != int(year):
            raise ValueError(
                f"telemetry manifest {path} declares year {manifest_year}, expected {year}"
            )
        evidence.append(
            {
                "path": _portable(path, root),
                "sha256": _sha256(path),
                "schema_version": str(payload.get("schema_version") or ""),
                "event_key": int(payload["event_key"]),
                "qualifying_start_utc": str(payload.get("qualifying_start_utc") or ""),
                "feature_record_count": int(len(payload.get("feature_records", []))),
                "rejected_feature_record_count": int(
                    len(payload.get("rejected_feature_records", []))
                ),
            }
        )
    return evidence


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def main() -> int:
    root = find_repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--round", type=int, action="append", dest="rounds")
    parser.add_argument(
        "--output-root", type=Path, default=root / "data" / "f1" / "telemetry" / "pre_qualifying"
    )
    parser.add_argument("--cache-dir", type=Path, default=root / ".cache" / "fastf1")
    parser.add_argument("--maximum-laps-per-driver", type=int, default=3)
    parser.add_argument("--distance-bins", type=int, default=DEFAULT_DISTANCE_BINS)
    parser.add_argument(
        "--minimum-distance-coverage",
        type=float,
        default=DEFAULT_MINIMUM_DISTANCE_COVERAGE,
    )
    parser.add_argument("--include-training-targets", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--minimum-independent-events",
        type=int,
        default=1,
        help=(
            "caller-defined complete-event requirement for a concrete split; "
            "this is not a neural-network capacity or promotion threshold"
        ),
    )
    parser.add_argument("--minimum-drivers-per-event", type=int, default=18)
    parser.add_argument(
        "--audit-output",
        type=Path,
        help="write the immutable audit payload with exclusive-create semantics",
    )
    args = parser.parse_args()

    captured_manifests: list[str] = []
    if not args.audit_only:
        if not args.rounds:
            raise SystemExit("at least one --round is required unless --audit-only is set")
        for round_number in sorted(set(args.rounds)):
            path = capture_round(
                year=args.year,
                round_number=round_number,
                output_root=args.output_root,
                cache_dir=args.cache_dir,
                maximum_laps_per_driver=args.maximum_laps_per_driver,
                distance_bins=args.distance_bins,
                minimum_distance_coverage=args.minimum_distance_coverage,
                include_training_targets=bool(args.include_training_targets),
            )
            captured_manifests.append(str(path))

    audited_manifests = _manifest_evidence(
        args.output_root,
        year=args.year,
        root=root,
    )
    audit = audit_telemetry_cache_manifests(
        _load_records(args.output_root, year=args.year),
        root=root,
        minimum_independent_events=args.minimum_independent_events,
        minimum_drivers_per_event=args.minimum_drivers_per_event,
    )
    payload = {
        "schema_version": "f1_prequal_telemetry_cache_audit_v2",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "year": int(args.year),
        "captured_manifests": captured_manifests,
        "manifests": audited_manifests,
        "manifest_set_sha256": _canonical_sha256(audited_manifests),
        "audit": audit.to_payload(),
    }
    if args.audit_output is not None:
        output = args.audit_output.expanduser()
        if not output.is_absolute():
            output = root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Suggested commit name: feat(f1-telemetry): capture timestamped pre-quali tensors
