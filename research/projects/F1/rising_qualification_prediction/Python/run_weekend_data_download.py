#!/usr/bin/env python3
"""Download raw FastF1 data for a set of race weekends."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import fastf1
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "FastF1 is required for weekend downloads. Install with: pip install fastf1"
    ) from exc


LAP_COLUMNS = [
    "Time",
    "Driver",
    "DriverNumber",
    "LapNumber",
    "LapTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "Stint",
    "Compound",
    "TyreLife",
    "FreshTyre",
    "Team",
    "TrackStatus",
    "Position",
    "IsAccurate",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
    "PitOutTime",
    "PitInTime",
    "Deleted",
    "DeletedReason",
    "FastF1Generated",
]

WEEKEND_METADATA_SCHEMA_VERSION = "f1_weekend_snapshot_v2_point_in_time"


def default_output_dir() -> str:
    project_root = Path(__file__).resolve().parents[5]
    return str(project_root / "data" / "f1" / "raw" / "weekends")


def default_cache_dir() -> str:
    project_root = Path(__file__).resolve().parents[5]
    return str(project_root / ".cache" / "fastf1")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "unknown"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_datetime(value: object) -> Optional[str]:
    if is_missing(value):
        return None
    if hasattr(value, "isoformat"):
        try:
            text = value.isoformat()  # type: ignore[union-attr]
            return str(text).replace("+00:00", "Z")
        except Exception:
            pass
    text = str(value).strip()
    return text or None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path, *, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _file_evidence(path: Optional[Path], *, project_root: Path, rows: int) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    return {
        "path": _portable_path(path, project_root=project_root),
        "sha256": _sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "rows": int(rows),
    }


def is_missing(value: object) -> bool:
    if value is None:
        return True
    return bool(pd.isna(value))


def normalize_for_csv(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    normalized = frame.copy()
    for col in normalized.columns:
        if pd.api.types.is_timedelta64_dtype(normalized[col]):
            normalized[col] = normalized[col].dt.total_seconds()
            continue
        if pd.api.types.is_datetime64_any_dtype(normalized[col]):
            normalized[col] = normalized[col].astype(str)
    return normalized


def session_kind(session_name: str) -> Optional[str]:
    normalized = session_name.strip().lower()
    if normalized.startswith("practice"):
        return "free_practice"
    if normalized == "qualifying":
        return "qualifying"
    if normalized in {"sprint qualifying", "sprint shootout"}:
        return "sprint_qualifying"
    if normalized == "sprint":
        return "sprint_race"
    if normalized == "race":
        return "race"
    return None


def export_laps(session: fastf1.core.Session) -> pd.DataFrame:
    try:
        laps = session.laps
    except Exception:
        return pd.DataFrame()
    if laps is None or laps.empty:
        return pd.DataFrame()
    available_cols = [col for col in LAP_COLUMNS if col in laps.columns]
    if not available_cols:
        return pd.DataFrame()
    frame = laps[available_cols].copy()
    if "DriverNumber" in frame.columns:
        frame["DriverNumber"] = frame["DriverNumber"].astype(str)
    return frame


def export_results(session: fastf1.core.Session) -> pd.DataFrame:
    try:
        results = session.results
    except Exception:
        return pd.DataFrame()
    if results is None or results.empty:
        return pd.DataFrame()
    frame = results.reset_index(drop=True).copy()
    if "DriverNumber" in frame.columns:
        frame["DriverNumber"] = frame["DriverNumber"].astype(str)
    return frame


def export_optional_frame(session: fastf1.core.Session, attribute: str) -> pd.DataFrame:
    try:
        value = getattr(session, attribute)
    except Exception:
        return pd.DataFrame()
    if value is None:
        return pd.DataFrame()
    try:
        frame = value.copy()
    except Exception:
        return pd.DataFrame()
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def pick_rounds(
    schedule: pd.DataFrame,
    start_round: int,
    weekends: int,
) -> List[Dict[str, Any]]:
    races = schedule[schedule["RoundNumber"] > 0].copy()
    races["RoundNumber"] = pd.to_numeric(races["RoundNumber"], errors="coerce")
    races = races.dropna(subset=["RoundNumber"])
    races["RoundNumber"] = races["RoundNumber"].astype(int)
    races = races[races["RoundNumber"] >= start_round]
    races = races.sort_values("RoundNumber")
    selected = races.head(weekends)
    rows: List[Dict[str, Any]] = []
    for _, row in selected.iterrows():
        rows.append(
            {
                "round_number": int(row["RoundNumber"]),
                "event_name": str(row["EventName"]),
                "event_format": str(row.get("EventFormat", "")),
            }
        )
    return rows


def download_weekend(
    year: int,
    round_number: int,
    event_name: str,
    event_format: str,
    output_root: Path,
    max_session_order: Optional[int] = None,
) -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parents[5]
    snapshot_started_at = _utc_now()
    event_dir = output_root / str(year) / f"round_{round_number:02d}_{slugify(event_name)}"
    event_dir.mkdir(parents=True, exist_ok=True)

    event = fastf1.get_event(year, round_number)
    downloaded_sessions: List[Dict[str, Any]] = []
    notes: List[str] = []

    for i in range(1, 6):
        if max_session_order is not None and i > int(max_session_order):
            continue
        session_field = f"Session{i}"
        if session_field not in event.index:
            continue
        raw_name = event[session_field]
        if is_missing(raw_name):
            continue
        session_name = str(raw_name)
        kind = session_kind(session_name)
        if kind is None:
            continue

        try:
            session = fastf1.get_session(year, round_number, session_name)
            session.load(laps=True, telemetry=False, weather=True, messages=True)
        except Exception as exc:
            notes.append(f"{session_name}: {exc}")
            continue

        laps = normalize_for_csv(export_laps(session))
        results = normalize_for_csv(export_results(session))
        weather = normalize_for_csv(export_optional_frame(session, "weather_data"))
        messages = normalize_for_csv(export_optional_frame(session, "race_control_messages"))
        if laps.empty and results.empty:
            notes.append(f"{session_name}: no lap/results data available after load")
            continue
        if results.empty:
            notes.append(
                f"{session_name}: partial snapshot rejected because no post-session classification is available",
            )
            continue
        session_slug = slugify(session_name)

        laps_file: Optional[Path] = None
        if not laps.empty:
            laps_file = event_dir / f"{i:02d}_{session_slug}_laps.csv"
            laps.to_csv(laps_file, index=False)

        results_file = event_dir / f"{i:02d}_{session_slug}_results.csv"
        results.to_csv(results_file, index=False)

        weather_file: Optional[Path] = None
        if not weather.empty:
            weather_file = event_dir / f"{i:02d}_{session_slug}_weather.csv"
            weather.to_csv(weather_file, index=False)

        messages_file: Optional[Path] = None
        if not messages.empty:
            messages_file = event_dir / f"{i:02d}_{session_slug}_race_control_messages.csv"
            messages.to_csv(messages_file, index=False)

        file_evidence = {
            "laps": _file_evidence(laps_file, project_root=project_root, rows=len(laps)),
            "results": _file_evidence(results_file, project_root=project_root, rows=len(results)),
            "weather": _file_evidence(weather_file, project_root=project_root, rows=len(weather)),
            "race_control_messages": _file_evidence(
                messages_file,
                project_root=project_root,
                rows=len(messages),
            ),
        }
        captured_at = _utc_now()

        downloaded_sessions.append(
            {
                "session_order": i,
                "session_name": session_name,
                "session_type": kind,
                "availability_phase": "post_race" if kind == "race" else "post_session",
                "completion_status": "completed_provider_classification",
                "classification_status": "provider_post_session_snapshot",
                "completed": True,
                "available_at": captured_at,
                "captured_at": captured_at,
                "scheduled_start_utc": _json_datetime(event.get(f"Session{i}DateUtc")),
                "source_session_id": f"fastf1:{year}:{round_number}:{session_name}",
                "laps_rows": int(len(laps)),
                "results_rows": int(len(results)),
                "weather_rows": int(len(weather)),
                "race_control_messages_rows": int(len(messages)),
                "laps_path": file_evidence["laps"]["path"] if file_evidence["laps"] else None,
                "results_path": file_evidence["results"]["path"] if file_evidence["results"] else None,
                "weather_path": file_evidence["weather"]["path"] if file_evidence["weather"] else None,
                "race_control_messages_path": (
                    file_evidence["race_control_messages"]["path"]
                    if file_evidence["race_control_messages"]
                    else None
                ),
                "files": file_evidence,
            }
        )

    snapshot_completed_at = _utc_now()
    metadata = {
        "schema_version": WEEKEND_METADATA_SCHEMA_VERSION,
        "source": "fastf1",
        "source_version": str(getattr(fastf1, "__version__", "unknown")),
        "year": year,
        "round_number": round_number,
        "event_name": event_name,
        "event_format": event_format,
        "scheduled_event_date": _json_datetime(event.get("EventDate")),
        "max_session_order": max_session_order,
        "sessions": downloaded_sessions,
        "notes": notes,
        "snapshot_started_at": snapshot_started_at,
        "generated_at": snapshot_completed_at,
        "snapshot_semantics": {
            "immutable_as_of": snapshot_completed_at,
            "partial_sessions_rejected": True,
            "official_fia_classification_guaranteed": False,
            "paths_portable_when_inside_project": True,
        },
    }
    metadata["snapshot_sha256"] = hashlib.sha256(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8"),
    ).hexdigest()
    metadata_path = event_dir / "weekend_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return {
        "round_number": round_number,
        "event_name": event_name,
        "event_format": event_format,
        "event_dir": str(event_dir),
        "metadata_path": str(metadata_path),
        "snapshot_sha256": metadata["snapshot_sha256"],
        "session_count": len(downloaded_sessions),
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download raw FastF1 sessions for local algorithm testing.",
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--start-round", type=int, default=1)
    parser.add_argument("--weekends", type=int, default=5)
    parser.add_argument("--output-dir", default=default_output_dir())
    parser.add_argument("--cache-dir", default=default_cache_dir())
    parser.add_argument("--max-session-order", type=int, default=None)
    parser.add_argument("--output-format", choices=["text", "json"], default="text")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    schedule = fastf1.get_event_schedule(args.year)
    selected_rounds = pick_rounds(
        schedule=schedule,
        start_round=args.start_round,
        weekends=args.weekends,
    )
    if not selected_rounds:
        raise SystemExit(
            f"No race weekends found for year={args.year}, start_round={args.start_round}."
        )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    downloads: List[Dict[str, Any]] = []
    for round_meta in selected_rounds:
        downloads.append(
            download_weekend(
                year=args.year,
                round_number=round_meta["round_number"],
                event_name=round_meta["event_name"],
                event_format=round_meta["event_format"],
                output_root=output_root,
                max_session_order=args.max_session_order,
            )
        )

    payload = {
        "sport": "F1",
        "source": "fastf1",
        "year": args.year,
        "start_round": args.start_round,
        "weekends_requested": args.weekends,
        "weekends_downloaded": len(downloads),
        "max_session_order": args.max_session_order,
        "output_dir": str(output_root),
        "cache_dir": str(cache_dir),
        "downloads": downloads,
        "metadata_schema_version": WEEKEND_METADATA_SCHEMA_VERSION,
        "generated_at": _utc_now(),
    }

    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    if args.quiet:
        return

    if args.output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("=" * 72)
    print("FastF1 weekend data download")
    print("=" * 72)
    print(f"Year: {args.year}")
    print(f"Start round: {args.start_round}")
    print(f"Weekends requested: {args.weekends}")
    print(f"Weekends downloaded: {len(downloads)}")
    print(f"Max session order: {args.max_session_order}")
    print(f"Output directory: {output_root}")
    print(f"Cache directory: {cache_dir}")
    print("\nDownloaded weekends:")
    for item in downloads:
        print(
            f"- R{item['round_number']:02d} {item['event_name']} "
            f"(sessions: {item['session_count']}) -> {item['event_dir']}"
        )
        for note in item["notes"]:
            print(f"  note: {note}")


if __name__ == "__main__":
    main()
