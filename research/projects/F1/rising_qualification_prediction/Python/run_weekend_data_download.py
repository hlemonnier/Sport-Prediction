#!/usr/bin/env python3
"""Download raw FastF1 data for a set of race weekends."""

from __future__ import annotations

import argparse
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
]


def default_output_dir() -> str:
    project_root = Path(__file__).resolve().parents[5]
    return str(project_root / "data" / "f1" / "raw" / "weekends")


def default_cache_dir() -> str:
    project_root = Path(__file__).resolve().parents[5]
    return str(project_root / ".cache" / "fastf1")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "unknown"


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
    if session.laps is None or session.laps.empty:
        return pd.DataFrame()
    available_cols = [col for col in LAP_COLUMNS if col in session.laps.columns]
    if not available_cols:
        return pd.DataFrame()
    frame = session.laps[available_cols].copy()
    if "DriverNumber" in frame.columns:
        frame["DriverNumber"] = frame["DriverNumber"].astype(str)
    return frame


def export_results(session: fastf1.core.Session) -> pd.DataFrame:
    if session.results is None or session.results.empty:
        return pd.DataFrame()
    frame = session.results.reset_index(drop=True).copy()
    if "DriverNumber" in frame.columns:
        frame["DriverNumber"] = frame["DriverNumber"].astype(str)
    return frame


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
    output_root: Path,
) -> Dict[str, Any]:
    event_dir = output_root / str(year) / f"round_{round_number:02d}_{slugify(event_name)}"
    event_dir.mkdir(parents=True, exist_ok=True)

    event = fastf1.get_event(year, round_number)
    downloaded_sessions: List[Dict[str, Any]] = []
    notes: List[str] = []

    for i in range(1, 6):
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
            session.load(laps=True, telemetry=False, weather=False, messages=False)
        except Exception as exc:
            notes.append(f"{session_name}: {exc}")
            continue

        laps = normalize_for_csv(export_laps(session))
        results = normalize_for_csv(export_results(session))
        session_slug = slugify(session_name)

        laps_path = None
        if not laps.empty:
            laps_file = event_dir / f"{i:02d}_{session_slug}_laps.csv"
            laps.to_csv(laps_file, index=False)
            laps_path = str(laps_file)

        results_path = None
        if not results.empty:
            results_file = event_dir / f"{i:02d}_{session_slug}_results.csv"
            results.to_csv(results_file, index=False)
            results_path = str(results_file)

        downloaded_sessions.append(
            {
                "session_order": i,
                "session_name": session_name,
                "session_type": kind,
                "laps_rows": int(len(laps)),
                "results_rows": int(len(results)),
                "laps_path": laps_path,
                "results_path": results_path,
            }
        )

    metadata = {
        "year": year,
        "round_number": round_number,
        "event_name": event_name,
        "sessions": downloaded_sessions,
        "notes": notes,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    metadata_path = event_dir / "weekend_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return {
        "round_number": round_number,
        "event_name": event_name,
        "event_dir": str(event_dir),
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
                output_root=output_root,
            )
        )

    payload = {
        "sport": "F1",
        "source": "fastf1",
        "year": args.year,
        "start_round": args.start_round,
        "weekends_requested": args.weekends,
        "weekends_downloaded": len(downloads),
        "output_dir": str(output_root),
        "cache_dir": str(cache_dir),
        "downloads": downloads,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
