"""Live lap observation sources for local replay and FastF1."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from packages.sports_core.paths import find_repo_root

from packages.f1.data.schemas.session import PredictionConfig
from packages.f1.data.utils import first_available

try:
    import fastf1
except Exception:  # pragma: no cover - optional dependency
    fastf1 = None


@dataclass
class LiveSourceResult:
    frame: pd.DataFrame
    source_used: str
    notes: list[str]


def _normalize_driver_id(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return ""
    if re.fullmatch(r"\d+(\.0+)?", text):
        try:
            return str(int(float(text)))
        except Exception:
            return text
    return text


def _to_seconds(series: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() > 0:
        return numeric
    try:
        parsed = pd.to_timedelta(series, errors="coerce")
        if parsed.notna().sum() > 0:
            return parsed.dt.total_seconds()
    except Exception:
        pass
    return pd.Series(index=series.index, dtype=float)


def _to_bool(series: pd.Series, default: bool = False) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(default)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "yes", "y", "t"})


def _resolve_session_path(base_dir: Path, raw_path: object) -> Optional[Path]:
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    candidates: list[Path]
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [
            base_dir / path,
            base_dir / path.name,
            Path.cwd() / path,
            Path.cwd() / path.name,
        ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _read_local_laps_from_metadata(weekend_dir: Path) -> pd.DataFrame:
    metadata_path = weekend_dir / "weekend_metadata.json"
    if not metadata_path.exists():
        return pd.DataFrame()
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()
    if not isinstance(payload, dict):
        return pd.DataFrame()

    sessions = payload.get("sessions")
    if not isinstance(sessions, list):
        return pd.DataFrame()

    race_sessions = [
        s for s in sessions if isinstance(s, dict) and str(s.get("session_type", "")).strip().lower() == "race"
    ]
    race_sessions.sort(key=lambda item: int(item.get("session_order") or 999))
    if not race_sessions:
        return pd.DataFrame()

    laps_path = _resolve_session_path(weekend_dir, race_sessions[0].get("laps_path"))
    if laps_path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(laps_path)
    except Exception:
        return pd.DataFrame()


def _round_folder(weekends_root: Path, year: int, round_number: int) -> Optional[Path]:
    year_dir = weekends_root / str(int(year))
    if not year_dir.exists():
        return None
    prefix = f"round_{int(round_number):02d}_"
    candidates = [p for p in sorted(year_dir.iterdir(), key=lambda item: item.name) if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        return None
    return candidates[0]


def _build_stint_id(frame: pd.DataFrame) -> pd.Series:
    stint_col = first_available(frame, ["Stint", "stint"])
    if stint_col is not None:
        stint = pd.to_numeric(frame[stint_col], errors="coerce")
        if stint.notna().sum() > 0:
            return stint.ffill().bfill().fillna(1).astype(int)

    pit_in_col = first_available(frame, ["PitInTime", "pit_in_time", "pit_in"])
    pit_out_col = first_available(frame, ["PitOutTime", "pit_out_time", "pit_out"])
    pit_in = frame[pit_in_col].notna() if pit_in_col else pd.Series(False, index=frame.index, dtype=bool)
    pit_out = frame[pit_out_col].notna() if pit_out_col else pd.Series(False, index=frame.index, dtype=bool)
    box = pit_in | pit_out

    if pit_out.any():
        return (pit_out.astype(int).cumsum() + 1).astype(int)
    if pit_in.any():
        return (pit_in.astype(int).cumsum() + 1).astype(int)

    box_to_track = (box.shift(fill_value=False) & (~box)).astype(int)
    stint = (box_to_track.cumsum() + 1).astype(int)

    if int(stint.max()) <= 1:
        lap_time_col = first_available(frame, ["LapTime", "lap_time", "duration"])
        if lap_time_col:
            lap_time = _to_seconds(frame[lap_time_col])
            rolling = lap_time.rolling(window=5, min_periods=3).median()
            slow = lap_time > (rolling * 1.35)
            bump = (slow.fillna(False) & (~slow.shift(fill_value=False))).astype(int)
            stint = (bump.cumsum() + 1).astype(int)

    return stint


def _compute_tyre_age(work: pd.DataFrame) -> pd.Series:
    age = pd.Series(index=work.index, data=0, dtype=int)
    for _, idx in work.groupby(["driver_id", "stint_id"], sort=False).groups.items():
        subset = work.loc[idx].sort_values(["lap_number", "timestamp"], kind="mergesort")
        counter = 0
        values: dict[int, int] = {}
        for row_index, row in subset.iterrows():
            is_box = bool(row.get("is_box_lap", False))
            is_accurate = bool(row.get("is_accurate", False))
            lap_time = pd.to_numeric(pd.Series([row.get("lap_time_seconds")]), errors="coerce").iloc[0]
            assimilable = (not is_box) and is_accurate and pd.notna(lap_time) and float(lap_time) > 0.0
            if assimilable:
                values[int(row_index)] = int(counter)
                counter += 1
            else:
                values[int(row_index)] = int(counter)
        for row_index, row_age in values.items():
            age.loc[row_index] = int(row_age)
    return age


def _build_race_time_seconds(work: pd.DataFrame) -> pd.Series:
    existing_col = first_available(work, ["race_time_seconds", "RaceTimeSeconds"])
    if existing_col:
        existing = pd.to_numeric(work[existing_col], errors="coerce")
        if existing.notna().sum() > 0:
            return existing

    lap_time = pd.to_numeric(work.get("lap_time_seconds"), errors="coerce")

    # Locked convention: race_time = LapStartTime + LapTime whenever available.
    lap_start_col = first_available(work, ["LapStartTime", "lap_start_time"])
    if lap_start_col and "lap_time_seconds" in work.columns:
        lap_start = _to_seconds(work[lap_start_col])
        combined = lap_start + lap_time
        if combined.notna().sum() > 0:
            return combined

    # Secondary fallback: cumulative clean lap times per driver.
    if lap_time.notna().sum() == 0:
        return pd.Series(index=work.index, dtype=float)

    out = pd.Series(index=work.index, dtype=float)
    for _, idx in work.groupby("driver_id", sort=False).groups.items():
        subset = work.loc[idx].sort_values(["lap_number", "timestamp"], kind="mergesort")
        # Preserve unknown lap times as unknown: never coerce missing values to 0.
        # Forward-fill keeps the last known cumulative race time without creating
        # artificial progress for laps with missing timing.
        cumulative = pd.to_numeric(subset["lap_time_seconds"], errors="coerce").cumsum().ffill()
        out.loc[subset.index] = cumulative
    return out


def _standardize_laps(
    laps: pd.DataFrame,
    *,
    event_key: int,
    source_used: str,
    session_name: str = "Race",
) -> pd.DataFrame:
    if laps.empty:
        return pd.DataFrame()

    driver_col = first_available(laps, ["DriverNumber", "driver_number", "Driver", "driver_id"])
    lap_col = first_available(laps, ["LapNumber", "lap_number"])
    lap_time_col = first_available(laps, ["LapTime", "lap_time", "duration"])
    if driver_col is None or lap_col is None or lap_time_col is None:
        return pd.DataFrame()

    work = laps.copy()
    work["driver_id"] = work[driver_col].map(_normalize_driver_id)
    work = work[work["driver_id"] != ""].copy()
    if work.empty:
        return pd.DataFrame()

    name_col = first_available(work, ["Driver", "Abbreviation", "BroadcastName", "driver_name"])
    work["driver_name"] = work[name_col].astype(str) if name_col else work["driver_id"]

    work["lap_number"] = pd.to_numeric(work[lap_col], errors="coerce")
    work = work[work["lap_number"].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    work["lap_number"] = work["lap_number"].astype(int)

    work["lap_time_seconds"] = _to_seconds(work[lap_time_col])

    compound_col = first_available(work, ["Compound", "compound"])
    work["compound"] = work[compound_col].astype(str).str.strip() if compound_col else "UNKNOWN"

    track_status_col = first_available(work, ["TrackStatus", "track_status"])
    if track_status_col:
        work["track_status"] = work[track_status_col].astype(str).str.strip().replace({"nan": ""})
    else:
        work["track_status"] = ""

    accurate_col = first_available(work, ["IsAccurate", "is_accurate"])
    if accurate_col:
        work["is_accurate"] = _to_bool(work[accurate_col], default=False)
    else:
        work["is_accurate"] = True

    pit_in_col = first_available(work, ["PitInTime", "pit_in_time", "pit_in"])
    pit_out_col = first_available(work, ["PitOutTime", "pit_out_time", "pit_out"])
    pit_in = work[pit_in_col].notna() if pit_in_col else pd.Series(False, index=work.index, dtype=bool)
    pit_out = work[pit_out_col].notna() if pit_out_col else pd.Series(False, index=work.index, dtype=bool)
    work["is_box_lap"] = (pit_in | pit_out).astype(bool)

    work["stint_id"] = _build_stint_id(work)

    time_col = first_available(work, ["Time", "time", "LapStartTime", "lap_start_time"])
    if time_col:
        work["timestamp"] = _to_seconds(work[time_col])
    else:
        work["timestamp"] = pd.to_numeric(work["lap_number"], errors="coerce")

    work["race_time_seconds"] = _build_race_time_seconds(work)
    work["gap_to_leader_seconds"] = pd.to_numeric(work.get("gap_to_leader_seconds"), errors="coerce")

    work = work.sort_values(["driver_id", "lap_number", "timestamp"], kind="mergesort").copy()
    work["tyre_age"] = _compute_tyre_age(work)

    out = pd.DataFrame(index=work.index)
    out["event_key"] = int(event_key)
    out["session"] = str(session_name)
    out["driver_id"] = work["driver_id"].astype(str)
    out["driver_name"] = work["driver_name"].fillna(work["driver_id"]).astype(str)
    out["lap_number"] = pd.to_numeric(work["lap_number"], errors="coerce").astype(int)
    out["stint_id"] = pd.to_numeric(work["stint_id"], errors="coerce").fillna(1).astype(int)
    out["compound"] = work["compound"].astype(str)
    out["tyre_age"] = pd.to_numeric(work["tyre_age"], errors="coerce").fillna(0).astype(int)
    out["is_box_lap"] = work["is_box_lap"].astype(bool)
    out["is_accurate"] = work["is_accurate"].astype(bool)
    out["track_status"] = work["track_status"].astype(str)
    out["lap_time_seconds"] = pd.to_numeric(work["lap_time_seconds"], errors="coerce")
    out["timestamp"] = pd.to_numeric(work["timestamp"], errors="coerce")
    out["race_time_seconds"] = pd.to_numeric(work["race_time_seconds"], errors="coerce")
    out["gap_to_leader_seconds"] = pd.to_numeric(work["gap_to_leader_seconds"], errors="coerce")
    out["source"] = str(source_used)
    out["tyre_life_raw"] = pd.to_numeric(work.get("TyreLife"), errors="coerce")

    out = out.sort_values(["lap_number", "timestamp", "driver_id"], kind="mergesort").reset_index(drop=True)
    return out


def _load_local_source(config: PredictionConfig) -> LiveSourceResult:
    notes: list[str] = []
    event_key = (int(config.year) * 100) + int(config.round_number)

    if config.f1_live_replay_path:
        path = Path(config.f1_live_replay_path).expanduser()
        if path.exists() and path.is_file():
            try:
                frame = pd.read_csv(path)
                std = _standardize_laps(frame, event_key=event_key, source_used="local", session_name="Race")
                if not std.empty:
                    return LiveSourceResult(frame=std, source_used="local", notes=notes)
            except Exception as exc:
                notes.append(f"Replay file read failed: {exc}")
        elif path.exists() and path.is_dir():
            frame = _read_local_laps_from_metadata(path)
            std = _standardize_laps(frame, event_key=event_key, source_used="local", session_name="Race")
            if not std.empty:
                return LiveSourceResult(frame=std, source_used="local", notes=notes)
            notes.append("Replay directory found but no race laps loaded.")
        else:
            notes.append(f"Replay path not found: {path}")

    weekends_root = Path(config.weekends_dir or "data/f1/raw/weekends").expanduser()
    if not weekends_root.is_absolute():
        project_root = find_repo_root(__file__)
        weekends_root = (project_root / weekends_root).resolve()

    weekend_dir = _round_folder(weekends_root, config.year, config.round_number)
    if weekend_dir is None:
        notes.append(f"Weekend folder not found for {config.year} R{config.round_number:02d}: {weekends_root}")
        return LiveSourceResult(frame=pd.DataFrame(), source_used="local", notes=notes)

    frame = _read_local_laps_from_metadata(weekend_dir)
    std = _standardize_laps(frame, event_key=event_key, source_used="local", session_name="Race")
    if std.empty:
        notes.append(f"Race laps unavailable in weekend metadata: {weekend_dir}")
    return LiveSourceResult(frame=std, source_used="local", notes=notes)


def _load_fastf1_source(config: PredictionConfig) -> LiveSourceResult:
    notes: list[str] = []
    if fastf1 is None:
        notes.append("FastF1 unavailable: install package fastf1.")
        return LiveSourceResult(frame=pd.DataFrame(), source_used="fastf1", notes=notes)

    cache_dir = config.f1_live_cache_dir or config.cache_dir
    if cache_dir:
        cache_path = Path(cache_dir).expanduser()
        cache_path.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_path))

    try:
        session = fastf1.get_session(int(config.year), int(config.round_number), "R")
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as exc:
        notes.append(f"FastF1 race session load failed: {exc}")
        return LiveSourceResult(frame=pd.DataFrame(), source_used="fastf1", notes=notes)

    try:
        laps = session.laps.copy() if session.laps is not None else pd.DataFrame()
    except Exception as exc:
        notes.append(f"FastF1 race laps unavailable: {exc}")
        return LiveSourceResult(frame=pd.DataFrame(), source_used="fastf1", notes=notes)
    if laps.empty:
        notes.append("FastF1 race laps empty.")
        return LiveSourceResult(frame=pd.DataFrame(), source_used="fastf1", notes=notes)

    event_key = (int(config.year) * 100) + int(config.round_number)
    std = _standardize_laps(laps, event_key=event_key, source_used="fastf1", session_name="Race")
    if std.empty:
        notes.append("FastF1 race laps could not be standardized.")
    return LiveSourceResult(frame=std, source_used="fastf1", notes=notes)


def load_live_observations(config: PredictionConfig) -> LiveSourceResult:
    source = str(config.f1_live_source or "auto").strip().lower()
    if source == "openf1":
        return LiveSourceResult(
            frame=pd.DataFrame(),
            source_used=source,
            notes=["OpenF1 live source is not supported in Horizon B v1. Use local, fastf1 or auto."],
        )

    if source not in {"auto", "local", "fastf1"}:
        return LiveSourceResult(
            frame=pd.DataFrame(),
            source_used=source,
            notes=[f"Unsupported f1_live_source={source}. Allowed: auto, local, fastf1."],
        )

    if source == "local":
        return _load_local_source(config)

    if source == "fastf1":
        return _load_fastf1_source(config)

    local = _load_local_source(config)
    if not local.frame.empty:
        local.notes.append("Source auto: local replay selected.")
        return local

    fast = _load_fastf1_source(config)
    fast.notes = list(local.notes) + ["Source auto: local unavailable, fallback to fastf1."] + list(fast.notes)
    return fast
