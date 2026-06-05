"""Data providers for FastF1 and OpenF1."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from .constants import POINTS_TABLE
from .utils import first_available, merge_fp_frames, normalize_event_name

try:
    import fastf1
except Exception:  # pragma: no cover - optional dependency
    fastf1 = None


class BaseProvider:
    def list_rounds(self, year: int) -> List[Dict[str, object]]:
        raise NotImplementedError

    def get_fp_features(self, year: int, round_number: int) -> pd.DataFrame:
        raise NotImplementedError

    def get_qualifying_results(self, year: int, round_number: int) -> pd.DataFrame:
        raise NotImplementedError

    def get_race_results(self, year: int, round_number: int) -> pd.DataFrame:
        raise NotImplementedError

    def get_standings(self, year: int, round_number: int) -> Optional[pd.DataFrame]:
        return None

    def get_track_stats(self, year: int, round_number: int) -> Optional[dict[str, float]]:
        return None


class FastF1Provider(BaseProvider):
    def __init__(self, cache_dir: Optional[str]) -> None:
        if fastf1 is None:
            raise SystemExit(
                "FastF1 is not installed. Install with: pip install fastf1"
            )
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            fastf1.Cache.enable_cache(cache_dir)

    def list_rounds(self, year: int) -> List[Dict[str, object]]:
        schedule = fastf1.get_event_schedule(year)
        rounds: List[Dict[str, object]] = []
        for _, row in schedule.iterrows():
            rounds.append({
                "round_number": int(row["RoundNumber"]),
                "event_name": row["EventName"],
            })
        return rounds

    def _session_best_laps(self, year: int, round_number: int, session_name: str) -> pd.DataFrame:
        session = fastf1.get_session(year, round_number, session_name)
        session.load()
        laps = session.laps
        laps = laps[["Driver", "LapTime"]].dropna()
        if laps.empty:
            return pd.DataFrame(columns=["driver_id", "driver_name", "best_lap"])
        best = laps.groupby("Driver")["LapTime"].min()
        best_seconds = best.dt.total_seconds()
        df = best_seconds.reset_index().rename(columns={"Driver": "driver_id", "LapTime": "best_lap"})
        df["driver_name"] = df["driver_id"]
        return df

    def get_fp_features(self, year: int, round_number: int) -> pd.DataFrame:
        fp_sessions = ["FP1", "FP2", "FP3"]
        frames: List[pd.DataFrame] = []
        for sess in fp_sessions:
            df = self._session_best_laps(year, round_number, sess)
            if df.empty:
                continue
            df = df.copy()
            df["delta"] = df["best_lap"] - df["best_lap"].min()
            df["rank"] = df["best_lap"].rank(method="min").astype(int)
            df["session"] = sess
            frames.append(df[["driver_id", "driver_name", "delta", "rank", "session"]])
        return merge_fp_frames(frames)

    def get_qualifying_results(self, year: int, round_number: int) -> pd.DataFrame:
        session = fastf1.get_session(year, round_number, "Q")
        session.load()
        results = session.results.copy()
        if results.empty:
            return pd.DataFrame()
        driver_col = first_available(results, ["Abbreviation", "Driver", "DriverNumber", "FullName"])
        pos_col = first_available(results, ["Position", "GridPosition"])
        q3_col = "Q3" if "Q3" in results.columns else None
        if driver_col is None:
            return pd.DataFrame()
        df = results[[driver_col]].copy()
        df = df.rename(columns={driver_col: "driver_id"})
        df["driver_name"] = df["driver_id"].astype(str)
        if pos_col:
            df["position"] = pd.to_numeric(results[pos_col], errors="coerce")
        if q3_col:
            q3 = results[q3_col]
            df["q3_time"] = q3.dt.total_seconds()
        return df

    def get_race_results(self, year: int, round_number: int) -> pd.DataFrame:
        session = fastf1.get_session(year, round_number, "R")
        session.load()
        results = session.results.copy()
        if results.empty:
            return pd.DataFrame()
        driver_col = first_available(results, ["Abbreviation", "Driver", "DriverNumber", "FullName"])
        pos_col = first_available(results, ["Position", "ClassifiedPosition"])
        grid_col = first_available(results, ["GridPosition", "Grid", "StartingGridPosition"])
        if driver_col is None or pos_col is None:
            return pd.DataFrame()
        df = results[[driver_col, pos_col]].copy()
        df = df.rename(columns={driver_col: "driver_id", pos_col: "position"})
        df["driver_name"] = df["driver_id"].astype(str)
        df["position"] = pd.to_numeric(df["position"], errors="coerce")
        if grid_col:
            df["grid_position"] = pd.to_numeric(results[grid_col], errors="coerce")
        return df

    def get_standings(self, year: int, round_number: int) -> Optional[pd.DataFrame]:
        if round_number <= 1:
            return None
        standings: Dict[str, int] = {}
        for rnd in range(1, round_number):
            race = self.get_race_results(year, rnd)
            if race.empty:
                continue
            for _, row in race.iterrows():
                pos = int(row["position"]) if not pd.isna(row["position"]) else None
                if pos is None or pos > 10:
                    continue
                driver = str(row["driver_id"])
                standings[driver] = standings.get(driver, 0) + POINTS_TABLE.get(pos, 0)
        if not standings:
            return None
        df = pd.DataFrame(
            [(k, v) for k, v in standings.items()],
            columns=["driver_id", "points"],
        )
        df["position_start"] = df["points"].rank(method="min", ascending=False).astype(int)
        df["driver_name"] = df["driver_id"]
        return df[["driver_id", "driver_name", "position_start"]]


class OpenF1Provider(BaseProvider):
    def __init__(
        self,
        cache_dir: Optional[str],
        target_round: Optional[int] = None,
        meeting_name: Optional[str] = None,
        country_name: Optional[str] = None,
    ) -> None:
        self.base_url = "https://api.openf1.org/v1"
        self.cache_dir = cache_dir
        self.target_round = target_round
        self.meeting_name = meeting_name
        self.country_name = country_name
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, url: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        digest = hashlib.md5(url.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{digest}.json")

    def _get_json(self, endpoint: str, params: Dict[str, object]) -> List[Dict[str, object]]:
        query = "&".join(f"{k}={params[k]}" for k in sorted(params))
        url = f"{self.base_url}/{endpoint}?{query}" if query else f"{self.base_url}/{endpoint}"
        cache_path = self._cache_path(url)
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 429 and attempt < 2:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_seconds = max(1.0, float(retry_after))
                        except ValueError:
                            wait_seconds = float(attempt + 1)
                    else:
                        wait_seconds = float(attempt + 1)
                    time.sleep(wait_seconds)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.RequestException as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(float(attempt + 1))
                    continue
                raise
        else:
            if last_error is not None:
                raise last_error
            raise RuntimeError(f"Failed to fetch {url}")
        if cache_path:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        return data

    def list_rounds(self, year: int) -> List[Dict[str, object]]:
        meetings = self._get_json("meetings", {"year": year})
        meetings_sorted = sorted(meetings, key=lambda m: m.get("date_start", ""))
        rounds: List[Dict[str, object]] = []
        for idx, meeting in enumerate(meetings_sorted, start=1):
            rounds.append({
                "round_number": idx,
                "meeting_key": meeting.get("meeting_key"),
                "meeting_name": meeting.get("meeting_name"),
                "country_name": meeting.get("country_name"),
            })
        return rounds

    def _meeting_for_round(
        self,
        year: int,
        round_number: int,
        meeting_name: Optional[str],
        country_name: Optional[str],
    ) -> Dict[str, object]:
        if meeting_name:
            meetings = self._get_json("meetings", {"year": year, "meeting_name": meeting_name})
            if not meetings:
                raise SystemExit(f"No meeting found for meeting_name={meeting_name}")
            return meetings[0]
        if country_name:
            meetings = self._get_json("meetings", {"year": year, "country_name": country_name})
            if not meetings:
                raise SystemExit(f"No meeting found for country_name={country_name}")
            return meetings[0]
        rounds = self.list_rounds(year)
        if round_number < 1 or round_number > len(rounds):
            raise SystemExit(f"Round {round_number} is out of range for year {year}")
        match = rounds[round_number - 1]
        meeting_key = match.get("meeting_key")
        meetings = self._get_json("meetings", {"year": year, "meeting_key": meeting_key})
        if not meetings:
            raise SystemExit(f"No meeting found for meeting_key={meeting_key}")
        return meetings[0]

    def _meeting_filters(self, round_number: int) -> Tuple[Optional[str], Optional[str]]:
        if self.target_round is not None and round_number == self.target_round:
            return self.meeting_name, self.country_name
        return None, None

    def _session_key(self, meeting_key: int, session_name: str) -> Optional[int]:
        sessions = self._get_json("sessions", {"meeting_key": meeting_key, "session_name": session_name})
        if not sessions:
            return None
        return sessions[0].get("session_key")

    def _drivers_for_session(self, session_key: int) -> Dict[str, str]:
        drivers = self._get_json("drivers", {"session_key": session_key})
        mapping: Dict[str, str] = {}
        for d in drivers:
            number = str(d.get("driver_number"))
            acronym = d.get("name_acronym") or number
            mapping[number] = acronym
        return mapping

    def get_fp_features(self, year: int, round_number: int) -> pd.DataFrame:
        meeting_name, country_name = self._meeting_filters(round_number)
        meeting = self._meeting_for_round(year, round_number, meeting_name, country_name)
        meeting_key = meeting.get("meeting_key")
        frames: List[pd.DataFrame] = []
        for sess_name, label in [("Practice 1", "FP1"), ("Practice 2", "FP2"), ("Practice 3", "FP3")]:
            session_key = self._session_key(meeting_key, sess_name)
            if not session_key:
                continue
            results = self._get_json("session_result", {"session_key": session_key})
            if not results:
                continue
            driver_map = self._drivers_for_session(session_key)
            rows = []
            for r in results:
                duration = r.get("duration")
                if duration is None:
                    continue
                driver_number = str(r.get("driver_number"))
                rows.append({
                    "driver_id": driver_number,
                    "driver_name": driver_map.get(driver_number, driver_number),
                    "best_lap": float(duration),
                })
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["delta"] = df["best_lap"] - df["best_lap"].min()
            df["rank"] = df["best_lap"].rank(method="min").astype(int)
            df["session"] = label
            frames.append(df[["driver_id", "driver_name", "delta", "rank", "session"]])
        return merge_fp_frames(frames)

    def get_qualifying_results(self, year: int, round_number: int) -> pd.DataFrame:
        meeting_name, country_name = self._meeting_filters(round_number)
        meeting = self._meeting_for_round(year, round_number, meeting_name, country_name)
        meeting_key = meeting.get("meeting_key")
        session_key = self._session_key(meeting_key, "Qualifying")
        if not session_key:
            return pd.DataFrame()
        results = self._get_json("session_result", {"session_key": session_key})
        if not results:
            return pd.DataFrame()
        driver_map = self._drivers_for_session(session_key)
        rows = []
        for r in results:
            duration = r.get("duration")
            q3_time = None
            if isinstance(duration, list) and len(duration) >= 3:
                q3_time = duration[2]
            rows.append({
                "driver_id": str(r.get("driver_number")),
                "driver_name": driver_map.get(str(r.get("driver_number")), str(r.get("driver_number"))),
                "position": r.get("position"),
                "q3_time": q3_time,
            })
        return pd.DataFrame(rows)

    def get_race_results(self, year: int, round_number: int) -> pd.DataFrame:
        meeting_name, country_name = self._meeting_filters(round_number)
        meeting = self._meeting_for_round(year, round_number, meeting_name, country_name)
        meeting_key = meeting.get("meeting_key")
        session_key = self._session_key(meeting_key, "Race")
        if not session_key:
            return pd.DataFrame()
        results = self._get_json("session_result", {"session_key": session_key})
        if not results:
            return pd.DataFrame()
        driver_map = self._drivers_for_session(session_key)
        rows = []
        for r in results:
            rows.append({
                "driver_id": str(r.get("driver_number")),
                "driver_name": driver_map.get(str(r.get("driver_number")), str(r.get("driver_number"))),
                "position": r.get("position"),
                "grid_position": r.get("grid_position") or r.get("starting_grid_position"),
            })
        df = pd.DataFrame(rows)
        df["position"] = pd.to_numeric(df["position"], errors="coerce")
        df["grid_position"] = pd.to_numeric(df["grid_position"], errors="coerce")
        return df

    def get_standings(self, year: int, round_number: int) -> Optional[pd.DataFrame]:
        if round_number <= 1:
            return None
        meeting = self._meeting_for_round(year, round_number - 1, None, None)
        meeting_key = meeting.get("meeting_key")
        session_key = self._session_key(meeting_key, "Race")
        if not session_key:
            return None
        standings = self._get_json("championship_drivers", {"session_key": session_key})
        if not standings:
            return None
        rows = []
        for s in standings:
            rows.append({
                "driver_id": str(s.get("driver_number")),
                "driver_name": str(s.get("driver_number")),
                "position_start": s.get("position_start") or s.get("position_current"),
            })
        df = pd.DataFrame(rows)
        df["position_start"] = pd.to_numeric(df["position_start"], errors="coerce")
        return df


class LocalWeekendProvider(BaseProvider):
    SLOW_LAP_DELTA_SEC = 5.0
    QUALI_SIM_DELTA_SEC = 1.4
    QUALI_SIM_TYRE_LIFE_MAX = 3.0
    RACE_SIM_MIN_STINT_LAPS = 5
    RACE_SIM_MIN_DELTA_SEC = 0.8
    RACE_SIM_MAX_DELTA_SEC = 5.5

    def __init__(self, weekends_dir: Optional[str] = None) -> None:
        self.project_root = Path(__file__).resolve().parents[6]
        if weekends_dir:
            self.weekends_root = Path(weekends_dir).expanduser()
            if not self.weekends_root.is_absolute():
                self.weekends_root = self.project_root / self.weekends_root
        else:
            self.weekends_root = self.project_root / "data" / "f1" / "raw" / "weekends"
        self._event_summary_cache: dict[tuple[int, int], Optional[dict[str, object]]] = {}

    @staticmethod
    def _round_number_from_name(name: str) -> Optional[int]:
        match = re.search(r"round_(\d+)", name)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _safe_int(value: object, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_driver_id(value: object) -> str:
        if value is None or pd.isna(value):
            return ""
        text = str(value).strip()
        if not text:
            return ""
        if re.fullmatch(r"\d+(\.0+)?", text):
            return str(int(float(text)))
        return text

    @staticmethod
    def _mode_or_first(values: pd.Series, fallback: str) -> str:
        clean = values.dropna().astype(str).str.strip()
        clean = clean[clean != ""]
        if clean.empty:
            return fallback
        mode = clean.mode(dropna=True)
        if mode.empty:
            return str(clean.iloc[0])
        return str(mode.iloc[0])

    def _year_dir(self, year: int) -> Path:
        return self.weekends_root / str(year)

    def _read_weekend_meta(self, weekend_dir: Path) -> dict[str, object]:
        meta_path = weekend_dir / "weekend_metadata.json"
        if not meta_path.exists():
            return {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                return payload
            return {}
        except Exception:
            return {}

    def _weekends_for_year(self, year: int) -> list[dict[str, object]]:
        year_dir = self._year_dir(year)
        if not year_dir.exists():
            return []
        weekends: list[dict[str, object]] = []
        for candidate in sorted(year_dir.iterdir(), key=lambda p: p.name):
            if not candidate.is_dir():
                continue
            round_number = self._round_number_from_name(candidate.name)
            if round_number is None:
                continue
            meta = self._read_weekend_meta(candidate)
            round_number = self._safe_int(meta.get("round_number"), default=round_number)
            event_name = str(meta.get("event_name") or f"Round {round_number}")
            sessions = meta.get("sessions")
            if not isinstance(sessions, list):
                sessions = []
            weekends.append(
                {
                    "round_number": round_number,
                    "event_name": event_name,
                    "event_dir": candidate,
                    "sessions": sessions,
                },
            )
        weekends.sort(key=lambda item: int(item["round_number"]))
        return weekends

    def _weekend_for_round(self, year: int, round_number: int) -> Optional[dict[str, object]]:
        for weekend in self._weekends_for_year(year):
            if int(weekend["round_number"]) == int(round_number):
                return weekend
        return None

    def _resolve_session_path(self, weekend_dir: Path, raw_path: object) -> Optional[Path]:
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
                self.project_root / path,
                weekend_dir / path.name,
                weekend_dir / path,
            ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _read_session_csv(
        self,
        weekend_dir: Path,
        session_entry: dict[str, object],
        key: str,
    ) -> pd.DataFrame:
        path = self._resolve_session_path(weekend_dir, session_entry.get(key))
        if path is None:
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    def _session_entries(self, year: int, round_number: int) -> tuple[Path, list[dict[str, object]]]:
        weekend = self._weekend_for_round(year, round_number)
        if weekend is None:
            return Path(), []
        weekend_dir = weekend["event_dir"]
        if not isinstance(weekend_dir, Path):
            weekend_dir = Path(str(weekend_dir))
        sessions_raw = weekend.get("sessions")
        sessions: list[dict[str, object]] = []
        if isinstance(sessions_raw, list):
            for entry in sessions_raw:
                if isinstance(entry, dict):
                    sessions.append(dict(entry))
        sessions.sort(key=lambda s: self._safe_int(s.get("session_order"), default=999))
        return weekend_dir, sessions

    def _select_fp_sessions(self, sessions: list[dict[str, object]]) -> list[dict[str, object]]:
        qualifying_orders = [
            self._safe_int(s.get("session_order"), default=999)
            for s in sessions
            if str(s.get("session_type", "")).strip().lower() == "qualifying"
        ]
        qualifying_order = min(qualifying_orders) if qualifying_orders else 999
        selected = [
            s
            for s in sessions
            if self._safe_int(s.get("session_order"), default=999) < qualifying_order
            and str(s.get("session_type", "")).strip().lower()
            in {"free_practice", "sprint_qualifying", "sprint_race"}
        ]
        if selected:
            return selected
        return [
            s
            for s in sessions
            if str(s.get("session_type", "")).strip().lower() == "free_practice"
        ]

    def _session_label(self, session_type: str, free_practice_idx: int) -> str:
        normalized = session_type.strip().lower()
        if normalized == "free_practice":
            return f"FP{free_practice_idx}"
        if normalized == "sprint_qualifying":
            return "SQ"
        if normalized == "sprint_race":
            return "Sprint"
        return "Session"

    def _session_pace_features(self, laps: pd.DataFrame, label: str) -> pd.DataFrame:
        if laps.empty:
            return pd.DataFrame()
        driver_col = first_available(laps, ["DriverNumber", "driver_number", "Driver", "driver_id"])
        lap_col = first_available(laps, ["LapTime", "lap_time", "duration"])
        if driver_col is None or lap_col is None:
            return pd.DataFrame()

        work = laps.copy()
        work["driver_id"] = work[driver_col].map(self._normalize_driver_id)
        work = work[work["driver_id"] != ""]
        if work.empty:
            return pd.DataFrame()

        name_col = first_available(work, ["Driver", "Abbreviation", "BroadcastName", "driver_name"])
        team_col = first_available(work, ["Team", "TeamName", "team_name"])
        lap_number_col = first_available(work, ["LapNumber", "lap_number"])
        stint_col = first_available(work, ["Stint", "stint"])
        tyre_life_col = first_available(work, ["TyreLife", "tyre_life"])
        fresh_tyre_col = first_available(work, ["FreshTyre", "fresh_tyre"])
        pit_out_col = first_available(work, ["PitOutTime", "pit_out_time", "pit_out"])
        if name_col:
            work["driver_name"] = work[name_col].astype(str)
        else:
            work["driver_name"] = work["driver_id"]
        if team_col:
            work["team_name"] = work[team_col]
        else:
            work["team_name"] = pd.NA

        work["lap_time"] = pd.to_numeric(work[lap_col], errors="coerce")
        work = work[work["lap_time"].notna() & (work["lap_time"] > 0.0)]
        if work.empty:
            return pd.DataFrame()

        if "IsAccurate" in work.columns:
            accurate = work["IsAccurate"]
            if accurate.dtype == bool:
                accurate_mask = accurate
            else:
                accurate_mask = (
                    accurate.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})
                )
            if accurate_mask.any():
                work = work[accurate_mask]
        if work.empty:
            return pd.DataFrame()

        if lap_number_col:
            work["lap_order"] = pd.to_numeric(work[lap_number_col], errors="coerce")
        else:
            work["lap_order"] = range(1, len(work) + 1)
        work["lap_order"] = pd.to_numeric(work["lap_order"], errors="coerce").fillna(9999.0)

        rows: list[dict[str, object]] = []
        for driver_id, group in work.groupby("driver_id", sort=False):
            group = group.sort_values("lap_order", kind="mergesort").copy()
            lap_times = pd.to_numeric(group["lap_time"], errors="coerce").dropna().sort_values()
            if lap_times.empty:
                continue
            top_count = min(3, len(lap_times))
            best_lap = float(lap_times.iloc[0])
            top3_lap = float(lap_times.iloc[:top_count].mean())
            median_lap = float(lap_times.median())
            lap_std = float(lap_times.std(ddof=0)) if len(lap_times) > 1 else 0.0

            if stint_col:
                stint_id = pd.to_numeric(group[stint_col], errors="coerce")
            else:
                stint_id = pd.Series(float("nan"), index=group.index, dtype=float)
            if stint_id.notna().sum() == 0:
                if pit_out_col:
                    pit_out = group[pit_out_col].notna()
                    stint_id = pit_out.cumsum() + 1
                else:
                    stint_id = pd.Series(1, index=group.index, dtype=float)
            group["stint_id"] = stint_id.ffill().bfill().fillna(1).astype(int)
            stint_sizes = group.groupby("stint_id", sort=False)["lap_time"].transform("size").astype(float)

            if tyre_life_col:
                tyre_life = pd.to_numeric(group[tyre_life_col], errors="coerce")
            else:
                tyre_life = pd.Series(float("nan"), index=group.index, dtype=float)
            if fresh_tyre_col:
                fresh_raw = group[fresh_tyre_col]
                if fresh_raw.dtype == bool:
                    fresh_tyre = fresh_raw.fillna(False)
                else:
                    fresh_tyre = (
                        fresh_raw.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})
                    )
            else:
                fresh_tyre = pd.Series(False, index=group.index, dtype=bool)

            driver_lap_time = pd.to_numeric(group["lap_time"], errors="coerce")
            lap_delta = driver_lap_time - best_lap
            slow_mask = lap_delta > self.SLOW_LAP_DELTA_SEC
            usable_mask = (~slow_mask) & driver_lap_time.notna()

            quali_mask = (
                usable_mask
                & (lap_delta <= self.QUALI_SIM_DELTA_SEC)
                & (
                    (tyre_life <= self.QUALI_SIM_TYRE_LIFE_MAX)
                    | fresh_tyre
                    | (stint_sizes <= 3)
                )
            )
            quali_laps = driver_lap_time[quali_mask].dropna().sort_values()
            if quali_laps.empty:
                fallback_quali = driver_lap_time[usable_mask].dropna().sort_values()
                quali_laps = fallback_quali.head(min(2, len(fallback_quali)))
            if not quali_laps.empty:
                quali_sim_lap = float(quali_laps.mean())
                quali_sim_lap_count = int(len(quali_laps))
            else:
                quali_sim_lap = float("nan")
                quali_sim_lap_count = 0

            race_mask = (
                usable_mask
                & (stint_sizes >= self.RACE_SIM_MIN_STINT_LAPS)
                & (lap_delta >= self.RACE_SIM_MIN_DELTA_SEC)
                & (lap_delta <= self.RACE_SIM_MAX_DELTA_SEC)
            )
            race_laps = driver_lap_time[race_mask].dropna()
            if race_laps.empty:
                long_stint_laps = driver_lap_time[usable_mask & (stint_sizes >= self.RACE_SIM_MIN_STINT_LAPS)].dropna()
                if not long_stint_laps.empty:
                    race_laps = long_stint_laps
            if race_laps.empty:
                fallback_race = driver_lap_time[usable_mask].dropna().sort_values()
                if len(fallback_race) > 4:
                    race_laps = fallback_race.iloc[2:]
                else:
                    race_laps = fallback_race
            if not race_laps.empty:
                race_sim_lap = float(race_laps.mean())
                race_sim_lap_count = int(len(race_laps))
            else:
                race_sim_lap = float("nan")
                race_sim_lap_count = 0

            slow_lap_ratio = float(slow_mask.mean()) if len(slow_mask) > 0 else 0.0
            if pd.notna(quali_sim_lap) and pd.notna(race_sim_lap):
                quali_vs_race_gap = float(race_sim_lap - quali_sim_lap)
            else:
                quali_vs_race_gap = float("nan")

            rows.append(
                {
                    "driver_id": str(driver_id),
                    "driver_name": self._mode_or_first(group["driver_name"], fallback=str(driver_id)),
                    "team_name": self._mode_or_first(group["team_name"], fallback=""),
                    "best_lap": best_lap,
                    "top3_lap": top3_lap,
                    "median_lap": median_lap,
                    "lap_std": lap_std,
                    "lap_count": int(len(lap_times)),
                    "slow_lap_ratio": slow_lap_ratio,
                    "quali_sim_lap": quali_sim_lap,
                    "quali_sim_lap_count": quali_sim_lap_count,
                    "race_sim_lap": race_sim_lap,
                    "race_sim_lap_count": race_sim_lap_count,
                    "quali_vs_race_gap": quali_vs_race_gap,
                },
            )
        if not rows:
            return pd.DataFrame()

        frame = pd.DataFrame(rows)
        frame["delta"] = frame["best_lap"] - frame["best_lap"].min()
        frame["rank"] = frame["best_lap"].rank(method="min").astype(int)
        frame["top3_delta"] = frame["top3_lap"] - frame["top3_lap"].min()
        frame["median_delta"] = frame["median_lap"] - frame["median_lap"].min()
        if frame["quali_sim_lap"].notna().sum() > 0:
            frame["quali_sim_delta"] = frame["quali_sim_lap"] - frame["quali_sim_lap"].min(skipna=True)
            frame["quali_sim_rank"] = frame["quali_sim_lap"].rank(method="min")
        else:
            frame["quali_sim_delta"] = float("nan")
            frame["quali_sim_rank"] = float("nan")
        if frame["race_sim_lap"].notna().sum() > 0:
            frame["race_sim_delta"] = frame["race_sim_lap"] - frame["race_sim_lap"].min(skipna=True)
            frame["race_sim_rank"] = frame["race_sim_lap"].rank(method="min")
        else:
            frame["race_sim_delta"] = float("nan")
            frame["race_sim_rank"] = float("nan")
        frame["session"] = label
        return frame[
            [
                "driver_id",
                "driver_name",
                "team_name",
                "delta",
                "rank",
                "top3_delta",
                "median_delta",
                "lap_std",
                "lap_count",
                "slow_lap_ratio",
                "quali_sim_delta",
                "quali_sim_rank",
                "quali_sim_lap_count",
                "race_sim_delta",
                "race_sim_rank",
                "race_sim_lap_count",
                "quali_vs_race_gap",
                "session",
            ]
        ]

    def _find_session_entry(
        self,
        sessions: list[dict[str, object]],
        session_type: str,
    ) -> Optional[dict[str, object]]:
        normalized = session_type.strip().lower()
        matches = [
            s for s in sessions if str(s.get("session_type", "")).strip().lower() == normalized
        ]
        if not matches:
            return None
        matches.sort(key=lambda s: self._safe_int(s.get("session_order"), default=999))
        return matches[0]

    @staticmethod
    def _track_status_codes(value: object) -> set[str]:
        if value is None:
            return set()
        try:
            if pd.isna(value):
                return set()
        except Exception:
            pass
        text = str(value).strip()
        if not text:
            return set()
        return {char for char in text if char.isdigit()}

    def _race_event_summary(self, year: int, round_number: int) -> Optional[dict[str, object]]:
        cache_key = (int(year), int(round_number))
        if cache_key in self._event_summary_cache:
            return self._event_summary_cache[cache_key]

        weekend = self._weekend_for_round(year, round_number)
        if weekend is None:
            self._event_summary_cache[cache_key] = None
            return None
        event_name_norm = normalize_event_name(weekend.get("event_name"))

        qualy = self.get_qualifying_results(year, round_number)
        race = self.get_race_results(year, round_number)
        if qualy.empty or race.empty:
            self._event_summary_cache[cache_key] = None
            return None

        qualy_pos = pd.DataFrame(
            {
                "driver_id": qualy["driver_id"].astype(str),
                "qualy_position": pd.to_numeric(qualy.get("position"), errors="coerce"),
            },
        )
        race_pos = pd.DataFrame(
            {
                "driver_id": race["driver_id"].astype(str),
                "race_position": pd.to_numeric(race.get("position"), errors="coerce"),
                "grid_position": pd.to_numeric(race.get("grid_position"), errors="coerce"),
            },
        )
        merged = qualy_pos.merge(race_pos, on="driver_id", how="inner")
        merged["start_position"] = merged["grid_position"].where(
            merged["grid_position"].notna(),
            merged["qualy_position"],
        )
        merged = merged.dropna(subset=["start_position", "race_position"])
        if merged.empty:
            self._event_summary_cache[cache_key] = None
            return None

        grid_corr = merged["start_position"].corr(merged["race_position"], method="spearman")
        grid_stability = 0.5 if pd.isna(grid_corr) else float(min(1.0, max(0.0, grid_corr)))

        weekend_dir, sessions = self._session_entries(year, round_number)
        race_entry = self._find_session_entry(sessions, "race")
        laps = self._read_session_csv(weekend_dir, race_entry, "laps_path") if race_entry else pd.DataFrame()
        race_results_raw = (
            self._read_session_csv(weekend_dir, race_entry, "results_path") if race_entry else pd.DataFrame()
        )

        safety_car_presence = 0.0
        sc_lap_ratio = 0.0
        vsc_lap_ratio = 0.0
        pit_stop_intensity = 0.0
        weather_uncertainty = 0.0
        if not laps.empty:
            status_col = first_available(laps, ["TrackStatus", "track_status"])
            lap_number_col = first_available(laps, ["LapNumber", "lap_number"])
            if status_col:
                if lap_number_col:
                    lap_numbers = pd.to_numeric(laps[lap_number_col], errors="coerce")
                    lap_df = pd.DataFrame({"lap": lap_numbers, "status": laps[status_col]})
                    lap_df = lap_df.dropna(subset=["lap"])
                    status_sets = lap_df.groupby("lap", sort=False)["status"].apply(
                        lambda values: set().union(*(self._track_status_codes(v) for v in values)),
                    )
                else:
                    status_sets = laps[status_col].apply(self._track_status_codes)
                if not status_sets.empty:
                    sc_flags = status_sets.apply(lambda codes: ("4" in codes) or ("6" in codes) or ("7" in codes))
                    vsc_flags = status_sets.apply(lambda codes: ("6" in codes) or ("7" in codes))
                    sc_lap_ratio = float(sc_flags.mean())
                    vsc_lap_ratio = float(vsc_flags.mean())
                    safety_car_presence = float(sc_flags.any())

            driver_col = first_available(laps, ["DriverNumber", "driver_number", "Driver", "driver_id"])
            pit_in_col = first_available(laps, ["PitInTime", "pit_in_time", "pit_in"])
            if driver_col and pit_in_col:
                work = pd.DataFrame(
                    {
                        "driver_id": laps[driver_col].map(self._normalize_driver_id),
                        "pit_in": laps[pit_in_col],
                    },
                )
                work = work[(work["driver_id"] != "") & work["pit_in"].notna()]
                if not work.empty:
                    pit_count = work.groupby("driver_id", sort=False)["pit_in"].size()
                    pit_stop_intensity = float(pit_count.mean())
            compound_col = first_available(laps, ["Compound", "compound"])
            if compound_col:
                compound = laps[compound_col].astype(str).str.strip().str.lower()
                wet_tyre_ratio = float(compound.str.contains("wet|intermediate", regex=True).mean())
                weather_uncertainty = max(weather_uncertainty, wet_tyre_ratio)
            rain_col = first_available(laps, ["Rainfall", "rainfall", "rain"])
            if rain_col:
                rain = pd.to_numeric(laps[rain_col], errors="coerce")
                if rain.notna().sum() > 0:
                    weather_uncertainty = max(weather_uncertainty, float((rain.fillna(0.0) > 0.0).mean()))

        dnf_rate = 0.0
        dnf_driver_ids: set[str] = set()
        if not race_results_raw.empty:
            driver_col_raw = first_available(
                race_results_raw,
                ["DriverNumber", "driver_number", "Abbreviation", "Driver", "DriverId", "FullName"],
            )
            status_col = first_available(race_results_raw, ["Status", "status"])
            class_col = first_available(race_results_raw, ["ClassifiedPosition", "Classified", "Position"])
            dnf_mask = pd.Series(False, index=race_results_raw.index, dtype=bool)
            if status_col:
                status_text = race_results_raw[status_col].astype(str).str.strip().str.lower()
                dnf_mask = dnf_mask | status_text.str.contains(
                    r"retired|accident|disqual|dnf|did not|not classified|collision|damage|engine|gearbox|brake",
                    regex=True,
                )
            if class_col:
                classified = pd.to_numeric(race_results_raw[class_col], errors="coerce")
                dnf_mask = dnf_mask | classified.isna()
            dnf_rate = float(dnf_mask.mean())
            if driver_col_raw:
                dnf_driver_ids = set(
                    race_results_raw.loc[dnf_mask, driver_col_raw].map(self._normalize_driver_id).astype(str).tolist()
                )

        overtake_frame = merged.copy()
        if dnf_driver_ids:
            overtake_frame = overtake_frame[~overtake_frame["driver_id"].astype(str).isin(dnf_driver_ids)]
        if overtake_frame.empty:
            overtake_frame = merged
        field_size = float(len(overtake_frame))
        pos_delta = overtake_frame["start_position"] - overtake_frame["race_position"]
        overtake_propensity = float(pos_delta.abs().mean() / max(1.0, field_size - 1.0))
        overtake_propensity = float(min(1.0, max(0.0, overtake_propensity)))

        summary: dict[str, object] = {
            "event_name_norm": event_name_norm,
            "event_year": int(year),
            "event_round": int(round_number),
            "overtake_propensity": overtake_propensity,
            "grid_stability": grid_stability,
            "safety_car_presence": float(safety_car_presence),
            "sc_lap_ratio": float(sc_lap_ratio),
            "vsc_lap_ratio": float(vsc_lap_ratio),
            "dnf_rate": float(max(0.0, min(1.0, dnf_rate))),
            "pit_stop_intensity": float(max(0.0, pit_stop_intensity)),
            "weather_uncertainty": float(max(0.0, min(1.0, weather_uncertainty))),
        }
        self._event_summary_cache[cache_key] = summary
        return summary

    def get_track_stats(self, year: int, round_number: int) -> Optional[dict[str, float]]:
        weekend = self._weekend_for_round(year, round_number)
        if weekend is None:
            return None
        if not self.weekends_root.exists():
            return None
        target_event_norm = normalize_event_name(weekend.get("event_name"))

        history: list[dict[str, object]] = []
        for year_dir in sorted(self.weekends_root.iterdir(), key=lambda p: p.name):
            if not year_dir.is_dir():
                continue
            try:
                event_year = int(year_dir.name)
            except ValueError:
                continue
            if event_year > int(year):
                continue
            for round_meta in self._weekends_for_year(event_year):
                event_round = int(round_meta.get("round_number", 0))
                if event_round <= 0:
                    continue
                if event_year == int(year) and event_round >= int(round_number):
                    continue
                summary = self._race_event_summary(event_year, event_round)
                if summary is not None:
                    history.append(summary)

        if not history:
            return None

        same_track = [
            s
            for s in history
            if normalize_event_name(s.get("event_name_norm")) == target_event_norm
        ]
        source = same_track if same_track else history
        source_frame = pd.DataFrame(source)
        if source_frame.empty:
            return None

        overtake = pd.to_numeric(source_frame.get("overtake_propensity"), errors="coerce")
        grid_stability = pd.to_numeric(source_frame.get("grid_stability"), errors="coerce")
        sc_presence = pd.to_numeric(source_frame.get("safety_car_presence"), errors="coerce")
        sc_lap_ratio = pd.to_numeric(source_frame.get("sc_lap_ratio"), errors="coerce")
        vsc_lap_ratio = pd.to_numeric(source_frame.get("vsc_lap_ratio"), errors="coerce")
        dnf_rate = pd.to_numeric(source_frame.get("dnf_rate"), errors="coerce")
        pit_stop = pd.to_numeric(source_frame.get("pit_stop_intensity"), errors="coerce")
        weather = pd.to_numeric(source_frame.get("weather_uncertainty"), errors="coerce")

        same_track_count = float(len(same_track))
        history_count = float(len(source))
        if same_track_count > 0:
            reliability = min(1.0, same_track_count / 3.0)
        else:
            reliability = min(0.4, history_count / 10.0)

        pit_variance = (pit_stop.fillna(pit_stop.mean(skipna=True)) / 3.0).clip(lower=0.0, upper=1.0)
        chaos = (
            (0.45 * sc_presence.fillna(sc_presence.mean(skipna=True)))
            + (0.20 * sc_lap_ratio.fillna(sc_lap_ratio.mean(skipna=True)))
            + (0.20 * dnf_rate.fillna(dnf_rate.mean(skipna=True)))
            + (0.15 * pit_variance)
        )
        chaos_value = float(chaos.mean(skipna=True)) if chaos.notna().sum() > 0 else 0.5

        return {
            "track_overtake_propensity": float(overtake.mean(skipna=True)),
            "track_grid_stability": float(grid_stability.mean(skipna=True)),
            "track_safety_car_propensity": float(sc_presence.mean(skipna=True)),
            "track_sc_lap_ratio": float(sc_lap_ratio.mean(skipna=True)),
            "track_vsc_lap_ratio": float(vsc_lap_ratio.mean(skipna=True)),
            "track_dnf_rate": float(dnf_rate.mean(skipna=True)),
            "track_pit_stop_intensity": float(pit_stop.mean(skipna=True)),
            "track_weather_uncertainty": float(weather.mean(skipna=True)),
            "track_same_event_count": same_track_count,
            "track_history_count": history_count,
            "track_stats_reliability": float(reliability),
            "track_chaos_index": float(min(1.0, max(0.0, chaos_value))),
        }

    def list_rounds(self, year: int) -> List[Dict[str, object]]:
        rounds: List[Dict[str, object]] = []
        for weekend in self._weekends_for_year(year):
            rounds.append(
                {
                    "round_number": int(weekend["round_number"]),
                    "event_name": str(weekend["event_name"]),
                },
            )
        return rounds

    def get_fp_features(self, year: int, round_number: int) -> pd.DataFrame:
        weekend_dir, sessions = self._session_entries(year, round_number)
        if not sessions:
            return pd.DataFrame()
        selected = self._select_fp_sessions(sessions)
        if not selected:
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        fp_idx = 0
        for entry in selected:
            session_type = str(entry.get("session_type", ""))
            if session_type.strip().lower() == "free_practice":
                fp_idx += 1
                label = self._session_label(session_type, fp_idx)
            else:
                label = self._session_label(session_type, fp_idx)
            laps = self._read_session_csv(weekend_dir, entry, "laps_path")
            session_frame = self._session_pace_features(laps, label=label)
            if session_frame.empty:
                continue
            frames.append(session_frame)
        return merge_fp_frames(frames)

    def get_qualifying_results(self, year: int, round_number: int) -> pd.DataFrame:
        weekend_dir, sessions = self._session_entries(year, round_number)
        if not sessions:
            return pd.DataFrame()
        entry = self._find_session_entry(sessions, "qualifying")
        if entry is None:
            return pd.DataFrame()
        results = self._read_session_csv(weekend_dir, entry, "results_path")
        if results.empty:
            return pd.DataFrame()

        driver_col = first_available(
            results,
            ["DriverNumber", "driver_number", "Abbreviation", "Driver", "DriverId", "FullName"],
        )
        if driver_col is None:
            return pd.DataFrame()
        name_col = first_available(results, ["Abbreviation", "BroadcastName", "FullName", "Driver"])
        pos_col = first_available(results, ["Position", "ClassifiedPosition", "GridPosition"])
        q3_col = first_available(results, ["Q3", "q3_time"])
        team_col = first_available(results, ["TeamName", "Team", "team_name"])

        frame = pd.DataFrame()
        frame["driver_id"] = results[driver_col].map(self._normalize_driver_id)
        if name_col:
            frame["driver_name"] = results[name_col].fillna(frame["driver_id"]).astype(str)
        else:
            frame["driver_name"] = frame["driver_id"]
        if pos_col:
            frame["position"] = pd.to_numeric(results[pos_col], errors="coerce")
        if q3_col:
            frame["q3_time"] = pd.to_numeric(results[q3_col], errors="coerce")
        if team_col:
            frame["team_name"] = results[team_col].astype(str)
        frame = frame[frame["driver_id"] != ""]
        return frame

    def get_race_results(self, year: int, round_number: int) -> pd.DataFrame:
        weekend_dir, sessions = self._session_entries(year, round_number)
        if not sessions:
            return pd.DataFrame()
        entry = self._find_session_entry(sessions, "race")
        if entry is None:
            return pd.DataFrame()
        results = self._read_session_csv(weekend_dir, entry, "results_path")
        if results.empty:
            return pd.DataFrame()

        driver_col = first_available(
            results,
            ["DriverNumber", "driver_number", "Abbreviation", "Driver", "DriverId", "FullName"],
        )
        pos_col = first_available(results, ["Position", "ClassifiedPosition"])
        if driver_col is None or pos_col is None:
            return pd.DataFrame()

        name_col = first_available(results, ["Abbreviation", "BroadcastName", "FullName", "Driver"])
        team_col = first_available(results, ["TeamName", "Team", "team_name"])
        grid_col = first_available(results, ["GridPosition", "Grid", "StartingGridPosition", "grid_position"])
        frame = pd.DataFrame()
        frame["driver_id"] = results[driver_col].map(self._normalize_driver_id)
        if name_col:
            frame["driver_name"] = results[name_col].fillna(frame["driver_id"]).astype(str)
        else:
            frame["driver_name"] = frame["driver_id"]
        frame["position"] = pd.to_numeric(results[pos_col], errors="coerce")
        if grid_col:
            frame["grid_position"] = pd.to_numeric(results[grid_col], errors="coerce")
        if team_col:
            frame["team_name"] = results[team_col].astype(str)
        frame = frame[frame["driver_id"] != ""]
        return frame

    def get_standings(self, year: int, round_number: int) -> Optional[pd.DataFrame]:
        if round_number <= 1:
            return None
        standings: dict[str, int] = {}
        driver_name: dict[str, str] = {}
        rounds = self.list_rounds(year)
        for round_meta in rounds:
            rnd = int(round_meta.get("round_number", 0))
            if rnd <= 0 or rnd >= round_number:
                continue
            race = self.get_race_results(year, rnd)
            if race.empty:
                continue
            for _, row in race.iterrows():
                pos = pd.to_numeric(pd.Series([row.get("position")]), errors="coerce").iloc[0]
                if pd.isna(pos):
                    continue
                pos_int = int(pos)
                if pos_int < 1 or pos_int > 10:
                    continue
                driver_id = str(row.get("driver_id", "")).strip()
                if not driver_id:
                    continue
                standings[driver_id] = standings.get(driver_id, 0) + POINTS_TABLE.get(pos_int, 0)
                driver_name[driver_id] = str(row.get("driver_name", driver_id))
        if not standings:
            return None
        frame = pd.DataFrame(
            [(driver_id, pts, driver_name.get(driver_id, driver_id)) for driver_id, pts in standings.items()],
            columns=["driver_id", "points", "driver_name"],
        )
        frame["position_start"] = frame["points"].rank(method="min", ascending=False).astype(int)
        return frame[["driver_id", "driver_name", "position_start"]]
