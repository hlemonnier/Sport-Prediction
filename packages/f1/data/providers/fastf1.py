"""Concrete provider adapter."""

from __future__ import annotations

from .base import (
    BaseProvider,
    Dict,
    List,
    Optional,
    Path,
    POINTS_TABLE,
    Tuple,
    _assign_pit_lane_grid_positions,
    _parse_grid_position_status,
    _standardize_grid_columns,
    fastf1,
    find_repo_root,
    first_available,
    hashlib,
    json,
    merge_fp_frames,
    normalize_event_name,
    os,
    pd,
    re,
    requests,
    time,
)

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
