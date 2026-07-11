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
    PredictionTarget,
    Session,
    SessionCutoff,
    _assign_pit_lane_grid_positions,
    _eligible_pace_session_indices,
    _parse_grid_position_status,
    _standardize_grid_columns,
    canonicalize_session_sequence,
    complete_classification_positions,
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
from .practice_features import FP_FEATURE_CONTRACT_VERSION, build_session_pace_features, normalize_driver_id

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

    def _pre_qualifying_sessions(
        self,
        year: int,
        round_number: int,
        *,
        session_cutoff: str | SessionCutoff | None = None,
        prediction_target: str | PredictionTarget = "qualifying",
    ) -> list[tuple[str, str]]:
        schedule = fastf1.get_event_schedule(year)
        round_values = pd.to_numeric(schedule.get("RoundNumber"), errors="coerce")
        matches = schedule.loc[round_values == int(round_number)]
        if matches.empty:
            self._last_session_cutoff_resolved = "unavailable_schedule"
            self._last_weekend_format_version = "unknown"
            return []
        event = matches.iloc[0]
        ordered_names = [str(event.get(f"Session{idx}") or "").strip() for idx in range(1, 6)]
        session_names = [name for name in ordered_names if name]
        selected_indices, cutoff_label, weekend_format = _eligible_pace_session_indices(
            year=year,
            session_names=session_names,
            prediction_target=prediction_target,
            session_cutoff=session_cutoff,
            event_format_hint=str(event.get("EventFormat") or ""),
        )
        canonical_sessions = canonicalize_session_sequence(year, session_names)
        selected: list[tuple[str, str]] = []
        label_by_session = {
            Session.FP1: "FP1",
            Session.FP2: "FP2",
            Session.FP3: "FP3",
            Session.SPRINT_QUALIFYING: "SQ",
            Session.SPRINT: "Sprint",
        }
        for index in selected_indices:
            session = canonical_sessions[index]
            if session in label_by_session:
                selected.append((session_names[index], label_by_session[session]))
        self._last_session_cutoff_resolved = cutoff_label
        self._last_weekend_format_version = weekend_format
        return selected

    def _session_pace_features(self, year: int, round_number: int, session_name: str) -> pd.DataFrame:
        session = fastf1.get_session(year, round_number, session_name)
        session.load()
        try:
            results = session.results
        except Exception:
            return pd.DataFrame()
        if results is None or results.empty:
            return pd.DataFrame()
        laps = session.laps.copy()
        return build_session_pace_features(laps, session_name, provider="fastf1")

    def get_fp_features(
        self,
        year: int,
        round_number: int,
        *,
        session_cutoff: str | SessionCutoff | None = None,
        prediction_target: str | PredictionTarget = "qualifying",
        prediction_as_of: Optional[str] = None,
    ) -> pd.DataFrame:
        if prediction_as_of is not None:
            raise ValueError(
                "FastF1 retrospective sessions do not expose immutable first-seen timestamps; "
                "prediction_as_of requires a frozen local weekend snapshot.",
            )
        frames: List[pd.DataFrame] = []
        for session_name, label in self._pre_qualifying_sessions(
            year,
            round_number,
            session_cutoff=session_cutoff,
            prediction_target=prediction_target,
        ):
            df = self._session_pace_features(year, round_number, session_name)
            if df.empty:
                continue
            df["session"] = label
            frames.append(df)
        merged = merge_fp_frames(frames)
        if not merged.empty:
            merged["fp_feature_contract_version"] = FP_FEATURE_CONTRACT_VERSION
            merged["fp_feature_source"] = "fastf1"
            merged["session_cutoff_resolved"] = getattr(
                self,
                "_last_session_cutoff_resolved",
                "unknown",
            )
            merged["weekend_format_version"] = getattr(
                self,
                "_last_weekend_format_version",
                "unknown",
            )
        return merged

    def get_qualifying_results(self, year: int, round_number: int) -> pd.DataFrame:
        session = fastf1.get_session(year, round_number, "Q")
        session.load()
        results = session.results.copy()
        if results.empty:
            return pd.DataFrame()
        driver_col = first_available(results, ["DriverNumber", "Abbreviation", "Driver", "FullName"])
        name_col = first_available(results, ["Abbreviation", "Driver", "BroadcastName", "FullName"])
        pos_col = first_available(results, ["Position", "GridPosition"])
        q3_col = "Q3" if "Q3" in results.columns else None
        if driver_col is None:
            return pd.DataFrame()
        df = pd.DataFrame(index=results.index)
        df["driver_id"] = results[driver_col].map(normalize_driver_id)
        df["driver_name"] = (
            results[name_col].fillna(df["driver_id"]).astype(str)
            if name_col
            else df["driver_id"]
        )
        if pos_col:
            df["position"] = pd.to_numeric(results[pos_col], errors="coerce")
        if q3_col:
            q3 = results[q3_col]
            df["q3_time"] = q3.dt.total_seconds()
        return complete_classification_positions(df)

    def get_race_results(self, year: int, round_number: int) -> pd.DataFrame:
        session = fastf1.get_session(year, round_number, "R")
        session.load()
        results = session.results.copy()
        if results.empty:
            return pd.DataFrame()
        driver_col = first_available(results, ["DriverNumber", "Abbreviation", "Driver", "FullName"])
        name_col = first_available(results, ["Abbreviation", "Driver", "BroadcastName", "FullName"])
        pos_col = first_available(results, ["Position", "ClassifiedPosition"])
        grid_col = first_available(results, ["GridPosition", "Grid", "StartingGridPosition"])
        if driver_col is None or pos_col is None:
            return pd.DataFrame()
        df = pd.DataFrame(index=results.index)
        df["driver_id"] = results[driver_col].map(normalize_driver_id)
        df["driver_name"] = (
            results[name_col].fillna(df["driver_id"]).astype(str)
            if name_col
            else df["driver_id"]
        )
        df["position"] = pd.to_numeric(results[pos_col], errors="coerce")
        if grid_col:
            df["grid_position"] = pd.to_numeric(results[grid_col], errors="coerce")
        return complete_classification_positions(df)

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
