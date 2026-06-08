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
