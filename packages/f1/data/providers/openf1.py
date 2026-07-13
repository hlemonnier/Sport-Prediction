"""Concrete provider adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

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
from .practice_features import FP_FEATURE_CONTRACT_VERSION, build_session_pace_features
from packages.f1.features.race import (
    aggregate_race_practice_evidence,
    derive_race_practice_evidence,
)
from packages.f1.domain.starting_grid import (
    GridAdjustmentKind,
    GridEntryStatus,
    GridRevisionPhase,
    OfficialGridDecision,
    OfficialGridEntry,
    OfficialGridRevision,
    RaceGridCapture,
    build_race_grid_capture,
    persist_race_grid_capture,
)
from packages.f1.domain.weekend import build_weekend_contract

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

    def _get_json_first_seen(
        self,
        endpoint: str,
        params: Dict[str, object],
    ) -> List[Dict[str, object]]:
        """Fetch mutable evidence without replaying a stale response cache."""

        cache_dir = self.cache_dir
        self.cache_dir = None
        try:
            return self._get_json(endpoint, params)
        finally:
            self.cache_dir = cache_dir

    @staticmethod
    def _capture_timestamp(value: object | None) -> str:
        parsed = pd.to_datetime(
            value if value is not None else datetime.now(timezone.utc),
            errors="coerce",
            utc=True,
        )
        if pd.isna(parsed):
            raise ValueError("grid capture timestamp must be timezone-aware")
        return pd.Timestamp(parsed).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _first_value(row: Dict[str, object], *keys: str) -> object | None:
        for key in keys:
            if key in row and row[key] is not None and str(row[key]).strip():
                return row[key]
        return None

    @classmethod
    def _official_grid_entry(
        cls,
        row: Dict[str, object],
        *,
        revision_id: str,
    ) -> OfficialGridEntry | None:
        raw_driver = cls._first_value(
            row,
            "driver_number",
            "driver_id",
            "DriverNumber",
        )
        if raw_driver is None:
            return None
        driver_id = str(raw_driver).strip()
        if re.fullmatch(r"\d+(?:\.0+)?", driver_id):
            driver_id = str(int(float(driver_id)))
        raw_position = cls._first_value(
            row,
            "position",
            "grid_position",
            "starting_grid_position",
            "GridPosition",
        )
        status_text = str(
            cls._first_value(row, "status", "grid_status", "start_status") or ""
        ).strip().lower().replace("-", "_").replace(" ", "_")
        numeric_position = pd.to_numeric(pd.Series([raw_position]), errors="coerce").iloc[0]
        position = None if pd.isna(numeric_position) else int(numeric_position)
        if "pit" in status_text or position == 0:
            status = GridEntryStatus.PIT_LANE
            position = None
        elif any(token in status_text for token in ("withdraw", "wd")):
            status = GridEntryStatus.WITHDRAWN
            position = None
        elif any(token in status_text for token in ("did_not_start", "dns", "nonstarter")):
            status = GridEntryStatus.DID_NOT_START
            position = None
        elif any(token in status_text for token in ("disqual", "dsq", "excluded")):
            status = GridEntryStatus.DISQUALIFIED
            position = None
        elif position is not None and position > 0:
            status = GridEntryStatus.GRID
        else:
            status = GridEntryStatus.UNRESOLVED
            position = None

        decisions: list[OfficialGridDecision] = []
        status_decision = {
            GridEntryStatus.PIT_LANE: GridAdjustmentKind.PIT_LANE_START,
            GridEntryStatus.WITHDRAWN: GridAdjustmentKind.WITHDRAWAL,
            GridEntryStatus.DISQUALIFIED: GridAdjustmentKind.DISQUALIFICATION,
        }.get(status)
        penalty_places_raw = cls._first_value(row, "penalty_places", "grid_drop_places")
        penalty_places_numeric = pd.to_numeric(
            pd.Series([penalty_places_raw]), errors="coerce"
        ).iloc[0]
        penalty_places = (
            None
            if pd.isna(penalty_places_numeric) or int(penalty_places_numeric) <= 0
            else int(penalty_places_numeric)
        )
        reason_raw = cls._first_value(row, "penalty_reason", "reason", "decision_reason")
        reason = None if reason_raw is None else str(reason_raw).strip() or None
        if status_decision is not None:
            decisions.append(
                OfficialGridDecision(
                    kind=status_decision,
                    evidence_id=revision_id,
                    reason=reason,
                )
            )
        elif penalty_places is not None:
            decisions.append(
                OfficialGridDecision(
                    kind=GridAdjustmentKind.GRID_DROP,
                    places=penalty_places,
                    reason=reason,
                    evidence_id=revision_id,
                )
            )
        evidence_complete = status is not GridEntryStatus.UNRESOLVED
        return OfficialGridEntry(
            driver_id=driver_id,
            position=position,
            status=status,
            decisions=tuple(decisions),
            evidence_complete=evidence_complete,
        )

    def capture_starting_grid_snapshot(
        self,
        year: int,
        round_number: int,
        *,
        output_dir: str | Path,
        revision_phase: GridRevisionPhase = GridRevisionPhase.PROVISIONAL_PRE_RACE,
        captured_at: object | None = None,
        first_published_at: object | None = None,
        source_document_url: str | None = None,
        source_document_sha256: str | None = None,
        expected_position_driver_pairs: Sequence[tuple[int, str]] | None = None,
    ) -> tuple[RaceGridCapture, Path]:
        """Capture the mutable OpenF1 starting-grid response exactly once.

        By default the capture is provisional and therefore unavailable to the
        ``post_grid_pre_race`` model.  Passing an authoritative FIA publication
        timestamp plus document URL/hash permits later download/backfill while
        keeping the causal publication time distinct from ``captured_at``.
        The optional PDF-extracted pairs must match the OpenF1 structured order.
        """

        if not isinstance(revision_phase, GridRevisionPhase):
            raise TypeError("revision_phase must be a GridRevisionPhase")
        captured_text = self._capture_timestamp(captured_at)
        if first_published_at is None:
            published_text = captured_text
            semantics = "first_seen_upper_bound"
        else:
            published_text = self._capture_timestamp(first_published_at)
            semantics = "authoritative_document_timestamp"
        meeting_name, country_name = self._meeting_filters(round_number)
        meeting = self._meeting_for_round(year, round_number, meeting_name, country_name)
        meeting_key = meeting.get("meeting_key")
        if meeting_key is None:
            raise ValueError("OpenF1 meeting is missing meeting_key")
        session_key = self._session_key(int(meeting_key), "Race")
        if session_key is None:
            raise ValueError("OpenF1 meeting has no Race session")
        session_rows = self._get_json("sessions", {"meeting_key": int(meeting_key)})
        race_session = next(
            (
                row
                for row in session_rows
                if int(row.get("session_key") or -1) == int(session_key)
            ),
            None,
        )
        race_start_at = None if race_session is None else race_session.get("date_start")
        if race_start_at is None or pd.isna(
            pd.to_datetime(race_start_at, errors="coerce", utc=True)
        ):
            raise ValueError("OpenF1 Race session lacks a provable start timestamp")
        raw_rows = self._get_json_first_seen("starting_grid", {"session_key": int(session_key)})
        if not raw_rows:
            raise ValueError("OpenF1 starting_grid endpoint returned no rows")
        raw_hash = hashlib.sha256(
            json.dumps(raw_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        revision_id = (
            f"openf1:{session_key}:{published_text}:"
            f"{(source_document_sha256 or raw_hash)[:16]}"
        )
        entries = tuple(
            entry
            for entry in (
                self._official_grid_entry(dict(row), revision_id=revision_id)
                for row in raw_rows
            )
            if entry is not None
        )
        expected_field_size = build_weekend_contract(int(year)).eligible_cars
        driver_ids = [entry.driver_id for entry in entries]
        positions = [
            entry.position
            for entry in entries
            if entry.status is GridEntryStatus.GRID and entry.position is not None
        ]
        complete = bool(
            len(entries) == expected_field_size
            and len(driver_ids) == len(set(driver_ids))
            and len(positions) == len(set(positions))
            and all(entry.evidence_complete for entry in entries)
        )
        if expected_position_driver_pairs is not None:
            expected_pairs = sorted(
                (int(position), str(driver_id))
                for position, driver_id in expected_position_driver_pairs
            )
            observed_pairs = sorted(
                (int(entry.position), entry.driver_id)
                for entry in entries
                if entry.position is not None
            )
            if observed_pairs != expected_pairs:
                raise ValueError(
                    "OpenF1 starting grid does not match FIA document position/driver pairs"
                )
        revision = OfficialGridRevision(
            revision_id=revision_id,
            phase=revision_phase,
            entries=entries,
            as_of=published_text,
            evidence_complete=complete,
        )
        capture = build_race_grid_capture(
            build_weekend_contract(int(year)),
            year=int(year),
            round_number=int(round_number),
            provider="openf1",
            source_endpoint=f"{self.base_url}/starting_grid?session_key={session_key}",
            meeting_key=str(meeting_key),
            session_key=str(session_key),
            captured_at=captured_text,
            first_published_at=published_text,
            race_start_at=str(race_start_at),
            publication_time_semantics=semantics,
            source_document_url=source_document_url,
            source_document_sha256=source_document_sha256,
            revision=revision,
            raw_payload=raw_rows,
        )
        output = persist_race_grid_capture(capture, output_dir)
        return capture, output

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

    def _driver_metadata_for_session(self, session_key: int) -> Dict[str, Dict[str, str]]:
        rows = self._get_json("drivers", {"session_key": session_key})
        return {
            str(row.get("driver_number")): {
                "driver_name": str(row.get("name_acronym") or row.get("full_name") or row.get("driver_number")),
                "team_name": str(row.get("team_name") or ""),
            }
            for row in rows
            if row.get("driver_number") is not None
        }

    def _pre_qualifying_sessions(
        self,
        meeting_key: int,
        *,
        year: int = 2026,
        session_cutoff: str | SessionCutoff | None = None,
        prediction_target: str | PredictionTarget = "qualifying",
    ) -> list[tuple[int, str]]:
        sessions = self._get_json("sessions", {"meeting_key": meeting_key})
        sessions = sorted(sessions, key=lambda row: str(row.get("date_start") or ""))
        session_rows = [
            row
            for row in sessions
            if row.get("session_key") is not None and str(row.get("session_name") or "").strip()
        ]
        session_names = [str(row.get("session_name") or "").strip() for row in session_rows]
        selected_indices, cutoff_label, weekend_format = _eligible_pace_session_indices(
            year=year,
            session_names=session_names,
            prediction_target=prediction_target,
            session_cutoff=session_cutoff,
        )
        canonical_sessions = canonicalize_session_sequence(year, session_names)
        selected: list[tuple[int, str]] = []
        label_by_session = {
            Session.FP1: "FP1",
            Session.FP2: "FP2",
            Session.FP3: "FP3",
            Session.SPRINT_QUALIFYING: "SQ",
            Session.SPRINT: "Sprint",
        }
        for index in selected_indices:
            row = session_rows[index]
            session_key = int(row["session_key"])
            if not self._get_json("session_result", {"session_key": session_key}):
                continue
            session = canonical_sessions[index]
            if session in label_by_session:
                selected.append((session_key, label_by_session[session]))
        self._last_session_cutoff_resolved = cutoff_label
        self._last_weekend_format_version = weekend_format
        return selected

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
                "OpenF1 retrospective responses do not expose immutable first-seen timestamps; "
                "prediction_as_of requires a frozen local weekend snapshot.",
            )
        meeting_name, country_name = self._meeting_filters(round_number)
        meeting = self._meeting_for_round(year, round_number, meeting_name, country_name)
        meeting_key = meeting.get("meeting_key")
        frames: List[pd.DataFrame] = []
        selected_sessions = self._pre_qualifying_sessions(
            int(meeting_key),
            year=year,
            session_cutoff=session_cutoff,
            prediction_target=prediction_target,
        )
        for session_key, label in selected_sessions:
            lap_rows = self._get_json("laps", {"session_key": session_key})
            if not lap_rows:
                continue
            laps = pd.DataFrame(lap_rows)
            metadata = self._driver_metadata_for_session(session_key)
            numbers = laps.get("driver_number", pd.Series(index=laps.index, dtype=object)).astype(str)
            laps["driver_name"] = numbers.map(lambda value: metadata.get(value, {}).get("driver_name", value))
            laps["team_name"] = numbers.map(lambda value: metadata.get(value, {}).get("team_name", ""))
            stints = pd.DataFrame(self._get_json("stints", {"session_key": session_key}))
            feature_rows = build_session_pace_features(
                laps,
                label,
                provider="openf1",
                stints=stints,
            )
            if not feature_rows.empty:
                race_evidence = derive_race_practice_evidence(
                    laps,
                    session_label=label,
                    stints=stints,
                ).drop(columns=["session"], errors="ignore")
                if not race_evidence.empty:
                    feature_rows = feature_rows.merge(
                        race_evidence,
                        on="driver_id",
                        how="left",
                        validate="one_to_one",
                    )
                frames.append(feature_rows)
        merged = merge_fp_frames(frames)
        if not merged.empty:
            merged = aggregate_race_practice_evidence(
                merged,
                expected_sessions=len(selected_sessions),
            )
            merged["fp_feature_contract_version"] = FP_FEATURE_CONTRACT_VERSION
            merged["fp_feature_source"] = "openf1"
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
        return complete_classification_positions(pd.DataFrame(rows))

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
        return complete_classification_positions(df)

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
