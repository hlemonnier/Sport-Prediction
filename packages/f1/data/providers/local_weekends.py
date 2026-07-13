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
from .practice_features import FP_FEATURE_CONTRACT_VERSION, PracticeFeatureConfig, build_session_pace_features
from packages.f1.features.race import (
    aggregate_race_practice_evidence,
    derive_race_practice_evidence,
)
from packages.f1.domain.starting_grid import (
    GridEntryStatus,
    RaceGridCapture,
    load_race_grid_capture,
)


def _season_entry_list_power_unit(year: int, team_name: object) -> str | None:
    """Causal constructor-to-PU contract from the season entry list."""

    team = re.sub(r"[^a-z0-9]+", " ", str(team_name).lower()).strip()
    if not team or team in {"nan", "none", "null"}:
        return None
    if int(year) >= 2026:
        if "audi" in team:
            return "Audi"
        if any(token in team for token in ("ferrari", "haas", "cadillac")):
            return "Ferrari"
        if "aston martin" in team:
            return "Honda"
        if "red bull" in team or "racing bulls" in team:
            return "Red Bull Ford Powertrains"
        if any(token in team for token in ("mercedes", "mclaren", "williams", "alpine")):
            return "Mercedes"
        return None
    if "alpine" in team:
        return "Renault"
    if any(token in team for token in ("red bull", "alphatauri", "racing bulls")) or team == "rb":
        return "Honda RBPT"
    if any(token in team for token in ("ferrari", "haas", "alfa romeo", "sauber")):
        return "Ferrari"
    if any(token in team for token in ("mercedes", "mclaren", "aston martin", "williams")):
        return "Mercedes"
    return None


class LocalWeekendProvider(BaseProvider):
    SLOW_LAP_DELTA_SEC = 5.0
    QUALI_SIM_DELTA_SEC = 1.4
    QUALI_SIM_TYRE_LIFE_MAX = 3.0
    RACE_SIM_MIN_STINT_LAPS = 5
    RACE_SIM_MIN_DELTA_SEC = 0.8
    RACE_SIM_MAX_DELTA_SEC = 5.5

    def __init__(self, weekends_dir: Optional[str] = None) -> None:
        self.project_root = find_repo_root(__file__)
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
                    "event_format": meta.get("event_format"),
                    "metadata_schema_version": meta.get("schema_version"),
                    "snapshot_sha256": meta.get("snapshot_sha256"),
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

    @staticmethod
    def _session_is_complete(
        session: dict[str, object],
        prediction_as_of: Optional[str] = None,
    ) -> bool:
        if prediction_as_of is not None:
            available_at = session.get("available_at") or session.get("first_published_at")
            if not available_at:
                return False
            try:
                available_timestamp = pd.Timestamp(str(available_at))
                cutoff_timestamp = pd.Timestamp(str(prediction_as_of))
                if available_timestamp.tzinfo is None:
                    available_timestamp = available_timestamp.tz_localize("UTC")
                if cutoff_timestamp.tzinfo is None:
                    cutoff_timestamp = cutoff_timestamp.tz_localize("UTC")
                if available_timestamp > cutoff_timestamp:
                    return False
            except Exception:
                return False
        if session.get("completed") is False:
            return False
        status = str(session.get("completion_status") or "").strip().lower()
        if status:
            if any(token in status for token in ("partial", "ongoing", "scheduled", "suspended")):
                return False
            if any(token in status for token in ("complete", "final", "chequered")):
                return True
        # Backward-compatible read of immutable legacy snapshots.  A provider
        # classification file is required; laps alone are not proof that an
        # in-progress session had finished.
        results_rows = LocalWeekendProvider._safe_int(session.get("results_rows"), default=0)
        return results_rows > 0 and bool(session.get("results_path"))

    def _select_fp_sessions(
        self,
        *,
        year: int,
        sessions: list[dict[str, object]],
        prediction_target: str | PredictionTarget,
        session_cutoff: str | SessionCutoff | None,
        event_format_hint: Optional[str],
        prediction_as_of: Optional[str],
    ) -> tuple[list[dict[str, object]], str, str]:
        session_rows = [
            session
            for session in sessions
            if str(session.get("session_name") or session.get("session_type") or "").strip()
        ]
        session_names = [
            str(session.get("session_name") or session.get("session_type") or "").strip()
            for session in session_rows
        ]
        selected_indices, cutoff_label, weekend_format = _eligible_pace_session_indices(
            year=year,
            session_names=session_names,
            prediction_target=prediction_target,
            session_cutoff=session_cutoff,
            event_format_hint=event_format_hint,
        )
        selected = [
            session_rows[index]
            for index in selected_indices
            if self._session_is_complete(session_rows[index], prediction_as_of)
        ]
        return selected, cutoff_label, weekend_format

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
        return build_session_pace_features(
            laps,
            label,
            provider="local_weekends",
            config=PracticeFeatureConfig(
                slow_lap_delta_seconds=self.SLOW_LAP_DELTA_SEC,
                qualifying_sim_delta_seconds=self.QUALI_SIM_DELTA_SEC,
                qualifying_sim_tyre_life_max=self.QUALI_SIM_TYRE_LIFE_MAX,
                race_sim_min_stint_laps=self.RACE_SIM_MIN_STINT_LAPS,
                race_sim_min_delta_seconds=self.RACE_SIM_MIN_DELTA_SEC,
                race_sim_max_delta_seconds=self.RACE_SIM_MAX_DELTA_SEC,
            ),
        )

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

        mobility_frame = merged.copy()
        if dnf_driver_ids:
            mobility_frame = mobility_frame[~mobility_frame["driver_id"].astype(str).isin(dnf_driver_ids)]
        if mobility_frame.empty:
            mobility_frame = merged
        field_size = float(len(mobility_frame))
        pos_delta = mobility_frame["start_position"] - mobility_frame["race_position"]
        finish_order_mobility = float(pos_delta.abs().mean() / max(1.0, field_size - 1.0))
        finish_order_mobility = float(min(1.0, max(0.0, finish_order_mobility)))

        summary: dict[str, object] = {
            "event_name_norm": event_name_norm,
            "event_year": int(year),
            "event_round": int(round_number),
            "finish_order_mobility": finish_order_mobility,
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

        mobility_source = (
            source_frame.get("finish_order_mobility")
            if "finish_order_mobility" in source_frame.columns
            else source_frame.get("overtake_propensity")
        )
        finish_order_mobility = pd.to_numeric(mobility_source, errors="coerce")
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
            "track_finish_order_mobility": float(finish_order_mobility.mean(skipna=True)),
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

    def get_fp_features(
        self,
        year: int,
        round_number: int,
        *,
        session_cutoff: str | SessionCutoff | None = None,
        prediction_target: str | PredictionTarget = "qualifying",
        prediction_as_of: Optional[str] = None,
    ) -> pd.DataFrame:
        weekend_dir, sessions = self._session_entries(year, round_number)
        if not sessions:
            return pd.DataFrame()
        weekend = self._weekend_for_round(year, round_number) or {}
        selected, cutoff_label, weekend_format = self._select_fp_sessions(
            year=year,
            sessions=sessions,
            prediction_target=prediction_target,
            session_cutoff=session_cutoff,
            event_format_hint=(
                str(weekend.get("event_format")) if weekend.get("event_format") is not None else None
            ),
            prediction_as_of=prediction_as_of,
        )
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
            race_evidence = derive_race_practice_evidence(
                laps,
                session_label=label,
            ).drop(columns=["session"], errors="ignore")
            if not race_evidence.empty:
                session_frame = session_frame.merge(
                    race_evidence,
                    on="driver_id",
                    how="left",
                    validate="one_to_one",
                )
            frames.append(session_frame)
        merged = merge_fp_frames(frames)
        if not merged.empty:
            merged = aggregate_race_practice_evidence(
                merged,
                expected_sessions=len(selected),
            )
            merged["fp_feature_contract_version"] = FP_FEATURE_CONTRACT_VERSION
            merged["fp_feature_source"] = "local_weekends"
            merged["session_cutoff_resolved"] = cutoff_label
            merged["weekend_format_version"] = weekend_format
            snapshot_sha256 = weekend.get("snapshot_sha256")
            if snapshot_sha256:
                merged["weekend_snapshot_sha256"] = str(snapshot_sha256)
        return merged

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
        power_unit_col = first_available(
            results,
            ["PowerUnit", "PowerUnitManufacturer", "EngineManufacturer", "power_unit"],
        )

        frame = pd.DataFrame()
        frame["driver_id"] = results[driver_col].map(self._normalize_driver_id)
        car_number_col = first_available(results, ["DriverNumber", "driver_number"])
        if car_number_col:
            frame["car_number"] = results[car_number_col].map(self._normalize_driver_id)
        abbreviation_col = first_available(results, ["Abbreviation", "driver_abbreviation"])
        if abbreviation_col:
            frame["driver_abbreviation"] = results[abbreviation_col].astype("string").str.strip()
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
        if power_unit_col:
            frame["power_unit"] = results[power_unit_col].astype("string").str.strip()
            frame["power_unit_source"] = "provider_result_field"
        elif "team_name" in frame.columns:
            frame["power_unit"] = frame["team_name"].map(
                lambda value: _season_entry_list_power_unit(int(year), value)
            )
            frame["power_unit_source"] = "causal_season_entry_list_team_mapping"
        frame = frame[frame["driver_id"] != ""]
        return complete_classification_positions(frame)

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
        power_unit_col = first_available(
            results,
            ["PowerUnit", "PowerUnitManufacturer", "EngineManufacturer", "power_unit"],
        )
        grid_col = first_available(results, ["GridPosition", "Grid", "StartingGridPosition", "grid_position"])
        status_col = first_available(results, ["Status", "status", "ResultStatus"])
        laps_col = first_available(results, ["Laps", "laps", "LapsCompleted"])
        frame = pd.DataFrame()
        frame["driver_id"] = results[driver_col].map(self._normalize_driver_id)
        if name_col:
            frame["driver_name"] = results[name_col].fillna(frame["driver_id"]).astype(str)
        else:
            frame["driver_name"] = frame["driver_id"]
        frame["position"] = pd.to_numeric(results[pos_col], errors="coerce")
        if grid_col:
            grid_parsed = results[grid_col].apply(_parse_grid_position_status)
            frame["grid_position"] = grid_parsed.map(lambda item: item[0])
            frame["grid_status"] = grid_parsed.map(lambda item: item[1])
            frame = _assign_pit_lane_grid_positions(frame)
        if team_col:
            frame["team_name"] = results[team_col].astype(str)
        if power_unit_col:
            frame["power_unit"] = results[power_unit_col].astype("string").str.strip()
            frame["power_unit_source"] = "provider_result_field"
        elif "team_name" in frame.columns:
            frame["power_unit"] = frame["team_name"].map(
                lambda value: _season_entry_list_power_unit(int(year), value)
            )
            frame["power_unit_source"] = "causal_season_entry_list_team_mapping"
        if status_col:
            raw_status = results[status_col].astype("string").str.strip()
            valid_status = raw_status.notna() & raw_status.ne("") & ~raw_status.str.lower().isin(
                {"nan", "none", "null", "unknown"}
            )
            frame["race_status_raw"] = raw_status.where(valid_status)
            frame["race_status_evidence_complete"] = valid_status.astype(bool)
        else:
            # A terminal-status target cannot be reconstructed from final
            # position alone; retain an explicit unavailable contract.
            frame["race_status_raw"] = pd.NA
            frame["race_status_evidence_complete"] = False
        if laps_col:
            frame["laps_completed"] = pd.to_numeric(results[laps_col], errors="coerce")
            winner_laps = frame["laps_completed"].max(skipna=True)
            if pd.notna(winner_laps) and float(winner_laps) > 0.0:
                frame["retirement_fraction"] = (
                    frame["laps_completed"] / float(winner_laps)
                ).clip(0.0, 1.0)
            else:
                frame["retirement_fraction"] = float("nan")
        else:
            frame["laps_completed"] = float("nan")
            frame["retirement_fraction"] = float("nan")
        frame = frame[frame["driver_id"] != ""]
        return complete_classification_positions(frame)

    @staticmethod
    def _capture_frame(capture: RaceGridCapture, path: Path) -> pd.DataFrame:
        snapshot = capture.snapshot
        raw_rows = (
            capture.raw_payload.get("grid_rows", [])
            if isinstance(capture.raw_payload, dict)
            else []
        )
        raw_identity = {
            str(row.get("driver_number")): row
            for row in raw_rows
            if isinstance(row, dict) and row.get("driver_number") is not None
        }
        rows: list[dict[str, object]] = []
        for entry in snapshot.entries:
            identity = raw_identity.get(str(entry.driver_id), {})
            rows.append(
                {
                    "driver_id": entry.driver_id,
                    "grid_car_number": str(
                        identity.get("driver_number", entry.driver_id)
                    ),
                    "grid_official_driver_name": identity.get("driver_name"),
                    "grid_position": entry.grid_position,
                    "grid_status": entry.status.value,
                    "grid_starter_eligible": entry.starter_eligible,
                    "grid_pit_lane_start": entry.pit_lane_start,
                    "grid_source": "first_seen_official_grid",
                    "grid_revision_phase": capture.revision_phase.value,
                    "grid_evidence_as_of": capture.first_published_at,
                    "grid_evidence_id": "|".join(snapshot.revision_ids),
                    "grid_evidence_complete": bool(
                        snapshot.available and entry.evidence_complete
                    ),
                    "grid_resolution_status": snapshot.resolution_status.value,
                    "grid_snapshot_available": snapshot.available,
                    "race_information_horizon": snapshot.horizon.value,
                    "grid_capture_id": capture.capture_id,
                    "grid_capture_path": str(path),
                    "grid_captured_at": capture.captured_at,
                    "grid_first_published_at": capture.first_published_at,
                    "grid_race_start_at": capture.race_start_at,
                    "grid_publication_pre_race_verified": (
                        capture.publication_pre_race_verified
                    ),
                    "grid_publication_time_semantics": (
                        capture.publication_time_semantics
                    ),
                    "grid_provider": capture.provider,
                    "grid_source_endpoint": capture.source_endpoint,
                    "grid_source_document_url": capture.source_document_url,
                    "grid_source_document_sha256": capture.source_document_sha256,
                    "grid_raw_payload_sha256": capture.raw_payload_sha256,
                    "grid_penalty_evidence_ids": "|".join(
                        adjustment.evidence_id
                        for adjustment in entry.penalty_evidence
                    ),
                    "grid_penalty_kinds": "|".join(
                        adjustment.kind.value
                        for adjustment in entry.penalty_evidence
                    ),
                    "grid_penalty_reasons": "|".join(
                        adjustment.reason or ""
                        for adjustment in entry.penalty_evidence
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _first_seen_starting_grid(
        self,
        weekend_dir: Path,
        *,
        year: int,
        round_number: int,
        prediction_as_of: Optional[str],
    ) -> tuple[bool, pd.DataFrame]:
        capture_dir = weekend_dir / "first_seen_grid_snapshots"
        paths = sorted(capture_dir.glob("grid_*.json")) if capture_dir.exists() else []
        if not paths:
            return False, pd.DataFrame()
        cutoff = None
        if prediction_as_of is not None:
            cutoff = pd.to_datetime(prediction_as_of, errors="coerce", utc=True)
            if pd.isna(cutoff):
                return True, pd.DataFrame()
        candidates: list[tuple[pd.Timestamp, pd.Timestamp, Path, RaceGridCapture]] = []
        try:
            for path in paths:
                capture = load_race_grid_capture(path)
                if capture.year != int(year) or capture.round_number != int(round_number):
                    raise ValueError("grid capture event identity does not match weekend directory")
                published = pd.to_datetime(
                    capture.first_published_at, errors="coerce", utc=True
                )
                captured = pd.to_datetime(capture.captured_at, errors="coerce", utc=True)
                if pd.isna(published) or pd.isna(captured):
                    raise ValueError("grid capture has invalid causal timestamps")
                if cutoff is not None and published > cutoff:
                    continue
                candidates.append((pd.Timestamp(published), pd.Timestamp(captured), path, capture))
        except (OSError, TypeError, ValueError):
            # Once a first-seen store exists, corrupt provenance must not be
            # bypassed by a retrospective metadata/grid fallback.
            return True, pd.DataFrame()
        if not candidates:
            return True, pd.DataFrame()
        latest_publication = max(item[0] for item in candidates)
        latest = [item for item in candidates if item[0] == latest_publication]
        signatures = {
            (
                item[3].raw_payload_sha256,
                tuple(item[3].snapshot.revision_ids),
                item[3].snapshot.available,
            )
            for item in latest
        }
        if len(signatures) != 1:
            return True, pd.DataFrame()
        # Earliest observation of the same authoritative publication is the
        # immutable first-seen representative; later identical downloads add
        # no information.
        chosen = min(latest, key=lambda item: (item[1], str(item[2])))
        return True, self._capture_frame(chosen[3], chosen[2])

    def get_starting_grid(
        self,
        year: int,
        round_number: int,
        *,
        prediction_as_of: Optional[str] = None,
    ) -> pd.DataFrame:
        weekend = self._weekend_for_round(year, round_number)
        if weekend is None:
            return pd.DataFrame()
        weekend_dir = weekend.get("event_dir")
        if not isinstance(weekend_dir, Path):
            weekend_dir = Path(str(weekend_dir))
        capture_store_exists, captured = self._first_seen_starting_grid(
            weekend_dir,
            year=int(year),
            round_number=int(round_number),
            prediction_as_of=prediction_as_of,
        )
        if capture_store_exists:
            return captured
        meta = self._read_weekend_meta(weekend_dir)
        sessions_raw = meta.get("sessions") if meta else weekend.get("sessions")
        entries = [dict(entry) for entry in sessions_raw if isinstance(entry, dict)] if isinstance(sessions_raw, list) else []
        grid_refs: list[tuple[object, object, bool, dict[str, object]]] = [
            (meta.get("grid_path"), meta.get("grid_availability_phase"), False, meta),
            (meta.get("starting_grid_path"), meta.get("grid_availability_phase"), False, meta),
            (meta.get("pre_race_grid_path"), "pre_race", True, meta),
        ]
        for entry in entries:
            grid_refs.extend(
                [
                    (entry.get("grid_path"), entry.get("grid_availability_phase") or entry.get("availability_phase"), False, entry),
                    (entry.get("starting_grid_path"), entry.get("grid_availability_phase") or entry.get("availability_phase"), False, entry),
                    (entry.get("pre_race_grid_path"), "pre_race", True, entry),
                ],
            )
        cutoff = None
        if prediction_as_of is not None:
            cutoff = pd.to_datetime(prediction_as_of, utc=True, errors="coerce")
            if pd.isna(cutoff):
                return pd.DataFrame()
        candidates: list[tuple[object, int, pd.DataFrame]] = []
        unknown_time_candidates: list[tuple[int, pd.DataFrame]] = []
        for ref_index, (raw_path, phase, trusted_by_key, evidence_owner) in enumerate(grid_refs):
            if raw_path is None:
                continue
            if phase is None and not trusted_by_key:
                continue
            phase_text = str(phase or "pre_race").strip().lower()
            if phase_text in {"post_race", "race_results", "retrospective", "actuals"}:
                continue
            if phase_text not in {"pre_race", "official_pre_race", "starting_grid", "published_pre_race"}:
                continue
            path = self._resolve_session_path(weekend_dir, raw_path)
            if path is None:
                continue
            try:
                grid_raw = pd.read_csv(path)
            except Exception:
                continue
            grid = _standardize_grid_columns(grid_raw, source="pre_race_official_grid")
            if grid.empty:
                continue
            grid["driver_id"] = grid["driver_id"].map(self._normalize_driver_id)
            grid = grid[grid["driver_id"] != ""]
            if grid.empty:
                continue

            def evidence_value(*keys: str) -> object:
                for source in (evidence_owner, meta):
                    for key in keys:
                        value = source.get(key)
                        if value is not None and str(value).strip():
                            return value
                return None

            revision_phase = evidence_value("grid_revision_phase", "official_grid_revision_phase")
            evidence_as_of = evidence_value(
                "grid_evidence_as_of",
                "pre_race_grid_as_of",
                "grid_as_of",
            )
            evidence_id = evidence_value("grid_evidence_id", "grid_revision_id")
            complete_raw = evidence_value("grid_evidence_complete", "grid_complete")
            explicitly_complete = complete_raw is True or str(complete_raw).strip().lower() in {
                "1",
                "true",
                "yes",
            }
            phase_is_final = str(revision_phase or "").strip().lower() in {
                "final",
                "final_pre_race",
            }
            open_adjustment_raw = evidence_value(
                "grid_adjustments_pending",
                "grid_open_adjustments",
                "grid_evidence_superseded",
            )
            has_open_adjustment = open_adjustment_raw is True or str(
                open_adjustment_raw
            ).strip().lower() in {"1", "true", "yes"}
            grid["grid_revision_phase"] = revision_phase
            grid["grid_evidence_as_of"] = evidence_as_of
            grid["grid_evidence_id"] = evidence_id
            grid["grid_evidence_complete"] = bool(
                explicitly_complete
                and phase_is_final
                and evidence_as_of
                and evidence_id
                and not has_open_adjustment
            )
            grid["grid_resolution_status"] = (
                "unvalidated_complete_revision"
                if bool(grid["grid_evidence_complete"].iloc[0])
                else "incomplete_evidence"
            )
            weekend_format = evidence_value("weekend_format_version", "weekend_format")
            if weekend_format is not None:
                grid["weekend_format_version"] = str(weekend_format).strip().lower()
            if prediction_as_of is not None:
                grid["prediction_as_of"] = str(prediction_as_of)

            evidence_timestamp = pd.to_datetime(evidence_as_of, utc=True, errors="coerce")
            if pd.isna(evidence_timestamp):
                grid["grid_evidence_complete"] = False
                grid["grid_resolution_status"] = "incomplete_evidence"
                unknown_time_candidates.append((ref_index, grid))
                continue
            if cutoff is not None and evidence_timestamp > cutoff:
                continue
            candidates.append((evidence_timestamp, ref_index, grid))

        if candidates:
            return max(candidates, key=lambda item: (item[0], item[1]))[2]
        if unknown_time_candidates:
            return max(unknown_time_candidates, key=lambda item: item[0])[1]
        return pd.DataFrame()

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
