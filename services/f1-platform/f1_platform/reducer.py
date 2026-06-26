"""OpenF1-style event reducer for live and replay sessions."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from .schemas import (
    CustomMicroSectorPassage,
    DriverState,
    F1Event,
    JsonObject,
    LapPoint,
    SessionSnapshot,
    StateUpdate,
    StintSegment,
)
from .time import utc_now_iso
from .track_geometry import TrackProjectionProvider

CUSTOM_MICRO_SECTOR_COUNT = 25


class F1StateReducer:
    """Reduce raw OpenF1 events into compact session state.

    OpenF1 live messages can update the same logical object multiple times. The
    contract here is one accepted state per `(topic, source_key)` identity, with
    the highest `source_id` winning.
    """

    def __init__(
        self,
        session_key: int | str,
        *,
        source: str = "openf1-replay",
        track_projector: TrackProjectionProvider | None = None,
    ) -> None:
        self.session_key = session_key
        self.source = source
        self.track_projector = track_projector
        self.seq = 0
        self.events_by_identity: dict[tuple[str, str], F1Event] = {}
        self.session_info: JsonObject | None = None
        self.drivers: dict[int, DriverState] = {}
        self.lap_points_by_key: dict[tuple[int, int], LapPoint] = {}
        self.stints_by_key: dict[tuple[int, int], StintSegment] = {}
        self.race_control: list[JsonObject] = []
        self.pit_stops_by_key: dict[str, JsonObject] = {}
        self.overtakes: list[JsonObject] = []
        self.session_results_by_driver: dict[int, JsonObject] = {}
        self.custom_micro_sectors_by_key: dict[tuple[int, int | None, int], CustomMicroSectorPassage] = {}
        self._micro_sector_entries: dict[int, JsonObject] = {}
        self._micro_sector_personal_bests: dict[tuple[int, int], float] = {}
        self._micro_sector_session_bests: dict[int, float] = {}
        self._latest_micro_sector_by_driver: dict[tuple[int, int], CustomMicroSectorPassage] = {}
        self.weather_samples_by_key: dict[str, JsonObject] = {}
        self.weather: JsonObject | None = None
        self.topic_watermarks: dict[str, int] = defaultdict(int)
        self.replay_meta: JsonObject = {"mode": "sample", "eventCount": 0}

    def ingest(self, event: F1Event) -> StateUpdate | None:
        identity = event.identity()
        existing = self.events_by_identity.get(identity)
        if existing is not None and existing.source_id >= event.source_id:
            return None

        self.events_by_identity[identity] = event
        self.seq += 1
        self.topic_watermarks[event.topic] = max(self.topic_watermarks[event.topic], event.source_id)
        self.replay_meta["eventCount"] = int(self.replay_meta.get("eventCount", 0)) + 1

        update_type = self._apply_event(event)
        payload = self._compact_payload(event)
        return StateUpdate(
            seq=self.seq,
            type=update_type,
            event_time=event.event_time,
            driver_number=event.driver_number,
            payload=payload,
        )

    def snapshot(self) -> SessionSnapshot:
        drivers = sorted(
            self.drivers.values(),
            key=lambda driver: (
                driver.position if driver.position is not None else 10_000,
                driver.driver_number,
            ),
        )
        lap_points = sorted(
            self.lap_points_by_key.values(),
            key=lambda point: (point.lap, point.driver_number),
        )
        stints = sorted(
            self.stints_by_key.values(),
            key=lambda segment: (segment.driver_number, segment.stint_number),
        )
        pit_stops = sorted(
            self.pit_stops_by_key.values(),
            key=lambda stop: (
                _optional_int(stop.get("lap_number")) or 10_000,
                _optional_int(stop.get("driver_number")) or 10_000,
                str(stop.get("event_time") or stop.get("date") or ""),
            ),
        )
        session_results = sorted(
            self.session_results_by_driver.values(),
            key=lambda result: (
                _optional_int(result.get("position")) or 10_000,
                _optional_int(result.get("driver_number")) or 10_000,
            ),
        )
        custom_micro_sectors = sorted(
            self.custom_micro_sectors_by_key.values(),
            key=lambda passage: (
                passage.seq,
                passage.driver_number,
                passage.sector_index,
            ),
        )
        weather_samples = sorted(
            self.weather_samples_by_key.values(),
            key=lambda sample: (
                str(sample.get("event_time") or sample.get("date") or ""),
                _optional_int(sample.get("source_id")) or 0,
                str(sample.get("source_key") or ""),
            ),
        )
        return SessionSnapshot(
            session_key=self.session_key,
            seq=self.seq,
            generated_at=utc_now_iso(),
            source=self.source,
            drivers=drivers,
            session_info=dict(self.session_info) if self.session_info else None,
            lap_chart=lap_points,
            strategy_timeline=stints,
            race_control=list(self.race_control[-30:]),
            pit_stops=pit_stops[-120:],
            overtakes=list(self.overtakes[-80:]),
            session_results=session_results,
            custom_micro_sectors=custom_micro_sectors[-300:],
            weather=dict(self.weather) if self.weather else None,
            weather_samples=weather_samples[-240:],
            predictions=[],
            topic_watermarks=dict(self.topic_watermarks),
            replay=dict(self.replay_meta),
        )

    def _driver(self, driver_number: int | None) -> DriverState | None:
        if driver_number is None:
            return None
        if driver_number not in self.drivers:
            self.drivers[driver_number] = DriverState(driver_number=driver_number)
        return self.drivers[driver_number]

    def _apply_event(self, event: F1Event) -> str:
        topic = event.topic.removeprefix("v1/")
        if topic == "sessions":
            return self._apply_session(event)
        if topic == "drivers":
            return self._apply_driver(event)
        if topic == "laps":
            return self._apply_lap(event)
        if topic == "position":
            return self._apply_position(event)
        if topic == "intervals":
            return self._apply_interval(event)
        if topic == "stints":
            return self._apply_stint(event)
        if topic == "pit":
            return self._apply_pit(event)
        if topic == "race_control":
            return self._apply_race_control(event)
        if topic == "weather":
            return self._apply_weather(event)
        if topic == "car_data":
            return self._apply_car_data(event)
        if topic == "location":
            return self._apply_location(event)
        if topic == "overtakes":
            return self._apply_overtake(event)
        if topic in {"session_result", "results"}:
            return self._apply_session_result(event)
        return f"{topic}.updated"

    def _apply_session(self, event: F1Event) -> str:
        payload = _json_safe(event.payload)
        session_info = {
            "session_key": payload.get("session_key", self.session_key),
            "meeting_key": _optional_int(payload.get("meeting_key", event.meeting_key)),
            "session_name": _first_text(payload, "session_name", "name"),
            "session_type": _first_text(payload, "session_type", "type"),
            "date_start": _first_text(payload, "date_start", "date"),
            "date_end": _first_text(payload, "date_end"),
            "gmt_offset": _first_text(payload, "gmt_offset"),
            "location": _first_text(payload, "location"),
            "year": _optional_int(payload.get("year")),
            "is_cancelled": _optional_bool(payload.get("is_cancelled")),
            "source_id": event.source_id,
            "source_key": event.source_key,
            "event_time": event.event_time or _first_text(payload, "date_start", "date"),
            "raw": payload,
        }
        self.session_info = {key: value for key, value in session_info.items() if value is not None}
        self.replay_meta["sessionKey"] = self.session_info.get("session_key", self.session_key)
        if self.session_info.get("meeting_key") is not None:
            self.replay_meta["meetingKey"] = self.session_info["meeting_key"]
        if self.session_info.get("session_name"):
            self.replay_meta["sessionName"] = self.session_info["session_name"]
        if self.session_info.get("session_type"):
            self.replay_meta["sessionType"] = self.session_info["session_type"]
        if self.session_info.get("year") is not None:
            self.replay_meta["year"] = self.session_info["year"]
        return "session.updated"

    def _apply_driver(self, event: F1Event) -> str:
        driver = self._driver(event.driver_number)
        if driver is None:
            return "driver.ignored"
        payload = event.payload
        driver.acronym = _first_text(payload, "name_acronym", "acronym", "broadcast_name") or driver.acronym
        driver.full_name = _first_text(payload, "full_name", "driver_name", "broadcast_name") or driver.full_name
        driver.team_name = _first_text(payload, "team_name", "team") or driver.team_name
        driver.team_colour = _first_text(payload, "team_colour", "team_color") or driver.team_colour
        driver.last_update_seq = self.seq
        return "driver.updated"

    def _apply_lap(self, event: F1Event) -> str:
        driver = self._driver(event.driver_number)
        if driver is None:
            return "lap.ignored"
        payload = event.payload
        lap_number = _optional_int(payload.get("lap_number"))
        lap_time = _seconds(payload.get("lap_duration", payload.get("lap_time")))
        if lap_number is not None:
            driver.current_lap = max(driver.current_lap or 0, lap_number)
        if lap_time is not None:
            driver.last_lap_time = lap_time
            if driver.best_lap_time is None or lap_time < driver.best_lap_time:
                driver.best_lap_time = lap_time
            if lap_number is not None:
                self.lap_points_by_key[(driver.driver_number, lap_number)] = LapPoint(
                    lap=lap_number,
                    driver_number=driver.driver_number,
                    value=lap_time,
                )
        has_sector_update = False
        for key in ("duration_sector_1", "duration_sector_2", "duration_sector_3"):
            value = _seconds(payload.get(key))
            if value is not None:
                driver.sector_times[key.replace("duration_", "")] = value
                has_sector_update = True
        driver.last_update_seq = self.seq
        if lap_time is not None:
            return "lap.updated"
        if lap_number is not None or has_sector_update:
            return "lap.partial"
        return "lap.ignored"

    def _apply_position(self, event: F1Event) -> str:
        driver = self._driver(event.driver_number)
        if driver is None:
            return "position.ignored"
        driver.position = _optional_int(event.payload.get("position")) or driver.position
        driver.last_update_seq = self.seq
        return "position.updated"

    def _apply_interval(self, event: F1Event) -> str:
        driver = self._driver(event.driver_number)
        if driver is None:
            return "interval.ignored"
        payload = event.payload
        driver.interval = _first_text(payload, "interval", "interval_to_position_ahead") or driver.interval
        driver.gap_to_leader = _first_text(payload, "gap_to_leader") or driver.gap_to_leader
        driver.last_update_seq = self.seq
        return "interval.updated"

    def _apply_stint(self, event: F1Event) -> str:
        driver = self._driver(event.driver_number)
        if driver is None:
            return "stint.ignored"
        payload = event.payload
        stint_number = _optional_int(payload.get("stint_number")) or driver.stint_number or 1
        compound = _first_text(payload, "compound", "tyre_compound") or driver.current_compound or "UNKNOWN"
        lap_start = _optional_int(payload.get("lap_start", payload.get("start_lap"))) or driver.current_lap or 1
        lap_end = _optional_int(payload.get("lap_end", payload.get("end_lap")))
        tyre_age_start = _optional_int(payload.get("tyre_age_at_start", payload.get("tyre_age_start"))) or 0
        if driver.current_lap is not None:
            driver.tyre_age = tyre_age_start + max(0, driver.current_lap - lap_start)
        else:
            driver.tyre_age = tyre_age_start
        driver.current_compound = compound
        driver.stint_number = stint_number
        self.stints_by_key[(driver.driver_number, stint_number)] = StintSegment(
            driver_number=driver.driver_number,
            stint_number=stint_number,
            compound=compound,
            start_lap=lap_start,
            end_lap=lap_end,
            tyre_age_start=tyre_age_start,
        )
        driver.last_update_seq = self.seq
        return "stint.updated"

    def _apply_pit(self, event: F1Event) -> str:
        driver = self._driver(event.driver_number)
        if driver is None:
            return "pit.ignored"
        payload = event.payload
        lap_number = _optional_int(payload.get("lap_number"))
        pit_duration = _seconds(payload.get("pit_duration"))
        pit_stop = _json_safe(payload)
        pit_stop["driver_number"] = driver.driver_number
        if lap_number is not None:
            pit_stop["lap_number"] = lap_number
        if pit_duration is not None:
            pit_stop["pit_duration"] = pit_duration
        pit_stop["source_id"] = event.source_id
        pit_stop["source_key"] = event.source_key
        pit_stop["event_time"] = event.event_time
        if pit_duration is None:
            driver.pit_status = f"pit lap {lap_number}" if lap_number else "pit"
        else:
            driver.pit_status = f"{pit_duration:.1f}s stop"
        pit_stop["pit_status"] = driver.pit_status
        self.pit_stops_by_key[event.source_key] = pit_stop
        driver.last_update_seq = self.seq
        return "pit.updated"

    def _apply_race_control(self, event: F1Event) -> str:
        payload = _json_safe(event.payload)
        status = _race_status(payload)
        if status is not None:
            payload["track_status"] = status
            self.replay_meta["trackStatus"] = status
            self.replay_meta["trackStatusSeq"] = self.seq
            for driver in self.drivers.values():
                driver.track_status = status
                driver.last_update_seq = self.seq
        self.race_control.append(payload)
        return "race_control.updated"

    def _apply_weather(self, event: F1Event) -> str:
        payload = _json_safe(event.payload)
        sample = {
            **payload,
            "source_id": event.source_id,
            "source_key": event.source_key,
            "event_time": event.event_time or _first_text(payload, "date"),
        }
        self.weather_samples_by_key[event.source_key] = sample
        self.weather = sample
        self.replay_meta["weatherSampleCount"] = len(self.weather_samples_by_key)
        return "weather.updated"

    def _apply_car_data(self, event: F1Event) -> str:
        driver = self._driver(event.driver_number)
        if driver is None:
            return "car_data.ignored"
        driver.last_speed = _seconds(event.payload.get("speed")) or driver.last_speed
        driver.drs = _optional_int(event.payload.get("drs")) or driver.drs
        driver.last_update_seq = self.seq
        return "car_data.updated"

    def _apply_location(self, event: F1Event) -> str:
        driver = self._driver(event.driver_number)
        if driver is None:
            return "location.ignored"
        x = _seconds(event.payload.get("x"))
        y = _seconds(event.payload.get("y"))
        z = _seconds(event.payload.get("z"))
        location = {}
        if x is not None:
            location["x"] = x
        if y is not None:
            location["y"] = y
        if z is not None:
            location["z"] = z
        if location:
            driver.last_location = location
            projection = self.track_projector.project(self.session_key, location) if self.track_projector else None
            if projection is not None:
                driver.track_progress = round(projection.progress, 9)
                driver.last_location.update(
                    {
                        "projected_distance": round(projection.distance, 6),
                        "projected_x": round(projection.x, 6),
                        "projected_y": round(projection.y, 6),
                        "projected_z": round(projection.z, 6),
                        "projection_error": round(projection.error, 6),
                    }
                )
                self.replay_meta["trackProjection"] = projection.source
            else:
                driver.track_progress = ((abs(location.get("x", 0.0)) + abs(location.get("y", 0.0))) % 10_000.0) / 10_000.0
                self.replay_meta.setdefault("trackProjection", "coordinate-fallback")
            self._update_custom_micro_sector(event, driver)
        driver.last_update_seq = self.seq
        return "location.updated"

    def _update_custom_micro_sector(self, event: F1Event, driver: DriverState) -> None:
        progress = driver.track_progress
        if progress is None:
            return

        timestamp = _timestamp_seconds(event.event_time or event.received_at)
        sector_index = _micro_sector_index(progress, CUSTOM_MICRO_SECTOR_COUNT)
        existing = self._micro_sector_entries.get(driver.driver_number)
        if existing is None:
            self._micro_sector_entries[driver.driver_number] = {
                "sector_index": sector_index,
                "timestamp": timestamp,
                "event_time": event.event_time,
                "lap": driver.current_lap,
            }
            return
        if existing.get("sector_index") == sector_index:
            return

        previous_sector = _optional_int(existing.get("sector_index"))
        previous_timestamp = _optional_float(existing.get("timestamp"))
        if previous_sector is None or previous_timestamp is None:
            self._micro_sector_entries[driver.driver_number] = {
                "sector_index": sector_index,
                "timestamp": timestamp,
                "event_time": event.event_time,
                "lap": driver.current_lap,
            }
            return
        passage_time = timestamp - previous_timestamp
        if passage_time <= 0:
            self._micro_sector_entries[driver.driver_number] = {
                "sector_index": sector_index,
                "timestamp": timestamp,
                "event_time": event.event_time,
                "lap": driver.current_lap,
            }
            return

        lap = _optional_int(existing.get("lap")) or driver.current_lap
        personal_key = (driver.driver_number, previous_sector)
        personal_best = self._micro_sector_personal_bests.get(personal_key)
        session_best = self._micro_sector_session_bests.get(previous_sector)
        car_ahead = self._car_ahead(driver)
        teammate = self._nearest_teammate(driver)
        car_ahead_passage = self._latest_micro_sector_by_driver.get((car_ahead.driver_number, previous_sector)) if car_ahead else None
        teammate_passage = self._latest_micro_sector_by_driver.get((teammate.driver_number, previous_sector)) if teammate else None

        passage = CustomMicroSectorPassage(
            driver_number=driver.driver_number,
            lap=lap,
            sector_index=previous_sector,
            sector_count=CUSTOM_MICRO_SECTOR_COUNT,
            progress_start=round((previous_sector - 1) / CUSTOM_MICRO_SECTOR_COUNT, 6),
            progress_end=round(previous_sector / CUSTOM_MICRO_SECTOR_COUNT, 6),
            passage_time=round(passage_time, 3),
            personal_best_delta=_rounded_delta(passage_time, personal_best),
            session_best_delta=_rounded_delta(passage_time, session_best),
            car_ahead_delta=_rounded_delta(passage_time, car_ahead_passage.passage_time if car_ahead_passage else None),
            teammate_delta=_rounded_delta(passage_time, teammate_passage.passage_time if teammate_passage else None),
            label="custom micro-sector",
            source=str(self.replay_meta.get("trackProjection", "track-progress")),
            event_time=event.event_time,
            seq=self.seq,
        )
        self.custom_micro_sectors_by_key[(driver.driver_number, lap, previous_sector)] = passage
        self._latest_micro_sector_by_driver[(driver.driver_number, previous_sector)] = passage
        if personal_best is None or passage_time < personal_best:
            self._micro_sector_personal_bests[personal_key] = passage_time
        if session_best is None or passage_time < session_best:
            self._micro_sector_session_bests[previous_sector] = passage_time
        self.replay_meta["customMicroSectorCount"] = CUSTOM_MICRO_SECTOR_COUNT
        self.replay_meta["customMicroSectorSource"] = passage.source
        self._micro_sector_entries[driver.driver_number] = {
            "sector_index": sector_index,
            "timestamp": timestamp,
            "event_time": event.event_time,
            "lap": driver.current_lap,
        }

    def _car_ahead(self, driver: DriverState) -> DriverState | None:
        if driver.position is None or driver.position <= 1:
            return None
        target_position = driver.position - 1
        for candidate in self.drivers.values():
            if candidate.position == target_position:
                return candidate
        return None

    def _nearest_teammate(self, driver: DriverState) -> DriverState | None:
        if not driver.team_name:
            return None
        teammates = [
            candidate
            for candidate in self.drivers.values()
            if candidate.driver_number != driver.driver_number and candidate.team_name == driver.team_name
        ]
        if not teammates:
            return None
        return sorted(
            teammates,
            key=lambda candidate: (
                abs((candidate.position or 10_000) - (driver.position or 10_000)),
                candidate.driver_number,
            ),
        )[0]

    def _apply_overtake(self, event: F1Event) -> str:
        payload = _json_safe(event.payload)
        overtaking_driver = _optional_int(
            payload.get(
                "overtaking_driver_number",
                payload.get("driver_overtaking", payload.get("driver_number_overtaking", event.driver_number)),
            )
        )
        overtaken_driver = _optional_int(
            payload.get(
                "overtaken_driver_number",
                payload.get("driver_overtaken", payload.get("driver_number_overtaken")),
            )
        )
        lap_number = _optional_int(payload.get("lap_number"))
        if overtaking_driver is not None:
            payload["overtaking_driver_number"] = overtaking_driver
            driver = self._driver(overtaking_driver)
            if driver is not None:
                driver.last_update_seq = self.seq
        if overtaken_driver is not None:
            payload["overtaken_driver_number"] = overtaken_driver
        if lap_number is not None:
            payload["lap_number"] = lap_number
        payload["source_id"] = event.source_id
        payload["source_key"] = event.source_key
        payload["event_time"] = event.event_time
        self.overtakes.append(payload)
        return "overtake.updated"

    def _apply_session_result(self, event: F1Event) -> str:
        driver = self._driver(event.driver_number)
        if driver is None:
            return "session_result.ignored"
        payload = _json_safe(event.payload)
        position = _optional_int(payload.get("position"))
        number_of_laps = _optional_int(payload.get("number_of_laps"))
        duration = _seconds(payload.get("duration"))
        gap = payload.get("gap_to_leader")

        payload["driver_number"] = driver.driver_number
        if position is not None:
            payload["position"] = position
            driver.position = position
        if number_of_laps is not None:
            payload["number_of_laps"] = number_of_laps
            driver.current_lap = max(driver.current_lap or 0, number_of_laps)
        if duration is not None:
            payload["duration"] = duration
        if gap is not None:
            driver.gap_to_leader = str(gap)
        payload["source_id"] = event.source_id
        payload["source_key"] = event.source_key
        payload["event_time"] = event.event_time

        self.session_results_by_driver[driver.driver_number] = payload
        driver.last_update_seq = self.seq
        return "session_result.updated"

    def _compact_payload(self, event: F1Event) -> JsonObject:
        keys = (
            "position",
            "gap_to_leader",
            "interval",
            "lap_number",
            "lap_duration",
            "lap_time",
            "duration_sector_1",
            "duration_sector_2",
            "duration_sector_3",
            "compound",
            "tyre_age_at_start",
            "pit_duration",
            "speed",
            "message",
            "overtaking_driver_number",
            "overtaken_driver_number",
            "duration",
            "number_of_laps",
            "session_name",
            "session_type",
            "date_start",
            "date_end",
            "gmt_offset",
            "location",
            "year",
            "is_cancelled",
            "dnf",
            "dns",
            "dsq",
        )
        compact = {key: event.payload[key] for key in keys if key in event.payload}
        compact["topic"] = event.topic
        compact["sourceId"] = event.source_id
        compact["sourceKey"] = event.source_key
        return compact


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _seconds(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _optional_float(value: Any) -> float | None:
    return _seconds(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _first_text(payload: JsonObject, *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _json_safe(payload: JsonObject) -> JsonObject:
    return {str(key): value for key, value in payload.items()}


def _micro_sector_index(progress: float, sector_count: int) -> int:
    if progress < 0:
        progress = 0.0
    if progress > 1:
        progress = 1.0
    if progress >= 1.0:
        return sector_count
    return int(progress * sector_count) + 1


def _timestamp_seconds(value: Any) -> float:
    if value is None or value == "":
        return datetime.now(timezone.utc).timestamp()
    text = str(value)
    try:
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return datetime.now(timezone.utc).timestamp()


def _rounded_delta(value: float, reference: float | None) -> float | None:
    if reference is None:
        return None
    return round(value - reference, 3)


def _race_status(payload: JsonObject) -> str | None:
    text = " ".join(
        str(payload.get(key, ""))
        for key in (
            "flag",
            "category",
            "scope",
            "status",
            "message",
            "reason",
        )
    ).lower()
    if not text.strip():
        return None
    if "red" in text or "suspended" in text:
        return "red_flag"
    if "safety car" in text and "virtual" not in text:
        if "ending" in text or "in this lap" in text:
            return "safety_car_ending"
        return "safety_car"
    if "virtual safety car" in text or " vsc" in f" {text}":
        if "ending" in text:
            return "vsc_ending"
        return "vsc"
    if "yellow" in text:
        return "yellow"
    if "green" in text or "track clear" in text or "clear" in text:
        return "green"
    if "restart" in text or "resume" in text or "resumed" in text:
        return "restarted"
    return None
