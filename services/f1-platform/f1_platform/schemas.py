"""Stable contracts for the F1 platform service.

The reducer core intentionally uses dataclasses instead of framework models so
it can be tested without FastAPI/Pydantic imports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


JsonObject = dict[str, Any]


@dataclass(slots=True)
class F1Event:
    source: str
    topic: str
    source_id: int
    source_key: str
    meeting_key: int | None
    session_key: int | str
    driver_number: int | None
    event_time: str | None
    received_at: str
    payload: JsonObject

    @classmethod
    def from_payload(
        cls,
        raw: JsonObject,
        *,
        topic: str | None = None,
        source: str = "openf1",
        received_at: str,
        session_key: int | str | None = None,
    ) -> "F1Event":
        payload = dict(raw.get("payload", raw))
        event_topic = str(topic or raw.get("topic") or payload.get("topic") or "unknown")
        event_session_key = raw.get("session_key", payload.get("session_key", session_key))
        if event_session_key is None:
            raise ValueError("session_key is required")

        source_id = raw.get("_id", raw.get("source_id", payload.get("_id", payload.get("id", 0))))
        source_key = raw.get("_key", raw.get("source_key", payload.get("_key")))
        if not source_key:
            source_key = _derive_source_key(event_topic, payload)

        driver_number = raw.get("driver_number", payload.get("driver_number"))
        return cls(
            source=str(raw.get("source", source)),
            topic=event_topic,
            source_id=int(source_id),
            source_key=str(source_key),
            meeting_key=_optional_int(raw.get("meeting_key", payload.get("meeting_key"))),
            session_key=event_session_key,
            driver_number=_optional_int(driver_number),
            event_time=_optional_str(raw.get("date", raw.get("event_time", payload.get("date")))),
            received_at=received_at,
            payload=payload,
        )

    @classmethod
    def from_record(cls, raw: JsonObject) -> "F1Event":
        """Rehydrate an event previously emitted by ``to_dict``."""

        if not isinstance(raw, dict):
            raise ValueError("event record must be a JSON object")
        payload = raw.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("event record payload must be a JSON object")
        session_key = raw.get("session_key")
        if session_key is None:
            raise ValueError("event record session_key is required")
        return cls(
            source=_required_str(raw, "source"),
            topic=_required_str(raw, "topic"),
            source_id=_required_int(raw.get("source_id"), "source_id"),
            source_key=_required_str(raw, "source_key"),
            meeting_key=_optional_int(raw.get("meeting_key")),
            session_key=session_key,
            driver_number=_optional_int(raw.get("driver_number")),
            event_time=_optional_str(raw.get("event_time")),
            received_at=_required_str(raw, "received_at"),
            payload=dict(payload),
        )

    def identity(self) -> tuple[str, str]:
        return (self.topic, self.source_key)

    def to_dict(self) -> JsonObject:
        return asdict(self)


@dataclass(slots=True)
class DriverState:
    driver_number: int
    acronym: str | None = None
    full_name: str | None = None
    team_name: str | None = None
    team_colour: str | None = None
    position: int | None = None
    interval: str | None = None
    gap_to_leader: str | None = None
    current_lap: int | None = None
    last_lap_time: float | None = None
    best_lap_time: float | None = None
    sector_times: dict[str, float | None] = field(default_factory=dict)
    current_compound: str | None = None
    tyre_age: int | None = None
    stint_number: int | None = None
    pit_status: str | None = None
    track_status: str | None = None
    last_speed: float | None = None
    last_location: dict[str, float] | None = None
    track_progress: float | None = None
    drs: int | None = None
    last_update_seq: int = 0

    def to_dict(self) -> JsonObject:
        return asdict(self)


@dataclass(slots=True)
class LapPoint:
    lap: int
    driver_number: int
    value: float

    def to_dict(self) -> JsonObject:
        return asdict(self)


@dataclass(slots=True)
class StintSegment:
    driver_number: int
    stint_number: int
    compound: str
    start_lap: int
    end_lap: int | None
    tyre_age_start: int

    def to_dict(self) -> JsonObject:
        return asdict(self)


@dataclass(slots=True)
class PredictionSnapshot:
    model_version: str
    prediction_time: str
    source_event_sequence: int
    features_version: str
    driver_number: int
    expected_position: float | None
    position_distribution: dict[str, float]
    win_probability: float
    podium_probability: float
    points_probability: float
    dnf_probability: float
    confidence: float
    position_p10: float | None = None
    position_p90: float | None = None
    prediction_kind: str = "race"
    position_semantics: str = "race_finish_order"
    strategy: JsonObject | None = None
    forecast_available: bool = True
    unavailable_reason: str | None = None
    eligibility_status: str = "classification_eligible"
    participation_status: str = "running_or_unknown"

    def to_dict(self) -> JsonObject:
        return asdict(self)


@dataclass(slots=True)
class CustomMicroSectorPassage:
    driver_number: int
    lap: int | None
    sector_index: int
    sector_count: int
    progress_start: float
    progress_end: float
    passage_time: float
    personal_best_delta: float | None
    session_best_delta: float | None
    car_ahead_delta: float | None
    teammate_delta: float | None
    label: str
    source: str
    event_time: str | None
    seq: int

    def to_dict(self) -> JsonObject:
        return asdict(self)


@dataclass(slots=True)
class SessionSnapshot:
    session_key: int | str
    seq: int
    generated_at: str
    source: str
    drivers: list[DriverState]
    session_info: JsonObject | None = None
    lap_chart: list[LapPoint] = field(default_factory=list)
    strategy_timeline: list[StintSegment] = field(default_factory=list)
    race_control: list[JsonObject] = field(default_factory=list)
    pit_stops: list[JsonObject] = field(default_factory=list)
    overtakes: list[JsonObject] = field(default_factory=list)
    session_results: list[JsonObject] = field(default_factory=list)
    custom_micro_sectors: list[CustomMicroSectorPassage] = field(default_factory=list)
    weather: JsonObject | None = None
    weather_samples: list[JsonObject] = field(default_factory=list)
    predictions: list[PredictionSnapshot] = field(default_factory=list)
    topic_watermarks: dict[str, int] = field(default_factory=dict)
    replay: JsonObject = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        return {
            "sessionKey": self.session_key,
            "seq": self.seq,
            "generatedAt": self.generated_at,
            "source": self.source,
            "sessionInfo": self.session_info,
            "drivers": [driver.to_dict() for driver in self.drivers],
            "lapChart": [point.to_dict() for point in self.lap_chart],
            "strategyTimeline": [segment.to_dict() for segment in self.strategy_timeline],
            "raceControl": self.race_control,
            "pitStops": self.pit_stops,
            "overtakes": self.overtakes,
            "sessionResults": self.session_results,
            "customMicroSectors": [sector.to_dict() for sector in self.custom_micro_sectors],
            "weather": self.weather,
            "weatherSamples": self.weather_samples,
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "topicWatermarks": dict(self.topic_watermarks),
            "replay": dict(self.replay),
        }


@dataclass(slots=True)
class StateUpdate:
    seq: int
    type: str
    event_time: str | None
    driver_number: int | None
    payload: JsonObject

    def to_dict(self) -> JsonObject:
        return {
            "seq": self.seq,
            "type": self.type,
            "eventTime": self.event_time,
            "driverNumber": self.driver_number,
            "payload": self.payload,
        }


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _required_str(payload: JsonObject, key: str) -> str:
    value = payload.get(key)
    if value is None or value == "":
        raise ValueError(f"event record {key} is required")
    return str(value)


def _required_int(value: Any, key: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"event record {key} must be an integer") from exc


def _derive_source_key(topic: str, payload: JsonObject) -> str:
    driver = payload.get("driver_number", "session")
    if topic.endswith("sessions") and payload.get("session_key") is not None:
        return f"session:{payload['session_key']}"
    if "lap_number" in payload:
        return f"{driver}:lap:{payload['lap_number']}"
    if "stint_number" in payload:
        return f"{driver}:stint:{payload['stint_number']}"
    if "date" in payload:
        return f"{driver}:{payload['date']}"
    if "date_start" in payload:
        return f"{driver}:{payload['date_start']}"
    return f"{driver}:{topic}"
