import pytest

from f1_platform.schemas import F1Event


def test_f1_event_record_round_trips_full_contract():
    event = F1Event(
        source="openf1",
        topic="v1/laps",
        source_id=123,
        source_key="44:lap:12",
        meeting_key=2026001,
        session_key="session-a",
        driver_number=44,
        event_time="2026-06-25T20:00:00Z",
        received_at="2026-06-25T20:00:03Z",
        payload={"session_key": "session-a", "driver_number": 44, "lap_number": 12},
    )

    loaded = F1Event.from_record(event.to_dict())

    assert loaded == event
    assert loaded.received_at == "2026-06-25T20:00:03Z"


def test_f1_event_record_rejects_missing_payload():
    with pytest.raises(ValueError, match="payload"):
        F1Event.from_record(
            {
                "source": "openf1",
                "topic": "v1/laps",
                "source_id": 123,
                "source_key": "44:lap:12",
                "session_key": "session-a",
                "received_at": "2026-06-25T20:00:03Z",
            }
        )
