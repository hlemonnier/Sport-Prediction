from io import BytesIO
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

from f1_platform.openf1_rest import OpenF1ImportRequest, OpenF1RestClient


def test_openf1_rest_import_builds_deterministic_reducer_events():
    def transport(url: str):
        parsed = urlparse(url)
        endpoint = parsed.path.rsplit("/", 1)[-1]
        query = parse_qs(parsed.query)
        if endpoint == "sessions" and "session_key" in query:
            return [
                {
                    "session_key": 123,
                    "meeting_key": 9,
                    "session_name": "Race",
                    "session_type": "Race",
                    "date_start": "2026-06-01T12:00:00Z",
                    "gmt_offset": "+02:00",
                    "location": "Monaco",
                    "year": 2026,
                    "is_cancelled": False,
                }
            ]
        if endpoint == "drivers":
            return [
                {
                    "session_key": 123,
                    "driver_number": 1,
                    "name_acronym": "VER",
                    "team_name": "Red Bull Racing",
                }
            ]
        if endpoint == "position":
            return [
                {
                    "session_key": 123,
                    "driver_number": 1,
                    "position": 1,
                    "date": "2026-06-01T12:00:01Z",
                }
            ]
        if endpoint == "laps":
            return [
                {
                    "session_key": 123,
                    "driver_number": 1,
                    "lap_number": 2,
                    "lap_duration": 68.4,
                    "date_start": "2026-06-01T12:02:00Z",
                }
            ]
        if endpoint == "overtakes":
            return [
                {
                    "session_key": 123,
                    "overtaking_driver_number": 1,
                    "overtaken_driver_number": 2,
                    "lap_number": 3,
                    "date": "2026-06-01T12:03:00Z",
                }
            ]
        if endpoint == "session_result":
            return [
                {
                    "session_key": 123,
                    "driver_number": 1,
                    "position": 1,
                    "number_of_laps": 57,
                    "duration": 5441.2,
                    "gap_to_leader": 0,
                    "dnf": False,
                    "dns": False,
                    "dsq": False,
                }
            ]
        return []

    client = OpenF1RestClient(transport=transport, request_interval_seconds=0)
    result = client.import_session(
        OpenF1ImportRequest(
            session_key=123,
            topics=("sessions", "drivers", "position", "laps", "overtakes", "session_result"),
        )
    )

    assert result.session_key == 123
    assert result.topic_counts == {
        "sessions": 1,
        "drivers": 1,
        "position": 1,
        "laps": 1,
        "overtakes": 1,
        "session_result": 1,
    }
    assert [event.topic for event in result.events] == [
        "v1/sessions",
        "v1/drivers",
        "v1/position",
        "v1/laps",
        "v1/overtakes",
        "v1/session_result",
    ]
    assert [event.source_id for event in result.events] == [1, 2, 3, 4, 5, 6]
    assert result.events[0].source_key == "session:123"
    assert result.events[1].source_key == "driver:1"
    assert result.events[-2].source_key.startswith("1:overtakes:2:3")
    assert result.events[-1].source_key == "1:session_result"


def test_openf1_rest_resolves_current_live_session():
    def transport(url: str):
        parsed = urlparse(url)
        endpoint = parsed.path.rsplit("/", 1)[-1]
        query = parse_qs(parsed.query)
        if endpoint == "sessions" and query.get("year") == ["2026"]:
            return [
                {
                    "session_key": 101,
                    "meeting_key": 1,
                    "session_name": "Practice 1",
                    "session_type": "Practice",
                    "date_start": "2026-06-26T11:30:00Z",
                    "date_end": "2026-06-26T12:30:00Z",
                    "location": "Spielberg",
                    "year": 2026,
                    "is_cancelled": False,
                },
                {
                    "session_key": 102,
                    "meeting_key": 1,
                    "session_name": "Practice 2",
                    "session_type": "Practice",
                    "date_start": "2026-06-26T15:00:00Z",
                    "date_end": "2026-06-26T16:00:00Z",
                    "location": "Spielberg",
                    "year": 2026,
                    "is_cancelled": False,
                },
            ]
        return []

    client = OpenF1RestClient(transport=transport, request_interval_seconds=0)
    result = client.resolve_live_or_next_session(now="2026-06-26T12:00:00Z", year=2026)

    assert result["status"] == "live"
    assert result["session"]["session_key"] == 101
    assert result["nextSession"]["session_key"] == 102
    assert result["secondsUntilStart"] == 0
    assert result["secondsUntilEnd"] == 1800


def test_openf1_rest_resolves_next_session_when_none_live():
    def transport(url: str):
        parsed = urlparse(url)
        endpoint = parsed.path.rsplit("/", 1)[-1]
        query = parse_qs(parsed.query)
        if endpoint == "sessions" and query.get("year") == ["2026"]:
            return [
                {
                    "session_key": 201,
                    "meeting_key": 8,
                    "session_name": "Practice 1",
                    "session_type": "Practice",
                    "date_start": "2026-06-26T11:30:00Z",
                    "date_end": "2026-06-26T12:30:00Z",
                    "location": "Spielberg",
                    "year": 2026,
                    "is_cancelled": False,
                },
                {
                    "session_key": 202,
                    "meeting_key": 8,
                    "session_name": "Qualifying",
                    "session_type": "Qualifying",
                    "date_start": "2026-06-27T14:00:00Z",
                    "date_end": "2026-06-27T15:00:00Z",
                    "location": "Spielberg",
                    "year": 2026,
                    "is_cancelled": False,
                },
            ]
        return []

    client = OpenF1RestClient(transport=transport, request_interval_seconds=0)
    result = client.resolve_live_or_next_session(now="2026-06-26T13:00:00Z", year=2026)

    assert result["status"] == "upcoming"
    assert result["session"] is None
    assert result["nextSession"]["session_key"] == 202
    assert result["secondsUntilStart"] == 90000


def test_openf1_rest_treats_openf1_no_results_404_as_empty_schedule_page():
    def transport(url: str):
        parsed = urlparse(url)
        endpoint = parsed.path.rsplit("/", 1)[-1]
        query = parse_qs(parsed.query)
        if endpoint == "sessions" and query.get("year") == ["2026"]:
            return [
                {
                    "session_key": 301,
                    "meeting_key": 9,
                    "session_name": "Practice 1",
                    "session_type": "Practice",
                    "date_start": "2026-06-26T11:30:00Z",
                    "date_end": "2026-06-26T12:30:00Z",
                    "location": "Spielberg",
                    "year": 2026,
                    "is_cancelled": False,
                },
            ]
        if endpoint == "sessions" and query.get("year") == ["2027"]:
            raise HTTPError(
                url,
                404,
                "Not Found",
                hdrs=None,
                fp=BytesIO(b'{"detail":"No results found."}'),
            )
        return []

    client = OpenF1RestClient(transport=transport, request_interval_seconds=0)
    result = client.resolve_live_or_next_session(now="2026-06-26T13:00:00Z")

    assert result["status"] == "unavailable"
    assert result["session"] is None
    assert result["nextSession"] is None
    assert result["secondsUntilStart"] is None
    assert "2026, 2027" in result["message"]
