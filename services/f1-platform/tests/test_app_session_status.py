import asyncio

import pytest


def _session_status_route(app):
    return next(
        route
        for route in app.routes
        if getattr(route, "path", "") == "/api/f1/session-status"
        and "GET" in getattr(route, "methods", set())
    )


def _fastf1_live_payload():
    return {
        "status": "live",
        "source": "fastf1-schedule",
        "resolvedAt": "2026-06-27T10:40:00Z",
        "message": "FastF1 schedule reports an ongoing session.",
        "session": {
            "session_key": "fastf1:2026:austrian-grand-prix:fp3",
            "meeting_key": 8,
            "session_name": "Practice 3",
            "date_start": "2026-06-27T10:30:00Z",
            "date_end": "2026-06-27T11:30:00Z",
        },
        "nextSession": None,
        "secondsUntilStart": 0,
        "secondsUntilEnd": 3000,
    }


class _FastF1Schedule:
    def __init__(self, payload):
        self.payload = payload

    def resolve_live_or_next_session(self, *, year=None, now=None):
        return dict(self.payload)


class _OpenF1LiveClient:
    def resolve_live_or_next_session(self, *, year=None, now=None):
        return {
            "status": "live",
            "source": "openf1-rest",
            "resolvedAt": "2026-06-27T10:40:00Z",
            "message": "OpenF1 reports an ongoing session.",
            "session": {
                "session_key": 10008,
                "meeting_key": 8,
                "session_name": "Practice 3",
                "date_start": "2026-06-27T10:30:00Z",
                "date_end": "2026-06-27T11:30:00Z",
            },
            "nextSession": None,
            "secondsUntilStart": 0,
            "secondsUntilEnd": 3000,
        }


class _OpenF1UnavailableClient:
    def resolve_live_or_next_session(self, *, year=None, now=None):
        raise RuntimeError("OpenF1 REST API requires authentication")


def _create_test_app(monkeypatch, tmp_path, openf1_client):
    try:
        from f1_platform.app import create_app
        from f1_platform.openf1_rest import OpenF1RestClient
    except ImportError as exc:
        pytest.skip(f"FastAPI unavailable: {exc}")

    monkeypatch.setenv("F1_PLATFORM_EVENT_STORE", str(tmp_path / "events"))
    monkeypatch.setenv("F1_PLATFORM_SQLITE_PROJECTION_STORE", str(tmp_path / "projection.sqlite"))
    monkeypatch.setenv("F1_PLATFORM_FASTF1_ARTIFACT_STORE", str(tmp_path / "artifacts"))
    monkeypatch.delenv("F1_PLATFORM_REDIS_URL", raising=False)
    monkeypatch.setattr(OpenF1RestClient, "from_env", classmethod(lambda cls: openf1_client))
    app = create_app()
    app.state.fastf1_schedule = _FastF1Schedule(_fastf1_live_payload())
    return app


def test_session_status_prefers_openf1_when_live_timing_is_available(monkeypatch, tmp_path):
    app = _create_test_app(monkeypatch, tmp_path, _OpenF1LiveClient())
    route = _session_status_route(app)

    response = asyncio.run(route.endpoint(year=2026, now="2026-06-27T10:40:00Z"))

    assert response["source"] == "openf1-rest"
    assert response["session"]["session_key"] == 10008
    assert response["scheduleSource"] == "fastf1-schedule"
    assert response["scheduleSession"]["session_key"] == "fastf1:2026:austrian-grand-prix:fp3"


def test_session_status_keeps_fastf1_live_when_openf1_auth_is_missing(monkeypatch, tmp_path):
    app = _create_test_app(monkeypatch, tmp_path, _OpenF1UnavailableClient())
    route = _session_status_route(app)

    response = asyncio.run(route.endpoint(year=2026, now="2026-06-27T10:40:00Z"))

    assert response["source"] == "fastf1-schedule"
    assert response["session"]["session_key"] == "fastf1:2026:austrian-grand-prix:fp3"
    assert "OpenF1 live timing unavailable" in response["fallbackReason"]
