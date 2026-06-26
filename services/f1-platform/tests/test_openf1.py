import asyncio
from urllib.parse import parse_qs

from f1_platform.openf1 import (
    OpenF1AuthConfig,
    OpenF1TokenManager,
    normalize_openf1_message,
)


def test_token_manager_posts_form_credentials_and_caches_token():
    requests = []

    def transport(request):
        requests.append(request)
        body = parse_qs(request.data.decode("utf-8"))
        assert request.full_url == "https://api.openf1.org/token"
        assert request.get_header("Content-type") == "application/x-www-form-urlencoded"
        assert body == {"username": ["user@example.com"], "password": ["secret"]}
        return {"access_token": "token-1", "expires_in": "3600", "token_type": "bearer"}

    manager = OpenF1TokenManager(
        OpenF1AuthConfig(username="user@example.com", password="secret", refresh_margin_seconds=60),
        transport=transport,
    )

    assert asyncio.run(manager.token()) == "token-1"
    assert asyncio.run(manager.token()) == "token-1"
    assert len(requests) == 1


def test_token_manager_from_env(monkeypatch):
    monkeypatch.setenv("OPENF1_USERNAME", "env-user")
    monkeypatch.setenv("OPENF1_PASSWORD", "env-password")

    manager = OpenF1TokenManager.from_env()

    assert manager.config.username == "env-user"
    assert manager.config.password == "env-password"


def test_normalize_openf1_message_keeps_raw_payload_and_keys():
    normalized = normalize_openf1_message(
        "v1/laps",
        {
            "_id": 42,
            "_key": "63:lap:12",
            "meeting_key": 100,
            "session_key": 200,
            "driver_number": 63,
            "date": "2026-06-25T10:00:00Z",
            "lap_number": 12,
        },
    )

    assert normalized["source"] == "openf1"
    assert normalized["topic"] == "v1/laps"
    assert normalized["source_id"] == 42
    assert normalized["source_key"] == "63:lap:12"
    assert normalized["payload"]["lap_number"] == 12


def test_normalize_openf1_session_message_uses_date_start_as_event_time():
    normalized = normalize_openf1_message(
        "v1/sessions",
        {
            "_id": 7,
            "_key": "session:latest",
            "meeting_key": 100,
            "session_key": 200,
            "session_name": "Race",
            "session_type": "Race",
            "date_start": "2026-06-25T13:00:00Z",
        },
    )

    assert normalized["topic"] == "v1/sessions"
    assert normalized["event_time"] == "2026-06-25T13:00:00Z"
    assert normalized["payload"]["session_name"] == "Race"
