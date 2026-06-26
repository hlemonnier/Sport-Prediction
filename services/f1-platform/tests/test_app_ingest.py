import asyncio
import json

import pytest

from f1_platform.replay import raw_event


def test_api_ingress_persists_live_events_to_replay_store(monkeypatch, tmp_path):
    try:
        from f1_platform.app import create_app
    except ImportError as exc:
        pytest.skip(f"FastAPI unavailable: {exc}")

    monkeypatch.setenv("F1_PLATFORM_EVENT_STORE", str(tmp_path / "events"))
    monkeypatch.setenv("F1_PLATFORM_SQLITE_PROJECTION_STORE", str(tmp_path / "projection.sqlite"))
    monkeypatch.setenv("F1_PLATFORM_FASTF1_ARTIFACT_STORE", str(tmp_path / "artifacts"))
    monkeypatch.delenv("F1_PLATFORM_REDIS_URL", raising=False)
    app = create_app()
    route = next(
        route
        for route in app.routes
        if getattr(route, "path", "") == "/api/f1/sessions/{session_key}/events"
        and "POST" in getattr(route, "methods", set())
    )

    first = raw_event(
        10,
        "v1/laps",
        "63:lap:22",
        "live-s1",
        {"lap_number": 22, "lap_duration": 69.1},
        driver_number=63,
    )
    stale = raw_event(
        9,
        "v1/laps",
        "63:lap:22",
        "live-s1",
        {"lap_number": 22, "lap_duration": 67.0},
        driver_number=63,
    )

    first_response = asyncio.run(route.endpoint(session_key="live-s1", body=first.to_dict()))
    stale_response = asyncio.run(route.endpoint(session_key="live-s1", body=stale.to_dict()))

    assert first_response["accepted"] is True
    assert stale_response == {"accepted": False, "reason": "duplicate_or_stale"}
    replay_path = app.state.event_store.path_for_session("live-s1")
    rows = [json.loads(line) for line in replay_path.read_text(encoding="utf-8").splitlines()]
    assert [row["source_id"] for row in rows] == [10, 9]
    assert rows[0]["payload"]["lap_duration"] == 69.1


def test_track_geometry_endpoint_exposes_fastf1_centerline(monkeypatch, tmp_path):
    try:
        from f1_platform.app import create_app
    except ImportError as exc:
        pytest.skip(f"FastAPI unavailable: {exc}")

    monkeypatch.setenv("F1_PLATFORM_EVENT_STORE", str(tmp_path / "events"))
    monkeypatch.setenv("F1_PLATFORM_SQLITE_PROJECTION_STORE", str(tmp_path / "projection.sqlite"))
    monkeypatch.setenv("F1_PLATFORM_FASTF1_ARTIFACT_STORE", str(tmp_path / "artifacts"))
    monkeypatch.delenv("F1_PLATFORM_REDIS_URL", raising=False)
    app = create_app()
    app.state.fastf1_artifacts.artifact_store.write_table(
        [
            {"Distance": 0, "Progress": 0, "X": 100, "Y": 100, "Z": 0},
            {"Distance": 50, "Progress": 0.5, "X": 140, "Y": 120, "Z": 0},
            {"Distance": 100, "Progress": 1, "X": 160, "Y": 80, "Z": 0},
        ],
        "centerline/year=2026/event=austria/session=r/canonical",
        preferred_format="jsonl",
        metadata={"kind": "fastf1_centerline", "sessionKey": "fastf1:2026:austria:r"},
    )
    route = next(
        route
        for route in app.routes
        if getattr(route, "path", "") == "/api/f1/sessions/{session_key}/track-geometry"
        and "GET" in getattr(route, "methods", set())
    )

    response = asyncio.run(
        route.endpoint(
            session_key="sample-race",
            centerline_session_key="fastf1:2026:austria:r",
            limit=2,
        )
    )

    assert response["sessionKey"] == "fastf1:2026:austria:r"
    assert response["pointCount"] == 3
    assert response["sampledPointCount"] == 2
    assert response["points"][0] == {"distance": 0, "progress": 0, "x": 100, "y": 100, "z": 0}
    assert response["points"][-1]["progress"] == 1
