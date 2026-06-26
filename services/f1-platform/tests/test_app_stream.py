import asyncio

import pytest


def test_websocket_stream_sends_periodic_full_snapshots(monkeypatch, tmp_path):
    try:
        from fastapi import WebSocketDisconnect
        from f1_platform.app import create_app
    except ImportError as exc:
        pytest.skip(f"FastAPI unavailable: {exc}")

    monkeypatch.setenv("F1_PLATFORM_EVENT_STORE", str(tmp_path / "events"))
    monkeypatch.setenv("F1_PLATFORM_SQLITE_PROJECTION_STORE", str(tmp_path / "projection.sqlite"))
    monkeypatch.setenv("F1_PLATFORM_FASTF1_ARTIFACT_STORE", str(tmp_path / "artifacts"))
    monkeypatch.setenv("F1_PLATFORM_WS_SNAPSHOT_INTERVAL_SECONDS", "0.001")
    app = create_app()
    route = next(route for route in app.routes if getattr(route, "path", "") == "/api/f1/sessions/{session_key}/stream")
    websocket = CapturingWebSocket(WebSocketDisconnect, max_messages=2)

    asyncio.run(route.endpoint(websocket, "sample-race"))

    assert websocket.accepted is True
    assert [message["type"] for message in websocket.messages] == ["snapshot", "snapshot"]
    assert [message["reason"] for message in websocket.messages] == ["initial", "periodic"]
    assert websocket.messages[0]["payload"]["sessionKey"] == "sample-race"
    assert not app.state.runtime.subscribers["sample-race"]


class CapturingWebSocket:
    def __init__(self, disconnect_error, *, max_messages: int):
        self.disconnect_error = disconnect_error
        self.max_messages = max_messages
        self.accepted = False
        self.messages = []

    async def accept(self):
        self.accepted = True

    async def send_json(self, payload):
        self.messages.append(payload)
        if len(self.messages) >= self.max_messages:
            raise self.disconnect_error(code=1000)
