"""OpenF1 authentication and topic boundary.

This module keeps live OpenF1 credentials server-side. The first production
ingestor can plug an MQTT/WebSocket client into `OPENF1_TOPICS` and feed
`F1Event.from_payload` without changing the API contract.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from os import environ
from time import monotonic
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

OPENF1_TOPICS = (
    "v1/sessions",
    "v1/drivers",
    "v1/laps",
    "v1/position",
    "v1/intervals",
    "v1/stints",
    "v1/pit",
    "v1/race_control",
    "v1/weather",
    "v1/car_data",
    "v1/location",
    "v1/overtakes",
)


@dataclass(slots=True)
class OpenF1AuthConfig:
    username: str
    password: str
    auth_url: str = "https://api.openf1.org/token"
    refresh_margin_seconds: int = 120


class OpenF1TokenManager:
    """Small token-renewal boundary for the future live ingestor."""

    def __init__(
        self,
        config: OpenF1AuthConfig,
        *,
        transport: Callable[[Request], dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self._transport = transport
        self._token: str | None = None
        self._expires_at_monotonic = 0.0
        self._lock = asyncio.Lock()

    async def token(self) -> str:
        async with self._lock:
            if self._token and monotonic() < self._expires_at_monotonic:
                return self._token
            token, expires_in = await self._request_token()
            self._token = token
            self._expires_at_monotonic = monotonic() + max(
                1.0,
                float(expires_in) - float(self.config.refresh_margin_seconds),
            )
            return token

    async def _request_token(self) -> tuple[str, int]:
        return await asyncio.to_thread(self._request_token_sync)

    def _request_token_sync(self) -> tuple[str, int]:
        payload = urlencode({"username": self.config.username, "password": self.config.password}).encode("utf-8")
        request = Request(
            self.config.auth_url,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            response_payload = self._transport(request) if self._transport is not None else _urlopen_json(request)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenF1 token request failed with HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"OpenF1 token request failed: {exc.reason}") from exc

        access_token = response_payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise RuntimeError("OpenF1 token response did not include access_token")
        return access_token, int(float(response_payload.get("expires_in") or 3600))

    @classmethod
    def from_env(cls) -> "OpenF1TokenManager":
        username = environ.get("OPENF1_USERNAME")
        password = environ.get("OPENF1_PASSWORD")
        if not username or not password:
            raise RuntimeError("OPENF1_USERNAME and OPENF1_PASSWORD are required for live OpenF1 access")
        return cls(OpenF1AuthConfig(username=username, password=password))


def normalize_openf1_message(topic: str, message: dict[str, Any]) -> dict[str, Any]:
    """Normalize an OpenF1 stream message into the raw platform event shape."""

    payload = dict(message)
    return {
        "topic": topic,
        "source": "openf1",
        "source_id": payload.get("_id", payload.get("id", 0)),
        "source_key": payload.get("_key"),
        "meeting_key": payload.get("meeting_key"),
        "session_key": payload.get("session_key"),
        "driver_number": payload.get("driver_number"),
        "event_time": payload.get("date", payload.get("date_start")),
        "payload": payload,
    }


def _urlopen_json(request: Request) -> dict[str, Any]:
    with urlopen(request, timeout=20.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("OpenF1 token response was not a JSON object")
    return payload
