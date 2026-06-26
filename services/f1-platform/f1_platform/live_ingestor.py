"""OpenF1 MQTT live ingestor.

The API process owns reduction, projections, and WebSocket fan-out. This module
is a separate process boundary: it connects to OpenF1 MQTT, normalizes each
message into the stable event contract, then submits it to the FastAPI event
ingress endpoint.
"""

from __future__ import annotations

import asyncio
import json
import os
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .openf1 import OPENF1_TOPICS, OpenF1TokenManager, normalize_openf1_message
from .schemas import F1Event, JsonObject
from .time import utc_now_iso


class EventSink(Protocol):
    async def submit(self, event: F1Event) -> None:
        ...


class DeadLetterSpool(Protocol):
    async def append(self, event: F1Event, *, error: str) -> None:
        ...


@dataclass(slots=True)
class OpenF1MqttIngestorConfig:
    broker: str = "mqtt.openf1.org"
    port: int = 8883
    keepalive_seconds: int = 60
    mqtt_username: str = "openf1-live-ingestor"
    topics: tuple[str, ...] = OPENF1_TOPICS
    fallback_session_key: str | None = None
    queue_size: int = 10_000
    token_reconnect_seconds: int = 3_300
    reconnect_delay_seconds: float = 5.0
    sink_retry_attempts: int = 3
    sink_retry_backoff_seconds: float = 0.5
    dead_letter_path: str | None = None

    @classmethod
    def from_env(cls) -> "OpenF1MqttIngestorConfig":
        topics = _env_topics(os.environ.get("OPENF1_MQTT_TOPICS")) or OPENF1_TOPICS
        return cls(
            broker=os.environ.get("OPENF1_MQTT_BROKER", "mqtt.openf1.org"),
            port=_env_int("OPENF1_MQTT_PORT", 8883),
            keepalive_seconds=_env_int("OPENF1_MQTT_KEEPALIVE_SECONDS", 60),
            mqtt_username=(
                os.environ.get("OPENF1_MQTT_USERNAME")
                or os.environ.get("OPENF1_USERNAME")
                or "openf1-live-ingestor"
            ),
            topics=topics,
            fallback_session_key=os.environ.get("OPENF1_FALLBACK_SESSION_KEY") or None,
            queue_size=_env_int("OPENF1_INGEST_QUEUE_SIZE", 10_000),
            token_reconnect_seconds=_env_int("OPENF1_TOKEN_RECONNECT_SECONDS", 3_300),
            reconnect_delay_seconds=_env_float("OPENF1_RECONNECT_DELAY_SECONDS", 5.0),
            sink_retry_attempts=max(1, _env_int("OPENF1_SINK_RETRY_ATTEMPTS", 3)),
            sink_retry_backoff_seconds=_env_float("OPENF1_SINK_RETRY_BACKOFF_SECONDS", 0.5),
            dead_letter_path=_env_dead_letter_path(os.environ.get("OPENF1_DEAD_LETTER_PATH")),
        )


class OpenF1ApiEventSink:
    """Submit normalized live events into the platform API event endpoint."""

    def __init__(
        self,
        api_base_url: str,
        *,
        timeout_seconds: float = 10.0,
        transport: Callable[[Request], Any] | None = None,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    async def submit(self, event: F1Event) -> None:
        await asyncio.to_thread(self._submit_sync, event)

    def _submit_sync(self, event: F1Event) -> None:
        session_key = quote(str(event.session_key), safe="")
        url = f"{self.api_base_url}/api/f1/sessions/{session_key}/events"
        body = json.dumps(event.to_dict(), separators=(",", ":")).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            if self._transport is not None:
                response = self._transport(request)
                status = int(getattr(response, "status", getattr(response, "code", 200)))
                payload = response.read() if hasattr(response, "read") else b""
            else:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    status = int(response.status)
                    payload = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"F1 API event submit failed with HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"F1 API event submit failed: {exc.reason}") from exc
        if status < 200 or status >= 300:
            detail = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else str(payload)
            raise RuntimeError(f"F1 API event submit failed with HTTP {status}: {detail}")


class JsonlDeadLetterSpool:
    """Append failed live-ingest events to a local JSONL spool."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    async def append(self, event: F1Event, *, error: str) -> None:
        await asyncio.to_thread(self._append_sync, event, error)

    def _append_sync(self, event: F1Event, error: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "failedAt": utc_now_iso(),
            "error": error,
            "event": event.to_dict(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


class OpenF1MqttLiveIngestor:
    """Long-running MQTT client with an async submission worker."""

    def __init__(
        self,
        token_manager: OpenF1TokenManager,
        sink: EventSink,
        *,
        config: OpenF1MqttIngestorConfig | None = None,
        sleep: Callable[[float], Any] | None = None,
        dead_letter_spool: DeadLetterSpool | None = None,
    ) -> None:
        self.token_manager = token_manager
        self.sink = sink
        self.config = config or OpenF1MqttIngestorConfig()
        self.status: JsonObject = {
            "connected": False,
            "received": 0,
            "submitted": 0,
            "invalid": 0,
            "dropped": 0,
            "failed": 0,
            "deadLettered": 0,
            "submitRetries": 0,
            "reconnects": 0,
            "lastError": None,
        }
        self._queue: asyncio.Queue[F1Event] = asyncio.Queue(maxsize=self.config.queue_size)
        self._stop_event = asyncio.Event()
        self._client: Any | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reconnect_task: asyncio.Task | None = None
        self._sleep = sleep or asyncio.sleep
        self.dead_letter_spool = dead_letter_spool or (
            JsonlDeadLetterSpool(self.config.dead_letter_path) if self.config.dead_letter_path else None
        )

    async def run_forever(self) -> None:
        self._loop = asyncio.get_running_loop()
        submitter = asyncio.create_task(self._submit_worker(), name="openf1-submit-worker")
        refresher = asyncio.create_task(self._token_refresh_loop(), name="openf1-token-refresh")
        try:
            await self._connect_client()
            await self._stop_event.wait()
        finally:
            await self._disconnect_client()
            if self._reconnect_task is not None:
                self._reconnect_task.cancel()
            refresher.cancel()
            submitter.cancel()
            tasks = [task for task in (refresher, submitter, self._reconnect_task) if task is not None]
            await asyncio.gather(*tasks, return_exceptions=True)

    def stop(self) -> None:
        self._stop_event.set()

    async def _connect_client(self) -> None:
        mqtt = _import_paho_mqtt()
        token = await self.token_manager.token()
        client = _new_mqtt_client(mqtt)
        client.username_pw_set(username=self.config.mqtt_username, password=token)
        client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS_CLIENT)
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        self._client = client
        await asyncio.to_thread(client.connect, self.config.broker, self.config.port, self.config.keepalive_seconds)
        client.loop_start()

    async def _disconnect_client(self) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        try:
            await asyncio.to_thread(client.disconnect)
        finally:
            client.loop_stop()
            self.status["connected"] = False

    async def _token_refresh_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.config.token_reconnect_seconds)
            except asyncio.TimeoutError:
                await self._disconnect_client()
                await self._connect_client()

    async def _submit_worker(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                await self._submit_with_retries(event)
                self.status["submitted"] = int(self.status["submitted"]) + 1
            except Exception as exc:
                self.status["failed"] = int(self.status["failed"]) + 1
                self.status["lastError"] = str(exc)
                await self._dead_letter(event, error=str(exc))
            finally:
                self._queue.task_done()

    async def _dead_letter(self, event: F1Event, *, error: str) -> None:
        if self.dead_letter_spool is None:
            return
        try:
            await self.dead_letter_spool.append(event, error=error)
            self.status["deadLettered"] = int(self.status["deadLettered"]) + 1
        except Exception as exc:
            self.status["lastError"] = f"{error}; dead-letter failed: {exc}"

    async def _submit_with_retries(self, event: F1Event) -> None:
        attempts = max(1, self.config.sink_retry_attempts)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                await self.sink.submit(event)
                return
            except Exception as exc:
                last_error = exc
                self.status["lastError"] = str(exc)
                if attempt >= attempts:
                    break
                self.status["submitRetries"] = int(self.status["submitRetries"]) + 1
                await self._sleep(max(0.0, self.config.sink_retry_backoff_seconds))
        assert last_error is not None
        raise last_error

    def _on_connect(self, client: Any, _userdata: Any, _flags: Any, reason_code: Any, _properties: Any = None) -> None:
        if _reason_code_ok(reason_code):
            self.status["connected"] = True
            self.status["lastError"] = None
            for topic in self.config.topics:
                client.subscribe(topic)
        else:
            self.status["connected"] = False
            self.status["lastError"] = f"MQTT connect failed: {reason_code}"

    def _on_disconnect(self, _client: Any, _userdata: Any, *args: Any) -> None:
        reason_code = args[1] if len(args) >= 2 else args[0] if args else 0
        self.status["connected"] = False
        if not _reason_code_ok(reason_code):
            self.status["lastError"] = f"MQTT disconnected: {reason_code}"
            loop = self._loop
            if loop is not None:
                loop.call_soon_threadsafe(self._schedule_reconnect)

    def _schedule_reconnect(self) -> None:
        if self._stop_event.is_set():
            return
        if self._reconnect_task is not None and not self._reconnect_task.done():
            return
        self._reconnect_task = asyncio.create_task(self._reconnect_after_delay(), name="openf1-mqtt-reconnect")

    async def _reconnect_after_delay(self) -> None:
        while not self._stop_event.is_set():
            await self._sleep(max(0.0, self.config.reconnect_delay_seconds))
            if self._stop_event.is_set():
                return
            try:
                await self._disconnect_client()
                await self._connect_client()
                self.status["reconnects"] = int(self.status["reconnects"]) + 1
                return
            except Exception as exc:
                self.status["lastError"] = str(exc)

    def _on_message(self, _client: Any, _userdata: Any, message: Any) -> None:
        try:
            event = decode_openf1_mqtt_payload(
                str(message.topic),
                message.payload,
                received_at=utc_now_iso(),
                fallback_session_key=self.config.fallback_session_key,
            )
        except Exception as exc:
            self.status["invalid"] = int(self.status["invalid"]) + 1
            self.status["lastError"] = str(exc)
            return
        self.status["received"] = int(self.status["received"]) + 1
        loop = self._loop
        if loop is None:
            self.status["dropped"] = int(self.status["dropped"]) + 1
            return
        loop.call_soon_threadsafe(self._enqueue_event, event)

    def _enqueue_event(self, event: F1Event) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self.status["dropped"] = int(self.status["dropped"]) + 1


def decode_openf1_mqtt_payload(
    topic: str,
    payload: bytes | bytearray | str,
    *,
    received_at: str,
    fallback_session_key: int | str | None = None,
) -> F1Event:
    if isinstance(payload, (bytes, bytearray)):
        text = bytes(payload).decode("utf-8")
    else:
        text = payload
    decoded = json.loads(text)
    if not isinstance(decoded, dict):
        raise ValueError("OpenF1 MQTT payload must be a JSON object")
    normalized = normalize_openf1_message(topic, decoded)
    if normalized.get("session_key") is None and fallback_session_key is not None:
        normalized["session_key"] = fallback_session_key
    return F1Event.from_payload(normalized, topic=topic, received_at=received_at)


def _import_paho_mqtt():
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        raise RuntimeError("Install paho-mqtt to enable OpenF1 MQTT ingestion") from exc
    return mqtt


def _new_mqtt_client(mqtt: Any):
    callback_api_version = getattr(mqtt, "CallbackAPIVersion", None)
    if callback_api_version is not None:
        return mqtt.Client(callback_api_version.VERSION2)
    return mqtt.Client()


def _reason_code_ok(reason_code: Any) -> bool:
    try:
        return int(reason_code) == 0
    except (TypeError, ValueError):
        return str(reason_code).lower() in {"0", "success", "normal disconnection"}


def _env_topics(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    topics = tuple(topic.strip() for topic in value.split(",") if topic.strip())
    return topics or None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_dead_letter_path(value: str | None) -> str | None:
    text = str(value or "").strip()
    if text.lower() in {"0", "false", "no", "off", "disabled", "none"}:
        return None
    if text:
        return text
    return str(_default_dead_letter_path())


def _default_dead_letter_path() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / ".git").exists() and (candidate / "data" / "raw" / "f1").exists():
            return candidate / "data" / "raw" / "f1" / "openf1-dead-letter.jsonl"
    return Path("data/raw/f1/openf1-dead-letter.jsonl")


async def _amain() -> None:
    api_base_url = os.environ.get("F1_PLATFORM_API_URL", "http://127.0.0.1:8001")
    sink = OpenF1ApiEventSink(api_base_url)
    ingestor = OpenF1MqttLiveIngestor(
        OpenF1TokenManager.from_env(),
        sink,
        config=OpenF1MqttIngestorConfig.from_env(),
    )
    await ingestor.run_forever()


def main() -> None:
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
