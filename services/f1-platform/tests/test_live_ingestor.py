import asyncio
import json

import pytest

from f1_platform.live_ingestor import (
    JsonlDeadLetterSpool,
    OpenF1ApiEventSink,
    OpenF1MqttIngestorConfig,
    OpenF1MqttLiveIngestor,
    decode_openf1_mqtt_payload,
)
from f1_platform.replay import raw_event


def test_decode_openf1_mqtt_payload_normalizes_stream_message():
    event = decode_openf1_mqtt_payload(
        "v1/car_data",
        json.dumps(
            {
                "meeting_key": 1257,
                "session_key": 10007,
                "driver_number": 31,
                "date": "2025-04-11T11:21:16.603025+00:00",
                "speed": 312,
                "_key": "1744370476603_31",
                "_id": 1747235800206,
            }
        ).encode("utf-8"),
        received_at="2026-06-25T00:00:00Z",
    )

    assert event.topic == "v1/car_data"
    assert event.source == "openf1"
    assert event.source_id == 1747235800206
    assert event.source_key == "1744370476603_31"
    assert event.session_key == 10007
    assert event.driver_number == 31
    assert event.event_time == "2025-04-11T11:21:16.603025+00:00"
    assert event.payload["speed"] == 312


def test_decode_openf1_mqtt_payload_uses_fallback_session_key():
    event = decode_openf1_mqtt_payload(
        "v1/weather",
        '{"_id": 9, "_key": "weather:latest", "rainfall": 0}',
        received_at="2026-06-25T00:00:00Z",
        fallback_session_key="live-session",
    )

    assert event.session_key == "live-session"
    assert event.source_key == "weather:latest"


def test_decode_openf1_mqtt_payload_rejects_non_object_json():
    with pytest.raises(ValueError, match="JSON object"):
        decode_openf1_mqtt_payload(
            "v1/laps",
            '[{"session_key": 1}]',
            received_at="2026-06-25T00:00:00Z",
        )


def test_api_event_sink_posts_to_session_ingress():
    requests = []

    class Response:
        status = 200

        def read(self):
            return b'{"accepted":true}'

    def transport(request):
        requests.append(request)
        return Response()

    sink = OpenF1ApiEventSink("http://f1-api.local/", transport=transport)
    event = raw_event(
        12,
        "v1/laps",
        "44:lap:7",
        "session/with space",
        {"lap_number": 7, "lap_duration": 68.5},
        driver_number=44,
    )

    asyncio.run(sink.submit(event))

    assert len(requests) == 1
    assert requests[0].full_url == "http://f1-api.local/api/f1/sessions/session%2Fwith%20space/events"
    body = json.loads(requests[0].data.decode("utf-8"))
    assert body["topic"] == "v1/laps"
    assert body["source_id"] == 12
    assert body["payload"]["lap_duration"] == 68.5


def test_openf1_mqtt_config_reads_environment(monkeypatch):
    monkeypatch.setenv("OPENF1_MQTT_BROKER", "mqtt.example.test")
    monkeypatch.setenv("OPENF1_MQTT_PORT", "1883")
    monkeypatch.setenv("OPENF1_MQTT_TOPICS", "v1/laps, v1/position")
    monkeypatch.setenv("OPENF1_USERNAME", "operator@example.test")
    monkeypatch.setenv("OPENF1_FALLBACK_SESSION_KEY", "fallback")
    monkeypatch.setenv("OPENF1_DEAD_LETTER_PATH", "/tmp/f1-dead-letter.jsonl")

    config = OpenF1MqttIngestorConfig.from_env()

    assert config.broker == "mqtt.example.test"
    assert config.port == 1883
    assert config.topics == ("v1/laps", "v1/position")
    assert config.mqtt_username == "operator@example.test"
    assert config.fallback_session_key == "fallback"
    assert config.dead_letter_path == "/tmp/f1-dead-letter.jsonl"


def test_openf1_mqtt_config_can_disable_dead_letter(monkeypatch):
    monkeypatch.setenv("OPENF1_DEAD_LETTER_PATH", "off")

    config = OpenF1MqttIngestorConfig.from_env()

    assert config.dead_letter_path is None


def test_openf1_mqtt_submit_retries_transient_sink_failure():
    async def run():
        sleep_calls = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        sink = FlakySink(failures_before_success=1)
        ingestor = OpenF1MqttLiveIngestor(
            object(),
            sink,
            config=OpenF1MqttIngestorConfig(sink_retry_attempts=3, sink_retry_backoff_seconds=0.25),
            sleep=fake_sleep,
        )
        event = raw_event(1, "v1/laps", "1:lap:1", "s1", {"lap_number": 1}, driver_number=1)

        await ingestor._submit_with_retries(event)

        assert sink.attempts == 2
        assert int(ingestor.status["submitRetries"]) == 1
        assert sleep_calls == [0.25]

    asyncio.run(run())


def test_openf1_mqtt_unexpected_disconnect_schedules_single_reconnect():
    async def run():
        async def fake_sleep(_delay):
            return None

        ingestor = OpenF1MqttLiveIngestor(
            object(),
            FlakySink(),
            config=OpenF1MqttIngestorConfig(reconnect_delay_seconds=0),
            sleep=fake_sleep,
        )
        calls = []

        async def fake_disconnect():
            calls.append("disconnect")

        async def fake_connect():
            calls.append("connect")

        ingestor._loop = asyncio.get_running_loop()
        ingestor._disconnect_client = fake_disconnect
        ingestor._connect_client = fake_connect

        ingestor._on_disconnect(None, None, 7)
        ingestor._on_disconnect(None, None, 7)
        await asyncio.sleep(0)
        await ingestor._reconnect_task

        assert calls == ["disconnect", "connect"]
        assert int(ingestor.status["reconnects"]) == 1
        assert ingestor.status["lastError"] == "MQTT disconnected: 7"

    asyncio.run(run())


def test_jsonl_dead_letter_spool_writes_failed_event(tmp_path):
    async def run():
        event = raw_event(1, "v1/laps", "1:lap:1", "s1", {"lap_number": 1}, driver_number=1)
        spool = JsonlDeadLetterSpool(tmp_path / "dead-letter.jsonl")

        await spool.append(event, error="api down")

        rows = [json.loads(line) for line in (tmp_path / "dead-letter.jsonl").read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["error"] == "api down"
        assert rows[0]["event"]["source_key"] == "1:lap:1"
        assert rows[0]["event"]["payload"]["lap_number"] == 1

    asyncio.run(run())


def test_submit_worker_dead_letters_after_exhausted_retries(tmp_path):
    async def run():
        async def fake_sleep(_delay):
            return None

        sink = FlakySink(failures_before_success=10)
        ingestor = OpenF1MqttLiveIngestor(
            object(),
            sink,
            config=OpenF1MqttIngestorConfig(
                sink_retry_attempts=2,
                sink_retry_backoff_seconds=0,
                dead_letter_path=str(tmp_path / "dead-letter.jsonl"),
            ),
            sleep=fake_sleep,
        )
        event = raw_event(1, "v1/laps", "1:lap:1", "s1", {"lap_number": 1}, driver_number=1)
        worker = asyncio.create_task(ingestor._submit_worker())

        await ingestor._queue.put(event)
        await ingestor._queue.join()
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)

        rows = [json.loads(line) for line in (tmp_path / "dead-letter.jsonl").read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["event"]["source_key"] == "1:lap:1"
        assert int(ingestor.status["failed"]) == 1
        assert int(ingestor.status["deadLettered"]) == 1
        assert int(ingestor.status["submitRetries"]) == 1

    asyncio.run(run())


class FlakySink:
    def __init__(self, failures_before_success=0):
        self.failures_before_success = failures_before_success
        self.attempts = 0

    async def submit(self, _event):
        self.attempts += 1
        if self.attempts <= self.failures_before_success:
            raise RuntimeError("temporary sink failure")
