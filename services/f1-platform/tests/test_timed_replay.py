import asyncio

from f1_platform.event_stream import InMemoryEventStream
from f1_platform.replay import raw_event
from f1_platform.runtime import F1PlatformRuntime
from f1_platform.sample_data import SAMPLE_SESSION_KEY, sample_events
from f1_platform.storage import JsonlEventStore
from f1_platform.timed_replay import TimedReplayController, replay_delay_seconds


def test_replay_delay_seconds_scales_event_time_delta():
    first = raw_event(
        1,
        "v1/position",
        "44:position:1",
        "s1",
        {"position": 1},
        driver_number=44,
        event_time="2026-06-25T12:00:00Z",
    )
    second = raw_event(
        2,
        "v1/position",
        "44:position:2",
        "s1",
        {"position": 2},
        driver_number=44,
        event_time="2026-06-25T12:00:10Z",
    )

    assert replay_delay_seconds(first, second, speed=5) == 2.0
    assert replay_delay_seconds(None, second, speed=5) == 0.0


def test_timed_replay_feeds_stored_events_through_live_ingest_path(tmp_path):
    async def run():
        sleep_calls = []

        async def fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        event_stream = InMemoryEventStream()
        runtime = F1PlatformRuntime(event_stream=event_stream)
        store = JsonlEventStore(tmp_path)
        events = sample_events(SAMPLE_SESSION_KEY)
        store.replace(SAMPLE_SESSION_KEY, events)
        controller = TimedReplayController(runtime, store, sleep=fake_sleep)

        status = await controller.start(SAMPLE_SESSION_KEY, speed=20, max_delay_seconds=0)
        assert status.state == "starting"

        finished = await controller.wait(SAMPLE_SESSION_KEY)
        snapshot = await runtime.snapshot(SAMPLE_SESSION_KEY)
        records = await event_stream.read_session(SAMPLE_SESSION_KEY, count=1_000)

        assert finished.state == "finished"
        assert finished.cursor == len(events)
        assert snapshot.seq == len(records)
        assert snapshot.drivers
        assert all(delay == 0 for delay in sleep_calls)

    asyncio.run(run())


def test_timed_replay_reports_missing_fixture(tmp_path):
    async def run():
        controller = TimedReplayController(F1PlatformRuntime(), JsonlEventStore(tmp_path))

        try:
            await controller.start("missing-session")
        except FileNotFoundError as exc:
            assert "missing-session" in str(exc)
        else:
            raise AssertionError("expected FileNotFoundError")

    asyncio.run(run())
