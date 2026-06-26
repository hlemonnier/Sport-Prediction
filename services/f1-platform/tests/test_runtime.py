import asyncio

from f1_platform.replay import load_jsonl_events, raw_event
from f1_platform.runtime import F1PlatformRuntime
from f1_platform.sample_data import SAMPLE_SESSION_KEY, sample_events
from f1_platform.storage import JsonlEventStore


def test_runtime_persists_prediction_timeline():
    asyncio.run(_assert_runtime_persists_prediction_timeline())


async def _assert_runtime_persists_prediction_timeline():
    runtime = F1PlatformRuntime()
    await runtime.reset_session(SAMPLE_SESSION_KEY, sample_events(SAMPLE_SESSION_KEY))
    state = await runtime.snapshot(SAMPLE_SESSION_KEY)

    assert state.predictions
    first_count = len(state.predictions)

    event = raw_event(
        999,
        "v1/position",
        "63:position",
        SAMPLE_SESSION_KEY,
        {"position": 1},
        driver_number=63,
    )
    update = await runtime.ingest(event)
    state_after = await runtime.snapshot(SAMPLE_SESSION_KEY)

    assert update is not None
    assert len(state_after.predictions) > first_count
    assert state_after.predictions[-1].source_event_sequence == state_after.seq


def test_jsonl_event_store_round_trips_replay(tmp_path):
    events = sample_events(SAMPLE_SESSION_KEY)
    store = JsonlEventStore(tmp_path)
    replay_path = store.replace(SAMPLE_SESSION_KEY, events)

    loaded = load_jsonl_events(replay_path, session_key=SAMPLE_SESSION_KEY)

    assert len(loaded) == len(events)
    assert loaded[0].topic == events[0].topic
    assert loaded[0].received_at == events[0].received_at
    assert loaded[-1].source_key == events[-1].source_key


def test_runtime_normalizes_numeric_session_keys():
    async def run():
        runtime = F1PlatformRuntime()
        await runtime.reset_session(9165, sample_events(9165), source="openf1-rest")

        numeric = await runtime.snapshot(9165)
        text = await runtime.snapshot("9165")

        assert numeric.seq == text.seq
        assert len(text.drivers) == len(numeric.drivers)
        assert text.session_key == "9165"

    asyncio.run(run())


def test_runtime_session_summary_exposes_session_metadata():
    async def run():
        runtime = F1PlatformRuntime()
        await runtime.reset_session(SAMPLE_SESSION_KEY, sample_events(SAMPLE_SESSION_KEY))

        [summary] = runtime.list_sessions()

        assert summary["sessionKey"] == SAMPLE_SESSION_KEY
        assert summary["sessionName"] == "Race"
        assert summary["sessionType"] == "Race"
        assert summary["meetingKey"] == 2026001
        assert summary["year"] == 2026

    asyncio.run(run())


def test_runtime_raw_stream_records_stale_events_before_reduction():
    async def run():
        runtime = F1PlatformRuntime()
        first = raw_event(
            10,
            "v1/laps",
            "63:lap:22",
            "s1",
            {"lap_number": 22, "lap_duration": 69.1},
            driver_number=63,
        )
        stale = raw_event(
            9,
            "v1/laps",
            "63:lap:22",
            "s1",
            {"lap_number": 22, "lap_duration": 67.0},
            driver_number=63,
        )

        assert await runtime.ingest(first) is not None
        assert await runtime.ingest(stale) is None

        records = await runtime.recent_events("s1", count=10)
        assert [record["event"]["source_id"] for record in records] == [10, 9]

    asyncio.run(run())
