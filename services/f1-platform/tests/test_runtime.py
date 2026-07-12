import asyncio
import threading
from dataclasses import replace

from f1_platform.predictions import PredictionService
from f1_platform.replay import load_jsonl_events, raw_event
from f1_platform.runtime import F1PlatformRuntime
from f1_platform.sample_data import SAMPLE_SESSION_KEY, sample_events
from f1_platform.schemas import PredictionSnapshot, SessionSnapshot
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


def test_runtime_routes_sprint_shootout_to_qualifying_end_to_end():
    async def run():
        service = _RecordingPredictionService()
        runtime = F1PlatformRuntime(prediction_service=service)
        session_key = "sprint-shootout"
        session = raw_event(
            1,
            "v1/sessions",
            "session:sprint-shootout",
            session_key,
            {"session_name": "Sprint Shootout", "session_type": "Race"},
        )

        await runtime.reset_session(session_key, [session])
        await runtime.ingest(
            raw_event(
                2,
                "v1/position",
                "63:position",
                session_key,
                {"position": 1},
                driver_number=63,
            )
        )

        assert service.calls == ["qualifying", "qualifying"]

    asyncio.run(run())


def test_runtime_releases_ingestion_lock_and_rejects_stale_inference():
    async def run():
        service = _GatedPredictionService()
        runtime = F1PlatformRuntime(prediction_service=service)
        session_key = "concurrent-inference"
        first_event = raw_event(
            1,
            "v1/position",
            "63:position",
            session_key,
            {"position": 2},
            driver_number=63,
        )
        second_event = raw_event(
            2,
            "v1/position",
            "63:position",
            session_key,
            {"position": 1},
            driver_number=63,
        )

        first_task = asyncio.create_task(runtime.ingest(first_event))
        await service.wait_until_started(1)

        visible_while_inference_waits = await asyncio.wait_for(runtime.snapshot(session_key), timeout=2.0)
        assert visible_while_inference_waits.seq == 1
        assert visible_while_inference_waits.predictions == []

        second_task = asyncio.create_task(runtime.ingest(second_event))
        await service.wait_until_started(2)
        service.release(2)
        assert await asyncio.wait_for(second_task, timeout=2.0) is not None

        service.release(1)
        assert await asyncio.wait_for(first_task, timeout=2.0) is not None

        snapshot = await runtime.snapshot(session_key)
        assert snapshot.seq == 2
        assert [prediction.source_event_sequence for prediction in snapshot.predictions] == [2]
        assert [prediction.prediction_kind for prediction in snapshot.predictions] == ["race"]

    asyncio.run(run())


def test_runtime_serializes_projection_writes_by_session_sequence():
    async def run():
        store = _BlockingProjectionStore(block_sequence=1)
        runtime = F1PlatformRuntime(projection_store=store)

        first_task = asyncio.create_task(
            runtime.ingest(
                raw_event(
                    1,
                    "v1/position",
                    "63:position",
                    "projection-order",
                    {"position": 2},
                    driver_number=63,
                )
            )
        )
        assert await asyncio.to_thread(store.started.wait, 2.0)

        second_task = asyncio.create_task(
            runtime.ingest(
                raw_event(
                    2,
                    "v1/position",
                    "63:position",
                    "projection-order",
                    {"position": 1},
                    driver_number=63,
                )
            )
        )
        await asyncio.sleep(0.05)
        assert store.sequences == []

        store.release.set()
        await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=2.0)

        assert store.sequences == [1, 2]
        assert store.latest_sequence == 2

    asyncio.run(run())


def test_runtime_skips_projection_from_an_older_sequence_or_reset_generation():
    async def run():
        store = _RecordingProjectionStore()
        runtime = F1PlatformRuntime(projection_store=store)
        current = await runtime.reset_session("projection-stale", [])
        generation = runtime._projection_generation["projection-stale"]

        newer = replace(current, seq=4)
        older = replace(current, seq=3)
        await runtime._project(newer, generation=generation)
        await runtime._project(older, generation=generation)
        await runtime._project(newer, generation=generation - 1)

        assert store.sequences == [0, 4]

    asyncio.run(run())


class _RecordingPredictionService(PredictionService):
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def _record(self, kind: str, _state: SessionSnapshot) -> list[PredictionSnapshot]:
        self.calls.append(kind)
        return []

    async def predict_qualifying(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        return await self._record("qualifying", state)

    async def predict_race(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        return await self._record("race", state)

    async def predict_next_lap(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        return await self._record("next-lap", state)

    async def predict_strategy(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        return await self._record("strategy", state)


class _GatedPredictionService(PredictionService):
    def __init__(self) -> None:
        self.started: dict[int, asyncio.Event] = {}
        self.releases: dict[int, asyncio.Event] = {}

    async def _run(self, kind: str, state: SessionSnapshot) -> list[PredictionSnapshot]:
        started = self.started.setdefault(state.seq, asyncio.Event())
        release = self.releases.setdefault(state.seq, asyncio.Event())
        started.set()
        await release.wait()
        driver_number = state.drivers[0].driver_number
        return [
            PredictionSnapshot(
                model_version=f"gated_{kind}_v1",
                prediction_time="2026-07-12T00:00:00Z",
                source_event_sequence=state.seq,
                features_version="gated_features_v1",
                driver_number=driver_number,
                expected_position=1.0,
                position_distribution={"1": 1.0},
                win_probability=1.0,
                podium_probability=1.0,
                points_probability=1.0,
                dnf_probability=0.0,
                confidence=1.0,
                prediction_kind=kind,
            )
        ]

    async def wait_until_started(self, sequence: int) -> None:
        async def wait() -> None:
            while sequence not in self.started:
                await asyncio.sleep(0)
            await self.started[sequence].wait()

        await asyncio.wait_for(wait(), timeout=2.0)

    def release(self, sequence: int) -> None:
        self.releases[sequence].set()

    async def predict_qualifying(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        return await self._run("qualifying", state)

    async def predict_race(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        return await self._run("race", state)

    async def predict_next_lap(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        return await self._run("next-lap", state)

    async def predict_strategy(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        return await self._run("strategy", state)


class _RecordingProjectionStore:
    kind = "test"

    def __init__(self) -> None:
        self.sequences: list[int] = []

    def initialize(self) -> None:
        return None

    def project_snapshot(self, snapshot: SessionSnapshot) -> None:
        self.sequences.append(snapshot.seq)

    def session_counts(self, _session_key):
        return {}

    def derived_analytics(self, _session_key):
        return {}


class _BlockingProjectionStore(_RecordingProjectionStore):
    def __init__(self, *, block_sequence: int) -> None:
        super().__init__()
        self.block_sequence = block_sequence
        self.started = threading.Event()
        self.release = threading.Event()

    @property
    def latest_sequence(self) -> int | None:
        return self.sequences[-1] if self.sequences else None

    def project_snapshot(self, snapshot: SessionSnapshot) -> None:
        if snapshot.seq == self.block_sequence:
            self.started.set()
            if not self.release.wait(timeout=2.0):
                raise TimeoutError("projection test release was not signalled")
        super().project_snapshot(snapshot)
