import asyncio

from f1_platform.event_stream import InMemoryEventStream
from f1_platform.replay import raw_event


def test_in_memory_event_stream_preserves_session_order():
    async def run():
        stream = InMemoryEventStream()
        first = raw_event(1, "v1/laps", "44:lap:1", "s1", {"lap_number": 1}, driver_number=44)
        second = raw_event(2, "v1/laps", "44:lap:2", "s1", {"lap_number": 2}, driver_number=44)

        assert await stream.append(first) == "1-0"
        assert await stream.append(second) == "2-0"

        records = await stream.read_session("s1")
        assert [record.event.source_key for record in records] == ["44:lap:1", "44:lap:2"]
        assert records[-1].to_dict()["event"]["source_id"] == 2

    asyncio.run(run())


def test_in_memory_event_stream_bounds_recent_records():
    async def run():
        stream = InMemoryEventStream()
        for source_id in range(1, 6):
            await stream.append(
                raw_event(
                    source_id,
                    "v1/position",
                    f"44:position:{source_id}",
                    "s1",
                    {"position": source_id},
                    driver_number=44,
                )
            )

        records = await stream.read_session("s1", count=2)
        assert [record.id for record in records] == ["4-0", "5-0"]

    asyncio.run(run())
