import asyncio
import json

from f1_platform.dead_letter import replay_dead_letters
from f1_platform.replay import raw_event


def test_replay_dead_letters_submits_events_and_removes_empty_spool(tmp_path):
    event_a = raw_event(1, "v1/laps", "1:lap:1", "s1", {"lap_number": 1}, driver_number=1)
    event_b = raw_event(2, "v1/laps", "2:lap:1", "s1", {"lap_number": 1}, driver_number=2)
    path = tmp_path / "dead-letter.jsonl"
    _write_spool(path, event_a, event_b)
    sink = RecordingSink()

    result = asyncio.run(replay_dead_letters(path, sink))

    assert result.processed == 2
    assert result.submitted == 2
    assert result.kept == 0
    assert result.failed == 0
    assert not path.exists()
    assert [event.source_key for event in sink.events] == ["1:lap:1", "2:lap:1"]


def test_replay_dead_letters_keeps_failed_rows_with_replay_error(tmp_path):
    event = raw_event(1, "v1/laps", "1:lap:1", "s1", {"lap_number": 1}, driver_number=1)
    path = tmp_path / "dead-letter.jsonl"
    _write_spool(path, event)

    result = asyncio.run(replay_dead_letters(path, AlwaysFailSink()))

    assert result.processed == 1
    assert result.submitted == 0
    assert result.failed == 1
    assert result.kept == 1
    [row] = [json.loads(line) for line in path.read_text().splitlines()]
    assert row["event"]["source_key"] == "1:lap:1"
    assert row["lastReplayError"] == "api still down"
    assert row["replayAttempts"] == 1


def test_replay_dead_letters_keeps_malformed_rows_and_compacts_successes(tmp_path):
    event = raw_event(1, "v1/laps", "1:lap:1", "s1", {"lap_number": 1}, driver_number=1)
    path = tmp_path / "dead-letter.jsonl"
    path.write_text("not-json\n" + _spool_line(event) + "\n", encoding="utf-8")
    sink = RecordingSink()

    result = asyncio.run(replay_dead_letters(path, sink))

    assert result.processed == 2
    assert result.invalid == 1
    assert result.submitted == 1
    assert result.kept == 1
    assert path.read_text(encoding="utf-8") == "not-json\n"
    assert [event.source_key for event in sink.events] == ["1:lap:1"]


def test_replay_dead_letters_dry_run_does_not_submit_or_rewrite(tmp_path):
    event = raw_event(1, "v1/laps", "1:lap:1", "s1", {"lap_number": 1}, driver_number=1)
    path = tmp_path / "dead-letter.jsonl"
    _write_spool(path, event)
    before = path.read_text(encoding="utf-8")
    sink = RecordingSink()

    result = asyncio.run(replay_dead_letters(path, sink, dry_run=True))

    assert result.dry_run is True
    assert result.processed == 1
    assert result.submitted == 0
    assert result.kept == 1
    assert sink.events == []
    assert path.read_text(encoding="utf-8") == before


class RecordingSink:
    def __init__(self):
        self.events = []

    async def submit(self, event):
        self.events.append(event)


class AlwaysFailSink:
    async def submit(self, _event):
        raise RuntimeError("api still down")


def _write_spool(path, *events):
    path.write_text("".join(_spool_line(event) + "\n" for event in events), encoding="utf-8")


def _spool_line(event):
    return json.dumps(
        {
            "failedAt": "2026-06-25T00:00:00Z",
            "error": "api down",
            "event": event.to_dict(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
