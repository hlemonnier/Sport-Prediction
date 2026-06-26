"""Dead-letter replay and compaction tools for live OpenF1 ingestion."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .live_ingestor import EventSink, OpenF1ApiEventSink
from .schemas import F1Event, JsonObject
from .time import utc_now_iso


@dataclass(slots=True)
class DeadLetterReplayResult:
    path: str
    dry_run: bool
    processed: int = 0
    submitted: int = 0
    kept: int = 0
    invalid: int = 0
    failed: int = 0
    skipped: int = 0

    def to_dict(self) -> JsonObject:
        return asdict(self)


async def replay_dead_letters(
    path: str | Path,
    sink: EventSink,
    *,
    dry_run: bool = False,
    limit: int | None = None,
) -> DeadLetterReplayResult:
    """Submit recoverable dead-letter rows and compact successes out of JSONL."""

    spool_path = Path(path)
    result = DeadLetterReplayResult(path=str(spool_path), dry_run=dry_run)
    if not spool_path.exists():
        return result

    lines = spool_path.read_text(encoding="utf-8").splitlines()
    kept_lines: list[str] = []
    max_rows = max(0, limit) if limit is not None else None

    for raw_line in lines:
        if max_rows is not None and result.processed >= max_rows:
            result.skipped += 1
            kept_lines.append(raw_line)
            continue

        result.processed += 1
        try:
            record = _parse_dead_letter_line(raw_line)
            event = event_from_dead_letter_record(record)
        except Exception:
            result.invalid += 1
            kept_lines.append(raw_line)
            continue

        if dry_run:
            kept_lines.append(raw_line)
            continue

        try:
            await sink.submit(event)
            result.submitted += 1
        except Exception as exc:
            result.failed += 1
            kept_lines.append(_serialize_retry_failure(record, error=str(exc)))

    result.kept = len(kept_lines)
    if not dry_run:
        _replace_spool(spool_path, kept_lines)
    return result


def event_from_dead_letter_record(record: JsonObject) -> F1Event:
    event_payload = record.get("event")
    if not isinstance(event_payload, dict):
        raise ValueError("dead-letter record must contain an event object")
    return F1Event.from_record(event_payload)


def _parse_dead_letter_line(raw_line: str) -> JsonObject:
    record = json.loads(raw_line)
    if not isinstance(record, dict):
        raise ValueError("dead-letter line must be a JSON object")
    return record


def _serialize_retry_failure(record: JsonObject, *, error: str) -> str:
    updated = dict(record)
    updated["lastReplayAt"] = utc_now_iso()
    updated["lastReplayError"] = error
    updated["replayAttempts"] = _optional_int(updated.get("replayAttempts")) or 0
    updated["replayAttempts"] += 1
    return _json_line(updated)


def _replace_spool(path: Path, kept_lines: list[str]) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    if kept_lines:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
        temp_path.replace(path)
        return

    temp_path.unlink(missing_ok=True)
    path.unlink(missing_ok=True)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _json_line(value: JsonObject) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


async def _replay_from_args(args: argparse.Namespace) -> DeadLetterReplayResult:
    sink = OpenF1ApiEventSink(args.api_url, timeout_seconds=args.timeout_seconds)
    return await replay_dead_letters(args.path, sink, dry_run=args.dry_run, limit=args.limit)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay F1 live-ingest dead-letter events.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    replay = subcommands.add_parser("replay", help="submit recoverable events and compact successes")
    replay.add_argument(
        "--path",
        default=os.environ.get("OPENF1_DEAD_LETTER_PATH") or "data/raw/f1/openf1-dead-letter.jsonl",
        help="dead-letter JSONL path",
    )
    replay.add_argument(
        "--api-url",
        default=os.environ.get("F1_PLATFORM_API_URL", "http://127.0.0.1:8001"),
        help="F1 platform API base URL",
    )
    replay.add_argument("--timeout-seconds", type=float, default=10.0, help="API submit timeout")
    replay.add_argument("--limit", type=int, default=None, help="maximum rows to examine")
    replay.add_argument("--dry-run", action="store_true", help="validate rows without submitting or rewriting")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "replay":
        parser.error(f"unknown command: {args.command}")

    result = asyncio.run(_replay_from_args(args))
    print(json.dumps(result.to_dict(), sort_keys=True))
    if result.failed or result.invalid:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
