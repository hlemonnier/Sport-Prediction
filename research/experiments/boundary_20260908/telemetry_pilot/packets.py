"""Bounded raw CarData.z diagnostics using physical archive availability.

Archive-prefix clocks are not certified client receipt times or exact sample
times. Source Utc is retained for quality checks only; no session-origin estimate,
lap table, interpolation, feature fitting, or target is used here.
"""
from __future__ import annotations

import argparse
import base64
import binascii
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Iterator
import zlib

MAX_RECORD_BYTES = 2 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 8 * 1024 * 1024
CHANNEL_NAMES = {"0": "RPM", "2": "Speed", "3": "nGear", "4": "Throttle",
                 "5": "Brake", "45": "DRS"}
_CLOCK = re.compile(rb"([0-9]{2}):([0-5][0-9]):([0-5][0-9])\.([0-9]{3})\Z")
_UTC = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2})(?:\.([0-9]{1,9}))?Z\Z")


class PacketError(ValueError):
    """A malformed record makes the requested stream/prefix invalid."""


def _integer(value, name, *, positive=False):
    if type(value) is not int or value < int(positive):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def clock_milliseconds(value: str | bytes) -> int:
    """Parse exactly HH:MM:SS.mmm without floating-point rounding."""
    if isinstance(value, str):
        try:
            value = value.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("invalid archive clock") from exc
    if not isinstance(value, bytes):
        raise ValueError("archive clock must be text or bytes")
    match = _CLOCK.fullmatch(value)
    if match is None:
        raise ValueError("expected 12-byte HH:MM:SS.mmm archive clock")
    hour, minute, second, millis = map(int, match.groups())
    return ((hour * 60 + minute) * 60 + second) * 1000 + millis


def utc_nanoseconds(value) -> int | None:
    """Exact source-UTC diagnostic; None means missing/invalid, never a join key."""
    if not isinstance(value, str):
        return None
    match = _UTC.fullmatch(value)
    if match is None:
        return None
    try:
        whole = datetime.strptime(match[1], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    elapsed = whole - datetime(1970, 1, 1, tzinfo=timezone.utc)
    seconds = elapsed.days * 86400 + elapsed.seconds
    return seconds * 1_000_000_000 + int((match[2] or "").ljust(9, "0"))


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("nonfinite JSON constant")


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("nonfinite JSON number")
    return parsed


def _json(value):
    return json.loads(value, object_pairs_hook=_object, parse_constant=_nonfinite,
                      parse_float=_finite_float)


def decode_payload(encoded: bytes, *, max_decompressed_bytes=MAX_DECOMPRESSED_BYTES) -> dict:
    """Strict JSON-string/base64/raw-DEFLATE decoding with a hard output cap."""
    _integer(max_decompressed_bytes, "max_decompressed_bytes", positive=True)
    quoted = _json(encoded.decode("utf-8-sig"))
    if not isinstance(quoted, str):
        raise ValueError("CarData.z envelope must be a JSON string")
    compressed = base64.b64decode(quoted.encode("ascii"), validate=True)
    decoder = zlib.decompressobj(-zlib.MAX_WBITS)
    decoded = decoder.decompress(compressed, max_decompressed_bytes + 1)
    if len(decoded) > max_decompressed_bytes or decoder.unconsumed_tail:
        raise ValueError("decompressed packet exceeds byte limit")
    if not decoder.eof:
        raise ValueError("truncated raw-DEFLATE packet")
    if decoder.unused_data:
        raise ValueError("trailing data after raw-DEFLATE packet")
    payload = _json(decoded.decode("utf-8-sig"))
    if not isinstance(payload, dict) or not isinstance(payload.get("Entries"), list):
        raise ValueError("payload must contain an Entries list")
    for entry in payload["Entries"]:
        if not isinstance(entry, dict):
            raise ValueError("each Entries item must be an object")
        # Missing source/channel fields are retained, not silently zero-filled.
        cars = entry.get("Cars")
        if cars is None:
            continue
        if not isinstance(cars, dict):
            raise ValueError("Cars must be an object when present")
        for driver, car in cars.items():
            if not driver or not isinstance(car, dict):
                raise ValueError("driver keys and car objects must be valid")
            if car.get("Channels") is not None and not isinstance(car["Channels"], dict):
                raise ValueError("Channels must be an object when present")
    return payload


def iter_packets(path, *, before_ms=None, stats=None,
                 max_record_bytes=MAX_RECORD_BYTES,
                 max_decompressed_bytes=MAX_DECOMPRESSED_BYTES) -> Iterator[dict]:
    """Yield validated packets in physical order, strictly before an optional clock.

    ``available_ms`` is the cumulative maximum of physical prefix clocks. At the
    first availability >= cutoff the generator stops before payload validation;
    even a subsequently regressing record cannot be moved into the past. An
    invalid clock encountered earlier fails closed because its position on the
    requested clock is unknown. Only full exhaustion validates the full stream.
    """
    if before_ms is not None:
        _integer(before_ms, "before_ms")
    _integer(max_record_bytes, "max_record_bytes", positive=True)
    _integer(max_decompressed_bytes, "max_decompressed_bytes", positive=True)
    if max_record_bytes < 13:
        raise ValueError("max_record_bytes must allow a clock and payload")
    stats = {} if stats is None else stats
    stats.update(records=0, physical_lines_examined=0, blank_lines=0,
                 timestamp_regressions=0, adjacent_duplicate_recorded_clocks=0,
                 duplicate_availability_clocks=0, first_recorded_ms=None,
                 last_recorded_ms=None, first_available_ms=None,
                 last_available_ms=None, max_availability_gap_ms=0,
                 complete=False, stream_input_valid=None,
                 prefix_input_valid=False, stopped_at_cutoff=False)
    highwater, previous = -1, None
    sequence = 0
    try:
        with Path(path).open("rb") as stream:
            while True:
                line = stream.readline(max_record_bytes + 1)
                if not line:
                    stats.update(complete=True, stream_input_valid=True, prefix_input_valid=True)
                    return
                if sequence == 0:
                    line = line.removeprefix(b"\xef\xbb\xbf")
                stats["physical_lines_examined"] += 1
                current_sequence = sequence
                sequence += 1
                if not line.strip():
                    if len(line) > max_record_bytes:
                        raise ValueError("record exceeds byte limit")
                    stats["blank_lines"] += 1
                    continue
                recorded = clock_milliseconds(line[:12])
                available = max(highwater, recorded)
                if before_ms is not None and available >= before_ms:
                    stats.update(prefix_input_valid=True, stopped_at_cutoff=True)
                    return
                if len(line) > max_record_bytes:
                    raise ValueError("record exceeds byte limit")
                payload = decode_payload(line[12:], max_decompressed_bytes=max_decompressed_bytes)
                packet = {"sequence": current_sequence, "recorded_ms": recorded,
                          "available_ms": available, "timestamp_regressed": recorded < highwater,
                          "payload": payload}
                stats["records"] += 1
                stats["timestamp_regressions"] += int(recorded < highwater)
                stats["adjacent_duplicate_recorded_clocks"] += int(recorded == previous)
                stats["duplicate_availability_clocks"] += int(available == highwater)
                if stats["first_recorded_ms"] is None:
                    stats["first_recorded_ms"], stats["first_available_ms"] = recorded, available
                if highwater >= 0:
                    stats["max_availability_gap_ms"] = max(stats["max_availability_gap_ms"], available-highwater)
                stats["last_recorded_ms"], stats["last_available_ms"] = recorded, available
                previous, highwater = recorded, available
                yield packet
    except (ValueError, UnicodeError, binascii.Error, zlib.error, RecursionError, OSError) as exc:
        stats.update(prefix_input_valid=False, stream_input_valid=False, complete=False,
                     failure={"sequence": max(sequence-1, 0), "reason": str(exc)})
        raise PacketError(f"{Path(path).name}: physical record {max(sequence-1, 0)}: {exc}") from exc


def iter_car_rows(packet) -> Iterator[dict]:
    """Flatten for diagnostics while retaining atomic bundle provenance and raw values."""
    for index, entry in enumerate(packet["payload"]["Entries"]):
        for driver, car in (entry.get("Cars") or {}).items():
            yield {"packet_sequence": packet["sequence"], "entry_index": index,
                   "recorded_ms": packet["recorded_ms"], "available_ms": packet["available_ms"],
                   "utc": entry.get("Utc"), "driver": driver,
                   "channels_present": isinstance(car.get("Channels"), dict),
                   "channels": dict(car.get("Channels") or {})}


def summarize(path, *, before_ms=None, max_record_bytes=MAX_RECORD_BYTES,
              max_decompressed_bytes=MAX_DECOMPRESSED_BYTES) -> dict:
    """One-pass, bounded-memory quality inventory; does not aggregate model features."""
    stats, channel_counts, drivers = {}, {}, {}
    counts = Counter({key: 0 for key in ("entries", "empty_packets", "bundled_packets",
        "max_entries_per_packet", "entries_missing_utc", "entries_invalid_utc",
        "entries_missing_cars", "car_rows", "car_rows_missing_channels")})
    prior_utc = {}
    for packet in iter_packets(path, before_ms=before_ms, stats=stats,
                               max_record_bytes=max_record_bytes,
                               max_decompressed_bytes=max_decompressed_bytes):
        entries = packet["payload"]["Entries"]
        counts["entries"] += len(entries)
        counts["empty_packets"] += int(not entries)
        counts["bundled_packets"] += int(len(entries) > 1)
        counts["max_entries_per_packet"] = max(counts["max_entries_per_packet"], len(entries))
        for entry in entries:
            utc = utc_nanoseconds(entry.get("Utc"))
            counts["entries_missing_utc"] += int(entry.get("Utc") is None)
            counts["entries_invalid_utc"] += int(entry.get("Utc") is not None and utc is None)
            counts["entries_missing_cars"] += int(entry.get("Cars") is None)
        for row in iter_car_rows(packet):
            counts["car_rows"] += 1
            counts["car_rows_missing_channels"] += int(not row["channels_present"])
            driver = row["driver"]
            d = drivers.setdefault(driver, {"car_rows": 0, "first_available_ms": row["available_ms"],
                "last_available_ms": row["available_ms"], "max_availability_gap_ms": 0,
                "source_utc_regressions": 0, "source_utc_duplicates": 0,
                "source_utc_valid": 0, "source_utc_missing_or_invalid": 0})
            d["car_rows"] += 1
            d["max_availability_gap_ms"] = max(d["max_availability_gap_ms"], row["available_ms"]-d["last_available_ms"])
            d["last_available_ms"] = row["available_ms"]
            utc = utc_nanoseconds(row["utc"])
            d["source_utc_valid"] += int(utc is not None)
            d["source_utc_missing_or_invalid"] += int(utc is None)
            if utc is not None:
                if driver in prior_utc:
                    d["source_utc_regressions"] += int(utc < prior_utc[driver])
                    d["source_utc_duplicates"] += int(utc == prior_utc[driver])
                prior_utc[driver] = utc
            for channel, value in row["channels"].items():
                c = channel_counts.setdefault(channel, {"present": 0, "null": 0, "zero": 0,
                    "boolean": 0, "nonnumeric": 0, "minimum": None, "maximum": None})
                c["present"] += 1
                c["null"] += int(value is None)
                c["boolean"] += int(isinstance(value, bool))
                numeric = isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))
                c["nonnumeric"] += int(value is not None and not numeric)
                if numeric:
                    c["zero"] += int(value == 0)
                    c["minimum"] = value if c["minimum"] is None else min(c["minimum"], value)
                    c["maximum"] = value if c["maximum"] is None else max(c["maximum"], value)
    for channel in sorted(set(CHANNEL_NAMES) | set(channel_counts)):
        c = channel_counts.setdefault(channel, {"present": 0, "null": 0, "zero": 0,
            "boolean": 0, "nonnumeric": 0, "minimum": None, "maximum": None})
        c["absent"] = counts["car_rows"]-c["present"]
        c["known_name"] = CHANNEL_NAMES.get(channel)
    return {"schema": "raw_car_data_prefix_diagnostics_v1", "path": str(Path(path)),
            "before_ms": before_ms, "packet_validation": stats, "counts": dict(counts),
            "channels": dict(sorted(channel_counts.items())), "drivers": dict(sorted(drivers.items())),
            "clock_contract": "physical archive order; cummax recorded prefix milliseconds; strict before cutoff",
            "limitations": ["archive clock is not historical client receipt or precise sample time",
                "Utc is diagnostic only; no full-session t0_date or lap alignment",
                "bundled rows are not independent or evenly spaced observations",
                "missing channels including DRS stay missing; unknown DRS meanings are not inferred"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--before-ms", type=int)
    args = parser.parse_args()
    print(json.dumps(summarize(args.path, before_ms=args.before_ms), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
