"""Incremental, packet-weighted telemetry90 measurements; no laps or targets.

Call one cursor per delivery lag with monotonically increasing integer-ns
cutoffs. The cursor reads only a future packet's header, never its payload,
until its cumulative archive availability is strictly before the query cutoff.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path

import numpy as np

from research.experiments.boundary_20260908.telemetry_pilot import packets

ROOT = Path(__file__).resolve().parents[4]
NS_PER_SECOND = 1_000_000_000
NS_PER_MS = 1_000_000
WINDOWS_SECONDS = (30, 90, 180)
DEPENDENCY_BINDINGS = {
    "research/experiments/boundary_20260908/telemetry_pilot/packets.py":
        "18e169a21abcb88dd7315f1be6bc5aa552d66033e6f6987a79313fcfa2736680",
    "research/experiments/boundary_20260908/telemetry_pilot/future_protocol_proposal.md":
        "b4b5fc382b9f3f9e3a26d1fc9348c95e254740c610a41cb32cd7a442c35da3a3",
}
CHANNELS = (("0", "rpm"), ("2", "speed"), ("3", "gear"),
            ("4", "throttle"), ("5", "brake"), ("45", "drs"))
CONTENT_SUFFIXES = (
    "speed_mean", "speed_packet_sd", "rpm_mean", "throttle_mean",
    "throttle_ge95_fraction", "brake_code100_fraction",
    "low_throttle_zero_brake_fraction", "gear_mean",
    "drs_codes_10_12_14_fraction",
)
QUALITY_SUFFIXES = (
    "packet_count", "availability_span_seconds", "newest_packet_age_seconds",
    "max_packet_or_boundary_gap_seconds", "mean_bundle_size",
    *(item for _, name in CHANNELS for item in (f"{name}_present_fraction", f"{name}_valid_fraction")),
    "throttle_code104_fraction", "brake_code104_fraction",
    "repeated_final_state_fraction", "joint_throttle_brake_valid_fraction",
)
FEATURE_NAMES = tuple(f"telemetry_w{window}_{name}" for window in WINDOWS_SECONDS
                      for name in (*CONTENT_SUFFIXES, *QUALITY_SUFFIXES))
CONTENT_INDICES = tuple(30 * block + i for block in range(3) for i in range(9))
QUALITY_INDICES = tuple(i for i in range(90) if i not in CONTENT_INDICES)
assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 90
assert len(CONTENT_INDICES) == 27 and len(QUALITY_INDICES) == 63


def dependency_bindings():
    """Check the committed measurement contract and immutable parser helpers."""
    for path, expected in DEPENDENCY_BINDINGS.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen telemetry dependency changed: {path}")
    return dict(DEPENDENCY_BINDINGS)


def _ns(value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("cutoff_ns must be an exact integer, not a float or boolean")
    return int(value)


def _mean(values):
    # Every measurement entering this helper is nonnegative. Normalizing by
    # its maximum bounds the sum by n and the final mean by that maximum;
    # summing value/n can still overflow for three float64-maximum values.
    if not values:
        return 0.0
    scale = max(values)
    return (math.fsum(value / scale for value in values) / len(values)) * scale if scale else 0.0


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _valid(channel, value):
    number = _number(value)
    if number is None or number < 0:
        return False
    if channel == "4":
        return number <= 100
    if channel == "3":
        return number.is_integer() and number <= 8
    if channel == "5":
        return number in (0, 100)
    if channel == "45":
        return number.is_integer()
    return True


def _identity(channels):
    # JSON spellings preserve null/missing, booleans and int/float distinctions;
    # native Python equality would wrongly identify False and integer zero.
    return tuple((channel in channels, json.dumps(channels.get(channel),
        sort_keys=True, separators=(",", ":"), allow_nan=False)) for channel, _ in CHANNELS)


@dataclass(frozen=True)
class _PacketSummary:
    availability_ns: int
    sequence: int
    bundle_size: int
    content: tuple[float | None, ...]
    fractions: tuple[float, ...]
    final_state: tuple


@dataclass(frozen=True)
class FeatureSnapshot:
    values: np.ndarray
    supported: bool
    provenance: dict

    def as_dict(self):
        return dict(zip(FEATURE_NAMES, self.values.tolist(), strict=True))


def _driver(driver_id):
    if not isinstance(driver_id, str) or not driver_id:
        raise ValueError("driver_id must be a nonempty string")
    return driver_id


def empty_snapshot(driver_id: str, *, cutoff_ns: int) -> FeatureSnapshot:
    """Pure missing-stream representation; never substitutes for a corrupt stream."""
    driver_id, cutoff_ns = _driver(driver_id), _ns(cutoff_ns)
    values = np.array([x for seconds in WINDOWS_SECONDS for x in _window([], cutoff_ns, seconds)], dtype=float)
    values.setflags(write=False)
    return FeatureSnapshot(values, False, _provenance(driver_id, cutoff_ns, [], 0, None, None))


def _provenance(driver_id, cutoff_ns, rows, consumed, sequence, availability):
    return {
        "driver_id": driver_id, "cutoff_ns": cutoff_ns,
        "clock": "physical_archive_cummax_milliseconds_not_certified_client_receipt",
        "history_packet_count_180s": len(rows),
        "oldest_driver_packet_availability_ns": rows[0].availability_ns if rows else None,
        "newest_driver_packet_availability_ns": rows[-1].availability_ns if rows else None,
        "driver_packet_sequences": [row.sequence for row in rows],
        "consumed_packets": consumed,
        "last_consumed_sequence": sequence,
        "last_consumed_availability_ns": availability,
        "measurement_contract_sha256": DEPENDENCY_BINDINGS[
            "research/experiments/boundary_20260908/telemetry_pilot/future_protocol_proposal.md"],
    }


def _summarize_packet(availability_ns, sequence, entries):
    """One driver's entries in one atomic packet; no source-UTC use."""
    count = len(entries)
    usable = {channel: [float(row[channel]) for row in entries
                        if channel in row and _valid(channel, row[channel])]
              for channel, _ in CHANNELS}
    fractions = []
    for channel, _ in CHANNELS:
        fractions.extend((sum(channel in row for row in entries)/count,
                          len(usable[channel])/count))
    for channel in ("4", "5"):
        fractions.append(sum(_number(row.get(channel)) == 104 for row in entries)/count)
    pairs = [(float(row["4"]), float(row["5"])) for row in entries
             if _valid("4", row.get("4")) and _valid("5", row.get("5"))]
    fractions.append(len(pairs)/count)
    content = (
        _mean(usable["2"]) if usable["2"] else None,
        None,  # speed dispersion is calculated over packet means at query time
        _mean(usable["0"]) if usable["0"] else None,
        _mean(usable["4"]) if usable["4"] else None,
        _mean([float(x >= 95) for x in usable["4"]]) if usable["4"] else None,
        _mean([float(x == 100) for x in usable["5"]]) if usable["5"] else None,
        _mean([float(throttle <= 5 and brake == 0) for throttle, brake in pairs]) if pairs else None,
        _mean(usable["3"]) if usable["3"] else None,
        _mean([float(x in (10, 12, 14)) for x in usable["45"]]) if usable["45"] else None,
    )
    return _PacketSummary(availability_ns, sequence, count, content, tuple(fractions), _identity(entries[-1]))


def _window(rows, cutoff_ns, seconds):
    left = cutoff_ns-seconds*NS_PER_SECOND
    rows = [row for row in rows if row.availability_ns >= left]
    if not rows:
        return [0.0]*9 + [0.0, 0.0, float(seconds), float(seconds), 0.0] + [0.0]*16
    n = len(rows)
    content = [_mean([row.content[i] for row in rows if row.content[i] is not None]) for i in range(9)]
    speed = [row.content[0] for row in rows if row.content[0] is not None]
    if speed:
        scale = max(speed)
        # Center in normalized units too: centering tiny raw values around an
        # already underflowed mean otherwise overstates their population SD.
        normalized = [x/scale for x in speed] if scale else [0.0]*len(speed)
        center = _mean(normalized)
        content[1] = scale * math.sqrt(_mean([(x-center)**2 for x in normalized]))
    edges = [left, *(row.availability_ns for row in rows), cutoff_ns]
    quality = [float(n), (rows[-1].availability_ns-rows[0].availability_ns)/NS_PER_SECOND,
        (cutoff_ns-rows[-1].availability_ns)/NS_PER_SECOND,
        max(b-a for a,b in zip(edges, edges[1:]))/NS_PER_SECOND,
        _mean([float(row.bundle_size) for row in rows]),
        *[_mean([row.fractions[i] for row in rows]) for i in range(14)],
        sum(a.final_state == b.final_state for a,b in zip(rows, rows[1:]))/(n-1) if n > 1 else 0.0,
        _mean([row.fractions[14] for row in rows])]
    assert len(content) == 9 and len(quality) == 21
    return content + quality


class TelemetryCursor:
    """Streaming physical-prefix reader with at most180 seconds of driver state.

    Cutoffs are already delivery-lag adjusted. Negative cutoffs are empty and
    unsupported. A cursor cannot move backwards; use separate cursors for0/2s
    lag. Public snapshots contain past provenance only, not pending headers or
    EOF status, so appending future data cannot alter an earlier snapshot.
    """

    def __init__(self, path):
        dependency_bindings()
        self.path = Path(path)
        self._file = self.path.open("rb")
        prefix = self._file.read(3)
        if prefix != b"\xef\xbb\xbf":
            self._file.seek(0)
        self._pending = None
        self._next_sequence = 0
        self._highwater_ms = -1
        self._history = {}
        self._last_cutoff_ns = None
        self._last_consumed_sequence = None
        self._last_consumed_availability_ns = None
        self._consumed_packets = 0
        self._closed = False
        self._failed = None
        self._eof = False

    def close(self):
        if not self._closed:
            self._file.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def retained_packet_count(self):
        return sum(len(rows) for rows in self._history.values())

    def _read_header(self):
        if self._pending is not None or self._eof:
            return
        while True:
            prefix = self._file.readline(12)
            if not prefix:
                self._eof = True
                return
            sequence = self._next_sequence
            self._next_sequence += 1
            if not prefix.strip():
                if not prefix.endswith(b"\n"):
                    rest = self._file.readline(packets.MAX_RECORD_BYTES-12+1)
                    if len(prefix)+len(rest) > packets.MAX_RECORD_BYTES or rest.strip():
                        raise ValueError("invalid or oversized blank record")
                continue
            recorded = packets.clock_milliseconds(prefix)
            self._highwater_ms = max(self._highwater_ms, recorded)
            self._pending = (sequence, recorded, self._highwater_ms*NS_PER_MS)
            return

    def _prune(self, instant_ns):
        left = instant_ns-180*NS_PER_SECOND
        for driver in list(self._history):
            rows = self._history[driver]
            while rows and rows[0].availability_ns < left:
                rows.popleft()
            if not rows:
                del self._history[driver]

    def _advance(self, cutoff_ns):
        if cutoff_ns < 0:
            return
        while True:
            self._read_header()
            if self._pending is None or self._pending[2] >= cutoff_ns:
                break
            sequence, recorded_ms, availability_ns = self._pending
            body = self._file.readline(packets.MAX_RECORD_BYTES-12+1)
            if len(body)+12 > packets.MAX_RECORD_BYTES:
                raise ValueError("record exceeds byte limit")
            payload = packets.decode_payload(body)
            groups = {}
            for row in packets.iter_car_rows({"sequence": sequence, "recorded_ms": recorded_ms,
                    "available_ms": availability_ns//NS_PER_MS, "payload": payload}):
                groups.setdefault(row["driver"], []).append(row["channels"])
            for driver, entries in groups.items():
                summary = _summarize_packet(availability_ns, sequence, entries)
                self._history.setdefault(driver, deque()).append(summary)
            self._prune(availability_ns)
            self._pending = None
            self._last_consumed_sequence = sequence
            self._last_consumed_availability_ns = availability_ns
            self._consumed_packets += 1
        self._prune(cutoff_ns)

    def query(self, driver_id: str, *, cutoff_ns: int) -> FeatureSnapshot:
        cutoff_ns = _ns(cutoff_ns)
        _driver(driver_id)
        if self._closed:
            raise ValueError("telemetry cursor is closed")
        if self._failed is not None:
            raise packets.PacketError("telemetry cursor previously failed") from self._failed
        if self._last_cutoff_ns is not None and cutoff_ns < self._last_cutoff_ns:
            raise ValueError("cutoff_ns must be nondecreasing; use one cursor per delivery lag")
        try:
            self._advance(cutoff_ns)
        except Exception as exc:
            self._failed = exc
            raise packets.PacketError(f"{self.path.name}: before cutoff_ns={cutoff_ns}: {exc}") from exc
        self._last_cutoff_ns = cutoff_ns
        rows = list(self._history.get(driver_id, ()))
        values = np.array([value for seconds in WINDOWS_SECONDS for value in _window(rows, cutoff_ns, seconds)], dtype=float)
        if values.shape != (90,) or not np.isfinite(values).all():
            raise ValueError("telemetry90 encoding must contain exactly90 finite values")
        # The gate uses integer ns rather than feature floats at equality edges.
        recent = [row for row in rows if row.availability_ns >= cutoff_ns-30*NS_PER_SECOND]
        supported = bool(len(recent) >= 6
            and recent[-1].availability_ns-recent[0].availability_ns >= 24*NS_PER_SECOND
            and cutoff_ns-recent[-1].availability_ns <= 5*NS_PER_SECOND)
        values.setflags(write=False)
        return FeatureSnapshot(values, supported, _provenance(driver_id, cutoff_ns, rows,
            self._consumed_packets, self._last_consumed_sequence, self._last_consumed_availability_ns))
