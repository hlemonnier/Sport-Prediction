"""Causal 128x34 packet sequences; no lap table, outcome, UTC or fitted scaler.

Suggested commit: research(f1-live): encode causal packet order with matched controls
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

from research.experiments.boundary_20260908.telemetry import features as frozen
from research.experiments.boundary_20260908.telemetry_pilot import packets

ROOT = Path(__file__).resolve().parents[4]
MAX_TOKENS = 128
WINDOW_SECONDS = 180
NS_PER_SECOND = 1_000_000_000
SEED = 20260908
CONTENT_POSITIONS = (0, 2, 3, 4, 5, 6, 7, 8)
CONTENT_NAMES = tuple(frozen.CONTENT_SUFFIXES[i] for i in CONTENT_POSITIONS)
QUALITY_NAMES = (
    *(item for _, name in frozen.CHANNELS for item in (f"{name}_present_fraction", f"{name}_valid_fraction")),
    "throttle_code104_fraction", "brake_code104_fraction", "joint_throttle_brake_valid_fraction",
)
TOKEN_NAMES = (*CONTENT_NAMES, *(name+"_valid" for name in CONTENT_NAMES), *QUALITY_NAMES,
               "log_bundle_size_scaled", "previous_driver_packet_gap_scaled", "padding")
CONTENT_INDICES = tuple(range(8))
VALIDITY_INDICES = tuple(range(8, 16))
QUALITY_INDICES = tuple(range(16, 31))
BUNDLE_INDEX, GAP_INDEX, PADDING_INDEX = 31, 32, 33
SHUFFLE_INDICES = tuple(range(32))
FIXED_INDICES = (GAP_INDEX, PADDING_INDEX)
CONTENT_SCALES = (350., 15000., 100., 1., 1., 1., 8., 1.)
DEPENDENCY_BINDINGS = {
    "research/experiments/boundary_20260908/telemetry/features.py":
        "7354b6675a4725d311c5bbf11f329e615118abf33fef5ef4bf4ee69474c1371f",
    "research/experiments/boundary_20260908/telemetry_next/idea.md":
        "3ef0f7545c0c58cff0fa61f6d007e97e7e8add75b796068f0fbf146b5ba8aed3",
    **frozen.DEPENDENCY_BINDINGS,
}
assert len(TOKEN_NAMES) == len(set(TOKEN_NAMES)) == 34


def dependency_bindings():
    for name, expected in DEPENDENCY_BINDINGS.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen token dependency changed: "+name)
    return dict(DEPENDENCY_BINDINGS)


def _integer(value, name, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(name+" must be an exact integer")
    value = int(value)
    if minimum is not None and value < minimum: raise ValueError(name+" is below its allowed minimum")
    if not -(2**63) <= value < 2**63: raise ValueError(name+" does not fit int64")
    return value


def _driver(value):
    if not isinstance(value, str) or not value: raise ValueError("driver_id must be a nonempty string")
    return value


def _readonly(values, dtype):
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PacketToken:
    driver_id: str
    availability_ns: int
    packet_sequence: int
    previous_availability_ns: int | None
    values: np.ndarray

    @property
    def sequence(self): return self.packet_sequence


@dataclass(frozen=True)
class TokenSnapshot:
    values: np.ndarray
    supported: bool
    source_availability_ns: np.ndarray
    source_sequences: np.ndarray
    previous_availability_ns: np.ndarray
    provenance: dict


def reduce_packet(driver_id, availability_ns, packet_sequence, entries, *, previous_availability_ns=None):
    """Reduce an entire atomic driver packet using the frozen channel semantics.

    The true predecessor clock remains metadata even if outside the 180s window.
    First-ever packet gap is zero; later equal-clock packets also have zero gap.
    Neither value is an invented sample interval or physical sampling rate.
    """
    driver_id = _driver(driver_id)
    available = _integer(availability_ns, "availability_ns", minimum=0)
    sequence = _integer(packet_sequence, "packet_sequence", minimum=0)
    previous = None if previous_availability_ns is None else _integer(previous_availability_ns, "previous_availability_ns", minimum=0)
    if previous is not None and previous > available: raise ValueError("Driver predecessor is in the future")
    if not isinstance(entries, (list, tuple)) or not entries or not all(isinstance(x, dict) for x in entries):
        raise ValueError("An atomic driver packet needs nonempty channel dictionaries")
    summary = frozen._summarize_packet(available, sequence, entries)
    values = [summary.content[i] for i in CONTENT_POSITIONS]
    content = [0. if value is None else min(float(value)/scale, 5.) for value, scale in zip(values, CONTENT_SCALES)]
    validity = [float(value is not None) for value in values]
    bundle = min(math.log(summary.bundle_size)/math.log(18), 5.)
    gap = 0. if previous is None else min((available-previous)/(5*NS_PER_SECOND), 5.)
    encoded = _readonly([*content, *validity, *summary.fractions, bundle, gap, 0.], np.float32)
    if encoded.shape != (34,) or not np.isfinite(encoded).all() or (encoded < 0).any() or (encoded > 5).any():
        raise ValueError("Packet token must contain exactly 34 bounded finite values")
    return PacketToken(driver_id, available, sequence, previous, encoded)


def _groups(packet):
    groups = {}
    for row in packets.iter_car_rows(packet):
        groups.setdefault(row["driver"], []).append(row["channels"])
    return groups


def iter_packet_tokens(path, *, stats=None):
    """Read a completed stream for SSL only after the caller's source/data lock.

    Rows are emitted in physical packet order, then lexical driver order. The
    whole current packet is decoded and reduced atomically before any row yields.
    This function has no label or year access: the caller must restrict sources
    to its frozen fitting period. Exhaustion validates the entire raw stream.
    """
    dependency_bindings();previous = {}
    for packet in packets.iter_packets(path, stats=stats):
        available = packet["available_ms"]*1_000_000;groups = _groups(packet)
        current = [reduce_packet(driver, available, packet["sequence"], groups[driver],
            previous_availability_ns=previous.get(driver)) for driver in sorted(groups)]
        for token in current: previous[token.driver_id] = token.availability_ns
        yield from current


def context_from_tokens(records, *, cutoff_ns, driver_id):
    """Select history by metadata before inspecting token values.

    Require strictly earlier availability; equal-clock rows are excluded. Input
    may include future records, whose measurement arrays are never inspected.
    The uncapped recent history determines support, while only the last128 rows
    enter the numeric context. Precomputed token gaps must not be recomputed at
    the selected-window boundary.
    """
    cutoff = _integer(cutoff_ns, "cutoff_ns");driver = _driver(driver_id)
    left = cutoff-WINDOW_SECONDS*NS_PER_SECOND
    selected = []
    for token in records:
        available = _integer(token.availability_ns, "availability_ns", minimum=0)
        if available >= cutoff or available < left: continue
        if token.driver_id != driver: continue
        sequence = _integer(token.packet_sequence, "packet_sequence", minimum=0)
        if selected and (available < selected[-1].availability_ns or sequence <= selected[-1].packet_sequence):
            raise ValueError("Own packet history must follow physical sequence and cumulative clock order")
        selected.append(token)
    recent = [row for row in selected if row.availability_ns >= cutoff-30*NS_PER_SECOND]
    supported = bool(len(recent) >= 6 and recent[-1].availability_ns-recent[0].availability_ns >= 24*NS_PER_SECOND
                     and cutoff-recent[-1].availability_ns <= 5*NS_PER_SECOND)
    kept = selected[-MAX_TOKENS:];offset = MAX_TOKENS-len(kept)
    values = np.zeros((MAX_TOKENS, 34), dtype=np.float32);values[:offset, PADDING_INDEX] = 1.
    availability = np.full(MAX_TOKENS, -1, dtype=np.int64)
    sequences = np.full(MAX_TOKENS, -1, dtype=np.int64)
    previous = np.full(MAX_TOKENS, -1, dtype=np.int64)
    for i, token in enumerate(kept, start=offset):
        numeric = np.asarray(token.values)
        if numeric.shape != (34,) or not np.isfinite(numeric).all() or (numeric < 0).any() or (numeric > 5).any() or numeric[PADDING_INDEX] != 0:
            raise ValueError("Real packet token has invalid shape, values or padding")
        prev = token.previous_availability_ns
        if prev is not None:
            prev = _integer(prev, "previous_availability_ns", minimum=0)
            if prev > token.availability_ns: raise ValueError("Packet predecessor is in the future")
        expected_gap = 0. if prev is None else min((token.availability_ns-prev)/(5*NS_PER_SECOND), 5.)
        if numeric[GAP_INDEX] != np.float32(expected_gap): raise ValueError("Token gap does not match causal predecessor metadata")
        values[i] = numeric;availability[i] = token.availability_ns;sequences[i] = token.packet_sequence
        previous[i] = -1 if prev is None else prev
    provenance = {"driver_id": driver, "cutoff_ns": cutoff,
        "clock": "physical_archive_cummax_milliseconds_not_certified_client_receipt",
        "history_packet_count_180s": len(selected), "context_packet_count": len(kept),
        "support_packet_count_30s": len(recent), "older_in_window_packets_omitted": max(0, len(selected)-MAX_TOKENS),
        "driver_packet_sequences": [row.packet_sequence for row in kept],
        "oldest_context_availability_ns": kept[0].availability_ns if kept else None,
        "newest_context_availability_ns": kept[-1].availability_ns if kept else None,
        "last_admitted_driver_packet_sequence": kept[-1].packet_sequence if kept else None}
    return TokenSnapshot(_readonly(values, np.float32), supported, _readonly(availability, np.int64),
                         _readonly(sequences, np.int64), _readonly(previous, np.int64), provenance)


def empty_snapshot(driver_id, *, cutoff_ns):
    return context_from_tokens((), cutoff_ns=cutoff_ns, driver_id=driver_id)


def permutation_key(event_key, driver_id, packet_sequence):
    event = _integer(event_key, "event_key", minimum=0)
    sequence = None if packet_sequence is None else _integer(packet_sequence, "packet_sequence", minimum=0)
    return json.dumps([event, _driver(driver_id), sequence], separators=(",", ":"), ensure_ascii=True)


def permute_context(values, *, key, seed=SEED):
    """Move whole measurement/quality bundles; keep newest, gaps and pads fixed.

    Key must identify only event/driver/last-admitted physical packet sequence.
    Never pass a whole-file hash or query's future data. Different prefix keys
    avoid teaching the control a single globally invertible permutation.
    """
    seed = _integer(seed, "seed", minimum=0)
    if not isinstance(key, str) or not key: raise ValueError("A nonempty past-only prefix key is required")
    value = np.asarray(values)
    if value.shape != (MAX_TOKENS, 34) or not np.isfinite(value).all() or (value < 0).any() or (value > 5).any():
        raise ValueError("Expected a finite bounded 128x34 context")
    pad = value[:, PADDING_INDEX]
    if not np.isin(pad, (0, 1)).all() or np.any(np.diff(pad) > 0): raise ValueError("Padding must be binary and on the left")
    padded = pad == 1
    if np.any(value[padded, :PADDING_INDEX] != 0): raise ValueError("Padded coordinates must otherwise be zero")
    positions = np.flatnonzero(~padded)[:-1]
    digest = hashlib.sha256(json.dumps([seed, key], separators=(",", ":")).encode()).digest()
    # Hash sorting is deterministic across NumPy/Python RNG implementations.
    order = sorted(positions.tolist(), key=lambda i: (hashlib.sha256(digest+int(i).to_bytes(4, "big")).digest(), i))
    result = np.array(value, dtype=np.float32, copy=True)
    if len(positions): result[np.ix_(positions, SHUFFLE_INDICES)] = value[np.ix_(order, SHUFFLE_INDICES)]
    result.setflags(write=False)
    return result


class TokenCursor(frozen.TelemetryCursor):
    """Frozen header-only reader plus new immutable token reduction/history.

    Inherited header parsing, cumulative clocks, closing and window pruning are
    hash-bound to telemetry90. Each lag requires its own monotone cursor.
    """

    def __init__(self, path):
        dependency_bindings();super().__init__(path)
        self._driver_predecessors = {}

    def _advance(self, cutoff_ns):
        if cutoff_ns < 0: return
        while True:
            self._read_header()
            if self._pending is None or self._pending[2] >= cutoff_ns: break
            sequence, recorded_ms, available = self._pending
            body = self._file.readline(packets.MAX_RECORD_BYTES-12+1)
            if len(body)+12 > packets.MAX_RECORD_BYTES: raise ValueError("record exceeds byte limit")
            payload = packets.decode_payload(body)
            groups = _groups({"sequence": sequence, "recorded_ms": recorded_ms,
                "available_ms": available//1_000_000, "payload": payload})
            current = [reduce_packet(driver, available, sequence, groups[driver],
                previous_availability_ns=self._driver_predecessors.get(driver)) for driver in sorted(groups)]
            for token in current:
                self._history.setdefault(token.driver_id, deque()).append(token)
                self._driver_predecessors[token.driver_id] = available
            self._prune(available)
            self._pending = None;self._last_consumed_sequence = sequence
            self._last_consumed_availability_ns = available;self._consumed_packets += 1
        self._prune(cutoff_ns)

    def query(self, driver_id, *, cutoff_ns):
        cutoff = _integer(cutoff_ns, "cutoff_ns");driver = _driver(driver_id)
        if self._closed: raise ValueError("token cursor is closed")
        if self._failed is not None: raise packets.PacketError("token cursor previously failed") from self._failed
        if self._last_cutoff_ns is not None and cutoff < self._last_cutoff_ns:
            raise ValueError("cutoff_ns must be nondecreasing; use one cursor per lag")
        try: self._advance(cutoff)
        except Exception as exc:
            self._failed = exc
            raise packets.PacketError(f"{self.path.name}: before cutoff_ns={cutoff}: {exc}") from exc
        self._last_cutoff_ns = cutoff
        return context_from_tokens(self._history.get(driver, ()), cutoff_ns=cutoff, driver_id=driver)
