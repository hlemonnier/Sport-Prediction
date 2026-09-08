"""Synthetic token math, strict availability, prefix and matched-control tests."""
from dataclasses import replace
from copy import deepcopy
import base64
import json
import math
import sys
import zlib

import numpy as np
import pytest

from . import tokens as t
from research.experiments.boundary_20260908.telemetry import data, features
from research.experiments.boundary_20260908.telemetry.test_data import laps

NS = 1_000_000_000


def channels(speed=280, **patch):
    value = {"0": 12000, "2": speed, "3": 7, "4": 98, "5": 0, "45": 12}
    value.update(patch);return value


def packet(ms, entries=None, *, speed=280, utc="2022-01-01T00:00:00Z"):
    if entries is None: entries = [{"Utc": utc, "Cars": {"1": {"Channels": channels(speed)}}}]
    codec = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    payload = json.dumps({"Entries": entries}).encode()
    body = base64.b64encode(codec.compress(payload)+codec.flush()).decode()
    h, rest = divmod(ms, 3_600_000);m, rest = divmod(rest, 60_000);s, milli = divmod(rest, 1000)
    return f"{h:02}:{m:02}:{s:02}.{milli:03}".encode()+json.dumps(body).encode()+b"\n"


def path(tmp_path, content, name="stream"):
    target = tmp_path/name;target.write_bytes(content);return target


def token(sequence, instant, *, speed=280, previous=None, driver="1"):
    return t.reduce_packet(driver, instant, sequence, [channels(speed)], previous_availability_ns=previous)


def assert_snapshot_same(a, b):
    assert a.supported == b.supported and a.provenance == b.provenance
    for name in ("values", "source_availability_ns", "source_sequences", "previous_availability_ns"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_exact_34_order_and_normalization():
    assert len(t.TOKEN_NAMES) == len(set(t.TOKEN_NAMES)) == 34
    assert t.TOKEN_NAMES[:8] == ("speed_mean", "rpm_mean", "throttle_mean", "throttle_ge95_fraction", "brake_code100_fraction",
        "low_throttle_zero_brake_fraction", "gear_mean", "drs_codes_10_12_14_fraction")
    assert (t.BUNDLE_INDEX, t.GAP_INDEX, t.PADDING_INDEX) == (31, 32, 33)
    value = t.reduce_packet("1", 15*NS, 4, [channels(350), channels(350)], previous_availability_ns=5*NS)
    expected = np.array([1, .8, .98, 1, 0, 0, .875, 1], dtype=np.float32)
    np.testing.assert_array_equal(value.values[:8], expected)
    np.testing.assert_array_equal(value.values[8:16], np.ones(8))
    assert value.values[31] == np.float32(math.log(2)/math.log(18))
    assert value.values[32] == 2 and value.values[33] == 0
    assert value.values.dtype == np.float32 and not value.values.flags.writeable


def test_atomic_bundle_fraction_is_not_equal_weighted_driver_entries_across_packets(tmp_path):
    entries = [{"Cars": {"1": {"Channels": channels(v)}}} for v in (0, 100, 200)]
    p = path(tmp_path, packet(1000, entries)+packet(2000, speed=350))
    rows = list(t.iter_packet_tokens(p))
    assert len(rows) == 2
    assert rows[0].values[0] == np.float32(100/350)
    assert rows[1].values[0] == 1
    assert rows[0].values[31] == np.float32(math.log(3)/math.log(18))


def test_zero_missing_false_null_and_104_stay_distinct():
    observed = t.reduce_packet("1", 0, 0, [channels(0, **{"4": 0, "5": 0, "45": 0})])
    missing = t.reduce_packet("1", 0, 0, [{"0": 0, "3": 0}])
    invalid = t.reduce_packet("1", 0, 0, [channels(False, **{"4": 104, "5": 104, "45": None})])
    assert observed.values[0] == missing.values[0] == invalid.values[0] == 0
    assert observed.values[8] == 1 and missing.values[8] == invalid.values[8] == 0
    assert observed.values[5] == 1 and observed.values[13] == 1
    assert missing.values[13] == 0 and invalid.values[13] == 0
    assert invalid.values[t.TOKEN_NAMES.index("speed_present_fraction")] == 1
    assert missing.values[t.TOKEN_NAMES.index("speed_present_fraction")] == 0
    assert invalid.values[t.TOKEN_NAMES.index("throttle_code104_fraction")] == 1
    assert invalid.values[t.TOKEN_NAMES.index("brake_code104_fraction")] == 1


def test_joint_disjoint_validity_has_no_fabricated_coast_measurement():
    rows = [channels(**{"4": 0, "5": 104}), channels(**{"4": 104, "5": 0})]
    value = t.reduce_packet("1", 0, 0, rows).values
    assert value[5] == 0 and value[13] == 0
    assert value[t.TOKEN_NAMES.index("throttle_valid_fraction")] == .5
    assert value[t.TOKEN_NAMES.index("brake_valid_fraction")] == .5
    assert value[t.TOKEN_NAMES.index("joint_throttle_brake_valid_fraction")] == 0


@pytest.mark.parametrize("speed", [sys.float_info.max, 5e-324, 0])
def test_extreme_positive_values_remain_finite_bounded_float32(speed):
    value = t.reduce_packet("1", 0, 0, [channels(speed), channels(speed), channels(speed)]).values
    assert np.isfinite(value).all() and (value >= 0).all() and (value <= 5).all()
    assert value[8] == 1
    if speed == sys.float_info.max: assert value[0] == 5


@pytest.mark.parametrize("bad", [True, 1.0, np.float64(1), "1000", None])
def test_clocks_do_not_accept_float_rounding_or_bool(bad):
    with pytest.raises(ValueError): t.empty_snapshot("1", cutoff_ns=bad)
    with pytest.raises(ValueError): t.reduce_packet("1", bad, 0, [channels()])


def test_empty_padding_missing_stream_and_negative_cutoff():
    value = t.empty_snapshot("1", cutoff_ns=-2*NS)
    assert value.values.shape == (128, 34) and not value.supported
    np.testing.assert_array_equal(value.values[:, :33], 0)
    np.testing.assert_array_equal(value.values[:, 33], 1)
    np.testing.assert_array_equal(value.source_sequences, -1)
    assert value.provenance["context_packet_count"] == 0
    key = t.permutation_key(202201, "1", None)
    np.testing.assert_array_equal(t.permute_context(value.values, key=key), value.values)


def test_history_filters_future_and_stale_values_before_reading_them():
    class Poison:
        def __array__(self, *args, **kwargs): raise AssertionError("future or stale numeric values read")
    past = token(2, 200*NS)
    future = replace(past, packet_sequence=3, availability_ns=201*NS, values=Poison())
    stale = replace(past, packet_sequence=1, availability_ns=19*NS, values=Poison())
    expected = t.context_from_tokens([past], cutoff_ns=201*NS, driver_id="1")
    actual = t.context_from_tokens([stale, past, future], cutoff_ns=201*NS, driver_id="1")
    assert_snapshot_same(actual, expected)


def test_180_second_left_boundary_inclusive_and_cutoff_strict():
    rows = [token(0, 20*NS), token(1, 200*NS, previous=20*NS)]
    value = t.context_from_tokens(rows, cutoff_ns=200*NS, driver_id="1")
    assert value.provenance["context_packet_count"] == 1
    assert value.source_availability_ns[-1] == 20*NS
    after = t.context_from_tokens(rows, cutoff_ns=200*NS+1, driver_id="1")
    assert after.provenance["context_packet_count"] == 1
    assert after.source_availability_ns[-1] == 200*NS


def test_support_uses_uncapped_history_not_latest128():
    times = [NS]+[27*NS+i*1_000_000 for i in range(149)]
    rows = [token(i, instant, previous=times[i-1] if i else None) for i, instant in enumerate(times)]
    value = t.context_from_tokens(rows, cutoff_ns=30*NS, driver_id="1")
    assert value.supported
    assert value.provenance["support_packet_count_30s"] == 150
    assert value.provenance["context_packet_count"] == 128
    assert value.provenance["older_in_window_packets_omitted"] == 22
    assert value.source_availability_ns[-1]-value.source_availability_ns[0] < NS


def test_true_predecessor_gap_survives_window_eviction(tmp_path):
    p = path(tmp_path, packet(0)+packet(200_000)+packet(201_000))
    with t.TokenCursor(p) as cursor:
        value = cursor.query("1", cutoff_ns=201*NS)
        assert value.provenance["context_packet_count"] == 1
        assert value.previous_availability_ns[-1] == 0
        assert value.values[-1, t.GAP_INDEX] == 5
    rows = list(t.iter_packet_tokens(p))
    assert rows[0].values[t.GAP_INDEX] == 0 and rows[1].values[t.GAP_INDEX] == 5
    assert_snapshot_same(value, t.context_from_tokens(rows, cutoff_ns=201*NS, driver_id="1"))


def test_repeated_and_regressed_archive_clocks_stay_in_physical_order(tmp_path):
    p = path(tmp_path, packet(1000, speed=100)+packet(900, speed=200)+packet(1000, speed=300))
    rows = list(t.iter_packet_tokens(p))
    assert [r.availability_ns for r in rows] == [NS]*3
    assert [r.packet_sequence for r in rows] == [0, 1, 2]
    with t.TokenCursor(p) as cursor:
        assert cursor.query("1", cutoff_ns=NS).provenance["context_packet_count"] == 0
        snap = cursor.query("1", cutoff_ns=NS+1)
    assert snap.source_sequences[-3:].tolist() == [0, 1, 2]
    np.testing.assert_array_equal(snap.values[-3:, t.GAP_INDEX], 0)


def test_packet_array_and_context_input_immutability():
    entries = [channels(100), channels(200)];before = deepcopy(entries)
    row = t.reduce_packet("1", 0, 0, entries)
    assert entries == before
    with pytest.raises(ValueError): row.values[0] = 100
    snapshot = t.context_from_tokens([row], cutoff_ns=1, driver_id="1")
    with pytest.raises(ValueError): snapshot.values[0, 0] = 100


def test_minimal_identical170_summaries_but_different_sequence_order(tmp_path):
    raw = laps(last=5);original, _ = data._original(raw, 202201)
    base80 = original.iloc[0][list(data.BASE_FEATURES)].to_numpy(float)
    times = (275000, 280000, 285000, 290000, 295000, 299000)
    a = (100, 150, 200, 250, 300, 275);b = (300, 200, 100, 250, 150, 275)
    first = path(tmp_path, b"".join(packet(ms, speed=v) for ms, v in zip(times, a)), "ordered")
    second = path(tmp_path, b"".join(packet(ms, speed=v) for ms, v in zip(times, b)), "different_order")
    with features.TelemetryCursor(first) as x, features.TelemetryCursor(second) as y:
        sx, sy = x.query("1", cutoff_ns=300*NS), y.query("1", cutoff_ns=300*NS)
    np.testing.assert_array_equal(np.r_[base80, sx.values], np.r_[base80, sy.values])
    assert sx.supported and sy.supported
    with t.TokenCursor(first) as x, t.TokenCursor(second) as y:
        tx, ty = x.query("1", cutoff_ns=300*NS), y.query("1", cutoff_ns=300*NS)
    assert tx.supported and ty.supported and not np.array_equal(tx.values, ty.values)
    np.testing.assert_array_equal(tx.values[-1], ty.values[-1])


def test_order_destroyed_control_moves_whole_bundles_and_preserves_newest_gaps_padding():
    rows = [token(i, i*5*NS, speed=100+i*10, previous=(i-1)*5*NS if i else None) for i in range(20)]
    snapshot = t.context_from_tokens(rows, cutoff_ns=96*NS, driver_id="1")
    key = t.permutation_key(202201, "1", 19);before = snapshot.values.copy()
    changed = t.permute_context(snapshot.values, key=key)
    np.testing.assert_array_equal(snapshot.values, before)
    np.testing.assert_array_equal(changed[:, t.FIXED_INDICES], before[:, t.FIXED_INDICES])
    np.testing.assert_array_equal(changed[-1], before[-1])
    np.testing.assert_array_equal(changed[:108], before[:108])
    assert sorted(map(tuple, changed[108:-1, :32])) == sorted(map(tuple, before[108:-1, :32]))
    assert not np.array_equal(changed, before)
    np.testing.assert_array_equal(changed, t.permute_context(before, key=key))
    assert not np.array_equal(changed, t.permute_context(before, key=t.permutation_key(202201, "1", 20)))
    assert not changed.flags.writeable


def test_control_same_admitted_prefix_uses_same_key_across_cutoffs():
    rows = [token(i, i*5*NS) for i in range(7)]
    a = t.context_from_tokens(rows, cutoff_ns=31*NS, driver_id="1")
    b = t.context_from_tokens(rows, cutoff_ns=33*NS, driver_id="1")
    ka = t.permutation_key(202201, "1", a.provenance["last_admitted_driver_packet_sequence"])
    kb = t.permutation_key(202201, "1", b.provenance["last_admitted_driver_packet_sequence"])
    np.testing.assert_array_equal(t.permute_context(a.values, key=ka), t.permute_context(b.values, key=kb))


@pytest.mark.parametrize("corruption", ["nonleft_padding", "pad_values", "nonfinite", "wrong_shape"])
def test_control_rejects_malformed_context(corruption):
    value = t.empty_snapshot("1", cutoff_ns=0).values.copy()
    if corruption == "nonleft_padding": value[0, 33] = 0
    elif corruption == "pad_values": value[0, 0] = 1
    elif corruption == "nonfinite": value[0, 0] = np.nan
    else: value = value[:, :33]
    with pytest.raises(ValueError): t.permute_context(value, key="past-prefix")


def test_header_lookahead_does_not_decode_equal_or_future_payload(tmp_path, monkeypatch):
    past = packet(1000);future = b"00:00:02.000"+b"invalid future body"*1000+b"\n"
    p = path(tmp_path, past+future)
    original = t.packets.decode_payload;decoded = []
    def decoder(body): decoded.append(body);return original(body)
    monkeypatch.setattr(t.packets, "decode_payload", decoder)
    with t.TokenCursor(p) as cursor:
        early = cursor.query("1", cutoff_ns=2*NS)
        assert len(decoded) == 1 and cursor._file.tell() == len(past)+12
        assert early.provenance["context_packet_count"] == 1
        with pytest.raises(t.packets.PacketError): cursor.query("1", cutoff_ns=2*NS+1)
        with pytest.raises(t.packets.PacketError, match="previously failed"): cursor.query("1", cutoff_ns=3*NS)


def test_future_append_and_utc_poison_preserve_snapshot_and_strict_prefix(tmp_path):
    past = b"".join(packet(ms) for ms in (0, 5000, 10000, 15000, 20000, 25000))
    poisoned = b"".join(packet(ms, utc="not-a-date") for ms in (0, 5000, 10000, 15000, 20000, 25000))
    a = path(tmp_path, past, "prefix")
    b = path(tmp_path, poisoned+packet(30000, speed=99999)+b"00:00:31.000malformed", "future")
    with t.TokenCursor(a) as x, t.TokenCursor(b) as y:
        assert_snapshot_same(x.query("1", cutoff_ns=30*NS), y.query("1", cutoff_ns=30*NS))


def test_monotone_query_and_missing_other_driver(tmp_path):
    p = path(tmp_path, packet(1000))
    with t.TokenCursor(p) as cursor:
        assert not cursor.query("44", cutoff_ns=2*NS).supported
        assert cursor.query("1", cutoff_ns=2*NS).provenance["context_packet_count"] == 1
        with pytest.raises(ValueError, match="nondecreasing"): cursor.query("1", cutoff_ns=NS)
    with pytest.raises(ValueError, match="closed"): cursor.query("1", cutoff_ns=3*NS)


def test_cursor_iterator_exact_parity_atomic_driver_order_and_final_stats(tmp_path):
    entries = [{"Cars": {"2": {"Channels": channels(200)}, "1": {"Channels": channels(100)}}},
               {"Cars": {"1": {"Channels": channels(300)}}}]
    p = path(tmp_path, b"\xef\xbb\xbf"+packet(1000, entries)+packet(2000))
    stats = {};rows = list(t.iter_packet_tokens(p, stats=stats))
    assert [(r.packet_sequence, r.driver_id) for r in rows] == [(0, "1"), (0, "2"), (1, "1")]
    assert stats["complete"] is True and stats["stream_input_valid"] is True
    with t.TokenCursor(p) as cursor:
        for driver in ("1", "2"):
            assert_snapshot_same(cursor.query(driver, cutoff_ns=2*NS+1),
                t.context_from_tokens(rows, cutoff_ns=2*NS+1, driver_id=driver))


def test_supported_matches_frozen_summary_gate_at_exact_edges(tmp_path):
    p = path(tmp_path, b"".join(packet(ms) for ms in (1000, 5000, 10000, 15000, 20000, 25000)))
    with t.TokenCursor(p) as new, features.TelemetryCursor(p) as old:
        for cutoff in (25*NS, 25*NS+1, 30*NS, 30*NS+1, 31*NS):
            assert new.query("1", cutoff_ns=cutoff).supported == old.query("1", cutoff_ns=cutoff).supported


def test_history_rejects_reordering_duplicate_sequences_and_false_gap():
    a, b = token(0, NS), token(1, 2*NS, previous=NS)
    for rows in ([b, a], [a, a], [a, replace(b, previous_availability_ns=0)]):
        with pytest.raises(ValueError): t.context_from_tokens(rows, cutoff_ns=3*NS, driver_id="1")
