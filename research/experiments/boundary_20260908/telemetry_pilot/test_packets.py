"""Synthetic parser/clock tests; no provider access, cached races, or models."""
import base64
import json
import math
import zlib

import pytest

from research.experiments.boundary_20260908.telemetry_pilot import packets as p


def envelope(payload, *, raw=False):
    encoded = payload if raw else json.dumps(payload, separators=(",", ":")).encode()
    encoder = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    compressed = encoder.compress(encoded) + encoder.flush()
    return json.dumps(base64.b64encode(compressed).decode()).encode()


def record(ms, payload):
    h, remaining = divmod(ms, 3600000)
    m, remaining = divmod(remaining, 60000)
    s, millis = divmod(remaining, 1000)
    return f"{h:02}:{m:02}:{s:02}.{millis:03}".encode()+envelope(payload)+b"\n"


def body(channels=None, utc="2022-03-20T15:00:00.1234567Z", driver="1"):
    return {"Entries": [{"Utc": utc, "Cars": {driver: {"Channels": {"2": 250, "5": False} if channels is None else channels}}}]}


def save(tmp_path, value):
    path = tmp_path / "CarData.z.jsonStream"
    path.write_bytes(value)
    return path


@pytest.mark.parametrize("millis", [0, 1, 9, 29, 999, 1001, 59999, 60000, 3599999, 3600000, 86399999, 359999999])
def test_clock_is_exact_integer_without_binary_rounding(millis):
    header = record(millis, body())[:12]
    assert p.clock_milliseconds(header) == millis
    assert type(p.clock_milliseconds(header.decode())) is int


@pytest.mark.parametrize("bad", ["0:00:00.000", "00:60:00.000", "00:00:60.000", "-1:00:00.000",
    "00:00:00:000", "00:00:00.00", "00:00:00.0001", "00:00:00.nan", "00:00:00.０00", 1.0, None])
def test_invalid_clocks_fail(bad):
    with pytest.raises(ValueError):
        p.clock_milliseconds(bad)


def test_utc_ns_precision_and_quality_only():
    origin = p.utc_nanoseconds("2022-03-20T15:00:00Z")
    assert p.utc_nanoseconds("2022-03-20T15:00:00.123456789Z")-origin == 123456789
    assert p.utc_nanoseconds("2022-03-20T15:00:00.1234567Z")-origin == 123456700
    assert p.utc_nanoseconds("1969-12-31T23:59:59.999999999Z") == -1
    for value in [None, False, 1.0, "2022-02-30T15:00:00Z", "2022-03-20T15:00:00", "nonsense"]:
        assert p.utc_nanoseconds(value) is None


def test_physical_order_regressions_and_strict_cutoff(tmp_path):
    path = save(tmp_path, b"".join(record(t, body({"2": t})) for t in [10, 20, 12, 20, 30]))
    stats = {}
    rows = list(p.iter_packets(path, stats=stats))
    assert [r["sequence"] for r in rows] == [0, 1, 2, 3, 4]
    assert [r["recorded_ms"] for r in rows] == [10, 20, 12, 20, 30]
    assert [r["available_ms"] for r in rows] == [10, 20, 20, 20, 30]
    assert [r["timestamp_regressed"] for r in rows] == [False, False, True, False, False]
    assert stats["stream_input_valid"] is True and stats["complete"] is True
    stats = {}
    assert list(p.iter_packets(path, before_ms=20, stats=stats)) == rows[:1]
    assert stats["prefix_input_valid"] is True and stats["stream_input_valid"] is None
    assert stats["stopped_at_cutoff"] is True and stats["complete"] is False
    assert list(p.iter_packets(path, before_ms=21)) == rows[:4]


@pytest.mark.parametrize("poison", [b"not-json", b'"invalid-base64"', b"x"*500])
def test_future_payload_poison_and_later_clock_regression_cannot_change_prefix(tmp_path, poison):
    past = record(10, body())
    expected = list(p.iter_packets(save(tmp_path, past), before_ms=20))
    changed = past+b"00:00:00.020"+poison+b"\n"+record(11, body({"2": -1000}))
    assert list(p.iter_packets(save(tmp_path, changed), before_ms=20, max_record_bytes=256)) == expected


def test_full_and_all_prefix_replay_are_identical(tmp_path):
    raw = b"".join(record(t, body({"2": i}, utc=f"2022-03-20T15:00:00.{i:09}Z"))
        for i, t in enumerate([10, 10, 20, 18, 35, 50]))
    path = save(tmp_path, raw)
    full = list(p.iter_packets(path))
    for cutoff in [0, 10, 11, 20, 21, 36, 50, 51]:
        expected = [row for row in full if row["available_ms"] < cutoff]
        assert list(p.iter_packets(path, before_ms=cutoff)) == expected


def test_source_utc_does_not_reorder_or_define_availability(tmp_path):
    raw = record(100, body(utc="2099-01-01T00:00:00Z"))+record(200, body(utc="1970-01-01T00:00:00Z"))
    path = save(tmp_path, raw)
    packets = list(p.iter_packets(path, before_ms=201))
    assert [x["available_ms"] for x in packets] == [100, 200]
    report = p.summarize(path)
    assert report["drivers"]["1"]["source_utc_regressions"] == 1
    assert report["packet_validation"]["timestamp_regressions"] == 0


def test_bundle_provenance_missing_zero_false_and_unknown_channel_retained(tmp_path):
    payload = {"Entries": [
        {"Utc": "2022-03-20T15:00:00.2Z", "Cars": {"1": {"Channels": {"2": 0, "5": False, "99": 17}}, "2": {}}},
        {"Utc": "2022-03-20T15:00:00.1Z", "Cars": {"1": {"Channels": {"2": 3, "5": True, "45": None}}}},
        {"Utc": "broken", "Cars": {}}, {}]}
    path = save(tmp_path, record(100, payload))
    packet = next(p.iter_packets(path))
    rows = list(p.iter_car_rows(packet))
    assert [(x["entry_index"], x["driver"], x["available_ms"]) for x in rows] == [(0, "1", 100), (0, "2", 100), (1, "1", 100)]
    assert rows[0]["channels"]["2"] == 0 and rows[0]["channels"]["5"] is False
    assert "45" not in rows[0]["channels"] and rows[1]["channels"] == {}
    assert rows[1]["channels_present"] is False and rows[2]["channels"]["45"] is None
    rows[0]["channels"]["2"] = 999
    assert packet["payload"]["Entries"][0]["Cars"]["1"]["Channels"]["2"] == 0
    report = p.summarize(path)
    assert report["counts"]["entries"] == 4 and report["counts"]["bundled_packets"] == 1
    assert report["counts"]["car_rows"] == 3 and report["counts"]["car_rows_missing_channels"] == 1
    assert report["counts"]["entries_missing_utc"] == 1 and report["counts"]["entries_invalid_utc"] == 1
    assert report["channels"]["45"]["present"] == 1 and report["channels"]["45"]["absent"] == 2
    assert report["channels"]["45"]["null"] == 1 and report["channels"]["45"]["zero"] == 0
    assert report["channels"]["5"]["boolean"] == 2 and report["channels"]["5"]["zero"] == 1
    assert report["channels"]["99"]["known_name"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("payload", [[], {}, {"Entries": {}}, {"Entries": [1]},
    {"Entries": [{"Cars": []}]}, {"Entries": [{"Cars": {"1": []}}]},
    {"Entries": [{"Cars": {"1": {"Channels": []}}}]}])
def test_malformed_decoded_structure_fails_explicitly(tmp_path, payload):
    stats = {}
    with pytest.raises(p.PacketError, match="physical record 0"):
        list(p.iter_packets(save(tmp_path, record(1, payload)), stats=stats))
    assert stats["stream_input_valid"] is False and stats["prefix_input_valid"] is False


@pytest.mark.parametrize("payload", [b'"not base64!"', b'{}', b'null', b'""', b'"abc', b'"e30="'])
def test_malformed_envelope_or_compression_fails_explicitly(tmp_path, payload):
    with pytest.raises(p.PacketError):
        list(p.iter_packets(save(tmp_path, b"00:00:00.001"+payload+b"\n")))


@pytest.mark.parametrize("payload", [b'{"Entries":[],"Entries":[]}',
    b'{"Entries":[{"Cars":{"1":{"Channels":{"2":NaN}}}}]}',
    b'{"Entries":[{"Cars":{"1":{"Channels":{"2":Infinity}}}}]}',
    b'{"Entries":[{"Cars":{"1":{"Channels":{"2":1e400}}}}]}'])
def test_duplicate_keys_nonfinite_and_overflow_are_rejected(tmp_path, payload):
    with pytest.raises(p.PacketError):
        list(p.iter_packets(save(tmp_path, b"00:00:00.001"+envelope(payload, raw=True)+b"\n")))


def test_truncated_and_trailing_compressed_bytes_rejected():
    compressed = base64.b64decode(json.loads(envelope(body())))
    for value in [compressed[:-1], compressed+b"extra"]:
        with pytest.raises(ValueError):
            p.decode_payload(json.dumps(base64.b64encode(value).decode()).encode())


def test_decompression_cap_and_exact_boundary():
    raw = json.dumps({"Entries": [], "padding": "x"*10000}).encode()
    encoded = envelope(raw, raw=True)
    with pytest.raises(ValueError, match="byte limit"):
        p.decode_payload(encoded, max_decompressed_bytes=128)
    assert p.decode_payload(encoded, max_decompressed_bytes=len(raw))["Entries"] == []


def test_record_cap_bom_crlf_and_blank_lines(tmp_path):
    raw = record(1, body())
    assert len(list(p.iter_packets(save(tmp_path, b"\xef\xbb\xbf"+raw.rstrip(b"\n")+b"\r\n\n")))) == 1
    with pytest.raises(p.PacketError, match="record exceeds"):
        list(p.iter_packets(save(tmp_path, raw), max_record_bytes=len(raw)-1))


def test_invalid_unknown_clock_before_cutoff_fails_closed(tmp_path):
    with pytest.raises(p.PacketError, match="archive clock"):
        list(p.iter_packets(save(tmp_path, record(1, body())+b"not-a-clock invalid\n"), before_ms=20))


def test_full_stream_validation_not_claimed_before_exhaustion(tmp_path):
    stats = {}
    gen = p.iter_packets(save(tmp_path, record(1, body())+record(2, body())), stats=stats)
    first = next(gen)
    before = json.dumps(first, sort_keys=True)
    assert stats["stream_input_valid"] is None and stats["complete"] is False
    next(gen)
    assert json.dumps(first, sort_keys=True) == before
    assert stats["stream_input_valid"] is None
    assert next(gen, None) is None
    assert stats["stream_input_valid"] is True


def test_empty_stream_is_explicit_empty_valid_diagnostic(tmp_path):
    report = p.summarize(save(tmp_path, b""))
    assert report["packet_validation"]["complete"] is True
    assert report["counts"]["car_rows"] == 0 and report["counts"]["entries"] == 0
    assert report["drivers"] == {} and report["channels"]["45"]["absent"] == 0
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("invalid", [True, False, 1.0, 1.1, -1, math.nan, math.inf, "1"])
def test_cutoff_rejects_noninteger_including_rounded_float(invalid, tmp_path):
    with pytest.raises(ValueError, match="nonnegative integer"):
        list(p.iter_packets(save(tmp_path, b""), before_ms=invalid))


@pytest.mark.parametrize("name", ["max_record_bytes", "max_decompressed_bytes"])
@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5, math.nan])
def test_limits_are_strict_positive_integers(name, invalid, tmp_path):
    with pytest.raises(ValueError):
        list(p.iter_packets(save(tmp_path, b""), **{name: invalid}))
