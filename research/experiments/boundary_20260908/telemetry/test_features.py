"""Synthetic telemetry90 math and streaming-causality tests; no real data."""
import base64
import hashlib
import json
import math
import zlib

import numpy as np
import pytest

from research.experiments.boundary_20260908.telemetry import features as f


def encode(ms, entries):
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    prefix = f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}".encode()
    obj = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    raw = json.dumps({"Entries": entries}, separators=(",", ":")).encode()
    compressed = obj.compress(raw)+obj.flush()
    return prefix+json.dumps(base64.b64encode(compressed).decode()).encode()+b"\n"


def entry(channels=None, *, driver="1", utc="2022-01-01T00:00:00Z"):
    return {"Utc": utc, "Cars": {driver: {"Channels": channels}}}


def channels(speed=100, rpm=5000, throttle=50, brake=0, gear=4, drs=0):
    return {"0": rpm, "2": speed, "3": gear, "4": throttle, "5": brake, "45": drs}


def write(tmp_path, raw, name="stream.jsonStream"):
    path = tmp_path/name
    path.write_bytes(raw)
    return path


def snapshot(tmp_path, raw, cutoff_ns, driver="1"):
    with f.TelemetryCursor(write(tmp_path, raw)) as cursor:
        return cursor.query(driver, cutoff_ns=cutoff_ns)


def val(result, suffix, window=30):
    return result.as_dict()[f"telemetry_w{window}_{suffix}"]


def same(a, b):
    np.testing.assert_array_equal(a.values, b.values)
    assert a.supported == b.supported and a.provenance == b.provenance


def test_exact_layout_and_distinct_content_quality_indices():
    assert len(f.FEATURE_NAMES) == len(set(f.FEATURE_NAMES)) == 90
    assert f.CONTENT_INDICES == (*range(9), *range(30,39), *range(60,69))
    assert len(f.QUALITY_INDICES) == 63
    assert set(f.CONTENT_INDICES).isdisjoint(f.QUALITY_INDICES)
    assert sorted((*f.CONTENT_INDICES, *f.QUALITY_INDICES)) == list(range(90))
    for offset, seconds in zip((0,30,60), (30,90,180)):
        assert f.FEATURE_NAMES[offset] == f"telemetry_w{seconds}_speed_mean"
        assert f.FEATURE_NAMES[offset+29] == f"telemetry_w{seconds}_joint_throttle_brake_valid_fraction"
    data = np.arange(90, dtype=float)
    masked = data.copy()
    masked[list(f.CONTENT_INDICES)] = 0
    np.testing.assert_array_equal(masked[list(f.QUALITY_INDICES)], data[list(f.QUALITY_INDICES)])


@pytest.mark.parametrize("cutoff", [-10000000000, -1, 0, 999999999999])
def test_empty_snapshot_is_exact_finite_encoding_and_cursor_agrees(tmp_path, cutoff):
    expected = f.empty_snapshot("1", cutoff_ns=cutoff)
    actual = snapshot(tmp_path, b"", cutoff)
    same(expected, actual)
    assert not actual.supported and not actual.values.flags.writeable
    for window in f.WINDOWS_SECONDS:
        assert val(actual, "newest_packet_age_seconds", window) == window
        assert val(actual, "max_packet_or_boundary_gap_seconds", window) == window
        assert val(actual, "packet_count", window) == 0
    assert np.count_nonzero(actual.values) == 6
    json.dumps(actual.provenance, allow_nan=False)


def test_negative_cutoff_does_not_parse_even_first_unknown_clock(tmp_path):
    result = snapshot(tmp_path, b"invalid clock and payload", -1)
    same(result, f.empty_snapshot("1", cutoff_ns=-1))


def test_manual_packet_weighted_content_and_quality_math(tmp_path):
    a = entry(channels(speed=0, rpm=1000, throttle=0, brake=0, gear=1, drs=0))
    b = entry(channels(speed=100, rpm=9000, throttle=100, brake=100, gear=8, drs=12))
    result = snapshot(tmp_path, encode(0,[a])+encode(10000,[b]*9), 15*f.NS_PER_SECOND)
    expected_content = [50, 50, 5000, 50, .5, .5, .5, 4.5, .5]
    for window in (30,90,180):
        np.testing.assert_allclose([val(result,s,window) for s in f.CONTENT_SUFFIXES], expected_content, rtol=0, atol=1e-12)
        assert val(result,"packet_count",window) == 2
        assert val(result,"mean_bundle_size",window) == 5
        assert val(result,"availability_span_seconds",window) == 10
        assert val(result,"newest_packet_age_seconds",window) == 5
        assert val(result,"max_packet_or_boundary_gap_seconds",window) == window-15
        assert val(result,"joint_throttle_brake_valid_fraction",window) == 1
        assert val(result,"repeated_final_state_fraction",window) == 0
    assert not result.supported
    # A frame-weighted implementation would return90 rather than50.
    assert val(result, "speed_mean") != 90


def test_content_means_exclude_unsupported_packets_but_quality_keeps_them(tmp_path):
    raw = encode(0,[entry({"2": None, "4": 104})])+encode(1000,[entry({"2": 200, "4": 100})])
    result = snapshot(tmp_path, raw, 2*f.NS_PER_SECOND)
    assert val(result,"speed_mean") == 200 and val(result,"speed_packet_sd") == 0
    assert val(result,"throttle_mean") == 100
    assert val(result,"speed_present_fraction") == 1 and val(result,"speed_valid_fraction") == .5
    assert val(result,"throttle_code104_fraction") == .5


def test_disjoint_valid_channels_have_zero_joint_support(tmp_path):
    raw = encode(0,[entry({"4": 0, "5": 104}), entry({"4": 104, "5": 0})])
    result = snapshot(tmp_path, raw, 1)
    assert val(result,"throttle_valid_fraction") == .5 and val(result,"brake_valid_fraction") == .5
    assert val(result,"joint_throttle_brake_valid_fraction") == 0
    assert val(result,"low_throttle_zero_brake_fraction") == 0
    supported_zero = snapshot(tmp_path, encode(0,[entry({"4": 100,"5": 0})]),1)
    assert val(supported_zero,"low_throttle_zero_brake_fraction") == 0
    assert val(supported_zero,"joint_throttle_brake_valid_fraction") == 1


def test_joint_support_and_content_have_equal_packet_weights(tmp_path):
    first = [entry({"4": 0,"5": 0})]
    second = [entry({"4": 100,"5": 0})]+[entry({"4": 104,"5": 104})]*9
    result = snapshot(tmp_path, encode(0,first)+encode(1000,second),2*f.NS_PER_SECOND)
    assert val(result,"joint_throttle_brake_valid_fraction") == pytest.approx(.55)
    assert val(result,"low_throttle_zero_brake_fraction") == .5
    assert val(result,"throttle_code104_fraction") == .45


@pytest.mark.parametrize("channel,suffix,invalid", [
    ("2","speed",-1), ("2","speed",True), ("0","rpm",-1), ("0","rpm",False),
    ("3","gear",49), ("3","gear",1.5), ("3","gear",True),
    ("4","throttle",104), ("4","throttle",-1), ("4","throttle",True),
    ("5","brake",104), ("5","brake",1), ("5","brake",False),
    ("45","drs",-1), ("45","drs",1.5), ("45","drs",True),
])
def test_invalid_values_remain_present_without_physical_measurement(tmp_path, channel, suffix, invalid):
    result = snapshot(tmp_path, encode(0,[entry({channel: invalid})]),1)
    assert val(result,f"{suffix}_present_fraction") == 1
    assert val(result,f"{suffix}_valid_fraction") == 0


def test_raw_zero_null_false_missing_and_code104_are_distinct(tmp_path):
    rows = [entry({}),entry({"4":None,"5":None}),entry({"4":False,"5":False}),
            entry({"4":0,"5":0}),entry({"4":104,"5":104})]
    raw = encode(0,rows)
    path = write(tmp_path,raw)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with f.TelemetryCursor(path) as cursor:
        result = cursor.query("1",cutoff_ns=1)
    assert val(result,"throttle_present_fraction") == .8
    assert val(result,"throttle_valid_fraction") == .2
    assert val(result,"throttle_code104_fraction") == .2
    assert val(result,"brake_code104_fraction") == .2
    assert val(result,"joint_throttle_brake_valid_fraction") == .2
    assert val(result,"throttle_mean") == 0 and val(result,"low_throttle_zero_brake_fraction") == 1
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_repetition_preserves_raw_type_and_null_presence(tmp_path):
    states = [{"2":False},{"2":0},{"2":None},{},{"2":0},{"2":0}]
    raw = b"".join(encode(i*1000,[entry(state)]) for i,state in enumerate(states))
    result = snapshot(tmp_path,raw,6*f.NS_PER_SECOND)
    assert val(result,"repeated_final_state_fraction") == .2


def test_integer_like_gear_and_raw_drs_group_without_binary_semantics(tmp_path):
    rows = [entry({"3": 8.0, "45": code}) for code in [0,8,9,10,12,13,14,99]]
    result = snapshot(tmp_path,encode(0,rows),1)
    assert val(result,"gear_mean") == 8
    assert val(result,"drs_valid_fraction") == 1
    assert val(result,"drs_codes_10_12_14_fraction") == 3/8


def test_gate_uses_common_packet_support_and_inclusive_thresholds(tmp_path):
    raw = b"".join(encode(ms,[entry({})]) for ms in [0,5000,10000,15000,20000,24000])
    with f.TelemetryCursor(write(tmp_path,raw)) as cursor:
        exact = cursor.query("1",cutoff_ns=29*f.NS_PER_SECOND)
        too_old = cursor.query("1",cutoff_ns=29*f.NS_PER_SECOND+1)
    assert exact.supported is True and too_old.supported is False
    # No DRS or other channel is required by the purely packet-support gate.
    assert val(exact,"drs_valid_fraction") == 0
    assert val(exact,"packet_count") == 6 and val(exact,"availability_span_seconds") == 24


@pytest.mark.parametrize("times,cutoff", [([0,5000,10000,15000,24000],29000),
    ([0,5000,10000,15000,20000,23999],28999), ([0,5000,10000,15000,20000,24000],29001)])
def test_each_gate_condition_is_necessary(tmp_path,times,cutoff):
    raw = b"".join(encode(ms,[entry(channels())]) for ms in times)
    assert not snapshot(tmp_path,raw,cutoff*f.NS_PER_MS).supported


def test_window_lower_inclusive_upper_strict_at_nanosecond_precision(tmp_path):
    raw = encode(0,[entry(channels(speed=0))])+encode(30000,[entry(channels(speed=300))])
    with f.TelemetryCursor(write(tmp_path,raw)) as cursor:
        at = cursor.query("1",cutoff_ns=30*f.NS_PER_SECOND)
        after = cursor.query("1",cutoff_ns=30*f.NS_PER_SECOND+1)
    assert val(at,"packet_count") == 1 and val(at,"speed_mean") == 0
    assert val(after,"packet_count") == 1 and val(after,"speed_mean") == 300
    assert val(after,"newest_packet_age_seconds") == 1e-9
    assert val(after,"packet_count",90) == 2


@pytest.mark.parametrize("future_payload", [b"not-json\n",b'"invalid compressed bytes"\n',b"x"*10000+b"\n"])
def test_no_future_payload_prefetch_even_for_poisoned_or_oversized_record(tmp_path,monkeypatch,future_payload):
    past = encode(0,[entry(channels())])
    path = write(tmp_path,past+b"00:00:01.000"+future_payload)
    calls=[]
    original=f.packets.decode_payload
    def spy(payload,**kwargs):
        calls.append(payload)
        return original(payload,**kwargs)
    monkeypatch.setattr(f.packets,"decode_payload",spy)
    with f.TelemetryCursor(path) as cursor:
        result=cursor.query("1",cutoff_ns=f.NS_PER_SECOND)
        assert len(calls)==1
        assert cursor._file.tell() == len(past)+12  # header only; no payload read
        again=cursor.query("1",cutoff_ns=f.NS_PER_SECOND)
        same(result,again)
    expected=snapshot(tmp_path,past,f.NS_PER_SECOND)
    same(result,expected)


def test_payload_is_validated_when_its_availability_becomes_past(tmp_path):
    path=write(tmp_path,encode(0,[entry(channels())])+b"00:00:01.000broken\n")
    with f.TelemetryCursor(path) as cursor:
        cursor.query("1",cutoff_ns=f.NS_PER_SECOND)
        with pytest.raises(f.packets.PacketError):
            cursor.query("1",cutoff_ns=f.NS_PER_SECOND+1)
        with pytest.raises(f.packets.PacketError,match="previously failed"):
            cursor.query("1",cutoff_ns=2*f.NS_PER_SECOND)


def test_regressing_packet_cannot_be_backdated_past_preceding_future_header(tmp_path):
    raw=b"".join(encode(ms,[entry(channels(speed=speed))]) for ms,speed in [(1000,10),(3000,30),(2000,20)])
    with f.TelemetryCursor(write(tmp_path,raw)) as cursor:
        first=cursor.query("1",cutoff_ns=3*f.NS_PER_SECOND)
        later=cursor.query("1",cutoff_ns=3*f.NS_PER_SECOND+1)
    assert val(first,"packet_count")==1 and val(first,"speed_mean")==10
    assert val(later,"packet_count")==3 and val(later,"speed_mean")==20
    assert later.provenance["newest_driver_packet_availability_ns"]==3*f.NS_PER_SECOND


def test_all_queries_match_independent_prefix_files_including_provenance(tmp_path):
    times=[0,0,9000,8500,30000,100000,190000]
    records=[encode(ms,[entry(channels(speed=i))]) for i,ms in enumerate(times)]
    path=write(tmp_path,b"".join(records),"full")
    highwater=np.maximum.accumulate(times)
    with f.TelemetryCursor(path) as cursor:
        for cutoff in [0,1,9000,9001,30000,30001,100001,190001,400000]:
            actual=cursor.query("1",cutoff_ns=cutoff*f.NS_PER_MS)
            kept=[row for row,clock in zip(records,highwater) if clock<cutoff]
            with f.TelemetryCursor(write(tmp_path,b"".join(kept),"prefix")) as prefix:
                expected=prefix.query("1",cutoff_ns=cutoff*f.NS_PER_MS)
            same(actual,expected)


def test_bundles_atomic_equal_time_driver_order_invariant_and_utc_unused(tmp_path):
    entries=[entry(channels(speed=100),driver="1",utc="2099-01-01T00:00:00Z"),
             entry(channels(speed=200),driver="2",utc={"future": "not a clock"})]
    path=write(tmp_path,encode(1000,entries))
    with f.TelemetryCursor(path) as a, f.TelemetryCursor(path) as b:
        a1=a.query("1",cutoff_ns=f.NS_PER_SECOND+1)
        a2=a.query("2",cutoff_ns=f.NS_PER_SECOND+1)
        b2=b.query("2",cutoff_ns=f.NS_PER_SECOND+1)
        b1=b.query("1",cutoff_ns=f.NS_PER_SECOND+1)
    same(a1,b1);same(a2,b2)
    entries[0]["Utc"]="1970-01-01T00:00:00Z"
    entries[1]["Utc"]=None
    same(a1,snapshot(tmp_path,encode(1000,entries),f.NS_PER_SECOND+1))


def test_driver_isolation_missing_car_does_not_forward_fill(tmp_path):
    raw=encode(0,[entry(channels(speed=100),driver="1")])+encode(1000,[entry(channels(speed=200),driver="2")])
    result=snapshot(tmp_path,raw,2*f.NS_PER_SECOND)
    assert val(result,"packet_count")==1 and val(result,"speed_mean")==100
    assert val(result,"newest_packet_age_seconds")==2
    missing=snapshot(tmp_path,raw,2*f.NS_PER_SECOND,driver="9")
    assert val(missing,"packet_count")==0 and not missing.supported


def test_bounded_180s_state_and_immutable_prior_snapshot(tmp_path):
    raw=b"".join(encode(ms,[entry(channels(speed=ms/1000))]) for ms in range(0,500000,1000))
    with f.TelemetryCursor(write(tmp_path,raw)) as cursor:
        first=cursor.query("1",cutoff_ns=10*f.NS_PER_SECOND)
        saved=first.values.copy();provenance=json.dumps(first.provenance,sort_keys=True)
        last=cursor.query("1",cutoff_ns=500*f.NS_PER_SECOND)
        assert cursor.retained_packet_count==180
        assert last.provenance["oldest_driver_packet_availability_ns"]==320*f.NS_PER_SECOND
        assert val(last,"packet_count",180)==180
        np.testing.assert_array_equal(first.values,saved)
        assert json.dumps(first.provenance,sort_keys=True)==provenance
        cursor.query("1",cutoff_ns=1000*f.NS_PER_SECOND)
        assert cursor.retained_packet_count==0


def test_bom_blank_lines_empty_payload_and_null_channel_container(tmp_path):
    raw=b"\xef\xbb\xbf\n             \n"+encode(0,[])+encode(1,[entry(None)])
    result=snapshot(tmp_path,raw,2*f.NS_PER_MS)
    assert val(result,"packet_count")==1 and val(result,"mean_bundle_size")==1
    assert val(result,"rpm_present_fraction")==0
    assert result.provenance["driver_packet_sequences"]==[3]


@pytest.mark.parametrize("payload", [b"bad-clock!!!body\n", b"00:00:00.000null\n", b"00:00:00.000\n"])
def test_invalid_past_clock_or_body_fails_without_missing_fallback(tmp_path,payload):
    with f.TelemetryCursor(write(tmp_path,payload)) as cursor:
        with pytest.raises(f.packets.PacketError):
            cursor.query("1",cutoff_ns=f.NS_PER_SECOND)


def test_read_and_decompression_caps_apply_to_admitted_packets(tmp_path,monkeypatch):
    raw=encode(0,[entry(channels())])
    monkeypatch.setattr(f.packets,"MAX_RECORD_BYTES",len(raw)-1)
    with f.TelemetryCursor(write(tmp_path,raw)) as cursor:
        with pytest.raises(f.packets.PacketError,match="byte limit"):
            cursor.query("1",cutoff_ns=1)


def test_frozen_dependency_mismatch_fails_before_reading_input(monkeypatch,tmp_path):
    monkeypatch.setattr(f,"DEPENDENCY_BINDINGS",{next(iter(f.DEPENDENCY_BINDINGS)):"0"*64})
    with pytest.raises(RuntimeError,match="dependency changed"):
        f.TelemetryCursor(tmp_path/"does-not-exist")


@pytest.mark.parametrize("bad", [True,False,1.0,1.5,math.nan,math.inf,"1",None])
def test_cutoff_requires_exact_integer(tmp_path,bad):
    with f.TelemetryCursor(write(tmp_path,b"")) as cursor:
        with pytest.raises(ValueError,match="exact integer"):
            cursor.query("1",cutoff_ns=bad)
    with pytest.raises(ValueError):
        f.empty_snapshot("1",cutoff_ns=bad)


def test_nondecreasing_time_and_closed_cursor_guards(tmp_path):
    cursor=f.TelemetryCursor(write(tmp_path,b""))
    cursor.query("1",cutoff_ns=np.int64(1000))
    with pytest.raises(ValueError,match="nondecreasing"):
        cursor.query("1",cutoff_ns=999)
    cursor.close();cursor.close()
    with pytest.raises(ValueError,match="closed"):
        cursor.query("1",cutoff_ns=1001)


@pytest.mark.parametrize("bad", ["",1,None,False])
def test_driver_identifier_is_explicit_string(tmp_path,bad):
    with f.TelemetryCursor(write(tmp_path,b"")) as cursor:
        with pytest.raises(ValueError,match="driver_id"):
            cursor.query(bad,cutoff_ns=0)


def test_large_finite_nonnegative_measurements_do_not_overflow_summary(tmp_path):
    raw=encode(0,[entry({"2":1e308,"0":1e308})]*3)+encode(1000,[entry({"2":0,"0":0})])
    result=snapshot(tmp_path,raw,2*f.NS_PER_SECOND)
    assert np.isfinite(result.values).all()
    assert val(result,"speed_mean")==5e307 and val(result,"speed_packet_sd")==5e307


@pytest.mark.parametrize('count', [1, 2, 3, 6, 7, 9, 23, 99])
def test_float64_maximum_constant_bundles_and_windows_stay_finite(tmp_path, count):
    maximum = np.finfo(np.float64).max
    raw = b''.join(encode(i*100, [entry({'2': float(maximum), '0': float(maximum)})]*count)
                   for i in range(count))
    result = snapshot(tmp_path, raw, 10*f.NS_PER_SECOND)
    assert np.isfinite(result.values).all()
    for window in f.WINDOWS_SECONDS:
        assert val(result, 'speed_mean', window) == maximum
        assert val(result, 'rpm_mean', window) == maximum
        assert val(result, 'speed_packet_sd', window) == 0


@pytest.mark.parametrize('maximum', [float(np.finfo(np.float64).max), 1e308, 1e-300])
def test_extreme_mixture_preserves_mean_and_population_dispersion(tmp_path, maximum):
    raw = encode(0, [entry({'2': maximum})])+encode(1000, [entry({'2': 0})])
    result = snapshot(tmp_path, raw, 2*f.NS_PER_SECOND)
    assert val(result, 'speed_mean') == maximum/2
    assert val(result, 'speed_packet_sd') == maximum/2


def test_smallest_subnormal_constant_mean_is_not_lost_by_premature_division(tmp_path):
    minimum = float(np.nextafter(0., 1.))
    raw = encode(0, [entry({'2': minimum, '0': minimum})]*3)
    result = snapshot(tmp_path, raw, 1)
    assert val(result, 'speed_mean') == val(result, 'rpm_mean') == minimum


def test_subnormal_population_dispersion_does_not_center_on_underflowed_mean(tmp_path):
    minimum = float(np.nextafter(0., 1.))
    raw = encode(0, [entry({'2': minimum})])+encode(1000, [entry({'2': 0.})])
    result = snapshot(tmp_path, raw, 2*f.NS_PER_SECOND)
    # Exact mean and population SD are both minimum/2: round-to-even gives0.
    assert val(result, 'speed_mean') == val(result, 'speed_packet_sd') == minimum/2 == 0
