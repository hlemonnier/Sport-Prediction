"""Synthetic original-horizon parity, exact clocks and feature/label separation."""
from copy import deepcopy
from decimal import Decimal
import base64
import json
from pathlib import Path
import zlib

import numpy as np
import pandas as pd
import pytest

from . import data, features


def laps(last=9):
    return pd.DataFrame([{
        "DriverNumber": str(driver), "LapNumber": lap, "Time": lap*100.+offset,
        "LapTime": 90.+lap*.07+driver*.03, "IsAccurate": True, "Stint": 1.,
        "Compound": "MEDIUM", "TyreLife": float(lap), "FreshTyre": False,
        "PitInTime": np.nan, "PitOutTime": np.nan, "TrackStatus": "1",
        "Sector1Time": 30.+lap*.02, "Sector2Time": 30., "Sector3Time": 30.,
        "Position": float(driver), "SpeedI1": 280., "SpeedI2": 290., "SpeedFL": 295., "SpeedST": 300.,
    } for lap in range(1, last+1) for driver, offset in ((1, 0), (2, 0), (3, 1), (4, 2))])


def packet(millis, value=260.):
    channels = {"0": 11000, "2": value, "3": 7, "4": 98, "5": 0, "45": 12}
    payload = {"Entries": [{"Cars": {str(d): {"Channels": channels} for d in range(1, 5)}}]}
    codec = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    compressed = codec.compress(json.dumps(payload).encode())+codec.flush()
    h, rest = divmod(millis, 3_600_000);m, rest = divmod(rest, 60_000);s, ms = divmod(rest, 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}".encode()+json.dumps(base64.b64encode(compressed).decode()).encode()+b"\n"


def telemetry(tmp_path, *, suffix=b""):
    path = tmp_path/"CarData.z.jsonStream"
    path.write_bytes(b"".join(packet(t, 250+(t//5000)%30) for t in range(5000, 1_000_001, 5000))+suffix)
    return path


def clock(raw):
    return data.make_clock_index(raw, [str(t) for t in raw.Time])


def source(tmp_path, raw=None):
    path = tmp_path/"laps.csv"
    (laps() if raw is None else raw).to_csv(path, index=False)
    return data.read_laps(path)


def event(tmp_path, raw=None, *, unavailable=False):
    raw, clocks = source(tmp_path, raw)
    path = None if unavailable else telemetry(tmp_path)
    return data.build_event(raw, 202201, clocks, path,
                            source_status="unavailable" if unavailable else "downloaded")


def closure(tmp_path, bundle):
    design = tmp_path/"design_lock.json";design.write_text('{"synthetic":true}\n')
    return data.close_features([bundle], tmp_path/"features", design_lock_path=design,
                               design_sha256=data.sha(design), expected_event_keys=[202201])


@pytest.mark.parametrize("text,expected", [("0", 0), ("0.029", 29_000_000), ("0.999999999", 999_999_999),
    ("1.000000001", 1_000_000_001), ("12345.678900001", 12_345_678_900_001), ("1e-9", 1),
    (Decimal("1.25"), 1_250_000_000), (2, 2_000_000_000)])
def test_exact_seconds_to_ns_has_no_rounding(text, expected):
    assert data.seconds_to_ns(text) == expected
    assert type(data.seconds_to_ns(text)) is int


@pytest.mark.parametrize("bad", [1.0, True, np.float64(1), None, "nan", "Infinity", "-0.1", "0.0000000001", "1e999999", "9223372037"])
def test_ambiguous_nonfinite_subns_and_float_clocks_fail(bad):
    with pytest.raises((ValueError, TypeError)): data.seconds_to_ns(bad)


def test_csv_time_lexical_ns_keeps_frozen_float_keys_untouched(tmp_path):
    raw = laps(last=4)
    path = tmp_path/"laps.csv";raw.to_csv(path, index=False)
    text = path.read_text().replace(",300.0,", ",300.000000001,")
    path.write_text(text)
    parsed, clocks = data.read_laps(path)
    assert clocks[("1", 3)].nanoseconds == 300_000_000_001
    expected, _ = data.frontier.enrich(pd.read_csv(path), 202201)
    built = data.build_event(parsed, 202201, clocks, None, source_status="unavailable")
    for frame in built.frames_by_lag.values():
        pd.testing.assert_frame_equal(frame[list(expected.columns)], expected, check_exact=True)
    assert built.frames_by_lag[2].query("driver_id == '1' and issued_after_lap_number == 3").issued_at_ns.iloc[0] == 300_000_000_001


def test_all_original80_features_anchors_issuances_and_unmatched_are_preserved(tmp_path):
    raw = laps();before = raw.copy(deep=True)
    expected, matched = data.frontier.enrich(raw, 202201)
    path = telemetry(tmp_path)
    built = data.build_event(raw, 202201, clock(raw), path)
    assert len(data.BASE_FEATURES) == 80
    assert tuple(data.frontier.base.FEATURES)+tuple(c for c in expected if c.startswith("x_")) == data.BASE_FEATURES
    assert len(expected)-len(matched) == 4
    for lag, frame in built.frames_by_lag.items():
        pd.testing.assert_frame_equal(frame[list(expected.columns)], expected, check_exact=True)
        assert len(frame) == len(expected) and not frame.issuance_id.duplicated().any()
        assert (frame.issued_at_ns-frame.telemetry_cutoff_ns == lag*1_000_000_000).all()
        assert not set(data.TARGET_FIELDS).intersection(frame)
        assert all(len(v) == 90 and np.isfinite(v).all() for v in frame.telemetry_values)
    assert built.frames_by_lag[0].issuance_id.tolist() == built.frames_by_lag[2].issuance_id.tolist()
    pd.testing.assert_frame_equal(raw, before, check_exact=True)


def test_equal_clock_telemetry_is_excluded_and_same_time_drivers_are_retained(tmp_path):
    raw = laps(last=4);path = telemetry(tmp_path)
    built = data.build_event(raw, 202201, clock(raw), path)
    for lag, frame in built.frames_by_lag.items():
        rows = frame.loc[frame.issued_at_timestamp == 300.]
        assert set(rows.driver_id) == {"1", "2"}
        for row in rows.to_dict("records"):
            assert row["telemetry_provenance"]["last_consumed_availability_ns"] < row["telemetry_cutoff_ns"]
            assert row["telemetry_provenance"]["last_consumed_availability_ns"] == 295_000_000_000
    # Source is shared across cars, but both queries see only strictly prior bundles.
    assert built.frames_by_lag[0].query("issued_at_timestamp == 300").telemetry_provenance.map(lambda p:p["last_consumed_sequence"]).nunique() == 1


def test_submillisecond_cutoff_preserves_integer_ns_boundary(tmp_path):
    raw = laps(last=3);raw.loc[raw.DriverNumber.eq("1") & raw.LapNumber.eq(3), "Time"] = 300.000000001
    path = telemetry(tmp_path);built = data.build_event(raw, 202201, clock(raw), path)
    one = built.frames_by_lag[0].query("driver_id == '1'").iloc[0]
    two = built.frames_by_lag[0].query("driver_id == '2'").iloc[0]
    assert one.telemetry_provenance["last_consumed_availability_ns"] == 300_000_000_000
    assert two.telemetry_provenance["last_consumed_availability_ns"] == 295_000_000_000


def test_future_lap_value_poison_does_not_change_prior_enriched_issuances(tmp_path):
    raw = laps();path = telemetry(tmp_path)
    full = data.build_event(raw, 202201, clock(raw), path)
    changed = raw.copy();future = changed.Time > 602.
    changed.loc[future, ["LapTime", "Sector1Time", "SpeedST", "TyreLife"]] = 999999.
    changed.loc[future, "IsAccurate"] = False
    changed.loc[future, "Compound"] = "WET";changed.loc[future, "Stint"] = 777.
    poisoned = data.build_event(changed, 202201, clock(changed), path)
    for lag in data.LAGS:
        expected = full.frames_by_lag[lag].query("issued_at_timestamp <= 602").reset_index(drop=True)
        actual = poisoned.frames_by_lag[lag].query("issued_at_timestamp <= 602").reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)


def test_future_telemetry_append_payload_poison_cannot_change_prefix(tmp_path):
    raw = laps();prefix = raw.loc[raw.Time <= 602].copy();path = telemetry(tmp_path)
    complete = data.build_event(prefix, 202201, clock(prefix), path)
    past = b"".join(packet(t, 250+(t//5000)%30) for t in range(5000, 600001, 5000))
    path.write_bytes(past+b"00:10:05.000this is an invalid future payload\n")
    poisoned = data.build_event(prefix, 202201, clock(prefix), path)
    for lag in data.LAGS:
        pd.testing.assert_frame_equal(complete.frames_by_lag[lag], poisoned.frames_by_lag[lag], check_exact=True)


def test_empty_warmup_and_unavailable_source_do_not_invent_forecasts(tmp_path):
    raw = laps(last=2)
    def forbidden(_): raise AssertionError("Unavailable stream opened")
    result = data.build_event(raw, 202201, clock(raw), None, source_status="unavailable", cursor_factory=forbidden)
    assert all(frame.empty for frame in result.frames_by_lag.values())
    raw = laps(last=3)
    result = data.build_event(raw, 202201, clock(raw), None, source_status="unavailable", cursor_factory=forbidden)
    for lag, frame in result.frames_by_lag.items():
        assert len(frame) == 4 and not frame.telemetry_supported.any()
        assert set(frame.telemetry_source_status) == {"unavailable"}
        for row in frame.to_dict("records"):
            assert row["telemetry_values"] == features.empty_snapshot(row["driver_id"], cutoff_ns=row["telemetry_cutoff_ns"]).values.tolist()


def test_exact_global_query_order_and_original_output_order(tmp_path):
    raw = laps(last=4);path = telemetry(tmp_path);calls = []
    class Cursor:
        def __init__(self, _): self.last = None;calls.append([]);self.calls = calls[-1]
        def query(self, driver, *, cutoff_ns):
            assert self.last is None or cutoff_ns >= self.last
            self.last = cutoff_ns;self.calls.append((cutoff_ns, driver))
            return features.empty_snapshot(driver, cutoff_ns=cutoff_ns)
        def close(self): pass
    result = data.build_event(raw, 202201, clock(raw), path, cursor_factory=Cursor)
    assert len(calls) == 2 and all(sequence == sorted(sequence) for sequence in calls)
    for frame in result.frames_by_lag.values():
        assert frame.issued_at_timestamp.tolist() == data.frontier.enrich(raw, 202201)[0].issued_at_timestamp.tolist()


def test_external_label_closure_keeps_exact_original_targets_and_unmatched(tmp_path):
    built = event(tmp_path);closed = closure(tmp_path, built)
    saved = data.verify_feature_closure(closed)
    before = {p["path"]:data.sha(p["path"]) for e in saved["events"] for p in e["ledgers"].values()}
    validation = data.validate_downloaded_streams(closed, tmp_path/"input_validation.json")
    label_path = data.attach_labels(closed, tmp_path/"labels", input_validation_path=validation)
    receipt = json.loads(label_path.read_text())
    rows = data._read_rows(receipt["events"][0]["path"])
    raw, _ = data.read_laps(built.source["lap_path"]);original, matched = data.frontier.enrich(raw, 202201)
    assert len(rows) == len(original) and receipt["matched"] == len(matched) and receipt["unmatched"] == 4
    by_key = {tuple(r[k] for k in data.KEYS):r for r in rows}
    for row in matched.to_dict("records"):
        actual = by_key[tuple(row[k] for k in data.KEYS)]
        assert actual["outcome_status"] == "matched"
        assert all(actual[k] == row[k] for k in data.TARGET_FIELDS)
        assert actual["target_at_ns"] > actual["issued_at_ns"]
        assert actual["target_id"] == f"202201/{row['driver_id']}/{int(row['target_lap_number'])}"
    for row in rows:
        if row["outcome_status"] == "unmatched":
            assert all(row[k] is None for k in (*data.TARGET_FIELDS, "target_id", "target_at_ns"))
    assert all(data.sha(path) == value for path, value in before.items())
    with pytest.raises(FileExistsError): data.attach_labels(closed, tmp_path/"labels", input_validation_path=validation)


def test_unavailable_event_retained_in_full_label_population(tmp_path):
    built = event(tmp_path, unavailable=True);closed = closure(tmp_path, built)
    validation = data.validate_downloaded_streams(closed, tmp_path/"validation.json")
    assert json.loads(validation.read_text())["events"][0]["status"] == "unavailable"
    report = data.attach_labels(closed, tmp_path/"labels", input_validation_path=validation)
    assert json.loads(report.read_text())["all_issuances"] == len(built.frames_by_lag[2])


def test_corrupt_download_after_last_query_stops_validation_without_rewriting(tmp_path):
    raw, clocks = source(tmp_path, laps(last=4));path = telemetry(tmp_path, suffix=b"01:00:00.000corrupt\n")
    built = data.build_event(raw, 202201, clocks, path);closed = closure(tmp_path, built)
    files = json.loads(closed.read_text())["events"][0]["ledgers"]
    before = {item["path"]:data.sha(item["path"]) for item in files.values()}
    validation = tmp_path/"invalid_validation.json"
    with pytest.raises(data.packets.PacketError): data.validate_downloaded_streams(closed, validation)
    assert json.loads(validation.read_text())["status"] == "invalid_downloaded_input"
    assert all(data.sha(path) == value for path, value in before.items())
    with pytest.raises(ValueError, match="validation must pass"):
        data.attach_labels(closed, tmp_path/"labels", input_validation_path=validation)
    assert not (tmp_path/"labels").exists()


def test_missing_closure_or_modified_feature_file_blocks_label_attachment(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.attach_labels(tmp_path/"missing.json", tmp_path/"labels", input_validation_path=tmp_path/"validation.json")
    built = event(tmp_path, unavailable=True);closed = closure(tmp_path, built)
    validation = data.validate_downloaded_streams(closed, tmp_path/"validation.json")
    ledger = json.loads(closed.read_text())["events"][0]["ledgers"]["2"]["path"]
    with Path(ledger).open("a") as stream: stream.write("{}\n")
    with pytest.raises(ValueError, match="ledger changed"):
        data.attach_labels(closed, tmp_path/"labels", input_validation_path=validation)


def test_event_set_change_is_not_silently_dropped(tmp_path):
    built = event(tmp_path, unavailable=True)
    design = tmp_path/"design.json";design.write_text("{}")
    with pytest.raises(ValueError, match="event set"):
        data.close_features([built], tmp_path/"features", design_lock_path=design,
                            design_sha256=data.sha(design), expected_event_keys=[202201, 202202])
    assert (tmp_path/"features"/"202201_lag2_issued.jsonl").exists()
    assert not (tmp_path/"features"/"feature_closure.json").exists()
