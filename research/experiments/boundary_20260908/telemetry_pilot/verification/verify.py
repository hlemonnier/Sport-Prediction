"""Independent closed-pilot replay using FastF1's decoder, without lap targets.

This does not import the pilot parser/acquirer/diagnoser. It verifies immutable
receipts, replays physical raw records, and inventories raw channel values.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from fastf1 import _api

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "artifacts/research/boundary_20260908/telemetry_pilot"
ACQUISITION_SHA = "9d39fc16e23db3a3c9066b94e2add265d6a68e639d73417b3d3a813c348ca0e4"
DIAGNOSTICS_SHA = "9f1650bd43d6868874835398debebe12a1727c7bc18df8ae6aa1ceaa5ea92a4f"


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def replay(path):
    # Independent aggregation of all diagnostic values used below. No import of
    # packets.py, no full-session t0_date, no lap table and no outcome access.
    counts = Counter({k: 0 for k in ("entries", "empty_packets", "bundled_packets",
        "max_entries_per_packet", "entries_missing_utc", "entries_invalid_utc",
        "entries_missing_cars", "car_rows", "car_rows_missing_channels")})
    clocks, driver_rows, values, utc_state = [], {}, {}, {}
    first, previous, highwater = None, None, -1
    regressions = repeated = same_available = 0
    for sequence, raw in enumerate(path.read_bytes().splitlines()):
        if not raw.strip():
            raise AssertionError("Unexpected blank record in acquired pilot")
        record = raw.decode("utf-8-sig")
        assert re.fullmatch(r"[0-9]{2}:[0-5][0-9]:[0-5][0-9]\.[0-9]{3}", record[:12])
        hour, minute, seconds = record[:12].split(":")
        second, milli = seconds.split(".")
        timestamp = 3600000*int(hour)+60000*int(minute)+1000*int(second)+int(milli)
        available = max(highwater, timestamp)
        regressions += timestamp < highwater
        repeated += timestamp == previous
        same_available += available == highwater
        if first is None:
            first = timestamp
        clocks.append(available)
        previous, highwater = timestamp, available
        payload = _api.parse(record[12:].strip(), zipped=True)
        assert isinstance(payload, dict) and isinstance(payload["Entries"], list)
        entries = payload["Entries"]
        counts["entries"] += len(entries)
        counts["empty_packets"] += not entries
        counts["bundled_packets"] += len(entries) > 1
        counts["max_entries_per_packet"] = max(counts["max_entries_per_packet"], len(entries))
        for entry in entries:
            assert isinstance(entry, dict)
            utc = entry.get("Utc")
            counts["entries_missing_utc"] += utc is None
            try:
                # Validates date and ordering independently. Native samples here
                # are spaced far beyond sub-microsecond precision; raw strings
                # remain available and are never converted into lap join clocks.
                parsed_utc = datetime.fromisoformat(utc.replace("Z", "+00:00"))
                assert parsed_utc.utcoffset().total_seconds() == 0
            except (AttributeError, ValueError, AssertionError):
                parsed_utc = None
                counts["entries_invalid_utc"] += utc is not None
            cars = entry.get("Cars")
            counts["entries_missing_cars"] += cars is None
            for driver, car in (cars or {}).items():
                d = driver_rows.setdefault(driver, {"car_rows": 0, "first_available_ms": available,
                    "last_available_ms": available, "max_availability_gap_ms": 0,
                    "source_utc_valid": 0, "source_utc_missing_or_invalid": 0,
                    "source_utc_regressions": 0, "source_utc_duplicates": 0})
                d["car_rows"] += 1
                d["max_availability_gap_ms"] = max(d["max_availability_gap_ms"], available-d["last_available_ms"])
                d["last_available_ms"] = available
                d["source_utc_valid"] += parsed_utc is not None
                d["source_utc_missing_or_invalid"] += parsed_utc is None
                if parsed_utc is not None:
                    if driver in utc_state:
                        d["source_utc_regressions"] += parsed_utc < utc_state[driver]
                        d["source_utc_duplicates"] += parsed_utc == utc_state[driver]
                    utc_state[driver] = parsed_utc
                channels = car.get("Channels")
                counts["car_rows"] += 1
                counts["car_rows_missing_channels"] += not isinstance(channels, dict)
                for channel, value in (channels or {}).items():
                    assert type(value) is int, "Pilot raw channel type differs from integer contract"
                    values.setdefault(channel, Counter())[value] += 1
    return {"counts": dict(counts), "driver_rows": driver_rows,
        "packet_summary": {"records": len(clocks), "physical_lines_examined": len(clocks),
            "blank_lines": 0, "first_recorded_ms": first, "last_recorded_ms": previous,
            "first_available_ms": clocks[0], "last_available_ms": clocks[-1],
            "max_availability_gap_ms": max(b-a for a,b in zip(clocks,clocks[1:])),
            "timestamp_regressions": regressions, "adjacent_duplicate_recorded_clocks": repeated,
            "duplicate_availability_clocks": same_available},
        "value_frequencies": values}


def main():
    destination = OUT / "independent_verification.json"
    if destination.exists():
        raise FileExistsError("Independent verification is immutable")
    checked = {}
    def bind(path, expected, size=None):
        path = ROOT / path
        assert sha(path) == expected, path
        if size is not None:
            assert path.stat().st_size == size, path
        checked[str(path.relative_to(ROOT))] = {"sha256": expected, "bytes": path.stat().st_size}
        return path

    acquisition_path = bind(OUT / "acquisition.json", ACQUISITION_SHA)
    diagnostic_path = bind(OUT / "diagnostics.json", DIAGNOSTICS_SHA)
    acquisition, diagnostics = read(acquisition_path), read(diagnostic_path)
    assert diagnostics["status"] == "passed" and "completed_at_utc" in diagnostics
    assert acquisition["downloaded"] == 2
    assert [row["event_key"] for row in acquisition["streams"]] == [202201, 202301]
    acquisition_lock = read(bind(OUT / "acquisition_lock.json", acquisition["acquisition_lock_sha256"]))
    for path, expected in acquisition["source_files"].items():
        bind(path, expected)
    assert acquisition_lock["source_files"] == acquisition["source_files"]
    spec = read(ROOT / "research/experiments/boundary_20260908/telemetry_pilot/specification.json")
    assert acquisition_lock["specification_sha256"] == sha(ROOT / "research/experiments/boundary_20260908/telemetry_pilot/specification.json")
    parent = read(bind(spec["parent_session_manifest"]["path"], spec["parent_session_manifest"]["sha256"]))
    originals = {s["event_key"]: s for s in parent["sessions"]}
    assert acquisition["sessions"] == spec["sessions"]
    assert all(s == originals[s["event_key"]] for s in spec["sessions"])
    assert diagnostics["acquisition_sha256"] == ACQUISITION_SHA
    diagnostic_lock = read(OUT / "diagnostics_lock.json")
    bind(OUT / "diagnostics_lock.json", sha(OUT / "diagnostics_lock.json"))
    for key, value in diagnostic_lock.items():
        assert diagnostics[key] == value
    for path, expected in diagnostics["source_files"].items():
        bind(path, expected)
    for path, expected in diagnostics["inputs"].items():
        bind(path, expected)

    rows = []
    total_attempts = 0
    for stream, expected in zip(acquisition["streams"], diagnostics["streams"], strict=True):
        event = stream["event_key"]
        assert expected["event_key"] == event
        assert stream["status"] == "downloaded_unparsed"
        assert expected["packet_validation"]["complete"] is True
        assert expected["packet_validation"]["stream_input_valid"] is True
        body = bind(stream["decoded_body_path"], stream["decoded_body_sha256"], stream["decoded_body_bytes"])
        source = body.parent / f"{event}_CarData.z.source.json"
        bind(source, sha(source))
        assert read(source) == stream
        assert len(stream["attempts"]) <= 2
        total_attempts += len(stream["attempts"])
        for ordinal, attempt in enumerate(stream["attempts"], 1):
            url = spec["base_urls"][ordinal-1]+"/static/"+stream["session_path"]+"CarData.z.jsonStream"
            assert attempt["actual_url"] == attempt["requested_url"] == url
            bind(attempt["wire_body_path"], attempt["wire_body_sha256"], attempt["wire_body_bytes"])
            assert attempt["wire_body_complete"] is True
            if "Content-Length" in attempt["response_headers"]:
                assert int(attempt["response_headers"]["Content-Length"]) == attempt["wire_body_bytes"]
            receipt = body.parent / f"{event}_CarData.z.attempt{ordinal}.receipt.json"
            bind(receipt, sha(receipt))
            assert read(receipt) == attempt
            assert attempt["wire_body_bytes"] <= spec["max_http_body_bytes"]
        assert stream["attempts"][-1]["http_status"] == 200
        assert stream["decoded_body_bytes"] <= spec["max_decoded_body_bytes"]
        raw = replay(body)
        assert raw["counts"] == expected["counts"], event
        assert raw["driver_rows"] == expected["drivers"], event
        for key, value in raw["packet_summary"].items():
            assert value == expected["packet_validation"][key], (event, key)
        for channel, frequencies in raw["value_frequencies"].items():
            reported = expected["channels"][channel]
            assert sum(frequencies.values()) == reported["present"]
            assert frequencies[0] == reported["zero"]
            assert min(frequencies) == reported["minimum"] and max(frequencies) == reported["maximum"]
            assert reported["null"] == reported["boolean"] == reported["nonnumeric"] == 0
            assert reported["absent"] == raw["counts"]["car_rows"]-reported["present"]
        assert set(raw["value_frequencies"]) == set(expected["channels"])
        frequencies = raw["value_frequencies"]
        anomalies = {}
        for channel, low, high, name in [("3", 0, 8, "gear_outside_0_to_8"),
                                       ("4", 0, 100, "throttle_outside_0_to_100")]:
            bad = {str(k): v for k,v in sorted(frequencies[channel].items()) if not low <= k <= high}
            anomalies[name] = {"rows": sum(bad.values()), "frequency": bad,
                "fraction": sum(bad.values())/raw["counts"]["car_rows"]}
        raw["value_frequencies"] = {k: {str(value): count for value,count in sorted(v.items())}
                                    for k,v in sorted(frequencies.items())}
        raw.update(event_key=event, source_body_path=stream["decoded_body_path"], range_diagnostics=anomalies,
            brake_contract="Raw channel 5 codes retained; nonzero indicator is not analog brake pressure.")
        rows.append(raw)
        print(json.dumps({"event_key": event, "records": raw["packet_summary"]["records"],
            "car_rows": raw["counts"]["car_rows"], "range_diagnostics": anomalies,
            "raw_brake_frequency": raw["value_frequencies"]["5"]}), flush=True)
    assert total_attempts <= 4
    bind(__file__, sha(__file__))
    decoder_source = Path(_api.__file__)
    result = {"status": "passed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": checked, "unique_bindings_checked": len(checked), "http_attempts": total_attempts,
        "implementation": "Independent physical-record replay using installed FastF1 _api.parse; no pilot parser imports",
        "fastf1_decoder": {"path": str(decoder_source), "sha256": sha(decoder_source)},
        "streams": rows, "models_fitted": 0, "predictive_scores_computed": 0,
        "lap_tables_opened": 0, "network_calls": 0,
        "limits": ["Raw full-session availability coverage is not original-issuance window support.",
            "UTC order check uses independent microsecond datetime parsing; raw channel/packet values are exact.",
            "Archive availability is not certified historical client receipt latency.",
            "No range anomalies are clipped, dropped or interpreted as physical pressure/SOC."],
        "suggested_commit": "research(f1-live): verify raw telemetry pilot and freeze measurement proposal"}
    with destination.open("x") as file:
        json.dump(result,file,indent=2,sort_keys=True,allow_nan=False)
        file.write("\n")
    print(json.dumps({"path": str(destination.relative_to(ROOT)), "sha256": sha(destination), "status": "passed"}))


if __name__ == "__main__":
    main()
