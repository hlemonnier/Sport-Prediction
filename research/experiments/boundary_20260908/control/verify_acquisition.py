"""Verify the discovery-only acquisition without reading prediction outcomes."""
from collections import Counter
import inspect
import json
from pathlib import Path

import fastf1
from fastf1 import _api
import pandas as pd

from research.experiments.boundary_20260908.control import acquire


def verify():
    root, out = acquire.ROOT, acquire.OUT
    path = out / "input_manifest.json"
    manifest = json.loads(path.read_text())
    assert not manifest["failed_events"] and len(manifest["events"]) == 44
    assert acquire.sha(acquire.__file__) == manifest["source_sha256"]
    assert acquire.sha(acquire.WEATHER_SOURCE) == manifest["audited_fetch_source_sha256"]
    assert acquire.sha(acquire.WEATHER_MANIFEST) == manifest["weather_manifest_sha256"]
    weather = json.loads(acquire.WEATHER_MANIFEST.read_text())
    event_map = {row["event_key"]: row for row in weather["events"]}
    native = inspect.unwrap(_api.track_status_data)
    counts, checked, paired, size = Counter(), 0, 0, 0
    duplicates = []
    for event in manifest["events"]:
        assert event["event_key"] // 100 in (2022, 2023)
        source = event_map[event["event_key"]]
        for field in ("meeting_name", "meeting_key", "session_key", "session_path"):
            assert event[field] == source[field]
        for file_field, hash_field in (("raw_path", "raw_sha256"), ("csv_path", "csv_sha256"),
                                       ("source_receipt_path", "source_receipt_sha256")):
            file = root / event[file_field]
            assert acquire.sha(file) == event[hash_field], file
            checked += 1
            size += file.stat().st_size
        raw = (root / event["raw_path"]).read_bytes()
        parsed = acquire.parse(raw)
        saved = pd.read_csv(root / event["csv_path"], dtype={"Status": str, "Message": str},
                            keep_default_na=False)
        pd.testing.assert_frame_equal(saved, parsed, check_exact=False, atol=1e-12, rtol=0)
        response = [[line[:12], json.loads(line[12:])] for line in raw.decode("utf-8-sig").splitlines()
                    if line.strip()]
        reference = native("unused", response=response)
        assert reference["Status"] == parsed.Status.tolist()
        assert reference["Message"] == parsed.Message.tolist()
        assert [t.total_seconds() for t in reference["Time"]] == parsed.Time.tolist()
        paired += len(parsed)
        counts.update(parsed.Status.tolist())
        if parsed.Time.duplicated().any():
            duplicates.append({"event_key": event["event_key"],
                               "rows": parsed.loc[parsed.Time.duplicated(keep=False)].to_dict("records")})
        receipt = json.loads((root / event["source_receipt_path"]).read_text())
        assert receipt["sha256"] == event["raw_sha256"]
        assert receipt["url"] == event["actual_download_url"]
    result = {
        "status": "acquisition_verified_model_not_started",
        "fastf1_version": fastf1.__version__,
        "source_sha256": acquire.sha(__file__),
        "acquisition_source_sha256": acquire.sha(acquire.__file__),
        "input_manifest_sha256": acquire.sha(path),
        "fastf1_api_source_sha256": acquire.sha(_api.__file__),
        "events": len(manifest["events"]), "verified_input_files": checked,
        "input_bytes": size, "native_parser_exact_rows": paired,
        "status_counts": dict(counts), "ordered_duplicate_timestamp_records": duplicates,
        "unknown_status_rows": sum(n for status, n in counts.items() if status not in ("1", "2", "4", "5", "6", "7")),
        "partial_status_rows": sum(e["partial_status_rows"] for e in manifest["events"]),
        "time_contract": "Raw 12-character session clock; native FastF1 parser agrees exactly. No clock offset.",
        "receipt_limit": "Download receipts verify acquisition provenance; historical message receipt latency is unknown.",
        "model_status": "Paused before feature engineering, grid freeze, fitting or score inspection; no transfer inputs acquired.",
    }
    acquire.write(out / "acquisition_verification.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "ordered_duplicate_timestamp_records"}, indent=2))
    return result


if __name__ == "__main__":
    verify()
