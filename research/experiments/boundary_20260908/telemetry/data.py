"""Original lap-issuance telemetry assembly; no model fitting or scoring.

Historical orchestration must freeze this source and inputs before calling it.
Features, downloaded-stream validation, and external labels close separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
from datetime import datetime, timezone
from decimal import Decimal, DecimalException, localcontext
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from research.experiments.frontier_20260907 import live_frontier as frontier
from research.experiments.boundary_20260908.telemetry_pilot import packets

ROOT = Path(__file__).resolve().parents[4]
KEYS = tuple(frontier.KEYS)
LAGS = (0, 2)
TARGET_FIELDS = ("target_lap_number", "target_timestamp", "lap_time_seconds", "target_same_stint")
_extra = ["observed_lap", "position", "lap_duration", "fresh_tyre", "history_count"]
_extra += [f"lag_{k}_{suffix}" for k in range(1, 7) for suffix in ("gap", "lap_distance")]
_extra += [f"{prefix}_{k}" + ("_gap" if prefix in ("mean", "median") else "")
           for k in (2, 3, 5, 8) for prefix in ("mean", "median", "mad", "trend", "range")]
_extra += [f"reset_ewma_{a}_gap" for a in (.25, .5, .75)]
_extra += [name+suffix for name in (*frontier.SECTORS, *frontier.SPEEDS) for suffix in ("", "_missing", "_median_gap")]
_extra += ["compound_"+name for name in ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")]
_extra += ["peer_count", "peer_delta_mean", "peer_delta_median", "peer_delta_mad", "relative_field_pace"]
BASE_FEATURES = tuple(frontier.base.FEATURES)+tuple("x_"+name for name in _extra)
assert len(BASE_FEATURES) == 80 and len(set(BASE_FEATURES)) == 80
ADDED_COLUMNS = ("year", "issuance_id", "issued_at_ns", "telemetry_lag_seconds", "telemetry_cutoff_ns",
                 "telemetry_source_status", "telemetry_supported", "telemetry_values", "telemetry_provenance")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json_value(value):
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float):
        if math.isnan(value): return None
        if not math.isfinite(value): raise ValueError("Infinity cannot enter an issuance ledger")
    if value is pd.NA: return None
    if value is None or isinstance(value, (str, int, float, bool)): return value
    raise TypeError("Unsupported ledger value: " + type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_json(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_json_value(value), stream, indent=2, sort_keys=True, allow_nan=False);stream.write("\n")


def _read_json(path):
    def bad(value): raise ValueError("Nonstandard JSON: " + value)
    return json.loads(Path(path).read_text(), parse_constant=bad)


def _write_rows(path, rows):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(_json_value(row), sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n")


def _read_rows(path):
    def bad(value): raise ValueError("Nonstandard JSON: " + value)
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line, parse_constant=bad) for line in stream]


def seconds_to_ns(text):
    """Convert original decimal text exactly; floats and sub-ns values are errors."""
    if isinstance(text, bool) or not isinstance(text, (str, int, Decimal)):
        raise TypeError("Exact clock requires original decimal text, Decimal or integer seconds")
    try:
        with localcontext() as context:
            context.prec = max(50, len(str(text))+12)
            value = Decimal(text)
            if not value.is_finite() or value < 0:
                raise ValueError("Clock must be finite and nonnegative")
            nanos = value * 1_000_000_000
            if nanos != nanos.to_integral_value():
                raise ValueError("Original CSV Time is not an integral number of nanoseconds; no rounding is allowed")
            if nanos >= 2**63:
                raise ValueError("Clock exceeds signed 64-bit nanoseconds")
            return int(nanos)
    except (DecimalException, OverflowError) as exc:
        raise ValueError("Invalid exact decimal clock") from exc


@dataclass(frozen=True)
class LapClock:
    nanoseconds: int
    original_float: float
    source_row: int


def make_clock_index(raw, time_texts):
    if len(raw) != len(time_texts): raise ValueError("CSV clock text population differs")
    result = {}
    for index, ((driver, lap, stamp), text) in enumerate(zip(raw[["DriverNumber", "LapNumber", "Time"]].itertuples(index=False, name=None), time_texts)):
        driver = frontier.base.driver_key(driver)
        number = frontier.base.numeric(lap)
        stamp = frontier.base.numeric(stamp)
        if not math.isfinite(number) or number != int(number) or number <= 0 or not math.isfinite(stamp):
            raise ValueError("Invalid original driver/lap/clock metadata")
        key = (driver, int(number))
        if key in result: raise ValueError("Duplicate original driver/lap identity")
        result[key] = LapClock(seconds_to_ns(text), float(stamp), index)
    return result


def read_laps(path, *, expected_sha256=None):
    path = Path(path);before = sha(path)
    if expected_sha256 is not None and before != expected_sha256: raise ValueError("Original lap input hash changed")
    raw = pd.read_csv(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        lexical = list(csv.DictReader(stream))
    if len(raw) != len(lexical): raise ValueError("CSV parser row populations differ")
    for row, original in zip(raw[["DriverNumber", "LapNumber"]].itertuples(index=False, name=None), lexical):
        if frontier.base.driver_key(row[0]) != frontier.base.driver_key(original["DriverNumber"]) or Decimal(str(row[1])) != Decimal(original["LapNumber"]):
            raise ValueError("Original CSV row order/identity differs")
    clocks = make_clock_index(raw, [row["Time"] for row in lexical])
    if sha(path) != before: raise ValueError("Lap source changed while reading")
    raw.attrs["source_path"] = str(path.resolve())
    raw.attrs["source_sha256"] = before
    return raw, clocks


def _original(raw, event_key):
    try:
        issued, paired = frontier.enrich(raw, event_key)
    except (AttributeError, KeyError):
        # Frozen enrich lacks schemas for both zero issuances and first issues
        # with zero subsequent targets. The published inference-only encoder
        # preserves the same features without requiring a matched target table.
        old, old_matched = frontier.base.stream_event(raw, event_key)
        if old.empty:
            columns = list(dict.fromkeys([*KEYS, "forecast_naive_seconds", *BASE_FEATURES]))
            return pd.DataFrame(columns=columns), pd.DataFrame(columns=[*KEYS, *TARGET_FIELDS])
        if not old_matched.empty: raise
        from research.experiments.frontier_20260907.feature_encoder import observed_features
        issued = observed_features(raw, event_key)
        paired = pd.DataFrame(columns=[*KEYS, *TARGET_FIELDS])
    if issued.duplicated(list(KEYS)).any(): raise ValueError("Duplicate original issuance")
    if tuple(frontier.base.FEATURES)+tuple(c for c in issued if c.startswith("x_")) != BASE_FEATURES:
        raise ValueError("Original ordered 80-feature contract changed")
    if set(ADDED_COLUMNS) & set(issued): raise ValueError("New columns would overwrite an original field")
    if set(TARGET_FIELDS) & set(issued): raise ValueError("Original issuance contains targets")
    if not np.isfinite(issued[list(BASE_FEATURES)].to_numpy(float)).all(): raise ValueError("Original feature is not finite")
    return issued, paired


def _clock(row, clocks, *, target=False):
    lap = int(row["target_lap_number"] if target else row["issued_after_lap_number"])
    stamp = float(row["target_timestamp"] if target else row["issued_at_timestamp"])
    result = clocks[(str(row["driver_id"]), lap)]
    if result.original_float != stamp: raise ValueError("Exact clock sidecar disagrees with frozen float key")
    return result


def _issuance_id(event, driver, lap, nanos):
    return f"{int(event)}/{driver}/{int(lap)}/{int(nanos)}"


@dataclass
class EventFeatures:
    event_key: int
    frames_by_lag: dict[int, pd.DataFrame]
    original_columns: tuple[str, ...]
    telemetry_feature_names: tuple[str, ...]
    source: dict


def build_event(raw, event_key, clocks, telemetry_path, *, source_status="downloaded",
                cursor_factory=None, empty_factory=None, telemetry_feature_names=None,
                telemetry_sha256=None):
    """Build both fixed-lag, all-issuance frames; discard internal target matches."""
    if type(event_key) is not int or event_key <= 0: raise ValueError("Integer event key required")
    if source_status not in ("downloaded", "unavailable"): raise ValueError("Unknown telemetry acquisition status")
    if cursor_factory is None or empty_factory is None or telemetry_feature_names is None:
        from . import features
        cursor_factory = features.TelemetryCursor if cursor_factory is None else cursor_factory
        empty_factory = features.empty_snapshot if empty_factory is None else empty_factory
        telemetry_feature_names = features.FEATURE_NAMES if telemetry_feature_names is None else telemetry_feature_names
    names = tuple(telemetry_feature_names)
    if len(names) != 90 or len(set(names)) != 90: raise ValueError("Exactly 90 ordered telemetry features are required")
    if source_status == "unavailable" and telemetry_path is not None: raise ValueError("Unavailable source cannot name a downloaded body")
    actual_telemetry_sha = None
    if source_status == "downloaded":
        if telemetry_path is None: raise ValueError("Downloaded source needs a path")
        actual_telemetry_sha = sha(telemetry_path)
        if telemetry_sha256 is not None and actual_telemetry_sha != telemetry_sha256: raise ValueError("Telemetry source hash changed")
    issued, _discarded_matches = _original(raw, event_key)
    del _discarded_matches
    records = issued.to_dict("records")
    exact = [_clock(row, clocks).nanoseconds for row in records]
    ids = [_issuance_id(event_key, row["driver_id"], row["issued_after_lap_number"], ns) for row, ns in zip(records, exact)]
    if len(set(ids)) != len(ids): raise ValueError("Duplicate exact-clock issuance identity")
    order = sorted(range(len(records)), key=lambda i: (exact[i], str(records[i]["driver_id"]), int(records[i]["issued_after_lap_number"])))
    frames = {}
    for lag in LAGS:
        snapshots = [None]*len(records)
        cursor = cursor_factory(telemetry_path) if source_status == "downloaded" else None
        try:
            for index in order:
                row = records[index];cutoff = exact[index]-lag*1_000_000_000
                snapshot = (cursor.query(str(row["driver_id"]), cutoff_ns=cutoff) if cursor is not None
                            else empty_factory(str(row["driver_id"]), cutoff_ns=cutoff))
                values = np.asarray(snapshot.values, dtype=float)
                if values.shape != (90,) or not np.isfinite(values).all() or type(snapshot.supported) is not bool:
                    raise ValueError("Invalid fixed-shape telemetry snapshot")
                if source_status == "unavailable" and snapshot.supported: raise ValueError("Unavailable source claims support")
                snapshots[index] = (values.tolist(), snapshot.supported, _json_value(snapshot.provenance))
        finally:
            if cursor is not None: cursor.close()
        frame = issued.copy(deep=True)
        frame["year"] = event_key//100
        frame["issuance_id"] = ids
        frame["issued_at_ns"] = pd.Series(exact, index=frame.index, dtype="int64")
        frame["telemetry_lag_seconds"] = lag
        frame["telemetry_cutoff_ns"] = frame["issued_at_ns"]-lag*1_000_000_000
        frame["telemetry_source_status"] = source_status
        frame["telemetry_supported"] = pd.Series([v[1] for v in snapshots], index=frame.index, dtype=bool)
        frame["telemetry_values"] = [v[0] for v in snapshots]
        frame["telemetry_provenance"] = [v[2] for v in snapshots]
        pd.testing.assert_frame_equal(frame[list(issued.columns)], issued, check_exact=True)
        frames[lag] = frame
    if actual_telemetry_sha is not None and sha(telemetry_path) != actual_telemetry_sha: raise ValueError("Telemetry source changed while streaming")
    source = {"lap_path": raw.attrs.get("source_path"), "lap_sha256": raw.attrs.get("source_sha256"),
              "telemetry_status": source_status, "telemetry_path": str(Path(telemetry_path).resolve()) if telemetry_path is not None else None,
              "telemetry_sha256": actual_telemetry_sha}
    return EventFeatures(event_key, frames, tuple(issued.columns), names, source)


def close_features(events, output_dir, *, design_lock_path, design_sha256, expected_event_keys):
    """Exclusively write all events and an independent feature-closure receipt."""
    design_lock_path = Path(design_lock_path).resolve()
    if sha(design_lock_path) != design_sha256: raise ValueError("Design lock binding mismatch")
    output_dir = Path(output_dir).resolve();entries = [];seen = set();names = None
    for event in events:
        if event.event_key in seen: raise ValueError("Duplicate materialized event")
        seen.add(event.event_key)
        if set(event.frames_by_lag) != set(LAGS): raise ValueError("Both fixed lag ledgers are required")
        if names is None: names = event.telemetry_feature_names
        if names != event.telemetry_feature_names: raise ValueError("Telemetry feature order changed between events")
        source = event.source
        if not source["lap_path"] or sha(source["lap_path"]) != source["lap_sha256"]: raise ValueError("Unbound original lap source")
        if source["telemetry_status"] == "downloaded" and sha(source["telemetry_path"]) != source["telemetry_sha256"]: raise ValueError("Unbound telemetry source")
        primary = event.frames_by_lag[2]
        paths = {}
        for lag, frame in event.frames_by_lag.items():
            pd.testing.assert_frame_equal(frame[list(event.original_columns)], primary[list(event.original_columns)], check_exact=True)
            if frame.issuance_id.tolist() != primary.issuance_id.tolist(): raise ValueError("Lag population mismatch")
            if any(name in frame for name in (*TARGET_FIELDS, "outcome_status", "target_id")): raise ValueError("External targets entered feature closure")
            path = output_dir/f"{event.event_key}_lag{lag}_issued.jsonl"
            _write_rows(path, frame.to_dict("records"))
            paths[str(lag)] = {"path": str(path), "sha256": sha(path), "rows": len(frame)}
        entries.append({"event_key": event.event_key, "source": source, "original_columns": list(event.original_columns),
                        "issuance_ids_sha256": digest(primary.issuance_id.tolist()), "rows": len(primary), "ledgers": paths})
    if seen != set(expected_event_keys) or len(seen) != len(expected_event_keys): raise ValueError("Materialized event set differs from locked inputs")
    if sha(design_lock_path) != design_sha256: raise ValueError("Design lock changed during feature construction")
    receipt = {"schema": "original_issuance_telemetry_features_v1", "closed_at_utc": _now(),
               "design_lock": {"path": str(design_lock_path), "sha256": design_sha256},
               "base_features": list(BASE_FEATURES), "telemetry_features": list(names or ()), "lags_seconds": list(LAGS),
               "events": entries, "issuances_per_lag": sum(r["rows"] for r in entries),
               "external_labels_attached": False, "model_fits_performed": 0,
               "downloaded_stream_integrity": "pending_separate_validation"}
    path = output_dir/"feature_closure.json"
    _write_json(path, receipt)
    return path


def verify_feature_closure(path):
    receipt = _read_json(path)
    if receipt["external_labels_attached"] is not False or receipt["base_features"] != list(BASE_FEATURES): raise ValueError("Invalid target-free feature contract")
    binding = receipt["design_lock"]
    if sha(binding["path"]) != binding["sha256"]: raise ValueError("Design lock changed")
    for event in receipt["events"]:
        if sha(event["source"]["lap_path"]) != event["source"]["lap_sha256"]: raise ValueError("Lap source changed after feature closure")
        source = event["source"]
        if source["telemetry_status"] == "downloaded" and sha(source["telemetry_path"]) != source["telemetry_sha256"]: raise ValueError("Telemetry source changed after feature closure")
        for ledger in event["ledgers"].values():
            if sha(ledger["path"]) != ledger["sha256"]: raise ValueError("Target-free ledger changed")
    return receipt


def validate_downloaded_streams(feature_closure_path, output_path):
    """Exhaust downloaded sources after closure; preserve failures, never rewrite."""
    receipt = verify_feature_closure(feature_closure_path);checks = []
    base = {"feature_closure_sha256": sha(feature_closure_path), "started_at_utc": _now()}
    try:
        for event in receipt["events"]:
            source = event["source"]
            if source["telemetry_status"] == "unavailable":
                checks.append({"event_key": event["event_key"], "status": "unavailable", "parsed": False});continue
            stats = {}
            for _ in packets.iter_packets(source["telemetry_path"], stats=stats): pass
            if stats.get("stream_input_valid") is not True or stats.get("complete") is not True: raise ValueError("Incomplete downloaded stream validation")
            checks.append({"event_key": event["event_key"], "status": "valid_downloaded", "parsed": True, "statistics": stats})
    except Exception as exc:
        _write_json(output_path, {**base, "status": "invalid_downloaded_input", "completed_at_utc": _now(),
                                 "events": checks, "failing_event_key": event["event_key"], "failure": type(exc).__name__+": "+str(exc)})
        raise
    _write_json(output_path, {**base, "status": "PASS", "completed_at_utc": _now(), "events": checks})
    return Path(output_path)


def attach_labels(feature_closure_path, output_dir, *, input_validation_path):
    """Attach original next-eligible targets only after immutable feature closure."""
    receipt = verify_feature_closure(feature_closure_path)
    validation = _read_json(input_validation_path)
    if validation.get("status") != "PASS" or validation.get("feature_closure_sha256") != sha(feature_closure_path):
        raise ValueError("Full source validation must pass before external label attachment")
    output_dir = Path(output_dir).resolve();entries = []
    for event in receipt["events"]:
        event_key = event["event_key"]
        raw, clocks = read_laps(event["source"]["lap_path"], expected_sha256=event["source"]["lap_sha256"])
        original, paired = _original(raw, event_key)
        primary = _read_rows(event["ledgers"]["2"]["path"])
        if len(primary) != len(original): raise ValueError("Original issuance population changed before labels")
        for saved, actual in zip(primary, original.to_dict("records")):
            if {k: saved[k] for k in event["original_columns"]} != _json_value(actual): raise ValueError("Original features changed before labels")
        if digest([r["issuance_id"] for r in primary]) != event["issuance_ids_sha256"]: raise ValueError("Issuance identity hash mismatch")
        matches = {tuple(row[k] for k in KEYS): row for row in paired.to_dict("records")}
        if len(matches) != len(paired): raise ValueError("Duplicate original target association")
        labels = []
        for row in primary:
            match = matches.get(tuple(row[k] for k in KEYS))
            label = {k: row[k] for k in (*KEYS, "year", "issuance_id", "issued_at_ns")}
            label.update(outcome_status="unmatched", target_id=None, target_at_ns=None, **{k: None for k in TARGET_FIELDS})
            if match is not None:
                target_clock = _clock(match, clocks, target=True)
                if target_clock.nanoseconds <= row["issued_at_ns"]: raise ValueError("Target is not strictly after issuance")
                label.update({k: match[k] for k in TARGET_FIELDS})
                label.update(outcome_status="matched", target_at_ns=target_clock.nanoseconds,
                             target_id=f"{event_key}/{row['driver_id']}/{int(match['target_lap_number'])}")
            labels.append(label)
        if sum(r["outcome_status"] == "matched" for r in labels) != len(paired): raise ValueError("Matched/unmatched retention mismatch")
        path = output_dir/f"{event_key}_labels.jsonl";_write_rows(path, labels)
        entries.append({"event_key": event_key, "path": str(path), "sha256": sha(path), "rows": len(labels),
                        "matched": len(paired), "unmatched": len(labels)-len(paired)})
    verify_feature_closure(feature_closure_path)
    report = {"schema": "original_issuance_telemetry_labels_v1", "closed_at_utc": _now(),
              "feature_closure": {"path": str(Path(feature_closure_path).resolve()), "sha256": sha(feature_closure_path)},
              "input_validation": {"path": str(Path(input_validation_path).resolve()), "sha256": sha(input_validation_path)},
              "events": entries, "all_issuances": sum(r["rows"] for r in entries), "matched": sum(r["matched"] for r in entries),
              "unmatched": sum(r["unmatched"] for r in entries), "feature_ledgers_modified": False}
    path = output_dir/"label_closure.json";_write_json(path, report)
    return path
