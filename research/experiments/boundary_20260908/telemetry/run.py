"""Immutable freeze/prepare/select execution for the original-issuance experiment.

Suggested commit: research(f1-live): execute locked original-horizon telemetry study
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import traceback

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from . import data, evaluate, features, models

ROOT = data.ROOT
HERE = Path(__file__).resolve().parent
OUT = ROOT/"artifacts/research/boundary_20260908/telemetry/execution"
ACQUISITION = ROOT/"artifacts/research/boundary_20260908/telemetry/acquisition.json"
TARGET_COLUMNS = ("outcome_status", "target_id", "target_at_ns", *data.TARGET_FIELDS)


def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return data._read_json(path)
def save(path, value): return data._write_json(path, value)
def record(path): return {"path": str(Path(path).resolve()), "sha256": data.sha(path)}
def progress(**value): print(json.dumps(value, allow_nan=False), flush=True)


def check_record(value):
    if data.sha(value["path"]) != value["sha256"]: raise ValueError("Hash binding changed: "+value["path"])
    return Path(value["path"])


def bound_path(spec, suffix):
    matches = [ROOT/r["path"] for r in spec["original_frontier_bindings"] if r["path"].endswith(suffix)]
    if len(matches) != 1: raise ValueError("Unique original binding required: "+suffix)
    return matches[0]


def validate_spec(spec):
    expected = list(range(202201, 202223))+list(range(202301, 202323))
    if (spec["discovery"]["event_keys"] != expected or spec["discovery"]["train_years"] != [2022]
            or spec["discovery"]["selection_years"] != [2023]): raise ValueError("Unexpected discovery dates")
    f, m, gate = spec["features"], spec["models"], spec["selection_gate"]
    if (f["primary_lag_seconds"], f["sensitivity_lag_seconds"], f["sensitivity_refit"]) != (2, 0, False): raise ValueError("Fixed latency policy changed")
    if (len(data.BASE_FEATURES), len(features.FEATURE_NAMES), len(features.CONTENT_INDICES), len(features.QUALITY_INDICES)) != (80, 90, 27, 63): raise ValueError("Feature layout changed")
    if tuple(data.BASE_FEATURES) != tuple(models.BASE_FEATURES): raise ValueError("Data/model feature orders differ")
    expected_model = {"names": list(models.MODEL_NAMES), "fits_total": 3, "max_leaf_nodes": 15, "max_iter": 150,
        "loss": "absolute_error", "learning_rate": .06, "min_samples_leaf": 80, "l2_regularization": 10,
        "early_stopping": False, "random_state": 20260907, "correction_clip_seconds": 3,
        "expected_base_selection_event_mae_seconds": models.EXPECTED_SELECTION_BASE_MAE}
    if any(m[key] != value for key, value in expected_model.items()): raise ValueError("Frozen model configuration changed")
    if gate["candidate"] != "telemetry_hgb" or gate["references"] != ["base_hgb", "quality_hgb"] or gate["minimum_relative_event_mae_reduction_each_reference"] != .01:
        raise ValueError("Frozen candidate/reference gate changed")
    if spec["uncertainty"]["seed"] != 20260907 or spec["uncertainty"]["resamples_each"] != 20000: raise ValueError("Uncertainty protocol changed")


def sources():
    spec = read(HERE/"specification.json")
    paths = {p for pattern in ("*.py", "*.json", "*.md") for p in HERE.glob(pattern)}
    for binding in spec["original_frontier_bindings"]:
        if binding["path"].startswith("research/"):
            path = ROOT/binding["path"]
            if data.sha(path) != binding["sha256"]: raise ValueError("Frozen incumbent source changed")
            paths.add(path)
    for name, expected in features.dependency_bindings().items():
        path = ROOT/name
        if data.sha(path) != expected: raise ValueError("Telemetry source dependency changed")
        paths.add(path)
    paths.add(ROOT/"research/experiments/performance_20260907/live/spec.json")
    return {str(p.relative_to(ROOT)): data.sha(p) for p in sorted(paths)}


def input_bindings(spec, acquisition_path):
    bindings = {r["path"]: r["sha256"] for r in spec["original_frontier_bindings"]}
    for key in ("acquisition_contract", "measurement_contract"):
        bindings[spec[key]["path"]] = spec[key]["sha256"]
    for path, expected in bindings.items():
        if data.sha(ROOT/path) != expected: raise ValueError("Specification input drift: "+path)
    contract = read(ROOT/spec["acquisition_contract"]["path"])
    original = read(bound_path(spec, "/selection.json"))
    inventory = original["input_manifest"]
    if original["features"] != list(data.BASE_FEATURES): raise ValueError("Original 80-feature order differs")
    if [r["event_key"] for r in inventory] != spec["discovery"]["event_keys"]: raise ValueError("Original discovery event order differs")
    if sum(r["issuances"] for r in inventory) != spec["discovery"]["expected_original_issuances"] or sum(r["matched_rows"] for r in inventory) != spec["discovery"]["matched_rows"]:
        raise ValueError("Original discovery denominator differs")
    acquired = read(acquisition_path)
    if acquired["status"] != "all_discovery_attempts_recorded" or acquired["contract_sha256"] != spec["acquisition_contract"]["sha256"]:
        raise ValueError("Acquisition is not closed under the declared contract")
    if acquired["sessions"] != contract["sessions"] or [r["event_key"] for r in acquired["streams"]] != spec["discovery"]["event_keys"]:
        raise ValueError("Acquired session identity/order differs")
    bindings[str(Path(acquisition_path).resolve())] = data.sha(acquisition_path)
    lock_path = Path(acquisition_path).parent/"acquisition_lock.json"
    bindings[str(lock_path.resolve())] = acquired["acquisition_lock_sha256"]
    if data.sha(lock_path) != acquired["acquisition_lock_sha256"]: raise ValueError("Acquisition lock changed")
    acquisition_lock = read(lock_path)
    bindings[str((lock_path.parent/"acquisition_review.json").resolve())] = acquisition_lock["acquisition_review_sha256"]
    def add_receipts(receipts):
        for path, binding in receipts.items():
            if (ROOT/path).stat().st_size != binding["bytes"]: raise ValueError("Acquisition receipt byte count changed: "+path)
            if path in bindings and bindings[path] != binding["sha256"]: raise ValueError("Conflicting receipt binding: "+path)
            bindings[path] = binding["sha256"]
    add_receipts(acquired["output_bindings"])
    for path, value in acquired["source_files"].items(): bindings[path] = value
    for item in [contract["parent_manifest"], *contract["bindings"]]: bindings[item["path"]] = item["sha256"]
    pilot_verifications = [item for item in contract["bindings"] if item["path"].endswith("/independent_verification.json")]
    if len(pilot_verifications) != 1: raise ValueError("One reviewed reused-pilot closure required")
    pilot = pilot_verifications[0]
    if data.sha(ROOT/pilot["path"]) != pilot["sha256"]: raise ValueError("Reused-pilot verification changed")
    add_receipts(read(ROOT/pilot["path"])["bindings"])
    sessions = {r["event_key"]: r for r in contract["sessions"]}
    for item, stream in zip(inventory, acquired["streams"]):
        session = sessions[item["event_key"]]
        if item["path"] != session["original_laps_path"] or item["sha256"] != session["original_laps_sha256"]:
            raise ValueError("Acquisition/original lap binding differs")
        bindings[item["path"]] = item["sha256"]
        if stream["status"] not in ("downloaded_unparsed", "reused_verified_pilot", "unavailable"):
            raise ValueError("Unknown acquisition source state")
        if stream["session_path"] != session["session_path"]: raise ValueError("Wrong telemetry race session")
        if stream["status"] != "unavailable":
            bindings[stream["decoded_body_path"]] = stream["decoded_body_sha256"]
            if (ROOT/stream["decoded_body_path"]).stat().st_size != stream["decoded_body_bytes"]: raise ValueError("Telemetry body byte count changed")
        for attempt in stream.get("attempts", []):
            if "wire_body_path" in attempt: bindings[attempt["wire_body_path"]] = attempt["wire_body_sha256"]
    for path, expected in bindings.items():
        if data.sha(ROOT/path) != expected: raise ValueError("Acquired/input file hash mismatch: "+path)
    return bindings, inventory


def freeze(out, review_path, acquisition_path):
    out = Path(out)
    if (out/"design_lock.json").exists(): raise FileExistsError("Existing design is immutable")
    spec = read(HERE/"specification.json");validate_spec(spec)
    before = sources();review = read(review_path)
    if review.get("approved_for_execution_lock") is not True or review.get("source_files") != before:
        raise ValueError("Independent review must approve exactly these source bytes")
    bindings, inventory = input_bindings(spec, acquisition_path)
    command = [sys.executable, "-m", "pytest", "-q", "--import-mode=importlib", "-p", "no:cacheprovider", str(HERE)]
    tested = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    save(out/"pre_fit_tests.json", {"command": command, "exit_code": tested.returncode, "stdout": tested.stdout,
                                  "stderr": tested.stderr, "source_files": before, "completed_at_utc": now()})
    if tested.returncode: raise ValueError("Pre-fit test suite failed")
    if sources() != before: raise ValueError("Sources changed during tests")
    for name, expected in bindings.items():
        if data.sha(ROOT/name) != expected: raise ValueError("Inputs changed during tests")
    save(out/"design_lock.json", {"closed_at_utc": now(), "sources": before, "inputs": bindings,
        "specification": record(HERE/"specification.json"), "independent_review": record(review_path),
        "pre_fit_tests": record(out/"pre_fit_tests.json"), "acquisition_manifest": record(acquisition_path),
        "original_input_manifest": inventory, "base_features": list(data.BASE_FEATURES),
        "telemetry_features": list(features.FEATURE_NAMES), "historical_feature_builds_before_lock": 0,
        "candidate_fits_before_lock": 0, "runtime": {"python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "threads": 1}})
    progress(stage="frozen", design_lock_sha256=data.sha(out/"design_lock.json"))


def verify_design(out):
    lock = read(Path(out)/"design_lock.json")
    if sources() != lock["sources"]: raise ValueError("Current source differs from reviewed design")
    for name, expected in lock["inputs"].items():
        if data.sha(ROOT/name) != expected: raise ValueError("Frozen input changed: "+name)
    for name in ("specification", "independent_review", "pre_fit_tests", "acquisition_manifest"): check_record(lock[name])
    spec = read(check_record(lock["specification"]));validate_spec(spec)
    return lock, spec


def original_reference(raw, event_key, clocks):
    """Independently materialize original issuances, never from telemetry rows."""
    original, _discarded = data._original(raw, event_key)
    del _discarded
    result = original.copy()
    records = result.to_dict("records")
    exact = [data._clock(row, clocks).nanoseconds for row in records]
    result["issued_at_ns"] = pd.Series(exact, index=result.index, dtype="int64")
    result["issuance_id"] = [data._issuance_id(event_key, row["driver_id"], row["issued_after_lap_number"], ns) for row, ns in zip(records, exact)]
    result["year"] = event_key//100
    return result


def prepare_year(out, year, lock, acquired):
    items = [r for r in lock["original_input_manifest"] if r["event_key"]//100 == year]
    streams = {r["event_key"]: r for r in acquired["streams"]}
    references = []
    def events():
        for item in items:
            key = item["event_key"]
            raw, clocks = data.read_laps(ROOT/item["path"], expected_sha256=item["sha256"])
            if len(raw) != item["raw_rows"]: raise ValueError("Original raw row count changed")
            reference = original_reference(raw, key, clocks)
            if len(reference) != item["issuances"]: raise ValueError("Original issuance count changed")
            path = out/"original_references"/f"{key}_issued.jsonl"
            data._write_rows(path, reference.to_dict("records"));references.append({"event_key": key, **record(path), "rows": len(reference)})
            stream = streams[key];unavailable = stream["status"] == "unavailable"
            result = data.build_event(raw, key, clocks, None if unavailable else ROOT/stream["decoded_body_path"],
                source_status="unavailable" if unavailable else "downloaded",
                telemetry_sha256=None if unavailable else stream["decoded_body_sha256"])
            for frame in result.frames_by_lag.values():
                assert_original_parity(frame, reference)
            progress(stage="features_event", event_key=key, rows=len(reference), lags=[0, 2], source_status=result.source["telemetry_status"])
            yield result
    path = data.close_features(events(), out/f"features_{year}", design_lock_path=out/"design_lock.json",
        design_sha256=data.sha(out/"design_lock.json"), expected_event_keys=[r["event_key"] for r in items])
    return {"feature_closure": record(path), "original_references": references}


def validate_year(feature_closure, output):
    """Exhaust downloaded streams after closure with per-event progress."""
    receipt = data.verify_feature_closure(feature_closure);events = [];started = now()
    try:
        for event in receipt["events"]:
            source = event["source"]
            if source["telemetry_status"] == "unavailable":
                entry = {"event_key": event["event_key"], "status": "unavailable", "parsed": False}
            else:
                stats = {}
                for _ in data.packets.iter_packets(source["telemetry_path"], stats=stats): pass
                if stats.get("complete") is not True or stats.get("stream_input_valid") is not True: raise ValueError("Downloaded stream validation incomplete")
                entry = {"event_key": event["event_key"], "status": "valid_downloaded", "parsed": True, "statistics": stats}
            events.append(entry);progress(stage="validated_event", event_key=event["event_key"], source_status=entry["status"])
    except Exception as exc:
        save(output, {"status": "invalid_downloaded_input", "feature_closure_sha256": data.sha(feature_closure),
            "started_at_utc": started, "completed_at_utc": now(), "events": events, "failing_event_key": event["event_key"], "failure": str(exc)})
        raise
    save(output, {"status": "PASS", "feature_closure_sha256": data.sha(feature_closure),
                  "started_at_utc": started, "completed_at_utc": now(), "events": events})
    return record(output)


def prepare(out):
    out = Path(out);lock, spec = verify_design(out)
    save(out/"prepare_attempt.json", {"started_at_utc": now(), "design_lock": record(out/"design_lock.json")})
    acquired = read(check_record(lock["acquisition_manifest"]))
    years = {str(year): prepare_year(out, year, lock, acquired) for year in (2022, 2023)}
    # Every target-free event/lag ledger closes before exhaustive integrity checks.
    for year in (2022, 2023):
        years[str(year)]["input_validation"] = validate_year(check_record(years[str(year)]["feature_closure"]), out/f"input_validation_{year}.json")
    count = sum(read(check_record(v["feature_closure"]))["issuances_per_lag"] for v in years.values())
    if count != spec["discovery"]["expected_original_issuances"]: raise ValueError("Original all-issuance population changed")
    verify_design(out)
    save(out/"data_lock.json", {"closed_at_utc": now(), "design_lock": record(out/"design_lock.json"), "years": years,
        "original_issuances_per_lag": count, "external_labels_attached": False, "model_fits": 0})
    progress(stage="prepared", events=44, original_issuances_per_lag=count, data_lock_sha256=data.sha(out/"data_lock.json"))


def load_year(data_lock, year, lag):
    info = data_lock["years"][str(year)]
    closure = data.verify_feature_closure(check_record(info["feature_closure"]))
    validation = read(check_record(info["input_validation"]))
    if validation["status"] != "PASS" or validation["feature_closure_sha256"] != info["feature_closure"]["sha256"]: raise ValueError("Year input validation mismatch")
    rows = []
    for event in closure["events"]:
        item = event["ledgers"][str(lag)];rows.extend(data._read_rows(check_record(item)))
    frame = pd.DataFrame(rows)
    if any(name in frame for name in TARGET_COLUMNS): raise ValueError("Target entered saved feature ledger")
    if not frame.year.eq(year).all(): raise ValueError("Mixed year feature ledger")
    if frame.issuance_id.duplicated().any(): raise ValueError("Duplicate saved issuance")
    return frame


def reference_year(data_lock, year):
    rows = []
    for item in data_lock["years"][str(year)]["original_references"]:
        rows.extend(data._read_rows(check_record(item)))
    return pd.DataFrame(rows)


def assert_original_parity(frame, expected):
    columns = [*data.KEYS, "forecast_naive_seconds", *data.BASE_FEATURES, "issuance_id", "issued_at_ns"]
    if len(frame) != len(expected): raise ValueError("Original full issuance population changed")
    for name in columns:
        if not pd.Series(frame[name].to_numpy(), dtype=object).equals(pd.Series(expected[name].to_numpy(), dtype=object)):
            raise ValueError("Original all-issuance field/order changed: "+name)


def label_year(out, data_lock, year):
    info = data_lock["years"][str(year)]
    closed = data.attach_labels(check_record(info["feature_closure"]), out/f"labels_{year}", input_validation_path=check_record(info["input_validation"]))
    return record(closed)


def join_labels(frame, label_receipt):
    closed = read(check_record(label_receipt));records = []
    for event in closed["events"]: records.extend(data._read_rows(check_record(event)))
    by_id = {r["issuance_id"]: r for r in records}
    if len(by_id) != len(records) or set(by_id) != set(frame.issuance_id): raise ValueError("Labels must retain every original issuance")
    out = frame.copy(deep=True);ordered = []
    for row in frame.to_dict("records"):
        label = by_id[row["issuance_id"]]
        if any(label[key] != row[key] for key in (*data.KEYS, "year", "issued_at_ns")): raise ValueError("Label/issuance metadata mismatch")
        ordered.append(label)
    for name in TARGET_COLUMNS:
        out[name] = pd.Series([r[name] for r in ordered], index=out.index, dtype=object)
    return out


def assert_matched_parity(actual, expected):
    columns = [*data.KEYS, "forecast_naive_seconds", *data.BASE_FEATURES, *data.TARGET_FIELDS]
    if len(actual) != len(expected): raise ValueError("Original matched population differs")
    for name in columns:
        if not pd.Series(actual[name].to_numpy(), dtype=object).equals(pd.Series(expected[name].to_numpy(), dtype=object)):
            raise ValueError("Original matched value/order changed: "+name)


def verify_discovery_cache(spec, training, selection, *, forecast_closure_path):
    # Explicit closure guard occurs before unpickling the combined historical table.
    closure = read(forecast_closure_path)
    if closure["selection_labels_attached"] is not False: raise ValueError("Forecasts did not close target-free")
    for item in closure["forecasts"].values(): check_record(item)
    path = bound_path(spec, "/discovery_data.pkl")
    expected_hash = next(r["sha256"] for r in spec["original_frontier_bindings"] if (ROOT/r["path"]) == path)
    if data.sha(path) != expected_hash: raise ValueError("Original matched-only cache changed")
    cached = pd.read_pickle(path)
    if len(cached) != spec["discovery"]["matched_rows"]: raise ValueError("Original discovery cache denominator")
    for year, frame in ((2022, training), (2023, selection)):
        expected = cached.loc[cached.year.eq(year)]
        assert_matched_parity(frame, expected)
    return {"cache": record(path), "matched_rows": len(cached), "train_rows": len(training), "selection_rows": len(selection),
            "ordered_values_exact": True, "cache_first_read_after_forecast_closure": True}


@contextmanager
def fit_progress():
    original_fit = models.original.fit_model;calls = []
    def observed(*args, **kwargs):
        index = len(calls);name = models.MODEL_NAMES[index] if index < 3 else "unexpected_extra_fit"
        progress(stage="model_fit_started", model=name, rows=len(args[0]))
        result = original_fit(*args, **kwargs);calls.append(name)
        progress(stage="model_fit_completed", model=name);return result
    models.original.fit_model = observed
    try: yield
    finally: models.original.fit_model = original_fit


def forecast_rows(frame, predictions, lag):
    if set(predictions) != set(models.MODEL_NAMES): raise ValueError("All three forecasts required")
    for value in predictions.values():
        if np.asarray(value).shape != (len(frame),) or not np.isfinite(value).all(): raise ValueError("Invalid full issuance forecasts")
    rows = []
    for i, row in enumerate(frame.to_dict("records")):
        if set(TARGET_COLUMNS) & set(row): raise ValueError("Selection labels were accessed before forecast closure")
        rows.append({**{k: row[k] for k in (*data.KEYS, "issuance_id", "issued_at_ns", "telemetry_supported")},
            "lag_seconds": lag, "feature_row_sha256": data.digest(row),
            "predictions": {name: float(predictions[name][i]) for name in models.MODEL_NAMES}})
    return rows


def load_forecasts(frame, item, lag):
    rows = data._read_rows(check_record(item))
    if len(rows) != len(frame): raise ValueError("Saved forecasts dropped issuances")
    for saved, original in zip(rows, frame.to_dict("records")):
        if saved["issuance_id"] != original["issuance_id"] or saved["feature_row_sha256"] != data.digest(original) or saved["lag_seconds"] != lag:
            raise ValueError("Saved prediction/feature-row binding differs")
    return {name: np.array([r["predictions"][name] for r in rows]) for name in models.MODEL_NAMES}


def select(out):
    out = Path(out);design, spec = verify_design(out)
    locked = read(out/"data_lock.json");check_record(locked["design_lock"])
    if locked["design_lock"]["sha256"] != data.sha(out/"design_lock.json") or locked["external_labels_attached"] is not False or locked["model_fits"] != 0: raise ValueError("Data closure does not precede targets/fits")
    save(out/"selection_attempt.json", {"started_at_utc": now(), "data_lock": record(out/"data_lock.json")})
    train_all = load_year(locked, 2022, 2);assert_original_parity(train_all, reference_year(locked, 2022))
    train_labels = label_year(out, locked, 2022)
    train_labeled = join_labels(train_all, train_labels)
    training = train_labeled.loc[train_labeled.outcome_status.eq("matched")].copy()
    if len(training) != spec["discovery"]["train_matched_rows"]: raise ValueError("Full matched training population changed")
    # Event ascending, then original issued order; matching only filters rows,
    # exactly as frozen frontier.build/enrich produced discovery_data.pkl.
    with threadpool_limits(limits=1), fit_progress(): bundle = models.fit_models(training)
    model_path = out/"models.pkl"
    with model_path.open("xb") as stream: pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)
    model_binding = record(model_path)
    save(out/"fit_lock.json", {"closed_at_utc": now(), "models": model_binding, "training_labels": train_labels,
        "data_lock": record(out/"data_lock.json"), "fit_summary": bundle["fit_summary"], "external_selection_labels_attached": False})
    with check_record(model_binding).open("rb") as stream: reloaded = pickle.load(stream)
    selection_frames, forecasts = {}, {}
    expected_issuances = reference_year(locked, 2023)
    for lag in (2, 0):
        frame = load_year(locked, 2023, lag);assert_original_parity(frame, expected_issuances)
        before = data.digest(frame.to_dict("records"))
        predictions = models.predict_models(reloaded, frame, latency_seconds=lag)
        if data.digest(frame.to_dict("records")) != before: raise ValueError("Prediction mutated target-free features")
        path = out/f"selection_lag{lag}_forecasts.jsonl"
        data._write_rows(path, forecast_rows(frame, predictions, lag));forecasts[str(lag)] = {**record(path), "rows": len(frame)}
        selection_frames[lag] = frame
        progress(stage="selection_forecasts_saved", lag_seconds=lag, rows=len(frame))
    verify_design(out)
    issuance_lock_path = out/"selection_issuance_lock.json"
    save(issuance_lock_path, {"closed_at_utc": now(), "data_lock": record(out/"data_lock.json"), "fit_lock": record(out/"fit_lock.json"),
        "forecasts": forecasts, "original_reference_files": locked["years"]["2023"]["original_references"],
        "selection_labels_attached": False, "selection_matched_cache_read": False})
    selection_labels = label_year(out, locked, 2023)
    labeled = {lag: join_labels(frame, selection_labels) for lag, frame in selection_frames.items()}
    scored = labeled[2].loc[labeled[2].outcome_status.eq("matched")]
    if len(scored) != spec["discovery"]["selection_matched_rows"]: raise ValueError("Original selection matched population changed")
    parity = verify_discovery_cache(spec, training, scored, forecast_closure_path=issuance_lock_path)
    # The matched-only cache is a separately pinned reference and is now safe to
    # inspect. It never supplied issuance eligibility or training row ordering.
    cached = pd.read_pickle(bound_path(spec, "/discovery_data.pkl"));expected_matched = cached.loc[cached.year.eq(2023)]
    predictions = {lag: load_forecasts(frame, forecasts[str(lag)], lag) for lag, frame in selection_frames.items()}
    summary = evaluate.evaluate_selection(labeled[2], predictions[2], labeled[0], predictions[0],
        expected_matched=expected_matched, expected_issuances=expected_issuances)
    save(out/"original_population_parity.json", parity)
    verify_design(out)
    save(out/"selection.json", {"completed_at_utc": now(), "status": "retrospective_telemetry_research_no_promotion",
        "summary": summary, "design_lock": record(out/"design_lock.json"), "data_lock": record(out/"data_lock.json"),
        "fit_lock": record(out/"fit_lock.json"), "issuance_lock": record(issuance_lock_path),
        "selection_labels": selection_labels, "original_population_parity": record(out/"original_population_parity.json")})
    save(out/"selection_lock.json", {"closed_at_utc": now(), "selection": record(out/"selection.json"),
        "design_lock": record(out/"design_lock.json"), "selected_candidate": "telemetry_hgb",
        "advances_to_later_evaluation": summary["advances_to_later_evaluation"], "promotion": False})
    progress(stage="selection_complete", summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "prepare", "select"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--acquisition-manifest", type=Path, default=ACQUISITION)
    args = parser.parse_args()
    if args.stage == "freeze" and args.review is None: parser.error("--review must approve exact source bytes")
    try:
        if args.stage == "freeze": freeze(args.out, args.review, args.acquisition_manifest)
        elif args.stage == "prepare": prepare(args.out)
        else: select(args.out)
    except Exception as exc:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        save(args.out/f"{args.stage}_failure_{stamp}.json", {"failed_at_utc": now(), "stage": args.stage,
            "failure": type(exc).__name__+": "+str(exc), "traceback": traceback.format_exc(),
            "existing_outputs_preserved": True})
        raise


if __name__ == "__main__": main()
