"""Freeze, assemble target-free discovery streams, then fit and select once."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd
import sklearn

from research.experiments.boundary_20260908.sector_forecast import models
from . import data, evaluate, features

ROOT = Path(__file__).resolve().parents[5]
LANE = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/research/boundary_20260908/sector_forecast/execution_v2"


def now(): return datetime.now(timezone.utc).isoformat()
def relative(path): return str(Path(path).resolve().relative_to(ROOT))
def sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()
def read(path): return json.loads(Path(path).read_text())


def safe_json(value):
    if isinstance(value, dict): return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        if math.isinf(value): raise ValueError("Infinity cannot enter a research artifact")
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    return value


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(safe_json(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def save_rows(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(safe_json(row), allow_nan=False, separators=(",", ":")) + "\n")


def load_rows(path):
    with Path(path).open() as stream: return [json.loads(line) for line in stream]


def sources():
    # Future independent verification/transfer modules belong in subdirectories;
    # this exact execution and every consumed dependency stay immutable.
    files = list(LANE.glob("*.py")) + list(LANE.glob("*.json"))
    files += [LANE.parent / name for name in ["models.py", "baselines.py", "completed.py", "context.py", "completed_contract.json"]]
    pilot = LANE.parent.parent / "sector_pilot"
    files += [pilot / name for name in ["ledger.py", "run.py", "specification.json"]]
    return {relative(p): sha(p) for p in sorted(set(files))}


def manifest():
    spec = read(LANE / "specification.json")
    path = ROOT / spec["discovery_acquisition"]["path"]
    if sha(path) != spec["discovery_acquisition"]["sha256"]: raise ValueError("Acquisition manifest changed")
    return read(path)


def inputs():
    acquired = manifest(); result = {}
    spec = read(LANE / "specification.json")
    result[spec["discovery_acquisition"]["path"]] = spec["discovery_acquisition"]["sha256"]
    for row in acquired["streams"]: result[row["decoded_body_path"]] = row["decoded_body_sha256"]
    for row in acquired["existing_track_status"]: result[row["stream"]["path"]] = row["stream"]["sha256"]
    for row in acquired["sessions"]: result[row["original_laps_path"]] = row["original_laps_sha256"]
    return result


def verify_lock(out):
    lock = read(out / "design_lock.json")
    if lock["sources"] != sources(): raise ValueError("Execution source changed after pre-fit lock")
    if sha(out / "pre_fit_tests.json") != lock["pre_fit_tests_sha256"]: raise ValueError("Pre-fit test receipt changed")
    for path, expected in lock["inputs"].items():
        if sha(ROOT / path) != expected: raise ValueError(f"Locked input changed: {path}")
    if sha(LANE / "mathematical_review.md") != lock["mathematical_review_sha256"]:
        raise ValueError("Pre-fit mathematical review changed")
    return lock


def freeze(out):
    if (out / "design_lock.json").exists(): raise ValueError("Existing immutable design lock")
    before = sources()
    spec = read(LANE / "specification.json")
    if sorted(spec["discovery_events"]) != sorted(r["event_key"] for r in manifest()["sessions"]):
        raise ValueError("Declared discovery scope differs from acquired sessions")
    command = [sys.executable, "-m", "pytest", "-q", "--import-mode=importlib", "-p", "no:cacheprovider",
               relative(LANE), relative(LANE.parent / "test_baselines.py"),
               relative(LANE.parent / "test_models.py"), relative(LANE.parent / "test_completed.py"),
               relative(LANE.parent / "test_context.py"), relative(LANE.parent.parent / "sector_pilot/test_ledger.py")]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if sources() != before: raise ValueError("Sources changed during pre-fit tests")
    save(out / "pre_fit_tests.json", {"closed_at_utc": now(), "command": command,
         "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "sources": before})
    print(result.stdout, flush=True)
    if result.returncode: raise ValueError("Pre-fit synthetic contract tests failed")
    bound = inputs()
    for path, expected in bound.items():
        if sha(ROOT / path) != expected: raise ValueError(f"Input hash mismatch: {path}")
    save(out / "design_lock.json", {"locked_at_utc": now(), "sources": before, "inputs": bound,
         "pre_fit_tests_sha256": sha(out / "pre_fit_tests.json"),
         "mathematical_review_sha256": sha(LANE / "mathematical_review.md"),
         "sector_models_fitted_before_lock": False, "sector_predictive_scores_before_lock": False,
         "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": sklearn.__version__, "numerical_threads": 1, "event_assembly_workers": 2}})
    print(json.dumps({"stage": "frozen", "sha256": sha(out / "design_lock.json")}), flush=True)


def _assemble(event):
    return event, data.build_event(data.load_event_paths(event), event)


def build(out):
    verify_lock(out)
    if (out / "data_lock.json").exists(): raise ValueError("Existing immutable target-free data lock")
    spec = read(LANE / "specification.json"); rows = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        pending = {executor.submit(_assemble, event): event for event in spec["discovery_events"]}
        for future in as_completed(pending):
            event, (records, diagnostics) = future.result()
            path = out / "target_free" / f"{event}.jsonl"
            save_rows(path, records)
            record = {"event_key": event, "path": relative(path), "sha256": sha(path),
                      "diagnostics": diagnostics, "closed_at_utc": now()}
            save(out / "target_free" / f"{event}.json", record); rows.append(record)
            print(json.dumps({"stage": "target_free", "event": event, "records": len(records),
                              "statuses": diagnostics["status_counts"]}), flush=True)
    if sources() != read(out / "design_lock.json")["sources"]: raise ValueError("Sources changed during assembly")
    save(out / "data_lock.json", {"closed_at_utc": now(), "design_lock_sha256": sha(out / "design_lock.json"),
         "events": sorted(rows, key=lambda r: r["event_key"]), "target_labels_read": False,
         "sector_models_fitted": 0, "sector_predictive_scores_computed": False})


def locked_records(out, years):
    lock = read(out / "data_lock.json")
    if lock["design_lock_sha256"] != sha(out / "design_lock.json"): raise ValueError("Data lock changed lineage")
    result = []
    for row in lock["events"]:
        if row["event_key"] // 100 not in years: continue
        path = ROOT / row["path"]
        if sha(path) != row["sha256"]: raise ValueError("Target-free ledger changed")
        result.append((row, load_rows(path)))
    return result


def labels(out, years):
    sessions = {r["event_key"]: r for r in manifest()["sessions"]}; all_rows = []
    for metadata, records in locked_records(out, years):
        session = sessions[metadata["event_key"]]
        terminal = metadata["diagnostics"]["terminal"]
        all_rows.extend(data.attach_targets(records, ROOT / session["original_laps_path"], terminal))
    return all_rows


def frame(records):
    result = []
    for row in records:
        if row["status"] != data.ISSUED_STATUS: continue
        item = {"event_key": str(row["event_key"]), "driver": row["driver"],
                "ledger_id": row["issuance_id"], "sector": row["sector"], "checkpoint_ms": row["checkpoint_ms"],
                **row["features"], **row["points"]}
        if "outcome_status" in row:
            item.update(outcome_status=row["outcome_status"], target_id=row["target_id"],
                        target_time_seconds=row["target_time_seconds"], y_true=row["y"])
        result.append(item)
    table = pd.DataFrame(result)
    for name in features.FEATURES: table[name] = pd.to_numeric(table[name], errors="raise")
    if "y_true" in table: table["y_true"] = pd.to_numeric(table["y_true"], errors="raise")
    return table


def select(out):
    verify_lock(out)
    if (out / "selection_attempt.json").exists(): raise ValueError("Selection already attempted; preserve its outcome")
    spec = read(LANE / "specification.json"); refs = spec["references"]
    candidates = [c["name"] for c in spec["candidates"]]
    save(out / "selection_attempt.json", {"started_at_utc": now(), "data_lock_sha256": sha(out / "data_lock.json"),
         "design_lock_sha256": sha(out / "design_lock.json"), "candidates": candidates})
    training_records = labels(out, spec["split"]["train_years"])
    save_rows(out / "training_labeled.jsonl", training_records)
    training = frame(training_records)
    evaluate.validate(training, refs + [candidates[0]])
    resolved = training.loc[training["outcome_status"] == evaluate.MATCHED]
    target_free = [r for _, records in locked_records(out, spec["split"]["selection_years"]) for r in records]
    blind = frame(target_free)
    if any(key in blind for key in ["y_true", "target_id", "outcome_status"]): raise ValueError("Selection prediction inputs contain outcomes")
    fitted = {}
    for config in spec["candidates"][1:]:
        print(json.dumps({"stage": "fit", "candidate": config["name"], "rows": len(resolved)}), flush=True)
        model = models.fit_residual(resolved, list(features.FEATURES), spec["hgb"]["anchor"], config)
        blind[config["name"]] = model.predict(blind)
        path = out / "models" / f"{config['name']}.pkl"; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream: pickle.dump(model, stream, protocol=5)
        fitted[config["name"]] = {"path": relative(path), "sha256": sha(path)}
    issued_path = out / "selection_issued.jsonl"
    issued = blind[["event_key", "driver", "ledger_id", "sector", "checkpoint_ms", *refs, *candidates]]
    save_rows(issued_path, issued.to_dict(orient="records"))
    save(out / "selection_issuance_lock.json", {"closed_at_utc": now(), "path": relative(issued_path),
         "sha256": sha(issued_path), "issued_rows": len(issued), "models": fitted,
         "training_labeled_sha256": sha(out / "training_labeled.jsonl"), "selection_targets_read": False})
    selection_records = labels(out, spec["split"]["selection_years"])
    save_rows(out / "selection_labeled.jsonl", selection_records)
    scored = frame(selection_records)
    if scored["ledger_id"].tolist() != blind["ledger_id"].tolist(): raise ValueError("Label attachment changed the issued cohort")
    unchanged = [*features.FEATURES, *refs, candidates[0]]
    if not np.array_equal(scored[unchanged].to_numpy(float), blind[unchanged].to_numpy(float), equal_nan=True):
        raise ValueError("Label attachment changed a feature or reference prediction")
    for candidate in candidates[1:]: scored[candidate] = blind[candidate].to_numpy()
    result = evaluate.selection_report(scored, candidates, refs, spec)
    result.update({"completed_at_utc": now(), "design_lock_sha256": sha(out / "design_lock.json"),
                   "data_lock_sha256": sha(out / "data_lock.json"),
                   "selection_issuance_lock_sha256": sha(out / "selection_issuance_lock.json"),
                   "selection_labeled_sha256": sha(out / "selection_labeled.jsonl"),
                   "feature_names": list(features.FEATURES), "transfer_acquired_or_evaluated": False})
    save(out / "selection.json", result)
    save(out / "selection_lock.json", {"locked_at_utc": now(), "selected": result["selected"],
         "selection_passed": result["selection_passed"], "selection_sha256": sha(out / "selection.json"),
         "next_action": "Frozen transfer acquisition may begin" if result["selection_passed"] else "Stop this candidate family; no transfer acquisition or evaluation"})
    print(json.dumps({"stage": "selection_complete", "selected": result["selected"],
          "selection_passed": result["selection_passed"], "metrics": {
              name: {metric: values[metric] for metric in evaluate.METRICS}
              for name, values in result["summary"]["models"].items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["freeze", "build", "select"])
    parser.add_argument("--out", type=Path, default=OUT); args = parser.parse_args()
    try:
        {"freeze": freeze, "build": build, "select": select}[args.stage](args.out)
    except Exception:
        save(args.out / f"{args.stage}_failure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.json",
             {"failed_at_utc": now(), "stage": args.stage, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__": main()
