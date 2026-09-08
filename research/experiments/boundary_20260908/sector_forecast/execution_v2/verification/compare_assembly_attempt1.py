"""Read-only comparison of frozen v1/v2 target-free ledgers, never outcomes.

Run only after the parent confirms the v2 build is complete. This standalone
verifier imports neither assembly, feature, model, baseline nor target code.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[6]
ARTIFACTS = ROOT / "artifacts/research/boundary_20260908/sector_forecast"
ORIGINAL_EVENTS = tuple(range(202201, 202213))
FORBIDDEN_LABELS = {"y", "y_true", "target", "target_id", "target_group_id",
                    "target_time_seconds", "outcome_status"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def differences(left, right, path=""):
    """Exact semantic equality, except that explicit NaN equals explicit NaN."""
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(left.keys() | right.keys()):
            child = f"{path}.{key}" if path else key
            if key not in left or key not in right:
                yield child
            else:
                yield from differences(left[key], right[key], child)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            yield path + ".length"
        for index, (a, b) in enumerate(zip(left, right)):
            yield from differences(a, b, f"{path}[{index}]")
    elif isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right):
        return
    elif type(left) is not type(right) or left != right:
        yield path


def load_event(directory, event, data_lock_entry=None):
    path = directory / "target_free" / f"{event}.jsonl"
    metadata_path = path.with_suffix(".json")
    metadata = read(metadata_path)
    digest = sha(path)
    require(metadata["event_key"] == event, "Mismatched event metadata")
    require(metadata["path"] == relative(path), "Mismatched ledger path")
    require(metadata["sha256"] == digest, "Ledger hash differs from metadata")
    if data_lock_entry is not None:
        require(metadata == data_lock_entry, "Metadata differs from the completed data lock")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row["issuance_id"] for row in rows]
    require(len(ids) == len(set(ids)), "Duplicate issuance IDs")
    require(len(rows) == metadata["diagnostics"]["records"], "Incomplete ledger")
    require(metadata["diagnostics"]["target_free"] is True, "Expected target-free metadata")
    require(metadata["diagnostics"]["canonical_csv_read"] is False, "Target pass has entered assembly metadata")
    for row in rows:
        require(row["event_key"] == event, "Mixed event rows")
        require(not FORBIDDEN_LABELS.intersection(row), "Refusing a labeled input")
        if row["status"] == "issued":
            require(len(row["features"]) == 75, "Unexpected feature population")
            require(len(row["points"]) == 5, "Unexpected reference population")
            require(all(type(x) in (int, float) and not math.isinf(x) for x in row["features"].values()),
                    "Invalid feature value")
            require(all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in row["points"].values()),
                    "Invalid reference point")
        else:
            require(row["features"] == {} and row["points"] == {}, "Unsupported row has a forecast")
    return rows, {"path": relative(path), "sha256": digest,
                  "metadata_path": relative(metadata_path), "metadata_sha256": sha(metadata_path)}


def strip_only_new_provenance(row):
    result = deepcopy(row)
    result["source_provenance"].pop("s1_prefix", None)
    return result


def compare_event(original, amended, event):
    ids1 = [row["issuance_id"] for row in original]
    ids2 = [row["issuance_id"] for row in amended]
    old = {row["issuance_id"]: row for row in original}
    new = {row["issuance_id"]: row for row in amended}
    mismatches = []
    categories = Counter()
    provenance_rows = 0
    for identity in old.keys() & new.keys():
        a, b = old[identity], new[identity]
        require("s1_prefix" not in a["source_provenance"], "Original unexpectedly has v2 provenance")
        if b["status"] == "issued" and b["sector"] == 2:
            require("s1_prefix" in b["source_provenance"], "Missing amended S1 source provenance")
            provenance_rows += 1
        for path in differences(a, strip_only_new_provenance(b)):
            categories[path.split(".", 1)[0]] += 1
            if len(mismatches) < 20:
                mismatches.append({"issuance_id": identity, "field": path})
    return {"event_key": event, "original_rows": len(original), "amended_rows": len(amended),
            "original_status_counts": dict(Counter(row["status"] for row in original)),
            "amended_status_counts": dict(Counter(row["status"] for row in amended)),
            "issuance_order_identical": ids1 == ids2,
            "missing_issuance_ids": sorted(old.keys() - new.keys()),
            "extra_issuance_ids": sorted(new.keys() - old.keys()),
            "mismatch_count": sum(categories.values()), "mismatch_categories": dict(categories),
            "mismatch_examples": mismatches, "new_s1_provenance_rows": provenance_rows,
            "all_original_record_values_unchanged": ids1 == ids2 and not categories}


def verify_hungary(rows):
    selected = [r for r in rows if r["driver"] == "47" and r["packet_sequence"] == 24357 and r["sector"] == 2]
    require(len(selected) == 1, "Expected exactly one Hungary correction S2 row")
    row = selected[0]
    require(row["status"] == "issued", "Hungary correction failed to issue")
    require(row["checkpoint_ms"] == 5977490 and row["history_count"] == 20, "Unexpected correction checkpoint/history")
    source = row["source_provenance"]["s1_prefix"]
    prior, latest = source["prior_support"], source["value_source"]
    require(prior["packet_sequence"] == 24356 and prior["available_ms"] == 5977489 and prior["value"] == 30.526,
            "Incorrect prior S1 support")
    require(latest["sequence"] == 24357 and latest["available_ms"] == 5977490 and latest["value"] == 30.486,
            "Corrected S1 value was lost or backdated")
    require(source["same_packet_value_update"] is True and source["value_changed_from_prior_support"] is True,
            "Correction flags are inconsistent")
    scale = row["points"]["sector_pace_scaled_median5"]
    encoded = [row["features"][name] * scale for name in ("s1_over_b4", "s2_over_b4")]
    require(all(math.isclose(x, y, rel_tol=0, abs_tol=1e-10) for x, y in zip(encoded, (30.486, 31.233))),
            "Encoded prefix differs from the current atomic S1/S2 values")
    earlier = [r for r in rows if r["driver"] == "47" and r["packet_sequence"] == 24356 and r["sector"] == 1]
    require(len(earlier) == 1 and earlier[0]["status"] == "issued", "Earlier S1 forecast is missing")
    earlier_scale = earlier[0]["points"]["sector_pace_scaled_median5"]
    require(math.isclose(earlier[0]["features"]["s1_over_b4"] * earlier_scale, 30.526, rel_tol=0, abs_tol=1e-10),
            "Earlier S1 feature was revised retroactively")
    return {"event_key": 202213, "issuance_id": row["issuance_id"], "checkpoint_ms": row["checkpoint_ms"],
            "history_count": row["history_count"], "prior_support": prior, "current_value_source": latest,
            "encoded_prefix_seconds": encoded, "earlier_s1_value_preserved": True,
            "corrected_prefix_and_actual_source_clock_verified": True}


def compare(v1, v2):
    expected = {f"{event}.jsonl" for event in ORIGINAL_EVENTS}
    require({p.name for p in (v1 / "target_free").glob("*.jsonl")} == expected,
            "The original saved population is not exactly the expected twelve events")
    lock_path = v2 / "data_lock.json"
    lock = read(lock_path)
    require(lock["target_labels_read"] is False and lock["sector_models_fitted"] == 0,
            "Expected a completed target-free data lock")
    require(lock["sector_predictive_scores_computed"] is False, "Unexpected scored data lock")
    require(lock["design_lock_sha256"] == sha(v2 / "design_lock.json"), "Design/data lock lineage mismatch")
    events = {entry["event_key"]: entry for entry in lock["events"]}
    require(len(events) == len(lock["events"]) == 44, "Incomplete or duplicate v2 event lock")
    output = {"schema_version": 1, "scope": "Read-only target-free assembly comparison; no CSV, labels, fits or scores.",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "verifier": {"path": relative(__file__), "sha256": sha(__file__)},
              "v2_data_lock": {"path": relative(lock_path), "sha256": sha(lock_path)},
              "ignored_difference": "Only the new source_provenance.s1_prefix field is excluded from original-row equality.",
              "comparison_rule": "Exact recursive values and types, with explicit NaN equal to explicit NaN; issuance order must also agree.",
              "events": []}
    bindings = []
    for event in ORIGINAL_EVENTS:
        old, a = load_event(v1, event)
        new, b = load_event(v2, event, events[event])
        result = compare_event(old, new, event)
        result.update({"original_input": a, "amended_input": b})
        output["events"].append(result)
        bindings.extend((a, b))
    hungary, bound = load_event(v2, 202213, events[202213])
    output["hungary_correction"] = {**verify_hungary(hungary), "input": bound}
    bindings.append(bound)
    for item in bindings:
        require(sha(ROOT / item["path"]) == item["sha256"], "Input changed during comparison")
        require(sha(ROOT / item["metadata_path"]) == item["metadata_sha256"], "Metadata changed during comparison")
    require(sha(lock_path) == output["v2_data_lock"]["sha256"], "Data lock changed during comparison")
    output["passed"] = all(r["all_original_record_values_unchanged"] for r in output["events"])
    output["original_rows_compared"] = sum(r["original_rows"] for r in output["events"])
    output["all_read_bindings_unchanged_after"] = True
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-confirmed-build-complete", action="store_true", required=True)
    parser.add_argument("--v1", type=Path, default=ARTIFACTS / "execution")
    parser.add_argument("--v2", type=Path, default=ARTIFACTS / "execution_v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to replace a previous comparison receipt")
    report = compare(args.v1, args.v2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"path": str(args.output), "sha256": sha(args.output),
                      "passed": report["passed"], "original_rows_compared": report["original_rows_compared"]}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
