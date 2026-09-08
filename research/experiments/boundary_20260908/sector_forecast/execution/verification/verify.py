"""Independent post-selection verification; never fits or imports the evaluator.

The author also wrote execution/features.py. This independently checks saved
cohorts, labels, fitted-model replay and scoring, not independent feature design.
Run --self-test without reading event artifacts. Historical verification requires
an explicit --selection-complete flag and an existing completed selection lock.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[6]
LANE = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts/research/boundary_20260908/sector_forecast/execution"
METRICS = ("checkpoint_event_mae", "target_balanced_event_mae")
MATCHED = "matched_recorded_future_eligible_lap"
UNMATCHED = {"unmatched_in_terminal_recorded_archive", "unresolved_in_incomplete_archive"}
LABEL_FIELDS = {"outcome_status", "target", "target_id", "target_group_id", "y",
                "target_time_seconds", "target_lap_number", "target_csv_row"}
ISSUANCE_STATUSES = {"issued", "pilot_excluded", "retired_or_stopped", "insufficient_history"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def strict_json(text):
    def reject(value):
        raise ValueError(f"Nonstandard JSON constant: {value}")
    def real(value):
        result = float(value)
        require(math.isfinite(result), "JSON number is not representable as finite data")
        return result
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, f"Duplicate JSON key: {key}")
            value[key] = item
        return value
    return json.loads(text, parse_constant=reject, parse_float=real, object_pairs_hook=unique)


def read(path):
    return strict_json(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            require(bool(line.strip()), "Blank JSONL record")
            yield strict_json(line)


def local(path):
    path = (ROOT / path).resolve()
    require(path.is_relative_to(ROOT), "Artifact/source path escapes repository")
    return path


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def finite(value, name, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), f"Invalid numeric {name}")
    require(not positive or value > 0, f"Nonpositive {name}")
    return float(value)


def integer(value, name):
    require(type(value) is int and value >= 0, f"Invalid integer {name}")
    return value


def stamp(value):
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, "Execution timestamp lacks timezone")
    return parsed.astimezone(timezone.utc)


def close(actual, expected, path="value"):
    """Exact structure and discrete values; absolute 1e-12 numerical tolerance."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"Keys differ: {path}")
        for key in expected:
            close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"Length differs: {path}")
        for i, (left, right) in enumerate(zip(actual, expected)):
            close(left, right, f"{path}[{i}]")
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual)
                and abs(actual - expected) <= 1e-12, f"Numeric mismatch: {path}: {actual!r} != {expected!r}")
    else:
        require(type(actual) is type(expected) and actual == expected, f"Mismatch: {path}")


def event_identity(row, issued=False):
    raw = row["event_key"]
    require((type(raw) is str if issued else type(raw) is int), "Unexpected event serialization")
    event = str(raw)
    require(re.fullmatch(r"20\d{4}", event) is not None, "Invalid event identity")
    require(isinstance(row["driver"], str) and row["driver"], "Invalid driver identity")
    require(type(row["sector"]) is int and row["sector"] in (1, 2), "Invalid sector")
    require(finite(row["checkpoint_ms"], "checkpoint clock") >= 0, "Negative checkpoint clock")
    return event


def mean(values):
    require(len(values) > 0, "Empty mean")
    return math.fsum(values) / len(values)


def summary(records, names):
    """Independent nested aggregation using lists and compensated sums."""
    require(records, "Empty issued cohort")
    ids = set()
    event_rows = defaultdict(list)
    target_signature = {}
    time_identity = {}
    for row in records:
        identity = row["ledger_id"]
        require(identity and identity not in ids, "Duplicate issuance identity")
        ids.add(identity)
        event = event_identity(row, issued=True)
        event_rows[event].append(row)
        status = row["outcome_status"]
        require(status in {MATCHED, *UNMATCHED}, "Unknown issued outcome")
        for name in names:
            finite(row[name], f"prediction {name}", positive=True)
        if status == MATCHED:
            truth = finite(row["y_true"], "target seconds", positive=True)
            target_time = finite(row["target_time_seconds"], "target clock")
            require(target_time > row["checkpoint_ms"] / 1000, "Target is not strictly later")
            target = row["target_id"]
            require(isinstance(target, str) and target, "Missing target identity")
            key = (event, target)
            signature = (row["driver"], truth, target_time)
            require(key not in target_signature or target_signature[key] == signature, "Inconsistent repeated target")
            target_signature[key] = signature
            time_key = (event, row["driver"], target_time)
            require(time_key not in time_identity or time_identity[time_key] == target, "One target split across sectors/IDs")
            time_identity[time_key] = target
        else:
            require(row["y_true"] is None and row["target_id"] is None
                    and row["target_time_seconds"] is None, "Unmatched row invented a target")
    resolved = {event: [r for r in group if r["outcome_status"] == MATCHED]
                for event, group in event_rows.items()}
    resolved = {event: group for event, group in resolved.items() if group}
    require(resolved, "No resolved event")
    matched_count = sum(map(len, resolved.values()))
    result = {"issued": len(records), "resolved_issued": matched_count,
              "unmatched_issued": len(records) - matched_count,
              "issued_events": sorted(event_rows), "resolved_events": sorted(resolved),
              "resolved_event_targets": len(target_signature), "per_event_outcomes": {}, "models": {}}
    for event in sorted(event_rows):
        group = event_rows[event]
        matched = resolved.get(event, [])
        result["per_event_outcomes"][event] = {
            "issued": len(group), "resolved": len(matched), "unmatched": len(group) - len(matched),
            "resolved_targets": len({r["target_id"] for r in matched})}
    for name in names:
        checkpoint, balanced = {}, {}
        for event in sorted(resolved):
            errors = []
            target_errors = defaultdict(list)
            for row in resolved[event]:
                error = abs(row[name] - row["y_true"])
                errors.append(error)
                target_errors[row["target_id"]].append(error)
            checkpoint[event] = mean(errors)
            balanced[event] = mean([mean(v) for v in target_errors.values()])
        result["models"][name] = {METRICS[0]: mean(list(checkpoint.values())),
                                  METRICS[1]: mean(list(balanced.values())),
                                  "per_event": {METRICS[0]: checkpoint, METRICS[1]: balanced}}
    return result


def percentiles(values):
    ordered = np.sort(values)
    answers = []
    for fraction in (.025, .975):
        position = fraction * (len(ordered) - 1)
        lower, upper = math.floor(position), math.ceil(position)
        weight = position - lower
        answers.append(float((1 - weight) * ordered[lower] + weight * ordered[upper]))
    return answers


def intervals(differences, keys, resamples, seed):
    """Draw the declared indices, accumulate each draw independently of evaluate."""
    require(len(keys) == len(differences) >= 2 and keys == sorted(keys), "Invalid paired event series")
    require(all(math.isfinite(v) for v in differences), "Invalid event difference")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(keys), size=(resamples, len(keys)))
    values = np.asarray(differences, dtype=float)
    ordinary = np.zeros(resamples)
    for position in range(len(keys)):
        ordinary += values[indexes[:, position]] / len(keys)
    block_means = np.zeros(resamples)
    for year in sorted({key[:4] for key in keys}):
        values_in_year = np.asarray([value for key, value in zip(keys, differences) if key[:4] == year])
        count = len(values_in_year)
        starts = rng.integers(0, count, size=(resamples, math.ceil(count / 3)))
        for position in range(count):
            indexes = (starts[:, position // 3] + position % 3) % count
            block_means += values_in_year[indexes] / len(keys)
    omitted = [mean([v for j, v in enumerate(differences) if j != i]) for i in range(len(keys))]
    return {"difference_candidate_minus_reference": mean(differences),
            "event_percentile_95_interval": percentiles(ordinary),
            "three_event_percentile_95_interval": percentiles(block_means),
            "leave_one_event_out_max_difference": max(omitted),
            "events_won": sum(v < 0 for v in differences), "events_lost": sum(v > 0 for v in differences),
            "events_tied": sum(v == 0 for v in differences), "resamples": resamples, "seed": seed}


def selection(records, candidates, references, spec):
    statistics = summary(records, references + candidates)
    chosen = min(enumerate(candidates), key=lambda pair: (statistics["models"][pair[1]][METRICS[1]], pair[0]))[1]
    comparisons = {}
    options = spec["evaluation"]["uncertainty"]
    for candidate in candidates:
        comparisons[candidate] = {}
        for reference in references:
            comparisons[candidate][reference] = {}
            for metric in METRICS:
                c, r = statistics["models"][candidate], statistics["models"][reference]
                keys = sorted(c["per_event"][metric])
                require(keys == sorted(r["per_event"][metric]), "Model event populations differ")
                delta = [c["per_event"][metric][event] - r["per_event"][metric][event] for event in keys]
                comparisons[candidate][reference][metric] = {
                    "candidate": c[metric], "reference": r[metric],
                    "relative_gain_fraction": 1 - c[metric] / r[metric] if r[metric] > 0 else None,
                    **intervals(delta, keys, options["resamples"], options["seed"])}
    checks = {"complete_selection_event_coverage": statistics["issued_events"] == statistics["resolved_events"]
              and len(statistics["resolved_events"]) == spec["split"]["required_selection_resolved_events"]}
    for reference in references:
        point, balanced = (comparisons[chosen][reference][metric] for metric in METRICS)
        gain = balanced["relative_gain_fraction"]
        checks[reference] = (gain is not None and gain >= spec["selection"]["gain_fraction_vs_every_reference"]
                             and balanced["three_event_percentile_95_interval"][1] < 0
                             and point["candidate"] <= point["reference"])
    return {"summary": statistics, "comparisons": comparisons, "selected": chosen,
            "selection_checks": checks, "selection_passed": all(checks.values()),
            "subgroups": {str(stage): summary([r for r in records if r["sector"] == stage], references + candidates)
                          for stage in (1, 2)}, "new_substantial_gain_established": False}


def csv_number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def canonical_targets(path):
    """Independent CSV scan; no pilot target helpers or final feature reads."""
    grouped = defaultdict(list)
    total = 0
    with Path(path).open() as stream:
        for ordinal, row in enumerate(csv.DictReader(stream)):
            total += 1
            lap = csv_number(row.get("LapTime"))
            eligible = (math.isfinite(lap) and lap > 0
                        and str(row.get("IsAccurate", "")).lower() in ("true", "1")
                        and not math.isfinite(csv_number(row.get("PitInTime")))
                        and not math.isfinite(csv_number(row.get("PitOutTime")))
                        and not set("4567").intersection(str(row.get("TrackStatus", ""))))
            if not eligible:
                continue
            clock, number = csv_number(row.get("Time")), csv_number(row.get("DriverNumber"))
            if not math.isfinite(clock) or not math.isfinite(number):
                continue
            driver = str(int(number))
            grouped[driver].append({"csv_row": ordinal, "recorded_time_seconds": clock,
                                    "lap_number": csv_number(row.get("LapNumber")), "lap_time_seconds": lap})
    result = {}
    for driver, targets in grouped.items():
        targets.sort(key=lambda target: (target["recorded_time_seconds"], target["csv_row"]))
        result[driver] = ([target["recorded_time_seconds"] for target in targets], targets)
    return result, total


def expected_label(row, index, terminal):
    clocks, targets = index.get(row["driver"], ([], []))
    position = bisect_right(clocks, row["checkpoint_ms"] / 1000)
    target = targets[position] if position < len(targets) else None
    if target is None:
        status = "unmatched_in_terminal_recorded_archive" if terminal else "unresolved_in_incomplete_archive"
        return {"outcome_status": status, "target": None, "target_id": None, "target_group_id": None,
                "y": None, "target_time_seconds": None, "target_lap_number": None, "target_csv_row": None}
    identity = f'{row["event_key"]}:{row["driver"]}:csvrow:{target["csv_row"]}'
    return {"outcome_status": MATCHED, "target": target, "target_id": identity, "target_group_id": identity,
            "y": target["lap_time_seconds"], "target_time_seconds": target["recorded_time_seconds"],
            "target_lap_number": target["lap_number"], "target_csv_row": target["csv_row"]}


def source_inventory(lane):
    paths = [*lane.glob("*.py"), *lane.glob("*.json")]
    paths += [lane.parent / name for name in ("models.py", "baselines.py", "completed.py", "context.py", "completed_contract.json")]
    paths += [lane.parent.parent / "sector_pilot" / name for name in ("ledger.py", "run.py", "specification.json")]
    return {relative(path) for path in paths}


def verify(out, execution_dir=LANE, design_sha256=None):
    out = Path(out).resolve()
    lane = Path(execution_dir).resolve()
    require(out.is_relative_to(ROOT), "Verification output must be inside repository")
    require(lane.is_relative_to(ROOT) and lane.parent == LANE.parent, "Execution source must be an explicit sector_forecast sibling directory")
    require(not (out / "verification.json").exists(), "Immutable verification already exists")
    checked = {}

    def check(path, expected=None):
        path = local(path)
        actual = digest(path)
        if expected is not None:
            require(actual == expected, f"Hash mismatch: {relative(path)}")
        key = relative(path)
        require(key not in checked or checked[key] == actual, f"Input changed during verification: {key}")
        checked[key] = actual
        return actual

    verifier_hash = check(Path(__file__))
    check(out / "selection_lock.json")
    selected_lock = read(out / "selection_lock.json")
    check(out / "selection.json", selected_lock["selection_sha256"])
    reported = read(out / "selection.json")
    if design_sha256 is not None:
        require(reported["design_lock_sha256"] == design_sha256, "Externally supplied design digest differs")
    for name, field in (("design_lock.json", "design_lock_sha256"), ("data_lock.json", "data_lock_sha256"),
                        ("selection_issuance_lock.json", "selection_issuance_lock_sha256"),
                        ("selection_labeled.jsonl", "selection_labeled_sha256")):
        check(out / name, reported[field])
    design = read(out / "design_lock.json")
    require(set(design["sources"]) == source_inventory(lane), "Frozen execution source inventory changed")
    for path, expected in {**design["sources"], **design["inputs"]}.items():
        check(path, expected)
    check(out / "pre_fit_tests.json", design["pre_fit_tests_sha256"])
    pre_fit = read(out / "pre_fit_tests.json")
    require(pre_fit["exit_code"] == 0 and pre_fit["sources"] == design["sources"], "Pre-fit test binding failed")
    check(lane / "mathematical_review.md", design["mathematical_review_sha256"])
    require(design["sector_models_fitted_before_lock"] is False
            and design["sector_predictive_scores_before_lock"] is False, "Invalid pre-fit declaration")
    spec = read(lane / "specification.json")
    feature_names = read(lane / "feature_contract.json")["ordered_features"]
    require(len(feature_names) == len(set(feature_names)) == 75, "Feature contract changed")
    require(reported["feature_names"] == feature_names, "Reported feature order differs")
    require(spec["split"]["train_years"] == [2022] and spec["split"]["selection_years"] == [2023], "Unexpected discovery split")
    references = spec["references"]
    candidates = [config["name"] for config in spec["candidates"]]
    require(len(references) == 4 and len(candidates) == 3, "Unexpected frozen candidate/reference family")
    names = references + candidates
    acquisition_path = spec["discovery_acquisition"]["path"]
    check(acquisition_path, spec["discovery_acquisition"]["sha256"])
    acquisition = read(local(acquisition_path))
    expected_inputs = {acquisition_path: spec["discovery_acquisition"]["sha256"]}
    for record in acquisition["streams"]:
        expected_inputs[record["decoded_body_path"]] = record["decoded_body_sha256"]
    for record in acquisition["existing_track_status"]:
        expected_inputs[record["stream"]["path"]] = record["stream"]["sha256"]
    sessions = {record["event_key"]: record for record in acquisition["sessions"]}
    for record in sessions.values():
        expected_inputs[record["original_laps_path"]] = record["original_laps_sha256"]
    require(design["inputs"] == expected_inputs, "Input closure differs from acquired discovery data")
    require(sorted(sessions) == spec["discovery_events"], "Acquisition event set changed")
    data_lock = read(out / "data_lock.json")
    require(data_lock["design_lock_sha256"] == checked[relative(out / "design_lock.json")], "Data lock design binding failed")
    require(data_lock["target_labels_read"] is False and data_lock["sector_predictive_scores_computed"] is False
            and type(data_lock["sector_models_fitted"]) is int and data_lock["sector_models_fitted"] == 0,
            "Target-free closure declaration failed")
    events = data_lock["events"]
    require([row["event_key"] for row in events] == spec["discovery_events"], "Data-lock event order/population changed")
    for metadata in events:
        check(metadata["path"], metadata["sha256"])
        check(out / "target_free" / f'{metadata["event_key"]}.json')
        require(read(out / "target_free" / f'{metadata["event_key"]}.json') == metadata, "Per-event data metadata changed")
    issuance_lock = read(out / "selection_issuance_lock.json")
    check(issuance_lock["path"], issuance_lock["sha256"])
    require(local(issuance_lock["path"]) == out / "selection_issued.jsonl", "Unexpected issuance file")
    check(out / "training_labeled.jsonl", issuance_lock["training_labeled_sha256"])
    require(issuance_lock["selection_targets_read"] is False, "Forecast closure did not precede selection labels")
    require(set(issuance_lock["models"]) == set(candidates[1:]), "Fitted model set differs from frozen candidates")
    for model in issuance_lock["models"].values():
        check(model["path"], model["sha256"])
    check(out / "selection_attempt.json")
    attempt = read(out / "selection_attempt.json")
    require(attempt["data_lock_sha256"] == reported["data_lock_sha256"]
            and attempt["design_lock_sha256"] == reported["design_lock_sha256"]
            and attempt["candidates"] == candidates, "Selection attempt binding failed")
    times = [pre_fit["closed_at_utc"], design["locked_at_utc"], data_lock["closed_at_utc"],
             attempt["started_at_utc"], issuance_lock["closed_at_utc"], reported["completed_at_utc"], selected_lock["locked_at_utc"]]
    require([stamp(value) for value in times] == sorted(stamp(value) for value in times), "Invalid declared execution order")
    require(all(stamp(m["closed_at_utc"]) <= stamp(data_lock["closed_at_utc"]) for m in events), "Data lock predates event closure")

    # Replay models on target-free rows BEFORE this verifier opens any labels/CSV.
    selected_events = [metadata for metadata in events if metadata["event_key"] // 100 == 2023]
    expected_events = [str(event) for event in spec["discovery_events"] if event // 100 == 2023]
    require(len(expected_events) == 22, "Selection requires the exact 22 declared 2023 events")
    issued = list(rows(out / "selection_issued.jsonl"))
    require(len(issued) == issuance_lock["issued_rows"], "Issuance count binding failed")
    require(len({row["ledger_id"] for row in issued}) == len(issued), "Duplicate serialized issuance")
    require(sorted({event_identity(row, issued=True) for row in issued}) == expected_events, "Serialized issuance event cohort changed")
    matrix, anchors = [], []
    index = 0
    for metadata in selected_events:
        for row in rows(local(metadata["path"])):
            require(not LABEL_FIELDS.intersection(row), "Target-free ledger contains outcome fields")
            if row["status"] != "issued":
                continue
            require(index < len(issued), "Target-free issuance absent from forecast closure")
            serialized = issued[index]
            expected_identity = {"event_key": str(row["event_key"]), "driver": row["driver"],
                                 "ledger_id": row["issuance_id"], "sector": row["sector"], "checkpoint_ms": row["checkpoint_ms"]}
            require(set(serialized) == set(expected_identity) | set(names), "Forecast closure contains unexpected fields")
            require({key: serialized[key] for key in expected_identity} == expected_identity, "Forecast order/identity changed")
            require(list(row["features"]) == feature_names, "Target-free feature order changed")
            for name in references + [candidates[0]]:
                require(serialized[name] == row["points"][name], "Prelabel reference point changed")
            feature_row = [float("nan") if row["features"][name] is None else finite(row["features"][name], name)
                           for name in feature_names]
            matrix.append(feature_row)
            anchors.append(finite(row["points"][spec["hgb"]["anchor"]], "anchor", positive=True))
            index += 1
    require(index == len(issued), "Forecast closure adds unsupported rows")
    matrix = np.asarray(matrix, dtype=float)
    anchors = np.asarray(anchors, dtype=float)
    model_replay = {}
    for config in spec["candidates"][1:]:
        name = config["name"]
        binding = issuance_lock["models"][name]
        check(binding["path"], binding["sha256"])
        with local(binding["path"]).open("rb") as stream:
            model = pickle.load(stream)
        require(model.features == feature_names and model.anchor == spec["hgb"]["anchor"], "Serialized model feature/anchor binding failed")
        require(model.correction_clip == 5. == spec["hgb"]["prediction_correction_clip_seconds"][1], "Prediction clipping changed")
        parameters = model.estimator.get_params()
        for key in ("loss", "learning_rate", "max_iter", "min_samples_leaf", "l2_regularization", "early_stopping", "random_state"):
            require(parameters[key] == spec["hgb"][key], f"Estimator parameter changed: {key}")
        require(parameters["max_leaf_nodes"] == config["max_leaf_nodes"] and model.estimator.n_features_in_ == len(feature_names), "Estimator shape changed")
        with threadpool_limits(limits=1):
            corrections = model.estimator.predict(matrix)
        predictions = anchors + np.maximum(-5., np.minimum(5., corrections))
        saved = np.asarray([finite(row[name], name, positive=True) for row in issued])
        require(np.isfinite(predictions).all() and (predictions > 0).all(), "Invalid serialized-model replay")
        discrepancy = float(np.max(np.abs(predictions - saved)))
        require(discrepancy <= 1e-12, f"Serialized prediction mismatch: {name}: {discrepancy}")
        model_replay[name] = {"rows": len(saved), "maximum_absolute_difference_seconds": discrepancy,
                              "method": "Direct estimator.predict on target-free columns, then independently apply fixed anchor and [-5,+5] correction clip."}

    # Independently re-resolve ALL retained rows, including unsupported records.
    selected_rows = []
    retention = {}
    csv_rows = 0
    target_associations = 0
    for year, labeled_path in ((2022, out / "training_labeled.jsonl"), (2023, out / "selection_labeled.jsonl")):
        labeled = iter(rows(labeled_path))
        ids = set()
        issuance_cursor = 0
        for metadata in [m for m in events if m["event_key"] // 100 == year]:
            event = metadata["event_key"]
            diagnostic = metadata["diagnostics"]
            require(diagnostic["target_free"] is True and diagnostic["canonical_csv_read"] is False
                    and diagnostic["stream_input_valid"] is True, "Invalid target-free event declaration")
            require(diagnostic["terminal"] is (diagnostic["terminal_status"] in {"Finished", "Finalised", "Ends"}), "Terminal archive classification changed")
            index_by_driver, original_count = canonical_targets(local(sessions[event]["original_laps_path"]))
            csv_rows += original_count
            statuses, outcomes = Counter(), Counter()
            for original in rows(local(metadata["path"])):
                actual = next(labeled, None)
                require(actual is not None, "Labeled ledger dropped original rows")
                require(set(actual) == set(original) | LABEL_FIELDS, "Label attachment changed record schema")
                require({key: actual[key] for key in original} == original, "Label attachment changed a feature, point, provenance or population")
                require(not LABEL_FIELDS.intersection(original), "Target-free ledger includes labels")
                require(event_identity(original) == str(event), "Row belongs to wrong event")
                identity = original["issuance_id"]
                packet = integer(original["packet_sequence"], "packet sequence")
                require(identity == f'{event}:{original["driver"]}:{packet}:S{original["sector"]}', "Noncanonical issuance identity")
                require(identity not in ids, "Raw ledger has duplicate issuance")
                ids.add(identity)
                require(original["status"] in ISSUANCE_STATUSES, "Unknown issuance support status")
                statuses[original["status"]] += 1
                if original["status"] == "issued":
                    require(original["candidate_checkpoint"] is True and original["history_count"] >= 3, "Unsupported issued row")
                    require(original["history_last_available_ms"] < original["checkpoint_ms"], "Noncausal final history support")
                    require(set(original["points"]) == set(references + [candidates[0]])
                            and list(original["features"]) == feature_names, "Issued rows lack common model/features")
                else:
                    require(original["points"] == {} and original["features"] == {}, "Unsupported row acquired model output")
                expected = expected_label(original, index_by_driver, diagnostic["terminal"])
                close({key: actual[key] for key in LABEL_FIELDS}, expected, f"canonical_label.{identity}")
                target_associations += 1
                outcomes[actual["outcome_status"]] += 1
                if year == 2023 and original["status"] == "issued":
                    require(issuance_cursor < len(issued) and issued[issuance_cursor]["ledger_id"] == identity,
                            "Labeled issued cohort differs from prelabel forecasts")
                    selected_rows.append({**issued[issuance_cursor], "outcome_status": actual["outcome_status"],
                                          "target_id": actual["target_id"], "target_time_seconds": actual["target_time_seconds"],
                                          "y_true": actual["y"]})
                    issuance_cursor += 1
            require(dict(statuses) == diagnostic["status_counts"] and sum(statuses.values()) == diagnostic["records"], "All-row retention count changed")
            require(sum(statuses.values()) == diagnostic["pilot_parser"]["ledger_rows"], "Positive raw update retention changed")
            retention[str(event)] = {"records": sum(statuses.values()), "support_statuses": dict(statuses), "outcome_statuses": dict(outcomes)}
        require(next(labeled, None) is None, "Labeled ledger added rows")
        if year == 2023:
            require(issuance_cursor == len(issued), "Labeled ledger dropped issued forecasts")
    computed = selection(selected_rows, candidates, references, spec)
    require(computed["summary"]["issued_events"] == expected_events, "Scored exact 2023 event set changed")
    # Evaluator prose is not a computed metric; compare every other returned field.
    comparable = {key: reported[key] for key in computed}
    comparable["summary"] = {key: value for key, value in comparable["summary"].items() if key != "scope"}
    comparable["subgroups"] = {stage: {key: value for key, value in item.items() if key != "scope"}
                                for stage, item in comparable["subgroups"].items()}
    close(comparable, computed, "selection")
    require(selected_lock["selected"] == computed["selected"]
            and selected_lock["selection_passed"] is computed["selection_passed"], "Selection lock does not match independently reproduced decision")
    expected_action = ("Frozen transfer acquisition may begin" if computed["selection_passed"] else
                       "Stop this candidate family; no transfer acquisition or evaluation")
    require(selected_lock["next_action"] == expected_action, "Next action conflicts with gate result")
    require(reported["transfer_acquired_or_evaluated"] is False, "Selection artifact claims transfer was already read")
    for path, expected in checked.items():
        require(digest(local(path)) == expected, f"Verified evidence changed while checking: {path}")
    result = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS",
              "verifier_path": relative(Path(__file__)), "verifier_sha256": verifier_hash,
              "execution_directory": relative(lane), "explicit_design_sha256": design_sha256,
              "independence": "No run/evaluate imports, no fitting, independent CSV target resolution and numerical aggregation. The verifier author also authored the feature encoder; this is not independent feature-design review.",
              "limits": ["Declared lock timestamps and hash closure establish repository-internal ordering, not an external timestamp attestation.",
                         "Archived recorded-clock causality is a proxy; historical client receipt time and prospective performance remain uncertified.",
                         "Serialized models are replayed, not independently refitted."],
              "verified_files": checked, "verified_file_count": len(checked),
              "canonical_csv_rows_independently_read": csv_rows,
              "all_retained_rows_independently_relabeled": target_associations,
              "event_retention": retention, "serialized_model_replay": model_replay,
              "selection": computed, "all_checks_passed": True,
              "suggested_commit": "research(f1): independently verify sector selection and causal forecast closure"}
    # Exclusive final write occurs only after every closure/numerical check passes.
    with (out / "verification.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {"path": relative(out / "verification.json"), "sha256": digest(out / "verification.json"),
            "verified_files": len(checked), "serialized_rows_per_hgb": len(issued),
            "all_rows_independently_relabeled": target_associations, "selection_passed": computed["selection_passed"]}


def self_test():
    """Hand-computed synthetic checks; never open a historical file or fit."""
    records = []
    # Event1: targets errors [0,0] and [6], checkpoint2, target3. Event2: error4.
    for index, (event, target, truth, forecast) in enumerate((
            ("202301", "A", 10., 10.), ("202301", "A", 10., 10.),
            ("202301", "B", 10., 16.), ("202302", "C", 10., 14.))):
        records.append({"event_key": event, "driver": "1", "ledger_id": str(index), "sector": 1 + index % 2,
                        "checkpoint_ms": 1000., "outcome_status": MATCHED, "y_true": truth,
                        "target_id": target, "target_time_seconds": 10. if target != "B" else 20., "p": forecast})
    result = summary(records, ["p"])
    require(result["models"]["p"][METRICS[0]] == 3., "Synthetic checkpoint denominator failed")
    require(result["models"]["p"][METRICS[1]] == 3.5, "Synthetic target denominator failed")
    constant = intervals([-.5] * 4, ["202301", "202302", "202401", "202402"], 100, 12)
    close(constant["event_percentile_95_interval"], [-.5, -.5])
    close(constant["three_event_percentile_95_interval"], [-.5, -.5])
    require(constant["leave_one_event_out_max_difference"] == -.5, "Synthetic LOO failed")
    # Uneven year lengths exercise year preservation, circular wrap, final-block
    # truncation and RNG stream ordering; compare independent literal resamples.
    values = [1., -2., 4., 3., -5.]
    keys = ["202301", "202302", "202401", "202402", "202403"]
    calculated = intervals(values, keys, 40, 71)
    rng = np.random.default_rng(71)
    independent = [mean([values[i] for i in draw]) for draw in rng.integers(0, 5, size=(40, 5))]
    years = [values[:2], values[2:]]
    block_draws = [rng.integers(0, len(group), size=(40, math.ceil(len(group) / 3))) for group in years]
    brute = []
    for trial in range(40):
        draw = []
        for group, starts in zip(years, block_draws):
            year_draw = []
            for start in starts[trial]:
                year_draw.extend(group[(start + offset) % len(group)] for offset in (0, 1, 2))
            draw.extend(year_draw[:len(group)])
        brute.append(mean(draw))
    close(calculated["event_percentile_95_interval"], percentiles(independent))
    close(calculated["three_event_percentile_95_interval"], percentiles(brute))
    close(calculated["leave_one_event_out_max_difference"], max(mean(values[:i] + values[i + 1:]) for i in range(5)))
    index = {"1": ([1., 2.], [{"csv_row": 0, "recorded_time_seconds": 1., "lap_number": 1., "lap_time_seconds": 10.},
                              {"csv_row": 1, "recorded_time_seconds": 2., "lap_number": 2., "lap_time_seconds": 11.}])}
    row = {"event_key": 202301, "driver": "1", "checkpoint_ms": 1000.}
    label = expected_label(row, index, True)
    require(label["target_id"] == "202301:1:csvrow:1" and label["y"] == 11., "Strict target time/identity failed")
    require(expected_label({**row, "checkpoint_ms": 2000.}, index, True)["outcome_status"] in UNMATCHED, "Unmatched retention failed")
    for bad in ('{"x":NaN}', '{"x":1,"x":2}', '{"x":1e999}'):
        try:
            strict_json(bad)
        except ValueError:
            pass
        else:
            raise ValueError("Strict JSON regression failed")
    return {"status": "PASS", "scope": "Synthetic equations, unequal target multiplicity, multiyear constant-block/LOO, strict target time, unmatched retention, strict JSON"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--execution-dir", type=Path, default=LANE)
    parser.add_argument("--design-sha256", help="Optional coordinating-task digest to bind the completed execution")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--selection-complete", action="store_true",
                        help="Explicitly confirm the coordinating task closed selection before reading historical outputs")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
    else:
        require(args.selection_complete, "Do not inspect historical outputs until the coordinating task confirms selection is complete")
        print(json.dumps(verify(args.out, args.execution_dir, args.design_sha256)))


if __name__ == "__main__":
    main()
