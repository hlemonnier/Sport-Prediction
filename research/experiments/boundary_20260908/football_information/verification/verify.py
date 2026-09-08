"""Independent saved-forecast verification; never fit or import experiment models.

Run only after selection closes. The immediate experiment sources are immutable;
this nested verifier is intentionally outside their source glob. Each execution
gets an exclusive attempt receipt, including failures. Only a full pass writes
verification.json. No later-season outcomes or forecasts are scored.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
LANE = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts/research/boundary_20260908/football_information"
INTERNAL = "shot_strength_90d_ridge0.1"
OLD = ("production_default_dc_auto", "dc365_elo50", "dc_180", INTERNAL)
FITTED = ("market_shared", "pool_shared", "market_classwise", "pool_classwise")
META = ("match_id", "league", "season", "home", "away", "day", "forecast_cutoff_utc",
        "result_available_at", "fit_id", "fit_cutoff_utc")
TOL = 1e-12


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    def invalid(value):
        raise ValueError("Nonstandard JSON constant: " + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def utc(value):
    value = datetime.fromisoformat(value)
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def same(left, right, label, tolerance=TOL):
    if isinstance(right, dict):
        require(isinstance(left, dict) and set(left) == set(right), label + ": keys")
        for key in right:
            same(left[key], right[key], label + "." + key, tolerance)
    elif isinstance(right, list):
        require(isinstance(left, list) and len(left) == len(right), label + ": length")
        for index, (a, b) in enumerate(zip(left, right)):
            same(a, b, f"{label}[{index}]", tolerance)
    elif isinstance(right, bool) or right is None or isinstance(right, str):
        require(type(left) is type(right) and left == right, label + ": exact value")
    else:
        require(not isinstance(left, bool) and math.isfinite(float(left)) and
                math.isfinite(float(right)) and abs(float(left) - float(right)) <= tolerance,
                f"{label}: {left!r} != {right!r}")


def unique(rows):
    result = {}
    for row in rows:
        key = row["match_id"]
        require(key == f"{row['league']}:{row['season']}:{row['home']}:{row['away']}", "Fixture metadata identity")
        require(key not in result, "Duplicate fixture: " + key)
        require(utc(row["fit_cutoff_utc"]) <= utc(row["forecast_cutoff_utc"]) < utc(row["result_available_at"]),
                "Fixture clock ordering: " + key)
        result[key] = row
    return result


def softmax(logits):
    values = np.asarray(logits, dtype=float)
    weights = np.exp(values - values.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)


def simplex(values, floor=None):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 2 and values.shape[1] == 3 and len(values), "N by three probabilities required")
    require(np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all(), "Invalid probability")
    require(np.allclose(values.sum(1), 1, atol=1e-10, rtol=0), "Probability simplex sum")
    if floor is not None:
        values = np.maximum(values, floor)
        values = values / values.sum(1, keepdims=True)
    return values


def pool(saved, rows, spec):
    """Replay affine logits with independently constructed sum-zero contrasts."""
    width = 1 if saved["family"] == "shared" else 3
    require(saved["family"] in ("shared", "classwise") and type(saved["internal"]) is bool, "Pool type")
    internal = saved["internal"]
    theta = np.asarray(saved["theta"], dtype=float)
    require(theta.shape == (width * (1 + internal) + 2,), "Pool parameter shape")
    require(np.isfinite(theta).all() and (theta[:-2] >= 0).all() and (theta[:-2] <= 4).all(), "Pool coefficient bounds")
    u, v = theta[-2:]
    bias = np.array([u / math.sqrt(2) + v / math.sqrt(6), -u / math.sqrt(2) + v / math.sqrt(6), -2 * v / math.sqrt(6)])
    lm = np.log(simplex([r["probabilities"]["avg_power"] for r in rows], spec["pooling"]["probability_floor"]))
    lp = np.log(simplex([r["probabilities"][INTERNAL] for r in rows], spec["pooling"]["probability_floor"])) if internal else None
    logits = lm * theta[:width] + bias
    if internal:
        logits += lp * theta[width:2*width]
    return softmax(logits), (theta, bias, lm, lp, logits)


def pool_optimum(saved, training, labels, spec):
    predictions, (theta, bias, lm, lp, logits) = pool(saved, training, spec)
    n = len(training)
    y = np.asarray(labels)
    width = 1 if saved["family"] == "shared" else 3
    penalty = spec["pooling"]["penalty"]
    a = theta[:width]
    c = theta[width:2*width] if saved["internal"] else np.zeros(width)
    loss = -np.log(predictions[np.arange(n), y]).mean() + penalty/2*(sum((a-1)**2)+sum(c*c)+sum(bias*bias))
    residual = predictions.copy()
    residual[np.arange(n), y] -= 1
    residual /= n
    ga = (residual*lm).sum(0)
    components = [(np.array([ga.sum()]) if width == 1 else ga) + penalty*(a-1)]
    if saved["internal"]:
        gc = (residual*lp).sum(0)
        components.append((np.array([gc.sum()]) if width == 1 else gc) + penalty*c)
    gb = residual.sum(0)
    components.append(np.array([(gb[0]-gb[1])/math.sqrt(2), (gb[0]+gb[1]-2*gb[2])/math.sqrt(6)]) + penalty*theta[-2:])
    gradient = np.concatenate(components)
    for i in range(len(theta)-2):
        if theta[i] <= 1e-8:
            gradient[i] = min(gradient[i], 0)
        if theta[i] >= 4-1e-8:
            gradient[i] = max(gradient[i], 0)
    kkt = float(np.abs(gradient).max())
    require(kkt <= spec["pooling"]["maximum_projected_gradient"] + TOL, "Saved model is not a constrained optimum")
    diagnostic = saved["diagnostics"]
    require(diagnostic["n"] == n and diagnostic["solver_success"] is True, "Fit diagnostic population/convergence")
    same(loss, diagnostic["objective"], "Saved objective")
    same(kkt, diagnostic["projected_gradient_max"], "Saved KKT")
    same(penalty, diagnostic["penalty"], "Saved penalty")
    return kkt


def shot_features(rows):
    output = []
    for row in rows:
        p = row["probabilities"]
        mix, dc, elo = (np.asarray(p[key]) for key in ("dc365_elo50", "dc_365", "elo_component"))
        output.append([np.log(mix[0]/mix[2]), np.log(mix[1]/mix[2]),
                       np.log(dc[0]/dc[2])-np.log(elo[0]/elo[2]), np.log(dc[1]/dc[2])-np.log(elo[1]/elo[2]),
                       float(row["league"] == "SP1"), float(row["league"] == "I1"), *row["log_shot_means"]["90"]])
    return np.asarray(output)


def metrics(rows, name, labels):
    p = simplex([r["probabilities"][name] for r in rows])
    require((p > 0).all(), "Scored zero probability")
    y = [labels[r["match_id"]] for r in rows]
    n = len(rows)
    loss = math.fsum(-math.log(p[i, target]) for i, target in enumerate(y))/n
    brier = math.fsum(math.fsum((float(p[i, j])-int(j == target))**2 for j in range(3)) for i, target in enumerate(y))/n
    hits = [int(int(np.argmax(p[i])) == target) for i, target in enumerate(y)]
    confidence = [float(max(vector)) for vector in p]
    bins, ece = [], 0.
    for k in range(10):
        indices = [i for i, c in enumerate(confidence) if c >= k/10 and (c < (k+1)/10 if k < 9 else c <= 1)]
        if indices:
            m = math.fsum(confidence[i] for i in indices)/len(indices)
            hit = sum(hits[i] for i in indices)/len(indices)
            ece += len(indices)/n*abs(m-hit)
            bins.append({"lower": k/10, "upper": (k+1)/10, "n": len(indices), "mean_confidence": m, "accuracy": hit})
    return {"n": n, "log_loss": loss, "brier_sum_classes": brier, "accuracy": sum(hits)/n,
            "top_label_ece_10_bins": ece, "calibration_bins": bins,
            "class_mean_probability_minus_frequency": [math.fsum(float(row[j]) for row in p)/n-y.count(j)/n for j in range(3)]}


def percentile(values, probability):
    values = sorted(float(v) for v in values)
    index = (len(values)-1)*probability
    left = int(math.floor(index));right = int(math.ceil(index))
    return values[left] + (index-left)*(values[right]-values[left])


def uncertainty(rows, candidate, reference, days, labels, spec):
    rng = np.random.default_rng(spec["uncertainty"]["seed"])
    samples = spec["uncertainty"]["resamples"]
    boot = np.zeros(samples)
    all_delta = []
    counts = {}
    strata = sorted({f"{r['league']}:{r['season']}" for r in rows})
    for stratum in strata:
        local = [r for r in rows if f"{r['league']}:{r['season']}" == stratum]
        origin = min(date.fromisoformat(r["day"]) for r in local)
        blocks = defaultdict(list)
        for row in local:
            target = labels[row["match_id"]]
            delta = -math.log(row["probabilities"][candidate][target])+math.log(row["probabilities"][reference][target])
            all_delta.append(delta)
            blocks[(date.fromisoformat(row["day"])-origin).days//days].append(delta)
        keys = list(blocks)
        totals = np.array([math.fsum(blocks[k]) for k in keys])
        sizes = np.array([len(blocks[k]) for k in keys])
        draws = rng.integers(0, len(keys), size=(samples, len(keys)))
        # Independently sum each sampled block position, rather than borrowing the evaluator.
        numerator = np.zeros(samples);denominator = np.zeros(samples)
        for position in range(len(keys)):
            numerator += totals[draws[:, position]]
            denominator += sizes[draws[:, position]]
        boot += len(local)/len(rows)*(numerator/denominator)
        counts[stratum] = len(keys)
    return {"difference_candidate_minus_reference": math.fsum(all_delta)/len(all_delta),
            "percentile_95_interval": [percentile(boot, .025), percentile(boot, .975)],
            "bootstrap_fraction_difference_below_zero": float(sum(boot < 0)/samples),
            "block_calendar_days": days, "blocks_by_season": counts, "resamples": samples,
            "reference": reference}


def raw_source_rows(paths):
    rows = {}
    for path in paths:
        parts = path.stem.split("_")
        year = int(parts[-2]);league = parts[0] if parts[0] == "E0" else parts[1]
        require(2019 <= year <= 2023, "Later or unnecessary raw season access")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            average = "Avg" if "AvgH" in reader.fieldnames else "BbAv"
            for row in reader:
                key = f"{league}:{year}:{row['HomeTeam'].strip()}:{row['AwayTeam'].strip()}"
                require(key not in rows, "Duplicate canonical CSV fixture")
                day = None
                for pattern in ("%d/%m/%Y", "%d/%m/%y"):
                    try:
                        day = datetime.strptime(row["Date"], pattern).date().isoformat();break
                    except ValueError:
                        pass
                require(day is not None and row["FTR"] in "HDA" and len(row["FTR"]) == 1, "CSV date/result")
                goals = [int(row["FTHG"]), int(row["FTAG"])]
                label = 0 if goals[0] > goals[1] else (1 if goals[0] == goals[1] else 2)
                require(label == "HDA".index(row["FTR"]), "Canonical score/result disagreement")
                rows[key] = {"day": day, "label": label, "goals": goals,
                             "avg": [row.get(average+c) for c in "HDA"], "b365": [row.get("B365"+c) for c in "HDA"],
                             "fields": [average+c for c in "HDA"]+["B365"+c for c in "HDA"], "path": path}
    return rows


def verify(out, expected_design):
    checks = {"hash_bindings": 0}
    def binding(path, expected):
        require(sha(path) == expected, "SHA256 mismatch: " + str(path))
        checks["hash_bindings"] += 1
    binding(out/"design_lock.json", expected_design)
    design = read(out/"design_lock.json")
    for name, value in design["sources"].items(): binding(ROOT/name, value)
    for name, value in design["inputs"].items(): binding(ROOT/name, value)
    binding(ROOT/design["independent_review"]["path"], design["independent_review"]["sha256"])
    review = read(ROOT/design["independent_review"]["path"])
    require(review["approved_for_execution_lock"] is True and review["source_files"] == design["sources"], "Exact reviewed source approval")
    parent = read(ROOT/"artifacts/research/boundary_20260908/football/selection.json")
    inventory = {str(path.relative_to(ROOT)) for pattern in ("*.py", "*.json", "*.md") for path in LANE.glob(pattern)} | set(parent["source_files"])
    require(inventory == set(design["sources"]), "Execution source inventory changed")
    for name, value in parent["source_files"].items(): binding(ROOT/name, value)
    binding(out/"pre_fit_tests.json", design["pre_fit_tests_sha256"])
    tests = read(out/"pre_fit_tests.json")
    require(tests["exit_code"] == 0 and tests["source_files"] == design["sources"], "Frozen tests do not bind sources")
    spec = read(LANE/"specification.json")
    binding(ROOT/spec["proposal"]["path"], spec["proposal"]["sha256"])
    require(spec["candidates"] == ["pool_shared", "pool_classwise"], "Unexpected candidates")
    require(spec["selection"]["league"] == "E0" and spec["selection"]["seasons"] == [2022, 2023], "Selection dates changed")
    require(spec["training"]["history_elapsed_days"] == 1460, "Training window changed")
    data_lock = read(out/"data_lock.json")
    binding(out/"design_lock.json", data_lock["design_lock_sha256"])
    binding(out/"prepared_inputs.json", data_lock["prepared_inputs_sha256"])
    binding(out/"incumbent_training_reconstruction.json", data_lock["incumbent_reconstruction_sha256"])
    for path, value in data_lock["quote_input_bindings"].items(): binding(ROOT/path, value)
    require(data_lock["labels_attached"] is False and data_lock["pooling_models_fitted"] == 0 and data_lock["new_candidate_scores_computed"] is False, "Data closure status")
    issuance_lock = read(out/"selection_issuance_lock.json")
    binding(out/"selection_issued.json", issuance_lock["issued_sha256"])
    binding(out/"selection_fits.json", issuance_lock["fits_sha256"])
    binding(out/"data_lock.json", issuance_lock["data_lock_sha256"])
    require(issuance_lock["current_row_labels_attached"] is False and issuance_lock["earlier_selection_labels_used_only_after_strict_availability"] is True, "Prelabel issuance flags")
    result = read(out/"selection.json");selection_lock = read(out/"selection_lock.json")
    for payload in (result, selection_lock):
        binding(out/"design_lock.json", payload["design_lock_sha256"])
        binding(out/"data_lock.json", payload["data_lock_sha256"])
    binding(out/"selection_issuance_lock.json", result["issuance_lock_sha256"])
    binding(out/"selection.json", selection_lock["selection_sha256"])
    prepare_attempt = read(out/"prepare_attempt.json");selection_attempt = read(out/"selection_attempt.json")
    binding(out/"design_lock.json", prepare_attempt["design_lock_sha256"])
    binding(out/"data_lock.json", selection_attempt["data_lock_sha256"])
    stamps = [tests["completed_at_utc"], design["locked_at_utc"], prepare_attempt["started_at_utc"], data_lock["closed_at_utc"],
              selection_attempt["started_at_utc"], issuance_lock["closed_at_utc"], result["completed_at_utc"], selection_lock["closed_at_utc"]]
    require(all(utc(a) <= utc(b) for a, b in zip(stamps, stamps[1:])), "Declared issuance/label phase chronology")

    feature_rows = []
    for league in spec["leagues"]:
        path = ROOT/f"artifacts/research/boundary_20260908/football/features_selection_{league}.json"
        payload = read(path)
        for name, value in payload["input_hashes"].items(): binding(ROOT/name, value)
        require(all(r["league"] == league for r in payload["rows"]), "Mixed feature league")
        feature_rows.extend(payload["rows"])
    feature_rows.sort(key=lambda row: (utc(row["forecast_cutoff_utc"]), row["match_id"]))
    features = unique(feature_rows)
    require(len(features) == 5700 and {r["season"] for r in feature_rows} == set(range(2019, 2024)), "Feature population")
    paths = [ROOT/name for name in design["inputs"] if name.endswith(".csv") and 2019 <= int(Path(name).stem.split("_")[-2]) <= 2023]
    canonical = raw_source_rows(paths)
    require(set(canonical) == set(features), "Canonical CSV/features identity set")
    for key, row in features.items():
        require(row["day"] == canonical[key]["day"] and row["label"] == canonical[key]["label"] and row["goals"] == canonical[key]["goals"], "Canonical target pairing: " + key)
    labels = {key: row["label"] for key, row in canonical.items()}
    prepared = read(out/"prepared_inputs.json");prepared_by_id = unique(prepared)
    expected_ids = {r["match_id"] for r in feature_rows if r["season"] in [2020, 2021, 2022, 2023]}
    require(set(prepared_by_id) == expected_ids and len(prepared) == 4560 == data_lock["rows"], "Prepared common population")
    require(prepared == sorted(prepared, key=lambda row: (utc(row["forecast_cutoff_utc"]), row["match_id"])), "Prepared chronological ordering")
    frozen = unique(parent["predictions"])
    require(len(frozen) == 760 and set(frozen) == {k for k in expected_ids if features[k]["league"] == "E0" and features[k]["season"] in (2022, 2023)}, "Exact frozen EPL cohort")
    require(parent["selected_model"] == INTERNAL, "Strong internal comparator changed")
    for key, row in prepared_by_id.items():
        original = features[key]
        require(all(row[k] == original[k] for k in META), "Prepared metadata changed")
        require("label" not in row and "goals" not in row and row["labels_attached"] is False, "Prepared target contamination")
        require(row["internal_vector_reused_without_transformation"] is True and type(row["market_ready"]) is bool, "Prepared provenance flags")
        for name in OLD:
            if key in frozen:
                require(row["probabilities"][name] == frozen[key]["probabilities"][name], "Frozen vector changed")
            elif name != INTERNAL:
                require(row["probabilities"][name] == original["probabilities"][name], "Baseline training vector changed")
        q = row["quote"];raw = canonical[key]
        require(q["match_id"] == key and q["day"] == raw["day"] and q["avg"] == raw["avg"] and q["b365"] == raw["b365"] and q["fields"] == raw["fields"], "Raw quote pairing")
        require(q["source_path"] == str(raw["path"].relative_to(ROOT)) and q["source_sha256"] == design["inputs"][q["source_path"]], "Quote source binding")
        require(q["quote_observed_at"] is None and q["receipt_certified"] is False and q["product"] == "provider_preclosing_snapshot", "Quote clock falsely certified")
        try:
            triples = {book: np.array([float(v) for v in q[book]]) for book in ("avg", "b365")}
            ready = all(np.isfinite(v).all() and (v > 1).all() for v in triples.values())
        except (ValueError, TypeError):
            ready = False
        require(row["market_ready"] is ready, "Market readiness")
        if ready:
            require(row["fallback_reason"] is None, "Unexpected fallback")
            for book, odds in triples.items():
                raw_probability = 1/odds
                same((raw_probability/raw_probability.sum()).tolist(), row["probabilities"][book+"_normalized"], "Normalized odds")
                power = row["power_exponents"][book]
                require(math.isfinite(power) and power > 0, "Power root positivity")
                powered = np.exp(-power*np.log(odds))
                require(abs(float(powered.sum())-1) <= 1e-10, "Power root equation")
                same((powered/powered.sum()).tolist(), row["probabilities"][book+"_power"], "Power odds")
            same(((np.array(row["probabilities"][INTERNAL])+row["probabilities"]["avg_power"])/2).tolist(), row["probabilities"]["arithmetic_half"], "Arithmetic reference")
        else:
            require(row["fallback_reason"] == "incomplete_or_invalid_preclosing_snapshot", "Missing quote reason")
            for name in ("avg_normalized", "avg_power", "b365_normalized", "b365_power", "arithmetic_half"):
                require(row["probabilities"][name] == row["probabilities"][INTERNAL], "Common missing quote fallback")
    checks["canonical_rows_independently_labeled"] = len(canonical)
    checks["prepared_rows"] = len(prepared)
    checks["frozen_selection_vectors_exact"] = len(frozen)*len(OLD)

    reconstruction = read(out/"incumbent_training_reconstruction.json")
    require(reconstruction["configuration"] == ["shot_strength_90d", .1] and reconstruction["generated_rows"] == 3800 and reconstruction["verbatim_reused_selection_rows"] == 760, "Incumbent reconstruction configuration")
    generated_ids = set();generated_error = 0.
    for fit in reconstruction["fits"]:
        cutoff = utc(fit["cutoff"]);lower = cutoff-timedelta(days=1460)
        training = [r for r in feature_rows if lower <= utc(r["forecast_cutoff_utc"]) < cutoff and utc(r["result_available_at"]) <= cutoff]
        ids = [r["match_id"] for r in training]
        require(ids == fit["training_ids"] and digest(ids) == fit["training_ids_sha256"], "Incumbent prequential training membership")
        batch = [r for r in feature_rows if r["match_id"] in expected_ids-set(frozen) and r["league"] == fit["league"] and r["fit_id"] == fit["fit_id"]]
        require(batch and not generated_ids.intersection(r["match_id"] for r in batch), "Incumbent reconstruction batch")
        require(all(utc(r["fit_cutoff_utc"]) == cutoff for r in batch), "Incumbent reconstruction cutoff")
        diagnostic = fit["models"][INTERNAL]
        x = shot_features(training);mean = x.mean(0);scale = x.std(0);scale = np.where(scale > 1e-12, scale, 1.)
        same(mean.tolist(), diagnostic["mean"], "Incumbent fitted scaler mean")
        same(scale.tolist(), diagnostic["scale"], "Incumbent fitted scaler scale")
        require(diagnostic["fit_n"] == len(training) and diagnostic["penalty"] == .1 and diagnostic["converged"] is True, "Incumbent fit metadata")
        xb = np.column_stack((np.ones(len(batch)), (shot_features(batch)-mean)/scale))
        values = softmax(np.log(np.asarray([r["probabilities"]["dc365_elo50"] for r in batch]))+xb@np.asarray(diagnostic["weights"]))
        for row, vector in zip(batch, values):
            generated_error = max(generated_error, float(np.max(np.abs(vector-prepared_by_id[row["match_id"]]["probabilities"][INTERNAL]))))
            generated_ids.add(row["match_id"])
    require(generated_ids == expected_ids-set(frozen) and generated_error <= TOL, "Reconstructed internal vectors differ")
    checks["reconstructed_internal_vectors_replayed"] = len(generated_ids)
    checks["reconstructed_internal_max_absolute_difference"] = generated_error

    issued = read(out/"selection_issued.json");issued_by_id = unique(issued)
    require(set(issued_by_id) == set(frozen) and len(issued) == 760 == issuance_lock["rows"] == data_lock["selection_rows"], "Selection full cohort")
    require(issued == sorted(issued, key=lambda row: (utc(row["forecast_cutoff_utc"]), row["match_id"])), "Selection order")
    require(all(type(row["season"]) is int and row["season"] in (2022, 2023) and row["league"] == "E0" for row in issued), "Selection exact role")
    for key, row in issued_by_id.items():
        original = prepared_by_id[key]
        projected = {k: v for k, v in row.items() if k not in ("pool_fit_id", "pool_fit_cutoff_utc", "probabilities")}
        require(projected == {k: v for k, v in original.items() if k != "probabilities"}, "Issuance metadata/target-free projection changed")
        require(set(row["probabilities"]) == set(original["probabilities"]) | set(FITTED), "Forecast family population")
        require(all(row["probabilities"][name] == value for name, value in original["probabilities"].items()), "Original reference changed while issuing")
        require(row["market_ready"] is True and frozen[key]["label"] == labels[key], "Selection availability/label alignment")
    fits = read(out/"selection_fits.json")
    seen = set();maximum_error = 0.;max_kkt = 0.;fit_ids = set()
    for fit in fits:
        cutoff = utc(fit["cutoff_utc"]);lower = cutoff-timedelta(days=1460)
        key = (fit["cutoff_utc"], fit["fit_id"])
        require(key not in fit_ids, "Duplicate pool fit");fit_ids.add(key)
        eligible = [r for r in prepared if lower <= utc(r["forecast_cutoff_utc"]) < cutoff and utc(r["result_available_at"]) < cutoff]
        training = [r for r in eligible if r["market_ready"] is True]
        require(fit["training_ids"] == [r["match_id"] for r in training] and fit["training_ids_sha256"] == digest(fit["training_ids"]), "Pool common training membership")
        require(fit["excluded_missing_snapshot_ids"] == [r["match_id"] for r in eligible if r["market_ready"] is False], "Common excluded training population")
        require(fit["latest_training_result_available_at"] == max((r["result_available_at"] for r in training), key=utc), "Latest training result")
        batch = [r for r in issued if r["pool_fit_id"] == fit["fit_id"] and r["pool_fit_cutoff_utc"] == fit["cutoff_utc"]]
        require(batch and not seen.intersection(r["match_id"] for r in batch) and not set(fit["training_ids"]).intersection(r["match_id"] for r in batch), "Pool batch overlap")
        require(set(fit["models"]) == set(FITTED), "Matched pool family fits")
        for name in FITTED:
            saved = fit["models"][name]
            require(saved["family"] == name.split("_")[1] and saved["internal"] is name.startswith("pool_"), "Serialized model role")
            replay, _ = pool(saved, batch, spec)
            expected = np.asarray([r["probabilities"][name] for r in batch])
            maximum_error = max(maximum_error, float(np.max(np.abs(replay-expected))))
            max_kkt = max(max_kkt, pool_optimum(saved, training, [labels[r["match_id"]] for r in training], spec))
        seen.update(r["match_id"] for r in batch)
    require(seen == set(frozen) and maximum_error <= TOL, "Serialized forecast replay")
    require(fit_ids == {(r["fit_cutoff_utc"], r["fit_id"]) for r in issued}, "Inherited fit bucket set")
    checks.update(selection_rows=len(issued), fitted_pool_models=4*len(fits), serialized_probability_vectors_replayed=4*len(issued),
                  pool_max_absolute_prediction_difference=maximum_error, maximum_independent_projected_gradient=max_kkt)

    summary = result["summary"];names = spec["candidates"]+spec["references"]
    calculated = {name: metrics(issued, name, labels) for name in names}
    same(calculated, summary["metrics"], "Pooled metrics")
    internal_ablation = {}
    for family in ("shared", "classwise"):
        candidate, reference = "pool_"+family, "market_"+family
        width = 1 if family == "shared" else 3
        coefficients = [fit["models"][candidate]["theta"][width:2*width] for fit in fits]
        all_zero = all(value == 0. for vector in coefficients for value in vector)
        maximum_probability_gap = max(abs(float(a)-float(b)) for row in issued
            for a, b in zip(row["probabilities"][candidate], row["probabilities"][reference]))
        internal_ablation[family] = {
            "refit_blocks": len(fits), "internal_coefficients_by_refit": coefficients,
            "all_internal_coefficients_exactly_zero": all_zero,
            "candidate_minus_matched_market_nll": calculated[candidate]["log_loss"]-calculated[reference]["log_loss"],
            "maximum_absolute_probability_difference": maximum_probability_gap,
            "interpretation": ("Every internal coefficient is zero, so the fitted pool uses no internal-model information. Any discrepancy from the matched market-only fit is numerical optimization error, not evidence of added information."
                if all_zero else "Some internal coefficients are nonzero; the frozen materiality and uncertainty gates still determine advancement.")}
    yearly = {str(year): {name: metrics([r for r in issued if r["season"] == year], name, labels) for name in names} for year in (2022, 2023)}
    same(yearly, summary["by_season"], "Season metrics")
    selected = min(spec["candidates"], key=lambda name: (calculated[name]["log_loss"], name))
    comparisons = {};comparison_count = 0
    for candidate in spec["candidates"]:
        comparisons[candidate] = {}
        for reference in spec["references"]:
            comparisons[candidate][reference] = {}
            for days in spec["uncertainty"]["blocks_calendar_days"]:
                independent = uncertainty(issued, candidate, reference, days, labels, spec)
                saved = summary["candidate_comparisons"][candidate][reference][str(days)]
                same(independent, {k: v for k, v in saved.items() if k != "note"}, f"Paired bootstrap {candidate}/{reference}/{days}")
                comparisons[candidate][reference][str(days)] = independent
                comparison_count += 1
    gates = {reference: {"relative_nll_gain": 1-calculated[selected]["log_loss"]/calculated[reference]["log_loss"],
                        "minimum_gain_met": calculated[selected]["log_loss"] <= (1-spec["selection"]["minimum_relative_nll_gain_vs_every_reference"])*calculated[reference]["log_loss"],
                        "negative_upper_28day_interval": comparisons[selected][reference]["28"]["percentile_95_interval"][1] < 0}
             for reference in spec["references"]}
    advanced = all(v["minimum_gain_met"] and v["negative_upper_28day_interval"] for v in gates.values())
    same(gates, summary["all_reference_gate_checks"], "Every-reference advancement gates")
    require(selected == summary["selected_candidate"] == selection_lock["selected_candidate"], "Candidate choice")
    require(summary["advances_to_later_evaluation"] is advanced and selection_lock["advances_to_later_evaluation"] is advanced, "Advancement boolean")
    require(summary["n"] == len(issued) and summary["population_sha256"] == digest([r["match_id"] for r in issued]), "Summary denominator/population")
    require(summary["decision"] == ("eligible_for_locked_later_research" if advanced else "stop_no_new_transfer_or_promotion"), "Selection stop decision")
    checks.update(pooled_and_season_metric_groups=len(names)*3, independent_bootstrap_comparisons=comparison_count,
                  selection_reference_gates=len(gates))
    # Recheck every frozen source and phase artifact used above before success.
    for name, value in design["sources"].items(): binding(ROOT/name, value)
    binding(out/"design_lock.json", expected_design)
    for path, value in (("prepared_inputs.json", data_lock["prepared_inputs_sha256"]),
                        ("incumbent_training_reconstruction.json", data_lock["incumbent_reconstruction_sha256"]),
                        ("selection_issued.json", issuance_lock["issued_sha256"]), ("selection_fits.json", issuance_lock["fits_sha256"]),
                        ("selection.json", selection_lock["selection_sha256"])):
        binding(out/path, value)
    return {"status": "PASS", "design_lock_sha256": expected_design,
            "source_files_sha256": digest(design["sources"]), "selection_sha256": sha(out/"selection.json"),
            "checks": checks, "selected_candidate": selected, "advances_to_later_evaluation": advanced,
            "independently_recomputed_metrics": calculated, "independently_recomputed_gate_checks": gates,
            "matched_market_only_ablation": internal_ablation,
            "limitations": ["No fitting or experiment model/evaluation helper was imported or called.",
                "Historical internal vectors were replayed from saved coefficients and independent feature equations; this is not independent refitting of predecessor models.",
                "Source receipt timestamps and phase timestamps are recorded assertions, not external historical publication attestations.",
                "Unknown quote times prevent certification of availability before the snapshot or at a fixed forecast lead time.",
                "Only 2019-2023 canonical CSV outcomes and the closed 2022-2023 selection were inspected; no later season was scored.",
                "The verifier author also reviewed the experiment before execution; implementation and metric recomputation are independent of its author."],
            "suggested_commit": "research(football): independently verify market pooling selection evidence"}


def self_test():
    rows = [{"match_id": "A", "day": "2022-01-01", "league": "E0", "season": 2022,
             "probabilities": {"a": [.5, .25, .25], "b": [.25, .5, .25]}}]
    measured = metrics(rows, "a", {"A": 0})
    same(measured["log_loss"], math.log(2), "Synthetic NLL")
    same(measured["brier_sum_classes"], .375, "Synthetic Brier")
    paired = uncertainty(rows, "a", "b", 28, {"A": 0}, {"uncertainty": {"seed": 1, "resamples": 40}})
    same(paired["percentile_95_interval"], [-math.log(2)]*2, "Synthetic interval")
    require(utc("2022-01-01T01:00:00+01:00") == utc("2022-01-01T00:00:00"), "Synthetic UTC normalization")
    require(percentile([1., 2., 3., 4.], .25) == 1.75, "Independent percentile interpolation")
    print(json.dumps({"self_test": "PASS", "historical_artifacts_read": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--design-sha256")
    parser.add_argument("--selection-complete", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test();return
    if not args.selection_complete or not args.design_sha256:
        parser.error("A closed selection and explicit --selection-complete --design-sha256 are required")
    require((args.out/"selection_lock.json").exists(), "Selection is not closed")
    require(not (args.out/"verification.json").exists(), "Existing successful verification is immutable")
    started = datetime.now(timezone.utc)
    source_hash = sha(__file__)
    attempt = args.out/("verification_attempt_"+started.strftime("%Y%m%dT%H%M%S%fZ")+".json")
    base = {"started_at_utc": started.isoformat(), "verifier_sha256": source_hash, "design_sha256": args.design_sha256}
    try:
        result = verify(args.out, args.design_sha256)
        require(sha(__file__) == source_hash, "Verifier source changed during execution")
        result.update(verifier_sha256=source_hash, completed_at_utc=datetime.now(timezone.utc).isoformat())
        save(attempt, {**base, "status": "PASS", "completed_at_utc": result["completed_at_utc"]})
        result["attempt"] = {"path": str(attempt.relative_to(ROOT)), "sha256": sha(attempt)}
        save(args.out/"verification.json", result)
        print(json.dumps({"status": "PASS", "checks": result["checks"], "selected_candidate": result["selected_candidate"],
                          "advances_to_later_evaluation": result["advances_to_later_evaluation"], "verification_sha256": sha(args.out/"verification.json")}, indent=2))
    except Exception as exc:
        if not attempt.exists():
            save(attempt, {**base, "status": "FAIL", "failure": type(exc).__name__+": "+str(exc), "traceback": traceback.format_exc(),
                           "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    main()
