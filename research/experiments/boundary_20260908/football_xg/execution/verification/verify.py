"""Independent frozen-artifact replay; no fitting, model imports or new selection.

Checks raw xG eligibility, serialized convex scores, original matched forecasts,
metrics and paired block resampling using separately expressed mathematics.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[6]
OUT = ROOT / "artifacts/research/boundary_20260908/football_xg/execution"
OLD = ROOT / "artifacts/research/boundary_20260908/football"
SOURCE = ROOT / "research/experiments/boundary_20260908/football_xg"
SHOT = "shot_strength_90d_ridge0.1"
REFS = ["dc365_elo50", "dc_180", SHOT]
BOUND = {}
CHECKS = defaultdict(int)
MAXIMA = defaultdict(float)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    BOUND[str(path.relative_to(ROOT))] = sha(path)
    return json.loads(path.read_text())


def close(actual, expected, tolerance=2e-11, label="numerical"):
    a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == e.shape and np.isfinite(a).all() and np.isfinite(e).all(), label
    error = float(np.max(np.abs(a-e))) if a.size else 0.
    assert np.allclose(a, e, rtol=tolerance, atol=tolerance), (label, error)
    MAXIMA[label] = max(MAXIMA[label], error)


def normalized(logits):
    value = np.exp(logits-np.max(logits, axis=1, keepdims=True))
    return value/value.sum(1, keepdims=True)


def probability(rows, name):
    p = np.asarray([r["probabilities"][name] for r in rows], dtype=float)
    assert p.shape == (len(rows), 3) and np.isfinite(p).all() and (p > 0).all()
    close(p.sum(1), np.ones(len(rows)), label="probability_sum")
    return p


def availability(row, delay, zones):
    assert row["accepted_exact_join"] in (True, "True")
    assert row["canonical_date"] == row["provider_date"]
    for side in ("home", "away"):
        assert float(row[side+"_goals"]) == float(row["canonical_"+side+"_goals"])
    day = date.fromisoformat(row["canonical_date"])
    local = datetime.combine(day+timedelta(days=1), time.min, ZoneInfo(zones[row["league"]]))
    return (local.astimezone(timezone.utc)+timedelta(hours=24*delay)).replace(tzinfo=None)


def design(teams, fixtures):
    indexes = {team: i for i, team in enumerate(teams)}
    x = np.zeros((2*len(fixtures), 2*len(teams)+1))
    for index, (home, away) in enumerate(fixtures):
        for n, attack, defense in ((2*index, home, away), (2*index+1, away, home)):
            if attack in indexes:
                x[n, indexes[attack]] = 1
            if defense in indexes:
                x[n, len(teams)+indexes[defense]] = 1
        x[2*index, -1] = 1
    return x


def features(row, config):
    mix, dc, elo = [np.asarray(row["probabilities"][k]) for k in ("dc365_elo50", "dc_365", "elo_component")]
    base = [np.log(mix[0])-np.log(mix[2]), np.log(mix[1])-np.log(mix[2]),
            np.log(dc[0])-np.log(dc[2])-np.log(elo[0])+np.log(elo[2]),
            np.log(dc[1])-np.log(dc[2])-np.log(elo[1])+np.log(elo[2]),
            float(row["league"] == "SP1"), float(row["league"] == "I1")]
    if config["mode"] in ("add", "shot"):
        base += row["log_shot_means"]["90"]
    if config["mode"] != "shot":
        base += row["log_xg_means"][str(config["half_life"])]
    return base


def ordered(rows):
    return sorted(rows, key=lambda r: (r["forecast_cutoff_utc"], r["match_id"]))


def check_feature_cache(phase, delay, raw, spec, lock):
    combined = []
    for league in spec["leagues"]:
        path = OUT/f"features_{phase}_{league}_delay{delay}.json"
        value, inherited = read(path), read(OLD/f"features_{phase}_{league}.json")
        assert value["sources"] == lock["sources"]
        assert value["design_lock_sha256"] == sha(OUT/"design_lock.json")
        assert value["inherited_feature_cache_sha256"] == sha(OLD/f"features_{phase}_{league}.json")
        rows = value["rows"]
        assert len(rows) == len(inherited["rows"])
        assert len({r["match_id"] for r in rows}) == len(rows)
        for row, old in zip(rows, inherited["rows"]):
            assert {key: row[key] for key in old} == old
        original_fits = {r["base"]["fit_id"]: r["base"] for r in inherited["fits"]}
        assert len(value["fits"]) == len(original_fits)
        for fit in value["fits"]:
            base = original_fits[fit["fit_id"]]
            cutoff = datetime.fromisoformat(base["cutoff_utc"])
            lower = cutoff-timedelta(days=spec["xg_strength"]["history_days"])
            identifiers = set(base["fit_match_ids"])
            history = [r for r in raw if r["league"] == league and r["accepted_exact_join"] in (True, "True")
                       and r["canonical_match_id"] in identifiers
                       and lower <= datetime.fromisoformat(r["canonical_date"]) < cutoff
                       and availability(r, delay, spec["leagues"]) <= cutoff]
            ids = [r["canonical_match_id"] for r in history]
            assert ids == fit["training_ids"] and len(ids) == len(set(ids)) > 0
            assert fit["canonical_reference_fit_ids_sha256"] == digest(base["fit_match_ids"])
            assert fit["latest_xg_available_at"] == max(availability(r, delay, spec["leagues"]) for r in history).isoformat()
            batch = [r for r in rows if r["fit_id"] == base["fit_id"]]
            assert not set(ids).intersection(r["match_id"] for r in batch)
            assert all(r["fit_cutoff_utc"] == base["cutoff_utc"] for r in batch)
            teams = sorted({r[k] for r in history for k in ("home", "away")})
            x = design(teams, [(r["home"], r["away"]) for r in history])
            target = np.asarray([float(r[k]) for r in history for k in ("home_xg", "away_xg")])
            assert np.isfinite(target).all() and (target >= 0).all()
            ages = np.asarray([(cutoff-datetime.fromisoformat(r["canonical_date"])).total_seconds()/86400 for r in history])
            future_x = design(teams, [(r["home"], r["away"]) for r in batch])
            assert set(fit["models"]) == {str(h) for h in spec["xg_strength"]["half_lives_days"]}
            for half in spec["xg_strength"]["half_lives_days"]:
                model = fit["models"][str(half)]
                assert model["teams"] == teams and model["training_ids"] == ids
                assert model["half_life"] == half and model["delay_days"] == delay
                assert model["alpha"] == spec["xg_strength"]["alpha"] and model["fit_n"] == len(history)
                coef, intercept = np.asarray(model["coef"]), model["intercept"]
                eta = x@coef+intercept
                weights = np.repeat(np.exp2(-ages/half), 2)
                weights /= weights.sum()
                residual = weights*(np.exp(eta)-target)
                gradient = np.r_[x.T@residual+spec["xg_strength"]["alpha"]*coef, residual.sum()]
                error = float(abs(gradient).max())
                assert error <= 1e-6
                MAXIMA["quasi_gradient"] = max(MAXIMA["quasi_gradient"], error)
                close(error, model["gradient_max"], label="recorded_quasi_gradient")
                objective = weights@(np.exp(eta)-target*eta)+spec["xg_strength"]["alpha"]/2*(coef@coef)
                close(objective, model["objective_value"], label="quasi_objective")
                close(1/(weights@weights)/2, model["effective_match_count"], label="effective_match_count")
                log_means = (future_x@coef+intercept).reshape(-1, 2)
                close(log_means, [r["log_xg_means"][str(half)] for r in batch], label="serialized_xg_log_means")
                CHECKS["xg_mean_models"] += 1
            for row in batch:
                assert row["xg_team_support"] == [row["home"] in teams, row["away"] in teams]
            CHECKS["xg_feature_fits"] += 1
        CHECKS["feature_rows"] += len(rows)
        combined += rows
    return ordered(combined)


def check_predictions(rows, all_features, phase, delay, configs):
    previous = read(OLD/f"{phase}.json")
    reference = {r["match_id"]: r for r in previous["predictions"]}
    assert len(rows) == len(reference) == (760 if phase == "selection" else 2280)
    assert len({r["match_id"] for r in rows}) == len(rows)
    assert set(reference) == {r["match_id"] for r in rows}
    by_id = {r["match_id"]: r for r in rows}
    source = {r["match_id"]: r for r in all_features}
    issued = read(OUT/f"{phase}_issued_delay{delay}.json")
    assert len(issued) == len(rows) and {r["match_id"] for r in issued} == set(reference)
    assert all(set(r) == {"match_id", "forecast_cutoff_utc", "fit_cutoff_utc", "probabilities"} for r in issued)
    for row in rows:
        old = reference[row["match_id"]]
        for key in old:
            if key == "probabilities":
                for name in [*REFS, "production_default_dc_auto", "dc_365", "elo_component"]:
                    assert row[key][name] == old[key][name]
            else:
                assert row[key] == old[key]
        for key in source[row["match_id"]]:
            if key != "probabilities":
                assert row[key] == source[row["match_id"]][key]
    for item in issued:
        row = by_id[item["match_id"]]
        assert item["forecast_cutoff_utc"] == row["forecast_cutoff_utc"]
        assert item["fit_cutoff_utc"] == row["fit_cutoff_utc"]
        assert set(item["probabilities"]) == set(configs)
        for name in configs:
            assert item["probabilities"][name] == row["probabilities"][name]
    fits = read(OUT/f"{phase}_fits_delay{delay}.json")
    seen = set()
    for fit in fits:
        cutoff = datetime.fromisoformat(fit["cutoff_utc"])
        training = [r for r in all_features if cutoff-timedelta(days=1460) <= datetime.fromisoformat(r["forecast_cutoff_utc"]) < cutoff
                    and datetime.fromisoformat(r["result_available_at"]) <= cutoff]
        assert [r["match_id"] for r in training] == fit["training_ids"]
        batch = [r for r in rows if r["league"] == fit["league"] and r["fit_id"] == fit["fit_id"]]
        assert batch and not seen.intersection(r["match_id"] for r in batch)
        seen.update(r["match_id"] for r in batch)
        assert fit["identity_shot_max_probability_difference"] == 0
        oldfit = next(f for f in previous["fits"] if f["league"] == fit["league"] and f["fit_id"] == fit["fit_id"])
        assert oldfit["training_ids"] == fit["training_ids"]
        assert set(fit["models"]) == set(configs)
        model_items = [(name, config, fit["models"][name]) for name, config in configs.items()]
        model_items.append((SHOT, {"mode": "shot"}, oldfit["models"][SHOT]))
        for name, config, model in model_items:
            x = np.asarray([features(r, config) for r in training])
            mean, scale = x.mean(0), x.std(0)
            scale[scale <= 1e-12] = 1.
            close(mean, model["mean"], label="training_means")
            close(scale, model["scale"], label="training_scales")
            z = np.column_stack((np.ones(len(x)), (x-mean)/scale))
            coef = np.asarray(model["weights"])
            logits = np.log(probability(training, "dc365_elo50"))+z@coef
            p = normalized(logits)
            labels = np.asarray([r["label"] for r in training], dtype=int)
            residual = p-np.eye(3)[labels]
            gradient = z.T@residual/len(training)+.1*coef
            error = float(abs(gradient).max())
            assert error <= 1e-6
            MAXIMA["offset_gradient"] = max(MAXIMA["offset_gradient"], error)
            assert model["penalty"] == .1 and model["fit_n"] == len(training)
            close(error, model["gradient_max"] if name != SHOT else model["gradient_inf_norm"], label="recorded_offset_gradient")
            objective = -np.log(p[np.arange(len(p)), labels]).mean()+.1/2*np.square(coef).sum()
            close(objective, model["objective"], label="offset_objective")
            fx = np.asarray([features(r, config) for r in batch])
            fz = np.column_stack((np.ones(len(fx)), (fx-model["mean"])/model["scale"]))
            predicted = normalized(np.log(probability(batch, "dc365_elo50"))+fz@coef)
            close(predicted, probability(batch, name), label="serialized_offset_probabilities")
            CHECKS["offset_models_including_incumbent"] += 1
    assert seen == set(reference)
    CHECKS["scored_rows"] += len(rows)


def metrics(rows, name):
    p = probability(rows, name)
    y = np.asarray([r["label"] for r in rows], dtype=int)
    assert set(y).issubset({0, 1, 2})
    truth = np.eye(3)[y]
    confidence, correct = p.max(1), p.argmax(1) == y
    bins, ece = [], 0.
    for index in range(10):
        mask = (confidence >= index/10) & ((confidence < (index+1)/10) if index < 9 else (confidence <= 1))
        count = int(mask.sum())
        if count:
            average, accuracy = float(confidence[mask].mean()), float(correct[mask].mean())
            ece += count/len(rows)*abs(average-accuracy)
            bins.append({"lower": index/10, "upper": (index+1)/10, "n": count,
                         "mean_confidence": average, "accuracy": accuracy})
    return {"n": len(rows), "log_loss": float(-np.log(p[np.arange(len(rows)), y]).mean()),
            "brier_sum_classes": float(np.square(p-truth).sum(1).mean()), "accuracy": float(correct.mean()),
            "top_label_ece_10_bins": ece, "calibration_bins": bins,
            "class_mean_probability_minus_frequency": (p.mean(0)-truth.mean(0)).tolist()}


def same(actual, expected):
    if isinstance(expected, dict):
        for key in expected:
            assert key in actual, key
            same(actual[key], expected[key])
    elif isinstance(expected, list) and expected and isinstance(expected[0], dict):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            same(a, e)
    elif isinstance(expected, (int, float, list)):
        close(actual, expected, label="scores_and_bootstrap")
    else:
        assert actual == expected


def uncertainty(rows, candidate, reference, days, spec):
    rng = np.random.default_rng(spec["uncertainty"]["seed"])
    boot = np.zeros(spec["uncertainty"]["resamples"])
    deltas, counts_by_stratum = [], {}
    for stratum in sorted({f"{r['league']}:{r['season']}" for r in rows}):
        subset = [r for r in rows if f"{r['league']}:{r['season']}" == stratum]
        start = min(date.fromisoformat(r["day"]) for r in subset)
        blocks = defaultdict(list)
        for row in subset:
            delta = -np.log(row["probabilities"][candidate][row["label"]])+np.log(row["probabilities"][reference][row["label"]])
            deltas.append(delta)
            blocks[(date.fromisoformat(row["day"])-start).days//days].append(delta)
        totals = np.asarray([sum(v) for v in blocks.values()])
        count = np.asarray([len(v) for v in blocks.values()])
        draws = rng.integers(len(blocks), size=(len(boot), len(blocks)))
        boot += len(subset)/len(rows)*(totals[draws].sum(1)/count[draws].sum(1))
        counts_by_stratum[stratum] = len(blocks)
    return {"difference_candidate_minus_reference": float(np.mean(deltas)),
            "percentile_95_interval": np.quantile(boot, [.025, .975]).tolist(),
            "bootstrap_fraction_difference_below_zero": float((boot < 0).mean()),
            "block_calendar_days": days, "blocks_by_season": counts_by_stratum,
            "resamples": len(boot), "reference": reference}


def check_summary(saved, rows, candidate, spec):
    assert saved["n"] == len(rows) and saved["population_sha256"] == digest([r["match_id"] for r in rows])
    for name in [*spec["comparators"], candidate]:
        same(saved["metrics"][name], metrics(rows, name))
    for ref in REFS:
        for days in spec["uncertainty"]["blocks_calendar_days"]:
            same(saved["paired"][ref][str(days)], uncertainty(rows, candidate, ref, days, spec))
            CHECKS["paired_bootstrap_replays"] += 1


def main():
    target = OUT/"verification.json"
    assert not target.exists(), "Preserve closed verification artifact"
    lock = read(OUT/"design_lock.json")
    for relative, expected in {**lock["sources"], **lock["inputs"], **lock["pre_fit_prerequisites"]}.items():
        assert sha(ROOT/relative) == expected, relative
        BOUND[relative] = expected
        CHECKS["locked_hashes"] += 1
    spec = read(SOURCE/"specification.json")
    for relative in lock["pre_fit_prerequisites"]:
        prerequisite = read(ROOT/relative)
        assert prerequisite["status"] == "passed" and prerequisite["sources"] == lock["sources"]
    contract = read(SOURCE/"execution/execution_contract.json")
    assert contract["parent_spec_sha256"] == sha(SOURCE/"specification.json")
    data = read(ROOT/"artifacts/research/boundary_20260908/football_xg_data/availability_contract.json")
    raw = []
    for relative in data["joined_files"]:
        with (ROOT/relative).open(newline="") as stream:
            raw += list(csv.DictReader(stream))
    assert len(raw) == 10260 and len({r["canonical_match_id"] for r in raw}) == 10260
    assert sum(r["accepted_exact_join"] == "True" for r in raw) == 10249
    caches = {}
    for path in sorted(OUT.glob("features_*_E0_delay*.json")):
        _, phase, _, suffix = path.stem.split("_")
        delay = int(suffix.removeprefix("delay"))
        caches[(phase, delay)] = check_feature_cache(phase, delay, raw, spec, lock)
    primary = read(OUT/"selection_7day.json")
    selection_lock = read(OUT/"selection_lock.json")
    assert selection_lock["sources"] == lock["sources"]
    assert selection_lock["selection_7day_sha256"] == sha(OUT/"selection_7day.json")
    check_predictions(primary["predictions"], caches[("selection", 7)], "selection", 7, spec["candidates"])
    for name in [*spec["comparators"], *spec["candidates"]]:
        same(primary["metrics"][name], metrics(primary["predictions"], name))
    selected = min(spec["candidates"], key=lambda name: (primary["metrics"][name]["log_loss"], name))
    assert primary["selected"] == selection_lock["selected"] == selected
    check_summary(primary["summary"], primary["predictions"], selected, spec)
    gains = {ref: 1-primary["metrics"][selected]["log_loss"]/primary["metrics"][ref]["log_loss"] for ref in REFS}
    same(primary["relative_gain"], gains)
    passed7 = all(gain >= .005 for gain in gains.values())
    assert primary["screen_passed"] is passed7 and selection_lock["seven_day_passed"] is passed7
    passed14 = False
    if passed7:
        sensitivity = read(OUT/"selection_14day.json")
        assert selection_lock["selection_14day_sha256"] == sha(OUT/"selection_14day.json")
        assert sensitivity["selected"] == selected
        check_predictions(sensitivity["predictions"], caches[("selection", 14)], "selection", 14, {selected: spec["candidates"][selected]})
        check_summary(sensitivity["summary"], sensitivity["predictions"], selected, spec)
        sm = sensitivity["summary"]["metrics"]
        passed14 = all(sm[selected]["log_loss"] < sm[ref]["log_loss"] for ref in REFS)
        assert sensitivity["screen_passed"] is passed14
    else:
        assert not (OUT/"selection_14day.json").exists() and ("selection", 14) not in caches
        assert selection_lock["selection_14day_sha256"] is None
    assert selection_lock["fourteen_day_evaluated"] is passed7
    assert selection_lock["fourteen_day_passed"] is passed14
    advanced = passed7 and passed14
    assert selection_lock["advancement_passed"] is advanced
    evaluated = (OUT/"evaluation.json").exists()
    if not advanced:
        assert not evaluated and not (OUT/"evaluation_attempt.json").exists()
        assert not any(phase == "evaluation" for phase, delay in caches)
    if evaluated:
        evaluation = read(OUT/"evaluation.json")
        assert evaluation["selected"] == selected and evaluation["promotion"] is False and evaluation["prospective_validation"] is False
        for delay in (7, 14):
            rows = evaluation["predictions"][str(delay)]
            check_predictions(rows, ordered(caches[("selection", delay)]+caches[("evaluation", delay)]), "evaluation", delay, {selected: spec["candidates"][selected]})
            summaries = evaluation["summaries"][str(delay)]
            check_summary(summaries["pooled"], rows, selected, spec)
            assert set(summaries["by_league"]) == set(spec["leagues"])
            assert set(summaries["by_pooled_season"]) == {"2024", "2025"}
            for league, saved in summaries["by_league"].items():
                check_summary(saved, [r for r in rows if r["league"] == league], selected, spec)
            for season, saved in summaries["by_pooled_season"].items():
                check_summary(saved, [r for r in rows if r["season"] == int(season)], selected, spec)
            check_summary(summaries["resumed_fixture_excluded_diagnostic"], [r for r in rows if r["match_id"] != "I1:2024:Fiorentina:Inter"], selected, spec)
        p, s = [evaluation["summaries"][str(delay)] for delay in (7, 14)]
        m = p["pooled"]["metrics"]
        gates = {
            "pooled_two_percent_and_negative_28day_ci": all(m[selected]["log_loss"] <= .98*m[ref]["log_loss"] and p["pooled"]["paired"][ref]["28"]["percentile_95_interval"][1] < 0 for ref in REFS),
            "each_league_improves_shot": all(v["metrics"][selected]["log_loss"] < v["metrics"][SHOT]["log_loss"] for v in p["by_league"].values()),
            "each_pooled_season_improves_shot": all(v["metrics"][selected]["log_loss"] < v["metrics"][SHOT]["log_loss"] for v in p["by_pooled_season"].values()),
            "brier_not_worse": m[selected]["brier_sum_classes"] <= m[SHOT]["brier_sum_classes"],
            "ece_increase_at_most_one_point": m[selected]["top_label_ece_10_bins"] <= m[SHOT]["top_label_ece_10_bins"]+.01,
            "fourteen_day_pooled_improves_all_references": all(s["pooled"]["metrics"][selected]["log_loss"] < s["pooled"]["metrics"][ref]["log_loss"] for ref in REFS),
            "fourteen_day_each_league_improves_shot": all(v["metrics"][selected]["log_loss"] < v["metrics"][SHOT]["log_loss"] for v in s["by_league"].values())}
        assert evaluation["gates"] == gates and evaluation["substantial_gate_passed"] is all(gates.values())
    for relative, expected in BOUND.items():
        assert sha(ROOT/relative) == expected, f"Input changed during verification: {relative}"
    output = {"status": "passed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "verifier_sha256": sha(Path(__file__)), "checks": dict(CHECKS), "maximum_errors": dict(MAXIMA),
              "selected": selected, "selection_advancement_passed": advanced, "transfer_evaluated": evaluated,
              "fits_performed_by_verifier": 0, "new_candidates": 0, "bindings": BOUND,
              "scope": "Independent raw eligibility, mean-model quasi-score and coefficient replay; offset training, gradient and probabilities; matched population/incumbents; metrics, historical block bootstrap and frozen gates. No prospective or original publication-time claim."}
    with target.open("x") as stream:
        json.dump(output, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: output[key] for key in ("status", "checks", "maximum_errors", "selection_advancement_passed", "transfer_evaluated")}, indent=2))


if __name__ == "__main__":
    main()
