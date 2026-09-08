"""Predeclared delayed-xG experiment on the frozen pre-match football population."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
import platform

import numpy as np
from threadpoolctl import threadpool_limits

from research.experiments.boundary_20260908.football import run as previous
from research.experiments.boundary_20260908.football import models as previous_models
from research.experiments.boundary_20260908.football_xg.execution import models

b = previous.b
LANE = Path(__file__).resolve().parent
OUT = b.ROOT / "artifacts/research/boundary_20260908/football_xg/execution"
OLD = previous.OUT
SPEC = LANE.parent / "specification.json"
CONTRACT = LANE / "execution_contract.json"
DATA_CONTRACT = b.ROOT / "artifacts/research/boundary_20260908/football_xg_data/availability_contract.json"
SHOT = "shot_strength_90d_ridge0.1"
REFS = ["dc365_elo50", "dc_180", SHOT]


def write_new(path, value):
    if path.exists():
        raise ValueError(f"Refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    b.write_json(path, value)


def sources():
    paths = [*sorted(LANE.glob("*.py")), SPEC, CONTRACT, LANE.parent / "models.py",
             DATA_CONTRACT, b.ROOT / "research/experiments/boundary_20260908/football_xg_data/availability.py"]
    result = {str(p.relative_to(b.ROOT)): b.file_sha(p) for p in paths}
    result.update(previous.sources())
    return result


def input_files():
    paths = [OLD / "selection.json", OLD / "selection_lock.json", OLD / "evaluation.json"]
    data = json.loads(DATA_CONTRACT.read_text())
    paths += [b.ROOT / p for p in data["joined_files"]]
    for phase in ["selection", "evaluation"]:
        for league in ["E0", "SP1", "I1"]:
            cache = OLD / f"features_{phase}_{league}.json"
            paths.append(cache)
            payload = json.loads(cache.read_text())
            paths += [b.ROOT / p for p in payload["input_hashes"]]
    paths += [b.ROOT / p for p in data["artifact_manifest"]]
    return {str(p.relative_to(b.ROOT)): b.file_sha(p) for p in sorted(set(paths))}


def verify_design():
    lock = json.loads((OUT / "design_lock.json").read_text())
    assert lock["sources"] == sources(), "Execution source changed after design lock"
    for path, sha in lock["inputs"].items():
        assert b.file_sha(b.ROOT / path) == sha, path
    for path, sha in lock["pre_fit_prerequisites"].items():
        assert b.file_sha(b.ROOT / path) == sha, path
    assert json.loads(CONTRACT.read_text())["parent_spec_sha256"] == b.file_sha(SPEC)
    for path, sha in json.loads(DATA_CONTRACT.read_text())["artifact_manifest"].items():
        assert b.file_sha(b.ROOT / path) == sha, path
    return lock


def freeze():
    prerequisites = {}
    for name in ["pre_fit_tests.json", "independent_pre_fit_review.json"]:
        path = OUT / name
        record = json.loads(path.read_text())
        assert record["status"] == "passed" and record["sources"] == sources(), name
        prerequisites[str(path.relative_to(b.ROOT))] = b.file_sha(path)
    write_new(OUT / "design_lock.json", {
        "locked_at_utc": b.now(), "sources": sources(), "inputs": input_files(),
        "pre_fit_prerequisites": prerequisites,
        "real_xg_model_fitted_before_lock": False, "new_candidate_scores_before_lock": False,
        "parent_spec_sha256": b.file_sha(SPEC), "runtime": {"python": platform.python_version(),
        "numpy": np.__version__, "sklearn": b.sklearn.__version__, "threads": 1}})


def joined_history():
    records = []
    for relative in json.loads(DATA_CONTRACT.read_text())["joined_files"]:
        with (b.ROOT / relative).open(newline="") as stream:
            records.extend(csv.DictReader(stream))
    assert len(records) == 10260
    ids = [r["canonical_match_id"] for r in records]
    assert len(set(ids)) == len(ids)
    return records


def order(rows):
    return sorted(rows, key=lambda r: (r["forecast_cutoff_utc"], r["match_id"]))


def feature_cache(phase, delay, spec):
    """Every xG strength uses only earlier available measurements, never match targets."""
    output, provenance = [], {}
    raw = joined_history()
    for league in spec["leagues"]:
        target = OUT / f"features_{phase}_{league}_delay{delay}.json"
        if target.exists():
            payload = json.loads(target.read_text())
            assert payload["sources"] == sources()
            assert payload["design_lock_sha256"] == b.file_sha(OUT / "design_lock.json")
            output.extend(payload["rows"])
            provenance[str(target.relative_to(b.ROOT))] = b.file_sha(target)
            continue
        inherited_path = OLD / f"features_{phase}_{league}.json"
        inherited = json.loads(inherited_path.read_text())
        assert inherited["source_files"] == previous.sources()
        for path, sha in inherited["input_hashes"].items():
            assert b.file_sha(b.ROOT / path) == sha
        rows = copy.deepcopy(inherited["rows"])
        fits = []
        for bundle in inherited["fits"]:
            base = bundle["base"]
            cutoff = datetime.fromisoformat(base["cutoff_utc"])
            batch = [r for r in rows if r["fit_id"] == base["fit_id"]]
            assert batch and all(r["fit_cutoff_utc"] == base["cutoff_utc"] for r in batch)
            canonical_ids = set(base["fit_match_ids"])
            eligible = models.eligible_history(raw, league, cutoff, delay, spec["xg_strength"]["history_days"])
            history = [r for r in eligible if r["canonical_match_id"] in canonical_ids]
            assert history and all(models.available_at(r, delay) <= cutoff for r in history)
            assert not {r["canonical_match_id"] for r in history}.intersection(r["match_id"] for r in batch)
            fixtures = [(r["home"], r["away"]) for r in batch]
            diagnostics = {}
            for half in spec["xg_strength"]["half_lives_days"]:
                strength = models.fit_strength(history, cutoff, half, {**spec["xg_strength"], "delay_days": delay})
                means = strength.predict(fixtures)
                diagnostics[str(half)] = strength.diagnostics
                known = set(strength.teams)
                for r, value in zip(batch, means):
                    r.setdefault("log_xg_means", {})[str(half)] = value.tolist()
                    r["xg_team_support"] = [r["home"] in known, r["away"] in known]
            fits.append({"league": league, "fit_id": base["fit_id"], "cutoff_utc": base["cutoff_utc"],
                         "delay_days": delay, "training_ids": [r["canonical_match_id"] for r in history],
                         "latest_xg_available_at": max(models.available_at(r, delay).isoformat() for r in history),
                         "canonical_reference_fit_ids_sha256": b.digest(base["fit_match_ids"]),
                         "models": diagnostics})
        write_new(target, {"sources": sources(), "design_lock_sha256": b.file_sha(OUT / "design_lock.json"),
            "inherited_feature_cache_sha256": b.file_sha(inherited_path), "delay_days": delay,
            "phase": phase, "league": league, "rows": rows, "fits": fits})
        output.extend(rows)
        provenance[str(target.relative_to(b.ROOT))] = b.file_sha(target)
        print(json.dumps({"feature_phase": phase, "delay": delay, "league": league, "rows": len(rows), "fits": len(fits)}), flush=True)
    return order(output), provenance


def forecast_input(row):
    return {k: copy.deepcopy(v) for k, v in row.items() if k not in {"label", "goals", "result_available_at"}}


def predict(feature_rows, phase, configs, delay):
    reference_payload = json.loads((OLD / f"{phase}.json").read_text())
    references = {r["match_id"]: r for r in reference_payload["predictions"]}
    scoring = [r for r in feature_rows if r["match_id"] in references]
    assert len(scoring) == (760 if phase == "selection" else 2280)
    assert set(references) == {r["match_id"] for r in scoring}
    output, fits, issued = [], [], []
    keys = sorted({(r["league"], r["fit_id"], r["fit_cutoff_utc"]) for r in scoring}, key=lambda x: (x[2], x[0]))
    for league, fit_id, instant in keys:
        batch = [r for r in scoring if r["league"] == league and r["fit_id"] == fit_id]
        training = previous.prior_training_rows(feature_rows, datetime.fromisoformat(instant))
        assert training and not {r["match_id"] for r in training}.intersection(r["match_id"] for r in batch)
        inputs = [forecast_input(r) for r in batch]
        fitted, probabilities = {}, {}
        # This verifies that the prior strong reference is an exact matched model.
        identity = previous_models.fit_offset(training, "shot_strength_90d", .1)
        identity_p = identity.predict(inputs)
        reference_p = np.asarray([references[r["match_id"]]["probabilities"][SHOT] for r in batch])
        assert np.array_equal(identity_p, reference_p), "Prior shot identity ablation changed"
        for name, config in configs.items():
            model = models.fit_offset(training, config, .1)
            probabilities[name] = model.predict(inputs)
            b.validate_probabilities(probabilities[name])
            fitted[name] = model.diagnostics
        for index, row in enumerate(batch):
            ref = references[row["match_id"]]
            assert all(row[k] == ref[k] for k in ["day", "league", "season", "home", "away", "label", "goals", "forecast_cutoff_utc", "fit_cutoff_utc"])
            assert all(row["probabilities"][k] == ref["probabilities"][k] for k in row["probabilities"])
            point = {"match_id": row["match_id"], "forecast_cutoff_utc": row["forecast_cutoff_utc"],
                     "fit_cutoff_utc": instant, "probabilities": {name: values[index].tolist() for name, values in probabilities.items()}}
            issued.append(point)
            merged = copy.deepcopy(row)
            merged["probabilities"][SHOT] = ref["probabilities"][SHOT]
            merged["probabilities"].update(point["probabilities"])
            output.append(merged)
        fits.append({"league": league, "fit_id": fit_id, "cutoff_utc": instant, "delay_days": delay,
                     "training_ids": [r["match_id"] for r in training], "models": fitted,
                     "identity_shot_max_probability_difference": float(np.max(np.abs(identity_p-reference_p)))})
    # Scores are only computed after the target-free issued forecast artifact is closed.
    write_new(OUT / f"{phase}_issued_delay{delay}.json", issued)
    write_new(OUT / f"{phase}_fits_delay{delay}.json", fits)
    return order(output)


def summary(rows, candidate, spec):
    names = [*spec["comparators"], candidate]
    return {"n": len(rows), "population_sha256": b.digest([r["match_id"] for r in rows]),
            "metrics": {name: b.metrics(rows, name) for name in names},
            "paired": {ref: {str(days): previous.uncertainty(rows, candidate, ref, days, spec)
                       for days in spec["uncertainty"]["blocks_calendar_days"]} for ref in REFS}}


def selection():
    verify_design()
    if (OUT / "selection_attempt.json").exists():
        raise ValueError("Selection already attempted; preserve the frozen attempt")
    write_new(OUT / "selection_attempt.json", {"started_at_utc": b.now(), "sources": sources()})
    spec = json.loads(SPEC.read_text())
    features, provenance = feature_cache("selection", 7, spec)
    rows = predict(features, "selection", spec["candidates"], 7)
    metrics = {name: b.metrics(rows, name) for name in [*spec["comparators"], *spec["candidates"]]}
    selected = min(spec["candidates"], key=lambda k: (metrics[k]["log_loss"], k))
    gain = {ref: 1-metrics[selected]["log_loss"]/metrics[ref]["log_loss"] for ref in REFS}
    passed7 = all(v >= .005 for v in gain.values())
    write_new(OUT / "selection_7day.json", {"selected": selected, "metrics": metrics, "relative_gain": gain,
        "screen_passed": passed7, "predictions": rows, "features": provenance, "summary": summary(rows, selected, spec)})
    sensitivity = None
    passed14 = False
    if passed7:
        delayed, delayed_provenance = feature_cache("selection", 14, spec)
        sensitivity_rows = predict(delayed, "selection", {selected: spec["candidates"][selected]}, 14)
        sensitivity = summary(sensitivity_rows, selected, spec)
        passed14 = all(sensitivity["metrics"][selected]["log_loss"] < sensitivity["metrics"][ref]["log_loss"] for ref in REFS)
        write_new(OUT / "selection_14day.json", {"selected": selected, "predictions": sensitivity_rows,
            "features": delayed_provenance, "summary": sensitivity, "screen_passed": passed14})
    advanced = bool(passed7 and passed14)
    write_new(OUT / "selection_lock.json", {"selected": selected, "locked_at_utc": b.now(),
        "sources": sources(), "advancement_passed": advanced, "seven_day_passed": passed7,
        "fourteen_day_evaluated": passed7, "fourteen_day_passed": passed14,
        "selection_7day_sha256": b.file_sha(OUT / "selection_7day.json"),
        "selection_14day_sha256": b.file_sha(OUT / "selection_14day.json") if passed7 else None})
    print(json.dumps({"selected": selected, "metrics": {k: v["log_loss"] for k,v in metrics.items()},
                      "relative_gain": gain, "advancement_passed": advanced}, indent=2), flush=True)


def transfer_gate(pooled, by_league, by_season, sensitivity, sensitivity_leagues, candidate):
    metrics = pooled["metrics"]
    return {
        "pooled_two_percent_and_negative_28day_ci": all(
            metrics[candidate]["log_loss"] <= .98*metrics[ref]["log_loss"] and
            pooled["paired"][ref]["28"]["percentile_95_interval"][1] < 0 for ref in REFS),
        "each_league_improves_shot": all(s["metrics"][candidate]["log_loss"] < s["metrics"][SHOT]["log_loss"] for s in by_league.values()),
        "each_pooled_season_improves_shot": all(s["metrics"][candidate]["log_loss"] < s["metrics"][SHOT]["log_loss"] for s in by_season.values()),
        "brier_not_worse": metrics[candidate]["brier_sum_classes"] <= metrics[SHOT]["brier_sum_classes"],
        "ece_increase_at_most_one_point": metrics[candidate]["top_label_ece_10_bins"] <= metrics[SHOT]["top_label_ece_10_bins"]+.01,
        "fourteen_day_pooled_improves_all_references": all(sensitivity["metrics"][candidate]["log_loss"] < sensitivity["metrics"][ref]["log_loss"] for ref in REFS),
        "fourteen_day_each_league_improves_shot": all(s["metrics"][candidate]["log_loss"] < s["metrics"][SHOT]["log_loss"] for s in sensitivity_leagues.values())}


def evaluate():
    verify_design()
    lock = json.loads((OUT / "selection_lock.json").read_text())
    assert lock["sources"] == sources() and lock["advancement_passed"], "Selection failed; transfer is forbidden"
    for delay in [7, 14]:
        assert b.file_sha(OUT / f"selection_{delay}day.json") == lock[f"selection_{delay}day_sha256"]
    write_new(OUT / "evaluation_attempt.json", {"started_at_utc": b.now(), "selection_lock_sha256": b.file_sha(OUT / "selection_lock.json")})
    spec, candidate = json.loads(SPEC.read_text()), lock["selected"]
    all_rows, all_summaries, all_provenance = {}, {}, {}
    for delay in [7, 14]:
        earlier, _ = feature_cache("selection", delay, spec)
        later, provenance = feature_cache("evaluation", delay, spec)
        rows = predict(order(earlier+later), "evaluation", {candidate: spec["candidates"][candidate]}, delay)
        all_rows[str(delay)] = rows
        all_provenance[str(delay)] = provenance
        all_summaries[str(delay)] = {"pooled": summary(rows, candidate, spec),
            "by_league": {league: summary([r for r in rows if r["league"] == league], candidate, spec) for league in spec["leagues"]},
            "by_pooled_season": {str(year): summary([r for r in rows if r["season"] == year], candidate, spec) for year in [2024,2025]},
            "resumed_fixture_excluded_diagnostic": summary([r for r in rows if r["match_id"] != "I1:2024:Fiorentina:Inter"], candidate, spec)}
    primary, sensitivity = all_summaries["7"], all_summaries["14"]
    gates = transfer_gate(primary["pooled"], primary["by_league"], primary["by_pooled_season"], sensitivity["pooled"], sensitivity["by_league"], candidate)
    write_new(OUT / "evaluation.json", {"selected": candidate, "predictions": all_rows, "summaries": all_summaries,
        "features": all_provenance, "gates": gates, "substantial_gate_passed": all(gates.values()),
        "promotion": False, "prospective_validation": False, "sources": sources()})
    print(json.dumps({"selected": candidate, "gates": gates, "substantial_gate_passed": all(gates.values())}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "selection", "evaluation"])
    phase = parser.parse_args().phase
    with threadpool_limits(limits=1):
        {"freeze": freeze, "selection": selection, "evaluation": evaluate}[phase]()


if __name__ == "__main__":
    main()
