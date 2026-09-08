"""Six frozen football challengers; source/data inputs remain read-only."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timedelta
import json
import platform
from pathlib import Path
from zoneinfo import ZoneInfo
from threadpoolctl import threadpool_limits
from research.experiments.performance_20260907.football import benchmark as b
from research.experiments.performance_20260907.football import transfer_run as transfer
from research.experiments.boundary_20260908.football import models

LANE = Path(__file__).resolve().parent
OUT = b.ROOT / "artifacts/research/boundary_20260908/football"
SPEC_PATH = LANE / "specification.json"


def specification():
    spec = json.loads(SPEC_PATH.read_text())
    lock = json.loads((OUT / "design_lock.json").read_text())
    assert lock["spec_sha256"] == b.file_sha(SPEC_PATH)
    return spec


def sources():
    files = [*sorted(LANE.glob("*.py")), SPEC_PATH,
             b.LANE / "benchmark.py", b.LANE / "transfer_run.py", *sorted((b.ROOT / "packages/football/mrp").glob("*.py"))]
    return {str(p.relative_to(b.ROOT)): b.file_sha(p) for p in files}


@contextmanager
def league_context(timezone):
    old_zone, old_dc = b.LONDON, b.DC
    b.LONDON, b.DC = ZoneInfo(timezone), {"dc_equal": None, "dc_365": 365., "dc_180": 180.}
    try:
        yield
    finally:
        b.LONDON, b.DC = old_zone, old_dc


def load_league(league, years, spec):
    parent = json.loads((b.LANE / "spec.json").read_text())
    if league == "E0":
        rows, populations = b.load_rows(years, parent)
        pattern = lambda year: f"E0_{year}_{year+1}.csv"
    else:
        transfer_spec = json.loads(transfer.SPEC_PATH.read_text())
        transfer_spec["input_season_start_years"] = years
        rows, populations = transfer.load_league(league, transfer_spec)
        pattern = lambda year: f"transfer_{league}_{year}_{year+1}.csv"
    stats, hashes = {}, {}
    for year in years:
        path = b.DATA / pattern(year)
        hashes[str(path.relative_to(b.ROOT))] = b.file_sha(path)
        with path.open(encoding="utf-8-sig", newline="") as f:
            for raw in csv.DictReader(f):
                if not raw.get("HomeTeam"):
                    continue
                mid = f"{league}:{year}:{raw['HomeTeam'].strip()}:{raw['AwayTeam'].strip()}"
                stats[mid] = [float(raw[key]) for key in spec["shot_strength_model"]["statistics"]]
    assert set(stats) == {r.match.match_id for r in rows}
    return rows, stats, populations, hashes


def build_features(phase, spec):
    cutoff_year = 2023 if phase == "selection" else 2025
    scored_years = list(range(2019, 2024)) if phase == "selection" else [2024, 2025]
    all_rows, provenance = [], {}
    for league, timezone in spec["leagues"].items():
        path = OUT / f"features_{phase}_{league}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            assert cached["source_sha256"] == b.digest(sources()), "Feature cache code changed"
            for name, sha in cached["input_hashes"].items():
                assert b.file_sha(b.ROOT/name) == sha
            all_rows.extend(cached["rows"])
            provenance[league] = b.file_sha(path)
            continue
        with league_context(timezone), threadpool_limits(limits=1):
            rows, stats, populations, input_hashes = load_league(league, list(range(2017, cutoff_year+1)), spec)
            by_id = {r.match.match_id: r for r in rows}
            parent_spec = json.loads((b.LANE / "spec.json").read_text())
            forecasts, base_fits = b.predict_phase(rows, scored_years, parent_spec)
            fits = []
            for fit in base_fits:
                batch = [r for r in forecasts if r["fit_id"] == fit["fit_id"]]
                matches = [by_id[mid].match for mid in fit["fit_match_ids"]]
                cutoff = datetime.fromisoformat(fit["cutoff_utc"])
                assert all(b.match_available_at(m) <= cutoff for m in matches)
                default, calibrator, groups = models.production_default_fit(matches)
                fixture_pairs = [(r["home"], r["away"]) for r in batch]
                shot_predictions, shot_diagnostics = {}, {}
                for half in spec["shot_strength_model"]["half_life_days"]:
                    strength = models.fit_shot_strength(matches, stats, cutoff, half, spec["shot_strength_model"])
                    shot_predictions[str(half)] = strength.predict(fixture_pairs)
                    shot_diagnostics[str(half)] = strength.diagnostics
                for i, row in enumerate(batch):
                    row["league"] = league
                    row["fit_cutoff_utc"] = fit["cutoff_utc"]
                    row["result_available_at"] = b.match_available_at(by_id[row["match_id"]].match).isoformat()
                    row["probabilities"].pop("closing_market", None)
                    row["probabilities"]["production_default_dc_auto"] = list(models.production_default_predict(default, calibrator, *fixture_pairs[i]))
                    row["log_shot_means"] = {half: matrix[i].tolist() for half, matrix in shot_predictions.items()}
                metadata = groups.metadata()
                fits.append({"base": fit, "shot_strength": shot_diagnostics,
                    "production_default": {"model_diagnostics": default.diagnostics, "calibration": calibrator.method,
                        "populations": metadata}})
                print(json.dumps({"feature_block": fit["fit_id"], "league": league, "phase": phase, "rows": len(batch)}), flush=True)
        b.write_json(path, {"source_sha256": b.digest(sources()), "spec_sha256": b.file_sha(SPEC_PATH),
            "source_files": sources(), "input_hashes": input_hashes, "populations": populations,
            "rows": forecasts, "fits": fits})
        all_rows.extend(forecasts)
        provenance[league] = b.file_sha(path)
    return sorted(all_rows, key=lambda r: (r["forecast_cutoff_utc"], r["match_id"])), provenance


def configurations(spec):
    return {f"{mechanism}_ridge{penalty:g}": (mechanism, penalty)
            for mechanism in spec["mechanisms"] for penalty in spec["ridge_penalties"]}


def prior_training_rows(rows, cutoff):
    lower = cutoff-timedelta(days=1460)
    return [r for r in rows if lower <= datetime.fromisoformat(r["forecast_cutoff_utc"]) < cutoff
            and datetime.fromisoformat(r["result_available_at"]) <= cutoff]


def predict_corrections(feature_rows, scoring_rows, config, phase):
    output, fits = [], []
    keys = sorted({(r["league"], r["fit_id"], r["fit_cutoff_utc"]) for r in scoring_rows}, key=lambda x: (x[2], x[0]))
    for league, fit_id, date_string in keys:
        batch = [r for r in scoring_rows if r["league"] == league and r["fit_id"] == fit_id]
        training = prior_training_rows(feature_rows, datetime.fromisoformat(date_string))
        ids = [r["match_id"] for r in training]
        assert not set(ids).intersection(r["match_id"] for r in batch)
        local = [{**r, "probabilities": dict(r["probabilities"])} for r in batch]
        fitted = {}
        for name, (mechanism, penalty) in config.items():
            model = models.fit_offset(training, mechanism, penalty)
            probabilities = model.predict(batch)
            b.validate_probabilities(probabilities)
            for row, p in zip(local, probabilities):
                row["probabilities"][name] = p.tolist()
            fitted[name] = model.diagnostics
        fits.append({"league": league, "fit_id": fit_id, "cutoff": date_string,
                     "training_ids": ids, "training_ids_sha256": b.digest(ids),
                     "latest_training_result_available_at": max(r["result_available_at"] for r in training),
                     "models": fitted})
        output.extend(local)
        print(json.dumps({"correction_block": fit_id, "league": league, "phase": phase, "fit_n": len(training), "models": list(config)}), flush=True)
    return sorted(output, key=lambda r: (r["forecast_cutoff_utc"], r["match_id"])), fits


def uncertainty(rows, candidate, reference, block_days, spec):
    adjusted = [{**r, "season": f"{r['league']}:{r['season']}",
                 "probabilities": {**r["probabilities"], "dc_equal": r["probabilities"][reference]}} for r in rows]
    u_spec = {"uncertainty": {"resamples": spec["uncertainty"]["resamples"], "seed": spec["uncertainty"]["seed"]}}
    result = b.paired_uncertainty(adjusted, candidate, u_spec, block_days)
    result["difference_candidate_minus_reference"] = result.pop("difference_candidate_minus_dc_equal")
    result["reference"] = reference
    return result


def summarize(rows, candidate, spec):
    names = list(dict.fromkeys([*spec["comparators"], candidate]))
    metrics = {name: b.metrics(rows, name) for name in names}
    result = {"n": len(rows), "population_sha256": b.digest([r["match_id"] for r in rows]), "models": metrics,
              "paired_against_references": {ref: {str(days): uncertainty(rows, candidate, ref, days, spec)
                  for days in spec["uncertainty"]["blocks_calendar_days"]}
                  for ref in ["production_default_dc_auto", *spec["required_research_references"]]}}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["selection", "evaluation"], required=True)
    args = parser.parse_args()
    spec = specification()
    result_path = OUT / f"{args.phase}.json"
    if result_path.exists():
        raise ValueError("A completed phase is frozen; do not overwrite it")
    attempt = OUT / f"{args.phase}_attempt.json"
    if not attempt.exists():
        b.write_json(attempt, {"started_at_utc": b.now(), "spec_sha256": b.file_sha(SPEC_PATH),
            "source_files": sources(), "source_sha256": b.digest(sources()),
            "runtime": {"python": platform.python_version(), "numpy": b.np.__version__,
                        "scipy": b.scipy.__version__, "sklearn": b.sklearn.__version__, "threads": 1}})
    else:
        assert json.loads(attempt.read_text())["source_sha256"] == b.digest(sources())
    if args.phase == "evaluation":
        lock = json.loads((OUT / "selection_lock.json").read_text())
        assert lock["source_sha256"] == b.digest(sources())
        assert lock["spec_sha256"] == b.file_sha(SPEC_PATH)
        assert lock["selection_artifact_sha256"] == b.file_sha(OUT / "selection.json")
        selection_rows = []
        for league in spec["leagues"]:
            cache = OUT / f"features_selection_{league}.json"
            assert b.file_sha(cache) == lock["features_sha256"][league]
            selection_rows.extend(json.loads(cache.read_text())["rows"])
        evaluation_rows, provenance = build_features("evaluation", spec)
        feature_rows = sorted(selection_rows+evaluation_rows, key=lambda r: (r["forecast_cutoff_utc"], r["match_id"]))
        candidate = lock["selected_model"]
        if candidate == "dc365_elo50":
            predictions, fits = evaluation_rows, []
        else:
            with threadpool_limits(limits=1):
                predictions, fits = predict_corrections(feature_rows, evaluation_rows,
                    {candidate: configurations(spec)[candidate]}, "evaluation")
        pooled = summarize(predictions, candidate, spec)
        leagues = {league: summarize([r for r in predictions if r["league"] == league], candidate, spec) for league in spec["leagues"]}
        seasons = {f"{league}:{year}": summarize([r for r in predictions if r["league"] == league and r["season"] == year], candidate, spec)
                   for league in spec["leagues"] for year in spec["evaluation_seasons"]}
        substantial = all(pooled["models"][candidate]["log_loss"] <= .98*pooled["models"][ref]["log_loss"]
                          and pooled["paired_against_references"][ref]["28"]["percentile_95_interval"][1] < 0
                          for ref in spec["required_research_references"])
        substantial &= pooled["models"][candidate]["log_loss"] < pooled["models"]["production_default_dc_auto"]["log_loss"]
        summary = {"pooled": pooled, "by_league": leagues, "by_league_season": seasons,
                   "substantial_gain_target_met": bool(substantial)}
    else:
        feature_rows, provenance = build_features("selection", spec)
        scoring_rows = [r for r in feature_rows if r["league"] == spec["selection"]["league"] and r["season"] in spec["selection"]["seasons"]]
        with threadpool_limits(limits=1):
            predictions, fits = predict_corrections(feature_rows, scoring_rows, configurations(spec), "selection")
        names = ["dc365_elo50", *configurations(spec)]
        selection_metrics = {name: b.metrics(predictions, name) for name in names}
        candidate = min(names, key=lambda name: (selection_metrics[name]["log_loss"], name))
        summary = {"selection_metrics": selection_metrics}
    b.write_json(result_path, {"phase": args.phase, "completed_at_utc": b.now(), "selected_model": candidate,
        "status": "retrospective_reused_dates_no_promotion", "specification": spec, "spec_sha256": b.file_sha(SPEC_PATH),
        "source_files": sources(), "source_sha256": b.digest(sources()), "feature_artifacts_sha256": provenance,
        "summary": summary, "predictions": predictions, "fits": fits})
    if args.phase == "selection":
        b.write_json(OUT / "selection_lock.json", {"selected_model": candidate, "selected_at_utc": b.now(),
            "spec_sha256": b.file_sha(SPEC_PATH), "source_sha256": b.digest(sources()), "source_files": sources(),
            "selection_artifact_sha256": b.file_sha(result_path), "features_sha256": provenance,
            "all_six_candidates": list(configurations(spec)), "selection_metrics": summary["selection_metrics"],
            "new_candidate_evaluation_metrics_not_yet_computed": True,
            "previous_test_date_exposure": spec["known_exposure"]})
    print(json.dumps({"phase": args.phase, "selected_model": candidate, "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
