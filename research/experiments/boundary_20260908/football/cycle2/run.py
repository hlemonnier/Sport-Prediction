"""Immutable-data, three-candidate nonlinear residual experiment."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from threadpoolctl import threadpool_limits
from research.experiments.boundary_20260908.football import run as parent
from research.experiments.boundary_20260908.football.cycle2 import model
from research.experiments.performance_20260907.football import benchmark as b

LANE = Path(__file__).resolve().parent
OUT = parent.OUT / "cycle2"
SPEC = LANE / "specification.json"


def specification():
    spec = json.loads(SPEC.read_text())
    assert b.file_sha(SPEC) == json.loads((OUT/"design_lock.json").read_text())["spec_sha256"]
    return spec


def sources():
    files = {str(p.relative_to(b.ROOT)): b.file_sha(p) for p in [*sorted(LANE.glob("*.py")), SPEC]}
    return {**parent.sources(), **files}


def inputs(phase, spec):
    parent_artifacts = {}
    for name in ("selection", "evaluation"):
        path = parent.OUT/f"{name}.json"
        assert b.file_sha(path) == spec[f"parent_{name}_artifact_sha256"]
        parent_artifacts[name] = json.loads(path.read_text())
    features, hashes = [], {}
    for name in (("selection",) if phase == "selection" else ("selection", "evaluation")):
        for league in spec["leagues"]:
            path = parent.OUT/f"features_{name}_{league}.json"
            assert b.file_sha(path) == parent_artifacts[name]["feature_artifacts_sha256"][league]
            cache = json.loads(path.read_text())
            for filepath, digest in cache["input_hashes"].items():
                assert b.file_sha(b.ROOT/filepath) == digest
            hashes[str(path.relative_to(b.ROOT))] = b.file_sha(path)
            features.extend(cache["rows"])
    rows = sorted(features, key=lambda r: (r["forecast_cutoff_utc"], r["match_id"]))
    return rows, parent_artifacts[phase]["predictions"], hashes


def corrections(feature_rows, score_rows, candidates, spec):
    outputs, fits = [], []
    keys = sorted({(r["league"], r["fit_id"], r["fit_cutoff_utc"]) for r in score_rows}, key=lambda k: (k[2], k[0]))
    for league, fit_id, date in keys:
        training = parent.prior_training_rows(feature_rows, datetime.fromisoformat(date))
        ids = [r["match_id"] for r in training]
        batch = [{**r, "probabilities": dict(r["probabilities"])} for r in score_rows if r["league"] == league and r["fit_id"] == fit_id]
        assert not set(ids).intersection(r["match_id"] for r in batch)
        fitted = {}
        for name, config in candidates.items():
            fitted[name] = model.fit(training, config, spec)
            for row, p in zip(batch, model.predict(batch, fitted[name])):
                row["probabilities"][name] = p.tolist()
        fits.append({"league": league, "fit_id": fit_id, "cutoff": date, "training_ids": ids,
                     "training_ids_sha256": b.digest(ids), "models": fitted})
        outputs.extend(batch)
        print(json.dumps({"fitted": fit_id, "league": league, "n": len(training), "candidates": list(candidates)}), flush=True)
    return sorted(outputs, key=lambda r: (r["forecast_cutoff_utc"], r["match_id"])), fits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["selection", "evaluation"], required=True)
    args = parser.parse_args()
    spec = specification()
    path = OUT/f"{args.phase}.json"
    if path.exists():
        raise ValueError("Completed phase is frozen")
    attempt = OUT/f"{args.phase}_attempt.json"
    if attempt.exists():
        assert json.loads(attempt.read_text())["source_sha256"] == b.digest(sources())
    else:
        b.write_json(attempt, {"started_at_utc": b.now(), "spec_sha256": b.file_sha(SPEC), "source_files": sources(), "source_sha256": b.digest(sources()), "threads": 1})
    if args.phase == "evaluation":
        lock = json.loads((OUT/"selection_lock.json").read_text())
        assert lock["source_sha256"] == b.digest(sources())
        assert lock["selection_artifact_sha256"] == b.file_sha(OUT/"selection.json")
        candidate = lock["selected_model"]
    feature_rows, score_rows, hashes = inputs(args.phase, spec)
    if args.phase == "selection" or candidate not in spec["identity_fallbacks"]:
        candidates = spec["candidates"] if args.phase == "selection" else {candidate: spec["candidates"][candidate]}
        with threadpool_limits(limits=1):
            predictions, fits = corrections(feature_rows, score_rows, candidates, spec)
    else:
        predictions, fits = score_rows, []
    if args.phase == "selection":
        names = [*spec["identity_fallbacks"], *spec["candidates"]]
        metrics = {name: b.metrics(predictions, name) for name in names}
        candidate = min(names, key=lambda name: (metrics[name]["log_loss"], name))
        summary = {"selection_metrics": metrics}
    else:
        pooled = parent.summarize(predictions, candidate, spec)
        substantial = all(pooled["models"][candidate]["log_loss"] <= .98*pooled["models"][ref]["log_loss"]
                          and pooled["paired_against_references"][ref]["28"]["percentile_95_interval"][1] < 0 for ref in spec["required_research_references"])
        substantial &= pooled["models"][candidate]["log_loss"] < pooled["models"]["production_default_dc_auto"]["log_loss"]
        summary = {"pooled": pooled,
            "by_league": {league: parent.summarize([r for r in predictions if r["league"] == league], candidate, spec) for league in spec["leagues"]},
            "by_league_season": {f"{league}:{year}": parent.summarize([r for r in predictions if r["league"] == league and r["season"] == year], candidate, spec)
                for league in spec["leagues"] for year in spec["evaluation_seasons"]}, "substantial_gain_target_met": bool(substantial)}
    b.write_json(path, {"phase": args.phase, "completed_at_utc": b.now(), "spec_sha256": b.file_sha(SPEC),
        "source_files": sources(), "source_sha256": b.digest(sources()), "input_feature_artifacts_sha256": hashes,
        "selected_model": candidate, "summary": summary, "predictions": predictions, "fits": fits})
    if args.phase == "selection":
        b.write_json(OUT/"selection_lock.json", {"selected_at_utc": b.now(), "selected_model": candidate,
            "source_sha256": b.digest(sources()), "spec_sha256": b.file_sha(SPEC), "selection_artifact_sha256": b.file_sha(path),
            "selection_metrics": summary["selection_metrics"], "new_cycle2_evaluation_metrics_inspected": False})
    print(json.dumps({"phase": args.phase, "selected_model": candidate, "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
