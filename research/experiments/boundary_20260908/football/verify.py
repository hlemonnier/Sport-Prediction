"""Recompute evidence, timing membership, and serialized forecasts without fitting."""
from datetime import datetime
import json
import numpy as np
from research.experiments.boundary_20260908.football import run, models
from research.experiments.performance_20260907.football import benchmark as b


def main():
    spec = run.specification()
    selection = json.loads((run.OUT / "selection.json").read_text())
    evaluation = json.loads((run.OUT / "evaluation.json").read_text())
    lock = json.loads((run.OUT / "selection_lock.json").read_text())
    assert b.file_sha(run.OUT / "selection.json") == lock["selection_artifact_sha256"]
    assert evaluation["selected_model"] == lock["selected_model"]
    assert len(selection["predictions"]) == 760 and len(evaluation["predictions"]) == 2280
    by_id, features, base_blocks, shot_models = {}, [], 0, 0
    data_quality = {"matches": 0, "count_fields_complete_nonnegative_integer": True, "targets_exceeding_shots": []}
    for league, zone in spec["leagues"].items():
        with run.league_context(zone):
            rows, stats, _, _ = run.load_league(league, spec["input_years"], spec)
        data_quality["matches"] += len(rows)
        values = np.asarray(list(stats.values()))
        assert np.isfinite(values).all() and (values >= 0).all() and (values == np.floor(values)).all()
        data_quality["targets_exceeding_shots"].extend({"match_id": mid, "HS_AS_HST_AST": v} for mid, v in stats.items() if v[2]>v[0] or v[3]>v[1])
        by_id.update({r.match.match_id: r for r in rows})
    for phase, artifact in (("selection", selection), ("evaluation", evaluation)):
        assert artifact["source_sha256"] == b.digest(run.sources())
        assert artifact["spec_sha256"] == b.file_sha(run.SPEC_PATH)
        for league in spec["leagues"]:
            path = run.OUT / f"features_{phase}_{league}.json"
            assert b.file_sha(path) == artifact["feature_artifacts_sha256"][league]
            cache = json.loads(path.read_text())
            assert cache["source_sha256"] == artifact["source_sha256"]
            for name, sha in cache["input_hashes"].items():
                assert b.file_sha(b.ROOT/name) == sha
            features.extend(cache["rows"])
            for fit in cache["fits"]:
                base = fit["base"]
                cutoff = datetime.fromisoformat(base["cutoff_utc"])
                history = [by_id[mid].match for mid in base["fit_match_ids"]]
                assert all(m.date < cutoff and b.match_available_at(m) <= cutoff for m in history)
                batch = [r for r in cache["rows"] if r["fit_id"] == base["fit_id"]]
                assert not set(base["fit_match_ids"]).intersection(r["match_id"] for r in batch)
                for half, info in fit["shot_strength"].items():
                    x = models.count_design(info["teams"], [(r["home"], r["away"]) for r in batch])
                    means = [np.exp(x@np.asarray(m["coef"])+m["intercept"]).reshape(-1, 2) for m in info["models"]]
                    reconstructed = np.log(np.column_stack(means))
                    np.testing.assert_allclose(reconstructed, [r["log_shot_means"][half] for r in batch], atol=1e-12, rtol=0)
                    shot_models += len(info["models"])
                base_blocks += 1
    features.sort(key=lambda r: (r["forecast_cutoff_utc"], r["match_id"]))
    assert len({r["match_id"] for r in features}) == len(features) == 7980
    corrected_blocks, corrected_models, max_replay_error = 0, 0, 0.
    for artifact in (selection, evaluation):
        for fit in artifact["fits"]:
            cutoff = datetime.fromisoformat(fit["cutoff"])
            training = run.prior_training_rows(features, cutoff)
            ids = [r["match_id"] for r in training]
            assert ids == fit["training_ids"] and b.digest(ids) == fit["training_ids_sha256"]
            batch = [r for r in artifact["predictions"] if r["league"] == fit["league"] and r["fit_id"] == fit["fit_id"]]
            for name, info in fit["models"].items():
                mechanism, penalty = run.configurations(spec)[name]
                f = np.asarray([models.feature_vector(r, mechanism) for r in training])
                np.testing.assert_allclose(f.mean(0), info["mean"], atol=1e-12, rtol=0)
                expected_scale = np.where(f.std(0)>1e-12, f.std(0), 1.)
                np.testing.assert_allclose(expected_scale, info["scale"], atol=1e-12, rtol=0)
                model = models.OffsetModel(mechanism, np.asarray(info["mean"]), np.asarray(info["scale"]), np.asarray(info["weights"]), info)
                predicted = model.predict(batch)
                stored = np.asarray([r["probabilities"][name] for r in batch])
                max_replay_error = max(max_replay_error, float(np.max(np.abs(predicted-stored))))
                np.testing.assert_allclose(predicted, stored, atol=1e-12, rtol=0)
                corrected_models += 1
            corrected_blocks += 1
    # The inherited strong comparators must reproduce the previous frozen run.
    old_files = {"E0": "test.json", "SP1": "transfer_SP1.json", "I1": "transfer_I1.json"}
    baseline_comparisons, baseline_max_error = 0, 0.
    for league, filename in old_files.items():
        old = json.loads((b.OUT / filename).read_text())
        old_rows = {r["match_id"]: r for r in old["predictions"]}
        for r in evaluation["predictions"]:
            if r["league"] != league:
                continue
            prior = old_rows[r["match_id"]]
            for name in ("dc_equal", "dc_365", "dc_180", "dc365_elo50"):
                if name not in prior["probabilities"]:
                    continue
                err = float(np.max(np.abs(np.asarray(r["probabilities"][name])-prior["probabilities"][name])))
                baseline_max_error = max(baseline_max_error, err)
                assert err <= 1e-10
                baseline_comparisons += 1
    candidate = evaluation["selected_model"]
    summary = run.summarize(evaluation["predictions"], candidate, spec)
    assert summary == evaluation["summary"]["pooled"]
    # Excludes the already documented completion-day rather than original-kickoff row.
    resumed = "I1:2024:Fiorentina:Inter"
    filtered = [r for r in evaluation["predictions"] if r["match_id"] != resumed]
    assert len(filtered) == 2279
    sensitivity = run.summarize(filtered, candidate, spec)
    verification = {"verified_at_utc": b.now(), "source_sha256": b.digest(run.sources()),
        "selection_sha256": b.file_sha(run.OUT / "selection.json"),
        "evaluation_sha256": b.file_sha(run.OUT / "evaluation.json"),
        "status": "passed", "prequential_feature_rows": len(features), "base_blocks": base_blocks,
        "shot_models_reconstructed": shot_models, "corrected_blocks": corrected_blocks,
        "corrected_models_reconstructed": corrected_models, "max_serialized_probability_replay_error": max_replay_error,
        "prior_comparator_vectors_checked": baseline_comparisons, "prior_comparator_max_error": baseline_max_error,
        "checks": ["source and input hashes", "exact correction training populations and availability", "shot model serialized parameter replay",
                   "training-only scaler replay", "selected correction serialized probability replay", "frozen prior comparator agreement", "pooled metrics and paired intervals recomputed"]}
    b.write_json(run.OUT / "verification.json", verification)
    b.write_json(run.OUT / "evidence.json", {"experiment_id": spec["id"], "created_at_utc": b.now(),
        "status": spec["status"], "promotion": False, "selected_model": candidate,
        "known_exposure": spec["known_exposure"], "spec_sha256": b.file_sha(run.SPEC_PATH),
        "source_files": run.sources(), "source_sha256": b.digest(run.sources()),
        "artifacts": {name: b.file_sha(run.OUT/name) for name in ["design_lock.json", "selection_lock.json", "selection.json", "evaluation.json", "verification.json"]},
        "all_candidates_selection_metrics": selection["summary"]["selection_metrics"],
        "evaluation_summary": evaluation["summary"], "resumed_fixture_exclusion": {"excluded_match_id": resumed, "summary": sensitivity},
        "verification": verification, "limitations": spec["limitations"],
        "timing_contract": spec["timing"], "pooling_contract": spec["offset_model"]["pooling"],
        "data_quality": data_quality,
        "count_anomaly_policy": "One known provider target-count > total-shot anomaly retained unchanged; separate Poisson count GLMs do not require a binomial subset interpretation. No performance-driven repair or exclusion.",
        "references": spec["source_references"]})
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
