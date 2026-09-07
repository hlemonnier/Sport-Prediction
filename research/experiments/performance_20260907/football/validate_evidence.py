"""Independent artifact-level reconstruction and chronology verification."""
from datetime import date
import json
import numpy as np
from research.experiments.performance_20260907.football import benchmark as b
from packages.football.mrp.joint import DixonColesModel
from packages.football.mrp.data import match_available_at


def main():
    spec = json.loads((b.LANE / "spec.json").read_text())
    rows, _ = b.load_rows(spec["input_season_start_years"], spec)
    by_id = {r.match.match_id: r for r in rows}
    features = b.causal_elo_features(rows, spec["elo"])
    selection = json.loads((b.OUT / "selection_lock.json").read_text())
    test = json.loads((b.OUT / "test.json").read_text())
    val = json.loads((b.OUT / "validation.json").read_text())
    assert selection["selected_at_utc"] < test["completed_at_utc"]
    assert selection["test_labels_loaded"] is False
    assert set(r["match_id"] for r in val["predictions"]).isdisjoint(r["match_id"] for r in test["predictions"])
    max_error = 0.0
    for report in (val, test):
        fits = {f["fit_id"]: f for f in report["fits"]}
        for f in fits.values():
            cutoff = b.datetime.fromisoformat(f["cutoff_utc"])
            assert all(match_available_at(by_id[mid].match) <= cutoff for mid in f["fit_match_ids"])
            assert b.digest(f["fit_match_ids"]) == f["fit_ids_sha256"]
            assert all(obj["diagnostics"]["converged"] for obj in f["dixon_coles"].values())
        for r in report["predictions"]:
            f = fits[r["fit_id"]]
            assert r["match_id"] not in f["fit_match_ids"]
            assert f["cutoff_utc"] <= r["forecast_cutoff_utc"]
            assert r["forecast_cutoff_utc"] == b.utc_midnight(date.fromisoformat(r["day"])).isoformat()
            assert r["elo_features"] == features[r["match_id"]]
            for name in b.DC:
                params = f["dixon_coles"][name]
                model = DixonColesModel(params["attack"], params["defense"], params["home_intercept"], params["away_intercept"], params["rho"])
                p = b.build_score_distribution(*model.expected_goals(r["home"], r["away"]), model.rho).outcome_probabilities
                max_error = max(max_error, float(np.max(np.abs(np.array(p)-r["probabilities"][name]))))
            mapping = f["elo_mapping"]
            logits = np.array(mapping["coefficients"]) @ np.array(r["elo_features"]) + mapping["intercept"]
            p = np.exp(logits-logits.max()); p /= p.sum()
            max_error = max(max_error, float(np.max(np.abs(p-r["probabilities"]["elo_component"]))))
            mixture = (.5*np.array(r["probabilities"]["dc_365"])+.5*p)
            max_error = max(max_error, float(np.max(np.abs(mixture-r["probabilities"]["dc365_elo50"]))))
            counts = np.bincount([by_id[mid].label for mid in f["fit_match_ids"]], minlength=3)+1
            max_error = max(max_error, float(np.max(np.abs(counts/counts.sum()-r["probabilities"]["league_frequency"]))))
        assert b.summarize(report["predictions"]) == report["summary"]
    assert max_error < 1e-12
    result = {"status": "passed", "prediction_rows": len(val["predictions"])+len(test["predictions"]),
              "maximum_probability_reconstruction_error": max_error,
              "checks": ["complete primary schedules", "disjoint validation/test IDs", "selection lock before test completion",
                         "all fit result availability before cutoff", "no target in its fit IDs", "normalized day-start UTC cutoff",
                         "Dixon-Coles reconstruction", "Elo multinomial softmax reconstruction", "fixed mixture reconstruction",
                         "smoothed league prior reconstruction", "all metrics recomputed"],
              "source_script_sha256": b.file_sha(b.LANE / "validate_evidence.py")}
    b.write_json(b.OUT / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
