"""Independent artifact, metric and real-prefix checks; no fitting or mutation."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from research.experiments.boundary_20260908.checkpoint import run_experiment as exp


def main():
    if (exp.OUT/"verification.json").exists(): raise FileExistsError("existing verification")
    lock = exp.verify_lock()
    result = json.loads((exp.OUT/"results.json").read_text())
    selection = json.loads((exp.OUT/"selection.json").read_text())
    assert result["fit_lock_sha256"] == exp.sha(exp.OUT/"fit_lock.json")
    assert result["forecasts_sha256"] == exp.sha(exp.OUT/"transfer_forecasts.pkl")
    assert result["amended_input_lock_sha256"] == exp.sha(exp.OUT/"amended_input_lock.json")
    amendment = json.loads((exp.OUT/"amended_input_lock.json").read_text())
    assert amendment["adapter_sha256"] == exp.sha(exp.HERE/"run_transfer_full_schema.py")
    assert amendment["original_fit_lock_sha256"] == exp.sha(exp.OUT/"fit_lock.json")
    assert result["failed_pre_scoring_attempt_sha256"] == exp.sha(exp.OUT/"failed_transfer_attempt.json")
    with (exp.OUT/"models.pkl").open("rb") as f: models = pickle.load(f)
    frame = pd.read_pickle(exp.OUT/"transfer_forecasts.pkl")
    reference = exp.base_points(frame)
    np.testing.assert_array_equal(reference, frame.reference)
    checked_models = []
    for variant in exp.SPEC["candidate_family"]["grid"]:
        name = variant["name"]
        values = exp.predict_expert(frame, reference, models["final_experts"][name], variant)
        np.testing.assert_array_equal(values, frame["prediction_"+name])
        np.testing.assert_array_equal(values[frame.eligible], reference[frame.eligible])
        checked_models.append(name)
    checks = []
    for item in [*selection["input_manifest"], *result["input_manifest"]]:
        assert exp.sha(exp.ROOT/item["path"]) == item["sha256"]
        checks.append(item["event_key"])
    # Direct weighted MAE reconstruction, independent of the reporting helper.
    selected = result["selected"]
    for label, mask in [("2024", frame.year.eq(2024)), ("2025", frame.year.eq(2025)),
                        ("2024_2025", frame.year.isin([2024, 2025])),
                        ("2026_exposed", frame.year.eq(2026)),
                        ("2026_recent_exposed", frame.event_key.ge(202610))]:
        f = frame.loc[mask].copy()
        f["base_error"] = abs(f.reference-f.target_seconds)
        f["candidate_error"] = abs(f["prediction_"+selected]-f.target_seconds)
        ev = f.groupby("event_key")[["base_error", "candidate_error"]].mean()
        m = result["results"][label][selected]
        np.testing.assert_allclose([ev.base_error.mean(), ev.candidate_error.mean()],
            [m["full_checkpoint"]["baseline_mae"], m["full_checkpoint"]["candidate_mae"]], atol=1e-12, rtol=0)
        targets = f.groupby(exp.TARGET_KEYS)[["base_error", "candidate_error"]].mean()
        ev_targets = targets.groupby("event_key").mean()
        np.testing.assert_allclose([ev_targets.base_error.mean(), ev_targets.candidate_error.mean()],
            [m["target_balanced"]["baseline_mae"], m["target_balanced"]["candidate_mae"]], atol=1e-12, rtol=0)
    # Selection reference uses only the 2022-fitted base; 2022 expert support
    # uses expanding folds with all score events later than every fit event.
    val = pd.read_pickle(exp.OUT/"selection_forecasts.pkl")
    np.testing.assert_array_equal(exp.base_points(val, models["base_models"]["base_2022"]), val.reference)
    for fold in selection["folds"]:
        assert max(fold["train_events"]) < min(fold["score_events"])
        assert not set(fold["train_events"]) & set(fold["score_events"])
    training = pd.read_pickle(exp.OUT/"crossfit_residual_training.pkl")
    assert training.year.isin([2022, 2023]).all()
    assert not training.event_key.between(202201, 202206).any()
    # Actual stored-event prefixes and poisoned future records; validate every
    # candidate's issued points, not merely matched targets or feature shapes.
    prefix_checks = []
    inventory = selection["input_manifest"] + result["input_manifest"]
    for key in [202201, 202313, 202506, 202613]:
        item = next(i for i in inventory if i["event_key"] == key)
        raw = pd.read_csv(exp.ROOT/item["path"])
        full, _, _ = exp.checkpoint_event(raw, key)
        cutoff = float(raw.Time.quantile(.6))
        expected = full.loc[full.checkpoint_time <= cutoff].reset_index(drop=True)
        prefix, _, _ = exp.checkpoint_event(raw.loc[raw.Time <= cutoff], key)
        poison = raw.copy()
        future = poison.Time > cutoff
        poison.loc[future, ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "TyreLife"]] = 9999.
        poison.loc[future, "IsAccurate"] = False
        poisoned, _, _ = exp.checkpoint_event(poison, key)
        pd.testing.assert_frame_equal(expected, prefix, check_exact=True)
        pd.testing.assert_frame_equal(expected, poisoned.loc[poisoned.checkpoint_time <= cutoff].reset_index(drop=True), check_exact=True)
        for variant in exp.SPEC["candidate_family"]["grid"]:
            name = variant["name"]
            a = exp.predict_expert(expected, exp.base_points(expected), models["final_experts"][name], variant)
            b = exp.predict_expert(prefix, exp.base_points(prefix), models["final_experts"][name], variant)
            np.testing.assert_array_equal(a, b)
        prefix_checks.append(dict(event_key=key, cutoff=cutoff, issuances=len(expected), all_four_candidates_exact=True))
    exp.write(exp.OUT/"verification.json", dict(results_sha256=exp.sha(exp.OUT/"results.json"),
              verifier_sha256=exp.sha(Path(__file__)), source_dependencies_unchanged=True,
              input_hashes_checked=len(checks), all_transfer_predictions_recomputed=True,
              models_checked=checked_models, full_and_target_balanced_mae_independently_recomputed=True,
              eligible_predictions_exactly_unchanged=True, selection_base_unseen_year_verified=True,
              expanding_fold_disjointness_verified=True, prefix_invariance=prefix_checks,
              tests_sha256=exp.sha(exp.HERE/"test_experiment.py")))
    print(json.dumps(dict(verified=True, rows=len(frame), inputs=len(checks), prefix_cases=len(prefix_checks)), indent=2))


if __name__ == "__main__": main()
