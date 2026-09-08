"""Independent pairing, chronology, prefix and metric checks for frozen discovery."""
from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from research.experiments.boundary_20260908.control.model import run_experiment as run


def independent_means(frame, reference, candidate, target_balanced):
    groups = {}
    for row, base, pred in zip(frame.itertuples(), reference, candidate):
        key = (row.event_key, row.driver_id, row.target_lap) if target_balanced else (row.event_key,)
        groups.setdefault(key, []).append((abs(base-row.target_seconds), abs(pred-row.target_seconds)))
    event_values = {}
    for key, values in groups.items():
        event_values.setdefault(key[0], []).append(np.mean(values, axis=0))
    return np.mean([np.mean(values, axis=0) for values in event_values.values()], axis=0)


def verify():
    lock = run.verify_design()
    result = json.loads((run.OUT / "results.json").read_text())
    for name, expected in result["output_hashes"].items():
        assert run.c1.sha(run.OUT / name) == expected
    original, streams, _ = run.load_inputs()
    prepared = run.enrich(original, streams)
    saved_prepared = pd.read_pickle(run.OUT / "prepared_discovery.pkl")
    pd.testing.assert_frame_equal(saved_prepared, prepared)
    pd.testing.assert_frame_equal(prepared[original.columns], original)
    with (run.c1.OUT / "models.pkl").open("rb") as handle:
        c1_models = pickle.load(handle)
    with (run.c2.OUT / "models.pkl").open("rb") as handle:
        c2_models = pickle.load(handle)
    chronology_rows = 0
    for i, fold in enumerate(run.c2.SPEC["training"]["2022_cycle1_expert_crossfit"]):
        assert max(fold["train_events"]) < min(fold["score_events"])
        rows = original.loc[original.event_key.isin(fold["score_events"])]
        n = 11 if i == 0 else 16
        reference = run.c1.base_points(rows, c1_models["base_models"][f"base_first_{n}"])
        np.testing.assert_array_equal(reference, rows.reference)
        values = run.c1.predict_expert(rows, reference, c2_models["stage1_crossfit"][str(i)], run.c2.C1_VARIANT)
        np.testing.assert_array_equal(values, rows.cycle1_prediction)
        chronology_rows += len(rows)
    rows = original.loc[original.year.eq(2023)]
    reference = run.c1.base_points(rows, c1_models["base_models"]["base_2022"])
    np.testing.assert_array_equal(reference, rows.reference)
    values = run.c1.predict_expert(rows, reference,
        c1_models["discovery_experts"][run.c2.C1_VARIANT["name"]], run.c2.C1_VARIANT)
    np.testing.assert_array_equal(values, rows.cycle1_prediction)
    chronology_rows += len(rows)
    with (run.OUT / "models.pkl").open("rb") as handle:
        models = pickle.load(handle)
    saved = pd.read_pickle(run.OUT / "selection_predictions.pkl")
    validation = prepared.loc[prepared.year.eq(2023)]
    pd.testing.assert_frame_equal(saved[validation.columns], validation)
    recomputed, total = {}, 0
    for variant in run.SPEC["training"]["grid"]:
        name = variant["name"]
        if name not in models:
            assert name in result["failed_variants"]
            continue
        candidate = run.predict(validation, models[name], variant)
        np.testing.assert_array_equal(candidate, saved["prediction_"+name])
        report = run.report(validation, candidate, variant)
        assert report == result["all_variants"][name]
        for reference_name, reference in (("vs_hgb", validation.reference), ("vs_c1", validation.cycle1_prediction)):
            for target_balanced, metric in ((False, "full_checkpoint"), (True, "target_balanced")):
                means = independent_means(validation, reference, candidate, target_balanced)
                np.testing.assert_allclose(means, [report[reference_name][metric]["baseline_mae"],
                    report[reference_name][metric]["candidate_mae"]], rtol=0, atol=2e-14)
        recomputed[name] = report
        total += len(candidate)
    selected = min((v for v in run.SPEC["training"]["grid"] if v["name"] in recomputed),
                   key=lambda v: recomputed[v["name"]]["vs_c1"]["full_checkpoint"]["candidate_mae"])
    assert selected == result["selected"]
    sensitivity = {}
    for lag in (0, 60):
        delayed = run.enrich(original.loc[original.year.eq(2023)], streams, lag)
        candidate = run.predict(delayed, models[selected["name"]], selected)
        np.testing.assert_array_equal(candidate, saved["prediction_selected_lag"+str(lag)])
        sensitivity[str(lag)] = run.report(delayed, candidate, selected)
        assert sensitivity[str(lag)] == result["selected_latency_sensitivity"][str(lag)]
        total += len(candidate)
    assert run.advancement(recomputed[selected["name"]], sensitivity) == result["advancement_criteria"]
    assert result["advanced"] == (all(result["advancement_criteria"].values()) and not result["failed_variants"])
    prefixes = []
    for key in (202212, 202303):
        event = prepared.loc[prepared.event_key.eq(key)]
        sample = event.iloc[np.linspace(0, len(event)-1, 40, dtype=int)]
        for row in sample.itertuples():
            point = event.loc[[row.Index]]
            cutoff = row.checkpoint_time-15.
            raw = streams[key]
            truncated = raw.loc[raw.Time.lt(cutoff)]
            poisoned = raw.copy()
            poisoned.loc[poisoned.Time.ge(cutoff), "Status"] = "3"
            actual = run.state.event_features(raw, point)
            pd.testing.assert_frame_equal(actual, run.state.event_features(truncated, point))
            pd.testing.assert_frame_equal(actual, run.state.event_features(poisoned, point))
        prefixes.append({"event_key": key, "checkpoints": len(sample), "prefix_and_future_poison_equal": True})
    payload = {"status": "passed", "source_sha256": run.c1.sha(__file__),
        "results_sha256": run.c1.sha(run.OUT / "results.json"),
        "design_sha256": run.c1.sha(run.OUT / "design_lock.json"),
        "input_files": len(lock["inputs"]), "dependencies": len(lock["dependencies"]),
        "control_feature_rows_exact": len(prepared), "base_and_c1_chronology_predictions_exact": chronology_rows,
        "candidate_and_latency_predictions_exact": total, "independent_event_and_target_metric_means": True,
        "all_saved_metrics_and_uncertainty_exact": True, "model_feature_and_target_pairing_unchanged": True,
        "real_prefix_checks": prefixes, "advanced": result["advanced"], "transfer_executed": False}
    run.c1.write(run.OUT / "verification.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    verify()
