"""Recompute saved discovery evidence and test real stream prefix invariance."""
from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from research.experiments.boundary_20260908.weather.model import run_experiment as w


def verify():
    lock = json.loads((w.OUT/"selection_lock.json").read_text())
    for path, digest in lock["dependencies"].items():
        assert w.c1.sha(w.ROOT/path) == digest, path
    for name, digest in lock["artifacts"].items():
        assert w.c1.sha(w.OUT/name) == digest, name
    frame, issued, weather, base, manifest = w.load_inputs()
    pd.testing.assert_frame_equal(frame, pd.read_pickle(w.OUT/"discovery_original_pairs.pkl"))
    features = w.augment(frame, weather)
    pd.testing.assert_frame_equal(features, pd.read_pickle(w.OUT/"weather_features_lag60.pkl"))
    with (w.OUT/"models.pkl").open("rb") as handle: models = pickle.load(handle)
    results = json.loads((w.OUT/"results.json").read_text())
    predictions = pd.read_pickle(w.OUT/"selection_forecasts.pkl")
    valid = frame.year.eq(2023)
    reference = w.c1.base_points(frame.loc[valid], base)
    np.testing.assert_array_equal(reference, predictions.baseline)
    recomputed = 0
    for variant in w.SPEC["family"]["grid"]:
        point = w.candidate_points(frame.loc[valid], features.loc[valid], reference,
                                   models[variant["max_leaf_nodes"]], variant)
        np.testing.assert_array_equal(point, predictions[variant["name"]])
        assert w.reports(frame.loc[valid], features.loc[valid], reference, point) == results["selection"][variant["name"]]
        recomputed += len(point)
    selected = min(w.SPEC["family"]["grid"], key=lambda v:results["selection"][v["name"]]["full"]["candidate_mae"])
    assert selected == results["selected"]
    for lag in [0, 120]:
        added = w.augment(frame.loc[valid], weather, lag)
        point = w.candidate_points(frame.loc[valid], added, reference, models[selected["max_leaf_nodes"]], selected)
        np.testing.assert_array_equal(point, predictions[f"selected_lag{lag}"])
        assert w.reports(frame.loc[valid], added, reference, point) == results["lag_sensitivity"][str(lag)]
        recomputed += len(point)
    cases = []
    for key in [202204, 202313]:
        event = next(e for e in manifest["events"] if e["event_key"] == key)
        raw = pd.read_csv(w.ROOT/event["original_laps_path"])
        boundary = float(np.quantile(raw.Time, .65))
        prefix_raw = raw.loc[raw.Time <= boundary].copy()
        full = w.c1.encoder.observed_features(raw, key)
        prefix = w.c1.encoder.observed_features(prefix_raw, key)
        expected = full.loc[full.issued_at_timestamp <= boundary].reset_index(drop=True)
        pd.testing.assert_frame_equal(prefix.reset_index(drop=True), expected)
        # The unmatched issuance ledger includes terminal points: target presence
        # cannot determine the weather issuance population or its predictions.
        rows = issued.loc[issued.event_key.eq(key) & issued.checkpoint_time.le(boundary)]
        assert len(rows) == len(prefix)
        ordered = rows.sort_values(["driver_id", "checkpoint_lap"])
        p = prefix.sort_values(["driver_id", "issued_after_lap_number"])
        np.testing.assert_array_equal(ordered.checkpoint_time, p.issued_at_timestamp)
        np.testing.assert_array_equal(w.c1.prior_features(ordered), p[w.c1.BASE_FEATURES])
        original_weather = weather[key]
        prefix_weather = original_weather.loc[original_weather.Time < boundary-60].copy()
        poisoned_weather = original_weather.copy()
        future = poisoned_weather.Time >= boundary-60
        poisoned_weather.loc[future, w.CHANNELS] = 9999.0
        poisoned_weather.loc[future, "Rainfall"] = 1.0
        a = w.augment(rows, {key:original_weather})
        b = w.augment(rows, {key:prefix_weather})
        c = w.augment(rows, {key:poisoned_weather})
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(a, c)
        refs = w.c1.base_points(rows, base)
        for variant in w.SPEC["family"]["grid"]:
            model = models[variant["max_leaf_nodes"]]
            first = w.candidate_points(rows, a, refs, model, variant)
            np.testing.assert_array_equal(first, w.candidate_points(rows, b, refs, model, variant))
            np.testing.assert_array_equal(first, w.candidate_points(rows, c, refs, model, variant))
        cases.append(dict(event_key=key, prefix_issuances=len(rows), cutoff=boundary, variants=4,
                          legacy_features_and_population_identical=True, weather_and_points_identical=True))
    checks = results["advancement"]["checks"]
    selected_report = results["selection"][selected["name"]]
    main = selected_report["full"]
    expected_checks = dict(full_gain_at_least_one_percent=main["relative_reduction"] >= .01,
        event_ci95_upper_negative=main["event_ci95"][1] < 0,
        block3_ci95_upper_negative=main["block3_ci95"][1] < 0,
        all_event_leave_one_out_improve=main["loo_max_delta"] < 0,
        changed_events_at_least_three=selected_report["events_with_changed_predictions"] >= 3,
        lag0_and_lag120_positive=all(r["full"]["relative_reduction"] > 0 for r in results["lag_sensitivity"].values()),
        original_population_features_targets_unchanged=True)
    assert checks == expected_checks
    assert results["advancement"]["passed"] == all(checks.values())
    if not all(checks.values()):
        assert results["status"] == "rejected_2023_no_transfer"
        assert not (w.OUT/"transfer_forecasts.pkl").exists()
    report = dict(passed=True, source_sha256=w.c1.sha(Path(__file__)),
                  results_sha256=w.c1.sha(w.OUT/"results.json"),
                  predictions_recomputed=recomputed, max_prediction_error=0.0,
                  original_pairs_recomputed=len(frame), weather_features_recomputed=len(features),
                  raw_weather_lap_hashes_verified=3*len(manifest["events"]),
                  matched_channel_coverage={c:dict(missing_rows=int(features[f"w_{c}_missing"].sum()),
                      age_min_median_p99_max=np.quantile(features[f"w_{c}_age_seconds"], [0,.5,.99,1]).tolist())
                      for c in w.CHANNELS},
                  real_prefix_cases=cases,
                  transfer_run=False)
    w.c1.write(w.OUT/"verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    verify()
