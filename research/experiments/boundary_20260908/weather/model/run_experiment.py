"""Bounded original-issuance weather experiment; recorded-time causality only."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from research.experiments.boundary_20260908.checkpoint import run_experiment as c1

HERE = Path(__file__).resolve().parent
ROOT = c1.ROOT
OUT = ROOT / "artifacts/research/boundary_20260908/weather/model"
MANIFEST = ROOT / "artifacts/research/boundary_20260908/weather/input_manifest.json"
SPEC = json.loads((HERE / "spec.json").read_text())
CHANNELS = SPEC["feature_contract"]["channels"]
SLOPES = ["AirTemp", "Humidity", "Pressure", "TrackTemp", "WindSpeed"]


def weather_features(weather, issuance_times, lag=60.0):
    """Only the issuance clock enters; targets and future lap rows are absent."""
    t = np.asarray(issuance_times, dtype=float)
    if not np.isfinite(t).all() or not np.isfinite(lag) or lag < 0:
        raise ValueError("finite issuance times and nonnegative lag required")
    times = weather.Time.to_numpy(dtype=float)
    if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError("weather times must be finite, unique and increasing")
    cutoff, result, sources = t-lag, {}, {}
    available = np.zeros(len(t), dtype=bool)
    values, missing = {}, {}
    for channel in CHANNELS:
        raw = pd.to_numeric(weather[channel], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(raw)
        ct, cv = times[finite], raw[finite]
        index = np.searchsorted(ct, cutoff, side="left")-1
        known = index >= 0
        source = np.full(len(t), np.nan)
        value = np.zeros(len(t))
        source[known] = ct[index[known]]
        value[known] = cv[index[known]]
        age = np.where(known, t-source, 0.0)
        valid = known & (age <= 600.0)
        values[channel], missing[channel] = np.where(valid, value, 0.0), ~valid
        available |= valid
        sources[channel] = source
        if channel != "WindDirection":
            result[f"w_{channel}"] = values[channel]
        result[f"w_{channel}_age_seconds"] = np.minimum(age, 86400.0)
        result[f"w_{channel}_missing"] = (~valid).astype(float)
        result[f"w_{channel}_never_observed"] = (~known).astype(float)
        if channel in SLOPES:
            old_index = np.searchsorted(ct, cutoff-300.0, side="left")-1
            supported = valid & (old_index >= 0)
            span, slope = np.zeros(len(t)), np.zeros(len(t))
            span[supported] = source[supported]-ct[old_index[supported]]
            supported &= span > 0
            slope[supported] = (value[supported]-cv[old_index[supported]])/(span[supported]/60.0)
            result[f"w_{channel}_slope_per_minute"] = slope
            result[f"w_{channel}_slope_span_seconds"] = span
            result[f"w_{channel}_slope_missing"] = (~supported).astype(float)
    direction = np.deg2rad(values["WindDirection"] % 360)
    wind_valid = ~missing["WindDirection"]
    both = wind_valid & ~missing["WindSpeed"]
    result["w_wind_sin"] = np.where(wind_valid, np.sin(direction), 0.0)
    result["w_wind_cos"] = np.where(wind_valid, np.cos(direction), 0.0)
    result["w_wind_x"] = np.where(both, np.cos(direction)*values["WindSpeed"], 0.0)
    result["w_wind_y"] = np.where(both, np.sin(direction)*values["WindSpeed"], 0.0)
    result["w_wind_vector_missing"] = (~both).astype(float)
    rain = pd.to_numeric(weather.Rainfall, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(rain)
    if not np.isin(rain[finite], [0.0, 1.0]).all():
        raise ValueError("observed Rainfall must be binary or missing")
    rt, rv = times[finite], rain[finite]
    lo = np.searchsorted(rt, t-900.0, side="left")
    hi = np.searchsorted(rt, cutoff, side="left")
    lo = np.minimum(lo, hi)
    n = np.maximum(hi-lo, 0)
    cumulative = np.r_[0.0, np.cumsum(rv)]
    positives = np.maximum(cumulative[hi]-cumulative[lo], 0.0)
    # Both endpoints of each transition must be inside the observed window.
    changes = np.r_[0.0, np.cumsum(np.r_[0.0, np.diff(rv) != 0])] if len(rv) else np.array([0.0])
    count_changes = changes[hi]-changes[np.minimum(lo+1, hi)]
    positive_t = rt[rv == 1]
    index = np.searchsorted(positive_t, cutoff, side="left")-1
    rain_known = index >= 0
    rain_age = np.zeros(len(t))
    rain_age[rain_known] = np.minimum(t[rain_known]-positive_t[index[rain_known]], 86400.0)
    result.update(w_rain_sample_count=n.astype(float), w_rain_positive_count=positives,
                  w_rain_fraction=np.divide(positives, n, out=np.zeros(len(t)), where=n>0),
                  w_rain_changes=count_changes, w_rain_history_missing=(n == 0).astype(float),
                  w_last_positive_rain_age_seconds=rain_age,
                  w_positive_rain_never_observed=(~rain_known).astype(float),
                  w_any_channel_available=available.astype(float),
                  w_recent_rain=(positives > 0).astype(float))
    features = pd.DataFrame(result)
    assert np.isfinite(features.to_numpy()).all()
    return features, pd.DataFrame(sources)


def augment(frame, weather_by_event, lag=60.0):
    pieces = []
    for key, group in frame.groupby("event_key", sort=True):
        added, _ = weather_features(weather_by_event[int(key)], group.checkpoint_time, lag)
        added.index = group.index
        pieces.append(added)
    features = pd.concat(pieces).loc[frame.index]
    assert not features.index.duplicated().any()
    return features


def model_features(frame, weather):
    assert frame.index.equals(weather.index)
    original = c1.prior_features(frame)
    assert not set(original.columns).intersection(weather.columns)
    return pd.concat([original, weather], axis=1)


def candidate_points(frame, weather, reference, model, variant):
    gate = weather.w_any_channel_available.eq(1).to_numpy()
    if variant["scope"] == "observed_recent_rain":
        gate &= weather.w_recent_rain.eq(1).to_numpy()
    point = np.asarray(reference).copy()
    if gate.any():
        correction = model.predict(model_features(frame.loc[gate], weather.loc[gate]))
        point[gate] = frame.loc[gate].prior_naive_seconds.to_numpy()+np.clip(correction, -3, 3)
    assert np.array_equal(point[~gate], np.asarray(reference)[~gate])
    return point


def reports(frame, weather, reference, point):
    masks = {"observed_recent_rain":weather.w_recent_rain.eq(1).to_numpy(),
             "no_observed_recent_rain":weather.w_recent_rain.eq(0).to_numpy()}
    return dict(full=c1.metrics(frame, reference, point),
                subsets={k:c1.metrics(frame.loc[m], reference[m], point[m]) for k,m in masks.items()},
                events_with_changed_predictions=int(frame.loc[np.asarray(point) != reference].event_key.nunique()))


def load_inputs():
    c1.verify_lock()
    manifest = json.loads(MANIFEST.read_text())
    original_manifest = json.loads((c1.OUT/"selection.json").read_text())["input_manifest"]
    originals = {int(e["event_key"]):e for e in original_manifest}
    weather = {}
    for event in manifest["events"]:
        key = int(event["event_key"])
        assert key in originals and key//100 in [2022, 2023]
        assert event["status"] in {"downloaded", "local_reuse"}
        for path_key, hash_key in [("csv_path", "csv_sha256"), ("raw_path", "raw_sha256"),
                                   ("original_laps_path", "original_laps_sha256")]:
            assert c1.sha(ROOT/event[path_key]) == event[hash_key]
        assert event["original_laps_sha256"] == originals[key]["sha256"]
        weather[key] = pd.read_csv(ROOT/event["csv_path"])
    assert set(weather) == set(originals) and len(weather) == 44
    all_frame = pd.read_pickle(c1.OUT/"discovery_checkpoints.pkl")
    all_issued = pd.read_pickle(c1.OUT/"discovery_issuances.pkl")
    frame = all_frame.loc[all_frame.eligible].copy().reset_index(drop=True)
    issued = all_issued.loc[all_issued.eligible].copy().reset_index(drop=True)
    assert not frame.duplicated(c1.TARGET_KEYS).any()
    assert frame.checkpoint_time.equals(frame.prior_time)
    assert frame.target_time.gt(frame.checkpoint_time).all()
    with (c1.OUT/"models.pkl").open("rb") as handle:
        base = pickle.load(handle)["base_models"]["base_2022"]
    assert base.get_params()["max_leaf_nodes"] == 15
    assert list(base.feature_names_in_) == c1.BASE_FEATURES
    return frame, issued, weather, base, manifest


def dependencies():
    paths = [Path(__file__), HERE/"spec.json", Path(c1.__file__), c1.OUT/"models.pkl",
             c1.OUT/"fit_lock.json", c1.OUT/"selection.json", c1.OUT/"discovery_checkpoints.pkl",
             c1.OUT/"discovery_issuances.pkl", MANIFEST,
             Path(c1.next_lap.__file__), Path(c1.encoder.__file__), c1.next_lap.MODEL_PATH]
    return {str(p.relative_to(ROOT)):c1.sha(p) for p in paths}


def discover():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/"design_lock.json").exists():
        raise FileExistsError("existing frozen weather experiment")
    frame, issued, weather_by_event, base, manifest = load_inputs()
    c1.write(OUT/"design_lock.json", dict(frozen_before_fitting=datetime.now(timezone.utc).isoformat(),
                                        dependencies=dependencies(), specification=SPEC))
    added = augment(frame, weather_by_event)
    issued_added = augment(issued, weather_by_event)
    # Features computed from issuance clocks alone must not depend on target matching.
    paired = frame[c1.KEYS].merge(pd.concat([issued[c1.KEYS], issued_added], axis=1),
                                 on=c1.KEYS, how="left", validate="one_to_one")
    np.testing.assert_array_equal(added.to_numpy(), paired[added.columns].to_numpy())
    frame.to_pickle(OUT/"discovery_original_pairs.pkl")
    added.to_pickle(OUT/"weather_features_lag60.pkl")
    train, validation = frame.year.eq(2022), frame.year.eq(2023)
    reference = c1.base_points(frame.loc[validation], base)
    family = SPEC["family"]
    params = {k:family[k] for k in ["loss", "learning_rate", "max_iter", "min_samples_leaf", "l2_regularization", "early_stopping", "random_state"]}
    models, results, predictions = {}, {}, frame.loc[validation, c1.KEYS+c1.TARGET_KEYS[2:]+["target_time", "target_seconds", "year"]].copy()
    predictions["baseline"] = reference
    for variant in family["grid"]:
        leaves = variant["max_leaf_nodes"]
        if leaves not in models:
            model = HistGradientBoostingRegressor(max_leaf_nodes=leaves, **params)
            model.fit(model_features(frame.loc[train], added.loc[train]),
                      np.clip(frame.loc[train].target_seconds-frame.loc[train].prior_naive_seconds, -5, 5),
                      sample_weight=c1.event_weights(frame.loc[train]))
            models[leaves] = model
        point = candidate_points(frame.loc[validation], added.loc[validation], reference, models[leaves], variant)
        predictions[variant["name"]] = point
        results[variant["name"]] = reports(frame.loc[validation], added.loc[validation], reference, point)
        print(variant["name"], results[variant["name"]]["full"]["relative_reduction"], flush=True)
    selected = min(family["grid"], key=lambda v:results[v["name"]]["full"]["candidate_mae"])
    sensitivities = {}
    for lag in [0, 120]:
        sensitivity_features = augment(frame.loc[validation], weather_by_event, lag)
        point = candidate_points(frame.loc[validation], sensitivity_features, reference, models[selected["max_leaf_nodes"]], selected)
        sensitivities[str(lag)] = reports(frame.loc[validation], sensitivity_features, reference, point)
        predictions[f"selected_lag{lag}"] = point
    primary = results[selected["name"]]
    m = primary["full"]
    checks = dict(full_gain_at_least_one_percent=m["relative_reduction"] >= .01,
                  event_ci95_upper_negative=m["event_ci95"][1] < 0,
                  block3_ci95_upper_negative=m["block3_ci95"][1] < 0,
                  all_event_leave_one_out_improve=m["loo_max_delta"] < 0,
                  changed_events_at_least_three=primary["events_with_changed_predictions"] >= 3,
                  lag0_and_lag120_positive=all(s["full"]["relative_reduction"] > 0 for s in sensitivities.values()),
                  original_population_features_targets_unchanged=True)
    with (OUT/"models.pkl").open("wb") as handle: pickle.dump(models, handle)
    predictions.to_pickle(OUT/"selection_forecasts.pkl")
    coverage = {str(year):dict(events=int(frame.loc[frame.year.eq(year)].event_key.nunique()),
                              matched=int(frame.year.eq(year).sum()), issued=int(issued.year.eq(year).sum()),
                              rain_matched=int(added.loc[frame.year.eq(year)].w_recent_rain.sum())) for year in [2022, 2023]}
    c1.write(OUT/"results.json", dict(selected=selected, selection=results, lag_sensitivity=sensitivities,
        advancement=dict(passed=all(checks.values()), checks=checks), coverage=coverage,
        weather_rows=sum(len(w) for w in weather_by_event.values()),
        features=dict(legacy_count=len(c1.BASE_FEATURES), weather_count=len(added.columns), weather_names=list(added.columns)),
        status="discovery_passed_pending_frozen_transfer" if all(checks.values()) else "rejected_2023_no_transfer",
        limitations=["Recorded weather stream time with assumed delivery lag; no historical receipt times.",
                     "Original retrospective lap eligibility and recorded data corrections remain unchanged.",
                     "2023 is selected validation, not independent evidence of the winning variant's effect.",
                     "2024–2026 weather not acquired or scored by this experiment."],
        weather_manifest_sha256=c1.sha(MANIFEST), design_lock_sha256=c1.sha(OUT/"design_lock.json")))
    c1.write(OUT/"selection_lock.json", dict(frozen_at=datetime.now(timezone.utc).isoformat(), dependencies=dependencies(),
        artifacts={p.name:c1.sha(p) for p in [OUT/"models.pkl", OUT/"results.json", OUT/"selection_forecasts.pkl",
                                            OUT/"discovery_original_pairs.pkl", OUT/"weather_features_lag60.pkl"]}))
    print("selected", selected["name"], "advance", all(checks.values()), flush=True)


if __name__ == "__main__":
    discover()
