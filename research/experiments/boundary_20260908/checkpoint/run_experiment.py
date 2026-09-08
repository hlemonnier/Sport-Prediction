"""Frozen-grid checkpoint expert research; no serving or acquisition changes."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from packages.f1.models.live_race import next_lap, next_lap_features as encoder
from research.experiments.performance_20260907.live.run_experiment import stream_event

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / "artifacts/research/boundary_20260908/checkpoint"
SPEC_PATH = HERE / "spec.json"
SPEC = json.loads(SPEC_PATH.read_text())
SEED = SPEC["seed"]
BASE_FEATURES = next_lap.load_model()["features"]
KEYS = ["event_key", "driver_id", "checkpoint_lap", "checkpoint_time"]
TARGET_KEYS = ["event_key", "driver_id", "target_lap"]
COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def clean(row):
    y = encoder.numeric(row.get("LapTime"))
    return (np.isfinite(y) and y > 0 and encoder.truth(row.get("IsAccurate"))
            and not any(np.isfinite(encoder.numeric(row.get(k))) for k in ["PitInTime", "PitOutTime"])
            and not any(c in str(row.get("TrackStatus", "")) for c in "4567"))


def available_pit(row, column):
    value = encoder.numeric(row.get(column))
    return bool(np.isfinite(value) and value <= float(row["Time"]))


def compound_name(value):
    name = str(value).strip().upper()
    return name if name in COMPOUNDS else "UNKNOWN"


def current_features(row, previous, state):
    """No target values, future pit timestamps or future-presence indicators."""
    time, lap = float(row["Time"]), int(row["LapNumber"])
    pit_in, pit_out = available_pit(row, "PitInTime"), available_pit(row, "PitOutTime")
    status = str(row.get("TrackStatus", ""))
    compound = compound_name(row.get("Compound"))
    stint = encoder.numeric(row.get("Stint"))
    result = dict(
        observed_lap=float(lap), elapsed_laps=float(lap-previous["issued_after_lap_number"]),
        elapsed_seconds=float(time-previous["issued_at_timestamp"]),
        pit_in=float(pit_in), pit_out=float(pit_out),
        observed_stint_transition=float(state["generation"] != previous["checkpoint_generation"]),
        known_provider_stint_change=float(np.isfinite(stint) and np.isfinite(previous["provider_stint"])
                                         and stint != previous["provider_stint"]),
        compound_missing=float(compound == "UNKNOWN"),
        compound_changed=float(compound != "UNKNOWN" and previous["compound"] in COMPOUNDS
                               and compound != previous["compound"]),
        fresh_tyre=float(encoder.truth(row.get("FreshTyre"))),
        fresh_tyre_missing=float(pd.isna(row.get("FreshTyre"))),
        track_status_missing=float(status.lower() in {"", "nan", "none"}),
    )
    for code in "1234567":
        result[f"flag_{code}"] = float(code in status)
    for name in COMPOUNDS:
        result[f"compound_{name}"] = float(compound == name)
    for column in ["LapTime", "TyreLife", "Position", *encoder.SECTORS, *encoder.SPEEDS]:
        v = encoder.numeric(row.get(column))
        result[column] = float(v) if np.isfinite(v) else 0.0
        result[column + "_missing"] = float(not np.isfinite(v))
        if column in encoder.SECTORS + encoder.SPEEDS:
            previous_value = previous["x_" + column]
            previous_missing = previous["x_" + column + "_missing"]
            result[column + "_prior_gap"] = float(np.clip(v-previous_value, -120, 120)) if np.isfinite(v) and not previous_missing else 0.0
            result[column + "_prior_gap_missing"] = float(not np.isfinite(v) or previous_missing)
    # Sector 2/3 can remain useful when an outlap's total duration is absent.
    s2, s3 = encoder.numeric(row.get("Sector2Time")), encoder.numeric(row.get("Sector3Time"))
    result["sector23_sum"] = float(s2+s3) if np.isfinite(s2+s3) else 0.0
    result["sector23_missing"] = float(not np.isfinite(s2+s3))
    return {"c_"+k: float(v) for k, v in result.items()}


def checkpoint_event(raw, event_key):
    """Resolve old checkpoints before issuing after the current observation."""
    base = encoder.observed_features(raw, event_key)
    if base.empty:
        return pd.DataFrame(), pd.DataFrame(), base
    lookup = {(r["driver_id"], r["issued_after_lap_number"], r["issued_at_timestamp"]): r
              for r in base.to_dict("records")}
    states, previous, pending = {}, {}, defaultdict(list)
    issued, matched = [], []
    ordered = raw.assign(_driver=raw.DriverNumber.map(encoder.driver_key)).sort_values(
        ["Time", "_driver", "LapNumber"], kind="mergesort")
    for row in ordered.to_dict("records"):
        driver, lap, time = row["_driver"], int(row["LapNumber"]), float(row["Time"])
        state = states.setdefault(driver, dict(generation=0, stint=np.nan, compound="UNKNOWN"))
        compound, stint = compound_name(row.get("Compound")), encoder.numeric(row.get("Stint"))
        if (state["generation"] == 0 or available_pit(row, "PitOutTime")
                or (compound != "UNKNOWN" and state["compound"] != "UNKNOWN" and compound != state["compound"])
                or (np.isfinite(stint) and np.isfinite(state["stint"]) and stint != state["stint"])):
            state["generation"] += 1
        if np.isfinite(stint): state["stint"] = stint
        if compound != "UNKNOWN": state["compound"] = compound
        eligible = clean(row)
        current = lookup.get((driver, lap, time))
        if eligible:
            for old in pending.pop(driver, []):
                assert old["checkpoint_time"] < time and old["checkpoint_lap"] < lap
                matched.append({**old, "target_lap": lap, "target_time": time,
                                "target_seconds": float(row["LapTime"]),
                                "target_stint_transition": old["prior_stint_generation"] != current["stint_generation"]})
            if current is not None:
                previous[driver] = {**current, "provider_stint": stint,
                                    "checkpoint_generation": state["generation"]}
        if driver not in previous: continue
        prior = previous[driver]
        c = current_features(row, prior, state)
        category = ("eligible" if eligible else "pit_out" if c["c_pit_out"] else
                    "pit_in" if c["c_pit_in"] else "neutralized" if any(c[f"c_flag_{k}"] for k in "4567")
                    else "other_ineligible")
        record = dict(event_key=event_key, driver_id=driver, checkpoint_lap=lap,
                      checkpoint_time=time, eligible=eligible, category=category,
                      prior_lap=prior["issued_after_lap_number"], prior_time=prior["issued_at_timestamp"],
                      prior_naive_seconds=prior["forecast_naive_seconds"],
                      prior_stint_generation=prior["stint_generation"],
                      **{"p_"+k: float(prior[k]) for k in BASE_FEATURES}, **c)
        issued.append(record)
        pending[driver].append(record)
    emitted, pairs = pd.DataFrame(issued), pd.DataFrame(matched)
    assert not emitted.duplicated(KEYS).any() and not pairs.duplicated(KEYS).any()
    # Independent legacy matching guarantees unchanged original eligibility
    # and target pairing, including cross-stint and skipped-numbered laps.
    _, old = stream_event(raw, event_key)
    eligible = pairs.loc[pairs.eligible].sort_values(["driver_id", "checkpoint_lap"])
    old = old.sort_values(["driver_id", "issued_after_lap_number"])
    assert len(eligible) == len(old)
    for left, right in [("checkpoint_lap", "issued_after_lap_number"), ("checkpoint_time", "issued_at_timestamp"),
                        ("target_lap", "target_lap_number"), ("target_time", "target_timestamp"),
                        ("target_seconds", "lap_time_seconds"), ("prior_naive_seconds", "forecast_naive_seconds")]:
        np.testing.assert_array_equal(eligible[left], old[right])
    return emitted, pairs, base


def input_paths(years):
    paths = []
    for year in years:
        paths.extend(sorted((ROOT/"data/f1/raw/weekends"/str(year)).glob("*/*_race_laps.csv")))
    if 2026 in years:
        paths.extend(sorted((ROOT/"data/f1/performance_20260907/live_recent_final").glob("*_race_laps.csv")))
    return paths


def build(years):
    frames, issuance_frames, manifest, seen = [], [], [], set()
    for path in input_paths(years):
        if "live_recent_final" in str(path): year, rnd = 2026, int(path.name.split("_")[2])
        else: year, rnd = int(path.parts[-3]), int(path.parent.name.split("_")[1])
        key = year*100+rnd
        assert key not in seen; seen.add(key)
        raw = pd.read_csv(path)
        emitted, matched, _ = checkpoint_event(raw, key)
        matched["year"] = year; emitted["year"] = year
        frames.append(matched); issuance_frames.append(emitted)
        manifest.append(dict(event_key=key, path=str(path.relative_to(ROOT)), sha256=sha(path),
                             raw_rows=len(raw), issued=len(emitted), matched=len(matched),
                             eligible_matched=int(matched.eligible.sum()),
                             future_pit_values_masked=sum(int((pd.to_numeric(raw[c], errors="coerce") > raw.Time).sum())
                                                          for c in ["PitInTime", "PitOutTime"])))
        print("checkpoints", key, len(matched), flush=True)
    return pd.concat(frames, ignore_index=True), pd.concat(issuance_frames, ignore_index=True), manifest


def event_weights(frame):
    n = frame.event_key.value_counts()
    return len(frame)/(len(n)*frame.event_key.map(n).to_numpy())


def prior_features(frame):
    return frame[["p_"+c for c in BASE_FEATURES]].rename(columns={"p_"+c:c for c in BASE_FEATURES})


def fit_base(frame):
    eligible = frame.loc[frame.eligible]
    s = SPEC["base_hgb"]
    model = HistGradientBoostingRegressor(**{k:s[k] for k in ["loss", "learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization", "early_stopping", "random_state"]})
    model.fit(prior_features(eligible), np.clip(eligible.target_seconds-eligible.prior_naive_seconds, -5, 5),
              sample_weight=event_weights(eligible))
    return model


def base_points(frame, model=None):
    correction = next_lap.predict_correction(prior_features(frame)) if model is None else np.clip(model.predict(prior_features(frame)), -3, 3)
    return frame.prior_naive_seconds.to_numpy()+correction


def expert_features(frame, reference):
    names = ["p_"+c for c in BASE_FEATURES]+sorted(c for c in frame if c.startswith("c_"))
    x = frame[names].copy()
    x["reference_seconds"] = reference
    x["reference_correction_seconds"] = reference-frame.prior_naive_seconds.to_numpy()
    x["observed_duration_gap_to_reference"] = np.where(frame.c_LapTime_missing, 0., np.clip(frame.c_LapTime-reference, -120, 120))
    assert np.isfinite(x.to_numpy()).all()
    return x


def fit_expert(frame, reference, variant):
    gate = ~frame.eligible.to_numpy()
    s = SPEC["candidate_family"]
    model = HistGradientBoostingRegressor(max_leaf_nodes=variant["max_leaf_nodes"],
        **{k:s[k] for k in ["loss", "learning_rate", "max_iter", "min_samples_leaf", "l2_regularization", "early_stopping", "random_state"]})
    model.fit(expert_features(frame.loc[gate], reference[gate]),
              np.clip(frame.loc[gate, "target_seconds"].to_numpy()-reference[gate], -10, 10),
              sample_weight=event_weights(frame.loc[gate]))
    return model


def predict_expert(frame, reference, model, variant):
    gate = ~frame.eligible.to_numpy()
    predicted = np.array(reference, copy=True)
    if gate.any():
        raw = model.predict(expert_features(frame.loc[gate], reference[gate]))
        predicted[gate] += np.clip(raw, -variant["prediction_clip_seconds"], variant["prediction_clip_seconds"])
    assert np.array_equal(predicted[~gate], reference[~gate])
    assert np.isfinite(predicted).all() and (predicted > 0).all()
    return predicted


def metrics(frame, reference, candidate, target_balanced=False):
    if frame.empty: return dict(events=0, rows=0)
    work = frame[TARGET_KEYS+(["year"] if "year" in frame else [])].copy()
    work["base"] = np.abs(np.asarray(reference)-frame.target_seconds.to_numpy())
    work["candidate"] = np.abs(np.asarray(candidate)-frame.target_seconds.to_numpy())
    work["weight"] = 1.0
    if target_balanced:
        work["weight"] = 1.0/work.groupby(TARGET_KEYS).base.transform("size")
    work["wb"], work["wc"] = work.weight*work.base, work.weight*work.candidate
    sums = work.groupby("event_key")[["wb", "wc", "weight"]].sum()
    base, pred = sums.wb/sums.weight, sums.wc/sums.weight
    delta = (pred-base).to_numpy(); keys = base.index.to_numpy(); n = len(keys)
    rng = np.random.default_rng(SEED)
    bootstrap = delta[rng.integers(n, size=(20000, n))].mean(axis=1)
    block_sums = np.zeros(20000)
    for year in np.unique(keys//100):
        subset = delta[keys//100 == year]; m = len(subset)
        starts = rng.integers(m, size=(20000, int(np.ceil(m/3))))
        indices = ((starts[:, :, None]+np.arange(3)) % m).reshape(20000, -1)[:, :m]
        block_sums += subset[indices].sum(axis=1)
    return dict(events=n, rows=len(frame), distinct_targets=len(work.drop_duplicates(TARGET_KEYS)),
                baseline_mae=float(base.mean()), candidate_mae=float(pred.mean()), delta=float(delta.mean()),
                relative_reduction=float(1-pred.mean()/base.mean()),
                event_ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
                block3_ci95=np.quantile(block_sums/n, [.025, .975]).tolist(),
                events_improved=int((delta < 0).sum()),
                loo_max_delta=float(max((delta.sum()-d)/(n-1) for d in delta)) if n>1 else None,
                per_event=[dict(event_key=int(k), baseline_mae=float(base.loc[k]), candidate_mae=float(pred.loc[k]),
                                delta=float(pred.loc[k]-base.loc[k])) for k in keys])


def reports(frame, reference, candidate):
    masks = dict(eligible=frame.eligible, all_ineligible=~frame.eligible,
                 observed_stint_transition=frame.c_observed_stint_transition.eq(1),
                 future_target_stint_transition_diagnostic_only=frame.target_stint_transition)
    masks.update({name:frame.category.eq(name) for name in ["pit_in", "pit_out", "neutralized", "other_ineligible"]})
    return dict(full_checkpoint=metrics(frame, reference, candidate),
                target_balanced=metrics(frame, reference, candidate, True),
                subsets={name:metrics(frame.loc[mask], reference[mask], candidate[mask]) for name, mask in masks.items()})


def dependencies():
    return {str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), SPEC_PATH,
            Path(next_lap.__file__), Path(encoder.__file__), next_lap.MODEL_PATH,
            ROOT/"research/experiments/performance_20260907/live/run_experiment.py"]}


def discover():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/"design_lock.json").exists(): raise FileExistsError("existing discovery lock")
    write(OUT/"design_lock.json", dict(frozen_at=datetime.now(timezone.utc).isoformat(), dependencies=dependencies(), specification=SPEC))
    frame, issued, manifest = build([2022, 2023])
    frame.to_pickle(OUT/"discovery_checkpoints.pkl"); issued.to_pickle(OUT/"discovery_issuances.pkl")
    train = frame.loc[frame.year.eq(2022)].copy()
    validation = frame.loc[frame.year.eq(2023)].copy()
    events = sorted(train.event_key.unique()); assert len(events) == 22
    crossfit, models, fold_records = [], {}, []
    for fold in SPEC["residual_training"]["2022_expanding_folds"]:
        n = fold["train_event_count"]; low, high = fold["score_event_indices"]
        fitting = train.loc[train.event_key.isin(events[:n])]
        scoring = train.loc[train.event_key.isin(events[low:high])].copy()
        model = fit_base(fitting); models[f"base_first_{n}"] = model
        scoring["reference"] = base_points(scoring, model)
        crossfit.append(scoring)
        fold_records.append(dict(train_events=[int(k) for k in events[:n]], score_events=[int(k) for k in events[low:high]],
                                 gated_fit_support=int((~scoring.eligible).sum()), target_leakage=False))
        print("base-fold", n, len(scoring), flush=True)
    crossfit = pd.concat(crossfit, ignore_index=True)
    model2022 = fit_base(train); models["base_2022"] = model2022
    validation["reference"] = base_points(validation, model2022)
    scores, selection_predictions, discovery_experts = {}, {}, {}
    for variant in SPEC["candidate_family"]["grid"]:
        model = fit_expert(crossfit, crossfit.reference.to_numpy(), variant)
        candidate = predict_expert(validation, validation.reference.to_numpy(), model, variant)
        scores[variant["name"]] = reports(validation, validation.reference.to_numpy(), candidate)
        selection_predictions[variant["name"]] = candidate
        discovery_experts[variant["name"]] = model
        print("selection", variant["name"], scores[variant["name"]]["full_checkpoint"]["relative_reduction"], flush=True)
    selected = min(SPEC["candidate_family"]["grid"], key=lambda v:scores[v["name"]]["full_checkpoint"]["candidate_mae"])
    final_training = pd.concat([crossfit, validation], ignore_index=True)
    # Fit every predeclared candidate for transparent loser transfer reporting;
    # only the 2023-selected candidate can satisfy the research gain criteria.
    final_models = {v["name"]:fit_expert(final_training, final_training.reference.to_numpy(), v)
                    for v in SPEC["candidate_family"]["grid"]}
    payload = dict(base_models=models, discovery_experts=discovery_experts, final_experts=final_models)
    with (OUT/"models.pkl").open("wb") as f: pickle.dump(payload, f)
    final_training.to_pickle(OUT/"crossfit_residual_training.pkl")
    for name, values in selection_predictions.items(): validation["prediction_"+name] = values
    validation.to_pickle(OUT/"selection_forecasts.pkl")
    selection = dict(selected=selected, all_selection_results=scores, folds=fold_records,
                     final_residual_training_events=sorted(int(k) for k in final_training.event_key.unique()),
                     final_training_rows=len(final_training), final_gated_training_rows=int((~final_training.eligible).sum()),
                     input_manifest=manifest, base_fit_years_for_selection=[2022],
                     base_model_for_transfer=next_lap.MODEL_ID, base_sha256=next_lap.MODEL_SHA256,
                     design_sha256=sha(OUT/"design_lock.json"), promotion=False)
    write(OUT/"selection.json", selection)
    write(OUT/"fit_lock.json", dict(frozen_before_transfer=datetime.now(timezone.utc).isoformat(), dependencies=dependencies(),
          design_sha256=sha(OUT/"design_lock.json"), selection_sha256=sha(OUT/"selection.json"),
          models_sha256=sha(OUT/"models.pkl"), residual_training_sha256=sha(OUT/"crossfit_residual_training.pkl"),
          selection_forecasts_sha256=sha(OUT/"selection_forecasts.pkl")))


def verify_lock():
    lock = json.loads((OUT/"fit_lock.json").read_text())
    assert lock["dependencies"] == dependencies()
    for filename, field in [("design_lock.json", "design_sha256"), ("selection.json", "selection_sha256"),
                            ("models.pkl", "models_sha256"), ("crossfit_residual_training.pkl", "residual_training_sha256"),
                            ("selection_forecasts.pkl", "selection_forecasts_sha256")]:
        assert sha(OUT/filename) == lock[field]
    return lock


def transfer():
    verify_lock()
    if (OUT/"results.json").exists(): raise FileExistsError("completed transfer")
    selection = json.loads((OUT/"selection.json").read_text())
    with (OUT/"models.pkl").open("rb") as f: models = pickle.load(f)
    frame, issued, manifest = build([2024, 2025, 2026])
    reference = base_points(frame)
    predictions = {v["name"]:predict_expert(frame, reference, models["final_experts"][v["name"]], v)
                   for v in SPEC["candidate_family"]["grid"]}
    masks = {"2024":frame.year.eq(2024), "2025":frame.year.eq(2025),
             "2024_2025":frame.year.isin([2024, 2025]), "2026_exposed":frame.year.eq(2026),
             "2026_recent_exposed":frame.event_key.ge(202610)}
    results = {period:{name:reports(frame.loc[mask], reference[mask], values[mask]) for name, values in predictions.items()}
               for period, mask in masks.items()}
    preferred = selection["selected"]["name"]
    hist = results["2024_2025"][preferred]
    criteria = dict(historical_checkpoint_gain_at_least_10pct=hist["full_checkpoint"]["relative_reduction"] >= .10,
                    historical_target_balanced_gain_at_least_5pct=hist["target_balanced"]["relative_reduction"] >= .05,
                    historical_gated_gain_at_least_20pct=hist["subsets"]["all_ineligible"]["relative_reduction"] >= .20,
                    each_historical_year_improves=all(results[y][preferred]["full_checkpoint"]["delta"] < 0 for y in ["2024", "2025"]),
                    exposed_2026_improves=results["2026_exposed"][preferred]["full_checkpoint"]["delta"] < 0,
                    historical_block3_ci_upper_negative=hist["full_checkpoint"]["block3_ci95"][1] < 0,
                    historical_all_loo_improve=hist["full_checkpoint"]["loo_max_delta"] < 0,
                    eligible_predictions_exactly_unchanged=all(np.array_equal(p[frame.eligible], reference[frame.eligible]) for p in predictions.values()))
    frame["reference"] = reference
    for name, values in predictions.items(): frame["prediction_"+name] = values
    frame.to_pickle(OUT/"transfer_forecasts.pkl"); issued.to_pickle(OUT/"transfer_issuances.pkl")
    frame[[*KEYS, *[k for k in TARGET_KEYS if k not in KEYS], "eligible", "category", "target_time", "target_seconds",
           "prior_lap", "prior_time", "reference", *["prediction_"+v["name"] for v in SPEC["candidate_family"]["grid"]]]].to_csv(OUT/"transfer_forecasts.csv.gz", index=False)
    verify_lock()
    write(OUT/"results.json", dict(experiment=SPEC["experiment"], selected=preferred, results=results,
          gain_criteria=criteria, substantial_research_gain_passed=all(criteria.values()), promotion=False,
          input_manifest=manifest, fit_lock_sha256=sha(OUT/"fit_lock.json"),
          forecasts_sha256=sha(OUT/"transfer_forecasts.pkl"), csv_sha256=sha(OUT/"transfer_forecasts.csv.gz"),
          eligible_issuance_identity_and_target_pairing_verified=True,
          limits=["Previously exposed retrospective seasons, not prospective validation.",
                  "Recorded eligibility and provider field receipt latency remain retrospective assumptions.",
                  "Original eligible issuance point predictions unchanged; different checkpoint information sets are explicit."]))
    print(json.dumps(dict(selected=preferred, criteria=criteria, summaries={k:v[preferred]["full_checkpoint"] for k,v in results.items()}), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=["discover", "transfer"])
    args = parser.parse_args()
    discover() if args.phase == "discover" else transfer()
