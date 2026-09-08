"""Frozen sequential residual experiment; no production imports or mutations."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

from research.experiments.frontier_20260907 import live_frontier as original

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / "artifacts/research/boundary_20260908/online_residual"
OLD = original.OUT
SEED = 20260908
BASE_CONFIG = {"kind": "hgb", "leaves": 15, "iterations": 150}
CONTEXT = [
    "level_gap", "own_last_delta", "own_robust_trend", "common_increment",
    "log_stint_clean_count", "tyre_age", "clean_lap_gap", "wet_compound",
    "x_observed_lap", "x_position", "x_history_count", "x_peer_delta_mean",
    "x_peer_delta_median", "x_peer_delta_mad", "x_relative_field_pace",
    "x_mad_3", "x_mad_8", "x_range_3", "x_range_8",
]
ONLINE = [
    "own_last", "own_ewm02", "own_ewm05", "own_median3", "own_count",
    "own_age_seconds", "peer_median", "peer_mean", "peer_mad", "peer_count",
    "peer_same_compound", "peer_same_compound_count", "peer_latest_age",
]
META_FEATURES = CONTEXT + ["base_correction"] + ["r_" + x for x in ONLINE]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def assimilate(frame, baseline):
    """Replay label arrivals; peer state is frozen before an equal-time batch.

    frame may contain labels beyond the present; only target-arrival actions can
    read them. No residual is used to construct its own original prediction.
    """
    frame = frame.reset_index(drop=True)
    baseline = np.asarray(baseline, float)
    if len(frame) != len(baseline) or not np.isfinite(baseline).all():
        raise ValueError("finite baseline must align with rows")
    if not (frame.target_timestamp > frame.issued_at_timestamp).all():
        raise ValueError("targets must be strictly later than issuance")
    result = np.zeros((len(frame), len(ONLINE)))
    for _, event in frame.groupby("event_key", sort=True):
        records = event.to_dict("index")
        actions = {}
        for idx, row in records.items():
            actions.setdefault(float(row["issued_at_timestamp"]), [[], []])[0].append(idx)
            actions.setdefault(float(row["target_timestamp"]), [[], []])[1].append(idx)
        own, latest = {}, {}
        for timestamp, (issues, resolved) in sorted(actions.items()):
            # Copies of immutable dictionaries preserve strictly earlier peers.
            peers_before = dict(latest)
            for idx in sorted(resolved):
                row = records[idx]
                if not bool(row["target_same_stint"]):
                    continue
                driver = str(row["driver_id"])
                generation = int(row["stint_generation"])
                state = own.get(driver)
                if state is None or state["generation"] != generation:
                    state = {"generation": generation, "history": deque(maxlen=3),
                             "count": 0, "ewm02": 0.0, "ewm05": 0.0}
                    own[driver] = state
                error = float(np.clip(float(row["lap_time_seconds"])-baseline[idx], -3, 3))
                state["count"] += 1
                state["history"].append(error)
                for key, alpha in [("ewm02", .2), ("ewm05", .5)]:
                    state[key] += alpha * (error - state[key])
                state["time"] = timestamp
                latest[driver] = {"time": timestamp, "lap": int(row["target_lap_number"]),
                                  "error": error, "compound": row["compound"]}
            for idx in sorted(issues):
                row = records[idx]
                driver = str(row["driver_id"])
                generation = int(row["stint_generation"])
                state = own.get(driver)
                x = dict.fromkeys(ONLINE, 0.0)
                if state is not None and state["generation"] == generation:
                    x.update(own_last=state["history"][-1], own_ewm02=state["ewm02"],
                             own_ewm05=state["ewm05"], own_median3=float(np.median(state["history"])),
                             own_count=min(state["count"], 20)/20,
                             own_age_seconds=min(timestamp-state["time"], 300)/300)
                others = [p for d, p in peers_before.items()
                          if d != driver and 0 < timestamp-p["time"] <= 180
                          and abs(int(row["issued_after_lap_number"])-p["lap"]) <= 2]
                if others:
                    errors = np.array([p["error"] for p in others])
                    med = float(np.median(errors))
                    x.update(peer_median=med, peer_mean=float(errors.mean()),
                             peer_mad=float(np.median(np.abs(errors-med))),
                             peer_count=len(errors)/20,
                             peer_latest_age=min(timestamp-p["time"] for p in others)/180)
                    same = [p["error"] for p in others if p["compound"] == row["compound"]]
                    if same:
                        x["peer_same_compound"] = float(np.median(same))
                        x["peer_same_compound_count"] = len(same)/20
                result[idx] = [x[k] for k in ONLINE]
    out = frame[CONTEXT].copy()
    out["base_correction"] = baseline-frame.forecast_naive_seconds.to_numpy()
    out[["r_" + k for k in ONLINE]] = result
    assert np.isfinite(out.to_numpy()).all()
    return out


def metric(frame, prediction, baseline):
    ev = pd.DataFrame({"event_key": frame.event_key.to_numpy(),
        "baseline": np.abs(np.asarray(baseline)-frame.lap_time_seconds.to_numpy()),
        "candidate": np.abs(np.asarray(prediction)-frame.lap_time_seconds.to_numpy())}).groupby("event_key").mean()
    delta = (ev.candidate-ev.baseline).to_numpy()
    n = len(ev)
    rng = np.random.default_rng(SEED)
    draws = delta[rng.integers(n, size=(20000, n))].mean(axis=1)
    starts = rng.integers(n, size=(20000, int(np.ceil(n/3))))
    idx = ((starts[:, :, None]+np.arange(3)) % n).reshape(20000, -1)[:, :n]
    block = delta[idx].mean(axis=1)
    return {"rows": len(frame), "events": n, "baseline_mae": float(ev.baseline.mean()),
            "candidate_mae": float(ev.candidate.mean()),
            "relative_gain": float(1-ev.candidate.mean()/ev.baseline.mean()),
            "delta": float(delta.mean()), "event_ci95": np.quantile(draws, [.025,.975]).tolist(),
            "block3_ci95": np.quantile(block, [.025,.975]).tolist(),
            "events_won": int((delta < 0).sum()),
            "loo_max_delta": float(max((delta.sum()-v)/(n-1) for v in delta)) if n > 1 else None,
            "per_event": [{"event_key": int(k), "baseline_mae": float(r.baseline),
                           "candidate_mae": float(r.candidate)} for k, r in ev.iterrows()]}


def fit_meta(kind, option, frame, xx, baseline):
    y = np.clip(frame.lap_time_seconds.to_numpy()-baseline, -3, 3)
    w = original.weights(frame)
    if kind == "huber":
        scale = StandardScaler().fit(xx, sample_weight=w)
        model = HuberRegressor(epsilon=1.35, alpha=option, max_iter=1000, tol=1e-7)
        model.fit(scale.transform(xx), y, sample_weight=w)
    else:
        scale = None
        model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=150,
            learning_rate=.04, max_leaf_nodes=option, min_samples_leaf=100,
            l2_regularization=10, early_stopping=False, random_state=SEED)
        model.fit(xx, y, sample_weight=w)
    return {"kind": kind, "option": option, "model": model, "scale": scale}


def correction(bundle, xx):
    if bundle["kind"] == "filter":
        cfg = bundle["config"]
        # Shrink field median when fewer than three peers are observed.
        peer = xx.r_peer_median*np.minimum(xx.r_peer_count*20/3, 1)
        values = cfg["peer"]*peer + cfg["own"]*xx.r_own_ewm05
    else:
        x = bundle["scale"].transform(xx) if bundle["scale"] is not None else xx
        values = bundle["model"].predict(x)
    return np.clip(values, -1.5, 1.5)


def crossfit(discovery, features):
    train = discovery.loc[discovery.year.eq(2022)].copy()
    events = sorted(train.event_key.unique())
    rows, preds, provenance = [], [], []
    for n_fit, stop in [(6, 11), (11, 16), (16, len(events))]:
        fit = train.loc[train.event_key.isin(events[:n_fit])]
        val = train.loc[train.event_key.isin(events[n_fit:stop])]
        fitted = original.fit_model(fit, BASE_CONFIG, features)
        predictions = original.predict_model(val, fitted)
        rows.append(val)
        preds.append(predictions)
        provenance.append({"fit_events": [int(x) for x in events[:n_fit]],
                           "prediction_events": [int(x) for x in events[n_fit:stop]],
                           "fit_rows": len(fit), "prediction_rows": len(val)})
        print("crossfit", n_fit, len(val), flush=True)
    return pd.concat(rows, ignore_index=True), np.concatenate(preds), provenance


def discover():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "selection.json").exists():
        raise FileExistsError("Frozen selection already exists")
    source_spec = json.loads((HERE / "specification.json").read_text())
    write(OUT / "design_lock.json", {"frozen_at": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": sha(HERE / "specification.json"), "source_sha256": sha(__file__),
        "original_source_sha256": sha(original.__file__),
        "discovery_data_sha256": sha(OLD / "discovery_data.pkl"), "specification": source_spec})
    discovery = pd.read_pickle(OLD / "discovery_data.pkl")
    selection = json.loads((OLD / "selection.json").read_text())
    features = selection["features"]
    for item in selection["input_manifest"]:
        assert sha(ROOT / item["path"]) == item["sha256"]
    train, pt, provenance = crossfit(discovery, features)
    base = original.fit_model(discovery.loc[discovery.year.eq(2022)], BASE_CONFIG, features)
    val = discovery.loc[discovery.year.eq(2023)].reset_index(drop=True)
    pv = original.predict_model(val, base)
    xx, vx = assimilate(train, pt), assimilate(val, pv)
    models = {x["name"]: {"kind": "filter", "config": x} for x in source_spec["candidates"]["filters"]}
    for kind, options in [("huber", [.1, 10.]), ("hgb", [7, 15])]:
        for option in options:
            name = f"{kind}_{option}"
            models[name] = fit_meta(kind, option, train, xx, pt)
            print("meta-fit", name, flush=True)
    predictions, metrics = {}, {}
    for name, bundle in models.items():
        predictions[name] = pv+correction(bundle, vx)
        metrics[name] = metric(val, predictions[name], pv)
        print("validation", name, metrics[name]["relative_gain"], flush=True)
    preferred = min(metrics, key=lambda n: (metrics[n]["candidate_mae"], n))
    selected = {"preferred": preferred, "metrics": metrics,
        "validation_gain_screen_passed": metrics[preferred]["relative_gain"] >= .005,
        "crossfit_blocks": provenance, "features": META_FEATURES,
        "design_lock_sha256": sha(OUT / "design_lock.json"),
        "selection_data_year": 2023, "fit_static_labels_years": [2022]}
    write(OUT / "selection.json", selected)
    for name, values in predictions.items():
        val["prediction_"+name] = values
    val["strong_baseline"] = pv
    val.to_pickle(OUT / "selection_predictions.pkl")
    with (OUT / "selection_models.pkl").open("wb") as f:
        pickle.dump(models, f)
    if selected["validation_gain_screen_passed"]:
        final_frame = pd.concat([train, val], ignore_index=True)
        final_base = np.concatenate([pt, pv])
        fx = pd.concat([xx, vx], ignore_index=True)
        chosen = models[preferred]
        final = chosen if chosen["kind"] == "filter" else fit_meta(
            chosen["kind"], chosen["option"], final_frame, fx, final_base)
        with (OUT / "frozen_model.pkl").open("wb") as f:
            pickle.dump(final, f)
        write(OUT / "fit_lock.json", {"selection_sha256": sha(OUT / "selection.json"),
            "model_sha256": sha(OUT / "frozen_model.pkl"), "source_sha256": sha(__file__),
            "fit_event_keys": sorted(int(x) for x in final_frame.event_key.unique()),
            "fit_rows": len(final_frame), "base_production_model_sha256": sha(OLD / "candidate/model.pkl")})
    print(json.dumps({"preferred": preferred, "validation_gain": metrics[preferred]["relative_gain"],
        "transfer_allowed": selected["validation_gain_screen_passed"]}), flush=True)


def transfer():
    if (OUT / "results.json").exists():
        raise FileExistsError("Frozen results already exist")
    selected = json.loads((OUT / "selection.json").read_text())
    assert selected["validation_gain_screen_passed"]
    lock = json.loads((OUT / "fit_lock.json").read_text())
    assert lock["source_sha256"] == sha(__file__)
    assert lock["selection_sha256"] == sha(OUT / "selection.json")
    assert lock["model_sha256"] == sha(OUT / "frozen_model.pkl")
    assert lock["base_production_model_sha256"] == sha(OLD / "candidate/model.pkl")
    with (OUT / "frozen_model.pkl").open("rb") as f:
        model = pickle.load(f)
    with (OLD / "candidate/model.pkl").open("rb") as f:
        base = pickle.load(f)["model"]
    path = OLD / "corrected_input_contract/transfer_data_and_forecasts.pkl"
    data = pd.read_pickle(path)
    baseline = original.predict_model(data, base)
    np.testing.assert_allclose(baseline, data.prediction_hgb_l15_i150.to_numpy(), atol=1e-12, rtol=0)
    xx = assimilate(data, baseline)
    predictions = baseline+correction(model, xx)
    metrics = {}
    for name, mask in [("2024", data.year.eq(2024)), ("2025", data.year.eq(2025)),
                       ("2024_2025", data.year.isin([2024,2025])), ("2026", data.year.eq(2026))]:
        metrics[name] = metric(data.loc[mask], predictions[mask], baseline[mask])
    historic = metrics["2024_2025"]
    gate = historic["relative_gain"] >= .10 and historic["block3_ci95"][1] < 0 and historic["loo_max_delta"] < 0
    gate = gate and all(metrics[y]["relative_gain"] > 0 for y in ["2024", "2025", "2026"])
    data["online_candidate"] = predictions
    data["strong_baseline"] = baseline
    data.to_pickle(OUT / "transfer_predictions.pkl")
    write(OUT / "results.json", {"preferred": selected["preferred"], "metrics": metrics,
        "substantial_gate_passed": bool(gate), "original_transfer_sha256": sha(path),
        "fit_lock_sha256": sha(OUT / "fit_lock.json"),
        "predictions_sha256": sha(OUT / "transfer_predictions.pkl"), "production_change": False,
        "all_test_dates_previously_exposed": True})
    print(json.dumps({"gate": bool(gate), "gains": {k:v["relative_gain"] for k,v in metrics.items()}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["discover", "transfer"])
    args = parser.parse_args()
    {"discover": discover, "transfer": transfer}[args.command]()
