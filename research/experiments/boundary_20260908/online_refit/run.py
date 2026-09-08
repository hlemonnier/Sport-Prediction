"""Causal expanding race-event fits against the exact frozen strong baseline."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from research.experiments.boundary_20260908.online_residual.run import metric, sha, write
from research.experiments.frontier_20260907 import live_frontier as old

HERE = Path(__file__).resolve().parent
OUT = old.ROOT / "artifacts/research/boundary_20260908/online_refit"


def plan(frame, year, stride=4):
    targets = sorted(int(x) for x in frame.loc[frame.year.eq(year), "event_key"].unique())
    blocks = []
    for start in range(0, len(targets), stride):
        predicted = targets[start:start+stride]
        fitted = sorted(int(x) for x in frame.loc[frame.event_key < predicted[0], "event_key"].unique())
        if not fitted:
            raise ValueError("No prior events available")
        assert max(fitted) < min(predicted)
        blocks.append((fitted, predicted))
    return blocks


def event_weights(frame, half_life):
    events = sorted(frame.event_key.unique())
    counts = frame.event_key.value_counts()
    ages = {event: len(events)-1-i for i, event in enumerate(events)}
    w = 1/frame.event_key.map(counts).to_numpy(float)
    if half_life is not None:
        w *= np.exp2(-frame.event_key.map(ages).to_numpy()/half_life)
    return w/w.mean()


def fit_predict(frame, features, years, cfg):
    output = pd.Series(index=frame.index, dtype=float)
    inventory = []
    for year in years:
        for fitted, predicted in plan(frame, year):
            train = frame.loc[frame.event_key.isin(fitted)]
            test = frame.loc[frame.event_key.isin(predicted)]
            model = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=.06,
                max_iter=150, max_leaf_nodes=cfg["leaves"], min_samples_leaf=80,
                l2_regularization=10, early_stopping=False, random_state=20260907)
            response = np.clip(train.lap_time_seconds-train.forecast_naive_seconds, -5, 5)
            model.fit(train[features], response, sample_weight=event_weights(train, cfg["half_life_events"]))
            p = test.forecast_naive_seconds.to_numpy()+np.clip(model.predict(test[features]), -3, 3)
            output.loc[test.index] = p
            inventory.append({"fit_events": fitted, "prediction_events": predicted,
                "fit_rows": len(train), "prediction_rows": len(test)})
            print(cfg["name"], year, predicted, len(train), flush=True)
    return output, inventory


def discover():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "selection.json").exists():
        raise FileExistsError("Selection cannot be overwritten")
    spec = json.loads((HERE / "specification.json").read_text())
    write(OUT / "design_lock.json", {"frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha(__file__), "spec_sha256": sha(HERE / "specification.json"),
        "discovery_sha256": sha(old.OUT / "discovery_data.pkl"), "spec": spec})
    frame = pd.read_pickle(old.OUT / "discovery_data.pkl")
    features = json.loads((old.OUT / "selection.json").read_text())["features"]
    vmask = frame.year.eq(2023)
    val = frame.loc[vmask].copy()
    baseline = old.predict_model(val, old.fit_model(frame.loc[frame.year.eq(2022)],
        {"kind":"hgb", "leaves":15, "iterations":150}, features))
    metrics, ledgers = {}, {}
    for cfg in spec["candidate_grid"]:
        pred, ledger = fit_predict(frame, features, [2023], cfg)
        values = pred.loc[val.index].to_numpy()
        metrics[cfg["name"]] = metric(val, values, baseline)
        ledgers[cfg["name"]] = ledger
        val[cfg["name"]] = values
        print("validation", cfg["name"], metrics[cfg["name"]]["relative_gain"], flush=True)
    chosen = min(metrics, key=lambda n: (metrics[n]["candidate_mae"], n))
    val["strong_baseline"] = baseline
    val.to_pickle(OUT / "selection_predictions.pkl")
    write(OUT / "selection.json", {"preferred": chosen, "metrics": metrics,
        "config": next(c for c in spec["candidate_grid"] if c["name"] == chosen),
        "validation_screen_passed": metrics[chosen]["relative_gain"] >= .005,
        "fit_ledgers": ledgers, "features": features, "source_sha256": sha(__file__),
        "design_lock_sha256": sha(OUT / "design_lock.json"),
        "predictions_sha256": sha(OUT / "selection_predictions.pkl")})
    print(json.dumps({"preferred": chosen, "gain": metrics[chosen]["relative_gain"],
        "transfer_allowed": metrics[chosen]["relative_gain"] >= .005}), flush=True)


def transfer():
    if (OUT / "results.json").exists():
        raise FileExistsError("Results cannot be overwritten")
    selection = json.loads((OUT / "selection.json").read_text())
    assert selection["validation_screen_passed"]
    assert selection["source_sha256"] == sha(__file__)
    write(OUT / "transfer_lock.json", {"selection_sha256": sha(OUT / "selection.json"),
        "started_at": datetime.now(timezone.utc).isoformat(), "source_sha256": sha(__file__)})
    discovery = pd.read_pickle(old.OUT / "discovery_data.pkl")
    path = old.OUT / "corrected_input_contract/transfer_data_and_forecasts.pkl"
    test = pd.read_pickle(path)
    frame = pd.concat([discovery, test], ignore_index=True)
    predictions, ledgers = fit_predict(frame, selection["features"], [2024,2025,2026], selection["config"])
    data = frame.loc[frame.year >= 2024].copy()
    with (old.OUT / "candidate/model.pkl").open("rb") as f:
        base = pickle.load(f)["model"]
    baseline = old.predict_model(data, base)
    np.testing.assert_allclose(baseline, data.prediction_hgb_l15_i150, rtol=0, atol=1e-12)
    values = predictions.loc[data.index].to_numpy()
    metrics = {}
    for name, mask in [("2024",data.year.eq(2024)),("2025",data.year.eq(2025)),
                       ("2024_2025",data.year.isin([2024,2025])),("2026",data.year.eq(2026))]:
        metrics[name] = metric(data.loc[mask], values[mask], baseline[mask])
    historic = metrics["2024_2025"]
    gate = (historic["relative_gain"] >= .10 and historic["block3_ci95"][1] < 0
        and historic["loo_max_delta"] < 0
        and all(metrics[y]["relative_gain"] > 0 for y in ["2024","2025","2026"]))
    data["candidate"] = values
    data["strong_baseline"] = baseline
    data.to_pickle(OUT / "transfer_predictions.pkl")
    write(OUT / "results.json", {"preferred": selection["preferred"], "metrics": metrics,
        "fit_ledgers": ledgers, "substantial_gate_passed": bool(gate),
        "source_transfer_sha256": sha(path), "predictions_sha256": sha(OUT / "transfer_predictions.pkl"),
        "transfer_lock_sha256": sha(OUT / "transfer_lock.json"), "production_change": False,
        "all_evaluation_dates_previously_exposed": True})
    print(json.dumps({"gate": bool(gate), "gains": {k:v["relative_gain"] for k,v in metrics.items()}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["discover", "transfer"])
    {"discover": discover, "transfer": transfer}[parser.parse_args().command]()
