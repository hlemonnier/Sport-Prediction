"""Bounded global-control residual study; discovery gate precedes any later feed."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from research.experiments.boundary_20260908.checkpoint import run_experiment as c1
from research.experiments.boundary_20260908.checkpoint.cycle2 import run_experiment as c2
from research.experiments.boundary_20260908.control import acquire
from research.experiments.boundary_20260908.control.model import features as state

HERE = Path(__file__).resolve().parent
OUT = acquire.OUT / "model"
SPEC_PATH = HERE / "specification.json"
SPEC = json.loads(SPEC_PATH.read_text())
MANIFEST_PATH = acquire.OUT / "input_manifest.json"
CACHE_PATH = c2.OUT / "crossfit_training.pkl"


def dependencies():
    paths = [Path(__file__), Path(state.__file__), SPEC_PATH, Path(acquire.__file__),
             HERE / "test_model.py", HERE / "verify.py", MANIFEST_PATH,
             acquire.OUT / "acquisition_verification.json", acquire.WEATHER_MANIFEST,
             acquire.WEATHER_SOURCE, Path(c1.__file__), c1.SPEC_PATH,
             Path(c2.__file__), c2.SPEC_PATH, c2.OUT / "fit_lock.json", CACHE_PATH,
             c2.OUT / "models.pkl", c1.OUT / "selection_forecasts.pkl", c1.OUT / "models.pkl",
             c1.OUT / "fit_lock.json"]
    return {str(p.relative_to(c1.ROOT)): c1.sha(p) for p in paths}


def load_inputs():
    c2.verify_lock()
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert len(manifest["events"]) == 44 and not manifest["failed_events"]
    assert acquire.sha(acquire.__file__) == manifest["source_sha256"]
    streams, inputs = {}, {}
    for item in manifest["events"]:
        assert item["event_key"] // 100 in (2022, 2023)
        for field, hash_field in (("raw_path", "raw_sha256"), ("csv_path", "csv_sha256"),
                                  ("source_receipt_path", "source_receipt_sha256")):
            path = c1.ROOT / item[field]
            assert c1.sha(path) == item[hash_field], path
            inputs[item[field]] = item[hash_field]
        streams[item["event_key"]] = pd.read_csv(c1.ROOT / item["csv_path"],
            dtype={"Status": str, "Message": str}, keep_default_na=False)
    frame = pd.read_pickle(CACHE_PATH)
    assert len(frame) == 32575 and not frame.duplicated(c1.KEYS).any()
    assert sorted(int(k) for k in frame.loc[frame.year.eq(2022), "event_key"].unique()) == SPEC["training"]["fit_events"]
    assert frame.loc[frame.year.eq(2023), "event_key"].nunique() == 22
    assert set(frame.year.unique()) == {2022, 2023}
    assert np.isfinite(frame[["target_seconds", "reference", "cycle1_prediction"]]).all().all()
    np.testing.assert_array_equal(frame.loc[frame.eligible, "reference"], frame.loc[frame.eligible, "cycle1_prediction"])
    return frame, streams, inputs


def enrich(frame, streams, lag=15):
    parts = [state.event_features(streams[int(key)], points, lag)
             for key, points in frame.groupby("event_key", sort=False)]
    extra = pd.concat(parts).reindex(frame.index)
    assert not set(extra.columns) & set(frame.columns)
    result = pd.concat([frame, extra], axis=1)
    pd.testing.assert_frame_equal(result[frame.columns], frame)
    observed = result.control_observed_time
    assert (observed.isna() | observed.lt(result.checkpoint_time-lag)).all()
    return result


def model_features(frame):
    result = c1.expert_features(frame, frame.reference.to_numpy())
    result["cycle1_point_seconds"] = frame.cycle1_prediction.to_numpy()
    result["cycle1_correction_seconds"] = frame.cycle1_prediction.to_numpy()-frame.reference.to_numpy()
    for name in sorted(c for c in frame if c.startswith("g_")):
        result[name] = frame[name].to_numpy()
    assert not any(name.startswith("a_") for name in result)
    assert np.isfinite(result.to_numpy()).all()
    return result


def fit(frame, variant):
    selected = frame.loc[state.activation(frame, variant)]
    if len(selected) < 80 or selected.event_key.nunique() < 3:
        raise ValueError("fewer than 80 active rows or three active fitting events")
    model = HistGradientBoostingRegressor(max_leaf_nodes=variant["max_leaf_nodes"], **SPEC["training"]["estimator"])
    model.fit(model_features(selected), np.clip(selected.target_seconds-selected.cycle1_prediction, -12, 12),
              sample_weight=c1.event_weights(selected))
    return model


def predict(frame, model, variant):
    output = frame.cycle1_prediction.to_numpy(copy=True)
    gate = state.activation(frame, variant)
    if gate.any():
        residual = model.predict(model_features(frame.loc[gate]))
        output[gate] = np.maximum(1., output[gate]+np.clip(residual, -6., 6.))
    np.testing.assert_array_equal(output[~gate], frame.loc[~gate, "cycle1_prediction"])
    np.testing.assert_array_equal(output[frame.eligible], frame.loc[frame.eligible, "reference"])
    assert np.isfinite(output).all() and (output > 0).all()
    return output


def report(frame, candidate, variant):
    return {
        "vs_hgb": c1.reports(frame, frame.reference.to_numpy(), candidate),
        "vs_c1": c1.reports(frame, frame.cycle1_prediction.to_numpy(), candidate),
        "active_rows": int(state.activation(frame, variant).sum()),
    }


def advancement(primary, sensitivity):
    full, target = primary["vs_c1"]["full_checkpoint"], primary["vs_c1"]["target_balanced"]
    return {
        "full_gain_vs_c1_at_least_half_percent": full["relative_reduction"] >= .005,
        "target_balanced_improves_vs_c1": target["delta"] < 0,
        "full_block3_ci_upper_vs_c1_negative": full["block3_ci95"][1] < 0,
        "full_loo_max_delta_vs_c1_negative": full["loo_max_delta"] < 0,
        "lag0_and_lag60_full_and_target_improve_vs_c1": all(
            sensitivity[str(lag)]["vs_c1"][weight]["delta"] < 0
            for lag in (0, 60) for weight in ("full_checkpoint", "target_balanced")),
        "eligible_and_unsupported_predictions_exact": True,
    }


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "design_lock.json").exists():
        raise FileExistsError("immutable model design already exists")
    frame, streams, inputs = load_inputs()
    lock = {"frozen_before_fitting_at": datetime.now(timezone.utc).isoformat(),
            "specification": SPEC, "dependencies": dependencies(), "inputs": inputs,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "sklearn": sklearn.__version__}}
    c1.write(OUT / "design_lock.json", lock)
    prepared = enrich(frame, streams)
    inventory = {}
    for year, points in prepared.groupby("year"):
        inventory[str(year)] = {"events": int(points.event_key.nunique()), "rows": len(points),
            "eligible_rows": int(points.eligible.sum()), "known_control_rows": int(points.g_known.sum()),
            "current_status_counts": {code: int(points["g_current_"+code].sum()) for code in (*state.KNOWN, "unknown")},
            "variant_active_rows": {v["name"]: int(state.activation(points, v).sum()) for v in SPEC["training"]["grid"]}}
    prepared.to_pickle(OUT / "prepared_discovery.pkl")
    c1.write(OUT / "label_blind_inventory.json", {"populations": inventory, "feature_count": len(model_features(prepared).columns),
        "control_feature_count": len([c for c in prepared if c.startswith("g_")]),
        "prepared_sha256": c1.sha(OUT / "prepared_discovery.pkl"), "outcomes_scored": False})
    verify_design()
    print(json.dumps(inventory, indent=2), flush=True)


def verify_design():
    lock = json.loads((OUT / "design_lock.json").read_text())
    assert dependencies() == lock["dependencies"]
    for path, expected in lock["inputs"].items():
        assert c1.sha(c1.ROOT / path) == expected, path
    c2.verify_lock()
    return lock


def discover():
    verify_design()
    if (OUT / "results.json").exists() or (OUT / "models.pkl").exists():
        raise FileExistsError("immutable discovery results already exist")
    inventory = json.loads((OUT / "label_blind_inventory.json").read_text())
    assert c1.sha(OUT / "prepared_discovery.pkl") == inventory["prepared_sha256"]
    frame = pd.read_pickle(OUT / "prepared_discovery.pkl")
    train, validation = frame.loc[frame.year.eq(2022)], frame.loc[frame.year.eq(2023)].copy()
    before = model_features(validation).copy(deep=True)
    models, results, failures = {}, {}, {}
    for variant in SPEC["training"]["grid"]:
        name = variant["name"]
        try:
            model = fit(train, variant)
            points = predict(validation, model, variant)
            results[name] = report(validation, points, variant)
            models[name] = model
            validation["prediction_"+name] = points
            print("selection", name, results[name]["vs_c1"]["full_checkpoint"]["relative_reduction"], flush=True)
        except Exception as exc:
            failures[name] = repr(exc)
            print("candidate-failed", name, repr(exc), flush=True)
        pd.testing.assert_frame_equal(before, model_features(validation))
    if not results:
        c1.write(OUT / "results.json", {"status": "all_candidates_failed", "failures": failures, "transfer_executed": False})
        return
    selected = min((v for v in SPEC["training"]["grid"] if v["name"] in results),
                   key=lambda v: results[v["name"]]["vs_c1"]["full_checkpoint"]["candidate_mae"])
    _, streams, _ = load_inputs()
    original = pd.read_pickle(CACHE_PATH)
    original = original.loc[original.year.eq(2023)]
    sensitivity = {}
    for lag in SPEC["clock"]["sensitivity_lags_seconds"]:
        delayed = enrich(original, streams, lag)
        values = predict(delayed, models[selected["name"]], selected)
        sensitivity[str(lag)] = report(delayed, values, selected)
        validation["prediction_selected_lag"+str(lag)] = values
    gates = advancement(results[selected["name"]], sensitivity)
    advanced = all(gates.values()) and not failures
    with (OUT / "models.pkl").open("wb") as handle:
        pickle.dump(models, handle)
    validation.to_pickle(OUT / "selection_predictions.pkl")
    columns = [*c1.KEYS, "target_lap", "target_time", "target_seconds", "eligible", "category",
               "reference", "cycle1_prediction", *[c for c in validation if c.startswith("prediction_")]]
    validation[columns].to_csv(OUT / "selection_predictions.csv.gz", index=False)
    payload = {"status": "advanced_for_frozen_transfer" if advanced else "stopped_at_2023_selection",
        "scope": SPEC["scope"], "specification_sha256": c1.sha(SPEC_PATH), "design_sha256": c1.sha(OUT / "design_lock.json"),
        "training_events": sorted(int(k) for k in train.event_key.unique()), "training_rows": len(train),
        "selection_events": int(validation.event_key.nunique()), "selection_rows": len(validation),
        "selected": selected, "all_variants": results, "failed_variants": failures,
        "selected_latency_sensitivity": sensitivity, "advancement_criteria": gates,
        "advanced": advanced, "transfer_executed": False, "later_control_inputs_acquired": False,
        "production_change": False, "output_hashes": {name: c1.sha(OUT / name)
            for name in ("models.pkl", "selection_predictions.pkl", "selection_predictions.csv.gz")}}
    c1.write(OUT / "results.json", payload)
    verify_design()
    print(json.dumps({"selected": selected, "advancement_criteria": gates, "advanced": advanced}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "discover"))
    args = parser.parse_args()
    {"prepare": prepare, "discover": discover}[args.phase]()
