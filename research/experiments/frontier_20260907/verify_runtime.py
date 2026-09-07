"""Reconcile the production encoder and portable trees with frozen transfer rows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from packages.f1.models.live_race.next_lap import MODEL_ID, MODEL_SHA256, load_model, predict_correction
from packages.f1.models.live_race.next_lap_features import KEYS, observed_features
from packages.f1.models.live_race.sources import _standardize_laps

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "artifacts/research/frontier_20260907/live/corrected_input_contract/results.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start = time.perf_counter()
    if sha(RESULTS) != "c9c230e889f178413e6d9f841cb2456d8a39b7e6e59b4ae3bf6d4092ba2da097":
        raise ValueError("canonical transfer results changed")
    results = json.loads(RESULTS.read_text())
    replay_path = RESULTS.with_name("transfer_data_and_forecasts.pkl")
    assert sha(replay_path) == results["forecasts_sha256"]
    reference = pd.read_pickle(replay_path)
    model = load_model()
    portable = reference.forecast_naive_seconds.to_numpy() + predict_correction(reference)
    np.testing.assert_array_equal(portable, reference.prediction_hgb_l15_i150)
    events = []
    for source in results["input_manifest"]:
        path = ROOT / source["path"]
        assert sha(path) == source["sha256"]
        raw = pd.read_csv(path)
        standardized = _standardize_laps(raw, event_key=source["event_key"], source_used="verification")
        features = observed_features(standardized, source["event_key"])
        expected = reference.loc[reference.event_key.eq(source["event_key"])]
        aligned = expected[KEYS].merge(features, on=KEYS, validate="one_to_one")
        assert len(aligned) == len(expected) == source["matched_rows"]
        np.testing.assert_array_equal(aligned[model["features"]].to_numpy(), expected[model["features"]].to_numpy())
        forecast = aligned.forecast_naive_seconds.to_numpy() + predict_correction(aligned)
        np.testing.assert_array_equal(forecast, expected.prediction_hgb_l15_i150)
        events.append({"event_key": source["event_key"], "source_sha256": source["sha256"],
                       "matched_forecasts": len(aligned), "issuances": len(features),
                       "max_feature_difference": 0.0, "max_forecast_difference_seconds": 0.0})
        if len(events) % 10 == 0:
            print(f"Verified {len(events)}/{len(results['input_manifest'])} event encodings", flush=True)
    paths = ["packages/f1/models/live_race/next_lap.py", "packages/f1/models/live_race/next_lap_features.py",
             "packages/f1/models/live_race/sources.py", "packages/f1/models/live_race/predict.py",
             "research/experiments/frontier_20260907/verify_runtime.py"]
    report = {"status": "passed", "model_id": MODEL_ID, "model_sha256": MODEL_SHA256,
              "canonical_results_sha256": sha(RESULTS), "frozen_transfer_rows": len(reference),
              "portable_prediction_max_difference_seconds": 0.0,
              "production_source_and_encoder_events": events, "source_hashes": {p: sha(ROOT / p) for p in paths},
              "seconds_elapsed": time.perf_counter() - start,
              "scope": "numeric equivalence to frozen retrospective evidence; no prospective claim"}
    out = ROOT / "artifacts/research/frontier_20260907/integration/runtime_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"path": str(out), "status": "passed", "events": len(events), "matched_rows": len(reference)}))


if __name__ == "__main__":
    main()
