"""Export the trusted, frozen research bundle as numeric trees (no refitting)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle

ROOT = Path(__file__).resolve().parents[3]
MODEL_ID = "frontier_live_hgb_l15_i150_20260907"
PICKLE_SHA256 = "8845658bc0e37c9cc2d4878d846b643d1449efb342469ff509c3c6ec41ab484c"


def main():
    source = ROOT / "artifacts/research/frontier_20260907/live/candidate/model.pkl"
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PICKLE_SHA256:
        raise ValueError(
            "The research candidate differs from the verified frozen model"
        )
    bundle = pickle.loads(
        raw
    )  # Only this explicitly hash-verified local research file.
    estimator = bundle["model"]["model"]
    trees = []
    for iteration in estimator._predictors:
        if len(iteration) != 1:
            raise ValueError("Only scalar regression is supported")
        nodes = iteration[0].nodes
        if nodes["is_categorical"].any():
            raise ValueError("Categorical splits require a different runtime format")
        trees.append(
            [
                [
                    int(n["feature_idx"]),
                    float(n["num_threshold"]),
                    int(n["left"]),
                    int(n["right"]),
                    bool(n["missing_go_to_left"]),
                    bool(n["is_leaf"]),
                    float(n["value"]),
                ]
                for n in nodes
            ]
        )
    exported = {
        "format": "numeric_hgb_regression_v1",
        "model_id": MODEL_ID,
        "features": bundle["model"]["features"],
        "baseline": float(estimator._baseline_prediction.item()),
        "correction_clip_seconds": 3.0,
        "node_columns": [
            "feature",
            "threshold",
            "left",
            "right",
            "missing_left",
            "leaf",
            "value",
        ],
        "trees": trees,
        "provenance": {k: v for k, v in bundle.items() if k != "model"},
        "source_pickle_sha256": PICKLE_SHA256,
        "value_semantics": "leaf values already include learning_rate; add baseline and all trees",
        "runtime_activation": "default_when_observed_input_contract_is_satisfied",
        "evidence_status": "retrospective_multi_season_transfer_passed_not_prospective",
        "interval_calibrated": False,
    }
    destination = ROOT / f"packages/f1/models/live_race/assets/{MODEL_ID}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(exported, separators=(",", ":"), allow_nan=False) + "\n"
    destination.write_text(serialized)
    print(
        json.dumps(
            {
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
