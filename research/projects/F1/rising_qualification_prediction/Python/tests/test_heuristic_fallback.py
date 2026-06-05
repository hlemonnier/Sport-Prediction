from __future__ import annotations

import pandas as pd

from rqp.prediction import _hierarchical_fallback


def test_qualifying_fallback_prioritizes_empirical_fp_pace_blend() -> None:
    features = pd.DataFrame(
        {
            "driver_id": ["a", "b", "c"],
            "event_pace_index": [0.20, 0.60, 0.40],
            "fp_mean_rank": [1.0, 3.0, 2.0],
            "fp_quali_sim_rank": [2.0, 1.0, 3.0],
            "fp_weighted_delta": [9.0, 0.0, 1.0],
        }
    )

    score = _hierarchical_fallback(features, fallback_cols=["fp_weighted_delta"])

    assert score.idxmin() == 0
    assert score.loc[0] < score.loc[2] < score.loc[1]


def test_race_fallback_uses_qualifying_position_when_available() -> None:
    features = pd.DataFrame(
        {
            "driver_id": ["a", "b", "c"],
            "qualy_position": [3.0, 1.0, 2.0],
            "event_pace_index": [0.01, 0.99, 0.50],
            "fp_race_sim_rank": [1.0, 3.0, 2.0],
        }
    )

    score = _hierarchical_fallback(features, fallback_cols=["event_pace_index"])

    assert score.idxmin() == 1
    assert score.loc[1] < score.loc[2] < score.loc[0]
