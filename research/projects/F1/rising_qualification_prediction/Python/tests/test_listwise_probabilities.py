from __future__ import annotations

import numpy as np
import pandas as pd

from run_experiment import build_parser
from packages.f1.orchestration.prediction import _pl_gumbel_listwise


def _sample_frame() -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.DataFrame(
        {
            "event_key": [202501] * 5 + [202502] * 5,
            "event_year": [2025] * 10,
            "event_round": [1] * 5 + [2] * 5,
            "event_name_norm": ["alpha"] * 5 + ["beta"] * 5,
            "driver_id": [f"d{i}" for i in range(10)],
        }
    )
    preds = pd.Series(
        [1.0, 2.0, 3.5, 5.0, 6.0, 1.2, 2.3, 3.1, 3.8, 6.1],
        index=frame.index,
        dtype=float,
    )
    return frame, preds


def test_pl_listwise_probabilities_are_valid_and_normalized_per_event() -> None:
    frame, preds = _sample_frame()
    out = _pl_gumbel_listwise(
        frame=frame,
        preds=preds,
        mode="qualifying",
        samples=2000,
        temperature=1.0,
        seed=42,
    )
    assert ((out["p_win"] >= 0.0) & (out["p_win"] <= 1.0)).all()
    assert ((out["p_top3"] >= 0.0) & (out["p_top3"] <= 1.0)).all()
    assert ((out["p_top10"] >= 0.0) & (out["p_top10"] <= 1.0)).all()
    assert (out["p_top3"] <= (out["p_top10"] + 1e-9)).all()

    for event_key, event_rows in out.groupby(frame["event_key"], sort=False):
        _ = event_key
        assert abs(float(event_rows["p_win"].sum()) - 1.0) < 0.02
        assert abs(float(event_rows["p_top3"].sum()) - 3.0) < 0.02
        assert abs(float(event_rows["p_top10"].sum()) - float(len(event_rows))) < 0.02


def test_pl_listwise_is_deterministic_with_fixed_seed() -> None:
    frame, preds = _sample_frame()
    out_a = _pl_gumbel_listwise(
        frame=frame,
        preds=preds,
        mode="qualifying",
        samples=2000,
        temperature=1.0,
        seed=123,
    )
    out_b = _pl_gumbel_listwise(
        frame=frame,
        preds=preds,
        mode="qualifying",
        samples=2000,
        temperature=1.0,
        seed=123,
    )
    cols = ["p_win", "p_top3", "p_top10", "exp_pos", "pos_p10", "pos_p50", "pos_p90"]
    assert np.allclose(out_a[cols].to_numpy(dtype=float), out_b[cols].to_numpy(dtype=float))


def test_cli_help_exposes_horizon_a_flags() -> None:
    stdout = build_parser("prediction").format_help()
    assert "--f1_model" in stdout
    assert "--f1_listwise" in stdout
    assert "--f1_pl_samples" in stdout
    assert "--f1_pl_temperature" in stdout
    assert "--f1_listwise_seed" in stdout
    assert "--shadow_eval" in stdout
