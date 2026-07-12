from __future__ import annotations

import numpy as np
import pandas as pd

from packages.f1.models.live_race import evaluate as live_evaluate


def test_naive_comparison_uses_warmup_history_and_identical_rows() -> None:
    trace = pd.DataFrame(
        {
            "driver_id": ["driver"] * 5,
            "lap_number": [1, 2, 3, 4, 5],
            "lap_time_seconds": [100.0, 102.0, 104.0, 106.0, 108.0],
            "one_step_pred_mean": [np.nan, np.nan, 103.0, 105.0, 107.0],
            "one_step_pred_std": [1.0] * 5,
            "assim_laps_driver": [1, 2, 3, 4, 5],
            "eval_included": [True] * 5,
        }
    )

    result = live_evaluate.evaluate_live_replay(trace, warmup_laps=3)

    assert result["model"]["rows"] == 3
    assert result["model_on_naive_rows"]["rows"] == 3
    assert result["naive_last_lap"]["rows"] == 3
    assert result["model_on_naive_rows"]["mae"] == 1.0
    assert result["naive_last_lap"]["mae"] == 2.0
    assert result["mae_gain_vs_naive"] == 1.0


def test_arima_baseline_is_expanding_origin_not_full_series_fit(monkeypatch) -> None:
    fitted_histories: list[np.ndarray] = []

    class FakeFit:
        def __init__(self, history: np.ndarray) -> None:
            self.history = history

        def fit(self) -> "FakeFit":
            fitted_histories.append(self.history.copy())
            return self

        def forecast(self, steps: int) -> np.ndarray:
            assert steps == 1
            return np.asarray([self.history[-1]], dtype=float)

    def fake_arima(history: np.ndarray, **_: object) -> FakeFit:
        return FakeFit(np.asarray(history, dtype=float))

    monkeypatch.setattr(live_evaluate, "ARIMA", fake_arima)
    frame = pd.DataFrame(
        {
            "driver_id": ["driver"] * 10,
            "lap_number": range(1, 11),
            "lap_time_seconds": range(101, 111),
        }
    )

    predictions, available = live_evaluate._arima_baseline_predictions(frame)

    assert available is True
    assert [len(history) for history in fitted_histories] == [8, 9]
    assert fitted_histories[0][-1] == 108.0
    assert fitted_histories[1][-1] == 109.0
    assert predictions.iloc[:8].isna().all()
    assert predictions.iloc[8:].tolist() == [108.0, 109.0]
