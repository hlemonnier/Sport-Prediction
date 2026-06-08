from __future__ import annotations

import numpy as np
import pandas as pd

from packages.f1.models.ultimate_lap_time.tabular_quantile import (
    TabularQuantileConfig,
    fit_tabular_quantile_model,
    predict_tabular_quantiles,
)


def _training_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event_idx, circuit in enumerate(["bahrain", "jeddah", "monaco"]):
        for driver_idx, driver in enumerate(["VER", "LEC", "NOR", "PIA"]):
            for lap_idx in range(3):
                rows.append(
                    {
                        "season": 2026,
                        "event_key": f"event-{event_idx}",
                        "circuit_id": circuit,
                        "driver_id": driver,
                        "team_id": "team",
                        "session": "Q",
                        "compound": "SOFT" if lap_idx < 2 else "MEDIUM",
                        "tyre_age": lap_idx,
                        "track_temp_c": 35.0 + event_idx,
                        "lap_time_seconds": 88.0 + event_idx + (0.2 * driver_idx) + (0.05 * lap_idx),
                    }
                )
    return pd.DataFrame(rows)


def test_tabular_quantile_model_fits_with_graceful_backend_and_monotonic_predictions() -> None:
    train = _training_frame()
    config = TabularQuantileConfig(
        n_estimators=12,
        min_rows_for_boosting=6,
        feature_columns=("circuit_id", "driver_id", "compound", "tyre_age", "track_temp_c"),
    )

    model = fit_tabular_quantile_model(train, config=config)
    predictions = predict_tabular_quantiles(model, train.head(5))

    assert model.backend_name in {"lightgbm", "xgboost", "sklearn_hist", "sklearn_gbr", "empirical"}
    assert model.training_summary["rows_used"] == len(train)
    assert list(predictions.columns) == ["lap_p05", "lap_p50", "lap_p90", "model"]
    assert np.isfinite(predictions[["lap_p05", "lap_p50", "lap_p90"]].to_numpy()).all()
    assert (predictions["lap_p05"] <= predictions["lap_p50"]).all()
    assert (predictions["lap_p50"] <= predictions["lap_p90"]).all()


def test_tabular_quantile_small_data_uses_empirical_fallback() -> None:
    train = _training_frame().head(3)
    model = fit_tabular_quantile_model(
        train,
        config=TabularQuantileConfig(min_rows_for_boosting=10, feature_columns=("circuit_id", "tyre_age")),
    )

    predictions = model.predict(train)

    assert model.backend_name == "empirical"
    assert predictions["lap_p05"].nunique() == 1
    assert predictions["lap_p90"].iloc[0] >= predictions["lap_p05"].iloc[0]
