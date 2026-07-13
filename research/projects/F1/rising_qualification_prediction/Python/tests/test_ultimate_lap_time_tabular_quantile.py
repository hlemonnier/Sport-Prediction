from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import packages.f1.models.ultimate_lap_time.tabular_quantile as tabular_quantile
from packages.f1.models.ultimate_lap_time.tabular_quantile import (
    TabularQuantileBackendUnavailable,
    TabularQuantileConfig,
    fit_tabular_quantile_model,
    predict_tabular_quantiles,
)
from packages.f1.orchestration.model_runtime import inspect_optional_model_runtime


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
    assert model.training_summary["quantile_semantics"]["lap_p05"]["alpha"] == 0.05
    assert model.training_summary["quantile_semantics"]["lap_p50"]["alpha"] == 0.50
    assert model.training_summary["quantile_semantics"]["lap_p90"]["alpha"] == 0.90
    assert model.training_summary["quantile_semantics"]["p05_to_p90_nominal_coverage"] == 0.85
    assert model.training_summary["config_sha256"] == model.config.fingerprint


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
    assert model.training_summary["status"] == "available_fallback"


def test_quantile_columns_are_bound_to_alpha_not_config_tuple_position() -> None:
    train = _training_frame()
    config = TabularQuantileConfig(
        backend="empirical",
        quantiles=(0.90, 0.05, 0.50),
        feature_columns=("tyre_age",),
    )
    model = fit_tabular_quantile_model(train, config=config)
    predictions = model.predict(train.head(1))
    target = train["lap_time_seconds"].to_numpy(dtype=float)

    assert predictions.loc[0, "lap_p05"] == pytest.approx(np.quantile(target, 0.05))
    assert predictions.loc[0, "lap_p50"] == pytest.approx(np.quantile(target, 0.50))
    assert predictions.loc[0, "lap_p90"] == pytest.approx(np.quantile(target, 0.90))


def test_quantile_absolute_lap_fit_is_same_season_by_default() -> None:
    current = _training_frame()
    prior = current.assign(
        season=2025,
        event_key=lambda frame: "prior-" + frame["event_key"].astype(str),
        lap_time_seconds=lambda frame: frame["lap_time_seconds"] + 4.0,
    )
    combined = pd.concat([prior, current], ignore_index=True)
    model = fit_tabular_quantile_model(
        combined,
        config=TabularQuantileConfig(
            backend="empirical",
            feature_columns=("tyre_age",),
            target_season=2026,
        ),
    )

    expected = np.quantile(current["lap_time_seconds"].to_numpy(dtype=float), 0.50)
    prediction = model.predict(current.head(1))
    assert prediction.loc[0, "lap_p50"] == pytest.approx(expected)
    assert model.training_summary["training_season"] == 2026
    assert model.training_summary["other_season_rows_excluded_from_fit"] == len(prior)
    assert model.training_summary["season_transfer_policy"] == "same_season_absolute_lap_time_only"


def test_explicit_quantile_backend_reports_unavailable_without_silent_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        tabular_quantile,
        "inspect_tabular_quantile_backend",
        lambda backend: {
            "backend": backend,
            "status": "unavailable",
            "available": False,
            "version": None,
            "reason": "runtime_missing_for_test",
        },
    )

    with pytest.raises(TabularQuantileBackendUnavailable) as captured:
        fit_tabular_quantile_model(
            _training_frame(),
            config=TabularQuantileConfig(
                backend="lightgbm",
                feature_columns=("tyre_age", "track_temp_c"),
                n_estimators=4,
            ),
        )

    payload = captured.value.to_payload()
    assert payload["status"] == "unavailable"
    assert payload["requested_backend"] == "lightgbm"
    assert [attempt["backend"] for attempt in payload["backend_attempts"]] == ["lightgbm"]


def test_lightgbm_quantile_objectives_when_runtime_is_available() -> None:
    runtime = inspect_optional_model_runtime("lightgbm")
    if not runtime.available:
        pytest.skip(f"LightGBM optional runtime unavailable: {runtime.issue}")
    model = fit_tabular_quantile_model(
        _training_frame(),
        config=TabularQuantileConfig(
            backend="lightgbm",
            feature_columns=("tyre_age", "track_temp_c", "driver_id"),
            n_estimators=5,
        ),
    )

    attempts = model.training_summary["backend_attempts"]
    assert model.backend_name == "lightgbm"
    assert attempts[-1]["status"] == "selected"
    assert attempts[-1]["quantile_objectives"]["lap_p05"]["alpha"] == 0.05
    assert attempts[-1]["quantile_objectives"]["lap_p50"]["alpha"] == 0.50
    assert attempts[-1]["quantile_objectives"]["lap_p90"]["alpha"] == 0.90
