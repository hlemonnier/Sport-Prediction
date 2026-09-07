"""Runtime contracts; the golden values come from the frozen research model."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from packages.f1.data.schemas.session import PredictionConfig
from packages.f1.models.live_race import next_lap
from packages.f1.models.live_race.next_lap_features import observed_features
from packages.f1.models.live_race.sources import _standardize_laps
from packages.f1.models.live_race.evaluate import _safe_metrics
from packages.f1.orchestration.prediction import run_prediction

FIXTURE = Path(__file__).with_name("fixtures") / "live_frontier_prefix.json"


def raw_fixture():
    fixture = json.loads(FIXTURE.read_text())
    raw = pd.DataFrame(fixture["observations"])
    for column in next_lap.NUMERIC_COLUMNS:
        raw[column] = pd.to_numeric(raw[column]).astype(float)
    return raw, fixture


def test_portable_runtime_reproduces_frozen_reference():
    raw, fixture = raw_fixture()
    batch = next_lap.forecast_laps(raw, 202601)
    assert batch.status == "available", batch.reason
    np.testing.assert_allclose(
        batch.frame.forecast_seconds,
        fixture["frozen_forecast_seconds"],
        rtol=0,
        atol=1e-12,
    )
    assert not any(column.startswith("target_") for column in batch.frame)


def test_global_time_prefix_and_same_timestamp_peer_invariance():
    raw, _ = raw_fixture()
    full = observed_features(raw, 202601)
    prefix = observed_features(raw.loc[raw.Time <= 600], 202601)
    expected = full.loc[full.issued_at_timestamp <= 600].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, prefix, check_exact=True)
    poisoned = raw.copy()
    poisoned.loc[poisoned.Time > 600, ["LapTime", "Sector1Time", "Position"]] = 9999.0
    after = observed_features(poisoned, 202601)
    pd.testing.assert_frame_equal(
        expected,
        after.loc[after.issued_at_timestamp <= 600].reset_index(drop=True),
        check_exact=True,
    )
    # Driver 2's observation at the same instant cannot affect driver 1.
    poisoned = raw.copy()
    poisoned.loc[(poisoned.Time == 600) & poisoned.DriverNumber.eq("2"), "LapTime"] = (
        190.0
    )
    changed = observed_features(poisoned, 202601)
    selector = full.issued_at_timestamp.eq(600) & full.driver_id.eq("1")
    pd.testing.assert_frame_equal(
        full.loc[selector], changed.loc[selector], check_exact=True
    )


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda raw: raw.drop(columns=["Sector1Time"]), "Sector1Time"),
        (lambda raw: raw.assign(Time=np.nan), "finite global timestamp"),
        (
            lambda raw: raw.assign(
                Sector1Time=pd.to_timedelta(raw.Sector1Time, unit="s")
            ),
            "numeric seconds",
        ),
        (lambda raw: pd.concat([raw, raw.iloc[[0]]]), "chronology"),
    ],
)
def test_missing_inputs_and_invalid_chronology_trigger_fallback(mutation, reason):
    raw, _ = raw_fixture()
    result = next_lap.forecast_laps(mutation(raw), 202601)
    assert result.status == "fallback" and result.frame.empty
    assert reason in result.reason


def test_cold_start_and_integrity_failure(monkeypatch, tmp_path):
    raw, _ = raw_fixture()
    assert (
        next_lap.forecast_laps(raw.loc[raw.LapNumber <= 2], 202601).status
        == "warming_up"
    )
    path = tmp_path / "model.json"
    path.write_text("{}")
    next_lap.load_model.cache_clear()
    monkeypatch.setattr(next_lap, "MODEL_PATH", path)
    result = next_lap.forecast_laps(raw, 202601)
    assert result.status == "fallback" and "SHA256" in result.reason
    next_lap.load_model.cache_clear()


def test_source_conversion_and_global_dispatch_use_the_model(monkeypatch, tmp_path):
    raw, _ = raw_fixture()
    numeric = _standardize_laps(raw, event_key=202601, source_used="test")
    timedeltas = raw.copy()
    for column in next_lap.SECONDS_COLUMNS:
        timedeltas[column] = pd.to_timedelta(timedeltas[column], unit="s")
    converted = _standardize_laps(timedeltas, event_key=202601, source_used="test")
    np.testing.assert_allclose(
        next_lap.forecast_laps(numeric, 202601).frame.forecast_seconds,
        next_lap.forecast_laps(converted, 202601).frame.forecast_seconds,
        rtol=0,
        atol=1e-12,
    )
    path = tmp_path / "observed.csv"
    raw.to_csv(path, index=False)
    config = PredictionConfig(
        source="local",
        mode="race",
        year=2026,
        round_number=1,
        train_seasons=[],
        include_standings=False,
        cache_dir=None,
        meeting_name=None,
        country_name=None,
        weekends_dir=None,
        f1_mode="live",
        f1_live_source="local",
        f1_live_replay_path=str(path),
        f1_live_replay_cutoff_time_seconds=600,
    )
    monkeypatch.setattr(
        "packages.f1.models.live_race.predict._write_trace", lambda *a, **k: {}
    )
    monkeypatch.setattr("packages.f1.models.live_race.evaluate.ARIMA", None)
    result = run_prediction(config)
    summary = result.extras["live_summary"]
    assert summary["next_lap_point_model"] == next_lap.MODEL_ID
    expected = (
        next_lap.forecast_laps(raw.loc[raw.Time <= 600], 202601)
        .frame.groupby("driver_id")
        .tail(1)
    )
    for row in result.table.to_dict("records"):
        reference = expected.loc[expected.driver_id.eq(row["driver_id"])].iloc[0]
        assert row["next_lap_mean"] == reference.forecast_seconds
        assert row["next_lap_point_model"] == next_lap.MODEL_ID
        assert np.isnan(row["next_lap_std"]) and np.isnan(row["next_lap_pi90_low"])
        assert np.isfinite(row["next_lap_std_ssm"])
    baseline = run_prediction(replace(config, f1_live_next_lap_point_model="baseline"))
    assert baseline.table.next_lap_point_model.eq("last_clean_lap_naive_v1").all()
    np.testing.assert_array_equal(
        baseline.table.next_lap_mean, baseline.table.next_lap_mean_naive
    )


def test_point_only_metrics_do_not_invent_a_likelihood():
    metrics = _safe_metrics(
        np.array([90.0, 91.0]), np.array([90.5, 90.5]), np.array([np.nan, np.nan])
    )
    assert metrics.mae == 0.5 and metrics.nll_like is None
