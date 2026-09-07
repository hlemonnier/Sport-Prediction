"""Missing timing rows must not remove elapsed latent-process transitions."""

import numpy as np
import pandas as pd
import pytest

import packages.f1.models.live_race.predict as live
from packages.f1.data.schemas.session import PredictionConfig
from packages.f1.models.live_race.sources import LiveSourceResult, _standardize_laps
from packages.f1.models.live_race.state import FilterConfig, FilterState, initialize_filter_state, predict_state


def _run(monkeypatch, rows):
    config = PredictionConfig(source="local", mode="race", year=2026, round_number=1,
        train_seasons=[], include_standings=False, cache_dir=None, meeting_name=None,
        country_name=None, weekends_dir=None, f1_live_horizon_laps=1)
    monkeypatch.setattr(live, "load_live_observations", lambda _: LiveSourceResult(pd.DataFrame(rows), "local", []))
    monkeypatch.setattr(live, "_write_trace", lambda *args, **kwargs: {})
    monkeypatch.setattr(live, "evaluate_live_replay", lambda *args, **kwargs: {"available": False})
    monkeypatch.setattr(live, "_mc_position_distribution", lambda snapshot, **kwargs: (snapshot, {"position_dist_enabled": False}))
    return live.run_live_race_prediction(config)


def _row(lap, stint=1, compound="SOFT", **kwargs):
    return dict(event_key=202601, driver_id="A", lap_number=lap, stint_id=stint,
        compound=compound, is_box_lap=False, is_accurate=True, track_status="4",
        lap_time_seconds=90., race_time_seconds=90.*lap, timestamp=90.*lap, **kwargs)


@pytest.mark.parametrize("steps", [0, 1, 3, 17])
def test_multi_lap_transition_matches_closed_form_moments(steps):
    cfg = FilterConfig(phi=.8, q_pace=.13, q_deg=.027)
    state = FilterState(np.array([.3, .07]), np.array([[.4, .021], [.021, .03]]))
    a = np.array([[1., 1.], [0., cfg.phi]])
    q = np.diag([cfg.q_pace**2, cfg.q_deg**2])
    ah = np.linalg.matrix_power(a, steps)
    expected_cov = ah @ state.cov @ ah.T
    for exponent in range(steps):
        aj = np.linalg.matrix_power(a, exponent)
        expected_cov += aj @ q @ aj.T
    mean, cov = predict_state(state, cfg, steps=steps)
    np.testing.assert_allclose(mean, ah @ state.mean, atol=1e-12)
    np.testing.assert_allclose(cov, expected_cov, atol=1e-12)


def test_sparse_same_stint_matches_all_elapsed_laps_including_covariance(monkeypatch):
    dense = _run(monkeypatch, [_row(1), _row(2), _row(3)])
    sparse = _run(monkeypatch, [_row(1), _row(3)])
    columns = ["pace_penalty_mean", "pace_penalty_std", "deg_rate_mean", "deg_rate_std", "next_lap_mean_ssm", "next_lap_std"]
    np.testing.assert_allclose(sparse.trace.iloc[-1][columns].astype(float), dense.trace.iloc[-1][columns].astype(float))
    assert sparse.trace["prediction_transition_laps"].tolist() == [1, 2]
    assert sparse.trace["tyre_age"].tolist() == [1, 3]
    assert sparse.trace.iloc[-1]["pace_penalty_mean"] == pytest.approx(.1355)
    assert sparse.trace.iloc[-1]["deg_rate_mean"] == pytest.approx(.03645)
    assert sparse.summary["unobserved_lap_transition_count"] == 1


def test_explicit_stint_start_advances_only_new_stint_laps_after_reset(monkeypatch):
    result = _run(monkeypatch, [_row(1), _row(5, stint=2, compound="HARD", stint_start_lap=3)])
    current = result.trace.iloc[-1]
    expected_mean, expected_cov = predict_state(initialize_filter_state("HARD", FilterConfig()), FilterConfig(), steps=3)
    np.testing.assert_allclose(current[["pace_penalty_mean", "deg_rate_mean"]].astype(float), expected_mean)
    np.testing.assert_allclose(current[["pace_penalty_std", "deg_rate_std"]].astype(float), np.sqrt(np.diag(expected_cov)))
    assert current["prediction_transition_laps"] == 3
    assert current["completed_laps_since_previous_observation"] == 4
    assert current["tyre_age"] == 3
    assert result.snapshot.iloc[0]["stint_start_lap"] == 3
    assert result.snapshot.iloc[0]["reset_boundary"] == "explicit_stint_start_lap"


def test_scrubbed_tyre_age_does_not_fabricate_unobserved_stint_start(monkeypatch):
    result = _run(monkeypatch, [_row(1), _row(5, stint=2, compound="HARD", tyre_age=9),
                                _row(7, stint=2, compound="HARD", tyre_age=11)])
    assert result.trace["prediction_transition_laps"].tolist() == [1, 1, 2]
    assert result.trace["tyre_age"].tolist() == [1, 9, 11]
    assert result.trace.iloc[1]["reset_boundary"] == "first_observed_stint_lap_start_unknown"
    assert result.summary["unknown_stint_reset_boundary_count"] == 1


@pytest.mark.parametrize("rows,message", [
    ([_row(1), _row(1)], "unique"),
    ([_row(1), {**_row(3), "timestamp": 89.}], "timestamps"),
    ([_row(1), _row(3, stint=2, stint_start_lap=1)], "stint_start_lap"),
    ([_row(1), _row(3, stint=2, stint_start_lap=4)], "stint_start_lap"),
])
def test_inconsistent_lap_and_stint_chronology_rejected(monkeypatch, rows, message):
    with pytest.raises(ValueError, match=message):
        _run(monkeypatch, rows)


def test_standard_source_preserves_sparse_distance_and_explicit_stint_start(monkeypatch):
    raw = pd.DataFrame(dict(DriverNumber=["A"]*3, LapNumber=[1, 3, 7], LapTime=[90.]*3,
        Time=[90., 270., 630.], Compound=["SOFT", "SOFT", "HARD"], Stint=[1, 1, 2],
        TrackStatus=["4"]*3, IsAccurate=[True]*3, StintStartLap=[1, 1, 5]))
    standardized = _standardize_laps(raw, event_key=202601, source_used="local")
    assert standardized["tyre_age"].tolist() == [1, 3, 3]
    result = _run(monkeypatch, standardized)
    assert result.trace["prediction_transition_laps"].tolist() == [1, 2, 3]
    assert result.snapshot.iloc[0]["stint_start_lap"] == 5
