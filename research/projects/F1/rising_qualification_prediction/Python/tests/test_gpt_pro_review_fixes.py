from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from run_experiment import _assert_probability_audit_schema
from rqp.betting import BettingConfig, build_betting_recommendations
from rqp.data import _add_temporal_features_train, build_current_features, build_training_data
from rqp.evaluation import evaluate_prediction_rows
from rqp.prediction import (
    _build_oof_qualifying_signal_frame,
    _hierarchical_fallback,
    _predict_probability,
    _race_stochastic_score_layer,
    _rank_based_probability,
)
from rqp.providers import LocalWeekendProvider
from rqp.training import (
    CandidateSpec,
    StrategicRaceDeltaModel,
    TargetOffsetModel,
    _fit_candidate,
    _fit_pl_temperature_from_oof,
    _infer_target_spec,
    _probability_audit_from_oof,
    _pl_probabilities_for_oof_audit,
    _oof_scores_for_selection,
    train_model,
)


class MeanRegressor:
    def fit(self, _x, y):
        self.mean_ = float(np.mean(y))

    def predict(self, x):
        return np.full(len(x), self.mean_, dtype=float)


class LargeDeltaModel:
    def predict(self, frame):
        out = np.zeros(len(frame), dtype=float)
        if len(out) > 0:
            out[0] = 6.0
        if len(out) > 9:
            out[9] = -6.0
        return out


class DriftProbabilityModel:
    def predict_probabilities(self, scores):
        return {"top10": pd.Series(0.99, index=scores.index, dtype=float)}


class GridSourceProvider:
    def list_rounds(self, year):
        return [{"round_number": 8, "event_name": "Monaco Grand Prix"}]

    def get_fp_features(self, year, round_number):
        return pd.DataFrame(
            {
                "driver_id": ["a", "b", "c"],
                "driver_name": ["A", "B", "C"],
                "event_pace_index": [0.1, 0.2, 0.3],
            }
        )

    def get_qualifying_results(self, year, round_number):
        return pd.DataFrame(
            {
                "driver_id": ["a", "b", "c"],
                "position": [1.0, 2.0, 3.0],
            }
        )

    def get_race_results(self, year, round_number):
        return pd.DataFrame(
            {
                "driver_id": ["a", "b", "c"],
                "grid_position": [9.0, 9.0, 9.0],
            }
        )

    def get_starting_grid(self, year, round_number):
        return pd.DataFrame(
            {
                "driver_id": ["a", "b", "c"],
                "grid_position": [1.0, np.nan, 7.0],
                "grid_source": ["pre_race_official_grid", "pre_race_official_grid", "pre_race_official_grid"],
                "grid_status": ["grid", "missing", "grid"],
            }
        )

    def get_track_stats(self, year, round_number):
        return {}


class PostRaceGridOnlyProvider(GridSourceProvider):
    def get_starting_grid(self, year, round_number):
        return pd.DataFrame()


def test_evaluation_prefers_driver_id_over_brittle_names() -> None:
    predicted = [
        {"rank": 1, "driver_id": "44", "driver_name": "Different Name"},
        {"rank": 2, "driver_id": "1", "driver_name": "Another Name"},
    ]
    actual = pd.DataFrame(
        {
            "driver_id": ["44", "1"],
            "driver_name": ["Lewis Hamilton", "Max Verstappen"],
            "position": [1, 2],
        }
    )

    result = evaluate_prediction_rows(predicted, actual, "position")

    assert result["available"] is True
    assert result["rows_common"] == 2
    assert result["mae_on_common"] == 0.0
    assert result["field_mae"] == 0.0
    assert result["mae_valid"] is True


def test_evaluation_incomplete_field_cannot_report_headline_mae() -> None:
    predicted = [{"rank": idx, "driver_id": str(idx)} for idx in range(1, 11)]
    actual = pd.DataFrame(
        {
            "DriverNumber": list(range(1, 21)),
            "Abbreviation": [f"D{idx}" for idx in range(1, 21)],
            "position": list(range(1, 21)),
        }
    )

    result = evaluate_prediction_rows(predicted, actual, "position")

    assert result["available"] is True
    assert result["rows_actual"] == 20
    assert result["rows_common"] == 10
    assert result["field_coverage"] == 0.5
    assert result["mae_valid"] is False
    assert result["field_mae"] is None
    assert result["mae_on_common"] == 0.0
    assert result["field_mae_penalized"] > result["mae_on_common"]


def test_driver_identity_resolution_matches_number_code_and_full_name() -> None:
    predicted = [
        {"rank": 1, "driver_id": "44", "driver_name": "Hamilton"},
        {"rank": 2, "driver_id": "VER", "driver_name": "Max Verstappen"},
        {"rank": 3, "driver_name": "Charles Leclerc"},
    ]
    actual = pd.DataFrame(
        {
            "DriverNumber": [44, 1, 16],
            "Abbreviation": ["HAM", "VER", "LEC"],
            "FullName": ["Lewis Hamilton", "Max Verstappen", "Charles Leclerc"],
            "position": [1, 2, 3],
        }
    )

    result = evaluate_prediction_rows(predicted, actual, "position")

    assert result["available"] is True
    assert result["rows_common"] == 3
    assert result["field_mae"] == 0.0


def test_driver_identity_resolution_refuses_ambiguous_surname_only_match() -> None:
    predicted = [{"rank": 1, "driver_name": "Schumacher"}]
    actual = pd.DataFrame(
        {
            "FullName": ["Mick Schumacher", "Ralf Schumacher"],
            "DriverNumber": [47, 5],
            "position": [1, 2],
        }
    )

    result = evaluate_prediction_rows(predicted, actual, "position")

    assert result["available"] is False
    assert result["reason"] == "no_common_drivers"
    assert result["rows_common"] == 0


def test_current_race_features_mark_official_grid_vs_qualifying_fallback() -> None:
    current, notes = build_current_features(
        GridSourceProvider(),
        mode="race",
        year=2025,
        round_number=8,
        include_standings=False,
    )

    by_driver = current.set_index("driver_id")

    assert notes == []
    assert by_driver.loc["a", "grid_source"] == "pre_race_official_grid"
    assert float(by_driver.loc["a", "grid_position"]) == 1.0
    assert by_driver.loc["b", "grid_source"] == "qualifying_fallback"
    assert float(by_driver.loc["b", "grid_position"]) == 2.0
    assert by_driver.loc["c", "grid_source"] == "pre_race_official_grid"
    assert float(by_driver.loc["c", "grid_position"]) == 7.0
    assert by_driver.loc["a", "grid_status"] == "grid"


def test_current_race_features_do_not_use_post_race_grid_as_pre_race_source() -> None:
    current, notes = build_current_features(
        PostRaceGridOnlyProvider(),
        mode="race",
        year=2025,
        round_number=8,
        include_standings=False,
    )

    by_driver = current.set_index("driver_id")

    assert notes == []
    assert by_driver.loc["a", "grid_source"] == "qualifying_fallback"
    assert float(by_driver.loc["a", "grid_position"]) == 1.0
    assert by_driver.loc["c", "grid_source"] == "qualifying_fallback"
    assert float(by_driver.loc["c", "grid_position"]) == 3.0


def test_team_temporal_features_are_event_level_not_teammate_row_leakage() -> None:
    frame = pd.DataFrame(
        {
            "driver_id": ["a", "b", "a", "b"],
            "team_name": ["red", "red", "red", "red"],
            "event_name": ["Round 1", "Round 1", "Round 2", "Round 2"],
            "event_key": [202501, 202501, 202502, 202502],
            "fp_mean_delta": [1.0, 3.0, 100.0, 200.0],
            "fp_weighted_delta": [2.0, 4.0, 300.0, 500.0],
        }
    )

    out = _add_temporal_features_train(frame)
    event2 = out[pd.to_numeric(out["event_key"], errors="coerce") == 202502]

    assert event2["team_form_3_fp_mean_delta"].tolist() == [2.0, 2.0]
    assert event2["team_form_3_fp_weighted_delta"].tolist() == [3.0, 3.0]


def test_local_provider_exposes_grid_and_negative_grid_corr_is_not_stable(tmp_path) -> None:
    weekend = tmp_path / "2025" / "round_01_test_grand_prix"
    weekend.mkdir(parents=True)
    (weekend / "weekend_metadata.json").write_text(
        json.dumps(
            {
                "round_number": 1,
                "event_name": "Test Grand Prix",
                "sessions": [
                    {
                        "session_type": "qualifying",
                        "session_order": 4,
                        "results_path": "qualifying_results.csv",
                    },
                    {
                        "session_type": "race",
                        "session_order": 5,
                        "results_path": "race_results.csv",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "DriverNumber": [1, 2, 3],
            "Abbreviation": ["A", "B", "C"],
            "Position": [1, 2, 3],
        }
    ).to_csv(weekend / "qualifying_results.csv", index=False)
    pd.DataFrame(
        {
            "DriverNumber": [1, 2, 3],
            "Abbreviation": ["A", "B", "C"],
            "GridPosition": [1, 2, 3],
            "Position": [3, 2, 1],
        }
    ).to_csv(weekend / "race_results.csv", index=False)
    provider = LocalWeekendProvider(weekends_dir=str(tmp_path))

    race = provider.get_race_results(2025, 1)
    summary = provider._race_event_summary(2025, 1)

    assert race["grid_position"].tolist() == [1, 2, 3]
    assert summary is not None
    assert float(summary["grid_stability"]) == 0.0


def test_local_provider_starting_grid_uses_pre_race_grid_path_and_encodes_status(tmp_path) -> None:
    weekend = tmp_path / "2025" / "round_01_test_grand_prix"
    weekend.mkdir(parents=True)
    (weekend / "weekend_metadata.json").write_text(
        json.dumps(
            {
                "round_number": 1,
                "event_name": "Test Grand Prix",
                "grid_path": "starting_grid.csv",
                "grid_availability_phase": "pre_race",
                "sessions": [],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "DriverNumber": [1, 2, 3, 4],
            "GridPosition": [1, "PL", "DNS", 0],
        }
    ).to_csv(weekend / "starting_grid.csv", index=False)
    provider = LocalWeekendProvider(weekends_dir=str(tmp_path))

    grid = provider.get_starting_grid(2025, 1).set_index("driver_id")

    assert grid.loc["1", "grid_source"] == "pre_race_official_grid"
    assert float(grid.loc["1", "grid_position"]) == 1.0
    assert grid.loc["2", "grid_status"] == "pit_lane"
    assert float(grid.loc["2", "grid_position"]) == 2.0
    assert grid.loc["3", "grid_status"] == "dns"
    assert pd.isna(grid.loc["3", "grid_position"])
    assert grid.loc["4", "grid_status"] == "pit_lane"


def test_local_provider_race_results_encode_pit_lane_after_grid(tmp_path) -> None:
    weekend = tmp_path / "2025" / "round_01_test_grand_prix"
    weekend.mkdir(parents=True)
    (weekend / "weekend_metadata.json").write_text(
        json.dumps(
            {
                "round_number": 1,
                "event_name": "Test Grand Prix",
                "sessions": [
                    {
                        "session_type": "Race",
                        "results_path": "race_results.csv",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "DriverNumber": [1, 2, 3, 4],
            "Abbreviation": ["A", "B", "C", "D"],
            "GridPosition": [1, "PL", "DNS", 0],
            "Position": [1, 2, np.nan, 3],
        }
    ).to_csv(weekend / "race_results.csv", index=False)
    provider = LocalWeekendProvider(weekends_dir=str(tmp_path))

    race = provider.get_race_results(2025, 1).set_index("driver_id")

    assert float(race.loc["1", "grid_position"]) == 1.0
    assert race.loc["2", "grid_status"] == "pit_lane"
    assert float(race.loc["2", "grid_position"]) == 2.0
    assert race.loc["3", "grid_status"] == "dns"
    assert pd.isna(race.loc["3", "grid_position"])
    assert race.loc["4", "grid_status"] == "pit_lane"
    assert float(race.loc["4", "grid_position"]) == 3.0


class DnsTrainingProvider:
    def list_rounds(self, year):
        return [{"round_number": 1, "event_name": "Test Grand Prix"}]

    def get_fp_features(self, year, round_number):
        return pd.DataFrame(
            {
                "driver_id": ["1", "2", "3"],
                "driver_name": ["A", "B", "C"],
                "event_pace_index": [0.1, 0.2, 0.3],
                "fp_weighted_delta": [0.1, 0.2, 0.3],
            }
        )

    def get_qualifying_results(self, year, round_number):
        return pd.DataFrame(
            {
                "driver_id": ["1", "2", "3"],
                "driver_name": ["A", "B", "C"],
                "position": [1.0, 2.0, 3.0],
            }
        )

    def get_race_results(self, year, round_number):
        return pd.DataFrame(
            {
                "driver_id": ["1", "2", "3"],
                "driver_name": ["A", "B", "C"],
                "position": [1.0, 2.0, np.nan],
                "grid_position": [1.0, 4.0, np.nan],
                "grid_status": ["grid", "pit_lane", "dns"],
            }
        )

    def get_track_stats(self, year, round_number):
        return {}


def test_training_race_data_excludes_dns_instead_of_qualifying_grid_fallback() -> None:
    frame, notes = build_training_data(
        DnsTrainingProvider(),  # type: ignore[arg-type]
        mode="race",
        train_seasons=[2025],
        target_year=2026,
        target_round=1,
        include_standings=False,
    )

    assert set(frame["driver_id"]) == {"1", "2"}
    assert "3" not in set(frame["driver_id"])
    assert float(frame.loc[frame["driver_id"] == "2", "grid_position"].iloc[0]) == 4.0
    assert any("DNS" in note for note in notes)


def test_local_provider_ignores_unphased_generic_grid_path(tmp_path) -> None:
    weekend = tmp_path / "2025" / "round_01_test_grand_prix"
    weekend.mkdir(parents=True)
    (weekend / "weekend_metadata.json").write_text(
        json.dumps(
            {
                "round_number": 1,
                "event_name": "Test Grand Prix",
                "grid_path": "ambiguous_grid.csv",
                "sessions": [],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"DriverNumber": [1, 2], "GridPosition": [10, 11]}).to_csv(
        weekend / "ambiguous_grid.csv",
        index=False,
    )
    provider = LocalWeekendProvider(weekends_dir=str(tmp_path))

    grid = provider.get_starting_grid(2025, 1)

    assert grid.empty


def test_real_monaco_current_grid_does_not_use_post_race_results_when_pre_race_grid_missing() -> None:
    project_root = Path(__file__).resolve().parents[6]
    weekends_dir = project_root / "data" / "f1" / "raw" / "weekends"
    if not (weekends_dir / "2025" / "round_08_monaco_grand_prix").exists():
        pytest.skip("local F1 weekend fixture unavailable")
    provider = LocalWeekendProvider(weekends_dir=str(weekends_dir))

    grid = provider.get_starting_grid(2025, 8)
    current, _ = build_current_features(provider, mode="race", year=2025, round_number=8, include_standings=False)

    assert grid.empty
    assert not current.empty
    assert set(current["grid_source"].dropna().astype(str)) == {"qualifying_fallback"}
    assert "retrospective_results_grid" not in set(current["grid_source"].dropna().astype(str))


def test_real_historical_training_grid_is_labeled_retrospective_not_pre_race() -> None:
    project_root = Path(__file__).resolve().parents[6]
    weekends_dir = project_root / "data" / "f1" / "raw" / "weekends"
    if not (weekends_dir / "2025" / "round_08_monaco_grand_prix").exists():
        pytest.skip("local F1 weekend fixture unavailable")
    provider = LocalWeekendProvider(weekends_dir=str(weekends_dir))

    train, _ = build_training_data(
        provider,
        mode="race",
        train_seasons=[2025],
        target_year=2026,
        target_round=1,
        include_standings=False,
    )

    monaco = train[pd.to_numeric(train["event_key"], errors="coerce") == 202508]
    assert not monaco.empty
    assert "retrospective_results_grid" in set(monaco["grid_source"].dropna().astype(str))
    assert "pre_race_official_grid" not in set(monaco["grid_source"].dropna().astype(str))


def test_real_circuit_summary_uses_finish_order_mobility_not_overtake_metric() -> None:
    project_root = Path(__file__).resolve().parents[6]
    weekends_dir = project_root / "data" / "f1" / "raw" / "weekends"
    if not (weekends_dir / "2025" / "round_08_monaco_grand_prix").exists():
        pytest.skip("local F1 weekend fixture unavailable")
    provider = LocalWeekendProvider(weekends_dir=str(weekends_dir))

    summary = provider._race_event_summary(2025, 8)

    assert summary is not None
    assert "finish_order_mobility" in summary
    assert "overtake_propensity" not in summary
    assert 0.0 <= float(summary["finish_order_mobility"]) <= 1.0


def test_race_fallback_is_grid_constrained_at_monaco_but_allows_monza_comeback() -> None:
    base = {
        "driver_id": ["grid_p1_slow", "grid_p4_fast", "mid_a", "mid_b"],
        "grid_position": [1.0, 4.0, 2.0, 3.0],
        "fp_race_sim_rank": [4.0, 1.0, 2.0, 3.0],
        "event_pace_index": [0.9, 0.1, 0.3, 0.6],
    }
    monaco = pd.DataFrame(
        {
            **base,
            "track_finish_order_mobility": [0.05] * 4,
            "circuit_drs_effectiveness": [0.20] * 4,
            "circuit_overtaking_difficulty": [0.98] * 4,
            "track_chaos_index": [0.10] * 4,
        }
    )
    monza = pd.DataFrame(
        {
            **base,
            "track_finish_order_mobility": [0.75] * 4,
            "circuit_drs_effectiveness": [0.90] * 4,
            "circuit_overtaking_difficulty": [0.25] * 4,
            "track_chaos_index": [0.10] * 4,
        }
    )

    monaco_score = _hierarchical_fallback(monaco, fallback_cols=["event_pace_index"])
    monza_score = _hierarchical_fallback(monza, fallback_cols=["event_pace_index"])

    assert monaco_score.iloc[0] < monaco_score.iloc[1]
    assert monza_score.iloc[1] < monza_score.iloc[0]


def test_rank_based_probabilities_preserve_event_totals_and_monotonicity() -> None:
    scores = pd.Series(range(1, 21), dtype=float)

    win = _rank_based_probability(scores, 1)
    top3 = _rank_based_probability(scores, 3)
    top10 = _rank_based_probability(scores, 10)

    assert abs(float(win.sum()) - 1.0) < 1e-9
    assert abs(float(top3.sum()) - 3.0) < 1e-9
    assert abs(float(top10.sum()) - 10.0) < 1e-9
    assert ((win <= top3 + 1e-12) & (top3 <= top10 + 1e-12)).all()


def test_calibrated_probability_path_is_renormalized_to_event_total() -> None:
    scores = pd.Series(range(1, 21), dtype=float)

    top10 = _predict_probability(DriftProbabilityModel(), scores, label="top10", k=10)

    assert abs(float(top10.sum()) - 10.0) < 1e-9
    assert (top10 <= 1.0).all()


def test_oof_qualifying_context_never_trains_on_same_event(monkeypatch) -> None:
    qual_train = pd.DataFrame(
        {
            "driver_id": ["a", "b", "a", "b", "a", "b"],
            "event_key": [202501, 202501, 202502, 202502, 202503, 202503],
            "event_pace_index": [0.1, 0.2, 0.2, 0.1, 0.3, 0.1],
            "fp_mean_rank": [1, 2, 2, 1, 2, 1],
        }
    )

    def fake_train_model(train, *_args, **_kwargs):
        return SimpleNamespace(model=None, model_name="fake", notes=[])

    monkeypatch.setattr("rqp.prediction.train_model", fake_train_model)
    signal = _build_oof_qualifying_signal_frame(
        qual_train=qual_train,
        qual_feature_cols=["event_pace_index", "fp_mean_rank"],
        qual_fallback_cols=["event_pace_index", "fp_mean_rank"],
        config=SimpleNamespace(
            enable_dl_candidates=False,
            compare_families=["ml"],
            dl_device="auto",
            dl_arch="mlp_tabular_v1",
            dl_hyperparams={},
            dl_seed=42,
            f1_model="auto",
        ),
        notes=[],
        min_prior_events=1,
    )

    trained = signal.dropna(subset=["qualy_pred_training_event_max"])
    assert not trained.empty
    assert (
        pd.to_numeric(trained["qualy_pred_training_event_max"], errors="coerce")
        < pd.to_numeric(trained["event_key"], errors="coerce")
    ).all()


def test_race_training_target_uses_grid_delta_and_reconstructs_finish_score() -> None:
    train = pd.DataFrame(
        {
            "driver_id": ["a", "b", "c"],
            "event_key": [202501, 202501, 202501],
            "grid_position": [1.0, 7.0, 13.0],
            "target": [1.0, 7.0, 13.0],
            "race_delta_target": [0.0, 0.0, 0.0],
            "pace": [0.2, 0.5, 0.8],
        }
    )
    spec = _infer_target_spec(train)
    candidate = CandidateSpec(
        name="mean_delta",
        build_model=MeanRegressor,
        task="regression",
        family="ml",
    )

    fitted = _fit_candidate(train, ["pace"], candidate, target_spec=spec)
    assert fitted is not None
    current = pd.DataFrame({"grid_position": [3.0, 11.0], "pace": [0.1, 0.9]})
    pred = pd.Series(fitted.predict(current), dtype=float)

    assert spec.train_col == "race_delta_target"
    assert pred.tolist() == [3.0, 11.0]


def test_strategic_race_delta_model_uses_race_features_not_grid_clone() -> None:
    train = pd.DataFrame(
        {
            "driver_id": ["front", "mover", "front", "mover"],
            "team_name": ["williams", "ferrari", "williams", "ferrari"],
            "event_key": [202601, 202601, 202602, 202602],
            "grid_position": [1.0, 2.0, 1.0, 2.0],
            "target": [4.0, 1.0, 3.0, 1.0],
            "race_delta_target": [3.0, -1.0, 2.0, -1.0],
            "circuit_card_id": ["monaco", "monaco", "monaco", "monaco"],
            "circuit_archetype": ["street_max_downforce"] * 4,
        },
    )
    current = pd.DataFrame(
        {
            "driver_id": ["front", "mover"],
            "team_name": ["williams", "ferrari"],
            "grid_position": [1.0, 2.0],
            "qualy_position": [1.0, 2.0],
            "fp_race_sim_rank": [2.0, 1.0],
            "fp_mean_rank": [2.0, 1.0],
            "fp_quali_sim_rank": [2.0, 1.0],
            "circuit_fit_index": [0.80, 0.20],
            "driver_archetype_form_3_fp_weighted_delta": [0.90, 0.10],
            "team_archetype_form_3_fp_weighted_delta": [0.90, 0.10],
            "driver_circuit_hist_fp_weighted_delta": [0.90, 0.10],
            "team_circuit_hist_fp_weighted_delta": [0.90, 0.10],
            "track_finish_order_mobility": [0.80, 0.80],
            "track_safety_car_prior": [0.40, 0.40],
            "track_chaos_index": [0.50, 0.50],
            "track_strategy_variance_prior": [0.50, 0.50],
            "track_dnf_prior": [0.10, 0.10],
            "circuit_downforce_demand": [1.0, 1.0],
            "circuit_power_sensitivity": [0.12, 0.12],
            "circuit_low_speed_corner_demand": [1.0, 1.0],
            "circuit_high_speed_corner_demand": [0.22, 0.22],
            "circuit_traction_demand": [0.86, 0.86],
            "circuit_braking_demand": [0.52, 0.52],
            "circuit_tyre_degradation": [0.32, 0.32],
        },
    )

    model = StrategicRaceDeltaModel().fit(train)
    pred = pd.Series(model.predict(current), index=current["driver_id"], dtype=float)

    assert pred["mover"] < pred["front"]
    assert pred.tolist() != current["grid_position"].tolist()


def test_race_baseline_request_selects_unified_strategic_model() -> None:
    train = pd.DataFrame(
        {
            "driver_id": ["a", "b", "a", "b", "a", "b"],
            "team_name": ["williams", "ferrari", "williams", "ferrari", "williams", "ferrari"],
            "event_key": [202601, 202601, 202602, 202602, 202603, 202603],
            "grid_position": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "target": [2.0, 1.0, 3.0, 1.0, 2.0, 1.0],
            "race_delta_target": [1.0, -1.0, 2.0, -1.0, 1.0, -1.0],
            "fp_race_sim_rank": [2.0, 1.0, 2.0, 1.0, 2.0, 1.0],
            "fp_mean_rank": [2.0, 1.0, 2.0, 1.0, 2.0, 1.0],
            "track_finish_order_mobility": [0.70] * 6,
            "circuit_overtaking_difficulty": [0.35] * 6,
        },
    )

    result = train_model(
        train,
        ["grid_position", "fp_race_sim_rank", "fp_mean_rank", "track_finish_order_mobility"],
        f1_model="baseline",
    )

    assert result.model_name == "strategic_race_delta"
    assert any("unified strategic_race_delta" in note for note in result.notes)


def test_trained_race_delta_wrapper_is_circuit_mobility_constrained() -> None:
    model = TargetOffsetModel(base_model=LargeDeltaModel(), base_column="grid_position", base_fill=10.0)
    grid = list(range(1, 21))
    monaco = pd.DataFrame(
        {
            "grid_position": grid,
            "track_finish_order_mobility": [0.05] * 20,
            "circuit_drs_effectiveness": [0.08] * 20,
            "circuit_overtaking_difficulty": [1.0] * 20,
            "race_generation_variance_prior": [0.10] * 20,
            "track_strategy_variance_prior": [0.20] * 20,
        }
    )
    monza = pd.DataFrame(
        {
            "grid_position": grid,
            "track_finish_order_mobility": [0.80] * 20,
            "circuit_drs_effectiveness": [0.90] * 20,
            "circuit_overtaking_difficulty": [0.25] * 20,
            "race_generation_variance_prior": [0.45] * 20,
            "track_strategy_variance_prior": [0.55] * 20,
        }
    )

    monaco_pred = pd.Series(model.predict(monaco), dtype=float)
    monza_pred = pd.Series(model.predict(monza), dtype=float)

    assert monaco_pred.iloc[0] <= 2.0
    assert monaco_pred.iloc[9] >= 9.0
    assert monza_pred.iloc[0] > monaco_pred.iloc[0]
    assert monza_pred.iloc[9] < monaco_pred.iloc[9]


def test_race_delta_wrapper_can_run_unconstrained_for_baseline_ladder() -> None:
    model = TargetOffsetModel(
        base_model=LargeDeltaModel(),
        base_column="grid_position",
        base_fill=10.0,
        constraint_mode="unconstrained",
    )
    frame = pd.DataFrame(
        {
            "grid_position": list(range(1, 21)),
            "track_finish_order_mobility": [0.01] * 20,
            "circuit_overtaking_difficulty": [1.0] * 20,
        }
    )

    pred = pd.Series(model.predict(frame), dtype=float)

    assert pred.iloc[0] == 7.0
    assert pred.iloc[9] == 4.0


def test_race_stochastic_layer_uses_reliability_strategy_dnf_priors() -> None:
    base = pd.DataFrame(
        {
            "driver_id": ["a", "b", "c", "d"],
            "grid_position": [1.0, 2.0, 3.0, 4.0],
            "track_finish_order_mobility": [0.20, 0.20, 0.20, 0.20],
            "fp_slow_lap_ratio": [0.0, 0.2, 0.4, 0.8],
            "fp_delta_std": [0.1, 0.2, 0.3, 0.4],
        }
    )
    low = base.assign(
        track_safety_car_prior=0.05,
        track_dnf_prior=0.02,
        track_strategy_variance_prior=0.05,
        track_weather_uncertainty_prior=0.02,
        race_generation_variance_prior=0.05,
    )
    high = base.assign(
        track_safety_car_prior=0.90,
        track_dnf_prior=0.35,
        track_strategy_variance_prior=0.90,
        track_weather_uncertainty_prior=0.80,
        race_generation_variance_prior=0.90,
    )
    preds = pd.Series([1.0, 2.5, 3.5, 4.0], index=base.index, dtype=float)

    low_out = _race_stochastic_score_layer(low, preds)
    high_out = _race_stochastic_score_layer(high, preds)

    assert float(high_out["race_stochastic_sigma"].mean()) > float(low_out["race_stochastic_sigma"].mean())
    assert float(high_out["race_stochastic_dnf_probability"].mean()) > float(
        low_out["race_stochastic_dnf_probability"].mean(),
    )
    high_raw_spread = float(high_out["race_stochastic_score"].max() - high_out["race_stochastic_score"].min())
    high_pl_spread = float(high_out["race_stochastic_pl_score"].max() - high_out["race_stochastic_pl_score"].min())
    assert high_pl_spread < high_raw_spread


def test_pl_temperature_is_fit_from_oof_event_likelihood() -> None:
    scores = pd.Series([1.0, 2.0, 3.0, 1.2, 2.2, 3.2, 1.4, 2.4, 3.4])
    actual = pd.Series([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    event_key = pd.Series([1, 1, 1, 2, 2, 2, 3, 3, 3])

    temperature, audit = _fit_pl_temperature_from_oof(scores, actual, event_key)

    assert temperature is not None
    assert audit["available"] is True
    assert audit["source"] == "walk_forward_oof"
    assert 0.1 <= float(temperature) <= 10.0


def test_oof_probability_audit_uses_deployed_pl_gumbel_layer() -> None:
    scores = pd.Series(list(range(1, 21)) + list(range(1, 21)), dtype=float)
    actual = pd.Series(list(range(1, 21)) + list(range(1, 21)), dtype=float)
    event_key = pd.Series([1] * 20 + [2] * 20)

    probabilities = _pl_probabilities_for_oof_audit(scores, event_key, 1.0, samples=1600, seed=17)
    audit = _probability_audit_from_oof(
        scores,
        actual,
        event_key,
        temperature=1.0,
        temperature_audit={"available": True, "source": "walk_forward_oof"},
        samples=1600,
        seed=17,
    )

    assert audit["schema_version"] == "pl_gumbel_probability_audit_v2"
    assert audit["probability_layer"] == "pl_gumbel"
    assert audit["same_probability_layer_as_production"] is True
    assert audit["samples"] == 1600
    assert audit["seed"] == 17
    assert audit["event_total_audit"]["passed"] is True
    assert audit["passed"] is False
    assert "insufficient_probability_audit_events" in audit["reason"]
    assert audit["thresholds"]["min_oof_events"] >= 5
    assert audit["metrics"]["win"]["bootstrap"]["available"] is True
    assert "brier_delta_ci95" in audit["metrics"]["win"]["bootstrap"]
    assert "ci_gate_passed" in audit["metrics"]["win"]
    for _, idx in event_key.groupby(event_key, sort=False).groups.items():
        event_prob = probabilities.loc[idx]
        assert abs(float(event_prob["win"].sum()) - 1.0) < 1e-9
        assert abs(float(event_prob["top3"].sum()) - 3.0) < 1e-9
        assert abs(float(event_prob["top10"].sum()) - 10.0) < 1e-9


def test_experiment_writer_rejects_stale_probability_audit_schema() -> None:
    payload = {
        "probability_audit": {
            "available": True,
            "passed": False,
            "source": "walk_forward_oof",
            "reason": "top10_calibration_failed",
            "metrics": {},
        }
    }

    with pytest.raises(RuntimeError, match="stale_probability_audit_schema"):
        _assert_probability_audit_schema(payload)


def test_oof_score_reconstruction_blocks_dl_backed_pace_blends() -> None:
    train = pd.DataFrame(
        {
            "event_key": [202501, 202501, 202502, 202502, 202503, 202503],
            "target": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "event_pace_index": [0.1, 0.2, 0.2, 0.1, 0.1, 0.2],
            "feature": [0.1, 0.3, 0.2, 0.4, 0.1, 0.3],
        }
    )
    dl_candidate = CandidateSpec(
        name="torch_mlp_tabular_v1",
        build_model=MeanRegressor,
        task="regression",
        family="dl",
    )

    scores, actual, event_key = _oof_scores_for_selection(
        train=train,
        feature_cols=["feature"],
        selected_name="pace_blend::torch_mlp_tabular_v1::event_pace_index::0.50",
        candidate_lookup={dl_candidate.name: dl_candidate},
        folds=[({202501}, 202502), ({202501, 202502}, 202503)],
        race_baseline_col="event_pace_index",
    )

    assert scores.empty
    assert actual.empty
    assert event_key.empty


def test_oof_probability_audit_fails_without_usable_calibration_slope() -> None:
    scores = pd.Series([0.0] * 20, dtype=float)
    actual = pd.Series(list(range(1, 11)) + list(range(1, 11)), dtype=float)
    event_key = pd.Series([1] * 10 + [2] * 10)

    audit = _probability_audit_from_oof(
        scores,
        actual,
        event_key,
        temperature=1.0,
        temperature_audit={"available": True, "source": "walk_forward_oof"},
    )

    assert audit["available"] is True
    assert audit["passed"] is False
    assert "calibration_failed" in audit["reason"]


def test_betting_uses_fair_edge_only_for_plausible_complete_market() -> None:
    predictions = pd.DataFrame(
        [
            {"driver_name": "Driver A", "driver_id": "a", "rank": 1, "proba_win": 0.49},
            {"driver_name": "Driver B", "driver_id": "b", "rank": 2, "proba_win": 0.18},
        ]
    )
    odds = pd.DataFrame(
        [
            {"market": "winner", "driver_name": "Driver A", "decimal_odds": 2.10, "bookmaker": "book"},
            {"market": "winner", "driver_name": "Driver B", "decimal_odds": 1.60, "bookmaker": "book"},
        ]
    )

    recommendations = build_betting_recommendations(
        predictions,
        odds,
        BettingConfig(
            min_edge=0.03,
            min_expected_roi=0.02,
            max_bet_fraction=0.01,
            require_probability_gate=False,
            require_oof_probability_audit=False,
            require_odds_timestamp=False,
            fair_market_min_selection_count=2,
        ),
    )
    driver_a = recommendations[recommendations["driver_name"] == "Driver A"].iloc[0]

    assert driver_a["edge_source"] == "fair_market"
    assert bool(driver_a["fair_edge_available"]) is True
    assert float(driver_a["probability_edge"]) < 0.03
    assert float(driver_a["edge_used"]) >= 0.03
    assert driver_a["status"] == "bet"


def test_betting_default_gate_blocks_incomplete_probability_sums() -> None:
    predictions = pd.DataFrame(
        [
            {"driver_name": "Driver A", "driver_id": "a", "rank": 1, "proba_win": 0.49},
            {"driver_name": "Driver B", "driver_id": "b", "rank": 2, "proba_win": 0.18},
        ]
    )
    odds = pd.DataFrame(
        [{"market": "winner", "driver_name": "Driver A", "decimal_odds": 2.10, "bookmaker": "book"}]
    )

    recommendations = build_betting_recommendations(
        predictions,
        odds,
        BettingConfig(min_edge=0.01, min_expected_roi=0.01),
    )
    row = recommendations.iloc[0]

    assert bool(row["probability_gate_passed"]) is False
    assert row["reject_reason"] == "probability_gate_failed"
    assert row["status"] == "skip"


def test_betting_default_gate_blocks_missing_oof_probability_audit_when_invariants_pass() -> None:
    predictions = pd.DataFrame(
        [
            {"driver_name": "Driver A", "driver_id": "a", "rank": 1, "proba_win": 0.60, "proba_top3": 1.0, "proba_top10": 1.0},
            {"driver_name": "Driver B", "driver_id": "b", "rank": 2, "proba_win": 0.40, "proba_top3": 1.0, "proba_top10": 1.0},
        ]
    )
    odds = pd.DataFrame(
        [{"market": "winner", "driver_name": "Driver A", "decimal_odds": 2.20, "bookmaker": "book"}]
    )

    recommendations = build_betting_recommendations(
        predictions,
        odds,
        BettingConfig(min_edge=0.01, min_expected_roi=0.01),
    )
    row = recommendations.iloc[0]

    assert bool(row["probability_gate_passed"]) is True
    assert bool(row["probability_audit_passed"]) is False
    assert row["reject_reason"] == "probability_audit_failed"
    assert row["status"] == "skip"


def test_betting_skips_when_complete_market_fair_edge_turns_raw_edge_negative() -> None:
    predictions = pd.DataFrame(
        [
            {
                "driver_name": f"Driver {idx}",
                "driver_id": str(idx).lower(),
                "rank": idx,
                "proba_win": 0.49 if idx == 1 else 0.51 / 9.0,
            }
            for idx in range(1, 11)
        ]
    )
    odds = pd.DataFrame(
        [
            {
                "market": "winner",
                "driver_name": "Driver 1",
                "decimal_odds": 2.10,
                "bookmaker": "book",
            },
            *[
                {
                    "market": "winner",
                    "driver_name": f"Driver {idx}",
                    "decimal_odds": 19.0,
                    "bookmaker": "book",
                }
                for idx in range(2, 11)
            ],
        ]
    )

    recommendations = build_betting_recommendations(
        predictions,
        odds,
        BettingConfig(
            min_edge=0.01,
            min_expected_roi=0.01,
            require_probability_gate=False,
            require_oof_probability_audit=False,
            require_odds_timestamp=False,
            fair_market_min_selection_count=10,
            fair_market_overround_min=0.90,
        ),
    )
    row = recommendations[recommendations["driver_name"] == "Driver 1"].iloc[0]

    assert bool(row["fair_edge_available"]) is True
    assert float(row["probability_edge"]) > 0.0
    assert float(row["fair_probability_edge"]) < 0.0
    assert row["edge_source"] == "fair_market"
    assert row["reject_reason"] == "edge_below_min"
    assert row["status"] == "skip"
