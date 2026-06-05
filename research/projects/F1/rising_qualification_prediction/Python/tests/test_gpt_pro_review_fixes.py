from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from rqp.betting import BettingConfig, build_betting_recommendations
from rqp.data import _add_temporal_features_train, build_current_features
from rqp.evaluation import evaluate_prediction_rows
from rqp.prediction import (
    _build_oof_qualifying_signal_frame,
    _hierarchical_fallback,
    _predict_probability,
    _rank_based_probability,
)
from rqp.providers import LocalWeekendProvider
from rqp.training import (
    CandidateSpec,
    TargetOffsetModel,
    _fit_candidate,
    _fit_pl_temperature_from_oof,
    _infer_target_spec,
    _probability_audit_from_oof,
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
                "grid_position": [1.0, np.nan, 7.0],
            }
        )

    def get_track_stats(self, year, round_number):
        return {}


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
    assert by_driver.loc["a", "grid_source"] == "official_grid"
    assert float(by_driver.loc["a", "grid_position"]) == 1.0
    assert by_driver.loc["b", "grid_source"] == "qualifying_fallback"
    assert float(by_driver.loc["b", "grid_position"]) == 2.0
    assert by_driver.loc["c", "grid_source"] == "official_grid"
    assert float(by_driver.loc["c", "grid_position"]) == 7.0


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
            "track_overtake_propensity": [0.05] * 4,
            "circuit_drs_effectiveness": [0.20] * 4,
            "circuit_overtaking_difficulty": [0.98] * 4,
            "track_chaos_index": [0.10] * 4,
        }
    )
    monza = pd.DataFrame(
        {
            **base,
            "track_overtake_propensity": [0.75] * 4,
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


def test_trained_race_delta_wrapper_is_circuit_mobility_constrained() -> None:
    model = TargetOffsetModel(base_model=LargeDeltaModel(), base_column="grid_position", base_fill=10.0)
    grid = list(range(1, 21))
    monaco = pd.DataFrame(
        {
            "grid_position": grid,
            "track_overtake_propensity": [0.05] * 20,
            "circuit_drs_effectiveness": [0.08] * 20,
            "circuit_overtaking_difficulty": [1.0] * 20,
            "race_generation_variance_prior": [0.10] * 20,
            "track_strategy_variance_prior": [0.20] * 20,
        }
    )
    monza = pd.DataFrame(
        {
            "grid_position": grid,
            "track_overtake_propensity": [0.80] * 20,
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


def test_pl_temperature_is_fit_from_oof_event_likelihood() -> None:
    scores = pd.Series([1.0, 2.0, 3.0, 1.2, 2.2, 3.2, 1.4, 2.4, 3.4])
    actual = pd.Series([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    event_key = pd.Series([1, 1, 1, 2, 2, 2, 3, 3, 3])

    temperature, audit = _fit_pl_temperature_from_oof(scores, actual, event_key)

    assert temperature is not None
    assert audit["available"] is True
    assert audit["source"] == "walk_forward_oof"
    assert 0.1 <= float(temperature) <= 10.0


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
