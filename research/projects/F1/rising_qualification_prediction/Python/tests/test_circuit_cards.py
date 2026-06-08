from __future__ import annotations

import sys
import types

import pandas as pd

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.RequestException = Exception
    sys.modules["requests"] = requests_stub

from packages.f1.data.schemas.circuit import circuit_card_from_event
from packages.f1.data.schemas.session import PredictionConfig
from packages.f1.features.assembly import _attach_temporal_features_current, _attach_track_stats
from packages.f1.orchestration.prediction import _qualifying_feature_sets, _race_feature_sets
from run_circuit_card_ablation import _decision_from_summary


class StubTrackProvider:
    def get_track_stats(self, year: int, round_number: int) -> dict[str, float]:
        return {
            "track_finish_order_mobility": 0.08,
            "track_grid_stability": 0.92,
            "track_safety_car_propensity": 0.70,
            "track_dnf_rate": 0.20,
            "track_pit_stop_intensity": 1.20,
            "track_weather_uncertainty": 0.30,
            "track_stats_reliability": 0.80,
            "track_chaos_index": 0.55,
        }


class NoStatsProvider:
    def get_track_stats(self, year: int, round_number: int) -> dict[str, float]:
        return {}


def test_monaco_card_encodes_high_downforce_low_overtaking_profile() -> None:
    card = circuit_card_from_event("Monaco Grand Prix")

    assert card.card_id == "monaco"
    assert card.downforce_demand > 0.95
    assert card.power_sensitivity < 0.20
    assert card.overtaking_difficulty > 0.95
    assert card.qualifying_importance > 0.90


def test_accented_event_names_match_static_cards() -> None:
    card = circuit_card_from_event("São Paulo Grand Prix")

    assert card.card_id == "interlagos"
    assert card.archetype == "balanced_bumpy_sprint"


def test_track_stats_refine_circuit_card_without_losing_static_identity() -> None:
    card = circuit_card_from_event(
        "Monaco Grand Prix",
        {
            "track_finish_order_mobility": 0.20,
            "track_grid_stability": 0.80,
            "track_safety_car_propensity": 0.90,
            "track_stats_reliability": 1.00,
        },
    )

    assert card.card_id == "monaco"
    assert 0.80 <= card.overtaking_difficulty <= 1.00
    assert card.safety_car_probability > 0.75
    assert card.data_reliability == 1.00


def test_attach_track_stats_adds_card_features_and_interactions() -> None:
    frame = pd.DataFrame(
        {
            "driver_id": ["a", "b"],
            "event_name": ["Monaco Grand Prix", "Monaco Grand Prix"],
            "fp_weighted_delta": [0.10, 0.40],
            "fp_quali_sim_delta": [0.05, 0.30],
            "fp_race_sim_delta": [0.20, 0.60],
            "qualy_position": [1.0, 5.0],
        }
    )

    out = _attach_track_stats(frame, StubTrackProvider(), year=2026, round_number=8, notes=[])

    assert set(out["circuit_card_id"]) == {"monaco"}
    assert float(out["circuit_downforce_demand"].iloc[0]) > 0.95
    assert float(out["track_qualy_importance"].iloc[0]) > 0.75
    assert "track_safety_car_prior" in out.columns
    assert "track_finish_order_mobility" in out.columns
    assert "track_overtake_propensity" in out.columns
    assert float(out["track_finish_order_mobility"].iloc[0]) == float(out["track_overtake_propensity"].iloc[0])
    assert "track_dnf_prior" in out.columns
    assert "track_strategy_variance_prior" in out.columns
    assert "track_weather_uncertainty_prior" in out.columns
    assert "race_generation_variance_prior" in out.columns
    assert "fp_weighted_delta_downforce_adj" in out.columns
    assert "qualy_position_circuit_importance_adj" in out.columns
    assert "circuit_fit_index" in out.columns


def test_prediction_config_quarantines_circuit_features_by_default() -> None:
    config = PredictionConfig(
        source="local",
        mode="race",
        year=2026,
        round_number=1,
        train_seasons=[2024, 2025],
        include_standings=False,
        cache_dir=None,
        meeting_name=None,
        country_name=None,
        weekends_dir=None,
    )

    assert config.disable_circuit_features is True


def test_feature_sets_consume_circuit_card_columns() -> None:
    qualifying_features, qualifying_fallback = _qualifying_feature_sets()
    race_features, race_fallback = _race_feature_sets(include_standings=False)
    qualifying_no_cards, _ = _qualifying_feature_sets(disable_circuit=True)
    race_no_cards, _ = _race_feature_sets(include_standings=False, disable_circuit=True)

    for columns in [qualifying_features, qualifying_fallback, race_features, race_fallback]:
        assert "circuit_downforce_demand" in columns
        assert "circuit_power_sensitivity" in columns
        assert "team_archetype_form_3_fp_weighted_delta" in columns
        assert "driver_circuit_hist_fp_weighted_delta" in columns
    for columns in [race_features, race_fallback]:
        assert "track_safety_car_prior" in columns
        assert "track_dnf_prior" in columns
        assert "track_strategy_variance_prior" in columns
        assert "track_weather_uncertainty_prior" in columns
        assert "race_generation_variance_prior" in columns
    assert "circuit_downforce_demand" not in qualifying_no_cards
    assert "circuit_downforce_demand" not in race_no_cards
    assert "driver_circuit_hist_fp_weighted_delta" not in race_no_cards
    assert "track_qualy_importance" in race_no_cards


def test_circuit_fit_index_changes_relative_order_across_circuit_traits() -> None:
    base = pd.DataFrame(
        {
            "driver_id": ["quali_car", "race_car"],
            "fp_weighted_delta": [0.40, 0.40],
            "fp_quali_sim_delta": [0.10, 1.00],
            "fp_race_sim_delta": [1.00, 0.10],
            "fp_delta_std": [0.20, 0.20],
            "event_name": ["Monaco Grand Prix", "Monaco Grand Prix"],
        }
    )
    monaco = _attach_track_stats(base, NoStatsProvider(), year=2026, round_number=8, notes=[])
    monza_frame = base.copy()
    monza_frame["event_name"] = "Italian Grand Prix"
    monza = _attach_track_stats(monza_frame, NoStatsProvider(), year=2026, round_number=16, notes=[])

    monaco_fit = monaco.set_index("driver_id")["circuit_fit_index"]
    monza_fit = monza.set_index("driver_id")["circuit_fit_index"]

    assert float(monaco_fit["quali_car"]) < float(monaco_fit["race_car"])
    assert float(monza_fit["race_car"]) < float(monza_fit["quali_car"])


def test_current_features_map_team_and_driver_archetype_history() -> None:
    history = pd.DataFrame(
        {
            "driver_id": ["a", "a", "b"],
            "team_name": ["red", "red", "blue"],
            "event_name": ["Monaco Grand Prix", "Italian Grand Prix", "Monaco Grand Prix"],
            "event_key": [202408, 202416, 202408],
            "event_year": [2024, 2024, 2024],
            "event_round": [8, 16, 8],
            "circuit_card_id": ["monaco", "monza", "monaco"],
            "circuit_archetype": ["street_max_downforce", "temple_of_speed", "street_max_downforce"],
            "fp_weighted_delta": [0.12, 0.95, 0.80],
        }
    )
    current = pd.DataFrame(
        {
            "driver_id": ["a"],
            "team_name": ["red"],
            "event_name": ["Monaco Grand Prix"],
            "event_key": [202608],
            "circuit_card_id": ["monaco"],
            "circuit_archetype": ["street_max_downforce"],
            "fp_weighted_delta": [0.30],
        }
    )

    out = _attach_temporal_features_current(current, history)

    assert float(out["driver_archetype_form_3_fp_weighted_delta"].iloc[0]) == 0.12
    assert float(out["team_archetype_form_3_fp_weighted_delta"].iloc[0]) == 0.12
    assert float(out["driver_circuit_hist_fp_weighted_delta"].iloc[0]) == 0.12
    assert float(out["team_circuit_hist_fp_weighted_delta"].iloc[0]) == 0.12


def test_zero_effect_circuit_card_ablation_decision_is_quarantine() -> None:
    decision = _decision_from_summary(
        {
            "available": True,
            "paired_deltas": {
                "mae_with_minus_without": {
                    "mean": 0.0,
                    "ci95_low": 0.0,
                    "ci95_high": 0.0,
                },
                "top10_with_minus_without": {"mean": 0.0},
            },
        }
    )

    assert decision["state"] == "quarantine"
