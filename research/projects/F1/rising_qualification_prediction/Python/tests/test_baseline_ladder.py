from __future__ import annotations

import pandas as pd

from run_f1_baseline_ladder import (
    _qualifying_ladder_specs,
    _race_ladder_specs,
    _score_grid_only,
    _summarize_paired_ladder,
)


def test_baseline_ladder_declares_required_race_variants() -> None:
    specs = _race_ladder_specs()
    names = [spec["name"] for spec in specs]

    assert names == [
        "grid_only",
        "strategic_race_delta",
    ]
    strategic = next(spec for spec in specs if spec["name"] == "strategic_race_delta")
    assert strategic.get("f1_model") == "baseline"
    assert strategic.get("disable_circuit_features") is False
    assert strategic.get("race_delta_constraint_mode") == "constrained"


def test_baseline_ladder_declares_required_qualifying_variants() -> None:
    names = [spec["name"] for spec in _qualifying_ladder_specs()]

    assert names == [
        "fp_weighted_rank_baseline",
        "fp_plus_driver_team_rolling_priors",
        "current_full_model",
        "current_full_model_plus_circuit_cards",
    ]


def test_baseline_ladder_summary_uses_identical_event_sets() -> None:
    rows = [
        {"mode": "race", "round": 1, "variant": "grid_only", "metric_available": True, "field_mae": 2.0, "top10_hit": 0.8},
        {"mode": "race", "round": 2, "variant": "grid_only", "metric_available": True, "field_mae": 4.0, "top10_hit": 0.7},
        {"mode": "race", "round": 1, "variant": "strategic_race_delta", "metric_available": True, "field_mae": 1.0, "top10_hit": 0.9},
        {"mode": "race", "round": 2, "variant": "strategic_race_delta", "metric_available": False, "field_mae": None, "top10_hit": None},
    ]

    summary = _summarize_paired_ladder(rows)

    assert summary["common_event_count"] == 1
    assert summary["variant_metrics"]["grid_only"]["field_mae_avg"] == 2.0
    assert summary["variant_metrics"]["strategic_race_delta"]["field_mae_avg"] == 1.0
    paired = summary["paired_comparisons"]["strategic_race_delta_vs_grid_only_field_mae"]
    assert paired["available"] is True
    assert paired["event_count"] == 1
    assert paired["mean_improvement"] == 1.0
    assert paired["improved_events"] == 1
    assert paired["worse_events"] == 0
    assert paired["sign_test_p_value_two_sided"] == 1.0
    assert paired["event_deltas"][0]["raw_delta_challenger_minus_baseline"] == -1.0


def test_baseline_ladder_summary_labels_duplicate_config_aliases() -> None:
    signature = {
        "kind": "model",
        "f1_model": "baseline",
        "disable_circuit_features": True,
        "race_delta_constraint_mode": "constrained",
    }
    rows = [
        {
            "mode": "race",
            "round": 1,
            "variant": "grid_delta_constrained",
            "metric_available": True,
            "field_mae": 2.0,
            "top10_hit": 0.8,
            "config_signature": signature,
        },
        {
            "mode": "race",
            "round": 1,
            "variant": "current_full_model",
            "metric_available": True,
            "field_mae": 2.0,
            "top10_hit": 0.8,
            "config_signature": signature,
        },
    ]

    summary = _summarize_paired_ladder(rows)

    assert summary["configuration_aliases"] == [
        {
            "variants": ["current_full_model", "grid_delta_constrained"],
            "config_signature": signature,
        }
    ]


def test_baseline_ladder_summary_pairs_events_separately_by_mode() -> None:
    rows = [
        {"mode": "qualifying", "round": 1, "variant": "q_a", "metric_available": True, "field_mae": 2.0, "top10_hit": 0.8},
        {"mode": "qualifying", "round": 1, "variant": "q_b", "metric_available": True, "field_mae": 1.0, "top10_hit": 0.9},
        {"mode": "race", "round": 1, "variant": "r_a", "metric_available": True, "field_mae": 4.0, "top10_hit": 0.7},
        {"mode": "race", "round": 1, "variant": "r_b", "metric_available": True, "field_mae": 3.0, "top10_hit": 0.8},
    ]

    summary = _summarize_paired_ladder(rows)

    assert summary["common_event_count_by_mode"] == {"qualifying": 1, "race": 1}
    assert summary["mode_summaries"]["qualifying"]["variant_metrics"]["q_b"]["field_mae_avg"] == 1.0
    assert summary["mode_summaries"]["race"]["variant_metrics"]["r_b"]["field_mae_avg"] == 3.0


class PostRaceGridProvider:
    def get_starting_grid(self, year: int, round_number: int) -> pd.DataFrame:
        return pd.DataFrame()

    def get_qualifying_results(self, year: int, round_number: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "driver_id": ["a", "b", "c"],
                "driver_name": ["A", "B", "C"],
                "position": [1.0, 2.0, 3.0],
            },
        )

    def get_race_results(self, year: int, round_number: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "driver_id": ["a", "b", "c"],
                "driver_name": ["A", "B", "C"],
                "position": [1.0, 2.0, 3.0],
                "grid_position": [9.0, 8.0, 7.0],
            },
        )


def test_grid_only_ladder_uses_pre_race_grid_or_qualifying_fallback_not_post_race_grid() -> None:
    rows = _score_grid_only(PostRaceGridProvider(), 2026, 1)  # type: ignore[arg-type]

    assert [row["driver_id"] for row in rows] == ["a", "b", "c"]
    assert [row["pred"] for row in rows] == [1.0, 2.0, 3.0]
    assert {row["grid_source"] for row in rows} == {"qualifying_fallback"}


class PreRaceGridProvider(PostRaceGridProvider):
    def get_starting_grid(self, year: int, round_number: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "driver_id": ["a", "b", "c"],
                "driver_name": ["A", "B", "C"],
                "grid_position": [3.0, 1.0, 2.0],
                "grid_source": ["pre_race_official_grid"] * 3,
                "grid_status": ["grid", "grid", "grid"],
            },
        )


def test_grid_ladder_rows_preserve_grid_source_labels() -> None:
    rows = _score_grid_only(PreRaceGridProvider(), 2026, 1)  # type: ignore[arg-type]

    assert [row["driver_id"] for row in rows] == ["b", "c", "a"]
    assert {row["grid_source"] for row in rows} == {"pre_race_official_grid"}
    assert {row["grid_status"] for row in rows} == {"grid"}
