from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


_MODULE_PATH = Path(__file__).resolve().parents[1] / "run_horizon_a_vs_b_lap_snapshots.py"
_PYTHON_ROOT = str(_MODULE_PATH.parent)
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)
_MODULE_SPEC = importlib.util.spec_from_file_location("run_horizon_a_vs_b_lap_snapshots", _MODULE_PATH)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
snapshots = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(snapshots)


def test_distance_cutoff_plan_maps_percentages_to_real_laps() -> None:
    plan = snapshots._build_cutoff_plan(
        cutoff_mode="distance_pct",
        total_laps_completed=53,
        pct_cutoffs=[5, 10, 20],
        lap_cutoffs=[5, 10],
    )
    assert [row["cutoff_label"] for row in plan] == ["5%", "10%", "20%"]
    assert [row["lap_cutoff"] for row in plan] == [3, 6, 11]
    assert abs(float(plan[0]["cutoff_pct_realized"]) - (300.0 / 53.0)) < 1e-9


def test_chaos_profile_counts_sc_vsc_red_laps() -> None:
    trace = pd.DataFrame(
        {
            "lap_number": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "track_status": ["1", "1", "4", "1", "6", "6", "5", "1", "2", "2"],
        }
    )
    profile = snapshots._build_chaos_profile(
        trace,
        total_laps_completed=5,
        clean_max_chaos_fraction=0.02,
        chaotic_min_chaos_fraction=0.05,
    )
    assert int(profile["sc_vsc_laps"]) == 3
    assert abs(float(profile["chaos_fraction"]) - 0.6) < 1e-9
    assert bool(profile["has_sc_vsc_or_red"]) is True
    assert profile["chaos_segment"] == "chaotic"


def test_crossover_with_epsilon_and_never_bucket() -> None:
    per_round = pd.DataFrame(
        [
            {
                "round": 1,
                "cutoff_mode": "distance_pct",
                "cutoff_label": "5%",
                "cutoff_sort_value": 5.0,
                "lap_cutoff": 3,
                "cutoff_pct_realized": 6.0,
                "total_laps_completed": 50,
                "sc_vsc_laps": 0,
                "chaos_fraction": 0.0,
                "chaos_segment": "clean",
                "chaos_has_incident": False,
                "A_mae": 2.9,
                "B_mae": 3.1,
            },
            {
                "round": 1,
                "cutoff_mode": "distance_pct",
                "cutoff_label": "10%",
                "cutoff_sort_value": 10.0,
                "lap_cutoff": 5,
                "cutoff_pct_realized": 10.0,
                "total_laps_completed": 50,
                "sc_vsc_laps": 0,
                "chaos_fraction": 0.0,
                "chaos_segment": "clean",
                "chaos_has_incident": False,
                "A_mae": 2.9,
                "B_mae": 2.85,
            },
            {
                "round": 1,
                "cutoff_mode": "distance_pct",
                "cutoff_label": "20%",
                "cutoff_sort_value": 20.0,
                "lap_cutoff": 10,
                "cutoff_pct_realized": 20.0,
                "total_laps_completed": 50,
                "sc_vsc_laps": 0,
                "chaos_fraction": 0.0,
                "chaos_segment": "clean",
                "chaos_has_incident": False,
                "A_mae": 2.9,
                "B_mae": 2.79,
            },
            {
                "round": 2,
                "cutoff_mode": "distance_pct",
                "cutoff_label": "5%",
                "cutoff_sort_value": 5.0,
                "lap_cutoff": 3,
                "cutoff_pct_realized": 6.0,
                "total_laps_completed": 50,
                "sc_vsc_laps": 5,
                "chaos_fraction": 0.1,
                "chaos_segment": "chaotic",
                "chaos_has_incident": True,
                "A_mae": 2.9,
                "B_mae": 3.05,
            },
            {
                "round": 2,
                "cutoff_mode": "distance_pct",
                "cutoff_label": "10%",
                "cutoff_sort_value": 10.0,
                "lap_cutoff": 5,
                "cutoff_pct_realized": 10.0,
                "total_laps_completed": 50,
                "sc_vsc_laps": 5,
                "chaos_fraction": 0.1,
                "chaos_segment": "chaotic",
                "chaos_has_incident": True,
                "A_mae": 2.9,
                "B_mae": 2.95,
            },
            {
                "round": 2,
                "cutoff_mode": "distance_pct",
                "cutoff_label": "20%",
                "cutoff_sort_value": 20.0,
                "lap_cutoff": 10,
                "cutoff_pct_realized": 20.0,
                "total_laps_completed": 50,
                "sc_vsc_laps": 5,
                "chaos_fraction": 0.1,
                "chaos_segment": "chaotic",
                "chaos_has_incident": True,
                "A_mae": 2.9,
                "B_mae": 2.85,
            },
        ]
    )
    crossover = snapshots._build_crossover_per_round(
        per_round,
        metric_specs=[
            {
                "metric": "mae",
                "a_col": "A_mae",
                "b_col": "B_mae",
                "higher_is_better": False,
                "epsilon": 0.1,
            }
        ],
    )

    mae = crossover.set_index("round")
    assert bool(mae.loc[1, "crossover_found"]) is True
    assert mae.loc[1, "crossover_bucket"] == "20%"
    assert bool(mae.loc[2, "crossover_found"]) is False
    assert mae.loc[2, "crossover_bucket"] == snapshots.NEVER_BEFORE_FINISH

    distribution = snapshots._crossover_distribution(crossover)
    dist_all = distribution[(distribution["metric"] == "mae") & (distribution["chaos_segment"] == "all")]
    bucket_to_rounds = dict(zip(dist_all["crossover_bucket"], dist_all["rounds"]))
    assert bucket_to_rounds["20%"] == 1
    assert bucket_to_rounds[snapshots.NEVER_BEFORE_FINISH] == 1

    survival = snapshots._crossover_survival_curve(crossover, per_round)
    surv_all = survival[(survival["metric"] == "mae") & (survival["chaos_segment"] == "all")].set_index("cutoff_label")
    assert abs(float(surv_all.loc["10%", "crossed_share"]) - 0.0) < 1e-9
    assert abs(float(surv_all.loc["20%", "crossed_share"]) - 0.5) < 1e-9


def test_round_observability_builds_core_tables() -> None:
    trace = pd.DataFrame(
        [
            {
                "driver_id": "1",
                "driver_name": "A",
                "lap_number": 1,
                "timestamp": 1.0,
                "baseline_lap": 90.0,
                "lap_time_seconds": 90.5,
                "one_step_pred_mean": 90.3,
                "one_step_pred_std": 0.4,
                "innovation": 0.2,
                "innovation_var": 0.16,
                "race_time_seconds": 90.5,
                "tyre_age": 1,
                "deg_rate_mean": 0.04,
                "pace_penalty_mean": 0.2,
                "pace_penalty_std": 0.3,
                "deg_rate_std": 0.1,
                "eval_included": True,
                "is_box_lap": False,
                "is_accurate": True,
                "gate_skip_update": False,
                "robust_applied": False,
                "reset_applied": False,
                "is_red": False,
                "is_sc_vsc": False,
                "is_yellow": False,
                "compound": "MEDIUM",
                "gate_mode": "green_nominal",
                "stint_id": 1,
            },
            {
                "driver_id": "1",
                "driver_name": "A",
                "lap_number": 2,
                "timestamp": 2.0,
                "baseline_lap": 90.2,
                "lap_time_seconds": 90.9,
                "one_step_pred_mean": 90.6,
                "one_step_pred_std": 0.5,
                "innovation": 0.3,
                "innovation_var": 0.25,
                "race_time_seconds": 181.4,
                "tyre_age": 2,
                "deg_rate_mean": 0.05,
                "pace_penalty_mean": 0.3,
                "pace_penalty_std": 0.3,
                "deg_rate_std": 0.1,
                "eval_included": True,
                "is_box_lap": False,
                "is_accurate": True,
                "gate_skip_update": False,
                "robust_applied": True,
                "reset_applied": True,
                "is_red": False,
                "is_sc_vsc": False,
                "is_yellow": True,
                "compound": "MEDIUM",
                "gate_mode": "yellow_inflate",
                "stint_id": 2,
            },
            {
                "driver_id": "2",
                "driver_name": "B",
                "lap_number": 1,
                "timestamp": 1.0,
                "baseline_lap": 90.0,
                "lap_time_seconds": 91.2,
                "one_step_pred_mean": 90.8,
                "one_step_pred_std": 0.5,
                "innovation": 0.4,
                "innovation_var": 0.25,
                "race_time_seconds": 91.2,
                "tyre_age": 1,
                "deg_rate_mean": 0.03,
                "pace_penalty_mean": 0.4,
                "pace_penalty_std": 0.35,
                "deg_rate_std": 0.1,
                "eval_included": False,
                "is_box_lap": False,
                "is_accurate": True,
                "gate_skip_update": True,
                "robust_applied": False,
                "reset_applied": False,
                "is_red": False,
                "is_sc_vsc": True,
                "is_yellow": False,
                "compound": "HARD",
                "gate_mode": "skip_sc_vsc_red",
                "stint_id": 1,
            },
        ]
    )

    out = snapshots._build_round_observability(
        round_number=6,
        trace=trace,
        horizon_laps=10,
        pit_window_laps=3,
    )
    assert not out["a1_lap_availability"].empty
    assert not out["b7_update_behavior"].empty
    assert not out["c10_interval_coverage"].empty
    assert not out["e16_pit_hazard_curve"].empty
    assert not out["e19_strategy_posterior"].empty


def test_cutoff_observability_uses_position_and_pairwise_payloads() -> None:
    snapshot = pd.DataFrame(
        [
            {"driver_id": "1", "driver_name": "A", "rank": 1, "exp_pos_H": 1.2, "p_win_H": 0.55, "p_top3_H": 0.9, "p_top10_H": 1.0},
            {"driver_id": "2", "driver_name": "B", "rank": 2, "exp_pos_H": 2.1, "p_win_H": 0.30, "p_top3_H": 0.85, "p_top10_H": 1.0},
            {"driver_id": "3", "driver_name": "C", "rank": 3, "exp_pos_H": 3.4, "p_win_H": 0.15, "p_top3_H": 0.60, "p_top10_H": 1.0},
        ]
    )
    dist_summary = {
        "position_probabilities": [
            {"driver_id": "1", "position": 1, "probability": 0.55},
            {"driver_id": "1", "position": 2, "probability": 0.30},
            {"driver_id": "2", "position": 1, "probability": 0.30},
            {"driver_id": "3", "position": 3, "probability": 0.60},
        ],
        "pairwise_ahead_probabilities": [
            {"driver_a": "1", "driver_b": "2", "probability_a_ahead_b": 0.7},
            {"driver_a": "2", "driver_b": "1", "probability_a_ahead_b": 0.3},
        ],
        "mc_samples_requested": 200,
        "mc_samples_effective": 180,
        "sum_p_win": 1.0,
    }
    cutoff = {
        "cutoff_mode": "distance_pct",
        "cutoff_label": "20%",
        "cutoff_sort_value": 20.0,
        "lap_cutoff": 10,
        "cutoff_pct_realized": 20.0,
    }

    out = snapshots._build_cutoff_observability(
        round_number=6,
        cutoff=cutoff,
        snapshot=snapshot,
        dist_summary=dist_summary,
    )
    assert not out["d12_position_distribution_top5"].empty
    assert not out["d13_pairwise_ahead_top10"].empty
    assert not out["d14_ranking_curve"].empty
    assert float(out["d15_mc_health"].iloc[0]["mc_samples_effective"]) == 180.0
