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
