"""Race feature contracts for pre-race F1 prediction."""

from __future__ import annotations

import pandas as pd


RACE_CONTEXT_COLUMNS: tuple[str, ...] = (
    "grid_position",
    "grid_status",
    "fp_race_sim_delta",
    "fp_race_sim_delta_track_adj",
    "track_grid_stability",
    "track_chaos_index",
    "track_finish_order_mobility",
    "race_generation_variance_prior",
)


def available_race_context(frame: pd.DataFrame) -> list[str]:
    """Return race-context columns present in a feature frame."""

    return [column for column in RACE_CONTEXT_COLUMNS if column in frame.columns]


def has_grid_signal(frame: pd.DataFrame) -> bool:
    """Check whether the race model has a usable grid column."""

    return "grid_position" in frame.columns and frame["grid_position"].notna().any()
