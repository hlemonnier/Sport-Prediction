"""Qualifying feature contracts for the F1 pre-quali and race models."""

from __future__ import annotations

import pandas as pd


QUALIFYING_CONTEXT_COLUMNS: tuple[str, ...] = (
    "qualy_position",
    "qualy_q3_time",
    "qualy_pred_position",
    "qualy_pred_rank",
    "qualy_position_track_adj",
    "qualy_context_position_track_adj",
)


def available_qualifying_context(frame: pd.DataFrame) -> list[str]:
    """Return official or predicted qualifying context columns present in a frame."""

    return [column for column in QUALIFYING_CONTEXT_COLUMNS if column in frame.columns]


def grid_source(frame: pd.DataFrame) -> str:
    """Classify whether race prediction uses official or predicted qualifying context."""

    if "grid_source" in frame.columns and frame["grid_source"].notna().any():
        return str(frame["grid_source"].dropna().iloc[0])
    if "qualy_position" in frame.columns and frame["qualy_position"].notna().any():
        return "official_qualifying"
    if "qualy_pred_rank" in frame.columns and frame["qualy_pred_rank"].notna().any():
        return "predicted_qualifying_grid"
    return "unknown"
