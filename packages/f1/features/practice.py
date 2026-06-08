"""Practice-session feature contracts for F1 models."""

from __future__ import annotations

import pandas as pd


PRACTICE_FEATURE_COLUMNS: tuple[str, ...] = (
    "fp1_delta",
    "fp2_delta",
    "fp3_delta",
    "fp_mean_delta",
    "fp_rank_mean",
    "fp_quali_sim_delta",
    "fp_race_sim_delta",
)


def available_practice_features(frame: pd.DataFrame) -> list[str]:
    """Return practice feature columns present in a feature frame."""

    return [column for column in PRACTICE_FEATURE_COLUMNS if column in frame.columns]


def has_practice_signal(frame: pd.DataFrame) -> bool:
    """Check whether a frame has at least one non-null practice pace feature."""

    columns = available_practice_features(frame)
    return bool(columns) and frame[columns].notna().any(axis=None)
