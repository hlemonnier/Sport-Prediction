"""Shared helpers for data handling."""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


def first_available(df: pd.DataFrame, columns: Iterable[str]) -> Optional[str]:
    for col in columns:
        if col in df.columns:
            return col
    return None


def merge_fp_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    merged = None
    for frame in frames:
        label = frame["session"].iloc[0].lower()
        frame = frame.copy()
        frame = frame.rename(columns={
            "delta": f"{label}_delta",
            "rank": f"{label}_rank",
        })
        frame = frame.drop(columns=["session"])
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=["driver_id", "driver_name"], how="outer")
    if merged is None:
        return pd.DataFrame()
    delta_cols = [c for c in merged.columns if c.endswith("_delta")]
    rank_cols = [c for c in merged.columns if c.endswith("_rank")]
    merged["fp_mean_delta"] = merged[delta_cols].mean(axis=1, skipna=True)
    merged["fp_mean_rank"] = merged[rank_cols].mean(axis=1, skipna=True)
    return merged


def format_prediction_table(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy().sort_values("pred", ascending=True).head(top_n)
    df["rank"] = range(1, len(df) + 1)
    return df[["rank", "driver_name", "pred"]]
