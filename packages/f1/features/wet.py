"""Causal wet-practice feature interactions shared by train and prediction."""

from __future__ import annotations

import pandas as pd


WET_PACE_FEATURES = (
    "fp_wet_sim_delta",
    "fp_wet_sim_rank",
    "fp_wet_sim_laps",
    "wet_sim_sessions_available",
    "fp_wet_sim_delta_weather_adj",
    "fp_wet_sim_rank_weather_adj",
)


def add_f1_wet_pace_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach pre-event wet-pace x wet-risk features without target data."""

    if frame.empty:
        return frame
    out = frame.copy()
    risk = pd.to_numeric(
        out.get(
            "track_weather_uncertainty_prior",
            out.get("track_weather_uncertainty", pd.Series(0.0, index=out.index)),
        ),
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0)
    laps = pd.to_numeric(
        out.get("fp_wet_sim_laps", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    sessions = pd.to_numeric(
        out.get("wet_sim_sessions_available", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    reliability = ((laps / 12.0).clip(0.0, 1.0) * 0.65) + ((sessions / 3.0).clip(0.0, 1.0) * 0.35)
    exposure = risk * reliability

    wet_delta = pd.to_numeric(
        out.get("fp_wet_sim_delta", pd.Series(float("nan"), index=out.index)),
        errors="coerce",
    )
    wet_rank = pd.to_numeric(
        out.get("fp_wet_sim_rank", pd.Series(float("nan"), index=out.index)),
        errors="coerce",
    )
    out["fp_wet_sim_delta_weather_adj"] = wet_delta * exposure
    out["fp_wet_sim_rank_weather_adj"] = wet_rank * exposure
    out["wet_pace_evidence_reliability"] = reliability
    return out


def wet_pace_evidence_rows(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    delta = pd.to_numeric(
        frame.get("fp_wet_sim_delta", pd.Series(float("nan"), index=frame.index)),
        errors="coerce",
    )
    rank = pd.to_numeric(
        frame.get("fp_wet_sim_rank", pd.Series(float("nan"), index=frame.index)),
        errors="coerce",
    )
    laps = pd.to_numeric(
        frame.get("fp_wet_sim_laps", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    sessions = pd.to_numeric(
        frame.get("wet_sim_sessions_available", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    return int(((delta.notna() | rank.notna()) & (laps.gt(0.0) | sessions.gt(0.0))).sum())
