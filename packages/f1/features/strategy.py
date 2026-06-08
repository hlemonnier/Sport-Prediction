"""Strategy feature helpers for race and live-race F1 models."""

from __future__ import annotations

import pandas as pd


def strategy_volatility_score(frame: pd.DataFrame) -> pd.Series:
    """Combine track and weather variance priors into a bounded strategy score."""

    if frame.empty:
        return pd.Series(dtype=float)
    index = frame.index
    score = pd.Series(0.0, index=index, dtype=float)
    for column, weight in (
        ("track_chaos_index", 0.45),
        ("track_safety_car_probability", 0.25),
        ("race_generation_variance_prior", 0.20),
        ("open_meteo_wet_risk", 0.10),
    ):
        if column not in frame.columns:
            continue
        score = score + pd.to_numeric(frame[column], errors="coerce").fillna(0.0) * weight
    return score.clip(lower=0.0, upper=1.0)


def attach_strategy_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach race strategy volatility features to a feature frame."""

    if frame.empty:
        return frame
    out = frame.copy()
    out["strategy_volatility_score"] = strategy_volatility_score(out)
    return out
