"""Shared race score transformation used by calibration and production.

The model prediction is already a finish-position score.  This layer adds the
expected reliability penalty and converts event-level uncertainty into the
score scale consumed by Plackett-Luce/Gumbel sampling.  It deliberately does
not shrink predicted movement by circuit mobility: race models and their
target-offset wrappers own that mean constraint, so applying it again here
would double-count the same prior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


RACE_PROBABILITY_SCORE_LAYER = "race_stochastic_pl_score_v2"


def _first_numeric_series(frame: pd.DataFrame, columns: list[str]) -> pd.Series | None:
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            return values
    return None


def _rank_percentile(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(0.5, index=numeric.index, dtype=float)
    return numeric.rank(method="average", pct=True)


def race_stochastic_score_layer(features: pd.DataFrame, predictions: pd.Series) -> pd.DataFrame:
    """Transform finish-position means into the exact deployed PL score.

    Lower scores remain better.  The transformation is deterministic and is
    shared by OOF temperature fitting, OOF probability audit, and production
    scoring.  The returned sigma describes event/driver uncertainty; it is not
    a second movement constraint.
    """

    out = pd.DataFrame(index=features.index)
    base = _first_numeric_series(
        features,
        ["grid_position", "qualy_context_position", "qualy_position", "qualy_pred_rank", "qualy_pred_position"],
    )
    pred = pd.to_numeric(predictions, errors="coerce").reindex(features.index)
    if base is None or base.notna().sum() == 0:
        out["race_stochastic_score"] = pred
        out["race_stochastic_pl_score"] = pred
        out["race_stochastic_sigma"] = 1.0
        out["race_stochastic_dnf_probability"] = 0.0
        out["race_stochastic_layer"] = RACE_PROBABILITY_SCORE_LAYER
        return out

    field_size = max(1.0, float(len(features)))
    base = base.reindex(features.index).fillna(base.median(skipna=True))
    pred = pred.fillna(base)

    safety = pd.to_numeric(
        features.get(
            "track_safety_car_prior",
            features.get("track_safety_car_propensity", pd.Series(0.25, index=features.index)),
        ),
        errors="coerce",
    ).fillna(0.25).clip(0.0, 1.0)
    dnf = pd.to_numeric(
        features.get("track_dnf_prior", features.get("track_dnf_rate", pd.Series(0.08, index=features.index))),
        errors="coerce",
    ).fillna(0.08).clip(0.0, 0.60)
    strategy = pd.to_numeric(
        features.get("track_strategy_variance_prior", pd.Series(0.35, index=features.index)),
        errors="coerce",
    ).fillna(0.35).clip(0.0, 1.0)
    weather = pd.to_numeric(
        features.get(
            "track_weather_uncertainty_prior",
            features.get("track_weather_uncertainty", pd.Series(0.15, index=features.index)),
        ),
        errors="coerce",
    ).fillna(0.15).clip(0.0, 1.0)
    variance = pd.to_numeric(
        features.get("race_generation_variance_prior", pd.Series(0.25, index=features.index)),
        errors="coerce",
    ).fillna(0.25).clip(0.0, 1.0)
    slow_lap = pd.to_numeric(
        features.get("fp_slow_lap_ratio", pd.Series(0.0, index=features.index)),
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0)
    pace_vol = (
        _rank_percentile(features["fp_delta_std"])
        if "fp_delta_std" in features.columns
        else pd.Series(0.5, index=features.index, dtype=float)
    )
    pace_vol = pd.to_numeric(pace_vol, errors="coerce").fillna(0.5).clip(0.0, 1.0)

    race_variance = ((0.34 * safety) + (0.26 * dnf) + (0.25 * strategy) + (0.15 * weather)).clip(0.0, 1.0)
    race_variance = ((0.5 * race_variance) + (0.5 * variance)).clip(0.0, 1.0)
    driver_dnf = (dnf * (0.75 + (0.35 * slow_lap) + (0.20 * pace_vol))).clip(0.0, 0.75)
    expected_dnf_penalty = driver_dnf * max(1.0, field_size - 1.0)

    # Mean movement was already constrained by the fitted race model/wrapper.
    # Applying mobility to (pred - grid) here would shrink it a second time.
    stochastic_score = pred + expected_dnf_penalty
    sigma = (0.45 + (2.25 * race_variance) + (0.75 * strategy) + (0.65 * weather)).clip(0.35, 4.50)
    event_center = float(stochastic_score.median(skipna=True)) if stochastic_score.notna().any() else 0.0
    pl_score = event_center + ((stochastic_score - event_center) / sigma.replace(0.0, 1.0))

    out["race_stochastic_score"] = stochastic_score
    out["race_stochastic_pl_score"] = pl_score
    out["race_stochastic_sigma"] = sigma
    out["race_stochastic_dnf_probability"] = driver_dnf
    out["race_stochastic_layer"] = RACE_PROBABILITY_SCORE_LAYER
    return out
