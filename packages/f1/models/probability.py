"""Shared probability helpers for F1 ranking outputs."""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import pandas as pd


def stable_event_hash(value: object) -> int:
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def pl_gumbel_probabilities(
    *,
    scores: pd.Series,
    event_key: pd.Series,
    samples: int,
    temperature: float,
    seed: int,
    event_seed_key: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Sample PL/Gumbel full-order probabilities per event.

    Lower scores are better. The same helper is used by production prediction
    and OOF probability audit so calibration gates audit the deployed layer.
    """

    numeric_scores = pd.to_numeric(scores, errors="coerce")
    groups = event_key.reindex(numeric_scores.index)
    seed_groups = event_seed_key.reindex(numeric_scores.index) if event_seed_key is not None else groups
    output = pd.DataFrame(index=numeric_scores.index)
    output["utility"] = np.nan
    output["p_win"] = 0.0
    output["p_top3"] = 0.0
    output["p_top10"] = 0.0
    output["exp_pos"] = np.nan
    output["pos_p10"] = np.nan
    output["pos_p50"] = np.nan
    output["pos_p90"] = np.nan

    valid = numeric_scores.notna() & groups.notna()
    if valid.sum() == 0:
        return output

    n_samples = int(max(1, samples))
    temperature_safe = float(max(temperature, 1e-6))
    for group_value in groups.loc[valid].dropna().unique().tolist():
        idx = groups.index[(groups == group_value) & valid]
        if len(idx) == 0:
            continue
        event_scores = numeric_scores.loc[idx]
        utility = -event_scores.to_numpy(dtype=float)
        output.loc[idx, "utility"] = utility
        n_drivers = len(idx)
        if n_drivers == 1:
            output.loc[idx, ["p_win", "p_top3", "p_top10"]] = 1.0
            output.loc[idx, ["exp_pos", "pos_p10", "pos_p50", "pos_p90"]] = 1.0
            continue

        seed_value = group_value
        seed_candidates = seed_groups.loc[idx].dropna()
        if not seed_candidates.empty:
            seed_value = seed_candidates.iloc[0]
        event_seed = (int(seed) + stable_event_hash(seed_value)) % (2**32 - 1)
        if event_seed <= 0:
            event_seed = 1
        rng = np.random.default_rng(event_seed)

        noise = rng.gumbel(size=(n_samples, n_drivers))
        sampled_scores = (utility / temperature_safe)[np.newaxis, :] + noise
        sampled_order = np.argsort(-sampled_scores, axis=1)
        sampled_positions = np.empty_like(sampled_order)
        sampled_positions[np.arange(n_samples)[:, np.newaxis], sampled_order] = np.arange(1, n_drivers + 1)

        output.loc[idx, "p_win"] = (sampled_positions == 1).mean(axis=0)
        output.loc[idx, "p_top3"] = (sampled_positions <= min(3, n_drivers)).mean(axis=0)
        output.loc[idx, "p_top10"] = (sampled_positions <= min(10, n_drivers)).mean(axis=0)
        output.loc[idx, "exp_pos"] = sampled_positions.mean(axis=0)
        q10, q50, q90 = np.percentile(sampled_positions, [10, 50, 90], axis=0)
        output.loc[idx, "pos_p10"] = q10
        output.loc[idx, "pos_p50"] = q50
        output.loc[idx, "pos_p90"] = q90

    output["p_top3"] = np.minimum(output["p_top3"], output["p_top10"])
    output["p_win"] = np.minimum(output["p_win"], output["p_top3"])
    return output.clip(lower=0.0)
