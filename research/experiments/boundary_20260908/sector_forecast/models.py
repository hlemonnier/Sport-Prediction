"""Frozen-shape candidate machinery for sector-checkpoint research.

The final feature/issuance protocol must be locked before real-data fitting.
No runtime model is loaded or changed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def training_weights(frame):
    """Equal event weight; each resolved driver/target within an event has weight one.

    Multiple sector issuances for a target divide that target's weight. Thus an
    extra observation of an already represented target does not change its total
    fitting influence. Weights are finally normalized to mean one for HGB L2.
    """
    if len(frame) == 0 or frame[["event_key", "target_id"]].isna().any().any():
        raise ValueError("Training requires nonempty resolved event/target identities")
    multiplicity = frame.groupby(["event_key", "target_id"])["target_id"].transform("size").to_numpy(float)
    target_counts = frame.groupby("event_key")["target_id"].transform("nunique").to_numpy(float)
    weights = 1 / (multiplicity * target_counts)
    return weights / weights.mean()


def error_summary(frame, prediction):
    """Conditional point error and complete availability/outcome accounting."""
    if not len(frame):
        raise ValueError("No issued forecasts")
    y = frame["y_true"].to_numpy(float)
    p = frame[prediction].to_numpy(float)
    if not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("Every issued forecast must be positive and finite")
    matched = np.isfinite(y)
    if not matched.any() or (y[matched] <= 0).any():
        raise ValueError("No valid positive resolved targets")
    subset = frame.loc[matched, ["event_key", "target_id"]].copy()
    if subset["target_id"].isna().any():
        raise ValueError("Resolved target lacks identity")
    subset["absolute_error"] = np.abs(p[matched]-y[matched])
    point_events = subset.groupby("event_key")["absolute_error"].mean()
    target_events = subset.groupby(["event_key", "target_id"])["absolute_error"].mean().groupby("event_key").mean()
    return {
        "issued": len(frame), "resolved_issued": int(matched.sum()), "unmatched_issued": int((~matched).sum()),
        "events_with_issuances": int(frame["event_key"].nunique()), "events_with_resolved_targets": len(target_events),
        "resolved_unique_event_targets": len(subset.groupby(["event_key", "target_id"])),
        "checkpoint_event_mae": float(point_events.mean()), "target_balanced_event_mae": float(target_events.mean()),
        "per_event_checkpoint_mae": {str(k): float(v) for k,v in point_events.items()},
        "per_event_target_balanced_mae": {str(k): float(v) for k,v in target_events.items()},
        "point_metric_scope": "Conditional on a subsequent recorded eligible clean lap existing; unmatched issuances retained as unavailable outcomes, not invented errors."}


def paired_uncertainty(frame, candidate, reference, resamples=20000, seed=20260908):
    """Paired event bootstrap and within-year circular three-event blocks."""
    c, r = error_summary(frame, candidate), error_summary(frame, reference)
    assert c["issued"] == r["issued"] and c["resolved_issued"] == r["resolved_issued"]
    by_event = c["per_event_target_balanced_mae"]
    keys = sorted(by_event)
    differences = np.asarray([by_event[k]-r["per_event_target_balanced_mae"][k] for k in keys])
    if len(keys) < 2:
        raise ValueError("At least two events required for paired uncertainty")
    rng = np.random.default_rng(seed)
    ordinary = differences[rng.integers(0, len(keys), (resamples, len(keys)))].mean(1)
    blocks = np.zeros(resamples)
    for year in sorted({k[:4] for k in keys}):
        indexes = np.asarray([i for i,k in enumerate(keys) if k[:4] == year])
        local = differences[indexes]
        starts = rng.integers(0, len(local), (resamples, int(np.ceil(len(local)/3))))
        draws = ((starts[...,None]+np.arange(3)) % len(local)).reshape(resamples,-1)[:, :len(local)]
        blocks += len(local)/len(keys) * local[draws].mean(1)
    omissions = [(differences.sum()-v)/(len(differences)-1) for v in differences]
    return {"difference_candidate_minus_reference": float(differences.mean()),
            "event_percentile_95_interval": np.quantile(ordinary,[.025,.975]).tolist(),
            "three_event_percentile_95_interval": np.quantile(blocks,[.025,.975]).tolist(),
            "leave_one_event_out_max_difference": float(max(omissions)),
            "events_won": int((differences < 0).sum()), "events_lost": int((differences > 0).sum()),
            "events_tied": int((differences == 0).sum()), "resamples": resamples, "seed": seed,
            "scope": "Descriptive historical uncertainty conditional on fixed candidates/cohort; no multiple-search or prospective guarantee."}


@dataclass
class ResidualModel:
    estimator: HistGradientBoostingRegressor
    features: list[str]
    anchor: str
    correction_clip: float

    def predict(self, frame):
        inputs = frame[self.features].to_numpy(float)
        if np.isinf(inputs).any():
            raise ValueError("Features can be missing but not infinite")
        anchor = frame[self.anchor].to_numpy(float)
        if not np.isfinite(anchor).all() or (anchor <= 0).any():
            raise ValueError("Anchor must be positive and finite")
        result = anchor+np.clip(self.estimator.predict(inputs), -self.correction_clip, self.correction_clip)
        if not np.isfinite(result).all() or (result <= 0).any():
            raise ValueError("Invalid corrected point forecasts")
        return result


def fit_residual(frame, features, anchor, config):
    if not features or len(set(features)) != len(features):
        raise ValueError("Features must be nonempty and unique")
    if set(features).intersection({"y_true", "target_id", "target_time", "target_lap", "target_available_ms"}):
        raise ValueError("Future outcome fields cannot be features")
    y = frame["y_true"].to_numpy(float)
    base = frame[anchor].to_numpy(float)
    if not np.isfinite(y).all() or not np.isfinite(base).all() or (y <= 0).any() or (base <= 0).any():
        raise ValueError("Fit requires positive finite resolved labels and anchors")
    inputs = frame[features].to_numpy(float)
    if np.isinf(inputs).any():
        raise ValueError("Infinite feature value")
    model = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=.05, max_iter=150,
        max_leaf_nodes=config["max_leaf_nodes"], min_samples_leaf=60, l2_regularization=20,
        early_stopping=False, random_state=20260908)
    model.fit(inputs, np.clip(y-base,-10.,10.), sample_weight=training_weights(frame))
    return ResidualModel(model,list(features),anchor,5.)
