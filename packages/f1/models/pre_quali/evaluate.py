"""Pre-qualifying evaluation and chronological challenger entrypoints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from packages.f1.models.pre_quali.pairwise import (
    PairwiseRankerConfig,
    fit_pairwise_qualifying_ranker,
)
from packages.f1.orchestration.backtest import evaluate_prediction_rows


def evaluate_pre_quali_predictions(*args: object, **kwargs: object) -> object:
    """Evaluate predicted qualifying order against actual session results."""

    return evaluate_prediction_rows(*args, **kwargs)


@dataclass(frozen=True)
class PairwiseWalkForwardResult:
    """Auditable per-entrant and per-event outputs from a local walk-forward."""

    predictions: pd.DataFrame
    per_event_metrics: pd.DataFrame
    skipped_event_keys: tuple[int, ...]


def walk_forward_pairwise_qualifying(
    frame: pd.DataFrame,
    *,
    config: PairwiseRankerConfig,
    evaluation_event_keys: tuple[int, ...] | None = None,
) -> PairwiseWalkForwardResult:
    """Fit on all strictly earlier events and score complete later fields.

    Required columns are the event, driver, target, baseline, rehearsal source,
    and explicit numeric feature columns declared by ``config``.  This helper
    is intentionally local and deterministic; the repository-wide runner owns
    immutable manifests, bootstrap inference, and promotion decisions.
    """

    if frame.empty:
        return PairwiseWalkForwardResult(pd.DataFrame(), pd.DataFrame(), ())
    if config.event_column not in frame.columns:
        raise ValueError(f"missing event column: {config.event_column}")
    numeric_events = pd.to_numeric(frame[config.event_column], errors="coerce")
    event_array = numeric_events.to_numpy(dtype=float)
    if (
        numeric_events.isna().any()
        or not np.isfinite(event_array).all()
        or not np.allclose(event_array, np.rint(event_array))
    ):
        raise ValueError("walk-forward event keys must be finite integers")
    event_keys = tuple(sorted(int(value) for value in numeric_events.astype(int).unique().tolist()))
    requested = set(event_keys if evaluation_event_keys is None else evaluation_event_keys)
    unknown = requested - set(event_keys)
    if unknown:
        raise ValueError(f"evaluation_event_keys are absent from frame: {sorted(unknown)}")

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int]] = []
    skipped: list[int] = []
    for event_key in event_keys:
        if event_key not in requested:
            continue
        prior_keys = [value for value in event_keys if value < event_key]
        if len(prior_keys) < config.minimum_training_events:
            skipped.append(event_key)
            continue
        history = frame.loc[numeric_events.isin(prior_keys)].copy()
        current = frame.loc[numeric_events.eq(event_key)].copy()
        model = fit_pairwise_qualifying_ranker(
            history,
            config=config,
            target_event_key=event_key,
        )
        forecast = model.predict_event(current, samples=1, seed=config.random_state)
        scored = forecast.point_order[
            [
                config.event_column,
                config.driver_column,
                "baseline_rank_prior",
                "pairwise_expected_wins",
                "predicted_qualifying_position",
                "movement_from_baseline",
                "ranking_model",
                config.target_column,
            ]
        ].copy()
        scored = scored.rename(columns={config.target_column: "actual_qualifying_position"})
        actual = pd.to_numeric(scored["actual_qualifying_position"], errors="coerce")
        predicted = pd.to_numeric(scored["predicted_qualifying_position"], errors="coerce")
        baseline = pd.to_numeric(scored["baseline_rank_prior"], errors="coerce")
        observed = actual.notna() & predicted.notna()
        if observed.any():
            challenger_mae = float((predicted.loc[observed] - actual.loc[observed]).abs().mean())
            baseline_mae = float((baseline.loc[observed] - actual.loc[observed]).abs().mean())
            kendall = float(predicted.loc[observed].corr(actual.loc[observed], method="kendall"))
        else:
            challenger_mae = float("nan")
            baseline_mae = float("nan")
            kendall = float("nan")
        metric_rows.append(
            {
                config.event_column: int(event_key),
                "entrants": int(len(current)),
                "observed_targets": int(observed.sum()),
                "challenger_mae": challenger_mae,
                "baseline_mae": baseline_mae,
                "mae_improvement": baseline_mae - challenger_mae,
                "challenger_kendall_tau_b": kendall,
            }
        )
        prediction_frames.append(scored)

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    return PairwiseWalkForwardResult(
        predictions=predictions,
        per_event_metrics=metrics,
        skipped_event_keys=tuple(skipped),
    )


__all__ = [
    "PairwiseWalkForwardResult",
    "evaluate_pre_quali_predictions",
    "evaluate_prediction_rows",
    "walk_forward_pairwise_qualifying",
]
