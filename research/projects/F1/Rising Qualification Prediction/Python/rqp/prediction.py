"""Prediction orchestration."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .config import PredictionConfig, PredictionResult
from .data import build_current_features, build_training_data
from .providers import FastF1Provider, OpenF1Provider, BaseProvider
from .training import train_model
from .utils import format_prediction_table


def compute_version(round_number: int, include_standings: bool) -> str:
    suffix = "S" if include_standings else "B"
    if round_number <= 1:
        return f"V1-{suffix}"
    return f"V{round_number}-{suffix}"


def _rank_percentile(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(0.5, index=numeric.index, dtype=float)
    return numeric.rank(method="average", pct=True)


def _average_rank_component(features: pd.DataFrame, columns: List[str]) -> Optional[pd.Series]:
    parts: list[pd.Series] = []
    for col in columns:
        if col not in features.columns:
            continue
        ranked = _rank_percentile(features[col])
        if ranked.notna().sum() == 0:
            continue
        parts.append(ranked)
    if not parts:
        return None
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def _hierarchical_fallback(
    features: pd.DataFrame,
    fallback_cols: List[str],
) -> pd.Series:
    if features.empty:
        return pd.Series(dtype=float)

    components: list[tuple[float, pd.Series]] = []
    if "qualy_position" in features.columns:
        components.append((0.55, _rank_percentile(features["qualy_position"])))

    driver_form = _average_rank_component(
        features,
        [
            "driver_form_3_fp_mean_delta",
            "driver_form_5_fp_mean_delta",
            "driver_ewma_fp_mean_delta",
            "event_driver_hist_idx",
            "fp_mean_delta",
            "fp_mean_rank",
        ],
    )
    if driver_form is not None:
        components.append((0.30, driver_form))

    team_form = _average_rank_component(
        features,
        [
            "team_form_3_fp_mean_delta",
            "team_form_5_fp_mean_delta",
            "team_ewma_fp_mean_delta",
        ],
    )
    if team_form is not None:
        components.append((0.15, team_form))

    if not components:
        fallback = features.reindex(columns=fallback_cols).copy()
        if fallback.empty:
            return pd.Series(0.5, index=features.index, dtype=float)
        fallback = fallback.apply(pd.to_numeric, errors="coerce")
        fallback = fallback.fillna(fallback.median(numeric_only=True))
        fallback = fallback.fillna(0.0)
        return _rank_percentile(fallback.mean(axis=1))

    weighted_sum = pd.Series(0.0, index=features.index, dtype=float)
    weight_total = pd.Series(0.0, index=features.index, dtype=float)
    for weight, values in components:
        valid = values.notna()
        weighted_sum.loc[valid] = weighted_sum.loc[valid] + (weight * values.loc[valid])
        weight_total.loc[valid] = weight_total.loc[valid] + weight
    score = weighted_sum.divide(weight_total.where(weight_total > 0.0))
    if score.notna().sum() == 0:
        return pd.Series(0.5, index=features.index, dtype=float)
    return score.fillna(float(score.median(skipna=True)))


def predict_with_model(
    model: Optional[object],
    features: pd.DataFrame,
    feature_cols: List[str],
    fallback_cols: List[str],
) -> pd.Series:
    if features.empty:
        return pd.Series(dtype=float)
    if model is not None:
        try:
            raw = model.predict(features)
            pred = pd.Series(raw, index=features.index, dtype=float)
            return pred
        except Exception:
            pass
        X = features.reindex(columns=feature_cols).copy()
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median(numeric_only=True))
        X = X.fillna(0.0)
        return pd.Series(model.predict(X), index=features.index)
    return _hierarchical_fallback(features, fallback_cols)


def _rank_based_probability(scores: pd.Series, k: int) -> pd.Series:
    numeric = pd.to_numeric(scores, errors="coerce")
    valid = numeric.notna()
    proba = pd.Series(0.0, index=numeric.index, dtype=float)
    if valid.sum() == 0:
        return proba
    ranked = numeric.loc[valid].rank(method="average", ascending=True)
    n = float(len(ranked))
    expected_hits = min(float(k), n)
    decay = max(1.0, n / 6.0)
    weights = np.exp(-((ranked.to_numpy(dtype=float) - 1.0) / decay))
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        proba.loc[valid] = expected_hits / n
        return proba.clip(0.0, 1.0)
    values = np.clip((expected_hits * weights) / weight_sum, 0.0, 1.0)
    proba.loc[valid] = values
    return proba


def _predict_probabilities(model: Optional[object], preds: pd.Series) -> tuple[pd.Series, pd.Series]:
    if preds.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    top10 = pd.Series(dtype=float)
    top3 = pd.Series(dtype=float)
    if model is not None and hasattr(model, "predict_probabilities"):
        try:
            calibrated = model.predict_probabilities(preds)
            if isinstance(calibrated, dict):
                if "top10" in calibrated:
                    top10 = pd.Series(calibrated["top10"], index=preds.index, dtype=float)
                if "top3" in calibrated:
                    top3 = pd.Series(calibrated["top3"], index=preds.index, dtype=float)
        except Exception:
            top10 = pd.Series(dtype=float)
            top3 = pd.Series(dtype=float)

    if top10.empty:
        top10 = _rank_based_probability(preds, k=10)
    if top3.empty:
        top3 = _rank_based_probability(preds, k=3)

    top10 = top10.reindex(preds.index).fillna(_rank_based_probability(preds, k=10))
    top3 = top3.reindex(preds.index).fillna(_rank_based_probability(preds, k=3))
    top10 = top10.clip(0.0, 1.0)
    top3 = top3.clip(0.0, 1.0)
    top3 = np.minimum(top3, top10)
    return top10, pd.Series(top3, index=preds.index, dtype=float)


def run_prediction(config: PredictionConfig) -> PredictionResult:
    provider: BaseProvider
    if config.source == "fastf1":
        provider = FastF1Provider(config.cache_dir)
    else:
        provider = OpenF1Provider(
            cache_dir=config.cache_dir,
            target_round=config.round_number,
            meeting_name=config.meeting_name,
            country_name=config.country_name,
        )

    train, notes = build_training_data(
        provider=provider,
        mode=config.mode,
        train_seasons=config.train_seasons,
        target_year=config.year,
        target_round=config.round_number,
        include_standings=config.include_standings,
    )

    features, feature_notes = build_current_features(
        provider=provider,
        mode=config.mode,
        year=config.year,
        round_number=config.round_number,
        include_standings=config.include_standings,
        history=train,
    )
    notes.extend(feature_notes)

    if config.mode == "qualifying":
        feature_cols = [
            "fp1_delta",
            "fp2_delta",
            "fp3_delta",
            "fp_mean_delta",
            "fp1_rank",
            "fp2_rank",
            "fp3_rank",
            "fp_mean_rank",
            "driver_ewma_fp_mean_delta",
            "driver_form_3_fp_mean_delta",
            "driver_form_5_fp_mean_delta",
            "team_ewma_fp_mean_delta",
            "team_form_3_fp_mean_delta",
            "team_form_5_fp_mean_delta",
            "event_driver_hist_idx",
        ]
        fallback_cols = [
            "fp_mean_delta",
            "driver_form_3_fp_mean_delta",
            "driver_form_5_fp_mean_delta",
            "driver_ewma_fp_mean_delta",
            "team_form_3_fp_mean_delta",
            "team_form_5_fp_mean_delta",
            "event_driver_hist_idx",
        ]
    else:
        feature_cols = [
            "fp1_delta",
            "fp2_delta",
            "fp3_delta",
            "fp_mean_delta",
            "fp1_rank",
            "fp2_rank",
            "fp3_rank",
            "fp_mean_rank",
            "qualy_position",
            "driver_ewma_fp_mean_delta",
            "driver_form_3_fp_mean_delta",
            "driver_form_5_fp_mean_delta",
            "team_ewma_fp_mean_delta",
            "team_form_3_fp_mean_delta",
            "team_form_5_fp_mean_delta",
            "event_driver_hist_idx",
        ]
        if config.include_standings:
            feature_cols.append("position_start")
        fallback_cols = [
            "qualy_position",
            "position_start",
            "driver_form_3_fp_mean_delta",
            "driver_form_5_fp_mean_delta",
            "driver_ewma_fp_mean_delta",
            "team_form_3_fp_mean_delta",
            "team_form_5_fp_mean_delta",
            "event_driver_hist_idx",
        ]

    training_result = train_model(train, feature_cols)
    notes.extend(training_result.notes)
    if training_result.model is None:
        notes.append(
            "Fallback heuristique hierarchique actif: qualy_position + forme pilote + forme ecurie (quand dispo).",
        )
    preds = predict_with_model(training_result.model, features, feature_cols, fallback_cols)
    output = features.copy()
    output["pred"] = preds
    proba_top10, proba_top3 = _predict_probabilities(training_result.model, preds)
    output["proba_top10"] = proba_top10
    output["proba_top3"] = proba_top3
    if "driver_name" not in output.columns:
        if "driver_id" in output.columns:
            output["driver_name"] = output["driver_id"]
        else:
            output["driver_name"] = pd.Series(dtype=str)
    elif "driver_id" in output.columns:
        output["driver_name"] = output["driver_name"].fillna(output["driver_id"])

    version = compute_version(config.round_number, config.include_standings)
    table = format_prediction_table(output, top_n=10)
    return PredictionResult(version=version, table=table, notes=notes)
