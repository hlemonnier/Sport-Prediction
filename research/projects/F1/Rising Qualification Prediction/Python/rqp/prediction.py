"""Prediction orchestration."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .config import PredictionConfig, PredictionResult
from .data import build_current_features, build_training_data
from .providers import FastF1Provider, OpenF1Provider, BaseProvider
from .training import train_model
from .utils import format_prediction_table


def compute_version(round_number: int, include_standings: bool) -> str:
    if round_number <= 1:
        return "V1"
    if include_standings:
        return f"V{round_number}"
    return f"V{round_number}"


def predict_with_model(
    model: Optional[object],
    features: pd.DataFrame,
    feature_cols: List[str],
    fallback_cols: List[str],
) -> pd.Series:
    if features.empty:
        return pd.Series(dtype=float)
    if model is not None:
        X = features.reindex(columns=feature_cols).copy()
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median(numeric_only=True))
        X = X.fillna(0.0)
        return pd.Series(model.predict(X), index=features.index)
    fallback = features.reindex(columns=fallback_cols).copy()
    if fallback.empty:
        return pd.Series(0.0, index=features.index)
    fallback = fallback.apply(pd.to_numeric, errors="coerce")
    fallback = fallback.fillna(fallback.median(numeric_only=True))
    fallback = fallback.fillna(0.0)
    return fallback.mean(axis=1).fillna(0.0)


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
        ]
        fallback_cols = [c for c in feature_cols if c.endswith("_delta")]
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
        ]
        if config.include_standings:
            feature_cols.append("position_start")
        fallback_cols = ["qualy_position"]

    training_result = train_model(train, feature_cols)
    notes.extend(training_result.notes)
    preds = predict_with_model(training_result.model, features, feature_cols, fallback_cols)
    output = features.copy()
    output["pred"] = preds
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
