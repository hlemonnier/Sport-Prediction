"""Prediction orchestration."""

from __future__ import annotations

import hashlib
import json
from typing import List, Optional

import numpy as np
import pandas as pd

from .circuit_cards import (
    CIRCUIT_INTERACTION_FEATURES,
    CIRCUIT_NUMERIC_FEATURES,
    circuit_card_payload_from_frame,
)
from .config import PredictionConfig, PredictionResult
from .data import build_current_features, build_training_data
from .live_runner import run_live_race_prediction
from .providers import BaseProvider, FastF1Provider, LocalWeekendProvider, OpenF1Provider
from .training import train_model
from .utils import format_prediction_table

RUNSIM_EXACT_COLUMNS = {"fp_slow_lap_ratio", "fp_quali_vs_race_gap"}


def _is_runsim_column(column: str) -> bool:
    return (
        column.startswith("fp_quali_sim_")
        or column.startswith("fp_race_sim_")
        or column in RUNSIM_EXACT_COLUMNS
    )


def _apply_runsim_ablation(feature_cols: List[str], fallback_cols: List[str], disable: bool) -> tuple[List[str], List[str]]:
    if not disable:
        return feature_cols, fallback_cols
    filtered_features = [col for col in feature_cols if not _is_runsim_column(col)]
    filtered_fallback = [col for col in fallback_cols if not _is_runsim_column(col)]
    return filtered_features, filtered_fallback


def _drop_runsim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    cols = [col for col in frame.columns if _is_runsim_column(col)]
    if not cols:
        return frame
    return frame.drop(columns=cols, errors="ignore")


def _is_circuit_column(column: str) -> bool:
    return (
        column in CIRCUIT_NUMERIC_FEATURES
        or column in CIRCUIT_INTERACTION_FEATURES
        or column.startswith("circuit_")
        or column in {"driver_archetype_form_3_fp_weighted_delta", "team_archetype_form_3_fp_weighted_delta"}
        or column in {"driver_circuit_hist_fp_weighted_delta", "team_circuit_hist_fp_weighted_delta"}
        or column in {
            "track_qualy_importance",
            "track_safety_car_prior",
            "track_strategy_variance_prior",
            "track_weather_uncertainty_prior",
            "race_generation_variance_prior",
        }
    )


def _apply_circuit_ablation(feature_cols: List[str], fallback_cols: List[str], disable: bool) -> tuple[List[str], List[str]]:
    if not disable:
        return feature_cols, fallback_cols
    filtered_features = [col for col in feature_cols if not _is_circuit_column(col)]
    filtered_fallback = [col for col in fallback_cols if not _is_circuit_column(col)]
    return filtered_features, filtered_fallback


def _drop_circuit_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    cols = [col for col in frame.columns if _is_circuit_column(col)]
    if not cols:
        return frame
    return frame.drop(columns=cols, errors="ignore")


def _apply_season_sample_weighting(train: pd.DataFrame, config: PredictionConfig, notes: List[str]) -> pd.DataFrame:
    if train.empty:
        return train
    multiplier = float(config.season_weight_multiplier or 1.0)
    if multiplier <= 1.0:
        return train
    weight_year = int(config.season_weight_year or config.year)
    if "event_year" not in train.columns:
        notes.append("Season sample weighting skipped: event_year column unavailable.")
        return train
    out = train.copy()
    event_year = pd.to_numeric(out["event_year"], errors="coerce")
    mask = event_year == weight_year
    out["_sample_weight"] = 1.0
    out.loc[mask, "_sample_weight"] = multiplier
    weighted_events = 0
    if "event_key" in out.columns:
        weighted_events = int(pd.to_numeric(out.loc[mask, "event_key"], errors="coerce").dropna().nunique())
    notes.append(
        "Season sample weighting active: "
        f"event_year={weight_year}, multiplier={multiplier:.2f}, "
        f"weighted_rows={int(mask.sum())}, weighted_events={weighted_events}.",
    )
    return out


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


def _weighted_rank_score(
    features: pd.DataFrame,
    components: list[tuple[str, float, bool]],
) -> Optional[pd.Series]:
    weighted_sum = pd.Series(0.0, index=features.index, dtype=float)
    weight_total = pd.Series(0.0, index=features.index, dtype=float)
    used = False
    for col, weight, ascending in components:
        if col not in features.columns or weight <= 0.0:
            continue
        values = pd.to_numeric(features[col], errors="coerce")
        if values.notna().sum() == 0:
            continue
        ranked = _rank_percentile(values)
        if not ascending:
            ranked = 1.0 - ranked
        valid = ranked.notna()
        if not valid.any():
            continue
        weighted_sum.loc[valid] = weighted_sum.loc[valid] + (float(weight) * ranked.loc[valid])
        weight_total.loc[valid] = weight_total.loc[valid] + float(weight)
        used = True
    if not used:
        return None
    score = weighted_sum.divide(weight_total.where(weight_total > 0.0))
    if score.notna().sum() == 0:
        return None
    return score.fillna(float(score.median(skipna=True)))


def _first_numeric_series(features: pd.DataFrame, columns: list[str]) -> Optional[pd.Series]:
    for column in columns:
        if column not in features.columns:
            continue
        values = pd.to_numeric(features[column], errors="coerce")
        if values.notna().any():
            return values
    return None


def _race_grid_delta_fallback(features: pd.DataFrame) -> Optional[pd.Series]:
    start = _first_numeric_series(features, ["grid_position", "qualy_context_position", "qualy_position"])
    if start is None or start.notna().sum() == 0:
        return None

    n = max(1, len(features))
    start = start.fillna(start.median(skipna=True))
    pace_parts: list[pd.Series] = []
    for col in [
        "fp_race_sim_rank",
        "event_pace_index",
        "fp_weighted_delta",
        "fp_race_sim_delta",
        "fp_quali_sim_delta",
        "team_archetype_form_3_fp_weighted_delta",
        "driver_archetype_form_3_fp_weighted_delta",
        "team_circuit_hist_fp_weighted_delta",
        "driver_circuit_hist_fp_weighted_delta",
        "circuit_fit_index",
    ]:
        if col in features.columns:
            pace_parts.append(_rank_percentile(features[col]))
    if pace_parts:
        pace_pct = pd.concat(pace_parts, axis=1).mean(axis=1, skipna=True).fillna(0.5)
    else:
        pace_pct = _rank_percentile(start)
    pace_rank = 1.0 + ((float(n) - 1.0) * pace_pct.clip(0.0, 1.0))

    overtake = pd.to_numeric(
        features.get("track_overtake_propensity", pd.Series(0.35, index=features.index)),
        errors="coerce",
    ).fillna(0.35).clip(0.0, 1.0)
    drs = pd.to_numeric(
        features.get("circuit_drs_effectiveness", pd.Series(0.45, index=features.index)),
        errors="coerce",
    ).fillna(0.45).clip(0.0, 1.0)
    difficulty = pd.to_numeric(
        features.get("circuit_overtaking_difficulty", pd.Series(0.50, index=features.index)),
        errors="coerce",
    ).fillna(0.50).clip(0.0, 1.0)
    chaos = pd.to_numeric(
        features.get("track_chaos_index", pd.Series(0.20, index=features.index)),
        errors="coerce",
    ).fillna(0.20).clip(0.0, 1.0)
    safety_prior = pd.to_numeric(
        features.get("track_safety_car_prior", pd.Series(0.25, index=features.index)),
        errors="coerce",
    ).fillna(0.25).clip(0.0, 1.0)
    dnf_prior = pd.to_numeric(
        features.get("track_dnf_prior", pd.Series(0.10, index=features.index)),
        errors="coerce",
    ).fillna(0.10).clip(0.0, 1.0)
    strategy_prior = pd.to_numeric(
        features.get("track_strategy_variance_prior", pd.Series(0.35, index=features.index)),
        errors="coerce",
    ).fillna(0.35).clip(0.0, 1.0)
    weather_prior = pd.to_numeric(
        features.get("track_weather_uncertainty_prior", pd.Series(0.15, index=features.index)),
        errors="coerce",
    ).fillna(0.15).clip(0.0, 1.0)

    kappa = (0.04 + (0.62 * overtake) + (0.18 * drs) - (0.48 * difficulty)).clip(0.03, 0.78)
    residual = pace_rank - start
    score = start + (kappa * residual)

    # Keep clean low-overtake races tightly grid-constrained, while allowing
    # chaos/reliability to soften the grid prior without treating overtaking as chaos.
    race_variance = (
        (0.30 * chaos)
        + (0.25 * safety_prior)
        + (0.20 * dnf_prior)
        + (0.15 * strategy_prior)
        + (0.10 * weather_prior)
    ).clip(0.0, 1.0)
    chaos_weight = (0.05 + (0.12 * race_variance)).clip(0.05, 0.17)
    score = ((1.0 - chaos_weight) * score) + (chaos_weight * pace_rank)
    return pd.Series(score, index=features.index, dtype=float)


def _hierarchical_fallback(
    features: pd.DataFrame,
    fallback_cols: List[str],
) -> pd.Series:
    if features.empty:
        return pd.Series(dtype=float)

    # Empirically strongest no-training fallbacks on the local 2025 holdout:
    # qualifying: FP mean rank + event pace + qualifying-sim rank;
    # race: actual qualifying position. These avoid overfitting a model when no
    # historical seasons are available while still preserving event-local signal.
    race_score = _race_grid_delta_fallback(features)
    if race_score is not None:
        return race_score

    qualifying_score = _weighted_rank_score(
        features,
        [
            ("event_pace_index", 2.0, True),
            ("fp_mean_rank", 2.0, True),
            ("fp_quali_sim_rank", 1.0, True),
            ("fp_quali_sim_delta_downforce_adj", 0.75, True),
            ("fp_weighted_delta_downforce_adj", 0.55, True),
            ("fp_weighted_delta_power_adj", 0.45, True),
            ("circuit_fit_index", 0.85, True),
            ("team_archetype_form_3_fp_weighted_delta", 0.50, True),
            ("driver_archetype_form_3_fp_weighted_delta", 0.35, True),
        ],
    )
    if qualifying_score is not None:
        return qualifying_score

    components: list[tuple[float, pd.Series]] = []
    if "qualy_position" in features.columns:
        qualy_rank = _rank_percentile(features["qualy_position"])
        if "track_qualy_importance" in features.columns:
            qualy_importance = pd.to_numeric(features["track_qualy_importance"], errors="coerce")
            qualy_importance = qualy_importance.clip(lower=0.0, upper=1.0).fillna(0.6)
            qualy_rank = (qualy_importance * qualy_rank) + ((1.0 - qualy_importance) * 0.5)
        components.append((0.45, qualy_rank))
    if "qualy_context_position" in features.columns:
        components.append((0.25, _rank_percentile(features["qualy_context_position"])))
    if "qualy_pred_position" in features.columns:
        components.append((0.10, _rank_percentile(features["qualy_pred_position"])))
    if "qualy_pred_rank_pct" in features.columns:
        pred_pct = pd.to_numeric(features["qualy_pred_rank_pct"], errors="coerce")
        components.append((0.10, pred_pct.clip(lower=0.0, upper=1.0)))
    if "qualy_pred_top10_proba" in features.columns:
        pred_top10 = pd.to_numeric(features["qualy_pred_top10_proba"], errors="coerce")
        components.append((0.05, _rank_percentile(1.0 - pred_top10)))
    if "qualy_gap_to_best" in features.columns:
        components.append((0.10, _rank_percentile(features["qualy_gap_to_best"])))

    driver_form = _average_rank_component(
        features,
        [
            "fp_quali_sim_delta",
            "fp_quali_sim_rank",
            "fp_race_sim_delta",
            "fp_race_sim_rank",
            "fp_slow_lap_ratio",
            "fp_quali_vs_race_gap",
            "fp_weighted_delta",
            "fp_delta_std",
            "fp_mean_top3_delta",
            "fp_mean_lap_std",
            "driver_form_3_fp_quali_sim_delta",
            "driver_ewma_fp_quali_sim_delta",
            "driver_form_3_fp_race_sim_delta",
            "driver_ewma_fp_race_sim_delta",
            "driver_form_3_fp_weighted_delta",
            "driver_ewma_fp_weighted_delta",
            "driver_form_3_fp_mean_delta",
            "driver_form_5_fp_mean_delta",
            "driver_ewma_fp_mean_delta",
            "event_driver_hist_idx",
            "driver_archetype_form_3_fp_weighted_delta",
            "team_archetype_form_3_fp_weighted_delta",
            "driver_circuit_hist_fp_weighted_delta",
            "team_circuit_hist_fp_weighted_delta",
            "fp_weighted_delta_downforce_adj",
            "fp_weighted_delta_power_adj",
            "fp_quali_sim_delta_downforce_adj",
            "fp_race_sim_delta_tyre_adj",
            "fp_race_sim_delta_power_adj",
            "fp_mean_delta",
            "fp_mean_rank",
            "qualy_context_position",
            "qualy_context_position_track_adj",
            "qualy_position_track_adj",
            "qualy_position_circuit_importance_adj",
            "qualy_pred_position",
            "qualy_pred_position_track_adj",
            "qualy_pred_position_circuit_importance_adj",
            "qualy_pred_vs_actual_gap",
            "track_chaos_index",
            "circuit_overtaking_difficulty",
            "circuit_qualifying_importance",
            "circuit_downforce_demand",
            "circuit_power_sensitivity",
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
            "team_form_3_fp_weighted_delta",
            "team_ewma_fp_weighted_delta",
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


def _capped_probability_allocation(weights: np.ndarray, expected_hits: float) -> np.ndarray:
    n = int(len(weights))
    if n == 0:
        return weights
    total = float(max(0.0, min(expected_hits, float(n))))
    if total <= 0.0:
        return np.zeros(n, dtype=float)
    if total >= float(n):
        return np.ones(n, dtype=float)

    clean = np.asarray(weights, dtype=float)
    clean = np.where(np.isfinite(clean) & (clean > 0.0), clean, 0.0)
    if float(clean.sum()) <= 0.0:
        clean = np.ones(n, dtype=float)

    values = np.zeros(n, dtype=float)
    remaining = np.ones(n, dtype=bool)
    remaining_total = total
    while remaining.any() and remaining_total > 1e-12:
        w = clean[remaining]
        if float(w.sum()) <= 0.0:
            allocation = np.full(int(remaining.sum()), remaining_total / float(remaining.sum()))
        else:
            allocation = remaining_total * (w / float(w.sum()))
        capped = allocation >= 1.0
        remaining_idx = np.flatnonzero(remaining)
        if not capped.any():
            values[remaining_idx] = allocation
            break
        cap_idx = remaining_idx[capped]
        values[cap_idx] = 1.0
        remaining_total -= float(len(cap_idx))
        remaining[cap_idx] = False
        if remaining_total <= 0.0:
            break
    return np.clip(values, 0.0, 1.0)


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
        proba.loc[valid] = _capped_probability_allocation(np.ones(len(ranked), dtype=float), expected_hits)
        return proba.clip(0.0, 1.0)
    values = _capped_probability_allocation(weights, expected_hits)
    proba.loc[valid] = values
    return proba


def _event_total_probability(scores: pd.Series, probabilities: pd.Series, k: int) -> pd.Series:
    numeric = pd.to_numeric(scores, errors="coerce")
    raw = pd.to_numeric(probabilities, errors="coerce").reindex(numeric.index)
    valid = numeric.notna() & raw.notna()
    output = pd.Series(0.0, index=numeric.index, dtype=float)
    if valid.sum() == 0:
        return output
    weights = raw.loc[valid].clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
    if float(weights.sum()) <= 0.0:
        ranked = numeric.loc[valid].rank(method="average", ascending=True)
        weights = np.exp(-((ranked.to_numpy(dtype=float) - 1.0) / max(1.0, float(len(ranked)) / 6.0)))
    output.loc[valid] = _capped_probability_allocation(weights, min(float(k), float(valid.sum())))
    return output.clip(0.0, 1.0)


def _predict_probability(model: Optional[object], preds: pd.Series, *, label: str, k: int) -> pd.Series:
    if preds.empty:
        return pd.Series(dtype=float)

    calibrated_probability = pd.Series(dtype=float)
    if model is not None and hasattr(model, "predict_probabilities"):
        try:
            calibrated = model.predict_probabilities(preds)
            if isinstance(calibrated, dict):
                if label in calibrated:
                    calibrated_probability = pd.Series(calibrated[label], index=preds.index, dtype=float)
        except Exception:
            calibrated_probability = pd.Series(dtype=float)

    fallback = _rank_based_probability(preds, k=k)
    if calibrated_probability.empty:
        return fallback
    combined = calibrated_probability.reindex(preds.index).fillna(fallback).clip(0.0, 1.0)
    return _event_total_probability(preds, combined, k=k)


def _predict_probabilities(model: Optional[object], preds: pd.Series) -> tuple[pd.Series, pd.Series]:
    if preds.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    top10 = _predict_probability(model, preds, label="top10", k=10)
    top3 = _predict_probability(model, preds, label="top3", k=3)
    top10 = top10.clip(0.0, 1.0)
    top3 = top3.clip(0.0, 1.0)
    top3 = np.minimum(top3, top10)
    return top10, pd.Series(top3, index=preds.index, dtype=float)


def _prediction_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records"))


def _normalize_event_name_for_key(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "unknown_event"
    return " ".join(text.split())


def _event_group_id(frame: pd.DataFrame, mode: str) -> pd.Series:
    if "event_key" in frame.columns:
        event_key = pd.to_numeric(frame["event_key"], errors="coerce")
        if event_key.notna().any():
            return event_key.fillna(-1).astype(int).astype(str)

    season_col = "event_year" if "event_year" in frame.columns else "year"
    round_col = "event_round" if "event_round" in frame.columns else "round_number"
    season = (
        pd.to_numeric(frame[season_col], errors="coerce")
        if season_col in frame.columns
        else pd.Series(index=frame.index, data=np.nan, dtype=float)
    )
    round_number = (
        pd.to_numeric(frame[round_col], errors="coerce")
        if round_col in frame.columns
        else pd.Series(index=frame.index, data=np.nan, dtype=float)
    )
    event_name = (
        frame["event_name_norm"].map(_normalize_event_name_for_key)
        if "event_name_norm" in frame.columns
        else frame.get("event_name", pd.Series(index=frame.index, dtype=object)).map(_normalize_event_name_for_key)
    )
    season_token = season.fillna(-1).astype(int).astype(str)
    round_token = round_number.fillna(-1).astype(int).astype(str)
    mode_token = pd.Series(str(mode), index=frame.index, dtype=str)
    return season_token + "-" + round_token + "-" + mode_token + "-" + event_name.astype(str)


def _canonical_event_key(event_rows: pd.DataFrame, mode: str, fallback_group_id: str) -> str:
    season_value: Optional[int] = None
    for season_col in ("event_year", "year", "season"):
        if season_col not in event_rows.columns:
            continue
        season_series = pd.to_numeric(event_rows[season_col], errors="coerce").dropna()
        if not season_series.empty:
            season_value = int(season_series.iloc[0])
            break

    round_value: Optional[int] = None
    for round_col in ("event_round", "round_number", "round"):
        if round_col not in event_rows.columns:
            continue
        round_series = pd.to_numeric(event_rows[round_col], errors="coerce").dropna()
        if not round_series.empty:
            round_value = int(round_series.iloc[0])
            break

    event_name_value = "unknown_event"
    if "event_name_norm" in event_rows.columns:
        series = event_rows["event_name_norm"]
    elif "event_name" in event_rows.columns:
        series = event_rows["event_name"]
    else:
        series = pd.Series(dtype=object)
    if not series.empty:
        first_non_empty = series.dropna()
        if not first_non_empty.empty:
            event_name_value = _normalize_event_name_for_key(first_non_empty.iloc[0])

    if season_value is not None and round_value is not None:
        return f"{season_value}-{round_value}-{mode}-{event_name_value}"

    if "event_key" in event_rows.columns:
        event_series = pd.to_numeric(event_rows["event_key"], errors="coerce").dropna()
        if not event_series.empty:
            return str(int(event_series.iloc[0]))

    return str(fallback_group_id)


def _stable_event_hash(event_key: str) -> int:
    digest = hashlib.md5(str(event_key).encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def _pl_gumbel_listwise(
    *,
    frame: pd.DataFrame,
    preds: pd.Series,
    mode: str,
    samples: int,
    temperature: float,
    seed: int,
) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    output["utility"] = np.nan
    output["p_win"] = 0.0
    output["p_top3"] = 0.0
    output["p_top10"] = 0.0
    output["exp_pos"] = np.nan
    output["pos_p10"] = np.nan
    output["pos_p50"] = np.nan
    output["pos_p90"] = np.nan
    output["listwise_method"] = "pl_gumbel"
    output["listwise_samples"] = int(max(1, samples))
    output["temperature"] = float(max(temperature, 1e-6))
    output["listwise_enabled"] = True

    valid_mask = pd.to_numeric(preds, errors="coerce").notna()
    if valid_mask.sum() == 0:
        output["listwise_enabled"] = False
        return output

    group_id = _event_group_id(frame, mode=mode)
    temperature_safe = float(max(temperature, 1e-6))
    n_samples = int(max(1, samples))
    for event_group in group_id.loc[valid_mask].dropna().unique().tolist():
        idx = group_id.index[(group_id == event_group) & valid_mask]
        if len(idx) == 0:
            continue
        event_rows = frame.loc[idx]
        event_preds = pd.to_numeric(preds.loc[idx], errors="coerce")
        utility = -event_preds.to_numpy(dtype=float)
        output.loc[idx, "utility"] = utility

        n_drivers = len(idx)
        if n_drivers == 1:
            output.loc[idx, "p_win"] = 1.0
            output.loc[idx, "p_top3"] = 1.0
            output.loc[idx, "p_top10"] = 1.0
            output.loc[idx, "exp_pos"] = 1.0
            output.loc[idx, "pos_p10"] = 1.0
            output.loc[idx, "pos_p50"] = 1.0
            output.loc[idx, "pos_p90"] = 1.0
            continue

        canonical_key = _canonical_event_key(event_rows, mode=mode, fallback_group_id=str(event_group))
        h = _stable_event_hash(canonical_key)
        event_seed = (int(seed) + h) % (2**32 - 1)
        if event_seed <= 0:
            event_seed = 1
        rng = np.random.default_rng(event_seed)

        noise = rng.gumbel(size=(n_samples, n_drivers))
        sampled_scores = (utility / temperature_safe)[np.newaxis, :] + noise
        sampled_order = np.argsort(-sampled_scores, axis=1)
        sampled_positions = np.empty_like(sampled_order)
        sampled_positions[np.arange(n_samples)[:, np.newaxis], sampled_order] = np.arange(1, n_drivers + 1)

        p_win = (sampled_positions == 1).mean(axis=0)
        p_top3 = (sampled_positions <= 3).mean(axis=0)
        p_top10 = (sampled_positions <= 10).mean(axis=0)
        expected_position = sampled_positions.mean(axis=0)
        q10, q50, q90 = np.percentile(sampled_positions, [10, 50, 90], axis=0)

        output.loc[idx, "p_win"] = np.clip(p_win, 0.0, 1.0)
        output.loc[idx, "p_top3"] = np.clip(p_top3, 0.0, 1.0)
        output.loc[idx, "p_top10"] = np.clip(p_top10, 0.0, 1.0)
        output.loc[idx, "exp_pos"] = expected_position
        output.loc[idx, "pos_p10"] = q10
        output.loc[idx, "pos_p50"] = q50
        output.loc[idx, "pos_p90"] = q90

    output["p_top3"] = np.minimum(output["p_top3"], output["p_top10"])
    return output


def _qualifying_feature_sets(
    disable_runsim: bool = False,
    disable_circuit: bool = False,
) -> tuple[List[str], List[str]]:
    feature_cols = [
        "fp1_delta",
        "fp2_delta",
        "fp3_delta",
        "sq_delta",
        "sprint_delta",
        "fp_mean_delta",
        "fp_weighted_delta",
        "fp_quali_sim_delta",
        "fp_quali_sim_rank",
        "fp_quali_sim_laps",
        "quali_sim_sessions_available",
        "fp_race_sim_delta",
        "fp_race_sim_rank",
        "fp_race_sim_laps",
        "race_sim_sessions_available",
        "fp_slow_lap_ratio",
        "fp_quali_vs_race_gap",
        "fp_delta_std",
        "pace_sessions_available",
        "fp_mean_top3_delta",
        "fp_mean_lap_std",
        "fp_total_laps",
        "fp1_rank",
        "fp2_rank",
        "fp3_rank",
        "sq_rank",
        "sprint_rank",
        "fp_mean_rank",
        "driver_ewma_fp_mean_delta",
        "driver_form_3_fp_mean_delta",
        "driver_form_5_fp_mean_delta",
        "driver_ewma_fp_weighted_delta",
        "driver_form_3_fp_weighted_delta",
        "driver_ewma_fp_quali_sim_delta",
        "driver_form_3_fp_quali_sim_delta",
        "driver_ewma_fp_race_sim_delta",
        "driver_form_3_fp_race_sim_delta",
        "driver_form_3_vs_team_fp_weighted_delta",
        "team_ewma_fp_mean_delta",
        "team_form_3_fp_mean_delta",
        "team_form_5_fp_mean_delta",
        "team_ewma_fp_weighted_delta",
        "team_form_3_fp_weighted_delta",
        "event_driver_hist_idx",
        "event_pace_index",
        "driver_vs_team_fp_weighted_delta",
    ]
    feature_cols.extend(CIRCUIT_NUMERIC_FEATURES)
    feature_cols.extend(CIRCUIT_INTERACTION_FEATURES)
    fallback_cols = [
        "fp_mean_delta",
        "fp_weighted_delta",
        "fp_quali_sim_delta",
        "fp_quali_sim_rank",
        "fp_quali_sim_laps",
        "fp_race_sim_delta",
        "fp_race_sim_rank",
        "fp_race_sim_laps",
        "fp_slow_lap_ratio",
        "fp_quali_vs_race_gap",
        "fp_delta_std",
        "pace_sessions_available",
        "fp_mean_top3_delta",
        "fp_mean_lap_std",
        "fp_total_laps",
        "driver_form_3_fp_mean_delta",
        "driver_form_5_fp_mean_delta",
        "driver_ewma_fp_mean_delta",
        "driver_form_3_fp_weighted_delta",
        "driver_form_3_fp_quali_sim_delta",
        "driver_ewma_fp_quali_sim_delta",
        "driver_form_3_fp_race_sim_delta",
        "driver_ewma_fp_race_sim_delta",
        "driver_form_3_vs_team_fp_weighted_delta",
        "driver_ewma_fp_weighted_delta",
        "team_form_3_fp_mean_delta",
        "team_form_5_fp_mean_delta",
        "event_pace_index",
        "driver_vs_team_fp_weighted_delta",
        "event_driver_hist_idx",
    ]
    fallback_cols.extend(CIRCUIT_NUMERIC_FEATURES)
    fallback_cols.extend(CIRCUIT_INTERACTION_FEATURES)
    feature_cols, fallback_cols = _apply_runsim_ablation(feature_cols, fallback_cols, disable_runsim)
    return _apply_circuit_ablation(feature_cols, fallback_cols, disable_circuit)


def _race_feature_sets(
    include_standings: bool,
    disable_runsim: bool = False,
    disable_circuit: bool = False,
) -> tuple[List[str], List[str]]:
    feature_cols = [
        "fp1_delta",
        "fp2_delta",
        "fp3_delta",
        "sq_delta",
        "sprint_delta",
        "fp_mean_delta",
        "fp_weighted_delta",
        "fp_weighted_delta_track_adj",
        "fp_quali_sim_delta",
        "fp_quali_sim_rank",
        "fp_quali_sim_laps",
        "quali_sim_sessions_available",
        "fp_race_sim_delta",
        "fp_race_sim_delta_track_adj",
        "fp_race_sim_rank",
        "fp_race_sim_laps",
        "race_sim_sessions_available",
        "fp_slow_lap_ratio",
        "fp_quali_vs_race_gap",
        "fp_delta_std",
        "pace_sessions_available",
        "fp_mean_top3_delta",
        "fp_mean_lap_std",
        "fp_total_laps",
        "fp1_rank",
        "fp2_rank",
        "fp3_rank",
        "sq_rank",
        "sprint_rank",
        "fp_mean_rank",
        "grid_position",
        "grid_position_track_adj",
        "qualy_position",
        "qualy_context_position",
        "qualy_context_position_track_adj",
        "qualy_position_track_adj",
        "qualy_gap_to_best",
        "qualy_gap_track_adj",
        "qualy_pred_position",
        "qualy_pred_rank_pct",
        "qualy_pred_top10_proba",
        "qualy_pred_top3_proba",
        "qualy_pred_vs_actual_gap",
        "qualy_pred_position_track_adj",
        "driver_ewma_fp_mean_delta",
        "driver_form_3_fp_mean_delta",
        "driver_form_5_fp_mean_delta",
        "driver_ewma_fp_weighted_delta",
        "driver_form_3_fp_weighted_delta",
        "driver_ewma_fp_quali_sim_delta",
        "driver_form_3_fp_quali_sim_delta",
        "driver_ewma_fp_race_sim_delta",
        "driver_form_3_fp_race_sim_delta",
        "driver_form_3_vs_team_fp_weighted_delta",
        "team_ewma_fp_mean_delta",
        "team_form_3_fp_mean_delta",
        "team_form_5_fp_mean_delta",
        "team_ewma_fp_weighted_delta",
        "team_form_3_fp_weighted_delta",
        "event_driver_hist_idx",
        "event_pace_index",
        "driver_vs_team_fp_weighted_delta",
        "track_overtake_propensity",
        "track_grid_stability",
        "track_safety_car_propensity",
        "track_sc_lap_ratio",
        "track_vsc_lap_ratio",
        "track_dnf_rate",
        "track_pit_stop_intensity",
        "track_weather_uncertainty",
        "track_safety_car_prior",
        "track_dnf_prior",
        "track_strategy_variance_prior",
        "track_weather_uncertainty_prior",
        "race_generation_variance_prior",
        "track_same_event_count",
        "track_history_count",
        "track_stats_reliability",
        "track_chaos_index",
        "track_qualy_importance",
    ]
    feature_cols.extend(CIRCUIT_NUMERIC_FEATURES)
    feature_cols.extend(CIRCUIT_INTERACTION_FEATURES)
    if include_standings:
        feature_cols.append("position_start")

    fallback_cols = [
        "qualy_position",
        "grid_position",
        "grid_position_track_adj",
        "qualy_context_position",
        "qualy_context_position_track_adj",
        "qualy_position_track_adj",
        "qualy_gap_to_best",
        "qualy_gap_track_adj",
        "qualy_pred_position",
        "qualy_pred_rank_pct",
        "qualy_pred_top10_proba",
        "position_start",
        "fp_weighted_delta",
        "fp_weighted_delta_track_adj",
        "fp_quali_sim_delta",
        "fp_quali_sim_rank",
        "fp_quali_sim_laps",
        "fp_race_sim_delta",
        "fp_race_sim_delta_track_adj",
        "fp_race_sim_rank",
        "fp_race_sim_laps",
        "fp_slow_lap_ratio",
        "fp_quali_vs_race_gap",
        "pace_sessions_available",
        "fp_delta_std",
        "fp_mean_top3_delta",
        "driver_form_3_fp_mean_delta",
        "driver_form_5_fp_mean_delta",
        "driver_ewma_fp_mean_delta",
        "driver_form_3_fp_weighted_delta",
        "driver_form_3_fp_quali_sim_delta",
        "driver_ewma_fp_quali_sim_delta",
        "driver_form_3_fp_race_sim_delta",
        "driver_ewma_fp_race_sim_delta",
        "driver_form_3_vs_team_fp_weighted_delta",
        "driver_ewma_fp_weighted_delta",
        "team_form_3_fp_mean_delta",
        "team_form_5_fp_mean_delta",
        "team_form_3_fp_weighted_delta",
        "team_ewma_fp_weighted_delta",
        "event_pace_index",
        "driver_vs_team_fp_weighted_delta",
        "event_driver_hist_idx",
        "track_overtake_propensity",
        "track_safety_car_propensity",
        "track_chaos_index",
        "track_qualy_importance",
        "track_safety_car_prior",
        "track_dnf_prior",
        "track_strategy_variance_prior",
        "track_weather_uncertainty_prior",
        "race_generation_variance_prior",
    ]
    fallback_cols.extend(CIRCUIT_NUMERIC_FEATURES)
    fallback_cols.extend(CIRCUIT_INTERACTION_FEATURES)
    feature_cols, fallback_cols = _apply_runsim_ablation(feature_cols, fallback_cols, disable_runsim)
    return _apply_circuit_ablation(feature_cols, fallback_cols, disable_circuit)


def _build_qualifying_signal_frame(
    frame: pd.DataFrame,
    preds: pd.Series,
    top10: pd.Series,
    top3: pd.Series,
    *,
    source: str = "",
    training_event_max: Optional[float] = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    if "driver_id" not in frame.columns:
        return pd.DataFrame()
    out = pd.DataFrame(index=frame.index)
    out["driver_id"] = frame["driver_id"].astype(str)
    if "event_key" in frame.columns:
        out["event_key"] = pd.to_numeric(frame["event_key"], errors="coerce")
    out["qualy_pred_position"] = pd.to_numeric(preds, errors="coerce")
    out["qualy_pred_top10_proba"] = pd.to_numeric(top10, errors="coerce")
    out["qualy_pred_top3_proba"] = pd.to_numeric(top3, errors="coerce")
    if source:
        out["qualy_pred_source"] = source
    if training_event_max is not None:
        out["qualy_pred_training_event_max"] = float(training_event_max)
    if "event_key" in out.columns:
        valid = out["event_key"].notna()
        out["qualy_pred_rank_pct"] = float("nan")
        if valid.any():
            out.loc[valid, "qualy_pred_rank_pct"] = (
                out.loc[valid]
                .groupby("event_key", sort=False)["qualy_pred_position"]
                .rank(method="average", pct=True, ascending=True)
            )
    else:
        out["qualy_pred_rank_pct"] = _rank_percentile(out["qualy_pred_position"])
    subset = ["driver_id"]
    if "event_key" in out.columns:
        subset.insert(0, "event_key")
    out = out.drop_duplicates(subset=subset, keep="last")
    return out.reset_index(drop=True)


def _build_oof_qualifying_signal_frame(
    *,
    qual_train: pd.DataFrame,
    qual_feature_cols: List[str],
    qual_fallback_cols: List[str],
    config: PredictionConfig,
    notes: List[str],
    min_prior_events: int = 3,
) -> pd.DataFrame:
    if qual_train.empty or "event_key" not in qual_train.columns:
        return pd.DataFrame()
    event_key = pd.to_numeric(qual_train["event_key"], errors="coerce")
    valid_events = sorted(event_key.dropna().unique().tolist())
    signals: list[pd.DataFrame] = []
    for event in valid_events:
        val_mask = event_key == event
        train_mask = event_key < event
        val_q = qual_train.loc[val_mask].copy()
        train_q = qual_train.loc[train_mask].copy()
        if val_q.empty:
            continue
        prior_event_count = int(pd.to_numeric(train_q.get("event_key"), errors="coerce").dropna().nunique()) if not train_q.empty else 0
        model_obj: Optional[object] = None
        source = "oof_fallback"
        train_max: Optional[float] = None
        if prior_event_count > 0:
            train_max = float(pd.to_numeric(train_q["event_key"], errors="coerce").max())
        if prior_event_count >= int(min_prior_events):
            qual_model = train_model(
                train_q,
                qual_feature_cols,
                enable_dl_candidates=config.enable_dl_candidates,
                compare_families=config.compare_families,
                dl_device=config.dl_device,
                dl_arch=config.dl_arch,
                dl_hyperparams=config.dl_hyperparams,
                dl_seed=config.dl_seed,
                f1_model=config.f1_model,
            )
            model_obj = qual_model.model
            source = f"oof_{qual_model.model_name}"
        preds = predict_with_model(model_obj, val_q, qual_feature_cols, qual_fallback_cols)
        top10, top3 = _predict_probabilities(model_obj, preds)
        signal = _build_qualifying_signal_frame(
            val_q,
            preds,
            top10,
            top3,
            source=source,
            training_event_max=train_max,
        )
        signals.append(signal)
    if not signals:
        return pd.DataFrame()
    notes.append(
        "[Race<-Quali] Signal qualif historique genere out-of-fold par event "
        f"({len(signals)} events, min_prior_events={int(min_prior_events)}).",
    )
    return pd.concat(signals, ignore_index=True)


def _add_race_context_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    numeric_cols = [
        "qualy_position",
        "grid_position",
        "qualy_pred_position",
        "track_qualy_importance",
        "track_overtake_propensity",
        "track_safety_car_propensity",
        "track_chaos_index",
        "track_safety_car_prior",
        "track_dnf_prior",
        "track_strategy_variance_prior",
        "track_weather_uncertainty_prior",
        "race_generation_variance_prior",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "qualy_pred_position" in out.columns and "qualy_position" in out.columns:
        out["qualy_pred_vs_actual_gap"] = (out["qualy_position"] - out["qualy_pred_position"]).abs()
    else:
        out["qualy_pred_vs_actual_gap"] = float("nan")

    if "track_qualy_importance" in out.columns:
        qualy_importance = out["track_qualy_importance"].clip(lower=0.0, upper=1.0).fillna(0.5)
    else:
        overtake_raw = out["track_overtake_propensity"] if "track_overtake_propensity" in out.columns else pd.Series(
            0.5,
            index=out.index,
            dtype=float,
        )
        safety_raw = out["track_safety_car_propensity"] if "track_safety_car_propensity" in out.columns else pd.Series(
            0.2,
            index=out.index,
            dtype=float,
        )
        overtake = pd.to_numeric(overtake_raw, errors="coerce").reindex(out.index).fillna(0.5)
        safety = pd.to_numeric(safety_raw, errors="coerce").reindex(out.index).fillna(0.2)
        qualy_importance = (1.0 - (0.65 * overtake) - (0.35 * safety)).clip(lower=0.0, upper=1.0)
        out["track_qualy_importance"] = qualy_importance

    if "qualy_pred_position" in out.columns:
        out["qualy_pred_position_track_adj"] = out["qualy_pred_position"] * (0.35 + qualy_importance)
    else:
        out["qualy_pred_position_track_adj"] = float("nan")
    circuit_qualy_importance = (
        pd.to_numeric(out["circuit_qualifying_importance"], errors="coerce")
        if "circuit_qualifying_importance" in out.columns
        else qualy_importance
    )
    circuit_qualy_importance = circuit_qualy_importance.reindex(out.index).fillna(qualy_importance).clip(
        lower=0.0,
        upper=1.0,
    )

    qualy_actual = pd.to_numeric(out.get("qualy_position"), errors="coerce")
    grid_position = pd.to_numeric(out.get("grid_position"), errors="coerce")
    if isinstance(grid_position, pd.Series):
        out["grid_position"] = grid_position.where(grid_position.notna(), qualy_actual)
        out["grid_position_track_adj"] = out["grid_position"] * (0.35 + qualy_importance)
    else:
        out["grid_position"] = qualy_actual if isinstance(qualy_actual, pd.Series) else float("nan")
        out["grid_position_track_adj"] = float("nan")
    race_start = pd.to_numeric(out.get("grid_position"), errors="coerce")
    qualy_pred = pd.to_numeric(out.get("qualy_pred_position"), errors="coerce")
    if isinstance(race_start, pd.Series) and isinstance(qualy_pred, pd.Series):
        out["qualy_context_position"] = (qualy_importance * race_start) + ((1.0 - qualy_importance) * qualy_pred)
        out["qualy_context_position"] = out["qualy_context_position"].where(
            out["qualy_context_position"].notna(),
            race_start,
        )
        out["qualy_context_position"] = out["qualy_context_position"].where(
            out["qualy_context_position"].notna(),
            qualy_pred,
        )
    elif isinstance(race_start, pd.Series):
        out["qualy_context_position"] = race_start
    elif isinstance(qualy_actual, pd.Series):
        out["qualy_context_position"] = qualy_actual
    elif isinstance(qualy_pred, pd.Series):
        out["qualy_context_position"] = qualy_pred
    else:
        out["qualy_context_position"] = float("nan")

    out["qualy_context_position_track_adj"] = (
        pd.to_numeric(out["qualy_context_position"], errors="coerce") * (0.35 + qualy_importance)
    )
    out["qualy_position_circuit_importance_adj"] = (
        pd.to_numeric(out.get("qualy_position"), errors="coerce") * (0.35 + circuit_qualy_importance)
    )
    out["qualy_pred_position_circuit_importance_adj"] = (
        pd.to_numeric(out.get("qualy_pred_position"), errors="coerce") * (0.35 + circuit_qualy_importance)
    )

    return out


def _merge_predicted_qualifying_context(
    provider: BaseProvider,
    config: PredictionConfig,
    race_train: pd.DataFrame,
    race_features: pd.DataFrame,
    notes: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if race_train.empty and race_features.empty:
        return race_train, race_features

    qual_train, qual_notes = build_training_data(
        provider=provider,
        mode="qualifying",
        train_seasons=config.train_seasons,
        target_year=config.year,
        target_round=config.round_number,
        include_standings=False,
    )
    qual_train = _apply_season_sample_weighting(qual_train, config, notes)
    qual_features, qual_feature_notes = build_current_features(
        provider=provider,
        mode="qualifying",
        year=config.year,
        round_number=config.round_number,
        include_standings=False,
        history=qual_train,
    )
    for note in qual_notes + qual_feature_notes:
        note_lower = note.lower()
        if note_lower.startswith("echec") or "indisponible" in note_lower or "aucune" in note_lower or "pas assez" in note_lower:
            notes.append(f"[Race<-Quali] {note}")

    if qual_train.empty and qual_features.empty:
        notes.append("[Race<-Quali] Impossible de construire un signal de qualif predit.")
        return _add_race_context_interactions(race_train), _add_race_context_interactions(race_features)

    if config.disable_runsim_features:
        qual_train = _drop_runsim_columns(qual_train)
        qual_features = _drop_runsim_columns(qual_features)
    if config.disable_circuit_features:
        qual_train = _drop_circuit_columns(qual_train)
        qual_features = _drop_circuit_columns(qual_features)

    qual_feature_cols, qual_fallback_cols = _qualifying_feature_sets(
        disable_runsim=config.disable_runsim_features,
        disable_circuit=config.disable_circuit_features,
    )
    qual_model = train_model(
        qual_train,
        qual_feature_cols,
        enable_dl_candidates=config.enable_dl_candidates,
        compare_families=config.compare_families,
        dl_device=config.dl_device,
        dl_arch=config.dl_arch,
        dl_hyperparams=config.dl_hyperparams,
        dl_seed=config.dl_seed,
        f1_model=config.f1_model,
    )
    notes.append(f"[Race<-Quali] Modele qualif contexte: {qual_model.model_name}.")
    for note in qual_model.notes:
        if note.startswith("Modele retenu:") or note.startswith("Prediction qualif:") or "fallback" in note.lower():
            notes.append(f"[Race<-Quali] {note}")

    train_signal = pd.DataFrame()
    if not qual_train.empty:
        train_signal = _build_oof_qualifying_signal_frame(
            qual_train=qual_train,
            qual_feature_cols=qual_feature_cols,
            qual_fallback_cols=qual_fallback_cols,
            config=config,
            notes=notes,
        )

    current_signal = pd.DataFrame()
    if not qual_features.empty:
        qual_current_pred = predict_with_model(qual_model.model, qual_features, qual_feature_cols, qual_fallback_cols)
        qual_current_top10, qual_current_top3 = _predict_probabilities(qual_model.model, qual_current_pred)
        current_signal = _build_qualifying_signal_frame(
            qual_features,
            qual_current_pred,
            qual_current_top10,
            qual_current_top3,
            source=f"current_{qual_model.model_name}",
        )

    if not race_train.empty and not train_signal.empty:
        left = race_train.copy()
        left["driver_id"] = left["driver_id"].astype(str)
        if "event_key" in left.columns and "event_key" in train_signal.columns:
            left["event_key"] = pd.to_numeric(left["event_key"], errors="coerce")
            race_train = left.merge(train_signal, on=["event_key", "driver_id"], how="left")
        else:
            race_train = left.merge(
                train_signal.drop(columns=["event_key"], errors="ignore"),
                on=["driver_id"],
                how="left",
            )

    if not race_features.empty and not current_signal.empty:
        right = race_features.copy()
        right["driver_id"] = right["driver_id"].astype(str)
        if "event_key" in right.columns and "event_key" in current_signal.columns:
            right["event_key"] = pd.to_numeric(right["event_key"], errors="coerce")
            race_features = right.merge(current_signal, on=["event_key", "driver_id"], how="left")
        else:
            race_features = right.merge(
                current_signal.drop(columns=["event_key"], errors="ignore"),
                on=["driver_id"],
                how="left",
            )

    race_train = _add_race_context_interactions(race_train)
    race_features = _add_race_context_interactions(race_features)
    return race_train, race_features


def run_prediction(config: PredictionConfig) -> PredictionResult:
    mode_live = str(config.f1_mode or "offline").strip().lower() == "live"
    if mode_live:
        if str(config.mode).strip().lower() != "race":
            notes = ["f1_mode=live is only supported with mode=race in Horizon B v1."]
            return PredictionResult(
                version=compute_version(config.round_number, config.include_standings),
                table=pd.DataFrame(),
                notes=notes,
                model_name="live_disabled",
                model_family="live",
                device_used=None,
                dl_available=False,
                candidate_leaderboard=[],
                extras={
                    "live_summary": {
                        "available": False,
                        "reason": "live_mode_requires_race",
                        "f1_mode": str(config.f1_mode),
                    }
                },
            )
        live_result = run_live_race_prediction(config)
        live_summary = dict(live_result.summary)
        version = compute_version(config.round_number, config.include_standings)
        return PredictionResult(
            version=version,
            table=live_result.snapshot,
            notes=live_result.notes,
            model_name=str(config.f1_live_model or "ssm_v1"),
            model_family="live_ssm",
            device_used=None,
            dl_available=False,
            candidate_leaderboard=[],
            extras={
                "live_summary": live_summary,
                "trace_path": live_summary.get("trace_path"),
                "trace_path_jsonl": live_summary.get("trace_path_jsonl"),
                "trace_format_effective": live_summary.get("trace_format_effective"),
            },
        )

    provider: BaseProvider
    if config.source == "fastf1":
        provider = FastF1Provider(config.cache_dir)
    elif config.source == "openf1":
        provider = OpenF1Provider(
            cache_dir=config.cache_dir,
            target_round=config.round_number,
            meeting_name=config.meeting_name,
            country_name=config.country_name,
        )
    else:
        provider = LocalWeekendProvider(weekends_dir=config.weekends_dir)

    train, notes = build_training_data(
        provider=provider,
        mode=config.mode,
        train_seasons=config.train_seasons,
        target_year=config.year,
        target_round=config.round_number,
        include_standings=config.include_standings,
    )
    train = _apply_season_sample_weighting(train, config, notes)

    features, feature_notes = build_current_features(
        provider=provider,
        mode=config.mode,
        year=config.year,
        round_number=config.round_number,
        include_standings=config.include_standings,
        history=train,
    )
    notes.extend(feature_notes)

    if config.mode == "race":
        train, features = _merge_predicted_qualifying_context(
            provider=provider,
            config=config,
            race_train=train,
            race_features=features,
            notes=notes,
        )

    if config.disable_runsim_features:
        train = _drop_runsim_columns(train)
        features = _drop_runsim_columns(features)
        notes.append("Ablation active: run-sim features supprimees (fp_quali_sim_*, fp_race_sim_*, fp_slow_lap_ratio, fp_quali_vs_race_gap).")
    if config.disable_circuit_features:
        train = _drop_circuit_columns(train)
        features = _drop_circuit_columns(features)
        notes.append("Ablation active: circuit-card features and interactions supprimes.")

    if config.mode == "qualifying":
        feature_cols, fallback_cols = _qualifying_feature_sets(
            disable_runsim=config.disable_runsim_features,
            disable_circuit=config.disable_circuit_features,
        )
    else:
        feature_cols, fallback_cols = _race_feature_sets(
            include_standings=config.include_standings,
            disable_runsim=config.disable_runsim_features,
            disable_circuit=config.disable_circuit_features,
        )

    training_result = train_model(
        train,
        feature_cols,
        enable_dl_candidates=config.enable_dl_candidates,
        compare_families=config.compare_families,
        dl_device=config.dl_device,
        dl_arch=config.dl_arch,
        dl_hyperparams=config.dl_hyperparams,
        dl_seed=config.dl_seed,
        f1_model=config.f1_model,
    )
    notes.extend(training_result.notes)
    if training_result.model is None:
        notes.append(
            "Fallback heuristique actif: qualif=blend FP pace empirique; race=position qualif si disponible.",
        )
    preds = predict_with_model(training_result.model, features, feature_cols, fallback_cols)
    output = features.copy()
    output["pred"] = preds
    proba_win = _predict_probability(training_result.model, preds, label="win", k=1)
    proba_top10, proba_top3 = _predict_probabilities(training_result.model, preds)
    proba_win = pd.Series(np.minimum(proba_win.clip(0.0, 1.0), proba_top3), index=preds.index, dtype=float)
    output["proba_win"] = proba_win
    output["proba_top10"] = proba_top10
    output["proba_top3"] = proba_top3

    if str(config.f1_listwise).strip().lower() == "pl_gumbel":
        effective_temperature = (
            float(training_result.listwise_temperature)
            if training_result.listwise_temperature is not None
            else float(config.f1_pl_temperature)
        )
        listwise = _pl_gumbel_listwise(
            frame=output,
            preds=preds,
            mode=config.mode,
            samples=int(config.f1_pl_samples),
            temperature=effective_temperature,
            seed=int(config.f1_listwise_seed),
        )
        output = output.join(listwise)
        output["old_rank_based_top10"] = proba_top10
        output["old_rank_based_top3"] = proba_top3
        output["old_rank_based_win"] = proba_win
        output["proba_win"] = output["p_win"]
        output["proba_top10"] = output["p_top10"]
        output["proba_top3"] = output["p_top3"]
        notes.append(
            "Listwise PL active: proba_win/proba_top10/proba_top3 remplaces par p_win/p_top10/p_top3 "
            f"(seed stable par event, temperature={effective_temperature:.3f}).",
        )
    else:
        output["listwise_enabled"] = False

    if "driver_name" not in output.columns:
        if "driver_id" in output.columns:
            output["driver_name"] = output["driver_id"]
        else:
            output["driver_name"] = pd.Series(dtype=str)
    elif "driver_id" in output.columns:
        output["driver_name"] = output["driver_name"].fillna(output["driver_id"])

    version = compute_version(config.round_number, config.include_standings)
    all_rows = format_prediction_table(output, top_n=None)
    table = format_prediction_table(output, top_n=10)
    circuit_card = circuit_card_payload_from_frame(output)
    return PredictionResult(
        version=version,
        table=table,
        notes=notes,
        model_name=training_result.model_name,
        model_family=training_result.model_family,
        device_used=training_result.device_used,
        dl_available=training_result.dl_available,
        candidate_leaderboard=training_result.candidate_leaderboard,
        extras={
            "all_prediction_rows": _prediction_records(all_rows),
            "circuit_card": circuit_card,
            "circuit_feature_columns": list(CIRCUIT_NUMERIC_FEATURES + CIRCUIT_INTERACTION_FEATURES),
            "listwise_temperature": training_result.listwise_temperature,
            "probability_audit": training_result.probability_audit,
        },
    )
