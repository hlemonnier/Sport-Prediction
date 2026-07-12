"""Prediction orchestration."""

from __future__ import annotations

import json
from typing import List, Optional

import numpy as np
import pandas as pd

from packages.f1.data.schemas.circuit import (
    CIRCUIT_INTERACTION_FEATURES,
    CIRCUIT_NUMERIC_FEATURES,
    circuit_card_payload_from_frame,
)
from packages.f1.domain import (
    EvidenceSource,
    GridEntryStatus,
    GridRevisionPhase,
    OfficialGridEntry,
    OfficialGridRevision,
    ResolutionStatus,
    WeekendFormat,
    build_weekend_contract,
    resolve_grand_prix_start,
)
from packages.f1.orchestration.contracts import architecture_payload
from packages.f1.data.schemas import PredictionConfig, PredictionResult
from packages.f1.features.assembly import build_current_features, build_training_data
from packages.f1.models.live_race.predict import run_live_race_prediction
from packages.f1.data.providers import BaseProvider, FastF1Provider, LocalWeekendProvider, OpenF1Provider
from packages.f1.models.probability import pl_gumbel_probabilities
from packages.f1.models.race_probability import race_stochastic_score_layer as _race_stochastic_score_layer
from packages.f1.models.training import train_model
from packages.f1.data.utils import format_prediction_table
from packages.f1.features.weather import apply_f1_weather_to_features, fetch_f1_weather_summary
from packages.f1.features.wet import add_f1_wet_pace_interactions, wet_pace_evidence_rows

RUNSIM_EXACT_COLUMNS = {"fp_slow_lap_ratio", "fp_quali_vs_race_gap"}
WEATHER_SCENARIO_COLUMNS = (
    "track_weather_uncertainty",
    "track_weather_uncertainty_prior",
    "race_generation_variance_prior",
)
QUALIFYING_CIRCUIT_NUMERIC_FEATURES = (
    "circuit_downforce_demand",
    "circuit_power_sensitivity",
    "circuit_low_speed_corner_demand",
    "circuit_high_speed_corner_demand",
    "circuit_traction_demand",
    "circuit_braking_demand",
    "circuit_tyre_degradation",
    "circuit_kerb_bump_penalty",
    "circuit_drs_effectiveness",
    "circuit_card_reliability",
)
QUALIFYING_CIRCUIT_INTERACTION_FEATURES = (
    "fp_weighted_delta_downforce_adj",
    "fp_weighted_delta_power_adj",
    "fp_quali_sim_delta_downforce_adj",
    "driver_archetype_form_3_fp_weighted_delta",
    "team_archetype_form_3_fp_weighted_delta",
    "driver_circuit_hist_fp_weighted_delta",
    "team_circuit_hist_fp_weighted_delta",
)
RACE_CONTEXT_EVIDENCE_COLUMNS = (
    "track_finish_order_mobility",
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
)
RACE_STRENGTH_EVIDENCE_COLUMNS = (
    "fp_mean_delta",
    "fp_weighted_delta",
    "fp_quali_sim_delta",
    "fp_quali_sim_rank",
    "fp_race_sim_delta",
    "fp_race_sim_rank",
    "driver_ewma_fp_mean_delta",
    "driver_form_3_fp_mean_delta",
    "driver_form_5_fp_mean_delta",
    "driver_ewma_fp_weighted_delta",
    "driver_form_3_fp_weighted_delta",
    "driver_ewma_fp_race_sim_delta",
    "driver_form_3_fp_race_sim_delta",
    "team_ewma_fp_mean_delta",
    "team_form_3_fp_mean_delta",
    "team_form_5_fp_mean_delta",
    "team_ewma_fp_weighted_delta",
    "team_form_3_fp_weighted_delta",
    "event_pace_index",
    "driver_vs_team_fp_weighted_delta",
    "position_start",
)
CURRENT_WEEKEND_EVIDENCE_COLUMNS = (
    "pace_sessions_available",
    "fp_total_laps",
    "fp1_delta",
    "fp2_delta",
    "fp3_delta",
    "sq_delta",
    "sprint_delta",
    "latest_qualifying_rehearsal_rank",
    "latest_qualifying_rehearsal_delta",
    "latest_qualifying_rehearsal_coverage",
    "latest_qualifying_rehearsal_source",
    "qualy_position",
    "qualy_gap_to_best",
    "grid_position",
    "grid_source",
    "grid_status",
)
RACE_INFORMATION_HORIZONS = {
    "pre_fp_provisional",
    "post_fp_pre_qualifying",
    "post_qualifying_pre_grid",
    "post_grid_pre_race",
}
PRE_QUALIFYING_RACE_HORIZONS = {"pre_fp_provisional", "post_fp_pre_qualifying"}
CURRENT_WEEKEND_PRACTICE_EXACT_COLUMNS = {
    "pace_sessions_available",
    "quali_sim_sessions_available",
    "race_sim_sessions_available",
    "wet_sim_sessions_available",
    "wet_pace_evidence_reliability",
    "event_pace_index",
    "driver_vs_team_fp_weighted_delta",
    "circuit_fit_index",
}


def _stable_column_values(frame: pd.DataFrame, column: str) -> list[object]:
    """Return deterministic, JSON-safe non-null values for an evidence column."""

    if frame.empty or column not in frame.columns:
        return []
    values: list[object] = []
    for raw_value in frame[column].dropna().tolist():
        if isinstance(raw_value, np.generic):
            raw_value = raw_value.item()
        value: object
        if isinstance(raw_value, (bool, int, float, str)):
            value = raw_value
        else:
            value = str(raw_value)
        if value not in values:
            values.append(value)
    return sorted(values, key=lambda item: str(item))


def _practice_quality_evidence(features: pd.DataFrame) -> dict[str, object]:
    """Summarize the point-in-time practice sample without overstating run intent."""

    evidence: dict[str, object] = {
        "contract": "f1_practice_lap_features_v3_quality_weighted",
        "neutralised_laps_excluded_from_pace": True,
        "deleted_and_pit_transition_laps_excluded_from_pace": True,
        "run_intent_labels_calibrated": False,
        "degradation_is_fuel_corrected": False,
    }
    roster_policies = _stable_column_values(features, "fp_roster_policy")
    if roster_policies:
        evidence["roster_policy"] = roster_policies
    rehearsal_sources = _stable_column_values(features, "latest_qualifying_rehearsal_source")
    if rehearsal_sources:
        evidence["latest_qualifying_rehearsal_source"] = rehearsal_sources
    if "latest_qualifying_rehearsal_coverage" in features.columns:
        rehearsal_coverage = pd.to_numeric(
            features["latest_qualifying_rehearsal_coverage"],
            errors="coerce",
        ).dropna()
        if not rehearsal_coverage.empty:
            evidence["latest_qualifying_rehearsal_coverage"] = float(rehearsal_coverage.mean())
    if "latest_qualifying_rehearsal_imputed" in features.columns:
        evidence["latest_qualifying_rehearsal_imputed_rows"] = int(
            features["latest_qualifying_rehearsal_imputed"].fillna(True).astype(bool).sum()
        )
    for column in (
        "fp_roster_latest_session_driver_count",
        "fp_roster_active_driver_count",
        "fp_roster_carried_forward_driver_count",
        "fp_roster_superseded_driver_count",
    ):
        if column not in features.columns:
            continue
        numeric = pd.to_numeric(features[column], errors="coerce").dropna()
        if not numeric.empty:
            evidence[column] = sorted({int(value) for value in numeric.tolist()})
    for column in (
        "fp_raw_timed_laps",
        "fp_representative_laps",
        "fp_invalid_laps",
        "fp_deleted_laps",
        "fp_neutralised_laps",
        "fp_pit_out_laps",
        "fp_pit_in_laps",
    ):
        if column not in features.columns:
            continue
        numeric = pd.to_numeric(features[column], errors="coerce")
        evidence[column] = int(numeric.fillna(0.0).sum())
    for column in (
        "fp_lap_quality_ratio",
        "fp_quali_sim_evidence_share",
        "fp_race_sim_evidence_share",
        "fp_run_intent_unclassified_share",
        "fp_race_sim_raw_degradation_sec_per_lap",
        "fp_race_sim_raw_degradation_mad",
    ):
        if column not in features.columns:
            continue
        numeric = pd.to_numeric(features[column], errors="coerce").dropna()
        if not numeric.empty:
            evidence[f"{column}_field_mean"] = float(numeric.mean())
    return evidence


def _is_current_weekend_practice_column(column: str) -> bool:
    return (
        column.startswith(("fp_", "fp1_", "fp2_", "fp3_", "sq_", "sprint_"))
        or column.startswith("latest_qualifying_rehearsal_")
        or column in CURRENT_WEEKEND_PRACTICE_EXACT_COLUMNS
    )


def _scrub_current_weekend_practice(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in list(out.columns):
        if _is_current_weekend_practice_column(str(column)):
            out[column] = float("nan")
    return out


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
    """Return the honest race-start baseline when no trained model exists.

    Circuit mobility, DRS, difficulty, and chaos priors are intentionally not
    allowed to alter the mean finish here.  Those signals require empirical
    validation through a selected model; the no-training path stays anchored
    to the grid or causal qualifying estimate.
    """

    start = _first_numeric_series(
        features,
        ["grid_position", "qualy_context_position", "qualy_position", "qualy_pred_rank", "qualy_pred_position"],
    )
    if start is None or start.notna().sum() == 0:
        return None
    return pd.Series(start.fillna(start.median(skipna=True)), index=features.index, dtype=float)


def _hierarchical_fallback(
    features: pd.DataFrame,
    fallback_cols: List[str],
) -> pd.Series:
    if features.empty:
        return pd.Series(dtype=float)
    fallback_col_set = set(fallback_cols)
    race_context = bool(
        fallback_col_set.intersection(
            {
                "grid_position",
                "qualy_position",
                "qualy_context_position",
                "fp_race_sim_rank",
                "fp_race_sim_delta",
            }
        )
    )
    if not race_context and "qualy_position" in features.columns:
        race_context = True
    if not race_context and "grid_position" in features.columns:
        race_context = any(
            column in features.columns
            for column in (
                "fp_race_sim_rank",
                "fp_race_sim_delta",
                "track_finish_order_mobility",
            )
        )

    # Empirically strongest no-training fallbacks on the local 2025 holdout:
    # qualifying: FP mean rank + event pace + qualifying-sim rank;
    # race: actual qualifying position. These avoid overfitting a model when no
    # historical seasons are available while still preserving event-local signal.
    if race_context:
        race_score = _race_grid_delta_fallback(features)
        if race_score is not None:
            return race_score

    rehearsal_score = _first_numeric_series(features, ["latest_qualifying_rehearsal_rank"])
    if rehearsal_score is not None and rehearsal_score.notna().any():
        return rehearsal_score.fillna(float(rehearsal_score.median(skipna=True)))

    qualifying_score = _weighted_rank_score(
        features,
        [
            ("event_pace_index", 2.0, True),
            ("fp_mean_rank", 2.0, True),
            ("fp_quali_sim_rank", 1.0, True),
            ("fp_quali_sim_delta_downforce_adj", 0.75, True),
            ("fp_weighted_delta_downforce_adj", 0.55, True),
            ("fp_weighted_delta_power_adj", 0.45, True),
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
    if "qualy_pred_rank" in features.columns:
        components.append((0.10, _rank_percentile(features["qualy_pred_rank"])))
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
            "fp_weighted_delta",
            "fp_delta_std",
            "fp_mean_top3_delta",
            "fp_mean_lap_std",
            "driver_form_3_fp_quali_sim_delta",
            "driver_ewma_fp_quali_sim_delta",
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
            "fp_mean_delta",
            "fp_mean_rank",
            "circuit_downforce_demand",
            "circuit_power_sensitivity",
        ]
        + (
            [
                "fp_race_sim_delta",
                "fp_race_sim_rank",
                "fp_slow_lap_ratio",
                "fp_quali_vs_race_gap",
                "driver_form_3_fp_race_sim_delta",
                "driver_ewma_fp_race_sim_delta",
                "fp_race_sim_delta_tyre_adj",
                "fp_race_sim_delta_power_adj",
                "qualy_context_position",
                "qualy_context_position_track_adj",
                "qualy_position_track_adj",
                "qualy_position_circuit_importance_adj",
                "qualy_pred_rank",
                "qualy_pred_position",
                "qualy_pred_position_track_adj",
                "qualy_pred_position_circuit_importance_adj",
                "qualy_pred_vs_actual_gap",
                "track_chaos_index",
                "circuit_overtaking_difficulty",
                "circuit_qualifying_importance",
                "circuit_fit_index",
            ]
            if race_context
            else []
        ),
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


def _event_total_probability_with_caps(
    scores: pd.Series,
    probabilities: pd.Series,
    *,
    k: int,
    caps: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(scores, errors="coerce")
    raw = pd.to_numeric(probabilities, errors="coerce").reindex(numeric.index)
    cap_values = pd.to_numeric(caps, errors="coerce").reindex(numeric.index).fillna(0.0).clip(0.0, 1.0)
    valid = numeric.notna() & raw.notna() & (cap_values > 0.0)
    output = pd.Series(0.0, index=numeric.index, dtype=float)
    if valid.sum() == 0:
        return output
    expected = min(float(k), float(valid.sum()), float(cap_values.loc[valid].sum()))
    if expected <= 0.0:
        return output
    weights = raw.loc[valid].clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
    if float(weights.sum()) <= 0.0:
        ranked = numeric.loc[valid].rank(method="average", ascending=True)
        weights = np.exp(-((ranked.to_numpy(dtype=float) - 1.0) / max(1.0, float(len(ranked)) / 6.0)))
    caps_arr = cap_values.loc[valid].to_numpy(dtype=float)
    values = np.zeros(len(weights), dtype=float)
    remaining = np.ones(len(weights), dtype=bool)
    remaining_total = float(expected)
    for _ in range(len(weights) + 1):
        if remaining_total <= 1e-12 or not remaining.any():
            break
        w = weights[remaining]
        cap = caps_arr[remaining]
        if float(w.sum()) <= 0.0:
            allocation = np.full(int(remaining.sum()), remaining_total / float(remaining.sum()))
        else:
            allocation = remaining_total * (w / float(w.sum()))
        capped = allocation >= cap
        remaining_idx = np.flatnonzero(remaining)
        if not capped.any():
            values[remaining_idx] = allocation
            break
        cap_idx = remaining_idx[capped]
        values[cap_idx] = caps_arr[cap_idx]
        remaining_total -= float(caps_arr[cap_idx].sum())
        remaining[cap_idx] = False
    output.loc[valid] = values
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
    top3 = pd.Series(np.minimum(top3, top10), index=preds.index, dtype=float)
    top3 = _event_total_probability_with_caps(preds, top3, k=3, caps=top10)
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
    seed_key = pd.Series(index=frame.index, dtype=object)
    for event_group in group_id.loc[valid_mask].dropna().unique().tolist():
        idx = group_id.index[(group_id == event_group) & valid_mask]
        if len(idx) == 0:
            continue
        event_rows = frame.loc[idx]
        seed_key.loc[idx] = _canonical_event_key(event_rows, mode=mode, fallback_group_id=str(event_group))

    sampled = pl_gumbel_probabilities(
        scores=preds,
        event_key=group_id,
        samples=int(samples),
        temperature=float(temperature),
        seed=int(seed),
        event_seed_key=seed_key,
    )
    for column in ["utility", "p_win", "p_top3", "p_top10", "exp_pos", "pos_p10", "pos_p50", "pos_p90"]:
        output[column] = sampled[column].reindex(output.index)
    return output


def _qualifying_feature_sets(
    disable_runsim: bool = False,
    disable_circuit: bool = False,
) -> tuple[List[str], List[str]]:
    feature_cols = [
        "latest_qualifying_rehearsal_rank",
        "latest_qualifying_rehearsal_delta",
        "latest_qualifying_rehearsal_coverage",
        "latest_qualifying_rehearsal_imputed",
        "fp1_delta",
        "fp2_delta",
        "fp3_delta",
        "sq_delta",
        "fp_mean_delta",
        "fp_weighted_delta",
        "fp_quali_sim_delta",
        "fp_quali_sim_rank",
        "fp_quali_sim_laps",
        "quali_sim_sessions_available",
        "fp_wet_sim_laps",
        "wet_sim_sessions_available",
        "wet_pace_evidence_reliability",
        "fp_wet_sim_delta_weather_adj",
        "fp_wet_sim_rank_weather_adj",
        "fp_delta_std",
        "pace_sessions_available",
        "fp_mean_top3_delta",
        "fp_mean_lap_std",
        "fp_total_laps",
        "fp1_rank",
        "fp2_rank",
        "fp3_rank",
        "sq_rank",
        "fp_mean_rank",
        "driver_ewma_fp_mean_delta",
        "driver_form_3_fp_mean_delta",
        "driver_form_5_fp_mean_delta",
        "driver_ewma_fp_weighted_delta",
        "driver_form_3_fp_weighted_delta",
        "driver_ewma_fp_quali_sim_delta",
        "driver_form_3_fp_quali_sim_delta",
        "driver_form_3_vs_team_fp_weighted_delta",
        "team_ewma_fp_mean_delta",
        "team_form_3_fp_mean_delta",
        "team_form_5_fp_mean_delta",
        "team_ewma_fp_weighted_delta",
        "team_form_3_fp_weighted_delta",
        "event_driver_hist_idx",
        "event_pace_index",
        "driver_vs_team_fp_weighted_delta",
        "track_weather_uncertainty",
        "track_weather_uncertainty_prior",
    ]
    feature_cols.extend(QUALIFYING_CIRCUIT_NUMERIC_FEATURES)
    feature_cols.extend(QUALIFYING_CIRCUIT_INTERACTION_FEATURES)
    fallback_cols = [
        "latest_qualifying_rehearsal_rank",
        "latest_qualifying_rehearsal_delta",
        "latest_qualifying_rehearsal_coverage",
        "latest_qualifying_rehearsal_imputed",
        "fp_mean_delta",
        "fp_weighted_delta",
        "fp_quali_sim_delta",
        "fp_quali_sim_rank",
        "fp_quali_sim_laps",
        "fp_wet_sim_delta_weather_adj",
        "fp_wet_sim_rank_weather_adj",
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
        "driver_form_3_vs_team_fp_weighted_delta",
        "driver_ewma_fp_weighted_delta",
        "team_form_3_fp_mean_delta",
        "team_form_5_fp_mean_delta",
        "event_pace_index",
        "driver_vs_team_fp_weighted_delta",
        "event_driver_hist_idx",
        "track_weather_uncertainty",
        "track_weather_uncertainty_prior",
    ]
    fallback_cols.extend(QUALIFYING_CIRCUIT_NUMERIC_FEATURES)
    fallback_cols.extend(QUALIFYING_CIRCUIT_INTERACTION_FEATURES)
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
        "fp_wet_sim_laps",
        "wet_sim_sessions_available",
        "wet_pace_evidence_reliability",
        "fp_wet_sim_delta_weather_adj",
        "fp_wet_sim_rank_weather_adj",
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
        "qualy_pred_rank",
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
        "track_finish_order_mobility",
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
        "qualy_pred_rank",
        "qualy_pred_rank_pct",
        "qualy_pred_top10_proba",
        "position_start",
        "fp_weighted_delta",
        "fp_weighted_delta_track_adj",
        "fp_quali_sim_delta",
        "fp_quali_sim_rank",
        "fp_quali_sim_laps",
        "fp_wet_sim_delta_weather_adj",
        "fp_wet_sim_rank_weather_adj",
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
        "track_finish_order_mobility",
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
        out["qualy_pred_rank"] = float("nan")
        if valid.any():
            out.loc[valid, "qualy_pred_rank_pct"] = (
                out.loc[valid]
                .groupby("event_key", sort=False)["qualy_pred_position"]
                .rank(method="average", pct=True, ascending=True)
            )
            out.loc[valid, "qualy_pred_rank"] = (
                out.loc[valid]
                .groupby("event_key", sort=False)["qualy_pred_position"]
                .rank(method="first", ascending=True)
            )
    else:
        out["qualy_pred_rank_pct"] = _rank_percentile(out["qualy_pred_position"])
        out["qualy_pred_rank"] = out["qualy_pred_position"].rank(method="first", ascending=True)
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
                f1_pl_samples=getattr(config, "f1_pl_samples", 2000),
                f1_listwise_seed=getattr(config, "f1_listwise_seed", 42),
                race_delta_constraint_mode=getattr(config, "race_delta_constraint_mode", "constrained"),
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


def _resolve_race_information_horizon(
    features: pd.DataFrame,
    requested: str = "auto",
    *,
    prediction_as_of: Optional[str] = None,
) -> str:
    normalized = str(requested or "auto").strip().lower().replace("-", "_")
    aliases = {
        "pre_fp": "pre_fp_provisional",
        "prequal": "post_fp_pre_qualifying",
        "pre_qualifying": "post_fp_pre_qualifying",
        "postqual": "post_qualifying_pre_grid",
        "post_qualifying": "post_qualifying_pre_grid",
        "official_grid": "post_grid_pre_race",
        "post_grid": "post_grid_pre_race",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized != "auto":
        if normalized not in RACE_INFORMATION_HORIZONS:
            allowed = ", ".join(["auto", *sorted(RACE_INFORMATION_HORIZONS)])
            raise ValueError(f"Unknown race_information_horizon={requested!r}; expected one of: {allowed}.")
        return normalized

    if features.empty:
        return "pre_fp_provisional"
    # Metadata labels alone do not select the final-grid horizon.  The same
    # domain resolver used by the deployed score path must accept the full
    # publication first.
    if not _resolved_final_grid_frame(features, as_of=prediction_as_of).empty:
        return "post_grid_pre_race"
    qualy = pd.to_numeric(
        features.get("qualy_position", pd.Series(float("nan"), index=features.index)),
        errors="coerce",
    )
    if qualy.notna().any():
        return "post_qualifying_pre_grid"
    fp_sessions = pd.to_numeric(
        features.get("pace_sessions_available", pd.Series(0.0, index=features.index)),
        errors="coerce",
    ).fillna(0.0)
    fp_laps = pd.to_numeric(
        features.get("fp_total_laps", pd.Series(0.0, index=features.index)),
        errors="coerce",
    ).fillna(0.0)
    if (fp_sessions.gt(0.0) | fp_laps.gt(0.0)).any():
        return "post_fp_pre_qualifying"
    return "pre_fp_provisional"


def _resolved_final_grid_frame(
    frame: pd.DataFrame,
    *,
    as_of: Optional[str] = None,
) -> pd.DataFrame:
    """Validate a provider grid through the domain resolver or fail closed.

    A numeric ``grid_position`` plus a friendly source label is not proof of a
    point-in-time final grid.  The adapter requires one immutable publication
    timestamp, revision id, final phase, complete roster, and explicit row
    statuses before the scheduled order can enter the final-grid horizon.
    """

    required = {
        "driver_id",
        "grid_position",
        "grid_source",
        "grid_status",
        "grid_revision_phase",
        "grid_evidence_as_of",
        "grid_evidence_id",
        "grid_evidence_complete",
        "event_year",
    }
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()
    evidence = frame.loc[frame["grid_source"].astype(str).eq("pre_race_official_grid")].copy()
    if evidence.empty:
        return pd.DataFrame()

    if "event_key" in evidence.columns and evidence["event_key"].notna().any():
        group_columns = ["event_key"]
    elif "event_round" in evidence.columns and evidence["event_round"].notna().any():
        group_columns = ["event_year", "event_round"]
    else:
        group_columns = []
    groups = (
        [group for _, group in evidence.groupby(group_columns, sort=False, dropna=False)]
        if group_columns
        else [evidence]
    )
    resolved_groups = [
        resolved
        for group in groups
        if not (resolved := _resolved_final_grid_event(group, as_of=as_of)).empty
    ]
    if not resolved_groups:
        return pd.DataFrame()
    return pd.concat(resolved_groups, ignore_index=True)


def _resolved_final_grid_event(
    evidence: pd.DataFrame,
    *,
    as_of: Optional[str],
) -> pd.DataFrame:
    """Resolve exactly one event/revision at the requested prediction cutoff."""

    provenance_columns = (
        "event_year",
        "grid_evidence_as_of",
        "grid_evidence_id",
        "grid_revision_phase",
    )
    if any(
        evidence[column].isna().any()
        or evidence[column]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"", "nan", "none", "<na>", "null"})
        .any()
        for column in provenance_columns
    ):
        return pd.DataFrame()
    if not _strict_true_mask(evidence["grid_evidence_complete"]).all():
        return pd.DataFrame()

    numeric_years = pd.to_numeric(evidence["event_year"], errors="coerce")
    if numeric_years.isna().any() or not np.isfinite(numeric_years).all():
        return pd.DataFrame()
    if not np.equal(numeric_years, np.floor(numeric_years)).all():
        return pd.DataFrame()
    years = numeric_years.astype(int).unique()
    evidence_times = evidence["grid_evidence_as_of"].astype(str).str.strip().unique()
    evidence_ids = evidence["grid_evidence_id"].astype(str).str.strip().unique()
    phases = evidence["grid_revision_phase"].astype(str).str.strip().str.lower().unique()
    if len(years) != 1 or len(evidence_times) != 1 or len(evidence_ids) != 1 or len(phases) != 1:
        return pd.DataFrame()
    phase_aliases = {
        "final": GridRevisionPhase.FINAL_PRE_RACE,
        "final_pre_race": GridRevisionPhase.FINAL_PRE_RACE,
        "provisional": GridRevisionPhase.PROVISIONAL_PRE_RACE,
        "provisional_pre_race": GridRevisionPhase.PROVISIONAL_PRE_RACE,
    }
    phase = phase_aliases.get(str(phases[0]))
    if phase is None:
        return pd.DataFrame()

    raw_formats = evidence.get(
        "weekend_format_version",
        pd.Series("standard", index=evidence.index),
    )
    format_values = raw_formats.dropna().astype(str).str.strip().str.lower().unique()
    try:
        weekend_format = (
            WeekendFormat(str(format_values[0]))
            if len(format_values) == 1
            else WeekendFormat.STANDARD
        )
        contract = build_weekend_contract(int(years[0]), weekend_format)
    except (TypeError, ValueError):
        return pd.DataFrame()

    status_map = {
        "grid": GridEntryStatus.GRID,
        "pit_lane": GridEntryStatus.PIT_LANE,
        "dns": GridEntryStatus.DID_NOT_START,
        "did_not_start": GridEntryStatus.DID_NOT_START,
        "withdrawn": GridEntryStatus.WITHDRAWN,
        "disqualified": GridEntryStatus.DISQUALIFIED,
    }
    entries: list[OfficialGridEntry] = []
    for _, row in evidence.iterrows():
        status = status_map.get(str(row.get("grid_status") or "").strip().lower())
        if status is None:
            return pd.DataFrame()
        numeric_position = pd.to_numeric(pd.Series([row.get("grid_position")]), errors="coerce").iloc[0]
        if status is GridEntryStatus.GRID:
            if (
                pd.isna(numeric_position)
                or not np.isfinite(float(numeric_position))
                or float(numeric_position) <= 0.0
                or not float(numeric_position).is_integer()
            ):
                return pd.DataFrame()
            position = int(numeric_position)
        else:
            position = None
        try:
            entries.append(
                OfficialGridEntry(
                    driver_id=str(row["driver_id"]),
                    position=position,
                    status=status,
                    evidence_complete=True,
                )
            )
        except (TypeError, ValueError):
            return pd.DataFrame()

    cutoff = as_of
    if cutoff is None and "prediction_as_of" in evidence.columns:
        cutoffs = evidence["prediction_as_of"].dropna().astype(str).str.strip()
        cutoffs = cutoffs[~cutoffs.isin({"", "nan", "None", "<NA>"})].unique()
        if len(cutoffs) > 1:
            return pd.DataFrame()
        if len(cutoffs) == 1:
            cutoff = str(cutoffs[0])
    if cutoff is None:
        cutoff = str(evidence_times[0])
    try:
        resolution = resolve_grand_prix_start(
            contract,
            as_of=str(cutoff),
            official_grid_revisions=(
                OfficialGridRevision(
                    revision_id=str(evidence_ids[0]),
                    phase=phase,
                    entries=tuple(entries),
                    as_of=str(evidence_times[0]),
                    evidence_complete=True,
                ),
            ),
        )
    except (OverflowError, TypeError, ValueError):
        return pd.DataFrame()
    scheduled = resolution.scheduled_grid
    if (
        scheduled.status is not ResolutionStatus.RESOLVED
        or scheduled.source is not EvidenceSource.OFFICIAL_GRID_REVISION
        or not scheduled.evidence_complete
        or scheduled.phase != GridRevisionPhase.FINAL_PRE_RACE.value
    ):
        return pd.DataFrame()

    identity_columns = [
        column
        for column in ("event_key", "event_year", "event_round")
        if column in evidence.columns and evidence[column].notna().all()
    ]
    identity_values = {column: evidence[column].iloc[0] for column in identity_columns}
    resolved_rows: list[dict[str, object]] = []
    for entry in scheduled.entries:
        resolved_rows.append(
            {
                "driver_id": entry.driver_id,
                "grid_position": (
                    float(entry.position) if entry.position is not None else float("nan")
                ),
                "grid_status": entry.status.value,
                "grid_source": "pre_race_official_grid",
                "grid_evidence_complete": True,
                "grid_resolution_status": scheduled.status.value,
                "grid_revision_phase": scheduled.phase,
                "grid_evidence_as_of": scheduled.evidence_as_of,
                "grid_evidence_id": str(evidence_ids[0]),
                **identity_values,
            }
        )
    return pd.DataFrame(resolved_rows)


def _apply_race_information_horizon(
    frame: pd.DataFrame,
    *,
    horizon: str,
    training: bool,
    as_of: Optional[str] = None,
) -> pd.DataFrame:
    """Enforce the feature/target contract available at one race horizon."""

    if frame.empty:
        return frame
    if horizon not in RACE_INFORMATION_HORIZONS:
        raise ValueError(f"Unsupported race information horizon: {horizon!r}.")
    out = frame.copy()
    out["race_information_horizon"] = horizon

    predicted_grid = pd.to_numeric(
        out.get("qualy_pred_rank", pd.Series(float("nan"), index=out.index)),
        errors="coerce",
    )
    predicted_position = pd.to_numeric(
        out.get("qualy_pred_position", pd.Series(float("nan"), index=out.index)),
        errors="coerce",
    )
    predicted_grid = predicted_grid.where(predicted_grid.notna(), predicted_position)
    actual_qualy = pd.to_numeric(
        out.get("qualy_position", pd.Series(float("nan"), index=out.index)),
        errors="coerce",
    )

    if horizon in PRE_QUALIFYING_RACE_HORIZONS:
        out["grid_position"] = predicted_grid
        out["grid_source"] = "oof_predicted_qualifying_grid" if training else "predicted_qualifying_grid"
        out["grid_status"] = "predicted"
        # These columns are only observed after qualifying and must not be
        # imputed from retrospective results in a pre-qualifying model.
        for column in (
            "qualy_position",
            "qualy_gap_to_best",
            "qualy_position_track_adj",
            "qualy_gap_track_adj",
            "qualy_pred_vs_actual_gap",
            "qualy_position_circuit_importance_adj",
        ):
            out[column] = float("nan")
        if horizon == "pre_fp_provisional":
            # Historical training rows contain completed-weekend FP data. A
            # pre-FP current forecast cannot, so scrub the same direct
            # weekend-practice surface on both sides while retaining causal
            # rolling driver/team history columns.
            out = _scrub_current_weekend_practice(out)
    elif horizon == "post_qualifying_pre_grid":
        out["grid_position"] = actual_qualy
        out["grid_source"] = "historical_qualifying_proxy" if training else "qualifying_proxy"
        out["grid_status"] = "gp_grid_unresolved_proxy"
        weekend_format = out.get(
            "weekend_format_version",
            pd.Series("unknown", index=out.index, dtype=object),
        ).astype(str)
        original_sprint_format = weekend_format.eq("sprint_2021_2022")
        if original_sprint_format.any():
            # In 2021-22, Friday Qualifying set the Sprint grid; the Sprint
            # classification set the GP grid.  Q remains a useful causal proxy
            # at this horizon but must never be labelled as the GP grid.
            out.loc[original_sprint_format, "grid_source"] = (
                "historical_qualifying_proxy_pre_sprint"
                if training
                else "qualifying_proxy_pre_sprint"
            )
            out.loc[original_sprint_format, "grid_status"] = "gp_grid_unresolved_proxy"
        out["grid_evidence_complete"] = False
    elif horizon == "post_grid_pre_race":
        resolved_grid = _resolved_final_grid_frame(out, as_of=as_of)
        if resolved_grid.empty:
            out["grid_position"] = float("nan")
            out["grid_source"] = "unresolved_final_grid"
            out["grid_status"] = "unresolved"
            out["grid_evidence_complete"] = False
            out["grid_resolution_status"] = "unresolved"
        else:
            join_keys = ["driver_id"]
            join_keys.extend(
                column
                for column in ("event_key", "event_year", "event_round")
                if column in out.columns and column in resolved_grid.columns
            )
            replacement_columns = [
                column for column in resolved_grid.columns if column not in join_keys
            ]
            out = out.drop(columns=replacement_columns, errors="ignore").merge(
                resolved_grid,
                on=join_keys,
                how="left",
            )
            unresolved = ~out.get(
                "grid_resolution_status",
                pd.Series("unresolved", index=out.index),
            ).fillna("unresolved").astype(str).eq("resolved")
            out.loc[unresolved, "grid_position"] = float("nan")
            out.loc[unresolved, "grid_source"] = "unresolved_final_grid"
            out.loc[unresolved, "grid_status"] = "unresolved"
            out.loc[unresolved, "grid_evidence_complete"] = False
            out.loc[unresolved, "grid_resolution_status"] = "unresolved"
            nonstarters = out["grid_status"].astype(str).str.lower().isin(
                {"did_not_start", "withdrawn", "disqualified"}
            )
            out["race_prediction_available"] = ~nonstarters
            out["race_prediction_unavailable_reason"] = np.where(
                nonstarters,
                out["grid_status"].astype(str),
                None,
            )

    if training and "target" in out.columns:
        target = pd.to_numeric(out["target"], errors="coerce")
        grid = pd.to_numeric(out["grid_position"], errors="coerce")
        out["race_delta_target"] = target - grid
    return out


def _add_race_context_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    numeric_cols = [
        "qualy_position",
        "grid_position",
        "qualy_pred_position",
        "qualy_pred_rank",
        "track_qualy_importance",
        "track_finish_order_mobility",
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
        if "track_finish_order_mobility" in out.columns:
            mobility_raw = out["track_finish_order_mobility"]
        else:
            mobility_raw = pd.Series(0.5, index=out.index, dtype=float)
        safety_raw = out["track_safety_car_propensity"] if "track_safety_car_propensity" in out.columns else pd.Series(
            0.2,
            index=out.index,
            dtype=float,
        )
        mobility = pd.to_numeric(mobility_raw, errors="coerce").reindex(out.index).fillna(0.5)
        safety = pd.to_numeric(safety_raw, errors="coerce").reindex(out.index).fillna(0.2)
        qualy_importance = (1.0 - (0.65 * mobility) - (0.35 * safety)).clip(lower=0.0, upper=1.0)
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

    qualy_actual = (
        pd.to_numeric(out["qualy_position"], errors="coerce")
        if "qualy_position" in out.columns
        else pd.Series(float("nan"), index=out.index, dtype=float)
    )
    grid_position = (
        pd.to_numeric(out["grid_position"], errors="coerce")
        if "grid_position" in out.columns
        else pd.Series(float("nan"), index=out.index, dtype=float)
    )
    qualy_pred_rank = (
        pd.to_numeric(out["qualy_pred_rank"], errors="coerce")
        if "qualy_pred_rank" in out.columns
        else pd.Series(float("nan"), index=out.index, dtype=float)
    )
    qualy_pred_position = (
        pd.to_numeric(out["qualy_pred_position"], errors="coerce")
        if "qualy_pred_position" in out.columns
        else pd.Series(float("nan"), index=out.index, dtype=float)
    )
    predicted_grid = qualy_pred_rank.where(qualy_pred_rank.notna(), qualy_pred_position)
    filled_grid = grid_position.where(grid_position.notna(), qualy_actual)
    predicted_grid_mask = filled_grid.isna() & predicted_grid.notna()
    out["grid_position"] = filled_grid.where(filled_grid.notna(), predicted_grid)
    if predicted_grid_mask.any():
        if "grid_source" not in out.columns:
            out["grid_source"] = "unknown"
        grid_source = out["grid_source"].astype(object).where(out["grid_source"].notna(), "unknown")
        grid_source.loc[predicted_grid_mask] = "predicted_qualifying_grid"
        out["grid_source"] = grid_source
        if "grid_status" not in out.columns:
            out["grid_status"] = "unknown"
        grid_status = out["grid_status"].astype(object).where(out["grid_status"].notna(), "unknown")
        grid_status.loc[predicted_grid_mask] = "predicted"
        out["grid_status"] = grid_status
    out["grid_position_track_adj"] = out["grid_position"] * (0.35 + qualy_importance)
    race_start = pd.to_numeric(out.get("grid_position"), errors="coerce")
    qualy_pred = qualy_pred_position.where(qualy_pred_position.notna(), qualy_pred_rank)
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

    horizon = _resolve_race_information_horizon(
        race_features,
        getattr(config, "race_information_horizon", "auto"),
        prediction_as_of=getattr(config, "prediction_as_of", None),
    )

    qual_train, qual_notes = build_training_data(
        provider=provider,
        mode="qualifying",
        train_seasons=config.train_seasons,
        target_year=config.year,
        target_round=config.round_number,
        include_standings=False,
        session_cutoff=getattr(config, "qualifying_information_horizon", "auto"),
        prediction_as_of=getattr(config, "prediction_as_of", None),
    )
    qual_train = _apply_season_sample_weighting(qual_train, config, notes)
    qual_features, qual_feature_notes = build_current_features(
        provider=provider,
        mode="qualifying",
        year=config.year,
        round_number=config.round_number,
        include_standings=False,
        history=qual_train,
        session_cutoff=getattr(config, "qualifying_information_horizon", "auto"),
        prediction_as_of=getattr(config, "prediction_as_of", None),
    )
    for note in qual_notes + qual_feature_notes:
        note_lower = note.lower()
        if note_lower.startswith("echec") or "indisponible" in note_lower or "aucune" in note_lower or "pas assez" in note_lower:
            notes.append(f"[Race<-Quali] {note}")

    if qual_train.empty and qual_features.empty:
        notes.append("[Race<-Quali] Impossible de construire un signal de qualif predit.")
        # Fail closed: the requested race horizon still applies even when the
        # nested qualifying path cannot build a field.  Returning here used to
        # leave retrospective qualifying/grid columns untouched in pre-Q runs.
        race_train = _apply_race_information_horizon(
            race_train,
            horizon=horizon,
            training=True,
        )
        race_features = _apply_race_information_horizon(
            race_features,
            horizon=horizon,
            training=False,
            as_of=getattr(config, "prediction_as_of", None),
        )
        return _add_race_context_interactions(race_train), _add_race_context_interactions(race_features)

    if config.disable_runsim_features:
        qual_train = _drop_runsim_columns(qual_train)
        qual_features = _drop_runsim_columns(qual_features)
    if config.disable_circuit_features:
        qual_train = _drop_circuit_columns(qual_train)
        qual_features = _drop_circuit_columns(qual_features)
    if horizon == "pre_fp_provisional":
        qual_train = _scrub_current_weekend_practice(qual_train)
        qual_features = _scrub_current_weekend_practice(qual_features)
        notes.append(
            "[Race<-Quali] Pre-FP qualifying-grid model uses rolling/history context only; "
            "completed-weekend FP features are scrubbed in OOF training and current scoring."
        )

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
        f1_pl_samples=getattr(config, "f1_pl_samples", 2000),
        f1_listwise_seed=getattr(config, "f1_listwise_seed", 42),
        race_delta_constraint_mode=getattr(config, "race_delta_constraint_mode", "constrained"),
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

    race_train = _apply_race_information_horizon(race_train, horizon=horizon, training=True)
    race_features = _apply_race_information_horizon(
        race_features,
        horizon=horizon,
        training=False,
        as_of=getattr(config, "prediction_as_of", None),
    )
    if horizon == "post_grid_pre_race" and not race_train.empty:
        resolved_training_grid = race_train.get(
            "grid_resolution_status",
            pd.Series("unresolved", index=race_train.index),
        ).fillna("unresolved").astype(str).eq("resolved")
        numeric_training_grid = pd.to_numeric(
            race_train.get("grid_position"),
            errors="coerce",
        ).notna()
        valid_final_grid = resolved_training_grid & numeric_training_grid
        dropped = int((~valid_final_grid).sum())
        if dropped:
            notes.append(
                "[Race grid] Historical rows without a resolver-approved numeric final-grid slot "
                f"were excluded from training: rows={dropped}."
            )
            race_train = race_train.loc[valid_final_grid].copy()
    if horizon in PRE_QUALIFYING_RACE_HORIZONS and not race_train.empty:
        valid_predicted_grid = pd.to_numeric(race_train.get("grid_position"), errors="coerce").notna()
        dropped = int((~valid_predicted_grid).sum())
        if dropped:
            notes.append(
                "[Race<-Quali] Historical rows without a causal OOF qualifying-grid prediction "
                f"were excluded from the {horizon} race model: rows={dropped}."
            )
            race_train = race_train.loc[valid_predicted_grid].copy()
    race_train = _add_race_context_interactions(race_train)
    race_features = _add_race_context_interactions(race_features)
    notes.append(
        "[Race horizon] Explicit information contract active: "
        f"{horizon}; training and current rows use the same grid/qualifying availability."
    )
    if not race_features.empty and "grid_source" in race_features.columns:
        predicted_grid_rows = race_features["grid_source"].astype(str).eq("predicted_qualifying_grid")
        if predicted_grid_rows.any():
            notes.append(
                "[Race<-Quali] Pre-qualifying race path active: "
                f"qualy_pred_rank used as provisional grid for {int(predicted_grid_rows.sum())} current rows.",
            )
    return race_train, race_features


def _weather_uncertainty_series(frame: pd.DataFrame) -> pd.Series:
    for column in ("track_weather_uncertainty_prior", "track_weather_uncertainty"):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            return values.fillna(0.0).clip(lower=0.0, upper=1.0)
    return pd.Series(0.0, index=frame.index, dtype=float)


def _weather_neutral_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if frame.empty:
        return frame.copy(), []
    out = frame.copy()
    neutralized: list[str] = []
    for column in ("track_weather_uncertainty", "track_weather_uncertainty_prior"):
        if column in out.columns:
            out[column] = 0.0
            neutralized.append(column)
    if "race_generation_variance_prior" in out.columns:
        safety = pd.to_numeric(
            out.get("track_safety_car_prior", pd.Series(0.25, index=out.index)),
            errors="coerce",
        ).fillna(0.25).clip(lower=0.0, upper=1.0)
        dnf = pd.to_numeric(
            out.get("track_dnf_prior", pd.Series(0.08, index=out.index)),
            errors="coerce",
        ).fillna(0.08).clip(lower=0.0, upper=1.0)
        strategy = pd.to_numeric(
            out.get("track_strategy_variance_prior", pd.Series(0.35, index=out.index)),
            errors="coerce",
        ).fillna(0.35).clip(lower=0.0, upper=1.0)
        out["race_generation_variance_prior"] = (
            (0.34 * safety) + (0.26 * dnf) + (0.25 * strategy)
        ).clip(lower=0.0, upper=1.0)
        neutralized.append("race_generation_variance_prior")
    if "weather_mean_ranking_effect" in out.columns:
        out["weather_mean_ranking_effect"] = "weather_neutralized"
    return add_f1_wet_pace_interactions(out), sorted(set(neutralized))


def _weather_scenario_summary(frame: pd.DataFrame) -> dict[str, object]:
    weather = _weather_uncertainty_series(frame)
    effect_values = (
        frame["weather_mean_ranking_effect"].dropna().astype(str).unique().tolist()
        if "weather_mean_ranking_effect" in frame.columns
        else []
    )
    return {
        "rows": int(len(frame)),
        "weather_uncertainty_mean": float(weather.mean(skipna=True)) if not weather.empty else 0.0,
        "weather_uncertainty_max": float(weather.max(skipna=True)) if not weather.empty else 0.0,
        "weather_columns": [column for column in WEATHER_SCENARIO_COLUMNS if column in frame.columns],
        "source": "track_weather_uncertainty_prior",
        "wet_pace_evidence_rows": wet_pace_evidence_rows(frame),
        "mean_ranking_effect": effect_values[0] if len(effect_values) == 1 else "uncertainty_only",
    }


def _model_feature_columns(model: Optional[object]) -> set[str]:
    if model is None:
        return set()
    pending = [model]
    visited: set[int] = set()
    columns: set[str] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        preprocessor = getattr(current, "preprocessor", None)
        feature_cols = getattr(preprocessor, "feature_cols", None)
        if isinstance(feature_cols, list):
            columns.update(str(column) for column in feature_cols)
        for attribute in ("base_model", "primary_model"):
            nested = getattr(current, attribute, None)
            if nested is not None:
                pending.append(nested)
    return columns


def _score_prediction_output(
    *,
    config: PredictionConfig,
    features: pd.DataFrame,
    model: Optional[object],
    feature_cols: List[str],
    fallback_cols: List[str],
    listwise_temperature: Optional[float],
    scenario_name: str,
    emit_notes: bool,
) -> tuple[pd.DataFrame, list[str]]:
    scoring_notes: list[str] = []
    preds = predict_with_model(model, features, feature_cols, fallback_cols)
    output = features.copy()
    output["pred"] = preds
    output["prediction_scenario"] = scenario_name
    output["weather_scenario"] = scenario_name
    output["weather_uncertainty_level"] = _weather_uncertainty_series(output)
    listwise_preds = preds
    if config.mode == "race":
        race_stochastic = _race_stochastic_score_layer(output, preds)
        output = output.join(race_stochastic)
        if "race_stochastic_pl_score" in output.columns:
            listwise_preds = pd.to_numeric(output["race_stochastic_pl_score"], errors="coerce").fillna(preds)
    proba_win = _predict_probability(model, preds, label="win", k=1)
    proba_top10, proba_top3 = _predict_probabilities(model, preds)
    proba_win = pd.Series(np.minimum(proba_win.clip(0.0, 1.0), proba_top3), index=preds.index, dtype=float)
    proba_win = _event_total_probability_with_caps(preds, proba_win, k=1, caps=proba_top3)
    output["proba_win"] = proba_win
    output["proba_top10"] = proba_top10
    output["proba_top3"] = proba_top3

    if str(config.f1_listwise).strip().lower() == "pl_gumbel":
        effective_temperature = (
            float(listwise_temperature)
            if listwise_temperature is not None
            else float(config.f1_pl_temperature)
        )
        if config.mode == "qualifying":
            weather_level = float(output["weather_uncertainty_level"].mean(skipna=True))
            if np.isfinite(weather_level) and weather_level > 0.0:
                effective_temperature *= 1.0 + (0.70 * min(1.0, max(0.0, weather_level)))
        listwise = _pl_gumbel_listwise(
            frame=output,
            preds=listwise_preds,
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
        if emit_notes:
            scoring_notes.append(
                "Listwise PL active: proba_win/proba_top10/proba_top3 remplaces par p_win/p_top10/p_top3 "
                f"(seed stable par event, temperature={effective_temperature:.3f}, scenario={scenario_name}).",
            )
            if config.mode == "race":
                scoring_notes.append(
                    "Race stochastic layer active: model mean scores are adjusted by expected DNF/reliability "
                    "and rescaled by safety-car, strategy, and weather uncertainty before PL/Gumbel sampling; "
                    "circuit mobility is not applied a second time.",
                )
            elif scenario_name == "weather_integrated":
                scoring_notes.append(
                    "Qualifying weather scenario active: weather uncertainty priors widen PL/Gumbel temperature "
                    "for the integrated scenario.",
                )
    else:
        output["listwise_enabled"] = False

    output["probability_output_status"] = "diagnostic_not_promoted_uncalibrated"
    output["position_interval_output_status"] = "diagnostic_not_promoted_uncalibrated"
    if config.mode == "race":
        available = _strict_true_mask(
            output.get(
                "race_prediction_available",
                pd.Series(True, index=output.index),
            )
        )
        output["forecast_available"] = available
        output["forecast_unavailable_reason"] = output.get(
            "race_prediction_unavailable_reason",
            pd.Series(None, index=output.index, dtype=object),
        )
        unavailable = ~available
        if unavailable.any():
            for column in (
                "pred",
                "utility",
                "exp_pos",
                "pos_p10",
                "pos_p50",
                "pos_p90",
            ):
                if column in output.columns:
                    output.loc[unavailable, column] = float("nan")
            for column in (
                "proba_win",
                "proba_top3",
                "proba_top10",
                "p_win",
                "p_top3",
                "p_top10",
            ):
                if column in output.columns:
                    output.loc[unavailable, column] = 0.0
    else:
        output["forecast_available"] = True
        output["forecast_unavailable_reason"] = None

    if "driver_name" not in output.columns:
        if "driver_id" in output.columns:
            output["driver_name"] = output["driver_id"]
        else:
            output["driver_name"] = pd.Series(dtype=str)
    elif "driver_id" in output.columns:
        output["driver_name"] = output["driver_name"].fillna(output["driver_id"])
    return output, scoring_notes


def _scenario_records(output: pd.DataFrame) -> dict[str, object]:
    all_rows = format_prediction_table(output, top_n=None)
    rows = format_prediction_table(output, top_n=10)
    return {
        "rows": _prediction_records(rows),
        "all_prediction_rows": _prediction_records(all_rows),
        "summary": _weather_scenario_summary(output),
    }


def _weather_event_name(features: pd.DataFrame, config: PredictionConfig) -> str:
    if not features.empty and "event_name" in features.columns:
        values = features["event_name"].dropna().astype(str).str.strip()
        values = values[values != ""]
        if not values.empty:
            return str(values.iloc[0])
    for value in (config.meeting_name, config.country_name):
        if value is not None and str(value).strip():
            return str(value)
    return f"Round {config.round_number}"


def _non_null_numeric_count(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(pd.to_numeric(frame[column], errors="coerce").notna().sum())


def _positive_numeric_count(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    values = pd.to_numeric(frame[column], errors="coerce")
    return int((values.fillna(0.0) > 0.0).sum())


def _true_count(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(_strict_true_mask(frame[column]).sum())


def _strict_true_mask(values: pd.Series) -> pd.Series:
    """Parse only explicit true values; notably, the string ``false`` is false."""

    clean = values.astype("string").str.strip().str.lower()
    return clean.isin({"1", "1.0", "true", "yes"}).fillna(False)


def _numeric_feature_snapshot(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, float | None]:
    snapshot: dict[str, float | None] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.notna().any():
            value = float(values.mean(skipna=True))
            if np.isfinite(value):
                snapshot[column] = value
            else:
                snapshot[column] = None
    return snapshot


def _column_has_observed_value(frame: pd.DataFrame, column: str) -> bool:
    if frame.empty or column not in frame.columns:
        return False
    raw = frame[column]
    if pd.to_numeric(raw, errors="coerce").notna().any():
        return True
    clean = raw.dropna().astype(str).str.strip().str.lower()
    return bool((~clean.isin({"", "nan", "none", "<na>"})).any())


def _prediction_input_phase(config: PredictionConfig, features: pd.DataFrame) -> dict[str, object]:
    row_count = int(len(features))
    fp_session_rows = _positive_numeric_count(features, "pace_sessions_available")
    fp_lap_rows = _positive_numeric_count(features, "fp_total_laps")
    actual_fp_rows = max(fp_session_rows, fp_lap_rows)
    actual_qualifying_rows = _non_null_numeric_count(features, "qualy_position")
    provisional_rows = _true_count(features, "provisional_current_field")

    grid_source_counts: dict[str, int] = {}
    official_grid_rows = 0
    qualifying_grid_rows = 0
    predicted_grid_rows = 0
    if "grid_source" in features.columns:
        grid_sources = features["grid_source"].fillna("unknown").astype(str)
        grid_source_counts = {
            str(key): int(value)
            for key, value in grid_sources.value_counts(dropna=False).to_dict().items()
        }
        grid_position = (
            pd.to_numeric(features["grid_position"], errors="coerce")
            if "grid_position" in features.columns
            else pd.Series(float("nan"), index=features.index, dtype=float)
        )
        grid_evidence_complete = _strict_true_mask(features.get(
            "grid_evidence_complete",
            pd.Series(False, index=features.index),
        ))
        grid_resolution_status = features.get(
            "grid_resolution_status",
            pd.Series("unresolved", index=features.index),
        ).fillna("unresolved").astype(str).str.strip().str.lower()
        resolved_grid = _resolved_final_grid_frame(
            features,
            as_of=getattr(config, "prediction_as_of", None),
        )
        resolved_driver_ids = (
            set(resolved_grid["driver_id"].astype(str)) if not resolved_grid.empty else set()
        )
        driver_ids = features.get(
            "driver_id",
            pd.Series("", index=features.index),
        ).fillna("").astype(str)
        official_grid_mask = (
            grid_sources.eq("pre_race_official_grid")
            & grid_position.notna()
            & grid_evidence_complete
            & grid_resolution_status.eq("resolved")
            & driver_ids.isin(resolved_driver_ids)
        )
        qualifying_grid_mask = (
            grid_sources.isin({"qualifying_fallback", "qualifying_proxy"})
            & grid_position.notna()
        )
        predicted_grid_mask = grid_sources.eq("predicted_qualifying_grid") & grid_position.notna()
        official_grid_rows = int(official_grid_mask.sum())
        qualifying_grid_rows = int(qualifying_grid_mask.sum())
        predicted_grid_rows = int(predicted_grid_mask.sum())

    if row_count == 0:
        phase = "no_current_field"
    elif str(config.mode).strip().lower() == "qualifying":
        phase = "post_fp_pre_qualifying" if actual_fp_rows > 0 else "pre_fp_provisional"
    elif official_grid_rows > 0:
        phase = "post_grid_pre_race"
    elif qualifying_grid_rows > 0 or actual_qualifying_rows > 0:
        phase = "post_qualifying_pre_grid"
    elif actual_fp_rows > 0:
        phase = "post_fp_pre_qualifying"
    else:
        phase = "pre_fp_provisional"

    return {
        "phase": phase,
        "mode": str(config.mode),
        "source": str(config.source),
        "year": int(config.year),
        "round_number": int(config.round_number),
        "train_seasons": [int(year) for year in config.train_seasons],
        "row_count": row_count,
        "actual_current_fp_available": actual_fp_rows > 0,
        "actual_current_fp_rows": int(actual_fp_rows),
        "actual_current_fp_session_rows": int(fp_session_rows),
        "actual_current_fp_lap_rows": int(fp_lap_rows),
        "actual_current_qualifying_available": actual_qualifying_rows > 0,
        "actual_current_qualifying_rows": int(actual_qualifying_rows),
        "official_grid_available": official_grid_rows > 0,
        "official_grid_rows": int(official_grid_rows),
        "qualifying_grid_fallback_rows": int(qualifying_grid_rows),
        "predicted_grid_rows": int(predicted_grid_rows),
        "provisional_current_rows": int(provisional_rows),
        "provisional_current_field": provisional_rows > 0,
        "grid_source_counts": grid_source_counts,
    }


def _race_input_evidence(
    *,
    config: PredictionConfig,
    features: pd.DataFrame,
    feature_cols: List[str],
    fallback_cols: List[str],
    weather_summary: dict[str, object],
) -> dict[str, object]:
    model_features_present = [column for column in feature_cols if column in features.columns]
    fallback_features_present = [column for column in fallback_cols if column in features.columns]
    model_features_available = [
        column for column in model_features_present if _column_has_observed_value(features, column)
    ]
    fallback_features_available = [
        column for column in fallback_features_present if _column_has_observed_value(features, column)
    ]
    context_columns = [column for column in RACE_CONTEXT_EVIDENCE_COLUMNS if column in features.columns]
    strength_columns = [column for column in RACE_STRENGTH_EVIDENCE_COLUMNS if column in features.columns]
    current_weekend_columns = [column for column in CURRENT_WEEKEND_EVIDENCE_COLUMNS if column in features.columns]
    return {
        "race_context_columns_present": context_columns,
        "race_strength_columns_present": strength_columns,
        "current_weekend_columns_present": current_weekend_columns,
        "model_features_present": model_features_present,
        "fallback_features_present": fallback_features_present,
        "model_features_available": model_features_available,
        "fallback_features_available": fallback_features_available,
        "track_context_snapshot_mean": _numeric_feature_snapshot(features, RACE_CONTEXT_EVIDENCE_COLUMNS),
        "race_strength_snapshot_mean": _numeric_feature_snapshot(features, RACE_STRENGTH_EVIDENCE_COLUMNS),
        "current_weekend_snapshot_mean": _numeric_feature_snapshot(features, CURRENT_WEEKEND_EVIDENCE_COLUMNS),
        "circuit_feature_state": "quarantined" if config.disable_circuit_features else "enabled_research_only",
        "weather_enabled": bool(getattr(config, "weather_enabled", False)),
        "weather_available": bool(weather_summary.get("weather_available", False)),
        "weather_provider": weather_summary.get("provider") or getattr(config, "weather_provider", "open_meteo"),
        "finish_order_mobility_semantics": "observational_historical_grid_to_finish_mobility_not_causal_overtake_probability",
    }


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
                    "model_architecture": architecture_payload(),
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
                "model_architecture": architecture_payload(),
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
        session_cutoff=getattr(config, "qualifying_information_horizon", "auto"),
        prediction_as_of=getattr(config, "prediction_as_of", None),
    )
    train = _apply_season_sample_weighting(train, config, notes)

    features, feature_notes = build_current_features(
        provider=provider,
        mode=config.mode,
        year=config.year,
        round_number=config.round_number,
        include_standings=config.include_standings,
        history=train,
        session_cutoff=getattr(config, "qualifying_information_horizon", "auto"),
        prediction_as_of=getattr(config, "prediction_as_of", None),
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

    weather_summary: dict[str, object] = {
        "weather_enabled": bool(getattr(config, "weather_enabled", False)),
        "weather_available": False,
    }
    if getattr(config, "weather_enabled", False):
        weather_summary, weather_notes = fetch_f1_weather_summary(
            event_name=_weather_event_name(features, config),
            start=getattr(config, "weather_start", None),
            end=getattr(config, "weather_end", None),
            cache_root=getattr(config, "weather_cache_dir", None) or config.cache_dir,
            latitude=getattr(config, "weather_latitude", None),
            longitude=getattr(config, "weather_longitude", None),
            timezone=getattr(config, "weather_timezone", None),
            provider_name=getattr(config, "weather_provider", "open_meteo"),
        )
        weather_summary["weather_enabled"] = True
        notes.extend(weather_notes)
        features = apply_f1_weather_to_features(features, weather_summary)

    if config.disable_runsim_features:
        train = _drop_runsim_columns(train)
        features = _drop_runsim_columns(features)
        notes.append("Ablation active: run-sim features supprimees (fp_quali_sim_*, fp_race_sim_*, fp_slow_lap_ratio, fp_quali_vs_race_gap).")
    if config.disable_circuit_features:
        train = _drop_circuit_columns(train)
        features = _drop_circuit_columns(features)
        notes.append(
            "Circuit-card quarantine active: predictive circuit-card features/interactions are disabled "
            "unless an explicit ablation or research run enables them.",
        )

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
        f1_pl_samples=getattr(config, "f1_pl_samples", 2000),
        f1_listwise_seed=getattr(config, "f1_listwise_seed", 42),
        race_delta_constraint_mode=getattr(config, "race_delta_constraint_mode", "constrained"),
    )
    notes.extend(training_result.notes)
    wet_training_rows = wet_pace_evidence_rows(train)
    wet_current_rows = wet_pace_evidence_rows(features)
    selected_feature_columns = _model_feature_columns(training_result.model)
    learned_wet_interactions = {
        "fp_wet_sim_delta_weather_adj",
        "fp_wet_sim_rank_weather_adj",
    }.intersection(selected_feature_columns)
    if bool(getattr(config, "weather_enabled", False)):
        if wet_training_rows > 0 and wet_current_rows > 0 and learned_wet_interactions:
            weather_rank_effect = "learned_causal_wet_pace_interaction"
            notes.append(
                "Weather mean-ranking effect enabled: causal wet-practice pace x pre-event wet-risk "
                f"features are present in training/current data (train_rows={wet_training_rows}, "
                f"current_rows={wet_current_rows}).",
            )
        else:
            weather_rank_effect = "uncertainty_only"
            notes.append(
                "Weather is uncertainty-only for this prediction: no selected model had both historical and "
                "current causal wet-practice evidence, so wet risk does not claim a relative mean-pace adjustment.",
            )
    else:
        weather_rank_effect = "weather_disabled"
    weather_summary["mean_ranking_effect"] = weather_rank_effect
    weather_summary["wet_pace_training_rows"] = int(wet_training_rows)
    weather_summary["wet_pace_current_rows"] = int(wet_current_rows)
    weather_summary["selected_wet_interaction_features"] = sorted(learned_wet_interactions)
    features = features.copy()
    features["weather_mean_ranking_effect"] = weather_rank_effect
    if training_result.model is None:
        notes.append(
            "Fallback heuristique actif: qualif=blend FP pace empirique; race=position qualif si disponible.",
        )
    output, scoring_notes = _score_prediction_output(
        config=config,
        features=features,
        model=training_result.model,
        feature_cols=feature_cols,
        fallback_cols=fallback_cols,
        listwise_temperature=training_result.listwise_temperature,
        scenario_name="weather_integrated",
        emit_notes=True,
    )
    notes.extend(scoring_notes)
    weather_neutral_features, neutralized_weather_columns = _weather_neutral_frame(features)
    base_output, _ = _score_prediction_output(
        config=config,
        features=weather_neutral_features,
        model=training_result.model,
        feature_cols=feature_cols,
        fallback_cols=fallback_cols,
        listwise_temperature=training_result.listwise_temperature,
        scenario_name="base_no_weather",
        emit_notes=False,
    )
    prediction_scenarios = {
        "base_no_weather": {
            **_scenario_records(base_output),
            "weather_feature_state": "neutralized",
            "neutralized_columns": neutralized_weather_columns,
        },
        "weather_integrated": {
            **_scenario_records(output),
            "weather_feature_state": "integrated",
            "neutralized_columns": [],
        },
    }
    notes.append(
        "Weather scenarios active: base_no_weather neutralizes weather uncertainty priors; "
        "weather_integrated keeps the available track/weather priors.",
    )

    version = compute_version(config.round_number, config.include_standings)
    all_rows = format_prediction_table(output, top_n=None)
    table = format_prediction_table(output, top_n=10)
    circuit_card = circuit_card_payload_from_frame(output)
    grid_source_counts = (
        output["grid_source"].fillna("unknown").astype(str).value_counts(dropna=False).to_dict()
        if "grid_source" in output.columns
        else {}
    )
    grid_status_counts = (
        output["grid_status"].fillna("unknown").astype(str).value_counts(dropna=False).to_dict()
        if "grid_status" in output.columns
        else {}
    )
    prediction_phase = _prediction_input_phase(config, output)
    resolved_cutoffs = _stable_column_values(features, "session_cutoff_resolved")
    qualifying_information_horizon = {
        "requested_cutoff": str(getattr(config, "qualifying_information_horizon", "auto")),
        "resolved_cutoffs": resolved_cutoffs,
        "prediction_as_of": getattr(config, "prediction_as_of", None),
        "weekend_format_versions": _stable_column_values(features, "weekend_format_version"),
        "weekend_snapshot_sha256": _stable_column_values(features, "weekend_snapshot_sha256"),
        "session_completion_contract": "completed_session_with_classification",
        "future_or_partial_sessions_eligible": False,
    }
    if not resolved_cutoffs:
        notes.append(
            "Information-horizon evidence is incomplete: the provider did not emit a resolved session cutoff; "
            "the prediction artifact records this explicitly instead of inferring freshness from file presence.",
        )
    else:
        notes.append(
            "Qualifying information horizon: requested="
            f"{qualifying_information_horizon['requested_cutoff']}, resolved={','.join(map(str, resolved_cutoffs))}.",
        )
    practice_quality = _practice_quality_evidence(features)
    race_input_evidence: dict[str, object] = {}
    if config.mode == "race":
        race_input_evidence = _race_input_evidence(
            config=config,
            features=features,
            feature_cols=feature_cols,
            fallback_cols=fallback_cols,
            weather_summary=weather_summary,
        )
        notes.append(
            "Track mobility semantics: track_finish_order_mobility is an observational historical grid-to-finish "
            "movement feature, not a causal overtake probability; track_overtake_propensity is not a model feature.",
        )
        phase_name = str(prediction_phase.get("phase", "unknown"))
        if phase_name == "pre_fp_provisional":
            notes.append(
                "Race prediction phase: pre_fp_provisional; no current-weekend FP, real qualifying, or official grid "
                "was available, so the race ranking is expected to stay grid/form anchored until fresh session data is downloaded.",
            )
        elif phase_name == "post_fp_pre_qualifying":
            notes.append(
                "Race prediction phase: post_fp_pre_qualifying; current-weekend FP pace is available, "
                "but race grid still comes from predicted qualifying context.",
            )
        elif phase_name == "post_qualifying_pre_grid":
            notes.append(
                "Race prediction phase: post_qualifying_pre_grid; real qualifying is available and used as the race-start anchor.",
            )
        elif phase_name == "post_grid_pre_race":
            notes.append(
                "Race prediction phase: post_grid_pre_race; official starting-grid context is available and used.",
            )
        if not config.disable_circuit_features:
            notes.append(
                "Race context enabled: historical track mobility, overtake, safety-car, DNF, pit/strategy, "
                "weather-prior, and circuit-card features are available to the race model.",
            )
        if bool(getattr(config, "weather_enabled", False)):
            notes.append(
                "Race weather context requested: "
                f"weather_available={bool(weather_summary.get('weather_available', False))}.",
            )
    training_event_keys = sorted(
        {
            int(value)
            for value in pd.to_numeric(
                train.get("event_key", pd.Series(dtype=float)),
                errors="coerce",
            ).dropna()
        }
    )
    target_event_key = (int(config.year) * 100) + int(config.round_number)
    if target_event_key in training_event_keys:
        raise RuntimeError(
            f"Point-in-time violation: target event {target_event_key} is present in training rows"
        )
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
            "model_architecture": architecture_payload(),
            "all_prediction_rows": _prediction_records(all_rows),
            "prediction_scenarios": prediction_scenarios,
            "circuit_card": circuit_card,
            "circuit_feature_state": "quarantined" if config.disable_circuit_features else "enabled_research_only",
            "circuit_feature_columns": list(CIRCUIT_NUMERIC_FEATURES + CIRCUIT_INTERACTION_FEATURES),
            "weather_feature_columns": [column for column in WEATHER_SCENARIO_COLUMNS if column in features.columns],
            "weather": weather_summary,
            "prediction_phase": prediction_phase,
            "qualifying_information_horizon": qualifying_information_horizon,
            "practice_quality_evidence": practice_quality,
            "race_information_horizon": (
                str(output["race_information_horizon"].iloc[0])
                if config.mode == "race" and "race_information_horizon" in output.columns and not output.empty
                else None
            ),
            "race_input_evidence": race_input_evidence,
            "listwise_temperature": training_result.listwise_temperature,
            "probability_audit": training_result.probability_audit,
            "selection_audit": training_result.selection_audit,
            "grid_source_counts": grid_source_counts,
            "grid_status_counts": grid_status_counts,
            "training_event_keys": training_event_keys,
            "training_row_count": int(len(train)),
            "target_event_key_excluded_from_training": True,
        },
    )
