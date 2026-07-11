"""Data preparation and feature assembly."""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from packages.f1.data.schemas.circuit import (
    CIRCUIT_INTERACTION_FEATURES,
    CIRCUIT_NUMERIC_FEATURES,
    attach_circuit_card,
)
from packages.f1.data.providers import BaseProvider
from packages.f1.data.utils import normalize_event_name, team_column
from packages.f1.features.wet import add_f1_wet_pace_interactions


PACE_DELTA_WEIGHTS = {
    "fp1_delta": 0.18,
    "fp2_delta": 0.24,
    "fp3_delta": 0.32,
    "sq_delta": 0.12,
    "sprint_delta": 0.14,
}


def _round_event_name(round_meta: dict[str, object], round_number: int) -> str:
    for key in ("event_name", "meeting_name", "country_name"):
        value = round_meta.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"Round {round_number}"


def _event_name_for_round(
    provider: BaseProvider,
    year: int,
    round_number: int,
    notes: List[str],
) -> str:
    try:
        rounds = provider.list_rounds(year)
    except (Exception, SystemExit) as exc:
        notes.append(f"Echec listing rounds {year}: {exc}")
        return f"Round {round_number}"
    for rnd in rounds:
        try:
            if int(rnd.get("round_number", 0)) == round_number:
                return _round_event_name(rnd, round_number)
        except Exception:
            continue
    return f"Round {round_number}"


def _attach_track_stats(
    frame: pd.DataFrame,
    provider: BaseProvider,
    year: int,
    round_number: int,
    notes: List[str],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    event_name = _event_name_from_frame(frame, default=f"Round {round_number}")
    stats: Optional[dict[str, object]] = None
    getter = getattr(provider, "get_track_stats", None)
    if getter is not None:
        try:
            candidate_stats = getter(year, round_number)
            if isinstance(candidate_stats, dict) and candidate_stats:
                stats = dict(candidate_stats)
        except (Exception, SystemExit) as exc:
            notes.append(f"Echec stats circuit {year} round {round_number}: {exc}")
    out = attach_circuit_card(frame, event_name=event_name, track_stats=stats)
    if isinstance(stats, dict):
        for key, value in stats.items():
            if value is None:
                continue
            numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.isna(numeric):
                continue
            out[key] = float(numeric)
    if "track_finish_order_mobility" not in out.columns and "track_overtake_propensity" in out.columns:
        out["track_finish_order_mobility"] = out["track_overtake_propensity"]
    out = _add_track_interactions(out)
    return out


def _event_name_from_frame(frame: pd.DataFrame, default: str) -> str:
    if "event_name" not in frame.columns:
        return default
    values = frame["event_name"].dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return default
    return str(values.iloc[0])


def _add_track_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    track_cols = [
        "track_finish_order_mobility",
        "track_overtake_propensity",
        "track_grid_stability",
        "track_safety_car_propensity",
        "track_sc_lap_ratio",
        "track_vsc_lap_ratio",
        "track_dnf_rate",
        "track_pit_stop_intensity",
        "track_weather_uncertainty",
        "track_stats_reliability",
        "track_chaos_index",
    ]
    for col in track_cols + CIRCUIT_NUMERIC_FEATURES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "track_finish_order_mobility" not in out.columns and "track_overtake_propensity" in out.columns:
        out["track_finish_order_mobility"] = out["track_overtake_propensity"]
    mobility = (
        out["track_finish_order_mobility"]
        if "track_finish_order_mobility" in out.columns
        else pd.Series(0.5, index=out.index, dtype=float)
    )
    safety = (
        out["track_safety_car_propensity"]
        if "track_safety_car_propensity" in out.columns
        else pd.Series(0.2, index=out.index, dtype=float)
    )
    dnf = (
        out["track_dnf_rate"]
        if "track_dnf_rate" in out.columns
        else pd.Series(0.1, index=out.index, dtype=float)
    )
    pit = (
        out["track_pit_stop_intensity"]
        if "track_pit_stop_intensity" in out.columns
        else pd.Series(1.0, index=out.index, dtype=float)
    )
    weather = (
        out["track_weather_uncertainty"]
        if "track_weather_uncertainty" in out.columns
        else pd.Series(float("nan"), index=out.index, dtype=float)
    )
    pit_variance = (pd.to_numeric(pit, errors="coerce").fillna(1.0) / 3.0).clip(lower=0.0, upper=1.0)
    reliability = (
        pd.to_numeric(out["track_stats_reliability"], errors="coerce")
        if "track_stats_reliability" in out.columns
        else pd.Series(0.0, index=out.index, dtype=float)
    ).fillna(0.0).clip(lower=0.0, upper=1.0)
    circuit_safety = _numeric_feature(out, "circuit_safety_car_probability", default=0.35)
    circuit_strategy = _numeric_feature(out, "circuit_strategy_variance", default=0.55)

    if "track_chaos_index" not in out.columns:
        out["track_chaos_index"] = (
            (0.50 * safety.fillna(0.2))
            + (0.30 * dnf.fillna(0.1))
            + (0.20 * pit_variance)
        )
    out["track_chaos_index"] = out["track_chaos_index"].clip(lower=0.0, upper=1.0)
    observed_safety = pd.to_numeric(safety, errors="coerce").fillna(circuit_safety)
    observed_dnf = pd.to_numeric(dnf, errors="coerce").fillna(0.10).clip(lower=0.0, upper=1.0)
    observed_strategy = (
        (0.45 * pit_variance)
        + (0.25 * pd.to_numeric(out["track_chaos_index"], errors="coerce").fillna(0.20))
        + (0.20 * observed_safety)
        + (0.10 * observed_dnf)
    ).clip(lower=0.0, upper=1.0)
    observed_weather = pd.to_numeric(weather, errors="coerce")
    weather_default = (0.12 + (0.18 * circuit_strategy)).clip(lower=0.0, upper=1.0)
    observed_weather = observed_weather.fillna(weather_default).clip(lower=0.0, upper=1.0)
    observed_weight = (0.25 + (0.55 * reliability)).clip(lower=0.25, upper=0.80)
    out["track_safety_car_prior"] = (
        ((1.0 - observed_weight) * circuit_safety) + (observed_weight * observed_safety)
    ).clip(lower=0.0, upper=1.0)
    out["track_dnf_prior"] = observed_dnf
    out["track_strategy_variance_prior"] = (
        ((1.0 - observed_weight) * circuit_strategy) + (observed_weight * observed_strategy)
    ).clip(lower=0.0, upper=1.0)
    out["track_weather_uncertainty_prior"] = observed_weather
    out["race_generation_variance_prior"] = (
        (0.34 * out["track_safety_car_prior"])
        + (0.26 * out["track_dnf_prior"])
        + (0.25 * out["track_strategy_variance_prior"])
        + (0.15 * out["track_weather_uncertainty_prior"])
    ).clip(lower=0.0, upper=1.0)

    out["track_qualy_importance"] = (1.0 - (0.65 * mobility.fillna(0.5)) - (0.35 * safety.fillna(0.2))).clip(
        lower=0.0,
        upper=1.0,
    )
    if "circuit_qualifying_importance" in out.columns:
        circuit_qualy = pd.to_numeric(out["circuit_qualifying_importance"], errors="coerce").fillna(
            out["track_qualy_importance"],
        )
        reliability = (
            pd.to_numeric(out["circuit_card_reliability"], errors="coerce")
            if "circuit_card_reliability" in out.columns
            else pd.Series(0.35, index=out.index, dtype=float)
        ).fillna(0.35).clip(lower=0.0, upper=1.0)
        card_weight = (0.30 + (0.35 * reliability)).clip(lower=0.30, upper=0.65)
        out["track_qualy_importance"] = (
            ((1.0 - card_weight) * out["track_qualy_importance"]) + (card_weight * circuit_qualy)
        ).clip(lower=0.0, upper=1.0)

    if "qualy_position" in out.columns:
        qualy_pos = pd.to_numeric(out["qualy_position"], errors="coerce")
        out["qualy_position_track_adj"] = qualy_pos * (0.35 + out["track_qualy_importance"])
    else:
        out["qualy_position_track_adj"] = float("nan")

    if "qualy_gap_to_best" in out.columns:
        qualy_gap = pd.to_numeric(out["qualy_gap_to_best"], errors="coerce")
        out["qualy_gap_track_adj"] = qualy_gap * (0.35 + out["track_qualy_importance"])
    else:
        out["qualy_gap_track_adj"] = float("nan")

    if "fp_race_sim_delta" in out.columns:
        fp_race = pd.to_numeric(out["fp_race_sim_delta"], errors="coerce")
        out["fp_race_sim_delta_track_adj"] = fp_race * (
            1.0 + (0.75 * mobility.fillna(0.5)) + (0.35 * safety.fillna(0.2))
        )
    else:
        out["fp_race_sim_delta_track_adj"] = float("nan")

    if "fp_weighted_delta" in out.columns:
        fp_weighted = pd.to_numeric(out["fp_weighted_delta"], errors="coerce")
        out["fp_weighted_delta_track_adj"] = fp_weighted * (
            1.0 + (0.45 * mobility.fillna(0.5)) + (0.20 * safety.fillna(0.2))
        )
    else:
        out["fp_weighted_delta_track_adj"] = float("nan")

    if "qualy_pred_position" in out.columns:
        qualy_pred = pd.to_numeric(out["qualy_pred_position"], errors="coerce")
        out["qualy_pred_position_track_adj"] = qualy_pred * (0.35 + out["track_qualy_importance"])
    else:
        out["qualy_pred_position_track_adj"] = float("nan")

    downforce = _numeric_feature(out, "circuit_downforce_demand", default=0.55)
    power = _numeric_feature(out, "circuit_power_sensitivity", default=0.55)
    tyre = _numeric_feature(out, "circuit_tyre_degradation", default=0.55)
    qualy_importance = _numeric_feature(out, "circuit_qualifying_importance", default=0.60)

    if "fp_weighted_delta" in out.columns:
        fp_weighted = pd.to_numeric(out["fp_weighted_delta"], errors="coerce")
        out["fp_weighted_delta_downforce_adj"] = fp_weighted * (0.45 + downforce)
        out["fp_weighted_delta_power_adj"] = fp_weighted * (0.45 + power)
    else:
        out["fp_weighted_delta_downforce_adj"] = float("nan")
        out["fp_weighted_delta_power_adj"] = float("nan")

    if "fp_quali_sim_delta" in out.columns:
        fp_quali = pd.to_numeric(out["fp_quali_sim_delta"], errors="coerce")
        out["fp_quali_sim_delta_downforce_adj"] = fp_quali * (0.45 + downforce + (0.25 * qualy_importance))
    else:
        out["fp_quali_sim_delta_downforce_adj"] = float("nan")

    if "fp_race_sim_delta" in out.columns:
        fp_race = pd.to_numeric(out["fp_race_sim_delta"], errors="coerce")
        out["fp_race_sim_delta_tyre_adj"] = fp_race * (0.45 + tyre)
        out["fp_race_sim_delta_power_adj"] = fp_race * (0.45 + power)
    else:
        out["fp_race_sim_delta_tyre_adj"] = float("nan")
        out["fp_race_sim_delta_power_adj"] = float("nan")

    if "qualy_position" in out.columns:
        qualy_pos = pd.to_numeric(out["qualy_position"], errors="coerce")
        out["qualy_position_circuit_importance_adj"] = qualy_pos * (0.35 + qualy_importance)
    else:
        out["qualy_position_circuit_importance_adj"] = float("nan")

    if "qualy_pred_position" in out.columns:
        qualy_pred = pd.to_numeric(out["qualy_pred_position"], errors="coerce")
        out["qualy_pred_position_circuit_importance_adj"] = qualy_pred * (0.35 + qualy_importance)
    else:
        out["qualy_pred_position_circuit_importance_adj"] = float("nan")

    def _component(column: str, default: float = 0.5) -> pd.Series:
        if column not in out.columns:
            return pd.Series(float(default), index=out.index, dtype=float)
        ranked = _rank_percentile(out[column], ascending=True)
        return ranked.fillna(float(default)).clip(lower=0.0, upper=1.0)

    weighted_fit = _component("fp_weighted_delta")
    quali_fit = _component("fp_quali_sim_delta")
    race_fit = _component("fp_race_sim_delta")
    consistency_fit = _component("fp_delta_std")
    low_speed = _numeric_feature(out, "circuit_low_speed_corner_demand", default=0.55)
    traction = _numeric_feature(out, "circuit_traction_demand", default=0.55)
    out["circuit_fit_index"] = (
        downforce * ((0.60 * quali_fit) + (0.25 * weighted_fit) + (0.15 * consistency_fit))
        + power * ((0.55 * race_fit) + (0.30 * weighted_fit) + (0.15 * quali_fit))
        + tyre * ((0.55 * race_fit) + (0.25 * consistency_fit) + (0.20 * weighted_fit))
        + low_speed * ((0.55 * weighted_fit) + (0.30 * quali_fit) + (0.15 * consistency_fit))
        + traction * ((0.50 * weighted_fit) + (0.30 * race_fit) + (0.20 * consistency_fit))
    ) / (downforce + power + tyre + low_speed + traction).replace(0.0, 1.0)

    # Accept the legacy provider alias above, but never expose it downstream:
    # observed grid-to-finish movement is not an overtake probability.
    out = out.drop(columns=["track_overtake_propensity"], errors="ignore")
    return add_f1_wet_pace_interactions(out)


def _numeric_feature(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(float(default)).clip(lower=0.0, upper=1.0)


def _ensure_fp_mean_delta(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    delta_cols = [
        c
        for c in out.columns
        if c.endswith("_delta")
        and c not in {"fp_mean_delta", "fp_weighted_delta", "qualy_gap_to_best"}
        and not c.endswith("_top3_delta")
        and not c.endswith("_median_delta")
        and not c.endswith("_quali_sim_delta")
        and not c.endswith("_race_sim_delta")
    ]
    delta_frame = out[delta_cols].apply(pd.to_numeric, errors="coerce") if delta_cols else pd.DataFrame(index=out.index)

    if "fp_mean_delta" not in out.columns:
        if not delta_frame.empty:
            out["fp_mean_delta"] = delta_frame.mean(axis=1, skipna=True)
        else:
            out["fp_mean_delta"] = float("nan")
    else:
        out["fp_mean_delta"] = pd.to_numeric(out["fp_mean_delta"], errors="coerce")

    if "fp_delta_std" in out.columns:
        out["fp_delta_std"] = pd.to_numeric(out["fp_delta_std"], errors="coerce")
    elif not delta_frame.empty:
        out["fp_delta_std"] = delta_frame.std(axis=1, skipna=True)
    else:
        out["fp_delta_std"] = float("nan")

    if "pace_sessions_available" in out.columns:
        out["pace_sessions_available"] = pd.to_numeric(out["pace_sessions_available"], errors="coerce")
    elif not delta_frame.empty:
        out["pace_sessions_available"] = delta_frame.notna().sum(axis=1)
    else:
        out["pace_sessions_available"] = 0.0

    if "fp_quali_sim_delta" in out.columns:
        out["fp_quali_sim_delta"] = pd.to_numeric(out["fp_quali_sim_delta"], errors="coerce")
    else:
        quali_sim_delta_cols = [c for c in out.columns if c.endswith("_quali_sim_delta")]
        if quali_sim_delta_cols:
            out["fp_quali_sim_delta"] = (
                out[quali_sim_delta_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
            )
        else:
            out["fp_quali_sim_delta"] = float("nan")

    if "fp_race_sim_delta" in out.columns:
        out["fp_race_sim_delta"] = pd.to_numeric(out["fp_race_sim_delta"], errors="coerce")
    else:
        race_sim_delta_cols = [c for c in out.columns if c.endswith("_race_sim_delta")]
        if race_sim_delta_cols:
            out["fp_race_sim_delta"] = (
                out[race_sim_delta_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
            )
        else:
            out["fp_race_sim_delta"] = float("nan")

    if "fp_quali_sim_rank" in out.columns:
        out["fp_quali_sim_rank"] = pd.to_numeric(out["fp_quali_sim_rank"], errors="coerce")
    else:
        quali_sim_rank_cols = [c for c in out.columns if c.endswith("_quali_sim_rank")]
        if quali_sim_rank_cols:
            out["fp_quali_sim_rank"] = (
                out[quali_sim_rank_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
            )
        else:
            out["fp_quali_sim_rank"] = float("nan")

    if "fp_race_sim_rank" in out.columns:
        out["fp_race_sim_rank"] = pd.to_numeric(out["fp_race_sim_rank"], errors="coerce")
    else:
        race_sim_rank_cols = [c for c in out.columns if c.endswith("_race_sim_rank")]
        if race_sim_rank_cols:
            out["fp_race_sim_rank"] = (
                out[race_sim_rank_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
            )
        else:
            out["fp_race_sim_rank"] = float("nan")

    if "fp_quali_sim_laps" in out.columns:
        out["fp_quali_sim_laps"] = pd.to_numeric(out["fp_quali_sim_laps"], errors="coerce")
    else:
        quali_sim_lap_cols = [c for c in out.columns if c.endswith("_quali_sim_lap_count")]
        if quali_sim_lap_cols:
            out["fp_quali_sim_laps"] = (
                out[quali_sim_lap_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)
            )
        else:
            out["fp_quali_sim_laps"] = 0.0

    if "fp_race_sim_laps" in out.columns:
        out["fp_race_sim_laps"] = pd.to_numeric(out["fp_race_sim_laps"], errors="coerce")
    else:
        race_sim_lap_cols = [c for c in out.columns if c.endswith("_race_sim_lap_count")]
        if race_sim_lap_cols:
            out["fp_race_sim_laps"] = (
                out[race_sim_lap_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)
            )
        else:
            out["fp_race_sim_laps"] = 0.0

    if "quali_sim_sessions_available" in out.columns:
        out["quali_sim_sessions_available"] = pd.to_numeric(out["quali_sim_sessions_available"], errors="coerce")
    else:
        quali_sim_delta_cols = [c for c in out.columns if c.endswith("_quali_sim_delta")]
        if quali_sim_delta_cols:
            out["quali_sim_sessions_available"] = (
                out[quali_sim_delta_cols].apply(pd.to_numeric, errors="coerce").notna().sum(axis=1)
            )
        else:
            out["quali_sim_sessions_available"] = 0.0

    if "race_sim_sessions_available" in out.columns:
        out["race_sim_sessions_available"] = pd.to_numeric(out["race_sim_sessions_available"], errors="coerce")
    else:
        race_sim_delta_cols = [c for c in out.columns if c.endswith("_race_sim_delta")]
        if race_sim_delta_cols:
            out["race_sim_sessions_available"] = (
                out[race_sim_delta_cols].apply(pd.to_numeric, errors="coerce").notna().sum(axis=1)
            )
        else:
            out["race_sim_sessions_available"] = 0.0

    if "fp_slow_lap_ratio" in out.columns:
        out["fp_slow_lap_ratio"] = pd.to_numeric(out["fp_slow_lap_ratio"], errors="coerce")
    else:
        slow_cols = [c for c in out.columns if c.endswith("_slow_lap_ratio")]
        if slow_cols:
            out["fp_slow_lap_ratio"] = (
                out[slow_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
            )
        else:
            out["fp_slow_lap_ratio"] = float("nan")

    if "fp_quali_vs_race_gap" in out.columns:
        out["fp_quali_vs_race_gap"] = pd.to_numeric(out["fp_quali_vs_race_gap"], errors="coerce")
    else:
        gap_cols = [c for c in out.columns if c.endswith("_quali_vs_race_gap")]
        if gap_cols:
            out["fp_quali_vs_race_gap"] = (
                out[gap_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
            )
        else:
            out["fp_quali_vs_race_gap"] = out["fp_race_sim_delta"] - out["fp_quali_sim_delta"]

    if "fp_weighted_delta" in out.columns:
        out["fp_weighted_delta"] = pd.to_numeric(out["fp_weighted_delta"], errors="coerce")
    else:
        weighted_sum = pd.Series(0.0, index=out.index, dtype=float)
        weight_total = pd.Series(0.0, index=out.index, dtype=float)
        for col, weight in PACE_DELTA_WEIGHTS.items():
            if col not in out.columns:
                continue
            values = pd.to_numeric(out[col], errors="coerce")
            valid = values.notna()
            weighted_sum.loc[valid] = weighted_sum.loc[valid] + (values.loc[valid] * weight)
            weight_total.loc[valid] = weight_total.loc[valid] + weight
        out["fp_weighted_delta"] = weighted_sum.divide(weight_total.where(weight_total > 0.0))
        out["fp_weighted_delta"] = out["fp_weighted_delta"].fillna(out["fp_mean_delta"])

    if "fp_mean_top3_delta" in out.columns:
        out["fp_mean_top3_delta"] = pd.to_numeric(out["fp_mean_top3_delta"], errors="coerce")
    else:
        top3_cols = [c for c in out.columns if c.endswith("_top3_delta")]
        if top3_cols:
            out["fp_mean_top3_delta"] = (
                out[top3_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
            )
        else:
            out["fp_mean_top3_delta"] = float("nan")

    if "fp_mean_lap_std" in out.columns:
        out["fp_mean_lap_std"] = pd.to_numeric(out["fp_mean_lap_std"], errors="coerce")
    else:
        lap_std_cols = [c for c in out.columns if c.endswith("_lap_std")]
        if lap_std_cols:
            out["fp_mean_lap_std"] = (
                out[lap_std_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
            )
        else:
            out["fp_mean_lap_std"] = float("nan")

    if "fp_total_laps" in out.columns:
        out["fp_total_laps"] = pd.to_numeric(out["fp_total_laps"], errors="coerce")
    else:
        lap_count_cols = [c for c in out.columns if c.endswith("_lap_count")]
        if lap_count_cols:
            out["fp_total_laps"] = (
                out[lap_count_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)
            )
        else:
            out["fp_total_laps"] = 0.0

    return out


def _rank_percentile(values: pd.Series, ascending: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(0.5, index=numeric.index, dtype=float)
    return numeric.rank(method="average", pct=True, ascending=ascending)


def _average_event_rank_component(
    frame: pd.DataFrame,
    columns: list[str],
    ascending: bool,
) -> Optional[pd.Series]:
    parts: list[pd.Series] = []
    for col in columns:
        if col not in frame.columns:
            continue
        ranked = _rank_percentile(frame[col], ascending=ascending)
        if ranked.notna().sum() == 0:
            continue
        parts.append(ranked)
    if not parts:
        return None
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def _add_event_relative_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    out = frame.copy()
    if "event_key" in out.columns:
        event_key = pd.to_numeric(out["event_key"], errors="coerce").fillna(-1).astype(int)
    else:
        event_key = pd.Series(0, index=out.index, dtype=int)

    out["event_pace_index"] = float("nan")
    for _, idx in event_key.groupby(event_key, sort=False).groups.items():
        event_rows = out.loc[idx]
        weighted = pd.Series(0.0, index=event_rows.index, dtype=float)
        weight_total = pd.Series(0.0, index=event_rows.index, dtype=float)

        pace_core = _average_event_rank_component(
            event_rows,
            [
                "fp_quali_sim_delta",
                "fp_mean_rank",
                "fp_weighted_delta",
                "fp_mean_top3_delta",
                "fp_race_sim_delta",
            ],
            ascending=True,
        )
        if pace_core is not None:
            weighted = weighted + (0.55 * pace_core)
            weight_total = weight_total + 0.55

        consistency = _average_event_rank_component(
            event_rows,
            ["fp_delta_std", "fp_mean_lap_std", "fp_slow_lap_ratio"],
            ascending=True,
        )
        if consistency is not None:
            weighted = weighted + (0.20 * consistency)
            weight_total = weight_total + 0.20

        availability = _average_event_rank_component(
            event_rows,
            [
                "pace_sessions_available",
                "fp_total_laps",
                "quali_sim_sessions_available",
                "race_sim_sessions_available",
                "fp_quali_sim_laps",
                "fp_race_sim_laps",
            ],
            ascending=False,
        )
        if availability is not None:
            weighted = weighted + (0.25 * availability)
            weight_total = weight_total + 0.25

        score = weighted.divide(weight_total.where(weight_total > 0.0))
        if score.notna().sum() == 0:
            score = pd.Series(0.5, index=event_rows.index, dtype=float)
        out.loc[event_rows.index, "event_pace_index"] = score.fillna(float(score.median(skipna=True)))

    out["driver_vs_team_fp_weighted_delta"] = float("nan")
    team_col = team_column(out)
    if team_col and "fp_weighted_delta" in out.columns:
        team_key = (
            out[team_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"nan": pd.NA, "none": pd.NA, "<na>": pd.NA, "": pd.NA})
        )
        weighted = pd.to_numeric(out["fp_weighted_delta"], errors="coerce")
        team_mean = weighted.groupby([event_key, team_key], sort=False).transform("mean")
        event_mean = weighted.groupby(event_key, sort=False).transform("mean")
        rel = weighted - team_mean
        fallback_rel = weighted - event_mean
        out["driver_vs_team_fp_weighted_delta"] = rel.fillna(fallback_rel)

    return out


def _event_order_series(frame: pd.DataFrame) -> pd.Series:
    if "event_key" in frame.columns:
        return pd.to_numeric(frame["event_key"], errors="coerce")
    if "event_year" in frame.columns and "event_round" in frame.columns:
        year = pd.to_numeric(frame["event_year"], errors="coerce")
        round_number = pd.to_numeric(frame["event_round"], errors="coerce")
        return (year * 100) + round_number
    return pd.Series(range(len(frame)), index=frame.index, dtype=float)


def _assign_team_event_history(
    out: pd.DataFrame,
    *,
    team_col: Optional[str],
    value_col: str,
    ewma_col: Optional[str] = None,
    form3_col: Optional[str] = None,
    form5_col: Optional[str] = None,
) -> None:
    target_cols = [col for col in [ewma_col, form3_col, form5_col] if col]
    for col in target_cols:
        out[col] = float("nan")
    if not team_col or value_col not in out.columns:
        return

    work = pd.DataFrame(
        {
            "team_key": out[team_col],
            "event_order": _event_order_series(out),
            "value": pd.to_numeric(out[value_col], errors="coerce"),
        },
        index=out.index,
    )
    work = work.dropna(subset=["team_key", "event_order", "value"])
    if work.empty:
        return

    event_values = (
        work.groupby(["team_key", "event_order"], sort=True)["value"]
        .mean()
        .reset_index()
        .sort_values(["team_key", "event_order"], kind="mergesort")
    )
    team_values = event_values.groupby("team_key", sort=False)["value"]
    if ewma_col:
        event_values[ewma_col] = team_values.transform(
            lambda s: s.shift(1).ewm(alpha=0.5, adjust=False, min_periods=1).mean(),
        )
    if form3_col:
        event_values[form3_col] = team_values.transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).mean(),
        )
    if form5_col:
        event_values[form5_col] = team_values.transform(
            lambda s: s.shift(1).rolling(window=5, min_periods=1).mean(),
        )

    for col in target_cols:
        values = event_values.set_index(["team_key", "event_order"])[col]
        keys = list(zip(work["team_key"], work["event_order"]))
        mapped = pd.Series([values.get(key, float("nan")) for key in keys], index=work.index, dtype=float)
        out.loc[work.index, col] = mapped


def _assign_team_context_event_history(
    out: pd.DataFrame,
    *,
    team_col: Optional[str],
    context_col: str,
    value_col: str,
    output_col: str,
    window: Optional[int],
) -> None:
    out[output_col] = float("nan")
    if not team_col or context_col not in out.columns or value_col not in out.columns:
        return
    work = pd.DataFrame(
        {
            "team_key": out[team_col],
            "context_key": out[context_col],
            "event_order": _event_order_series(out),
            "value": pd.to_numeric(out[value_col], errors="coerce"),
        },
        index=out.index,
    )
    work = work.dropna(subset=["team_key", "context_key", "event_order", "value"])
    if work.empty:
        return
    event_values = (
        work.groupby(["team_key", "context_key", "event_order"], sort=True)["value"]
        .mean()
        .reset_index()
        .sort_values(["team_key", "context_key", "event_order"], kind="mergesort")
    )
    grouped = event_values.groupby(["team_key", "context_key"], sort=False)["value"]
    if window is None:
        event_values[output_col] = grouped.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    else:
        event_values[output_col] = grouped.transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
    values = event_values.set_index(["team_key", "context_key", "event_order"])[output_col]
    keys = list(zip(work["team_key"], work["context_key"], work["event_order"]))
    mapped = pd.Series([values.get(key, float("nan")) for key in keys], index=work.index, dtype=float)
    out.loc[work.index, output_col] = mapped


def _add_temporal_features_train(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = _ensure_fp_mean_delta(frame)
    if "driver_id" in out.columns:
        out["driver_id"] = out["driver_id"].astype(str)
    out["event_name"] = out.get("event_name", pd.Series(index=out.index, dtype=object))
    out["event_name_norm"] = out["event_name"].map(normalize_event_name)
    if "circuit_archetype" in out.columns:
        out["circuit_archetype"] = out["circuit_archetype"].astype(str).str.strip().str.lower()
    if "circuit_card_id" in out.columns:
        out["circuit_card_id"] = out["circuit_card_id"].astype(str).str.strip().str.lower()

    sort_cols = [c for c in ["event_key", "event_year", "event_round", "driver_id"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    driver_group = out.groupby("driver_id", sort=False)["fp_mean_delta"] if "driver_id" in out.columns else None
    if driver_group is not None:
        out["driver_ewma_fp_mean_delta"] = driver_group.transform(
            lambda s: s.shift(1).ewm(alpha=0.5, adjust=False, min_periods=1).mean(),
        )
        out["driver_form_3_fp_mean_delta"] = driver_group.transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).mean(),
        )
        out["driver_form_5_fp_mean_delta"] = driver_group.transform(
            lambda s: s.shift(1).rolling(window=5, min_periods=1).mean(),
        )
    else:
        out["driver_ewma_fp_mean_delta"] = float("nan")
        out["driver_form_3_fp_mean_delta"] = float("nan")
        out["driver_form_5_fp_mean_delta"] = float("nan")

    driver_weighted_group = (
        out.groupby("driver_id", sort=False)["fp_weighted_delta"] if "driver_id" in out.columns else None
    )
    if driver_weighted_group is not None:
        out["driver_ewma_fp_weighted_delta"] = driver_weighted_group.transform(
            lambda s: s.shift(1).ewm(alpha=0.5, adjust=False, min_periods=1).mean(),
        )
        out["driver_form_3_fp_weighted_delta"] = driver_weighted_group.transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).mean(),
        )
    else:
        out["driver_ewma_fp_weighted_delta"] = float("nan")
        out["driver_form_3_fp_weighted_delta"] = float("nan")

    driver_quali_sim_group = (
        out.groupby("driver_id", sort=False)["fp_quali_sim_delta"] if "driver_id" in out.columns else None
    )
    if driver_quali_sim_group is not None:
        out["driver_ewma_fp_quali_sim_delta"] = driver_quali_sim_group.transform(
            lambda s: s.shift(1).ewm(alpha=0.5, adjust=False, min_periods=1).mean(),
        )
        out["driver_form_3_fp_quali_sim_delta"] = driver_quali_sim_group.transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).mean(),
        )
    else:
        out["driver_ewma_fp_quali_sim_delta"] = float("nan")
        out["driver_form_3_fp_quali_sim_delta"] = float("nan")

    driver_race_sim_group = (
        out.groupby("driver_id", sort=False)["fp_race_sim_delta"] if "driver_id" in out.columns else None
    )
    if driver_race_sim_group is not None:
        out["driver_ewma_fp_race_sim_delta"] = driver_race_sim_group.transform(
            lambda s: s.shift(1).ewm(alpha=0.5, adjust=False, min_periods=1).mean(),
        )
        out["driver_form_3_fp_race_sim_delta"] = driver_race_sim_group.transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).mean(),
        )
    else:
        out["driver_ewma_fp_race_sim_delta"] = float("nan")
        out["driver_form_3_fp_race_sim_delta"] = float("nan")

    team_col = team_column(out)
    if team_col:
        out[team_col] = (
            out[team_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"nan": pd.NA, "none": pd.NA, "<na>": pd.NA, "": pd.NA})
        )
        _assign_team_event_history(
            out,
            team_col=team_col,
            value_col="fp_mean_delta",
            ewma_col="team_ewma_fp_mean_delta",
            form3_col="team_form_3_fp_mean_delta",
            form5_col="team_form_5_fp_mean_delta",
        )
    else:
        out["team_ewma_fp_mean_delta"] = float("nan")
        out["team_form_3_fp_mean_delta"] = float("nan")
        out["team_form_5_fp_mean_delta"] = float("nan")

    if team_col:
        _assign_team_event_history(
            out,
            team_col=team_col,
            value_col="fp_weighted_delta",
            ewma_col="team_ewma_fp_weighted_delta",
            form3_col="team_form_3_fp_weighted_delta",
        )
    else:
        out["team_ewma_fp_weighted_delta"] = float("nan")
        out["team_form_3_fp_weighted_delta"] = float("nan")

    out["driver_archetype_form_3_fp_weighted_delta"] = float("nan")
    out["team_archetype_form_3_fp_weighted_delta"] = float("nan")
    out["driver_circuit_hist_fp_weighted_delta"] = float("nan")
    out["team_circuit_hist_fp_weighted_delta"] = float("nan")

    if "fp_weighted_delta" in out.columns and "circuit_archetype" in out.columns:
        if "driver_id" in out.columns:
            driver_arch = out.groupby(["driver_id", "circuit_archetype"], sort=False)["fp_weighted_delta"]
            out["driver_archetype_form_3_fp_weighted_delta"] = driver_arch.transform(
                lambda s: s.shift(1).rolling(window=3, min_periods=1).mean(),
            )
        team_col_for_arch = team_column(out)
        if team_col_for_arch:
            _assign_team_context_event_history(
                out,
                team_col=team_col_for_arch,
                context_col="circuit_archetype",
                value_col="fp_weighted_delta",
                output_col="team_archetype_form_3_fp_weighted_delta",
                window=3,
            )
        if "driver_id" in out.columns and "circuit_card_id" in out.columns:
            driver_circuit = out.groupby(["driver_id", "circuit_card_id"], sort=False)["fp_weighted_delta"]
            out["driver_circuit_hist_fp_weighted_delta"] = driver_circuit.transform(
                lambda s: s.shift(1).expanding(min_periods=1).mean(),
            )
        if team_col_for_arch and "circuit_card_id" in out.columns:
            _assign_team_context_event_history(
                out,
                team_col=team_col_for_arch,
                context_col="circuit_card_id",
                value_col="fp_weighted_delta",
                output_col="team_circuit_hist_fp_weighted_delta",
                window=None,
            )

    if "driver_id" in out.columns:
        out["event_driver_hist_idx"] = out.groupby(
            ["event_name_norm", "driver_id"],
            sort=False,
        )["fp_mean_delta"].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean(),
        )
    else:
        out["event_driver_hist_idx"] = float("nan")

    out = _add_event_relative_features(out)
    if "driver_id" in out.columns:
        driver_vs_team_group = out.groupby("driver_id", sort=False)["driver_vs_team_fp_weighted_delta"]
        out["driver_form_3_vs_team_fp_weighted_delta"] = driver_vs_team_group.transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).mean(),
        )
    else:
        out["driver_form_3_vs_team_fp_weighted_delta"] = float("nan")

    return out


def _attach_temporal_features_current(
    current: pd.DataFrame,
    history: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if current.empty:
        return current
    out = _ensure_fp_mean_delta(current)
    out["driver_id"] = out["driver_id"].astype(str)
    out["event_name"] = out.get("event_name", pd.Series(index=out.index, dtype=object))
    out["event_name_norm"] = out["event_name"].map(normalize_event_name)
    if history is None or history.empty or "driver_id" not in history.columns:
        return _add_event_relative_features(out)

    # Run the exact shifted training transform on history + a marked current
    # event.  The shift excludes each current row while including every
    # completed historical event, including the latest one, exactly once.
    # Mapping the already-shifted tail row used to omit that latest event.
    original_index = out.index.copy()
    hist = history.copy()
    hist["__temporal_current_row"] = False
    hist["__temporal_current_order"] = -1
    current_rows = out.copy()
    current_rows["__temporal_current_row"] = True
    current_rows["__temporal_current_order"] = range(len(current_rows))
    combined = pd.concat([hist, current_rows], ignore_index=True, sort=False)
    transformed = _add_temporal_features_train(combined)
    current_out = transformed.loc[transformed["__temporal_current_row"].fillna(False)].copy()
    current_out = current_out.sort_values("__temporal_current_order", kind="mergesort")
    current_out = current_out.drop(columns=["__temporal_current_row", "__temporal_current_order"], errors="ignore")
    current_out.index = original_index
    return current_out


def _lookup_pair_series(
    *,
    keys_a: pd.Series,
    keys_b: pd.Series,
    values: pd.Series,
    index: pd.Index,
) -> pd.Series:
    if values.empty:
        return pd.Series(float("nan"), index=index, dtype=float)
    clean_values = values.copy()
    clean_values.index = pd.MultiIndex.from_tuples(
        [(str(a).strip().lower(), str(b).strip().lower()) for a, b in clean_values.index.tolist()],
    )
    keys = [(str(a).strip().lower(), str(b).strip().lower()) for a, b in zip(keys_a.tolist(), keys_b.tolist())]
    return pd.Series([clean_values.get(key, float("nan")) for key in keys], index=index, dtype=float)


def build_training_data(
    provider: BaseProvider,
    mode: str,
    train_seasons: List[int],
    target_year: int,
    target_round: int,
    include_standings: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[pd.DataFrame] = []
    notes: List[str] = []
    for year in train_seasons:
        try:
            rounds = provider.list_rounds(year)
        except (Exception, SystemExit) as exc:
            notes.append(f"Echec listing rounds {year}: {exc}")
            continue
        rounds_sorted = sorted(rounds, key=lambda r: int(r.get("round_number", 0)))
        for rnd in rounds_sorted:
            round_number = int(rnd.get("round_number", 0))
            event_name = _round_event_name(rnd, round_number)
            if year == target_year and round_number >= target_round:
                continue
            try:
                fp_features = provider.get_fp_features(year, round_number)
            except (Exception, SystemExit) as exc:
                notes.append(f"Echec FP {year} round {round_number}: {exc}")
                continue
            if fp_features.empty:
                continue
            fp_features = fp_features.copy()
            if "driver_id" in fp_features.columns:
                fp_features["driver_id"] = fp_features["driver_id"].astype(str)
            if mode == "qualifying":
                try:
                    qualy = provider.get_qualifying_results(year, round_number)
                except (Exception, SystemExit) as exc:
                    notes.append(f"Echec qualifs {year} round {round_number}: {exc}")
                    continue
                if qualy.empty:
                    continue
                qualy = qualy.copy()
                qualy["driver_id"] = qualy["driver_id"].astype(str)
                if "position" in qualy.columns:
                    qualy["position"] = pd.to_numeric(qualy["position"], errors="coerce")
                else:
                    qualy["position"] = float("nan")
                if "q3_time" in qualy.columns:
                    qualy["q3_time"] = pd.to_numeric(qualy["q3_time"], errors="coerce")
                else:
                    qualy["q3_time"] = float("nan")

                # Prefer official qualifying position for all drivers.
                # Fall back to q3 ranking only if position is unavailable.
                qualy["target"] = qualy["position"]
                if qualy["target"].notna().sum() == 0 and qualy["q3_time"].notna().sum() > 0:
                    qualy["target"] = qualy["q3_time"].rank(method="first", ascending=True)

                qualy = qualy.dropna(subset=["target"])
                if qualy.empty:
                    continue
                merged = fp_features.merge(qualy[["driver_id", "target"]], on="driver_id", how="inner")
                if merged.empty:
                    continue
                merged["event_name"] = event_name
                merged["event_year"] = year
                merged["event_round"] = round_number
                merged["event_key"] = (year * 100) + round_number
                merged = _attach_track_stats(
                    frame=merged,
                    provider=provider,
                    year=year,
                    round_number=round_number,
                    notes=notes,
                )
                rows.append(merged)
            else:
                try:
                    race = provider.get_race_results(year, round_number)
                    qualy = provider.get_qualifying_results(year, round_number)
                except (Exception, SystemExit) as exc:
                    notes.append(f"Echec race/qualifs {year} round {round_number}: {exc}")
                    continue
                if race.empty or qualy.empty:
                    continue
                qualy = qualy.copy()
                race = race.copy()
                qualy["driver_id"] = qualy["driver_id"].astype(str)
                race["driver_id"] = race["driver_id"].astype(str)
                qualy["position"] = pd.to_numeric(qualy["position"], errors="coerce")
                qualy_merge_cols = ["driver_id", "position"]
                if "q3_time" in qualy.columns:
                    qualy["q3_time"] = pd.to_numeric(qualy["q3_time"], errors="coerce")
                    best_q3 = qualy["q3_time"].min(skipna=True)
                    if pd.notna(best_q3):
                        qualy["qualy_gap_to_best"] = qualy["q3_time"] - best_q3
                        qualy_merge_cols.append("qualy_gap_to_best")
                merged = fp_features.merge(qualy[qualy_merge_cols], on="driver_id", how="inner")
                merged = merged.rename(columns={"position": "qualy_position"})
                race_merge_cols = ["driver_id", "position"]
                if "grid_position" in race.columns:
                    race["grid_position"] = pd.to_numeric(race["grid_position"], errors="coerce")
                    race_merge_cols.append("grid_position")
                if "grid_status" in race.columns:
                    race_merge_cols.append("grid_status")
                merged = merged.merge(race[race_merge_cols], on="driver_id", how="inner")
                merged = merged.rename(columns={"position": "target"})
                if "grid_position" not in merged.columns:
                    merged["grid_position"] = float("nan")
                if "grid_status" not in merged.columns:
                    merged["grid_status"] = "unknown"
                else:
                    merged["grid_status"] = merged["grid_status"].fillna("unknown")
                nonstarters = merged["grid_status"].astype(str).str.lower().isin(["dns"])
                if nonstarters.any():
                    notes.append(
                        f"{int(nonstarters.sum())} ligne(s) DNS exclues du target race pour eviter un fallback grille qualif."
                    )
                    merged = merged.loc[~nonstarters].copy()
                    if merged.empty:
                        continue
                grid_raw = pd.to_numeric(merged["grid_position"], errors="coerce")
                qualy_grid_fallback = pd.to_numeric(merged["qualy_position"], errors="coerce")
                merged["grid_source"] = pd.Series("qualifying_fallback", index=merged.index, dtype=object)
                merged.loc[grid_raw.notna(), "grid_source"] = "retrospective_results_grid"
                merged["grid_position"] = grid_raw.fillna(qualy_grid_fallback)
                merged["race_delta_target"] = pd.to_numeric(merged["target"], errors="coerce") - merged["grid_position"]
                if include_standings:
                    try:
                        standings = provider.get_standings(year, round_number)
                    except (Exception, SystemExit) as exc:
                        notes.append(f"Echec standings {year} round {round_number}: {exc}")
                        standings = None
                    if standings is not None and not standings.empty:
                        standings = standings.copy()
                        standings["driver_id"] = standings["driver_id"].astype(str)
                        merged = merged.merge(
                            standings[["driver_id", "position_start"]],
                            on="driver_id",
                            how="left",
                        )
                if merged.empty:
                    continue
                merged["event_name"] = event_name
                merged["event_year"] = year
                merged["event_round"] = round_number
                merged["event_key"] = (year * 100) + round_number
                merged = _attach_track_stats(
                    frame=merged,
                    provider=provider,
                    year=year,
                    round_number=round_number,
                    notes=notes,
                )
                rows.append(merged)
    if not rows:
        notes.append("Pas assez de data historique: fallback heuristique.")
        return pd.DataFrame(), notes
    train = pd.concat(rows, ignore_index=True)
    train = _add_temporal_features_train(train)
    return train, notes


def _latest_prior_field(
    provider: BaseProvider,
    year: int,
    round_number: int,
    notes: List[str],
) -> Tuple[pd.DataFrame, Optional[int]]:
    try:
        rounds = provider.list_rounds(year)
    except (Exception, SystemExit) as exc:
        notes.append(f"Echec listing rounds {year} pour field provisoire: {exc}")
        return pd.DataFrame(), None

    prior_rounds = sorted(
        {
            int(rnd.get("round_number", 0))
            for rnd in rounds
            if int(rnd.get("round_number", 0)) > 0 and int(rnd.get("round_number", 0)) < int(round_number)
        },
        reverse=True,
    )
    for prior_round in prior_rounds:
        for getter_name in ("get_race_results", "get_qualifying_results", "get_fp_features"):
            getter = getattr(provider, getter_name, None)
            if getter is None:
                continue
            try:
                candidate = getter(year, prior_round)
            except (Exception, SystemExit) as exc:
                notes.append(f"Echec field provisoire {year} round {prior_round} via {getter_name}: {exc}")
                continue
            if candidate is None or candidate.empty or "driver_id" not in candidate.columns:
                continue
            frame = candidate.copy()
            frame["driver_id"] = frame["driver_id"].astype(str)
            frame = frame[frame["driver_id"].str.strip() != ""]
            if frame.empty:
                continue
            if "driver_name" not in frame.columns:
                frame["driver_name"] = frame["driver_id"]
            team_col = team_column(frame)
            if team_col is None:
                frame["team_name"] = pd.NA
                team_col = "team_name"
            roster = (
                frame[["driver_id", "driver_name", team_col]]
                .rename(columns={team_col: "team_name"})
                .drop_duplicates(subset=["driver_id"], keep="last")
                .reset_index(drop=True)
            )
            return roster, prior_round
    return pd.DataFrame(), None


def _provisional_current_features_from_history(
    provider: BaseProvider,
    mode: str,
    year: int,
    round_number: int,
    include_standings: bool,
    history: Optional[pd.DataFrame],
    notes: List[str],
) -> pd.DataFrame:
    roster, source_round = _latest_prior_field(provider, year, round_number, notes)
    if roster.empty:
        notes.append("Field provisoire indisponible: aucun round precedent avec pilotes.")
        return pd.DataFrame()

    event_name = _event_name_for_round(provider, year, round_number, notes)
    current = roster.copy()
    current["event_name"] = event_name
    current["event_year"] = year
    current["event_round"] = round_number
    current["event_key"] = (year * 100) + round_number
    current["pace_sessions_available"] = 0.0
    current["fp_total_laps"] = 0.0
    current["provisional_current_field"] = True
    current["provisional_field_source_round"] = source_round

    if mode != "qualifying":
        current["qualy_position"] = float("nan")
        current["qualy_gap_to_best"] = float("nan")
        current["grid_position"] = float("nan")
        current["grid_source"] = "missing"
        current["grid_status"] = "unknown"

    if include_standings:
        try:
            standings = provider.get_standings(year, round_number)
        except (Exception, SystemExit) as exc:
            notes.append(f"Echec recuperation standings: {exc}")
            standings = None
        if standings is not None and not standings.empty:
            standings = standings.copy()
            standings["driver_id"] = standings["driver_id"].astype(str)
            current = current.merge(
                standings[["driver_id", "position_start"]],
                on="driver_id",
                how="left",
            )

    current = _attach_track_stats(
        frame=current,
        provider=provider,
        year=year,
        round_number=round_number,
        notes=notes,
    )
    current = _attach_temporal_features_current(current, history)
    current = _fill_provisional_pace_proxy(current)
    notes.append(
        "Aucune donnee FP disponible pour ce round: field provisoire construit "
        f"depuis le round {source_round}; proxy pace derive de la forme temporelle, "
        "aucune lap time cible synthetisee."
    )
    return current


def _first_numeric_feature(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            return values
    return pd.Series(float("nan"), index=frame.index, dtype=float)


def _fill_provisional_pace_proxy(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "provisional_current_field" not in frame.columns:
        return frame
    out = frame.copy()

    proxy_specs = {
        "fp_mean_delta": [
            "driver_ewma_fp_mean_delta",
            "driver_form_3_fp_mean_delta",
            "driver_form_5_fp_mean_delta",
            "team_ewma_fp_mean_delta",
            "team_form_3_fp_mean_delta",
        ],
        "fp_weighted_delta": [
            "driver_ewma_fp_weighted_delta",
            "driver_form_3_fp_weighted_delta",
            "team_ewma_fp_weighted_delta",
            "team_form_3_fp_weighted_delta",
            "driver_ewma_fp_mean_delta",
        ],
        "fp_quali_sim_delta": [
            "driver_ewma_fp_quali_sim_delta",
            "driver_form_3_fp_quali_sim_delta",
            "driver_ewma_fp_weighted_delta",
            "driver_form_3_fp_weighted_delta",
        ],
        "fp_race_sim_delta": [
            "driver_ewma_fp_race_sim_delta",
            "driver_form_3_fp_race_sim_delta",
            "driver_ewma_fp_weighted_delta",
            "driver_form_3_fp_weighted_delta",
        ],
    }
    for target, sources in proxy_specs.items():
        proxy = _first_numeric_feature(out, sources)
        current = pd.to_numeric(out[target], errors="coerce") if target in out.columns else proxy
        out[target] = current.where(current.notna(), proxy)

    if "fp_mean_top3_delta" not in out.columns:
        out["fp_mean_top3_delta"] = pd.to_numeric(out["fp_mean_delta"], errors="coerce")
    if "fp_quali_vs_race_gap" not in out.columns:
        out["fp_quali_vs_race_gap"] = (
            pd.to_numeric(out["fp_race_sim_delta"], errors="coerce")
            - pd.to_numeric(out["fp_quali_sim_delta"], errors="coerce")
        )
    for count_col in [
        "pace_sessions_available",
        "fp_total_laps",
        "quali_sim_sessions_available",
        "race_sim_sessions_available",
        "fp_quali_sim_laps",
        "fp_race_sim_laps",
    ]:
        out[count_col] = 0.0

    rank_specs = {
        "fp_mean_rank": "fp_mean_delta",
        "fp_quali_sim_rank": "fp_quali_sim_delta",
        "fp_race_sim_rank": "fp_race_sim_delta",
    }
    for rank_col, value_col in rank_specs.items():
        if value_col in out.columns:
            values = pd.to_numeric(out[value_col], errors="coerce")
        else:
            values = pd.Series(float("nan"), index=out.index, dtype=float)
        out[rank_col] = values.rank(method="average", ascending=True)

    out["provisional_pace_proxy"] = "temporal_form"
    return _add_event_relative_features(out)


def build_current_features(
    provider: BaseProvider,
    mode: str,
    year: int,
    round_number: int,
    include_standings: bool,
    history: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    try:
        fp_features = provider.get_fp_features(year, round_number)
    except (Exception, SystemExit) as exc:
        notes.append(f"Echec recuperation FP: {exc}")
        return pd.DataFrame(), notes
    if fp_features.empty:
        notes.append("Aucune donnee FP disponible pour ce round.")
        current = _provisional_current_features_from_history(
            provider=provider,
            mode=mode,
            year=year,
            round_number=round_number,
            include_standings=include_standings,
            history=history,
            notes=notes,
        )
        return current, notes
    fp_features = fp_features.copy()
    fp_features["driver_id"] = fp_features["driver_id"].astype(str)
    event_name = _event_name_for_round(provider, year, round_number, notes)
    if mode == "qualifying":
        fp_features["event_name"] = event_name
        fp_features["event_year"] = year
        fp_features["event_round"] = round_number
        fp_features["event_key"] = (year * 100) + round_number
        fp_features = _attach_track_stats(
            frame=fp_features,
            provider=provider,
            year=year,
            round_number=round_number,
            notes=notes,
        )
        current = _attach_temporal_features_current(fp_features, history)
        return current, notes
    qualy = pd.DataFrame()
    try:
        qualy = provider.get_qualifying_results(year, round_number)
    except (Exception, SystemExit) as exc:
        notes.append(f"Echec recuperation qualifications: {exc}")
        qualy = pd.DataFrame()

    if qualy.empty:
        merged = fp_features.copy()
        merged["qualy_position"] = float("nan")
        merged["qualy_gap_to_best"] = float("nan")
        merged["grid_position"] = float("nan")
        merged["grid_source"] = "missing"
        notes.append("Resultats qualifications indisponibles: mode FP-only active pour prediction race.")
    else:
        qualy = qualy.copy()
        qualy["driver_id"] = qualy["driver_id"].astype(str)
        qualy["position"] = pd.to_numeric(qualy["position"], errors="coerce")
        qualy_merge_cols = ["driver_id", "position"]
        if "q3_time" in qualy.columns:
            qualy["q3_time"] = pd.to_numeric(qualy["q3_time"], errors="coerce")
            best_q3 = qualy["q3_time"].min(skipna=True)
            if pd.notna(best_q3):
                qualy["qualy_gap_to_best"] = qualy["q3_time"] - best_q3
                qualy_merge_cols.append("qualy_gap_to_best")
        merged = fp_features.merge(qualy[qualy_merge_cols], on="driver_id", how="inner")
        merged = merged.rename(columns={"position": "qualy_position"})
        merged["grid_position"] = pd.to_numeric(merged["qualy_position"], errors="coerce")
        merged["grid_source"] = "qualifying_fallback"
    try:
        starting_grid = provider.get_starting_grid(year, round_number)
    except (Exception, SystemExit) as exc:
        notes.append(f"Echec recuperation grille: {exc}")
        starting_grid = pd.DataFrame()
    if not starting_grid.empty and "grid_position" in starting_grid.columns:
        grid_cols = ["driver_id", "grid_position"]
        if "grid_source" in starting_grid.columns:
            grid_cols.append("grid_source")
        if "grid_status" in starting_grid.columns:
            grid_cols.append("grid_status")
        grid = starting_grid[grid_cols].copy()
        grid["driver_id"] = grid["driver_id"].astype(str)
        grid["grid_position"] = pd.to_numeric(grid["grid_position"], errors="coerce")
        merged = merged.drop(columns=["grid_position", "grid_source", "grid_status"], errors="ignore").merge(
            grid,
            on="driver_id",
            how="left",
        )
        grid_raw = pd.to_numeric(merged["grid_position"], errors="coerce")
        qualy_grid_fallback = pd.to_numeric(merged.get("qualy_position"), errors="coerce")
        if "grid_source" not in merged.columns:
            merged["grid_source"] = "pre_race_official_grid"
        merged["grid_source"] = merged["grid_source"].fillna("qualifying_fallback")
        merged.loc[grid_raw.notna() & merged["grid_source"].eq("qualifying_fallback"), "grid_source"] = (
            "pre_race_official_grid"
        )
        status = merged.get("grid_status", pd.Series("unknown", index=merged.index, dtype=object)).fillna("unknown")
        nonstarters = status.astype(str).str.lower().isin(["dns"])
        if nonstarters.any():
            notes.append(f"{int(nonstarters.sum())} pilote(s) DNS exclus des features race courantes.")
            merged = merged.loc[~nonstarters].copy()
            status = status.loc[merged.index]
            grid_raw = pd.to_numeric(merged["grid_position"], errors="coerce")
            qualy_grid_fallback = pd.to_numeric(merged.get("qualy_position"), errors="coerce")
        missing_grid = grid_raw.isna() & status.isin(["missing", "unknown", "non_numeric"])
        merged.loc[missing_grid, "grid_source"] = "qualifying_fallback"
        merged["grid_position"] = grid_raw.fillna(qualy_grid_fallback)
        merged.loc[merged["grid_position"].isna(), "grid_source"] = "missing"
    if "grid_status" not in merged.columns:
        merged["grid_status"] = "unknown"
    else:
        merged["grid_status"] = merged["grid_status"].fillna("unknown")
    if include_standings:
        try:
            standings = provider.get_standings(year, round_number)
        except (Exception, SystemExit) as exc:
            notes.append(f"Echec recuperation standings: {exc}")
            standings = None
        if standings is not None and not standings.empty:
            standings = standings.copy()
            standings["driver_id"] = standings["driver_id"].astype(str)
            merged = merged.merge(
                standings[["driver_id", "position_start"]],
                on="driver_id",
                how="left",
            )
    merged["event_name"] = event_name
    merged["event_year"] = year
    merged["event_round"] = round_number
    merged["event_key"] = (year * 100) + round_number
    merged = _attach_track_stats(
        frame=merged,
        provider=provider,
        year=year,
        round_number=round_number,
        notes=notes,
    )
    current = _attach_temporal_features_current(merged, history)
    return current, notes
