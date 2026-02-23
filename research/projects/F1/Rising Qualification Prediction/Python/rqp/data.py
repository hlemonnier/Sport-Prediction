"""Data preparation and feature assembly."""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from .providers import BaseProvider
from .utils import normalize_event_name, team_column


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
    getter = getattr(provider, "get_track_stats", None)
    if getter is None:
        return _add_track_interactions(frame)
    try:
        stats = getter(year, round_number)
    except (Exception, SystemExit) as exc:
        notes.append(f"Echec stats circuit {year} round {round_number}: {exc}")
        return _add_track_interactions(frame)
    if not isinstance(stats, dict) or not stats:
        return _add_track_interactions(frame)
    out = frame.copy()
    for key, value in stats.items():
        if value is None:
            continue
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            continue
        out[key] = float(numeric)
    out = _add_track_interactions(out)
    return out


def _add_track_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    track_cols = [
        "track_overtake_propensity",
        "track_grid_stability",
        "track_safety_car_propensity",
        "track_sc_lap_ratio",
        "track_vsc_lap_ratio",
        "track_dnf_rate",
        "track_pit_stop_intensity",
        "track_stats_reliability",
        "track_chaos_index",
    ]
    for col in track_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    overtake = (
        out["track_overtake_propensity"]
        if "track_overtake_propensity" in out.columns
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

    if "track_chaos_index" not in out.columns:
        out["track_chaos_index"] = (0.45 * overtake.fillna(0.5)) + (0.35 * safety.fillna(0.2)) + (0.20 * dnf.fillna(0.1))
    out["track_chaos_index"] = out["track_chaos_index"].clip(lower=0.0, upper=1.0)

    out["track_qualy_importance"] = (1.0 - (0.65 * overtake.fillna(0.5)) - (0.35 * safety.fillna(0.2))).clip(
        lower=0.0,
        upper=1.0,
    )

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
            1.0 + (0.75 * overtake.fillna(0.5)) + (0.35 * safety.fillna(0.2))
        )
    else:
        out["fp_race_sim_delta_track_adj"] = float("nan")

    if "fp_weighted_delta" in out.columns:
        fp_weighted = pd.to_numeric(out["fp_weighted_delta"], errors="coerce")
        out["fp_weighted_delta_track_adj"] = fp_weighted * (
            1.0 + (0.45 * overtake.fillna(0.5)) + (0.20 * safety.fillna(0.2))
        )
    else:
        out["fp_weighted_delta_track_adj"] = float("nan")

    if "qualy_pred_position" in out.columns:
        qualy_pred = pd.to_numeric(out["qualy_pred_position"], errors="coerce")
        out["qualy_pred_position_track_adj"] = qualy_pred * (0.35 + out["track_qualy_importance"])
    else:
        out["qualy_pred_position_track_adj"] = float("nan")

    return out


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


def _add_temporal_features_train(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = _ensure_fp_mean_delta(frame)
    if "driver_id" in out.columns:
        out["driver_id"] = out["driver_id"].astype(str)
    out["event_name"] = out.get("event_name", pd.Series(index=out.index, dtype=object))
    out["event_name_norm"] = out["event_name"].map(normalize_event_name)

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
        team_group = out.groupby(out[team_col], sort=False)["fp_mean_delta"]
        out["team_ewma_fp_mean_delta"] = team_group.transform(
            lambda s: s.shift(1).ewm(alpha=0.5, adjust=False, min_periods=1).mean(),
        )
        out["team_form_3_fp_mean_delta"] = team_group.transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).mean(),
        )
        out["team_form_5_fp_mean_delta"] = team_group.transform(
            lambda s: s.shift(1).rolling(window=5, min_periods=1).mean(),
        )
    else:
        out["team_ewma_fp_mean_delta"] = float("nan")
        out["team_form_3_fp_mean_delta"] = float("nan")
        out["team_form_5_fp_mean_delta"] = float("nan")

    if team_col:
        weighted_team_group = out.groupby(out[team_col], sort=False)["fp_weighted_delta"]
        out["team_ewma_fp_weighted_delta"] = weighted_team_group.transform(
            lambda s: s.shift(1).ewm(alpha=0.5, adjust=False, min_periods=1).mean(),
        )
        out["team_form_3_fp_weighted_delta"] = weighted_team_group.transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).mean(),
        )
    else:
        out["team_ewma_fp_weighted_delta"] = float("nan")
        out["team_form_3_fp_weighted_delta"] = float("nan")

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

    temporal_cols = [
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
    ]
    for col in temporal_cols:
        out[col] = float("nan")

    if history is None or history.empty or "driver_id" not in history.columns:
        return _add_event_relative_features(out)

    hist = _add_temporal_features_train(history)
    hist = hist.copy()
    hist["driver_id"] = hist["driver_id"].astype(str)
    sort_cols = [c for c in ["event_key", "event_year", "event_round"] if c in hist.columns]
    if sort_cols:
        hist = hist.sort_values(sort_cols, kind="mergesort")

    driver_last = hist.groupby("driver_id", sort=False).tail(1).set_index("driver_id")
    for col in [
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
    ]:
        if col in driver_last.columns:
            out[col] = out["driver_id"].map(driver_last[col])

    hist_team_col = team_column(hist)
    current_team_col = team_column(out)
    if hist_team_col and current_team_col:
        hist[hist_team_col] = (
            hist[hist_team_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"nan": pd.NA, "none": pd.NA, "<na>": pd.NA, "": pd.NA})
        )
        out[current_team_col] = (
            out[current_team_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"nan": pd.NA, "none": pd.NA, "<na>": pd.NA, "": pd.NA})
        )
        team_last = hist.groupby(hist_team_col, sort=False).tail(1).set_index(hist_team_col)
        for col in [
            "team_ewma_fp_mean_delta",
            "team_form_3_fp_mean_delta",
            "team_form_5_fp_mean_delta",
            "team_ewma_fp_weighted_delta",
            "team_form_3_fp_weighted_delta",
        ]:
            if col in team_last.columns:
                out[col] = out[current_team_col].map(team_last[col])

    hist["event_name_norm"] = hist.get("event_name", pd.Series(index=hist.index, dtype=object)).map(
        normalize_event_name,
    )
    track_driver = hist.groupby(["event_name_norm", "driver_id"], sort=False)["fp_mean_delta"].mean()
    keys = list(zip(out["event_name_norm"], out["driver_id"]))
    out["event_driver_hist_idx"] = [track_driver.get(key, float("nan")) for key in keys]
    driver_mean = hist.groupby("driver_id", sort=False)["fp_mean_delta"].mean()
    out["event_driver_hist_idx"] = out["event_driver_hist_idx"].fillna(out["driver_id"].map(driver_mean))

    out = _add_event_relative_features(out)
    return out


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
                merged = merged.merge(race[["driver_id", "position"]], on="driver_id", how="inner")
                merged = merged.rename(columns={"position": "target"})
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
        return pd.DataFrame(), notes
    fp_features = fp_features.copy()
    fp_features["driver_id"] = fp_features["driver_id"].astype(str)
    event_name = _event_name_for_round(provider, year, round_number, notes)
    if mode == "qualifying":
        fp_features["event_name"] = event_name
        fp_features["event_year"] = year
        fp_features["event_round"] = round_number
        fp_features["event_key"] = (year * 100) + round_number
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
