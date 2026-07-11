"""Shared helpers for data handling."""

from __future__ import annotations

import re
from typing import Iterable, Optional

import pandas as pd


def first_available(df: pd.DataFrame, columns: Iterable[str]) -> Optional[str]:
    for col in columns:
        if col in df.columns:
            return col
    return None


def normalize_event_name(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    return " ".join(text.split())


def team_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "team_id",
        "constructor_id",
        "team_name",
        "constructor_name",
        "constructor",
        "team",
    ]
    for col in candidates:
        if col not in df.columns:
            continue
        values = df[col]
        if values.notna().sum() == 0:
            continue
        text = values.astype(str).str.strip().str.lower()
        text = text.replace({"nan": "", "none": "", "<na>": ""})
        if (text != "").any():
            return col
    return None


def complete_classification_positions(
    frame: pd.DataFrame,
    position_column: str = "position",
) -> pd.DataFrame:
    """Assign stable tail ranks to unclassified result rows.

    Official result feeds commonly leave Position empty for unclassified,
    disqualified, or no-time entrants while retaining them in authoritative
    classification order. Ranking targets still require one total order, so
    append those rows after the largest numeric position without reordering
    the source classification.
    """

    if frame.empty or position_column not in frame.columns:
        return frame
    out = frame.copy()
    positions = pd.to_numeric(out[position_column], errors="coerce").astype(float)
    missing = positions.isna() | positions.le(0.0)
    maximum = positions.loc[~missing].max(skipna=True)
    next_position = int(maximum) if pd.notna(maximum) else 0
    for index in out.index[missing]:
        next_position += 1
        positions.at[index] = float(next_position)
    out[position_column] = positions
    out["classification_position_imputed_tail"] = missing.astype(bool)
    return out


def _slug_token(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "session"


def merge_fp_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    merged = None
    latest_roster_ids: set[str] = set()
    all_roster_ids: set[str] = set()
    roster_snapshots: list[dict[str, str]] = []
    for frame in frames:
        if frame is None or frame.empty or "driver_id" not in frame.columns:
            continue
        label = _slug_token(frame["session"].iloc[0] if "session" in frame.columns else "session")
        frame = frame.copy()
        frame["driver_id"] = frame["driver_id"].astype(str)
        session_roster_ids = set(frame.loc[frame["driver_id"].str.strip().ne(""), "driver_id"])
        if session_roster_ids:
            latest_roster_ids = session_roster_ids
            all_roster_ids.update(session_roster_ids)
            team_col = team_column(frame)
            snapshot: dict[str, str] = {}
            for _, roster_row in frame.loc[frame["driver_id"].isin(session_roster_ids)].iterrows():
                team = str(roster_row.get(team_col, "") if team_col else "").strip()
                if team.lower() in {"", "nan", "none", "<na>"}:
                    team = ""
                snapshot[str(roster_row["driver_id"])] = team
            roster_snapshots.append(snapshot)
        if "driver_name" not in frame.columns:
            frame["driver_name"] = frame["driver_id"]
        frame["driver_name"] = frame["driver_name"].fillna(frame["driver_id"]).astype(str)
        rename_map = {}
        for col in frame.columns:
            if col in {"driver_id", "driver_name", "session"}:
                continue
            rename_map[col] = f"{label}_{_slug_token(col)}"
        frame = frame.rename(columns=rename_map)
        if "session" in frame.columns:
            frame = frame.drop(columns=["session"])
        frame = frame.drop_duplicates(subset=["driver_id"], keep="last")
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="driver_id", how="outer", suffixes=("", "_r"))
            if "driver_name_r" in merged.columns:
                if "driver_name" not in merged.columns:
                    merged["driver_name"] = merged["driver_name_r"]
                else:
                    merged["driver_name"] = merged["driver_name"].fillna(merged["driver_name_r"])
                merged = merged.drop(columns=["driver_name_r"])
    if merged is None:
        return pd.DataFrame()

    active_roster_ids = set(latest_roster_ids)
    latest_snapshot = roster_snapshots[-1] if roster_snapshots else {}
    active_team_counts: dict[str, int] = {}
    for driver_id in active_roster_ids:
        team = latest_snapshot.get(driver_id, "")
        if team:
            active_team_counts[team] = active_team_counts.get(team, 0) + 1
    # A car may miss every lap in the latest session after an accident or
    # mechanical failure. Carry the most recent earlier participant only when
    # that constructor has an unfilled two-car seat; earlier reserve drivers
    # remain excluded once both current seats are represented.
    for snapshot in reversed(roster_snapshots[:-1]):
        for driver_id, team in snapshot.items():
            if driver_id in active_roster_ids or not team:
                continue
            if active_team_counts.get(team, 0) >= 2:
                continue
            active_roster_ids.add(driver_id)
            active_team_counts[team] = active_team_counts.get(team, 0) + 1

    if active_roster_ids:
        merged = merged.loc[merged["driver_id"].isin(active_roster_ids)].copy()
    merged["fp_roster_policy"] = "latest_session_plus_team_seat_continuity"
    merged["fp_roster_latest_session_driver_count"] = int(len(latest_roster_ids))
    merged["fp_roster_active_driver_count"] = int(len(active_roster_ids))
    merged["fp_roster_carried_forward_driver_count"] = int(len(active_roster_ids - latest_roster_ids))
    merged["fp_roster_superseded_driver_count"] = int(len(all_roster_ids - active_roster_ids))

    team_candidates = [
        c
        for c in merged.columns
        if c.endswith("_team_name") or c.endswith("_team") or c.endswith("_constructor_name")
    ]
    if team_candidates and "team_name" not in merged.columns:
        team_name = pd.Series(index=merged.index, dtype=object)
        for col in team_candidates:
            values = merged[col]
            team_name = team_name.where(team_name.notna(), values)
        if team_name.notna().any():
            merged["team_name"] = team_name

    delta_cols = [
        c
        for c in merged.columns
        if c.endswith("_delta")
        and not c.endswith("_top3_delta")
        and not c.endswith("_median_delta")
        and not c.endswith("_quali_sim_delta")
        and not c.endswith("_race_sim_delta")
        and not c.endswith("_wet_sim_delta")
    ]
    rank_cols = [
        c
        for c in merged.columns
        if c.endswith("_rank")
        and not c.endswith("_quali_sim_rank")
        and not c.endswith("_race_sim_rank")
        and not c.endswith("_wet_sim_rank")
    ]
    if delta_cols:
        delta_frame = merged[delta_cols].apply(pd.to_numeric, errors="coerce")
        merged["fp_mean_delta"] = delta_frame.mean(axis=1, skipna=True)
        merged["fp_delta_std"] = delta_frame.std(axis=1, skipna=True)
        merged["pace_sessions_available"] = delta_frame.notna().sum(axis=1)
    else:
        merged["fp_mean_delta"] = float("nan")
        merged["fp_delta_std"] = float("nan")
        merged["pace_sessions_available"] = 0
    if rank_cols:
        rank_frame = merged[rank_cols].apply(pd.to_numeric, errors="coerce")
        merged["fp_mean_rank"] = rank_frame.mean(axis=1, skipna=True)
    else:
        merged["fp_mean_rank"] = float("nan")

    top3_cols = [c for c in merged.columns if c.endswith("_top3_delta")]
    if top3_cols:
        merged["fp_mean_top3_delta"] = (
            merged[top3_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        )
    lap_std_cols = [c for c in merged.columns if c.endswith("_lap_std")]
    if lap_std_cols:
        merged["fp_mean_lap_std"] = (
            merged[lap_std_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        )
    quality_count_tokens = (
        "raw_timed_",
        "invalid_",
        "deleted_",
        "neutralised_",
        "pit_out_",
        "pit_in_",
        "representative_",
    )
    lap_count_cols = [
        c
        for c in merged.columns
        if c.endswith("_lap_count")
        and not c.endswith("_sim_lap_count")
        and not any(token in c for token in quality_count_tokens)
    ]
    if lap_count_cols:
        merged["fp_total_laps"] = (
            merged[lap_count_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)
        )

    quali_sim_delta_cols = [c for c in merged.columns if c.endswith("_quali_sim_delta")]
    if quali_sim_delta_cols:
        quali_sim_delta_frame = merged[quali_sim_delta_cols].apply(pd.to_numeric, errors="coerce")
        merged["fp_quali_sim_delta"] = quali_sim_delta_frame.mean(axis=1, skipna=True)
        merged["quali_sim_sessions_available"] = quali_sim_delta_frame.notna().sum(axis=1)
    else:
        merged["fp_quali_sim_delta"] = float("nan")
        merged["quali_sim_sessions_available"] = 0

    race_sim_delta_cols = [c for c in merged.columns if c.endswith("_race_sim_delta")]
    if race_sim_delta_cols:
        race_sim_delta_frame = merged[race_sim_delta_cols].apply(pd.to_numeric, errors="coerce")
        merged["fp_race_sim_delta"] = race_sim_delta_frame.mean(axis=1, skipna=True)
        merged["race_sim_sessions_available"] = race_sim_delta_frame.notna().sum(axis=1)
    else:
        merged["fp_race_sim_delta"] = float("nan")
        merged["race_sim_sessions_available"] = 0

    quali_sim_rank_cols = [c for c in merged.columns if c.endswith("_quali_sim_rank")]
    if quali_sim_rank_cols:
        merged["fp_quali_sim_rank"] = (
            merged[quali_sim_rank_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        )
    else:
        merged["fp_quali_sim_rank"] = float("nan")

    race_sim_rank_cols = [c for c in merged.columns if c.endswith("_race_sim_rank")]
    if race_sim_rank_cols:
        merged["fp_race_sim_rank"] = (
            merged[race_sim_rank_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        )
    else:
        merged["fp_race_sim_rank"] = float("nan")

    wet_sim_delta_cols = [c for c in merged.columns if c.endswith("_wet_sim_delta")]
    if wet_sim_delta_cols:
        wet_delta = merged[wet_sim_delta_cols].apply(pd.to_numeric, errors="coerce")
        merged["fp_wet_sim_delta"] = wet_delta.mean(axis=1, skipna=True)
        merged["wet_sim_sessions_available"] = wet_delta.notna().sum(axis=1)
    else:
        merged["fp_wet_sim_delta"] = float("nan")
        merged["wet_sim_sessions_available"] = 0

    wet_sim_rank_cols = [c for c in merged.columns if c.endswith("_wet_sim_rank")]
    if wet_sim_rank_cols:
        merged["fp_wet_sim_rank"] = merged[wet_sim_rank_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
    else:
        merged["fp_wet_sim_rank"] = float("nan")

    quali_sim_lap_count_cols = [c for c in merged.columns if c.endswith("_quali_sim_lap_count")]
    if quali_sim_lap_count_cols:
        merged["fp_quali_sim_laps"] = (
            merged[quali_sim_lap_count_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)
        )
    else:
        merged["fp_quali_sim_laps"] = 0.0

    race_sim_lap_count_cols = [c for c in merged.columns if c.endswith("_race_sim_lap_count")]
    if race_sim_lap_count_cols:
        merged["fp_race_sim_laps"] = (
            merged[race_sim_lap_count_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)
        )
    else:
        merged["fp_race_sim_laps"] = 0.0

    wet_sim_lap_count_cols = [c for c in merged.columns if c.endswith("_wet_sim_lap_count")]
    if wet_sim_lap_count_cols:
        merged["fp_wet_sim_laps"] = merged[wet_sim_lap_count_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)
    else:
        merged["fp_wet_sim_laps"] = 0.0

    def _sum_session_metric(metric_suffix: str, output_column: str) -> None:
        columns = [c for c in merged.columns if c.endswith(metric_suffix)]
        if columns:
            merged[output_column] = (
                merged[columns].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)
            )
        else:
            merged[output_column] = 0.0

    _sum_session_metric("_raw_timed_lap_count", "fp_raw_timed_laps")
    _sum_session_metric("_invalid_lap_count", "fp_invalid_laps")
    _sum_session_metric("_deleted_lap_count", "fp_deleted_laps")
    _sum_session_metric("_neutralised_lap_count", "fp_neutralised_laps")
    _sum_session_metric("_pit_out_lap_count", "fp_pit_out_laps")
    _sum_session_metric("_pit_in_lap_count", "fp_pit_in_laps")
    _sum_session_metric("_representative_lap_count", "fp_representative_laps")
    raw_laps = pd.to_numeric(merged["fp_raw_timed_laps"], errors="coerce")
    representative_laps = pd.to_numeric(merged["fp_representative_laps"], errors="coerce")
    merged["fp_lap_quality_ratio"] = representative_laps.div(raw_laps.where(raw_laps > 0.0))

    for metric_suffix, output_column in (
        ("_quali_sim_evidence_share", "fp_quali_sim_evidence_share"),
        ("_race_sim_evidence_share", "fp_race_sim_evidence_share"),
        ("_run_intent_unclassified_share", "fp_run_intent_unclassified_share"),
    ):
        columns = [c for c in merged.columns if c.endswith(metric_suffix)]
        if columns:
            merged[output_column] = (
                merged[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
            )
        else:
            merged[output_column] = float("nan")

    degradation_columns = [
        c for c in merged.columns if c.endswith("_race_sim_raw_degradation_sec_per_lap")
    ]
    if degradation_columns:
        degradation = merged[degradation_columns].apply(pd.to_numeric, errors="coerce")
        merged["fp_race_sim_raw_degradation_sec_per_lap"] = degradation.mean(axis=1, skipna=True)
        merged["fp_race_sim_degradation_sessions_available"] = degradation.notna().sum(axis=1)
    else:
        merged["fp_race_sim_raw_degradation_sec_per_lap"] = float("nan")
        merged["fp_race_sim_degradation_sessions_available"] = 0

    degradation_mad_columns = [
        c for c in merged.columns if c.endswith("_race_sim_raw_degradation_mad")
    ]
    if degradation_mad_columns:
        merged["fp_race_sim_raw_degradation_mad"] = (
            merged[degradation_mad_columns]
            .apply(pd.to_numeric, errors="coerce")
            .mean(axis=1, skipna=True)
        )
    else:
        merged["fp_race_sim_raw_degradation_mad"] = float("nan")

    slow_lap_ratio_cols = [c for c in merged.columns if c.endswith("_slow_lap_ratio")]
    if slow_lap_ratio_cols:
        merged["fp_slow_lap_ratio"] = (
            merged[slow_lap_ratio_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        )
    else:
        merged["fp_slow_lap_ratio"] = float("nan")

    quali_vs_race_gap_cols = [c for c in merged.columns if c.endswith("_quali_vs_race_gap")]
    if quali_vs_race_gap_cols:
        merged["fp_quali_vs_race_gap"] = (
            merged[quali_vs_race_gap_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        )
    else:
        merged["fp_quali_vs_race_gap"] = float("nan")
    return merged


def format_prediction_table(df: pd.DataFrame, top_n: Optional[int] = 10) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy().sort_values("pred", ascending=True)
    if top_n is not None:
        df = df.head(max(0, int(top_n)))
    df["rank"] = range(1, len(df) + 1)
    cols = [
        "rank",
        "driver_name",
        "driver_id",
        "pred",
        "prediction_scenario",
        "weather_scenario",
        "weather_uncertainty_level",
        "proba_win",
        "proba_top3",
        "proba_top10",
        "grid_position",
        "grid_source",
        "grid_status",
        "qualy_pred_position",
        "qualy_pred_rank",
        "qualy_pred_rank_pct",
        "track_weather_uncertainty",
        "track_weather_uncertainty_prior",
        "race_stochastic_score",
        "race_stochastic_pl_score",
        "race_stochastic_sigma",
        "race_stochastic_dnf_probability",
        "race_stochastic_layer",
    ]
    ordered = [col for col in cols if col in df.columns]

    listwise_cols = [
        "utility",
        "p_win",
        "p_top3",
        "p_top10",
        "exp_pos",
        "pos_p10",
        "pos_p50",
        "pos_p90",
        "listwise_method",
        "listwise_samples",
        "temperature",
        "listwise_enabled",
        "old_rank_based_win",
        "old_rank_based_top10",
        "old_rank_based_top3",
    ]
    for col in listwise_cols:
        if col in df.columns:
            ordered.append(col)
    return df[ordered]
