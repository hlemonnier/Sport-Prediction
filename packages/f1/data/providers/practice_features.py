"""Provider-neutral practice-lap feature contract.

All concrete providers normalize their raw lap timing into this module before
model features are assembled.  This prevents ``source=local|fastf1|openf1``
from silently changing the meaning of FP pace, qualifying-simulation, and
race-simulation columns.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

import numpy as np
import pandas as pd

from packages.f1.data.utils import first_available


FP_FEATURE_CONTRACT_VERSION = "f1_practice_lap_features_v2"


@dataclass(frozen=True)
class PracticeFeatureConfig:
    slow_lap_delta_seconds: float = 5.0
    qualifying_sim_delta_seconds: float = 1.4
    qualifying_sim_tyre_life_max: float = 3.0
    race_sim_min_stint_laps: int = 5
    race_sim_min_delta_seconds: float = 0.8
    race_sim_max_delta_seconds: float = 5.5


def normalize_driver_id(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return str(int(float(text)))
    return text


def _mode_or_first(values: pd.Series, fallback: str) -> str:
    clean = values.dropna().astype(str).str.strip()
    clean = clean[~clean.str.lower().isin({"", "nan", "none", "<na>"})]
    if clean.empty:
        return fallback
    mode = clean.mode(dropna=True)
    return str(mode.iloc[0] if not mode.empty else clean.iloc[0])


def _seconds(values: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(values):
        return values.dt.total_seconds()
    numeric = pd.to_numeric(values, errors="coerce")
    # Some parquet/dataframe conversions expose timedelta nanoseconds as ints.
    finite = numeric[np.isfinite(numeric)]
    if not finite.empty and float(finite.median()) > 1_000_000.0:
        numeric = numeric / 1_000_000_000.0
    return numeric.astype(float)


def _truthy(values: pd.Series, *, default: bool = False) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    out = normalized.isin({"1", "true", "yes", "y", "t"})
    missing = values.isna() | normalized.isin({"", "nan", "none", "<na>"})
    return out.where(~missing, default).astype(bool)


def _stint_metadata_from_ranges(laps: pd.DataFrame, stints: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Attach OpenF1-style stint ranges to normalized lap rows."""

    if stints is None or stints.empty:
        return laps
    out = laps.copy()
    for column in ("stint", "tyre_life", "compound"):
        if column not in out.columns:
            out[column] = pd.NA
    for _, stint in stints.iterrows():
        driver = normalize_driver_id(stint.get("driver_number", stint.get("driver_id")))
        start = pd.to_numeric(pd.Series([stint.get("lap_start")]), errors="coerce").iloc[0]
        end = pd.to_numeric(pd.Series([stint.get("lap_end")]), errors="coerce").iloc[0]
        if not driver or pd.isna(start) or pd.isna(end):
            continue
        mask = (
            out["driver_id"].eq(driver)
            & out["lap_order"].between(float(start), float(end), inclusive="both")
        )
        if not mask.any():
            continue
        stint_number = stint.get("stint_number")
        tyre_age_start = pd.to_numeric(pd.Series([stint.get("tyre_age_at_start")]), errors="coerce").iloc[0]
        out.loc[mask, "stint"] = stint_number
        if pd.notna(tyre_age_start):
            out.loc[mask, "tyre_life"] = (
                float(tyre_age_start)
                + pd.to_numeric(out.loc[mask, "lap_order"], errors="coerce")
                - float(start)
            )
        out.loc[mask, "compound"] = stint.get("compound")
    return out


def build_session_pace_features(
    laps: pd.DataFrame,
    label: str,
    *,
    provider: str,
    stints: Optional[pd.DataFrame] = None,
    config: PracticeFeatureConfig | None = None,
) -> pd.DataFrame:
    """Build the canonical per-driver feature row for one practice session."""

    cfg = config or PracticeFeatureConfig()
    if laps is None or laps.empty:
        return pd.DataFrame()
    driver_col = first_available(laps, ["DriverNumber", "driver_number", "Driver", "driver_id"])
    lap_col = first_available(laps, ["LapTime", "lap_time", "lap_duration", "duration"])
    if driver_col is None or lap_col is None:
        return pd.DataFrame()

    work = laps.copy()
    work["driver_id"] = work[driver_col].map(normalize_driver_id)
    work = work[work["driver_id"] != ""].copy()
    if work.empty:
        return pd.DataFrame()

    name_col = first_available(work, ["Driver", "Abbreviation", "BroadcastName", "driver_name", "name_acronym"])
    team_col = first_available(work, ["Team", "TeamName", "team_name"])
    lap_number_col = first_available(work, ["LapNumber", "lap_number"])
    stint_col = first_available(work, ["Stint", "stint", "stint_number"])
    tyre_life_col = first_available(work, ["TyreLife", "tyre_life"])
    fresh_tyre_col = first_available(work, ["FreshTyre", "fresh_tyre"])
    compound_col = first_available(work, ["Compound", "compound"])
    pit_out_col = first_available(work, ["is_pit_out_lap", "PitOutTime", "pit_out_time", "pit_out"])
    pit_in_col = first_available(work, ["is_pit_in_lap", "PitInTime", "pit_in_time", "pit_in"])

    work["driver_name"] = work[name_col].astype(str) if name_col else work["driver_id"]
    work["team_name"] = work[team_col] if team_col else pd.NA
    work["lap_time"] = _seconds(work[lap_col])
    work["lap_order"] = (
        pd.to_numeric(work[lap_number_col], errors="coerce")
        if lap_number_col
        else pd.Series(range(1, len(work) + 1), index=work.index, dtype=float)
    )
    work = work[work["lap_time"].notna() & (work["lap_time"] > 0.0) & work["lap_order"].notna()].copy()

    accurate_col = first_available(work, ["IsAccurate", "is_accurate"])
    if accurate_col:
        accurate = _truthy(work[accurate_col], default=True)
        work = work[accurate].copy()
    deleted_col = first_available(work, ["Deleted", "deleted"])
    if deleted_col:
        work = work[~_truthy(work[deleted_col], default=False)].copy()
    if pit_out_col:
        pit_out = (
            _truthy(work[pit_out_col], default=False)
            if str(pit_out_col).lower().startswith("is_") or pd.api.types.is_bool_dtype(work[pit_out_col])
            else work[pit_out_col].notna()
        )
        work = work[~pit_out].copy()
    if pit_in_col:
        pit_in = (
            _truthy(work[pit_in_col], default=False)
            if str(pit_in_col).lower().startswith("is_") or pd.api.types.is_bool_dtype(work[pit_in_col])
            else work[pit_in_col].notna()
        )
        work = work[~pit_in].copy()
    if work.empty:
        return pd.DataFrame()

    work = _stint_metadata_from_ranges(work, stints)
    rows: list[dict[str, object]] = []
    for driver, group in work.groupby("driver_id", sort=False):
        group = group.sort_values("lap_order", kind="mergesort").copy()
        lap_times = pd.to_numeric(group["lap_time"], errors="coerce").dropna().sort_values()
        if lap_times.empty:
            continue
        best_lap = float(lap_times.iloc[0])
        top3_lap = float(lap_times.iloc[: min(3, len(lap_times))].mean())
        median_lap = float(lap_times.median())
        lap_std = float(lap_times.std(ddof=0)) if len(lap_times) > 1 else 0.0

        if stint_col and stint_col in group.columns:
            stint_id = pd.to_numeric(group[stint_col], errors="coerce")
        elif "stint" in group.columns:
            stint_id = pd.to_numeric(group["stint"], errors="coerce")
        else:
            stint_id = pd.Series(1.0, index=group.index, dtype=float)
        if stint_id.notna().sum() == 0:
            stint_id = pd.Series(1.0, index=group.index, dtype=float)
        group["_stint_id"] = stint_id.ffill().bfill().fillna(1).astype(int)
        stint_sizes = group.groupby("_stint_id", sort=False)["lap_time"].transform("size").astype(float)

        resolved_tyre_life_col = tyre_life_col if tyre_life_col in group.columns else ("tyre_life" if "tyre_life" in group.columns else None)
        tyre_life = (
            pd.to_numeric(group[resolved_tyre_life_col], errors="coerce")
            if resolved_tyre_life_col
            else pd.Series(float("nan"), index=group.index, dtype=float)
        )
        fresh_tyre = (
            _truthy(group[fresh_tyre_col], default=False)
            if fresh_tyre_col and fresh_tyre_col in group.columns
            else pd.Series(False, index=group.index, dtype=bool)
        )

        driver_lap_time = pd.to_numeric(group["lap_time"], errors="coerce")
        lap_delta = driver_lap_time - best_lap
        slow_mask = lap_delta > float(cfg.slow_lap_delta_seconds)
        usable = (~slow_mask) & driver_lap_time.notna()
        qualifying_mask = (
            usable
            & (lap_delta <= float(cfg.qualifying_sim_delta_seconds))
            & (
                (tyre_life <= float(cfg.qualifying_sim_tyre_life_max))
                | fresh_tyre
                | (stint_sizes <= 3)
            )
        )
        qualifying_laps = driver_lap_time[qualifying_mask].dropna().sort_values()
        if qualifying_laps.empty:
            qualifying_laps = driver_lap_time[usable].dropna().sort_values().head(2)

        race_mask = (
            usable
            & (stint_sizes >= int(cfg.race_sim_min_stint_laps))
            & (lap_delta >= float(cfg.race_sim_min_delta_seconds))
            & (lap_delta <= float(cfg.race_sim_max_delta_seconds))
        )
        race_laps = driver_lap_time[race_mask].dropna()
        if race_laps.empty:
            race_laps = driver_lap_time[usable & (stint_sizes >= int(cfg.race_sim_min_stint_laps))].dropna()
        if race_laps.empty:
            fallback = driver_lap_time[usable].dropna().sort_values()
            race_laps = fallback.iloc[2:] if len(fallback) > 4 else fallback

        qualifying_lap = float(qualifying_laps.mean()) if not qualifying_laps.empty else float("nan")
        race_lap = float(race_laps.mean()) if not race_laps.empty else float("nan")
        resolved_compound_col = compound_col if compound_col in group.columns else ("compound" if "compound" in group.columns else None)
        if resolved_compound_col:
            wet_mask = group[resolved_compound_col].astype(str).str.strip().str.upper().isin({"INTERMEDIATE", "WET"})
            wet_laps = driver_lap_time[usable & wet_mask].dropna()
        else:
            wet_laps = pd.Series(dtype=float)
        wet_lap = float(wet_laps.mean()) if not wet_laps.empty else float("nan")
        rows.append(
            {
                "driver_id": str(driver),
                "driver_name": _mode_or_first(group["driver_name"], str(driver)),
                "team_name": _mode_or_first(group["team_name"], ""),
                "best_lap": best_lap,
                "top3_lap": top3_lap,
                "median_lap": median_lap,
                "lap_std": lap_std,
                "lap_count": int(len(lap_times)),
                "slow_lap_ratio": float(slow_mask.mean()),
                "quali_sim_lap": qualifying_lap,
                "quali_sim_lap_count": int(len(qualifying_laps)),
                "race_sim_lap": race_lap,
                "race_sim_lap_count": int(len(race_laps)),
                "wet_sim_lap": wet_lap,
                "wet_sim_lap_count": int(len(wet_laps)),
                "quali_vs_race_gap": race_lap - qualifying_lap if np.isfinite(race_lap) and np.isfinite(qualifying_lap) else float("nan"),
            }
        )

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["delta"] = frame["best_lap"] - frame["best_lap"].min()
    frame["rank"] = frame["best_lap"].rank(method="min").astype(int)
    frame["top3_delta"] = frame["top3_lap"] - frame["top3_lap"].min()
    frame["median_delta"] = frame["median_lap"] - frame["median_lap"].min()
    for prefix in ("quali", "race"):
        lap_column = f"{prefix}_sim_lap"
        if frame[lap_column].notna().any():
            frame[f"{prefix}_sim_delta"] = frame[lap_column] - frame[lap_column].min(skipna=True)
            frame[f"{prefix}_sim_rank"] = frame[lap_column].rank(method="min")
        else:
            frame[f"{prefix}_sim_delta"] = float("nan")
            frame[f"{prefix}_sim_rank"] = float("nan")
    if frame["wet_sim_lap"].notna().sum() >= 2:
        frame["wet_sim_delta"] = frame["wet_sim_lap"] - frame["wet_sim_lap"].min(skipna=True)
        frame["wet_sim_rank"] = frame["wet_sim_lap"].rank(method="min")
    else:
        frame["wet_sim_delta"] = float("nan")
        frame["wet_sim_rank"] = float("nan")
    frame["session"] = str(label)
    frame["feature_contract_version"] = FP_FEATURE_CONTRACT_VERSION
    frame["feature_source"] = str(provider)
    return frame


__all__ = [
    "FP_FEATURE_CONTRACT_VERSION",
    "PracticeFeatureConfig",
    "build_session_pace_features",
    "normalize_driver_id",
]
