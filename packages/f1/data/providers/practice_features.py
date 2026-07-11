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


FP_FEATURE_CONTRACT_VERSION = "f1_practice_lap_features_v3_quality_weighted"


# FastF1 track-status digits: 1=all clear, 2=yellow, 4=Safety Car,
# 5=red flag, 6=VSC deployed, 7=VSC ending.  Status strings can contain
# multiple digits, so any neutralisation digit makes a lap unsuitable for a
# clean pace/degradation observation.  We retain the count as quality context.
NEUTRALISED_TRACK_STATUS_CODES = frozenset({"2", "4", "5", "6", "7"})


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


def _track_status_codes(value: object) -> set[str]:
    if value is None:
        return set()
    try:
        if pd.isna(value):
            return set()
    except Exception:
        pass
    return {character for character in str(value).strip() if character.isdigit()}


def _neutralised_track_status(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: bool(_track_status_codes(value).intersection(NEUTRALISED_TRACK_STATUS_CODES)),
    ).astype(bool)


def _robust_pairwise_slope(
    x: pd.Series,
    y: pd.Series,
) -> tuple[float, float, int]:
    """Return a Theil-Sen-style slope and pairwise MAD uncertainty.

    The value is deliberately labelled as a *raw* stint slope.  It still
    contains fuel-burn, traffic and track-evolution effects and must not be
    interpreted as pure tyre degradation.
    """

    clean = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        },
    ).dropna()
    clean = clean.sort_values("x", kind="mergesort").drop_duplicates(subset=["x"], keep="last")
    if len(clean) < 3:
        return float("nan"), float("nan"), 0
    x_values = clean["x"].to_numpy(dtype=float)
    y_values = clean["y"].to_numpy(dtype=float)
    slopes: list[float] = []
    for left in range(len(clean) - 1):
        dx = x_values[left + 1 :] - x_values[left]
        valid = np.abs(dx) > 1e-12
        if not valid.any():
            continue
        dy = y_values[left + 1 :] - y_values[left]
        slopes.extend((dy[valid] / dx[valid]).astype(float).tolist())
    if not slopes:
        return float("nan"), float("nan"), 0
    slope_values = np.asarray(slopes, dtype=float)
    median = float(np.median(slope_values))
    mad = float(np.median(np.abs(slope_values - median)))
    return median, mad, int(len(slope_values))


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
    if work.empty:
        return pd.DataFrame()

    accurate_col = first_available(work, ["IsAccurate", "is_accurate"])
    accurate = (
        _truthy(work[accurate_col], default=True)
        if accurate_col
        else pd.Series(True, index=work.index, dtype=bool)
    )
    deleted_col = first_available(work, ["Deleted", "deleted"])
    deleted = (
        _truthy(work[deleted_col], default=False)
        if deleted_col
        else pd.Series(False, index=work.index, dtype=bool)
    )
    if pit_out_col:
        pit_out = (
            _truthy(work[pit_out_col], default=False)
            if str(pit_out_col).lower().startswith("is_") or pd.api.types.is_bool_dtype(work[pit_out_col])
            else work[pit_out_col].notna()
        )
    else:
        pit_out = pd.Series(False, index=work.index, dtype=bool)
    if pit_in_col:
        pit_in = (
            _truthy(work[pit_in_col], default=False)
            if str(pit_in_col).lower().startswith("is_") or pd.api.types.is_bool_dtype(work[pit_in_col])
            else work[pit_in_col].notna()
        )
    else:
        pit_in = pd.Series(False, index=work.index, dtype=bool)

    track_status_col = first_available(work, ["TrackStatus", "track_status"])
    neutralised = (
        _neutralised_track_status(work[track_status_col])
        if track_status_col
        else pd.Series(False, index=work.index, dtype=bool)
    )
    work["_quality_accurate"] = accurate
    work["_quality_deleted"] = deleted
    work["_quality_pit_out"] = pit_out
    work["_quality_pit_in"] = pit_in
    work["_quality_neutralised"] = neutralised
    work["_quality_clean"] = accurate & ~deleted & ~pit_out & ~pit_in & ~neutralised

    quality_rows: dict[str, dict[str, float]] = {}
    identity_rows: dict[str, dict[str, str]] = {}
    for driver, quality_group in work.groupby("driver_id", sort=False):
        raw_count = int(len(quality_group))
        clean_count = int(quality_group["_quality_clean"].sum())
        identity_rows[str(driver)] = {
            "driver_name": _mode_or_first(quality_group["driver_name"], str(driver)),
            "team_name": _mode_or_first(quality_group["team_name"], ""),
        }
        quality_rows[str(driver)] = {
            "raw_timed_lap_count": raw_count,
            "invalid_lap_count": int((~quality_group["_quality_accurate"]).sum()),
            "deleted_lap_count": int(quality_group["_quality_deleted"].sum()),
            "neutralised_lap_count": int(quality_group["_quality_neutralised"].sum()),
            "pit_out_lap_count": int(quality_group["_quality_pit_out"].sum()),
            "pit_in_lap_count": int(quality_group["_quality_pit_in"].sum()),
            "representative_lap_count": clean_count,
            "lap_quality_ratio": float(clean_count / raw_count) if raw_count else 0.0,
        }

    work = work[work["_quality_clean"]].copy()
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
        threshold_epsilon = 1e-9
        slow_mask = lap_delta > (float(cfg.slow_lap_delta_seconds) + threshold_epsilon)
        usable = (~slow_mask) & driver_lap_time.notna()
        qualifying_mask = (
            usable
            & (lap_delta <= (float(cfg.qualifying_sim_delta_seconds) + threshold_epsilon))
            & (stint_sizes <= 3)
            & (
                (tyre_life <= float(cfg.qualifying_sim_tyre_life_max))
                | fresh_tyre
                | tyre_life.isna()
            )
        )
        qualifying_laps = driver_lap_time[qualifying_mask].dropna().sort_values()

        race_mask = (
            usable
            & (stint_sizes >= int(cfg.race_sim_min_stint_laps))
            & (lap_delta >= (float(cfg.race_sim_min_delta_seconds) - threshold_epsilon))
            & (lap_delta <= (float(cfg.race_sim_max_delta_seconds) + threshold_epsilon))
        )
        race_laps = driver_lap_time[race_mask].dropna()

        degradation_slopes: list[tuple[float, float, int]] = []
        for _, stint_group in group.loc[race_mask].groupby("_stint_id", sort=False):
            if len(stint_group) < 3:
                continue
            stint_tyre_life = (
                pd.to_numeric(stint_group[resolved_tyre_life_col], errors="coerce")
                if resolved_tyre_life_col
                else pd.to_numeric(stint_group["lap_order"], errors="coerce")
            )
            slope, slope_mad, pair_count = _robust_pairwise_slope(
                stint_tyre_life,
                pd.to_numeric(stint_group["lap_time"], errors="coerce"),
            )
            if np.isfinite(slope):
                degradation_slopes.append((slope, slope_mad, pair_count))
        if degradation_slopes:
            pair_weights = np.asarray([max(1, item[2]) for item in degradation_slopes], dtype=float)
            raw_degradation_slope = float(
                np.average(np.asarray([item[0] for item in degradation_slopes], dtype=float), weights=pair_weights),
            )
            raw_degradation_mad = float(
                np.average(np.asarray([item[1] for item in degradation_slopes], dtype=float), weights=pair_weights),
            )
            degradation_stint_count = int(len(degradation_slopes))
        else:
            raw_degradation_slope = float("nan")
            raw_degradation_mad = float("nan")
            degradation_stint_count = 0

        qualifying_lap = float(qualifying_laps.mean()) if not qualifying_laps.empty else float("nan")
        race_lap = float(race_laps.mean()) if not race_laps.empty else float("nan")
        resolved_compound_col = compound_col if compound_col in group.columns else ("compound" if "compound" in group.columns else None)
        if resolved_compound_col:
            wet_mask = group[resolved_compound_col].astype(str).str.strip().str.upper().isin({"INTERMEDIATE", "WET"})
            wet_laps = driver_lap_time[usable & wet_mask].dropna()
        else:
            wet_laps = pd.Series(dtype=float)
        wet_lap = float(wet_laps.mean()) if not wet_laps.empty else float("nan")
        usable_count = int(usable.sum())
        quality = quality_rows.get(str(driver), {})
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
                "quali_sim_evidence_share": (
                    float(len(qualifying_laps) / usable_count) if usable_count else 0.0
                ),
                "race_sim_lap": race_lap,
                "race_sim_lap_count": int(len(race_laps)),
                "race_sim_evidence_share": float(len(race_laps) / usable_count) if usable_count else 0.0,
                "run_intent_unclassified_share": (
                    float((usable & ~qualifying_mask & ~race_mask).sum() / usable_count)
                    if usable_count
                    else 0.0
                ),
                "race_sim_raw_degradation_sec_per_lap": raw_degradation_slope,
                "race_sim_raw_degradation_mad": raw_degradation_mad,
                "race_sim_degradation_stint_count": degradation_stint_count,
                "race_sim_degradation_is_fuel_corrected": False,
                "wet_sim_lap": wet_lap,
                "wet_sim_lap_count": int(len(wet_laps)),
                "quali_vs_race_gap": race_lap - qualifying_lap if np.isfinite(race_lap) and np.isfinite(qualifying_lap) else float("nan"),
                **quality,
            }
        )

    represented_drivers = {str(row["driver_id"]) for row in rows}
    for driver, quality in quality_rows.items():
        if driver in represented_drivers:
            continue
        identity = identity_rows.get(driver, {})
        rows.append(
            {
                "driver_id": driver,
                "driver_name": identity.get("driver_name", driver),
                "team_name": identity.get("team_name", ""),
                "lap_count": 0,
                "slow_lap_ratio": float("nan"),
                "quali_sim_lap_count": 0,
                "quali_sim_evidence_share": 0.0,
                "race_sim_lap_count": 0,
                "race_sim_evidence_share": 0.0,
                "run_intent_unclassified_share": 0.0,
                "race_sim_degradation_stint_count": 0,
                "race_sim_degradation_is_fuel_corrected": False,
                "wet_sim_lap_count": 0,
                **quality,
            },
        )

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    for column in (
        "best_lap",
        "top3_lap",
        "median_lap",
        "lap_std",
        "quali_sim_lap",
        "race_sim_lap",
        "race_sim_raw_degradation_sec_per_lap",
        "race_sim_raw_degradation_mad",
        "wet_sim_lap",
        "quali_vs_race_gap",
    ):
        if column not in frame.columns:
            frame[column] = float("nan")
    frame["delta"] = frame["best_lap"] - frame["best_lap"].min()
    frame["rank"] = frame["best_lap"].rank(method="min").astype("Int64")
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
    frame["run_intent_labels_calibrated"] = False
    return frame


__all__ = [
    "FP_FEATURE_CONTRACT_VERSION",
    "PracticeFeatureConfig",
    "build_session_pace_features",
    "normalize_driver_id",
]
