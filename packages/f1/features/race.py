"""Causal feature contracts for survival-aware pre-Race prediction.

The canonical columns below are deliberately small and inspectable.  Aliases
are accepted at the boundary, but missing evidence stays missing and is paired
with an explicit ``*_missing`` flag instead of being silently backfilled from
post-Race results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from packages.f1.domain.starting_grid import RaceGridSnapshot


RACE_CONTEXT_COLUMNS: tuple[str, ...] = (
    "grid_position",
    "grid_status",
    "fp_race_sim_delta",
    "fp_race_sim_delta_track_adj",
    "track_grid_stability",
    "track_chaos_index",
    "track_finish_order_mobility",
    "race_generation_variance_prior",
)

RACE_SURVIVAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "race_team_mechanical_rate",
    "race_power_unit_mechanical_rate",
    "race_driver_incident_rate",
    "race_weekend_stoppage_count",
    "race_missed_practice_share",
    "race_circuit_dnf_rate",
    "race_safety_car_probability",
    "race_wet_probability",
    "race_weather_uncertainty",
    "race_current_weekend_mechanical_stop_share",
    "race_power_unit_grid_penalty",
    "race_starter_eligible",
    "race_pit_lane_start",
)

RACE_ORDER_FEATURE_COLUMNS: tuple[str, ...] = (
    "race_grid_prior_score",
    "race_grid_mobility_score",
    "race_signed_qualifying_surprise_score",
    "race_team_strength_score",
    "race_driver_strength_score",
    "race_teammate_long_run_score",
    "race_long_run_pace_score",
    "race_compound_pace_score",
    "race_tyre_age_pace_score",
    "race_degradation_score",
    "race_longest_clean_stint_score",
    "race_long_run_evidence_score",
    "race_long_run_uncertainty_score",
    "race_order_evidence_reliability_score",
    "race_sprint_pace_score",
)

RACE_DISTANCE_FEATURE_COLUMNS: tuple[str, ...] = (
    "race_expected_lap_deficit",
    "race_expected_lap_seconds",
)


_ALIASES: dict[str, tuple[str, ...]] = {
    "race_team_mechanical_rate": (
        "team_mechanical_rate",
        "team_dnf_rate",
        "constructor_dnf_rate",
    ),
    "race_power_unit_mechanical_rate": (
        "power_unit_mechanical_rate",
        "pu_mechanical_rate",
        "engine_dnf_rate",
    ),
    "race_driver_incident_rate": (
        "driver_incident_rate",
        "driver_collision_rate",
        "driver_dnf_rate",
    ),
    "race_weekend_stoppage_count": (
        "weekend_stoppage_count",
        "current_weekend_stoppages",
        "fp_stoppage_count",
    ),
    "race_missed_practice_share": (
        "missed_practice_share",
        "fp_missed_session_share",
    ),
    "race_circuit_dnf_rate": ("track_dnf_rate", "track_dnf_prior"),
    "race_safety_car_probability": (
        "track_safety_car_propensity",
        "safety_car_probability",
        "track_sc_probability",
    ),
    "race_wet_probability": (
        "wet_probability",
        "race_wet_probability_prior",
        "rain_probability",
    ),
    "race_weather_uncertainty": (
        "track_weather_uncertainty",
        "track_weather_uncertainty_prior",
        "weather_uncertainty",
    ),
    "race_current_weekend_mechanical_stop_share": (
        "current_weekend_mechanical_stop_share",
        "weekend_mechanical_stop_share",
    ),
    "race_power_unit_grid_penalty": (
        "power_unit_grid_penalty",
        "pu_grid_penalty",
    ),
    "race_team_strength_score": (
        "team_strength_score",
        "constructor_strength",
        "team_rolling_strength",
    ),
    "race_driver_strength_score": (
        "driver_strength_score",
        "driver_rolling_strength",
    ),
    "race_teammate_long_run_delta": (
        "teammate_long_run_delta",
        "fp_teammate_long_run_delta",
    ),
    "race_long_run_pace_delta": (
        "long_run_pace_delta",
        "fp_race_sim_delta_track_adj",
        "fp_race_sim_delta",
    ),
    "race_compound_pace_delta": (
        "compound_pace_delta",
        "fp_compound_pace_delta",
        "fp_race_compound_pace_delta",
    ),
    "race_tyre_age_pace_delta": (
        "tyre_age_pace_delta",
        "fp_tyre_age_pace_delta",
        "fp_race_tyre_age_pace_delta",
    ),
    "race_fuel_track_adjusted_degradation": (
        "fuel_track_adjusted_degradation",
        "fp_degradation_track_adj",
        "fp_race_fuel_track_adjusted_degradation",
    ),
    "race_longest_clean_stint_laps": (
        "longest_clean_stint_laps",
        "fp_longest_clean_stint",
        "fp_long_run_laps",
    ),
    "race_long_run_evidence_share": (
        "long_run_evidence_share",
        "fp_race_sim_evidence_share",
    ),
    "race_long_run_uncertainty": (
        "long_run_uncertainty",
        "fp_race_sim_uncertainty",
        "fp_delta_std",
    ),
    "race_circuit_mobility": (
        "track_finish_order_mobility",
        "circuit_mobility",
    ),
    "race_sprint_pace_delta": (
        "sprint_pace_delta",
        "sprint_race_pace_delta",
    ),
    "race_expected_lap_deficit": (
        "expected_lap_deficit",
        "classified_lap_deficit_prior",
    ),
    "race_expected_lap_seconds": (
        "expected_race_lap_seconds",
        "track_reference_lap_seconds",
    ),
}


def _numeric_alias(frame: pd.DataFrame, canonical: str) -> pd.Series:
    """Select one declared causal alias; never synthesize post-cutoff values."""

    candidates = (canonical, *_ALIASES.get(canonical, ()))
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in candidates:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        out = out.where(out.notna(), values)
    return out


def _numeric_column(frame: pd.DataFrame, *columns: str) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in columns:
        if column in frame.columns:
            out = out.where(out.notna(), pd.to_numeric(frame[column], errors="coerce"))
    return out


def race_grid_snapshot_frame(
    snapshot: RaceGridSnapshot,
    *,
    require_available: bool = True,
) -> pd.DataFrame:
    """Convert immutable grid evidence to model rows without losing provenance."""

    if require_available:
        snapshot.require_available()
    rows = []
    for entry in snapshot.entries:
        rows.append(
            {
                "driver_id": entry.driver_id,
                "grid_position": entry.grid_position,
                "grid_status": entry.status.value,
                "grid_starter_eligible": entry.starter_eligible,
                "grid_pit_lane_start": entry.pit_lane_start,
                "grid_evidence_complete": entry.evidence_complete,
                "grid_evidence_ids": "|".join(entry.evidence_ids),
                "grid_penalty_evidence_ids": "|".join(
                    adjustment.evidence_id for adjustment in entry.penalty_evidence
                ),
                "grid_penalty_kinds": "|".join(
                    adjustment.kind.value for adjustment in entry.penalty_evidence
                ),
                "grid_penalty_reasons": "|".join(
                    adjustment.reason or "" for adjustment in entry.penalty_evidence
                ),
                "race_information_horizon": snapshot.horizon.value,
                "grid_prediction_as_of": snapshot.prediction_as_of,
                "grid_publication_as_of": snapshot.publication_as_of,
                "grid_resolution_status": snapshot.resolution_status.value,
                "grid_snapshot_available": snapshot.available,
            }
        )
    return pd.DataFrame(rows)


def engineer_survival_aware_race_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create canonical terminal-hazard and conditional-order interfaces.

    Positive order scores always mean *better expected running order*.  In
    particular, qualifying surprise is signed: a driver qualifying three places
    better than the causal prior receives ``+3`` rather than an absolute error.
    """

    out = frame.copy()
    for canonical in _ALIASES:
        out[canonical] = _numeric_alias(out, canonical)

    grid = _numeric_column(out, "grid_position")
    qualifying = _numeric_column(out, "qualy_position", "qualifying_position")
    causal_qualifying_prior = _numeric_column(
        out,
        "qualy_pred_rank",
        "qualy_pred_position",
        "expected_qualifying_position",
    )
    out["race_signed_qualifying_surprise"] = qualifying - causal_qualifying_prior
    out["race_signed_grid_vs_qualifying"] = grid - qualifying

    if "event_key" in out.columns:
        field_size = out.groupby("event_key", dropna=False)["driver_id"].transform("size")
    else:
        field_size = pd.Series(max(1, len(out)), index=out.index, dtype=float)
    field_size = pd.to_numeric(field_size, errors="coerce").clip(lower=1.0)
    # A pit-lane starter has no physical grid box and receives the weakest grid
    # prior only after starter eligibility is explicitly known.
    status = out.get("grid_status", pd.Series("", index=out.index)).astype(str).str.lower()
    pit_lane = out.get("grid_pit_lane_start")
    if pit_lane is None:
        pit_lane_flag = status.isin({"pit_lane", "started_pit_lane"})
    else:
        pit_lane_flag = pd.Series(pit_lane, index=out.index).fillna(False).astype(bool)
    grid_for_score = grid.where(grid.notna(), np.where(pit_lane_flag, field_size + 1.0, np.nan))
    out["race_grid_prior_score"] = -(grid_for_score - 1.0) / field_size

    mobility = out["race_circuit_mobility"].clip(lower=0.0, upper=1.0)
    out["race_grid_mobility_score"] = out["race_grid_prior_score"] * (1.0 - mobility)
    out["race_signed_qualifying_surprise_score"] = -out[
        "race_signed_qualifying_surprise"
    ]
    out["race_teammate_long_run_score"] = -out["race_teammate_long_run_delta"]
    out["race_long_run_pace_score"] = -out["race_long_run_pace_delta"]
    out["race_compound_pace_score"] = -out["race_compound_pace_delta"]
    out["race_tyre_age_pace_score"] = -out["race_tyre_age_pace_delta"]
    out["race_degradation_score"] = -out["race_fuel_track_adjusted_degradation"]
    out["race_longest_clean_stint_score"] = out["race_longest_clean_stint_laps"]
    out["race_long_run_evidence_score"] = out["race_long_run_evidence_share"]
    out["race_long_run_uncertainty_score"] = -out["race_long_run_uncertainty"]
    out["race_order_evidence_reliability_score"] = (
        out["race_long_run_evidence_share"].clip(lower=0.0, upper=1.0)
        / (1.0 + out["race_long_run_uncertainty"].clip(lower=0.0))
    )
    out["race_sprint_pace_score"] = -out["race_sprint_pace_delta"]

    starter = out.get("grid_starter_eligible")
    if starter is None:
        known_starter = status.isin({"grid", "pit_lane", "started", "started_pit_lane"})
        known_nonstarter = status.isin({"withdrawn", "did_not_start", "disqualified"})
        starter_values = pd.Series(np.nan, index=out.index, dtype=float)
        starter_values.loc[known_starter] = 1.0
        starter_values.loc[known_nonstarter] = 0.0
    else:
        starter_values = pd.to_numeric(starter, errors="coerce").clip(0.0, 1.0)
    out["race_starter_eligible"] = starter_values
    out["race_pit_lane_start"] = pit_lane_flag.astype(float)
    if "grid_penalty_reasons" in out.columns:
        penalty_text = out["grid_penalty_reasons"].fillna("").astype(str).str.lower()
        inferred_pu_penalty = penalty_text.str.contains(
            r"power.?unit|engine|gearbox|energy store|control electronics",
            regex=True,
        ).astype(float)
        out["race_power_unit_grid_penalty"] = out[
            "race_power_unit_grid_penalty"
        ].where(out["race_power_unit_grid_penalty"].notna(), inferred_pu_penalty)

    for column in (
        *RACE_SURVIVAL_FEATURE_COLUMNS,
        *RACE_ORDER_FEATURE_COLUMNS,
        *RACE_DISTANCE_FEATURE_COLUMNS,
    ):
        if column not in out.columns:
            out[column] = np.nan
        out[f"{column}_missing"] = pd.to_numeric(out[column], errors="coerce").isna()
    return out


def _lap_seconds(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() >= max(1, int(values.notna().sum() * 0.8)):
        return numeric.astype(float)
    timed = pd.to_timedelta(values, errors="coerce").dt.total_seconds()
    return numeric.where(numeric.notna(), timed).astype(float)


def _truthy(values: pd.Series, *, default: bool) -> pd.Series:
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
    }
    encoded = values.astype("string").str.strip().str.lower().map(mapping)
    return encoded.fillna(default).astype(bool)


def _first_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def _normalize_driver(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(numeric) and float(numeric).is_integer():
        return str(int(numeric))
    return text


def _attach_openf1_stints(laps: pd.DataFrame, stints: pd.DataFrame | None) -> pd.DataFrame:
    if stints is None or stints.empty:
        return laps
    out = laps.copy()
    lap_driver = _first_column(out, ("driver_number", "DriverNumber", "driver_id"))
    lap_number = _first_column(out, ("lap_number", "LapNumber"))
    stint_driver = _first_column(stints, ("driver_number", "DriverNumber", "driver_id"))
    lap_start = _first_column(stints, ("lap_start", "LapStart", "start_lap"))
    lap_end = _first_column(stints, ("lap_end", "LapEnd", "end_lap"))
    if None in {lap_driver, lap_number, stint_driver, lap_start, lap_end}:
        return out
    out_lap = pd.to_numeric(out[str(lap_number)], errors="coerce")
    out_driver = out[str(lap_driver)].map(_normalize_driver)
    for stint_index, stint in stints.iterrows():
        driver = _normalize_driver(stint.get(str(stint_driver)))
        start = pd.to_numeric(pd.Series([stint.get(str(lap_start))]), errors="coerce").iloc[0]
        end = pd.to_numeric(pd.Series([stint.get(str(lap_end))]), errors="coerce").iloc[0]
        if not driver or pd.isna(start) or pd.isna(end):
            continue
        mask = out_driver.eq(driver) & out_lap.between(float(start), float(end), inclusive="both")
        if not mask.any():
            continue
        if "stint" not in out.columns:
            out["stint"] = np.nan
        out.loc[mask, "stint"] = int(stint.get("stint_number") or stint_index + 1)
        for source, target in (
            ("compound", "compound"),
            ("tyre_age_at_start", "_tyre_age_at_start"),
        ):
            if source in stints.columns:
                if target not in out.columns:
                    out[target] = np.nan if "age" in target else pd.NA
                out.loc[mask, target] = stint.get(source)
        if "tyre_life" not in out.columns and "_tyre_age_at_start" in out.columns:
            out.loc[mask, "tyre_life"] = (
                out_lap.loc[mask] - float(start)
                + pd.to_numeric(
                    pd.Series([stint.get("tyre_age_at_start")]), errors="coerce"
                ).fillna(0.0).iloc[0]
            )
    return out


def derive_race_practice_evidence(
    laps: pd.DataFrame,
    *,
    session_label: str,
    stints: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Extract causal compound, stint, and relative-degradation evidence.

    Degradation is estimated from lap-time residuals after subtracting the
    same-lap field median.  That removes the shared fuel-burn and track-state
    trend instead of pretending the raw within-stint slope is pure tyre wear.
    The output remains relative degradation and carries explicit evidence and
    uncertainty fields.
    """

    if laps.empty:
        return pd.DataFrame()
    work = _attach_openf1_stints(laps, stints)
    driver_col = _first_column(
        work,
        ("DriverNumber", "driver_number", "driver_id", "Abbreviation", "Driver"),
    )
    lap_time_col = _first_column(work, ("LapTime", "lap_duration", "lap_time"))
    lap_number_col = _first_column(work, ("LapNumber", "lap_number"))
    if driver_col is None or lap_time_col is None:
        return pd.DataFrame()
    work = work.copy()
    work["driver_id"] = work[driver_col].map(_normalize_driver)
    work["_lap_seconds"] = _lap_seconds(work[lap_time_col])
    work["_lap_number"] = (
        pd.to_numeric(work[lap_number_col], errors="coerce")
        if lap_number_col is not None
        else pd.Series(np.arange(1, len(work) + 1), index=work.index, dtype=float)
    )
    accurate_col = _first_column(work, ("IsAccurate", "is_accurate"))
    deleted_col = _first_column(work, ("Deleted", "deleted"))
    accurate = (
        _truthy(work[accurate_col], default=True)
        if accurate_col is not None
        else pd.Series(True, index=work.index)
    )
    deleted = (
        _truthy(work[deleted_col], default=False)
        if deleted_col is not None
        else pd.Series(False, index=work.index)
    )
    pit_in_col = _first_column(work, ("PitInTime", "pit_in_time", "is_pit_in_lap"))
    pit_out_col = _first_column(work, ("PitOutTime", "pit_out_time", "is_pit_out_lap"))
    pit_in = (
        work[pit_in_col].notna()
        if pit_in_col is not None and not str(pit_in_col).startswith("is_")
        else _truthy(work[pit_in_col], default=False)
        if pit_in_col is not None
        else pd.Series(False, index=work.index)
    )
    pit_out = (
        work[pit_out_col].notna()
        if pit_out_col is not None and not str(pit_out_col).startswith("is_")
        else _truthy(work[pit_out_col], default=False)
        if pit_out_col is not None
        else pd.Series(False, index=work.index)
    )
    track_col = _first_column(work, ("TrackStatus", "track_status"))
    neutralised = (
        work[track_col].astype("string").str.contains(r"[4567]", regex=True, na=False)
        if track_col is not None
        else pd.Series(False, index=work.index)
    )
    clean = (
        work["driver_id"].ne("")
        & work["_lap_seconds"].between(30.0, 240.0, inclusive="both")
        & work["_lap_number"].notna()
        & accurate
        & ~deleted
        & ~pit_in
        & ~pit_out
        & ~neutralised
    )
    work = work.loc[clean].copy()
    if work.empty:
        return pd.DataFrame()

    stint_col = _first_column(work, ("Stint", "stint", "stint_number"))
    if stint_col is None:
        work["_stint"] = 1
    else:
        work["_stint"] = (
            pd.to_numeric(work[stint_col], errors="coerce")
            .groupby(work["driver_id"], sort=False)
            .ffill()
            .fillna(1)
            .astype(int)
        )
    tyre_life_col = _first_column(work, ("TyreLife", "tyre_life"))
    if tyre_life_col is None:
        work["_tyre_life"] = work.groupby(
            ["driver_id", "_stint"], sort=False
        ).cumcount().astype(float)
    else:
        fallback_age = work.groupby(["driver_id", "_stint"], sort=False).cumcount()
        work["_tyre_life"] = pd.to_numeric(
            work[tyre_life_col], errors="coerce"
        ).fillna(fallback_age).astype(float)
    compound_col = _first_column(work, ("Compound", "compound"))
    work["_compound"] = (
        work[compound_col].astype("string").str.strip().str.upper().fillna("UNKNOWN")
        if compound_col is not None
        else "UNKNOWN"
    )

    # Lap-matched field subtraction is causal inside the completed practice
    # session and removes common track evolution, weather and fuel-burn trend.
    lap_field = work.groupby("_lap_number", dropna=False)["_lap_seconds"].transform("median")
    session_field = float(work["_lap_seconds"].median())
    work["_track_residual"] = work["_lap_seconds"] - lap_field.fillna(session_field)
    compound_lap_field = work.groupby(
        ["_lap_number", "_compound"], dropna=False
    )["_lap_seconds"].transform("median")
    compound_field = work.groupby("_compound", dropna=False)["_lap_seconds"].transform("median")
    work["_compound_residual"] = work["_lap_seconds"] - compound_lap_field.fillna(
        compound_field
    )
    stint_size = work.groupby(["driver_id", "_stint"], sort=False)[
        "_lap_seconds"
    ].transform("size")
    driver_best = work.groupby("driver_id", sort=False)["_lap_seconds"].transform("min")
    representative = stint_size.ge(4) & work["_lap_seconds"].le(driver_best + 6.0)
    work["_representative"] = representative

    rows: list[dict[str, object]] = []
    for driver_id, driver in work.groupby("driver_id", sort=False):
        representative_driver = driver.loc[driver["_representative"]].copy()
        if representative_driver.empty:
            representative_driver = driver.copy()
        slopes: list[float] = []
        longest = 0
        for _, stint in representative_driver.groupby("_stint", sort=False):
            longest = max(longest, len(stint))
            if len(stint) < 4:
                continue
            x = stint["_tyre_life"].to_numpy(dtype=float)
            y = stint["_track_residual"].to_numpy(dtype=float)
            deltas: list[float] = []
            for left in range(len(stint)):
                for right in range(left + 1, len(stint)):
                    dx = x[right] - x[left]
                    if abs(dx) > 1e-9:
                        deltas.append(float((y[right] - y[left]) / dx))
            if deltas:
                slopes.append(float(np.median(deltas)))
        residual = representative_driver["_compound_residual"].astype(float)
        degradation = float(np.median(slopes)) if slopes else float("nan")
        uncertainty = (
            float(np.median(np.abs(residual - residual.median())))
            if len(residual)
            else float("nan")
        )
        rows.append(
            {
                "driver_id": str(driver_id),
                "session": str(session_label),
                "race_compound_pace_delta": float(residual.median()),
                "race_tyre_age_pace_delta": degradation,
                "race_fuel_track_adjusted_degradation": degradation,
                "race_longest_clean_stint_laps": int(longest),
                "race_practice_evidence_count": int(len(representative_driver)),
                "race_practice_evidence_share": float(
                    len(representative_driver) / max(1, len(driver))
                ),
                "race_practice_uncertainty": uncertainty,
                "race_degradation_stint_count": int(len(slopes)),
                "race_degradation_adjustment_method": (
                    "lap_matched_field_residual_common_fuel_track_removed"
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def aggregate_race_practice_evidence(
    frame: pd.DataFrame,
    *,
    expected_sessions: int,
) -> pd.DataFrame:
    """Aggregate prefixed session evidence without fabricating missing values."""

    out = frame.copy()

    def columns(suffix: str) -> list[str]:
        return [column for column in out.columns if column.endswith(suffix)]

    for suffix, output, reduction in (
        ("_race_compound_pace_delta", "race_compound_pace_delta", "median"),
        ("_race_tyre_age_pace_delta", "race_tyre_age_pace_delta", "median"),
        (
            "_race_fuel_track_adjusted_degradation",
            "race_fuel_track_adjusted_degradation",
            "median",
        ),
        ("_race_longest_clean_stint_laps", "race_longest_clean_stint_laps", "max"),
        ("_race_practice_evidence_count", "race_practice_evidence_count", "sum"),
        ("_race_practice_evidence_share", "race_practice_evidence_share", "mean"),
        ("_race_practice_uncertainty", "race_practice_uncertainty", "median"),
        ("_race_degradation_stint_count", "race_degradation_stint_count", "sum"),
    ):
        source_columns = columns(suffix)
        if not source_columns:
            out[output] = np.nan
            continue
        values = out[source_columns].apply(pd.to_numeric, errors="coerce")
        if reduction == "max":
            out[output] = values.max(axis=1, skipna=True)
        elif reduction == "sum":
            out[output] = values.sum(axis=1, min_count=1)
        elif reduction == "mean":
            out[output] = values.mean(axis=1, skipna=True)
        else:
            out[output] = values.median(axis=1, skipna=True)

    presence_columns = columns("_raw_timed_lap_count")
    if presence_columns and int(expected_sessions) > 0:
        present = out[presence_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).gt(0.0)
        out["race_missed_practice_share"] = (
            1.0 - present.sum(axis=1) / float(expected_sessions)
        ).clip(lower=0.0, upper=1.0)
    else:
        out["race_missed_practice_share"] = np.nan
    evidence_share = pd.to_numeric(out["race_practice_evidence_share"], errors="coerce")
    uncertainty = pd.to_numeric(out["race_practice_uncertainty"], errors="coerce")
    out["race_long_run_evidence_share"] = evidence_share
    out["race_long_run_uncertainty"] = uncertainty
    out["race_order_evidence_reliability"] = evidence_share.clip(0.0, 1.0) / (
        1.0 + uncertainty.clip(lower=0.0)
    )
    return out


def available_race_context(frame: pd.DataFrame) -> list[str]:
    """Return race-context columns present in a feature frame."""

    return [column for column in RACE_CONTEXT_COLUMNS if column in frame.columns]


def has_grid_signal(frame: pd.DataFrame) -> bool:
    """Check whether the race model has a usable grid column."""

    return "grid_position" in frame.columns and frame["grid_position"].notna().any()
