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
    "race_wet_probability",
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
    "race_degradation_score",
    "race_longest_clean_stint_score",
    "race_long_run_evidence_score",
    "race_long_run_uncertainty_score",
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
    "race_wet_probability": (
        "wet_probability",
        "race_wet_probability_prior",
        "rain_probability",
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
    ),
    "race_fuel_track_adjusted_degradation": (
        "fuel_track_adjusted_degradation",
        "fp_degradation_track_adj",
        "fp_deg_slope",
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
    out["race_degradation_score"] = -out["race_fuel_track_adjusted_degradation"]
    out["race_longest_clean_stint_score"] = out["race_longest_clean_stint_laps"]
    out["race_long_run_evidence_score"] = out["race_long_run_evidence_share"]
    out["race_long_run_uncertainty_score"] = -out["race_long_run_uncertainty"]
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

    for column in (
        *RACE_SURVIVAL_FEATURE_COLUMNS,
        *RACE_ORDER_FEATURE_COLUMNS,
        *RACE_DISTANCE_FEATURE_COLUMNS,
    ):
        if column not in out.columns:
            out[column] = np.nan
        out[f"{column}_missing"] = pd.to_numeric(out[column], errors="coerce").isna()
    return out


def available_race_context(frame: pd.DataFrame) -> list[str]:
    """Return race-context columns present in a feature frame."""

    return [column for column in RACE_CONTEXT_COLUMNS if column in frame.columns]


def has_grid_signal(frame: pd.DataFrame) -> bool:
    """Check whether the race model has a usable grid column."""

    return "grid_position" in frame.columns and frame["grid_position"].notna().any()
