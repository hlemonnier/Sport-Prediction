"""Causal, quality-aware rehearsal evidence for F1 qualifying pace.

The module deliberately keeps *valid observations* separate from evidence of
potential.  A deleted lap or a compatible-sector reconstruction can inform a
latent pace estimate, but it is never relabelled as a valid classified lap.

The returned frame is shared by Best Estimated Lap and Qualifying Prediction.
It contains one row per requested entrant, including entrants with no usable
lap, and records the fallback and uncertainty used to construct the anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DRIVER_COLUMNS: tuple[str, ...] = ("driver_id", "Driver", "driver", "DriverNumber")
TEAM_COLUMNS: tuple[str, ...] = ("team_id", "Team", "TeamName", "team")
EVENT_COLUMNS: tuple[str, ...] = ("event_key", "meeting_key", "event_id")
SESSION_COLUMNS: tuple[str, ...] = ("rehearsal_source", "session", "SessionName", "Session")
LAP_TIME_COLUMNS: tuple[str, ...] = (
    "lap_time_seconds",
    "LapTime",
    "lap_duration",
    "lap_time",
)
SECTOR_COLUMNS: tuple[tuple[str, ...], ...] = (
    ("sector1_seconds", "Sector1Time", "duration_sector_1"),
    ("sector2_seconds", "Sector2Time", "duration_sector_2"),
    ("sector3_seconds", "Sector3Time", "duration_sector_3"),
)
TIME_COLUMNS: tuple[str, ...] = ("lap_timestamp", "Date", "Time", "timestamp")


@dataclass(frozen=True)
class QualityAwareLapConfig:
    """Controls for causal anchor construction.

    Potential evidence receives an additive penalty because a deleted lap or
    stitched sector floor is not a guaranteed repeatable classified lap.
    The values are intentionally explicit and can be validated by event block.
    """

    min_lap_seconds: float = 40.0
    max_lap_seconds: float = 180.0
    potential_penalty_seconds: float = 0.25
    official_uncertainty_seconds: float = 0.30
    valid_uncertainty_seconds: float = 0.45
    potential_uncertainty_seconds: float = 0.90
    earlier_session_uncertainty_seconds: float = 1.25
    team_prior_uncertainty_seconds: float = 1.75
    push_lap_relative_threshold: float = 0.07
    sector_compatibility_fields: tuple[str, ...] = ("session", "compound")


def finite_lap_seconds(
    values: pd.Series | Sequence[Any] | Any,
    *,
    min_seconds: float = 40.0,
    max_seconds: float = 180.0,
) -> pd.Series:
    """Convert timing-like values to finite seconds without dropping scalars.

    A one-value sample is valid evidence.  In particular, this cleaner never
    requires variance, two observations, or a non-zero median absolute
    deviation before returning the value.
    """

    if isinstance(values, pd.Series):
        raw = values.copy()
    elif np.isscalar(values) or values is None:
        raw = pd.Series([values])
    else:
        raw = pd.Series(values)
    numeric = pd.to_numeric(raw, errors="coerce")
    unresolved = numeric.isna()
    if unresolved.any():
        timedeltas = pd.to_timedelta(raw.where(unresolved), errors="coerce")
        numeric = numeric.fillna(timedeltas.dt.total_seconds())
    numeric = pd.to_numeric(numeric, errors="coerce").astype(float)
    return numeric.where(
        np.isfinite(numeric) & numeric.between(float(min_seconds), float(max_seconds))
    )


def build_quality_aware_rehearsal_features(
    laps: pd.DataFrame,
    *,
    entrants: pd.DataFrame | Sequence[str] | None = None,
    earlier_laps: pd.DataFrame | None = None,
    team_priors: pd.DataFrame | Mapping[str, float] | None = None,
    as_of: Any | None = None,
    config: QualityAwareLapConfig | None = None,
) -> pd.DataFrame:
    """Return one causal, provenance-rich rehearsal row per entrant.

    ``laps`` is the target-aligned rehearsal snapshot (normally FP3 or Sprint
    Qualifying). ``earlier_laps`` is optional and is consulted only after
    official, valid, and potential target-aligned evidence are absent.
    ``entrants`` should be supplied for production inference so missing-timing
    entrants are preserved rather than silently removed.
    """

    if not isinstance(laps, pd.DataFrame):
        raise TypeError("laps must be a pandas DataFrame")
    if earlier_laps is not None and not isinstance(earlier_laps, pd.DataFrame):
        raise TypeError("earlier_laps must be a pandas DataFrame when supplied")
    cfg = config or QualityAwareLapConfig()
    current = _prepare_laps(laps, as_of=as_of, config=cfg)
    earlier = (
        _prepare_laps(earlier_laps, as_of=as_of, config=cfg)
        if earlier_laps is not None and not earlier_laps.empty
        else pd.DataFrame()
    )
    roster = _entrant_roster(entrants, current=current, earlier=earlier)
    if roster.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    current_features = _aggregate_session(current, cfg)
    if current_features.empty:
        current_features = pd.DataFrame(columns=["driver_id", "team_id"])
    earlier_features = _aggregate_session(earlier, cfg) if not earlier.empty else pd.DataFrame()
    prior_by_team = _team_prior_mapping(team_priors)

    result = roster.merge(current_features, how="left", on=["driver_id", "team_id"])
    result = _annotate_potential_credibility(result)
    if not earlier_features.empty:
        earlier_best = (
            earlier_features.sort_values(
                ["driver_id", "session_priority", "valid_clean_best_seconds"],
                na_position="last",
            )
            .groupby("driver_id", sort=False, as_index=False)
            .first()
        )
        earlier_best = earlier_best[
            ["driver_id", "valid_clean_best_seconds", "rehearsal_source"]
        ].rename(
            columns={
                "valid_clean_best_seconds": "earlier_session_best_seconds",
                "rehearsal_source": "earlier_session_source",
            }
        )
        result = result.merge(earlier_best, how="left", on="driver_id")
    else:
        result["earlier_session_best_seconds"] = np.nan
        result["earlier_session_source"] = None

    # First pass establishes all non-team-prior anchors.  The second pass uses
    # leave-driver-out teammate evidence for entrants still without an anchor.
    chosen = result.apply(lambda row: _choose_non_team_anchor(row, cfg), axis=1)
    chosen_frame = pd.DataFrame(chosen.tolist(), index=result.index)
    result = pd.concat([result, chosen_frame], axis=1)
    result = _add_latent_potential_adjusted_anchor(result, cfg)
    teammate_anchors: dict[str, float] = {}
    for team_id, rows in result.groupby("team_id", dropna=False, sort=False):
        # Teammate fallback consumes the accepted latent estimate, not a known
        # unrepresentative valid lap that the potential selector already fixed.
        values = pd.to_numeric(
            rows["latent_potential_adjusted_anchor_seconds"], errors="coerce"
        )
        finite = values[np.isfinite(values)]
        if len(finite):
            teammate_anchors[str(team_id)] = float(finite.median())

    missing = result["quality_aware_anchor_seconds"].isna()
    for index in result.index[missing]:
        team_id = str(result.at[index, "team_id"])
        if team_id in teammate_anchors:
            result.at[index, "quality_aware_anchor_seconds"] = teammate_anchors[team_id]
            result.at[index, "anchor_source"] = "teammate_partial_pool"
            result.at[index, "anchor_quality"] = "team_prior_widest_uncertainty"
            result.at[index, "anchor_uncertainty_seconds"] = cfg.team_prior_uncertainty_seconds
            result.at[index, "anchor_is_imputed"] = True
        elif team_id in prior_by_team:
            result.at[index, "quality_aware_anchor_seconds"] = prior_by_team[team_id]
            result.at[index, "anchor_source"] = "team_historical_prior"
            result.at[index, "anchor_quality"] = "team_prior_widest_uncertainty"
            result.at[index, "anchor_uncertainty_seconds"] = cfg.team_prior_uncertainty_seconds
            result.at[index, "anchor_is_imputed"] = True
        else:
            result.at[index, "anchor_source"] = "unavailable"
            result.at[index, "anchor_quality"] = "unavailable_no_causal_evidence"
            result.at[index, "anchor_is_imputed"] = True

    result = _add_latent_potential_adjusted_anchor(result, cfg)
    result["teammate_relative_anchor_seconds"] = result.apply(
        lambda row: _relative_to_group_median(
            row,
            result,
            group_column="team_id",
            value_column="quality_aware_anchor_seconds",
        ),
        axis=1,
    )
    field_median = pd.to_numeric(
        result["quality_aware_anchor_seconds"], errors="coerce"
    ).median(skipna=True)
    result["field_relative_anchor_seconds"] = (
        pd.to_numeric(result["quality_aware_anchor_seconds"], errors="coerce")
        - float(field_median)
        if np.isfinite(field_median)
        else np.nan
    )
    evidence_fields = [
        "official_classified_rehearsal_best_seconds",
        "valid_clean_best_seconds",
        "deleted_potential_best_seconds",
        "compatible_sector_potential_seconds",
        "earlier_session_best_seconds",
    ]
    for column in evidence_fields:
        if column not in result.columns:
            result[column] = np.nan
    result["evidence_item_count"] = result[evidence_fields].notna().sum(axis=1).astype(int)
    result["evidence_coverage_rate"] = result[evidence_fields].notna().mean(axis=1)
    result["feature_as_of"] = _as_of_label(as_of)
    result["feature_contract"] = "quality_aware_rehearsal_lap_v1"
    return result.reindex(columns=_OUTPUT_COLUMNS)


def _prepare_laps(
    laps: pd.DataFrame,
    *,
    as_of: Any | None,
    config: QualityAwareLapConfig,
) -> pd.DataFrame:
    if laps.empty:
        return pd.DataFrame()
    frame = laps.copy()
    driver_column = _first_existing(frame, DRIVER_COLUMNS)
    if driver_column is None:
        raise ValueError("lap evidence is missing a driver identifier")
    frame["driver_id"] = frame[driver_column].astype(str).str.strip()
    frame = frame[~frame["driver_id"].str.lower().isin({"", "nan", "none"})]
    team_column = _first_existing(frame, TEAM_COLUMNS)
    frame["team_id"] = (
        frame[team_column].astype(str).str.strip() if team_column else "unknown_team"
    )
    event_column = _first_existing(frame, EVENT_COLUMNS)
    frame["event_key"] = frame[event_column] if event_column else None
    session_column = _first_existing(frame, SESSION_COLUMNS)
    frame["rehearsal_source"] = (
        frame[session_column].map(_source_name) if session_column else "target_aligned_rehearsal"
    )
    frame["session_priority"] = frame["rehearsal_source"].map(_session_priority)

    lap_column = _first_existing(frame, LAP_TIME_COLUMNS)
    frame["_lap_seconds"] = (
        finite_lap_seconds(
            frame[lap_column],
            min_seconds=config.min_lap_seconds,
            max_seconds=config.max_lap_seconds,
        )
        if lap_column
        else np.nan
    )
    for number, aliases in enumerate(SECTOR_COLUMNS, start=1):
        column = _first_existing(frame, aliases)
        frame[f"_sector{number}"] = (
            _duration_seconds(frame[column]) if column else np.nan
        )
    deleted_column = _first_existing(frame, ("is_deleted", "Deleted", "deleted"))
    accurate_column = _first_existing(frame, ("is_accurate", "IsAccurate"))
    frame["_deleted"] = _bool_series(frame[deleted_column], default=False) if deleted_column else False
    frame["_accurate"] = _bool_series(frame[accurate_column], default=True) if accurate_column else True
    frame["_pit"] = False
    for column in ("PitInTime", "PitOutTime", "pit_in_time", "pit_out_time"):
        if column in frame.columns:
            frame["_pit"] |= frame[column].notna()
    for column in ("is_pit_lap", "is_box_lap"):
        if column in frame.columns:
            frame["_pit"] |= _bool_series(frame[column], default=False)
    official_column = _first_existing(
        frame, ("is_official_classified", "official_classified", "IsOfficialClassified")
    )
    frame["_official"] = (
        _bool_series(frame[official_column], default=False) if official_column else False
    )
    frame["_valid_clean"] = (
        frame["_lap_seconds"].notna()
        & ~frame["_deleted"]
        & frame["_accurate"]
        & ~frame["_pit"]
    )
    frame["_deleted_potential"] = (
        frame["_lap_seconds"].notna() & frame["_deleted"] & ~frame["_pit"]
    )
    time_column = _first_existing(frame, TIME_COLUMNS)
    frame["_lap_time_axis"] = _time_axis(frame[time_column]) if time_column else np.arange(len(frame))
    if as_of is not None and time_column is not None:
        mask = _causal_time_mask(frame[time_column], as_of)
        frame = frame.loc[mask].copy()
    return frame


def _aggregate_session(frame: pd.DataFrame, cfg: QualityAwareLapConfig) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (driver_id, team_id, source), group in frame.groupby(
        ["driver_id", "team_id", "rehearsal_source"], sort=False, dropna=False
    ):
        valid = group.loc[group["_valid_clean"]].sort_values("_lap_seconds")
        official = group.loc[group["_official"] & group["_valid_clean"]].sort_values("_lap_seconds")
        deleted = group.loc[group["_deleted_potential"]].sort_values("_lap_seconds")
        compatible = _compatible_sector_potential(group, cfg)
        valid_best = _first_finite(valid["_lap_seconds"])
        deleted_best = _first_finite(deleted["_lap_seconds"])
        potential_candidates = [value for value in (deleted_best, compatible) if np.isfinite(value)]
        potential = min(potential_candidates) if potential_candidates else float("nan")
        sorted_valid = valid["_lap_seconds"].dropna().astype(float).to_numpy()
        best_row = valid.iloc[0] if not valid.empty else None
        lap_axis = pd.to_numeric(group["_lap_time_axis"], errors="coerce")
        rows.append(
            {
                "driver_id": str(driver_id),
                "team_id": str(team_id),
                "event_key": _first_non_null(group["event_key"]),
                "rehearsal_source": str(source),
                "session_priority": _session_priority(source),
                "official_classified_rehearsal_best_seconds": _first_finite(
                    official["_lap_seconds"]
                ),
                "valid_clean_best_seconds": valid_best,
                "deleted_potential_best_seconds": deleted_best,
                "compatible_sector_potential_seconds": compatible,
                "potential_best_seconds": potential,
                "valid_minus_potential_seconds": (
                    float(valid_best - potential)
                    if np.isfinite(valid_best) and np.isfinite(potential)
                    else float("nan")
                ),
                "best_two_spread_seconds": _spread(sorted_valid, 2),
                "best_three_spread_seconds": _spread(sorted_valid, 3),
                "push_lap_count": _push_lap_count(sorted_valid, cfg.push_lap_relative_threshold),
                "lap_evidence_count": int(len(group)),
                "valid_clean_lap_count": int(len(valid)),
                "deleted_potential_lap_count": int(len(deleted)),
                "best_lap_recency_seconds": _best_lap_recency(best_row, lap_axis),
                "best_lap_session_progress": _best_lap_progress(best_row, lap_axis),
                "track_evolution_seconds_per_progress": _track_evolution(valid),
                "best_lap_compound": _row_value(best_row, ("compound", "Compound")),
                "best_lap_tyre_age_laps": _row_numeric(best_row, ("tyre_age_laps", "TyreLife")),
                "best_lap_fresh_tyre": _row_bool(best_row, ("fresh_tyre", "FreshTyre")),
                "best_lap_track_status": _row_value(best_row, ("track_status", "TrackStatus")),
                "best_lap_speed_trap": _row_numeric(best_row, ("speed_trap", "SpeedST")),
                "best_lap_is_accurate": bool(best_row["_accurate"]) if best_row is not None else False,
                "traffic_or_flag_evidence": _traffic_or_flag_evidence(group),
                "tyre_evidence_complete": _tyre_evidence_complete(best_row),
            }
        )
    return pd.DataFrame(rows)


def _choose_non_team_anchor(row: pd.Series, cfg: QualityAwareLapConfig) -> dict[str, Any]:
    official = _numeric(row.get("official_classified_rehearsal_best_seconds"))
    valid = _numeric(row.get("valid_clean_best_seconds"))
    potential = _numeric(row.get("potential_best_seconds"))
    earlier = _numeric(row.get("earlier_session_best_seconds"))
    if official is not None:
        return _anchor_payload(official, "official_classified_rehearsal", "official_high", cfg.official_uncertainty_seconds, False)
    if valid is not None:
        return _anchor_payload(valid, "valid_clean_rehearsal", "valid_high", cfg.valid_uncertainty_seconds, False)
    if potential is not None and bool(row.get("potential_is_credible", False)):
        return _anchor_payload(
            potential + cfg.potential_penalty_seconds,
            "deleted_or_sector_potential_with_penalty",
            "potential_only_not_valid",
            cfg.potential_uncertainty_seconds,
            True,
        )
    if earlier is not None:
        source = str(row.get("earlier_session_source") or "unknown")
        return _anchor_payload(
            earlier,
            f"earlier_session:{source}",
            "earlier_session_low",
            cfg.earlier_session_uncertainty_seconds,
            True,
        )
    return _anchor_payload(float("nan"), None, None, float("nan"), True)


def _annotate_potential_credibility(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject obviously non-representative deleted/sector evidence.

    Credibility is determined only from the rehearsal field and teammate pace.
    This prevents a slow aborted deleted lap (for example a 129-second lap in a
    92-second field) from becoming a fallback anchor, while retaining a fast
    deleted lap that agrees with the rest of the field.
    """

    result = frame.copy()
    for column in ("valid_clean_best_seconds", "potential_best_seconds"):
        if column not in result.columns:
            result[column] = np.nan
    valid = pd.to_numeric(result["valid_clean_best_seconds"], errors="coerce")
    finite_valid = valid[np.isfinite(valid)]
    field_median = float(finite_valid.median()) if len(finite_valid) else float("nan")
    field_mad = (
        float(np.median(np.abs(finite_valid - field_median)) * 1.4826)
        if len(finite_valid)
        else float("nan")
    )
    field_tolerance = max(4.0, min(8.0, 4.0 * field_mad)) if np.isfinite(field_mad) else 6.0
    credible: list[bool] = []
    reasons: list[str] = []
    for _, row in result.iterrows():
        potential = _numeric(row.get("potential_best_seconds"))
        if potential is None:
            credible.append(False)
            reasons.append("unavailable")
            continue
        peer_valid = pd.to_numeric(
            result.loc[
                (result["team_id"].astype(str) == str(row["team_id"]))
                & (result["driver_id"].astype(str) != str(row["driver_id"])),
                "valid_clean_best_seconds",
            ],
            errors="coerce",
        ).dropna()
        teammate_ok = bool(
            len(peer_valid) and abs(potential - float(peer_valid.median())) <= 4.0
        )
        field_ok = bool(
            np.isfinite(field_median) and abs(potential - field_median) <= field_tolerance
        )
        is_credible = teammate_ok or field_ok
        credible.append(is_credible)
        if teammate_ok:
            reasons.append("agrees_with_teammate_valid_pace")
        elif field_ok:
            reasons.append("agrees_with_rehearsal_field")
        else:
            reasons.append("rejected_field_and_teammate_outlier")
    result["potential_is_credible"] = credible
    result["potential_credibility_reason"] = reasons
    return result


def _add_latent_potential_adjusted_anchor(
    frame: pd.DataFrame,
    cfg: QualityAwareLapConfig,
) -> pd.DataFrame:
    result = frame.copy()
    latent: list[float] = []
    sources: list[str] = []
    used_potential: list[bool] = []
    for _, row in result.iterrows():
        anchor = _numeric(row.get("quality_aware_anchor_seconds"))
        valid = _numeric(row.get("valid_clean_best_seconds"))
        potential = _numeric(row.get("potential_best_seconds"))
        credible = bool(row.get("potential_is_credible", False))
        penalized = (
            potential + float(cfg.potential_penalty_seconds)
            if potential is not None and credible
            else None
        )
        if (
            valid is not None
            and penalized is not None
            and valid - penalized >= 2.0
        ):
            latent.append(float(penalized))
            sources.append("credible_potential_overrides_unrepresentative_valid_lap")
            used_potential.append(True)
        elif anchor is not None:
            latent.append(float(anchor))
            sources.append(str(row.get("anchor_source") or "quality_aware_anchor"))
            used_potential.append(bool("potential" in sources[-1]))
        else:
            latent.append(float("nan"))
            sources.append("unavailable")
            used_potential.append(False)
    result["latent_potential_adjusted_anchor_seconds"] = latent
    result["latent_anchor_source"] = sources
    result["latent_anchor_uses_potential"] = used_potential
    return result


def _anchor_payload(
    seconds: float,
    source: str | None,
    quality: str | None,
    uncertainty: float,
    imputed: bool,
) -> dict[str, Any]:
    return {
        "quality_aware_anchor_seconds": float(seconds),
        "anchor_source": source,
        "anchor_quality": quality,
        "anchor_uncertainty_seconds": float(uncertainty),
        "anchor_is_imputed": bool(imputed),
    }


def _compatible_sector_potential(group: pd.DataFrame, cfg: QualityAwareLapConfig) -> float:
    sector_valid = group[["_sector1", "_sector2", "_sector3"]].apply(
        pd.to_numeric, errors="coerce"
    )
    eligible = group.loc[~group["_pit"] & sector_valid.notna().all(axis=1)].copy()
    if eligible.empty:
        return float("nan")
    compatibility_columns: list[str] = []
    for field in cfg.sector_compatibility_fields:
        aliases = {
            "session": ("rehearsal_source", "session", "SessionName"),
            "compound": ("compound", "Compound"),
            "weather": ("weather", "Rainfall", "rainfall"),
            "setup": ("setup_version", "car_specification"),
        }.get(field, (field,))
        column = _first_existing(eligible, aliases)
        if column is not None:
            compatibility_columns.append(column)
    if not compatibility_columns:
        strata: Iterable[tuple[Any, pd.DataFrame]] = [("all", eligible)]
    elif len(compatibility_columns) == 1:
        strata = eligible.groupby(compatibility_columns[0], dropna=False, sort=False)
    else:
        strata = eligible.groupby(compatibility_columns, dropna=False, sort=False)
    candidates: list[float] = []
    for _, rows in strata:
        sectors = [pd.to_numeric(rows[f"_sector{i}"], errors="coerce").min() for i in (1, 2, 3)]
        if all(np.isfinite(value) and value > 0.0 for value in sectors):
            total = float(sum(sectors))
            if cfg.min_lap_seconds <= total <= cfg.max_lap_seconds:
                candidates.append(total)
    return min(candidates) if candidates else float("nan")


def _entrant_roster(
    entrants: pd.DataFrame | Sequence[str] | None,
    *,
    current: pd.DataFrame,
    earlier: pd.DataFrame,
) -> pd.DataFrame:
    evidence = pd.concat(
        [part[["driver_id", "team_id"]] for part in (current, earlier) if not part.empty],
        ignore_index=True,
    ) if (not current.empty or not earlier.empty) else pd.DataFrame(columns=["driver_id", "team_id"])
    evidence = evidence.drop_duplicates("driver_id", keep="first")
    if entrants is None:
        return evidence.reset_index(drop=True)
    if isinstance(entrants, pd.DataFrame):
        driver_column = _first_existing(entrants, DRIVER_COLUMNS)
        if driver_column is None:
            raise ValueError("entrants is missing a driver identifier")
        team_column = _first_existing(entrants, TEAM_COLUMNS)
        roster = pd.DataFrame({"driver_id": entrants[driver_column].astype(str).str.strip()})
        roster["team_id"] = (
            entrants[team_column].astype(str).str.strip().to_numpy()
            if team_column
            else roster["driver_id"].map(evidence.set_index("driver_id")["team_id"])
        )
    else:
        roster = pd.DataFrame({"driver_id": [str(value).strip() for value in entrants]})
        roster["team_id"] = roster["driver_id"].map(evidence.set_index("driver_id")["team_id"])
    roster["team_id"] = roster["team_id"].fillna("unknown_team").astype(str)
    return roster.drop_duplicates("driver_id", keep="first").reset_index(drop=True)


def _team_prior_mapping(priors: pd.DataFrame | Mapping[str, float] | None) -> dict[str, float]:
    if priors is None:
        return {}
    if isinstance(priors, Mapping):
        return {
            str(key): float(value)
            for key, value in priors.items()
            if _numeric(value) is not None
        }
    if not isinstance(priors, pd.DataFrame):
        raise TypeError("team_priors must be a mapping or pandas DataFrame")
    team_column = _first_existing(priors, TEAM_COLUMNS)
    value_column = _first_existing(
        priors, ("prior_lap_time_seconds", "team_prior_seconds", "lap_time_seconds")
    )
    if team_column is None or value_column is None:
        raise ValueError("team_priors requires team and prior lap-time columns")
    values = finite_lap_seconds(priors[value_column])
    return {
        str(team): float(value)
        for team, value in zip(priors[team_column], values)
        if np.isfinite(value)
    }


def _relative_to_group_median(
    row: pd.Series,
    frame: pd.DataFrame,
    *,
    group_column: str,
    value_column: str,
) -> float:
    value = _numeric(row.get(value_column))
    if value is None:
        return float("nan")
    peers = frame.loc[
        (frame[group_column].astype(str) == str(row[group_column]))
        & (frame["driver_id"].astype(str) != str(row["driver_id"])),
        value_column,
    ]
    peer_median = pd.to_numeric(peers, errors="coerce").median(skipna=True)
    return float(value - peer_median) if np.isfinite(peer_median) else float("nan")


def _duration_seconds(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    unresolved = numeric.isna()
    if unresolved.any():
        numeric = numeric.fillna(
            pd.to_timedelta(values.where(unresolved), errors="coerce").dt.total_seconds()
        )
    return pd.to_numeric(numeric, errors="coerce").astype(float).where(lambda x: x > 0.0)


def _time_axis(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return numeric.astype(float)
    timestamps = pd.to_datetime(values, errors="coerce", utc=True)
    if timestamps.notna().any():
        origin = timestamps.min()
        return (timestamps - origin).dt.total_seconds()
    durations = pd.to_timedelta(values, errors="coerce").dt.total_seconds()
    return durations.astype(float)


def _causal_time_mask(values: pd.Series, as_of: Any) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    try:
        numeric_cutoff = float(as_of)
    except (TypeError, ValueError):
        numeric_cutoff = float("nan")
    if numeric.notna().any() and np.isfinite(numeric_cutoff):
        return numeric.notna() & numeric.le(numeric_cutoff)
    timestamps = pd.to_datetime(values, errors="coerce", utc=True)
    cutoff = pd.to_datetime(as_of, errors="coerce", utc=True)
    if timestamps.notna().any() and not pd.isna(cutoff):
        return timestamps.notna() & timestamps.le(cutoff)
    raise ValueError("as_of is incompatible with the available lap timestamp column")


def _source_name(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "fp3": "practice_3",
        "practice3": "practice_3",
        "sq": "sprint_qualifying",
        "sprint_shootout": "sprint_qualifying",
        "fp2": "practice_2",
        "fp1": "practice_1",
    }
    return aliases.get(text, text or "target_aligned_rehearsal")


def _session_priority(source: Any) -> int:
    return {
        "practice_3": 0,
        "sprint_qualifying": 0,
        "practice_2": 1,
        "practice_1": 2,
        "sprint": 1,
    }.get(_source_name(source), 3)


def _bool_series(values: pd.Series, *, default: bool) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapped = normalized.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "false": False,
            "0": False,
            "no": False,
            "nan": default,
            "none": default,
            "": default,
        }
    )
    return mapped.fillna(default).astype(bool)


def _first_existing(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _first_finite(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    return float(finite.iloc[0]) if len(finite) else float("nan")


def _first_non_null(values: pd.Series) -> Any:
    usable = values.dropna()
    return usable.iloc[0] if len(usable) else None


def _numeric(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _spread(values: np.ndarray, count: int) -> float:
    if len(values) < count:
        return float("nan")
    return float(values[count - 1] - values[0])


def _push_lap_count(values: np.ndarray, threshold: float) -> int:
    if len(values) == 0:
        return 0
    return int(np.sum(values <= float(values[0]) * (1.0 + float(threshold))))


def _best_lap_recency(best_row: pd.Series | None, lap_axis: pd.Series) -> float:
    if best_row is None:
        return float("nan")
    best = _numeric(best_row.get("_lap_time_axis"))
    end = pd.to_numeric(lap_axis, errors="coerce").max(skipna=True)
    return float(end - best) if best is not None and np.isfinite(end) else float("nan")


def _best_lap_progress(best_row: pd.Series | None, lap_axis: pd.Series) -> float:
    if best_row is None:
        return float("nan")
    best = _numeric(best_row.get("_lap_time_axis"))
    start = pd.to_numeric(lap_axis, errors="coerce").min(skipna=True)
    end = pd.to_numeric(lap_axis, errors="coerce").max(skipna=True)
    if best is None or not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return float("nan")
    return float((best - start) / (end - start))


def _track_evolution(valid: pd.DataFrame) -> float:
    if len(valid) < 2:
        return float("nan")
    best = float(valid["_lap_seconds"].min())
    push = valid.loc[valid["_lap_seconds"] <= best * 1.07]
    x = pd.to_numeric(push["_lap_time_axis"], errors="coerce")
    y = pd.to_numeric(push["_lap_seconds"], errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or float(x[mask].max() - x[mask].min()) <= 0.0:
        return float("nan")
    x_valid = x[mask].reset_index(drop=True)
    y_valid = y[mask].reset_index(drop=True)
    progress = (x_valid - x_valid.min()) / (x_valid.max() - x_valid.min())
    slopes = [
        float((y_valid.iloc[j] - y_valid.iloc[i]) / (progress.iloc[j] - progress.iloc[i]))
        for i in range(len(progress))
        for j in range(i + 1, len(progress))
        if progress.iloc[j] != progress.iloc[i]
    ]
    return float(np.median(slopes)) if slopes else float("nan")


def _row_value(row: pd.Series | None, names: Sequence[str]) -> Any:
    if row is None:
        return None
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return None


def _row_numeric(row: pd.Series | None, names: Sequence[str]) -> float:
    return _numeric(_row_value(row, names)) or float("nan")


def _row_bool(row: pd.Series | None, names: Sequence[str]) -> bool | None:
    value = _row_value(row, names)
    if value is None:
        return None
    return bool(_bool_series(pd.Series([value]), default=False).iloc[0])


def _traffic_or_flag_evidence(group: pd.DataFrame) -> bool:
    status_column = _first_existing(group, ("track_status", "TrackStatus"))
    status_flag = False
    if status_column:
        statuses = group[status_column].dropna().astype(str).str.strip()
        status_flag = bool((statuses != "1").any())
    reason_column = _first_existing(group, ("DeletedReason", "deleted_reason"))
    reason_flag = False
    if reason_column:
        reason_flag = bool(
            group[reason_column]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains("imped|traffic|yellow|red flag", regex=True)
            .any()
        )
    return bool(status_flag or reason_flag)


def _tyre_evidence_complete(row: pd.Series | None) -> bool:
    if row is None:
        return False
    compound = _row_value(row, ("compound", "Compound"))
    age = _row_numeric(row, ("tyre_age_laps", "TyreLife"))
    fresh = _row_bool(row, ("fresh_tyre", "FreshTyre"))
    return compound is not None and np.isfinite(age) and fresh is not None


def _as_of_label(as_of: Any | None) -> str:
    return str(as_of) if as_of is not None else "snapshot_complete_pre_qualifying"


_OUTPUT_COLUMNS: tuple[str, ...] = (
    "event_key",
    "driver_id",
    "team_id",
    "rehearsal_source",
    "official_classified_rehearsal_best_seconds",
    "valid_clean_best_seconds",
    "deleted_potential_best_seconds",
    "compatible_sector_potential_seconds",
    "potential_best_seconds",
    "potential_is_credible",
    "potential_credibility_reason",
    "valid_minus_potential_seconds",
    "best_two_spread_seconds",
    "best_three_spread_seconds",
    "push_lap_count",
    "lap_evidence_count",
    "valid_clean_lap_count",
    "deleted_potential_lap_count",
    "best_lap_recency_seconds",
    "best_lap_session_progress",
    "track_evolution_seconds_per_progress",
    "best_lap_compound",
    "best_lap_tyre_age_laps",
    "best_lap_fresh_tyre",
    "best_lap_track_status",
    "best_lap_speed_trap",
    "best_lap_is_accurate",
    "traffic_or_flag_evidence",
    "tyre_evidence_complete",
    "earlier_session_best_seconds",
    "earlier_session_source",
    "quality_aware_anchor_seconds",
    "latent_potential_adjusted_anchor_seconds",
    "latent_anchor_source",
    "latent_anchor_uses_potential",
    "anchor_source",
    "anchor_quality",
    "anchor_uncertainty_seconds",
    "anchor_is_imputed",
    "teammate_relative_anchor_seconds",
    "field_relative_anchor_seconds",
    "evidence_item_count",
    "evidence_coverage_rate",
    "feature_as_of",
    "feature_contract",
)


__all__ = [
    "QualityAwareLapConfig",
    "build_quality_aware_rehearsal_features",
    "finite_lap_seconds",
]
