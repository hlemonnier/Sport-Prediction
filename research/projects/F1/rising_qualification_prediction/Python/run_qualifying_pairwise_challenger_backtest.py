#!/usr/bin/env python3
"""Same-season event-block backtest for the shared Qualifying challenger.

The canonical path reads one target season only. Cross-season weak-transfer
research remains available solely through the explicit legacy CLI flag.
"""

from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from packages.f1.domain.weekend import qualifying_elimination_rule
from packages.f1.features.qualifying_lap import build_quality_aware_rehearsal_features
from packages.f1.models.pre_quali.pairwise import (
    PairwiseRankerConfig,
    fit_pairwise_qualifying_ranker,
)
from packages.f1.models.pre_quali.train import train_shared_qualifying_latent_model
from packages.f1.models.ultimate_lap_time.achievable import (
    ACTUAL_LAP_COLUMN,
    DIRECT_PACE_POSITION_COLUMN,
    MEAL_POSITION_COLUMN,
    POINT_HEAD_DIRECT_PACE,
    POINT_HEAD_MINIMUM_EXPECTED_ABSOLUTE_LOSS,
    Q1_LAP_COLUMN,
    Q2_LAP_COLUMN,
    Q3_LAP_COLUMN,
    SHARED_QUALIFYING_ENABLE_ROBUST_RESIDUAL,
    SHARED_QUALIFYING_SAMPLE_COUNT,
    SHARED_QUALIFYING_SAMPLE_SEED_BASE,
    build_shared_qualifying_event_forecast,
    calibrate_achievable_best_lap_model,
    shared_qualifying_forecast_artifact,
)
from packages.f1.models.pre_quali.selection import (
    FrozenSelectorConfig,
    QualifyingModelEvidence,
    select_frozen_qualifying_model,
)
from packages.f1.orchestration.model_runtime import f1_model_runtime_doctor
from packages.f1.orchestration.non_live_validation import (
    EventError,
    evaluate_qualifying_promotion,
    validate_event_partitions,
)


ROUND_PATTERN = re.compile(r"round_(\d+)", re.IGNORECASE)
Q2_VALID_LAP_COLUMN = "has_valid_q2_lap"
Q3_VALID_LAP_COLUMN = "has_valid_q3_lap"
FEATURE_ALLOWLIST = (
    "quality_aware_anchor_seconds",
    "latent_potential_adjusted_anchor_seconds",
    "anchor_uncertainty_seconds",
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
    "best_lap_tyre_age_laps",
    "best_lap_fresh_tyre",
    "best_lap_speed_trap",
    "traffic_or_flag_evidence",
    "tyre_evidence_complete",
    "teammate_relative_anchor_seconds",
    "field_relative_anchor_seconds",
    "evidence_item_count",
    "evidence_coverage_rate",
)
QUALIFYING_BASELINE_MODEL_ID = "qualifying_rehearsal_rank_baseline_v1"
QUALIFYING_CHALLENGER_MODEL_ID = "shared_qualifying_latent_lap_v4"
QUALIFYING_ARTIFACT_SCHEMA_VERSION = "f1_shared_qualifying_latent_event_block_v6"


def _root() -> Path:
    return Path(__file__).resolve().parents[5]


def _json_ints(values: Iterable[object]) -> list[int]:
    """Return JSON-native event identifiers, including from pandas indexes."""

    return [int(value) for value in values]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_snapshot_path(root: Path, event_dir: Path, value: object) -> Path:
    candidate = Path(str(value)).expanduser()
    if candidate.is_absolute():
        return candidate
    root_candidate = root / candidate
    return root_candidate if root_candidate.exists() else event_dir / candidate.name


def _round_number(path: Path) -> int:
    match = ROUND_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"invalid round directory: {path}")
    return int(match.group(1))


def _driver_column(frame: pd.DataFrame) -> str:
    for column in ("Abbreviation", "Driver", "DriverId", "DriverNumber"):
        if column in frame.columns:
            return column
    raise ValueError("classification has no supported driver identifier")


def _team_column(frame: pd.DataFrame) -> str | None:
    return next((column for column in ("TeamName", "Team", "team_name") if column in frame.columns), None)


def _completed_positions(results: pd.DataFrame) -> pd.Series:
    for column in ("Position", "ClassifiedPosition"):
        if column in results.columns:
            values = pd.to_numeric(results[column], errors="coerce")
            break
    else:
        values = pd.Series(np.nan, index=results.index, dtype=float)
    used = {int(value) for value in values.dropna().tolist() if float(value) > 0}
    next_position = 1
    output = values.copy()
    for index in output.index[output.isna() | output.le(0)]:
        while next_position in used:
            next_position += 1
        output.loc[index] = next_position
        used.add(next_position)
    order = np.lexsort((np.arange(len(output)), output.to_numpy(dtype=float)))
    ranks = np.empty(len(output), dtype=int)
    ranks[order] = np.arange(1, len(output) + 1)
    return pd.Series(ranks, index=results.index, dtype=int)


def _has_time(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    text = frame[column].fillna("").astype(str).str.strip().str.lower()
    return ~text.isin({"", "nan", "nat", "none"})


def _lap_seconds(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    unresolved = numeric.isna()
    if unresolved.any():
        numeric = numeric.fillna(
            pd.to_timedelta(values.where(unresolved), errors="coerce").dt.total_seconds()
        )
    return pd.to_numeric(numeric, errors="coerce").astype(float)


def _pre_qualifying_roster(*parts: pd.DataFrame) -> pd.DataFrame:
    """Build the inference entrant set only from timestamped pre-Q evidence."""

    rows: list[pd.DataFrame] = []
    for part in parts:
        if part is None or part.empty:
            continue
        driver = next(
            (column for column in ("Driver", "Abbreviation", "DriverId") if column in part.columns),
            None,
        )
        if driver is None:
            continue
        team = next(
            (column for column in ("Team", "TeamName", "team_id") if column in part.columns),
            None,
        )
        item = pd.DataFrame({"driver_id": part[driver].astype(str).str.strip()})
        item["team_id"] = (
            part[team].astype(str).str.strip().to_numpy()
            if team is not None
            else "unknown_team"
        )
        rows.append(item)
    if not rows:
        raise ValueError("no pre-Qualifying roster evidence was available")
    roster = pd.concat(rows, ignore_index=True).loc[lambda frame: frame["driver_id"].ne("")]
    return roster.drop_duplicates("driver_id", keep="first").reset_index(drop=True)


def _causal_rehearsal_ranks(frame: pd.DataFrame) -> pd.Series:
    """Rank causal rehearsal anchors without inventing a driver-name signal.

    The provider's completed-session classification order is preserved by the
    entrant and feature builders. It is therefore the defensible deterministic
    tie-break for equal or missing anchors. Driver identifiers must never decide
    a sporting rank.
    """

    baseline_anchor = pd.to_numeric(frame.get("valid_clean_best_seconds"), errors="coerce")
    fallback = pd.to_numeric(frame.get("quality_aware_anchor_seconds"), errors="coerce")
    baseline_anchor = baseline_anchor.where(baseline_anchor.notna(), fallback)
    stable = pd.DataFrame(
        {
            "anchor": baseline_anchor.fillna(np.inf),
            "provider_order": np.arange(len(frame)),
        },
        index=frame.index,
    ).sort_values(["anchor", "provider_order"], kind="mergesort")
    ranks = pd.Series(np.arange(1, len(stable) + 1), index=stable.index)
    return ranks.reindex(frame.index).astype(int)


def _event_inference_frame(
    root: Path,
    event_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any], list[Path], Path]:
    """Build and freeze one event's pre-Qualifying inference snapshot.

    The returned frame contains no Grand Prix Qualifying result content.  The
    target path is resolved for later evaluation, but it is not opened here.
    """

    metadata_path = event_dir / "weekend_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sessions = [dict(value) for value in metadata.get("sessions", []) if isinstance(value, dict)]
    qualifying = next(
        (
            session
            for session in sessions
            if str(session.get("session_type", "")).strip().lower() == "qualifying"
        ),
        None,
    )
    if qualifying is None:
        raise ValueError(f"{event_dir.name}: qualifying session missing")
    qualifying_order = int(qualifying.get("session_order", 999))
    target_candidates = [
        session
        for session in sessions
        if int(session.get("session_order", 999)) < qualifying_order
        and (
            str(session.get("session_type", "")).strip().lower()
            == "sprint_qualifying"
            or str(session.get("session_name", ""))
            .strip()
            .lower()
            .replace(" ", "_")
            in {"practice_3", "fp3"}
        )
        and bool(session.get("completed", True))
    ]
    if not target_candidates:
        raise ValueError(
            f"{event_dir.name}: no causal FP3/Sprint-Qualifying target-aligned rehearsal"
        )
    target_aligned = max(
        target_candidates, key=lambda value: int(value.get("session_order", -1))
    )
    target_order = int(target_aligned.get("session_order", -1))
    completed_pre_q = [
        session
        for session in sessions
        if int(session.get("session_order", 999)) < qualifying_order
        and bool(session.get("completed", True))
    ]
    earlier_lap_sessions = [
        session
        for session in completed_pre_q
        if int(session.get("session_order", 999)) < target_order
        and (
            str(session.get("session_type", "")).strip().lower().startswith("practice")
            or str(session.get("session_name", ""))
            .strip()
            .lower()
            .replace(" ", "_")
            .startswith(("practice_", "fp"))
        )
        and session.get("laps_path")
    ]

    qualifying_results_path = _resolve_snapshot_path(
        root, event_dir, qualifying.get("results_path")
    )
    rehearsal_path = _resolve_snapshot_path(root, event_dir, target_aligned.get("laps_path"))
    rehearsal_results_path = _resolve_snapshot_path(
        root, event_dir, target_aligned.get("results_path")
    )
    rehearsal_laps = pd.read_csv(rehearsal_path)
    rehearsal_results = pd.read_csv(rehearsal_results_path)
    source_label = str(
        target_aligned.get("session_name") or target_aligned.get("session_type") or ""
    ).lower()
    source = "sprint_qualifying" if "sprint" in source_label else "practice_3"
    rehearsal_laps["rehearsal_source"] = source

    earlier_parts: list[pd.DataFrame] = []
    input_paths = [
        metadata_path,
        qualifying_results_path,
        rehearsal_path,
        rehearsal_results_path,
    ]
    for session in earlier_lap_sessions:
        path = _resolve_snapshot_path(root, event_dir, session.get("laps_path"))
        part = pd.read_csv(path)
        part["rehearsal_source"] = str(
            session.get("session_name") or session.get("session_type") or "earlier_rehearsal"
        )
        earlier_parts.append(part)
        input_paths.append(path)
    earlier_laps = pd.concat(earlier_parts, ignore_index=True) if earlier_parts else None

    # Freeze inference inputs before opening the target classification. This is
    # the production boundary. The latest target-aligned rehearsal owns the
    # active roster: unioning older FP1/FP2 classifications silently re-added
    # reserve drivers who had already surrendered their seats and produced
    # illegal 21/23/25-car fields in real replays. Missing entrants now remain a
    # visible coverage failure instead of being guessed from the target result.
    entrants = _pre_qualifying_roster(rehearsal_results)
    features = build_quality_aware_rehearsal_features(
        rehearsal_laps,
        entrants=entrants,
        earlier_laps=earlier_laps,
        official_session_timing=True,
    )
    event_key = int(metadata["year"]) * 100 + int(metadata["round_number"])
    frame = features.copy()
    frame["event_key"] = event_key
    frame["rehearsal_source"] = source
    frame["latest_qualifying_rehearsal_source"] = source
    frame["latest_qualifying_rehearsal_rank"] = _causal_rehearsal_ranks(frame)
    event_info = {
        "event_key": event_key,
        "year": int(metadata["year"]),
        "round": int(metadata["round_number"]),
        "event_name": str(metadata.get("event_name") or event_dir.name),
        "event_format": str(metadata.get("event_format") or "unknown"),
        "rehearsal_source": source,
        "field_size": int(len(frame)),
        "roster_source": "latest_target_aligned_pre_qualifying_session_only",
        "target_result_used_for_roster": False,
    }
    return frame, event_info, input_paths, qualifying_results_path


def _qualifying_target_frame(
    qualifying_results_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Open the evaluation-only Qualifying classification."""

    qualifying_results = pd.read_csv(qualifying_results_path)
    driver_column = _driver_column(qualifying_results)
    target_drivers = qualifying_results[driver_column].astype(str).str.strip()
    q1 = _has_time(qualifying_results, "Q1")
    q2 = _has_time(qualifying_results, "Q2")
    q3 = _has_time(qualifying_results, "Q3")
    stage_seconds = {
        stage: (
            _lap_seconds(qualifying_results[column])
            if column in qualifying_results.columns
            else pd.Series(np.nan, index=qualifying_results.index)
        )
        for stage, column in ((1, "Q1"), (2, "Q2"), (3, "Q3"))
    }
    official_positions = _completed_positions(qualifying_results)
    field_size = int(target_drivers.nunique())
    if field_size >= 12:
        try:
            elimination = qualifying_elimination_rule(field_size)
            q2_slots = int(elimination.period_2_cars)
            q3_slots = int(elimination.period_3_cars)
        except ValueError:
            # A historical result table can omit a withdrawn entrant and leave
            # an odd classified field. Preserve ten Q3 places and put the odd
            # elimination remainder after Q1 rather than inferring per-driver
            # advancement from whether a Q2 time happens to exist.
            q3_slots = min(10, field_size)
            q2_slots = q3_slots + ((field_size - q3_slots + 1) // 2)
    else:
        # Tiny synthetic fixtures do not represent an FIA three-period field.
        # Keep them internally nested without inventing elimination cut sizes.
        q2_slots = field_size
        q3_slots = field_size
    status = qualifying_results.get(
        "QualifyingStatus",
        qualifying_results.get("Status", pd.Series("", index=qualifying_results.index)),
    ).fillna("").astype(str).str.strip().str.lower()
    explicit_q3 = status.str.contains(
        r"(?:^|[^a-z0-9])q3(?:$|[^a-z0-9])|period[_ ]?3|segment[_ ]?3",
        regex=True,
    )
    explicit_q2 = explicit_q3 | status.str.contains(
        r"(?:^|[^a-z0-9])q2(?:$|[^a-z0-9])|period[_ ]?2|segment[_ ]?2",
        regex=True,
    )
    # Final classification can move a driver outside the segment cut after a
    # penalty or disqualification. A recorded Q2/Q3 time is therefore also
    # conclusive evidence that the driver reached that segment. Conversely,
    # the position/status branches preserve advanced-with-no-valid-time cases.
    reached_q2 = official_positions.le(q2_slots) | explicit_q2 | q2 | q3
    reached_q3 = official_positions.le(q3_slots) | explicit_q3 | q3
    reached_q2 = reached_q2 | reached_q3
    actual = pd.DataFrame(
        {
            "driver_id": target_drivers,
            "qualy_position": official_positions,
            # Provider tables occasionally omit a Q1 split while retaining a
            # later-stage time.  Any official Q1/Q2/Q3 time proves a valid lap;
            # the stage labels remain strictly nested.
            "has_valid_qualifying_lap": (q1 | q2 | q3).astype(int),
            # Advancement is an official classification/status fact. A driver
            # can advance and then set no valid time in the next segment, so
            # Q2/Q3 time presence must remain a separate label.
            "reached_q2": reached_q2.astype(int),
            "reached_q3": reached_q3.astype(int),
            Q2_VALID_LAP_COLUMN: q2.astype(int),
            Q3_VALID_LAP_COLUMN: q3.astype(int),
            Q1_LAP_COLUMN: stage_seconds[1],
            Q2_LAP_COLUMN: stage_seconds[2],
            Q3_LAP_COLUMN: stage_seconds[3],
            ACTUAL_LAP_COLUMN: pd.concat(list(stage_seconds.values()), axis=1).min(
                axis=1, skipna=True
            ),
        }
    )
    target_info = {
        "official_target_driver_count": int(target_drivers.nunique()),
        "official_target_driver_ids": sorted(target_drivers.unique().tolist()),
        "stage_advancement_label_source": (
            "official_classification_cut_slots_or_explicit_status_or_later_stage_time"
        ),
        "advanced_stage_time_validity_modeled_separately": True,
    }
    return actual, target_info


def _load_target_after_frozen_forecast(
    qualifying_results_path: Path,
    *,
    expected_event_key: int,
    frozen_forecast_artifact: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fail closed unless a complete event forecast exists before target I/O."""

    artifact_hash = str(frozen_forecast_artifact.get("artifact_sha256") or "")
    artifact_event_key = frozen_forecast_artifact.get("event_key")
    if re.fullmatch(r"[0-9a-f]{64}", artifact_hash) is None:
        raise RuntimeError("Qualifying target read requires a frozen forecast artifact")
    try:
        matching_event = int(artifact_event_key) == int(expected_event_key)
    except (TypeError, ValueError):
        matching_event = False
    if not matching_event:
        raise RuntimeError("Qualifying target read artifact belongs to another event")
    return _qualifying_target_frame(qualifying_results_path)


def _attach_qualifying_target(
    inference_frame: pd.DataFrame,
    target: pd.DataFrame,
) -> pd.DataFrame:
    target_columns = [column for column in target.columns if column != "driver_id"]
    return inference_frame.drop(columns=target_columns, errors="ignore").merge(
        target,
        on="driver_id",
        how="left",
        validate="one_to_one",
    )


def _event_frame(root: Path, event_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], list[Path]]:
    """Backward-compatible fully labelled frame for historical training/tests."""

    inference, info, input_paths, target_path = _event_inference_frame(root, event_dir)
    target, target_info = _qualifying_target_frame(target_path)
    info.update(target_info)
    return _attach_qualifying_target(inference, target), info, input_paths


def _metric_rows(
    predictions: pd.DataFrame,
    event_info: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for event_key, group in predictions.groupby("event_key", sort=True):
        info = event_info[int(event_key)]
        actual = pd.to_numeric(group["actual_qualifying_position"], errors="coerce")
        candidate = pd.to_numeric(group["predicted_qualifying_position"], errors="coerce")
        baseline = pd.to_numeric(group["baseline_rank_prior"], errors="coerce")
        comparator = pd.to_numeric(
            group["pairwise_comparator_position"]
            if "pairwise_comparator_position" in group.columns
            else pd.Series(np.nan, index=group.index),
            errors="coerce",
        )
        selected = pd.to_numeric(
            group["selected_predicted_qualifying_position"]
            if "selected_predicted_qualifying_position" in group.columns
            else candidate,
            errors="coerce",
        )
        candidate_mae = float((candidate - actual).abs().mean())
        baseline_mae = float((baseline - actual).abs().mean())
        selected_mae = float((selected - actual).abs().mean())
        candidate_order = set(group.nsmallest(3, "predicted_qualifying_position")["driver_id"])
        baseline_order = set(group.nsmallest(3, "baseline_rank_prior")["driver_id"])
        actual_top3 = set(group.nsmallest(3, "actual_qualifying_position")["driver_id"])
        actual_top10 = set(group.nsmallest(min(10, len(group)), "actual_qualifying_position")["driver_id"])
        candidate_top10 = set(group.nsmallest(min(10, len(group)), "predicted_qualifying_position")["driver_id"])
        baseline_top10 = set(group.nsmallest(min(10, len(group)), "baseline_rank_prior")["driver_id"])
        actual_winner = str(group.nsmallest(1, "actual_qualifying_position").iloc[0]["driver_id"])
        selected_order = set(
            group.assign(_selected_position=selected).nsmallest(3, "_selected_position")[
                "driver_id"
            ]
        )
        selected_top10 = set(
            group.assign(_selected_position=selected).nsmallest(
                min(10, len(group)), "_selected_position"
            )["driver_id"]
        )
        selected_winner = str(
            group.assign(_selected_position=selected)
            .nsmallest(1, "_selected_position")
            .iloc[0]["driver_id"]
        )
        availability_values = {
            str(value)
            for value in group.get(
                "candidate_availability",
                pd.Series("available", index=group.index),
            ).tolist()
        }
        if len(availability_values) != 1:
            raise ValueError(
                f"event {int(event_key)} has inconsistent candidate availability"
            )
        candidate_availability = next(iter(availability_values))
        candidate_available = candidate_availability == "available"
        events.append(
            {
                **info,
                "candidate_available": candidate_available,
                "candidate_availability": candidate_availability,
                "baseline_mae": baseline_mae,
                "candidate_mae": candidate_mae,
                "selected_output_mae": selected_mae,
                "delta_candidate_minus_baseline": candidate_mae - baseline_mae,
                "baseline_kendall": float(baseline.corr(actual, method="kendall")),
                "candidate_kendall": float(candidate.corr(actual, method="kendall")),
                "selected_output_kendall": float(selected.corr(actual, method="kendall")),
                "pairwise_comparator_mae": float((comparator - actual).abs().mean()),
                "pairwise_comparator_kendall": float(comparator.corr(actual, method="kendall")),
                "baseline_pole_hit": str(group.nsmallest(1, "baseline_rank_prior").iloc[0]["driver_id"]) == actual_winner,
                "candidate_pole_hit": str(group.nsmallest(1, "predicted_qualifying_position").iloc[0]["driver_id"]) == actual_winner,
                "selected_output_pole_hit": selected_winner == actual_winner,
                "baseline_top3_overlap": len(baseline_order & actual_top3) / 3.0,
                "candidate_top3_overlap": len(candidate_order & actual_top3) / 3.0,
                "selected_output_top3_overlap": len(selected_order & actual_top3) / 3.0,
                "baseline_top10_overlap": len(baseline_top10 & actual_top10) / float(min(10, len(group))),
                "candidate_top10_overlap": len(candidate_top10 & actual_top10) / float(min(10, len(group))),
                "selected_output_top10_overlap": len(selected_top10 & actual_top10) / float(min(10, len(group))),
            }
        )
        for record in group.to_dict(orient="records"):
            rows.append({**info, **record})
    return events, rows


def _mean(events: Sequence[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(event[key]) for event in events]))


def _locked_event_partitions(
    event_keys: Sequence[int],
    *,
    audit_year: int,
    same_season_only: bool = True,
) -> dict[str, tuple[int, ...]]:
    ordered = tuple(sorted({int(value) for value in event_keys}))
    if same_season_only:
        foreign = tuple(value for value in ordered if value // 100 != int(audit_year))
        if foreign:
            raise ValueError(
                "same-season Qualifying protocol rejects events outside the target year: "
                f"{list(foreign)}"
            )
        required = tuple(int(audit_year) * 100 + round_number for round_number in range(1, 7))
        missing_required = tuple(value for value in required if value not in set(ordered))
        audit = tuple(value for value in ordered if value % 100 >= 7)
        if missing_required or len(audit) < 2:
            raise ValueError(
                "same-season Qualifying evaluation requires exact R1-R6 plus at least "
                f"two R7+ audit events; missing={list(missing_required)}"
            )
        partitions = {
            "point_fit": required[:2],
            "selection": required[2:4],
            "calibration": required[4:6],
            "audit": audit,
        }
        issues = validate_event_partitions(
            development=[str(value) for value in partitions["point_fit"]],
            selection=[str(value) for value in partitions["selection"]],
            calibration=[str(value) for value in partitions["calibration"]],
            audit=[str(value) for value in partitions["audit"]],
        )
        if issues:
            raise ValueError(f"invalid same-season Qualifying event partitions: {list(issues)}")
        return partitions

    prior_years = sorted({value // 100 for value in ordered if value // 100 < int(audit_year)})
    audit_year_events = tuple(value for value in ordered if value // 100 == int(audit_year))
    if not prior_years or len(audit_year_events) < 6:
        raise ValueError(
            "shared Qualifying evaluation requires a prior selection season and at least "
            "six same-season calibration/audit events"
        )
    selection_year = prior_years[-1]
    partitions = {
        "development": tuple(value for value in ordered if value // 100 < selection_year),
        "selection": tuple(value for value in ordered if value // 100 == selection_year),
        "point_fit": audit_year_events[:2],
        "calibration": audit_year_events[2:4],
        "audit": audit_year_events[4:],
    }
    issues = validate_event_partitions(
        development=[str(value) for value in partitions["development"]],
        selection=[str(value) for value in partitions["selection"]],
        calibration=[str(value) for value in partitions["point_fit"]],
        audit=[
            str(value)
            for value in (*partitions["calibration"], *partitions["audit"])
        ],
    )
    if issues:
        raise ValueError(f"invalid Qualifying event partitions: {list(issues)}")
    return partitions


def _stage_metrics(rows: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for name, target, probability in (
        ("valid_lap", "has_valid_qualifying_lap", "p_valid_qualifying_lap"),
        ("reaches_q2", "reached_q2", "p_reaches_q2"),
        ("reaches_q3", "reached_q3", "p_reaches_q3"),
    ):
        if target not in rows.columns or probability not in rows.columns:
            continue
        actual = pd.to_numeric(rows[target], errors="coerce")
        predicted = pd.to_numeric(rows[probability], errors="coerce")
        valid = actual.isin([0.0, 1.0]) & predicted.between(0.0, 1.0)
        if not valid.any():
            continue
        y = actual.loc[valid].to_numpy(dtype=float)
        p = np.clip(predicted.loc[valid].to_numpy(dtype=float), 1e-12, 1.0 - 1e-12)
        output[name] = {
            "rows": int(len(y)),
            "base_rate": float(y.mean()),
            "mean_probability": float(p.mean()),
            "brier": float(np.mean(np.square(p - y))),
            "log_loss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
        }
    return output


def _qualifying_inference_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip every target field before a model sees an event."""

    return frame.drop(
        columns=[
            "qualy_position",
            ACTUAL_LAP_COLUMN,
            "has_valid_qualifying_lap",
            "reached_q2",
            "reached_q3",
            Q2_VALID_LAP_COLUMN,
            Q3_VALID_LAP_COLUMN,
            Q1_LAP_COLUMN,
            Q2_LAP_COLUMN,
            Q3_LAP_COLUMN,
        ],
        errors="ignore",
    )


def _freeze_shared_engine_on_selection(
    dataset: pd.DataFrame,
    *,
    partitions: dict[str, tuple[int, ...]],
    seed: int,
) -> tuple[dict[str, Any], Any]:
    """Freeze model family and residual setting before calibration/audit."""

    numeric_events = pd.to_numeric(dataset["event_key"], errors="coerce")
    development = dataset.loc[numeric_events.isin(partitions["development"])].copy()
    development["weak_transfer_prior"] = True
    if not development.empty:
        selection_year = min(partitions["selection"]) // 100
        development["history_weight"] = development["event_key"].map(
            lambda value: max(
                0.03,
                0.30 / max(1, selection_year - (int(value) // 100)),
            )
        )
    event_rows: list[dict[str, Any]] = []
    for event_key in partitions["selection"]:
        current = dataset.loc[numeric_events.eq(int(event_key))].copy()
        prior_keys = tuple(
            value for value in partitions["selection"] if value < int(event_key)
        )
        prior = dataset.loc[numeric_events.isin(prior_keys)].copy()
        prior["weak_transfer_prior"] = False
        prior["history_weight"] = 1.0
        history = pd.concat([development, prior], ignore_index=True, sort=False)
        if history.empty or current.empty:
            continue
        inference = _qualifying_inference_frame(current)
        actual = pd.to_numeric(current["qualy_position"], errors="coerce").to_numpy(
            dtype=float
        )
        predicted: dict[str, np.ndarray] = {}
        for name, robust in (("location", False), ("robust", True)):
            model = train_shared_qualifying_latent_model(
                history,
                target_event_key=int(event_key),
                calibration_event_keys=prior_keys,
                enable_robust_residual=robust,
            )
            point = model.predict_qualifying(
                inference,
                samples=2_000,
                seed=int(seed) + int(event_key),
                allow_diagnostic_stage_fallback=True,
            ).point_order.set_index("driver_id")
            predicted[name] = current["driver_id"].astype(str).map(
                point["predicted_qualifying_position"]
            ).to_numpy(dtype=float)
        baseline = pd.to_numeric(
            current["latest_qualifying_rehearsal_rank"], errors="coerce"
        ).to_numpy(dtype=float)
        finite = (
            np.isfinite(actual)
            & np.isfinite(baseline)
            & np.isfinite(predicted["location"])
            & np.isfinite(predicted["robust"])
        )
        if not finite.any():
            continue
        event_rows.append(
            {
                "event_key": int(event_key),
                "baseline_mae": float(np.mean(np.abs(baseline[finite] - actual[finite]))),
                "location_mae": float(
                    np.mean(np.abs(predicted["location"][finite] - actual[finite]))
                ),
                "robust_mae": float(
                    np.mean(np.abs(predicted["robust"][finite] - actual[finite]))
                ),
                "rows": int(finite.sum()),
            }
        )
    if len(event_rows) < 2:
        raise ValueError("Qualifying selection block has fewer than two scored events")
    robust_delta = np.asarray(
        [row["robust_mae"] - row["location_mae"] for row in event_rows],
        dtype=float,
    )
    leave_one_out = [
        float(np.delete(robust_delta, index).mean())
        for index in range(len(robust_delta))
    ]
    location_mean = float(np.mean([row["location_mae"] for row in event_rows]))
    robust_mean = float(np.mean([row["robust_mae"] for row in event_rows]))
    enable_robust = bool(
        location_mean - robust_mean >= 0.05
        and leave_one_out
        and all(value < 0.0 for value in leave_one_out)
    )
    candidate_column = "robust_mae" if enable_robust else "location_mae"
    selection_event_keys = tuple(int(row["event_key"]) for row in event_rows)
    selection_state = select_frozen_qualifying_model(
        [
            QualifyingModelEvidence(
                model_id="qualifying_rehearsal_rank_baseline_v1",
                mean_absolute_position_error=float(
                    np.mean([row["baseline_mae"] for row in event_rows])
                ),
                event_keys=selection_event_keys,
            ),
            QualifyingModelEvidence(
                model_id="shared_qualifying_latent_lap_v4",
                mean_absolute_position_error=float(
                    np.mean([row[candidate_column] for row in event_rows])
                ),
                event_keys=selection_event_keys,
                promotion_gates_passed=True,
            ),
        ],
        config=FrozenSelectorConfig(challenger_model_id="shared_qualifying_latent_lap_v4"),
    )
    selection_frame = dataset.loc[
        numeric_events.isin(partitions["selection"])
    ].sort_values(["event_key", "driver_id"], kind="mergesort")
    selection_data_sha256 = hashlib.sha256(
        selection_frame.to_csv(index=False, na_rep="<NA>").encode("utf-8")
    ).hexdigest()
    selected_config = {
        "model_id": selection_state.selected_model_id,
        "enable_robust_residual": enable_robust,
    }
    selected_config_sha256 = hashlib.sha256(
        json.dumps(selected_config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return (
        {
            "status": "frozen_before_calibration_and_audit",
            "selected_model_id": selection_state.selected_model_id,
            "decision": selection_state.decision,
            "selected_enable_robust_residual": enable_robust,
            "observed_event_keys": list(selection_state.observed_event_keys),
            "selection_event_keys": list(selection_state.selection_event_keys),
            "baseline_mae": selection_state.baseline_mae,
            "challenger_mae": selection_state.challenger_mae,
            "quality_location_event_mean_mae": location_mean,
            "robust_residual_event_mean_mae": robust_mean,
            "robust_minus_location_leave_one_event_out_deltas": leave_one_out,
            "robust_selection_rule": (
                "at_least_0_05_position_mae_gain_and_all_leave_one_event_out_deltas_negative"
            ),
            "event_evidence": event_rows,
            "selection_data_sha256": selection_data_sha256,
            "selected_config_sha256": selected_config_sha256,
            "calibration_or_audit_outcomes_used": False,
        },
        selection_state,
    )


def _freeze_same_season_engine_from_selection_forecasts(
    event_rows: Sequence[dict[str, Any]],
    *,
    selection_frame: pd.DataFrame,
) -> tuple[dict[str, Any], Any]:
    """Freeze residual family and public point model on the R3-R4 forecasts.

    ``event_rows`` must be built from forecasts that existed before the
    corresponding target was opened. The two residual candidates are first
    reduced to one configuration using their direct-pace heads. The direct and
    minimum-expected-absolute-loss point heads are then compared for that
    residual family, and the fully locked challenger is compared with the
    retained rehearsal-rank baseline. No R5+ outcome enters either choice.
    """

    ordered = tuple(sorted(event_rows, key=lambda row: int(row["event_key"])))
    if len(ordered) != 2:
        raise ValueError("same-season Qualifying selection requires exactly two events")
    selection_event_keys = tuple(int(row["event_key"]) for row in ordered)
    if len(set(selection_event_keys)) != len(selection_event_keys):
        raise ValueError("same-season Qualifying selection event keys must be unique")
    required = {
        "baseline_mae",
        "location_mae",
        "location_meal_mae",
        "robust_mae",
        "robust_meal_mae",
        "baseline_forecast_artifact_sha256",
        "location_forecast_artifact_sha256",
        "robust_forecast_artifact_sha256",
    }
    for row in ordered:
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"selection evidence is missing fields: {missing}")
        values = [
            float(row[column])
            for column in (
                "baseline_mae",
                "location_mae",
                "location_meal_mae",
                "robust_mae",
                "robust_meal_mae",
            )
        ]
        if not np.isfinite(values).all():
            raise ValueError("selection evidence must be finite")
        for name in (
            "baseline_forecast_artifact_sha256",
            "location_forecast_artifact_sha256",
            "robust_forecast_artifact_sha256",
        ):
            if re.fullmatch(r"[0-9a-f]{64}", str(row[name])) is None:
                raise ValueError(f"selection evidence has an invalid {name}")
        if not bool(row.get("forecast_frozen_before_target_read", False)):
            raise ValueError(
                "selection evidence is invalid unless its forecast was frozen before target I/O"
            )

    robust_delta = np.asarray(
        [float(row["robust_mae"]) - float(row["location_mae"]) for row in ordered],
        dtype=float,
    )
    leave_one_out = [
        float(np.delete(robust_delta, index).mean())
        for index in range(len(robust_delta))
    ]
    location_mean = float(np.mean([float(row["location_mae"]) for row in ordered]))
    robust_mean = float(np.mean([float(row["robust_mae"]) for row in ordered]))
    enable_robust = bool(
        location_mean - robust_mean >= 0.05
        and leave_one_out
        and all(value < 0.0 for value in leave_one_out)
    )
    candidate_column = "robust_mae" if enable_robust else "location_mae"
    meal_candidate_column = (
        "robust_meal_mae" if enable_robust else "location_meal_mae"
    )
    direct_pace_mean = float(
        np.mean([float(row[candidate_column]) for row in ordered])
    )
    minimum_loss_mean = float(
        np.mean([float(row[meal_candidate_column]) for row in ordered])
    )
    minimum_loss_delta = np.asarray(
        [
            float(row[meal_candidate_column]) - float(row[candidate_column])
            for row in ordered
        ],
        dtype=float,
    )
    point_head_leave_one_out = [
        float(np.delete(minimum_loss_delta, index).mean())
        for index in range(len(minimum_loss_delta))
    ]
    selected_point_head = (
        POINT_HEAD_MINIMUM_EXPECTED_ABSOLUTE_LOSS
        if direct_pace_mean - minimum_loss_mean >= 0.05
        and point_head_leave_one_out
        and all(value < 0.0 for value in point_head_leave_one_out)
        else POINT_HEAD_DIRECT_PACE
    )
    baseline_mean = float(np.mean([float(row["baseline_mae"]) for row in ordered]))
    challenger_mean = (
        minimum_loss_mean
        if selected_point_head == POINT_HEAD_MINIMUM_EXPECTED_ABSOLUTE_LOSS
        else direct_pace_mean
    )
    selection_contract_gates_passed = bool(
        all(bool(row.get("selection_contract_gates_passed", False)) for row in ordered)
    )
    selection_state = select_frozen_qualifying_model(
        [
            QualifyingModelEvidence(
                model_id="qualifying_rehearsal_rank_baseline_v1",
                mean_absolute_position_error=baseline_mean,
                event_keys=selection_event_keys,
            ),
            QualifyingModelEvidence(
                model_id="shared_qualifying_latent_lap_v4",
                mean_absolute_position_error=challenger_mean,
                event_keys=selection_event_keys,
                promotion_gates_passed=selection_contract_gates_passed,
            ),
        ],
        config=FrozenSelectorConfig(
            baseline_model_id="qualifying_rehearsal_rank_baseline_v1",
            challenger_model_id="shared_qualifying_latent_lap_v4",
            minimum_evidence_events=2,
            freeze_for_new_events=4,
            minimum_mae_improvement=0.15,
        ),
    )
    canonical_selection = selection_frame.sort_values(
        ["event_key", "driver_id"], kind="mergesort"
    )
    selection_data_sha256 = hashlib.sha256(
        canonical_selection.to_csv(index=False, na_rep="<NA>").encode("utf-8")
    ).hexdigest()
    selected_config = {
        "model_id": selection_state.selected_model_id,
        "enable_robust_residual": enable_robust,
        "point_head": selected_point_head,
    }
    selected_config_sha256 = hashlib.sha256(
        json.dumps(selected_config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return (
        {
            "status": "frozen_after_round_4_before_calibration_and_audit",
            "selected_model_id": selection_state.selected_model_id,
            "decision": selection_state.decision,
            "selected_enable_robust_residual": enable_robust,
            "selected_challenger_variant": "robust" if enable_robust else "location",
            "selected_point_head": selected_point_head,
            "observed_event_keys": list(selection_state.observed_event_keys),
            "selection_event_keys": list(selection_state.selection_event_keys),
            "baseline_mae": selection_state.baseline_mae,
            "challenger_mae": selection_state.challenger_mae,
            "quality_location_event_mean_mae": location_mean,
            "robust_residual_event_mean_mae": robust_mean,
            "robust_minus_location_leave_one_event_out_deltas": leave_one_out,
            "robust_selection_rule": (
                "at_least_0_05_position_mae_gain_and_all_leave_one_event_out_deltas_negative"
            ),
            "direct_pace_point_head_event_mean_mae": direct_pace_mean,
            "minimum_expected_absolute_loss_point_head_event_mean_mae": (
                minimum_loss_mean
            ),
            "minimum_loss_minus_direct_leave_one_event_out_deltas": (
                point_head_leave_one_out
            ),
            "point_head_selection_rule": (
                "meal_requires_at_least_0_05_position_mae_gain_and_all_"
                "leave_one_event_out_deltas_negative_else_direct_pace"
            ),
            "point_head_candidates_frozen_before_target_read": True,
            "material_challenger_gain_required_positions_mae": 0.15,
            "selection_contract_gates_passed": selection_contract_gates_passed,
            "event_evidence": [dict(row) for row in ordered],
            "selection_data_sha256": selection_data_sha256,
            "selected_config_sha256": selected_config_sha256,
            "calibration_or_audit_outcomes_used": False,
        },
        selection_state,
    )


def _qualifying_contract_gates(
    predictions: pd.DataFrame,
    *,
    event_info: dict[int, dict[str, Any]],
    event_keys: Sequence[int],
) -> dict[str, Any]:
    """Validate entrant coverage, legal permutations, and probability scope."""

    evidence: list[dict[str, Any]] = []
    numeric_events = pd.to_numeric(predictions["event_key"], errors="coerce")
    for event_key in sorted({int(value) for value in event_keys}):
        group = predictions.loc[numeric_events.eq(event_key)]
        expected = int(event_info[event_key]["field_size"])
        official_target_ids = set(event_info[event_key]["official_target_driver_ids"])
        drivers = group["driver_id"].astype(str)
        predicted_ids = set(drivers.tolist())
        positions = pd.to_numeric(
            group["predicted_qualifying_position"], errors="coerce"
        )
        probability_columns = [
            f"p_position_{position}" for position in range(1, expected + 1)
        ]
        matrix_legal = False
        if all(column in group.columns for column in probability_columns) and len(group) == expected:
            matrix = group[probability_columns].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=float)
            matrix_legal = bool(
                np.isfinite(matrix).all()
                and np.allclose(matrix.sum(axis=1), 1.0, atol=1e-9)
                and np.allclose(matrix.sum(axis=0), 1.0, atol=1e-9)
            )
        evidence.append(
            {
                "event_key": event_key,
                "expected_pre_q_entrants": expected,
                "official_target_entrants": int(len(official_target_ids)),
                "predicted_unique_entrants": int(drivers.nunique()),
                "entrant_coverage_complete": bool(
                    len(group) == expected and drivers.nunique() == expected
                ),
                "official_target_entrant_coverage_complete": bool(
                    predicted_ids == official_target_ids
                ),
                "missing_official_target_driver_ids": sorted(
                    official_target_ids - predicted_ids
                ),
                "extra_pre_q_driver_ids": sorted(predicted_ids - official_target_ids),
                "legal_full_field_point_permutation": bool(
                    sorted(positions.dropna().astype(int).tolist())
                    == list(range(1, expected + 1))
                ),
                "legal_full_field_probability_matrix": matrix_legal,
            }
        )
    coverage = bool(
        evidence
        and all(
            row["entrant_coverage_complete"]
            and row["official_target_entrant_coverage_complete"]
            for row in evidence
        )
    )
    legal = bool(
        evidence
        and all(
            row["legal_full_field_point_permutation"]
            and row["legal_full_field_probability_matrix"]
            for row in evidence
        )
    )
    calibrated = bool(
        len(predictions)
        and predictions.get(
            "position_marginals_calibrated",
            pd.Series(False, index=predictions.index),
        ).fillna(False).astype(bool).all()
    )
    fail_closed = bool(
        not calibrated
        and predictions.get(
            "probability_calibration_status",
            pd.Series("", index=predictions.index),
        ).isin(
            {
                "uncalibrated_joint_latent_samples",
                "unavailable_diagnostic_joint_stage_model",
            }
        ).all()
    )
    return {
        "pre_q_entrant_coverage_is_100_percent": coverage,
        "every_event_is_legal_full_field_permutation": legal,
        "position_probabilities_calibrated": calibrated,
        "position_probability_outputs_promoted": calibrated,
        "uncalibrated_probability_outputs_fail_closed": fail_closed,
        "probability_contract_satisfied": bool(calibrated or fail_closed),
        "point_contract_gates_passed": bool(
            coverage and legal and (calibrated or fail_closed)
        ),
        "event_evidence": evidence,
    }


def _build_shared_event_forecast(
    history: pd.DataFrame,
    inference: pd.DataFrame,
    *,
    target_event_key: int,
    interval_calibration_predictions: pd.DataFrame | None = None,
    enable_robust_residual: bool = SHARED_QUALIFYING_ENABLE_ROBUST_RESIDUAL,
    point_head: str = POINT_HEAD_DIRECT_PACE,
):
    """Qualifying runner adapter to the cross-mode frozen forecast path."""

    return build_shared_qualifying_event_forecast(
        history,
        inference,
        target_event_key=int(target_event_key),
        interval_calibration_predictions=interval_calibration_predictions,
        enable_robust_residual=bool(enable_robust_residual),
        point_head=str(point_head),
    )


def _baseline_event_forecast(
    inference: pd.DataFrame,
    *,
    target_event_key: int,
    phase: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Freeze the retained rehearsal-rank forecast without opening a target."""

    required = {"driver_id", "latest_qualifying_rehearsal_rank"}
    missing = sorted(required.difference(inference.columns))
    if missing:
        raise ValueError(f"baseline Qualifying forecast is missing columns: {missing}")
    if inference.empty or inference["driver_id"].astype(str).duplicated().any():
        raise ValueError("baseline Qualifying inference requires a non-empty unique driver field")
    ordered = pd.DataFrame(
        {
            "driver_id": inference["driver_id"].astype(str),
            "anchor_rank": pd.to_numeric(
                inference["latest_qualifying_rehearsal_rank"], errors="coerce"
            ),
            "row": np.arange(len(inference)),
        },
        index=inference.index,
    ).sort_values(["anchor_rank", "driver_id", "row"], kind="mergesort")
    if not np.isfinite(ordered["anchor_rank"]).all():
        raise ValueError("baseline Qualifying rehearsal ranks must be finite")
    rank = pd.Series(np.arange(1, len(ordered) + 1), index=ordered.index)
    point = pd.DataFrame(
        {
            "event_key": int(target_event_key),
            "driver_id": inference["driver_id"].astype(str),
            "predicted_qualifying_position": rank.reindex(inference.index).astype(int),
        }
    )
    field_size = len(point)
    for position in range(1, field_size + 1):
        point[f"p_position_{position}"] = (
            point["predicted_qualifying_position"].eq(position).astype(float)
        )
    point["expected_qualifying_position"] = point[
        "predicted_qualifying_position"
    ].astype(float)
    point["pole_probability"] = point["p_position_1"]
    point["top3_probability"] = point[
        [f"p_position_{value}" for value in range(1, min(3, field_size) + 1)]
    ].sum(axis=1)
    point["position_marginals_calibrated"] = False
    point["probability_calibration_status"] = (
        "unavailable_diagnostic_joint_stage_model"
    )
    payload: dict[str, object] = {
        "schema_version": "f1_qualifying_frozen_baseline_forecast_v1",
        "event_key": int(target_event_key),
        "phase": str(phase),
        "driver_ids": point["driver_id"].tolist(),
        "predicted_positions": point["predicted_qualifying_position"].tolist(),
        "target_columns_present": sorted(
            set(point.columns).intersection(
                {
                    "actual_qualifying_position",
                    "qualy_position",
                    ACTUAL_LAP_COLUMN,
                }
            )
        ),
    }
    payload["artifact_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return point, payload


def _event_actual_columns(current: pd.DataFrame) -> pd.DataFrame:
    return current[
        [
            "driver_id",
            "qualy_position",
            "has_valid_qualifying_lap",
            "reached_q2",
            "reached_q3",
            Q2_VALID_LAP_COLUMN,
            Q3_VALID_LAP_COLUMN,
            Q1_LAP_COLUMN,
            Q2_LAP_COLUMN,
            Q3_LAP_COLUMN,
            ACTUAL_LAP_COLUMN,
            "latest_qualifying_rehearsal_rank",
        ]
    ].rename(
        columns={
            "qualy_position": "actual_qualifying_position",
            "latest_qualifying_rehearsal_rank": "baseline_rank_prior",
        }
    )


def _score_baseline_only_event(
    baseline_point: pd.DataFrame,
    current: pd.DataFrame,
    *,
    artifact: dict[str, object],
    phase: str,
) -> pd.DataFrame:
    scored = baseline_point.merge(
        _event_actual_columns(current), on="driver_id", how="left", validate="one_to_one"
    )
    scored["pairwise_comparator_position"] = scored["baseline_rank_prior"]
    scored["pairwise_comparator_status"] = "baseline_substitute_no_fitted_candidate"
    scored["selected_predicted_qualifying_position"] = scored[
        "baseline_rank_prior"
    ]
    scored["selected_output_model_id"] = QUALIFYING_BASELINE_MODEL_ID
    scored["qualifying_model"] = QUALIFYING_BASELINE_MODEL_ID
    scored["forecast_phase"] = str(phase)
    scored["forecast_frozen_before_target_read"] = True
    scored["frozen_forecast_artifact_sha256"] = artifact["artifact_sha256"]
    scored["candidate_availability"] = "not_fit_until_point_fit_partition_complete"
    for column in (
        "p_valid_qualifying_lap",
        "p_q2_given_valid",
        "p_q3_given_q2",
        "p_reaches_q2",
        "p_reaches_q3",
    ):
        scored[column] = np.nan
    return scored


def _score_shared_event(
    forecast: Any,
    current: pd.DataFrame,
    *,
    artifact: dict[str, object],
    selected_model_id: str,
    selected_enable_robust_residual: bool,
    selected_point_head: str,
    phase: str,
) -> pd.DataFrame:
    shared = forecast.point_order.copy()
    if selected_point_head not in {
        POINT_HEAD_DIRECT_PACE,
        POINT_HEAD_MINIMUM_EXPECTED_ABSOLUTE_LOSS,
    }:
        raise ValueError(f"unsupported selected Qualifying point head: {selected_point_head}")
    point_head_column = (
        MEAL_POSITION_COLUMN
        if selected_point_head == POINT_HEAD_MINIMUM_EXPECTED_ABSOLUTE_LOSS
        else DIRECT_PACE_POSITION_COLUMN
    )
    if point_head_column not in shared.columns:
        raise ValueError(f"frozen Qualifying forecast is missing {point_head_column}")
    shared["predicted_qualifying_position"] = shared[point_head_column].astype(int)
    shared["point_prediction_source"] = str(selected_point_head)
    shared["point_head_uses_uncalibrated_joint_samples"] = bool(
        selected_point_head == POINT_HEAD_MINIMUM_EXPECTED_ABSOLUTE_LOSS
    )
    shared["point_tie_policy"] = (
        "hungarian_expected_absolute_loss_then_seeded_exchangeable_optimal_tie"
        if selected_point_head == POINT_HEAD_MINIMUM_EXPECTED_ABSOLUTE_LOSS
        else "sampled_expected_official_position_then_seeded_exchangeable_uncertainty"
    )
    shared["shared_forecast_artifact_sha256"] = artifact["artifact_sha256"]
    shared["shared_joint_samples_sha256"] = artifact["joint_samples_sha256"]
    scored = shared.merge(
        _event_actual_columns(current), on="driver_id", how="left", validate="one_to_one"
    )
    scored["pairwise_comparator_position"] = scored["baseline_rank_prior"]
    scored["pairwise_comparator_status"] = (
        "unavailable_two_structural_fit_events_baseline_substitute"
    )
    scored["selected_predicted_qualifying_position"] = (
        scored["predicted_qualifying_position"]
        if selected_model_id == QUALIFYING_CHALLENGER_MODEL_ID
        else scored["baseline_rank_prior"]
    )
    scored["selected_output_model_id"] = str(selected_model_id)
    scored["selected_enable_robust_residual"] = bool(
        selected_enable_robust_residual
    )
    scored["selected_point_head"] = str(selected_point_head)
    scored["candidate_availability"] = "available"
    scored["forecast_phase"] = str(phase)
    scored["forecast_frozen_before_target_read"] = True
    scored["p_valid_qualifying_lap"] = scored["valid_lap_probability"]
    scored["p_q2_given_valid"] = scored["q2_given_valid_probability"]
    scored["p_q3_given_q2"] = scored["q3_given_q2_probability"]
    scored["p_reaches_q2"] = (
        scored["valid_lap_probability"] * scored["q2_given_valid_probability"]
    )
    scored["p_reaches_q3"] = (
        scored["p_reaches_q2"] * scored["q3_given_q2_probability"]
    )
    return scored


def _apply_public_output_decision(
    predictions: pd.DataFrame,
    *,
    selection_block_winner_model_id: str,
    point_model_promoted: bool,
) -> tuple[pd.DataFrame, str]:
    """Separate the research-selected challenger from the retained public output."""

    scored = predictions.copy()
    research_position = pd.to_numeric(
        scored["selected_predicted_qualifying_position"], errors="raise"
    ).astype(int)
    research_model = scored["selected_output_model_id"].astype(str)
    scored["research_selected_predicted_qualifying_position"] = research_position
    scored["research_selected_output_model_id"] = research_model
    scored["selection_block_winner_model_id"] = str(
        selection_block_winner_model_id
    )

    public_model_id = (
        str(selection_block_winner_model_id)
        if point_model_promoted
        else QUALIFYING_BASELINE_MODEL_ID
    )
    if point_model_promoted:
        public_position = research_position
        public_row_model = research_model
    else:
        public_position = pd.to_numeric(
            scored["baseline_rank_prior"], errors="raise"
        ).astype(int)
        public_row_model = pd.Series(
            QUALIFYING_BASELINE_MODEL_ID,
            index=scored.index,
            dtype=object,
        )
    scored["public_output_predicted_qualifying_position"] = public_position
    scored["public_output_model_id"] = public_row_model
    # Backward-compatible selected-output fields now mean the actual retained
    # public output, never a rejected research challenger.
    scored["selected_predicted_qualifying_position"] = public_position
    scored["selected_output_model_id"] = public_row_model
    return scored, public_model_id


def _mean_absolute_position_error(
    point: pd.DataFrame,
    current: pd.DataFrame,
    *,
    prediction_column: str = "predicted_qualifying_position",
) -> float:
    actual = current.set_index("driver_id")["qualy_position"]
    if prediction_column not in point.columns:
        raise ValueError(f"Qualifying point forecast is missing {prediction_column}")
    predicted = point.set_index("driver_id")[prediction_column]
    aligned = pd.concat([predicted.rename("predicted"), actual.rename("actual")], axis=1)
    aligned = aligned.apply(pd.to_numeric, errors="coerce").dropna()
    if aligned.empty:
        raise ValueError("no complete driver rows for Qualifying selection score")
    return float((aligned["predicted"] - aligned["actual"]).abs().mean())


def _validate_same_season_cli_scope(
    years: Sequence[int],
    evaluation_years: Sequence[int],
) -> int:
    loaded = tuple(sorted({int(value) for value in years}))
    evaluated = tuple(sorted({int(value) for value in evaluation_years}))
    if len(loaded) != 1 or len(evaluated) != 1 or loaded != evaluated:
        raise ValueError(
            "same-season Qualifying mode requires one identical --years and "
            "--evaluation-years value; use --legacy-cross-season for transfer diagnostics"
        )
    return loaded[0]


def _build_locked_final_fit_history(
    selection_fit_history: pd.DataFrame,
    selection_outcomes: pd.DataFrame,
    *,
    partitions: dict[str, tuple[int, ...]],
) -> tuple[pd.DataFrame, tuple[int, ...]]:
    """Refit a selected configuration on exactly R1-R4, never R5+."""

    required_roles = {"point_fit", "selection"}
    missing_roles = sorted(required_roles.difference(partitions))
    if missing_roles:
        raise ValueError(f"final-fit partitions are missing roles: {missing_roles}")
    final_fit_history = pd.concat(
        [selection_fit_history, selection_outcomes], ignore_index=True, sort=False
    )
    if "event_key" not in final_fit_history.columns or final_fit_history.empty:
        raise ValueError("final Qualifying fit requires labelled R1-R4 event rows")
    final_fit_history["weak_transfer_prior"] = False
    final_fit_history["history_weight"] = 1.0
    final_fit_event_keys = tuple(
        sorted(
            pd.to_numeric(final_fit_history["event_key"], errors="raise")
            .astype(int)
            .unique()
        )
    )
    expected = tuple((*partitions["point_fit"], *partitions["selection"]))
    if final_fit_event_keys != expected:
        raise ValueError(
            "final Qualifying fit must contain exactly the locked R1-R4 events; "
            f"expected={list(expected)}, observed={list(final_fit_event_keys)}"
        )
    return final_fit_history, final_fit_event_keys


def _run_same_season(
    *,
    weekends_dir: Path,
    years: Sequence[int],
    evaluation_years: Sequence[int],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Run the canonical current-season-only Qualifying evidence protocol."""

    root = _root()
    target_year = _validate_same_season_cli_scope(years, evaluation_years)
    inference_frames: list[pd.DataFrame] = []
    infos: dict[int, dict[str, Any]] = {}
    target_paths: dict[int, Path] = {}
    inputs: set[Path] = set()
    skipped_no_causal_rehearsal: list[str] = []
    for event_dir in sorted(
        (weekends_dir / str(target_year)).glob("round_*"), key=_round_number
    ):
        try:
            frame, info, event_inputs, target_path = _event_inference_frame(root, event_dir)
        except ValueError as error:
            if "no causal FP3/Sprint-Qualifying" not in str(error):
                raise
            skipped_no_causal_rehearsal.append(str(event_dir.relative_to(root)))
            continue
        event_key = int(info["event_key"])
        if event_key // 100 != target_year:
            raise RuntimeError("same-season loader observed a foreign-season event")
        inference_frames.append(frame)
        infos[event_key] = info
        target_paths[event_key] = target_path
        inputs.update(event_inputs)
    if not inference_frames:
        raise ValueError(f"no local qualifying events found for {target_year}")
    inference_dataset = pd.concat(inference_frames, ignore_index=True)
    numeric_events = pd.to_numeric(inference_dataset["event_key"], errors="raise")
    if not numeric_events.floordiv(100).eq(target_year).all():
        raise RuntimeError("same-season inference dataset contains prior-season rows")
    partitions = _locked_event_partitions(
        tuple(infos), audit_year=target_year, same_season_only=True
    )
    feature_columns = tuple(
        column for column in FEATURE_ALLOWLIST if column in inference_dataset.columns
    )

    scored_frames: list[pd.DataFrame] = []
    point_fit_labeled: list[pd.DataFrame] = []
    target_read_log: list[dict[str, object]] = []

    def open_target(
        event_key: int,
        artifact: dict[str, object],
        *,
        phase: str,
        additional_frozen_artifacts: Sequence[dict[str, object]] = (),
    ) -> pd.DataFrame:
        frozen_artifacts = (artifact, *tuple(additional_frozen_artifacts))
        frozen_hashes: list[str] = []
        for frozen in frozen_artifacts:
            frozen_hash = str(frozen.get("artifact_sha256") or "")
            try:
                matching_event = int(frozen.get("event_key")) == int(event_key)
            except (TypeError, ValueError):
                matching_event = False
            if re.fullmatch(r"[0-9a-f]{64}", frozen_hash) is None or not matching_event:
                raise RuntimeError(
                    "Qualifying target read requires every declared forecast to be "
                    "frozen for the same event"
                )
            frozen_hashes.append(frozen_hash)
        target, target_info = _load_target_after_frozen_forecast(
            target_paths[event_key],
            expected_event_key=event_key,
            frozen_forecast_artifact=artifact,
        )
        infos[event_key].update(target_info)
        target_read_log.append(
            {
                "sequence": len(target_read_log) + 1,
                "event_key": event_key,
                "phase": phase,
                "forecast_artifact_sha256": artifact["artifact_sha256"],
                "all_frozen_forecast_artifact_sha256s": frozen_hashes,
                "forecast_frozen_before_target_read": True,
            }
        )
        current = inference_dataset.loc[numeric_events.eq(event_key)].copy()
        return _attach_qualifying_target(current, target)

    # R1-R2: freeze the only causally available baseline before each result is
    # opened, then use these two targets exclusively for structural point fit.
    for event_key in partitions["point_fit"]:
        inference = _qualifying_inference_frame(
            inference_dataset.loc[numeric_events.eq(event_key)].copy()
        )
        baseline_point, baseline_artifact = _baseline_event_forecast(
            inference, target_event_key=event_key, phase="point_fit"
        )
        current = open_target(event_key, baseline_artifact, phase="point_fit")
        point_fit_labeled.append(current)
        scored_frames.append(
            _score_baseline_only_event(
                baseline_point, current, artifact=baseline_artifact, phase="point_fit"
            )
        )
    structural_history = pd.concat(point_fit_labeled, ignore_index=True, sort=False)
    structural_history["weak_transfer_prior"] = False
    structural_history["history_weight"] = 1.0
    structural_event_keys = tuple(
        sorted(pd.to_numeric(structural_history["event_key"], errors="raise").unique())
    )
    if structural_event_keys != partitions["point_fit"]:
        raise RuntimeError("structural training escaped the locked R1-R2 point-fit partition")

    # R3-R4: both residual variants are forecast and hashed before each target
    # read. Their matched event errors select one challenger variant and then
    # the fail-closed baseline/challenger selector is frozen.
    selection_records: list[dict[str, Any]] = []
    selection_labeled: list[pd.DataFrame] = []
    selection_forecasts: dict[int, dict[str, tuple[Any, dict[str, object]]]] = {}
    for event_key in partitions["selection"]:
        inference = _qualifying_inference_frame(
            inference_dataset.loc[numeric_events.eq(event_key)].copy()
        )
        variants: dict[str, tuple[Any, dict[str, object]]] = {}
        for name, robust in (("location", False), ("robust", True)):
            _, forecast, artifact = _build_shared_event_forecast(
                structural_history,
                inference,
                target_event_key=event_key,
                interval_calibration_predictions=None,
                enable_robust_residual=robust,
            )
            variants[name] = (forecast, artifact)
        baseline_point, baseline_artifact = _baseline_event_forecast(
            inference, target_event_key=event_key, phase="model_selection"
        )
        current = open_target(
            event_key,
            variants["location"][1],
            phase="model_selection",
            additional_frozen_artifacts=(
                baseline_artifact,
                variants["robust"][1],
            ),
        )
        selection_labeled.append(current)
        selection_records.append(
            {
                "event_key": event_key,
                "baseline_mae": _mean_absolute_position_error(baseline_point, current),
                "location_mae": _mean_absolute_position_error(
                    variants["location"][0].point_order,
                    current,
                    prediction_column=DIRECT_PACE_POSITION_COLUMN,
                ),
                "location_meal_mae": _mean_absolute_position_error(
                    variants["location"][0].point_order,
                    current,
                    prediction_column=MEAL_POSITION_COLUMN,
                ),
                "robust_mae": _mean_absolute_position_error(
                    variants["robust"][0].point_order,
                    current,
                    prediction_column=DIRECT_PACE_POSITION_COLUMN,
                ),
                "robust_meal_mae": _mean_absolute_position_error(
                    variants["robust"][0].point_order,
                    current,
                    prediction_column=MEAL_POSITION_COLUMN,
                ),
                "location_forecast_artifact_sha256": variants["location"][1][
                    "artifact_sha256"
                ],
                "robust_forecast_artifact_sha256": variants["robust"][1][
                    "artifact_sha256"
                ],
                "baseline_forecast_artifact_sha256": baseline_artifact[
                    "artifact_sha256"
                ],
                "selection_contract_gates_passed": bool(
                    set(inference["driver_id"].astype(str))
                    == set(infos[event_key]["official_target_driver_ids"])
                    and len(inference)
                    == int(infos[event_key]["official_target_driver_count"])
                ),
                "forecast_frozen_before_target_read": True,
            }
        )
        selection_forecasts[event_key] = variants
    selection_frame = pd.concat(selection_labeled, ignore_index=True, sort=False)
    frozen_selection, selection = _freeze_same_season_engine_from_selection_forecasts(
        selection_records,
        selection_frame=selection_frame,
    )
    enable_selected_residual = bool(
        frozen_selection["selected_enable_robust_residual"]
    )
    selected_variant = "robust" if enable_selected_residual else "location"
    selected_model_id = str(selection.selected_model_id)
    selected_point_head = str(frozen_selection["selected_point_head"])

    # Model family and robust/location configuration are now locked. Refit that
    # fixed structure on all causally available R1-R4 outcomes before creating
    # any R5 forecast. R3-R4 can increase estimation sample size only after
    # selection; they cannot retroactively change the selected configuration.
    final_fit_history, final_fit_event_keys = _build_locked_final_fit_history(
        structural_history,
        selection_frame,
        partitions=partitions,
    )

    shared_forecast_artifacts: list[dict[str, object]] = []
    for current in selection_labeled:
        event_key = int(pd.to_numeric(current["event_key"], errors="raise").iloc[0])
        forecast, artifact = selection_forecasts[event_key][selected_variant]
        shared_forecast_artifacts.append(artifact)
        scored_frames.append(
            _score_shared_event(
                forecast,
                current,
                artifact=artifact,
                selected_model_id=selected_model_id,
                selected_enable_robust_residual=enable_selected_residual,
                selected_point_head=selected_point_head,
                phase="model_selection",
            )
        )

    # R5-R6: create raw, untouched forecasts with the R1-R4-refit point model.
    # This runner uses their outcomes for interval calibration only. Their raw
    # position marginals are retained as inputs to the separate probability
    # calibration diagnostic, but no probability transform is attached here.
    calibration_rows: list[pd.DataFrame] = []
    for event_key in partitions["calibration"]:
        inference = _qualifying_inference_frame(
            inference_dataset.loc[numeric_events.eq(event_key)].copy()
        )
        _, forecast, artifact = _build_shared_event_forecast(
            final_fit_history,
            inference,
            target_event_key=event_key,
            interval_calibration_predictions=None,
            enable_robust_residual=enable_selected_residual,
            point_head=selected_point_head,
        )
        current = open_target(
            event_key,
            artifact,
            phase="interval_calibration_probability_diagnostic_holdout",
        )
        shared_forecast_artifacts.append(artifact)
        scored_frames.append(
            _score_shared_event(
                forecast,
                current,
                artifact=artifact,
                selected_model_id=selected_model_id,
                selected_enable_robust_residual=enable_selected_residual,
                selected_point_head=selected_point_head,
                phase="interval_calibration_probability_diagnostic_holdout",
            )
        )
        calibration = forecast.lap_predictions[
            ["event_key", "driver_id", "rehearsal_source", "lap_p50"]
        ].copy()
        calibration[ACTUAL_LAP_COLUMN] = calibration["driver_id"].map(
            current.set_index("driver_id")[ACTUAL_LAP_COLUMN]
        )
        calibration_rows.append(
            calibration.dropna(subset=["lap_p50", ACTUAL_LAP_COLUMN])
        )
    held_out_interval_rows = pd.concat(calibration_rows, ignore_index=True)

    # R7+: final untouched audit. The selected configuration and R1-R4 final
    # point fit remain frozen; only the R5-R6 interval calibration is applied.
    for event_key in partitions["audit"]:
        inference = _qualifying_inference_frame(
            inference_dataset.loc[numeric_events.eq(event_key)].copy()
        )
        _, forecast, artifact = _build_shared_event_forecast(
            final_fit_history,
            inference,
            target_event_key=event_key,
            interval_calibration_predictions=held_out_interval_rows,
            enable_robust_residual=enable_selected_residual,
            point_head=selected_point_head,
        )
        current = open_target(event_key, artifact, phase="audit")
        shared_forecast_artifacts.append(artifact)
        scored_frames.append(
            _score_shared_event(
                forecast,
                current,
                artifact=artifact,
                selected_model_id=selected_model_id,
                selected_enable_robust_residual=enable_selected_residual,
                selected_point_head=selected_point_head,
                phase="audit",
            )
        )

    scored_predictions = pd.concat(scored_frames, ignore_index=True, sort=False)
    events, rows = _metric_rows(scored_predictions, infos)
    audit_events = [
        event for event in events if int(event["event_key"]) in set(partitions["audit"])
    ]
    if not audit_events:
        raise ValueError("same-season Qualifying audit partition produced no scored events")
    paired = [
        EventError(
            event_key=str(event["event_key"]),
            baseline_error=float(event["baseline_mae"]),
            candidate_error=float(event["candidate_mae"]),
            stratum=(
                "sprint"
                if "sprint" in str(event["event_format"]).lower()
                else "standard"
            ),
        )
        for event in audit_events
    ]
    exceptional_count = min(4, max(1, len(audit_events) // 4))
    retained_tail = sorted(
        audit_events, key=lambda value: float(value["baseline_mae"])
    )[:-exceptional_count]
    if not retained_tail:
        retained_tail = audit_events
    tail_delta = _mean(retained_tail, "candidate_mae") - _mean(
        retained_tail, "baseline_mae"
    )
    promotion = evaluate_qualifying_promotion(
        paired,
        baseline_kendall=_mean(audit_events, "baseline_kendall"),
        candidate_kendall=_mean(audit_events, "candidate_kendall"),
        pole_non_regression=_mean(audit_events, "candidate_pole_hit")
        >= _mean(audit_events, "baseline_pole_hit"),
        top3_non_regression=_mean(audit_events, "candidate_top3_overlap")
        >= _mean(audit_events, "baseline_top3_overlap"),
        top10_non_regression=_mean(audit_events, "candidate_top10_overlap")
        >= _mean(audit_events, "baseline_top10_overlap"),
        tail_excluded_delta=tail_delta,
        bootstrap_samples=int(bootstrap_samples),
        seed=int(seed),
    )
    audit_event_keys = tuple(int(event["event_key"]) for event in audit_events)
    contract_gates = _qualifying_contract_gates(
        scored_predictions,
        event_info=infos,
        event_keys=audit_event_keys,
    )
    preselected_challenger = selected_model_id == QUALIFYING_CHALLENGER_MODEL_ID
    point_promoted = bool(
        promotion.promoted
        and contract_gates["point_contract_gates_passed"]
        and preselected_challenger
    )
    promotion_payload = promotion.to_payload()
    promotion_reasons = list(promotion_payload["reasons"])
    if not preselected_challenger:
        promotion_reasons.append("gate_failed:challenger_not_selected_on_rounds_3_4")
    if not contract_gates["pre_q_entrant_coverage_is_100_percent"]:
        promotion_reasons.append("gate_failed:pre_q_entrant_coverage_is_100_percent")
    if not contract_gates["every_event_is_legal_full_field_permutation"]:
        promotion_reasons.append("gate_failed:every_event_is_legal_full_field_permutation")
    public_output_model_id = (
        selected_model_id if point_promoted else QUALIFYING_BASELINE_MODEL_ID
    )
    promotion_payload.update(
        {
            "promoted": point_promoted,
            "status": "promoted" if point_promoted else "rejected",
            "reasons": promotion_reasons,
            "point_model_promoted": point_promoted,
            "selection_block_winner_model_id": selected_model_id,
            "selected_public_output_model_id": public_output_model_id,
            "retained_public_output_model_id": public_output_model_id,
            "selected_challenger_point_head": selected_point_head,
            "position_probability_outputs_promoted": False,
            "mode_contract_gates": contract_gates,
        }
    )
    scored_predictions, public_output_model_id = _apply_public_output_decision(
        scored_predictions,
        selection_block_winner_model_id=selected_model_id,
        point_model_promoted=point_promoted,
    )
    events, rows = _metric_rows(scored_predictions, infos)
    audit_events = [
        event for event in events if int(event["event_key"]) in set(partitions["audit"])
    ]
    implementation_paths = [
        Path(__file__).resolve(),
        root / "packages/f1/features/qualifying_lap.py",
        root / "packages/f1/models/pre_quali/pairwise.py",
        root / "packages/f1/models/pre_quali/classification.py",
        root / "packages/f1/models/pre_quali/train.py",
        root / "packages/f1/models/pre_quali/selection.py",
        root / "packages/f1/models/ultimate_lap_time/achievable.py",
        root / "packages/f1/orchestration/non_live_validation.py",
    ]
    candidate_available_events = [
        event for event in events if bool(event.get("candidate_available"))
    ]
    return {
        "schema_version": QUALIFYING_ARTIFACT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": "qualifying_prediction",
        "target": "official_grand_prix_qualifying_classification",
        "protocol": {
            "training": "same_season_strictly_earlier_locked_event_partitions",
            "same_season_only": True,
            "legacy_cross_season_diagnostic_opt_in": False,
            "years_loaded": [target_year],
            "evaluation_years": [target_year],
            "prior_season_training_rows": 0,
            "prior_season_files_loaded": 0,
            "event_partitions": {
                name: list(values) for name, values in partitions.items()
            },
            "partition_semantics": (
                "R1-2_structural_point_fit_R3-4_model_residual_and_point_head_selection_"
                "then_locked_R1-4_final_refit_R5-6_interval_calibration_"
                "and_probability_diagnostic_holdout_R7_plus_audit"
            ),
            "selection_fit_event_keys": _json_ints(structural_event_keys),
            "structural_point_fit_event_keys": _json_ints(structural_event_keys),
            "selection_fit_role": (
                "R1_R2_only_for_matched_R3_R4_model_and_residual_selection"
            ),
            "final_fit_event_keys": _json_ints(final_fit_event_keys),
            "final_fit_role": (
                "selected_configuration_refit_on_R1_R4_before_any_R5_target_read"
            ),
            "model_selection_locked_before_final_refit": True,
            "selection_block_winner_model_id": selected_model_id,
            "selected_public_output_model_id": public_output_model_id,
            "retained_public_output_model_id": public_output_model_id,
            "selected_enable_robust_residual": enable_selected_residual,
            "selected_challenger_point_head": selected_point_head,
            "point_head_selection_event_keys": list(partitions["selection"]),
            "point_head_selection_locked_before_final_refit": True,
            "point_head_candidates": [
                POINT_HEAD_DIRECT_PACE,
                POINT_HEAD_MINIMUM_EXPECTED_ABSOLUTE_LOSS,
            ],
            "baseline": "latest_valid_target_aligned_rehearsal_rank",
            "candidate": "shared_latent_time_nested_hurdle_official_classification",
            "pairwise_role": "unavailable_with_only_two_structural_fit_events",
            "feature_allowlist": list(feature_columns),
            "position_probability_diagnostic_calibration_event_keys": list(
                partitions["calibration"]
            ),
            "held_out_interval_calibration_event_keys": list(
                partitions["calibration"]
            ),
            "position_probability_calibration_applied": False,
            "position_probability_calibration_status": (
                "raw_uncalibrated_marginals_only_use_separate_immutable_"
                "temperature_sinkhorn_diagnostic_runner"
            ),
            "position_probability_diagnostic_runner": (
                "run_qualifying_probability_calibration_audit.py"
            ),
            "within_run_audit_outcomes_reused_for_fit_or_selection": False,
            "prospective_development_evidence": False,
            "evidence_role": (
                "postdevelopment_replay_diagnostic_after_prior_R7_R9_inspection"
            ),
            "target_read_log": target_read_log,
            "all_targets_opened_after_immutable_forecast_within_current_replay": bool(
                target_read_log
                and all(
                    bool(row["forecast_frozen_before_target_read"])
                    for row in target_read_log
                )
            ),
            "skipped_no_causal_target_aligned_rehearsal": skipped_no_causal_rehearsal,
        },
        "aggregate": {
            "events": len(events),
            "baseline_mean_mae": _mean(events, "baseline_mae"),
            "candidate_available_events": len(candidate_available_events),
            "candidate_unavailable_cold_start_events": (
                len(events) - len(candidate_available_events)
            ),
            "candidate_mean_mae": _mean(
                candidate_available_events, "candidate_mae"
            ),
            "candidate_mean_mae_all_rounds_with_baseline_substitution": _mean(
                events, "candidate_mae"
            ),
            "selected_output_mean_mae": _mean(events, "selected_output_mae"),
            "baseline_mean_kendall": _mean(events, "baseline_kendall"),
            "candidate_mean_kendall": _mean(
                candidate_available_events, "candidate_kendall"
            ),
            "candidate_mean_kendall_all_rounds_with_baseline_substitution": _mean(
                events, "candidate_kendall"
            ),
            "selected_output_mean_kendall": _mean(
                events, "selected_output_kendall"
            ),
            "audit_events": len(audit_events),
            "audit_baseline_mean_mae": _mean(audit_events, "baseline_mae"),
            "audit_candidate_mean_mae": _mean(audit_events, "candidate_mae"),
            "audit_selected_output_mean_mae": _mean(
                audit_events, "selected_output_mae"
            ),
            "promotion_audit_year": target_year,
        },
        "promotion": promotion_payload,
        "frozen_selector": frozen_selection,
        "stage_probability_evaluation": _stage_metrics(scored_predictions),
        "shared_forecast_artifacts": shared_forecast_artifacts,
        "runtime": f1_model_runtime_doctor(),
        "events": events,
        "predictions": rows,
        "input_manifest": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in sorted(inputs)
        ],
        "implementation_manifest": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in implementation_paths
        ],
    }


def _run_legacy_cross_season(
    *,
    weekends_dir: Path,
    years: Sequence[int],
    evaluation_years: Sequence[int],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    root = _root()
    audit_year = max(int(value) for value in evaluation_years)
    frames: list[pd.DataFrame] = []
    infos: dict[int, dict[str, Any]] = {}
    deferred_evaluation_targets: dict[int, Path] = {}
    inputs: set[Path] = set()
    skipped_no_causal_rehearsal: list[str] = []
    for year in sorted(set(int(value) for value in years)):
        for event_dir in sorted(
            (weekends_dir / str(year)).glob("round_*"), key=_round_number
        ):
            try:
                if int(year) == audit_year:
                    frame, info, event_inputs, target_path = _event_inference_frame(
                        root,
                        event_dir,
                    )
                    deferred_evaluation_targets[int(info["event_key"])] = target_path
                else:
                    frame, info, event_inputs = _event_frame(root, event_dir)
            except ValueError as error:
                if "no causal FP3/Sprint-Qualifying" not in str(error):
                    raise
                skipped_no_causal_rehearsal.append(str(event_dir.relative_to(root)))
                continue
            frames.append(frame)
            infos[int(info["event_key"])] = info
            inputs.update(event_inputs)
    if not frames:
        raise ValueError("no local qualifying events found")
    dataset = pd.concat(frames, ignore_index=True)
    feature_columns = tuple(column for column in FEATURE_ALLOWLIST if column in dataset.columns)
    config = PairwiseRankerConfig(
        feature_columns=feature_columns,
        minimum_training_events=4,
        max_movement=3,
        random_state=int(seed),
    )
    partitions = _locked_event_partitions(
        tuple(infos), audit_year=audit_year, same_season_only=False
    )
    frozen_selection, selection = _freeze_shared_engine_on_selection(
        dataset,
        partitions=partitions,
        seed=int(seed),
    )
    enable_selected_residual = bool(
        frozen_selection["selected_enable_robust_residual"]
    )
    frozen_selection["shared_cross_mode_selected_enable_robust_residual"] = (
        enable_selected_residual
    )
    evaluation_keys = tuple(
        sorted(key for key in infos if key // 100 == audit_year)
    )
    numeric_events = pd.to_numeric(dataset["event_key"], errors="coerce")
    weak_prior = dataset.loc[numeric_events.lt(audit_year * 100)].copy()
    weak_prior["weak_transfer_prior"] = True
    weak_prior["history_weight"] = weak_prior["event_key"].map(
        lambda value: max(0.03, 0.30 / max(1, audit_year - (int(value) // 100)))
    )
    scored_frames: list[pd.DataFrame] = []
    shared_forecast_artifacts: list[dict[str, object]] = []
    interval_calibration_rows: list[pd.DataFrame] = []
    for event_key in evaluation_keys:
        current = dataset.loc[numeric_events.eq(event_key)].copy()
        if int(event_key) in set(
            (*partitions["calibration"], *partitions["audit"])
        ):
            same_season_prior_keys = tuple(partitions["point_fit"])
        else:
            same_season_prior_keys = tuple(
                value for value in partitions["point_fit"] if value < int(event_key)
            )
        same_season_history = dataset.loc[numeric_events.isin(same_season_prior_keys)].copy()
        same_season_history["weak_transfer_prior"] = False
        same_season_history["history_weight"] = 1.0
        shared_history = pd.concat(
            [weak_prior, same_season_history], ignore_index=True, sort=False
        )
        inference = _qualifying_inference_frame(current)
        if len(same_season_prior_keys) >= int(config.minimum_training_events):
            pairwise_model = fit_pairwise_qualifying_ranker(
                same_season_history,
                config=config,
                target_event_key=int(event_key),
            )
            comparator = pairwise_model.predict_event(
                inference,
                samples=2_000,
                seed=int(seed) + int(event_key),
            ).point_order[["driver_id", "predicted_qualifying_position"]].rename(
                columns={"predicted_qualifying_position": "pairwise_comparator_position"}
            )
            comparator["pairwise_comparator_status"] = "same_season_prior_events"
        else:
            comparator = inference[["driver_id", "latest_qualifying_rehearsal_rank"]].rename(
                columns={"latest_qualifying_rehearsal_rank": "pairwise_comparator_position"}
            )
            comparator["pairwise_comparator_status"] = (
                "unavailable_insufficient_same_season_events_baseline_substitute"
            )
        held_out_interval_rows = None
        if int(event_key) in set(partitions["audit"]):
            if not interval_calibration_rows:
                raise RuntimeError(
                    "Qualifying audit reached before held-out interval calibration completed"
                )
            held_out_interval_rows = pd.concat(
                interval_calibration_rows, ignore_index=True
            )
        shared_model, shared_forecast, artifact = _build_shared_event_forecast(
            shared_history,
            inference,
            target_event_key=int(event_key),
            interval_calibration_predictions=held_out_interval_rows,
            enable_robust_residual=enable_selected_residual,
        )
        target_path = deferred_evaluation_targets.get(int(event_key))
        if target_path is None:
            raise RuntimeError(f"evaluation event {event_key} has no deferred target path")
        target, target_info = _load_target_after_frozen_forecast(
            target_path,
            expected_event_key=int(event_key),
            frozen_forecast_artifact=artifact,
        )
        infos[int(event_key)].update(target_info)
        current = _attach_qualifying_target(current, target)
        current_target = target.set_index("driver_id")
        event_mask = numeric_events.eq(event_key)
        event_drivers = dataset.loc[event_mask, "driver_id"].astype(str)
        for target_column in target.columns:
            if target_column == "driver_id":
                continue
            dataset.loc[event_mask, target_column] = event_drivers.map(
                current_target[target_column]
            ).to_numpy()
        shared_forecast_artifacts.append(artifact)
        shared = shared_forecast.point_order.copy()
        shared["shared_forecast_artifact_sha256"] = artifact["artifact_sha256"]
        shared["shared_joint_samples_sha256"] = artifact["joint_samples_sha256"]
        actual_columns = current[
            [
                "driver_id",
                "qualy_position",
                "has_valid_qualifying_lap",
                "reached_q2",
                "reached_q3",
                Q1_LAP_COLUMN,
                Q2_LAP_COLUMN,
                Q3_LAP_COLUMN,
                "latest_qualifying_rehearsal_rank",
            ]
        ].rename(
            columns={
                "qualy_position": "actual_qualifying_position",
                "latest_qualifying_rehearsal_rank": "baseline_rank_prior",
            }
        )
        scored = (
            shared.merge(actual_columns, on="driver_id", how="left", validate="one_to_one")
            .merge(comparator, on="driver_id", how="left", validate="one_to_one")
        )
        scored["p_valid_qualifying_lap"] = scored["valid_lap_probability"]
        scored["p_q2_given_valid"] = scored["q2_given_valid_probability"]
        scored["p_q3_given_q2"] = scored["q3_given_q2_probability"]
        scored["p_reaches_q2"] = (
            scored["valid_lap_probability"] * scored["q2_given_valid_probability"]
        )
        scored["p_reaches_q3"] = (
            scored["p_reaches_q2"] * scored["q3_given_q2_probability"]
        )
        if int(event_key) in set(partitions["calibration"]):
            calibration = shared_forecast.lap_predictions[
                ["event_key", "driver_id", "rehearsal_source", "lap_p50"]
            ].copy()
            calibration[ACTUAL_LAP_COLUMN] = calibration["driver_id"].map(
                current.set_index("driver_id")[ACTUAL_LAP_COLUMN]
            )
            interval_calibration_rows.append(
                calibration.dropna(subset=["lap_p50", ACTUAL_LAP_COLUMN])
            )
        scored_frames.append(scored)
    scored_predictions = pd.concat(scored_frames, ignore_index=True)
    events, rows = _metric_rows(scored_predictions, infos)
    if len(events) < 2:
        raise ValueError("fewer than two complete evaluation events were scored")
    audit_events = [
        event for event in events if int(event["event_key"]) in set(partitions["audit"])
    ]
    if len(audit_events) < 2:
        raise ValueError(f"audit year {audit_year} has fewer than two scored events")
    paired = [
        EventError(
            event_key=str(event["event_key"]),
            baseline_error=float(event["baseline_mae"]),
            candidate_error=float(event["candidate_mae"]),
            stratum=("sprint" if "sprint" in str(event["event_format"]).lower() else "standard"),
        )
        for event in audit_events
    ]
    exceptional_count = min(4, max(1, len(audit_events) // 4))
    retained_tail = sorted(audit_events, key=lambda value: float(value["baseline_mae"]))[:-exceptional_count]
    tail_delta = _mean(retained_tail, "candidate_mae") - _mean(retained_tail, "baseline_mae")
    promotion = evaluate_qualifying_promotion(
        paired,
        baseline_kendall=_mean(audit_events, "baseline_kendall"),
        candidate_kendall=_mean(audit_events, "candidate_kendall"),
        pole_non_regression=_mean(audit_events, "candidate_pole_hit") >= _mean(audit_events, "baseline_pole_hit"),
        top3_non_regression=_mean(audit_events, "candidate_top3_overlap") >= _mean(audit_events, "baseline_top3_overlap"),
        top10_non_regression=_mean(audit_events, "candidate_top10_overlap") >= _mean(audit_events, "baseline_top10_overlap"),
        tail_excluded_delta=tail_delta,
        bootstrap_samples=int(bootstrap_samples),
        seed=int(seed),
    )
    audit_event_keys = tuple(sorted(int(event["event_key"]) for event in audit_events))
    contract_gates = _qualifying_contract_gates(
        scored_predictions,
        event_info=infos,
        event_keys=audit_event_keys,
    )
    preselected_challenger = bool(
        selection.selected_model_id == "shared_qualifying_latent_lap_v4"
    )
    point_promoted = bool(
        promotion.promoted
        and contract_gates["point_contract_gates_passed"]
        and preselected_challenger
    )
    promotion_payload = promotion.to_payload()
    promotion_reasons = list(promotion_payload["reasons"])
    if not contract_gates["pre_q_entrant_coverage_is_100_percent"]:
        promotion_reasons.append("gate_failed:pre_q_entrant_coverage_is_100_percent")
    if not contract_gates["every_event_is_legal_full_field_permutation"]:
        promotion_reasons.append("gate_failed:every_event_is_legal_full_field_permutation")
    if not contract_gates["probability_contract_satisfied"]:
        promotion_reasons.append("gate_failed:probability_contract_satisfied")
    if not preselected_challenger:
        promotion_reasons.append("gate_failed:challenger_not_frozen_on_selection_block")
    promotion_payload.update(
        {
            "promoted": point_promoted,
            "status": "promoted" if point_promoted else "rejected",
            "reasons": promotion_reasons,
            "point_model_promoted": point_promoted,
            "position_probability_outputs_promoted": bool(
                point_promoted and contract_gates["position_probabilities_calibrated"]
            ),
            "mode_contract_gates": contract_gates,
        }
    )
    implementation_paths = [
        Path(__file__).resolve(),
        root / "packages/f1/features/qualifying_lap.py",
        root / "packages/f1/models/pre_quali/pairwise.py",
        root / "packages/f1/models/pre_quali/classification.py",
        root / "packages/f1/models/pre_quali/train.py",
        root / "packages/f1/models/ultimate_lap_time/achievable.py",
        root / "packages/f1/models/pre_quali/evaluate.py",
        root / "packages/f1/orchestration/non_live_validation.py",
    ]
    return {
        "schema_version": "f1_shared_qualifying_latent_event_block_v4",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": "qualifying_prediction",
        "target": "official_grand_prix_qualifying_classification",
        "protocol": {
            "training": "strictly_earlier_complete_events",
            "same_season_only": False,
            "legacy_cross_season_diagnostic_opt_in": True,
            "years_loaded": sorted(set(int(value) for value in years)),
            "evaluation_years": sorted(set(int(value) for value in evaluation_years)),
            "prior_season_training_rows": int(
                pd.to_numeric(dataset["event_key"], errors="coerce")
                .lt(audit_year * 100)
                .sum()
            ),
            "baseline": "latest_valid_target_aligned_rehearsal_rank",
            "candidate": "shared_latent_time_nested_hurdle_official_classification",
            "pairwise_role": "same_season_only_comparator",
            "maximum_movement_positions": int(config.max_movement),
            "feature_allowlist": list(feature_columns),
            "event_partitions": {
                name: list(values) for name, values in partitions.items()
            },
            "partition_validation_issues": [],
            "final_event_specific_training": (
                "target_events_1_2_point_fit_only; predictor_frozen_before_events_3_4"
            ),
            "held_out_interval_calibration": (
                "target_events_3_4_final_predictor_residuals_only"
            ),
            "audit_outcomes_reused": False,
            "older_season_use": "weak_invariant_session_transition_and_reliability_priors_only",
            "skipped_no_causal_target_aligned_rehearsal": skipped_no_causal_rehearsal,
        },
        "aggregate": {
            "events": len(events),
            "baseline_mean_mae": _mean(events, "baseline_mae"),
            "candidate_mean_mae": _mean(events, "candidate_mae"),
            "baseline_mean_kendall": _mean(events, "baseline_kendall"),
            "candidate_mean_kendall": _mean(events, "candidate_kendall"),
            "pairwise_comparator_mean_mae": _mean(events, "pairwise_comparator_mae"),
            "pairwise_comparator_mean_kendall": _mean(events, "pairwise_comparator_kendall"),
            "baseline_pole_hit_rate": _mean(events, "baseline_pole_hit"),
            "candidate_pole_hit_rate": _mean(events, "candidate_pole_hit"),
            "baseline_top3_overlap": _mean(events, "baseline_top3_overlap"),
            "candidate_top3_overlap": _mean(events, "candidate_top3_overlap"),
            "baseline_top10_overlap": _mean(events, "baseline_top10_overlap"),
            "candidate_top10_overlap": _mean(events, "candidate_top10_overlap"),
            "tail_excluded_delta_candidate_minus_baseline": tail_delta,
            "by_year": {
                str(year): {
                    "events": len([event for event in events if int(event["year"]) == year]),
                    "baseline_mean_mae": _mean(
                        [event for event in events if int(event["year"]) == year], "baseline_mae"
                    ),
                    "candidate_mean_mae": _mean(
                        [event for event in events if int(event["year"]) == year], "candidate_mae"
                    ),
                }
                for year in sorted({int(event["year"]) for event in events})
            },
            "promotion_audit_year": audit_year,
        },
        "promotion": promotion_payload,
        "frozen_selector": frozen_selection,
        "stage_probability_evaluation": _stage_metrics(scored_predictions),
        "shared_forecast_artifacts": shared_forecast_artifacts,
        "runtime": f1_model_runtime_doctor(),
        "events": events,
        "predictions": rows,
        "input_manifest": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in sorted(inputs)
        ],
        "implementation_manifest": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in implementation_paths
        ],
    }


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def run(
    *,
    weekends_dir: Path,
    years: Sequence[int],
    evaluation_years: Sequence[int],
    bootstrap_samples: int,
    seed: int,
    same_season_only: bool = True,
) -> dict[str, Any]:
    """Dispatch to the canonical protocol or explicit legacy diagnostic."""

    runner = _run_same_season if same_season_only else _run_legacy_cross_season
    return runner(
        weekends_dir=weekends_dir,
        years=years,
        evaluation_years=evaluation_years,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weekends-dir", type=Path, default=_root() / "data/f1/raw/weekends")
    parser.add_argument(
        "--years",
        type=_csv_ints,
        default=(2026,),
        help="one target year by default; multiple years require --legacy-cross-season",
    )
    parser.add_argument("--evaluation-years", type=_csv_ints, default=(2026,))
    parser.add_argument(
        "--legacy-cross-season",
        action="store_true",
        help=(
            "explicitly opt into the prior weak-transfer diagnostic; the canonical "
            "Qualifying evidence protocol never reads prior-season files"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            _root()
            / "artifacts/backtests/f1/qualifying/shared_latent_same_season_v1.json"
        ),
    )
    return parser


def main() -> int:
    parser = _build_argument_parser()
    args = parser.parse_args()
    payload = run(
        weekends_dir=args.weekends_dir.expanduser().resolve(),
        years=args.years,
        evaluation_years=args.evaluation_years,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        same_season_only=not bool(args.legacy_cross_season),
    )
    output = args.output.expanduser()
    if not output.is_absolute():
        output = _root() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "aggregate": payload["aggregate"], "promotion": payload["promotion"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Suggested commit name: fix(f1-quali): enforce same-season causal model selection
