#!/usr/bin/env python3
"""Event-block backtest for the quality-aware Qualifying pairwise challenger."""

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

from packages.f1.features.qualifying_lap import build_quality_aware_rehearsal_features
from packages.f1.models.pre_quali.pairwise import (
    PairwiseRankerConfig,
    fit_pairwise_qualifying_ranker,
)
from packages.f1.models.pre_quali.train import train_shared_qualifying_latent_model
from packages.f1.models.ultimate_lap_time.achievable import (
    ACTUAL_LAP_COLUMN,
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


def _root() -> Path:
    return Path(__file__).resolve().parents[5]


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


def _event_frame(root: Path, event_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], list[Path]]:
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
    actual = pd.DataFrame(
        {
            "driver_id": target_drivers,
            "qualy_position": _completed_positions(qualifying_results),
            # Provider tables occasionally omit a Q1 split while retaining a
            # later-stage time.  Any official Q1/Q2/Q3 time proves a valid lap;
            # the stage labels remain strictly nested.
            "has_valid_qualifying_lap": (q1 | q2 | q3).astype(int),
            "reached_q2": (q2 | q3).astype(int),
            "reached_q3": q3.astype(int),
            Q1_LAP_COLUMN: stage_seconds[1],
            Q2_LAP_COLUMN: stage_seconds[2],
            Q3_LAP_COLUMN: stage_seconds[3],
            ACTUAL_LAP_COLUMN: pd.concat(list(stage_seconds.values()), axis=1).min(
                axis=1, skipna=True
            ),
        }
    )
    frame = features.merge(actual, on="driver_id", how="left", validate="one_to_one")
    frame["event_key"] = event_key
    frame["rehearsal_source"] = source
    frame["latest_qualifying_rehearsal_source"] = source
    baseline_anchor = pd.to_numeric(frame.get("valid_clean_best_seconds"), errors="coerce")
    fallback = pd.to_numeric(frame.get("quality_aware_anchor_seconds"), errors="coerce")
    baseline_anchor = baseline_anchor.where(baseline_anchor.notna(), fallback)
    stable = pd.DataFrame(
        {
            "anchor": baseline_anchor.fillna(np.inf),
            "driver": frame["driver_id"].astype(str),
            "row": np.arange(len(frame)),
        }
    ).sort_values(["anchor", "driver", "row"], kind="mergesort")
    ranks = pd.Series(np.arange(1, len(stable) + 1), index=stable.index)
    frame["latest_qualifying_rehearsal_rank"] = ranks.reindex(frame.index).astype(int)
    event_info = {
        "event_key": event_key,
        "year": int(metadata["year"]),
        "round": int(metadata["round_number"]),
        "event_name": str(metadata.get("event_name") or event_dir.name),
        "event_format": str(metadata.get("event_format") or "unknown"),
        "rehearsal_source": source,
        "field_size": int(len(frame)),
        "official_target_driver_count": int(target_drivers.nunique()),
        "official_target_driver_ids": sorted(target_drivers.unique().tolist()),
        "roster_source": "latest_target_aligned_pre_qualifying_session_only",
        "target_result_used_for_roster": False,
    }
    return frame, event_info, input_paths


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
        candidate_mae = float((candidate - actual).abs().mean())
        baseline_mae = float((baseline - actual).abs().mean())
        candidate_order = set(group.nsmallest(3, "predicted_qualifying_position")["driver_id"])
        baseline_order = set(group.nsmallest(3, "baseline_rank_prior")["driver_id"])
        actual_top3 = set(group.nsmallest(3, "actual_qualifying_position")["driver_id"])
        actual_top10 = set(group.nsmallest(min(10, len(group)), "actual_qualifying_position")["driver_id"])
        candidate_top10 = set(group.nsmallest(min(10, len(group)), "predicted_qualifying_position")["driver_id"])
        baseline_top10 = set(group.nsmallest(min(10, len(group)), "baseline_rank_prior")["driver_id"])
        actual_winner = str(group.nsmallest(1, "actual_qualifying_position").iloc[0]["driver_id"])
        events.append(
            {
                **info,
                "baseline_mae": baseline_mae,
                "candidate_mae": candidate_mae,
                "delta_candidate_minus_baseline": candidate_mae - baseline_mae,
                "baseline_kendall": float(baseline.corr(actual, method="kendall")),
                "candidate_kendall": float(candidate.corr(actual, method="kendall")),
                "pairwise_comparator_mae": float((comparator - actual).abs().mean()),
                "pairwise_comparator_kendall": float(comparator.corr(actual, method="kendall")),
                "baseline_pole_hit": str(group.nsmallest(1, "baseline_rank_prior").iloc[0]["driver_id"]) == actual_winner,
                "candidate_pole_hit": str(group.nsmallest(1, "predicted_qualifying_position").iloc[0]["driver_id"]) == actual_winner,
                "baseline_top3_overlap": len(baseline_order & actual_top3) / 3.0,
                "candidate_top3_overlap": len(candidate_order & actual_top3) / 3.0,
                "baseline_top10_overlap": len(baseline_top10 & actual_top10) / float(min(10, len(group))),
                "candidate_top10_overlap": len(candidate_top10 & actual_top10) / float(min(10, len(group))),
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
) -> dict[str, tuple[int, ...]]:
    ordered = tuple(sorted({int(value) for value in event_keys}))
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
                model_id="shared_qualifying_latent_lap_v3",
                mean_absolute_position_error=float(
                    np.mean([row[candidate_column] for row in event_rows])
                ),
                event_keys=selection_event_keys,
                promotion_gates_passed=True,
            ),
        ],
        config=FrozenSelectorConfig(challenger_model_id="shared_qualifying_latent_lap_v3"),
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
):
    """Qualifying runner adapter to the cross-mode frozen forecast path."""

    return build_shared_qualifying_event_forecast(
        history,
        inference,
        target_event_key=int(target_event_key),
        interval_calibration_predictions=interval_calibration_predictions,
    )


def run(
    *,
    weekends_dir: Path,
    years: Sequence[int],
    evaluation_years: Sequence[int],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    root = _root()
    frames: list[pd.DataFrame] = []
    infos: dict[int, dict[str, Any]] = {}
    inputs: set[Path] = set()
    skipped_no_causal_rehearsal: list[str] = []
    for year in sorted(set(int(value) for value in years)):
        for event_dir in sorted(
            (weekends_dir / str(year)).glob("round_*"), key=_round_number
        ):
            try:
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
    audit_year = max(int(value) for value in evaluation_years)
    partitions = _locked_event_partitions(tuple(infos), audit_year=audit_year)
    frozen_selection, selection = _freeze_shared_engine_on_selection(
        dataset,
        partitions=partitions,
        seed=int(seed),
    )
    enable_selected_residual = SHARED_QUALIFYING_ENABLE_ROBUST_RESIDUAL
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
        )
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
        selection.selected_model_id == "shared_qualifying_latent_lap_v3"
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
        "schema_version": "f1_shared_qualifying_latent_event_block_v3",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": "qualifying_prediction",
        "target": "official_grand_prix_qualifying_classification",
        "protocol": {
            "training": "strictly_earlier_complete_events",
            "years_loaded": sorted(set(int(value) for value in years)),
            "evaluation_years": sorted(set(int(value) for value in evaluation_years)),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weekends-dir", type=Path, default=_root() / "data/f1/raw/weekends")
    parser.add_argument("--years", type=_csv_ints, default=(2022, 2023, 2024, 2025, 2026))
    parser.add_argument("--evaluation-years", type=_csv_ints, default=(2025, 2026))
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--output",
        type=Path,
        default=_root() / "artifacts/backtests/f1/qualifying/quality_aware_pairwise_v1.json",
    )
    args = parser.parse_args()
    payload = run(
        weekends_dir=args.weekends_dir.expanduser().resolve(),
        years=args.years,
        evaluation_years=args.evaluation_years,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
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


# Suggested commit name: feat(f1-quali): add quality-aware pairwise walk-forward evidence
