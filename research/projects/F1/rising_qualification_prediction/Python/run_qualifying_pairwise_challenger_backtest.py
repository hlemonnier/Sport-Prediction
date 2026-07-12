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
from packages.f1.models.pre_quali.evaluate import walk_forward_pairwise_qualifying
from packages.f1.models.pre_quali.classification import (
    StageProbabilityConfig,
    fit_qualifying_stage_probability_model,
)
from packages.f1.models.pre_quali.pairwise import PairwiseRankerConfig
from packages.f1.models.pre_quali.selection import (
    FrozenSelectorConfig,
    QualifyingModelEvidence,
    select_frozen_qualifying_model,
)
from packages.f1.orchestration.model_runtime import f1_model_runtime_doctor
from packages.f1.orchestration.non_live_validation import (
    EventError,
    evaluate_qualifying_promotion,
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
    eligible = [
        session
        for session in sessions
        if int(session.get("session_order", 999)) < qualifying_order
        and str(session.get("session_type", "")).strip().lower()
        in {"free_practice", "sprint_qualifying"}
        and bool(session.get("completed", True))
    ]
    if not eligible:
        raise ValueError(f"{event_dir.name}: no completed pre-Qualifying rehearsal")
    target_aligned = max(eligible, key=lambda value: int(value.get("session_order", -1)))
    earlier = [value for value in eligible if value is not target_aligned]

    qualifying_results_path = _resolve_snapshot_path(
        root, event_dir, qualifying.get("results_path")
    )
    rehearsal_path = _resolve_snapshot_path(root, event_dir, target_aligned.get("laps_path"))
    qualifying_results = pd.read_csv(qualifying_results_path)
    rehearsal_laps = pd.read_csv(rehearsal_path)
    source = str(target_aligned.get("session_type") or target_aligned.get("session_name") or "rehearsal")
    rehearsal_laps["rehearsal_source"] = source

    earlier_parts: list[pd.DataFrame] = []
    input_paths = [metadata_path, qualifying_results_path, rehearsal_path]
    for session in earlier:
        path = _resolve_snapshot_path(root, event_dir, session.get("laps_path"))
        part = pd.read_csv(path)
        part["rehearsal_source"] = str(
            session.get("session_type") or session.get("session_name") or "earlier_rehearsal"
        )
        earlier_parts.append(part)
        input_paths.append(path)
    earlier_laps = pd.concat(earlier_parts, ignore_index=True) if earlier_parts else None

    driver_column = _driver_column(qualifying_results)
    team_column = _team_column(qualifying_results)
    entrants = pd.DataFrame(
        {
            "driver_id": qualifying_results[driver_column].astype(str).str.strip(),
            "team_id": (
                qualifying_results[team_column].astype(str).str.strip()
                if team_column
                else "unknown_team"
            ),
        }
    )
    features = build_quality_aware_rehearsal_features(
        rehearsal_laps,
        entrants=entrants,
        earlier_laps=earlier_laps,
    )
    event_key = int(metadata["year"]) * 100 + int(metadata["round_number"])
    actual = pd.DataFrame(
        {
            "driver_id": entrants["driver_id"],
            "qualy_position": _completed_positions(qualifying_results),
            "has_valid_qualifying_lap": _has_time(qualifying_results, "Q1").astype(int),
            "reached_q2": _has_time(qualifying_results, "Q2").astype(int),
            "reached_q3": _has_time(qualifying_results, "Q3").astype(int),
        }
    )
    frame = features.merge(actual, on="driver_id", how="left", validate="one_to_one")
    frame["event_key"] = event_key
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
    for year in sorted(set(int(value) for value in years)):
        for event_dir in sorted(
            (weekends_dir / str(year)).glob("round_*"), key=_round_number
        ):
            frame, info, event_inputs = _event_frame(root, event_dir)
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
    evaluation_keys = tuple(
        sorted(key for key in infos if key // 100 in set(int(value) for value in evaluation_years))
    )
    result = walk_forward_pairwise_qualifying(
        dataset,
        config=config,
        evaluation_event_keys=evaluation_keys,
    )
    stage_frames: list[pd.DataFrame] = []
    numeric_events = pd.to_numeric(dataset["event_key"], errors="coerce")
    for event_key in evaluation_keys:
        prior_keys = sorted(
            int(value) for value in numeric_events.loc[numeric_events.lt(event_key)].unique().tolist()
        )
        if len(prior_keys) < 4:
            continue
        history = dataset.loc[numeric_events.isin(prior_keys)].copy()
        current = dataset.loc[numeric_events.eq(event_key)].copy()
        stage_config = StageProbabilityConfig(
            feature_columns=feature_columns,
            minimum_training_events=4,
            random_state=int(seed),
        )
        stage_model = fit_qualifying_stage_probability_model(
            history,
            config=stage_config,
            target_event_key=event_key,
        )
        stage = stage_model.predict_event(current)
        for label in ("has_valid_qualifying_lap", "reached_q2", "reached_q3"):
            stage[label] = current[label].to_numpy()
        stage_frames.append(stage)
    stage_predictions = pd.concat(stage_frames, ignore_index=True) if stage_frames else pd.DataFrame()
    scored_predictions = result.predictions.copy()
    if not stage_predictions.empty:
        scored_predictions = scored_predictions.merge(
            stage_predictions,
            on=["event_key", "driver_id"],
            how="left",
            validate="one_to_one",
        )
    events, rows = _metric_rows(scored_predictions, infos)
    if len(events) < 2:
        raise ValueError("fewer than two complete evaluation events were scored")
    audit_year = max(int(value) for value in evaluation_years)
    audit_events = [event for event in events if int(event["year"]) == audit_year]
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
    selection = select_frozen_qualifying_model(
        [
            QualifyingModelEvidence(
                model_id="qualifying_rehearsal_rank_baseline_v1",
                mean_absolute_position_error=_mean(audit_events, "baseline_mae"),
                event_keys=audit_event_keys,
            ),
            QualifyingModelEvidence(
                model_id="qualifying_pairwise_logistic_residual_v1",
                mean_absolute_position_error=_mean(audit_events, "candidate_mae"),
                event_keys=audit_event_keys,
                promotion_gates_passed=bool(promotion.promoted),
            ),
        ],
        config=FrozenSelectorConfig(),
    )
    implementation_paths = [
        Path(__file__).resolve(),
        root / "packages/f1/features/qualifying_lap.py",
        root / "packages/f1/models/pre_quali/pairwise.py",
        root / "packages/f1/models/pre_quali/evaluate.py",
        root / "packages/f1/orchestration/non_live_validation.py",
    ]
    return {
        "schema_version": "f1_qualifying_pairwise_event_block_v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": "qualifying_prediction",
        "target": "official_grand_prix_qualifying_classification",
        "protocol": {
            "training": "strictly_earlier_complete_events",
            "years_loaded": sorted(set(int(value) for value in years)),
            "evaluation_years": sorted(set(int(value) for value in evaluation_years)),
            "baseline": "latest_valid_target_aligned_rehearsal_rank",
            "candidate": "quality_aware_regularized_pairwise_logistic_residual",
            "maximum_movement_positions": int(config.max_movement),
            "feature_allowlist": list(feature_columns),
            "skipped_insufficient_history": list(result.skipped_event_keys),
        },
        "aggregate": {
            "events": len(events),
            "baseline_mean_mae": _mean(events, "baseline_mae"),
            "candidate_mean_mae": _mean(events, "candidate_mae"),
            "baseline_mean_kendall": _mean(events, "baseline_kendall"),
            "candidate_mean_kendall": _mean(events, "candidate_kendall"),
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
        "promotion": promotion.to_payload(),
        "frozen_selector": {
            "selected_model_id": selection.selected_model_id,
            "decision": selection.decision,
            "observed_event_keys": list(selection.observed_event_keys),
            "selection_event_keys": list(selection.selection_event_keys),
            "baseline_mae": selection.baseline_mae,
            "challenger_mae": selection.challenger_mae,
        },
        "stage_probability_evaluation": _stage_metrics(scored_predictions),
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
