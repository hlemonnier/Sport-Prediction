#!/usr/bin/env python3
"""Evaluate optional F1 rank/quantile challengers on locked event blocks.

The command is evidence-only.  It never changes a production profile and it
never treats backend availability as model quality.  Prior-season rows form a
disjoint hyperparameter-selection record; absolute pace/rank fitting uses only
the current-season calibration events, and every audit event is later in time.
"""

from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import numpy as np
import pandas as pd

from packages.f1.models.grouped_ranking import (
    GroupedRankingConfig,
    fit_grouped_ranking_challenger,
)
from packages.f1.models.ultimate_lap_time.achievable import ACTUAL_LAP_COLUMN
from packages.f1.models.ultimate_lap_time.tabular_quantile import (
    TabularQuantileBackendUnavailable,
    TabularQuantileConfig,
    fit_tabular_quantile_model,
)
from packages.f1.orchestration.model_runtime import f1_model_runtime_doctor
from packages.f1.orchestration.non_live_validation import validate_event_partitions
from run_qualifying_pairwise_challenger_backtest import (
    FEATURE_ALLOWLIST,
    _event_frame,
)


ROUND_PATTERN = re.compile(r"round_(\d+)", re.IGNORECASE)
RANKING_BACKENDS = ("xgboost_lambdarank", "lightgbm_lambdarank")


def _root() -> Path:
    return Path(__file__).resolve().parents[5]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _round_number(path: Path) -> int:
    match = ROUND_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"invalid round directory: {path}")
    return int(match.group(1))


def _qualifying_start(metadata: Mapping[str, Any]) -> tuple[pd.Timestamp, str]:
    sessions = [value for value in metadata.get("sessions", []) if isinstance(value, Mapping)]
    qualifying = next(
        (
            session
            for session in sessions
            if str(session.get("session_type", "")).strip().lower() == "qualifying"
        ),
        None,
    )
    if qualifying is None:
        raise ValueError("weekend metadata has no Grand Prix Qualifying session")
    timestamp = pd.to_datetime(
        qualifying.get("scheduled_start_utc"), errors="coerce", utc=True
    )
    if not pd.isna(timestamp):
        return pd.Timestamp(timestamp), "scheduled_qualifying_start_utc"
    year = int(metadata.get("year"))
    round_number = int(metadata.get("round_number"))
    # Legacy snapshots lack wall-clock schedule fields.  These rows are used
    # only as prior-season selection/development records, where season/round is
    # the authoritative chronology and no availability cutoff is inferred.
    ordinal = pd.Timestamp(f"{year}-01-01T00:00:00Z") + pd.to_timedelta(
        int(round_number), unit="D"
    )
    return ordinal, "season_round_ordinal_selection_only"


def _load_event_dataset(
    weekends_dir: Path,
    *,
    years: Sequence[int],
) -> tuple[pd.DataFrame, dict[int, dict[str, Any]], tuple[Path, ...]]:
    root = _root()
    frames: list[pd.DataFrame] = []
    infos: dict[int, dict[str, Any]] = {}
    files: set[Path] = set()
    for year in sorted({int(value) for value in years}):
        for event_dir in sorted(
            (weekends_dir / str(year)).glob("round_*"), key=_round_number
        ):
            frame, info, event_files = _event_frame(root, event_dir)
            metadata_path = event_dir / "weekend_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            event_time, time_semantics = _qualifying_start(metadata)
            event_key = int(info["event_key"])
            frame = frame.copy()
            frame["season"] = int(year)
            frame["event_as_of"] = event_time.isoformat().replace("+00:00", "Z")
            frame["event_time_semantics"] = time_semantics
            frame["event_format"] = str(info.get("event_format") or "unknown")
            frames.append(frame)
            infos[event_key] = {
                **info,
                "event_as_of": event_time.isoformat().replace("+00:00", "Z"),
                "event_time_semantics": time_semantics,
            }
            files.update(Path(path).resolve() for path in event_files)
            files.add(metadata_path.resolve())
    if not frames:
        raise ValueError("no completed weekend snapshots were loaded")
    dataset = pd.concat(frames, ignore_index=True, sort=False)
    dataset["event_key"] = pd.to_numeric(dataset["event_key"], errors="raise").astype(int)
    return dataset, infos, tuple(sorted(files))


def _locked_partitions(
    event_keys: Sequence[int],
    *,
    target_year: int,
) -> dict[str, tuple[int, ...]]:
    ordered = tuple(sorted({int(value) for value in event_keys}))
    selection = tuple(value for value in ordered if value // 100 == int(target_year) - 1)
    target = tuple(value for value in ordered if value // 100 == int(target_year))
    if not selection:
        raise ValueError("optional challenger evidence requires the prior selection season")
    if len(target) < 6:
        raise ValueError("optional challenger evidence requires at least six target events")
    calibration_count = max(4, min(len(target) - 2, len(target) // 2))
    partitions = {
        "development": tuple(value for value in ordered if value // 100 < int(target_year) - 1),
        "selection": selection,
        "calibration": target[:calibration_count],
        "audit": target[calibration_count:],
    }
    issues = validate_event_partitions(
        **{name: [str(value) for value in values] for name, values in partitions.items()}
    )
    if issues:
        raise ValueError(f"invalid optional-model event partitions: {list(issues)}")
    return partitions


def _ranking_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event_key, event in frame.groupby("event_key", sort=True):
        actual = pd.to_numeric(event["actual_qualifying_position"], errors="coerce")
        predicted = pd.to_numeric(event["predicted_rank"], errors="coerce")
        baseline = pd.to_numeric(event["baseline_rank"], errors="coerce")
        rows.append(
            {
                "event_key": int(event_key),
                "event_format": str(event["event_format"].iloc[0]),
                "rows": int(len(event)),
                "candidate_mae": float((predicted - actual).abs().mean()),
                "baseline_mae": float((baseline - actual).abs().mean()),
                "candidate_kendall": float(predicted.corr(actual, method="kendall")),
                "baseline_kendall": float(baseline.corr(actual, method="kendall")),
            }
        )
    return rows


def _quantile_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event_key, event in frame.groupby("event_key", sort=True):
        actual = pd.to_numeric(event[ACTUAL_LAP_COLUMN], errors="coerce")
        predicted = pd.to_numeric(event["lap_p50"], errors="coerce")
        baseline = pd.to_numeric(event["baseline_lap_seconds"], errors="coerce")
        interval = (
            actual.ge(pd.to_numeric(event["lap_p05"], errors="coerce"))
            & actual.le(pd.to_numeric(event["lap_p90"], errors="coerce"))
        )
        rows.append(
            {
                "event_key": int(event_key),
                "event_format": str(event["event_format"].iloc[0]),
                "rows": int(len(event)),
                "candidate_mae_seconds": float((predicted - actual).abs().mean()),
                "baseline_mae_seconds": float((baseline - actual).abs().mean()),
                "p05_p90_coverage": float(interval.mean()),
                "p05_p90_mean_width_seconds": float(
                    (
                        pd.to_numeric(event["lap_p90"], errors="coerce")
                        - pd.to_numeric(event["lap_p05"], errors="coerce")
                    ).mean()
                ),
            }
        )
    return rows


def _aggregate_metrics(rows: Sequence[Mapping[str, Any]], *keys: str) -> dict[str, float]:
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in keys
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run(
    *,
    weekends_dir: Path,
    target_year: int,
) -> dict[str, Any]:
    selection_year = int(target_year) - 1
    dataset, infos, input_files = _load_event_dataset(
        weekends_dir,
        years=(selection_year - 1, selection_year, int(target_year)),
    )
    partitions = _locked_partitions(tuple(infos), target_year=int(target_year))
    numeric_event = pd.to_numeric(dataset["event_key"], errors="raise").astype(int)
    selection = dataset.loc[numeric_event.isin(partitions["selection"])].copy()
    calibration = dataset.loc[numeric_event.isin(partitions["calibration"])].copy()
    audit = dataset.loc[numeric_event.isin(partitions["audit"])].copy()
    if pd.to_datetime(calibration["event_as_of"], utc=True).max() >= pd.to_datetime(
        audit["event_as_of"], utc=True
    ).min():
        raise ValueError("optional challenger calibration must strictly precede audit")

    ranking_features = tuple(
        column
        for column in (*FEATURE_ALLOWLIST, "team_id", "rehearsal_source", "best_lap_compound")
        if column in dataset.columns
    )
    rank_results: dict[str, Any] = {}
    ranking_predictions: list[dict[str, Any]] = []
    for backend in RANKING_BACKENDS:
        config = GroupedRankingConfig(
            feature_columns=ranking_features,
            backend=backend,
            target_column="qualy_position",
            n_estimators=160,
            learning_rate=0.04,
            max_depth=3,
            num_leaves=15,
            minimum_training_events=2,
        )
        result = fit_grouped_ranking_challenger(
            calibration,
            config=config,
            target_season=int(target_year),
            selection_records=selection,
        )
        if not result.available:
            rank_results[backend] = {
                "status": result.status,
                "manifest": result.manifest,
                "reason": result.unavailable_reason,
            }
            continue
        inference = audit.drop(
            columns=[
                "qualy_position",
                ACTUAL_LAP_COLUMN,
                "has_valid_qualifying_lap",
                "reached_q2",
                "reached_q3",
            ],
            errors="ignore",
        )
        predictions = result.require_model().predict(inference)
        scored = pd.DataFrame(
            {
                "event_key": audit["event_key"].to_numpy(),
                "event_format": audit["event_format"].to_numpy(),
                "driver_id": audit["driver_id"].astype(str).to_numpy(),
                "actual_qualifying_position": pd.to_numeric(
                    audit["qualy_position"], errors="coerce"
                ).to_numpy(),
                "baseline_rank": pd.to_numeric(
                    audit["latest_qualifying_rehearsal_rank"], errors="coerce"
                ).to_numpy(),
                "ranking_score": predictions["ranking_score"].to_numpy(),
                "predicted_rank": predictions["predicted_rank"].to_numpy(),
                "backend": backend,
            }
        )
        metrics = _ranking_metrics(scored)
        rank_results[backend] = {
            "status": result.status,
            "manifest": result.manifest,
            "event_metrics": metrics,
            "aggregate": _aggregate_metrics(
                metrics,
                "candidate_mae",
                "baseline_mae",
                "candidate_kendall",
                "baseline_kendall",
            ),
        }
        ranking_predictions.extend(scored.to_dict(orient="records"))

    quantile_features = tuple(
        column
        for column in (
            *FEATURE_ALLOWLIST,
            "team_id",
            "rehearsal_source",
            "best_lap_compound",
        )
        if column in dataset.columns
    )
    quantile_training = calibration.copy()
    quantile_training["lap_residual_seconds"] = (
        pd.to_numeric(quantile_training[ACTUAL_LAP_COLUMN], errors="coerce")
        - pd.to_numeric(
            quantile_training["quality_aware_anchor_seconds"], errors="coerce"
        )
    )
    quantile_config = TabularQuantileConfig(
        backend="lightgbm",
        feature_columns=quantile_features,
        target_column="lap_residual_seconds",
        target_season=int(target_year),
        same_season_only=True,
        fit_before=str(audit["event_as_of"].min()),
        n_estimators=160,
        learning_rate=0.04,
        max_depth=3,
    )
    try:
        quantile_model = fit_tabular_quantile_model(
            quantile_training, config=quantile_config
        )
        quantile_inference = audit.drop(
            columns=[
                "qualy_position",
                ACTUAL_LAP_COLUMN,
                "has_valid_qualifying_lap",
                "reached_q2",
                "reached_q3",
            ],
            errors="ignore",
        )
        quantiles = quantile_model.predict(quantile_inference)
        baseline_lap = pd.to_numeric(
            audit["quality_aware_anchor_seconds"], errors="coerce"
        ).to_numpy()
        quantile_scored = pd.DataFrame(
            {
                "event_key": audit["event_key"].to_numpy(),
                "event_format": audit["event_format"].to_numpy(),
                "driver_id": audit["driver_id"].astype(str).to_numpy(),
                ACTUAL_LAP_COLUMN: pd.to_numeric(
                    audit[ACTUAL_LAP_COLUMN], errors="coerce"
                ).to_numpy(),
                "baseline_lap_seconds": baseline_lap,
                "lap_p05": baseline_lap + quantiles["lap_p05"].to_numpy(),
                "lap_p50": baseline_lap + quantiles["lap_p50"].to_numpy(),
                "lap_p90": baseline_lap + quantiles["lap_p90"].to_numpy(),
                "backend": "lightgbm_quantile",
            }
        ).dropna(subset=[ACTUAL_LAP_COLUMN, "baseline_lap_seconds"])
        quantile_metrics = _quantile_metrics(quantile_scored)
        quantile_result: dict[str, Any] = {
            "status": str(quantile_model.training_summary["status"]),
            "manifest": quantile_model.manifest(),
            "target_semantics": "actual_qualifying_best_minus_quality_aware_anchor_seconds",
            "event_metrics": quantile_metrics,
            "aggregate": _aggregate_metrics(
                quantile_metrics,
                "candidate_mae_seconds",
                "baseline_mae_seconds",
                "p05_p90_coverage",
                "p05_p90_mean_width_seconds",
            ),
            "predictions": quantile_scored.to_dict(orient="records"),
        }
    except TabularQuantileBackendUnavailable as exc:
        quantile_result = exc.to_payload()

    implementation_files = (
        Path(__file__).resolve(),
        _root()
        / "research/projects/F1/rising_qualification_prediction/Python/run_qualifying_pairwise_challenger_backtest.py",
        _root() / "packages/f1/features/qualifying_lap.py",
        _root() / "packages/f1/models/grouped_ranking.py",
        _root() / "packages/f1/models/ultimate_lap_time/tabular_quantile.py",
        _root() / "packages/f1/orchestration/model_runtime.py",
    )
    return _json_safe(
        {
            "schema_version": "f1_optional_non_live_event_block_evidence_v1",
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "target_year": int(target_year),
            "decision": {
                "production_profile_changed": False,
                "promotion_status": "evidence_only_requires_primary_mode_promotion_gates",
            },
            "protocol": {
                "event_partitions": {
                    name: list(values) for name, values in partitions.items()
                },
                "partition_validation_issues": [],
                "absolute_pace_fit": "target_season_calibration_events_only",
                "prior_season_role": "disjoint_hyperparameter_selection_record_only",
                "event_time_semantics": {
                    "target_season": "scheduled_qualifying_start_utc",
                    "legacy_prior_seasons": "season_round_ordinal_selection_only",
                },
                "audit_outcomes_used_for_fit": False,
                "training_threads_per_backend": 1,
                "training_queue": "sequential",
            },
            "runtime": f1_model_runtime_doctor(),
            "ranking": rank_results,
            "ranking_predictions": ranking_predictions,
            "lightgbm_quantile": quantile_result,
            "input_manifest": [
                {"path": str(path.relative_to(_root())), "sha256": _sha256(path)}
                for path in input_files
            ],
            "implementation_manifest": [
                {"path": str(path.relative_to(_root())), "sha256": _sha256(path)}
                for path in implementation_files
            ],
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weekends-dir",
        type=Path,
        default=_root() / "data/f1/raw/weekends",
    )
    parser.add_argument("--target-year", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=_root()
        / "artifacts/backtests/f1/optional_models/2026_event_block_challengers_v1.json",
    )
    args = parser.parse_args()
    payload = run(
        weekends_dir=args.weekends_dir.expanduser().resolve(),
        target_year=int(args.target_year),
    )
    output = args.output.expanduser()
    if not output.is_absolute():
        output = _root() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "ranking": {
                    backend: value.get("aggregate", value.get("status"))
                    for backend, value in payload["ranking"].items()
                },
                "lightgbm_quantile": payload["lightgbm_quantile"].get(
                    "aggregate", payload["lightgbm_quantile"].get("status")
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Suggested commit name: feat(f1): add real optional-model event-block evidence
