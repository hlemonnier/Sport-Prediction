#!/usr/bin/env python3
"""Causal same-season walk-forward backtest for emitted live next-lap forecasts.

One forecast is issued immediately after each eligible completed lap has been
assimilated.  That frozen ``next_lap_*`` forecast is scored against the same
driver's chronologically next eligible completed representative lap.  The
target row and observations arriving after issuance are never used to rebuild
the scored forecast.  For each target event, the convex weight is selected
using the mean event-level MAE of strictly earlier events only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from packages.f1.data.schemas.session import PredictionConfig
from packages.f1.models.live_race.predict import run_live_race_prediction
from packages.sports_core.paths import find_repo_root


SCHEMA_VERSION = "f1_live_next_lap_walk_forward_emitted_forecast_v3"
MODEL_NAME = "ssm_last_clean_lap_causal_blend_v1"
TARGET = "next_eligible_representative_lap_seconds"
ISSUANCE_PROTOCOL = "post_eligible_lap_assimilation_frozen_until_next_eligible_lap"
ROUND_PATTERN = re.compile(r"^round_(\d{2})_")
DEFAULT_WEIGHT_GRID = tuple(float(index) / 20.0 for index in range(21))


def _repo_root() -> Path:
    return find_repo_root(__file__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _git_head(root: Path) -> str | None:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value or None


def _git_dirty(root: Path) -> bool | None:
    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _implementation_paths(root: Path) -> list[Path]:
    return sorted(
        {
            Path(__file__).resolve(),
            (root / "packages/f1/data/schemas/session.py").resolve(),
            (root / "packages/sports_core/paths.py").resolve(),
            *(path.resolve() for path in (root / "packages/f1/models/live_race").rglob("*.py")),
        }
    )


def _hash_manifest(paths: Sequence[Path], *, root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): _sha256(path) for path in sorted(set(paths))}


def _assert_manifest_unchanged(
    before: dict[str, str],
    *,
    root: Path,
    label: str,
) -> None:
    after = {name: _sha256(root / name) for name in before}
    changed = sorted(name for name, digest in before.items() if after[name] != digest)
    if changed:
        raise RuntimeError(f"{label} changed during evaluation: {changed}")


def _round_number(directory: Path) -> int:
    match = ROUND_PATTERN.match(directory.name)
    if match is None:
        raise ValueError(f"not an F1 round directory: {directory}")
    return int(match.group(1))


def _discover_rounds(weekends_dir: Path, year: int) -> list[Path]:
    directories = [path for path in (weekends_dir / str(year)).glob("round_*") if path.is_dir()]
    return sorted(directories, key=_round_number)


def _race_laps_path(round_dir: Path) -> Path:
    candidates = sorted(round_dir.glob("*_race_laps.csv"))
    if len(candidates) != 1:
        raise ValueError(
            f"{round_dir} must contain exactly one race-laps CSV; found {len(candidates)}"
        )
    return candidates[0]


def _validate_weight_grid(values: Iterable[float]) -> tuple[float, ...]:
    grid = tuple(float(value) for value in values)
    if not grid:
        raise ValueError("weight grid cannot be empty")
    if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in grid):
        raise ValueError("every SSM weight must be finite and between zero and one")
    if tuple(sorted(set(grid))) != grid:
        raise ValueError("weight grid must be strictly increasing and contain no duplicates")
    return grid


def _as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapped = normalized.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "false": False,
            "0": False,
            "no": False,
        }
    )
    if mapped.isna().any():
        raise ValueError("boolean trace field contains unresolved values")
    return mapped.astype(bool)


def _validated_matched_rows(
    trace: pd.DataFrame,
    *,
    event_key: int,
    warmup_laps: int,
) -> pd.DataFrame:
    """Freeze emitted forecasts and match each to the next eligible target lap.

    Issuance occurs immediately after an eligible lap is assimilated.  The
    model and naive values are copied from that row's ``next_lap_*`` outputs,
    then joined to the same driver's next chronologically eligible completed
    lap.  No value computed on the target row is used as a forecast.
    """

    required = {
        "event_key",
        "driver_id",
        "lap_number",
        "timestamp",
        "timestamp_known",
        "baseline_information_order",
        "baseline_evidence_max_timestamp",
        "baseline_evidence_row_count",
        "eval_included",
        "assim_laps_driver",
        "lap_time_seconds",
        "next_lap_mean",
        "next_lap_mean_ssm",
        "next_lap_mean_naive",
        "next_lap_ssm_weight",
    }
    missing = sorted(required.difference(trace.columns))
    if missing:
        raise ValueError(f"live trace is missing required columns: {missing}")
    if trace.empty:
        raise ValueError("live trace is empty")

    work = trace.copy()
    event_values = pd.to_numeric(work["event_key"], errors="coerce")
    if event_values.isna().any() or set(event_values.astype(int)) != {int(event_key)}:
        raise ValueError("trace event key does not match the target event")

    timestamps = pd.to_numeric(work["timestamp"], errors="coerce")
    timestamp_known = _as_bool(work["timestamp_known"])
    if (~timestamp_known | ~np.isfinite(timestamps)).any():
        raise ValueError("noncausal trace rejected: every row needs a real global event timestamp")
    information_order = work["baseline_information_order"].astype(str)
    if not information_order.eq("global_event_time").all():
        bad = sorted(information_order[information_order.ne("global_event_time")].unique())
        raise ValueError(f"noncausal baseline information order rejected: {bad}")

    work["eval_included"] = _as_bool(work["eval_included"])
    for column in (
        "lap_number",
        "assim_laps_driver",
        "lap_time_seconds",
        "next_lap_mean",
        "next_lap_mean_ssm",
        "next_lap_mean_naive",
        "next_lap_ssm_weight",
        "baseline_evidence_max_timestamp",
        "baseline_evidence_row_count",
    ):
        work[column] = pd.to_numeric(work[column], errors="coerce")

    if work["baseline_evidence_row_count"].isna().any() or work[
        "baseline_evidence_row_count"
    ].lt(0).any():
        raise ValueError("baseline evidence row count must be a non-negative integer")
    evidence_present = work["baseline_evidence_row_count"].gt(0)
    evidence_max = work["baseline_evidence_max_timestamp"]
    if (evidence_present & ~np.isfinite(evidence_max)).any():
        raise ValueError("baseline evidence timestamp is missing despite non-empty evidence")
    if (
        np.isfinite(evidence_max)
        & evidence_max.ge(timestamps)
    ).any():
        raise ValueError("future baseline evidence reached or crossed the forecast issuance time")

    work["driver_id"] = work["driver_id"].astype(str)
    identity_columns = ["driver_id", "lap_number", "timestamp"]
    if work.duplicated(identity_columns).any():
        raise ValueError("live trace has duplicate driver/lap/event-time rows")

    matched: list[dict[str, Any]] = []
    for driver_id, driver_rows in work.groupby("driver_id", sort=False):
        ordered = driver_rows.sort_values(["timestamp", "lap_number"], kind="mergesort")
        eligible = ordered.loc[
            ordered["eval_included"] & np.isfinite(ordered["lap_time_seconds"])
        ].copy()
        if len(eligible) < 2:
            continue
        lap_values = eligible["lap_number"].to_numpy(dtype=float)
        time_values = eligible["timestamp"].to_numpy(dtype=float)
        if np.any(np.diff(lap_values) <= 0.0) or np.any(np.diff(time_values) <= 0.0):
            raise ValueError(
                f"eligible laps are not strictly chronological for driver {driver_id}"
            )

        eligible_rows = list(eligible.iterrows())
        for position in range(len(eligible_rows) - 1):
            _, issued = eligible_rows[position]
            _, target = eligible_rows[position + 1]
            if int(issued["assim_laps_driver"]) < int(warmup_laps):
                continue

            forecast_columns = (
                "next_lap_mean_ssm",
                "next_lap_mean_naive",
                "next_lap_mean",
                "next_lap_ssm_weight",
            )
            if not all(np.isfinite(float(issued[column])) for column in forecast_columns):
                raise ValueError(
                    "matched-population contract violated: an issuance row lacks SSM or naive forecast"
                )
            if not np.isclose(
                float(issued["next_lap_ssm_weight"]),
                1.0,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "core replay must run with SSM weight 1.0 before offline blend scoring"
                )
            if not np.isclose(
                float(issued["next_lap_mean"]),
                float(issued["next_lap_mean_ssm"]),
                rtol=0.0,
                atol=1e-10,
            ):
                raise ValueError(
                    "pure-SSM emitted forecast does not equal the recorded SSM component"
                )

            issued_at = float(issued["timestamp"])
            target_at = float(target["timestamp"])
            issued_after_lap = int(issued["lap_number"])
            target_lap = int(target["lap_number"])
            if target_at <= issued_at or target_lap <= issued_after_lap:
                raise ValueError("matched target must be strictly later than forecast issuance")
            evidence_cutoff = float(issued["baseline_evidence_max_timestamp"])
            if np.isfinite(evidence_cutoff) and evidence_cutoff >= issued_at:
                raise ValueError("emitted forecast contains global evidence unavailable at issuance")

            matched.append(
                {
                    "event_key": int(event_key),
                    "driver_id": str(driver_id),
                    "issued_after_lap_number": issued_after_lap,
                    "issued_at_timestamp": issued_at,
                    "target_lap_number": target_lap,
                    "target_timestamp": target_at,
                    "skipped_nonrepresentative_laps": max(
                        0, target_lap - issued_after_lap - 1
                    ),
                    "lap_time_seconds": float(target["lap_time_seconds"]),
                    "forecast_ssm_seconds": float(issued["next_lap_mean_ssm"]),
                    "forecast_naive_seconds": float(issued["next_lap_mean_naive"]),
                    "core_emitted_forecast_seconds": float(issued["next_lap_mean"]),
                    "core_replay_ssm_weight": float(issued["next_lap_ssm_weight"]),
                    "issuance_assimilated_laps": int(issued["assim_laps_driver"]),
                    "baseline_evidence_max_timestamp": (
                        evidence_cutoff if np.isfinite(evidence_cutoff) else float("nan")
                    ),
                    "baseline_evidence_row_count": int(
                        issued["baseline_evidence_row_count"]
                    ),
                    "issuance_protocol": ISSUANCE_PROTOCOL,
                }
            )

    rows = pd.DataFrame(matched)
    if rows.empty:
        raise ValueError("event has no emitted forecast matched after the causal warm-up")
    matched_identity = [
        "driver_id",
        "issued_after_lap_number",
        "issued_at_timestamp",
        "target_lap_number",
        "target_timestamp",
    ]
    if rows.duplicated(matched_identity).any():
        raise ValueError("matched scoring population contains duplicate forecast-target pairs")
    eligible_issuance = work.loc[
        work["eval_included"]
        & np.isfinite(work["lap_time_seconds"])
        & work["assim_laps_driver"].ge(int(warmup_laps))
        & np.isfinite(work["next_lap_mean_ssm"])
        & np.isfinite(work["next_lap_mean_naive"])
    ]
    result = rows.sort_values(matched_identity, kind="mergesort").reset_index(drop=True)
    result.attrs["eligible_issuance_rows"] = int(len(eligible_issuance))
    result.attrs["matched_next_eligible_rows"] = int(len(result))
    result.attrs["issuances_without_next_eligible_target"] = int(
        len(eligible_issuance) - len(result)
    )
    return result


def _prediction(rows: pd.DataFrame, weight: float) -> np.ndarray:
    ssm = rows["forecast_ssm_seconds"].to_numpy(dtype=float)
    naive = rows["forecast_naive_seconds"].to_numpy(dtype=float)
    return (float(weight) * ssm) + ((1.0 - float(weight)) * naive)


def _mae_for_weight(rows: pd.DataFrame, weight: float) -> float:
    actual = rows["lap_time_seconds"].to_numpy(dtype=float)
    return float(np.mean(np.abs(actual - _prediction(rows, weight))))


def _candidate_scores(
    prior_events: Sequence[pd.DataFrame],
    weight_grid: Sequence[float],
) -> dict[float, float]:
    if not prior_events:
        return {}
    return {
        float(weight): float(
            np.mean([_mae_for_weight(event_rows, float(weight)) for event_rows in prior_events])
        )
        for weight in weight_grid
    }


def _select_weight(
    prior_events: Sequence[pd.DataFrame],
    *,
    weight_grid: Sequence[float],
    cold_start_weight: float,
) -> tuple[float, dict[float, float]]:
    grid = _validate_weight_grid(weight_grid)
    cold = float(cold_start_weight)
    if cold not in grid:
        raise ValueError("cold-start weight must be declared in the weight grid")
    scores = _candidate_scores(prior_events, grid)
    if not scores:
        return cold, scores
    # Conservative deterministic tie-break: prefer less SSM exposure.
    selected = min(grid, key=lambda weight: (scores[float(weight)], float(weight)))
    return float(selected), scores


def _event_metrics(rows: pd.DataFrame, selected_weight: float) -> dict[str, Any]:
    actual = rows["lap_time_seconds"].to_numpy(dtype=float)
    blend = _prediction(rows, selected_weight)
    ssm = _prediction(rows, 1.0)
    naive = _prediction(rows, 0.0)

    def metrics(prediction: np.ndarray) -> tuple[float, float, float]:
        error = prediction - actual
        return (
            float(np.mean(np.abs(error))),
            float(np.sqrt(np.mean(np.square(error)))),
            float(np.mean(error)),
        )

    blend_mae, blend_rmse, blend_bias = metrics(blend)
    naive_mae, naive_rmse, naive_bias = metrics(naive)
    ssm_mae, ssm_rmse, ssm_bias = metrics(ssm)
    return {
        "rows": int(len(rows)),
        "selected_ssm_weight": float(selected_weight),
        "selected_naive_weight": float(1.0 - selected_weight),
        "blend_mae_seconds": blend_mae,
        "naive_mae_seconds": naive_mae,
        "ssm_mae_seconds": ssm_mae,
        "blend_rmse_seconds": blend_rmse,
        "naive_rmse_seconds": naive_rmse,
        "ssm_rmse_seconds": ssm_rmse,
        "blend_bias_seconds": blend_bias,
        "naive_bias_seconds": naive_bias,
        "ssm_bias_seconds": ssm_bias,
    }


def _paired_bootstrap(
    challenger: Sequence[float],
    baseline: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    challenger_values = np.asarray(challenger, dtype=float)
    baseline_values = np.asarray(baseline, dtype=float)
    if challenger_values.size == 0 or challenger_values.shape != baseline_values.shape:
        raise ValueError("paired bootstrap inputs must be non-empty and have equal shape")
    if int(samples) <= 0:
        raise ValueError("bootstrap samples must be positive")
    delta = challenger_values - baseline_values
    rng = np.random.default_rng(int(seed))
    draws = delta[
        rng.integers(0, len(delta), size=(int(samples), len(delta)))
    ].mean(axis=1)
    return {
        "mean_delta_seconds": float(delta.mean()),
        "ci95_seconds": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "bootstrap_probability_of_improvement": float(np.mean(draws < 0.0)),
        "samples": int(samples),
        "seed": int(seed),
        "unit": "event",
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _run_event(
    *,
    weekends_dir: Path,
    year: int,
    round_number: int,
    race_path: Path,
    live_seed: int,
    warmup_laps: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = PredictionConfig(
        source="local",
        mode="race",
        year=int(year),
        round_number=int(round_number),
        train_seasons=[int(year)],
        include_standings=False,
        cache_dir=None,
        meeting_name=None,
        country_name=None,
        weekends_dir=str(weekends_dir),
        enable_dl_candidates=False,
        compare_families=["baseline"],
        disable_runsim_features=True,
        disable_circuit_features=True,
        shadow_eval=False,
        f1_mode="live",
        f1_live_source="local",
        f1_live_model="ssm_v1",
        f1_live_horizon_laps=1,
        f1_live_seed=int(live_seed),
        f1_live_replay_path=str(race_path),
        f1_live_calibration_path=None,
        f1_live_next_lap_ssm_weight=1.0,
    )
    result = run_live_race_prediction(config)
    if not bool(result.summary.get("available", False)):
        raise ValueError(
            f"round {round_number} live replay unavailable: {result.summary.get('reason')}"
        )
    if str(result.summary.get("source_used")) != "local":
        raise ValueError(f"round {round_number} did not use the locked local replay input")
    event_key = (int(year) * 100) + int(round_number)
    rows = _validated_matched_rows(
        result.trace,
        event_key=event_key,
        warmup_laps=int(warmup_laps),
    )
    provenance = {
        "source_used": result.summary.get("source_used"),
        "trace_records": int(len(result.trace)),
        "matched_rows": int(len(rows)),
        "eligible_issuance_rows": int(rows.attrs.get("eligible_issuance_rows", len(rows))),
        "issuances_without_next_eligible_target": int(
            rows.attrs.get("issuances_without_next_eligible_target", 0)
        ),
        "matched_target_rate_given_eligible_issuance": float(
            len(rows) / max(1, int(rows.attrs.get("eligible_issuance_rows", len(rows))))
        ),
        "timestamp_rows_total": int(result.summary.get("timestamp_rows_total", 0)),
        "timestamp_rows_known": int(result.summary.get("timestamp_rows_known", 0)),
        "baseline_information_order_counts": result.summary.get(
            "baseline_information_order_counts", {}
        ),
        "filter_calibration": result.summary.get("filter_calibration"),
        "uses_hand_tuned_priors": bool(result.summary.get("uses_hand_tuned_priors", True)),
        "core_replay_ssm_weight": float(result.summary.get("next_lap_ssm_weight", 1.0)),
    }
    return rows, provenance


def run_backtest(
    *,
    weekends_dir: Path,
    year: int,
    rounds: Sequence[int] | None,
    weight_grid: Sequence[float],
    cold_start_weight: float,
    warmup_laps: int,
    live_seed: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    root = _repo_root()
    grid = _validate_weight_grid(weight_grid)
    requested = None if rounds is None else {int(value) for value in rounds}
    selected = [
        path
        for path in _discover_rounds(weekends_dir, int(year))
        if requested is None or _round_number(path) in requested
    ]
    if not selected:
        raise ValueError("no completed local rounds selected")
    if requested is not None and {_round_number(path) for path in selected} != requested:
        missing = sorted(requested.difference({_round_number(path) for path in selected}))
        raise ValueError(f"requested rounds are unavailable: {missing}")
    selected_rounds = [_round_number(path) for path in selected]
    if any(right <= left for left, right in zip(selected_rounds, selected_rounds[1:])):
        raise ValueError("events must be processed in strictly ascending round order")

    implementation_manifest = _hash_manifest(_implementation_paths(root), root=root)
    planned_input_paths = [_race_laps_path(path) for path in selected]
    input_manifest_before = {
        str(path.relative_to(root)): {
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
        }
        for path in planned_input_paths
    }

    prior_events: list[pd.DataFrame] = []
    prior_rounds: list[int] = []
    event_payloads: list[dict[str, Any]] = []
    input_paths: list[Path] = []

    for round_dir in selected:
        round_number = _round_number(round_dir)
        event_key = (int(year) * 100) + int(round_number)
        race_path = _race_laps_path(round_dir)
        input_paths.append(race_path)
        rows, replay_provenance = _run_event(
            weekends_dir=weekends_dir,
            year=int(year),
            round_number=round_number,
            race_path=race_path,
            live_seed=int(live_seed),
            warmup_laps=int(warmup_laps),
        )
        selected_weight, candidate_scores = _select_weight(
            prior_events,
            weight_grid=grid,
            cold_start_weight=float(cold_start_weight),
        )
        metrics = _event_metrics(rows, selected_weight)
        identity_payload = rows[
            [
                "driver_id",
                "issued_after_lap_number",
                "issued_at_timestamp",
                "target_lap_number",
                "target_timestamp",
            ]
        ].to_dict(orient="records")
        scoring_payload = rows[
            [
                "driver_id",
                "issued_after_lap_number",
                "issued_at_timestamp",
                "target_lap_number",
                "target_timestamp",
                "lap_time_seconds",
                "forecast_ssm_seconds",
                "forecast_naive_seconds",
            ]
        ].to_dict(orient="records")
        event_payloads.append(
            {
                "round": int(round_number),
                "event_key": int(event_key),
                "event_directory": str(round_dir.relative_to(root)),
                "race_laps_file": str(race_path.relative_to(root)),
                "selection_source": (
                    "explicit_cold_start" if not prior_events else "strictly_prior_event_mean_mae"
                ),
                "training_rounds": list(prior_rounds),
                "candidate_prior_event_mean_mae_seconds": {
                    f"{weight:.2f}": candidate_scores[float(weight)]
                    for weight in grid
                    if float(weight) in candidate_scores
                },
                "metrics": metrics,
                "replay_provenance": replay_provenance,
                "matched_row_identity_sha256": _sha256_json(identity_payload),
                "matched_scoring_payload_sha256": _sha256_json(scoring_payload),
                "matched_scoring_rows": scoring_payload,
            }
        )
        prior_events.append(rows)
        prior_rounds.append(int(round_number))

    metric_rows = [payload["metrics"] for payload in event_payloads]
    blend_mae = [float(item["blend_mae_seconds"]) for item in metric_rows]
    naive_mae = [float(item["naive_mae_seconds"]) for item in metric_rows]
    ssm_mae = [float(item["ssm_mae_seconds"]) for item in metric_rows]
    blend_vs_naive = _paired_bootstrap(
        blend_mae,
        naive_mae,
        samples=int(bootstrap_samples),
        seed=int(bootstrap_seed),
    )
    ssm_vs_naive = _paired_bootstrap(
        ssm_mae,
        naive_mae,
        samples=int(bootstrap_samples),
        seed=int(bootstrap_seed),
    )
    deployment_weight, deployment_scores = _select_weight(
        prior_events,
        weight_grid=grid,
        cold_start_weight=float(cold_start_weight),
    )
    point_forecast_retained = bool(blend_vs_naive["ci95_seconds"][1] < 0.0)
    deployment_ssm_weight = float(deployment_weight) if point_forecast_retained else 0.0
    if set(input_paths) != set(planned_input_paths):
        raise RuntimeError("Live next-lap evaluation accessed an unexpected input-file set")
    _assert_manifest_unchanged(
        implementation_manifest,
        root=root,
        label="Live next-lap implementation",
    )
    input_hashes = {name: payload["sha256"] for name, payload in input_manifest_before.items()}
    _assert_manifest_unchanged(
        input_hashes,
        root=root,
        label="Live next-lap input data",
    )

    return _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _utc_now(),
            "mode": "live_race_intelligence",
            "forecast_subcontract": "next_eligible_representative_lap_time",
            "target": TARGET,
            "target_definition": (
                "same driver's chronologically next eligible completed representative lap, "
                "predicted by the frozen next_lap forecast emitted immediately after the current "
                "eligible lap was assimilated"
            ),
            "evaluation_conditioning": (
                "timing error is conditional on another eligible representative lap occurring; "
                "issuances without a next eligible lap are reported as coverage outcomes"
            ),
            "model": MODEL_NAME,
            "invocation": {
                "year": int(year),
                "rounds": selected_rounds,
                "weight_grid": list(grid),
                "cold_start_ssm_weight": float(cold_start_weight),
                "warmup_laps": int(warmup_laps),
                "live_seed": int(live_seed),
                "bootstrap_samples": int(bootstrap_samples),
                "bootstrap_seed": int(bootstrap_seed),
            },
            "protocol": {
                "same_season_only": True,
                "year": int(year),
                "round_order": "strictly_ascending_sequential",
                "weight_selection": "minimum_strictly_prior_event_mean_mae",
                "target_event_excluded_from_weight_selection": True,
                "weight_grid": list(grid),
                "cold_start_ssm_weight": float(cold_start_weight),
                "tie_break": "lower_ssm_weight",
                "warmup_assimilated_laps": int(warmup_laps),
                "row_comparisons": "exact_matched_actual_ssm_naive_population_or_fail",
                "information_order": "global_event_time_only_or_fail",
                "issuance_protocol": ISSUANCE_PROTOCOL,
                "one_forecast_per_driver_per_eligible_assimilated_lap": True,
                "target_match": "same_driver_chronological_next_eligible_completed_lap",
                "target_row_forecast_columns_used": False,
                "forecast_refit_between_issuance_and_target": False,
                "global_evidence_must_precede_issuance": True,
                "bootstrap_unit": "event",
                "bootstrap_samples": int(bootstrap_samples),
                "bootstrap_seed": int(bootstrap_seed),
            },
            "events": event_payloads,
            "aggregate": {
                "rounds": int(len(event_payloads)),
                "rows": int(sum(int(item["rows"]) for item in metric_rows)),
                "eligible_issuance_rows": int(
                    sum(
                        int(item["replay_provenance"]["eligible_issuance_rows"])
                        for item in event_payloads
                    )
                ),
                "issuances_without_next_eligible_target": int(
                    sum(
                        int(
                            item["replay_provenance"][
                                "issuances_without_next_eligible_target"
                            ]
                        )
                        for item in event_payloads
                    )
                ),
                "matched_target_rate_given_eligible_issuance": float(
                    sum(int(item["rows"]) for item in metric_rows)
                    / max(
                        1,
                        sum(
                            int(item["replay_provenance"]["eligible_issuance_rows"])
                            for item in event_payloads
                        ),
                    )
                ),
                "event_mean_blend_mae_seconds": float(np.mean(blend_mae)),
                "event_mean_naive_mae_seconds": float(np.mean(naive_mae)),
                "event_mean_ssm_mae_seconds": float(np.mean(ssm_mae)),
                "event_mean_blend_rmse_seconds": float(
                    np.mean([float(item["blend_rmse_seconds"]) for item in metric_rows])
                ),
                "event_mean_naive_rmse_seconds": float(
                    np.mean([float(item["naive_rmse_seconds"]) for item in metric_rows])
                ),
                "event_mean_ssm_rmse_seconds": float(
                    np.mean([float(item["ssm_rmse_seconds"]) for item in metric_rows])
                ),
                "paired_blend_vs_naive": blend_vs_naive,
                "paired_ssm_vs_naive": ssm_vs_naive,
            },
            "next_event_configuration": {
                "research_selected_ssm_weight": float(deployment_weight),
                "research_selected_naive_weight": float(1.0 - deployment_weight),
                "research_runtime_default_ssm_weight": deployment_ssm_weight,
                "research_runtime_default_naive_weight": float(1.0 - deployment_ssm_weight),
                "research_runtime_gate": (
                    "paired_event_mae_passed"
                    if point_forecast_retained
                    else "blend_rejected_fallback_to_naive"
                ),
                "training_rounds": list(prior_rounds),
                "candidate_event_mean_mae_seconds": {
                    f"{weight:.2f}": deployment_scores[float(weight)] for weight in grid
                },
            },
            "decision": {
                "point_forecast_retained": point_forecast_retained,
                "pure_ssm_retained": bool(ssm_vs_naive["ci95_seconds"][1] < 0.0),
                "probabilistic_intervals_promoted": False,
                "rl_policy_promoted": False,
                "reason": (
                    "frozen emitted causal point blend clears paired event-level MAE gate"
                    if point_forecast_retained
                    else "frozen emitted blend does not clear the paired event-level MAE gate; naive fallback retained"
                ),
                "blockers": [
                    "hand_tuned_ssm_priors",
                    "only_nine_completed_same_season_events",
                    "interval_calibration_not_evaluated_by_point_mae",
                    "rl_requires_locked_simulator_ope_and_shadow_evidence",
                ],
            },
            "provenance": {
                "repo_root": str(root),
                "git_head": _git_head(root),
                "git_dirty": _git_dirty(root),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "source": "locked_local_race_laps_csv",
            },
            "input_manifest": input_manifest_before,
            "implementation_manifest": implementation_manifest,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weekends-dir",
        default=str(_repo_root() / "data/f1/raw/weekends"),
    )
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--rounds", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--weight-grid", default="0:1:0.05")
    parser.add_argument("--cold-start-ssm-weight", type=float, default=0.0)
    parser.add_argument("--warmup-laps", type=int, default=3)
    parser.add_argument("--live-seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=500_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260711)
    parser.add_argument(
        "--output",
        default=str(
            _repo_root()
            / "artifacts/backtests/f1/live_next_lap/2026_walk_forward_ssm_naive_emitted_forecast_v3.json"
        ),
    )
    return parser


def _parse_weight_grid(value: str) -> tuple[float, ...]:
    text = str(value).strip()
    if ":" not in text:
        return _validate_weight_grid(float(item.strip()) for item in text.split(",") if item.strip())
    parts = [float(item.strip()) for item in text.split(":")]
    if len(parts) != 3:
        raise ValueError("range weight grid must use start:stop:step")
    start, stop, step = parts
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("weight-grid step must be positive")
    count = int(round((stop - start) / step))
    if count < 0 or not np.isclose(start + (count * step), stop, atol=1e-12, rtol=0.0):
        raise ValueError("weight-grid range must terminate exactly at stop")
    return _validate_weight_grid(round(start + (index * step), 12) for index in range(count + 1))


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rounds = None
    if str(args.rounds).strip().lower() != "auto":
        rounds = [int(value.strip()) for value in str(args.rounds).split(",") if value.strip()]
    if int(args.warmup_laps) < 1:
        raise ValueError("warmup laps must be positive")
    payload = run_backtest(
        weekends_dir=Path(args.weekends_dir).expanduser().resolve(),
        year=int(args.year),
        rounds=rounds,
        weight_grid=_parse_weight_grid(args.weight_grid),
        cold_start_weight=float(args.cold_start_ssm_weight),
        warmup_laps=int(args.warmup_laps),
        live_seed=int(args.live_seed),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "aggregate": payload["aggregate"],
                "next_event_configuration": payload["next_event_configuration"],
                "decision": payload["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
