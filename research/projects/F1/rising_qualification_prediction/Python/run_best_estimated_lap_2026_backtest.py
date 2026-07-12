#!/usr/bin/env python3
"""Chronological 2026 backtest for achievable qualifying best-lap estimates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

from packages.f1.models.ultimate_lap_time.achievable import (
    ACTUAL_LAP_COLUMN,
    fit_achievable_best_lap_model,
)
from packages.f1.models.ultimate_lap_time.schemas import (
    ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT,
    TARGET_CONTRACT_SEMANTICS,
)
from packages.sports_core.paths import find_repo_root


SCHEMA_VERSION = "f1_best_estimated_lap_walk_forward_v2"
MODEL_NAME = "achievable_best_lap_rehearsal_shift_v1"
ROUND_PATTERN = re.compile(r"^round_(\d{2})_")
MIN_SUPPORTED_WEEKEND_YEAR = 2024


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


def _git_head(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


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
            (root / "packages/sports_core/paths.py").resolve(),
            *(path.resolve() for path in (root / "packages/f1/models/ultimate_lap_time").rglob("*.py")),
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


def _lap_seconds(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    unresolved = numeric.isna()
    if unresolved.any():
        timedeltas = pd.to_timedelta(values.where(unresolved), errors="coerce").dt.total_seconds()
        numeric = numeric.fillna(timedeltas)
    return pd.to_numeric(numeric, errors="coerce").astype(float)


def _bool_series(values: pd.Series, *, default: bool) -> pd.Series:
    if values.dtype == bool:
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


def _clean_driver_best_laps(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    required = {"Driver", "LapTime"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing lap columns: {missing}")
    seconds = _lap_seconds(frame["LapTime"])
    valid = seconds.between(40.0, 180.0) & np.isfinite(seconds)
    if "Deleted" in frame.columns:
        valid &= ~_bool_series(frame["Deleted"], default=False)
    if "IsAccurate" in frame.columns:
        valid &= _bool_series(frame["IsAccurate"], default=False)
    for column in ("PitInTime", "PitOutTime"):
        if column in frame.columns:
            valid &= frame[column].isna()
    clean = pd.DataFrame(
        {
            "driver_id": frame["Driver"].astype(str).str.strip(),
            "lap_time_seconds": seconds,
        },
        index=frame.index,
    ).loc[valid]
    clean = clean[clean["driver_id"] != ""]
    return clean.groupby("driver_id", sort=False)["lap_time_seconds"].min()


def _round_number(directory: Path) -> int:
    match = ROUND_PATTERN.match(directory.name)
    if match is None:
        raise ValueError(f"not an F1 round directory: {directory}")
    return int(match.group(1))


def _discover_rounds(weekends_dir: Path, year: int) -> list[Path]:
    directories = [path for path in (weekends_dir / str(year)).glob("round_*") if path.is_dir()]
    return sorted(directories, key=_round_number)


def _target_aligned_files(round_dir: Path) -> tuple[str, Path, Path]:
    sprint_qualifying = sorted(round_dir.glob("*_sprint_qualifying_laps.csv"))
    practice_3 = sorted(round_dir.glob("*_practice_3_laps.csv"))
    qualifying = sorted(
        path
        for path in round_dir.glob("*_qualifying_laps.csv")
        if "sprint_qualifying" not in path.name
    )
    if not qualifying:
        raise FileNotFoundError(f"{round_dir} has no Grand Prix qualifying laps")
    if sprint_qualifying:
        return "sprint_qualifying", sprint_qualifying[0], qualifying[0]
    if practice_3:
        return "practice_3", practice_3[0], qualifying[0]
    raise FileNotFoundError(
        f"{round_dir} has neither Sprint Qualifying nor FP3 as a target-aligned rehearsal"
    )


def _event_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    error = frame["lap_p50"] - frame[ACTUAL_LAP_COLUMN]
    raw_error = frame["rehearsal_lap_time_seconds"] - frame[ACTUAL_LAP_COLUMN]
    predicted_rank = frame["lap_p50"].rank(method="average", ascending=True)
    actual_rank = frame[ACTUAL_LAP_COLUMN].rank(method="average", ascending=True)
    interval_valid = frame["lap_p05"].notna() & frame["lap_p90"].notna()
    coverage = (
        (
            frame.loc[interval_valid, ACTUAL_LAP_COLUMN].ge(frame.loc[interval_valid, "lap_p05"])
            & frame.loc[interval_valid, ACTUAL_LAP_COLUMN].le(frame.loc[interval_valid, "lap_p90"])
        ).mean()
        if interval_valid.any()
        else float("nan")
    )
    predicted_top3 = set(frame.nsmallest(min(3, len(frame)), "lap_p50")["driver_id"])
    actual_top3 = set(frame.nsmallest(min(3, len(frame)), ACTUAL_LAP_COLUMN)["driver_id"])
    return {
        "rows": int(len(frame)),
        "p50_mae_seconds": float(error.abs().mean()),
        "raw_rehearsal_mae_seconds": float(raw_error.abs().mean()),
        "p50_rmse_seconds": float(np.sqrt(np.mean(np.square(error)))),
        "mean_error_seconds": float(error.mean()),
        "spearman": float(predicted_rank.corr(actual_rank, method="pearson")),
        "fastest_driver_hit": bool(frame.loc[frame["lap_p50"].idxmin(), "driver_id"] == frame.loc[frame[ACTUAL_LAP_COLUMN].idxmin(), "driver_id"]),
        "top3_overlap_rate": float(len(predicted_top3 & actual_top3) / max(1, min(3, len(frame)))),
        "interval_rows": int(interval_valid.sum()),
        "interval_coverage": float(coverage),
        "interval_mean_width_seconds": float(
            (frame.loc[interval_valid, "lap_p90"] - frame.loc[interval_valid, "lap_p05"]).mean()
        )
        if interval_valid.any()
        else float("nan"),
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
    delta = challenger_values - baseline_values
    rng = np.random.default_rng(int(seed))
    draws = delta[rng.integers(0, len(delta), size=(int(samples), len(delta)))].mean(axis=1)
    return {
        "mean_delta_seconds": float(delta.mean()),
        "ci95_seconds": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "bootstrap_probability_of_improvement": float(np.mean(draws < 0.0)),
        "samples": int(samples),
        "seed": int(seed),
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


def run_backtest(
    *,
    weekends_dir: Path,
    year: int,
    rounds: Sequence[int] | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if int(year) < MIN_SUPPORTED_WEEKEND_YEAR:
        raise ValueError(
            "best-lap rehearsal backtest supports 2024+ weekend chronology only; "
            "pre-2024 Sprint Qualifying/Sprint Shootout occurs after or does not exist "
            "before Grand Prix qualifying"
        )
    root = _repo_root()
    implementation_manifest = _hash_manifest(_implementation_paths(root), root=root)
    selected = [
        path
        for path in _discover_rounds(weekends_dir, year)
        if rounds is None or _round_number(path) in set(int(value) for value in rounds)
    ]
    if not selected:
        raise ValueError("no completed local rounds selected")
    selected_input_files = [
        path
        for round_dir in selected
        for path in _target_aligned_files(round_dir)[1:]
    ]
    input_manifest_before = _hash_manifest(selected_input_files, root=root)

    history_parts: list[pd.DataFrame] = []
    event_payloads: list[dict[str, Any]] = []
    all_scored: list[pd.DataFrame] = []
    input_files: list[Path] = []

    for round_dir in selected:
        round_number = _round_number(round_dir)
        event_key = (int(year) * 100) + int(round_number)
        source, rehearsal_path, qualifying_path = _target_aligned_files(round_dir)
        input_files.extend([rehearsal_path, qualifying_path])
        rehearsal = _clean_driver_best_laps(rehearsal_path)
        actual = _clean_driver_best_laps(qualifying_path)
        inference_ids = rehearsal.index
        common = inference_ids.intersection(actual.index)
        if inference_ids.empty or common.empty:
            raise ValueError(f"round {round_number} has no matched valid rehearsal/qualifying laps")

        history = (
            pd.concat(history_parts, ignore_index=True)
            if history_parts
            else pd.DataFrame()
        )
        model = fit_achievable_best_lap_model(history, target_event_key=event_key)
        inference = pd.DataFrame(
            {
                "event_key": event_key,
                "driver_id": inference_ids.astype(str),
                "rehearsal_source": source,
                "rehearsal_lap_time_seconds": rehearsal.loc[inference_ids].to_numpy(dtype=float),
            }
        )
        predictions = model.predict(inference)
        evaluated = predictions.copy()
        evaluated[ACTUAL_LAP_COLUMN] = evaluated["driver_id"].map(actual)
        evaluated["target_observed"] = evaluated[ACTUAL_LAP_COLUMN].notna()
        scored = evaluated.loc[evaluated["target_observed"]].copy()
        scored["round"] = int(round_number)
        scored["absolute_error_seconds"] = (
            scored["lap_p50"] - scored[ACTUAL_LAP_COLUMN]
        ).abs()
        metrics = _event_metrics(scored)
        event_payloads.append(
            {
                "round": int(round_number),
                "event_key": int(event_key),
                "event_directory": str(round_dir.relative_to(_repo_root())),
                "rehearsal_source": source,
                "rehearsal_file": str(rehearsal_path.relative_to(_repo_root())),
                "target_file": str(qualifying_path.relative_to(_repo_root())),
                "rehearsal_driver_count": int(len(rehearsal)),
                "target_driver_count": int(len(actual)),
                "matched_driver_count": int(len(common)),
                "causal_inference_driver_count": int(len(inference_ids)),
                "evaluation_union_driver_count": int(len(inference_ids.union(actual.index))),
                "target_observed_given_inference_rate": float(len(common) / len(inference_ids)),
                "inference_coverage_of_observed_target_rate": float(len(common) / len(actual)),
                "end_to_end_scored_union_rate": float(
                    len(common) / len(inference_ids.union(actual.index))
                ),
                "missing_target_drivers": sorted(set(rehearsal.index) - set(actual.index)),
                "missing_rehearsal_drivers": sorted(set(actual.index) - set(rehearsal.index)),
                "training_event_keys": list(model.training_event_keys),
                "metrics": metrics,
                "predictions": evaluated.to_dict(orient="records"),
            }
        )
        all_scored.append(scored)
        history_parts.append(
            pd.DataFrame(
                {
                    "event_key": event_key,
                    "driver_id": common.astype(str),
                    "rehearsal_source": source,
                    "rehearsal_lap_time_seconds": rehearsal.loc[common].to_numpy(dtype=float),
                    ACTUAL_LAP_COLUMN: actual.loc[common].to_numpy(dtype=float),
                }
            )
        )

    joined = pd.concat(all_scored, ignore_index=True)
    event_metrics = [payload["metrics"] for payload in event_payloads]
    paired = _paired_bootstrap(
        [float(item["p50_mae_seconds"]) for item in event_metrics],
        [float(item["raw_rehearsal_mae_seconds"]) for item in event_metrics],
        samples=int(bootstrap_samples),
        seed=int(bootstrap_seed),
    )
    point_retained = bool(paired["ci95_seconds"][1] < 0.0)
    interval_rows = joined["lap_p05"].notna() & joined["lap_p90"].notna()
    if set(input_files) != set(selected_input_files):
        raise RuntimeError("Best Lap accessed an unexpected input-file set")
    _assert_manifest_unchanged(
        implementation_manifest,
        root=root,
        label="Best Lap implementation",
    )
    _assert_manifest_unchanged(
        input_manifest_before,
        root=root,
        label="Best Lap input data",
    )
    return _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _utc_now(),
            "mode": "best_estimated_lap_time",
            "model": MODEL_NAME,
            "model_family": "robust_source_specific_rehearsal_calibration",
            "target_contract": ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT,
            "target_semantics": TARGET_CONTRACT_SEMANTICS[
                ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT
            ],
            "target_definition": "fastest valid session-end Grand Prix qualifying lap per driver",
            "theoretical_sector_floor_is_target": False,
            "invocation": {
                "year": int(year),
                "rounds": [_round_number(path) for path in selected],
                "bootstrap_samples": int(bootstrap_samples),
                "bootstrap_seed": int(bootstrap_seed),
            },
            "provenance": {
                "repo_root": str(root),
                "git_head": _git_head(root),
                "git_dirty": _git_dirty(root),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "source": "locked_local_weekend_csv",
            },
            "protocol": {
                "same_season_only": True,
                "training_window": "strictly_earlier_completed_2026_events",
                "round_order": "ascending_sequential",
                "standard_weekend_rehearsal": "FP3",
                "sprint_weekend_rehearsal": "Sprint Qualifying",
                "supported_weekend_era": "2024_plus",
                "target_outcome_available_to_inference": False,
                "one_heavy_training_job_at_a_time": True,
            },
            "aggregate": {
                "rounds": len(event_payloads),
                "rows": int(len(joined)),
                "causal_inference_rows": int(
                    sum(item["causal_inference_driver_count"] for item in event_payloads)
                ),
                "observed_target_rows": int(
                    sum(item["target_driver_count"] for item in event_payloads)
                ),
                "evaluation_union_rows": int(
                    sum(item["evaluation_union_driver_count"] for item in event_payloads)
                ),
                "conditional_event_mean_p50_mae_seconds": float(
                    np.mean([item["p50_mae_seconds"] for item in event_metrics])
                ),
                "conditional_row_weighted_p50_mae_seconds": float(
                    joined["absolute_error_seconds"].mean()
                ),
                "conditional_event_mean_raw_rehearsal_mae_seconds": float(
                    np.mean([item["raw_rehearsal_mae_seconds"] for item in event_metrics])
                ),
                "conditional_event_mean_spearman": float(
                    np.mean([item["spearman"] for item in event_metrics])
                ),
                "conditional_fastest_driver_hit_rate": float(
                    np.mean([item["fastest_driver_hit"] for item in event_metrics])
                ),
                "conditional_top3_overlap_rate": float(
                    np.mean([item["top3_overlap_rate"] for item in event_metrics])
                ),
                "target_observed_given_inference_rate": float(
                    len(joined)
                    / sum(item["causal_inference_driver_count"] for item in event_payloads)
                ),
                "inference_coverage_of_observed_target_rate": float(
                    len(joined) / sum(item["target_driver_count"] for item in event_payloads)
                ),
                "end_to_end_scored_union_rate": float(
                    len(joined)
                    / sum(item["evaluation_union_driver_count"] for item in event_payloads)
                ),
                "interval_rows": int(interval_rows.sum()),
                "interval_coverage": float(
                    (
                        joined.loc[interval_rows, ACTUAL_LAP_COLUMN].ge(joined.loc[interval_rows, "lap_p05"])
                        & joined.loc[interval_rows, ACTUAL_LAP_COLUMN].le(joined.loc[interval_rows, "lap_p90"])
                    ).mean()
                )
                if interval_rows.any()
                else float("nan"),
            },
            "paired_event_bootstrap_vs_raw_rehearsal_conditional_matched_population": paired,
            "decision": {
                "conditional_point_estimate_retained": point_retained,
                "full_mode_point_promoted": False,
                "probabilistic_intervals_promoted": False,
                "deep_model_promoted": False,
                "reason": (
                    "robust causal session-shift baseline improves conditional event-level MAE; full mode remains blocked by target availability and coverage"
                    if point_retained
                    else "point estimate did not clear paired event uncertainty"
                ),
                "full_mode_blockers": [
                    "valid_qualifying_lap_target_not_observed_for_every_causal_inference_driver",
                    "no_separate_probability_for_no_valid_qualifying_lap_or_non_participation",
                    "causal_rehearsal_roster_does_not_cover_every_observed_target_driver",
                ],
                "interval_blockers": [
                    "only_nine_completed_events",
                    "source_specific_calibration_has_fewer_events",
                    "no_disjoint_selection_calibration_final_audit_blocks",
                ],
                "deep_model_blockers": [
                    "no_2026_distance_normalized_car_telemetry_cache",
                    "rehearsal_feature_time_and_separate_q_target_time_provenance_required",
                    "deterministic_causal_baseline_must_be_beaten_first",
                ],
            },
            "events": event_payloads,
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
    parser.add_argument("--rounds", default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=200_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260711)
    parser.add_argument(
        "--output",
        default=str(
            _repo_root()
            / "artifacts/backtests/f1/best_estimated_lap/2026_walk_forward_rehearsal_shift_v1.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rounds = None
    if str(args.rounds).strip().lower() != "auto":
        rounds = [int(value.strip()) for value in str(args.rounds).split(",") if value.strip()]
    payload = run_backtest(
        weekends_dir=Path(args.weekends_dir).expanduser().resolve(),
        year=int(args.year),
        rounds=rounds,
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(output), "aggregate": payload["aggregate"], "decision": payload["decision"]}, indent=2))


if __name__ == "__main__":
    main()
