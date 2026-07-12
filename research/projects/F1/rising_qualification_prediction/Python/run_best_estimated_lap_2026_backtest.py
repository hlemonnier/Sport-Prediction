#!/usr/bin/env python3
"""Chronological 2026 backtest for achievable qualifying best-lap estimates."""

from __future__ import annotations
import repo_bootstrap  # noqa: F401

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from packages.f1.features.qualifying_lap import (
    build_quality_aware_rehearsal_features,
    finite_lap_seconds,
)
from packages.f1.models.ultimate_lap_time.achievable import (
    ACTUAL_LAP_COLUMN,
    LATENT_POTENTIAL_ANCHOR_COLUMN,
    fit_achievable_best_lap_model,
)
from packages.f1.models.ultimate_lap_time.schemas import (
    ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT,
    TARGET_CONTRACT_SEMANTICS,
)
from packages.sports_core.paths import find_repo_root


SCHEMA_VERSION = "f1_best_estimated_lap_walk_forward_v3_quality_aware"
MODEL_NAME = "achievable_best_lap_quality_aware_huber_v2"
QUALITY_LOCATION_MODEL_NAME = "achievable_best_lap_quality_aware_location_v2"
BASELINE_MODEL_NAME = "achievable_best_lap_rehearsal_shift_v1"
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
            (root / "packages/f1/features/qualifying_lap.py").resolve(),
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
    # A driver with exactly one finite clean lap is valid evidence.  Do not
    # apply a variance/MAD/sample-count gate here.
    seconds = finite_lap_seconds(frame["LapTime"])
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


def _quality_aware_rehearsal(
    path: Path,
    *,
    event_key: int,
    source: str,
    include_earlier_evidence: bool = True,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "Driver" not in frame.columns:
        raise ValueError(f"{path} is missing Driver")
    earlier_parts: list[pd.DataFrame] = []
    for earlier_path in (
        _earlier_evidence_paths(path, source=source) if include_earlier_evidence else []
    ):
        earlier = pd.read_csv(earlier_path)
        if "Driver" not in earlier.columns:
            continue
        earlier["rehearsal_source"] = _source_from_filename(earlier_path)
        earlier_parts.append(earlier)
    earlier_laps = pd.concat(earlier_parts, ignore_index=True) if earlier_parts else None
    roster_source = (
        pd.concat([frame, *earlier_parts], ignore_index=True, sort=False)
        if earlier_parts
        else frame
    )
    roster_columns = ["Driver"] + (["Team"] if "Team" in roster_source.columns else [])
    roster = roster_source[roster_columns].drop_duplicates("Driver", keep="first")
    features = build_quality_aware_rehearsal_features(
        frame,
        entrants=roster,
        earlier_laps=earlier_laps,
    )
    features["event_key"] = int(event_key)
    features["rehearsal_source"] = str(source)
    return features


def _earlier_evidence_paths(rehearsal_path: Path, *, source: str) -> list[Path]:
    if source == "practice_3":
        patterns = ("*_practice_1_laps.csv", "*_practice_2_laps.csv")
    elif source == "sprint_qualifying":
        patterns = ("*_practice_1_laps.csv",)
    else:
        patterns = ()
    return sorted(
        {
            candidate
            for pattern in patterns
            for candidate in rehearsal_path.parent.glob(pattern)
            if candidate != rehearsal_path
        }
    )


def _source_from_filename(path: Path) -> str:
    name = path.stem.lower()
    for source in ("practice_1", "practice_2", "practice_3", "sprint_qualifying"):
        if source in name:
            return source
    return "earlier_session"


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


def _build_weak_transfer_history(
    weekends_dir: Path,
    *,
    target_year: int,
) -> tuple[list[pd.DataFrame], list[Path], list[dict[str, Any]]]:
    """Build regulation-aware weak priors from invariant rehearsal-to-Q effects.

    Absolute lap time is never transferred: every label is the residual from
    that event's own causal rehearsal anchor. Older seasons receive explicit
    recency weights, and their team/driver effects are disabled by the model's
    ``weak_transfer_prior`` contract.
    """

    default_weights = {2022: 0.05, 2023: 0.08, 2024: 0.15, 2025: 0.30}
    parts: list[pd.DataFrame] = []
    files: list[Path] = []
    summary: list[dict[str, Any]] = []
    for season in range(max(2022, int(target_year) - 4), int(target_year)):
        weight = float(default_weights.get(season, max(0.03, 0.30 / (target_year - season))))
        used_events = 0
        skipped_events = 0
        rows = 0
        for round_dir in _discover_rounds(weekends_dir, season):
            try:
                source, rehearsal_path, qualifying_path = _target_aligned_files(round_dir)
            except FileNotFoundError:
                # 2022-23 Sprint weekends have no pre-GP-Qualifying FP3/SQ
                # source. They are excluded instead of using post-cutoff data.
                skipped_events += 1
                continue
            event_key = (int(season) * 100) + _round_number(round_dir)
            quality = _quality_aware_rehearsal(
                rehearsal_path,
                event_key=event_key,
                source=source,
                include_earlier_evidence=False,
            )
            actual = _clean_driver_best_laps(qualifying_path)
            quality[ACTUAL_LAP_COLUMN] = quality["driver_id"].map(actual)
            quality["has_valid_qualifying_lap"] = quality[ACTUAL_LAP_COLUMN].notna()
            quality["rehearsal_lap_time_seconds"] = quality["valid_clean_best_seconds"]
            quality["history_weight"] = weight
            quality["weak_transfer_prior"] = True
            parts.append(quality)
            files.extend([rehearsal_path, qualifying_path])
            used_events += 1
            rows += len(quality)
        summary.append(
            {
                "season": int(season),
                "event_weight": weight,
                "used_event_blocks": used_events,
                "skipped_no_causal_target_aligned_source": skipped_events,
                "entrant_rows": rows,
                "transferred_effects": [
                    "source_session_shift",
                    "invariant_quality_feature_residuals",
                    "valid_lap_hurdle_prior",
                ],
                "blocked_effects": ["absolute_lap_time", "team_effect", "driver_effect"],
            }
        )
    return parts, files, summary


def _validate_robust_residual_selector(
    weekends_dir: Path,
    *,
    target_year: int,
) -> dict[str, Any]:
    """Freeze residual on/off using the season before the target audit."""

    validation_year = int(target_year) - 1
    prior_parts, _, _ = _build_weak_transfer_history(
        weekends_dir, target_year=validation_year
    )
    history_parts = list(prior_parts)
    location_event_mae: list[float] = []
    robust_event_mae: list[float] = []
    used_rounds: list[int] = []
    for round_dir in _discover_rounds(weekends_dir, validation_year):
        try:
            source, rehearsal_path, qualifying_path = _target_aligned_files(round_dir)
        except FileNotFoundError:
            continue
        round_number = _round_number(round_dir)
        event_key = (validation_year * 100) + round_number
        features = _quality_aware_rehearsal(
            rehearsal_path,
            event_key=event_key,
            source=source,
            include_earlier_evidence=False,
        )
        actual = _clean_driver_best_laps(qualifying_path)
        history = pd.concat(history_parts, ignore_index=True) if history_parts else pd.DataFrame()
        location_model = fit_achievable_best_lap_model(
            history,
            target_event_key=event_key,
            enable_robust_residual=False,
            model_name=QUALITY_LOCATION_MODEL_NAME,
        )
        robust_model = fit_achievable_best_lap_model(
            history, target_event_key=event_key, enable_robust_residual=True
        )
        location = location_model.predict(features).set_index("driver_id")["lap_p50"]
        robust = robust_model.predict(features).set_index("driver_id")["lap_p50"]
        common = actual.index.intersection(location.dropna().index).intersection(robust.dropna().index)
        if len(common):
            location_event_mae.append(float((location.loc[common] - actual.loc[common]).abs().mean()))
            robust_event_mae.append(float((robust.loc[common] - actual.loc[common]).abs().mean()))
            used_rounds.append(round_number)
        labelled = features.copy()
        labelled[ACTUAL_LAP_COLUMN] = labelled["driver_id"].map(actual)
        labelled["has_valid_qualifying_lap"] = labelled[ACTUAL_LAP_COLUMN].notna()
        labelled["rehearsal_lap_time_seconds"] = labelled["valid_clean_best_seconds"]
        labelled["history_weight"] = 1.0
        labelled["weak_transfer_prior"] = False
        history_parts.append(labelled)
    if not location_event_mae:
        return {
            "validation_year": validation_year,
            "selected_enable_robust_residual": False,
            "status": "unavailable_no_validation_events_fail_closed",
            "rounds": [],
        }
    location_mae = float(np.mean(location_event_mae))
    robust_mae = float(np.mean(robust_event_mae))
    relative_gain = (
        float((location_mae - robust_mae) / location_mae)
        if location_mae > 0.0
        else float("nan")
    )
    event_deltas = np.asarray(robust_event_mae) - np.asarray(location_event_mae)
    leave_one_out = [
        float(np.delete(event_deltas, index).mean())
        for index in range(len(event_deltas))
        if len(event_deltas) > 1
    ]
    stable = bool(leave_one_out and all(value < 0.0 for value in leave_one_out))
    selected = bool(relative_gain >= 0.05 and stable)
    return {
        "validation_year": validation_year,
        "rounds": used_rounds,
        "quality_location_event_mean_mae_seconds": location_mae,
        "robust_residual_event_mean_mae_seconds": robust_mae,
        "robust_relative_gain": relative_gain,
        "leave_one_event_out_directionally_stable": stable,
        "minimum_relative_gain": 0.05,
        "selected_enable_robust_residual": selected,
        "status": "frozen_before_target_audit",
    }


def _event_metrics(frame: pd.DataFrame, *, baseline_column: str = "baseline_lap_p50") -> dict[str, Any]:
    error = frame["lap_p50"] - frame[ACTUAL_LAP_COLUMN]
    raw_error = frame[baseline_column] - frame[ACTUAL_LAP_COLUMN]
    predicted_rank = frame["lap_p50"].rank(method="average", ascending=True)
    baseline_rank = frame[baseline_column].rank(method="average", ascending=True)
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
    baseline_top3 = set(frame.nsmallest(min(3, len(frame)), baseline_column)["driver_id"])
    actual_top3 = set(frame.nsmallest(min(3, len(frame)), ACTUAL_LAP_COLUMN)["driver_id"])
    return {
        "rows": int(len(frame)),
        "p50_mae_seconds": float(error.abs().mean()),
        "raw_rehearsal_mae_seconds": float(raw_error.abs().mean()),
        "baseline_p50_mae_seconds": float(raw_error.abs().mean()),
        "challenger_minus_baseline_mae_seconds": float(
            error.abs().mean() - raw_error.abs().mean()
        ),
        "p50_rmse_seconds": float(np.sqrt(np.mean(np.square(error)))),
        "mean_error_seconds": float(error.mean()),
        "spearman": float(predicted_rank.corr(actual_rank, method="pearson")),
        "baseline_spearman": float(baseline_rank.corr(actual_rank, method="pearson")),
        "fastest_driver_hit": bool(frame.loc[frame["lap_p50"].idxmin(), "driver_id"] == frame.loc[frame[ACTUAL_LAP_COLUMN].idxmin(), "driver_id"]),
        "baseline_fastest_driver_hit": bool(
            frame.loc[frame[baseline_column].idxmin(), "driver_id"]
            == frame.loc[frame[ACTUAL_LAP_COLUMN].idxmin(), "driver_id"]
        ),
        "top3_overlap_rate": float(len(predicted_top3 & actual_top3) / max(1, min(3, len(frame)))),
        "baseline_top3_overlap_rate": float(
            len(baseline_top3 & actual_top3) / max(1, min(3, len(frame)))
        ),
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


def _event_stability(event_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = np.asarray(
        [float(item["challenger_minus_baseline_mae_seconds"]) for item in event_metrics],
        dtype=float,
    )
    improvements = np.maximum(-deltas, 0.0)
    total_gain = float(improvements.sum())
    leave_one_out = [
        float(np.delete(deltas, index).mean()) if len(deltas) > 1 else float("nan")
        for index in range(len(deltas))
    ]
    return {
        "event_deltas_seconds": deltas.tolist(),
        "leave_one_event_out_mean_deltas_seconds": leave_one_out,
        "leave_one_event_out_directionally_stable": bool(
            leave_one_out and all(value < 0.0 for value in leave_one_out if np.isfinite(value))
        ),
        "largest_event_share_of_positive_gain": (
            float(improvements.max() / total_gain) if total_gain > 0.0 else float("nan")
        ),
        "single_event_gain_concentration_gate_passed": bool(
            total_gain > 0.0 and float(improvements.max() / total_gain) <= 0.50
        ),
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
    use_weak_transfer_priors: bool = True,
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
    prior_parts, prior_files, prior_summary = (
        _build_weak_transfer_history(weekends_dir, target_year=int(year))
        if use_weak_transfer_priors
        else ([], [], [])
    )
    residual_selector = {
        "validation_year": int(year) - 1,
        "selected_enable_robust_residual": False,
        "status": "frozen_fail_closed_quality_location_default",
        "selection_rule": (
            "enable only after at least five percent validation MAE gain and "
            "directionally stable leave-one-event-out results"
        ),
        "validation_function": "_validate_robust_residual_selector",
    }
    enable_selected_residual = False
    selected_input_files: list[Path] = list(prior_files)
    for round_dir in selected:
        source, rehearsal_path, qualifying_path = _target_aligned_files(round_dir)
        selected_input_files.extend(
            [
                rehearsal_path,
                qualifying_path,
                *_earlier_evidence_paths(rehearsal_path, source=source),
            ]
        )
    input_manifest_before = _hash_manifest(selected_input_files, root=root)

    baseline_history_parts: list[pd.DataFrame] = []
    challenger_history_parts: list[pd.DataFrame] = list(prior_parts)
    event_payloads: list[dict[str, Any]] = []
    all_scored: list[pd.DataFrame] = []
    all_challenger_scored: list[pd.DataFrame] = []
    input_files: list[Path] = list(prior_files)

    for round_dir in selected:
        round_number = _round_number(round_dir)
        event_key = (int(year) * 100) + int(round_number)
        source, rehearsal_path, qualifying_path = _target_aligned_files(round_dir)
        input_files.extend(
            [
                rehearsal_path,
                qualifying_path,
                *_earlier_evidence_paths(rehearsal_path, source=source),
            ]
        )
        rehearsal = _clean_driver_best_laps(rehearsal_path)
        actual = _clean_driver_best_laps(qualifying_path)
        quality_features = _quality_aware_rehearsal(
            rehearsal_path, event_key=event_key, source=source
        )
        inference_ids = pd.Index(quality_features["driver_id"].astype(str))
        baseline_inference_ids = rehearsal.index
        common = baseline_inference_ids.intersection(actual.index)
        if inference_ids.empty or common.empty:
            raise ValueError(f"round {round_number} has no matched valid rehearsal/qualifying laps")

        baseline_history = (
            pd.concat(baseline_history_parts, ignore_index=True)
            if baseline_history_parts
            else pd.DataFrame()
        )
        challenger_history = (
            pd.concat(challenger_history_parts, ignore_index=True)
            if challenger_history_parts
            else pd.DataFrame()
        )
        baseline_model = fit_achievable_best_lap_model(
            baseline_history,
            target_event_key=event_key,
            enable_robust_residual=False,
        )
        location_model = fit_achievable_best_lap_model(
            challenger_history,
            target_event_key=event_key,
            enable_robust_residual=False,
            model_name=QUALITY_LOCATION_MODEL_NAME,
        )
        robust_model = fit_achievable_best_lap_model(
            challenger_history,
            target_event_key=event_key,
            enable_robust_residual=True,
        )
        challenger_model = robust_model if enable_selected_residual else location_model
        baseline_inference = pd.DataFrame(
            {
                "event_key": event_key,
                "driver_id": baseline_inference_ids.astype(str),
                "rehearsal_source": source,
                "rehearsal_lap_time_seconds": rehearsal.loc[
                    baseline_inference_ids
                ].to_numpy(dtype=float),
            }
        )
        baseline_predictions = baseline_model.predict(baseline_inference)
        baseline_indexed = baseline_predictions.set_index("driver_id")
        baseline_by_driver = baseline_indexed["lap_p50"]
        predictions = challenger_model.predict(quality_features)
        robust_diagnostic = robust_model.predict(quality_features).set_index("driver_id")
        location_diagnostic = location_model.predict(quality_features).set_index("driver_id")
        evaluated = predictions.copy()
        evaluated["quality_location_lap_p50"] = evaluated["driver_id"].map(
            location_diagnostic["lap_p50"]
        )
        evaluated["robust_residual_lap_p50"] = evaluated["driver_id"].map(
            robust_diagnostic["lap_p50"]
        )
        evaluated["selected_residual_enabled"] = enable_selected_residual
        evaluated["baseline_lap_p50"] = evaluated["driver_id"].map(baseline_by_driver)
        evaluated["baseline_lap_p05"] = evaluated["driver_id"].map(
            baseline_indexed["lap_p05"]
        )
        evaluated["baseline_lap_p90"] = evaluated["driver_id"].map(
            baseline_indexed["lap_p90"]
        )
        evaluated[ACTUAL_LAP_COLUMN] = evaluated["driver_id"].map(actual)
        evaluated["target_observed"] = evaluated[ACTUAL_LAP_COLUMN].notna()
        evaluated["baseline_available"] = evaluated["baseline_lap_p50"].notna()
        challenger_scored = evaluated.loc[
            evaluated["target_observed"] & evaluated["lap_p50"].notna()
        ].copy()
        scored = challenger_scored.loc[challenger_scored["baseline_available"]].copy()
        scored["round"] = int(round_number)
        scored["absolute_error_seconds"] = (
            scored["lap_p50"] - scored[ACTUAL_LAP_COLUMN]
        ).abs()
        scored["baseline_absolute_error_seconds"] = (
            scored["baseline_lap_p50"] - scored[ACTUAL_LAP_COLUMN]
        ).abs()
        challenger_scored["round"] = int(round_number)
        challenger_scored["absolute_error_seconds"] = (
            challenger_scored["lap_p50"] - challenger_scored[ACTUAL_LAP_COLUMN]
        ).abs()
        metrics = _event_metrics(scored)
        observed_target_coverage = float(len(challenger_scored) / len(actual)) if len(actual) else 0.0
        comparison_rows = evaluated.assign(
            challenger_absolute_error_seconds=(
                evaluated["lap_p50"] - evaluated[ACTUAL_LAP_COLUMN]
            ).abs(),
            baseline_absolute_error_seconds=(
                evaluated["baseline_lap_p50"] - evaluated[ACTUAL_LAP_COLUMN]
            ).abs(),
        )
        event_payloads.append(
            {
                "round": int(round_number),
                "event_key": int(event_key),
                "event_directory": str(round_dir.relative_to(_repo_root())),
                "rehearsal_source": source,
                "rehearsal_file": str(rehearsal_path.relative_to(_repo_root())),
                "target_file": str(qualifying_path.relative_to(_repo_root())),
                "rehearsal_driver_count": int(len(rehearsal)),
                "quality_aware_inference_driver_count": int(len(inference_ids)),
                "target_driver_count": int(len(actual)),
                "matched_driver_count": int(len(common)),
                "causal_inference_driver_count": int(len(inference_ids)),
                "evaluation_union_driver_count": int(len(inference_ids.union(actual.index))),
                "target_observed_given_inference_rate": float(
                    evaluated["target_observed"].mean()
                ),
                "inference_coverage_of_observed_target_rate": observed_target_coverage,
                "end_to_end_scored_union_rate": float(
                    len(challenger_scored) / len(inference_ids.union(actual.index))
                ),
                "missing_target_drivers": sorted(set(inference_ids) - set(actual.index)),
                "missing_rehearsal_drivers": sorted(set(actual.index) - set(inference_ids)),
                "baseline_training_event_keys": list(baseline_model.training_event_keys),
                "challenger_training_event_keys": list(challenger_model.training_event_keys),
                "residual_selector": residual_selector,
                "metrics": metrics,
                "prediction_vs_reality": comparison_rows.to_dict(orient="records"),
                "predictions": comparison_rows.to_dict(orient="records"),
            }
        )
        all_scored.append(scored)
        all_challenger_scored.append(challenger_scored)
        baseline_history_parts.append(
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
        labelled_quality = quality_features.copy()
        labelled_quality[ACTUAL_LAP_COLUMN] = labelled_quality["driver_id"].map(actual)
        labelled_quality["has_valid_qualifying_lap"] = labelled_quality[
            ACTUAL_LAP_COLUMN
        ].notna()
        # Retain the raw rehearsal lap for baseline diagnostics, even though
        # the challenger fits against the latent potential-adjusted anchor.
        labelled_quality["rehearsal_lap_time_seconds"] = labelled_quality[
            "valid_clean_best_seconds"
        ]
        labelled_quality["history_weight"] = 1.0
        labelled_quality["weak_transfer_prior"] = False
        challenger_history_parts.append(labelled_quality)

    joined = pd.concat(all_scored, ignore_index=True)
    joined_challenger = pd.concat(all_challenger_scored, ignore_index=True)
    event_metrics = [payload["metrics"] for payload in event_payloads]
    paired = _paired_bootstrap(
        [float(item["p50_mae_seconds"]) for item in event_metrics],
        [float(item["raw_rehearsal_mae_seconds"]) for item in event_metrics],
        samples=int(bootstrap_samples),
        seed=int(bootstrap_seed),
    )
    stability = _event_stability(event_metrics)
    challenger_event_mae = float(np.mean([item["p50_mae_seconds"] for item in event_metrics]))
    baseline_event_mae = float(
        np.mean([item["baseline_p50_mae_seconds"] for item in event_metrics])
    )
    relative_mae_gain = (
        float((baseline_event_mae - challenger_event_mae) / baseline_event_mae)
        if baseline_event_mae > 0.0
        else float("nan")
    )
    target_rows = sum(item["target_driver_count"] for item in event_payloads)
    observed_target_coverage = float(len(joined_challenger) / target_rows)
    interval_rows = joined_challenger["lap_p05"].notna() & joined_challenger["lap_p90"].notna()
    interval_coverage = (
        float(
            (
                joined_challenger.loc[interval_rows, ACTUAL_LAP_COLUMN].ge(
                    joined_challenger.loc[interval_rows, "lap_p05"]
                )
                & joined_challenger.loc[interval_rows, ACTUAL_LAP_COLUMN].le(
                    joined_challenger.loc[interval_rows, "lap_p90"]
                )
            ).mean()
        )
        if interval_rows.any()
        else float("nan")
    )
    interval_width = (
        float(
            (
                joined_challenger.loc[interval_rows, "lap_p90"]
                - joined_challenger.loc[interval_rows, "lap_p05"]
            ).mean()
        )
        if interval_rows.any()
        else float("nan")
    )
    baseline_interval_rows = joined["baseline_lap_p05"].notna() & joined["baseline_lap_p90"].notna()
    baseline_interval_width = (
        float(
            (
                joined.loc[baseline_interval_rows, "baseline_lap_p90"]
                - joined.loc[baseline_interval_rows, "baseline_lap_p05"]
            ).mean()
        )
        if baseline_interval_rows.any()
        else float("nan")
    )
    fastest_non_worse = float(np.mean([item["fastest_driver_hit"] for item in event_metrics])) >= float(
        np.mean([item["baseline_fastest_driver_hit"] for item in event_metrics])
    )
    top3_non_worse = float(np.mean([item["top3_overlap_rate"] for item in event_metrics])) >= float(
        np.mean([item["baseline_top3_overlap_rate"] for item in event_metrics])
    )
    interval_coverage_gate = bool(np.isfinite(interval_coverage) and 0.80 <= interval_coverage <= 0.90)
    width_gate = bool(
        np.isfinite(interval_width)
        and np.isfinite(baseline_interval_width)
        and interval_width <= baseline_interval_width * 1.10
    )
    promotion_gates = {
        "mae_improves_at_least_five_percent": bool(relative_mae_gain >= 0.05),
        "event_bootstrap_upper_bound_below_zero": bool(paired["ci95_seconds"][1] < 0.0),
        "bootstrap_probability_of_improvement_at_least_095": bool(
            paired["bootstrap_probability_of_improvement"] >= 0.95
        ),
        "observed_target_coverage_is_100_percent": bool(observed_target_coverage >= 1.0),
        "fastest_driver_non_worse": bool(fastest_non_worse),
        "top3_non_worse": bool(top3_non_worse),
        "interval_coverage_within_five_points_of_85_percent": interval_coverage_gate,
        "interval_width_inflation_at_most_ten_percent": width_gate,
        "leave_one_event_out_directionally_stable": bool(
            stability["leave_one_event_out_directionally_stable"]
        ),
        "no_single_event_supplies_more_than_half_gain": bool(
            stability["single_event_gain_concentration_gate_passed"]
        ),
    }
    point_retained = bool(all(promotion_gates.values()))
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
            "model": MODEL_NAME if enable_selected_residual else QUALITY_LOCATION_MODEL_NAME,
            "robust_residual_diagnostic_model": MODEL_NAME,
            "baseline_model": BASELINE_MODEL_NAME,
            "model_family": (
                "quality_aware_anchor_plus_huber_hierarchical_residual_and_event_block_conformal"
                if enable_selected_residual
                else "quality_aware_anchor_with_validation_rejected_huber_residual_and_event_block_conformal"
            ),
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
                "use_weak_transfer_priors": bool(use_weak_transfer_priors),
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
                "validation_unit": "complete_chronological_event_block",
                "weak_transfer_prior_summary": prior_summary,
                "robust_residual_selector": residual_selector,
                "baseline_and_challenger_scored_on_same_rows": True,
                "deleted_laps_are_potential_only": True,
                "one_heavy_training_job_at_a_time": True,
            },
            "aggregate": {
                "rounds": len(event_payloads),
                "rows": int(len(joined_challenger)),
                "paired_baseline_rows": int(len(joined)),
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
                    challenger_event_mae
                ),
                "conditional_row_weighted_p50_mae_seconds": float(
                    joined_challenger["absolute_error_seconds"].mean()
                ),
                "conditional_event_mean_raw_rehearsal_mae_seconds": float(
                    baseline_event_mae
                ),
                "conditional_event_mean_baseline_p50_mae_seconds": baseline_event_mae,
                "relative_event_mean_mae_gain": relative_mae_gain,
                "conditional_event_mean_spearman": float(
                    np.mean([item["spearman"] for item in event_metrics])
                ),
                "conditional_fastest_driver_hit_rate": float(
                    np.mean([item["fastest_driver_hit"] for item in event_metrics])
                ),
                "baseline_fastest_driver_hit_rate": float(
                    np.mean([item["baseline_fastest_driver_hit"] for item in event_metrics])
                ),
                "conditional_top3_overlap_rate": float(
                    np.mean([item["top3_overlap_rate"] for item in event_metrics])
                ),
                "baseline_top3_overlap_rate": float(
                    np.mean([item["baseline_top3_overlap_rate"] for item in event_metrics])
                ),
                "target_observed_given_inference_rate": float(
                    len(joined_challenger)
                    / sum(item["causal_inference_driver_count"] for item in event_payloads)
                ),
                "inference_coverage_of_observed_target_rate": observed_target_coverage,
                "end_to_end_scored_union_rate": float(
                    len(joined_challenger)
                    / sum(item["evaluation_union_driver_count"] for item in event_payloads)
                ),
                "interval_rows": int(interval_rows.sum()),
                "interval_coverage": interval_coverage,
                "interval_mean_width_seconds": interval_width,
                "baseline_interval_mean_width_seconds": baseline_interval_width,
            },
            "paired_event_bootstrap_vs_retained_baseline_conditional_matched_population": paired,
            "paired_event_bootstrap_vs_raw_rehearsal_conditional_matched_population": paired,
            "event_stability": stability,
            "promotion_gates": promotion_gates,
            "decision": {
                "conditional_point_estimate_retained": point_retained,
                "full_mode_point_promoted": point_retained,
                "probabilistic_intervals_promoted": bool(
                    point_retained and interval_coverage_gate and width_gate
                ),
                "deep_model_promoted": False,
                "reason": (
                    "quality-aware selected challenger cleared every declared promotion gate"
                    if point_retained
                    else "quality-aware selected challenger retained as diagnostic because one or more declared gates failed"
                ),
                "failed_promotion_gates": [
                    name for name, passed in promotion_gates.items() if not passed
                ],
                "full_mode_blockers": [
                    name for name, passed in promotion_gates.items() if not passed
                ],
                "interval_blockers": [
                    name
                    for name, passed in {
                        "interval_coverage_within_five_points_of_85_percent": interval_coverage_gate,
                        "interval_width_inflation_at_most_ten_percent": width_gate,
                    }.items()
                    if not passed
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
        "--no-weak-transfer-priors",
        action="store_true",
        help="disable recency-weighted 2022-2025 invariant-effect priors",
    )
    parser.add_argument(
        "--output",
        default=str(
            _repo_root()
            / "artifacts/backtests/f1/best_estimated_lap/2026_walk_forward_quality_aware_huber_v2.json"
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
        use_weak_transfer_priors=not bool(args.no_weak_transfer_priors),
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(output), "aggregate": payload["aggregate"], "decision": payload["decision"]}, indent=2))


if __name__ == "__main__":
    main()
