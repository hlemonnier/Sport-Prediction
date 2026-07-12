#!/usr/bin/env python3
"""Point-in-time 2026 F1 walk-forward backtest.

The default arm trains only on completed prior rounds from the target season.
Cross-regime transfer is available only as an explicitly named comparison arm;
it is never silently pooled into the 2026 baseline.
"""

from __future__ import annotations
import repo_bootstrap  # noqa: F401

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Optional, Sequence

_EARLY_RESOURCE_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _resource_variable in _EARLY_RESOURCE_ENVIRONMENT_VARIABLES:
    os.environ.setdefault(_resource_variable, "1")

import pandas as pd
import yaml

from packages.f1 import PredictionConfig, run_prediction
from packages.f1.orchestration.backtest import evaluate_prediction_rows
from packages.f1.data.providers import FastF1Provider, LocalWeekendProvider, OpenF1Provider
from packages.f1.orchestration.evidence import f1_runtime_manifest
from packages.f1.orchestration.runtime import parse_compare_families, parse_json_object


ROLLING_BACKTEST_SCHEMA_VERSION = "f1_rolling_2026_point_in_time_v2"
ROUND_DIRECTORY_PATTERN = re.compile(r"^round_(\d+)(?:_|$)", re.IGNORECASE)
RESOURCE_ENVIRONMENT_VARIABLES = _EARLY_RESOURCE_ENVIRONMENT_VARIABLES


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def default_output_dir() -> str:
    return str(_project_root() / "artifacts" / "backtests" / "f1" / "rolling_2026")


def default_weekends_dir() -> str:
    return str(_project_root() / "data" / "f1" / "raw" / "weekends")


def default_selection_csv() -> str:
    return str(_project_root() / "artifacts" / "reports" / "f1" / "docs" / "F1_2026_Rolling_Backtest_Selections.csv")


def default_summary_csv() -> str:
    return str(_project_root() / "artifacts" / "reports" / "f1" / "docs" / "F1_2026_Rolling_Backtest_Summary.csv")


def default_qualifying_profile() -> str:
    return str(_project_root() / "configs" / "f1" / "profiles" / "pre_quali.yaml")


def default_race_profile() -> str:
    return str(_project_root() / "configs" / "f1" / "profiles" / "pre_race.yaml")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path, *, project_root: Optional[Path] = None) -> str:
    root = (project_root or _project_root()).resolve()
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _file_manifest(
    paths: Sequence[Path],
    *,
    manifest_version: str,
    project_root: Optional[Path] = None,
) -> dict[str, Any]:
    root = (project_root or _project_root()).resolve()
    files: list[dict[str, Any]] = []
    for path in sorted({item.expanduser().resolve() for item in paths}, key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        files.append(
            {
                "path": _portable_path(path, project_root=root),
                "sha256": _sha256_file(path),
                "byte_size": int(path.stat().st_size),
            }
        )
    return {
        "manifest_version": manifest_version,
        "file_count": int(len(files)),
        "aggregate_sha256": _sha256_json(files),
        "files": files,
    }


def _git_state(project_root: Optional[Path] = None) -> dict[str, Any]:
    root = (project_root or _project_root()).resolve()

    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    status = run("status", "--short", "--untracked-files=all")
    return {
        "head_sha": run("rev-parse", "HEAD") or None,
        "dirty": bool(status),
        "status_porcelain_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "branch": run("branch", "--show-current") or None,
    }


def _runtime_manifest(max_threads: int) -> dict[str, Any]:
    runtime = dict(f1_runtime_manifest())
    extended_packages: dict[str, Optional[str]] = {}
    for distribution in (
        "scipy",
        "xgboost",
        "lightgbm",
        "catboost",
        "torch",
        "fastf1",
        "PyYAML",
        "requests",
    ):
        try:
            extended_packages[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            extended_packages[distribution] = None
    runtime.update(
        {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "pandas": pd.__version__,
            "extended_packages": extended_packages,
            "resource_limits": {
                "max_threads": int(max_threads),
                "environment": {
                    name: os.environ.get(name) for name in RESOURCE_ENVIRONMENT_VARIABLES
                },
            },
        }
    )
    return runtime


def _apply_resource_limits(max_threads: int) -> None:
    if int(max_threads) < 1:
        raise ValueError("--max-threads must be >= 1")
    value = str(int(max_threads))
    for name in RESOURCE_ENVIRONMENT_VARIABLES:
        os.environ[name] = value


def _round_number_from_directory(path: Path) -> Optional[int]:
    match = ROUND_DIRECTORY_PATTERN.match(path.name)
    return int(match.group(1)) if match else None


def _has_completed_backtest_targets(round_dir: Path) -> bool:
    qualifying = list(round_dir.glob("*_qualifying_results.csv"))
    race = list(round_dir.glob("*_race_results.csv"))

    def has_rows(path: Path) -> bool:
        try:
            return not pd.read_csv(path, nrows=1).empty
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            return False

    return any(has_rows(path) for path in qualifying) and any(has_rows(path) for path in race)


def _available_local_rounds(weekends_dir: str, year: int) -> list[int]:
    root = Path(weekends_dir).expanduser()
    if not root.is_absolute():
        root = _project_root() / root
    year_dir = root / str(int(year))
    if not year_dir.exists():
        return []
    available: list[int] = []
    for round_dir in year_dir.iterdir():
        if not round_dir.is_dir():
            continue
        round_number = _round_number_from_directory(round_dir)
        if round_number is not None and _has_completed_backtest_targets(round_dir):
            available.append(round_number)
    return sorted(set(available))


def _resolve_rounds(value: str, *, source: str, weekends_dir: str, year: int) -> list[int]:
    if str(value).strip().lower() not in {"auto", "available", "latest"}:
        rounds = sorted(set(_parse_int_list(value)))
        if not rounds or any(round_number <= 0 for round_number in rounds):
            raise ValueError("--rounds must contain positive round numbers")
        return rounds
    if source != "local":
        raise ValueError("--rounds auto is intentionally local-only; specify explicit completed rounds for remote providers")
    rounds = _available_local_rounds(weekends_dir, year)
    if not rounds:
        raise ValueError(f"No completed local qualifying-and-race rounds found for {year}")
    return rounds


def _resolve_training_protocol(
    *,
    year: int,
    experiment_arm: str,
    transfer_train_seasons: Optional[str],
    legacy_base_train_seasons: Optional[str],
    current_season_weight_multiplier: Optional[float],
) -> dict[str, Any]:
    transfer_value = transfer_train_seasons or legacy_base_train_seasons
    if experiment_arm == "same_season_walk_forward":
        if transfer_value:
            raise ValueError(
                "Cross-season inputs are forbidden in the default same-season arm; "
                "use --experiment-arm explicit_transfer with --transfer-train-seasons.",
            )
        if current_season_weight_multiplier not in {None, 1.0}:
            raise ValueError("The same-season arm has no cross-season reweighting; multiplier must be 1.0")
        return {
            "experiment_arm": "same_season_walk_forward",
            "same_season_only": True,
            "transfer_train_seasons": [],
            "train_seasons_used": [int(year)],
            "current_season_weight_multiplier": 1.0,
            "cross_regime_comparability": "primary_2026_same_regime_arm",
        }
    if experiment_arm != "explicit_transfer":
        raise ValueError(f"Unknown experiment arm: {experiment_arm}")
    if not transfer_value:
        raise ValueError("The explicit_transfer arm requires --transfer-train-seasons")
    transfer = sorted(set(_parse_int_list(transfer_value)))
    if not transfer or any(season >= int(year) for season in transfer):
        raise ValueError("Transfer seasons must be non-empty historical seasons strictly before the target season")
    multiplier = float(current_season_weight_multiplier if current_season_weight_multiplier is not None else 3.0)
    if multiplier <= 0.0:
        raise ValueError("--current-season-weight-multiplier must be > 0")
    return {
        "experiment_arm": "explicit_transfer",
        "same_season_only": False,
        "transfer_train_seasons": transfer,
        "train_seasons_used": sorted({*transfer, int(year)}),
        "current_season_weight_multiplier": multiplier,
        "cross_regime_comparability": "exploratory_non_primary_transfer_arm",
    }


def _implementation_manifest() -> dict[str, Any]:
    root = _project_root()
    paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("repo_bootstrap.py"),
        Path(__file__).resolve().with_name("pyproject.toml"),
        *sorted((root / "packages" / "f1").rglob("*.py")),
        *sorted((root / "packages" / "sports_core").rglob("*.py")),
    ]
    return _file_manifest(paths, manifest_version="f1_rolling_source_tree_v1", project_root=root)


def _configuration_file_manifest() -> dict[str, Any]:
    root = _project_root()
    config_root = root / "configs" / "f1"
    paths = [path for path in config_root.rglob("*") if path.is_file() and path.name != ".DS_Store"]
    return _file_manifest(paths, manifest_version="f1_configuration_files_v1", project_root=root)


def _data_paths_for_round(
    *,
    weekends_dir: str,
    target_year: int,
    target_round: int,
    transfer_train_seasons: Sequence[int],
) -> list[Path]:
    root = Path(weekends_dir).expanduser()
    if not root.is_absolute():
        root = _project_root() / root
    paths: list[Path] = []
    for season in sorted(set(int(value) for value in transfer_train_seasons)):
        season_dir = root / str(season)
        if season_dir.exists():
            paths.extend(path for path in season_dir.rglob("*") if path.is_file())
    target_dir = root / str(int(target_year))
    if target_dir.exists():
        for round_dir in target_dir.iterdir():
            if not round_dir.is_dir():
                continue
            round_number = _round_number_from_directory(round_dir)
            if round_number is not None and round_number <= int(target_round):
                paths.extend(path for path in round_dir.rglob("*") if path.is_file())
    return paths


@contextmanager
def _capture_csv_reads():
    """Capture actual pandas CSV reads during one prediction/evaluation call."""

    original = pd.read_csv
    accessed: list[Path] = []

    def traced(path_or_buffer: object, *args: object, **kwargs: object):
        if isinstance(path_or_buffer, (str, os.PathLike)):
            accessed.append(Path(path_or_buffer).expanduser().resolve())
        return original(path_or_buffer, *args, **kwargs)

    pd.read_csv = traced  # type: ignore[assignment]
    try:
        yield accessed
    finally:
        pd.read_csv = original  # type: ignore[assignment]


def _assert_no_current_target_label_access(
    paths: Sequence[Path],
    *,
    target_year: int,
    target_round: int,
    target: str,
) -> None:
    forbidden: list[str] = []
    for path in paths:
        round_number = _round_number_from_directory(path.parent)
        try:
            event_year = int(path.parent.parent.name)
        except (TypeError, ValueError):
            continue
        if event_year != int(target_year) or round_number != int(target_round):
            continue
        name = path.name.lower()
        is_gp_qualifying_label = (
            "_qualifying_results.csv" in name or "_qualifying_laps.csv" in name
        ) and "sprint_qualifying" not in name
        is_race_label = "_race_results.csv" in name or "_race_laps.csv" in name
        if (target == "qualifying" and (is_gp_qualifying_label or is_race_label)) or (
            target == "race" and is_race_label
        ):
            forbidden.append(_portable_path(path))
    if forbidden:
        raise RuntimeError(
            f"{target} inference read current-event evaluation labels: {sorted(set(forbidden))}"
        )


def _assert_manifest_stable(before: dict[str, Any], after: dict[str, Any], label: str) -> None:
    if before.get("aggregate_sha256") != after.get("aggregate_sha256"):
        raise RuntimeError(f"{label} changed during the rolling evaluation")


def _load_profile(path_value: str) -> tuple[Path, dict[str, Any]]:
    path = Path(path_value).expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"F1 profile must be a mapping: {path}")
    return path, payload


def _assert_profiles_match_invocation(
    *,
    qualifying: dict[str, Any],
    race: dict[str, Any],
    source: str,
    year: int,
    rounds: Sequence[int],
    experiment_arm: str,
    train_seasons: Sequence[int],
    compare_families: Sequence[str],
    qualifying_model: str,
    race_model: str,
    qualifying_horizon: str,
    race_horizon: str,
    qualifying_disable_runsim: bool,
    race_disable_runsim: bool,
    include_standings: bool,
    run_id: Optional[str],
) -> None:
    errors: list[str] = []
    for label, profile in (("qualifying", qualifying), ("race", race)):
        training = profile.get("training", {})
        if profile.get("source") != source:
            errors.append(f"{label}.source")
        if training.get("protocol") != experiment_arm:
            errors.append(f"{label}.training.protocol")
        if [int(value) for value in training.get("seasons", [])] != [
            int(value) for value in train_seasons
        ]:
            errors.append(f"{label}.training.seasons")
        frozen_rounds = profile.get("promotion", {}).get("frozen_rounds", [])
        if [int(value) for value in frozen_rounds] != [int(value) for value in rounds]:
            errors.append(f"{label}.promotion.frozen_rounds")
        if int(profile.get("field_size", 0)) != (22 if int(year) >= 2026 else 20):
            errors.append(f"{label}.field_size")
        if bool(profile.get("features", {}).get("standings", False)) != bool(include_standings):
            errors.append(f"{label}.features.standings")
        evidence_run_id = profile.get("promotion", {}).get("evidence_run_id")
        if run_id is not None and str(evidence_run_id) != str(run_id):
            errors.append(f"{label}.promotion.evidence_run_id")
    if qualifying.get("information_horizon") != qualifying_horizon:
        errors.append("qualifying.information_horizon")
    if race.get("information_horizon") != race_horizon:
        errors.append("race.information_horizon")
    if str(qualifying.get("model", {}).get("requested")) != str(qualifying_model):
        errors.append("qualifying.model.requested")
    if str(race.get("model", {}).get("requested")) != str(race_model):
        errors.append("race.model.requested")
    if list(qualifying.get("model", {}).get("compare_families", [])) != list(compare_families):
        errors.append("qualifying.model.compare_families")
    if bool(qualifying.get("features", {}).get("run_simulation_features", False)) == bool(
        qualifying_disable_runsim
    ):
        errors.append("qualifying.features.run_simulation_features")
    if bool(race.get("features", {}).get("run_simulation_features", False)) == bool(
        race_disable_runsim
    ):
        errors.append("race.features.run_simulation_features")
    if errors:
        raise ValueError(f"Bound profiles disagree with invocation: {sorted(errors)}")


def _normalize_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def _provider(source: str, cache_dir: Optional[str], weekends_dir: Optional[str], round_number: int) -> object:
    if source == "fastf1":
        return FastF1Provider(cache_dir=cache_dir)
    if source == "openf1":
        return OpenF1Provider(cache_dir=cache_dir, target_round=round_number)
    return LocalWeekendProvider(weekends_dir=weekends_dir)


def _actual_position_lookup(actual: pd.DataFrame) -> dict[str, float]:
    if actual.empty or "position" not in actual.columns:
        return {}
    frame = actual.copy()
    frame["position"] = pd.to_numeric(frame["position"], errors="coerce")
    frame = frame.dropna(subset=["position"])
    lookup: dict[str, float] = {}
    for _, row in frame.iterrows():
        position = float(row["position"])
        for col in ["driver_id", "driver_name", "Abbreviation", "Driver"]:
            if col not in frame.columns:
                continue
            key = _normalize_key(row.get(col))
            if key and key not in lookup:
                lookup[key] = position
    return lookup


def _actual_position_for_row(row: dict[str, Any], lookup: dict[str, float]) -> Optional[float]:
    for col in ["driver_id", "driver_name"]:
        key = _normalize_key(row.get(col))
        if key in lookup:
            return lookup[key]
    return None


def _event_name(provider: object, year: int, round_number: int) -> str:
    try:
        rounds = provider.list_rounds(year)  # type: ignore[attr-defined]
    except Exception:
        return f"Round {round_number}"
    for item in rounds:
        try:
            if int(item.get("round_number", 0)) == round_number:
                return str(item.get("event_name") or item.get("meeting_name") or item.get("country_name") or f"Round {round_number}")
        except Exception:
            continue
    return f"Round {round_number}"


def _prediction_rows(result: object) -> list[dict[str, Any]]:
    extras = getattr(result, "extras", None)
    if isinstance(extras, dict) and isinstance(extras.get("all_prediction_rows"), list):
        return list(extras["all_prediction_rows"])
    table = getattr(result, "table", pd.DataFrame())
    if isinstance(table, pd.DataFrame) and not table.empty:
        return json.loads(table.to_json(orient="records"))
    return []


def _selection_rows(
    *,
    year: int,
    round_number: int,
    race_name: str,
    target: str,
    information_cutoff: str,
    result: object,
    actual: pd.DataFrame,
    base_train_seasons: list[int],
    train_seasons_used: list[int],
    training_rounds_used: list[int],
    current_year_weight_multiplier: float,
    experiment_arm: str,
) -> list[dict[str, Any]]:
    rows = _prediction_rows(result)
    rows = sorted(rows, key=lambda r: float(r.get("rank", 9999)))
    match_eval = evaluate_prediction_rows(
        rows,
        actual,
        "position",
        min_field_coverage=0.0,
        require_complete_field=False,
        include_match_rows=True,
    )
    match_rows = match_eval.get("match_rows") if isinstance(match_eval, dict) else None
    actual_position_by_pred_index: dict[int, float] = {}
    if isinstance(match_rows, list):
        for match in match_rows:
            if not isinstance(match, dict):
                continue
            pred_index = match.get("pred_index")
            actual_rank = match.get("actual_rank")
            if pred_index is None or actual_rank is None or pd.isna(actual_rank):
                continue
            try:
                actual_position_by_pred_index[int(pred_index)] = float(actual_rank)
            except (TypeError, ValueError):
                continue
    output: list[dict[str, Any]] = []
    markets = [
        ("winner", 1, "proba_win"),
        ("podium", 3, "proba_top3"),
        ("top10", 10, "proba_top10"),
    ]
    for market, cutoff_rank, probability_col in markets:
        for row_index, row in enumerate(rows[:cutoff_rank]):
            actual_position = actual_position_by_pred_index.get(int(row_index))
            hit = None
            if actual_position is not None:
                hit = actual_position <= cutoff_rank
            probability = row.get(probability_col)
            output.append(
                {
                    "season": year,
                    "round": round_number,
                    "grand_prix": race_name,
                    "target": target,
                    "information_cutoff": information_cutoff,
                    "market": market,
                    "selection": row.get("driver_name") or row.get("driver_id"),
                    "driver_id": row.get("driver_id"),
                    "model_rank": int(row.get("rank")) if pd.notna(row.get("rank")) else None,
                    "model_probability_pct": round(float(probability) * 100.0, 3) if probability is not None and pd.notna(probability) else None,
                    "prediction_score": row.get("pred"),
                    "actual_position": actual_position,
                    "result": "hit" if hit is True else "miss" if hit is False else "unmatched",
                    "model_name": getattr(result, "model_name", None),
                    "model_family": getattr(result, "model_family", None),
                    "base_train_seasons": ",".join(str(y) for y in base_train_seasons),
                    "train_seasons_used": ",".join(str(y) for y in train_seasons_used),
                    "rolling_2026_rounds_used": ",".join(str(r) for r in training_rounds_used),
                    "current_year_weight_multiplier": float(current_year_weight_multiplier),
                    "experiment_arm": experiment_arm,
                }
            )
    return output


def _summary_row(
    *,
    year: int,
    round_number: int,
    race_name: str,
    target: str,
    information_cutoff: str,
    result: object,
    actual: pd.DataFrame,
    base_train_seasons: list[int],
    train_seasons_used: list[int],
    training_rounds_used: list[int],
    current_year_weight_multiplier: float,
    experiment_arm: str,
) -> dict[str, Any]:
    rows = _prediction_rows(result)
    evaluation = evaluate_prediction_rows(
        predicted_rows=rows,
        actual_results=actual,
        actual_position_col="position",
        include_podium_and_winner=True,
    )
    top_driver = rows[0] if rows else {}
    leaderboard = getattr(result, "candidate_leaderboard", []) or []
    top_candidate = leaderboard[0] if leaderboard else {}
    return {
        "season": year,
        "round": round_number,
        "grand_prix": race_name,
        "target": target,
        "information_cutoff": information_cutoff,
        "rows_predicted": evaluation.get("rows_predicted"),
        "rows_actual": evaluation.get("rows_actual"),
        "rows_common": evaluation.get("rows_common"),
        "field_coverage": evaluation.get("field_coverage"),
        "complete_field": evaluation.get("complete_field"),
        "missing_actual_count": evaluation.get("missing_actual_count"),
        "unexpected_prediction_count": evaluation.get("unexpected_prediction_count"),
        "evaluation_reason": evaluation.get("evaluation_reason"),
        "mae_valid": evaluation.get("mae_valid"),
        "field_mae": evaluation.get("field_mae"),
        "mae_on_common": evaluation.get("mae_on_common"),
        "field_mae_penalized": evaluation.get("field_mae_penalized"),
        "exact_position_accuracy": evaluation.get("exact_position_accuracy"),
        "kendall_tau_b": evaluation.get("kendall_tau_b"),
        "top3_hit": evaluation.get("top3_hit"),
        "top5_hit": evaluation.get("top5_hit"),
        "top10_hit_pct": round(float(evaluation["top10_hit"]) * 100.0, 2) if evaluation.get("top10_hit") is not None else None,
        "podium_hit_count": evaluation.get("podium_hit_count"),
        "winner_hit": evaluation.get("winner_hit"),
        "win_log_loss": evaluation.get("win_log_loss"),
        "win_brier": evaluation.get("win_brier"),
        "win_ece": evaluation.get("win_ece"),
        "top3_log_loss": evaluation.get("top3_log_loss"),
        "top3_brier": evaluation.get("top3_brier"),
        "top3_ece": evaluation.get("top3_ece"),
        "top10_log_loss": evaluation.get("top10_log_loss"),
        "top10_brier": evaluation.get("top10_brier"),
        "top10_ece": evaluation.get("top10_ece"),
        "position_interval_coverage": evaluation.get("position_interval_coverage"),
        "position_interval_mean_width": evaluation.get("position_interval_mean_width"),
        "top_model_selection": top_driver.get("driver_name") or top_driver.get("driver_id"),
        "top_model_probability_pct": round(float(top_driver.get("proba_win")) * 100.0, 3) if top_driver.get("proba_win") is not None and pd.notna(top_driver.get("proba_win")) else None,
        "model_name": getattr(result, "model_name", None),
        "model_family": getattr(result, "model_family", None),
        "cv_top_candidate": top_candidate.get("name"),
        "cv_top_composite": top_candidate.get("composite"),
        "base_train_seasons": ",".join(str(y) for y in base_train_seasons),
        "train_seasons_used": ",".join(str(y) for y in train_seasons_used),
        "rolling_2026_rounds_used": ",".join(str(r) for r in training_rounds_used),
        "current_year_weight_multiplier": float(current_year_weight_multiplier),
        "experiment_arm": experiment_arm,
    }


def _point_in_time_record(
    *,
    target: str,
    config: PredictionConfig,
    result: object,
) -> dict[str, Any]:
    extras = getattr(result, "extras", None)
    extras = extras if isinstance(extras, dict) else {}
    qualifying_horizon = extras.get("qualifying_information_horizon")
    if not isinstance(qualifying_horizon, dict):
        qualifying_horizon = {
            "requested_cutoff": str(config.qualifying_information_horizon),
            "resolved_cutoffs": [],
            "prediction_as_of": config.prediction_as_of,
        }
    record: dict[str, Any] = {
        "target": target,
        "target_round": int(config.round_number),
        "target_season": int(config.year),
        "training_event_rule": "strictly_prior_rounds_only",
        "target_round_allowed_in_training": False,
        "qualifying_session_cutoff": qualifying_horizon,
        "prediction_as_of": config.prediction_as_of,
        "as_of_semantics": "completed_session_boundary_when_timestamp_snapshot_is_unavailable",
        "actual_training_event_keys": extras.get("training_event_keys", []),
        "actual_training_row_count": extras.get("training_row_count", 0),
        "target_event_key_excluded_from_training": extras.get(
            "target_event_key_excluded_from_training",
            False,
        ),
    }
    if target == "race":
        record["information_cutoff"] = str(config.race_information_horizon)
        record["resolved_race_information_horizon"] = extras.get("race_information_horizon")
        record["prediction_phase"] = extras.get("prediction_phase")
    else:
        record["information_cutoff"] = "post_practice_pre_qualifying"
    return record


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_prediction_artifact(
    path: Path,
    config: PredictionConfig,
    result: object,
    *,
    point_in_time: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    table = getattr(result, "table", pd.DataFrame())
    config_payload = asdict(config)
    payload = {
        "schema_version": ROLLING_BACKTEST_SCHEMA_VERSION,
        "run_id": run_id,
        "sport": "F1",
        "project": "Rising Qualification Prediction",
        "config": config_payload,
        "config_sha256": _sha256_json(config_payload),
        "point_in_time": point_in_time,
        "rows": json.loads(table.to_json(orient="records")) if isinstance(table, pd.DataFrame) and not table.empty else [],
        "all_prediction_rows": _prediction_rows(result),
        "notes": getattr(result, "notes", []),
        "model_name": getattr(result, "model_name", None),
        "model_family": getattr(result, "model_family", None),
        "candidate_leaderboard": getattr(result, "candidate_leaderboard", []),
        "generated_at": _utc_now(),
    }
    _write_json_exclusive(path, payload)
    return {
        "artifact_type": f"{config.mode}_prediction",
        "path": _portable_path(path),
        "sha256": _sha256_file(path),
        "byte_size": int(path.stat().st_size),
        "config_sha256": payload["config_sha256"],
        "point_in_time": point_in_time,
    }


def _write_csv_exclusive(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
    return {
        "artifact_type": "table",
        "path": _portable_path(path),
        "sha256": _sha256_file(path),
        "byte_size": int(path.stat().st_size),
        "rows": int(len(frame)),
    }


def _new_run_id(*, year: int, experiment_arm: str, git_state: dict[str, Any], config: dict[str, Any]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    git_short = str(git_state.get("head_sha") or "nogit")[:10]
    config_short = _sha256_json(config)[:10]
    arm = re.sub(r"[^a-z0-9]+", "-", experiment_arm.lower()).strip("-")
    return f"{year}_{arm}_{timestamp}_{git_short}_{config_short}"


def _validated_run_id(value: str) -> str:
    normalized = str(value).strip()
    if normalized in {"", ".", ".."} or not re.fullmatch(r"[A-Za-z0-9._-]+", normalized):
        raise ValueError("--run-id may contain only letters, numbers, dot, underscore, and hyphen")
    return normalized


def _assert_complete_horizon_evidence(point_in_time: dict[str, Any]) -> None:
    qualifying = point_in_time.get("qualifying_session_cutoff")
    if not isinstance(qualifying, dict) or not qualifying.get("requested_cutoff"):
        raise RuntimeError("Prediction did not emit a requested qualifying-session cutoff")
    resolved = qualifying.get("resolved_cutoffs")
    if not isinstance(resolved, list) or not resolved:
        raise RuntimeError("Prediction did not emit a resolved qualifying-session cutoff")
    if point_in_time.get("target") == "race" and not point_in_time.get("resolved_race_information_horizon"):
        raise RuntimeError("Race prediction did not emit a resolved race information horizon")
    if not point_in_time.get("target_event_key_excluded_from_training"):
        raise RuntimeError("Prediction did not prove target-event exclusion from actual training rows")
    target_key = (int(point_in_time["target_season"]) * 100) + int(
        point_in_time["target_round"]
    )
    training_keys = [int(value) for value in point_in_time.get("actual_training_event_keys", [])]
    if any(value >= target_key and value // 100 == target_key // 100 for value in training_keys):
        raise RuntimeError("Actual training-event keys contain the target or a future same-season event")
    if (
        point_in_time.get("target") == "race"
        and point_in_time.get("resolved_race_information_horizon") == "post_grid_pre_race"
    ):
        phase = point_in_time.get("prediction_phase")
        if (
            not isinstance(phase, dict)
            or phase.get("phase") != "post_grid_pre_race"
            or not phase.get("official_grid_available")
            or int(phase.get("official_grid_rows", 0)) <= 0
        ):
            raise RuntimeError(
                "post_grid_pre_race requires resolver-approved complete official-grid evidence"
            )
    point_in_time["horizon_evidence_complete"] = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Point-in-time 2026 walk-forward backtest. The default primary arm uses only prior "
            "2026 rounds; historical transfer requires an explicit separate experiment arm."
        ),
    )
    parser.add_argument("--source", choices=["local", "fastf1", "openf1"], default="local")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument(
        "--rounds",
        default="auto",
        help="Comma-separated completed rounds, or auto to discover all completed local rounds through the latest snapshot.",
    )
    parser.add_argument(
        "--experiment-arm",
        choices=["same_season_walk_forward", "explicit_transfer"],
        default="same_season_walk_forward",
    )
    parser.add_argument(
        "--transfer-train-seasons",
        default=None,
        help="Explicit historical seasons for the non-primary transfer arm only.",
    )
    parser.add_argument(
        "--base-train-seasons",
        default=None,
        help="Deprecated alias for --transfer-train-seasons; valid only with --experiment-arm explicit_transfer.",
    )
    parser.add_argument("--current-season-weight-multiplier", type=float, default=None)
    parser.add_argument("--include-standings", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--weekends-dir", default=default_weekends_dir())
    parser.add_argument("--compare-families", default="ml")
    parser.add_argument("--enable-dl-candidates", action="store_true")
    parser.add_argument("--dl-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dl-arch", default="mlp_tabular_v1")
    parser.add_argument("--dl-hyperparams", default="{}")
    parser.add_argument("--dl-seed", type=int, default=42)
    parser.add_argument("--disable-runsim-features", action="store_true")
    parser.add_argument(
        "--qualifying-runsim-features",
        choices=["inherit", "enabled", "disabled"],
        default="inherit",
        help="Qualifying-only run-simulation policy; inherit preserves the legacy global flag.",
    )
    parser.add_argument(
        "--race-runsim-features",
        choices=["inherit", "enabled", "disabled"],
        default="inherit",
        help="Race-only run-simulation policy; inherit preserves the legacy global flag.",
    )
    parser.add_argument("--f1-model", default="auto")
    parser.add_argument(
        "--qualifying-model",
        default=None,
        help="Qualifying-only override. Defaults to --f1-model for backwards compatibility.",
    )
    parser.add_argument(
        "--race-model",
        default=None,
        help="Race-only override. Defaults to --f1-model for backwards compatibility.",
    )
    parser.add_argument("--qualifying-information-horizon", default="pre_qualifying")
    parser.add_argument(
        "--race-information-horizon",
        choices=["post_qualifying_pre_grid", "post_grid_pre_race"],
        default="post_qualifying_pre_grid",
    )
    parser.add_argument("--max-threads", type=int, default=1)
    parser.add_argument("--output-dir", default=default_output_dir())
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--qualifying-profile", default=default_qualifying_profile())
    parser.add_argument("--race-profile", default=default_race_profile())
    parser.add_argument(
        "--selection-csv",
        default=None,
        help="Optional immutable export path. Default: the unique run directory.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional immutable export path. Default: the unique run directory.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def _resolve_runsim_disable(*, legacy_disable: bool, target_policy: str) -> bool:
    if target_policy == "inherit":
        return bool(legacy_disable)
    return target_policy == "disabled"


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    _apply_resource_limits(args.max_threads)
    rounds = _resolve_rounds(
        args.rounds,
        source=args.source,
        weekends_dir=args.weekends_dir,
        year=args.year,
    )
    training_protocol = _resolve_training_protocol(
        year=args.year,
        experiment_arm=args.experiment_arm,
        transfer_train_seasons=args.transfer_train_seasons,
        legacy_base_train_seasons=args.base_train_seasons,
        current_season_weight_multiplier=args.current_season_weight_multiplier,
    )
    base_train_seasons = list(training_protocol["transfer_train_seasons"])
    train_seasons_used = list(training_protocol["train_seasons_used"])
    current_season_weight_multiplier = float(training_protocol["current_season_weight_multiplier"])
    compare_families = parse_compare_families(args.compare_families)
    dl_hyperparams = parse_json_object(args.dl_hyperparams, "--dl-hyperparams")
    qualifying_model = str(args.qualifying_model or args.f1_model)
    race_model = str(args.race_model or args.f1_model)
    qualifying_disable_runsim = _resolve_runsim_disable(
        legacy_disable=bool(args.disable_runsim_features),
        target_policy=str(args.qualifying_runsim_features),
    )
    race_disable_runsim = _resolve_runsim_disable(
        legacy_disable=bool(args.disable_runsim_features),
        target_policy=str(args.race_runsim_features),
    )
    qualifying_profile_path, qualifying_profile = _load_profile(args.qualifying_profile)
    race_profile_path, race_profile = _load_profile(args.race_profile)
    _assert_profiles_match_invocation(
        qualifying=qualifying_profile,
        race=race_profile,
        source=str(args.source),
        year=int(args.year),
        rounds=rounds,
        experiment_arm=str(args.experiment_arm),
        train_seasons=train_seasons_used,
        compare_families=compare_families,
        qualifying_model=qualifying_model,
        race_model=race_model,
        qualifying_horizon=str(args.qualifying_information_horizon),
        race_horizon=str(args.race_information_horizon),
        qualifying_disable_runsim=qualifying_disable_runsim,
        race_disable_runsim=race_disable_runsim,
        include_standings=bool(args.include_standings),
        run_id=args.run_id,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_configuration = {
        "source": args.source,
        "year": int(args.year),
        "rounds": rounds,
        "training_protocol": training_protocol,
        "include_standings": bool(args.include_standings),
        "weekends_dir": _portable_path(Path(args.weekends_dir)),
        "compare_families": compare_families,
        "enable_dl_candidates": bool(args.enable_dl_candidates),
        "dl_device": args.dl_device,
        "dl_arch": args.dl_arch,
        "dl_hyperparams": dl_hyperparams,
        "dl_seed": int(args.dl_seed),
        "disable_runsim_features": bool(args.disable_runsim_features),
        "target_disable_runsim_features": {
            "qualifying": qualifying_disable_runsim,
            "race": race_disable_runsim,
        },
        "f1_model": args.f1_model,
        "qualifying_model": qualifying_model,
        "race_model": race_model,
        "qualifying_information_horizon": args.qualifying_information_horizon,
        "race_information_horizon": args.race_information_horizon,
        "max_threads": int(args.max_threads),
        "round_execution": "ascending_sequential",
        "bound_profiles": {
            "qualifying": {
                "path": _portable_path(qualifying_profile_path),
                "sha256": _sha256_file(qualifying_profile_path),
            },
            "race": {
                "path": _portable_path(race_profile_path),
                "sha256": _sha256_file(race_profile_path),
            },
        },
    }
    git_state = _git_state()
    run_id = _validated_run_id(
        args.run_id
        or _new_run_id(
            year=int(args.year),
            experiment_arm=str(args.experiment_arm),
            git_state=git_state,
            config=run_configuration,
        )
    )
    run_root = output_dir / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    selection_path = Path(args.selection_csv) if args.selection_csv else run_root / "rolling_backtest_selections.csv"
    summary_path = Path(args.summary_csv) if args.summary_csv else run_root / "rolling_backtest_summary.csv"
    if selection_path.expanduser().resolve() == summary_path.expanduser().resolve():
        raise ValueError("--selection-csv and --summary-csv must be different immutable paths")
    for export_path in (selection_path, summary_path):
        if export_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {export_path}")

    selection_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    prediction_artifacts: list[dict[str, Any]] = []
    point_in_time_by_round: list[dict[str, Any]] = []
    data_by_round: dict[str, Any] = {}
    actual_csv_access_by_round: dict[str, Any] = {}
    available_local_rounds = _available_local_rounds(args.weekends_dir, args.year)
    implementation_before = _implementation_manifest()
    configuration_before = _configuration_file_manifest()
    data_stability_paths = _data_paths_for_round(
        weekends_dir=args.weekends_dir,
        target_year=args.year,
        target_round=max(rounds),
        transfer_train_seasons=base_train_seasons,
    )
    data_stability_before = _file_manifest(
        data_stability_paths,
        manifest_version="f1_rolling_input_stability_v1",
    )
    started_at = _utc_now()

    for round_number in rounds:
        provider = _provider(args.source, args.cache_dir, args.weekends_dir, round_number)
        race_name = _event_name(provider, args.year, round_number)
        training_rounds_used = [value for value in available_local_rounds if value < int(round_number)]

        common_config = {
            "source": args.source,
            "year": args.year,
            "round_number": round_number,
            "train_seasons": train_seasons_used,
            "include_standings": bool(args.include_standings),
            "cache_dir": args.cache_dir,
            "meeting_name": None,
            "country_name": None,
            "weekends_dir": args.weekends_dir,
            "enable_dl_candidates": bool(args.enable_dl_candidates),
            "compare_families": compare_families,
            "dl_device": args.dl_device,
            "dl_arch": args.dl_arch,
            "dl_hyperparams": dl_hyperparams,
            "dl_seed": int(args.dl_seed),
            "season_weight_year": int(args.year),
            "season_weight_multiplier": current_season_weight_multiplier,
            "qualifying_information_horizon": args.qualifying_information_horizon,
            "prediction_as_of": None,
        }

        qualifying_config = PredictionConfig(
            mode="qualifying",
            f1_model=qualifying_model,
            disable_runsim_features=qualifying_disable_runsim,
            **common_config,
        )
        with _capture_csv_reads() as qualifying_inference_reads:
            qualifying_result = run_prediction(qualifying_config)
        _assert_no_current_target_label_access(
            qualifying_inference_reads,
            target_year=args.year,
            target_round=round_number,
            target="qualifying",
        )
        qualifying_point_in_time = _point_in_time_record(
            target="qualifying",
            config=qualifying_config,
            result=qualifying_result,
        )
        _assert_complete_horizon_evidence(qualifying_point_in_time)
        prediction_artifacts.append(
            _write_prediction_artifact(
                run_root / str(args.year) / f"round_{round_number:02d}" / "qualifying_prediction.json",
                qualifying_config,
                qualifying_result,
                point_in_time=qualifying_point_in_time,
                run_id=run_id,
            )
        )
        point_in_time_by_round.append(qualifying_point_in_time)
        with _capture_csv_reads() as qualifying_evaluation_reads:
            actual_qualifying = provider.get_qualifying_results(args.year, round_number)  # type: ignore[attr-defined]
        if actual_qualifying.empty:
            raise RuntimeError(f"Round {round_number} has no qualifying outcome for evaluation")
        selection_rows.extend(
            _selection_rows(
                year=args.year,
                round_number=round_number,
                race_name=race_name,
                target="qualifying",
                information_cutoff="post_practice_pre_qualifying",
                result=qualifying_result,
                actual=actual_qualifying,
                base_train_seasons=base_train_seasons,
                train_seasons_used=train_seasons_used,
                training_rounds_used=training_rounds_used,
                current_year_weight_multiplier=current_season_weight_multiplier,
                experiment_arm=args.experiment_arm,
            )
        )
        summary_rows.append(
            _summary_row(
                year=args.year,
                round_number=round_number,
                race_name=race_name,
                target="qualifying",
                information_cutoff="post_practice_pre_qualifying",
                result=qualifying_result,
                actual=actual_qualifying,
                base_train_seasons=base_train_seasons,
                train_seasons_used=train_seasons_used,
                training_rounds_used=training_rounds_used,
                current_year_weight_multiplier=current_season_weight_multiplier,
                experiment_arm=args.experiment_arm,
            )
        )

        race_config = PredictionConfig(
            mode="race",
            f1_model=race_model,
            disable_runsim_features=race_disable_runsim,
            race_information_horizon=args.race_information_horizon,
            **common_config,
        )
        with _capture_csv_reads() as race_inference_reads:
            race_result = run_prediction(race_config)
        _assert_no_current_target_label_access(
            race_inference_reads,
            target_year=args.year,
            target_round=round_number,
            target="race",
        )
        race_point_in_time = _point_in_time_record(
            target="race",
            config=race_config,
            result=race_result,
        )
        _assert_complete_horizon_evidence(race_point_in_time)
        prediction_artifacts.append(
            _write_prediction_artifact(
                run_root / str(args.year) / f"round_{round_number:02d}" / "postqual_race_prediction.json",
                race_config,
                race_result,
                point_in_time=race_point_in_time,
                run_id=run_id,
            )
        )
        point_in_time_by_round.append(race_point_in_time)
        with _capture_csv_reads() as race_evaluation_reads:
            actual_race = provider.get_race_results(args.year, round_number)  # type: ignore[attr-defined]
        if actual_race.empty:
            raise RuntimeError(f"Round {round_number} has no race outcome for evaluation")
        selection_rows.extend(
            _selection_rows(
                year=args.year,
                round_number=round_number,
                race_name=race_name,
                target="race",
                information_cutoff=args.race_information_horizon,
                result=race_result,
                actual=actual_race,
                base_train_seasons=base_train_seasons,
                train_seasons_used=train_seasons_used,
                training_rounds_used=training_rounds_used,
                current_year_weight_multiplier=current_season_weight_multiplier,
                experiment_arm=args.experiment_arm,
            )
        )
        summary_rows.append(
            _summary_row(
                year=args.year,
                round_number=round_number,
                race_name=race_name,
                target="race",
                information_cutoff=args.race_information_horizon,
                result=race_result,
                actual=actual_race,
                base_train_seasons=base_train_seasons,
                train_seasons_used=train_seasons_used,
                training_rounds_used=training_rounds_used,
                current_year_weight_multiplier=current_season_weight_multiplier,
                experiment_arm=args.experiment_arm,
            )
        )

        data_manifest = _file_manifest(
            _data_paths_for_round(
                weekends_dir=args.weekends_dir,
                target_year=args.year,
                target_round=round_number,
                transfer_train_seasons=base_train_seasons,
            ),
            manifest_version="f1_point_in_time_round_data_v1",
        )
        data_manifest.update(
            {
                "target_season": int(args.year),
                "target_round": int(round_number),
                "target_season_future_rounds_included": False,
                "target_round_outcomes_role": "evaluation_only",
                "prior_target_rounds_role": "training",
                "transfer_seasons_role": "explicit_transfer_arm_only" if base_train_seasons else "not_used",
            }
        )
        data_by_round[str(round_number)] = data_manifest
        actual_csv_access_by_round[str(round_number)] = {
            "qualifying_inference": _file_manifest(
                qualifying_inference_reads,
                manifest_version="f1_actual_csv_access_v1",
            ),
            "qualifying_evaluation_labels": _file_manifest(
                qualifying_evaluation_reads,
                manifest_version="f1_actual_csv_access_v1",
            ),
            "race_inference": _file_manifest(
                race_inference_reads,
                manifest_version="f1_actual_csv_access_v1",
            ),
            "race_evaluation_labels": _file_manifest(
                race_evaluation_reads,
                manifest_version="f1_actual_csv_access_v1",
            ),
            "current_target_label_access_assertion": "passed",
        }

    _assert_manifest_stable(
        implementation_before,
        _implementation_manifest(),
        "implementation source tree",
    )
    _assert_manifest_stable(
        configuration_before,
        _configuration_file_manifest(),
        "F1 configuration files",
    )
    data_stability_after = _file_manifest(
        data_stability_paths,
        manifest_version="f1_rolling_input_stability_v1",
    )
    _assert_manifest_stable(
        data_stability_before,
        data_stability_after,
        "input data",
    )

    selection_artifact = _write_csv_exclusive(selection_path, pd.DataFrame(selection_rows))
    selection_artifact["artifact_type"] = "rolling_backtest_selections"
    summary_artifact = _write_csv_exclusive(summary_path, pd.DataFrame(summary_rows))
    summary_artifact["artifact_type"] = "rolling_backtest_summary"
    artifact_manifest = [*prediction_artifacts, selection_artifact, summary_artifact]

    manifest = {
        "schema_version": ROLLING_BACKTEST_SCHEMA_VERSION,
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "git": git_state,
        "implementation": implementation_before,
        "configuration_files": configuration_before,
        "run_configuration": {
            "payload": run_configuration,
            "sha256": _sha256_json(run_configuration),
        },
        "runtime": _runtime_manifest(args.max_threads),
        "training_protocol": {
            **training_protocol,
            "target_round_allowed_in_training": False,
            "target_season_training_window": "round_number_strictly_less_than_target_round",
            "execution_order": "round_ascending_then_qualifying_then_race",
            "parallel_rounds": False,
            "parallel_targets": False,
        },
        "point_in_time_by_round": point_in_time_by_round,
        "input_data_by_round": data_by_round,
        "actual_csv_access_by_round": actual_csv_access_by_round,
        "input_data_stability_snapshot": data_stability_before,
        "artifacts": artifact_manifest,
        "artifact_contract": {
            "immutable_paths": True,
            "write_mode": "exclusive_create",
            "run_root": _portable_path(run_root),
        },
    }
    manifest_path = run_root / "run_manifest.json"
    _write_json_exclusive(manifest_path, manifest)

    if not args.quiet:
        print(json.dumps(
            {
                "sport": "F1",
                "year": args.year,
                "rounds": rounds,
                "run_id": run_id,
                "experiment_arm": args.experiment_arm,
                "base_train_seasons": base_train_seasons,
                "train_seasons_used": train_seasons_used,
                "current_year_weight_multiplier": current_season_weight_multiplier,
                "selection_csv": str(selection_path),
                "summary_csv": str(summary_path),
                "artifact_dir": str(run_root),
                "manifest": str(manifest_path),
                "selection_rows": len(selection_rows),
                "summary_rows": len(summary_rows),
                "generated_at": _utc_now(),
            },
            ensure_ascii=False,
            indent=2,
        ))


if __name__ == "__main__":
    main()
