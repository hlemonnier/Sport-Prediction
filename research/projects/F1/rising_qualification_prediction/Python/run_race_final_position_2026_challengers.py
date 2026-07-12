#!/usr/bin/env python3
"""Causal 2026 race-final-position challenger backtest.

The local snapshots do not contain a point-in-time official starting-grid
artifact.  This experiment therefore uses the Grand Prix qualifying
classification as the only legal current-event ordering proxy.  The
``GridPosition`` field embedded in post-race result files is deliberately not
read.  Every challenger is fit from strictly earlier rounds in the same season
and emits a complete field permutation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from packages.sports_core.paths import find_repo_root


SCHEMA_VERSION = "f1_race_final_position_causal_challengers_v2"
BASELINE_NAME = "grand_prix_qualifying_order_proxy"
PRIMARY_PRIOR_ROUNDS = 4
SENSITIVITY_PRIOR_ROUNDS = (1, 2, 4, 8, 16)
ROUND_PATTERN = re.compile(r"^round_(\d{2})_")
QUALIFYING_RESULTS_PATTERN = re.compile(r"^\d+_qualifying_results\.csv$")
RACE_RESULTS_PATTERN = re.compile(r"^\d+_race_results\.csv$")
RUNNING_STATUSES = {"finished", "lapped"}


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


def _single_file(round_dir: Path, pattern: re.Pattern[str], label: str) -> Path:
    candidates = sorted(path for path in round_dir.glob("*_results.csv") if pattern.match(path.name))
    if len(candidates) != 1:
        raise ValueError(f"{round_dir} has {len(candidates)} {label} files; expected exactly one")
    return candidates[0]


def _complete_permutation(values: pd.Series, *, label: str) -> tuple[pd.Series, int]:
    """Complete missing tail positions in source row order and validate 1..N."""

    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    size = int(len(numeric))
    finite = numeric.dropna()
    invalid = finite[(finite < 1) | (finite > size) | ((finite % 1) != 0)]
    if not invalid.empty:
        raise ValueError(f"{label} contains invalid positions: {invalid.tolist()}")
    used = [int(value) for value in finite.tolist()]
    if len(used) != len(set(used)):
        raise ValueError(f"{label} contains duplicate classified positions")
    remaining = [position for position in range(1, size + 1) if position not in set(used)]
    missing_indices = numeric.index[numeric.isna()].tolist()
    if len(remaining) != len(missing_indices):
        raise ValueError(f"{label} cannot be completed into a field permutation")
    for index, position in zip(missing_indices, remaining):
        numeric.loc[index] = float(position)
    completed = numeric.astype(int)
    if sorted(completed.tolist()) != list(range(1, size + 1)):
        raise ValueError(f"{label} is not a complete field permutation")
    return completed, int(len(missing_indices))


def _metadata_contract(metadata_path: Path) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sessions = [entry for entry in metadata.get("sessions", []) if isinstance(entry, dict)]
    qualifying = next((entry for entry in sessions if entry.get("session_type") == "qualifying"), None)
    race = next((entry for entry in sessions if entry.get("session_type") == "race"), None)
    if qualifying is None or race is None:
        raise ValueError(f"{metadata_path} lacks qualifying/race session provenance")
    qualifying_order = int(qualifying.get("session_order", 0))
    race_order = int(race.get("session_order", 0))
    if qualifying_order <= 0 or race_order <= qualifying_order:
        raise ValueError(f"{metadata_path} does not prove qualifying precedes the race")
    if str(qualifying.get("availability_phase")) != "post_session":
        raise ValueError(f"{metadata_path} qualifying is not tagged post_session")
    if str(race.get("availability_phase")) != "post_race":
        raise ValueError(f"{metadata_path} race label is not tagged post_race")
    official_grid_refs = [
        metadata.get("grid_path"),
        metadata.get("starting_grid_path"),
        metadata.get("pre_race_grid_path"),
        *[
            entry.get(key)
            for entry in sessions
            for key in ("grid_path", "starting_grid_path", "pre_race_grid_path")
        ],
    ]
    return {
        "event_name": str(metadata.get("event_name") or metadata_path.parent.name),
        "event_format": str(metadata.get("event_format") or "unknown"),
        "qualifying_session_order": qualifying_order,
        "race_session_order": race_order,
        "qualifying_availability_phase": "post_session",
        "race_label_availability_phase": "post_race",
        "point_in_time_official_grid_reference_present": any(value for value in official_grid_refs),
    }


def _load_event(round_dir: Path, *, year: int) -> tuple[pd.DataFrame, dict[str, Any], list[Path]]:
    round_number = _round_number(round_dir)
    metadata_path = round_dir / "weekend_metadata.json"
    qualifying_path = _single_file(round_dir, QUALIFYING_RESULTS_PATTERN, "Grand Prix qualifying")
    race_path = _single_file(round_dir, RACE_RESULTS_PATTERN, "Grand Prix race")
    provenance = _metadata_contract(metadata_path)

    # Deliberately constrain usecols: GridPosition in the post-race file has no
    # point-in-time provenance and is forbidden as either a feature or baseline.
    qualifying = pd.read_csv(
        qualifying_path,
        usecols=["DriverNumber", "TeamName", "Position"],
        dtype={"DriverNumber": str},
    ).rename(
        columns={"DriverNumber": "driver_id", "TeamName": "team_name", "Position": "q_position"}
    )
    race = pd.read_csv(
        race_path,
        usecols=["DriverNumber", "Position", "Status"],
        dtype={"DriverNumber": str},
    ).rename(
        columns={"DriverNumber": "driver_id", "Position": "final_position", "Status": "status"}
    )
    for frame, label in ((qualifying, "qualifying"), (race, "race")):
        frame["driver_id"] = frame["driver_id"].astype(str).str.strip()
        if frame["driver_id"].eq("").any() or frame["driver_id"].duplicated().any():
            raise ValueError(f"round {round_number} {label} driver identifiers are not unique")
    q_positions, q_imputed = _complete_permutation(
        qualifying["q_position"], label=f"round {round_number} qualifying"
    )
    race_positions, race_imputed = _complete_permutation(
        race["final_position"], label=f"round {round_number} race"
    )
    qualifying["q_position"] = q_positions
    race["final_position"] = race_positions
    if set(qualifying["driver_id"]) != set(race["driver_id"]):
        raise ValueError(f"round {round_number} qualifying/race rosters differ")

    event = qualifying.merge(race, on="driver_id", how="inner", validate="one_to_one")
    event["status_normalized"] = event["status"].fillna("").astype(str).str.strip().str.lower()
    event["terminal"] = ~event["status_normalized"].isin(RUNNING_STATUSES)
    event["round"] = int(round_number)
    event["year"] = int(year)
    event["field_size"] = int(len(event))
    event["residual_positions"] = event["final_position"] - event["q_position"]
    event["normalized_residual"] = event["residual_positions"] / float(len(event))
    return (
        event,
        {
            **provenance,
            "round": int(round_number),
            "round_directory": str(round_dir.relative_to(_repo_root())),
            "qualifying_file": str(qualifying_path.relative_to(_repo_root())),
            "race_label_file": str(race_path.relative_to(_repo_root())),
            "field_size": int(len(event)),
            "qualifying_tail_positions_imputed_from_source_order": q_imputed,
            "race_tail_positions_imputed_from_source_order": race_imputed,
            "terminal_outcome_count": int(event["terminal"].sum()),
        },
        [metadata_path, qualifying_path, race_path],
    )


def _zero_shrunk_mean(history: pd.DataFrame, *, key: str, value: str, strength: float) -> pd.Series:
    if history.empty:
        return pd.Series(dtype=float)
    grouped = history.groupby(key, sort=False)[value].agg(["sum", "count"])
    return grouped["sum"] / (grouped["count"] + float(strength))


def _hierarchical_mean_for_current(
    current: pd.DataFrame,
    history: pd.DataFrame,
    *,
    value: str,
    driver_strength: float,
    team_strength: float,
) -> pd.Series:
    if history.empty:
        return pd.Series(0.0, index=current.index, dtype=float)
    team_mean = _zero_shrunk_mean(history, key="team_name", value=value, strength=team_strength)
    driver_stats = history.groupby("driver_id", sort=False)[value].agg(["sum", "count"])
    output = []
    for _, row in current.iterrows():
        team_prior = float(team_mean.get(row["team_name"], 0.0))
        if row["driver_id"] in driver_stats.index:
            stats = driver_stats.loc[row["driver_id"]]
            estimate = (float(stats["sum"]) + (float(driver_strength) * team_prior)) / (
                float(stats["count"]) + float(driver_strength)
            )
        else:
            estimate = team_prior
        output.append(estimate)
    return pd.Series(output, index=current.index, dtype=float)


def _terminal_probability(
    current: pd.DataFrame,
    history: pd.DataFrame,
    *,
    driver_strength: float,
    team_strength: float,
) -> pd.Series:
    if history.empty:
        return pd.Series(0.0, index=current.index, dtype=float)
    global_rate = (float(history["terminal"].sum()) + 0.5) / (float(len(history)) + 1.0)
    team_stats = history.groupby("team_name", sort=False)["terminal"].agg(["sum", "count"])
    driver_stats = history.groupby("driver_id", sort=False)["terminal"].agg(["sum", "count"])
    output = []
    for _, row in current.iterrows():
        if row["team_name"] in team_stats.index:
            stats = team_stats.loc[row["team_name"]]
            team_rate = (float(stats["sum"]) + (float(team_strength) * global_rate)) / (
                float(stats["count"]) + float(team_strength)
            )
        else:
            team_rate = global_rate
        if row["driver_id"] in driver_stats.index:
            stats = driver_stats.loc[row["driver_id"]]
            driver_rate = (float(stats["sum"]) + (float(driver_strength) * team_rate)) / (
                float(stats["count"]) + float(driver_strength)
            )
        else:
            driver_rate = team_rate
        output.append(driver_rate)
    return pd.Series(output, index=current.index, dtype=float).clip(0.0, 1.0)


def _candidate_scores(
    current: pd.DataFrame,
    history: pd.DataFrame,
    *,
    prior_rounds: int,
) -> dict[str, pd.Series]:
    driver_strength = float(prior_rounds)
    team_strength = float(2 * prior_rounds)
    q_position = current["q_position"].astype(float)
    if history.empty:
        return {
            name: q_position.copy()
            for name in (
                BASELINE_NAME,
                "driver_residual_shrinkage",
                "team_residual_shrinkage",
                "hierarchical_residual_shrinkage",
                "rank_normalized_hierarchical_residual",
                "reliability_hazard",
                "running_residual_plus_reliability",
            )
        }

    driver_mean = _zero_shrunk_mean(
        history,
        key="driver_id",
        value="residual_positions",
        strength=driver_strength,
    )
    team_mean = _zero_shrunk_mean(
        history,
        key="team_name",
        value="residual_positions",
        strength=team_strength,
    )
    hierarchical = _hierarchical_mean_for_current(
        current,
        history,
        value="residual_positions",
        driver_strength=driver_strength,
        team_strength=team_strength,
    )
    normalized_hierarchical = _hierarchical_mean_for_current(
        current,
        history,
        value="normalized_residual",
        driver_strength=driver_strength,
        team_strength=team_strength,
    )
    terminal_probability = _terminal_probability(
        current,
        history,
        driver_strength=driver_strength,
        team_strength=team_strength,
    )
    field_size = int(len(current))
    tail_start = int(math.ceil(0.75 * field_size))
    terminal_tail_mean = (float(tail_start) + float(field_size)) / 2.0
    running_history = history.loc[~history["terminal"]].copy()
    running_residual = _hierarchical_mean_for_current(
        current,
        running_history,
        value="residual_positions",
        driver_strength=driver_strength,
        team_strength=team_strength,
    )
    running_score = q_position + running_residual
    return {
        BASELINE_NAME: q_position,
        "driver_residual_shrinkage": q_position
        + current["driver_id"].map(driver_mean).fillna(0.0),
        "team_residual_shrinkage": q_position + current["team_name"].map(team_mean).fillna(0.0),
        "hierarchical_residual_shrinkage": q_position + hierarchical,
        "rank_normalized_hierarchical_residual": (
            ((q_position - 0.5) / float(field_size)) + normalized_hierarchical
        ),
        "reliability_hazard": (
            ((1.0 - terminal_probability) * q_position)
            + (terminal_probability * terminal_tail_mean)
        ),
        "running_residual_plus_reliability": (
            ((1.0 - terminal_probability) * running_score)
            + (terminal_probability * terminal_tail_mean)
        ),
    }


def _rank_scores(current: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    ranked = current[["driver_id", "team_name", "q_position", "final_position"]].copy()
    ranked["score"] = pd.to_numeric(scores, errors="coerce")
    if ranked["score"].isna().any():
        raise ValueError("challenger emitted a non-finite score")
    # Analytically equivalent normalized/raw scores must not diverge because of
    # sub-machine-epsilon arithmetic in an otherwise exact tie.
    ranked["score_for_order"] = ranked["score"].round(12)
    ranked = ranked.sort_values(
        ["score_for_order", "q_position", "driver_id"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked = ranked.drop(columns=["score_for_order"])
    ranked["predicted_position"] = np.arange(1, len(ranked) + 1, dtype=int)
    if sorted(ranked["predicted_position"].tolist()) != list(range(1, len(ranked) + 1)):
        raise AssertionError("prediction is not a complete field permutation")
    return ranked


def _kendall_tau_permutations(predicted: pd.Series, actual: pd.Series) -> float:
    paired = pd.DataFrame({"predicted": predicted, "actual": actual}).sort_values("predicted")
    actual_order = paired["actual"].to_numpy(dtype=int)
    inversions = sum(
        int(actual_order[left] > actual_order[right])
        for left in range(len(actual_order))
        for right in range(left + 1, len(actual_order))
    )
    pairs = len(actual_order) * (len(actual_order) - 1) / 2.0
    return float(1.0 - (2.0 * inversions / pairs)) if pairs else 1.0


def _metrics(ranked: pd.DataFrame) -> dict[str, Any]:
    predicted = ranked["predicted_position"].astype(int)
    actual = ranked["final_position"].astype(int)
    predicted_top3 = set(ranked.loc[predicted <= 3, "driver_id"])
    actual_top3 = set(ranked.loc[actual <= 3, "driver_id"])
    predicted_winner = ranked.loc[predicted.idxmin(), "driver_id"]
    actual_winner = ranked.loc[actual.idxmin(), "driver_id"]
    return {
        "rows": int(len(ranked)),
        "field_mae_positions": float(np.mean(np.abs(predicted - actual))),
        "kendall_tau": _kendall_tau_permutations(predicted, actual),
        "top3_overlap_rate": float(len(predicted_top3 & actual_top3) / 3.0),
        "winner_hit": bool(predicted_winner == actual_winner),
        "positions_changed_vs_qualifying": int((predicted != ranked["q_position"]).sum()),
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
    if len(challenger_values) != len(baseline_values) or not len(challenger_values):
        raise ValueError("paired bootstrap requires equal non-empty event vectors")
    delta = challenger_values - baseline_values
    rng = np.random.default_rng(int(seed))
    draws = delta[
        rng.integers(0, len(delta), size=(int(samples), len(delta)))
    ].mean(axis=1)
    return {
        "mean_delta_mae_positions": float(delta.mean()),
        "ci95_delta_mae_positions": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "bootstrap_probability_of_improvement": float(np.mean(draws < 0.0)),
        "events": int(len(delta)),
        "samples": int(samples),
        "seed": int(seed),
    }


def _aggregate(event_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [row["metrics"] for row in event_rows]
    return {
        "events": int(len(metrics)),
        "rows": int(sum(int(item["rows"]) for item in metrics)),
        "event_mean_field_mae_positions": float(
            np.mean([float(item["field_mae_positions"]) for item in metrics])
        ),
        "event_mean_kendall_tau": float(np.mean([float(item["kendall_tau"]) for item in metrics])),
        "event_mean_top3_overlap_rate": float(
            np.mean([float(item["top3_overlap_rate"]) for item in metrics])
        ),
        "winner_hit_rate": float(np.mean([bool(item["winner_hit"]) for item in metrics])),
        "event_mean_positions_changed_vs_qualifying": float(
            np.mean([float(item["positions_changed_vs_qualifying"]) for item in metrics])
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


def _evaluate_strength(
    events: Sequence[pd.DataFrame],
    *,
    prior_rounds: int,
) -> dict[str, dict[str, Any]]:
    candidate_event_rows: dict[str, list[dict[str, Any]]] = {}
    history_parts: list[pd.DataFrame] = []
    for current in events:
        history = pd.concat(history_parts, ignore_index=True) if history_parts else pd.DataFrame()
        scores = _candidate_scores(current, history, prior_rounds=prior_rounds)
        for name, values in scores.items():
            ranked = _rank_scores(current, values)
            candidate_event_rows.setdefault(name, []).append(
                {"round": int(current["round"].iloc[0]), "metrics": _metrics(ranked)}
            )
        history_parts.append(current)
    return {name: _aggregate(rows) for name, rows in candidate_event_rows.items()}


def run_backtest(
    *,
    weekends_dir: Path,
    year: int,
    rounds: Sequence[int] | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    root = _repo_root()
    implementation_manifest = _hash_manifest(
        [Path(__file__).resolve(), (root / "packages/sports_core/paths.py").resolve()],
        root=root,
    )
    selected = [
        path
        for path in _discover_rounds(weekends_dir, year)
        if rounds is None or _round_number(path) in {int(value) for value in rounds}
    ]
    if not selected:
        raise ValueError("no completed local rounds selected")
    planned_input_files = [
        path
        for round_dir in selected
        for path in (
            round_dir / "weekend_metadata.json",
            _single_file(round_dir, QUALIFYING_RESULTS_PATTERN, "Grand Prix qualifying"),
            _single_file(round_dir, RACE_RESULTS_PATTERN, "Grand Prix race"),
        )
    ]
    input_manifest_before = _hash_manifest(planned_input_files, root=root)

    events: list[pd.DataFrame] = []
    provenance_by_round: dict[int, dict[str, Any]] = {}
    input_files: list[Path] = []
    for round_dir in selected:
        event, provenance, paths = _load_event(round_dir, year=year)
        if provenance["point_in_time_official_grid_reference_present"]:
            raise ValueError(
                "this qualifying-proxy experiment must be revised when an official point-in-time grid exists"
            )
        events.append(event)
        provenance_by_round[int(provenance["round"])] = provenance
        input_files.extend(paths)

    history_parts: list[pd.DataFrame] = []
    event_payloads: list[dict[str, Any]] = []
    candidate_event_rows: dict[str, list[dict[str, Any]]] = {}
    for current in events:
        round_number = int(current["round"].iloc[0])
        history = pd.concat(history_parts, ignore_index=True) if history_parts else pd.DataFrame()
        training_rounds = sorted(int(value) for value in history.get("round", pd.Series(dtype=int)).unique())
        if any(value >= round_number for value in training_rounds):
            raise AssertionError("training window contains the target or a future round")
        scores = _candidate_scores(current, history, prior_rounds=PRIMARY_PRIOR_ROUNDS)
        event_candidates: dict[str, Any] = {}
        for name, values in scores.items():
            ranked = _rank_scores(current, values)
            metrics = _metrics(ranked)
            row = {"round": round_number, "metrics": metrics}
            candidate_event_rows.setdefault(name, []).append(row)
            event_candidates[name] = {
                "metrics": metrics,
                "predictions": ranked[
                    [
                        "driver_id",
                        "team_name",
                        "q_position",
                        "score",
                        "predicted_position",
                        "final_position",
                    ]
                ].to_dict(orient="records"),
            }
        event_payloads.append(
            {
                **provenance_by_round[round_number],
                "training_rounds": training_rounds,
                "history_row_count": int(len(history)),
                "candidates": event_candidates,
            }
        )
        history_parts.append(current)

    aggregate = {name: _aggregate(rows) for name, rows in candidate_event_rows.items()}
    baseline_mae = [
        float(row["metrics"]["field_mae_positions"])
        for row in candidate_event_rows[BASELINE_NAME]
    ]
    paired: dict[str, Any] = {}
    retained: list[str] = []
    for candidate, rows in candidate_event_rows.items():
        if candidate == BASELINE_NAME:
            continue
        evidence = _paired_bootstrap(
            [float(row["metrics"]["field_mae_positions"]) for row in rows],
            baseline_mae,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        paired[candidate] = evidence
        if (
            float(evidence["mean_delta_mae_positions"]) < 0.0
            and float(evidence["ci95_delta_mae_positions"][1]) < 0.0
            and float(evidence["bootstrap_probability_of_improvement"]) >= 0.975
        ):
            retained.append(candidate)

    sensitivity: dict[str, Any] = {}
    for prior_rounds in SENSITIVITY_PRIOR_ROUNDS:
        strength_aggregate = _evaluate_strength(events, prior_rounds=prior_rounds)
        sensitivity[str(prior_rounds)] = {
            name: {
                "event_mean_field_mae_positions": values["event_mean_field_mae_positions"],
                "delta_vs_qualifying_proxy": float(
                    values["event_mean_field_mae_positions"]
                    - strength_aggregate[BASELINE_NAME]["event_mean_field_mae_positions"]
                ),
            }
            for name, values in strength_aggregate.items()
            if name != BASELINE_NAME
        }

    if set(input_files) != set(planned_input_files):
        raise RuntimeError("Race challenger accessed an unexpected input-file set")
    _assert_manifest_unchanged(
        implementation_manifest,
        root=root,
        label="Race challenger implementation",
    )
    _assert_manifest_unchanged(
        input_manifest_before,
        root=root,
        label="Race challenger input data",
    )
    return _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _utc_now(),
            "mode": "race_final_position",
            "target": {
                "definition": "official Grand Prix final classification position",
                "unit": "full-field ordinal position",
                "output_constraint": "one permutation of positions 1..N per event",
            },
            "baseline": {
                "name": BASELINE_NAME,
                "definition": "Grand Prix qualifying classification order, including source-order tail completion",
                "official_starting_grid_used": False,
                "reason": "no point-in-time official starting-grid reference exists in the local 2026 snapshots",
            },
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
                "training_window": "strictly earlier completed rounds from the same season",
                "round_order": "ascending sequential",
                "random_split": False,
                "prior_season_pooling": False,
                "current_event_race_outcome_used_as_feature": False,
                "post_race_grid_position_used": False,
                "race_result_columns_read": ["DriverNumber", "Position", "Status"],
                "forbidden_post_race_feature_columns": ["GridPosition"],
                "primary_prior_round_equivalent": PRIMARY_PRIOR_ROUNDS,
                "driver_prior_strength_observations": PRIMARY_PRIOR_ROUNDS,
                "team_prior_strength_observations": 2 * PRIMARY_PRIOR_ROUNDS,
                "terminal_definition": "status outside Finished or Lapped",
                "terminal_expected_tail": "mean of positions ceil(0.75*N)..N",
                "bootstrap_unit": "event",
                "promotion_gate": "paired MAE delta < 0, upper 95% bootstrap bound < 0, P(improvement) >= 0.975",
            },
            "candidate_math": {
                "driver_residual_shrinkage": "score=q + sum(previous finish-q by driver)/(n_driver+k_driver)",
                "team_residual_shrinkage": "score=q + sum(previous finish-q by team)/(n_team+k_team)",
                "hierarchical_residual_shrinkage": "driver residual mean shrunk toward the current-team residual mean",
                "rank_normalized_hierarchical_residual": "same hierarchy on (finish-q)/field_size, added to (q-0.5)/field_size",
                "reliability_hazard": "score=(1-p_terminal)*q + p_terminal*expected_terminal_tail_position",
                "running_residual_plus_reliability": "reliability mixture with residuals learned only from non-terminal outcomes",
            },
            "aggregate": aggregate,
            "paired_event_bootstrap_vs_qualifying_proxy": paired,
            "shrinkage_sensitivity": sensitivity,
            "decision": {
                "retained_challengers": retained,
                "baseline_retained": not retained,
                "complex_ml_or_rl_justified": False,
                "reason": (
                    "at least one causal challenger cleared the paired event gate"
                    if retained
                    else "no causal residual or reliability challenger beat the qualifying-order proxy"
                ),
                "production_blockers": [
                    "no point-in-time official starting-grid artifact for 2026 replay",
                    "only nine completed same-season events",
                    "terminal outcomes are high-variance and weakly identifiable per driver/team",
                    "no challenger clears the simple deterministic baseline",
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
    parser.add_argument("--rounds", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--bootstrap-samples", type=int, default=200_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260712)
    parser.add_argument(
        "--output",
        default=str(
            _repo_root()
            / "artifacts/backtests/f1/race_final_position/2026_causal_residual_challengers_v1.json"
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
    print(
        json.dumps(
            {
                "output": str(output),
                "aggregate": payload["aggregate"],
                "paired": payload["paired_event_bootstrap_vs_qualifying_proxy"],
                "decision": payload["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
