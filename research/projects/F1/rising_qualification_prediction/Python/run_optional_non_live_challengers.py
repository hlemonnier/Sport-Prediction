#!/usr/bin/env python3
"""Evaluate optional F1 rank/quantile challengers on locked event blocks.

The command is evidence-only.  It never changes a production profile and it
never treats backend availability as model quality.  A bounded early/late
holdout inside the prior season selects each backend configuration; absolute
pace/rank fitting then uses only current-season calibration events, and every
audit event is later in time.  Quantile intervals remain raw model diagnostics,
not conformal or promotion-ready intervals.
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
from typing import Any, Iterable, Mapping, Sequence

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
from packages.f1.orchestration.non_live_validation import (
    EventError,
    paired_event_diagnostics,
    validate_event_partitions,
)
from run_qualifying_pairwise_challenger_backtest import (
    FEATURE_ALLOWLIST,
    _event_frame,
)


ROUND_PATTERN = re.compile(r"round_(\d+)", re.IGNORECASE)
RANKING_BACKENDS = ("xgboost_lambdarank", "lightgbm_lambdarank")
RANKING_CONFIG_CANDIDATES: tuple[dict[str, int | float], ...] = (
    {"n_estimators": 120, "learning_rate": 0.04, "max_depth": 3, "num_leaves": 15},
    {"n_estimators": 220, "learning_rate": 0.025, "max_depth": 4, "num_leaves": 31},
)
QUANTILE_CONFIG_CANDIDATES: tuple[dict[str, int | float], ...] = (
    {"n_estimators": 120, "learning_rate": 0.04, "max_depth": 3},
    {"n_estimators": 220, "learning_rate": 0.025, "max_depth": 4},
)
NOMINAL_INTERVAL_COVERAGE = 0.85


def _root() -> Path:
    return Path(__file__).resolve().parents[5]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_manifest(paths: Iterable[Path], *, root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for path in sorted({Path(value).resolve() for value in paths}):
        if not path.exists():
            raise FileNotFoundError(f"manifest input does not exist: {path}")
        try:
            label = str(path.relative_to(root))
        except ValueError:
            label = str(path)
        records.append({"path": label, "sha256": _sha256(path)})
    return records


def _resolve_metadata_reference(root: Path, event_dir: Path, value: object) -> Path:
    candidate = Path(str(value)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    root_candidate = (root / candidate).resolve()
    if root_candidate.exists():
        return root_candidate
    return (event_dir / candidate.name).resolve()


def _round_number(path: Path) -> int:
    match = ROUND_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"invalid round directory: {path}")
    return int(match.group(1))


def _qualifying_start(
    metadata: Mapping[str, Any],
    *,
    use_selection_ordinal: bool,
) -> tuple[pd.Timestamp, str]:
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
    year = int(metadata.get("year"))
    round_number = int(metadata.get("round_number"))
    if use_selection_ordinal:
        # Every prior-season record uses the same round-ordinal clock. Mixing
        # real timestamps with legacy fallbacks can reorder the season (for
        # example a missing round 10 becomes January 11 while round 2 remains
        # in March), which would leak late events into the early fit block.
        ordinal = pd.Timestamp(f"{year}-01-01T00:00:00Z") + pd.to_timedelta(
            round_number, unit="D"
        )
        return ordinal, "season_round_ordinal_selection_only"
    timestamp = pd.to_datetime(
        qualifying.get("scheduled_start_utc"), errors="coerce", utc=True
    )
    if pd.isna(timestamp):
        raise ValueError(
            "target-season calibration/audit requires scheduled_qualifying_start_utc; "
            "season/round ordinal fallback is selection-only"
        )
    resolved = pd.Timestamp(timestamp)
    if int(resolved.year) != year:
        raise ValueError(
            "scheduled_qualifying_start_utc year does not match weekend metadata year"
        )
    return resolved, "scheduled_qualifying_start_utc"


def _load_event_dataset(
    weekends_dir: Path,
    *,
    years: Sequence[int],
    target_year: int,
) -> tuple[
    pd.DataFrame,
    dict[int, dict[str, Any]],
    tuple[Path, ...],
    tuple[Path, ...],
    tuple[Path, ...],
]:
    root = _root()
    frames: list[pd.DataFrame] = []
    infos: dict[int, dict[str, Any]] = {}
    inference_files: set[Path] = set()
    target_files: set[Path] = set()
    protocol_files: set[Path] = set()
    for year in sorted({int(value) for value in years}):
        for event_dir in sorted(
            (weekends_dir / str(year)).glob("round_*"), key=_round_number
        ):
            frame, info, event_files = _event_frame(root, event_dir)
            metadata_path = event_dir / "weekend_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            directory_year = int(year)
            directory_round = _round_number(event_dir)
            metadata_year = int(metadata.get("year"))
            metadata_round = int(metadata.get("round_number"))
            expected_event_key = metadata_year * 100 + metadata_round
            info_event_key = int(info["event_key"])
            if metadata_year != directory_year:
                raise ValueError(
                    f"{event_dir}: metadata year {metadata_year} does not match "
                    f"directory year {directory_year}"
                )
            if metadata_round != directory_round:
                raise ValueError(
                    f"{event_dir}: metadata round {metadata_round} does not match "
                    f"directory round {directory_round}"
                )
            if info_event_key != expected_event_key:
                raise ValueError(
                    f"{event_dir}: event_key {info_event_key} does not match "
                    f"metadata event_key {expected_event_key}"
                )
            event_time, time_semantics = _qualifying_start(
                metadata,
                use_selection_ordinal=directory_year < int(target_year),
            )
            event_key = info_event_key
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
            protocol_files.add(metadata_path.resolve())
            qualifying = next(
                (
                    session
                    for session in metadata.get("sessions", [])
                    if isinstance(session, Mapping)
                    and str(session.get("session_type", "")).strip().lower()
                    == "qualifying"
                ),
                None,
            )
            target_paths: set[Path] = set()
            if qualifying is not None:
                for key in ("results_path", "laps_path"):
                    if qualifying.get(key):
                        target_paths.add(
                            _resolve_metadata_reference(
                                root, event_dir, qualifying.get(key)
                            )
                        )
            for path in (Path(value).resolve() for value in event_files):
                if path == metadata_path.resolve():
                    protocol_files.add(path)
                elif path in target_paths:
                    target_files.add(path)
                else:
                    inference_files.add(path)
    if not frames:
        raise ValueError("no completed weekend snapshots were loaded")
    dataset = pd.concat(frames, ignore_index=True, sort=False)
    dataset["event_key"] = pd.to_numeric(dataset["event_key"], errors="raise").astype(int)
    required = {
        "event_key",
        "driver_id",
        "qualy_position",
        ACTUAL_LAP_COLUMN,
        "quality_aware_anchor_seconds",
    }
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise ValueError(
            "shared event dataset is missing optional-challenger target fields: "
            f"{missing}"
        )
    return (
        dataset,
        infos,
        tuple(sorted(inference_files)),
        tuple(sorted(target_files)),
        tuple(sorted(protocol_files)),
    )


def _ordered_event_keys(frame: pd.DataFrame) -> tuple[int, ...]:
    required = {"event_key", "event_as_of"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"event ordering is missing columns: {missing}")
    work = frame[["event_key", "event_as_of"]].copy()
    work["event_key"] = pd.to_numeric(work["event_key"], errors="raise").astype(int)
    work["event_as_of"] = pd.to_datetime(work["event_as_of"], errors="coerce", utc=True)
    if work["event_as_of"].isna().any():
        raise ValueError("event ordering requires valid event_as_of timestamps")
    per_event = work.groupby("event_key", sort=False).agg(
        event_as_of=("event_as_of", "first"),
        timestamp_count=("event_as_of", "nunique"),
    )
    if per_event["timestamp_count"].ne(1).any():
        raise ValueError("every event must have exactly one event_as_of timestamp")
    ordered = per_event.reset_index().sort_values(
        ["event_as_of", "event_key"], kind="mergesort"
    )
    return tuple(ordered["event_key"].astype(int).tolist())


def _selection_holdout(
    selection: pd.DataFrame,
    *,
    minimum_fit_events: int = 4,
    maximum_validation_events: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ordered = _ordered_event_keys(selection)
    if len(ordered) < int(minimum_fit_events) + 2:
        raise ValueError(
            "bounded hyperparameter selection requires at least "
            f"{int(minimum_fit_events) + 2} selection-season events"
        )
    validation_count = min(
        int(maximum_validation_events), len(ordered) - int(minimum_fit_events)
    )
    validation_count = max(2, validation_count)
    fit_keys = ordered[:-validation_count]
    validation_keys = ordered[-validation_count:]
    numeric = pd.to_numeric(selection["event_key"], errors="raise").astype(int)
    fit = selection.loc[numeric.isin(fit_keys)].copy()
    validation = selection.loc[numeric.isin(validation_keys)].copy()
    fit_max = pd.to_datetime(fit["event_as_of"], utc=True).max()
    validation_min = pd.to_datetime(validation["event_as_of"], utc=True).min()
    if fit_max >= validation_min:
        raise ValueError("selection fit events must strictly precede validation events")
    return fit, validation, {
        "method": "bounded_within_selection_season_early_fit_late_validation",
        "fit_event_keys": list(fit_keys),
        "validation_event_keys": list(validation_keys),
        "candidate_count_bound": {
            "ranking_per_backend": len(RANKING_CONFIG_CANDIDATES),
            "quantile": len(QUANTILE_CONFIG_CANDIDATES),
        },
    }


def _select_candidate_evidence(
    candidates: Sequence[Mapping[str, Any]],
    *,
    objective_key: str,
    tie_break_keys: Sequence[str] = (),
) -> dict[str, Any]:
    available: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if str(candidate.get("status")) != "available":
            continue
        values = [candidate.get(objective_key), *(candidate.get(key) for key in tie_break_keys)]
        if not all(value is not None and np.isfinite(float(value)) for value in values):
            continue
        available.append({**dict(candidate), "_candidate_index": index})
    if not available:
        raise ValueError("no bounded selection candidate produced finite validation evidence")
    selected = min(
        available,
        key=lambda item: (
            float(item[objective_key]),
            *(float(item[key]) for key in tie_break_keys),
            int(item["_candidate_index"]),
        ),
    )
    selected.pop("_candidate_index", None)
    return selected


def _flatten_prediction_payload(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(value) for value in payload if isinstance(value, Mapping)]
    if not isinstance(payload, Mapping):
        raise ValueError("champion prediction artifact must be a JSON object or row list")
    direct = payload.get("predictions")
    if isinstance(direct, list):
        return [dict(value) for value in direct if isinstance(value, Mapping)]
    rows: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        if not isinstance(event, Mapping):
            continue
        nested = event.get("predictions")
        if not isinstance(nested, list):
            nested = event.get("prediction_vs_reality")
        if not isinstance(nested, list):
            continue
        for value in nested:
            if not isinstance(value, Mapping):
                continue
            row = dict(value)
            row.setdefault("event_key", event.get("event_key"))
            rows.append(row)
    return rows


def _first_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def _load_champion_predictions(path: Path, *, mode: str) -> pd.DataFrame:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"champion prediction artifact does not exist: {resolved}")
    if resolved.suffix.lower() == ".csv":
        frame = pd.read_csv(resolved)
    else:
        frame = pd.DataFrame(
            _flatten_prediction_payload(json.loads(resolved.read_text(encoding="utf-8")))
        )
    required = {"event_key", "driver_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"champion prediction artifact is missing columns: {missing}")
    output = frame[["event_key", "driver_id"]].copy()
    output["event_key"] = pd.to_numeric(output["event_key"], errors="raise").astype(int)
    output["driver_id"] = output["driver_id"].astype(str).str.strip()
    if mode == "qualifying":
        position = _first_column(
            frame,
            (
                "champion_qualifying_position",
                "predicted_qualifying_position",
                "predicted_position",
            ),
        )
        if position is None:
            raise ValueError("Qualifying champion artifact has no predicted position column")
        output["champion_rank"] = pd.to_numeric(frame[position], errors="coerce")
    elif mode == "best_lap":
        p50 = _first_column(frame, ("champion_lap_p50", "lap_p50"))
        if p50 is None:
            raise ValueError("Best-Lap champion artifact has no lap_p50 column")
        output["champion_lap_p50"] = pd.to_numeric(frame[p50], errors="coerce")
        for quantile in ("p05", "p90"):
            column = _first_column(
                frame, (f"champion_lap_{quantile}", f"lap_{quantile}")
            )
            output[f"champion_lap_{quantile}"] = (
                pd.to_numeric(frame[column], errors="coerce")
                if column is not None
                else np.nan
            )
    else:
        raise ValueError(f"unsupported champion mode: {mode}")
    if output.duplicated(["event_key", "driver_id"]).any():
        raise ValueError("champion predictions contain duplicate event-driver rows")
    return output


def _align_champion(
    audit: pd.DataFrame,
    champion: pd.DataFrame,
    *,
    mode: str,
) -> pd.DataFrame:
    keys = audit[["event_key", "driver_id"]].copy()
    keys["event_key"] = pd.to_numeric(keys["event_key"], errors="raise").astype(int)
    keys["driver_id"] = keys["driver_id"].astype(str).str.strip()
    audit_events = set(keys["event_key"].unique().tolist())
    selected = champion.loc[champion["event_key"].isin(audit_events)].copy()
    expected = set(map(tuple, keys[["event_key", "driver_id"]].to_records(index=False)))
    observed = set(
        map(tuple, selected[["event_key", "driver_id"]].to_records(index=False))
    )
    if expected != observed:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{mode} champion rows do not match audit inference rows; "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    aligned = keys.merge(selected, on=["event_key", "driver_id"], validate="one_to_one")
    if mode == "qualifying":
        numeric_rank = pd.to_numeric(aligned["champion_rank"], errors="coerce")
        if not np.isfinite(numeric_rank).all():
            raise ValueError("Qualifying champion positions must all be finite")
        if not np.equal(numeric_rank, np.floor(numeric_rank)).all():
            raise ValueError("Qualifying champion positions must all be integers")
        for _, event in aligned.groupby("event_key", sort=False):
            positions = sorted(
                pd.to_numeric(event["champion_rank"], errors="raise")
                .astype(int)
                .tolist()
            )
            if positions != list(range(1, len(event) + 1)):
                raise ValueError("Qualifying champion must be a legal full-field permutation")
    else:
        p50 = pd.to_numeric(aligned["champion_lap_p50"], errors="coerce")
        if not np.isfinite(p50).all():
            raise ValueError("Best-Lap champion lap_p50 must cover every audit entrant")
        if p50.le(0.0).any():
            raise ValueError("Best-Lap champion lap times must be positive")
        p05 = pd.to_numeric(aligned["champion_lap_p05"], errors="coerce")
        p90 = pd.to_numeric(aligned["champion_lap_p90"], errors="coerce")
        for label, values in (("p05", p05), ("p90", p90)):
            supplied = values.notna()
            if ((supplied & ~np.isfinite(values)) | (supplied & values.le(0.0))).any():
                raise ValueError(
                    f"Best-Lap champion {label} must be finite and positive when supplied"
                )
        if (p05.notna() & p05.gt(p50)).any() or (
            p90.notna() & p50.gt(p90)
        ).any():
            raise ValueError("Best-Lap champion quantiles must satisfy p05 <= p50 <= p90")
    return aligned


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
        champion = pd.to_numeric(event["champion_rank"], errors="coerce")
        rehearsal = pd.to_numeric(event["rehearsal_baseline_rank"], errors="coerce")
        observed = pd.Series(np.isfinite(actual), index=event.index)
        candidate_available = pd.Series(np.isfinite(predicted), index=event.index)
        champion_available = pd.Series(np.isfinite(champion), index=event.index)
        scored = observed & candidate_available & champion_available
        actual_scored = actual.loc[scored]
        candidate_scored = predicted.loc[scored]
        champion_scored = champion.loc[scored]
        rehearsal_scored = rehearsal.loc[scored]
        field_size = len(event)
        finite_candidate_positions = predicted.loc[candidate_available]
        candidate_positions_integral = bool(
            np.equal(
                finite_candidate_positions,
                np.floor(finite_candidate_positions),
            ).all()
        )
        actual_top3 = set(
            event.loc[observed]
            .nsmallest(min(3, int(observed.sum())), "actual_qualifying_position")[
                "driver_id"
            ]
        )
        actual_top10 = set(
            event.loc[observed]
            .nsmallest(min(10, int(observed.sum())), "actual_qualifying_position")[
                "driver_id"
            ]
        )
        candidate_top3 = set(event.nsmallest(min(3, field_size), "predicted_rank")["driver_id"])
        champion_top3 = set(event.nsmallest(min(3, field_size), "champion_rank")["driver_id"])
        candidate_top10 = set(event.nsmallest(min(10, field_size), "predicted_rank")["driver_id"])
        champion_top10 = set(event.nsmallest(min(10, field_size), "champion_rank")["driver_id"])
        actual_winner = (
            str(event.loc[observed].nsmallest(1, "actual_qualifying_position").iloc[0]["driver_id"])
            if observed.any()
            else None
        )
        rows.append(
            {
                "event_key": int(event_key),
                "event_format": str(event["event_format"].iloc[0]),
                "stratum": _weekend_stratum(str(event["event_format"].iloc[0])),
                "rows": field_size,
                "target_observed_rows": int(observed.sum()),
                "target_observed_rate": float(observed.mean()),
                "candidate_output_coverage": float(candidate_available.mean()),
                "champion_output_coverage": float(champion_available.mean()),
                "candidate_legal_permutation": bool(
                    candidate_available.all()
                    and candidate_positions_integral
                    and sorted(finite_candidate_positions.astype(int).tolist())
                    == list(range(1, field_size + 1))
                ),
                "candidate_mae": float((candidate_scored - actual_scored).abs().mean()),
                "champion_mae": float((champion_scored - actual_scored).abs().mean()),
                "rehearsal_baseline_mae": float(
                    (rehearsal_scored - actual_scored).abs().mean()
                ),
                "candidate_kendall": float(candidate_scored.corr(actual_scored, method="kendall")),
                "champion_kendall": float(champion_scored.corr(actual_scored, method="kendall")),
                "rehearsal_baseline_kendall": float(
                    rehearsal_scored.corr(actual_scored, method="kendall")
                ),
                "candidate_pole_hit": bool(
                    actual_winner is not None
                    and str(event.nsmallest(1, "predicted_rank").iloc[0]["driver_id"])
                    == actual_winner
                ),
                "champion_pole_hit": bool(
                    actual_winner is not None
                    and str(event.nsmallest(1, "champion_rank").iloc[0]["driver_id"])
                    == actual_winner
                ),
                "candidate_top3_overlap": (
                    float(len(candidate_top3 & actual_top3) / len(actual_top3))
                    if actual_top3
                    else float("nan")
                ),
                "champion_top3_overlap": (
                    float(len(champion_top3 & actual_top3) / len(actual_top3))
                    if actual_top3
                    else float("nan")
                ),
                "candidate_top10_overlap": (
                    float(len(candidate_top10 & actual_top10) / len(actual_top10))
                    if actual_top10
                    else float("nan")
                ),
                "champion_top10_overlap": (
                    float(len(champion_top10 & actual_top10) / len(actual_top10))
                    if actual_top10
                    else float("nan")
                ),
            }
        )
    return rows


def _quantile_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event_key, event in frame.groupby("event_key", sort=True):
        actual = pd.to_numeric(event[ACTUAL_LAP_COLUMN], errors="coerce")
        predicted = pd.to_numeric(event["lap_p50"], errors="coerce")
        raw_anchor = pd.to_numeric(event["raw_anchor_lap_seconds"], errors="coerce")
        champion = pd.to_numeric(event["champion_lap_p50"], errors="coerce")
        candidate_lower = pd.to_numeric(event["lap_p05"], errors="coerce")
        candidate_upper = pd.to_numeric(event["lap_p90"], errors="coerce")
        champion_lower = pd.to_numeric(event["champion_lap_p05"], errors="coerce")
        champion_upper = pd.to_numeric(event["champion_lap_p90"], errors="coerce")
        observed = pd.Series(np.isfinite(actual), index=event.index)
        candidate_available = pd.Series(np.isfinite(predicted), index=event.index)
        champion_available = pd.Series(np.isfinite(champion), index=event.index)
        candidate_interval_available = pd.Series(
            np.isfinite(candidate_lower) & np.isfinite(candidate_upper),
            index=event.index,
        )
        champion_interval_available = pd.Series(
            np.isfinite(champion_lower) & np.isfinite(champion_upper),
            index=event.index,
        )
        point_scored = observed & candidate_available & champion_available
        candidate_interval_scored = observed & candidate_interval_available
        champion_interval_scored = observed & champion_interval_available

        def pinball(prediction: pd.Series, quantile: float) -> float:
            mask = observed & np.isfinite(prediction)
            error = actual.loc[mask] - prediction.loc[mask]
            return float(np.maximum(quantile * error, (quantile - 1.0) * error).mean())

        rows.append(
            {
                "event_key": int(event_key),
                "event_format": str(event["event_format"].iloc[0]),
                "stratum": _weekend_stratum(str(event["event_format"].iloc[0])),
                "rows": int(len(event)),
                "target_observed_rows": int(observed.sum()),
                "target_observed_rate": float(observed.mean()),
                "candidate_output_coverage": float(candidate_available.mean()),
                "champion_output_coverage": float(champion_available.mean()),
                "candidate_interval_output_coverage": float(
                    candidate_interval_available.mean()
                ),
                "champion_interval_output_coverage": float(
                    champion_interval_available.mean()
                ),
                "candidate_mae_seconds": float(
                    (predicted.loc[point_scored] - actual.loc[point_scored]).abs().mean()
                ),
                "champion_mae_seconds": float(
                    (champion.loc[point_scored] - actual.loc[point_scored]).abs().mean()
                ),
                "raw_anchor_mae_seconds": float(
                    (raw_anchor.loc[observed] - actual.loc[observed]).abs().mean()
                ),
                "candidate_p05_pinball_loss": pinball(candidate_lower, 0.05),
                "candidate_p50_pinball_loss": pinball(predicted, 0.50),
                "candidate_p90_pinball_loss": pinball(candidate_upper, 0.90),
                "champion_p05_pinball_loss": pinball(champion_lower, 0.05),
                "champion_p50_pinball_loss": pinball(champion, 0.50),
                "champion_p90_pinball_loss": pinball(champion_upper, 0.90),
                "candidate_p05_p90_coverage": float(
                    (
                        actual.loc[candidate_interval_scored]
                        .ge(candidate_lower.loc[candidate_interval_scored])
                        & actual.loc[candidate_interval_scored].le(
                            candidate_upper.loc[candidate_interval_scored]
                        )
                    ).mean()
                ),
                "champion_p05_p90_coverage": float(
                    (
                        actual.loc[champion_interval_scored]
                        .ge(champion_lower.loc[champion_interval_scored])
                        & actual.loc[champion_interval_scored].le(
                            champion_upper.loc[champion_interval_scored]
                        )
                    ).mean()
                ),
                "candidate_p05_p90_mean_width_seconds": float(
                    (candidate_upper - candidate_lower).mean()
                ),
                "champion_p05_p90_mean_width_seconds": float(
                    (champion_upper - champion_lower).mean()
                ),
            }
        )
    return rows


def _weekend_stratum(event_format: str) -> str:
    return "sprint" if "sprint" in str(event_format).strip().lower() else "standard"


def _aggregate_metrics(rows: Sequence[Mapping[str, Any]], *keys: str) -> dict[str, float]:
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in keys
    }


def _stratum_aggregates(
    rows: Sequence[Mapping[str, Any]],
    *keys: str,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for stratum in ("standard", "sprint"):
        selected = [row for row in rows if str(row.get("stratum")) == stratum]
        if not selected:
            output[stratum] = {"event_count": 0}
            continue
        output[stratum] = {
            "event_count": len(selected),
            **_aggregate_metrics(selected, *keys),
        }
    return output


def _paired_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_key: str,
    champion_key: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row.get(candidate_key) is not None
        and row.get(champion_key) is not None
        and np.isfinite(float(row[candidate_key]))
        and np.isfinite(float(row[champion_key]))
    ]
    if len(eligible) < 2:
        return {
            "status": "unavailable_insufficient_complete_event_pairs",
            "event_count": len(eligible),
            "required_standard_and_sprint_present": False,
            "all_weekend_strata_improve": False,
        }
    diagnostics = paired_event_diagnostics(
        [
            EventError(
                event_key=str(row["event_key"]),
                baseline_error=float(row[champion_key]),
                candidate_error=float(row[candidate_key]),
                stratum=str(row.get("stratum") or "unknown"),
            )
            for row in eligible
        ],
        bootstrap_samples=int(bootstrap_samples),
        seed=int(seed),
    ).to_payload()
    stratum_deltas = diagnostics["stratum_mean_deltas"]
    diagnostics["required_standard_and_sprint_present"] = bool(
        {"standard", "sprint"}.issubset(stratum_deltas)
    )
    diagnostics["all_weekend_strata_improve"] = bool(
        diagnostics["required_standard_and_sprint_present"]
        and all(float(stratum_deltas[name]) < 0.0 for name in ("standard", "sprint"))
    )
    return diagnostics


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


def _ranking_selection_evidence(
    *,
    backend: str,
    feature_columns: tuple[str, ...],
    selection_fit: pd.DataFrame,
    selection_validation: pd.DataFrame,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    selection_years = pd.to_numeric(selection_fit["season"], errors="raise").astype(int)
    if selection_years.nunique() != 1:
        raise ValueError("ranking selection fit must contain exactly one season")
    selection_year = int(selection_years.iloc[0])
    candidates: list[dict[str, Any]] = []
    inference = selection_validation.drop(
        columns=[
            "qualy_position",
            ACTUAL_LAP_COLUMN,
            "has_valid_qualifying_lap",
            "reached_q2",
            "reached_q3",
        ],
        errors="ignore",
    )
    for candidate_index, parameters in enumerate(RANKING_CONFIG_CANDIDATES):
        config = GroupedRankingConfig(
            feature_columns=feature_columns,
            backend=backend,
            target_column="qualy_position",
            minimum_training_events=2,
            **parameters,
        )
        result = fit_grouped_ranking_challenger(
            selection_fit,
            config=config,
            target_season=selection_year,
        )
        record: dict[str, Any] = {
            "candidate_id": f"{backend}_candidate_{candidate_index + 1}",
            "config": config.to_payload(),
            "config_sha256": config.fingerprint,
            "status": "available" if result.available else "unavailable",
            "manifest": result.manifest,
        }
        if not result.available:
            record["reason"] = result.unavailable_reason
            candidates.append(record)
            continue
        predicted = result.require_model().predict(inference)
        scored = pd.DataFrame(
            {
                "event_key": selection_validation["event_key"].to_numpy(),
                "actual": pd.to_numeric(
                    selection_validation["qualy_position"], errors="coerce"
                ).to_numpy(),
                "predicted": predicted["predicted_rank"].to_numpy(),
            }
        )
        event_metrics: list[dict[str, Any]] = []
        for event_key, event in scored.groupby("event_key", sort=True):
            event_metrics.append(
                {
                    "event_key": int(event_key),
                    "mae": float((event["predicted"] - event["actual"]).abs().mean()),
                    "kendall": float(
                        event["predicted"].corr(event["actual"], method="kendall")
                    ),
                }
            )
        record.update(
            {
                "event_metrics": event_metrics,
                "event_mean_mae": float(
                    np.mean([value["mae"] for value in event_metrics])
                ),
                "event_mean_kendall": float(
                    np.mean([value["kendall"] for value in event_metrics])
                ),
            }
        )
        record["negative_event_mean_kendall"] = -float(record["event_mean_kendall"])
        candidates.append(record)
    try:
        selected = _select_candidate_evidence(
            candidates,
            objective_key="event_mean_mae",
            tie_break_keys=("negative_event_mean_kendall",),
        )
    except ValueError as exc:
        return None, {
            "status": "unavailable_no_candidate",
            "backend": backend,
            "selection_rule": "minimum_event_mean_mae_then_maximum_event_mean_kendall",
            "reason": str(exc),
            "candidates": candidates,
        }
    return {
        key: selected["config"][key]
        for key in ("n_estimators", "learning_rate", "max_depth", "num_leaves")
    }, {
        "status": "selected_on_prior_season_holdout",
        "backend": backend,
        "selection_rule": "minimum_event_mean_mae_then_maximum_event_mean_kendall",
        "selected_candidate_id": selected["candidate_id"],
        "selected_config_sha256": selected["config_sha256"],
        "candidates": candidates,
    }


def _quantile_selection_evidence(
    *,
    feature_columns: tuple[str, ...],
    selection_fit: pd.DataFrame,
    selection_validation: pd.DataFrame,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    selection_years = pd.to_numeric(selection_fit["season"], errors="raise").astype(int)
    if selection_years.nunique() != 1:
        raise ValueError("quantile selection fit must contain exactly one season")
    selection_year = int(selection_years.iloc[0])
    training = selection_fit.copy()
    training["lap_residual_seconds"] = (
        pd.to_numeric(training[ACTUAL_LAP_COLUMN], errors="coerce")
        - pd.to_numeric(training["quality_aware_anchor_seconds"], errors="coerce")
    )
    inference = selection_validation.drop(
        columns=[
            "qualy_position",
            ACTUAL_LAP_COLUMN,
            "has_valid_qualifying_lap",
            "reached_q2",
            "reached_q3",
        ],
        errors="ignore",
    )
    actual = pd.to_numeric(selection_validation[ACTUAL_LAP_COLUMN], errors="coerce")
    anchor = pd.to_numeric(
        selection_validation["quality_aware_anchor_seconds"], errors="coerce"
    )
    candidates: list[dict[str, Any]] = []
    fit_before = pd.to_datetime(
        selection_validation["event_as_of"], errors="raise", utc=True
    ).min().isoformat().replace("+00:00", "Z")
    for candidate_index, parameters in enumerate(QUANTILE_CONFIG_CANDIDATES):
        config = TabularQuantileConfig(
            backend="lightgbm",
            feature_columns=feature_columns,
            target_column="lap_residual_seconds",
            target_season=selection_year,
            same_season_only=True,
            fit_before=fit_before,
            **parameters,
        )
        record: dict[str, Any] = {
            "candidate_id": f"lightgbm_quantile_candidate_{candidate_index + 1}",
            "config": config.to_payload(),
            "config_sha256": config.fingerprint,
        }
        try:
            model = fit_tabular_quantile_model(training, config=config)
        except TabularQuantileBackendUnavailable as exc:
            record.update(
                {
                    "status": "unavailable",
                    "reason": str(exc),
                    "backend_attempts": list(exc.attempts),
                }
            )
            candidates.append(record)
            continue
        predicted = model.predict(inference)
        scored = pd.DataFrame(
            {
                "event_key": selection_validation["event_key"].to_numpy(),
                "actual": actual.to_numpy(),
                "lap_p05": anchor.to_numpy() + predicted["lap_p05"].to_numpy(),
                "lap_p50": anchor.to_numpy() + predicted["lap_p50"].to_numpy(),
                "lap_p90": anchor.to_numpy() + predicted["lap_p90"].to_numpy(),
            }
        )
        event_metrics: list[dict[str, Any]] = []
        for event_key, event in scored.groupby("event_key", sort=True):
            observed = pd.to_numeric(event["actual"], errors="coerce").notna()
            interval = (
                event.loc[observed, "actual"].ge(event.loc[observed, "lap_p05"])
                & event.loc[observed, "actual"].le(event.loc[observed, "lap_p90"])
            )
            event_metrics.append(
                {
                    "event_key": int(event_key),
                    "observed_rows": int(observed.sum()),
                    "mae_seconds": float(
                        (
                            event.loc[observed, "lap_p50"]
                            - event.loc[observed, "actual"]
                        )
                        .abs()
                        .mean()
                    ),
                    "coverage": float(interval.mean()),
                    "mean_width_seconds": float(
                        (event["lap_p90"] - event["lap_p05"]).mean()
                    ),
                }
            )
        event_mean_coverage = float(
            np.mean([value["coverage"] for value in event_metrics])
        )
        record.update(
            {
                "status": "available",
                "manifest": model.manifest(),
                "event_metrics": event_metrics,
                "event_mean_mae_seconds": float(
                    np.mean([value["mae_seconds"] for value in event_metrics])
                ),
                "coverage_absolute_error": abs(
                    event_mean_coverage - NOMINAL_INTERVAL_COVERAGE
                ),
                "event_mean_interval_width_seconds": float(
                    np.mean([value["mean_width_seconds"] for value in event_metrics])
                ),
            }
        )
        candidates.append(record)
    try:
        selected = _select_candidate_evidence(
            candidates,
            objective_key="event_mean_mae_seconds",
            tie_break_keys=(
                "coverage_absolute_error",
                "event_mean_interval_width_seconds",
            ),
        )
    except ValueError as exc:
        return None, {
            "status": "unavailable_no_candidate",
            "selection_rule": (
                "minimum_event_mean_p50_mae_then_coverage_error_then_interval_width"
            ),
            "reason": str(exc),
            "candidates": candidates,
        }
    return {
        key: selected["config"][key]
        for key in ("n_estimators", "learning_rate", "max_depth")
    }, {
        "status": "selected_on_prior_season_holdout",
        "selection_rule": (
            "minimum_event_mean_p50_mae_then_coverage_error_then_interval_width"
        ),
        "selected_candidate_id": selected["candidate_id"],
        "selected_config_sha256": selected["config_sha256"],
        "candidates": candidates,
    }


def run(
    *,
    weekends_dir: Path,
    target_year: int,
    qualifying_champion_predictions: Path,
    best_lap_champion_predictions: Path,
    bootstrap_samples: int = 20_000,
    seed: int = 20260713,
) -> dict[str, Any]:
    if int(bootstrap_samples) < 1_000:
        raise ValueError("bootstrap_samples must be at least 1000")
    root = _root()
    selection_year = int(target_year) - 1
    (
        dataset,
        infos,
        inference_files,
        target_files,
        protocol_files,
    ) = _load_event_dataset(
        weekends_dir,
        years=(selection_year - 1, selection_year, int(target_year)),
        target_year=int(target_year),
    )
    partitions = _locked_partitions(tuple(infos), target_year=int(target_year))
    numeric_event = pd.to_numeric(dataset["event_key"], errors="raise").astype(int)
    selection = dataset.loc[numeric_event.isin(partitions["selection"])].copy()
    calibration = dataset.loc[numeric_event.isin(partitions["calibration"])].copy()
    audit = dataset.loc[numeric_event.isin(partitions["audit"])].copy()
    selection_fit, selection_validation, selection_holdout = _selection_holdout(selection)
    if pd.to_datetime(calibration["event_as_of"], utc=True).max() >= pd.to_datetime(
        audit["event_as_of"], utc=True
    ).min():
        raise ValueError("optional challenger calibration must strictly precede audit")

    qualifying_champion = _align_champion(
        audit,
        _load_champion_predictions(
            qualifying_champion_predictions, mode="qualifying"
        ),
        mode="qualifying",
    )
    best_lap_champion = _align_champion(
        audit,
        _load_champion_predictions(best_lap_champion_predictions, mode="best_lap"),
        mode="best_lap",
    )

    ranking_features = tuple(
        column
        for column in (*FEATURE_ALLOWLIST, "team_id", "rehearsal_source", "best_lap_compound")
        if column in dataset.columns
    )
    rank_results: dict[str, Any] = {}
    ranking_predictions: list[dict[str, Any]] = []
    for backend in RANKING_BACKENDS:
        selected_parameters, selection_evidence = _ranking_selection_evidence(
            backend=backend,
            feature_columns=ranking_features,
            selection_fit=selection_fit,
            selection_validation=selection_validation,
        )
        if selected_parameters is None:
            rank_results[backend] = {
                "status": "unavailable_selection_failed",
                "selection_evidence": selection_evidence,
                "promotion_eligible": False,
            }
            continue
        config = GroupedRankingConfig(
            feature_columns=ranking_features,
            backend=backend,
            target_column="qualy_position",
            minimum_training_events=2,
            **selected_parameters,
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
                "selection_evidence": selection_evidence,
                "promotion_eligible": False,
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
                "target_observed": np.isfinite(
                    pd.to_numeric(audit["qualy_position"], errors="coerce")
                ),
                "rehearsal_baseline_rank": pd.to_numeric(
                    audit["latest_qualifying_rehearsal_rank"], errors="coerce"
                ).to_numpy(),
                "ranking_score": predictions["ranking_score"].to_numpy(),
                "predicted_rank": predictions["predicted_rank"].to_numpy(),
                "backend": backend,
            }
        ).merge(
            qualifying_champion,
            on=["event_key", "driver_id"],
            how="left",
            validate="one_to_one",
        )
        scored["candidate_output_available"] = np.isfinite(
            pd.to_numeric(scored["predicted_rank"], errors="coerce")
        )
        scored["champion_output_available"] = np.isfinite(
            pd.to_numeric(scored["champion_rank"], errors="coerce")
        )
        metrics = _ranking_metrics(scored)
        rank_results[backend] = {
            "status": result.status,
            "manifest": result.manifest,
            "selection_evidence": selection_evidence,
            "event_metrics": metrics,
            "aggregate": _aggregate_metrics(
                metrics,
                "candidate_mae",
                "champion_mae",
                "rehearsal_baseline_mae",
                "candidate_kendall",
                "champion_kendall",
                "candidate_pole_hit",
                "champion_pole_hit",
                "candidate_top3_overlap",
                "champion_top3_overlap",
                "candidate_top10_overlap",
                "champion_top10_overlap",
                "target_observed_rate",
                "candidate_output_coverage",
                "champion_output_coverage",
            ),
            "by_weekend_stratum": _stratum_aggregates(
                metrics,
                "candidate_mae",
                "champion_mae",
                "candidate_kendall",
                "champion_kendall",
            ),
            "paired_event_diagnostics_vs_shared_champion": _paired_diagnostics(
                metrics,
                candidate_key="candidate_mae",
                champion_key="champion_mae",
                bootstrap_samples=int(bootstrap_samples),
                seed=int(seed),
            ),
            "promotion_eligible": False,
            "promotion_status": "diagnostic_requires_primary_qualifying_gate",
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
    selected_quantile_parameters, quantile_selection = _quantile_selection_evidence(
        feature_columns=quantile_features,
        selection_fit=selection_fit,
        selection_validation=selection_validation,
    )
    if selected_quantile_parameters is None:
        quantile_result: dict[str, Any] = {
            "status": "unavailable_selection_failed",
            "selection_evidence": quantile_selection,
            "interval_status": "raw_quantile_diagnostic_not_conformal",
            "promotion_eligible": False,
        }
    else:
        quantile_training = calibration.copy()
        quantile_training["lap_residual_seconds"] = (
            pd.to_numeric(quantile_training[ACTUAL_LAP_COLUMN], errors="coerce")
            - pd.to_numeric(
                quantile_training["quality_aware_anchor_seconds"], errors="coerce"
            )
        )
        audit_min = pd.to_datetime(
            audit["event_as_of"], errors="raise", utc=True
        ).min().isoformat().replace("+00:00", "Z")
        quantile_config = TabularQuantileConfig(
            backend="lightgbm",
            feature_columns=quantile_features,
            target_column="lap_residual_seconds",
            target_season=int(target_year),
            same_season_only=True,
            fit_before=audit_min,
            **selected_quantile_parameters,
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
            raw_anchor = pd.to_numeric(
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
                    "target_observed": np.isfinite(
                        pd.to_numeric(audit[ACTUAL_LAP_COLUMN], errors="coerce")
                    ),
                    "raw_anchor_lap_seconds": raw_anchor,
                    "raw_anchor_available": np.isfinite(raw_anchor),
                    "lap_p05": raw_anchor + quantiles["lap_p05"].to_numpy(),
                    "lap_p50": raw_anchor + quantiles["lap_p50"].to_numpy(),
                    "lap_p90": raw_anchor + quantiles["lap_p90"].to_numpy(),
                    "backend": "lightgbm_quantile",
                    "interval_status": "raw_lightgbm_quantile_diagnostic_not_conformal",
                }
            ).merge(
                best_lap_champion,
                on=["event_key", "driver_id"],
                how="left",
                validate="one_to_one",
            )
            quantile_scored["candidate_output_available"] = np.isfinite(
                pd.to_numeric(quantile_scored["lap_p50"], errors="coerce")
            )
            quantile_scored["champion_output_available"] = np.isfinite(
                pd.to_numeric(quantile_scored["champion_lap_p50"], errors="coerce")
            )
            quantile_scored["candidate_interval_output_available"] = np.isfinite(
                pd.to_numeric(quantile_scored["lap_p05"], errors="coerce")
            ) & np.isfinite(
                pd.to_numeric(quantile_scored["lap_p90"], errors="coerce")
            )
            quantile_scored["champion_interval_output_available"] = np.isfinite(
                pd.to_numeric(quantile_scored["champion_lap_p05"], errors="coerce")
            ) & np.isfinite(
                pd.to_numeric(quantile_scored["champion_lap_p90"], errors="coerce")
            )
            quantile_metrics = _quantile_metrics(quantile_scored)
            quantile_result = {
                "status": str(quantile_model.training_summary["status"]),
                "manifest": quantile_model.manifest(),
                "selection_evidence": quantile_selection,
                "target_semantics": (
                    "conditional_observed_official_qualifying_best_minus_"
                    "quality_aware_anchor_seconds"
                ),
                "interval_status": "raw_lightgbm_quantile_diagnostic_not_conformal",
                "interval_nominal_mass": NOMINAL_INTERVAL_COVERAGE,
                "conformal_calibration_claimed": False,
                "event_metrics": quantile_metrics,
                "aggregate": _aggregate_metrics(
                    quantile_metrics,
                    "candidate_mae_seconds",
                    "champion_mae_seconds",
                    "raw_anchor_mae_seconds",
                    "candidate_p05_pinball_loss",
                    "candidate_p50_pinball_loss",
                    "candidate_p90_pinball_loss",
                    "champion_p05_pinball_loss",
                    "champion_p50_pinball_loss",
                    "champion_p90_pinball_loss",
                    "candidate_p05_p90_coverage",
                    "champion_p05_p90_coverage",
                    "candidate_p05_p90_mean_width_seconds",
                    "champion_p05_p90_mean_width_seconds",
                    "target_observed_rate",
                    "candidate_output_coverage",
                    "champion_output_coverage",
                ),
                "by_weekend_stratum": _stratum_aggregates(
                    quantile_metrics,
                    "candidate_mae_seconds",
                    "champion_mae_seconds",
                    "candidate_p05_p90_coverage",
                    "champion_p05_p90_coverage",
                ),
                "paired_event_diagnostics_vs_shared_champion": _paired_diagnostics(
                    quantile_metrics,
                    candidate_key="candidate_mae_seconds",
                    champion_key="champion_mae_seconds",
                    bootstrap_samples=int(bootstrap_samples),
                    seed=int(seed) + 1,
                ),
                "predictions": quantile_scored.to_dict(orient="records"),
                "promotion_eligible": False,
                "promotion_status": (
                    "raw_quantile_diagnostic_requires_disjoint_conformal_"
                    "calibration_and_primary_best_lap_gate"
                ),
            }
        except TabularQuantileBackendUnavailable as exc:
            quantile_result = {
                **exc.to_payload(),
                "selection_evidence": quantile_selection,
                "interval_status": "unavailable_raw_quantile_not_conformal",
                "promotion_eligible": False,
            }

    implementation_files = (
        Path(__file__).resolve(),
        root
        / (
            "research/projects/F1/rising_qualification_prediction/Python/"
            "run_qualifying_pairwise_challenger_backtest.py"
        ),
        root / "packages/f1/features/qualifying_lap.py",
        root / "packages/f1/models/grouped_ranking.py",
        root / "packages/f1/models/ultimate_lap_time/achievable.py",
        root / "packages/f1/models/ultimate_lap_time/tabular_quantile.py",
        root / "packages/f1/orchestration/model_runtime.py",
        root / "packages/f1/orchestration/non_live_validation.py",
    )
    inference_manifest = _file_manifest(inference_files, root=root)
    target_manifest = _file_manifest(target_files, root=root)
    protocol_input_manifest = _file_manifest(protocol_files, root=root)
    champion_manifest = _file_manifest(
        (qualifying_champion_predictions, best_lap_champion_predictions), root=root
    )
    implementation_manifest = _file_manifest(implementation_files, root=root)
    configuration_payload = {
        "ranking_candidate_grid": list(RANKING_CONFIG_CANDIDATES),
        "quantile_candidate_grid": list(QUANTILE_CONFIG_CANDIDATES),
        "ranking_feature_allowlist": list(ranking_features),
        "quantile_feature_allowlist": list(quantile_features),
        "ranking_selected_config_sha256": {
            backend: value.get("selection_evidence", {}).get(
                "selected_config_sha256"
            )
            for backend, value in rank_results.items()
        },
        "quantile_selected_config_sha256": quantile_selection.get(
            "selected_config_sha256"
        ),
    }
    protocol_payload = {
        "event_partitions": {
            name: list(values) for name, values in partitions.items()
        },
        "partition_validation_issues": [],
        "selection_holdout": selection_holdout,
        "absolute_pace_fit": "target_season_calibration_events_only",
        "prior_season_role": "bounded_early_fit_late_validation_config_selection",
        "event_time_semantics": {
            "target_season": "scheduled_qualifying_start_utc_required_fail_closed",
            "all_prior_seasons": "season_round_ordinal_selection_only",
            "per_event": {
                str(key): value["event_time_semantics"]
                for key, value in sorted(infos.items())
            },
        },
        "audit_outcomes_used_for_fit": False,
        "audit_target_files_separated_from_inference_manifest": True,
        "champion_comparison_required": True,
        "champion_rows_must_exactly_match_audit_inference_rows": True,
        "quantile_interval_semantics": (
            "raw_p05_p90_85pct_model_interval_diagnostic_not_conformal"
        ),
        "training_threads_per_backend": 1,
        "training_queue": "sequential",
        "bootstrap_samples": int(bootstrap_samples),
        "seed": int(seed),
    }
    return _json_safe(
        {
            "schema_version": "f1_optional_non_live_event_block_evidence_v2",
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "target_year": int(target_year),
            "decision": {
                "production_profile_changed": False,
                "promotion_eligible": False,
                "promotion_status": "fail_closed_diagnostic_evidence_only",
                "reasons": [
                    "optional_models_require_primary_mode_canonical_promotion_gates",
                    "raw_quantile_intervals_are_not_conformal",
                    "artifact_generation_never_changes_a_production_profile",
                ],
            },
            "protocol": protocol_payload,
            "protocol_sha256": _canonical_sha256(protocol_payload),
            "configuration": configuration_payload,
            "configuration_sha256": _canonical_sha256(configuration_payload),
            "runtime": f1_model_runtime_doctor(),
            "ranking": rank_results,
            "ranking_predictions": ranking_predictions,
            "lightgbm_quantile": quantile_result,
            "inference_input_manifest": inference_manifest,
            "inference_input_manifest_sha256": _canonical_sha256(inference_manifest),
            "target_evaluation_manifest": target_manifest,
            "target_evaluation_manifest_sha256": _canonical_sha256(target_manifest),
            "protocol_input_manifest": protocol_input_manifest,
            "protocol_input_manifest_sha256": _canonical_sha256(
                protocol_input_manifest
            ),
            "champion_prediction_manifest": champion_manifest,
            "champion_prediction_manifest_sha256": _canonical_sha256(
                champion_manifest
            ),
            "implementation_manifest": implementation_manifest,
            "implementation_manifest_sha256": _canonical_sha256(
                implementation_manifest
            ),
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
        "--qualifying-champion-predictions",
        type=Path,
        required=True,
        help="frozen shared-Qualifying artifact; audit event-driver rows must match exactly",
    )
    parser.add_argument(
        "--best-lap-champion-predictions",
        type=Path,
        required=True,
        help="frozen shared Best-Lap artifact; audit event-driver rows must match exactly",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--output",
        type=Path,
        default=_root()
        / "artifacts/backtests/f1/optional_models/2026_event_block_challengers_v3.json",
    )
    args = parser.parse_args()
    payload = run(
        weekends_dir=args.weekends_dir.expanduser().resolve(),
        target_year=int(args.target_year),
        qualifying_champion_predictions=args.qualifying_champion_predictions.expanduser().resolve(),
        best_lap_champion_predictions=args.best_lap_champion_predictions.expanduser().resolve(),
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed),
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
