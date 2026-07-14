#!/usr/bin/env python3
"""Walk-forward survival-aware Race Final Position challenger backtest."""

from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from packages.f1.data.providers import LocalWeekendProvider
from packages.f1.domain.starting_grid import RacePredictionHorizon
from packages.f1.models.pre_race.evaluate import evaluate_terminal_status_probabilities
from packages.f1.models.pre_race.joint import SurvivalAwareRaceModel
from packages.f1.models.pre_race.ranking import BradleyTerryOrderRanker, ConditionalOrderConfig
from packages.f1.models.pre_race.status import (
    TerminalStatus,
    reason_code_terminal_status,
    terminal_label_granularity,
)
from packages.f1.models.pre_race.survival import (
    BinaryTerminalCalibrator,
    PartialPooledTerminalHazard,
    TerminalHazardConfig,
)
from packages.f1.orchestration.model_runtime import f1_model_runtime_doctor
from packages.f1.orchestration.non_live_validation import (
    EventError,
    evaluate_race_promotion,
    validate_event_partitions,
)


_NO_SAME_HORIZON_ORDER_RESIDUAL_WEIGHT = 0.0
_SAME_SEASON_MINIMUM_PRIOR_EVENTS = 2
_LEGACY_CROSS_SEASON_MINIMUM_PRIOR_EVENTS = 4
_MINIMUM_RELATIVE_SELECTION_GAIN = 0.05
_MINIMUM_INDEPENDENT_LOCK_EVENTS = 4
_MINIMUM_INDEPENDENT_AUDIT_EVENTS = 3
RACE_BACKTEST_SCHEMA_VERSION = "f1_race_survival_order_event_block_v8"


def _root() -> Path:
    return Path(__file__).resolve().parents[5]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _same_season_event_partitions(
    event_keys: Sequence[int],
    *,
    target_year: int,
) -> dict[str, list[str]]:
    """Lock one season into chronological, evidence-adaptive event blocks.

    Nine available events preserve the original 2/2/2/3 diagnostic layout.
    As evidence arrives, selection expands first and calibration second.  At
    thirteen events the protocol reaches the formal 2/4/4/3 minimum, after
    which every additional event remains untouched audit evidence.

    Formal counts are still checked separately for each information horizon;
    pooling post-grid and provisional-grid weekends cannot clear those gates.
    """

    ordered = tuple(sorted({int(value) for value in event_keys}))
    foreign = [value for value in ordered if value // 100 != int(target_year)]
    if foreign:
        raise ValueError(
            "same-season Race partitions received events outside target year "
            f"{target_year}: {foreign}"
        )
    if len(ordered) < 8:
        raise ValueError(
            "same-season Race partitions require at least eight complete events "
            "(2 development, 2 selection, 2 calibration, and 2 audit)"
        )
    if len(ordered) < 9:
        selection_count = 2
        calibration_count = 2
    else:
        # Reserve two development and at least three untouched audit events.
        lock_capacity = len(ordered) - 5
        selection_count = min(4, max(2, (lock_capacity + 1) // 2))
        calibration_count = min(4, lock_capacity - selection_count)
    selection_start = 2
    calibration_start = selection_start + selection_count
    audit_start = calibration_start + calibration_count
    return {
        "development": [str(value) for value in ordered[:selection_start]],
        "selection": [
            str(value) for value in ordered[selection_start:calibration_start]
        ],
        "calibration": [
            str(value) for value in ordered[calibration_start:audit_start]
        ],
        "audit": [str(value) for value in ordered[audit_start:]],
    }


def _event_is_in_partition(
    event_key: int,
    partition_events: Mapping[str, Sequence[str]],
    partition: str,
) -> bool:
    if partition not in partition_events:
        raise ValueError(f"unknown Race event partition: {partition}")
    return str(int(event_key)) in {str(value) for value in partition_events[partition]}


def _strict_prior_history(
    history: pd.DataFrame,
    *,
    event_key: int,
    same_season_only: bool,
) -> pd.DataFrame:
    """Return only causally prior rows, optionally restricted to target season."""

    if history.empty:
        return history.copy()
    if "event_key" not in history.columns:
        raise ValueError("Race history is missing event_key")
    numeric_events = pd.to_numeric(history["event_key"], errors="coerce")
    if numeric_events.isna().any():
        raise ValueError("Race history event keys must be finite")
    mask = numeric_events.lt(int(event_key))
    if same_season_only:
        mask &= numeric_events.floordiv(100).eq(int(event_key) // 100)
    return history.loc[mask].copy()


def _metadata(weekends_dir: Path, year: int, round_number: int) -> tuple[dict[str, Any], Path]:
    matches = sorted((weekends_dir / str(year)).glob(f"round_{round_number:02d}_*"))
    if not matches:
        raise FileNotFoundError(f"missing local weekend {year} round {round_number}")
    path = matches[0] / "weekend_metadata.json"
    return json.loads(path.read_text(encoding="utf-8")), path


def _event_as_of(metadata: dict[str, Any]) -> str:
    raw = metadata.get("scheduled_event_date")
    parsed = pd.to_datetime(raw, errors="coerce", utc=True)
    if pd.isna(parsed):
        year = int(metadata["year"])
        round_number = int(metadata["round_number"])
        parsed = pd.Timestamp(year=year, month=1, day=1, tz="UTC") + timedelta(
            days=round_number * 7
        )
    return parsed.isoformat().replace("+00:00", "Z")


def _normalized_circuit(metadata: dict[str, Any]) -> str:
    text = str(metadata.get("event_name") or "unknown").lower()
    return "_".join(part for part in "".join(char if char.isalnum() else " " for char in text).split() if part)


def _session_snapshot_provenance(
    metadata: dict[str, Any],
    session_type: str,
    *,
    prediction_as_of: str,
) -> dict[str, Any]:
    """Describe capture time separately from the session's logical horizon."""

    normalized = str(session_type).strip().lower()
    entry = next(
        (
            session
            for session in metadata.get("sessions", [])
            if str(session.get("session_type", "")).strip().lower() == normalized
        ),
        {},
    )
    first_published_at = entry.get("first_published_at")
    captured_at = entry.get("available_at")
    published_timestamp = pd.to_datetime(
        first_published_at, errors="coerce", utc=True
    )
    logical_cutoff = pd.to_datetime(
        prediction_as_of, errors="coerce", utc=True
    )
    first_seen_verified = bool(
        first_published_at
        and pd.notna(published_timestamp)
        and pd.notna(logical_cutoff)
        and published_timestamp <= logical_cutoff
    )
    if first_seen_verified:
        semantics = "first_published_provider_snapshot_verified_pre_cutoff"
    elif first_published_at:
        semantics = "reported_first_publication_not_verified_pre_cutoff"
    elif captured_at:
        semantics = "retrospective_provider_post_session_snapshot_capture_time"
    else:
        semantics = "legacy_retrospective_provider_snapshot_capture_time_unavailable"
    payload: dict[str, Any] = {
        "session_type": normalized,
        "captured_at": str(captured_at) if captured_at else None,
        "first_published_at": (
            str(first_published_at) if first_published_at else None
        ),
        "first_seen_verified": first_seen_verified,
        "time_semantics": semantics,
        "logical_information_horizon": "completed_session_classification",
        "verification_cutoff": str(prediction_as_of),
    }
    return payload


def _rolling_group_mean(
    history: pd.DataFrame,
    *,
    key_column: str,
    value_column: str,
    prior_strength: float,
) -> dict[str, float]:
    if history.empty or key_column not in history.columns or value_column not in history.columns:
        return {}
    values = pd.to_numeric(history[value_column], errors="coerce")
    global_mean = float(values.mean()) if values.notna().any() else 0.0
    output: dict[str, float] = {}
    keys = history[key_column].astype("string").str.strip()
    valid_keys = keys.notna() & keys.ne("")
    for key, indexes in history.loc[valid_keys].groupby(keys.loc[valid_keys]).groups.items():
        group = values.loc[indexes].dropna()
        if group.empty:
            continue
        output[str(key)] = float(
            (group.sum() + prior_strength * global_mean) / (len(group) + prior_strength)
        )
    return output


def _rolling_rate(
    history: pd.DataFrame,
    *,
    key_column: str,
    mask: pd.Series,
    alpha: float = 1.0,
    beta: float = 4.0,
) -> dict[str, float]:
    if history.empty or key_column not in history.columns:
        return {}
    labels = mask.reindex(history.index).fillna(False).astype(float)
    output: dict[str, float] = {}
    keys = history[key_column].astype("string").str.strip()
    valid_keys = keys.notna() & keys.ne("")
    for key, indexes in history.loc[valid_keys].groupby(keys.loc[valid_keys]).groups.items():
        output[str(key)] = float((labels.loc[indexes].sum() + alpha) / (len(indexes) + alpha + beta))
    return output


def _resolve_session_reference(
    root: Path,
    weekend_dir: Path,
    reference: object,
) -> Path | None:
    """Resolve metadata paths with the same legacy fallback as the provider."""

    text = str(reference or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    candidates = (
        (path,)
        if path.is_absolute()
        else (root / path, weekend_dir / path.name, weekend_dir / path)
    )
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _pre_race_red_flag_count(
    root: Path,
    metadata: dict[str, Any],
    *,
    weekend_dir: Path,
) -> int:
    qualifying_order = min(
        (
            int(session.get("session_order", 999))
            for session in metadata.get("sessions", [])
            if str(session.get("session_type", "")).lower() == "qualifying"
        ),
        default=999,
    )
    messages: list[str] = []
    for session in metadata.get("sessions", []):
        if int(session.get("session_order", 999)) > qualifying_order:
            continue
        reference = session.get("race_control_messages_path")
        if not reference:
            continue
        path = _resolve_session_reference(root, weekend_dir, reference)
        if path is None:
            continue
        frame = pd.read_csv(path)
        for column in ("Flag", "Status", "Message"):
            if column in frame.columns:
                messages.extend(frame[column].fillna("").astype(str).str.lower().tolist())
    return int(sum("red" in value and "flag" in value for value in set(messages)))


def _pre_race_wet_evidence(
    root: Path,
    metadata: dict[str, Any],
    *,
    weekend_dir: Path,
) -> float:
    qualifying_order = min(
        (
            int(session.get("session_order", 999))
            for session in metadata.get("sessions", [])
            if str(session.get("session_type", "")).lower() == "qualifying"
        ),
        default=999,
    )
    evidence: list[float] = []
    for session in metadata.get("sessions", []):
        if int(session.get("session_order", 999)) > qualifying_order:
            continue
        reference = session.get("weather_path")
        if not reference:
            continue
        path = _resolve_session_reference(root, weekend_dir, reference)
        if path is None:
            continue
        frame = pd.read_csv(path)
        if "Rainfall" in frame.columns:
            rainfall = pd.to_numeric(frame["Rainfall"], errors="coerce")
            if rainfall.notna().any():
                evidence.append(float(rainfall.fillna(0.0).gt(0.0).mean()))
    return float(max(evidence, default=0.0))


def _pre_race_driver_mechanical_stop_share(
    root: Path,
    metadata: dict[str, Any],
    *,
    weekend_dir: Path,
) -> dict[str, float]:
    counts: dict[str, int] = {}
    mechanical: dict[str, int] = {}
    for session in metadata.get("sessions", []):
        if str(session.get("session_type", "")).strip().lower() == "race":
            continue
        reference = session.get("results_path")
        if not reference:
            continue
        path = _resolve_session_reference(root, weekend_dir, reference)
        if path is None:
            continue
        frame = pd.read_csv(path)
        driver_col = next(
            (
                column
                for column in (
                    "DriverNumber",
                    "driver_number",
                    "driver_id",
                    "Abbreviation",
                )
                if column in frame.columns
            ),
            None,
        )
        status_col = next(
            (
                column
                for column in ("Status", "status", "ResultStatus")
                if column in frame.columns
            ),
            None,
        )
        if driver_col is None or status_col is None:
            continue
        for _, row in frame.iterrows():
            driver = str(row.get(driver_col, "")).strip()
            numeric = pd.to_numeric(pd.Series([driver]), errors="coerce").iloc[0]
            if pd.notna(numeric) and float(numeric).is_integer():
                driver = str(int(numeric))
            if not driver or driver.lower() in {"nan", "none", "null"}:
                continue
            counts[driver] = counts.get(driver, 0) + 1
            if reason_code_terminal_status(row.get(status_col)) is TerminalStatus.MECHANICAL_POWER_UNIT:
                mechanical[driver] = mechanical.get(driver, 0) + 1
    return {
        driver: float(mechanical.get(driver, 0) / count)
        for driver, count in counts.items()
        if count > 0
    }


def _legal_grid_baseline(frame: pd.DataFrame) -> pd.Series:
    status = frame["grid_status"].astype(str).str.lower()
    physical = pd.to_numeric(frame["grid_position"], errors="coerce")
    group = np.where(
        status.eq("grid"),
        0,
        np.where(status.eq("pit_lane"), 1, 2),
    )
    order = pd.DataFrame(
        {
            "group": group,
            "physical": physical.fillna(np.inf),
            "provider_order": np.arange(len(frame)),
        },
        index=frame.index,
    ).sort_values(["group", "physical", "provider_order"], kind="mergesort")
    return pd.Series(
        np.arange(1, len(order) + 1, dtype=float), index=order.index
    ).reindex(frame.index)


def _causal_rolling_terminal_probability(
    prior: pd.DataFrame,
    current: pd.DataFrame,
) -> np.ndarray:
    """Beta-smoothed binary hazard baseline with no current-weekend covariates."""

    terminal = prior["terminal_status"].astype(str).ne(
        TerminalStatus.CLASSIFIED_FINISH.value
    ).astype(float)
    global_probability = float((terminal.sum() + 2.0) / (len(terminal) + 10.0))
    output = np.full(len(current), global_probability, dtype=float)
    for output_index, (_, row) in enumerate(current.iterrows()):
        logits: list[tuple[float, float]] = [(global_probability, 1.0)]
        for column, strength, weight in (
            ("team_name", 8.0, 0.45),
            ("power_unit", 10.0, 0.30),
            ("driver_id", 10.0, 0.25),
        ):
            if column not in prior.columns or column not in row.index or pd.isna(row[column]):
                continue
            mask = prior[column].astype(str).eq(str(row[column]))
            support = int(mask.sum())
            if support == 0:
                continue
            probability = float(
                (terminal.loc[mask].sum() + strength * global_probability)
                / (support + strength)
            )
            reliability = support / (support + strength)
            logits.append((probability, weight * reliability))
        base_logit = np.log(global_probability / max(1e-9, 1.0 - global_probability))
        blended_logit = base_logit
        for probability, weight in logits[1:]:
            value = np.clip(probability, 1e-6, 1.0 - 1e-6)
            blended_logit += weight * (
                np.log(value / (1.0 - value)) - base_logit
            )
        output[output_index] = 1.0 / (1.0 + np.exp(-blended_logit))
    return output


def _team_column(frame: pd.DataFrame) -> str | None:
    return next((column for column in ("team_name", "fp_team_name", "fp1_team_name") if column in frame.columns), None)


_QUALIFYING_PRIOR_FEATURES: tuple[str, ...] = (
    "fp_quali_sim_rank",
    "fp_mean_rank",
    "fp1_quali_sim_rank",
    "fp2_quali_sim_rank",
    "fp3_quali_sim_rank",
    "fp_quali_sim_delta",
    "fp_mean_delta",
    "fp_quali_sim_evidence_share",
    "fp_lap_quality_ratio",
)


def _deterministic_rank(values: np.ndarray) -> np.ndarray:
    """Rank scores with causal provider-row order as the only tie-break."""

    numeric = np.asarray(values, dtype=float)
    finite = np.isfinite(numeric)
    replacement = float(np.nanmedian(numeric[finite])) if finite.any() else 0.0
    order = np.lexsort(
        (
            np.arange(len(numeric)),
            np.where(finite, numeric, replacement),
        )
    )
    ranks = np.empty(len(numeric), dtype=float)
    ranks[order] = np.arange(1, len(numeric) + 1, dtype=float)
    return ranks


def _rolling_oof_qualifying_prior(
    history: pd.DataFrame,
    current: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict Qualifying rank from completed FP evidence using only prior events."""

    feature_columns = [
        column
        for column in _QUALIFYING_PRIOR_FEATURES
        if column in history.columns
        and column in current.columns
        and pd.to_numeric(history[column], errors="coerce").notna().any()
    ]
    training = (
        history.loc[
            pd.to_numeric(history["qualy_position"], errors="coerce").notna()
        ].copy()
        if not history.empty and "qualy_position" in history.columns
        else pd.DataFrame(columns=history.columns)
    )
    training_events = int(training["event_key"].nunique()) if not training.empty else 0
    if feature_columns and training_events >= 4:
        train_numeric = training[feature_columns].apply(pd.to_numeric, errors="coerce")
        current_numeric = current[feature_columns].apply(pd.to_numeric, errors="coerce")
        medians = train_numeric.median(axis=0, skipna=True).fillna(0.0)
        scales = train_numeric.std(axis=0, skipna=True).replace(0.0, np.nan).fillna(1.0)
        train_missing = train_numeric.isna().astype(float)
        current_missing = current_numeric.isna().astype(float)
        train_x = np.column_stack(
            [
                ((train_numeric.fillna(medians) - medians) / scales).to_numpy(dtype=float),
                train_missing.to_numpy(dtype=float),
            ]
        )
        current_x = np.column_stack(
            [
                ((current_numeric.fillna(medians) - medians) / scales).to_numpy(dtype=float),
                current_missing.to_numpy(dtype=float),
            ]
        )
        event_sizes = training.groupby("event_key", dropna=False)["driver_id"].transform(
            "size"
        )
        target = (
            pd.to_numeric(training["qualy_position"], errors="coerce") - 1.0
        ) / (pd.to_numeric(event_sizes, errors="coerce").clip(lower=2.0) - 1.0)
        try:
            from sklearn.linear_model import Ridge

            estimator = Ridge(alpha=10.0, fit_intercept=True)
            estimator.fit(train_x, target.to_numpy(dtype=float))
            score = estimator.predict(current_x)
            return _deterministic_rank(score), {
                "source": "rolling_out_of_event_ridge_from_pre_qualifying_practice",
                "training_events": training_events,
                "features": feature_columns,
                "regularization_alpha": 10.0,
                "strictly_prior_event_keys": True,
            }
        except Exception:
            pass

    fallback_column = next(
        (
            column
            for column in ("fp_quali_sim_rank", "fp_mean_rank")
            if column in current.columns
            and pd.to_numeric(current[column], errors="coerce").notna().any()
        ),
        None,
    )
    fallback_values = (
        pd.to_numeric(current[fallback_column], errors="coerce").to_numpy(dtype=float)
        if fallback_column is not None
        else np.zeros(len(current), dtype=float)
    )
    return _deterministic_rank(fallback_values), {
        "source": (
            f"causal_practice_rank_fallback:{fallback_column}"
            if fallback_column is not None
            else "provider_order_fallback_no_pre_qualifying_rank_evidence"
        ),
        "training_events": training_events,
        "features": [fallback_column] if fallback_column is not None else [],
        "regularization_alpha": None,
        "strictly_prior_event_keys": True,
    }


def _normalize_identity_token(value: object) -> str:
    """Normalize provider identity aliases without changing text identities."""

    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.upper() in {"NAN", "NONE", "NULL", "<NA>"}:
        return ""
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(numeric) and np.isfinite(float(numeric)) and float(numeric).is_integer():
        return str(int(numeric))
    return text


def _align_grid_driver_ids_from_qualifying(
    grid: pd.DataFrame,
    qualifying_roster: pd.DataFrame,
) -> pd.DataFrame:
    """Align FIA car numbers to model IDs using only pre-race Qualifying identity."""

    if "driver_id" not in grid.columns or "driver_id" not in qualifying_roster.columns:
        raise ValueError("grid alignment requires driver identity on both sources")
    out = grid.copy()
    out["driver_id"] = out["driver_id"].map(_normalize_identity_token)
    target = qualifying_roster["driver_id"].map(_normalize_identity_token)
    if out["driver_id"].eq("").any() or target.eq("").any():
        raise ValueError("grid alignment contains incomplete driver identity")
    if out["driver_id"].duplicated().any():
        raise ValueError("final grid contains duplicate provider driver identities")
    if target.duplicated().any():
        raise ValueError("Qualifying roster contains duplicate canonical driver identities")
    target_ids = set(target)
    if len(out) == len(target) and set(out["driver_id"]) == target_ids:
        out["grid_provider_driver_id"] = out["driver_id"]
        out["grid_identity_mapping_source"] = (
            "provider_driver_id_matches_pre_race_qualifying_canonical_id"
        )
        return out
    if "grid_car_number" not in out.columns or "car_number" not in qualifying_roster.columns:
        raise ValueError(
            "first-seen final grid identity differs from Qualifying and lacks car-number evidence"
        )
    qualifying_identity = qualifying_roster[["driver_id", "car_number"]].copy()
    qualifying_identity["driver_id"] = qualifying_identity["driver_id"].map(
        _normalize_identity_token
    )
    qualifying_identity["car_number"] = qualifying_identity["car_number"].map(
        _normalize_identity_token
    )
    if qualifying_identity["car_number"].eq("").any():
        raise ValueError("Qualifying roster contains incomplete car-number identity")
    if qualifying_identity["car_number"].duplicated().any():
        raise ValueError("Qualifying roster contains duplicate car-number identity")
    mapping = dict(
        zip(qualifying_identity["car_number"], qualifying_identity["driver_id"])
    )
    grid_car_numbers = out["grid_car_number"].map(_normalize_identity_token)
    aligned = grid_car_numbers.map(mapping)
    if (
        aligned.isna().any()
        or aligned.duplicated().any()
        or len(aligned) != len(target)
        or set(aligned.astype(str)) != target_ids
    ):
        raise ValueError(
            "FIA final-grid car numbers do not map one-to-one to the pre-race Qualifying roster"
        )
    out["grid_provider_driver_id"] = out["driver_id"]
    out["driver_id"] = aligned.astype(str)
    out["grid_identity_mapping_source"] = "pre_race_qualifying_car_number"
    return out


def _canonicalize_qualifying_driver_identity(
    qualifying: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Freeze one stable driver identity from the causal Qualifying roster.

    FastF1's primary ``driver_id`` is normally the car number.  Car numbers are
    not longitudinal driver identities (for example, a reigning champion may
    switch between number 1 and their permanent number).  The FIA three-letter
    abbreviation is present in the immutable Qualifying classification and is
    stable across the seasons covered by this walk-forward study.

    The returned lookup accepts the provider ID, car number, or canonical
    abbreviation so every other same-weekend source can be aligned without
    opening Race truth for identity repair.
    """

    required = {"driver_id", "car_number", "driver_abbreviation"}
    missing = sorted(required - set(qualifying.columns))
    if missing:
        raise ValueError(
            "causal Qualifying roster lacks stable driver identity fields: "
            f"{missing}"
        )
    out = qualifying.copy()
    provider_ids = out["driver_id"].map(_normalize_identity_token)
    car_numbers = out["car_number"].map(_normalize_identity_token)
    abbreviations = (
        out["driver_abbreviation"]
        .map(_normalize_identity_token)
        .astype(str)
        .str.upper()
    )
    invalid = provider_ids.eq("") | car_numbers.eq("") | abbreviations.eq("")
    if invalid.any():
        raise ValueError("causal Qualifying roster contains incomplete driver identity")
    if provider_ids.duplicated().any():
        raise ValueError("causal Qualifying roster contains duplicate provider identities")
    if car_numbers.duplicated().any():
        raise ValueError("causal Qualifying roster contains duplicate car numbers")
    if abbreviations.duplicated().any():
        raise ValueError("causal Qualifying roster contains duplicate stable driver identities")

    out["provider_driver_id"] = provider_ids.astype(str)
    out["car_number"] = car_numbers.astype(str)
    out["driver_abbreviation"] = abbreviations.astype(str)
    out["driver_id"] = abbreviations.astype(str)
    out["driver_identity_source"] = (
        "retrospective_provider_qualifying_fia_driver_abbreviation"
    )

    lookup: dict[str, str] = {}
    for row in out[
        ["driver_id", "provider_driver_id", "car_number"]
    ].itertuples(index=False):
        canonical = str(row.driver_id)
        for raw in (row.driver_id, row.provider_driver_id, row.car_number):
            key = _normalize_identity_token(raw)
            existing = lookup.get(key)
            if existing is not None and existing != canonical:
                raise ValueError(
                    "Qualifying identity aliases do not map one-to-one to drivers"
                )
            lookup[key] = canonical
    return out, lookup


def _map_driver_ids_from_qualifying(
    frame: pd.DataFrame,
    identity_lookup: dict[str, str],
    *,
    source_name: str,
    allow_non_roster_rows: bool,
) -> pd.DataFrame:
    """Map a same-weekend provider frame through the frozen Qualifying roster."""

    if frame.empty:
        return frame.copy()
    if "driver_id" not in frame.columns:
        raise ValueError(f"{source_name} lacks driver identity")
    out = frame.copy()
    provider_ids = out["driver_id"].map(_normalize_identity_token)
    canonical = provider_ids.map(identity_lookup)
    unmapped = canonical.isna()
    if unmapped.any() and not allow_non_roster_rows:
        unresolved = sorted(provider_ids.loc[unmapped].dropna().astype(str).unique())
        raise ValueError(
            f"{source_name} contains entrants absent from the causal pre-race roster: "
            f"{unresolved}"
        )
    out[f"{source_name}_provider_driver_id"] = provider_ids.astype(str)
    out = out.loc[~unmapped].copy()
    out["driver_id"] = canonical.loc[~unmapped].astype(str)
    if out["driver_id"].duplicated().any():
        raise ValueError(
            f"{source_name} maps multiple provider rows to one stable driver identity"
        )
    return out


def _resolve_missing_race_targets(
    frame: pd.DataFrame,
    *,
    final_capture: bool,
) -> pd.DataFrame:
    """Complete only officially proven nonstarters; leave all other gaps open."""

    out = frame.copy()
    observed = out["race_target_observed"].eq(True)
    unmatched = ~observed
    authoritative_nonstarter = pd.Series(False, index=out.index, dtype=bool)
    if final_capture:
        authoritative_nonstarter = (
            unmatched
            & out["grid_evidence_complete"].eq(True)
            & out["grid_publication_pre_race_verified"].eq(True)
            & pd.to_numeric(out["grid_starter_eligible"], errors="coerce")
            .fillna(1.0)
            .le(0.0)
        )
    if authoritative_nonstarter.any():
        out.loc[authoritative_nonstarter, "race_status_raw"] = "Did not start"
        out.loc[
            authoritative_nonstarter, "race_status_evidence_complete"
        ] = True
        out.loc[
            authoritative_nonstarter, "race_target_source"
        ] = "official_first_seen_grid_nonstarter"
        out.loc[authoritative_nonstarter, "race_target_observed"] = True
        used = {
            int(value)
            for value in pd.to_numeric(
                out["finish_position"], errors="coerce"
            ).dropna()
        }
        next_position = 1
        ordered_nonstarters = out.loc[authoritative_nonstarter].sort_values(
            ["grid_baseline_position", "driver_id"], kind="mergesort"
        )
        for index in ordered_nonstarters.index:
            while next_position in used:
                next_position += 1
            out.loc[index, "finish_position"] = next_position
            out.loc[index, "retirement_fraction"] = 0.0
            out.loc[index, "laps_completed"] = 0.0
            used.add(next_position)
    unproven_missing = unmatched & ~authoritative_nonstarter
    out.loc[unproven_missing, "race_target_observed"] = False
    out.loc[unproven_missing, "race_status_evidence_complete"] = False
    out.loc[
        unproven_missing, "race_target_source"
    ] = "unavailable_without_authoritative_nonstarter_evidence"
    return out


def _stable_provisional_grid_positions(frame: pd.DataFrame) -> pd.Series:
    """Turn a provider Qualifying classification into one physical proxy grid.

    Some archived FastF1 classifications contain tied numeric ``Position``
    values even though their source-row order is a total official order. A
    pre-grid forecast may use that order as a provisional grid, but it may not
    pass duplicate physical slots into the simulator.
    """

    positions = pd.to_numeric(frame.get("qualy_position"), errors="coerce")
    if positions.isna().any() or positions.le(0.0).any():
        raise ValueError("provisional Qualifying grid contains unresolved positions")
    order = pd.DataFrame(
        {
            "position": positions.to_numpy(dtype=float),
            "source_row": np.arange(len(frame), dtype=int),
        },
        index=frame.index,
    ).sort_values(["position", "source_row"], kind="mergesort")
    ranks = pd.Series(np.arange(1, len(order) + 1, dtype=float), index=order.index)
    return ranks.reindex(frame.index)


def _build_event_rows(
    *,
    root: Path,
    provider: LocalWeekendProvider,
    weekends_dir: Path,
    year: int,
    round_number: int,
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], list[Path]]:
    metadata, metadata_path = _metadata(weekends_dir, year, round_number)
    qualifying = provider.get_qualifying_results(year, round_number)
    practice = provider.get_fp_features(year, round_number, prediction_target="race")
    if qualifying.empty:
        raise ValueError(f"{year} round {round_number} has no causal Qualifying roster")

    q = qualifying[
        [
            column
            for column in (
                "driver_id",
                "position",
                "team_name",
                "power_unit",
                "power_unit_source",
                "car_number",
                "driver_abbreviation",
            )
            if column in qualifying
        ]
    ].copy()
    q = q.rename(
        columns={
            "position": "qualy_position",
            "power_unit_source": "qualy_power_unit_source",
        }
    )
    q, identity_lookup = _canonicalize_qualifying_driver_identity(q)
    frame = q.copy()
    if not practice.empty:
        practice = _map_driver_ids_from_qualifying(
            practice,
            identity_lookup,
            source_name="practice",
            allow_non_roster_rows=True,
        )
        # Qualifying owns entrant identity; practice may expose convenience
        # copies of team/name fields which must not suffix or replace it.
        overlapping_identity = sorted(
            (set(frame.columns) & set(practice.columns)) - {"driver_id"}
        )
        practice_features = practice.drop(
            columns=overlapping_identity,
            errors="ignore",
        )
        frame = frame.merge(
            practice_features,
            on="driver_id",
            how="left",
            validate="one_to_one",
        )
    frame["team_name"] = frame.get(
        "team_name", pd.Series(index=frame.index, dtype=object)
    ).fillna("unknown_team").astype(str)
    frame["power_unit"] = frame.get(
        "power_unit", pd.Series(index=frame.index, dtype=object)
    )
    frame["power_unit_source"] = frame.get(
        "qualy_power_unit_source", pd.Series(index=frame.index, dtype=object)
    )
    frame["causal_roster_source"] = (
        "retrospective_provider_post_qualifying_classification"
    )
    event_key = int(year) * 100 + int(round_number)
    frame["event_key"] = event_key
    frame["event_as_of"] = _event_as_of(metadata)
    frame["circuit_id"] = _normalized_circuit(metadata)
    starting_grid = provider.get_starting_grid(year, round_number)
    required_capture_columns = {
        "driver_id",
        "grid_position",
        "grid_status",
        "grid_starter_eligible",
        "grid_pit_lane_start",
        "grid_source",
        "grid_capture_id",
        "grid_evidence_complete",
        "grid_snapshot_available",
        "grid_publication_pre_race_verified",
        "grid_first_published_at",
        "grid_resolution_status",
        "race_information_horizon",
    }
    final_capture = bool(
        not starting_grid.empty
        and required_capture_columns.issubset(starting_grid.columns)
        and starting_grid["grid_source"].astype(str).eq("first_seen_official_grid").all()
        and starting_grid["grid_snapshot_available"].fillna(False).astype(bool).all()
        and starting_grid["grid_evidence_complete"].fillna(False).astype(bool).all()
        and starting_grid["grid_publication_pre_race_verified"].fillna(False).astype(bool).all()
        and starting_grid["grid_resolution_status"].astype(str).eq("resolved").all()
        and starting_grid["race_information_horizon"].astype(str).eq(
            RacePredictionHorizon.POST_GRID_PRE_RACE.value
        ).all()
    )
    if final_capture:
        grid = _align_grid_driver_ids_from_qualifying(starting_grid, q)
        if set(grid["driver_id"]) != set(frame["driver_id"].astype(str)):
            raise ValueError("first-seen final grid roster does not match event roster")
        grid_columns = [
            column
            for column in grid.columns
            if column == "driver_id" or column.startswith("grid_")
            or column == "race_information_horizon"
        ]
        frame = frame.drop(
            columns=[column for column in frame.columns if column.startswith("grid_")],
            errors="ignore",
        ).merge(grid[grid_columns], on="driver_id", how="left", validate="one_to_one")
        horizon = RacePredictionHorizon.POST_GRID_PRE_RACE
        prediction_as_of = str(grid["grid_first_published_at"].iloc[0])
    else:
        frame["grid_position"] = _stable_provisional_grid_positions(frame)
        frame["grid_status"] = "grid"
        frame["grid_starter_eligible"] = 1.0
        frame["grid_pit_lane_start"] = False
        frame["race_information_horizon"] = (
            RacePredictionHorizon.POST_QUALIFYING_PRE_GRID.value
        )
        horizon = RacePredictionHorizon.POST_QUALIFYING_PRE_GRID
        prediction_as_of = _event_as_of(metadata)
    qualifying_snapshot_provenance = _session_snapshot_provenance(
        metadata,
        "qualifying",
        prediction_as_of=prediction_as_of,
    )
    prediction_as_of_semantics = (
        "first_seen_final_grid_publication"
        if final_capture
        else "retrospective_race_start_upper_bound_without_first_seen_qualifying_time"
    )
    frame["feature_as_of"] = prediction_as_of
    frame["grid_baseline_position"] = _legal_grid_baseline(frame)

    classified_history = (
        history.get("terminal_status", pd.Series(index=history.index, dtype=object))
        .astype(str)
        .eq(TerminalStatus.CLASSIFIED_FINISH.value)
        if not history.empty
        else pd.Series(dtype=bool)
    )
    places_gained = (
        pd.to_numeric(history.get("grid_baseline_position"), errors="coerce")
        - pd.to_numeric(history.get("finish_position"), errors="coerce")
        if not history.empty
        else pd.Series(dtype=float)
    )
    places_gained = places_gained.where(classified_history)
    prior = history.copy()
    prior["places_gained"] = places_gained
    if prior.empty or "event_key" not in prior.columns:
        same_season_prior = prior.copy()
    else:
        same_season_prior = prior.loc[
            pd.to_numeric(prior["event_key"], errors="coerce")
            .floordiv(100)
            .eq(int(year))
        ].copy()
    team_strength = _rolling_group_mean(
        same_season_prior,
        key_column="team_name",
        value_column="places_gained",
        prior_strength=8.0,
    )
    driver_strength = _rolling_group_mean(
        same_season_prior,
        key_column="driver_id",
        value_column="places_gained",
        prior_strength=5.0,
    )
    terminal_mask = (
        prior.get("terminal_status", pd.Series(index=prior.index, dtype=object)).astype(str)
        != TerminalStatus.CLASSIFIED_FINISH.value
    )
    mechanical_mask = prior.get(
        "terminal_status", pd.Series(index=prior.index, dtype=object)
    ).astype(str).eq(TerminalStatus.MECHANICAL_POWER_UNIT.value)
    incident_mask = prior.get(
        "terminal_status", pd.Series(index=prior.index, dtype=object)
    ).astype(str).eq(TerminalStatus.COLLISION_INCIDENT.value)
    team_mechanical = _rolling_rate(
        prior, key_column="team_name", mask=mechanical_mask
    )
    driver_incident = _rolling_rate(
        prior, key_column="driver_id", mask=incident_mask
    )
    power_unit_mechanical = _rolling_rate(
        prior,
        key_column="power_unit",
        mask=mechanical_mask,
    )
    circuit_dnf = _rolling_rate(prior, key_column="circuit_id", mask=terminal_mask)

    frame["race_team_strength_score"] = frame["team_name"].map(team_strength).fillna(0.0)
    frame["race_driver_strength_score"] = frame["driver_id"].astype(str).map(driver_strength).fillna(0.0)
    frame["race_team_mechanical_rate"] = frame["team_name"].map(team_mechanical)
    frame["race_power_unit_mechanical_rate"] = frame["power_unit"].astype(str).map(
        power_unit_mechanical
    )
    frame["race_driver_incident_rate"] = frame["driver_id"].astype(str).map(driver_incident)
    frame["race_circuit_dnf_rate"] = frame["circuit_id"].map(circuit_dnf)
    frame["race_weekend_stoppage_count"] = float(
        _pre_race_red_flag_count(
            root, metadata, weekend_dir=metadata_path.parent
        )
    )
    frame["race_wet_probability"] = _pre_race_wet_evidence(
        root, metadata, weekend_dir=metadata_path.parent
    )
    raw_mechanical_stop_share = _pre_race_driver_mechanical_stop_share(
        root, metadata, weekend_dir=metadata_path.parent
    )
    mechanical_stop_share = {
        identity_lookup[_normalize_identity_token(raw_driver_id)]: value
        for raw_driver_id, value in raw_mechanical_stop_share.items()
        if _normalize_identity_token(raw_driver_id) in identity_lookup
    }
    frame["race_current_weekend_mechanical_stop_share"] = (
        frame["driver_id"].astype(str).map(mechanical_stop_share)
    )
    track_stats = provider.get_track_stats(year, round_number) or {}
    for key, value in track_stats.items():
        frame[key] = value
    if "track_safety_car_propensity" in frame.columns:
        frame["race_safety_car_probability"] = pd.to_numeric(
            frame["track_safety_car_propensity"], errors="coerce"
        )
    if "track_weather_uncertainty" in frame.columns:
        frame["race_weather_uncertainty"] = pd.to_numeric(
            frame["track_weather_uncertainty"], errors="coerce"
        )

    oof_qualifying_rank, qualifying_prior_manifest = _rolling_oof_qualifying_prior(
        history,
        frame,
    )
    frame["qualy_pred_rank"] = oof_qualifying_rank
    frame["qualy_prior_source"] = str(qualifying_prior_manifest["source"])
    frame["qualy_prior_training_events"] = int(
        qualifying_prior_manifest["training_events"]
    )

    if "fp_race_sim_delta" in frame.columns:
        frame["race_long_run_pace_delta"] = pd.to_numeric(frame["fp_race_sim_delta"], errors="coerce")
        frame["race_teammate_long_run_delta"] = frame["race_long_run_pace_delta"] - frame.groupby(
            "team_name", dropna=False
        )["race_long_run_pace_delta"].transform("median")
    if "race_practice_evidence_share" in frame.columns:
        frame["race_long_run_evidence_share"] = pd.to_numeric(
            frame["race_practice_evidence_share"], errors="coerce"
        )
    elif "fp_race_sim_evidence_share" in frame.columns:
        frame["race_long_run_evidence_share"] = pd.to_numeric(
            frame["fp_race_sim_evidence_share"], errors="coerce"
        )
    if "race_longest_clean_stint_laps" not in frame.columns and "fp_race_sim_laps" in frame.columns:
        frame["race_longest_clean_stint_laps"] = pd.to_numeric(
            frame["fp_race_sim_laps"], errors="coerce"
        )
    if "race_practice_uncertainty" in frame.columns:
        frame["race_long_run_uncertainty"] = pd.to_numeric(
            frame["race_practice_uncertainty"], errors="coerce"
        )
    elif "fp_race_sim_raw_degradation_mad" in frame.columns:
        frame["race_long_run_uncertainty"] = pd.to_numeric(
            frame["fp_race_sim_raw_degradation_mad"], errors="coerce"
        )
    if "sprint_race_sim_delta" in frame.columns:
        frame["race_sprint_pace_delta"] = pd.to_numeric(
            frame["sprint_race_sim_delta"], errors="coerce"
        )
    if "track_finish_order_mobility" not in frame.columns:
        same_circuit = prior.loc[prior.get("circuit_id", pd.Series(index=prior.index)).eq(frame["circuit_id"].iloc[0])]
        if same_circuit.empty:
            mobility = 0.5
        else:
            mobility = float(
                (
                    pd.to_numeric(same_circuit["finish_position"], errors="coerce")
                    - pd.to_numeric(same_circuit["grid_baseline_position"], errors="coerce")
                ).abs().mean()
                / max(1.0, len(frame))
            )
        frame["track_finish_order_mobility"] = float(np.clip(mobility, 0.0, 1.0))

    # Freeze the entire inference frame before opening the target-session
    # classification.  Race rows contribute labels only: they can never repair
    # the roster, team, PU identity, grid, or any model feature.
    inference_columns = tuple(frame.columns)
    inference_snapshot = frame.loc[:, inference_columns].copy()
    race = provider.get_race_results(year, round_number)
    if race.empty:
        raise ValueError(f"{year} round {round_number} has no Race target classification")
    target_columns = [
        column
        for column in (
            "driver_id",
            "position",
            "race_status_raw",
            "race_status_evidence_complete",
            "retirement_fraction",
            "laps_completed",
        )
        if column in race
    ]
    race_targets = race[target_columns].copy().rename(
        columns={"position": "finish_position"}
    )
    race_targets["race_target_source"] = "provider_race_classification"
    race_targets["race_target_observed"] = (
        pd.to_numeric(race_targets["finish_position"], errors="coerce").notna()
        & race_targets["race_status_evidence_complete"].fillna(False).astype(bool)
    )
    race_targets = _map_driver_ids_from_qualifying(
        race_targets,
        identity_lookup,
        source_name="race",
        allow_non_roster_rows=False,
    )
    feature_ids = set(inference_snapshot["driver_id"].astype(str))
    race_ids = set(race_targets["driver_id"].astype(str))
    target_only_ids = sorted(race_ids - feature_ids)
    if target_only_ids:
        raise ValueError(
            "Race target contains entrants absent from the causal pre-race roster: "
            f"{target_only_ids}"
        )
    frame = inference_snapshot.merge(
        race_targets,
        on="driver_id",
        how="left",
        validate="one_to_one",
    )
    frame = _resolve_missing_race_targets(frame, final_capture=final_capture)
    encoded_terminal = frame["race_status_raw"].map(reason_code_terminal_status)
    observed_target = frame["race_target_observed"].eq(True)
    unresolved_target = observed_target & encoded_terminal.isna()
    if unresolved_target.any():
        unresolved = sorted(
            frame.loc[unresolved_target, "race_status_raw"]
            .astype(str)
            .unique()
        )
        raise ValueError(f"unresolved terminal status labels: {unresolved}")
    frame["terminal_status"] = encoded_terminal.map(
        lambda value: value.value if isinstance(value, TerminalStatus) else None
    )
    frame["terminal_label_granularity"] = frame["race_status_raw"].map(
        terminal_label_granularity
    ).map(lambda value: value.value if value is not None else None)

    input_paths = [metadata_path]
    if final_capture and "grid_capture_path" in starting_grid.columns:
        capture_path = Path(str(starting_grid["grid_capture_path"].iloc[0]))
        if capture_path.exists():
            input_paths.append(capture_path)
    for session in metadata.get("sessions", []):
        for key in ("laps_path", "results_path", "weather_path", "race_control_messages_path"):
            reference = session.get(key)
            if not reference:
                continue
            path = _resolve_session_reference(
                root, metadata_path.parent, reference
            )
            if path is not None:
                input_paths.append(path)
    missing_target_driver_ids = sorted(
        frame.loc[~observed_target, "driver_id"].astype(str).tolist()
    )
    info = {
        "event_key": event_key,
        "year": int(year),
        "round": int(round_number),
        "event_name": str(metadata.get("event_name") or f"Round {round_number}"),
        "event_format": str(metadata.get("event_format") or "unknown"),
        "field_size": int(len(frame)),
        "information_horizon": horizon.value,
        "prediction_as_of": prediction_as_of,
        "prediction_as_of_semantics": prediction_as_of_semantics,
        "final_grid_snapshot_used": final_capture,
        "grid_capture_id": (
            str(starting_grid["grid_capture_id"].iloc[0]) if final_capture else None
        ),
        "causal_inference_columns": list(inference_columns),
        "race_truth_attached_after_inference_freeze": True,
        "race_target_coverage_complete": not missing_target_driver_ids,
        "race_target_missing_driver_ids": missing_target_driver_ids,
        "race_target_sources": {
            str(key): int(value)
            for key, value in frame["race_target_source"]
            .fillna("unavailable")
            .value_counts()
            .sort_index()
            .items()
        },
        "signed_qualifying_surprise_prior": qualifying_prior_manifest,
        "qualifying_snapshot_provenance": qualifying_snapshot_provenance,
        "driver_identity_contract": {
            "canonical_field": "driver_id",
            "canonical_source": (
                "retrospective_provider_qualifying_fia_driver_abbreviation"
            ),
            "provider_identity_retained_as": "provider_driver_id",
            "race_truth_allowed_to_repair_identity": False,
        },
    }
    return frame, info, input_paths


def _binary_metrics(actual: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0 - 1e-12)
    y = np.asarray(actual, dtype=float)
    return (
        float(np.mean(np.square(p - y))),
        float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
    )


def _set_prediction_order_residual_weight(
    model: SurvivalAwareRaceModel,
    residual_weight: float,
) -> None:
    """Set the score-time residual multiplier without refitting the ranker.

    ``ConditionalOrderConfig.residual_weight`` is consumed only by
    ``BradleyTerryOrderRanker.score``.  It is deliberately absent from its
    training matrix, regularization, and optimizer.  Keeping this operation
    explicit makes the event-level fit-cache contract reviewable and ensures a
    future fit-time use cannot be introduced here silently.
    """

    model.order_model.config = replace(
        model.order_model.config,
        residual_weight=float(residual_weight),
    )


def _mean(events: Sequence[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(event[key]) for event in events]))


def _optional_mean(events: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [
        float(event[key])
        for event in events
        if event.get(key) is not None and np.isfinite(float(event[key]))
    ]
    return float(np.mean(values)) if values else None


def _same_product_promotion_blockers(
    *,
    audit_event_count: int,
    same_product_selection_evidence: bool,
    same_product_calibration_evidence: bool,
    selection_event_count: int = _MINIMUM_INDEPENDENT_LOCK_EVENTS,
    calibration_event_count: int = _MINIMUM_INDEPENDENT_LOCK_EVENTS,
    challenger_selected_on_selection: bool = True,
    point_in_time_input_snapshot_verified: bool = True,
) -> tuple[str, ...]:
    """Return protocol blockers before any metric-based promotion decision."""

    reasons: list[str] = []
    if int(audit_event_count) < _MINIMUM_INDEPENDENT_AUDIT_EVENTS:
        reasons.append("fewer_than_three_same_horizon_audit_events")
    if not bool(same_product_selection_evidence):
        reasons.append("missing_same_product_selection_evidence")
    elif int(selection_event_count) < _MINIMUM_INDEPENDENT_LOCK_EVENTS:
        reasons.append("fewer_than_four_same_horizon_selection_events")
    if not bool(same_product_calibration_evidence):
        reasons.append("missing_same_product_calibration_evidence")
    elif int(calibration_event_count) < _MINIMUM_INDEPENDENT_LOCK_EVENTS:
        reasons.append("fewer_than_four_same_horizon_calibration_events")
    if not bool(challenger_selected_on_selection):
        reasons.append("challenger_not_selected_on_same_season_selection")
    if not bool(point_in_time_input_snapshot_verified):
        reasons.append("qualifying_snapshot_not_first_seen_verified")
    return tuple(reasons)


def _select_race_policy(
    challenger_rows: Sequence[dict[str, Any]],
    *,
    baseline_row: dict[str, Any] | None,
    diagnostic_temperature: float,
) -> dict[str, Any]:
    """Select the simulator only when it materially beats the retained grid.

    The simulator grid is still evaluated and its best configuration is kept
    for diagnostic audit scoring.  The public policy, however, remains the
    legal-grid baseline unless the selection-only gain reaches the same 5%
    materiality threshold used by the Race promotion contract.
    """

    if not challenger_rows:
        return {
            "information_horizon": (
                str(baseline_row["information_horizon"])
                if baseline_row is not None
                else RacePredictionHorizon.POST_QUALIFYING_PRE_GRID.value
            ),
            "selected_model_id": "legal_grid_baseline",
            "challenger_selected": False,
            "plackett_luce_temperature": float(diagnostic_temperature),
            "order_residual_weight": _NO_SAME_HORIZON_ORDER_RESIDUAL_WEIGHT,
            "mean_position_mae": None,
            "challenger_mean_position_mae": None,
            "baseline_mean_position_mae": (
                float(baseline_row["mean_position_mae"])
                if baseline_row is not None
                else None
            ),
            "relative_challenger_gain": None,
            "minimum_relative_selection_gain": _MINIMUM_RELATIVE_SELECTION_GAIN,
            "event_count": 0,
            "event_keys": [],
            "same_product_selection_evidence": False,
            "parameter_source": "fixed_diagnostic_default_no_same_product_selection",
        }
    challenger = min(
        challenger_rows,
        key=lambda row: (
            float(row["mean_position_mae"]),
            float(row["plackett_luce_temperature"]),
            float(row["order_residual_weight"]),
        ),
    )
    baseline_mae = (
        float(baseline_row["mean_position_mae"])
        if baseline_row is not None
        and baseline_row.get("mean_position_mae") is not None
        else None
    )
    challenger_mae = float(challenger["mean_position_mae"])
    comparable = bool(
        baseline_row is not None
        and tuple(baseline_row.get("event_keys", ()))
        == tuple(challenger.get("event_keys", ()))
        and int(baseline_row.get("event_count", 0))
        == int(challenger.get("event_count", 0))
    )
    relative_gain = (
        float((baseline_mae - challenger_mae) / baseline_mae)
        if comparable and baseline_mae is not None and baseline_mae > 0.0
        else None
    )
    challenger_selected = bool(
        relative_gain is not None
        and relative_gain >= _MINIMUM_RELATIVE_SELECTION_GAIN
    )
    return {
        **challenger,
        "selected_model_id": (
            "survival_aware_joint" if challenger_selected else "legal_grid_baseline"
        ),
        "challenger_selected": challenger_selected,
        "challenger_mean_position_mae": challenger_mae,
        "baseline_mean_position_mae": baseline_mae,
        "relative_challenger_gain": relative_gain,
        "minimum_relative_selection_gain": _MINIMUM_RELATIVE_SELECTION_GAIN,
        "same_product_selection_evidence": comparable,
        "parameter_source": (
            "declared_same_horizon_selection_partition_challenger_selected"
            if challenger_selected
            else "declared_same_horizon_selection_partition_baseline_retained"
        ),
    }


def _apply_selected_position_head(
    scored: pd.DataFrame,
    *,
    selected_model_id: str,
) -> pd.DataFrame:
    """Keep the challenger explicit and make the public alias selector-safe."""

    required = {
        "driver_id",
        "grid_baseline_position",
        "candidate_predicted_position",
    }
    missing = sorted(required.difference(scored.columns))
    if missing:
        raise ValueError(f"selected position head is missing columns {missing}")
    model_id = str(selected_model_id).strip()
    if model_id == "legal_grid_baseline":
        selected_source = "grid_baseline_position"
    elif model_id == "survival_aware_joint":
        selected_source = "candidate_predicted_position"
    else:
        raise ValueError(f"unsupported selected Race model {model_id!r}")

    out = scored.copy()
    field_size = len(out)
    expected = list(range(1, field_size + 1))
    for column in ("grid_baseline_position", "candidate_predicted_position"):
        values = pd.to_numeric(out[column], errors="coerce")
        if values.isna().any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{column} must be a complete integer permutation")
        if sorted(values.astype(int).tolist()) != expected:
            raise ValueError(f"{column} must be a complete integer permutation")

    selected = pd.to_numeric(out[selected_source], errors="raise").astype(int)
    out["selected_predicted_position"] = selected
    # ``predicted_position`` is the public/backward-compatible alias.  It must
    # never expose a challenger rejected by the selection-only policy.
    out["predicted_position"] = selected
    out["candidate_model_id"] = "survival_aware_joint"
    out["selected_model_id"] = model_id
    out["selected_position_source"] = selected_source
    return out


def _aggregate_race_events(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not group:
        raise ValueError("Race aggregate requires at least one event")
    return {
        "events": len(group),
        "baseline_mean_mae": _mean(group, "baseline_mae"),
        "candidate_mean_mae": _mean(group, "candidate_mae"),
        "selected_mean_mae": _mean(group, "selected_mae"),
        "baseline_mean_kendall": _mean(group, "baseline_kendall"),
        "candidate_mean_kendall": _mean(group, "candidate_kendall"),
        "selected_mean_kendall": _mean(group, "selected_kendall"),
        "baseline_status_brier": _mean(group, "baseline_status_brier"),
        "candidate_status_brier": _mean(group, "candidate_status_brier"),
        "selected_status_brier": _mean(group, "selected_status_brier"),
        "baseline_status_log_loss": _mean(group, "baseline_status_log_loss"),
        "candidate_status_log_loss": _mean(group, "candidate_status_log_loss"),
        "selected_status_log_loss": _mean(group, "selected_status_log_loss"),
        "candidate_status_terminal_ece": _mean(
            group, "candidate_status_terminal_ece"
        ),
        "candidate_retirement_fraction_mae": _optional_mean(
            group, "candidate_retirement_fraction_mae"
        ),
        "global_meal_mae": _mean(group, "global_meal_mae"),
        "dns_constrained_meal_mae": _mean(group, "dns_constrained_meal_mae"),
        "dns_constrained_minus_global_meal_mae": _mean(
            group, "dns_constrained_minus_global_meal_mae"
        ),
    }


def _audit_aggregate_payload(
    events: Sequence[Mapping[str, Any]],
    *,
    audit_event_keys: Sequence[int],
) -> dict[str, Any]:
    audit_key_set = {int(value) for value in audit_event_keys}
    audit_events = [
        event for event in events if int(event["event_key"]) in audit_key_set
    ]
    if not audit_events:
        raise ValueError("Race audit aggregate has no scored audit events")
    years = sorted({int(event["year"]) for event in audit_events})
    horizons = sorted(
        {str(event["information_horizon"]) for event in audit_events}
    )
    return {
        "partition_role": "audit",
        "event_keys": sorted(int(event["event_key"]) for event in audit_events),
        **_aggregate_race_events(audit_events),
        "by_year": {
            str(year): _aggregate_race_events(
                [event for event in audit_events if int(event["year"]) == year]
            )
            for year in years
        },
        "by_horizon": {
            horizon: _aggregate_race_events(
                [
                    event
                    for event in audit_events
                    if str(event["information_horizon"]) == horizon
                ]
            )
            for horizon in horizons
        },
    }


def _attach_result_sha256(payload: dict[str, Any]) -> dict[str, Any]:
    """Bind all result content while excluding only the digest field itself."""

    if "result_sha256" in payload:
        raise ValueError("result_sha256 must not exist before finalization")
    payload["result_sha256"] = _canonical_json_sha256(payload)
    return payload


def _fit_binary_terminal_calibrator(
    rows: pd.DataFrame,
) -> BinaryTerminalCalibrator:
    required = {"event_key", "actual_terminal", "predicted_terminal"}
    if not required.issubset(rows.columns) or rows.empty:
        raise ValueError("terminal calibration rows are incomplete")
    actual = pd.to_numeric(rows["actual_terminal"], errors="coerce").to_numpy(
        dtype=float
    )
    probability = np.clip(
        pd.to_numeric(rows["predicted_terminal"], errors="coerce").to_numpy(
            dtype=float
        ),
        1e-6,
        1.0 - 1e-6,
    )
    if not np.isfinite(actual).all() or not np.isfinite(probability).all():
        raise ValueError("terminal calibration rows contain non-finite values")
    if len(np.unique(actual)) != 2:
        raise ValueError("terminal calibration requires both outcome classes")
    logit = np.log(probability / (1.0 - probability)).reshape(-1, 1)
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:
        raise RuntimeError("terminal calibration requires scikit-learn") from exc
    estimator = LogisticRegression(
        C=10.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    )
    estimator.fit(logit, actual.astype(int))
    slope = float(estimator.coef_[0, 0])
    intercept = float(estimator.intercept_[0])
    if slope < 0.0:
        # Preserve monotonicity fail-closed; an inverse calibration curve is
        # evidence that this small block supports only its base rate.
        base_rate = float(np.clip(actual.mean(), 1e-6, 1.0 - 1e-6))
        slope = 0.0
        intercept = float(np.log(base_rate / (1.0 - base_rate)))
    return BinaryTerminalCalibrator(
        intercept=intercept,
        slope=slope,
        calibration_rows=len(rows),
        calibration_event_keys=tuple(
            str(value) for value in sorted(rows["event_key"].astype(str).unique())
        ),
    )


def _blocked_terminal_calibration_lock(
    rows: pd.DataFrame,
    *,
    information_horizon: str,
    simulations_per_event: int,
) -> dict[str, Any]:
    """Record raw final-distribution diagnostics without claiming calibration."""

    diagnostic_brier: float | None = None
    diagnostic_log_loss: float | None = None
    diagnostic_event_keys: list[str] = []
    if not rows.empty:
        required = {"event_key", "actual_terminal", "predicted_terminal"}
        if not required.issubset(rows.columns):
            raise ValueError("raw terminal calibration diagnostics are incomplete")
        actual = pd.to_numeric(rows["actual_terminal"], errors="coerce").to_numpy(
            dtype=float
        )
        probability = np.clip(
            pd.to_numeric(rows["predicted_terminal"], errors="coerce").to_numpy(
                dtype=float
            ),
            1e-12,
            1.0 - 1e-12,
        )
        if not np.isfinite(actual).all() or not np.isfinite(probability).all():
            raise ValueError("raw terminal calibration diagnostics are non-finite")
        diagnostic_brier = float(np.mean(np.square(probability - actual)))
        diagnostic_log_loss = float(
            -np.mean(
                actual * np.log(probability)
                + (1.0 - actual) * np.log(1.0 - probability)
            )
        )
        diagnostic_event_keys = sorted(
            rows["event_key"].astype(str).unique().tolist()
        )
    return {
        "information_horizon": str(information_horizon),
        "same_product_calibration_evidence": False,
        "calibration_status": (
            "blocked_requires_simulation_in_loop_or_marginal_preserving_copula"
        ),
        "calibration_fit_probability_source": None,
        "calibration_application_probability_source": None,
        "scored_probability_source": "empirical_post_shared_shock_joint_samples",
        "raw_calibration_diagnostic_probability_source": (
            "empirical_post_shared_shock_joint_samples"
        ),
        "simulations_per_event": int(simulations_per_event),
        "rows": int(len(rows)),
        "event_keys": diagnostic_event_keys,
        "intercept": None,
        "slope": None,
        "raw_terminal_brier": diagnostic_brier,
        "raw_terminal_log_loss": diagnostic_log_loss,
        "error": "zero_shock_platt_rejected_for_final_distribution_mismatch",
    }


def run(
    *,
    weekends_dir: Path,
    years: Sequence[int],
    evaluation_years: Sequence[int],
    simulations: int,
    bootstrap_samples: int,
    seed: int,
    temperature_candidates: Sequence[float] = (0.08, 0.12, 0.18, 0.25, 0.35),
    order_residual_candidates: Sequence[float] = (0.0, 0.65, 1.0, 1.5, 2.0),
    selection_simulations: int = 400,
    development_years: Sequence[int] = (2022, 2023),
    selection_years: Sequence[int] = (2024,),
    calibration_years: Sequence[int] = (2025,),
    audit_years: Sequence[int] = (2026,),
    same_season_only: bool = True,
) -> dict[str, Any]:
    root = _root()
    provider = LocalWeekendProvider(str(weekends_dir))
    loaded_years = tuple(sorted({int(value) for value in years}))
    evaluation_set = set(int(value) for value in evaluation_years)
    if same_season_only:
        if len(loaded_years) != 1:
            raise ValueError(
                "same-season Race mode requires exactly one loaded target year; "
                "use the explicit legacy cross-season opt-in for multiple years"
            )
        target_year = int(loaded_years[0])
        if evaluation_set != {target_year}:
            raise ValueError(
                "same-season Race mode requires evaluation_years to equal the one "
                f"loaded target year ({target_year})"
            )
    else:
        target_year = max(loaded_years)
    minimum_prior_events = (
        _SAME_SEASON_MINIMUM_PRIOR_EVENTS
        if same_season_only
        else _LEGACY_CROSS_SEASON_MINIMUM_PRIOR_EVENTS
    )
    history = pd.DataFrame()
    event_frames: dict[int, pd.DataFrame] = {}
    event_info: dict[int, dict[str, Any]] = {}
    inputs: set[Path] = set()
    excluded_incomplete_target_events: list[dict[str, Any]] = []
    for year in loaded_years:
        for item in provider.list_rounds(year):
            round_number = int(item["round_number"])
            event_key = int(year) * 100 + int(round_number)
            feature_history = _strict_prior_history(
                history,
                event_key=event_key,
                same_season_only=bool(same_season_only),
            )
            frame, info, event_inputs = _build_event_rows(
                root=root,
                provider=provider,
                weekends_dir=weekends_dir,
                year=year,
                round_number=round_number,
                history=feature_history,
            )
            inputs.update(event_inputs)
            if not bool(info["race_target_coverage_complete"]):
                excluded_incomplete_target_events.append(
                    {
                        "event_key": int(info["event_key"]),
                        "event_name": str(info["event_name"]),
                        "missing_driver_ids": list(
                            info["race_target_missing_driver_ids"]
                        ),
                        "reason": (
                            "incomplete_provider_race_target_without_"
                            "authoritative_nonstarter_evidence"
                        ),
                    }
                )
                continue
            event_frames[int(info["event_key"])] = frame
            event_info[int(info["event_key"])] = info
            history = pd.concat([history, frame], ignore_index=True)

    partition_years = {
        "development": set(int(value) for value in development_years),
        "selection": set(int(value) for value in selection_years),
        "calibration": set(int(value) for value in calibration_years),
        "audit": set(int(value) for value in audit_years),
    }
    if same_season_only:
        partition_events = _same_season_event_partitions(
            tuple(sorted(event_frames)), target_year=target_year
        )
    else:
        partition_events = {
            name: [
                str(event_key)
                for event_key in sorted(event_frames)
                if event_key // 100 in years_for_partition
            ]
            for name, years_for_partition in partition_years.items()
        }
    partition_issues = validate_event_partitions(
        development=partition_events["development"],
        selection=partition_events["selection"],
        calibration=partition_events["calibration"],
        audit=partition_events["audit"],
    )
    if partition_issues:
        raise ValueError(
            "invalid Race event partitions: " + ", ".join(partition_issues)
        )
    if same_season_only:
        ordered_boundaries = [
            max(int(value) for value in partition_events["development"]),
            min(int(value) for value in partition_events["selection"]),
            max(int(value) for value in partition_events["selection"]),
            min(int(value) for value in partition_events["calibration"]),
            max(int(value) for value in partition_events["calibration"]),
            min(int(value) for value in partition_events["audit"]),
        ]
        if not all(
            left < right
            for left, right in zip(ordered_boundaries[::2], ordered_boundaries[1::2])
        ):
            raise ValueError("same-season Race event partitions are not chronological")
    evaluation_event_keys = (
        {
            int(value)
            for name in ("selection", "calibration", "audit")
            for value in partition_events[name]
        }
        if same_season_only
        else {
            event_key
            for event_key in event_frames
            if event_key // 100 in evaluation_set
        }
    )
    audit_event_keys = {int(value) for value in partition_events["audit"]}
    for raw_event_key in partition_events["audit"]:
        audit_frame = event_frames[int(raw_event_key)]
        prior_source = str(
            event_info[int(raw_event_key)]["signed_qualifying_surprise_prior"][
                "source"
            ]
        )
        if not prior_source.startswith("rolling_out_of_event_ridge"):
            raise ValueError(
                f"audit event {raw_event_key} lacks OOF Qualifying-prior provenance"
            )
        if "power_unit" not in audit_frame.columns or audit_frame[
            "power_unit"
        ].isna().any():
            raise ValueError(
                f"audit event {raw_event_key} lacks complete causal power-unit identity"
            )

    target_columns = [
        "finish_position",
        "terminal_status",
        "race_status_raw",
        "race_status_evidence_complete",
        "retirement_fraction",
        "laps_completed",
        "terminal_label_granularity",
        "race_provider_driver_id",
        "race_target_observed",
        "race_target_source",
    ]

    def event_inputs(
        event_key: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, RacePredictionHorizon]:
        prior = _strict_prior_history(
            history,
            event_key=int(event_key),
            same_season_only=bool(same_season_only),
        )
        current = event_frames[event_key].copy()
        prediction_as_of = str(event_info[event_key]["prediction_as_of"])
        current_year = event_key // 100
        order_history = prior.loc[
            pd.to_numeric(prior["event_key"], errors="coerce")
            .floordiv(100)
            .eq(current_year)
        ].copy()
        inference = current.drop(columns=target_columns, errors="ignore")
        horizon = RacePredictionHorizon(event_info[event_key]["information_horizon"])
        return prior, current, order_history, prediction_as_of, horizon

    # Residual weight and Plackett-Luce temperature are both prediction-time
    # parameters.  The ranker fit is identical across the full selection grid,
    # so one immutable uncalibrated fit per event is the exact sufficient key.
    fitted_model_cache: dict[int, SurvivalAwareRaceModel] = {}
    model_fit_count = 0
    model_fit_cache_hits = 0

    def event_forecast(
        event_key: int,
        *,
        temperature: float,
        residual_weight: float,
        forecast_simulations: int,
        calibrator: BinaryTerminalCalibrator | None = None,
    ) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
        nonlocal model_fit_count, model_fit_cache_hits
        prior, current, order_history, prediction_as_of, horizon = event_inputs(
            event_key
        )
        cache_key = int(event_key)
        cache_allowed = calibrator is None
        model = fitted_model_cache.get(cache_key) if cache_allowed else None
        if model is None:
            terminal_model = PartialPooledTerminalHazard()
            model = SurvivalAwareRaceModel(
                terminal_model=terminal_model,
                order_model=BradleyTerryOrderRanker(
                    ConditionalOrderConfig(residual_weight=float(residual_weight))
                ),
            ).fit(
                prior,
                cutoff=prediction_as_of,
                order_history=order_history,
            )
            model_fit_count += 1
            if cache_allowed:
                fitted_model_cache[cache_key] = model
        else:
            model_fit_cache_hits += 1
        _set_prediction_order_residual_weight(model, residual_weight)
        if calibrator is not None:
            model.terminal_model.set_terminal_calibrator(calibrator)
        forecast = model.predict_joint(
            current.drop(columns=target_columns, errors="ignore"),
            horizon=horizon,
            prediction_as_of=prediction_as_of,
            simulations=int(forecast_simulations),
            seed=int(seed) + event_key,
            plackett_luce_temperature=float(temperature),
        )
        return forecast, prior, current

    temperatures = tuple(sorted({float(value) for value in temperature_candidates}))
    residual_weights = tuple(
        sorted({float(value) for value in order_residual_candidates})
    )
    if (
        not temperatures
        or len(temperatures) > 6
        or any(not np.isfinite(value) or value <= 0.0 for value in temperatures)
    ):
        raise ValueError("temperature candidate grid must contain 1-6 positive values")
    if (
        not residual_weights
        or len(residual_weights) > 6
        or any(not np.isfinite(value) or value < 0.0 for value in residual_weights)
    ):
        raise ValueError("order residual candidate grid must contain 1-6 non-negative values")
    bounded_selection_simulations = max(100, int(selection_simulations))
    horizon_values = tuple(horizon.value for horizon in RacePredictionHorizon)
    diagnostic_temperature = 0.25 if 0.25 in temperatures else temperatures[0]
    diagnostic_residual_weight = _NO_SAME_HORIZON_ORDER_RESIDUAL_WEIGHT
    selection_trace: list[dict[str, Any]] = []
    selected_by_horizon: dict[str, dict[str, Any]] = {}
    for horizon_value in horizon_values:
        selection_keys = [
            int(value)
            for value in partition_events["selection"]
            if event_info[int(value)]["information_horizon"] == horizon_value
        ]
        local_trace: list[dict[str, Any]] = []
        for temperature in temperatures:
            for residual_weight in residual_weights:
                errors: list[float] = []
                scored_keys: list[str] = []
                for event_key in selection_keys:
                    prior = _strict_prior_history(
                        history,
                        event_key=int(event_key),
                        same_season_only=bool(same_season_only),
                    )
                    if prior["event_key"].nunique() < minimum_prior_events:
                        continue
                    forecast, _, current = event_forecast(
                        event_key,
                        temperature=temperature,
                        residual_weight=residual_weight,
                        forecast_simulations=bounded_selection_simulations,
                    )
                    paired = current[["driver_id", "finish_position"]].merge(
                        forecast.point_classification[
                            ["driver_id", "predicted_position"]
                        ],
                        on="driver_id",
                        validate="one_to_one",
                    )
                    errors.append(
                        float(
                            (
                                pd.to_numeric(
                                    paired["finish_position"], errors="coerce"
                                )
                                - pd.to_numeric(
                                    paired["predicted_position"], errors="coerce"
                                )
                            )
                            .abs()
                            .mean()
                        )
                    )
                    scored_keys.append(str(event_key))
                if errors:
                    record = {
                        "model_id": "survival_aware_joint",
                        "information_horizon": horizon_value,
                        "plackett_luce_temperature": temperature,
                        "order_residual_weight": residual_weight,
                        "mean_position_mae": float(np.mean(errors)),
                        "event_count": len(errors),
                        "event_keys": scored_keys,
                    }
                    local_trace.append(record)
                    selection_trace.append(record)
        baseline_errors: list[float] = []
        baseline_keys: list[str] = []
        for event_key in selection_keys:
            _, current, _, _, _ = event_inputs(event_key)
            actual = pd.to_numeric(current["finish_position"], errors="coerce")
            baseline = pd.to_numeric(
                current["grid_baseline_position"], errors="coerce"
            )
            finite = actual.notna() & baseline.notna()
            if not finite.any():
                continue
            baseline_errors.append(
                float((actual.loc[finite] - baseline.loc[finite]).abs().mean())
            )
            baseline_keys.append(str(event_key))
        baseline_record = (
            {
                "model_id": "legal_grid_baseline",
                "information_horizon": horizon_value,
                "mean_position_mae": float(np.mean(baseline_errors)),
                "event_count": len(baseline_errors),
                "event_keys": baseline_keys,
            }
            if baseline_errors
            else None
        )
        if baseline_record is not None:
            selection_trace.append(baseline_record)
        selected_by_horizon[horizon_value] = _select_race_policy(
            local_trace,
            baseline_row=baseline_record,
            diagnostic_temperature=diagnostic_temperature,
        )

    terminal_calibrators: dict[str, BinaryTerminalCalibrator | None] = {}
    calibration_locks: dict[str, dict[str, Any]] = {}
    for horizon_value in horizon_values:
        calibration_records: list[dict[str, object]] = []
        selected = selected_by_horizon[horizon_value]
        calibration_keys = [
            int(value)
            for value in partition_events["calibration"]
            if event_info[int(value)]["information_horizon"] == horizon_value
        ]
        for event_key in calibration_keys:
            prior, current, _, _, _ = event_inputs(event_key)
            if prior["event_key"].nunique() < minimum_prior_events:
                continue
            forecast, _, current = event_forecast(
                event_key,
                temperature=float(selected["plackett_luce_temperature"]),
                residual_weight=float(selected["order_residual_weight"]),
                forecast_simulations=bounded_selection_simulations,
            )
            probability = forecast.status_probabilities.set_index("driver_id")
            for _, row in current.iterrows():
                driver_id = str(row["driver_id"])
                calibration_records.append(
                    {
                        "event_key": str(event_key),
                        "actual_terminal": float(
                            str(row["terminal_status"])
                            != TerminalStatus.CLASSIFIED_FINISH.value
                        ),
                        "predicted_terminal": float(
                            probability.loc[
                                driver_id, "p_terminal"
                            ]
                        ),
                    }
                )
        calibration_frame = pd.DataFrame.from_records(calibration_records)
        # A Platt curve fitted to zero-shock row probabilities does not
        # calibrate the final post-shock Monte-Carlo marginal: shared shocks,
        # per-bin caps and multi-bin survival are nonlinear.  Fitting a
        # post-hoc curve to the final marginal would in turn make reported
        # probabilities disagree with the draws used for classification.
        # Keep the coherent joint simulator raw and fail closed until either a
        # simulation-in-the-loop calibrator or a marginal-preserving copula is
        # implemented with enough independent events.
        terminal_calibrators[horizon_value] = None
        calibration_locks[horizon_value] = _blocked_terminal_calibration_lock(
            calibration_frame,
            information_horizon=horizon_value,
            simulations_per_event=bounded_selection_simulations,
        )

    events: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for event_key in sorted(event_frames):
        if event_key not in evaluation_event_keys:
            continue
        prior = _strict_prior_history(
            history,
            event_key=int(event_key),
            same_season_only=bool(same_season_only),
        )
        if prior["event_key"].nunique() < minimum_prior_events:
            continue
        horizon_value = str(event_info[event_key]["information_horizon"])
        selected = selected_by_horizon[horizon_value]
        same_product_calibrator = terminal_calibrators[horizon_value]
        calibration_applied = bool(
            _event_is_in_partition(event_key, partition_events, "audit")
            and same_product_calibrator is not None
        )
        forecast, prior, current = event_forecast(
            event_key,
            temperature=float(selected["plackett_luce_temperature"]),
            residual_weight=float(selected["order_residual_weight"]),
            forecast_simulations=int(simulations),
            calibrator=(same_product_calibrator if calibration_applied else None),
        )
        scored_identity_columns = [
            column
            for column in (
                "driver_id",
                "driver_abbreviation",
                "provider_driver_id",
                "car_number",
                "grid_provider_driver_id",
                "grid_identity_mapping_source",
                "race_provider_driver_id",
                "race_target_source",
            )
            if column in current.columns
        ]
        candidate_classification = forecast.point_classification[
            [
                "driver_id",
                "predicted_position",
                "global_meal_position",
                "dns_constrained_meal_position",
                "predicted_terminal_status",
                "expected_position",
            ]
        ].rename(columns={"predicted_position": "candidate_predicted_position"})
        scored = current[
            [
                *scored_identity_columns,
                "grid_position",
                "grid_baseline_position",
                "finish_position",
                "terminal_status",
                "terminal_label_granularity",
                "race_target_observed",
                "race_status_raw",
                "retirement_fraction",
            ]
        ].merge(
            candidate_classification,
            on="driver_id",
            validate="one_to_one",
        ).merge(
            forecast.status_probabilities,
            on="driver_id",
            validate="one_to_one",
        ).merge(
            forecast.position_probabilities,
            on="driver_id",
            validate="one_to_one",
        )
        baseline_by_driver = current[["driver_id"]].copy()
        baseline_by_driver["baseline_terminal_probability"] = (
            _causal_rolling_terminal_probability(prior, current)
        )
        scored = scored.merge(
            baseline_by_driver,
            on="driver_id",
            validate="one_to_one",
        )
        scored = _apply_selected_position_head(
            scored,
            selected_model_id=str(selected["selected_model_id"]),
        )
        actual_position = pd.to_numeric(scored["finish_position"], errors="coerce")
        baseline_position = pd.to_numeric(
            scored["grid_baseline_position"], errors="coerce"
        )
        candidate_position = pd.to_numeric(
            scored["candidate_predicted_position"], errors="coerce"
        )
        selected_position = pd.to_numeric(
            scored["selected_predicted_position"], errors="coerce"
        )
        global_meal_position = pd.to_numeric(
            scored["global_meal_position"], errors="coerce"
        )
        actual_terminal = scored["terminal_status"].ne(TerminalStatus.CLASSIFIED_FINISH.value).astype(float)
        candidate_terminal = pd.to_numeric(scored["p_terminal"], errors="coerce")
        baseline_probability = pd.to_numeric(
            scored["baseline_terminal_probability"], errors="coerce"
        ).to_numpy(dtype=float)
        baseline_brier, baseline_log_loss = _binary_metrics(
            actual_terminal.to_numpy(dtype=float), baseline_probability
        )
        candidate_brier, candidate_log_loss = _binary_metrics(
            actual_terminal.to_numpy(dtype=float), candidate_terminal.to_numpy(dtype=float)
        )
        scored["candidate_terminal_probability"] = candidate_terminal
        if str(selected["selected_model_id"]) == "survival_aware_joint":
            selected_terminal = candidate_terminal.to_numpy(dtype=float)
            selected_status_probability_source = "candidate_terminal_probability"
        else:
            selected_terminal = baseline_probability
            selected_status_probability_source = "baseline_terminal_probability"
        scored["selected_terminal_probability"] = selected_terminal
        scored["selected_status_probability_source"] = (
            selected_status_probability_source
        )
        selected_brier, selected_log_loss = _binary_metrics(
            actual_terminal.to_numpy(dtype=float),
            np.asarray(selected_terminal, dtype=float),
        )
        status_evaluation = evaluate_terminal_status_probabilities(
            scored[
                [
                    "driver_id",
                    "terminal_status",
                    "terminal_label_granularity",
                    "race_status_raw",
                    "retirement_fraction",
                ]
            ],
            forecast.status_probabilities,
        )
        event = {
            **event_info[event_key],
            "training_events": int(prior["event_key"].nunique()),
            "terminal_calibration_applied": calibration_applied,
            "same_product_selection_evidence": bool(
                selected["same_product_selection_evidence"]
            ),
            "selected_model_id": str(selected["selected_model_id"]),
            "challenger_selected_on_selection": bool(
                selected["challenger_selected"]
            ),
            "same_product_calibration_evidence": bool(
                calibration_locks[horizon_value][
                    "same_product_calibration_evidence"
                ]
            ),
            "plackett_luce_temperature": float(
                selected["plackett_luce_temperature"]
            ),
            "order_residual_weight": float(selected["order_residual_weight"]),
            "baseline_mae": float((baseline_position - actual_position).abs().mean()),
            "candidate_mae": float((candidate_position - actual_position).abs().mean()),
            "selected_mae": float((selected_position - actual_position).abs().mean()),
            "global_meal_mae": float(
                (global_meal_position - actual_position).abs().mean()
            ),
            "dns_constrained_meal_mae": float(
                (candidate_position - actual_position).abs().mean()
            ),
            "dns_constrained_minus_global_meal_mae": float(
                (candidate_position - actual_position).abs().mean()
                - (global_meal_position - actual_position).abs().mean()
            ),
            "baseline_kendall": float(baseline_position.corr(actual_position, method="kendall")),
            "candidate_kendall": float(candidate_position.corr(actual_position, method="kendall")),
            "selected_kendall": float(
                selected_position.corr(actual_position, method="kendall")
            ),
            "baseline_status_brier": baseline_brier,
            "candidate_status_brier": candidate_brier,
            "selected_status_brier": selected_brier,
            "baseline_status_log_loss": baseline_log_loss,
            "candidate_status_log_loss": candidate_log_loss,
            "selected_status_log_loss": selected_log_loss,
            "selected_status_probability_source": (
                selected_status_probability_source
            ),
            "candidate_status_multiclass_brier": status_evaluation["multiclass_brier"],
            "candidate_status_multiclass_log_loss": status_evaluation["multiclass_log_loss"],
            "candidate_status_terminal_ece": status_evaluation[
                "terminal_expected_calibration_error"
            ],
            "candidate_exact_reason_log_loss": status_evaluation[
                "exact_reason_log_loss"
            ],
            "candidate_exact_reason_rows": status_evaluation["exact_reason_rows"],
            "candidate_coarse_terminal_rows": status_evaluation[
                "coarse_terminal_rows"
            ],
            "candidate_retirement_timing_rows": status_evaluation[
                "retirement_timing_rows"
            ],
            "candidate_retirement_fraction_mae": status_evaluation[
                "retirement_fraction_mae"
            ],
            "candidate_retirement_fraction_mae_by_cause": status_evaluation[
                "retirement_fraction_mae_by_cause"
            ],
            "candidate_reason_recall": status_evaluation["reason_recall"],
            "candidate_terminal_calibration": status_evaluation["terminal_calibration"],
            "actual_terminal_rate": float(actual_terminal.mean()),
            "baseline_mean_terminal_probability": float(
                np.mean(baseline_probability)
            ),
            "candidate_mean_terminal_probability": float(candidate_terminal.mean()),
            "selected_mean_terminal_probability": float(
                np.mean(selected_terminal)
            ),
            "candidate_legal_permutation": (
                sorted(candidate_position.astype(int).tolist())
                == list(range(1, len(scored) + 1))
            ),
            "selected_legal_permutation": (
                sorted(selected_position.astype(int).tolist())
                == list(range(1, len(scored) + 1))
            ),
            "legal_permutation": (
                sorted(selected_position.astype(int).tolist())
                == list(range(1, len(scored) + 1))
            ),
            "entrant_coverage": float(len(scored) / len(current)),
        }
        event["delta_candidate_minus_baseline"] = event["candidate_mae"] - event["baseline_mae"]
        event["delta_selected_minus_baseline"] = (
            event["selected_mae"] - event["baseline_mae"]
        )
        events.append(event)
        for row in scored.to_dict(orient="records"):
            prediction_rows.append({**event_info[event_key], **row})

    if not events:
        raise ValueError("no Race evaluation events were scored")
    audit_events = [
        event for event in events if int(event["event_key"]) in audit_event_keys
    ]
    if len(audit_events) < 2:
        raise ValueError("declared audit partition has fewer than two scored events")

    def promotion_for(group: Sequence[dict[str, Any]]) -> dict[str, Any]:
        decision = evaluate_race_promotion(
            [
                EventError(
                    str(event["event_key"]),
                    float(event["baseline_mae"]),
                    float(event["candidate_mae"]),
                    (
                        "sprint"
                        if "sprint" in str(event["event_format"]).lower()
                        else "standard"
                    ),
                )
                for event in group
            ],
            baseline_kendall=_mean(group, "baseline_kendall"),
            candidate_kendall=_mean(group, "candidate_kendall"),
            baseline_status_brier=_mean(group, "baseline_status_brier"),
            candidate_status_brier=_mean(group, "candidate_status_brier"),
            baseline_status_log_loss=_mean(group, "baseline_status_log_loss"),
            candidate_status_log_loss=_mean(group, "candidate_status_log_loss"),
            entrant_coverage=min(float(event["entrant_coverage"]) for event in group),
            all_classifications_legal=all(
                bool(event["candidate_legal_permutation"]) for event in group
            ),
            bootstrap_samples=int(bootstrap_samples),
            seed=int(seed),
        )
        return decision.to_payload()

    promotion_by_horizon: dict[str, dict[str, Any]] = {}
    for horizon_value in (
        RacePredictionHorizon.POST_GRID_PRE_RACE.value,
        RacePredictionHorizon.POST_QUALIFYING_PRE_GRID.value,
    ):
        group = [
            event
            for event in audit_events
            if event["information_horizon"] == horizon_value
        ]
        protocol_reasons = _same_product_promotion_blockers(
            audit_event_count=len(group),
            same_product_selection_evidence=bool(
                selected_by_horizon[horizon_value][
                    "same_product_selection_evidence"
                ]
            ),
            same_product_calibration_evidence=bool(
                calibration_locks[horizon_value][
                    "same_product_calibration_evidence"
                ]
            ),
            selection_event_count=int(
                selected_by_horizon[horizon_value]["event_count"]
            ),
            calibration_event_count=len(
                calibration_locks[horizon_value]["event_keys"]
            ),
            challenger_selected_on_selection=bool(
                selected_by_horizon[horizon_value]["challenger_selected"]
            ),
            point_in_time_input_snapshot_verified=all(
                bool(
                    event_info[int(event["event_key"])][
                        "qualifying_snapshot_provenance"
                    ]["first_seen_verified"]
                )
                for event in group
            ),
        )
        if protocol_reasons:
            promotion_by_horizon[horizon_value] = {
                "mode": "race_final_position",
                "promoted": False,
                "status": "not_evaluated",
                "reasons": protocol_reasons,
                "event_count": len(group),
                "information_horizon": horizon_value,
                "diagnostic_only": True,
            }
        else:
            promotion_by_horizon[horizon_value] = promotion_for(group)
    post_grid_horizon = RacePredictionHorizon.POST_GRID_PRE_RACE.value
    primary_horizon = (
        post_grid_horizon
        if any(
            event["information_horizon"] == post_grid_horizon
            for event in audit_events
        )
        else RacePredictionHorizon.POST_QUALIFYING_PRE_GRID.value
    )
    promotion = promotion_by_horizon[primary_horizon]

    implementation_paths = [
        Path(__file__).resolve(),
        root / "research/projects/F1/rising_qualification_prediction/Python/capture_fia_final_grid_snapshot.py",
        root / "packages/f1/domain/starting_grid.py",
        root / "packages/f1/data/providers/local_weekends.py",
        root / "packages/f1/models/pre_race/joint.py",
        root / "packages/f1/models/pre_race/ranking.py",
        root / "packages/f1/models/pre_race/survival.py",
        root / "packages/f1/models/pre_race/status.py",
        root / "packages/f1/models/pre_race/evaluate.py",
        root / "packages/f1/features/race.py",
        root / "packages/f1/orchestration/non_live_validation.py",
    ]
    history_event_year = pd.to_numeric(
        history.get("event_key", pd.Series(index=history.index, dtype=float)),
        errors="coerce",
    ).floordiv(100)
    prior_season_training_rows = int(
        history_event_year.notna().to_numpy().sum()
        - history_event_year.eq(int(target_year)).to_numpy().sum()
    )
    if same_season_only and prior_season_training_rows != 0:
        raise RuntimeError("same-season Race mode retained prior-season training rows")
    payload: dict[str, Any] = {
        "schema_version": RACE_BACKTEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": "race_final_position",
        "target": "official_terminal_race_classification_and_status",
        "protocol": {
            "training": (
                "strictly_earlier_complete_events_in_target_year"
                if same_season_only
                else "legacy_strictly_earlier_cross_season_complete_events"
            ),
            "same_season_only": bool(same_season_only),
            "legacy_cross_season_diagnostic_opt_in": bool(not same_season_only),
            "partition_mode": (
                "target_year_2_development_adaptive_up_to_4_selection_"
                "and_4_calibration_reserving_at_least_3_audit"
                if same_season_only
                else "legacy_year_membership_diagnostic"
            ),
            "target_year": int(target_year),
            "minimum_prior_event_gate": int(minimum_prior_events),
            "prior_season_training_rows": prior_season_training_rows,
            "older_season_policy": (
                "forbidden_for_terminal_risk_and_conditional_order"
                if same_season_only
                else (
                    "legacy diagnostic: older seasons inform partial-pooled reliability; "
                    "conditional order remains current-season"
                )
            ),
            "years_loaded": list(loaded_years),
            "evaluation_years": sorted(evaluation_set),
            "evaluation_event_keys": sorted(evaluation_event_keys),
            "event_partitions": partition_events,
            "within_run_chronological_freeze_enforced": True,
            "prospective_development_evidence": False,
            "evidence_role": (
                "postdevelopment_replay_diagnostic_after_prior_R7_R9_inspection"
            ),
            "excluded_incomplete_target_events": excluded_incomplete_target_events,
            "driver_identity_semantics": {
                "driver_id": "uppercase_fia_driver_abbreviation",
                "provider_driver_id": "provider_or_car_number_from_qualifying",
                "car_number": "event_specific_car_number",
                "migration_from_v2": (
                    "v2 predictions used provider car numbers as driver_id"
                ),
            },
            "event_partition_issues": list(partition_issues),
            "audit_oof_qualifying_prior_verified": True,
            "audit_power_unit_identity_verified": True,
            "promotion_audit_event_keys": sorted(audit_event_keys),
            "promotion_audit_years": sorted(
                {event_key // 100 for event_key in audit_event_keys}
            ),
            "promotion_primary_horizon": primary_horizon,
            "hyperparameter_lock": {
                "locked_after_event": max(partition_events["selection"]),
                "locked_before_event": min(partition_events["calibration"]),
                "lock_uses_exact_event_partition_membership": True,
                "selection_partition_only": True,
                "constant_for_calibration_and_audit": True,
                "within_current_replay_audit_targets_read_for_tuning": False,
                "selection_objective": "mean_complete_event_final_position_mae",
                "selection_simulations": bounded_selection_simulations,
                "candidate_results": selection_trace,
                "selected_by_information_horizon": selected_by_horizon,
                "cross_horizon_parameter_reuse": False,
                "model_fit_cache": {
                    "key": ["event_key"],
                    "prediction_time_parameters_excluded": [
                        "order_residual_weight",
                        "plackett_luce_temperature",
                        "forecast_simulations",
                        "seed",
                    ],
                    "calibrated_models_cached": False,
                    "entries": len(fitted_model_cache),
                    "model_fits": model_fit_count,
                    "cache_hits": model_fit_cache_hits,
                },
                "hazard_covariate_l2_c": TerminalHazardConfig().covariate_l2_c,
                "hazard_covariate_z_clip": TerminalHazardConfig().covariate_z_clip,
                "conditional_order_fit": {
                    "likelihood": (
                        "bradley_terry_residual_conditional_on_fixed_grid_offset"
                    ),
                    "pair_orientation": "one_orientation_per_unordered_pair",
                    "event_balance": "within_event_pair_weights_sum_to_one",
                    "optimizer": "deterministic_bounded_projected_gradient",
                    "coefficient_bound": ConditionalOrderConfig().coefficient_bound,
                    "cold_start_event_k": ConditionalOrderConfig().cold_start_event_k,
                    "cold_start_formula": (
                        "configured_residual_weight*n_training_events/"
                        "(n_training_events+cold_start_event_k)"
                    ),
                    "no_same_horizon_residual_weight": (
                        _NO_SAME_HORIZON_ORDER_RESIDUAL_WEIGHT
                    ),
                },
                "qualifying_prior_ridge_alpha": 10.0,
            },
            "probability_calibration_lock": {
                "backend": "raw_post_shared_shock_joint_marginal_fail_closed",
                "status": (
                    "blocked_requires_simulation_in_loop_or_marginal_preserving_copula"
                ),
                "calibration_fit_probability_source": None,
                "calibration_application_probability_source": None,
                "scored_probability_source": (
                    "empirical_post_shared_shock_joint_samples"
                ),
                "zero_shock_platt_rejected": True,
                "rejection_reason": (
                    "mean_one_hazard_shocks_do_not_preserve_nonlinear_multi_bin_"
                    "terminal_probability"
                ),
                "calibration_partition_only": True,
                "locked_after_event": max(partition_events["calibration"]),
                "locked_before_event": min(partition_events["audit"]),
                "lock_uses_exact_event_partition_membership": True,
                "within_current_replay_audit_targets_read_for_calibration": False,
                "calibration_by_information_horizon": calibration_locks,
                "cross_horizon_calibration_reuse": False,
            },
            "joint_simulation_runtime": {
                "deterministic_row_hazards_prepared_once_per_forecast": True,
                "row_hazards_recomputed_per_simulation": False,
            },
            "inference_target_boundary": {
                "feature_roster": (
                    "retrospective_post_session_qualifying_snapshot_or_"
                    "immutable_final_grid_plus_pre_race_practice"
                ),
                "qualifying_snapshot_first_seen_verified": all(
                    bool(info["qualifying_snapshot_provenance"]["first_seen_verified"])
                    for info in event_info.values()
                ),
                "qualifying_snapshot_time_semantics": sorted(
                    {
                        str(info["qualifying_snapshot_provenance"]["time_semantics"])
                        for info in event_info.values()
                    }
                ),
                "race_result_read_after_inference_freeze": True,
                "race_result_fields_allowed": list(target_columns),
                "race_result_team_or_power_unit_fallback": False,
            },
            "horizons": sorted(
                {str(event["information_horizon"]) for event in events}
            ),
            "baseline_order": "legal_full_grid_permutation_at_matching_horizon",
            "baseline_status": (
                "causal_partial_pooled_rolling_binary_hazard_by_team_power_unit_driver"
            ),
            "position_output_contract": {
                "candidate_column": "candidate_predicted_position",
                "selected_column": "selected_predicted_position",
                "public_compatibility_alias": "predicted_position",
                "public_alias_semantics": "exact_copy_of_selected_predicted_position",
                "candidate_probability_columns": "p_position_1_through_field_size",
                "candidate_probability_promotion_status": "diagnostic_only_not_promoted",
            },
            "known_model_limitations": {
                "terminal_cause_factorization": (
                    "coarse non_classified remains an explicit competing cause; "
                    "binary terminal prediction followed by observed-cause "
                    "factorization remains future work"
                )
            },
            "final_grid_snapshot_events": int(
                sum(bool(event["final_grid_snapshot_used"]) for event in events)
            ),
            "simulations": int(simulations),
            "parameters_by_information_horizon": selected_by_horizon,
        },
        "aggregate": {
            "population": "all_scored_selection_calibration_and_audit_events",
            **_aggregate_race_events(events),
            "by_year": {
                str(year): _aggregate_race_events(
                    [event for event in events if int(event["year"]) == year]
                )
                for year in sorted({int(event["year"]) for event in events})
            },
            "by_horizon": {
                horizon_value: _aggregate_race_events(
                    [
                        event
                        for event in events
                        if event["information_horizon"] == horizon_value
                    ]
                )
                for horizon_value in sorted(
                    {str(event["information_horizon"]) for event in events}
                )
            },
        },
        "audit_aggregate": _audit_aggregate_payload(
            events,
            audit_event_keys=sorted(audit_event_keys),
        ),
        "promotion": promotion,
        "promotion_by_horizon": promotion_by_horizon,
        "runtime": f1_model_runtime_doctor(),
        "events": events,
        "predictions": prediction_rows,
        "input_manifest": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in sorted(inputs)
        ],
        "implementation_manifest": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in implementation_paths
        ],
        "artifact_contract": {
            "result_hash_algorithm": "sha256_canonical_json",
            "result_hash_excludes_only": "result_sha256",
            "all_event_outputs_bound_by_result_hash": True,
        },
    }
    configuration_manifest = {
        "weekends_dir": str(weekends_dir),
        "years": list(loaded_years),
        "evaluation_years": sorted(evaluation_set),
        "same_season_only": bool(same_season_only),
        "legacy_cross_season_diagnostic_opt_in": bool(not same_season_only),
        "target_year": int(target_year),
        "minimum_prior_event_gate": int(minimum_prior_events),
        "event_partitions": partition_events,
        "legacy_partition_year_arguments_ignored": bool(same_season_only),
        "development_years": (
            [] if same_season_only else sorted(partition_years["development"])
        ),
        "selection_years": (
            [] if same_season_only else sorted(partition_years["selection"])
        ),
        "calibration_years": (
            [] if same_season_only else sorted(partition_years["calibration"])
        ),
        "audit_years": (
            [] if same_season_only else sorted(partition_years["audit"])
        ),
        "simulations": int(simulations),
        "selection_simulations": int(selection_simulations),
        "bootstrap_samples": int(bootstrap_samples),
        "seed": int(seed),
        "temperature_candidates": list(temperatures),
        "order_residual_candidates": list(residual_weights),
    }
    payload["configuration_manifest"] = configuration_manifest
    payload["manifest_hashes"] = {
        "data_input_manifest_sha256": _canonical_json_sha256(
            payload["input_manifest"]
        ),
        "implementation_manifest_sha256": _canonical_json_sha256(
            payload["implementation_manifest"]
        ),
        "configuration_manifest_sha256": _canonical_json_sha256(
            configuration_manifest
        ),
        "protocol_sha256": _canonical_json_sha256(payload["protocol"]),
    }
    return _attach_result_sha256(payload)


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
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
            "diagnostic reproduction only: enable legacy year-based cross-season "
            "training and partition arguments"
        ),
    )
    parser.add_argument(
        "--development-years",
        type=_csv_ints,
        default=(2022, 2023),
        help="legacy cross-season diagnostic only",
    )
    parser.add_argument(
        "--selection-years",
        type=_csv_ints,
        default=(2024,),
        help="legacy cross-season diagnostic only",
    )
    parser.add_argument(
        "--calibration-years",
        type=_csv_ints,
        default=(2025,),
        help="legacy cross-season diagnostic only",
    )
    parser.add_argument(
        "--audit-years",
        type=_csv_ints,
        default=(2026,),
        help="legacy cross-season diagnostic only",
    )
    parser.add_argument("--simulations", type=int, default=2_000)
    parser.add_argument(
        "--temperature-candidates",
        type=_csv_floats,
        default=(0.08, 0.12, 0.18, 0.25, 0.35),
        help="bounded grid selected only on the declared selection partition",
    )
    parser.add_argument(
        "--order-residual-candidates",
        type=_csv_floats,
        default=(0.0, 0.65, 1.0, 1.5, 2.0),
        help="bounded grid selected only on the declared selection partition",
    )
    parser.add_argument("--selection-simulations", type=int, default=400)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            _root()
            / "artifacts/backtests/f1/race_final_position/"
            "survival_order_same_season_v1.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(
        weekends_dir=args.weekends_dir.expanduser().resolve(),
        years=args.years,
        evaluation_years=args.evaluation_years,
        simulations=args.simulations,
        temperature_candidates=args.temperature_candidates,
        order_residual_candidates=args.order_residual_candidates,
        selection_simulations=args.selection_simulations,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        development_years=args.development_years,
        selection_years=args.selection_years,
        calibration_years=args.calibration_years,
        audit_years=args.audit_years,
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


# Suggested commit name: feat(f1-race): add survival-aware order walk-forward evidence
