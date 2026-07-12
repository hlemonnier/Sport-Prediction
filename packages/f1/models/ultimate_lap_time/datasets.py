"""Dataset builders for Ultimate Lap-Time telemetry models."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from packages.f1.models.ultimate_lap_time.schemas import (
    DistanceNormalizedTelemetryTensor,
    UltimateLapMetadata,
    UltimateLapSplitKey,
    UltimateLapTargets,
    UltimateLapTelemetryBatch,
    UltimateLapTelemetryExample,
    UltimateLapTelemetryInput,
    assert_split_fields_are_leakage_safe,
    summarize_target_quantile_diagnostics,
)


DEFAULT_DISTANCE_BINS = 200
DEFAULT_MINIMUM_DISTANCE_COVERAGE = 0.95
DEFAULT_SPLIT_FIELDS: tuple[str, ...] = ("season", "event_key", "circuit_id", "session")
DEFAULT_TELEMETRY_CHANNELS: tuple[str, ...] = (
    "Speed",
    "Throttle",
    "Brake",
    "RPM",
    "nGear",
    "DRS",
)
DISTANCE_COLUMNS: tuple[str, ...] = (
    "distance",
    "Distance",
    "distance_m",
    "DistanceMeters",
    "lap_distance",
    "LapDistance",
)
EXPECTED_LAP_DISTANCE_COLUMNS: tuple[str, ...] = (
    "expected_lap_distance_m",
    "expected_lap_distance",
    "circuit_length_m",
    "track_length_m",
)
TIME_COLUMNS: tuple[str, ...] = ("time", "Time", "date", "Date", "timestamp", "SessionTime")
METADATA_COLUMNS: dict[str, tuple[str, ...]] = {
    "season": ("season", "year", "Season", "Year"),
    "event_key": ("event_key", "meeting_key", "weekend_key", "race_id", "event_id"),
    "circuit_id": ("circuit_id", "track_id", "event_name_norm", "event_name", "MeetingName"),
    "driver_id": ("driver_id", "driver_number", "DriverNumber", "Driver", "driver"),
    "team_id": ("team_id", "team_name", "constructor_name", "constructor", "TeamName", "Team", "team"),
    "session": ("session", "session_name", "SessionName", "SessionType"),
    "target_session": ("target_session", "target_session_name"),
    "feature_as_of": ("feature_as_of", "telemetry_as_of"),
    "target_as_of": ("target_as_of", "target_session_as_of"),
    "lap_number": ("lap_number", "LapNumber", "lap"),
    "source": ("source", "data_source"),
    "split_name": ("split_name", "split", "dataset_split"),
    "fold": ("fold", "cv_fold"),
}
TARGET_AND_PREDICTION_COLUMNS: frozenset[str] = frozenset(
    {
        "ideal_lap_time_seconds",
        "theoretical_sector_floor_seconds",
        "achievable_session_end_lap_time_seconds",
        "session_end_lap_time_seconds",
        "achievable_lap_time_seconds",
        "target_contract",
        "target_kind",
        "target_semantics",
        "quantile_target_semantics",
        "lap_time_seconds",
        "lap_duration",
        "LapTime",
        "lap_time",
        "duration",
        "sector1_seconds",
        "sector2_seconds",
        "sector3_seconds",
        "duration_sector_1",
        "duration_sector_2",
        "duration_sector_3",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
        "sector_1_time",
        "sector_2_time",
        "sector_3_time",
        "s1",
        "s2",
        "s3",
        "p05_target",
        "p50_target",
        "p90_target",
        "target_p05",
        "target_p50",
        "target_p90",
        "lap_p05",
        "lap_p50",
        "lap_p90",
        "prediction",
        "predicted_lap_time_seconds",
        "ultimate_lap_time_seconds",
    }
)


def _record_from_any(record: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    if isinstance(record, pd.Series):
        return record.to_dict()
    return dict(record)


def _first_present(record: Mapping[str, Any], candidates: Sequence[str], default: Any = None) -> Any:
    for candidate in candidates:
        if candidate in record and record[candidate] is not None:
            value = record[candidate]
            if isinstance(value, float) and np.isnan(value):
                continue
            return value
    return default


def _clean_str(value: Any, default: str | None = None) -> str:
    if value is None:
        if default is None:
            raise ValueError("required metadata value is missing")
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        if default is None:
            raise ValueError("required metadata value is missing")
        return default
    return text


def _find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        match = lowered.get(candidate.lower())
        if match is not None:
            return match
    return None


def _numeric_channel_columns(frame: pd.DataFrame, channel_names: Sequence[str] | None) -> tuple[str, ...]:
    if channel_names is not None:
        missing = [name for name in channel_names if name not in frame.columns]
        if missing:
            raise ValueError(f"telemetry is missing requested channels: {missing}")
        return tuple(channel_names)

    preferred = tuple(name for name in DEFAULT_TELEMETRY_CHANNELS if name in frame.columns)
    if preferred:
        return preferred

    excluded = set(DISTANCE_COLUMNS) | set(TIME_COLUMNS) | set(TARGET_AND_PREDICTION_COLUMNS)
    numeric_columns: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column]):
            numeric_columns.append(str(column))
    if not numeric_columns:
        raise ValueError("telemetry dataframe has no numeric channel columns")
    return tuple(numeric_columns)


def _resample_channel_matrix(values: np.ndarray, distance_bins: int) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("telemetry matrix must be two-dimensional")
    if values.shape[1] == distance_bins:
        return values.astype(np.float32, copy=False)
    old_x = np.linspace(0.0, 1.0, num=values.shape[1], dtype=float)
    new_x = np.linspace(0.0, 1.0, num=distance_bins, dtype=float)
    resampled = np.vstack([np.interp(new_x, old_x, row) for row in values])
    return resampled.astype(np.float32, copy=False)


def build_distance_normalized_telemetry(
    telemetry: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    *,
    distance_bins: int = DEFAULT_DISTANCE_BINS,
    channel_names: Sequence[str] | None = None,
    distance_column: str | None = None,
    already_distance_normalized: bool = False,
    expected_lap_distance: float | None = None,
    minimum_distance_coverage: float = DEFAULT_MINIMUM_DISTANCE_COVERAGE,
) -> DistanceNormalizedTelemetryTensor:
    """Build a finite channels x distance_bins telemetry tensor.

    DataFrame telemetry must have a distance column unless the caller explicitly
    marks it as already distance-normalized. Physical-distance payloads must
    declare the expected lap distance so truncated laps cannot be stretched to
    a full lap. Raw arrays are treated as already distance-normalized matrices.
    """

    bins = int(distance_bins)
    if bins <= 1:
        raise ValueError("distance_bins must be greater than one")
    minimum_coverage = float(minimum_distance_coverage)
    if not 0.5 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_distance_coverage must be between 0.5 and 1.0")

    if isinstance(telemetry, pd.DataFrame):
        if telemetry.empty:
            raise ValueError("telemetry dataframe is empty")
        channels = _numeric_channel_columns(telemetry, channel_names)
        distance_col = distance_column or _find_column(telemetry, DISTANCE_COLUMNS)
        if distance_col is None and not already_distance_normalized:
            time_col = _find_column(telemetry, TIME_COLUMNS)
            if time_col is not None:
                raise ValueError(
                    "telemetry appears time-indexed; provide a distance column or pre-normalized tensor"
                )
            raise ValueError("telemetry dataframe requires a distance column unless already_distance_normalized=True")

        channel_values = telemetry.loc[:, list(channels)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if distance_col is None:
            if not np.isfinite(channel_values).all():
                raise ValueError("telemetry channel values must be finite")
            return DistanceNormalizedTelemetryTensor(
                values=_resample_channel_matrix(channel_values.T, bins),
                channel_names=channels,
                distance_coverage=1.0,
            )

        if expected_lap_distance is None:
            attrs_distance = telemetry.attrs.get("expected_lap_distance_m")
            expected_lap_distance = attrs_distance if attrs_distance is not None else None
        try:
            expected_distance = float(expected_lap_distance)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "expected_lap_distance is required to validate physical-distance telemetry coverage"
            ) from exc
        if not np.isfinite(expected_distance) or expected_distance <= 0.0:
            raise ValueError("expected_lap_distance must be a positive finite distance")

        distance = pd.to_numeric(telemetry[distance_col], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(distance) & np.isfinite(channel_values).all(axis=1)
        if finite_mask.sum() < 2:
            raise ValueError("telemetry requires at least two finite distance samples")
        distance = distance[finite_mask]
        channel_values = channel_values[finite_mask]
        order = np.argsort(distance)
        distance = distance[order]
        channel_values = channel_values[order]
        unique_distance, unique_indices = np.unique(distance, return_index=True)
        if unique_distance.size < 2 or float(unique_distance[-1]) <= float(unique_distance[0]):
            raise ValueError("distance samples must span a positive lap distance")
        channel_values = channel_values[unique_indices]
        distance_start = float(unique_distance[0])
        distance_end = float(unique_distance[-1])
        covered_span = float(max(0.0, min(distance_end, expected_distance) - max(distance_start, 0.0)))
        coverage = float(covered_span / expected_distance)
        start_limit = float((1.0 - minimum_coverage) * expected_distance)
        if distance_start > start_limit or distance_end < minimum_coverage * expected_distance or coverage < minimum_coverage:
            raise ValueError(
                "incomplete lap distance coverage: "
                f"start={distance_start:.3f}, end={distance_end:.3f}, "
                f"expected={expected_distance:.3f}, coverage={coverage:.3f}, "
                f"minimum={minimum_coverage:.3f}"
            )
        if distance_end > expected_distance * 1.05:
            raise ValueError("telemetry distance exceeds expected lap distance by more than 5%")
        target_distance = np.linspace(0.0, expected_distance, num=bins, dtype=float)
        values = np.vstack(
            [np.interp(target_distance, unique_distance, channel_values[:, idx]) for idx in range(channel_values.shape[1])]
        )
        return DistanceNormalizedTelemetryTensor(
            values=values,
            channel_names=channels,
            raw_distance_start=distance_start,
            raw_distance_end=distance_end,
            expected_lap_distance=expected_distance,
            distance_coverage=coverage,
        )

    values = np.asarray(telemetry, dtype=float)
    if values.ndim != 2:
        raise ValueError("telemetry array must have shape channels x distance_bins")
    channels = tuple(channel_names or [f"channel_{idx}" for idx in range(values.shape[0])])
    if len(channels) != values.shape[0] and len(channels) == values.shape[1]:
        values = values.T
    if len(channels) != values.shape[0]:
        raise ValueError("channel_names length must match one telemetry array dimension")
    if not np.isfinite(values).all():
        raise ValueError("telemetry array values must be finite")
    return DistanceNormalizedTelemetryTensor(
        values=_resample_channel_matrix(values, bins),
        channel_names=channels,
        distance_coverage=1.0,
    )


def build_split_key(
    record: Mapping[str, Any] | pd.Series,
    *,
    split_fields: Sequence[str] = DEFAULT_SPLIT_FIELDS,
) -> UltimateLapSplitKey:
    """Build a no-leakage split key from pre-target identity fields."""

    assert_split_fields_are_leakage_safe(split_fields)
    row = _record_from_any(record)
    field_set = set(split_fields)
    if any(field not in field_set for field in ("event_key", "circuit_id", "session")):
        # The contract needs these fields regardless of the external grouping
        # choice, otherwise leakage checks cannot join weekend/session payloads.
        field_set.update(("event_key", "circuit_id", "session"))
    return UltimateLapSplitKey(
        season=_first_present(row, METADATA_COLUMNS["season"]) if "season" in field_set else None,
        event_key=_clean_str(_first_present(row, METADATA_COLUMNS["event_key"])),
        circuit_id=_clean_str(_first_present(row, METADATA_COLUMNS["circuit_id"])),
        session=_clean_str(_first_present(row, METADATA_COLUMNS["session"], default="unknown")),
        split_name=_first_present(row, METADATA_COLUMNS["split_name"]),
        fold=_first_present(row, METADATA_COLUMNS["fold"]),
    )


def build_metadata(
    record: Mapping[str, Any] | pd.Series,
    *,
    split_fields: Sequence[str] = DEFAULT_SPLIT_FIELDS,
) -> UltimateLapMetadata:
    """Build metadata required by the Ultimate Lap-Time dataset contract."""

    row = _record_from_any(record)
    split_key = build_split_key(row, split_fields=split_fields)
    return UltimateLapMetadata(
        season=_first_present(row, METADATA_COLUMNS["season"]),
        event_key=_clean_str(_first_present(row, METADATA_COLUMNS["event_key"])),
        circuit_id=_clean_str(_first_present(row, METADATA_COLUMNS["circuit_id"])),
        driver_id=_clean_str(_first_present(row, METADATA_COLUMNS["driver_id"])),
        team_id=_clean_str(_first_present(row, METADATA_COLUMNS["team_id"], default="unknown")),
        session=_clean_str(_first_present(row, METADATA_COLUMNS["session"], default="unknown")),
        split_key=split_key,
        lap_number=_first_present(row, METADATA_COLUMNS["lap_number"]),
        source=_first_present(row, METADATA_COLUMNS["source"]),
        target_session=_first_present(row, METADATA_COLUMNS["target_session"]),
        feature_as_of=_first_present(row, METADATA_COLUMNS["feature_as_of"]),
        target_as_of=_first_present(row, METADATA_COLUMNS["target_as_of"]),
    )


def extract_static_features(record: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    """Extract scalar non-target features from a lap/session record."""

    row = _record_from_any(record)
    excluded = set(TARGET_AND_PREDICTION_COLUMNS) | set(DISTANCE_COLUMNS) | set(TIME_COLUMNS)
    for candidates in METADATA_COLUMNS.values():
        excluded.update(candidates)
    static: dict[str, Any] = {}
    for key, value in row.items():
        if key in excluded:
            continue
        if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)) or value is None:
            if isinstance(value, float) and np.isnan(value):
                static[str(key)] = None
            else:
                static[str(key)] = value
    return static


def build_ultimate_lap_example(
    record: Mapping[str, Any] | pd.Series,
    telemetry: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    *,
    distance_bins: int = DEFAULT_DISTANCE_BINS,
    channel_names: Sequence[str] | None = None,
    split_fields: Sequence[str] = DEFAULT_SPLIT_FIELDS,
    already_distance_normalized: bool = False,
    expected_lap_distance: float | None = None,
    minimum_distance_coverage: float = DEFAULT_MINIMUM_DISTANCE_COVERAGE,
) -> UltimateLapTelemetryExample:
    """Build and validate one model-ready telemetry example."""

    row = _record_from_any(record)
    expected_distance = expected_lap_distance
    if expected_distance is None:
        expected_distance = _first_present(row, EXPECTED_LAP_DISTANCE_COLUMNS)
    return UltimateLapTelemetryExample(
        telemetry=build_distance_normalized_telemetry(
            telemetry,
            distance_bins=distance_bins,
            channel_names=channel_names,
            already_distance_normalized=already_distance_normalized,
            expected_lap_distance=expected_distance,
            minimum_distance_coverage=minimum_distance_coverage,
        ),
        static_features=extract_static_features(row),
        targets=UltimateLapTargets.from_mapping(row),
        metadata=build_metadata(row, split_fields=split_fields),
    )


def build_ultimate_lap_inference_input(
    record: Mapping[str, Any] | pd.Series,
    telemetry: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    *,
    distance_bins: int = DEFAULT_DISTANCE_BINS,
    channel_names: Sequence[str] | None = None,
    split_fields: Sequence[str] = DEFAULT_SPLIT_FIELDS,
    already_distance_normalized: bool = False,
    expected_lap_distance: float | None = None,
    minimum_distance_coverage: float = DEFAULT_MINIMUM_DISTANCE_COVERAGE,
) -> UltimateLapTelemetryInput:
    """Build one model-ready input without constructing or inventing targets."""

    row = _record_from_any(record)
    expected_distance = expected_lap_distance
    if expected_distance is None:
        expected_distance = _first_present(row, EXPECTED_LAP_DISTANCE_COLUMNS)
    return UltimateLapTelemetryInput(
        telemetry=build_distance_normalized_telemetry(
            telemetry,
            distance_bins=distance_bins,
            channel_names=channel_names,
            already_distance_normalized=already_distance_normalized,
            expected_lap_distance=expected_distance,
            minimum_distance_coverage=minimum_distance_coverage,
        ),
        static_features=extract_static_features(row),
        metadata=build_metadata(row, split_fields=split_fields),
    )


def _coerce_records(records: pd.DataFrame | Iterable[Mapping[str, Any] | pd.Series]) -> list[Mapping[str, Any]]:
    if isinstance(records, pd.DataFrame):
        return records.to_dict("records")
    return [_record_from_any(record) for record in records]


def _telemetry_lookup_key(record: Mapping[str, Any], fields: Sequence[str]) -> tuple[str, ...]:
    return tuple(_clean_str(_first_present(record, METADATA_COLUMNS.get(field, (field,)))) for field in fields)


def build_ultimate_lap_dataset(
    records: pd.DataFrame | Iterable[Mapping[str, Any] | pd.Series],
    *,
    telemetry: Sequence[pd.DataFrame | np.ndarray | Sequence[Sequence[float]]] | None = None,
    telemetry_by_key: Mapping[tuple[str, ...] | str, pd.DataFrame | np.ndarray | Sequence[Sequence[float]]] | None = None,
    telemetry_key_fields: Sequence[str] = ("event_key", "driver_id", "lap_number"),
    distance_bins: int = DEFAULT_DISTANCE_BINS,
    channel_names: Sequence[str] | None = None,
    split_fields: Sequence[str] = DEFAULT_SPLIT_FIELDS,
    already_distance_normalized: bool = False,
    expected_lap_distance: float | None = None,
    minimum_distance_coverage: float = DEFAULT_MINIMUM_DISTANCE_COVERAGE,
) -> list[UltimateLapTelemetryExample]:
    """Build examples from lap/session records and matching telemetry payloads."""

    rows = _coerce_records(records)
    if not rows:
        return []

    if telemetry is None and telemetry_by_key is None:
        raise ValueError("telemetry sequence or telemetry_by_key mapping is required for non-empty records")

    if telemetry is not None and len(telemetry) != len(rows):
        raise ValueError("telemetry sequence length must match records length")

    examples: list[UltimateLapTelemetryExample] = []
    for idx, row in enumerate(rows):
        if telemetry is not None:
            telemetry_payload = telemetry[idx]
        else:
            assert telemetry_by_key is not None
            key = _telemetry_lookup_key(row, telemetry_key_fields)
            telemetry_payload = telemetry_by_key.get(key)
            if telemetry_payload is None:
                telemetry_payload = telemetry_by_key.get("|".join(key))
            if telemetry_payload is None:
                raise KeyError(f"missing telemetry for key {key}")
        examples.append(
            build_ultimate_lap_example(
                row,
                telemetry_payload,
                distance_bins=distance_bins,
                channel_names=channel_names,
                split_fields=split_fields,
                already_distance_normalized=already_distance_normalized,
                expected_lap_distance=expected_lap_distance,
                minimum_distance_coverage=minimum_distance_coverage,
            )
        )
    return examples


def validate_ultimate_lap_examples(examples: Sequence[UltimateLapTelemetryExample]) -> UltimateLapTelemetryBatch:
    """Validate a sequence of examples and return a batch contract."""

    return UltimateLapTelemetryBatch.from_examples(examples)


def dataset_summary(examples: Sequence[UltimateLapTelemetryExample]) -> dict[str, Any]:
    """Summarize row counts and target availability for audit reports."""

    counters: dict[str, Counter[str]] = {
        "by_season": Counter(),
        "by_circuit": Counter(),
        "by_driver": Counter(),
        "by_session": Counter(),
    }
    target_availability: defaultdict[str, int] = defaultdict(int)
    for example in examples:
        metadata = example.metadata
        counters["by_season"][metadata.season or "unknown"] += 1
        counters["by_circuit"][metadata.circuit_id] += 1
        counters["by_driver"][metadata.driver_id] += 1
        counters["by_session"][metadata.session] += 1
        for key, value in example.targets.as_dict().items():
            try:
                available = value is not None and np.isfinite(float(value))
            except (TypeError, ValueError):
                available = value is not None
            if available:
                target_availability[key] += 1

    first_shape = examples[0].telemetry.shape if examples else (0, 0)
    return {
        "row_count": int(len(examples)),
        "telemetry_shape": {"channels": int(first_shape[0]), "distance_bins": int(first_shape[1])},
        "channel_names": list(examples[0].telemetry.channel_names) if examples else [],
        "by_season": dict(counters["by_season"]),
        "by_circuit": dict(counters["by_circuit"]),
        "by_driver": dict(counters["by_driver"]),
        "by_session": dict(counters["by_session"]),
        "target_availability": dict(target_availability),
        "target_diagnostics": summarize_target_quantile_diagnostics(
            [example.targets for example in examples]
        ),
    }


def leakage_issues_for_examples(examples: Sequence[UltimateLapTelemetryExample]) -> tuple[str, ...]:
    """Detect split metadata issues that would leak across train/validation/test."""

    issues: list[str] = []
    split_names_by_key: defaultdict[str, set[str]] = defaultdict(set)
    for example in examples:
        split_name = example.metadata.split_key.split_name
        if split_name:
            split_names_by_key[example.metadata.split_key.value].add(split_name.lower())
    for split_key, split_names in split_names_by_key.items():
        if len(split_names) > 1:
            issues.append(f"split_key {split_key!r} appears in multiple splits: {sorted(split_names)}")
    return tuple(issues)


__all__ = [
    "DEFAULT_DISTANCE_BINS",
    "DEFAULT_MINIMUM_DISTANCE_COVERAGE",
    "DEFAULT_SPLIT_FIELDS",
    "DEFAULT_TELEMETRY_CHANNELS",
    "EXPECTED_LAP_DISTANCE_COLUMNS",
    "build_distance_normalized_telemetry",
    "build_metadata",
    "build_split_key",
    "build_ultimate_lap_dataset",
    "build_ultimate_lap_example",
    "build_ultimate_lap_inference_input",
    "dataset_summary",
    "extract_static_features",
    "leakage_issues_for_examples",
    "validate_ultimate_lap_examples",
]
