"""Causal baseline for the achievable session-end best lap.

This model answers the user-facing question "what best lap should this driver
actually achieve in qualifying?"  It deliberately does not reuse the
theoretical sector-floor target.  The baseline starts from the latest
target-aligned rehearsal (FP3 or Sprint Qualifying) and learns only a robust,
source-specific session-to-session shift from earlier events in the same
season.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from packages.f1.models.ultimate_lap_time.schemas import (
    ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT,
    TARGET_CONTRACT_SEMANTICS,
)


REHEARSAL_LAP_COLUMN = "rehearsal_lap_time_seconds"
ACTUAL_LAP_COLUMN = "achievable_session_end_lap_time_seconds"
REHEARSAL_SOURCE_COLUMN = "rehearsal_source"
EVENT_KEY_COLUMN = "event_key"
DRIVER_ID_COLUMN = "driver_id"
ALLOWED_REHEARSAL_SOURCES = frozenset({"practice_3", "sprint_qualifying"})
FORBIDDEN_INFERENCE_COLUMNS = frozenset(
    {
        ACTUAL_LAP_COLUMN,
        "qualifying_best_lap_time_seconds",
        "actual_lap_time",
        "lap_p05",
        "lap_p50",
        "lap_p90",
        "target",
    }
)


def _source_name(value: object) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "fp3": "practice_3",
        "practice3": "practice_3",
        "sq": "sprint_qualifying",
        "sprint_shootout": "sprint_qualifying",
    }
    return aliases.get(normalized, normalized)


def _finite_seconds(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return values.where(np.isfinite(values) & values.between(40.0, 180.0))


@dataclass(frozen=True)
class AchievableLapSourceCalibration:
    source: str
    event_keys: tuple[int, ...]
    event_shifts_seconds: tuple[float, ...]
    prequential_residuals_seconds: tuple[float, ...]

    @property
    def shift_seconds(self) -> float:
        if not self.event_shifts_seconds:
            return 0.0
        return float(np.median(np.asarray(self.event_shifts_seconds, dtype=float)))

    @property
    def event_count(self) -> int:
        return len(self.event_keys)

    def centered_residual_quantiles(self) -> tuple[float, float]:
        if not self.prequential_residuals_seconds:
            return float("nan"), float("nan")
        residuals = np.asarray(self.prequential_residuals_seconds, dtype=float)
        q05, q50, q90 = np.quantile(residuals, [0.05, 0.50, 0.90])
        return float(q05 - q50), float(q90 - q50)


@dataclass(frozen=True)
class AchievableBestLapModel:
    target_event_key: int
    calibrations: Mapping[str, AchievableLapSourceCalibration]
    min_calibration_events: int = 2
    model_name: str = "achievable_best_lap_rehearsal_shift_v1"

    @property
    def training_event_keys(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    event_key
                    for calibration in self.calibrations.values()
                    for event_key in calibration.event_keys
                }
            )
        )

    def predict(self, inputs: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(inputs, pd.DataFrame):
            raise TypeError("achievable best-lap inference inputs must be a pandas DataFrame")
        leaked = sorted(FORBIDDEN_INFERENCE_COLUMNS.intersection(map(str, inputs.columns)))
        if leaked:
            raise ValueError(f"achievable best-lap inference contains target/outcome columns: {leaked}")
        required = {EVENT_KEY_COLUMN, DRIVER_ID_COLUMN, REHEARSAL_SOURCE_COLUMN, REHEARSAL_LAP_COLUMN}
        missing = sorted(required.difference(map(str, inputs.columns)))
        if missing:
            raise ValueError(f"achievable best-lap inference is missing columns: {missing}")
        if inputs.empty:
            return pd.DataFrame(index=inputs.index)

        event_keys = pd.to_numeric(inputs[EVENT_KEY_COLUMN], errors="coerce")
        if event_keys.isna().any() or set(event_keys.astype(int).tolist()) != {int(self.target_event_key)}:
            raise ValueError("inference rows must belong only to the model target_event_key")
        if self.training_event_keys and max(self.training_event_keys) >= int(self.target_event_key):
            raise ValueError("training history is not strictly earlier than the target event")

        rehearsal = _finite_seconds(inputs, REHEARSAL_LAP_COLUMN)
        if rehearsal.isna().any():
            raise ValueError("rehearsal lap inputs must be finite seconds between 40 and 180")

        rows: list[dict[str, object]] = []
        for index, row in inputs.iterrows():
            source = _source_name(row[REHEARSAL_SOURCE_COLUMN])
            if source not in ALLOWED_REHEARSAL_SOURCES:
                raise ValueError(f"unsupported target-aligned rehearsal source: {source!r}")
            calibration = self.calibrations.get(
                source,
                AchievableLapSourceCalibration(source, (), (), ()),
            )
            p50 = float(rehearsal.loc[index] + calibration.shift_seconds)
            lower_offset, upper_offset = calibration.centered_residual_quantiles()
            p05 = float(p50 + lower_offset) if np.isfinite(lower_offset) else float("nan")
            p90 = float(p50 + upper_offset) if np.isfinite(upper_offset) else float("nan")
            if np.isfinite(p05) and np.isfinite(p90):
                p05, p50, p90 = sorted((p05, p50, p90))
            interval_status = (
                "calibrated_minimum_event_count_met"
                if calibration.event_count >= int(self.min_calibration_events)
                else "diagnostic_underpowered"
                if calibration.event_count > 0
                else "unavailable_no_same_source_history"
            )
            rows.append(
                {
                    EVENT_KEY_COLUMN: int(self.target_event_key),
                    DRIVER_ID_COLUMN: str(row[DRIVER_ID_COLUMN]),
                    REHEARSAL_SOURCE_COLUMN: source,
                    REHEARSAL_LAP_COLUMN: float(rehearsal.loc[index]),
                    "lap_p05": p05,
                    "lap_p50": p50,
                    "lap_p90": p90,
                    "session_shift_seconds": calibration.shift_seconds,
                    "source_history_event_count": calibration.event_count,
                    "interval_status": interval_status,
                    "target_contract": ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT,
                    "target_semantics": TARGET_CONTRACT_SEMANTICS[
                        ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT
                    ],
                    "model": self.model_name,
                }
            )
        return pd.DataFrame(rows, index=inputs.index)


def fit_achievable_best_lap_model(
    history: pd.DataFrame,
    *,
    target_event_key: int,
    min_calibration_events: int = 2,
) -> AchievableBestLapModel:
    """Fit source-specific shifts from strictly earlier labelled events."""

    if not isinstance(history, pd.DataFrame):
        raise TypeError("achievable best-lap history must be a pandas DataFrame")
    if int(min_calibration_events) < 1:
        raise ValueError("min_calibration_events must be positive")
    required = {
        EVENT_KEY_COLUMN,
        DRIVER_ID_COLUMN,
        REHEARSAL_SOURCE_COLUMN,
        REHEARSAL_LAP_COLUMN,
        ACTUAL_LAP_COLUMN,
    }
    if history.empty:
        return AchievableBestLapModel(
            target_event_key=int(target_event_key),
            calibrations={},
            min_calibration_events=int(min_calibration_events),
        )
    missing = sorted(required.difference(map(str, history.columns)))
    if missing:
        raise ValueError(f"achievable best-lap history is missing columns: {missing}")

    frame = history.copy()
    frame[EVENT_KEY_COLUMN] = pd.to_numeric(frame[EVENT_KEY_COLUMN], errors="coerce")
    frame[REHEARSAL_LAP_COLUMN] = _finite_seconds(frame, REHEARSAL_LAP_COLUMN)
    frame[ACTUAL_LAP_COLUMN] = _finite_seconds(frame, ACTUAL_LAP_COLUMN)
    frame[REHEARSAL_SOURCE_COLUMN] = frame[REHEARSAL_SOURCE_COLUMN].map(_source_name)
    frame = frame.dropna(subset=[EVENT_KEY_COLUMN, REHEARSAL_LAP_COLUMN, ACTUAL_LAP_COLUMN])
    if frame.empty:
        raise ValueError("achievable best-lap history has no valid labelled rows")
    frame[EVENT_KEY_COLUMN] = frame[EVENT_KEY_COLUMN].astype(int)
    if int(frame[EVENT_KEY_COLUMN].max()) >= int(target_event_key):
        raise ValueError("history must contain only events strictly earlier than target_event_key")
    unknown_sources = sorted(set(frame[REHEARSAL_SOURCE_COLUMN]) - ALLOWED_REHEARSAL_SOURCES)
    if unknown_sources:
        raise ValueError(f"unsupported target-aligned rehearsal sources: {unknown_sources}")

    calibrations: dict[str, AchievableLapSourceCalibration] = {}
    for source, source_rows in frame.groupby(REHEARSAL_SOURCE_COLUMN, sort=True):
        event_keys: list[int] = []
        event_shifts: list[float] = []
        prequential_residuals: list[float] = []
        for event_key, event_rows in source_rows.groupby(EVENT_KEY_COLUMN, sort=True):
            forecast_shift = float(np.median(event_shifts)) if event_shifts else 0.0
            raw_shift = event_rows[ACTUAL_LAP_COLUMN] - event_rows[REHEARSAL_LAP_COLUMN]
            event_shift = float(raw_shift.median(skipna=True))
            event_errors = (
                event_rows[ACTUAL_LAP_COLUMN]
                - (event_rows[REHEARSAL_LAP_COLUMN] + forecast_shift)
            )
            prequential_residuals.extend(float(value) for value in event_errors.dropna().tolist())
            event_keys.append(int(event_key))
            event_shifts.append(event_shift)
        calibrations[str(source)] = AchievableLapSourceCalibration(
            source=str(source),
            event_keys=tuple(event_keys),
            event_shifts_seconds=tuple(event_shifts),
            prequential_residuals_seconds=tuple(prequential_residuals),
        )

    return AchievableBestLapModel(
        target_event_key=int(target_event_key),
        calibrations=calibrations,
        min_calibration_events=int(min_calibration_events),
    )


__all__ = [
    "ACTUAL_LAP_COLUMN",
    "AchievableBestLapModel",
    "AchievableLapSourceCalibration",
    "fit_achievable_best_lap_model",
]
