"""Deterministic baseline for F1 theoretical best lap pace.

The model estimates an ideal-lap lower envelope from clean timing rows. It
prefers sector minima when available, then falls back to fastest clean lap time.
Prediction uses shrunken historical anchors plus lightweight context
adjustments learned from the same dataframe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from packages.f1.models.ultimate_lap_time.schemas import IDEAL_LAP_TARGET_CONTRACT


DRIVER_COLUMNS: tuple[str, ...] = ("driver_id", "driver_number", "DriverNumber", "Driver", "driver")
TEAM_COLUMNS: tuple[str, ...] = (
    "team_id",
    "team_name",
    "constructor_name",
    "constructor",
    "TeamName",
    "Team",
    "team",
)
EVENT_COLUMNS: tuple[str, ...] = ("event_key", "meeting_key", "weekend_key", "race_id", "event_id")
CIRCUIT_COLUMNS: tuple[str, ...] = ("circuit_id", "track_id", "event_name_norm", "event_name", "MeetingName")
LAP_TIME_COLUMNS: tuple[str, ...] = ("lap_time_seconds", "lap_duration", "LapTime", "lap_time", "duration")
SECTOR_1_COLUMNS: tuple[str, ...] = ("sector1_seconds", "duration_sector_1", "Sector1Time", "sector_1_time", "s1")
SECTOR_2_COLUMNS: tuple[str, ...] = ("sector2_seconds", "duration_sector_2", "Sector2Time", "sector_2_time", "s2")
SECTOR_3_COLUMNS: tuple[str, ...] = ("sector3_seconds", "duration_sector_3", "Sector3Time", "sector_3_time", "s3")
COMPOUND_COLUMNS: tuple[str, ...] = ("compound", "tyre_compound", "tire_compound", "Compound")
TRACK_STATUS_COLUMNS: tuple[str, ...] = ("track_status", "TrackStatus")
PIT_LAP_COLUMNS: tuple[str, ...] = ("is_box_lap", "is_pit_lap", "is_pit_in_lap", "is_pit_out_lap")
PIT_TIME_COLUMNS: tuple[str, ...] = ("PitInTime", "PitOutTime", "pit_in_time", "pit_out_time")
ACCURACY_COLUMNS: tuple[str, ...] = ("is_accurate", "IsAccurate")
DELETED_COLUMNS: tuple[str, ...] = ("is_deleted", "Deleted")
SESSION_COLUMNS: tuple[str, ...] = ("session", "session_name", "SessionName", "SessionType")
SECTOR_COMPATIBILITY_FIELDS: dict[str, tuple[str, ...]] = {
    "session": SESSION_COLUMNS,
    "compound": COMPOUND_COLUMNS,
    "weather": ("weather", "weather_condition", "Rainfall", "rainfall"),
    "setup": ("setup_version", "car_specification", "upgrade_specification"),
}
IDEAL_LAP_TARGET_COLUMN = "ideal_lap_time_seconds"
TARGET_CONTRACT_COLUMN = "target_contract"


@dataclass(frozen=True)
class UltimateLapTimeConfig:
    """Training and prediction controls for the baseline pace model."""

    min_clean_laps_per_anchor: int = 1
    min_lap_seconds: float = 40.0
    max_lap_seconds: float = 180.0
    exclude_pit_laps: bool = True
    exclude_non_accurate_laps: bool = True
    exclude_deleted_laps: bool = True
    exclude_non_green_status: bool = True
    require_compatible_sector_context: bool = True
    sector_compatibility_fields: tuple[str, ...] = ("session", "compound")
    min_laps_per_sector_stratum: int = 1
    min_context_observations: int = 6
    context_adjustment_clip_seconds: float = 12.0
    driver_event_weight: float = 30.0
    driver_circuit_weight: float = 16.0
    team_event_weight: float = 12.0
    driver_weight: float = 10.0
    team_circuit_weight: float = 8.0
    team_weight: float = 5.0
    circuit_weight: float = 4.0
    event_weight: float = 4.0
    global_weight: float = 1.0


@dataclass(frozen=True)
class PaceAnchor:
    """Weighted ideal-lap estimate for a lookup key."""

    seconds: float
    weight: float


@dataclass(frozen=True)
class UltimateLapTimeTrainingSummary:
    """Small audit payload for tests, notebooks, and future orchestration."""

    rows_seen: int
    clean_laps_used: int
    anchor_groups: int
    global_anchor_seconds: float
    lap_time_column: str | None
    sector_columns: tuple[str | None, str | None, str | None]
    resolved_id_columns: dict[str, str | None]
    notes: tuple[str, ...] = ()


@dataclass
class UltimateLapTimeModel:
    """Fitted deterministic ultimate lap-time baseline."""

    config: UltimateLapTimeConfig
    global_anchor_seconds: float
    anchors: dict[str, dict[str, PaceAnchor]]
    compound_offsets: dict[str, float]
    context_coefficients: dict[str, float]
    context_references: dict[str, float]
    residual_quantiles: dict[str, float]
    training_summary: UltimateLapTimeTrainingSummary

    def predict_details(self, context: Mapping[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
        """Predict theoretical best lap seconds and diagnostics for context rows."""

        frame, _ = _coerce_context_frame(context)
        detail_columns = [
            "ultimate_lap_time_seconds",
            "pace_floor_seconds",
            "pace_ceiling_seconds",
            "anchor_seconds",
            "context_adjustment_seconds",
            "anchor_source",
            "model",
        ]
        if frame.empty:
            return pd.DataFrame(columns=detail_columns, index=frame.index)
        rows: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            anchor_seconds, anchor_source = self._anchor_prediction(row)
            adjustment = self._context_adjustment(row)
            prediction = float(anchor_seconds + adjustment)
            prediction = float(np.clip(prediction, self.config.min_lap_seconds, self.config.max_lap_seconds))
            envelope = self._envelope_width()
            rows.append(
                {
                    "ultimate_lap_time_seconds": prediction,
                    "pace_floor_seconds": float(max(self.config.min_lap_seconds, prediction - envelope)),
                    "pace_ceiling_seconds": float(min(self.config.max_lap_seconds, prediction + envelope)),
                    "anchor_seconds": float(anchor_seconds),
                    "context_adjustment_seconds": float(adjustment),
                    "anchor_source": anchor_source,
                    "model": "ultimate_lap_time_deterministic_baseline_v1",
                }
            )
        return pd.DataFrame(rows, index=frame.index, columns=detail_columns)

    def predict(self, context: Mapping[str, Any] | pd.Series | pd.DataFrame) -> float | pd.Series:
        """Return only predicted ultimate lap seconds."""

        _, is_single = _coerce_context_frame(context)
        details = self.predict_details(context)
        values = details["ultimate_lap_time_seconds"]
        if is_single:
            return float(values.iloc[0])
        return values

    def _anchor_prediction(self, row: pd.Series) -> tuple[float, str]:
        candidates: list[tuple[float, float, str]] = []
        for store_name, id_names, weight_attr in _ANCHOR_SPECS:
            key = _key_from_row(row, id_names)
            if key is None:
                continue
            anchor = self.anchors.get(store_name, {}).get(key)
            if anchor is None:
                continue
            evidence_weight = anchor.weight / (anchor.weight + 1.0)
            base_weight = float(getattr(self.config, weight_attr))
            weight = float(base_weight * evidence_weight)
            if weight > 0.0 and np.isfinite(anchor.seconds):
                candidates.append((float(anchor.seconds), weight, store_name))

        global_weight = float(max(self.config.global_weight, 1e-6))
        weighted_sum = self.global_anchor_seconds * global_weight
        weight_sum = global_weight
        best_source = "global"
        best_weight = global_weight
        for seconds, weight, source in candidates:
            weighted_sum += seconds * weight
            weight_sum += weight
            if weight > best_weight:
                best_weight = weight
                best_source = source
        return float(weighted_sum / weight_sum), best_source

    def _context_adjustment(self, row: pd.Series) -> float:
        adjustment = 0.0
        compound = _compound_from_row(row)
        if compound:
            adjustment += float(self.compound_offsets.get(compound, 0.0))

        for feature, coefficient in self.context_coefficients.items():
            value = _row_numeric(row, feature)
            reference = self.context_references.get(feature)
            if value is None or reference is None:
                continue
            adjustment += float(coefficient) * (float(value) - float(reference))
        clip = float(max(0.0, self.config.context_adjustment_clip_seconds))
        if clip > 0.0:
            adjustment = float(np.clip(adjustment, -clip, clip))
        return adjustment

    def _envelope_width(self) -> float:
        p10 = float(self.residual_quantiles.get("p10", 0.0))
        p90 = float(self.residual_quantiles.get("p90", 0.0))
        width = max(0.10, (p90 - p10) * 0.5)
        return float(min(width, 5.0))


def fit_ultimate_lap_time_model(
    laps: pd.DataFrame,
    *,
    config: UltimateLapTimeConfig | None = None,
) -> UltimateLapTimeModel:
    """Fit the deterministic ultimate lap-time baseline from lap/session rows."""

    if not isinstance(laps, pd.DataFrame):
        raise TypeError("laps must be a pandas DataFrame")
    if laps.empty:
        raise ValueError("laps must contain at least one timing row")

    cfg = config or UltimateLapTimeConfig()
    working, column_info = _prepare_laps(laps, cfg)
    clean = _clean_laps(working, cfg)
    if clean.empty:
        raise ValueError("no clean lap rows remain after ultimate lap-time filtering")

    group_cols = _anchor_group_columns(column_info)
    clean = clean.copy()
    clean["_anchor_group_key"] = _composite_key_frame(clean, group_cols) if group_cols else "all"

    anchor_rows = _build_anchor_rows(clean, group_cols, column_info, cfg)
    if anchor_rows.empty:
        raise ValueError("no valid ultimate lap-time anchors could be built from timing rows")

    global_anchor = _weighted_mean(anchor_rows["ideal_lap_seconds"], anchor_rows["clean_lap_count"])
    if not np.isfinite(global_anchor):
        raise ValueError("global ultimate lap-time anchor is not finite")

    anchors = _build_anchor_stores(anchor_rows)
    compound_offsets, coefficients, references, residual_quantiles = _fit_context_adjustments(clean, anchor_rows, cfg)

    summary = UltimateLapTimeTrainingSummary(
        rows_seen=int(len(laps)),
        clean_laps_used=int(len(clean)),
        anchor_groups=int(len(anchor_rows)),
        global_anchor_seconds=float(global_anchor),
        lap_time_column=column_info["lap_time"],
        sector_columns=(column_info["sector1"], column_info["sector2"], column_info["sector3"]),
        resolved_id_columns={
            "driver": column_info["driver"],
            "team": column_info["team"],
            "event": column_info["event"],
            "circuit": column_info["circuit"],
        },
        notes=tuple(_training_notes(column_info, clean, anchor_rows)),
    )
    return UltimateLapTimeModel(
        config=cfg,
        global_anchor_seconds=float(global_anchor),
        anchors=anchors,
        compound_offsets=compound_offsets,
        context_coefficients=coefficients,
        context_references=references,
        residual_quantiles=residual_quantiles,
        training_summary=summary,
    )


def aggregate_ideal_lap_holdout_targets(
    laps: pd.DataFrame,
    *,
    config: UltimateLapTimeConfig | None = None,
    group_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate raw holdout laps into explicit theoretical ideal-lap targets.

    One target is produced per driver/event/circuit/session group. Sector minima
    may come from different clean laps by definition. Outcome timing columns are
    removed from the returned prediction context so downstream evaluators cannot
    silently compare an ideal-lap prediction with arbitrary observed lap rows.
    """

    if not isinstance(laps, pd.DataFrame):
        raise TypeError("laps must be a pandas DataFrame")
    if laps.empty:
        raise ValueError("laps must contain at least one timing row")
    cfg = config or UltimateLapTimeConfig()
    working, column_info = _prepare_laps(laps, cfg)
    clean = _clean_laps(working, cfg)
    if clean.empty:
        raise ValueError("no clean holdout laps remain after ultimate lap-time filtering")

    if group_columns is None:
        resolved = [
            _first_existing(clean, ("season", "year", "Season", "Year")),
            column_info.get("event"),
            column_info.get("circuit"),
            _first_existing(clean, SESSION_COLUMNS),
            column_info.get("driver"),
        ]
        groups = [str(column) for column in resolved if column is not None]
        required_identities = {
            "event": column_info.get("event"),
            "session": _first_existing(clean, SESSION_COLUMNS),
            "driver": column_info.get("driver"),
        }
        missing = [name for name, column in required_identities.items() if column is None]
        if missing:
            raise ValueError(f"ideal-lap holdout aggregation requires identities: {tuple(missing)}")
    else:
        groups = [str(column) for column in group_columns]
        missing = [column for column in groups if column not in clean.columns]
        if missing:
            raise ValueError(f"ideal-lap holdout group columns are missing: {tuple(missing)}")
    if not groups:
        raise ValueError("ideal-lap holdout aggregation requires at least one group column")
    outcome_columns = {
        *LAP_TIME_COLUMNS,
        *SECTOR_1_COLUMNS,
        *SECTOR_2_COLUMNS,
        *SECTOR_3_COLUMNS,
        "p05_target",
        "p50_target",
        "p90_target",
        "lap_p05",
        "lap_p50",
        "lap_p90",
        "prediction",
        "predicted_lap_time_seconds",
        "ultimate_lap_time_seconds",
        IDEAL_LAP_TARGET_COLUMN,
        TARGET_CONTRACT_COLUMN,
    }

    rows: list[dict[str, Any]] = []
    for _, group in clean.groupby(groups, sort=False, dropna=False):
        ideal_details = _ideal_lap_details(group, cfg)
        ideal = float(ideal_details["seconds"])
        if not np.isfinite(ideal):
            continue
        representative = group.iloc[0]
        row = {
            str(column): representative[column]
            for column in laps.columns
            if column in representative.index and column not in outcome_columns
        }
        row[IDEAL_LAP_TARGET_COLUMN] = float(ideal)
        row[TARGET_CONTRACT_COLUMN] = IDEAL_LAP_TARGET_CONTRACT
        row["clean_lap_count"] = int(len(group))
        row["ideal_lap_construction"] = str(ideal_details["construction"])
        row["sector_compatibility_columns"] = list(ideal_details["compatibility_columns"])
        row["sector_compatibility_stratum"] = ideal_details["compatibility_stratum"]
        row["sector_compatibility_candidate_count"] = int(ideal_details["candidate_count"])
        rows.append(row)
    if not rows:
        raise ValueError("no finite theoretical ideal-lap holdout targets could be aggregated")
    return pd.DataFrame(rows)


_ANCHOR_SPECS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("driver_event", ("driver", "event"), "driver_event_weight"),
    ("driver_circuit", ("driver", "circuit"), "driver_circuit_weight"),
    ("team_event", ("team", "event"), "team_event_weight"),
    ("driver", ("driver",), "driver_weight"),
    ("team_circuit", ("team", "circuit"), "team_circuit_weight"),
    ("team", ("team",), "team_weight"),
    ("circuit", ("circuit",), "circuit_weight"),
    ("event", ("event",), "event_weight"),
)


def _coerce_context_frame(context: Mapping[str, Any] | pd.Series | pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if isinstance(context, pd.DataFrame):
        return context.copy(), False
    if isinstance(context, pd.Series):
        return context.to_frame().T, True
    if isinstance(context, Mapping):
        return pd.DataFrame([dict(context)]), True
    raise TypeError("context must be a mapping, pandas Series, or pandas DataFrame")


def _prepare_laps(laps: pd.DataFrame, cfg: UltimateLapTimeConfig) -> tuple[pd.DataFrame, dict[str, str | None]]:
    frame = laps.copy()
    lap_col = _first_existing(frame, LAP_TIME_COLUMNS)
    sector1 = _first_existing(frame, SECTOR_1_COLUMNS)
    sector2 = _first_existing(frame, SECTOR_2_COLUMNS)
    sector3 = _first_existing(frame, SECTOR_3_COLUMNS)
    if lap_col is None and not all((sector1, sector2, sector3)):
        raise ValueError(
            "ultimate lap-time training requires a lap time column or all three sector time columns"
        )

    if lap_col is not None:
        frame["_ultimate_lap_seconds"] = _to_seconds(frame[lap_col])
    else:
        frame["_ultimate_lap_seconds"] = np.nan

    for target, source in (
        ("_ultimate_sector1_seconds", sector1),
        ("_ultimate_sector2_seconds", sector2),
        ("_ultimate_sector3_seconds", sector3),
    ):
        frame[target] = _to_seconds(frame[source]) if source is not None else np.nan
    sector_cols = ["_ultimate_sector1_seconds", "_ultimate_sector2_seconds", "_ultimate_sector3_seconds"]
    frame["_ultimate_sector_sum_seconds"] = frame[sector_cols].sum(axis=1, min_count=3)
    if lap_col is None:
        frame["_ultimate_lap_seconds"] = frame["_ultimate_sector_sum_seconds"]
    else:
        missing_lap = frame["_ultimate_lap_seconds"].isna()
        frame.loc[missing_lap, "_ultimate_lap_seconds"] = frame.loc[missing_lap, "_ultimate_sector_sum_seconds"]

    frame["_ultimate_lap_valid"] = frame["_ultimate_lap_seconds"].between(
        float(cfg.min_lap_seconds),
        float(cfg.max_lap_seconds),
        inclusive="both",
    )
    sector_sum_valid = frame["_ultimate_sector_sum_seconds"].between(
        float(cfg.min_lap_seconds),
        float(cfg.max_lap_seconds),
        inclusive="both",
    )
    frame["_ultimate_sector_sum_valid"] = sector_sum_valid

    info = {
        "lap_time": lap_col,
        "sector1": sector1,
        "sector2": sector2,
        "sector3": sector3,
        "driver": _first_existing(frame, DRIVER_COLUMNS),
        "team": _first_existing(frame, TEAM_COLUMNS),
        "event": _first_existing(frame, EVENT_COLUMNS),
        "circuit": _first_existing(frame, CIRCUIT_COLUMNS),
    }
    return frame, info


def _clean_laps(frame: pd.DataFrame, cfg: UltimateLapTimeConfig) -> pd.DataFrame:
    mask = frame["_ultimate_lap_valid"].fillna(False).astype(bool)
    if all(col in frame.columns for col in ("_ultimate_sector1_seconds", "_ultimate_sector2_seconds", "_ultimate_sector3_seconds")):
        sector_mask = frame["_ultimate_sector_sum_valid"].fillna(False).astype(bool)
        mask = mask | sector_mask

    if cfg.exclude_non_accurate_laps:
        for col in ACCURACY_COLUMNS:
            if col in frame.columns:
                mask &= _truthy(frame[col]).fillna(True)
                break

    if cfg.exclude_deleted_laps:
        for col in DELETED_COLUMNS:
            if col in frame.columns:
                mask &= ~_truthy(frame[col]).fillna(False)
                break

    if cfg.exclude_pit_laps:
        for col in PIT_LAP_COLUMNS:
            if col in frame.columns:
                mask &= ~_truthy(frame[col]).fillna(False)
        for col in PIT_TIME_COLUMNS:
            if col in frame.columns:
                mask &= frame[col].isna()

    if cfg.exclude_non_green_status:
        status_col = _first_existing(frame, TRACK_STATUS_COLUMNS)
        if status_col is not None:
            mask &= frame[status_col].map(_is_green_track_status)

    return frame.loc[mask].copy()


def _build_anchor_rows(
    clean: pd.DataFrame,
    group_cols: Sequence[str],
    column_info: dict[str, str | None],
    cfg: UltimateLapTimeConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = clean.groupby("_anchor_group_key", sort=False, dropna=False)
    min_clean = int(max(1, cfg.min_clean_laps_per_anchor))
    id_cols = {name: col for name, col in column_info.items() if name in {"driver", "team", "event", "circuit"} and col}

    for group_key, group in grouped:
        if len(group) < min_clean:
            continue
        ideal_details = _ideal_lap_details(group, cfg)
        ideal = float(ideal_details["seconds"])
        if not np.isfinite(ideal):
            continue
        row: dict[str, Any] = {
            "anchor_group_key": str(group_key),
            "ideal_lap_seconds": float(ideal),
            "clean_lap_count": float(len(group)),
            "ideal_lap_construction": str(ideal_details["construction"]),
            "sector_compatibility_columns": "|".join(ideal_details["compatibility_columns"]),
            "sector_compatibility_stratum": ideal_details["compatibility_stratum"],
            "sector_compatibility_candidate_count": int(ideal_details["candidate_count"]),
        }
        for name, col in id_cols.items():
            row[name] = _first_non_missing(group[col])
        if group_cols:
            row["group_columns"] = "|".join(group_cols)
        else:
            row["group_columns"] = "all"
        rows.append(row)
    return pd.DataFrame(rows)


def _sector_compatibility_columns(
    group: pd.DataFrame,
    cfg: UltimateLapTimeConfig,
) -> list[str]:
    columns: list[str] = []
    for field in cfg.sector_compatibility_fields:
        aliases = SECTOR_COMPATIBILITY_FIELDS.get(str(field), (str(field),))
        column = _first_existing(group, aliases)
        if column is not None and column not in columns:
            columns.append(column)
    return columns


def _ideal_lap_details(
    group: pd.DataFrame,
    cfg: UltimateLapTimeConfig,
) -> dict[str, Any]:
    sector_cols = ["_ultimate_sector1_seconds", "_ultimate_sector2_seconds", "_ultimate_sector3_seconds"]
    sector_valid = group["_ultimate_sector_sum_valid"].fillna(False)
    if sector_valid.any():
        sector_group = group.loc[sector_valid, sector_cols]
        compatibility_columns = _sector_compatibility_columns(group, cfg)
        if not cfg.require_compatible_sector_context:
            sector_mins = [float(pd.to_numeric(sector_group[col], errors="coerce").min()) for col in sector_cols]
            if all(np.isfinite(value) and value > 0.0 for value in sector_mins):
                return {
                    "seconds": float(sum(sector_mins)),
                    "construction": "unconstrained_sector_lower_bound_research_only",
                    "compatibility_columns": (),
                    "compatibility_stratum": None,
                    "candidate_count": 1,
                }
        elif compatibility_columns:
            candidate_rows: list[tuple[float, str]] = []
            eligible = group.loc[sector_valid, [*compatibility_columns, *sector_cols]].copy()
            eligible = eligible.dropna(subset=compatibility_columns)
            groupby_key: str | list[str] = (
                compatibility_columns[0]
                if len(compatibility_columns) == 1
                else compatibility_columns
            )
            for stratum_key, stratum in eligible.groupby(groupby_key, sort=False, dropna=False):
                if len(stratum) < int(max(1, cfg.min_laps_per_sector_stratum)):
                    continue
                sector_mins = [
                    float(pd.to_numeric(stratum[col], errors="coerce").min())
                    for col in sector_cols
                ]
                if not all(np.isfinite(value) and value > 0.0 for value in sector_mins):
                    continue
                key_values = stratum_key if isinstance(stratum_key, tuple) else (stratum_key,)
                key_label = "|".join(
                    f"{column}={value}"
                    for column, value in zip(compatibility_columns, key_values)
                )
                candidate_rows.append((float(sum(sector_mins)), key_label))
            if candidate_rows:
                seconds, stratum = min(candidate_rows, key=lambda item: item[0])
                return {
                    "seconds": float(seconds),
                    "construction": "compatible_sector_lower_bound_v2",
                    "compatibility_columns": tuple(compatibility_columns),
                    "compatibility_stratum": stratum,
                    "candidate_count": int(len(candidate_rows)),
                }
    lap_values = pd.to_numeric(group["_ultimate_lap_seconds"], errors="coerce").dropna()
    return {
        "seconds": float(lap_values.min()) if not lap_values.empty else float("nan"),
        "construction": (
            "fastest_clean_lap_fallback_missing_compatibility_context"
            if sector_valid.any() and cfg.require_compatible_sector_context
            else "fastest_clean_lap_fallback"
        ),
        "compatibility_columns": tuple(_sector_compatibility_columns(group, cfg)),
        "compatibility_stratum": None,
        "candidate_count": 0,
    }


def _build_anchor_stores(anchor_rows: pd.DataFrame) -> dict[str, dict[str, PaceAnchor]]:
    stores: dict[str, dict[str, PaceAnchor]] = {name: {} for name, _, _ in _ANCHOR_SPECS}
    for store_name, id_names, _ in _ANCHOR_SPECS:
        cols = [name for name in id_names if name in anchor_rows.columns]
        if len(cols) != len(id_names):
            continue
        keyed = anchor_rows.copy()
        keyed["_store_key"] = _composite_key_frame(keyed, cols)
        keyed = keyed[keyed["_store_key"].notna()].copy()
        if keyed.empty:
            continue
        for key, group in keyed.groupby("_store_key", sort=False):
            weight = pd.to_numeric(group["clean_lap_count"], errors="coerce").fillna(1.0).clip(lower=1.0)
            seconds = _weighted_mean(group["ideal_lap_seconds"], weight)
            if np.isfinite(seconds):
                stores[store_name][str(key)] = PaceAnchor(seconds=float(seconds), weight=float(weight.sum()))
    return stores


def _fit_context_adjustments(
    clean: pd.DataFrame,
    anchor_rows: pd.DataFrame,
    cfg: UltimateLapTimeConfig,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    anchor_map = anchor_rows.set_index("anchor_group_key")["ideal_lap_seconds"].to_dict()
    rows = clean.copy()
    rows["_anchor_seconds"] = rows["_anchor_group_key"].map(anchor_map)
    residual = pd.to_numeric(rows["_ultimate_lap_seconds"], errors="coerce") - pd.to_numeric(
        rows["_anchor_seconds"],
        errors="coerce",
    )
    residual = residual.clip(lower=0.0)
    valid = residual.notna() & np.isfinite(residual)
    if valid.sum() == 0:
        return {}, {}, {}, {"p10": 0.0, "p50": 0.0, "p90": 0.0}

    rows = rows.loc[valid].copy()
    residual = residual.loc[valid].astype(float)
    residual_quantiles = {
        "p10": float(residual.quantile(0.10)),
        "p50": float(residual.quantile(0.50)),
        "p90": float(residual.quantile(0.90)),
    }

    compound_offsets = _fit_compound_offsets(rows, residual, cfg)
    working_residual = residual.copy()
    compound_col = _first_existing(rows, COMPOUND_COLUMNS)
    if compound_col is not None and compound_offsets:
        compound_adjustments = rows[compound_col].map(lambda value: compound_offsets.get(_normalize_compound(value), 0.0))
        working_residual = (working_residual - compound_adjustments.astype(float)).clip(lower=0.0)

    coefficients: dict[str, float] = {}
    references: dict[str, float] = {}
    for feature in ("tyre_age", "tire_age", "lap_number", "track_temperature", "air_temperature", "rainfall"):
        if feature not in rows.columns:
            continue
        x = pd.to_numeric(rows[feature], errors="coerce")
        feature_valid = x.notna() & working_residual.notna()
        if int(feature_valid.sum()) < int(max(3, cfg.min_context_observations)):
            continue
        reference = _feature_reference(feature, x.loc[feature_valid])
        slope = _robust_slope(x.loc[feature_valid], working_residual.loc[feature_valid])
        slope = _clip_context_slope(feature, slope)
        if not np.isfinite(slope) or abs(slope) < 1e-9:
            continue
        coefficients[feature] = float(slope)
        references[feature] = float(reference)
        working_residual = (working_residual - (slope * (x - reference))).clip(lower=0.0)
    return compound_offsets, coefficients, references, residual_quantiles


def _fit_compound_offsets(rows: pd.DataFrame, residual: pd.Series, cfg: UltimateLapTimeConfig) -> dict[str, float]:
    compound_col = _first_existing(rows, COMPOUND_COLUMNS)
    if compound_col is None:
        return {}
    grouped: dict[str, list[float]] = {}
    for idx, value in rows[compound_col].items():
        key = _normalize_compound(value)
        if not key:
            continue
        residual_value = float(residual.loc[idx])
        if np.isfinite(residual_value):
            grouped.setdefault(key, []).append(residual_value)
    min_obs = int(max(3, min(cfg.min_context_observations, 8)))
    medians = {
        key: float(np.median(values))
        for key, values in grouped.items()
        if len(values) >= min_obs and np.isfinite(np.median(values))
    }
    if len(medians) < 2:
        return {key: 0.0 for key in medians}
    best = min(medians.values())
    return {key: float(max(0.0, value - best)) for key, value in medians.items()}


def _anchor_group_columns(column_info: dict[str, str | None]) -> list[str]:
    driver = column_info.get("driver")
    team = column_info.get("team")
    event = column_info.get("event")
    circuit = column_info.get("circuit")
    if driver and event:
        return [driver, event]
    if driver and circuit:
        return [driver, circuit]
    if team and event:
        return [team, event]
    if driver:
        return [driver]
    if team:
        return [team]
    if event:
        return [event]
    if circuit:
        return [circuit]
    return []


def _training_notes(
    column_info: dict[str, str | None],
    clean: pd.DataFrame,
    anchor_rows: pd.DataFrame,
) -> list[str]:
    notes: list[str] = []
    if not all((column_info["sector1"], column_info["sector2"], column_info["sector3"])):
        notes.append("sector columns incomplete: fastest clean lap fallback was used where needed")
    if column_info["lap_time"] is None:
        notes.append("lap time column missing: sector sums were used as observed lap times")
    if len(anchor_rows) < len(clean):
        notes.append("clean laps were aggregated into lower-envelope anchor groups")
    constructions = set(anchor_rows.get("ideal_lap_construction", pd.Series(dtype=str)).dropna().astype(str))
    if "compatible_sector_lower_bound_v2" in constructions:
        notes.append(
            "sector minima were combined only inside declared compatible session/compound strata; "
            "the result remains a theoretical lower bound, not an achievable expected lap"
        )
    if "fastest_clean_lap_fallback_missing_compatibility_context" in constructions:
        notes.append(
            "sector compatibility context was unavailable for some anchors; fastest clean lap fallback used"
        )
    return notes


def _first_existing(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _to_seconds(values: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(values):
        return values.dt.total_seconds()
    numeric = pd.to_numeric(values, errors="coerce")
    needs_parse = numeric.isna() & values.notna()
    if not needs_parse.any():
        return numeric.astype(float)
    parsed = pd.to_timedelta(values.loc[needs_parse], errors="coerce").dt.total_seconds()
    numeric.loc[needs_parse] = parsed
    return numeric.astype(float)


def _truthy(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values
    normalized = values.astype(str).str.strip().str.lower()
    truth = normalized.isin({"true", "1", "yes", "y", "t"})
    false = normalized.isin({"false", "0", "no", "n", "f", "", "nan", "none", "<na>"})
    return truth.where(truth | false, values.astype(bool))


def _is_green_track_status(value: object) -> bool:
    if value is None or pd.isna(value):
        return True
    text = str(value).strip()
    if not text:
        return True
    codes = {char for char in text if char.isdigit()}
    if not codes:
        return True
    return codes.issubset({"1"})


def _first_non_missing(values: pd.Series) -> object:
    non_missing = values.dropna()
    if non_missing.empty:
        return None
    return non_missing.iloc[0]


def _clean_key(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "<na>"}:
        return None
    return text


def _composite_key_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    if not columns:
        return pd.Series("all", index=frame.index, dtype=object)
    keys: list[str | None] = []
    for _, row in frame[list(columns)].iterrows():
        parts = [_clean_key(row[col]) for col in columns]
        keys.append("||".join(str(part) for part in parts) if all(parts) else None)
    return pd.Series(keys, index=frame.index, dtype=object)


def _key_from_row(row: pd.Series, id_names: Sequence[str]) -> str | None:
    parts: list[str] = []
    for name in id_names:
        value = _id_value_from_row(row, name)
        key = _clean_key(value)
        if key is None:
            return None
        parts.append(key)
    return "||".join(parts)


def _id_value_from_row(row: pd.Series, name: str) -> object:
    candidates = {
        "driver": DRIVER_COLUMNS,
        "team": TEAM_COLUMNS,
        "event": EVENT_COLUMNS,
        "circuit": CIRCUIT_COLUMNS,
    }[name]
    for candidate in candidates:
        if candidate in row.index:
            return row[candidate]
    return None


def _compound_from_row(row: pd.Series) -> str | None:
    for candidate in COMPOUND_COLUMNS:
        if candidate in row.index:
            return _normalize_compound(row[candidate])
    return None


def _normalize_compound(value: object) -> str | None:
    key = _clean_key(value)
    if key is None:
        return None
    text = key.upper().replace(" ", "_")
    if text in {"S", "SOFT"}:
        return "SOFT"
    if text in {"M", "MEDIUM"}:
        return "MEDIUM"
    if text in {"H", "HARD"}:
        return "HARD"
    if "INTER" in text:
        return "INTERMEDIATE"
    if "WET" in text:
        return "WET"
    return text


def _row_numeric(row: pd.Series, feature: str) -> float | None:
    if feature not in row.index:
        return None
    value = pd.to_numeric(pd.Series([row[feature]]), errors="coerce").iloc[0]
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value)


def _feature_reference(feature: str, values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    if feature in {"tyre_age", "tire_age", "lap_number", "rainfall"}:
        return float(numeric.min())
    return float(numeric.median())


def _robust_slope(x: pd.Series, y: pd.Series) -> float:
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[valid]
    y_values = y_values[valid]
    if x_values.size < 3:
        return 0.0
    x_centered = x_values - float(np.median(x_values))
    y_centered = y_values - float(np.median(y_values))
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denom)


def _clip_context_slope(feature: str, slope: float) -> float:
    if not np.isfinite(slope):
        return 0.0
    bounds = {
        "tyre_age": (0.0, 0.25),
        "tire_age": (0.0, 0.25),
        "lap_number": (-0.15, 0.08),
        "track_temperature": (-0.08, 0.08),
        "air_temperature": (-0.05, 0.05),
        "rainfall": (0.0, 30.0),
    }
    low, high = bounds.get(feature, (-1.0, 1.0))
    return float(np.clip(float(slope), low, high))


def _weighted_mean(values: pd.Series, weights: pd.Series | Sequence[float]) -> float:
    value_array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    weight_array = pd.to_numeric(pd.Series(weights, index=values.index), errors="coerce").fillna(1.0).to_numpy(dtype=float)
    valid = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0.0)
    if not valid.any():
        return float("nan")
    return float(np.average(value_array[valid], weights=weight_array[valid]))


__all__ = [
    "IDEAL_LAP_TARGET_COLUMN",
    "PaceAnchor",
    "UltimateLapTimeConfig",
    "UltimateLapTimeModel",
    "UltimateLapTimeTrainingSummary",
    "aggregate_ideal_lap_holdout_targets",
    "fit_ultimate_lap_time_model",
]
