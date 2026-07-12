"""Typed data contracts for Ultimate Lap-Time telemetry models."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Mapping, Sequence

import numpy as np


ALLOWED_SPLIT_NAMES: frozenset[str] = frozenset(
    {"train", "validation", "val", "test", "holdout", "live", "backtest"}
)
IDEAL_LAP_TARGET_CONTRACT = "theoretical_ideal_lap_v1"
THEORETICAL_SECTOR_FLOOR_TARGET_CONTRACT = "theoretical_sector_floor_v1"
ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT = "achievable_session_end_lap_v1"
OBSERVED_LAP_TARGET_CONTRACT = "observed_lap_v1"
ALLOWED_TARGET_CONTRACTS: frozenset[str] = frozenset(
    {
        IDEAL_LAP_TARGET_CONTRACT,
        THEORETICAL_SECTOR_FLOOR_TARGET_CONTRACT,
        ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT,
        OBSERVED_LAP_TARGET_CONTRACT,
    }
)
THEORETICAL_TARGET_CONTRACTS: frozenset[str] = frozenset(
    {IDEAL_LAP_TARGET_CONTRACT, THEORETICAL_SECTOR_FLOOR_TARGET_CONTRACT}
)
TARGET_CONTRACT_SEMANTICS: Mapping[str, str] = {
    IDEAL_LAP_TARGET_CONTRACT: "theoretical_sector_floor",
    THEORETICAL_SECTOR_FLOOR_TARGET_CONTRACT: "theoretical_sector_floor",
    ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT: "achievable_session_end_lap",
    OBSERVED_LAP_TARGET_CONTRACT: "observed_lap",
}
DEGENERATE_QUANTILE_TOLERANCE_SECONDS = 1e-6
TARGET_VECTOR_COLUMNS: tuple[str, ...] = (
    "lap_p05",
    "lap_p50",
    "lap_p90",
    "sector1_seconds",
    "sector2_seconds",
    "sector3_seconds",
)
LEAKAGE_FIELD_TOKENS: tuple[str, ...] = (
    "lap_time",
    "lap_duration",
    "sector",
    "target",
    "p05",
    "p50",
    "p90",
    "prediction",
    "predicted",
    "actual",
    "result",
    "classified",
    "finish",
    "rank",
    "position",
    "gap",
    "interval",
)


def _as_clean_string(value: Any, *, field_name: str) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        raise ValueError(f"{field_name} must be a non-empty value")
    return text


def _optional_clean_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    return text


def _finite_positive(value: Any, *, field_name: str, required: bool) -> float | None:
    if value is None:
        if required:
            raise ValueError(f"{field_name} is required")
        return None
    try:
        if isinstance(value, timedelta):
            seconds = float(value.total_seconds())
        elif isinstance(value, np.timedelta64):
            seconds = float(value / np.timedelta64(1, "s"))
        elif hasattr(value, "total_seconds"):
            seconds = float(value.total_seconds())
        else:
            seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric seconds") from exc
    if not np.isfinite(seconds):
        if required:
            raise ValueError(f"{field_name} must be finite")
        return None
    if seconds <= 0.0:
        raise ValueError(f"{field_name} must be positive seconds")
    return float(seconds)


def _mapping_get_first(mapping: Mapping[str, Any], candidates: Sequence[str]) -> Any:
    for candidate in candidates:
        if candidate in mapping and mapping[candidate] is not None:
            return mapping[candidate]
    return None


@dataclass(frozen=True)
class UltimateLapSplitKey:
    """No-leakage split key derived from pre-target weekend identity."""

    event_key: str
    circuit_id: str
    session: str
    season: str | None = None
    split_name: str | None = None
    fold: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_key", _as_clean_string(self.event_key, field_name="event_key"))
        object.__setattr__(self, "circuit_id", _as_clean_string(self.circuit_id, field_name="circuit_id"))
        object.__setattr__(self, "session", _as_clean_string(self.session, field_name="session"))
        object.__setattr__(self, "season", _optional_clean_string(self.season))
        split_name = _optional_clean_string(self.split_name)
        if split_name is not None and split_name.lower() not in ALLOWED_SPLIT_NAMES:
            raise ValueError(f"split_name must be one of {sorted(ALLOWED_SPLIT_NAMES)}")
        object.__setattr__(self, "split_name", split_name)
        object.__setattr__(self, "fold", _optional_clean_string(self.fold))

    @property
    def value(self) -> str:
        parts = [self.season, self.event_key, self.circuit_id, self.session, self.fold]
        return "|".join(part for part in parts if part is not None)

    def as_dict(self) -> dict[str, str | None]:
        return {
            "season": self.season,
            "event_key": self.event_key,
            "circuit_id": self.circuit_id,
            "session": self.session,
            "split_name": self.split_name,
            "fold": self.fold,
            "split_key": self.value,
        }


@dataclass(frozen=True)
class UltimateLapTargets:
    """One realized lap outcome with explicit estimand diagnostics.

    ``IDEAL_LAP_TARGET_CONTRACT`` remains accepted as the legacy spelling for
    a theoretical sector floor. It is not an achievable/session-end lap and is
    therefore never silently treated as a promotion-grade forecast target.

    The optional ``p*_target`` fields are accepted only as legacy metadata.
    They are never used as quantile-regression labels: every pinball head trains
    against the same realized ``lap_time_seconds`` outcome.
    """

    lap_time_seconds: float
    sector1_seconds: float | None = None
    sector2_seconds: float | None = None
    sector3_seconds: float | None = None
    p05_target: float | None = None
    p50_target: float | None = None
    p90_target: float | None = None
    target_contract: str = OBSERVED_LAP_TARGET_CONTRACT
    quantile_targets_were_explicit: bool = field(init=False)
    quantile_targets_are_degenerate: bool = field(init=False)

    def __post_init__(self) -> None:
        quantiles_were_explicit = all(
            value is not None for value in (self.p05_target, self.p50_target, self.p90_target)
        )
        lap = _finite_positive(self.lap_time_seconds, field_name="lap_time_seconds", required=True)
        object.__setattr__(self, "lap_time_seconds", lap)
        for name in ("sector1_seconds", "sector2_seconds", "sector3_seconds"):
            object.__setattr__(
                self,
                name,
                _finite_positive(getattr(self, name), field_name=name, required=False),
            )
        p50 = _finite_positive(
            self.p50_target if self.p50_target is not None else lap,
            field_name="p50_target",
            required=True,
        )
        p05 = _finite_positive(
            self.p05_target if self.p05_target is not None else p50,
            field_name="p05_target",
            required=True,
        )
        p90 = _finite_positive(
            self.p90_target if self.p90_target is not None else p50,
            field_name="p90_target",
            required=True,
        )
        if not (p05 <= p50 <= p90):
            raise ValueError("quantile targets must satisfy p05_target <= p50_target <= p90_target")
        object.__setattr__(self, "p05_target", p05)
        object.__setattr__(self, "p50_target", p50)
        object.__setattr__(self, "p90_target", p90)
        object.__setattr__(self, "quantile_targets_were_explicit", quantiles_were_explicit)
        object.__setattr__(
            self,
            "quantile_targets_are_degenerate",
            bool(
                np.allclose(
                    np.asarray([p05, p50, p90], dtype=float),
                    float(p50),
                    atol=DEGENERATE_QUANTILE_TOLERANCE_SECONDS,
                    rtol=0.0,
                )
            ),
        )
        contract = _as_clean_string(self.target_contract, field_name="target_contract").lower()
        if contract not in ALLOWED_TARGET_CONTRACTS:
            raise ValueError(f"target_contract must be one of {sorted(ALLOWED_TARGET_CONTRACTS)}")
        object.__setattr__(self, "target_contract", contract)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "UltimateLapTargets":
        contract = _mapping_get_first(mapping, ("target_contract", "target_kind"))
        explicit_contract = str(contract).strip().lower() if contract is not None else None
        ideal_lap_time = _mapping_get_first(mapping, ("ideal_lap_time_seconds",))
        theoretical_sector_floor = _mapping_get_first(
            mapping, ("theoretical_sector_floor_seconds",)
        )
        achievable_session_end_lap = _mapping_get_first(
            mapping,
            (
                "achievable_session_end_lap_time_seconds",
                "session_end_lap_time_seconds",
                "achievable_lap_time_seconds",
            ),
        )
        if explicit_contract == IDEAL_LAP_TARGET_CONTRACT and ideal_lap_time is None:
            raise ValueError(
                "theoretical ideal-lap target requires an explicit ideal_lap_time_seconds value"
            )
        if (
            explicit_contract == THEORETICAL_SECTOR_FLOOR_TARGET_CONTRACT
            and theoretical_sector_floor is None
        ):
            raise ValueError(
                "theoretical sector-floor target requires an explicit "
                "theoretical_sector_floor_seconds value"
            )
        if (
            explicit_contract == ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT
            and achievable_session_end_lap is None
        ):
            raise ValueError(
                "achievable session-end target requires an explicit "
                "achievable_session_end_lap_time_seconds value"
            )
        semantic_values = {
            IDEAL_LAP_TARGET_CONTRACT: ideal_lap_time,
            THEORETICAL_SECTOR_FLOOR_TARGET_CONTRACT: theoretical_sector_floor,
            ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT: achievable_session_end_lap,
        }
        present_semantic_contracts = [
            semantic_contract
            for semantic_contract, semantic_value in semantic_values.items()
            if semantic_value is not None
        ]
        if len(present_semantic_contracts) > 1:
            raise ValueError(
                "target payload contains multiple incompatible estimands: "
                f"{sorted(present_semantic_contracts)}"
            )
        if explicit_contract is not None:
            for semantic_contract, semantic_value in semantic_values.items():
                if semantic_value is not None and semantic_contract != explicit_contract:
                    raise ValueError(
                        f"{semantic_contract} target value cannot use {explicit_contract}"
                    )

        if explicit_contract in semantic_values:
            lap_time = semantic_values[explicit_contract]
            contract = explicit_contract
        elif theoretical_sector_floor is not None:
            lap_time = theoretical_sector_floor
            contract = THEORETICAL_SECTOR_FLOOR_TARGET_CONTRACT
        elif ideal_lap_time is not None:
            lap_time = ideal_lap_time
            contract = IDEAL_LAP_TARGET_CONTRACT
        elif achievable_session_end_lap is not None:
            lap_time = achievable_session_end_lap
            contract = ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT
        else:
            lap_time = _mapping_get_first(
                mapping,
                ("lap_time_seconds", "lap_duration", "LapTime", "lap_time", "duration"),
            )
            contract = explicit_contract or OBSERVED_LAP_TARGET_CONTRACT
        return cls(
            lap_time_seconds=lap_time,
            sector1_seconds=_mapping_get_first(
                mapping,
                ("sector1_seconds", "duration_sector_1", "Sector1Time", "sector_1_time", "s1"),
            ),
            sector2_seconds=_mapping_get_first(
                mapping,
                ("sector2_seconds", "duration_sector_2", "Sector2Time", "sector_2_time", "s2"),
            ),
            sector3_seconds=_mapping_get_first(
                mapping,
                ("sector3_seconds", "duration_sector_3", "Sector3Time", "sector_3_time", "s3"),
            ),
            p05_target=_mapping_get_first(mapping, ("p05_target", "lap_p05", "p05", "target_p05")),
            p50_target=_mapping_get_first(mapping, ("p50_target", "lap_p50", "p50", "target_p50")),
            p90_target=_mapping_get_first(mapping, ("p90_target", "lap_p90", "p90", "target_p90")),
            target_contract=contract,
        )

    @property
    def target_semantics(self) -> str:
        return TARGET_CONTRACT_SEMANTICS[self.target_contract]

    @property
    def promotion_grade_quantiles_valid(self) -> bool:
        return self.target_contract == ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT

    @property
    def quantile_target_semantics(self) -> str:
        if self.quantile_targets_were_explicit and not self.quantile_targets_are_degenerate:
            return "legacy_per_row_quantiles_ignored"
        return "shared_realized_scalar"

    @property
    def promotion_blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if self.target_contract != ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT:
            blockers.append("target_is_not_achievable_session_end_lap")
        return tuple(blockers)

    def quantile_diagnostics(self) -> dict[str, Any]:
        return {
            "target_contract": self.target_contract,
            "target_semantics": self.target_semantics,
            "quantile_target_semantics": self.quantile_target_semantics,
            "quantile_targets_explicit": self.quantile_targets_were_explicit,
            "quantile_targets_degenerate": self.quantile_targets_are_degenerate,
            "quantile_training_target": "lap_time_seconds_shared_across_all_heads",
            "legacy_per_row_quantile_fields_used_for_training": False,
            "promotion_grade_quantiles_valid": self.promotion_grade_quantiles_valid,
            "promotion_blockers": list(self.promotion_blockers),
        }

    def as_dict(self) -> dict[str, float | str | None]:
        payload: dict[str, float | str | None] = {
            "lap_time_seconds": self.lap_time_seconds,
            "sector1_seconds": self.sector1_seconds,
            "sector2_seconds": self.sector2_seconds,
            "sector3_seconds": self.sector3_seconds,
            "p05_target": self.p05_target,
            "p50_target": self.p50_target,
            "p90_target": self.p90_target,
            "target_contract": self.target_contract,
            "target_semantics": self.target_semantics,
            "quantile_target_semantics": self.quantile_target_semantics,
        }
        if self.target_contract == IDEAL_LAP_TARGET_CONTRACT:
            payload["ideal_lap_time_seconds"] = self.lap_time_seconds
        elif self.target_contract == THEORETICAL_SECTOR_FLOOR_TARGET_CONTRACT:
            payload["theoretical_sector_floor_seconds"] = self.lap_time_seconds
        elif self.target_contract == ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT:
            payload["achievable_session_end_lap_time_seconds"] = self.lap_time_seconds
        return payload

    def target_vector(self) -> np.ndarray:
        values = [
            self.lap_time_seconds,
            self.lap_time_seconds,
            self.lap_time_seconds,
            self.sector1_seconds,
            self.sector2_seconds,
            self.sector3_seconds,
        ]
        return np.asarray([np.nan if value is None else float(value) for value in values], dtype=float)


@dataclass(frozen=True)
class UltimateLapMetadata:
    """Identity fields needed for no-leakage splits and grouped evaluation."""

    event_key: str
    circuit_id: str
    driver_id: str
    team_id: str
    session: str
    split_key: UltimateLapSplitKey
    season: str | None = None
    lap_number: str | None = None
    source: str | None = None
    target_session: str | None = None
    feature_as_of: str | None = None
    target_as_of: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_key", _as_clean_string(self.event_key, field_name="event_key"))
        object.__setattr__(self, "circuit_id", _as_clean_string(self.circuit_id, field_name="circuit_id"))
        object.__setattr__(self, "driver_id", _as_clean_string(self.driver_id, field_name="driver_id"))
        object.__setattr__(self, "team_id", _as_clean_string(self.team_id, field_name="team_id"))
        object.__setattr__(self, "session", _as_clean_string(self.session, field_name="session"))
        if not isinstance(self.split_key, UltimateLapSplitKey):
            raise TypeError("split_key must be an UltimateLapSplitKey")
        object.__setattr__(self, "season", _optional_clean_string(self.season))
        object.__setattr__(self, "lap_number", _optional_clean_string(self.lap_number))
        object.__setattr__(self, "source", _optional_clean_string(self.source))
        object.__setattr__(self, "target_session", _optional_clean_string(self.target_session))
        object.__setattr__(self, "feature_as_of", _optional_clean_string(self.feature_as_of))
        object.__setattr__(self, "target_as_of", _optional_clean_string(self.target_as_of))
        if self.split_key.event_key != self.event_key:
            raise ValueError("split_key event_key must match metadata event_key")
        if self.split_key.circuit_id != self.circuit_id:
            raise ValueError("split_key circuit_id must match metadata circuit_id")
        if self.split_key.session != self.session:
            raise ValueError("split_key session must match metadata session")

    def as_dict(self) -> dict[str, str | None]:
        return {
            "season": self.season,
            "event_key": self.event_key,
            "circuit_id": self.circuit_id,
            "driver_id": self.driver_id,
            "team_id": self.team_id,
            "session": self.session,
            "split_key": self.split_key.value,
            "split_name": self.split_key.split_name,
            "lap_number": self.lap_number,
            "source": self.source,
            "target_session": self.target_session,
            "feature_as_of": self.feature_as_of,
            "target_as_of": self.target_as_of,
        }


@dataclass(frozen=True)
class DistanceNormalizedTelemetryTensor:
    """Telemetry tensor with shape channels x distance_bins."""

    values: np.ndarray
    channel_names: tuple[str, ...]
    raw_distance_start: float | None = None
    raw_distance_end: float | None = None
    expected_lap_distance: float | None = None
    distance_coverage: float | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2:
            raise ValueError("telemetry values must have shape channels x distance_bins")
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("telemetry values must have at least one channel and one distance bin")
        if len(self.channel_names) != values.shape[0]:
            raise ValueError("channel_names length must match telemetry channel dimension")
        if not np.isfinite(values).all():
            raise ValueError("telemetry values must be finite")
        cleaned_channels = tuple(_as_clean_string(name, field_name="channel_name") for name in self.channel_names)
        object.__setattr__(self, "values", values.astype(np.float32, copy=False))
        object.__setattr__(self, "channel_names", cleaned_channels)
        for name in ("raw_distance_start", "raw_distance_end", "expected_lap_distance", "distance_coverage"):
            raw = getattr(self, name)
            if raw is None:
                continue
            numeric = float(raw)
            if not np.isfinite(numeric):
                raise ValueError(f"{name} must be finite when provided")
            object.__setattr__(self, name, numeric)
        if self.expected_lap_distance is not None and self.expected_lap_distance <= 0.0:
            raise ValueError("expected_lap_distance must be positive")
        if self.distance_coverage is not None and not 0.0 <= self.distance_coverage <= 1.05:
            raise ValueError("distance_coverage must be between 0 and 1.05")

    @property
    def channels(self) -> int:
        return int(self.values.shape[0])

    @property
    def distance_bins(self) -> int:
        return int(self.values.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.channels, self.distance_bins)


def _clean_static_features(static_features: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in dict(static_features).items():
        text_key = _as_clean_string(key, field_name="static feature key")
        if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)) or value is None:
            cleaned[text_key] = value
    return cleaned


def _validate_input_payload(
    telemetry: DistanceNormalizedTelemetryTensor,
    metadata: UltimateLapMetadata,
) -> None:
    if not isinstance(telemetry, DistanceNormalizedTelemetryTensor):
        raise TypeError("telemetry must be a DistanceNormalizedTelemetryTensor")
    if not isinstance(metadata, UltimateLapMetadata):
        raise TypeError("metadata must be an UltimateLapMetadata")


@dataclass(frozen=True)
class UltimateLapTelemetryInput:
    """Model-ready telemetry and metadata with no target dependency."""

    telemetry: DistanceNormalizedTelemetryTensor
    static_features: Mapping[str, Any]
    metadata: UltimateLapMetadata

    def __post_init__(self) -> None:
        _validate_input_payload(self.telemetry, self.metadata)
        object.__setattr__(self, "static_features", _clean_static_features(self.static_features))

    def as_flat_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {}
        record.update(self.metadata.as_dict())
        record.update(self.static_features)
        record["channels"] = self.telemetry.channels
        record["distance_bins"] = self.telemetry.distance_bins
        return record


@dataclass(frozen=True)
class UltimateLapTelemetryExample:
    """Single labelled, model-ready Ultimate Lap-Time example."""

    telemetry: DistanceNormalizedTelemetryTensor
    static_features: Mapping[str, Any]
    targets: UltimateLapTargets
    metadata: UltimateLapMetadata

    def __post_init__(self) -> None:
        _validate_input_payload(self.telemetry, self.metadata)
        if not isinstance(self.targets, UltimateLapTargets):
            raise TypeError("targets must be an UltimateLapTargets")
        object.__setattr__(self, "static_features", _clean_static_features(self.static_features))

    def without_targets(self) -> UltimateLapTelemetryInput:
        """Return the genuine unlabeled input used by inference APIs."""

        return UltimateLapTelemetryInput(
            telemetry=self.telemetry,
            static_features=self.static_features,
            metadata=self.metadata,
        )

    def as_flat_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {}
        record.update(self.metadata.as_dict())
        record.update(self.static_features)
        record.update(self.targets.as_dict())
        record["channels"] = self.telemetry.channels
        record["distance_bins"] = self.telemetry.distance_bins
        return record


@dataclass(frozen=True)
class UltimateLapTelemetryBatch:
    """Batch contract for telemetry models."""

    telemetry: np.ndarray
    static_features: tuple[Mapping[str, Any], ...]
    targets: tuple[UltimateLapTargets, ...]
    metadata: tuple[UltimateLapMetadata, ...]
    channel_names: tuple[str, ...]

    def __post_init__(self) -> None:
        telemetry = np.asarray(self.telemetry, dtype=float)
        if telemetry.ndim != 3:
            raise ValueError("batch telemetry must have shape batch x channels x distance_bins")
        batch_size = telemetry.shape[0]
        if not (
            len(self.static_features) == len(self.targets) == len(self.metadata) == batch_size
        ):
            raise ValueError("batch telemetry, static_features, targets, and metadata lengths must match")
        if batch_size and len(self.channel_names) != telemetry.shape[1]:
            raise ValueError("channel_names length must match batch channel dimension")
        if not np.isfinite(telemetry).all():
            raise ValueError("batch telemetry values must be finite")
        object.__setattr__(self, "telemetry", telemetry.astype(np.float32, copy=False))

    @classmethod
    def from_examples(cls, examples: Sequence[UltimateLapTelemetryExample]) -> "UltimateLapTelemetryBatch":
        if not examples:
            return cls(
                telemetry=np.empty((0, 0, 0), dtype=np.float32),
                static_features=(),
                targets=(),
                metadata=(),
                channel_names=(),
            )
        channel_names = examples[0].telemetry.channel_names
        shape = examples[0].telemetry.shape
        for example in examples:
            if example.telemetry.channel_names != channel_names:
                raise ValueError("all examples in a batch must have identical channel_names")
            if example.telemetry.shape != shape:
                raise ValueError("all examples in a batch must have identical telemetry shapes")
        return cls(
            telemetry=np.stack([example.telemetry.values for example in examples], axis=0),
            static_features=tuple(example.static_features for example in examples),
            targets=tuple(example.targets for example in examples),
            metadata=tuple(example.metadata for example in examples),
            channel_names=channel_names,
        )

    @property
    def batch_size(self) -> int:
        return int(self.telemetry.shape[0])

    @property
    def distance_bins(self) -> int:
        if self.telemetry.ndim != 3:
            return 0
        return int(self.telemetry.shape[2])

    def target_matrix(self) -> np.ndarray:
        if not self.targets:
            return np.empty((0, len(TARGET_VECTOR_COLUMNS)), dtype=float)
        return np.stack([target.target_vector() for target in self.targets], axis=0)


def split_fields_are_leakage_safe(fields: Sequence[str]) -> bool:
    """Return whether split fields avoid target/result leakage tokens."""

    for field in fields:
        lowered = str(field).lower()
        if any(token in lowered for token in LEAKAGE_FIELD_TOKENS):
            return False
    return True


def assert_split_fields_are_leakage_safe(fields: Sequence[str]) -> None:
    """Raise if split fields include target, result, or prediction-like fields."""

    if not split_fields_are_leakage_safe(fields):
        raise ValueError(f"split fields include leakage-prone columns: {tuple(fields)}")


def summarize_target_quantile_diagnostics(
    targets: Sequence[UltimateLapTargets],
) -> dict[str, Any]:
    """Aggregate target semantics and promotion-grade interval validity."""

    contracts = Counter(target.target_contract for target in targets)
    semantics = Counter(target.target_semantics for target in targets)
    quantile_semantics = Counter(target.quantile_target_semantics for target in targets)
    explicit_count = sum(target.quantile_targets_were_explicit for target in targets)
    degenerate_count = sum(target.quantile_targets_are_degenerate for target in targets)
    valid_count = sum(target.promotion_grade_quantiles_valid for target in targets)
    blockers = sorted({blocker for target in targets for blocker in target.promotion_blockers})
    row_count = len(targets)
    return {
        "row_count": row_count,
        "target_contracts": dict(contracts),
        "target_semantics": dict(semantics),
        "quantile_target_semantics": dict(quantile_semantics),
        "explicit_quantile_target_rows": int(explicit_count),
        "degenerate_quantile_target_rows": int(degenerate_count),
        "promotion_grade_quantile_rows": int(valid_count),
        "promotion_grade_validation_passed": bool(row_count and valid_count == row_count),
        "promotion_blockers": blockers,
    }


__all__ = [
    "ALLOWED_TARGET_CONTRACTS",
    "ALLOWED_SPLIT_NAMES",
    "ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT",
    "DEGENERATE_QUANTILE_TOLERANCE_SECONDS",
    "IDEAL_LAP_TARGET_CONTRACT",
    "OBSERVED_LAP_TARGET_CONTRACT",
    "TARGET_CONTRACT_SEMANTICS",
    "TARGET_VECTOR_COLUMNS",
    "THEORETICAL_SECTOR_FLOOR_TARGET_CONTRACT",
    "THEORETICAL_TARGET_CONTRACTS",
    "DistanceNormalizedTelemetryTensor",
    "UltimateLapMetadata",
    "UltimateLapSplitKey",
    "UltimateLapTargets",
    "UltimateLapTelemetryBatch",
    "UltimateLapTelemetryExample",
    "UltimateLapTelemetryInput",
    "assert_split_fields_are_leakage_safe",
    "split_fields_are_leakage_safe",
    "summarize_target_quantile_diagnostics",
]
