"""Pure, conservative Grand Prix grid and actual-start resolution.

This module deliberately keeps four facts separate:

* the Grand Prix qualifying classification;
* the classification that supplies the provisional Grand Prix grid;
* an explicitly published pre-race grid;
* the cars that actually started.

It does not implement FIA penalty ordering.  A penalty, withdrawal,
disqualification, or pit-lane decision published after the latest complete grid
revision makes that scheduled grid incomplete until a newer complete revision is
provided.  This is intentional: silently compacting positions or applying grid
drops in input order would invent regulatory evidence.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Union

from packages.f1.domain.weekend import GridTarget, Session, WeekendContract


Timestamp = Union[str, datetime]


class ResolutionStatus(str, Enum):
    """Confidence state for a classification, grid, or actual-start output."""

    RESOLVED = "resolved"
    PARTIAL = "partial"
    UNRESOLVED = "unresolved"
    CONFLICTED = "conflicted"


class EvidenceSource(str, Enum):
    """Authoritative source family used by a resolved output or row."""

    QUALIFYING_CLASSIFICATION = "qualifying_classification"
    SPRINT_CLASSIFICATION = "sprint_classification"
    OFFICIAL_GRID_REVISION = "official_grid_revision"
    GRID_ADJUSTMENT = "grid_adjustment"
    ACTUAL_START_EVIDENCE = "actual_start_evidence"
    COMPOSITE_RESOLVER = "composite_resolver"
    NONE = "none"


class ClassificationStatus(str, Enum):
    """Status in a session classification, not a starting-grid status."""

    CLASSIFIED = "classified"
    UNCLASSIFIED = "unclassified"
    DISQUALIFIED = "disqualified"
    DID_NOT_START = "did_not_start"
    WITHDRAWN = "withdrawn"
    UNKNOWN = "unknown"


class GridEntryStatus(str, Enum):
    """Status of a car in a scheduled grid or actual-start order."""

    GRID = "grid"
    PIT_LANE = "pit_lane"
    WITHDRAWN = "withdrawn"
    DISQUALIFIED = "disqualified"
    DID_NOT_START = "did_not_start"
    PENALTY_PENDING = "penalty_pending"
    STARTED = "started"
    STARTED_PIT_LANE = "started_pit_lane"
    UNRESOLVED = "unresolved"


class GridRevisionPhase(str, Enum):
    """Publication phase for an official grid revision."""

    PROVISIONAL_PRE_RACE = "provisional_pre_race"
    FINAL_PRE_RACE = "final_pre_race"
    POST_START = "post_start"


class GridAdjustmentKind(str, Enum):
    """Grid decision that cannot safely reorder the field on its own."""

    GRID_DROP = "grid_drop"
    BACK_OF_GRID = "back_of_grid"
    PIT_LANE_START = "pit_lane_start"
    DISQUALIFICATION = "disqualification"
    WITHDRAWAL = "withdrawal"


@dataclass(frozen=True)
class OfficialGridDecision:
    """One decision explicitly carried by an official grid publication.

    A fully published grid can contain the effect of a penalty while still
    exposing its reason.  Keeping that decision on the row avoids the former
    loss of ``places`` and ``reason`` when an :class:`OfficialGridEntry` was
    converted into model evidence.
    """

    kind: GridAdjustmentKind
    evidence_id: str
    places: int | None = None
    reason: str | None = None
    evidence_complete: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.kind, GridAdjustmentKind):
            raise TypeError("official grid decision kind must be a GridAdjustmentKind")
        evidence_id = str(self.evidence_id).strip()
        if not evidence_id:
            raise ValueError("official grid decision evidence_id must be non-empty")
        object.__setattr__(self, "evidence_id", evidence_id)
        object.__setattr__(
            self,
            "evidence_complete",
            _require_bool(self.evidence_complete, "official grid decision evidence_complete"),
        )
        if self.places is not None:
            if isinstance(self.places, bool) or not isinstance(self.places, int) or self.places <= 0:
                raise ValueError("official grid decision places must be a positive integer or None")
            if self.kind not in {GridAdjustmentKind.GRID_DROP, GridAdjustmentKind.BACK_OF_GRID}:
                raise ValueError("official grid decision places only apply to grid penalties")
        if self.reason is not None:
            reason = str(self.reason).strip()
            object.__setattr__(self, "reason", reason or None)


class ActualStartStatus(str, Enum):
    """Explicit observed start status, separate from the scheduled grid."""

    STARTED = "started"
    STARTED_PIT_LANE = "started_pit_lane"
    DID_NOT_START = "did_not_start"
    WITHDRAWN = "withdrawn"
    DISQUALIFIED = "disqualified"
    UNKNOWN = "unknown"


class RacePredictionHorizon(str, Enum):
    """Distinct information products for a pre-Race forecast.

    The post-Qualifying product is only a grid *proxy*.  It must never be
    relabelled as the published final grid product.
    """

    POST_QUALIFYING_PRE_GRID = "post_qualifying_pre_grid"
    POST_GRID_PRE_RACE = "post_grid_pre_race"


def _require_driver_id(value: str) -> str:
    driver_id = str(value).strip()
    if not driver_id or driver_id.lower() in {"nan", "none", "null", "<na>"}:
        raise ValueError("driver_id must be non-empty")
    return driver_id


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a bool")
    return value


def _require_optional_position(value: int | None, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer or None")
    return value


def _timestamp(value: Timestamp, label: str = "as_of") -> tuple[datetime, str]:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError(f"{label} must be a non-empty timezone-aware timestamp")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{label} is not a valid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    utc = parsed.astimezone(timezone.utc)
    return utc, utc.isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ClassificationEntry:
    """One immutable row from a qualifying or Sprint classification."""

    driver_id: str
    position: int | None
    status: ClassificationStatus = ClassificationStatus.CLASSIFIED
    evidence_complete: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "driver_id", _require_driver_id(self.driver_id))
        object.__setattr__(
            self,
            "evidence_complete",
            _require_bool(self.evidence_complete, "classification entry evidence_complete"),
        )
        if not isinstance(self.status, ClassificationStatus):
            raise TypeError("classification status must be a ClassificationStatus")
        object.__setattr__(
            self,
            "position",
            _require_optional_position(self.position, "classification position"),
        )


@dataclass(frozen=True)
class ClassificationSnapshot:
    """Session classification available at one explicit timestamp."""

    session: Session
    entries: tuple[ClassificationEntry, ...]
    as_of: Timestamp
    evidence_id: str
    evidence_complete: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.session, Session):
            raise TypeError("classification session must be a Session")
        object.__setattr__(self, "entries", tuple(self.entries))
        if any(not isinstance(entry, ClassificationEntry) for entry in self.entries):
            raise TypeError("classification entries must be ClassificationEntry values")
        evidence_id = str(self.evidence_id).strip()
        if not evidence_id:
            raise ValueError("classification evidence_id must be non-empty")
        object.__setattr__(self, "evidence_id", evidence_id)
        object.__setattr__(
            self,
            "evidence_complete",
            _require_bool(self.evidence_complete, "classification snapshot evidence_complete"),
        )
        _timestamp(self.as_of, "classification as_of")


@dataclass(frozen=True)
class OfficialGridEntry:
    """One row in an explicitly published official grid revision.

    ``position`` is a physical grid-box position.  Pit-lane cars and nonstarters
    therefore carry ``None`` instead of a synthetic position after the last grid
    box.
    """

    driver_id: str
    position: int | None
    status: GridEntryStatus = GridEntryStatus.GRID
    adjustments: tuple[GridAdjustmentKind, ...] = ()
    decisions: tuple[OfficialGridDecision, ...] = ()
    evidence_complete: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "driver_id", _require_driver_id(self.driver_id))
        object.__setattr__(
            self,
            "evidence_complete",
            _require_bool(self.evidence_complete, "official grid entry evidence_complete"),
        )
        if not isinstance(self.status, GridEntryStatus):
            raise TypeError("official grid status must be a GridEntryStatus")
        object.__setattr__(
            self,
            "position",
            _require_optional_position(self.position, "grid position"),
        )
        object.__setattr__(self, "adjustments", tuple(self.adjustments))
        if any(not isinstance(kind, GridAdjustmentKind) for kind in self.adjustments):
            raise TypeError("official grid adjustments must be GridAdjustmentKind values")
        object.__setattr__(self, "decisions", tuple(self.decisions))
        if any(not isinstance(decision, OfficialGridDecision) for decision in self.decisions):
            raise TypeError("official grid decisions must be OfficialGridDecision values")
        decision_kinds = tuple(decision.kind for decision in self.decisions)
        if self.decisions and self.adjustments and decision_kinds != self.adjustments:
            raise ValueError("official grid decisions and adjustment kinds disagree")
        if self.decisions and not self.adjustments:
            object.__setattr__(self, "adjustments", decision_kinds)


@dataclass(frozen=True)
class OfficialGridRevision:
    """Phase-tagged official grid publication."""

    revision_id: str
    phase: GridRevisionPhase
    entries: tuple[OfficialGridEntry, ...]
    as_of: Timestamp
    evidence_complete: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.phase, GridRevisionPhase):
            raise TypeError("official grid phase must be a GridRevisionPhase")
        revision_id = str(self.revision_id).strip()
        if not revision_id:
            raise ValueError("official grid revision_id must be non-empty")
        object.__setattr__(self, "revision_id", revision_id)
        object.__setattr__(self, "entries", tuple(self.entries))
        if any(not isinstance(entry, OfficialGridEntry) for entry in self.entries):
            raise TypeError("official grid entries must be OfficialGridEntry values")
        object.__setattr__(
            self,
            "evidence_complete",
            _require_bool(self.evidence_complete, "official grid revision evidence_complete"),
        )
        _timestamp(self.as_of, "official grid revision as_of")


@dataclass(frozen=True)
class GridAdjustment:
    """A published grid decision awaiting a complete revised order."""

    driver_id: str
    kind: GridAdjustmentKind
    as_of: Timestamp
    evidence_id: str
    places: int | None = None
    reason: str | None = None
    evidence_complete: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "driver_id", _require_driver_id(self.driver_id))
        object.__setattr__(
            self,
            "evidence_complete",
            _require_bool(self.evidence_complete, "grid adjustment evidence_complete"),
        )
        if not isinstance(self.kind, GridAdjustmentKind):
            raise TypeError("grid adjustment kind must be a GridAdjustmentKind")
        evidence_id = str(self.evidence_id).strip()
        if not evidence_id:
            raise ValueError("grid adjustment evidence_id must be non-empty")
        object.__setattr__(self, "evidence_id", evidence_id)
        if self.places is not None:
            if (
                isinstance(self.places, bool)
                or not isinstance(self.places, int)
                or self.places <= 0
            ):
                raise ValueError("grid adjustment places must be a positive integer or None")
            if self.kind not in {
                GridAdjustmentKind.GRID_DROP,
                GridAdjustmentKind.BACK_OF_GRID,
            }:
                raise ValueError("places is only valid for grid-drop or back-of-grid evidence")
        if self.reason is not None:
            reason = str(self.reason).strip()
            object.__setattr__(self, "reason", reason or None)
        _timestamp(self.as_of, "grid adjustment as_of")


@dataclass(frozen=True)
class ActualStartEntry:
    """One explicitly observed actual-start row."""

    driver_id: str
    status: ActualStartStatus
    start_order: int | None = None
    evidence_complete: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "driver_id", _require_driver_id(self.driver_id))
        object.__setattr__(
            self,
            "evidence_complete",
            _require_bool(self.evidence_complete, "actual start entry evidence_complete"),
        )
        if not isinstance(self.status, ActualStartStatus):
            raise TypeError("actual start status must be an ActualStartStatus")
        object.__setattr__(
            self,
            "start_order",
            _require_optional_position(self.start_order, "start order"),
        )


@dataclass(frozen=True)
class ActualStartSnapshot:
    """Observed starters and nonstarters at one explicit timestamp."""

    entries: tuple[ActualStartEntry, ...]
    as_of: Timestamp
    evidence_id: str
    evidence_complete: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))
        if any(not isinstance(entry, ActualStartEntry) for entry in self.entries):
            raise TypeError("actual-start entries must be ActualStartEntry values")
        evidence_id = str(self.evidence_id).strip()
        if not evidence_id:
            raise ValueError("actual-start evidence_id must be non-empty")
        object.__setattr__(self, "evidence_id", evidence_id)
        object.__setattr__(
            self,
            "evidence_complete",
            _require_bool(self.evidence_complete, "actual start snapshot evidence_complete"),
        )
        _timestamp(self.as_of, "actual-start as_of")


@dataclass(frozen=True)
class ResolvedClassificationEntry:
    """Classification row with explicit provenance."""

    driver_id: str
    position: int | None
    status: ClassificationStatus
    source: EvidenceSource
    as_of: str
    evidence_complete: bool
    evidence_id: str


@dataclass(frozen=True)
class ResolvedGridAdjustment:
    """One published grid decision preserved without applying an inferred order."""

    kind: GridAdjustmentKind
    places: int | None
    reason: str | None
    status: GridEntryStatus
    source: EvidenceSource
    as_of: str
    evidence_complete: bool
    evidence_id: str


@dataclass(frozen=True)
class ClassificationResolution:
    """Resolved classification as of the requested cutoff."""

    status: ResolutionStatus
    source: EvidenceSource
    session: Session
    as_of: str
    evidence_as_of: str | None
    evidence_complete: bool
    entries: tuple[ResolvedClassificationEntry, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    ignored_evidence_ids: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedOrderEntry:
    """Scheduled-grid or actual-start row with explicit provenance."""

    driver_id: str
    position: int | None
    status: GridEntryStatus
    source: EvidenceSource
    as_of: str
    evidence_complete: bool
    evidence_ids: tuple[str, ...] = ()
    adjustments: tuple[GridAdjustmentKind, ...] = ()
    adjustment_evidence: tuple[ResolvedGridAdjustment, ...] = ()
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class OrderResolution:
    """One provenance-bearing provisional, scheduled, or actual-start order."""

    status: ResolutionStatus
    source: EvidenceSource
    phase: str
    as_of: str
    evidence_as_of: str | None
    evidence_complete: bool
    entries: tuple[ResolvedOrderEntry, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    ignored_evidence_ids: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class GrandPrixStartResolution:
    """Complete separation of classification, grids, and actual starters."""

    status: ResolutionStatus
    source: EvidenceSource
    as_of: str
    evidence_complete: bool
    qualifying_classification: ClassificationResolution
    provisional_source_classification: ClassificationResolution
    provisional_grid: OrderResolution
    scheduled_grid: OrderResolution
    actual_start: OrderResolution
    evidence_ids: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()

    @property
    def actual_starters(self) -> tuple[ResolvedOrderEntry, ...]:
        """Observed starters only; never inferred from ``scheduled_grid``."""

        return tuple(
            entry
            for entry in self.actual_start.entries
            if entry.status in {GridEntryStatus.STARTED, GridEntryStatus.STARTED_PIT_LANE}
        )


@dataclass(frozen=True)
class RaceGridSnapshotEntry:
    """One immutable input row for a named pre-Race prediction horizon."""

    driver_id: str
    grid_position: int | None
    status: GridEntryStatus
    starter_eligible: bool
    pit_lane_start: bool
    penalty_evidence: tuple[ResolvedGridAdjustment, ...]
    evidence_ids: tuple[str, ...]
    evidence_complete: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "driver_id", _require_driver_id(self.driver_id))
        object.__setattr__(
            self,
            "grid_position",
            _require_optional_position(self.grid_position, "snapshot grid position"),
        )
        if not isinstance(self.status, GridEntryStatus):
            raise TypeError("snapshot grid status must be a GridEntryStatus")
        object.__setattr__(
            self,
            "starter_eligible",
            _require_bool(self.starter_eligible, "snapshot starter_eligible"),
        )
        object.__setattr__(
            self,
            "pit_lane_start",
            _require_bool(self.pit_lane_start, "snapshot pit_lane_start"),
        )
        object.__setattr__(self, "penalty_evidence", tuple(self.penalty_evidence))
        if any(
            not isinstance(adjustment, ResolvedGridAdjustment)
            for adjustment in self.penalty_evidence
        ):
            raise TypeError("penalty_evidence must contain ResolvedGridAdjustment values")
        evidence_ids = tuple(str(value).strip() for value in self.evidence_ids)
        if any(not value for value in evidence_ids):
            raise ValueError("snapshot evidence_ids must be non-empty strings")
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(
            self,
            "evidence_complete",
            _require_bool(self.evidence_complete, "snapshot entry evidence_complete"),
        )


@dataclass(frozen=True)
class RaceGridSnapshot:
    """Immutable, provenance-bearing race-grid model input.

    ``available`` is deliberately stricter than merely having rows.  The final
    grid horizon is available only from a complete ``FINAL_PRE_RACE`` official
    revision with no unresolved decisions.  Callers may inspect unavailable
    snapshots, but ``require_available`` prevents accidental forecast emission.
    """

    horizon: RacePredictionHorizon
    prediction_as_of: str
    publication_as_of: str | None
    resolution_status: ResolutionStatus
    source: EvidenceSource
    entries: tuple[RaceGridSnapshotEntry, ...]
    revision_ids: tuple[str, ...]
    available: bool
    evidence_complete: bool
    unresolved_state_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.horizon, RacePredictionHorizon):
            raise TypeError("snapshot horizon must be a RacePredictionHorizon")
        _, prediction_text = _timestamp(self.prediction_as_of, "snapshot prediction_as_of")
        object.__setattr__(self, "prediction_as_of", prediction_text)
        if self.publication_as_of is not None:
            publication_time, publication_text = _timestamp(
                self.publication_as_of,
                "snapshot publication_as_of",
            )
            prediction_time, _ = _timestamp(prediction_text, "snapshot prediction_as_of")
            if publication_time > prediction_time:
                raise ValueError("snapshot publication_as_of cannot be after prediction_as_of")
            object.__setattr__(self, "publication_as_of", publication_text)
        if not isinstance(self.resolution_status, ResolutionStatus):
            raise TypeError("snapshot resolution_status must be a ResolutionStatus")
        if not isinstance(self.source, EvidenceSource):
            raise TypeError("snapshot source must be an EvidenceSource")
        object.__setattr__(self, "entries", tuple(self.entries))
        if any(not isinstance(entry, RaceGridSnapshotEntry) for entry in self.entries):
            raise TypeError("snapshot entries must be RaceGridSnapshotEntry values")
        object.__setattr__(
            self,
            "revision_ids",
            tuple(str(value).strip() for value in self.revision_ids if str(value).strip()),
        )
        object.__setattr__(self, "available", _require_bool(self.available, "snapshot available"))
        object.__setattr__(
            self,
            "evidence_complete",
            _require_bool(self.evidence_complete, "snapshot evidence_complete"),
        )
        object.__setattr__(
            self,
            "unresolved_state_flags",
            tuple(str(value) for value in self.unresolved_state_flags),
        )
        if self.available and (not self.evidence_complete or not self.entries):
            raise ValueError("available snapshots require complete, non-empty evidence")

    def require_available(self) -> "RaceGridSnapshot":
        """Return this snapshot or fail closed before model inference."""

        if not self.available:
            detail = ",".join(self.unresolved_state_flags) or "grid_evidence_unavailable"
            raise ValueError(
                f"{self.horizon.value} snapshot is unavailable: {detail}"
            )
        return self


GRID_CAPTURE_SCHEMA_VERSION = "f1_first_seen_post_grid_pre_race_v1"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


@dataclass(frozen=True)
class RaceGridCapture:
    """Durable first-seen envelope around one immutable grid snapshot.

    ``captured_at`` is an observation timestamp, not a retrospectively inferred
    publication timestamp.  The canonical provider response is retained and
    hashed so a backtest can prove exactly what was first observed.  Captures
    are append-only; :func:`persist_race_grid_capture` never overwrites a file.
    """

    capture_id: str
    year: int
    round_number: int
    provider: str
    source_endpoint: str
    captured_at: Timestamp
    first_published_at: Timestamp
    race_start_at: Timestamp
    publication_pre_race_verified: bool
    raw_payload_json: str
    raw_payload_sha256: str
    snapshot: RaceGridSnapshot
    revision_phase: GridRevisionPhase
    meeting_key: str | None = None
    session_key: str | None = None
    source_document_url: str | None = None
    source_document_sha256: str | None = None
    publication_time_semantics: str = "first_seen_upper_bound"
    schema_version: str = GRID_CAPTURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        capture_id = str(self.capture_id).strip()
        if not capture_id:
            raise ValueError("grid capture_id must be non-empty")
        object.__setattr__(self, "capture_id", capture_id)
        if isinstance(self.year, bool) or int(self.year) < 1950:
            raise ValueError("grid capture year is invalid")
        object.__setattr__(self, "year", int(self.year))
        if isinstance(self.round_number, bool) or int(self.round_number) <= 0:
            raise ValueError("grid capture round_number must be positive")
        object.__setattr__(self, "round_number", int(self.round_number))
        provider = str(self.provider).strip()
        endpoint = str(self.source_endpoint).strip()
        if not provider or not endpoint:
            raise ValueError("grid capture provider and source_endpoint must be non-empty")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "source_endpoint", endpoint)
        _, captured_text = _timestamp(self.captured_at, "grid captured_at")
        object.__setattr__(self, "captured_at", captured_text)
        published_time, published_text = _timestamp(
            self.first_published_at, "grid first_published_at"
        )
        captured_time, _ = _timestamp(captured_text, "grid captured_at")
        if published_time > captured_time:
            raise ValueError("grid first_published_at cannot be after captured_at")
        object.__setattr__(self, "first_published_at", published_text)
        race_start_time, race_start_text = _timestamp(self.race_start_at, "grid race_start_at")
        object.__setattr__(self, "race_start_at", race_start_text)
        verified = _require_bool(
            self.publication_pre_race_verified,
            "grid publication_pre_race_verified",
        )
        if verified != bool(published_time < race_start_time):
            raise ValueError("grid pre-race verification disagrees with publication/start times")
        object.__setattr__(self, "publication_pre_race_verified", verified)
        if not isinstance(self.snapshot, RaceGridSnapshot):
            raise TypeError("grid capture snapshot must be a RaceGridSnapshot")
        if not isinstance(self.revision_phase, GridRevisionPhase):
            raise TypeError("grid capture revision_phase must be a GridRevisionPhase")
        if self.snapshot.horizon is not RacePredictionHorizon.POST_GRID_PRE_RACE:
            raise ValueError("first-seen grid capture must use post_grid_pre_race horizon")
        snapshot_time, _ = _timestamp(self.snapshot.prediction_as_of, "snapshot prediction_as_of")
        if snapshot_time != published_time:
            raise ValueError("snapshot prediction_as_of must equal first_published_at")
        if self.snapshot.publication_as_of != published_text:
            raise ValueError("snapshot publication_as_of must equal first_published_at")
        if self.snapshot.available and not verified:
            raise ValueError("post-grid snapshot cannot be available without proven pre-race publication")
        raw_text = str(self.raw_payload_json)
        try:
            decoded = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError("grid capture raw_payload_json is invalid") from exc
        canonical = _canonical_json(decoded)
        object.__setattr__(self, "raw_payload_json", canonical)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if str(self.raw_payload_sha256).strip().lower() != digest:
            raise ValueError("grid capture raw payload hash mismatch")
        object.__setattr__(self, "raw_payload_sha256", digest)
        for field_name in ("meeting_key", "session_key"):
            value = getattr(self, field_name)
            if value is not None:
                normalized = str(value).strip()
                object.__setattr__(self, field_name, normalized or None)
        document_url = (
            None if self.source_document_url is None else str(self.source_document_url).strip()
        )
        object.__setattr__(self, "source_document_url", document_url or None)
        document_hash = (
            None
            if self.source_document_sha256 is None
            else str(self.source_document_sha256).strip().lower()
        )
        if document_hash is not None and not re.fullmatch(r"[0-9a-f]{64}", document_hash):
            raise ValueError("source_document_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "source_document_sha256", document_hash)
        semantics = str(self.publication_time_semantics).strip()
        if semantics not in {
            "first_seen_upper_bound",
            "authoritative_document_timestamp",
        }:
            raise ValueError("unsupported grid publication_time_semantics")
        if semantics == "first_seen_upper_bound" and published_time != captured_time:
            raise ValueError("first-seen publication time must equal captured_at")
        if semantics == "authoritative_document_timestamp" and (
            not document_url or not document_hash
        ):
            raise ValueError(
                "authoritative publication time requires source document URL and SHA-256"
            )
        object.__setattr__(self, "publication_time_semantics", semantics)
        if self.schema_version != GRID_CAPTURE_SCHEMA_VERSION:
            raise ValueError(f"unsupported grid capture schema: {self.schema_version!r}")

    @property
    def raw_payload(self) -> object:
        return json.loads(self.raw_payload_json)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "capture_id": self.capture_id,
            "year": self.year,
            "round_number": self.round_number,
            "provider": self.provider,
            "source_endpoint": self.source_endpoint,
            "meeting_key": self.meeting_key,
            "session_key": self.session_key,
            "captured_at": self.captured_at,
            "first_published_at": self.first_published_at,
            "race_start_at": self.race_start_at,
            "publication_pre_race_verified": self.publication_pre_race_verified,
            "publication_time_semantics": self.publication_time_semantics,
            "source_document_url": self.source_document_url,
            "source_document_sha256": self.source_document_sha256,
            "raw_payload_sha256": self.raw_payload_sha256,
            "revision_phase": self.revision_phase.value,
            "raw_payload": self.raw_payload,
            "snapshot": _race_grid_snapshot_payload(self.snapshot),
        }


def _resolved_adjustment_payload(value: ResolvedGridAdjustment) -> dict[str, object]:
    return {
        "kind": value.kind.value,
        "places": value.places,
        "reason": value.reason,
        "status": value.status.value,
        "source": value.source.value,
        "as_of": value.as_of,
        "evidence_complete": value.evidence_complete,
        "evidence_id": value.evidence_id,
    }


def _race_grid_snapshot_payload(snapshot: RaceGridSnapshot) -> dict[str, object]:
    return {
        "horizon": snapshot.horizon.value,
        "prediction_as_of": snapshot.prediction_as_of,
        "publication_as_of": snapshot.publication_as_of,
        "resolution_status": snapshot.resolution_status.value,
        "source": snapshot.source.value,
        "revision_ids": list(snapshot.revision_ids),
        "available": snapshot.available,
        "evidence_complete": snapshot.evidence_complete,
        "unresolved_state_flags": list(snapshot.unresolved_state_flags),
        "entries": [
            {
                "driver_id": entry.driver_id,
                "grid_position": entry.grid_position,
                "status": entry.status.value,
                "starter_eligible": entry.starter_eligible,
                "pit_lane_start": entry.pit_lane_start,
                "penalty_evidence": [
                    _resolved_adjustment_payload(adjustment)
                    for adjustment in entry.penalty_evidence
                ],
                "evidence_ids": list(entry.evidence_ids),
                "evidence_complete": entry.evidence_complete,
            }
            for entry in snapshot.entries
        ],
    }


def _resolved_adjustment_from_payload(value: object) -> ResolvedGridAdjustment:
    row = _require_mapping(value, "grid adjustment evidence")
    return ResolvedGridAdjustment(
        kind=GridAdjustmentKind(str(row.get("kind"))),
        places=(None if row.get("places") is None else int(row["places"])),
        reason=(None if row.get("reason") is None else str(row["reason"])),
        status=GridEntryStatus(str(row.get("status"))),
        source=EvidenceSource(str(row.get("source"))),
        as_of=str(row.get("as_of")),
        evidence_complete=_require_bool(
            row.get("evidence_complete"), "grid adjustment evidence_complete"
        ),
        evidence_id=str(row.get("evidence_id")),
    )


def _race_grid_snapshot_from_payload(value: object) -> RaceGridSnapshot:
    payload = _require_mapping(value, "grid snapshot")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("grid snapshot entries must be a list")
    entries: list[RaceGridSnapshotEntry] = []
    for raw_entry in raw_entries:
        row = _require_mapping(raw_entry, "grid snapshot entry")
        penalty_evidence = row.get("penalty_evidence", [])
        if not isinstance(penalty_evidence, list):
            raise ValueError("grid snapshot penalty_evidence must be a list")
        evidence_ids = row.get("evidence_ids", [])
        if not isinstance(evidence_ids, list):
            raise ValueError("grid snapshot evidence_ids must be a list")
        entries.append(
            RaceGridSnapshotEntry(
                driver_id=str(row.get("driver_id")),
                grid_position=(
                    None if row.get("grid_position") is None else int(row["grid_position"])
                ),
                status=GridEntryStatus(str(row.get("status"))),
                starter_eligible=_require_bool(
                    row.get("starter_eligible"), "snapshot starter_eligible"
                ),
                pit_lane_start=_require_bool(
                    row.get("pit_lane_start"), "snapshot pit_lane_start"
                ),
                penalty_evidence=tuple(
                    _resolved_adjustment_from_payload(item) for item in penalty_evidence
                ),
                evidence_ids=tuple(str(item) for item in evidence_ids),
                evidence_complete=_require_bool(
                    row.get("evidence_complete"), "snapshot entry evidence_complete"
                ),
            )
        )
    revision_ids = payload.get("revision_ids", [])
    flags = payload.get("unresolved_state_flags", [])
    if not isinstance(revision_ids, list) or not isinstance(flags, list):
        raise ValueError("grid snapshot revision_ids and flags must be lists")
    return RaceGridSnapshot(
        horizon=RacePredictionHorizon(str(payload.get("horizon"))),
        prediction_as_of=str(payload.get("prediction_as_of")),
        publication_as_of=(
            None
            if payload.get("publication_as_of") is None
            else str(payload.get("publication_as_of"))
        ),
        resolution_status=ResolutionStatus(str(payload.get("resolution_status"))),
        source=EvidenceSource(str(payload.get("source"))),
        entries=tuple(entries),
        revision_ids=tuple(str(item) for item in revision_ids),
        available=_require_bool(payload.get("available"), "snapshot available"),
        evidence_complete=_require_bool(
            payload.get("evidence_complete"), "snapshot evidence_complete"
        ),
        unresolved_state_flags=tuple(str(item) for item in flags),
    )


def race_grid_capture_from_payload(value: object) -> RaceGridCapture:
    """Validate and reconstruct a first-seen capture payload."""

    payload = _require_mapping(value, "grid capture")
    raw_payload = payload.get("raw_payload")
    return RaceGridCapture(
        schema_version=str(payload.get("schema_version")),
        capture_id=str(payload.get("capture_id")),
        year=int(payload.get("year", 0)),
        round_number=int(payload.get("round_number", 0)),
        provider=str(payload.get("provider")),
        source_endpoint=str(payload.get("source_endpoint")),
        meeting_key=(
            None if payload.get("meeting_key") is None else str(payload.get("meeting_key"))
        ),
        session_key=(
            None if payload.get("session_key") is None else str(payload.get("session_key"))
        ),
        captured_at=str(payload.get("captured_at")),
        first_published_at=str(payload.get("first_published_at")),
        race_start_at=str(payload.get("race_start_at")),
        publication_pre_race_verified=_require_bool(
            payload.get("publication_pre_race_verified"),
            "grid publication_pre_race_verified",
        ),
        publication_time_semantics=str(payload.get("publication_time_semantics")),
        source_document_url=(
            None
            if payload.get("source_document_url") is None
            else str(payload.get("source_document_url"))
        ),
        source_document_sha256=(
            None
            if payload.get("source_document_sha256") is None
            else str(payload.get("source_document_sha256"))
        ),
        raw_payload_json=_canonical_json(raw_payload),
        raw_payload_sha256=str(payload.get("raw_payload_sha256")),
        revision_phase=GridRevisionPhase(str(payload.get("revision_phase"))),
        snapshot=_race_grid_snapshot_from_payload(payload.get("snapshot")),
    )


def load_race_grid_capture(path: str | Path) -> RaceGridCapture:
    """Load one capture with schema, provenance-hash, and type validation."""

    capture_path = Path(path)
    try:
        payload = json.loads(capture_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid first-seen grid capture: {capture_path}") from exc
    return race_grid_capture_from_payload(payload)


def persist_race_grid_capture(
    capture: RaceGridCapture,
    directory: str | Path,
) -> Path:
    """Append one capture and refuse every overwrite or content mutation."""

    if not isinstance(capture, RaceGridCapture):
        raise TypeError("capture must be a RaceGridCapture")
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_token = re.sub(r"[^0-9A-Za-z]+", "", str(capture.captured_at))
    filename = f"grid_{timestamp_token}_{capture.capture_id}.json"
    output = output_dir / filename
    encoded = json.dumps(capture.to_payload(), indent=2, sort_keys=True) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"immutable grid capture already exists with other content: {output}")
        return output
    try:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    except FileExistsError:
        if output.read_text(encoding="utf-8") != encoded:
            raise
    return output


def build_race_grid_capture(
    contract: WeekendContract,
    *,
    year: int,
    round_number: int,
    provider: str,
    source_endpoint: str,
    captured_at: Timestamp,
    first_published_at: Timestamp,
    race_start_at: Timestamp,
    revision: OfficialGridRevision,
    raw_payload: object,
    meeting_key: str | None = None,
    session_key: str | None = None,
    publication_time_semantics: str = "first_seen_upper_bound",
    source_document_url: str | None = None,
    source_document_sha256: str | None = None,
) -> RaceGridCapture:
    """Resolve and envelope an official grid revision without result leakage.

    The structured rows must come from the supplied publication; this helper
    never reads a race-result file and never repairs positions from a later
    classification.  A provisional or incomplete revision is still stored,
    but its nested snapshot remains unavailable for model inference.
    """

    if not isinstance(contract, WeekendContract):
        raise TypeError("contract must be a WeekendContract")
    if not isinstance(revision, OfficialGridRevision):
        raise TypeError("revision must be an OfficialGridRevision")
    _, published_text = _timestamp(first_published_at, "grid first_published_at")
    published_time, _ = _timestamp(published_text, "grid first_published_at")
    race_start_time, race_start_text = _timestamp(race_start_at, "grid race_start_at")
    publication_pre_race_verified = bool(published_time < race_start_time)
    _, revision_text = _timestamp(revision.as_of, "official grid revision as_of")
    if revision_text != published_text:
        raise ValueError("official grid revision as_of must equal first_published_at")
    effective_revision = (
        revision
        if publication_pre_race_verified
        else replace(revision, evidence_complete=False)
    )
    resolution = resolve_grand_prix_start(
        contract,
        as_of=published_text,
        official_grid_revisions=(effective_revision,),
    )
    snapshot = build_race_grid_snapshot(
        resolution,
        horizon=RacePredictionHorizon.POST_GRID_PRE_RACE,
    )
    raw_json = _canonical_json(raw_payload)
    raw_hash = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
    identity = _canonical_json(
        {
            "year": int(year),
            "round_number": int(round_number),
            "provider": str(provider),
            "source_endpoint": str(source_endpoint),
            "first_published_at": published_text,
            "revision_id": revision.revision_id,
            "raw_payload_sha256": raw_hash,
        }
    )
    capture_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return RaceGridCapture(
        capture_id=capture_id,
        year=int(year),
        round_number=int(round_number),
        provider=provider,
        source_endpoint=source_endpoint,
        meeting_key=meeting_key,
        session_key=session_key,
        captured_at=captured_at,
        first_published_at=published_text,
        race_start_at=race_start_text,
        publication_pre_race_verified=publication_pre_race_verified,
        publication_time_semantics=publication_time_semantics,
        source_document_url=source_document_url,
        source_document_sha256=source_document_sha256,
        raw_payload_json=raw_json,
        raw_payload_sha256=raw_hash,
        snapshot=snapshot,
        revision_phase=revision.phase,
    )


def _source_for_session(session: Session) -> EvidenceSource:
    if session is Session.QUALIFYING:
        return EvidenceSource.QUALIFYING_CLASSIFICATION
    if session is Session.SPRINT:
        return EvidenceSource.SPRINT_CLASSIFICATION
    return EvidenceSource.NONE


def _classification_signature(snapshot: ClassificationSnapshot) -> tuple[object, ...]:
    return (
        snapshot.session.value,
        bool(snapshot.evidence_complete),
        tuple(
            sorted(
                (
                    entry.driver_id,
                    entry.position,
                    entry.status.value,
                    bool(entry.evidence_complete),
                )
                for entry in snapshot.entries
            )
        ),
    )


def _revision_signature(revision: OfficialGridRevision) -> tuple[object, ...]:
    return (
        revision.phase.value,
        bool(revision.evidence_complete),
        tuple(
            sorted(
                (
                    entry.driver_id,
                    entry.position,
                    entry.status.value,
                    tuple(kind.value for kind in entry.adjustments),
                    tuple(
                        (
                            decision.kind.value,
                            decision.places,
                            decision.reason,
                            decision.evidence_id,
                            bool(decision.evidence_complete),
                        )
                        for decision in entry.decisions
                    ),
                    bool(entry.evidence_complete),
                )
                for entry in revision.entries
            )
        ),
    )


def _actual_start_signature(snapshot: ActualStartSnapshot) -> tuple[object, ...]:
    return (
        bool(snapshot.evidence_complete),
        tuple(
            sorted(
                (
                    entry.driver_id,
                    entry.status.value,
                    entry.start_order,
                    bool(entry.evidence_complete),
                )
                for entry in snapshot.entries
            )
        ),
    )


def _ordered_entries(entries: Iterable[ResolvedOrderEntry]) -> tuple[ResolvedOrderEntry, ...]:
    return tuple(
        sorted(
            entries,
            key=lambda entry: (
                entry.position is None,
                entry.position if entry.position is not None else 10**9,
                entry.driver_id,
            ),
        )
    )


def _classification_resolution(
    *,
    session: Session,
    snapshots: Sequence[ClassificationSnapshot],
    cutoff: datetime,
    cutoff_text: str,
    expected_field_size: int,
) -> ClassificationResolution:
    source = _source_for_session(session)
    eligible: list[tuple[datetime, ClassificationSnapshot, str]] = []
    ignored: list[str] = []
    for snapshot in snapshots:
        if snapshot.session is not session:
            continue
        evidence_time, evidence_text = _timestamp(snapshot.as_of, "classification as_of")
        if evidence_time > cutoff:
            ignored.append(str(snapshot.evidence_id))
            continue
        eligible.append((evidence_time, snapshot, evidence_text))

    if not eligible:
        return ClassificationResolution(
            status=ResolutionStatus.UNRESOLVED,
            source=source,
            session=session,
            as_of=cutoff_text,
            evidence_as_of=None,
            evidence_complete=False,
            ignored_evidence_ids=tuple(sorted(set(ignored))),
            issues=(f"{session.value}_classification_unavailable_at_cutoff",),
        )

    latest_time = max(item[0] for item in eligible)
    latest = [item for item in eligible if item[0] == latest_time]
    signatures = {_classification_signature(item[1]) for item in latest}
    evidence_ids = tuple(sorted({str(item[1].evidence_id) for item in latest}))
    older_ids = [str(item[1].evidence_id) for item in eligible if item[0] < latest_time]
    ignored.extend(older_ids)
    evidence_text = latest[0][2]
    if len(signatures) != 1:
        return ClassificationResolution(
            status=ResolutionStatus.CONFLICTED,
            source=source,
            session=session,
            as_of=cutoff_text,
            evidence_as_of=evidence_text,
            evidence_complete=False,
            evidence_ids=evidence_ids,
            ignored_evidence_ids=tuple(sorted(set(ignored))),
            issues=(f"conflicting_{session.value}_classifications_at_same_timestamp",),
        )

    snapshot = sorted(latest, key=lambda item: str(item[1].evidence_id))[0][1]
    driver_ids = [entry.driver_id for entry in snapshot.entries]
    numeric_positions = [entry.position for entry in snapshot.entries if entry.position is not None]
    issues: list[str] = []
    conflicted = False
    if len(driver_ids) != len(set(driver_ids)):
        issues.append("duplicate_driver_in_classification")
        conflicted = True
    if len(numeric_positions) != len(set(numeric_positions)):
        issues.append("duplicate_position_in_classification")
        conflicted = True
    if any(position > int(expected_field_size) for position in numeric_positions):
        issues.append("classification_position_exceeds_eligible_field")
        conflicted = True
    if len(driver_ids) != int(expected_field_size):
        issues.append(
            f"classification_field_size_{len(driver_ids)}_expected_{int(expected_field_size)}"
        )
    missing_classified_positions = sorted(
        entry.driver_id
        for entry in snapshot.entries
        if entry.status is ClassificationStatus.CLASSIFIED and entry.position is None
    )
    if missing_classified_positions:
        issues.append(
            "classified_drivers_missing_position:"
            f"{','.join(missing_classified_positions)}"
        )

    resolved_entries = tuple(
        ResolvedClassificationEntry(
            driver_id=entry.driver_id,
            position=entry.position,
            status=entry.status,
            source=source,
            as_of=evidence_text,
            evidence_complete=bool(snapshot.evidence_complete and entry.evidence_complete),
            evidence_id=str(snapshot.evidence_id),
        )
        for entry in snapshot.entries
    )
    complete = bool(
        not conflicted
        and snapshot.evidence_complete
        and len(driver_ids) == int(expected_field_size)
        and all(entry.evidence_complete for entry in snapshot.entries)
        and all(entry.status is not ClassificationStatus.UNKNOWN for entry in snapshot.entries)
        and not missing_classified_positions
    )
    status = (
        ResolutionStatus.CONFLICTED
        if conflicted
        else ResolutionStatus.RESOLVED
        if complete
        else ResolutionStatus.PARTIAL
    )
    return ClassificationResolution(
        status=status,
        source=source,
        session=session,
        as_of=cutoff_text,
        evidence_as_of=evidence_text,
        evidence_complete=complete,
        entries=resolved_entries,
        evidence_ids=evidence_ids,
        ignored_evidence_ids=tuple(sorted(set(ignored))),
        issues=tuple(issues),
    )


def _provisional_grid_from_classification(
    classification: ClassificationResolution,
    *,
    cutoff_text: str,
    expected_field_size: int,
) -> OrderResolution:
    if classification.status in {ResolutionStatus.UNRESOLVED, ResolutionStatus.CONFLICTED}:
        return OrderResolution(
            status=classification.status,
            source=classification.source,
            phase="classification_provisional",
            as_of=cutoff_text,
            evidence_as_of=classification.evidence_as_of,
            evidence_complete=False,
            evidence_ids=classification.evidence_ids,
            ignored_evidence_ids=classification.ignored_evidence_ids,
            issues=classification.issues,
        )

    issues = list(classification.issues)
    rows: list[ResolvedOrderEntry] = []
    for entry in classification.entries:
        position: int | None = None
        status = GridEntryStatus.UNRESOLVED
        if entry.status is ClassificationStatus.CLASSIFIED and entry.position is not None:
            position = entry.position
            status = GridEntryStatus.GRID
        elif entry.status is ClassificationStatus.DISQUALIFIED:
            status = GridEntryStatus.DISQUALIFIED
        elif entry.status is ClassificationStatus.WITHDRAWN:
            status = GridEntryStatus.WITHDRAWN
        elif entry.status is ClassificationStatus.DID_NOT_START:
            status = GridEntryStatus.DID_NOT_START
        else:
            issues.append(f"classification_does_not_resolve_grid_slot:{entry.driver_id}")
        rows.append(
            ResolvedOrderEntry(
                driver_id=entry.driver_id,
                position=position,
                status=status,
                source=classification.source,
                as_of=entry.as_of,
                evidence_complete=bool(
                    entry.evidence_complete
                    and status is GridEntryStatus.GRID
                    and position is not None
                ),
                evidence_ids=(entry.evidence_id,),
            )
        )

    complete = bool(
        classification.evidence_complete
        and len(rows) == int(expected_field_size)
        and all(row.evidence_complete for row in rows)
    )
    return OrderResolution(
        status=ResolutionStatus.RESOLVED if complete else ResolutionStatus.PARTIAL,
        source=classification.source,
        phase="classification_provisional",
        as_of=cutoff_text,
        evidence_as_of=classification.evidence_as_of,
        evidence_complete=complete,
        entries=_ordered_entries(rows),
        evidence_ids=classification.evidence_ids,
        ignored_evidence_ids=classification.ignored_evidence_ids,
        issues=tuple(dict.fromkeys(issues)),
    )


def _select_official_revision(
    revisions: Sequence[OfficialGridRevision],
    *,
    cutoff: datetime,
    cutoff_text: str,
) -> tuple[OfficialGridRevision | None, tuple[str, ...], OrderResolution | None]:
    eligible: list[tuple[datetime, OfficialGridRevision, str]] = []
    ignored: list[str] = []
    for revision in revisions:
        evidence_time, evidence_text = _timestamp(revision.as_of, "official grid revision as_of")
        if revision.phase is GridRevisionPhase.POST_START or evidence_time > cutoff:
            ignored.append(str(revision.revision_id))
            continue
        eligible.append((evidence_time, revision, evidence_text))

    if not eligible:
        return None, tuple(sorted(set(ignored))), None

    latest_time = max(item[0] for item in eligible)
    latest = [item for item in eligible if item[0] == latest_time]
    signatures = {_revision_signature(item[1]) for item in latest}
    evidence_ids = tuple(sorted({str(item[1].revision_id) for item in latest}))
    ignored.extend(str(item[1].revision_id) for item in eligible if item[0] < latest_time)
    if len(signatures) != 1:
        return (
            None,
            tuple(sorted(set(ignored))),
            OrderResolution(
                status=ResolutionStatus.CONFLICTED,
                source=EvidenceSource.OFFICIAL_GRID_REVISION,
                phase="conflicting_official_grid_revisions",
                as_of=cutoff_text,
                evidence_as_of=latest[0][2],
                evidence_complete=False,
                evidence_ids=evidence_ids,
                ignored_evidence_ids=tuple(sorted(set(ignored))),
                issues=("conflicting_official_grid_revisions_at_same_timestamp",),
            ),
        )
    chosen = sorted(latest, key=lambda item: str(item[1].revision_id))[0][1]
    return chosen, tuple(sorted(set(ignored))), None


def _order_from_official_revision(
    revision: OfficialGridRevision,
    *,
    cutoff_text: str,
    expected_field_size: int,
    ignored_evidence_ids: tuple[str, ...],
) -> OrderResolution:
    _, evidence_text = _timestamp(revision.as_of, "official grid revision as_of")
    driver_ids = [entry.driver_id for entry in revision.entries]
    grid_positions = [
        entry.position
        for entry in revision.entries
        if entry.status is GridEntryStatus.GRID and entry.position is not None
    ]
    issues: list[str] = []
    conflicted = False
    if len(driver_ids) != len(set(driver_ids)):
        issues.append("duplicate_driver_in_official_grid")
        conflicted = True
    if len(grid_positions) != len(set(grid_positions)):
        issues.append("duplicate_position_in_official_grid")
        conflicted = True
    if any(position > int(expected_field_size) for position in grid_positions):
        issues.append("official_grid_position_exceeds_eligible_field")
        conflicted = True
    if len(driver_ids) != int(expected_field_size):
        issues.append(
            f"official_grid_field_size_{len(driver_ids)}_expected_{int(expected_field_size)}"
        )

    rows: list[ResolvedOrderEntry] = []
    for entry in revision.entries:
        row_issues: list[str] = []
        position = entry.position
        row_complete = bool(revision.evidence_complete and entry.evidence_complete)
        if entry.status is GridEntryStatus.GRID and position is None:
            row_issues.append("grid_entry_missing_position")
            row_complete = False
        if entry.status is not GridEntryStatus.GRID and position is not None:
            row_issues.append("non_grid_status_has_synthetic_grid_position")
            position = None
            row_complete = False
        if entry.status in {GridEntryStatus.PENALTY_PENDING, GridEntryStatus.UNRESOLVED}:
            row_complete = False
        if entry.decisions:
            adjustment_evidence = tuple(
                ResolvedGridAdjustment(
                    kind=decision.kind,
                    places=decision.places,
                    reason=decision.reason,
                    status=_adjustment_status(decision.kind),
                    source=EvidenceSource.OFFICIAL_GRID_REVISION,
                    as_of=evidence_text,
                    evidence_complete=bool(row_complete and decision.evidence_complete),
                    evidence_id=decision.evidence_id,
                )
                for decision in entry.decisions
            )
        else:
            adjustment_evidence = tuple(
                ResolvedGridAdjustment(
                    kind=kind,
                    places=None,
                    reason=None,
                    status=_adjustment_status(kind),
                    source=EvidenceSource.OFFICIAL_GRID_REVISION,
                    as_of=evidence_text,
                    evidence_complete=row_complete,
                    evidence_id=str(revision.revision_id),
                )
                for kind in entry.adjustments
            )
        rows.append(
            ResolvedOrderEntry(
                driver_id=entry.driver_id,
                position=position,
                status=entry.status,
                source=EvidenceSource.OFFICIAL_GRID_REVISION,
                as_of=evidence_text,
                evidence_complete=row_complete,
                evidence_ids=(str(revision.revision_id),),
                adjustments=entry.adjustments,
                adjustment_evidence=adjustment_evidence,
                issues=tuple(row_issues),
            )
        )
        issues.extend(f"{entry.driver_id}:{issue}" for issue in row_issues)

    structurally_complete = bool(
        not conflicted
        and revision.evidence_complete
        and len(driver_ids) == int(expected_field_size)
        and all(row.evidence_complete for row in rows)
    )
    final_phase = revision.phase is GridRevisionPhase.FINAL_PRE_RACE
    evidence_complete = bool(structurally_complete and final_phase)
    if structurally_complete and not final_phase:
        issues.append("official_grid_revision_is_provisional_not_final")
    if conflicted:
        status = ResolutionStatus.CONFLICTED
    elif evidence_complete:
        status = ResolutionStatus.RESOLVED
    elif rows:
        status = ResolutionStatus.PARTIAL
    else:
        status = ResolutionStatus.UNRESOLVED
    return OrderResolution(
        status=status,
        source=EvidenceSource.OFFICIAL_GRID_REVISION,
        phase=revision.phase.value,
        as_of=cutoff_text,
        evidence_as_of=evidence_text,
        evidence_complete=evidence_complete,
        entries=_ordered_entries(rows),
        evidence_ids=(str(revision.revision_id),),
        ignored_evidence_ids=ignored_evidence_ids,
        issues=tuple(dict.fromkeys(issues)),
    )


def _scheduled_from_provisional(
    provisional: OrderResolution,
    *,
    cutoff_text: str,
    ignored_evidence_ids: tuple[str, ...],
) -> OrderResolution:
    issues = tuple(dict.fromkeys((*provisional.issues, "official_grid_revision_missing")))
    return OrderResolution(
        status=ResolutionStatus.UNRESOLVED,
        source=provisional.source,
        phase="classification_provisional_only",
        as_of=cutoff_text,
        evidence_as_of=provisional.evidence_as_of,
        evidence_complete=False,
        entries=provisional.entries,
        evidence_ids=provisional.evidence_ids,
        ignored_evidence_ids=tuple(
            sorted(set((*provisional.ignored_evidence_ids, *ignored_evidence_ids)))
        ),
        issues=issues,
    )


def _adjustment_status(kind: GridAdjustmentKind) -> GridEntryStatus:
    return {
        GridAdjustmentKind.GRID_DROP: GridEntryStatus.PENALTY_PENDING,
        GridAdjustmentKind.BACK_OF_GRID: GridEntryStatus.PENALTY_PENDING,
        GridAdjustmentKind.PIT_LANE_START: GridEntryStatus.PIT_LANE,
        GridAdjustmentKind.DISQUALIFICATION: GridEntryStatus.DISQUALIFIED,
        GridAdjustmentKind.WITHDRAWAL: GridEntryStatus.WITHDRAWN,
    }[kind]


def _apply_adjustments(
    base: OrderResolution,
    adjustments: Sequence[GridAdjustment],
    *,
    cutoff: datetime,
    cutoff_text: str,
) -> OrderResolution:
    base_time = (
        _timestamp(base.evidence_as_of, "grid evidence_as_of")[0]
        if base.source is EvidenceSource.OFFICIAL_GRID_REVISION and base.evidence_as_of
        else None
    )
    eligible: list[tuple[datetime, GridAdjustment, str]] = []
    ignored = list(base.ignored_evidence_ids)
    for adjustment in adjustments:
        evidence_time, evidence_text = _timestamp(adjustment.as_of, "grid adjustment as_of")
        if evidence_time > cutoff:
            ignored.append(str(adjustment.evidence_id))
            continue
        if base_time is not None and evidence_time <= base_time:
            # A later complete official publication is authoritative for all
            # earlier individual decisions.
            ignored.append(str(adjustment.evidence_id))
            continue
        eligible.append((evidence_time, adjustment, evidence_text))
    if not eligible:
        return replace(base, ignored_evidence_ids=tuple(sorted(set(ignored))))

    by_driver: dict[str, list[tuple[datetime, GridAdjustment, str]]] = {}
    for item in eligible:
        by_driver.setdefault(item[1].driver_id, []).append(item)

    rows = {entry.driver_id: entry for entry in base.entries}
    issues = list(base.issues)
    evidence_ids = list(base.evidence_ids)
    conflict = False
    open_order_penalty = False
    for driver_id, driver_items in sorted(by_driver.items()):
        latest_time = max(item[0] for item in driver_items)
        latest = [item for item in driver_items if item[0] == latest_time]
        decisions = {(item[1].kind, item[1].places) for item in latest}
        kinds = {kind for kind, _ in decisions}
        ignored.extend(str(item[1].evidence_id) for item in driver_items if item[0] < latest_time)
        latest_ids = tuple(sorted({str(item[1].evidence_id) for item in latest}))
        evidence_ids.extend(latest_ids)
        existing = rows.get(driver_id)
        if len(decisions) != 1:
            conflict = True
            issues.append(f"conflicting_grid_adjustments:{driver_id}")
            status = GridEntryStatus.UNRESOLVED
            adjustment_kinds = tuple(sorted(kinds, key=lambda kind: kind.value))
        else:
            kind, places = next(iter(decisions))
            status = _adjustment_status(kind)
            adjustment_kinds = (kind,)
            if kind in {GridAdjustmentKind.GRID_DROP, GridAdjustmentKind.BACK_OF_GRID}:
                open_order_penalty = True
            decision_label = f"{kind.value}:{places}" if places is not None else kind.value
            issues.append(
                "unresolved_grid_adjustment_without_new_full_revision:"
                f"{driver_id}:{decision_label}"
            )
        evidence_text = latest[0][2]
        complete = all(item[1].evidence_complete for item in latest)
        adjustment_evidence = tuple(
            ResolvedGridAdjustment(
                kind=item[1].kind,
                places=item[1].places,
                reason=item[1].reason,
                status=_adjustment_status(item[1].kind),
                source=EvidenceSource.GRID_ADJUSTMENT,
                as_of=item[2],
                evidence_complete=bool(item[1].evidence_complete),
                evidence_id=str(item[1].evidence_id),
            )
            for item in sorted(latest, key=lambda row: str(row[1].evidence_id))
        )
        rows[driver_id] = ResolvedOrderEntry(
            driver_id=driver_id,
            position=None,
            status=status,
            source=EvidenceSource.GRID_ADJUSTMENT,
            as_of=evidence_text,
            evidence_complete=False,
            evidence_ids=latest_ids,
            adjustments=adjustment_kinds,
            adjustment_evidence=adjustment_evidence,
            issues=(() if complete else ("adjustment_evidence_incomplete",)),
        )
        if existing is None:
            issues.append(f"adjustment_driver_absent_from_base_grid:{driver_id}")

    if conflict:
        status = ResolutionStatus.CONFLICTED
    elif base.status is ResolutionStatus.UNRESOLVED or open_order_penalty:
        status = ResolutionStatus.UNRESOLVED
    else:
        status = ResolutionStatus.PARTIAL
    latest_evidence = max(eligible, key=lambda item: item[0])[2]
    return OrderResolution(
        status=status,
        source=EvidenceSource.GRID_ADJUSTMENT,
        phase=f"{base.phase}_with_unresolved_adjustments",
        as_of=cutoff_text,
        evidence_as_of=latest_evidence,
        evidence_complete=False,
        entries=_ordered_entries(rows.values()),
        evidence_ids=tuple(dict.fromkeys(evidence_ids)),
        ignored_evidence_ids=tuple(sorted(set(ignored))),
        issues=tuple(dict.fromkeys(issues)),
    )


def _resolve_actual_start(
    snapshots: Sequence[ActualStartSnapshot],
    *,
    cutoff: datetime,
    cutoff_text: str,
    expected_field_size: int,
    scheduled: OrderResolution,
    provisional: OrderResolution,
) -> OrderResolution:
    eligible: list[tuple[datetime, ActualStartSnapshot, str]] = []
    ignored: list[str] = []
    for snapshot in snapshots:
        evidence_time, evidence_text = _timestamp(snapshot.as_of, "actual-start as_of")
        if evidence_time > cutoff:
            ignored.append(str(snapshot.evidence_id))
            continue
        eligible.append((evidence_time, snapshot, evidence_text))
    if not eligible:
        return OrderResolution(
            status=ResolutionStatus.UNRESOLVED,
            source=EvidenceSource.NONE,
            phase="actual_start",
            as_of=cutoff_text,
            evidence_as_of=None,
            evidence_complete=False,
            ignored_evidence_ids=tuple(sorted(set(ignored))),
            issues=("actual_start_evidence_missing",),
        )

    latest_time = max(item[0] for item in eligible)
    latest = [item for item in eligible if item[0] == latest_time]
    signatures = {_actual_start_signature(item[1]) for item in latest}
    evidence_ids = tuple(sorted({str(item[1].evidence_id) for item in latest}))
    ignored.extend(str(item[1].evidence_id) for item in eligible if item[0] < latest_time)
    evidence_text = latest[0][2]
    if len(signatures) != 1:
        return OrderResolution(
            status=ResolutionStatus.CONFLICTED,
            source=EvidenceSource.ACTUAL_START_EVIDENCE,
            phase="actual_start",
            as_of=cutoff_text,
            evidence_as_of=evidence_text,
            evidence_complete=False,
            evidence_ids=evidence_ids,
            ignored_evidence_ids=tuple(sorted(set(ignored))),
            issues=("conflicting_actual_start_snapshots_at_same_timestamp",),
        )

    snapshot = sorted(latest, key=lambda item: str(item[1].evidence_id))[0][1]
    driver_ids = [entry.driver_id for entry in snapshot.entries]
    started_orders = [
        entry.start_order
        for entry in snapshot.entries
        if entry.status in {ActualStartStatus.STARTED, ActualStartStatus.STARTED_PIT_LANE}
        and entry.start_order is not None
    ]
    issues: list[str] = []
    conflicted = False
    if len(driver_ids) != len(set(driver_ids)):
        issues.append("duplicate_driver_in_actual_start")
        conflicted = True
    if len(started_orders) != len(set(started_orders)):
        issues.append("duplicate_actual_start_order")
        conflicted = True
    observed_order_complete = set(started_orders) == set(
        range(1, len(started_orders) + 1)
    )
    if started_orders and not observed_order_complete:
        issues.append("actual_start_order_is_not_contiguous")

    expected_ids = {entry.driver_id for entry in scheduled.entries}
    if not expected_ids:
        expected_ids = {entry.driver_id for entry in provisional.entries}
    if expected_ids:
        missing = sorted(expected_ids - set(driver_ids))
        unexpected = sorted(set(driver_ids) - expected_ids)
        if missing:
            issues.append(f"actual_start_missing_drivers:{','.join(missing)}")
        if unexpected:
            issues.append(f"actual_start_unexpected_drivers:{','.join(unexpected)}")
    elif len(driver_ids) != int(expected_field_size):
        issues.append(
            f"actual_start_field_size_{len(driver_ids)}_expected_{int(expected_field_size)}"
        )

    status_map = {
        ActualStartStatus.STARTED: GridEntryStatus.STARTED,
        ActualStartStatus.STARTED_PIT_LANE: GridEntryStatus.STARTED_PIT_LANE,
        ActualStartStatus.DID_NOT_START: GridEntryStatus.DID_NOT_START,
        ActualStartStatus.WITHDRAWN: GridEntryStatus.WITHDRAWN,
        ActualStartStatus.DISQUALIFIED: GridEntryStatus.DISQUALIFIED,
        ActualStartStatus.UNKNOWN: GridEntryStatus.UNRESOLVED,
    }
    rows: list[ResolvedOrderEntry] = []
    for entry in snapshot.entries:
        row_status = status_map[entry.status]
        row_issues: list[str] = []
        start_order = entry.start_order
        if entry.status in {ActualStartStatus.STARTED, ActualStartStatus.STARTED_PIT_LANE}:
            if start_order is None:
                row_issues.append("starter_missing_observed_start_order")
        elif start_order is not None:
            row_issues.append("nonstarter_has_start_order")
            start_order = None
        if entry.status is ActualStartStatus.UNKNOWN:
            row_issues.append("actual_start_status_unknown")
        row_complete = bool(
            snapshot.evidence_complete
            and entry.evidence_complete
            and not row_issues
        )
        rows.append(
            ResolvedOrderEntry(
                driver_id=entry.driver_id,
                position=start_order,
                status=row_status,
                source=EvidenceSource.ACTUAL_START_EVIDENCE,
                as_of=evidence_text,
                evidence_complete=row_complete,
                evidence_ids=(str(snapshot.evidence_id),),
                issues=tuple(row_issues),
            )
        )
        issues.extend(f"{entry.driver_id}:{issue}" for issue in row_issues)

    roster_complete = (
        set(driver_ids) == expected_ids
        if expected_ids
        else len(driver_ids) == int(expected_field_size)
    )
    complete = bool(
        not conflicted
        and snapshot.evidence_complete
        and roster_complete
        and observed_order_complete
        and all(row.evidence_complete for row in rows)
    )
    status = (
        ResolutionStatus.CONFLICTED
        if conflicted
        else ResolutionStatus.RESOLVED
        if complete
        else ResolutionStatus.PARTIAL
    )
    return OrderResolution(
        status=status,
        source=EvidenceSource.ACTUAL_START_EVIDENCE,
        phase="actual_start",
        as_of=cutoff_text,
        evidence_as_of=evidence_text,
        evidence_complete=complete,
        entries=_ordered_entries(rows),
        evidence_ids=evidence_ids,
        ignored_evidence_ids=tuple(sorted(set(ignored))),
        issues=tuple(dict.fromkeys(issues)),
    )


def resolve_grand_prix_start(
    contract: WeekendContract,
    *,
    as_of: Timestamp,
    classifications: Iterable[ClassificationSnapshot] = (),
    official_grid_revisions: Iterable[OfficialGridRevision] = (),
    adjustments: Iterable[GridAdjustment] = (),
    actual_start_snapshots: Iterable[ActualStartSnapshot] = (),
) -> GrandPrixStartResolution:
    """Resolve classification, provisional grid, scheduled grid, and starters.

    Evidence published after ``as_of`` is ignored.  Same-timestamp conflicting
    evidence fails closed.  An individual penalty decision never reorders the
    field; only a subsequent complete official grid revision can do that.
    """

    cutoff, cutoff_text = _timestamp(as_of, "resolution as_of")
    classification_rows = tuple(classifications)
    revisions = tuple(official_grid_revisions)
    adjustment_rows = tuple(adjustments)
    start_rows = tuple(actual_start_snapshots)

    qualifying = _classification_resolution(
        session=Session.QUALIFYING,
        snapshots=classification_rows,
        cutoff=cutoff,
        cutoff_text=cutoff_text,
        expected_field_size=contract.eligible_cars,
    )
    source_session = contract.grid_source(GridTarget.RACE)
    if source_session is Session.QUALIFYING:
        source_classification = qualifying
    else:
        source_classification = _classification_resolution(
            session=source_session,
            snapshots=classification_rows,
            cutoff=cutoff,
            cutoff_text=cutoff_text,
            expected_field_size=contract.eligible_cars,
        )
    provisional = _provisional_grid_from_classification(
        source_classification,
        cutoff_text=cutoff_text,
        expected_field_size=contract.eligible_cars,
    )

    revision, ignored_revisions, revision_conflict = _select_official_revision(
        revisions,
        cutoff=cutoff,
        cutoff_text=cutoff_text,
    )
    if revision_conflict is not None:
        scheduled = revision_conflict
    elif revision is not None:
        scheduled = _order_from_official_revision(
            revision,
            cutoff_text=cutoff_text,
            expected_field_size=contract.eligible_cars,
            ignored_evidence_ids=ignored_revisions,
        )
    else:
        scheduled = _scheduled_from_provisional(
            provisional,
            cutoff_text=cutoff_text,
            ignored_evidence_ids=ignored_revisions,
        )
    if scheduled.status is not ResolutionStatus.CONFLICTED:
        scheduled = _apply_adjustments(
            scheduled,
            adjustment_rows,
            cutoff=cutoff,
            cutoff_text=cutoff_text,
        )

    actual_start = _resolve_actual_start(
        start_rows,
        cutoff=cutoff,
        cutoff_text=cutoff_text,
        expected_field_size=contract.eligible_cars,
        scheduled=scheduled,
        provisional=provisional,
    )

    parts = (qualifying, source_classification, provisional, scheduled, actual_start)
    if any(part.status is ResolutionStatus.CONFLICTED for part in parts):
        overall_status = ResolutionStatus.CONFLICTED
    elif all(part.status is ResolutionStatus.RESOLVED for part in parts):
        overall_status = ResolutionStatus.RESOLVED
    elif any(getattr(part, "entries", ()) for part in parts):
        overall_status = ResolutionStatus.PARTIAL
    else:
        overall_status = ResolutionStatus.UNRESOLVED
    evidence_complete = bool(all(part.evidence_complete for part in parts))
    evidence_ids = tuple(
        dict.fromkeys(
            evidence_id
            for part in parts
            for evidence_id in part.evidence_ids
        )
    )
    issues = tuple(
        dict.fromkeys(
            issue
            for part in parts
            for issue in part.issues
        )
    )
    return GrandPrixStartResolution(
        status=overall_status,
        source=EvidenceSource.COMPOSITE_RESOLVER,
        as_of=cutoff_text,
        evidence_complete=evidence_complete,
        qualifying_classification=qualifying,
        provisional_source_classification=source_classification,
        provisional_grid=provisional,
        scheduled_grid=scheduled,
        actual_start=actual_start,
        evidence_ids=evidence_ids,
        issues=issues,
    )


def build_race_grid_snapshot(
    resolution: GrandPrixStartResolution,
    *,
    horizon: RacePredictionHorizon = RacePredictionHorizon.POST_GRID_PRE_RACE,
) -> RaceGridSnapshot:
    """Freeze the exact grid evidence supplied to a pre-Race model.

    The final-grid horizon fails closed unless the resolver selected a complete
    official final pre-Race revision.  The post-Qualifying horizon intentionally
    uses ``provisional_grid`` and stays labelled as a proxy even when complete.
    """

    if not isinstance(resolution, GrandPrixStartResolution):
        raise TypeError("resolution must be a GrandPrixStartResolution")
    if not isinstance(horizon, RacePredictionHorizon):
        raise TypeError("horizon must be a RacePredictionHorizon")

    order = (
        resolution.scheduled_grid
        if horizon is RacePredictionHorizon.POST_GRID_PRE_RACE
        else resolution.provisional_grid
    )
    prediction_time, prediction_text = _timestamp(resolution.as_of, "resolution as_of")
    publication_text = order.evidence_as_of
    flags = list(order.issues)
    if publication_text is not None:
        publication_time, publication_text = _timestamp(
            publication_text,
            "grid evidence_as_of",
        )
        if publication_time > prediction_time:
            flags.append("grid_publication_after_prediction_cutoff")

    allowed_starter_statuses = {GridEntryStatus.GRID, GridEntryStatus.PIT_LANE}
    declared_nonstarters = {
        GridEntryStatus.WITHDRAWN,
        GridEntryStatus.DID_NOT_START,
        GridEntryStatus.DISQUALIFIED,
    }
    snapshot_entries: list[RaceGridSnapshotEntry] = []
    for row in order.entries:
        starter_eligible = row.status in allowed_starter_statuses
        status_resolved = row.status in allowed_starter_statuses | declared_nonstarters
        if not status_resolved:
            flags.append(f"unresolved_starter_eligibility:{row.driver_id}")
        snapshot_entries.append(
            RaceGridSnapshotEntry(
                driver_id=row.driver_id,
                grid_position=row.position,
                status=row.status,
                starter_eligible=starter_eligible,
                pit_lane_start=row.status is GridEntryStatus.PIT_LANE,
                penalty_evidence=row.adjustment_evidence,
                evidence_ids=row.evidence_ids,
                evidence_complete=bool(row.evidence_complete and status_resolved),
            )
        )

    duplicate_drivers = len({row.driver_id for row in snapshot_entries}) != len(snapshot_entries)
    if duplicate_drivers:
        flags.append("duplicate_driver_in_snapshot")

    if horizon is RacePredictionHorizon.POST_GRID_PRE_RACE:
        if order.source is not EvidenceSource.OFFICIAL_GRID_REVISION:
            flags.append("official_final_grid_revision_missing")
        if order.phase != GridRevisionPhase.FINAL_PRE_RACE.value:
            flags.append("grid_revision_not_final_pre_race")
        final_contract = bool(
            order.status is ResolutionStatus.RESOLVED
            and order.source is EvidenceSource.OFFICIAL_GRID_REVISION
            and order.phase == GridRevisionPhase.FINAL_PRE_RACE.value
        )
    else:
        # The proxy is useful but remains a distinct product.  It is never
        # upgraded to final-grid evidence by this constructor.
        final_contract = order.status is ResolutionStatus.RESOLVED

    complete = bool(
        final_contract
        and order.evidence_complete
        and publication_text is not None
        and not duplicate_drivers
        and snapshot_entries
        and all(row.evidence_complete for row in snapshot_entries)
        and "grid_publication_after_prediction_cutoff" not in flags
    )
    return RaceGridSnapshot(
        horizon=horizon,
        prediction_as_of=prediction_text,
        publication_as_of=publication_text,
        resolution_status=order.status,
        source=order.source,
        entries=tuple(snapshot_entries),
        revision_ids=order.evidence_ids,
        available=complete,
        evidence_complete=complete,
        unresolved_state_flags=tuple(dict.fromkeys(flags)),
    )


__all__ = [
    "ActualStartEntry",
    "ActualStartSnapshot",
    "ActualStartStatus",
    "ClassificationEntry",
    "ClassificationResolution",
    "ClassificationSnapshot",
    "ClassificationStatus",
    "EvidenceSource",
    "GrandPrixStartResolution",
    "GridAdjustment",
    "GridAdjustmentKind",
    "GridEntryStatus",
    "GridRevisionPhase",
    "GRID_CAPTURE_SCHEMA_VERSION",
    "OfficialGridEntry",
    "OfficialGridDecision",
    "OfficialGridRevision",
    "OrderResolution",
    "RaceGridCapture",
    "RaceGridSnapshot",
    "RaceGridSnapshotEntry",
    "RacePredictionHorizon",
    "ResolutionStatus",
    "ResolvedClassificationEntry",
    "ResolvedGridAdjustment",
    "ResolvedOrderEntry",
    "build_race_grid_snapshot",
    "build_race_grid_capture",
    "load_race_grid_capture",
    "persist_race_grid_capture",
    "race_grid_capture_from_payload",
    "resolve_grand_prix_start",
]
