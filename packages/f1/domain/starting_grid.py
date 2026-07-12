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
from typing import Union

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


class ActualStartStatus(str, Enum):
    """Explicit observed start status, separate from the scheduled grid."""

    STARTED = "started"
    STARTED_PIT_LANE = "started_pit_lane"
    DID_NOT_START = "did_not_start"
    WITHDRAWN = "withdrawn"
    DISQUALIFIED = "disqualified"
    UNKNOWN = "unknown"


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
                adjustment_evidence=tuple(
                    ResolvedGridAdjustment(
                        kind=kind,
                        places=None,
                        status=_adjustment_status(kind),
                        source=EvidenceSource.OFFICIAL_GRID_REVISION,
                        as_of=evidence_text,
                        evidence_complete=row_complete,
                        evidence_id=str(revision.revision_id),
                    )
                    for kind in entry.adjustments
                ),
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
    "OfficialGridEntry",
    "OfficialGridRevision",
    "OrderResolution",
    "ResolutionStatus",
    "ResolvedClassificationEntry",
    "ResolvedGridAdjustment",
    "ResolvedOrderEntry",
    "resolve_grand_prix_start",
]
