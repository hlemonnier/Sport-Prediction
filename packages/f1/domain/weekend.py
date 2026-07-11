"""Season-aware Formula 1 weekend and point-in-time information contracts.

The contract deliberately models sessions, not wall-clock timestamps.  A provider
must still prove that a session was complete at its prediction timestamp before it
marks the corresponding :class:`SessionCutoff` as available.

Regulatory anchors:

* 2021-2022: Friday Qualifying set the Sprint grid and the Sprint set the Race grid.
* 2023: Qualifying set the Race grid before a standalone Sprint Shootout/Sprint.
* 2024 onward (including the current 2026 Alternative Format): Sprint Qualifying
  and Sprint precede Race Qualifying, with setup freedom reopening between the
  Sprint and Race Qualifying.
* FIA 2026 Sporting Regulations B2/B3: a 22-car field eliminates six cars after
  Q1/Q2 and SQ1/SQ2, leaving ten cars in Q3/SQ3.

Final grids are not session classifications: penalties, disqualifications,
withdrawals and pit-lane starts must be applied by a later grid-resolution layer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import re
from typing import Any


class WeekendFormat(str, Enum):
    """Regulation-era weekend formats supported by the contract."""

    STANDARD = "standard"
    SPRINT_2021_2022 = "sprint_2021_2022"
    SPRINT_2023 = "sprint_2023"
    SPRINT_2024_PLUS = "sprint_2024_plus"


class Session(str, Enum):
    """Canonical whole-weekend sessions (qualifying periods are not sessions)."""

    FP1 = "FP1"
    FP2 = "FP2"
    FP3 = "FP3"
    QUALIFYING = "Q"
    SPRINT_QUALIFYING = "SQ"
    SPRINT = "Sprint"
    RACE = "Race"


class PredictionTarget(str, Enum):
    """Targets whose usable completed-session evidence can be requested."""

    GRAND_PRIX_QUALIFYING = "grand_prix_qualifying"
    SPRINT_QUALIFYING = "sprint_qualifying"
    SPRINT_STARTING_GRID = "sprint_starting_grid"
    SPRINT = "sprint"
    GRAND_PRIX_STARTING_GRID = "grand_prix_starting_grid"
    RACE = "race"


class GridTarget(str, Enum):
    """Classified session whose result supplies a provisional starting order."""

    SPRINT = "sprint"
    RACE = "race"


@dataclass(frozen=True)
class SessionEdge:
    """A directed chronological edge in the canonical session DAG."""

    predecessor: Session
    successor: Session


@dataclass(frozen=True)
class GridRule:
    """Source classification for a grid, before later grid decisions are applied."""

    target: GridTarget
    source_session: Session


@dataclass(frozen=True)
class ParcFermeWindow:
    """Setup-lock window from first pit exit in ``start`` to the start of ``end``."""

    start: Session
    end: Session


@dataclass(frozen=True)
class QualifyingEliminationRule:
    """Three-period elimination shape for Q and, where present, SQ."""

    eligible_cars: int
    eliminated_after_period_1: int
    eliminated_after_period_2: int
    period_3_cars: int = 10

    @property
    def period_2_cars(self) -> int:
        return self.eligible_cars - self.eliminated_after_period_1


@dataclass(frozen=True)
class SessionCutoff:
    """Session-level information cutoff.

    ``before_weekend`` exposes no same-event session. ``before(session)`` exposes
    only earlier completed sessions; ``after(session)`` also exposes that session.
    """

    session: Session | None = None
    include_session: bool = False

    def __post_init__(self) -> None:
        if self.session is None and self.include_session:
            raise ValueError("A before-weekend cutoff cannot include a session")

    @classmethod
    def before_weekend(cls) -> SessionCutoff:
        return cls()

    @classmethod
    def before(cls, session: Session) -> SessionCutoff:
        return cls(session=session, include_session=False)

    @classmethod
    def after(cls, session: Session) -> SessionCutoff:
        return cls(session=session, include_session=True)

    @property
    def label(self) -> str:
        if self.session is None:
            return "before_weekend"
        prefix = "after" if self.include_session else "before"
        return f"{prefix}_{self.session.value}"


@dataclass(frozen=True)
class AvailabilityEntry:
    """One explicit target/cutoff row in a feature-availability matrix."""

    target: PredictionTarget
    cutoff: SessionCutoff
    eligible_sessions: tuple[Session, ...]


@dataclass(frozen=True)
class WeekendContract:
    """Immutable season/format contract used by data and prediction layers."""

    season: int
    format: WeekendFormat
    session_order: tuple[Session, ...]
    grid_rules: tuple[GridRule, ...]
    parc_ferme_windows: tuple[ParcFermeWindow, ...]
    setup_reopens_after: tuple[Session, ...]
    eligible_cars: int
    race_qualifying_rule: QualifyingEliminationRule
    sprint_qualifying_rule: QualifyingEliminationRule | None

    def __post_init__(self) -> None:
        if len(self.session_order) != len(set(self.session_order)):
            raise ValueError("Canonical session order contains duplicates")
        if not self.session_order or self.session_order[0] is not Session.FP1:
            raise ValueError("A weekend contract must start with FP1")
        if self.session_order[-1] is not Session.RACE:
            raise ValueError("A weekend contract must end with the Race")
        if Session.QUALIFYING not in self.session_order:
            raise ValueError("A weekend contract must contain Race Qualifying")
        if self.race_qualifying_rule.eligible_cars != self.eligible_cars:
            raise ValueError("Race Qualifying rule does not match the weekend field")
        if (
            self.sprint_qualifying_rule is not None
            and self.sprint_qualifying_rule.eligible_cars != self.eligible_cars
        ):
            raise ValueError("Sprint Qualifying rule does not match the weekend field")

        order = {session: index for index, session in enumerate(self.session_order)}
        for rule in self.grid_rules:
            if rule.source_session not in order:
                raise ValueError(f"Grid source {rule.source_session.value} is absent")
        for window in self.parc_ferme_windows:
            if window.start not in order or window.end not in order:
                raise ValueError("Parc ferme window references an absent session")
            if order[window.start] >= order[window.end]:
                raise ValueError("Parc ferme window must point forward in time")
        for session in self.setup_reopens_after:
            if session not in order:
                raise ValueError("Setup-reopen marker references an absent session")

    @property
    def is_sprint(self) -> bool:
        return Session.SPRINT in self.session_order

    @property
    def session_dag(self) -> tuple[SessionEdge, ...]:
        """Immediate edges; transitive predecessors are available via ``ancestors``."""

        return tuple(
            SessionEdge(predecessor=left, successor=right)
            for left, right in zip(self.session_order, self.session_order[1:])
        )

    def ancestors(self, session: Session) -> tuple[Session, ...]:
        """All chronologically earlier sessions in canonical order."""

        try:
            index = self.session_order.index(session)
        except ValueError as exc:
            raise ValueError(f"{session.value} is not part of this weekend format") from exc
        return self.session_order[:index]

    def completed_sessions(self, cutoff: SessionCutoff) -> tuple[Session, ...]:
        """Same-event sessions complete at a session-level cutoff."""

        if cutoff.session is None:
            return ()
        try:
            index = self.session_order.index(cutoff.session)
        except ValueError as exc:
            raise ValueError(
                f"Cutoff session {cutoff.session.value} is absent from {self.format.value}"
            ) from exc
        stop = index + int(cutoff.include_session)
        return self.session_order[:stop]

    def supported_prediction_targets(self) -> tuple[PredictionTarget, ...]:
        targets = [PredictionTarget.GRAND_PRIX_QUALIFYING]
        if Session.SPRINT_QUALIFYING in self.session_order:
            targets.append(PredictionTarget.SPRINT_QUALIFYING)
        if Session.SPRINT in self.session_order:
            targets.extend(
                [PredictionTarget.SPRINT_STARTING_GRID, PredictionTarget.SPRINT]
            )
        targets.extend(
            [PredictionTarget.GRAND_PRIX_STARTING_GRID, PredictionTarget.RACE]
        )
        return tuple(targets)

    def eligible_sessions(
        self,
        target: PredictionTarget,
        cutoff: SessionCutoff,
    ) -> tuple[Session, ...]:
        """Completed, pre-target sessions safe for a target at ``cutoff``.

        Target and post-target sessions are always removed, even if the caller
        supplies a later cutoff.  This fail-closed behavior prevents label and
        future-session leakage at the contract boundary.
        """

        boundary = self._target_boundary(target)
        boundary_index = self.session_order.index(boundary)
        completed = self.completed_sessions(cutoff)
        return tuple(
            session
            for session in completed
            if self.session_order.index(session) < boundary_index
        )

    def default_cutoff(self, target: PredictionTarget) -> SessionCutoff:
        """Return the latest leakage-safe whole-session cutoff for a target."""

        return SessionCutoff.before(self._target_boundary(target))

    def availability_matrix(
        self,
        targets: Iterable[PredictionTarget] | None = None,
    ) -> tuple[AvailabilityEntry, ...]:
        """Return explicit pre-weekend and post-session availability rows."""

        selected = (
            tuple(targets)
            if targets is not None
            else self.supported_prediction_targets()
        )
        cutoffs = (SessionCutoff.before_weekend(),) + tuple(
            SessionCutoff.after(session) for session in self.session_order
        )
        return tuple(
            AvailabilityEntry(
                target=target,
                cutoff=cutoff,
                eligible_sessions=self.eligible_sessions(target, cutoff),
            )
            for target in selected
            for cutoff in cutoffs
        )

    def grid_source(self, target: GridTarget) -> Session:
        """Return the source classification, not a penalty-adjusted final grid."""

        for rule in self.grid_rules:
            if rule.target is target:
                return rule.source_session
        raise ValueError(f"This weekend has no {target.value} grid")

    def _target_boundary(self, target: PredictionTarget) -> Session:
        if target is PredictionTarget.GRAND_PRIX_QUALIFYING:
            return Session.QUALIFYING
        if target is PredictionTarget.SPRINT_QUALIFYING:
            boundary = Session.SPRINT_QUALIFYING
        elif target in {PredictionTarget.SPRINT_STARTING_GRID, PredictionTarget.SPRINT}:
            boundary = Session.SPRINT
        elif target in {
            PredictionTarget.GRAND_PRIX_STARTING_GRID,
            PredictionTarget.RACE,
        }:
            return Session.RACE
        else:  # pragma: no cover - defensive against a future enum member
            raise ValueError(f"Unsupported prediction target: {target}")

        if boundary not in self.session_order:
            raise ValueError(f"{target.value} is absent from {self.format.value}")
        return boundary


_SESSION_ORDERS: dict[WeekendFormat, tuple[Session, ...]] = {
    WeekendFormat.STANDARD: (
        Session.FP1,
        Session.FP2,
        Session.FP3,
        Session.QUALIFYING,
        Session.RACE,
    ),
    WeekendFormat.SPRINT_2021_2022: (
        Session.FP1,
        Session.QUALIFYING,
        Session.FP2,
        Session.SPRINT,
        Session.RACE,
    ),
    WeekendFormat.SPRINT_2023: (
        Session.FP1,
        Session.QUALIFYING,
        Session.SPRINT_QUALIFYING,
        Session.SPRINT,
        Session.RACE,
    ),
    WeekendFormat.SPRINT_2024_PLUS: (
        Session.FP1,
        Session.SPRINT_QUALIFYING,
        Session.SPRINT,
        Session.QUALIFYING,
        Session.RACE,
    ),
}


def default_field_size_for_season(season: int) -> int:
    """Known default championship field size for the project's modern data era.

    An explicit override is required outside 2017-2026 so a future or older field
    is never silently invented.
    """

    if 2017 <= season <= 2025:
        return 20
    if season == 2026:
        return 22
    raise ValueError(
        f"No frozen default field size for {season}; pass eligible_cars explicitly"
    )


def qualifying_elimination_rule(eligible_cars: int) -> QualifyingEliminationRule:
    """Build the FIA three-period rule while keeping ten cars for period three."""

    if eligible_cars < 12:
        raise ValueError("At least 12 eligible cars are required for three periods")
    if (eligible_cars - 10) % 2:
        raise ValueError(
            "The FIA equal Q1/Q2 elimination rule requires an even eligible field"
        )
    eliminated = (eligible_cars - 10) // 2
    return QualifyingEliminationRule(
        eligible_cars=eligible_cars,
        eliminated_after_period_1=eliminated,
        eliminated_after_period_2=eliminated,
    )


def build_weekend_contract(
    season: int,
    weekend_format: WeekendFormat = WeekendFormat.STANDARD,
    *,
    eligible_cars: int | None = None,
) -> WeekendContract:
    """Build a validated contract for an explicitly selected regulation era."""

    _validate_format_season(season, weekend_format)
    field_size = (
        default_field_size_for_season(season)
        if eligible_cars is None
        else eligible_cars
    )
    qualifying_rule = qualifying_elimination_rule(field_size)

    if weekend_format is WeekendFormat.STANDARD:
        grid_rules = (GridRule(GridTarget.RACE, Session.QUALIFYING),)
        parc_ferme = (ParcFermeWindow(Session.QUALIFYING, Session.RACE),)
        setup_reopens_after: tuple[Session, ...] = ()
        sprint_rule = None
    elif weekend_format is WeekendFormat.SPRINT_2021_2022:
        grid_rules = (
            GridRule(GridTarget.SPRINT, Session.QUALIFYING),
            GridRule(GridTarget.RACE, Session.SPRINT),
        )
        parc_ferme = (ParcFermeWindow(Session.QUALIFYING, Session.RACE),)
        setup_reopens_after = ()
        sprint_rule = None
    elif weekend_format is WeekendFormat.SPRINT_2023:
        grid_rules = (
            GridRule(GridTarget.SPRINT, Session.SPRINT_QUALIFYING),
            GridRule(GridTarget.RACE, Session.QUALIFYING),
        )
        parc_ferme = (ParcFermeWindow(Session.QUALIFYING, Session.RACE),)
        setup_reopens_after = ()
        sprint_rule = qualifying_rule
    else:
        grid_rules = (
            GridRule(GridTarget.SPRINT, Session.SPRINT_QUALIFYING),
            GridRule(GridTarget.RACE, Session.QUALIFYING),
        )
        parc_ferme = (
            ParcFermeWindow(Session.SPRINT_QUALIFYING, Session.SPRINT),
            ParcFermeWindow(Session.QUALIFYING, Session.RACE),
        )
        setup_reopens_after = (Session.SPRINT,)
        sprint_rule = qualifying_rule

    return WeekendContract(
        season=season,
        format=weekend_format,
        session_order=_SESSION_ORDERS[weekend_format],
        grid_rules=grid_rules,
        parc_ferme_windows=parc_ferme,
        setup_reopens_after=setup_reopens_after,
        eligible_cars=field_size,
        race_qualifying_rule=qualifying_rule,
        sprint_qualifying_rule=sprint_rule,
    )


def canonicalize_session_sequence(
    season: int,
    metadata: str | Session | Mapping[str, Any] | Iterable[str | Session | Mapping[str, Any]],
) -> tuple[Session, ...]:
    """Normalize provider-style session metadata without guessing unknown labels."""

    raw_sessions = _extract_raw_sessions(metadata)
    canonical = tuple(_canonical_session(season, raw) for raw in raw_sessions)
    if len(canonical) != len(set(canonical)):
        values = ", ".join(session.value for session in canonical)
        raise ValueError(f"Session metadata contains duplicate canonical sessions: {values}")
    return canonical


def infer_weekend_format(
    season: int,
    metadata: str | Session | Mapping[str, Any] | Iterable[str | Session | Mapping[str, Any]],
) -> WeekendFormat:
    """Infer the format from season plus a full or partial ordered session sequence."""

    sessions = canonicalize_session_sequence(season, metadata)
    has_sprint_signal = any(
        session in {Session.SPRINT_QUALIFYING, Session.SPRINT}
        for session in sessions
    )
    if not has_sprint_signal:
        inferred = WeekendFormat.STANDARD
    elif season in {2021, 2022}:
        inferred = WeekendFormat.SPRINT_2021_2022
    elif season == 2023:
        inferred = WeekendFormat.SPRINT_2023
    elif season >= 2024:
        inferred = WeekendFormat.SPRINT_2024_PLUS
    else:
        raise ValueError(f"Sprint metadata is incompatible with the {season} season")

    expected = _SESSION_ORDERS[inferred]
    if not _is_subsequence(sessions, expected):
        observed = ", ".join(session.value for session in sessions)
        canonical = ", ".join(session.value for session in expected)
        raise ValueError(
            f"Observed {season} session order [{observed}] conflicts with "
            f"{inferred.value} [{canonical}]"
        )
    return inferred


def infer_weekend_contract(
    season: int,
    metadata: str | Session | Mapping[str, Any] | Iterable[str | Session | Mapping[str, Any]],
    *,
    eligible_cars: int | None = None,
) -> WeekendContract:
    """Infer a format and return its validated weekend contract."""

    weekend_format = infer_weekend_format(season, metadata)
    return build_weekend_contract(
        season,
        weekend_format,
        eligible_cars=eligible_cars,
    )


def parse_session_cutoff(
    contract: WeekendContract,
    value: str | SessionCutoff | None,
    *,
    target: PredictionTarget,
) -> SessionCutoff:
    """Parse CLI/config cutoff aliases against an actual weekend contract.

    ``auto`` means the latest pre-target whole-session boundary, not "whatever
    files happen to exist".  A named session absent from the weekend fails
    closed instead of silently selecting a different information set.
    """

    if isinstance(value, SessionCutoff):
        cutoff = value
    else:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "auto").strip().lower()).strip("_")
        if normalized in {"", "auto", "default", "before_target", "post_all_pre_target"}:
            cutoff = contract.default_cutoff(target)
        elif normalized in {"pre_weekend", "before_weekend", "pre_fp", "pre_fp1"}:
            cutoff = SessionCutoff.before_weekend()
        else:
            aliases: dict[str, tuple[Session, bool]] = {
                "post_fp1": (Session.FP1, True),
                "after_fp1": (Session.FP1, True),
                "post_fp2": (Session.FP2, True),
                "after_fp2": (Session.FP2, True),
                "post_fp3": (Session.FP3, True),
                "post_fp3_pre_q": (Session.FP3, True),
                "post_fp3_pre_qualifying": (Session.FP3, True),
                "after_fp3": (Session.FP3, True),
                "post_sq": (Session.SPRINT_QUALIFYING, True),
                "post_sprint_qualifying": (Session.SPRINT_QUALIFYING, True),
                "after_sq": (Session.SPRINT_QUALIFYING, True),
                "post_sprint": (Session.SPRINT, True),
                "post_sprint_pre_q": (Session.SPRINT, True),
                "post_sprint_pre_qualifying": (Session.SPRINT, True),
                "after_sprint": (Session.SPRINT, True),
                "post_qualifying": (Session.QUALIFYING, True),
                "post_q": (Session.QUALIFYING, True),
                "after_q": (Session.QUALIFYING, True),
                "pre_qualifying": (Session.QUALIFYING, False),
                "before_qualifying": (Session.QUALIFYING, False),
                "post_all_pre_qualifying": (Session.QUALIFYING, False),
                "pre_race": (Session.RACE, False),
                "before_race": (Session.RACE, False),
                "post_all_pre_race": (Session.RACE, False),
            }
            if normalized not in aliases:
                raise ValueError(f"Unknown F1 session cutoff: {value!r}")
            session, include = aliases[normalized]
            cutoff = SessionCutoff.after(session) if include else SessionCutoff.before(session)

    # Validation is intentionally delegated to the contract so an FP2 cutoff
    # on a current Sprint weekend cannot degrade silently to FP1.
    contract.completed_sessions(cutoff)
    contract.eligible_sessions(target, cutoff)
    return cutoff


def _validate_format_season(season: int, weekend_format: WeekendFormat) -> None:
    if season < 1950:
        raise ValueError(f"Invalid Formula 1 season: {season}")
    if (
        weekend_format is WeekendFormat.SPRINT_2021_2022
        and season not in {2021, 2022}
    ):
        raise ValueError("The original Sprint format is valid only for 2021-2022")
    if weekend_format is WeekendFormat.SPRINT_2023 and season != 2023:
        raise ValueError("The standalone 2023 Sprint format is valid only for 2023")
    if weekend_format is WeekendFormat.SPRINT_2024_PLUS and season < 2024:
        raise ValueError("The current Alternative Format starts in 2024")


def _extract_raw_sessions(
    metadata: str | Session | Mapping[str, Any] | Iterable[str | Session | Mapping[str, Any]],
) -> tuple[str | Session, ...]:
    if isinstance(metadata, (str, Session)):
        return (metadata,)
    if isinstance(metadata, Mapping):
        indexed: list[tuple[int, Any]] = []
        for key, value in metadata.items():
            match = re.fullmatch(r"session\s*(\d+)", str(key), flags=re.IGNORECASE)
            if match and value is not None and value != "":
                indexed.append((int(match.group(1)), value))
        if indexed:
            return tuple(value for _, value in sorted(indexed))
        for key in ("sessions", "Sessions"):
            nested = metadata.get(key)
            if nested is not None:
                return _extract_raw_sessions(nested)
        return (_session_name_from_mapping(metadata),)

    extracted: list[str | Session] = []
    for entry in metadata:
        if isinstance(entry, Mapping):
            extracted.append(_session_name_from_mapping(entry))
        elif isinstance(entry, (str, Session)):
            if str(entry).strip():
                extracted.append(entry)
        else:
            raise TypeError(f"Unsupported session metadata entry: {entry!r}")
    return tuple(extracted)


def _session_name_from_mapping(metadata: Mapping[str, Any]) -> str | Session:
    for key in (
        "session_name",
        "SessionName",
        "name",
        "Name",
        "session",
        "Session",
        "session_type",
        "SessionType",
        "type",
    ):
        value = metadata.get(key)
        if isinstance(value, Session):
            return value
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError(f"Session metadata has no recognized name field: {metadata!r}")


def _canonical_session(season: int, raw: str | Session) -> Session:
    if isinstance(raw, Session):
        return raw
    normalized = re.sub(r"[^a-z0-9]+", " ", raw.strip().lower()).strip()
    compact = normalized.replace(" ", "")

    aliases = {
        "fp1": Session.FP1,
        "p1": Session.FP1,
        "practice1": Session.FP1,
        "freepractice1": Session.FP1,
        "firstpractice": Session.FP1,
        "fp2": Session.FP2,
        "p2": Session.FP2,
        "practice2": Session.FP2,
        "freepractice2": Session.FP2,
        "secondpractice": Session.FP2,
        "fp3": Session.FP3,
        "p3": Session.FP3,
        "practice3": Session.FP3,
        "freepractice3": Session.FP3,
        "thirdpractice": Session.FP3,
        "q": Session.QUALIFYING,
        "qualifying": Session.QUALIFYING,
        "racequalifying": Session.QUALIFYING,
        "qualifyingpractice": Session.QUALIFYING,
        "sq": Session.SPRINT_QUALIFYING,
        "sprintshootout": Session.SPRINT_QUALIFYING,
        "shootout": Session.SPRINT_QUALIFYING,
        "sp": Session.SPRINT,
        "sprint": Session.SPRINT,
        "sprintrace": Session.SPRINT,
        "race": Session.RACE,
        "r": Session.RACE,
        "grandprix": Session.RACE,
    }
    if compact == "sprintqualifying":
        # In 2021-2022 this label named the 100 km race; from 2023 it names
        # (or succeeded "Sprint Shootout" as) the Sprint grid-setting session.
        return Session.SPRINT if season <= 2022 else Session.SPRINT_QUALIFYING
    try:
        return aliases[compact]
    except KeyError as exc:
        raise ValueError(f"Unknown F1 session label: {raw!r}") from exc


def _is_subsequence(observed: Sequence[Session], expected: Sequence[Session]) -> bool:
    iterator = iter(expected)
    return all(any(candidate is session for candidate in iterator) for session in observed)


__all__ = [
    "AvailabilityEntry",
    "GridRule",
    "GridTarget",
    "ParcFermeWindow",
    "PredictionTarget",
    "QualifyingEliminationRule",
    "Session",
    "SessionCutoff",
    "SessionEdge",
    "WeekendContract",
    "WeekendFormat",
    "build_weekend_contract",
    "canonicalize_session_sequence",
    "default_field_size_for_season",
    "infer_weekend_contract",
    "infer_weekend_format",
    "qualifying_elimination_rule",
]
