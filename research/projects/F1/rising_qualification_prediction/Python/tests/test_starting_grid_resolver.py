from __future__ import annotations

from collections.abc import Iterable

import pytest

from packages.f1.domain.starting_grid import (
    ActualStartEntry,
    ActualStartSnapshot,
    ActualStartStatus,
    ClassificationEntry,
    ClassificationSnapshot,
    EvidenceSource,
    GridAdjustment,
    GridAdjustmentKind,
    GridEntryStatus,
    GridRevisionPhase,
    OfficialGridEntry,
    OfficialGridRevision,
    ResolutionStatus,
    resolve_grand_prix_start,
)
from packages.f1.domain.weekend import (
    Session,
    WeekendFormat,
    build_weekend_contract,
)


DRIVERS = tuple(f"DRV{number:02d}" for number in range(1, 21))
Q_TIME = "2025-07-05T15:00:00Z"
SPRINT_TIME = "2025-07-05T11:00:00Z"
GRID_TIME = "2025-07-06T12:00:00Z"
START_TIME = "2025-07-06T13:02:00Z"
CUTOFF = "2025-07-06T14:00:00Z"


def _classification(
    session: Session,
    order: Iterable[str],
    *,
    evidence_id: str,
    as_of: str | None = None,
) -> ClassificationSnapshot:
    ordered = tuple(order)
    return ClassificationSnapshot(
        session=session,
        entries=tuple(
            ClassificationEntry(driver_id=driver_id, position=position)
            for position, driver_id in enumerate(ordered, start=1)
        ),
        as_of=as_of or (SPRINT_TIME if session is Session.SPRINT else Q_TIME),
        evidence_id=evidence_id,
    )


def _official_grid(
    order: Iterable[str] = DRIVERS,
    *,
    revision_id: str = "grid-final-v1",
    phase: GridRevisionPhase = GridRevisionPhase.FINAL_PRE_RACE,
    as_of: str = GRID_TIME,
    replacements: dict[str, OfficialGridEntry] | None = None,
) -> OfficialGridRevision:
    replacements = replacements or {}
    entries = tuple(
        replacements.get(
            driver_id,
            OfficialGridEntry(driver_id=driver_id, position=position),
        )
        for position, driver_id in enumerate(tuple(order), start=1)
    )
    return OfficialGridRevision(
        revision_id=revision_id,
        phase=phase,
        entries=entries,
        as_of=as_of,
    )


def _standard_resolution(**kwargs: object):
    return resolve_grand_prix_start(
        build_weekend_contract(2025),
        as_of=CUTOFF,
        classifications=(_classification(Session.QUALIFYING, DRIVERS, evidence_id="q"),),
        **kwargs,
    )


def _positions(resolution: object) -> dict[str, int | None]:
    return {entry.driver_id: entry.position for entry in resolution.entries}


def _statuses(resolution: object) -> dict[str, GridEntryStatus]:
    return {entry.driver_id: entry.status for entry in resolution.entries}


@pytest.mark.parametrize("season", [2021, 2022])
def test_2021_and_2022_sprint_classification_supplies_provisional_gp_grid(
    season: int,
) -> None:
    contract = build_weekend_contract(season, WeekendFormat.SPRINT_2021_2022)
    sprint_order = tuple(reversed(DRIVERS))

    result = resolve_grand_prix_start(
        contract,
        as_of=CUTOFF,
        classifications=(
            _classification(Session.QUALIFYING, DRIVERS, evidence_id=f"q-{season}"),
            _classification(Session.SPRINT, sprint_order, evidence_id=f"sprint-{season}"),
        ),
    )

    assert result.qualifying_classification.session is Session.QUALIFYING
    assert result.qualifying_classification.entries[0].driver_id == DRIVERS[0]
    assert result.provisional_source_classification.session is Session.SPRINT
    assert result.provisional_grid.source is EvidenceSource.SPRINT_CLASSIFICATION
    assert result.provisional_grid.entries[0].driver_id == sprint_order[0]
    assert result.provisional_grid.status is ResolutionStatus.RESOLVED
    assert result.scheduled_grid.status is ResolutionStatus.UNRESOLVED
    assert "official_grid_revision_missing" in result.scheduled_grid.issues


def test_original_sprint_weekend_does_not_fall_back_to_qualifying_when_sprint_is_missing() -> None:
    result = resolve_grand_prix_start(
        build_weekend_contract(2022, WeekendFormat.SPRINT_2021_2022),
        as_of=CUTOFF,
        classifications=(_classification(Session.QUALIFYING, DRIVERS, evidence_id="q-only"),),
    )

    assert result.qualifying_classification.status is ResolutionStatus.RESOLVED
    assert result.provisional_source_classification.session is Session.SPRINT
    assert result.provisional_source_classification.status is ResolutionStatus.UNRESOLVED
    assert result.provisional_grid.entries == ()
    assert result.scheduled_grid.entries == ()


@pytest.mark.parametrize(
    ("season", "weekend_format"),
    [
        (2023, WeekendFormat.SPRINT_2023),
        (2024, WeekendFormat.SPRINT_2024_PLUS),
        (2025, WeekendFormat.SPRINT_2024_PLUS),
    ],
)
def test_2023_and_later_sprint_weekends_use_gp_qualifying_not_sprint(
    season: int,
    weekend_format: WeekendFormat,
) -> None:
    contract = build_weekend_contract(season, weekend_format)
    sprint_order = tuple(reversed(DRIVERS))

    result = resolve_grand_prix_start(
        contract,
        as_of=CUTOFF,
        classifications=(
            _classification(Session.QUALIFYING, DRIVERS, evidence_id=f"q-{season}"),
            _classification(Session.SPRINT, sprint_order, evidence_id=f"sprint-{season}"),
        ),
    )

    assert result.provisional_source_classification.session is Session.QUALIFYING
    assert result.provisional_grid.source is EvidenceSource.QUALIFYING_CLASSIFICATION
    assert tuple(entry.driver_id for entry in result.provisional_grid.entries) == DRIVERS


def test_2026_alternative_format_uses_qualifying_for_the_full_22_car_field() -> None:
    drivers = tuple(f"DRV{number:02d}" for number in range(1, 23))
    result = resolve_grand_prix_start(
        build_weekend_contract(2026, WeekendFormat.SPRINT_2024_PLUS),
        as_of=CUTOFF,
        classifications=(
            _classification(Session.SPRINT, reversed(drivers), evidence_id="sprint-2026"),
            _classification(Session.QUALIFYING, drivers, evidence_id="q-2026"),
        ),
    )

    assert result.provisional_source_classification.session is Session.QUALIFYING
    assert result.provisional_grid.status is ResolutionStatus.RESOLVED
    assert tuple(entry.driver_id for entry in result.provisional_grid.entries) == drivers


def test_standard_weekend_uses_gp_qualifying_classification() -> None:
    result = _standard_resolution()

    assert result.provisional_source_classification is result.qualifying_classification
    assert result.provisional_grid.source is EvidenceSource.QUALIFYING_CLASSIFICATION
    assert result.provisional_grid.status is ResolutionStatus.RESOLVED
    assert tuple(entry.driver_id for entry in result.provisional_grid.entries) == DRIVERS


def test_complete_final_official_revision_is_authoritative_over_classification() -> None:
    revised_order = (DRIVERS[1], DRIVERS[0], *DRIVERS[2:])
    result = _standard_resolution(
        official_grid_revisions=(
            _official_grid(revised_order, revision_id="official-final"),
        ),
    )

    assert result.qualifying_classification.entries[0].driver_id == DRIVERS[0]
    assert result.scheduled_grid.status is ResolutionStatus.RESOLVED
    assert result.scheduled_grid.source is EvidenceSource.OFFICIAL_GRID_REVISION
    assert result.scheduled_grid.phase == GridRevisionPhase.FINAL_PRE_RACE.value
    assert result.scheduled_grid.evidence_complete is True
    assert result.scheduled_grid.evidence_ids == ("official-final",)
    assert tuple(entry.driver_id for entry in result.scheduled_grid.entries) == revised_order
    assert all(
        entry.source is EvidenceSource.OFFICIAL_GRID_REVISION
        for entry in result.scheduled_grid.entries
    )


def test_complete_official_revision_can_authoritatively_encode_non_grid_cars() -> None:
    replacements = {
        DRIVERS[0]: OfficialGridEntry(
            driver_id=DRIVERS[0],
            position=None,
            status=GridEntryStatus.PIT_LANE,
            adjustments=(GridAdjustmentKind.PIT_LANE_START,),
        ),
        DRIVERS[4]: OfficialGridEntry(
            driver_id=DRIVERS[4],
            position=None,
            status=GridEntryStatus.WITHDRAWN,
            adjustments=(GridAdjustmentKind.WITHDRAWAL,),
        ),
        DRIVERS[9]: OfficialGridEntry(
            driver_id=DRIVERS[9],
            position=None,
            status=GridEntryStatus.DISQUALIFIED,
            adjustments=(GridAdjustmentKind.DISQUALIFICATION,),
        ),
    }
    result = _standard_resolution(
        official_grid_revisions=(
            _official_grid(revision_id="official-final-with-decisions", replacements=replacements),
        ),
    )

    positions = _positions(result.scheduled_grid)
    statuses = _statuses(result.scheduled_grid)
    assert result.scheduled_grid.status is ResolutionStatus.RESOLVED
    assert result.scheduled_grid.evidence_complete is True
    assert statuses[DRIVERS[0]] is GridEntryStatus.PIT_LANE
    assert statuses[DRIVERS[4]] is GridEntryStatus.WITHDRAWN
    assert statuses[DRIVERS[9]] is GridEntryStatus.DISQUALIFIED
    assert positions[DRIVERS[0]] is None
    assert positions[DRIVERS[1]] == 2
    assert positions[DRIVERS[5]] == 6
    assert positions[DRIVERS[10]] == 11


def test_late_official_revision_is_ignored_until_the_requested_cutoff() -> None:
    original = _official_grid(
        revision_id="grid-v1",
        as_of="2025-07-06T12:00:00Z",
    )
    revised_order = (DRIVERS[1], DRIVERS[0], *DRIVERS[2:])
    late = _official_grid(
        revised_order,
        revision_id="grid-v2",
        as_of="2025-07-06T13:00:00Z",
    )
    contract = build_weekend_contract(2025)
    classifications = (_classification(Session.QUALIFYING, DRIVERS, evidence_id="q"),)

    before = resolve_grand_prix_start(
        contract,
        as_of="2025-07-06T12:30:00Z",
        classifications=classifications,
        official_grid_revisions=(original, late),
    )
    after = resolve_grand_prix_start(
        contract,
        as_of="2025-07-06T13:30:00Z",
        classifications=classifications,
        official_grid_revisions=(original, late),
    )

    assert before.scheduled_grid.evidence_ids == ("grid-v1",)
    assert "grid-v2" in before.scheduled_grid.ignored_evidence_ids
    assert before.scheduled_grid.entries[0].driver_id == DRIVERS[0]
    assert after.scheduled_grid.evidence_ids == ("grid-v2",)
    assert "grid-v1" in after.scheduled_grid.ignored_evidence_ids
    assert after.scheduled_grid.entries[0].driver_id == DRIVERS[1]


def test_same_timestamp_conflicting_official_revisions_fail_closed() -> None:
    conflicting_order = (DRIVERS[1], DRIVERS[0], *DRIVERS[2:])
    result = _standard_resolution(
        official_grid_revisions=(
            _official_grid(revision_id="grid-a"),
            _official_grid(conflicting_order, revision_id="grid-b"),
        ),
    )

    assert result.scheduled_grid.status is ResolutionStatus.CONFLICTED
    assert result.scheduled_grid.evidence_complete is False
    assert result.scheduled_grid.entries == ()
    assert result.scheduled_grid.evidence_ids == ("grid-a", "grid-b")
    assert "conflicting_official_grid_revisions_at_same_timestamp" in result.scheduled_grid.issues


def test_provisional_and_post_start_revisions_cannot_masquerade_as_final_grid() -> None:
    provisional = _official_grid(
        revision_id="grid-provisional",
        phase=GridRevisionPhase.PROVISIONAL_PRE_RACE,
    )
    post_start = _official_grid(
        tuple(reversed(DRIVERS)),
        revision_id="grid-post-start",
        phase=GridRevisionPhase.POST_START,
        as_of="2025-07-06T13:05:00Z",
    )

    result = _standard_resolution(official_grid_revisions=(provisional, post_start))

    assert result.scheduled_grid.source is EvidenceSource.OFFICIAL_GRID_REVISION
    assert result.scheduled_grid.status is ResolutionStatus.PARTIAL
    assert result.scheduled_grid.evidence_complete is False
    assert result.scheduled_grid.phase == GridRevisionPhase.PROVISIONAL_PRE_RACE.value
    assert "official_grid_revision_is_provisional_not_final" in result.scheduled_grid.issues
    assert "grid-post-start" in result.scheduled_grid.ignored_evidence_ids


def test_grid_drop_is_recorded_without_inventing_the_reordered_field() -> None:
    result = _standard_resolution(
        official_grid_revisions=(_official_grid(),),
        adjustments=(
            GridAdjustment(
                driver_id=DRIVERS[0],
                kind=GridAdjustmentKind.GRID_DROP,
                places=5,
                as_of="2025-07-06T12:30:00Z",
                evidence_id="five-place-drop",
            ),
        ),
    )

    positions = _positions(result.scheduled_grid)
    statuses = _statuses(result.scheduled_grid)
    assert result.scheduled_grid.status is ResolutionStatus.UNRESOLVED
    assert result.scheduled_grid.source is EvidenceSource.GRID_ADJUSTMENT
    assert result.scheduled_grid.evidence_complete is False
    assert positions[DRIVERS[0]] is None
    assert statuses[DRIVERS[0]] is GridEntryStatus.PENALTY_PENDING
    assert result.scheduled_grid.entries[-1].adjustment_evidence[0].places == 5
    assert positions[DRIVERS[1]] == 2
    assert "five-place-drop" in result.scheduled_grid.evidence_ids


def test_individual_adjustment_cannot_upgrade_provisional_only_grid_to_partial() -> None:
    result = _standard_resolution(
        adjustments=(
            GridAdjustment(
                driver_id=DRIVERS[0],
                kind=GridAdjustmentKind.PIT_LANE_START,
                as_of="2025-07-06T12:20:00Z",
                evidence_id="pit-lane-without-final-grid",
            ),
        ),
    )

    assert result.scheduled_grid.status is ResolutionStatus.UNRESOLVED
    assert result.scheduled_grid.evidence_complete is False
    assert _statuses(result.scheduled_grid)[DRIVERS[0]] is GridEntryStatus.PIT_LANE
    assert "official_grid_revision_missing" in result.scheduled_grid.issues


def test_same_timestamp_penalties_with_different_magnitudes_are_conflicted() -> None:
    result = _standard_resolution(
        official_grid_revisions=(_official_grid(),),
        adjustments=(
            GridAdjustment(
                driver_id=DRIVERS[0],
                kind=GridAdjustmentKind.GRID_DROP,
                places=5,
                as_of="2025-07-06T12:30:00Z",
                evidence_id="five-place-report",
            ),
            GridAdjustment(
                driver_id=DRIVERS[0],
                kind=GridAdjustmentKind.GRID_DROP,
                places=10,
                as_of="2025-07-06T12:30:00Z",
                evidence_id="ten-place-report",
            ),
        ),
    )

    row = next(entry for entry in result.scheduled_grid.entries if entry.driver_id == DRIVERS[0])
    assert result.scheduled_grid.status is ResolutionStatus.CONFLICTED
    assert row.status is GridEntryStatus.UNRESOLVED
    assert row.position is None
    assert {decision.places for decision in row.adjustment_evidence} == {5, 10}
    assert "conflicting_grid_adjustments:DRV01" in result.scheduled_grid.issues


def test_pit_lane_withdrawal_and_disqualification_preserve_grid_gaps() -> None:
    result = _standard_resolution(
        official_grid_revisions=(_official_grid(),),
        adjustments=(
            GridAdjustment(
                driver_id=DRIVERS[0],
                kind=GridAdjustmentKind.PIT_LANE_START,
                as_of="2025-07-06T12:20:00Z",
                evidence_id="pit-lane",
            ),
            GridAdjustment(
                driver_id=DRIVERS[4],
                kind=GridAdjustmentKind.WITHDRAWAL,
                as_of="2025-07-06T12:25:00Z",
                evidence_id="withdrawal",
            ),
            GridAdjustment(
                driver_id=DRIVERS[9],
                kind=GridAdjustmentKind.DISQUALIFICATION,
                as_of="2025-07-06T12:30:00Z",
                evidence_id="dsq",
            ),
        ),
    )

    positions = _positions(result.scheduled_grid)
    statuses = _statuses(result.scheduled_grid)
    assert result.scheduled_grid.status is ResolutionStatus.PARTIAL
    assert positions[DRIVERS[0]] is None
    assert statuses[DRIVERS[0]] is GridEntryStatus.PIT_LANE
    assert positions[DRIVERS[4]] is None
    assert statuses[DRIVERS[4]] is GridEntryStatus.WITHDRAWN
    assert positions[DRIVERS[9]] is None
    assert statuses[DRIVERS[9]] is GridEntryStatus.DISQUALIFIED
    assert positions[DRIVERS[1]] == 2
    assert positions[DRIVERS[5]] == 6
    assert positions[DRIVERS[10]] == 11


def test_later_complete_official_grid_absorbs_earlier_adjustment_evidence() -> None:
    revision = _official_grid(
        (DRIVERS[1], DRIVERS[0], *DRIVERS[2:]),
        revision_id="grid-after-penalty",
        as_of="2025-07-06T12:40:00Z",
    )
    result = _standard_resolution(
        official_grid_revisions=(revision,),
        adjustments=(
            GridAdjustment(
                driver_id=DRIVERS[0],
                kind=GridAdjustmentKind.GRID_DROP,
                places=1,
                as_of="2025-07-06T12:30:00Z",
                evidence_id="penalty-before-grid",
            ),
        ),
    )

    assert result.scheduled_grid.status is ResolutionStatus.RESOLVED
    assert result.scheduled_grid.source is EvidenceSource.OFFICIAL_GRID_REVISION
    assert result.scheduled_grid.entries[0].driver_id == DRIVERS[1]
    assert "penalty-before-grid" in result.scheduled_grid.ignored_evidence_ids


def test_missing_actual_start_evidence_never_copies_the_scheduled_grid() -> None:
    result = _standard_resolution(official_grid_revisions=(_official_grid(),))

    assert result.scheduled_grid.status is ResolutionStatus.RESOLVED
    assert len(result.scheduled_grid.entries) == 20
    assert result.actual_start.status is ResolutionStatus.UNRESOLVED
    assert result.actual_start.source is EvidenceSource.NONE
    assert result.actual_start.entries == ()
    assert result.actual_starters == ()


def test_actual_starters_are_resolved_only_from_explicit_observed_evidence() -> None:
    actual_entries = tuple(
        ActualStartEntry(
            driver_id=driver_id,
            status=(
                ActualStartStatus.STARTED_PIT_LANE
                if index == 18
                else ActualStartStatus.DID_NOT_START
                if index == 19
                else ActualStartStatus.WITHDRAWN
                if index == 20
                else ActualStartStatus.STARTED
            ),
            start_order=index if index <= 18 else None,
        )
        for index, driver_id in enumerate(DRIVERS, start=1)
    )
    result = _standard_resolution(
        official_grid_revisions=(_official_grid(),),
        actual_start_snapshots=(
            ActualStartSnapshot(
                entries=actual_entries,
                as_of=START_TIME,
                evidence_id="timing-loop-start",
            ),
        ),
    )

    assert result.actual_start.status is ResolutionStatus.RESOLVED
    assert result.actual_start.source is EvidenceSource.ACTUAL_START_EVIDENCE
    assert result.actual_start.evidence_complete is True
    assert len(result.actual_starters) == 18
    assert result.actual_starters[-1].driver_id == DRIVERS[17]
    assert result.actual_starters[-1].status is GridEntryStatus.STARTED_PIT_LANE
    assert _statuses(result.actual_start)[DRIVERS[18]] is GridEntryStatus.DID_NOT_START
    assert _statuses(result.actual_start)[DRIVERS[19]] is GridEntryStatus.WITHDRAWN


def test_partial_actual_start_roster_stays_partial_and_reports_missing_driver() -> None:
    snapshot = ActualStartSnapshot(
        entries=tuple(
            ActualStartEntry(
                driver_id=driver_id,
                status=ActualStartStatus.STARTED,
                start_order=index,
            )
            for index, driver_id in enumerate(DRIVERS[:-1], start=1)
        ),
        as_of=START_TIME,
        evidence_id="partial-start",
    )
    result = _standard_resolution(
        official_grid_revisions=(_official_grid(),),
        actual_start_snapshots=(snapshot,),
    )

    assert result.actual_start.status is ResolutionStatus.PARTIAL
    assert result.actual_start.evidence_complete is False
    assert "actual_start_missing_drivers:DRV20" in result.actual_start.issues


def test_all_resolved_outputs_and_rows_carry_provenance_and_completeness() -> None:
    result = _standard_resolution(official_grid_revisions=(_official_grid(),))

    for output in (
        result.qualifying_classification,
        result.provisional_source_classification,
        result.provisional_grid,
        result.scheduled_grid,
        result.actual_start,
    ):
        assert isinstance(output.status, ResolutionStatus)
        assert isinstance(output.source, EvidenceSource)
        assert output.as_of == CUTOFF
        assert isinstance(output.evidence_complete, bool)
        for entry in output.entries:
            assert isinstance(entry.source, EvidenceSource)
            assert entry.as_of.endswith("Z")
            assert isinstance(entry.evidence_complete, bool)


def test_naive_cutoff_is_rejected_instead_of_silently_assuming_a_timezone() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        resolve_grand_prix_start(
            build_weekend_contract(2025),
            as_of="2025-07-06T14:00:00",
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: OfficialGridEntry(
            driver_id="DRV01",
            position=1,
            evidence_complete="false",  # type: ignore[arg-type]
        ),
        lambda: OfficialGridRevision(
            revision_id="bad-bool",
            phase=GridRevisionPhase.FINAL_PRE_RACE,
            entries=(OfficialGridEntry(driver_id="DRV01", position=1),),
            as_of=GRID_TIME,
            evidence_complete="false",  # type: ignore[arg-type]
        ),
        lambda: GridAdjustment(
            driver_id="DRV01",
            kind=GridAdjustmentKind.PIT_LANE_START,
            as_of=GRID_TIME,
            evidence_id="bad-bool",
            evidence_complete="false",  # type: ignore[arg-type]
        ),
    ],
)
def test_domain_evidence_completeness_rejects_string_booleans(factory) -> None:
    with pytest.raises(TypeError, match="must be a bool"):
        factory()


@pytest.mark.parametrize("driver_id", ["", "nan", "None", "null", "<NA>"])
def test_domain_driver_id_rejects_missing_sentinels(driver_id: str) -> None:
    with pytest.raises(ValueError, match="driver_id"):
        OfficialGridEntry(driver_id=driver_id, position=1)
