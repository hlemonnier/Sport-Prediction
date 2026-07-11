from __future__ import annotations

import pytest

from packages.f1.domain import (
    GridTarget,
    ParcFermeWindow,
    PredictionTarget,
    Session,
    SessionCutoff,
    WeekendFormat,
    build_weekend_contract,
    canonicalize_session_sequence,
    infer_weekend_contract,
    infer_weekend_format,
    parse_session_cutoff,
    qualifying_elimination_rule,
)


def test_standard_weekend_exposes_order_dag_grid_and_safe_cutoffs() -> None:
    contract = build_weekend_contract(2025)

    assert contract.format is WeekendFormat.STANDARD
    assert contract.session_order == (
        Session.FP1,
        Session.FP2,
        Session.FP3,
        Session.QUALIFYING,
        Session.RACE,
    )
    assert [(edge.predecessor, edge.successor) for edge in contract.session_dag] == [
        (Session.FP1, Session.FP2),
        (Session.FP2, Session.FP3),
        (Session.FP3, Session.QUALIFYING),
        (Session.QUALIFYING, Session.RACE),
    ]
    assert contract.ancestors(Session.QUALIFYING) == (
        Session.FP1,
        Session.FP2,
        Session.FP3,
    )
    assert contract.grid_source(GridTarget.RACE) is Session.QUALIFYING
    assert contract.parc_ferme_windows == (
        ParcFermeWindow(Session.QUALIFYING, Session.RACE),
    )

    assert contract.eligible_sessions(
        PredictionTarget.GRAND_PRIX_QUALIFYING,
        SessionCutoff.after(Session.FP3),
    ) == (Session.FP1, Session.FP2, Session.FP3)
    assert contract.eligible_sessions(
        PredictionTarget.RACE,
        SessionCutoff.before(Session.RACE),
    ) == (Session.FP1, Session.FP2, Session.FP3, Session.QUALIFYING)


@pytest.mark.parametrize("season", [2021, 2022])
def test_original_sprint_format_uses_sprint_as_race_grid_source(season: int) -> None:
    contract = infer_weekend_contract(
        season,
        {
            "Session1": "Practice 1",
            "Session2": "Qualifying",
            "Session3": "Practice 2",
            "Session4": "Sprint Qualifying",
            "Session5": "Race",
        },
    )

    assert contract.format is WeekendFormat.SPRINT_2021_2022
    assert contract.session_order == (
        Session.FP1,
        Session.QUALIFYING,
        Session.FP2,
        Session.SPRINT,
        Session.RACE,
    )
    assert contract.grid_source(GridTarget.SPRINT) is Session.QUALIFYING
    assert contract.grid_source(GridTarget.RACE) is Session.SPRINT
    assert contract.sprint_qualifying_rule is None
    assert contract.parc_ferme_windows == (
        ParcFermeWindow(Session.QUALIFYING, Session.RACE),
    )
    assert contract.setup_reopens_after == ()
    assert contract.eligible_sessions(
        PredictionTarget.GRAND_PRIX_QUALIFYING,
        SessionCutoff.before(Session.QUALIFYING),
    ) == (Session.FP1,)


def test_2023_standalone_sprint_keeps_race_qualifying_before_sprint() -> None:
    contract = infer_weekend_contract(
        2023,
        ["FP1", "Qualifying", "Sprint Shootout", "Sprint", "Grand Prix"],
    )

    assert contract.format is WeekendFormat.SPRINT_2023
    assert contract.session_order == (
        Session.FP1,
        Session.QUALIFYING,
        Session.SPRINT_QUALIFYING,
        Session.SPRINT,
        Session.RACE,
    )
    assert contract.grid_source(GridTarget.SPRINT) is Session.SPRINT_QUALIFYING
    assert contract.grid_source(GridTarget.RACE) is Session.QUALIFYING
    assert contract.parc_ferme_windows == (
        ParcFermeWindow(Session.QUALIFYING, Session.RACE),
    )
    assert contract.setup_reopens_after == ()
    assert contract.eligible_sessions(
        PredictionTarget.SPRINT,
        SessionCutoff.before(Session.SPRINT),
    ) == (Session.FP1, Session.QUALIFYING, Session.SPRINT_QUALIFYING)


def test_2026_alternative_format_has_22_cars_two_windows_and_setup_reset() -> None:
    contract = infer_weekend_contract(
        2026,
        [
            {"session_name": "Free Practice 1"},
            {"session_name": "Sprint Qualifying"},
            {"session_name": "Sprint"},
            {"session_name": "Race Qualifying"},
            {"session_name": "Race"},
        ],
    )

    assert contract.format is WeekendFormat.SPRINT_2024_PLUS
    assert contract.session_order == (
        Session.FP1,
        Session.SPRINT_QUALIFYING,
        Session.SPRINT,
        Session.QUALIFYING,
        Session.RACE,
    )
    assert contract.eligible_cars == 22
    assert contract.race_qualifying_rule.eliminated_after_period_1 == 6
    assert contract.race_qualifying_rule.eliminated_after_period_2 == 6
    assert contract.race_qualifying_rule.period_2_cars == 16
    assert contract.race_qualifying_rule.period_3_cars == 10
    assert contract.sprint_qualifying_rule == contract.race_qualifying_rule
    assert contract.grid_source(GridTarget.SPRINT) is Session.SPRINT_QUALIFYING
    assert contract.grid_source(GridTarget.RACE) is Session.QUALIFYING
    assert contract.parc_ferme_windows == (
        ParcFermeWindow(Session.SPRINT_QUALIFYING, Session.SPRINT),
        ParcFermeWindow(Session.QUALIFYING, Session.RACE),
    )
    assert contract.setup_reopens_after == (Session.SPRINT,)
    assert contract.eligible_sessions(
        PredictionTarget.GRAND_PRIX_QUALIFYING,
        SessionCutoff.before(Session.QUALIFYING),
    ) == (Session.FP1, Session.SPRINT_QUALIFYING, Session.SPRINT)


def test_target_boundary_fails_closed_for_late_cutoff() -> None:
    contract = build_weekend_contract(2026, WeekendFormat.SPRINT_2024_PLUS)

    assert contract.eligible_sessions(
        PredictionTarget.GRAND_PRIX_QUALIFYING,
        SessionCutoff.after(Session.RACE),
    ) == (Session.FP1, Session.SPRINT_QUALIFYING, Session.SPRINT)
    assert contract.eligible_sessions(
        PredictionTarget.RACE,
        SessionCutoff.after(Session.RACE),
    ) == (
        Session.FP1,
        Session.SPRINT_QUALIFYING,
        Session.SPRINT,
        Session.QUALIFYING,
    )
    assert contract.eligible_sessions(
        PredictionTarget.GRAND_PRIX_STARTING_GRID,
        SessionCutoff.after(Session.QUALIFYING),
    ) == (
        Session.FP1,
        Session.SPRINT_QUALIFYING,
        Session.SPRINT,
        Session.QUALIFYING,
    )


def test_availability_matrix_is_explicit_at_every_session_cutoff() -> None:
    contract = build_weekend_contract(2025)
    rows = contract.availability_matrix([PredictionTarget.GRAND_PRIX_QUALIFYING])

    assert len(rows) == len(contract.session_order) + 1
    assert rows[0].cutoff == SessionCutoff.before_weekend()
    assert rows[0].eligible_sessions == ()
    post_fp2 = next(row for row in rows if row.cutoff.label == "after_FP2")
    assert post_fp2.eligible_sessions == (Session.FP1, Session.FP2)
    post_race = next(row for row in rows if row.cutoff.label == "after_Race")
    assert post_race.eligible_sessions == (Session.FP1, Session.FP2, Session.FP3)


def test_inference_rejects_session_order_from_the_wrong_regulation_era() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        infer_weekend_format(
            2023,
            ["FP1", "Sprint Shootout", "Sprint", "Qualifying", "Race"],
        )


def test_canonicalization_rejects_unknown_or_duplicate_metadata() -> None:
    with pytest.raises(ValueError, match="Unknown F1 session label"):
        canonicalize_session_sequence(2026, ["FP1", "Fan Parade", "Race"])
    with pytest.raises(ValueError, match="duplicate canonical sessions"):
        canonicalize_session_sequence(2026, ["FP1", "Practice 1", "Race"])


def test_elimination_rule_handles_supported_even_fields_without_guessing_odd() -> None:
    twenty = qualifying_elimination_rule(20)
    twenty_four = qualifying_elimination_rule(24)

    assert (twenty.eliminated_after_period_1, twenty.period_2_cars) == (5, 15)
    assert (twenty_four.eliminated_after_period_1, twenty_four.period_2_cars) == (7, 17)
    with pytest.raises(ValueError, match="even eligible field"):
        qualifying_elimination_rule(21)


def test_format_specific_targets_and_cutoffs_reject_absent_sessions() -> None:
    standard = build_weekend_contract(2025)

    with pytest.raises(ValueError, match="absent"):
        standard.eligible_sessions(
            PredictionTarget.SPRINT,
            SessionCutoff.after(Session.FP1),
        )
    with pytest.raises(ValueError, match="absent"):
        standard.completed_sessions(SessionCutoff.after(Session.SPRINT))
    with pytest.raises(ValueError, match="no sprint grid"):
        standard.grid_source(GridTarget.SPRINT)


def test_config_cutoff_aliases_resolve_against_real_weekend_and_fail_closed() -> None:
    standard = build_weekend_contract(2025)
    sprint = build_weekend_contract(2026, WeekendFormat.SPRINT_2024_PLUS)

    assert parse_session_cutoff(
        standard,
        "post_fp2",
        target=PredictionTarget.GRAND_PRIX_QUALIFYING,
    ) == SessionCutoff.after(Session.FP2)
    assert parse_session_cutoff(
        sprint,
        "post_sprint_pre_qualifying",
        target=PredictionTarget.GRAND_PRIX_QUALIFYING,
    ) == SessionCutoff.after(Session.SPRINT)
    assert parse_session_cutoff(
        sprint,
        "auto",
        target=PredictionTarget.GRAND_PRIX_QUALIFYING,
    ) == SessionCutoff.before(Session.QUALIFYING)
    with pytest.raises(ValueError, match="absent"):
        parse_session_cutoff(
            sprint,
            "post_fp2",
            target=PredictionTarget.GRAND_PRIX_QUALIFYING,
        )
    with pytest.raises(ValueError, match="Unknown F1 session cutoff"):
        parse_session_cutoff(
            standard,
            "after_magic_session",
            target=PredictionTarget.RACE,
        )
