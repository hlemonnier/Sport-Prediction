from __future__ import annotations

from datetime import datetime

from mrp.data import FixtureRecord, MatchRecord
from mrp.training import (
    evaluate_match_probabilities,
    fit_frequency_baseline,
    rank_outcome_classes,
)


def _match(
    match_id: str,
    season: int,
    home_team_id: str,
    away_team_id: str,
    home_goals: int,
    away_goals: int,
) -> MatchRecord:
    return MatchRecord(
        match_id=match_id,
        date=datetime(season, 1, 1),
        season=season,
        league="epl",
        round_number=1,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_goals=home_goals,
        away_goals=away_goals,
        home_xg=None,
        away_xg=None,
    )


def test_frequency_baseline_is_deterministic_and_ranked() -> None:
    matches = [
        _match("m1", 2024, "a", "b", 2, 0),
        _match("m2", 2024, "c", "a", 1, 1),
        _match("m3", 2025, "a", "c", 3, 1),
        _match("m4", 2025, "b", "a", 1, 2),
    ]
    notes: list[str] = []
    model = fit_frequency_baseline(matches, notes)
    fixture = FixtureRecord(
        match_id="f1",
        date=datetime(2025, 2, 1),
        season=2025,
        league="epl",
        round_number=2,
        home_team_id="a",
        away_team_id="b",
    )

    probabilities_first = model.predict(fixture)
    probabilities_second = model.predict(fixture)
    ranked = rank_outcome_classes(probabilities_first)

    assert probabilities_first == probabilities_second
    assert abs(sum(probabilities_first) - 1.0) < 1e-9
    assert ranked[0] == 0  # HOME_WIN_CLASS
    assert probabilities_first[0] > probabilities_first[1]
    assert probabilities_first[0] > probabilities_first[2]


def test_probability_diagnostics_include_calibration_and_rank_metrics() -> None:
    matches = [
        _match("m1", 2025, "a", "b", 2, 0),
        _match("m2", 2025, "c", "d", 1, 1),
        _match("m3", 2025, "e", "f", 0, 3),
    ]

    def perfect_predictor(match: MatchRecord) -> tuple[float, float, float]:
        if match.home_goals is None or match.away_goals is None:
            return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        if match.home_goals > match.away_goals:
            return (1.0, 0.0, 0.0)
        if match.home_goals < match.away_goals:
            return (0.0, 0.0, 1.0)
        return (0.0, 1.0, 0.0)

    diagnostics = evaluate_match_probabilities(matches, perfect_predictor, bins=5)
    calibration = diagnostics["calibration"]
    ranking = diagnostics["ranking"]

    assert diagnostics["sample_size"] == 3
    assert calibration["brier"] == 0.0
    assert calibration["ece"] == 0.0
    assert calibration["log_loss"] < 1e-9
    assert ranking["top1_accuracy"] == 1.0
    assert ranking["top2_accuracy"] == 1.0
    assert ranking["mean_reciprocal_rank"] == 1.0
    assert ranking["avg_confidence"] == 1.0
