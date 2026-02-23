from __future__ import annotations

from datetime import datetime
import subprocess
import sys
from pathlib import Path

from mrp.data import FixtureRecord, MatchRecord
from mrp.prediction import _is_hybrid_eligible
from mrp.training import (
    evaluate_match_probabilities,
    fit_dixon_coles,
    fit_frequency_baseline,
    fit_probability_calibrator,
    fit_probability_calibrator_with_policy,
    normalize_probabilities,
    rank_outcome_classes,
    select_hybrid_weight,
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


def _many_matches(size: int = 40) -> list[MatchRecord]:
    matches: list[MatchRecord] = []
    teams = ["a", "b", "c", "d", "e", "f"]
    for idx in range(size):
        home = teams[idx % len(teams)]
        away = teams[(idx + 1) % len(teams)]
        home_goals = (idx * 3) % 4
        away_goals = (idx * 5) % 3
        matches.append(
            _match(
                match_id=f"mx_{idx}",
                season=2024 + (idx // 20),
                home_team_id=home,
                away_team_id=away,
                home_goals=home_goals,
                away_goals=away_goals,
            )
        )
    return matches


def test_hybrid_ineligible_path_sets_fallback_state() -> None:
    notes: list[str] = []
    training_matches = _many_matches(size=10)
    fixtures = [
        FixtureRecord(
            match_id="f1",
            date=datetime(2026, 1, 1),
            season=2026,
            league="epl",
            round_number=1,
            home_team_id="a",
            away_team_id="b",
        )
    ]
    eligible = _is_hybrid_eligible(
        training_matches=training_matches,
        fixtures=fixtures,
        validation_size=4,
        benchmark_available=False,
        notes=notes,
    )
    assert eligible is False
    assert any("Hybrid gate" in note for note in notes)


def test_hybrid_weight_selected_once_from_grid_and_probabilities_normalized() -> None:
    labels = [0, 1, 2, 0, 1, 2]
    dc_probs = [
        (0.50, 0.30, 0.20),
        (0.20, 0.50, 0.30),
        (0.25, 0.25, 0.50),
        (0.55, 0.25, 0.20),
        (0.30, 0.40, 0.30),
        (0.20, 0.30, 0.50),
    ]
    gbdt_probs = [
        (0.70, 0.20, 0.10),
        (0.10, 0.75, 0.15),
        (0.10, 0.20, 0.70),
        (0.65, 0.20, 0.15),
        (0.20, 0.65, 0.15),
        (0.10, 0.20, 0.70),
    ]
    notes: list[str] = []
    weight = select_hybrid_weight(labels, dc_probs, gbdt_probs, notes)
    assert weight in {round(x / 10.0, 1) for x in range(11)}
    blended = normalize_probabilities(
        (
            (weight * gbdt_probs[0][0]) + ((1.0 - weight) * dc_probs[0][0]),
            (weight * gbdt_probs[0][1]) + ((1.0 - weight) * dc_probs[0][1]),
            (weight * gbdt_probs[0][2]) + ((1.0 - weight) * dc_probs[0][2]),
        )
    )
    assert abs(sum(blended) - 1.0) < 1e-9


def test_dixon_auto_calibration_policy_matches_legacy_path() -> None:
    matches = _many_matches(size=45)
    model_notes: list[str] = []
    model = fit_dixon_coles(matches, model_notes)
    notes_direct: list[str] = []
    notes_policy: list[str] = []
    direct = fit_probability_calibrator(matches, model, notes_direct)
    with_policy = fit_probability_calibrator_with_policy(
        matches,
        model,
        notes_policy,
        policy="auto",
    )
    assert with_policy.method == direct.method


def test_cli_help_exposes_horizon_a_flags() -> None:
    project_python_dir = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(project_python_dir / "run_prediction.py"), "--help"]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stdout = completed.stdout
    assert "--football_model" in stdout
    assert "--football_calibration" in stdout
    assert "--shadow_eval" in stdout
