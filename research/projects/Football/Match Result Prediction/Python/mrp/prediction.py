"""Prediction orchestration for football match result prediction."""

from __future__ import annotations

from dataclasses import dataclass

from .config import PredictionConfig
from .constants import MODEL_VERSION
from .data import load_local_football_data, select_target_fixtures, select_training_matches
from .training import (
    build_history_from_matches,
    fit_dixon_coles,
    fit_probability_calibrator,
    fixture_feature_vector,
    most_likely_scoreline,
    outcome_probabilities,
    train_gradient_boosting_benchmark,
)
from .utils import datetime_to_iso, dedupe_preserve_order, format_decimal, format_probability


@dataclass(frozen=True)
class PredictionResult:
    version: str
    rows: list[dict[str, str]]
    notes: list[str]


def run_prediction(config: PredictionConfig) -> PredictionResult:
    notes: list[str] = []
    rows: list[dict[str, str]] = []

    dataset, data_notes = load_local_football_data(config)
    notes.extend(data_notes)

    fixtures, fixture_notes = select_target_fixtures(dataset, config)
    notes.extend(fixture_notes)

    if not fixtures:
        notes.append(
            "Aucune prediction generee: aucun fixture local correspondant aux filtres."
        )
        return PredictionResult(
            version=MODEL_VERSION,
            rows=[],
            notes=dedupe_preserve_order(notes),
        )

    training_matches, training_notes = select_training_matches(dataset, config, fixtures)
    notes.extend(training_notes)

    dixon_coles_model = fit_dixon_coles(training_matches, notes)
    calibrator = fit_probability_calibrator(training_matches, dixon_coles_model, notes)
    benchmark_model = train_gradient_boosting_benchmark(training_matches, notes)
    history = build_history_from_matches(training_matches)

    for fixture in fixtures:
        lambda_home, lambda_away = dixon_coles_model.expected_goals(
            fixture.home_team_id, fixture.away_team_id
        )
        raw_home, raw_draw, raw_away = outcome_probabilities(
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            rho=dixon_coles_model.rho,
        )
        calibrated_home, calibrated_draw, calibrated_away = calibrator.apply(
            (raw_home, raw_draw, raw_away)
        )
        score_home, score_away, scoreline_probability = most_likely_scoreline(
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            rho=dixon_coles_model.rho,
        )

        row: dict[str, str] = {
            "mode": config.mode,
            "match_id": fixture.match_id,
            "date": datetime_to_iso(fixture.date),
            "league": fixture.league or config.league,
            "season": str(fixture.season if fixture.season is not None else config.season),
            "round": str(
                fixture.round_number if fixture.round_number is not None else config.round_number
            ),
            "home_team_id": fixture.home_team_id,
            "away_team_id": fixture.away_team_id,
            "home_team_name": dataset.resolve_team_name(fixture.home_team_id),
            "away_team_name": dataset.resolve_team_name(fixture.away_team_id),
            "lambda_home_goals": format_decimal(lambda_home, 3),
            "lambda_away_goals": format_decimal(lambda_away, 3),
            "home_win_prob": format_probability(calibrated_home),
            "draw_prob": format_probability(calibrated_draw),
            "away_win_prob": format_probability(calibrated_away),
            "raw_home_win_prob": format_probability(raw_home),
            "raw_draw_prob": format_probability(raw_draw),
            "raw_away_win_prob": format_probability(raw_away),
            "predicted_scoreline": f"{score_home}-{score_away}",
            "scoreline_prob": format_probability(scoreline_probability),
            "calibration_method": calibrator.method,
            "primary_model": "dixon_coles",
        }

        if benchmark_model is not None:
            benchmark_probs = benchmark_model.predict(fixture_feature_vector(fixture, history))
            row["gbm_home_win_prob"] = format_probability(benchmark_probs[0])
            row["gbm_draw_prob"] = format_probability(benchmark_probs[1])
            row["gbm_away_win_prob"] = format_probability(benchmark_probs[2])
            row["gbm_log_loss"] = format_decimal(benchmark_model.log_loss_value, 4)
            row["gbm_brier"] = format_decimal(benchmark_model.brier_value, 4)

        rows.append(row)

    if config.mode == "scoreline":
        notes.append("Mode scoreline: score probable (0..6) fourni avec proba 1X2.")
    else:
        notes.append("Mode match_result: proba 1X2 et scoreline le plus probable fournis.")

    return PredictionResult(
        version=MODEL_VERSION,
        rows=rows,
        notes=dedupe_preserve_order(notes),
    )
