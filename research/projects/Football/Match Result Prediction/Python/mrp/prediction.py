"""Prediction orchestration for football match result prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import PredictionConfig
from .constants import AWAY_WIN_CLASS, DRAW_CLASS, HOME_WIN_CLASS, MODEL_VERSION
from .data import load_local_football_data, select_target_fixtures, select_training_matches
from .training import (
    build_history_from_matches,
    evaluate_match_probabilities,
    fit_dixon_coles,
    fit_frequency_baseline,
    fit_probability_calibrator,
    fixture_feature_vector,
    most_likely_scoreline,
    outcome_probabilities,
    rank_outcome_classes,
    train_gradient_boosting_benchmark,
)
from .utils import datetime_to_iso, dedupe_preserve_order, format_decimal, format_probability


@dataclass(frozen=True)
class PredictionResult:
    version: str
    rows: list[dict[str, str]]
    notes: list[str]
    diagnostics: dict[str, Any]


_OUTCOME_LABELS = {
    HOME_WIN_CLASS: "home_win",
    DRAW_CLASS: "draw",
    AWAY_WIN_CLASS: "away_win",
}


def _ranked_outcome_labels(probabilities: tuple[float, float, float]) -> list[str]:
    ranked_classes = rank_outcome_classes(probabilities)
    return [_OUTCOME_LABELS[outcome_class] for outcome_class in ranked_classes]


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
            diagnostics={
                "training_sample_size": 0,
                "models": {},
            },
        )

    training_matches, training_notes = select_training_matches(dataset, config, fixtures)
    notes.extend(training_notes)

    baseline_model = fit_frequency_baseline(training_matches, notes)
    dixon_coles_model = fit_dixon_coles(training_matches, notes)
    calibrator = fit_probability_calibrator(training_matches, dixon_coles_model, notes)
    benchmark_model = train_gradient_boosting_benchmark(training_matches, notes)
    history = build_history_from_matches(training_matches)

    def _predict_dixon_coles_calibrated(match_record: Any) -> tuple[float, float, float]:
        lambda_home_eval, lambda_away_eval = dixon_coles_model.expected_goals(
            match_record.home_team_id, match_record.away_team_id
        )
        raw_eval = outcome_probabilities(
            lambda_home=lambda_home_eval,
            lambda_away=lambda_away_eval,
            rho=dixon_coles_model.rho,
        )
        return calibrator.apply(raw_eval)

    diagnostics: dict[str, Any] = {
        "training_sample_size": len(training_matches),
        "models": {
            "baseline_frequency": evaluate_match_probabilities(
                training_matches, baseline_model.predict
            ),
            "dixon_coles_calibrated": evaluate_match_probabilities(
                training_matches, _predict_dixon_coles_calibrated
            ),
        },
    }

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
        baseline_home, baseline_draw, baseline_away = baseline_model.predict(fixture)
        primary_rank = _ranked_outcome_labels((calibrated_home, calibrated_draw, calibrated_away))
        baseline_rank = _ranked_outcome_labels((baseline_home, baseline_draw, baseline_away))
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
            "predicted_outcome": primary_rank[0],
            "prediction_confidence": format_probability(
                max(calibrated_home, calibrated_draw, calibrated_away)
            ),
            "outcome_rank_1": primary_rank[0],
            "outcome_rank_2": primary_rank[1],
            "outcome_rank_3": primary_rank[2],
            "raw_home_win_prob": format_probability(raw_home),
            "raw_draw_prob": format_probability(raw_draw),
            "raw_away_win_prob": format_probability(raw_away),
            "baseline_home_win_prob": format_probability(baseline_home),
            "baseline_draw_prob": format_probability(baseline_draw),
            "baseline_away_win_prob": format_probability(baseline_away),
            "baseline_predicted_outcome": baseline_rank[0],
            "baseline_confidence": format_probability(
                max(baseline_home, baseline_draw, baseline_away)
            ),
            "baseline_outcome_rank_1": baseline_rank[0],
            "baseline_outcome_rank_2": baseline_rank[1],
            "baseline_outcome_rank_3": baseline_rank[2],
            "predicted_scoreline": f"{score_home}-{score_away}",
            "scoreline_prob": format_probability(scoreline_probability),
            "calibration_method": calibrator.method,
            "primary_model": "dixon_coles_calibrated",
            "baseline_model": "frequency_by_season_and_team",
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
        diagnostics=diagnostics,
    )
