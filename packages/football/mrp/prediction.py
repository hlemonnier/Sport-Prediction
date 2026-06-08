"""Prediction orchestration for football match result prediction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .config import PredictionConfig
from .constants import (
    AWAY_WIN_CLASS,
    DRAW_CLASS,
    HOME_WIN_CLASS,
    HYBRID_MIN_TEAM_MATCHES,
    HYBRID_MIN_TRAIN_MATCHES,
    HYBRID_MIN_VALIDATION_SIZE,
    MODEL_VERSION,
)
from .data import load_local_football_data, select_target_fixtures, select_training_matches
from .training import (
    build_history_from_matches,
    evaluate_match_probabilities,
    fit_dixon_coles,
    fit_frequency_baseline,
    fit_probability_calibrator,
    fit_probability_calibrator_from_rows,
    fit_probability_calibrator_with_policy,
    fixture_feature_vector,
    most_likely_scoreline,
    normalize_probabilities,
    outcome_probabilities,
    rank_outcome_classes,
    select_hybrid_weight,
    train_gradient_boosting_benchmark,
)
from .utils import datetime_to_iso, dedupe_preserve_order, format_decimal, format_probability
from .weather import fetch_fixture_weather_summary


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


def _normalize_football_model(value: str) -> str:
    normalized = str(value or "dixon").strip().lower()
    if normalized in {"dixon", "gbdt", "hybrid"}:
        return normalized
    return "dixon"


def _normalize_calibration_policy(value: str) -> str:
    normalized = str(value or "auto").strip().lower()
    if normalized in {"off", "auto", "platt", "isotonic"}:
        return normalized
    return "auto"


def _format_weather_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format_decimal(value, 3)
    return str(value)


def _weather_row_fields(summary: dict[str, object] | None) -> dict[str, str]:
    if not summary:
        return {}
    keys = [
        "weather_available",
        "weather_kind",
        "weather_location_name",
        "weather_hour_count",
        "weather_wet_risk",
        "weather_precipitation_probability_max",
        "weather_precipitation_sum",
        "weather_rain_sum",
        "weather_temperature_2m_mean",
        "weather_wind_speed_10m_mean",
        "weather_wind_gusts_10m_max",
        "weather_cache_hit",
    ]
    return {key: _format_weather_value(summary.get(key)) for key in keys if key in summary}


def _team_match_counts(training_matches: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in training_matches:
        if match.home_goals is None or match.away_goals is None:
            continue
        counts[match.home_team_id] = counts.get(match.home_team_id, 0) + 1
        counts[match.away_team_id] = counts.get(match.away_team_id, 0) + 1
    return counts


def _is_hybrid_eligible(
    *,
    training_matches: list[Any],
    fixtures: list[Any],
    validation_size: int,
    benchmark_available: bool,
    notes: list[str],
) -> bool:
    if not benchmark_available:
        notes.append("Hybrid gate: GBDT indisponible.")
        return False
    if len(training_matches) < HYBRID_MIN_TRAIN_MATCHES:
        notes.append(
            f"Hybrid gate: historique insuffisant ({len(training_matches)}<{HYBRID_MIN_TRAIN_MATCHES})."
        )
        return False
    if validation_size < HYBRID_MIN_VALIDATION_SIZE:
        notes.append(
            f"Hybrid gate: validation insuffisante ({validation_size}<{HYBRID_MIN_VALIDATION_SIZE})."
        )
        return False

    team_counts = _team_match_counts(training_matches)
    for fixture in fixtures:
        home_count = team_counts.get(fixture.home_team_id, 0)
        away_count = team_counts.get(fixture.away_team_id, 0)
        if home_count < HYBRID_MIN_TEAM_MATCHES or away_count < HYBRID_MIN_TEAM_MATCHES:
            notes.append(
                "Hybrid gate: prior team insuffisant "
                f"({fixture.home_team_id}={home_count}, {fixture.away_team_id}={away_count}, "
                f"min={HYBRID_MIN_TEAM_MATCHES})."
            )
            return False
    return True


def run_prediction(config: PredictionConfig) -> PredictionResult:
    notes: list[str] = []
    rows: list[dict[str, str]] = []
    model_policy = _normalize_football_model(config.football_model)
    calibration_policy = _normalize_calibration_policy(config.football_calibration)

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

    weather_summaries: dict[str, dict[str, object]] = {}
    if config.weather_enabled:
        for fixture in fixtures:
            summary, weather_notes = fetch_fixture_weather_summary(
                fixture=fixture,
                cache_root=config.weather_cache_dir or config.cache_dir,
                provider_name=config.weather_provider,
                fallback_latitude=config.weather_latitude,
                fallback_longitude=config.weather_longitude,
                fallback_timezone=config.weather_timezone,
                hours_before=config.weather_hours_before,
                hours_after=config.weather_hours_after,
            )
            notes.extend(weather_notes)
            weather_summaries[fixture.match_id] = summary

    training_matches, training_notes = select_training_matches(dataset, config, fixtures)
    notes.extend(training_notes)

    baseline_model = fit_frequency_baseline(training_matches, notes)
    dixon_coles_model = fit_dixon_coles(training_matches, notes)
    benchmark_model = train_gradient_boosting_benchmark(training_matches, notes)
    history = build_history_from_matches(training_matches)

    if model_policy == "dixon" and calibration_policy == "auto":
        dixon_calibrator = fit_probability_calibrator(training_matches, dixon_coles_model, notes)
    else:
        dixon_calibrator = fit_probability_calibrator_with_policy(
            training_matches,
            dixon_coles_model,
            notes,
            policy=calibration_policy,
        )

    def _predict_dixon_coles_calibrated(match_record: Any) -> tuple[float, float, float]:
        lambda_home_eval, lambda_away_eval = dixon_coles_model.expected_goals(
            match_record.home_team_id, match_record.away_team_id
        )
        raw_eval = outcome_probabilities(
            lambda_home=lambda_home_eval,
            lambda_away=lambda_away_eval,
            rho=dixon_coles_model.rho,
        )
        return dixon_calibrator.apply(raw_eval)

    gbdt_calibrator = fit_probability_calibrator_from_rows(
        probabilities=list(getattr(benchmark_model, "validation_probabilities", []))
        if benchmark_model is not None
        else [],
        labels=list(getattr(benchmark_model, "validation_labels", []))
        if benchmark_model is not None
        else [],
        notes=notes,
        policy=calibration_policy,
        context="gbdt",
    )

    validation_labels = (
        list(benchmark_model.validation_labels) if benchmark_model is not None else []
    )
    validation_pairs = (
        list(benchmark_model.validation_pairs) if benchmark_model is not None else []
    )
    validation_gbdt = (
        [
            gbdt_calibrator.apply((row[0], row[1], row[2]))
            for row in benchmark_model.validation_probabilities
        ]
        if benchmark_model is not None
        else []
    )
    validation_dixon = []
    for home_team_id, away_team_id in validation_pairs:
        lambda_home_eval, lambda_away_eval = dixon_coles_model.expected_goals(
            home_team_id, away_team_id
        )
        raw_eval = outcome_probabilities(
            lambda_home=lambda_home_eval,
            lambda_away=lambda_away_eval,
            rho=dixon_coles_model.rho,
        )
        validation_dixon.append(dixon_calibrator.apply(raw_eval))

    gbdt_enabled = benchmark_model is not None
    hybrid_weight = 0.0
    model_used = model_policy
    calibration_effective = f"dixon:{dixon_calibrator.method}"

    if model_policy == "hybrid":
        eligible = _is_hybrid_eligible(
            training_matches=training_matches,
            fixtures=fixtures,
            validation_size=len(validation_labels),
            benchmark_available=benchmark_model is not None,
            notes=notes,
        )
        if eligible:
            hybrid_weight = select_hybrid_weight(
                labels=validation_labels,
                dc_probabilities=validation_dixon,
                gbdt_probabilities=validation_gbdt,
                notes=notes,
            )
            gbdt_enabled = True
            model_used = "hybrid"
            calibration_effective = f"dixon:{dixon_calibrator.method};gbdt:{gbdt_calibrator.method}"
        else:
            hybrid_weight = 0.0
            gbdt_enabled = False
            model_used = "dixon"
            notes.append("Hybrid ineligible: fallback Dixon-Coles.")
    elif model_policy == "gbdt":
        if benchmark_model is None:
            model_used = "dixon"
            gbdt_enabled = False
            notes.append("GBDT indisponible: fallback Dixon-Coles.")
        else:
            model_used = "gbdt"
            gbdt_enabled = True
            calibration_effective = f"gbdt:{gbdt_calibrator.method}"
    else:
        model_used = "dixon"
        gbdt_enabled = False

    diagnostics: dict[str, Any] = {
        "training_sample_size": len(training_matches),
        "model_used": model_used,
        "gbdt_enabled": bool(gbdt_enabled),
        "hybrid_weight_w": float(hybrid_weight),
        "calibration_method_effective": calibration_effective,
        "weather": {
            "enabled": bool(config.weather_enabled),
            "provider": config.weather_provider,
            "fixture_count": len(weather_summaries),
            "available_count": sum(
                1 for summary in weather_summaries.values() if bool(summary.get("weather_available"))
            ),
        },
        "models": {},
    }
    if config.shadow_eval:
        diagnostics["models"] = {
            "baseline_frequency": evaluate_match_probabilities(training_matches, baseline_model.predict),
            "dixon_coles_calibrated": evaluate_match_probabilities(training_matches, _predict_dixon_coles_calibrated),
        }
        if benchmark_model is not None:
            diagnostics["models"]["gbdt_validation"] = {
                "sample_size": len(validation_labels),
                "log_loss": benchmark_model.log_loss_value,
                "brier": benchmark_model.brier_value,
                "calibration_method": gbdt_calibrator.method,
            }
            if validation_labels and validation_dixon and validation_gbdt:
                blended_probs = [
                    normalize_probabilities(
                        (
                            (hybrid_weight * gbdt[HOME_WIN_CLASS]) + ((1.0 - hybrid_weight) * dc[HOME_WIN_CLASS]),
                            (hybrid_weight * gbdt[DRAW_CLASS]) + ((1.0 - hybrid_weight) * dc[DRAW_CLASS]),
                            (hybrid_weight * gbdt[AWAY_WIN_CLASS]) + ((1.0 - hybrid_weight) * dc[AWAY_WIN_CLASS]),
                        )
                    )
                    for dc, gbdt in zip(validation_dixon, validation_gbdt)
                ]
                if blended_probs:
                    eps = 1e-12
                    log_loss_sum = 0.0
                    for label, probs in zip(validation_labels, blended_probs):
                        log_loss_sum += -math.log(max(eps, min(1.0, probs[label])))
                    diagnostics["models"]["hybrid_validation"] = {
                        "sample_size": len(validation_labels),
                        "log_loss": log_loss_sum / float(len(validation_labels)),
                        "weight": hybrid_weight,
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
        calibrated_home, calibrated_draw, calibrated_away = dixon_calibrator.apply(
            (raw_home, raw_draw, raw_away)
        )
        gbdt_home = calibrated_home
        gbdt_draw = calibrated_draw
        gbdt_away = calibrated_away
        if benchmark_model is not None:
            gbdt_raw = benchmark_model.predict(fixture_feature_vector(fixture, history))
            gbdt_home, gbdt_draw, gbdt_away = gbdt_calibrator.apply(gbdt_raw)

        if model_used == "gbdt" and gbdt_enabled:
            selected = normalize_probabilities((gbdt_home, gbdt_draw, gbdt_away))
        elif model_used == "hybrid" and gbdt_enabled:
            selected = normalize_probabilities(
                (
                    (hybrid_weight * gbdt_home) + ((1.0 - hybrid_weight) * calibrated_home),
                    (hybrid_weight * gbdt_draw) + ((1.0 - hybrid_weight) * calibrated_draw),
                    (hybrid_weight * gbdt_away) + ((1.0 - hybrid_weight) * calibrated_away),
                )
            )
        else:
            selected = normalize_probabilities((calibrated_home, calibrated_draw, calibrated_away))

        baseline_home, baseline_draw, baseline_away = baseline_model.predict(fixture)
        primary_rank = _ranked_outcome_labels(selected)
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
            "home_win_prob": format_probability(selected[HOME_WIN_CLASS]),
            "draw_prob": format_probability(selected[DRAW_CLASS]),
            "away_win_prob": format_probability(selected[AWAY_WIN_CLASS]),
            "predicted_outcome": primary_rank[0],
            "prediction_confidence": format_probability(
                max(selected[HOME_WIN_CLASS], selected[DRAW_CLASS], selected[AWAY_WIN_CLASS])
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
            "calibration_method": dixon_calibrator.method,
            "calibration_method_effective": calibration_effective,
            "primary_model": model_used,
            "model_used": model_used,
            "gbdt_enabled": "true" if gbdt_enabled else "false",
            "hybrid_weight_w": format_decimal(hybrid_weight, 1),
            "baseline_model": "frequency_by_season_and_team",
        }
        row.update(_weather_row_fields(weather_summaries.get(fixture.match_id)))

        if benchmark_model is not None and config.shadow_eval:
            row["p_dc_home_win_prob"] = format_probability(calibrated_home)
            row["p_dc_draw_prob"] = format_probability(calibrated_draw)
            row["p_dc_away_win_prob"] = format_probability(calibrated_away)
            row["p_gbdt_home_win_prob"] = format_probability(gbdt_home)
            row["p_gbdt_draw_prob"] = format_probability(gbdt_draw)
            row["p_gbdt_away_win_prob"] = format_probability(gbdt_away)
            row["gbm_home_win_prob"] = format_probability(gbdt_home)
            row["gbm_draw_prob"] = format_probability(gbdt_draw)
            row["gbm_away_win_prob"] = format_probability(gbdt_away)
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
