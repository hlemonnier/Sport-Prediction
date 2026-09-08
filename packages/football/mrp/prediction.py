"""Prediction orchestration for football match result prediction."""

from __future__ import annotations

from copy import deepcopy
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
from .deployment import full_history_forecast
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
    train_gradient_boosting_model,
)
from .utils import datetime_to_iso, dedupe_preserve_order, format_decimal, format_probability, outcome_class
from .protocol import chronological_populations
from .score_distribution import build_score_distribution
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
    """Predict fixtures, using the verified full-history automatic DC default."""
    return _run_prediction(config, full_history_default=True)


def run_assessment_prediction(config: PredictionConfig) -> PredictionResult:
    """Preserve the original frozen-prefix policy for research replay."""
    return _run_prediction(config, full_history_default=False)


def _run_prediction(config: PredictionConfig, *, full_history_default: bool) -> PredictionResult:
    notes: list[str] = []
    model_policy = _normalize_football_model(config.football_model)
    calibration_policy = _normalize_calibration_policy(config.football_calibration)
    deploy_full_history = (full_history_default and model_policy == "dixon"
        and calibration_policy == "auto" and config.goal_strength_half_life_days is None)
    dataset, data_notes = load_local_football_data(config)
    notes.extend(data_notes)
    fixtures, fixture_notes = select_target_fixtures(dataset, config)
    notes.extend(fixture_notes)
    if not fixtures:
        return PredictionResult(MODEL_VERSION, [], dedupe_preserve_order(notes), {
            "training_sample_size": 0, "models": {}, "status": "unavailable",
            "reason": "No dated target fixtures match the requested population.",
            "maturity": "research_only", "predictive_edge_established": False,
        })
    history_matches, training_notes = select_training_matches(dataset, config, fixtures)
    notes.extend(training_notes)
    populations = chronological_populations(history_matches)
    fit_matches = populations.fit
    baseline = fit_frequency_baseline(fit_matches, notes)
    dixon = fit_dixon_coles(fit_matches, notes,
        half_life_days=config.goal_strength_half_life_days,
        reference_time=min(f.date for f in (populations.calibration or fixtures)))
    gbdt = train_gradient_boosting_model(fit_matches, notes)
    raw_cache: dict[str, tuple[tuple[float, float, float], tuple[float, float, float] | None]] = {}

    def raw_components(record: Any) -> tuple[tuple[float, float, float], tuple[float, float, float] | None]:
        if record.match_id not in raw_cache:
            raw_dc = outcome_probabilities(*dixon.expected_goals(record.home_team_id, record.away_team_id), dixon.rho)
            raw_gbdt = None
            if gbdt is not None:
                history = build_history_from_matches(history_matches, as_of=record.date)
                raw_gbdt = gbdt.predict(fixture_feature_vector(record, history))
            raw_cache[record.match_id] = raw_dc, raw_gbdt
        return raw_cache[record.match_id]

    labels_calibration = [outcome_class(m.home_goals, m.away_goals) for m in populations.calibration]
    if populations.calibration:
        dixon_calibrator = fit_probability_calibrator_with_policy(populations.calibration, dixon, notes, policy=calibration_policy)
    else:
        dixon_calibrator = fit_probability_calibrator_from_rows([], [], notes, policy="off", context="dixon_no_heldout_calibration")
    gbdt_calibrator = fit_probability_calibrator_from_rows(
        [raw_components(m)[1] for m in populations.calibration] if gbdt is not None else [],
        labels_calibration if gbdt is not None else [], notes,
        policy=calibration_policy, context="gbdt_heldout_calibration")

    def calibrated_components(record: Any) -> tuple[tuple[float, float, float], tuple[float, float, float] | None]:
        dc, gb = raw_components(record)
        return dixon_calibrator.apply(dc), gbdt_calibrator.apply(gb) if gb is not None else None

    hybrid_weight = 0.0
    model_used = model_policy
    if model_policy == "hybrid":
        eligible = _is_hybrid_eligible(training_matches=fit_matches, fixtures=fixtures,
            validation_size=len(populations.selection), benchmark_available=gbdt is not None, notes=notes)
        if eligible:
            selection_components = [calibrated_components(m) for m in populations.selection]
            hybrid_weight = select_hybrid_weight(
                [outcome_class(m.home_goals, m.away_goals) for m in populations.selection],
                [p[0] for p in selection_components], [p[1] for p in selection_components], notes)
        else:
            model_used = "dixon"
            notes.append("Hybrid unavailable without an independent selection population; using Dixon-Coles.")
    elif model_policy == "gbdt" and gbdt is None:
        model_used = "dixon"
    gbdt_enabled = model_used in {"gbdt", "hybrid"} and gbdt is not None
    calibration_effective = (f"gbdt:{gbdt_calibrator.method}" if model_used == "gbdt" else
        f"dixon:{dixon_calibrator.method};gbdt:{gbdt_calibrator.method}" if model_used == "hybrid" else
        f"dixon:{dixon_calibrator.method}")

    def selected_probabilities(record: Any) -> tuple[float, float, float]:
        dc, gb = calibrated_components(record)
        if model_used == "gbdt" and gb is not None:
            return gb
        if model_used == "hybrid" and gb is not None:
            return normalize_probabilities(tuple((1-hybrid_weight)*d + hybrid_weight*g for d,g in zip(dc,gb)))
        return dc

    diagnostics: dict[str, Any] = {
        "schema_version": "football_forecast_diagnostics_v2",
        "model_version": MODEL_VERSION,
        "status": "fitted" if fit_matches else "prior_only",
        "maturity": "research_only", "predictive_edge_established": False,
        "training_sample_size": len(fit_matches), "available_history_sample_size": len(history_matches),
        "model_used": model_used, "gbdt_enabled": gbdt_enabled,
        "hybrid_weight_w": hybrid_weight, "calibration_method_effective": calibration_effective,
        "protocol": populations.metadata(), "goal_model_fit": dixon.diagnostics,
        "models": {}, "fixture_distributions": {},
        "expected_goals_semantics": "Mean of the selected joint score distribution; lambda_* output fields are compatibility aliases.",
        "model_comparison_scope": "All reported metrics use the identical untouched outer test population.",
        "historical_xg_policy": "xG is used only with xg_available_at <= forecast cutoff; otherwise final goals are the explicit proxy.",
        "calibration_note": "Classwise calibration followed by normalization is assessed on the final joint forecast; exact multiclass calibration is not assumed.",
        "weather": {"enabled": bool(config.weather_enabled), "provider":config.weather_provider,
                    "fixture_count":0, "available_count":0, "used_by_model":False},
    }
    if config.shadow_eval and populations.test:
        predictors = {"baseline_frequency": baseline.predict,
                      "dixon_coles_raw": lambda m: raw_components(m)[0],
                      "dixon_coles_calibrated": lambda m: calibrated_components(m)[0],
                      "selected_model": selected_probabilities}
        if gbdt is not None:
            predictors["gbdt_raw"] = lambda m: raw_components(m)[1]
            predictors["gbdt_calibrated"] = lambda m: calibrated_components(m)[1]
        for name, predictor in predictors.items():
            metric = evaluate_match_probabilities(populations.test, predictor)
            metric["population_sha256"] = diagnostics["protocol"]["test"]["population_sha256"]
            metric["evaluation_kind"] = "untouched_chronological_test"
            diagnostics["models"][name] = metric
        score_losses = []
        evaluation_rows = []
        for match in populations.test:
            selected = selected_probabilities(match)
            joint = build_score_distribution(*dixon.expected_goals(match.home_team_id, match.away_team_id),dixon.rho,
                min_max_goals=max(match.home_goals,match.away_goals)).reconcile(selected)
            score_p = joint.matrix[match.home_goals][match.away_goals]
            score_losses.append(-math.log(max(1e-12, score_p)))
            evaluation_rows.append({"match_id":match.match_id,"kickoff":match.date.isoformat(),
                "outcome":outcome_class(match.home_goals,match.away_goals),
                "score":[match.home_goals,match.away_goals],
                "selected_probabilities":list(selected),"baseline_probabilities":list(baseline.predict(match)),
                "score_probability":score_p})
        diagnostics["models"]["selected_model"]["scoreline_log_loss"] = sum(score_losses)/len(score_losses)
        diagnostics["evaluation_rows"] = evaluation_rows
        diagnostics["protocol"]["metrics_status"] = "computed"
    else:
        diagnostics["protocol"]["metrics_status"] = "disabled" if not config.shadow_eval else "insufficient_history"
        notes.append("No out-of-sample metrics reported: evaluation disabled or insufficient disjoint history.")

    # Historical metrics above remain on the untouched fit-prefix estimator and
    # held-out calibrator. Only fixture forecasts use the separate full-history
    # estimator; no metric is recomputed on that estimator's training rows.
    fixture_outputs = None
    fixture_calibration_effective = calibration_effective
    if deploy_full_history:
        full = full_history_forecast(history_matches, fixtures,
            cutoff=min(fixture.date for fixture in fixtures))
        fixture_outputs = {state["match_id"]: state for state in full["fixtures"]}
        diagnostics["assessment_model"] = {
            "protocol": deepcopy(diagnostics["protocol"]),
            "goal_model_fit": deepcopy(diagnostics["goal_model_fit"]),
            "calibration_method_effective": calibration_effective,
        }
        diagnostics["schema_version"] = "football_forecast_diagnostics_v3"
        diagnostics["fixture_policy"] = "dc_full_admitted_equal_off"
        diagnostics["fixture_policy_version"] = "full_history_dc_deployment_v1"
        diagnostics["fixture_model"] = {key: deepcopy(value) for key,value in full.items() if key != "fixtures"}
        diagnostics["forecast_training_sample_size"] = len(history_matches)
        fixture_calibration_effective = "dixon:identity"
        diagnostics["fixture_calibration_method_effective"] = fixture_calibration_effective
        diagnostics["calibration_method_effective"] = fixture_calibration_effective
        diagnostics["protocol"]["parameter_policy"] = (
            "Historical assessment uses frozen fit-prefix parameters and held-out calibration; "
            "fixtures use a separate equal-weight model fitted to all admitted history, calibration off.")
        diagnostics["calibration_note"] = (
            "Historical assessment retains its held-out classwise calibration; "
            "the separate full-history Dixon fixture model uses calibration off.")
        notes.extend(full["notes"])
        notes.append("Automatic Dixon fixture policy: full admitted history, equal weights, calibration off; historical assessment remains on its original prefix.")

    rows: list[dict[str, str]] = []
    for fixture in fixtures:
        if fixture_outputs is None:
            lambda_home, lambda_away = dixon.expected_goals(fixture.home_team_id, fixture.away_team_id)
            raw = build_score_distribution(lambda_home,lambda_away,dixon.rho)
            selected = selected_probabilities(fixture)
            joint = raw.reconcile(selected)
            selected = joint.outcome_probabilities
            expected_home,expected_away = joint.expected_goals
            score_home,score_away,scoreline_probability = joint.most_likely_scoreline
            raw_hda = raw.outcome_probabilities
            joint_matrix = joint.matrix
            omitted_mass, tail_bound = joint.omitted_probability_mass, joint.tail_probability_bound
        else:
            state = fixture_outputs[fixture.match_id]
            lambda_home, lambda_away = state["lambda_home"], state["lambda_away"]
            selected, raw_hda = tuple(state["hda"]), tuple(state["raw_hda"])
            expected_home,expected_away = state["expected_goals"]
            score_home,score_away,scoreline_probability = state["mode"]
            joint_matrix = state["matrix"]
            omitted_mass, tail_bound = state["omitted_probability_mass"], state["tail_probability_bound"]
        baseline_probs = baseline.predict(fixture)
        ranked = _ranked_outcome_labels(selected)
        baseline_ranked = _ranked_outcome_labels(baseline_probs)
        row = {
            "mode":config.mode, "match_id":fixture.match_id,
            "date":fixture.date.isoformat(), "league":fixture.league or config.league,
            "season":str(fixture.season if fixture.season is not None else config.season),
            "round":str(fixture.round_number if fixture.round_number is not None else config.round_number),
            "home_team_id":fixture.home_team_id, "away_team_id":fixture.away_team_id,
            "home_team_name":dataset.resolve_team_name(fixture.home_team_id),
            "away_team_name":dataset.resolve_team_name(fixture.away_team_id),
            "lambda_home_goals":format_decimal(expected_home,6),"lambda_away_goals":format_decimal(expected_away,6),
            "expected_home_goals":format_decimal(expected_home,6),"expected_away_goals":format_decimal(expected_away,6),
            "dixon_base_lambda_home_goals":format_decimal(lambda_home,6),"dixon_base_lambda_away_goals":format_decimal(lambda_away,6),
            "home_win_prob":format_decimal(selected[0],12),"draw_prob":format_decimal(selected[1],12),"away_win_prob":format_decimal(selected[2],12),
            "predicted_outcome":ranked[0], "prediction_confidence":format_decimal(max(selected),12),
            "outcome_rank_1":ranked[0],"outcome_rank_2":ranked[1],"outcome_rank_3":ranked[2],
            "raw_home_win_prob":format_decimal(raw_hda[0],12),
            "raw_draw_prob":format_decimal(raw_hda[1],12),
            "raw_away_win_prob":format_decimal(raw_hda[2],12),
            "baseline_home_win_prob":format_decimal(baseline_probs[0],12),
            "baseline_draw_prob":format_decimal(baseline_probs[1],12),
            "baseline_away_win_prob":format_decimal(baseline_probs[2],12),
            "baseline_predicted_outcome":baseline_ranked[0],"baseline_confidence":format_decimal(max(baseline_probs),12),
            "baseline_outcome_rank_1":baseline_ranked[0],"baseline_outcome_rank_2":baseline_ranked[1],"baseline_outcome_rank_3":baseline_ranked[2],
            "predicted_scoreline":f"{score_home}-{score_away}","scoreline_prob":format_decimal(scoreline_probability,12),
            "score_distribution_omitted_mass":str(omitted_mass),
            "score_distribution_tail_error_bound":str(tail_bound),
            "calibration_method":fixture_calibration_effective,"calibration_method_effective":fixture_calibration_effective,
            "primary_model":model_used,"model_used":model_used,"gbdt_enabled":str(gbdt_enabled).lower(),
            "hybrid_weight_w":format_decimal(hybrid_weight,1),"baseline_model":"frequency_by_season_and_team",
            "forecast_status":diagnostics["status"],"maturity":"research_only",
        }
        diagnostics["fixture_distributions"][fixture.match_id] = {
            "score_probability_matrix":[list(r) for r in joint_matrix],
            "outcome_probabilities":list(selected),"expected_goals":[expected_home,expected_away],
            "base_omitted_probability_mass":omitted_mass,
            "reconciled_tail_error_bound":tail_bound,
            "reconciliation":"Preserve conditional score probabilities within each selected 1X2 region.",
        }
        if config.weather_enabled:
            summary,weather_notes = fetch_fixture_weather_summary(fixture=fixture,
                cache_root=config.weather_cache_dir or config.cache_dir,provider_name=config.weather_provider,
                fallback_latitude=config.weather_latitude,fallback_longitude=config.weather_longitude,
                fallback_timezone=config.weather_timezone,hours_before=config.weather_hours_before,hours_after=config.weather_hours_after)
            notes.extend(weather_notes);row.update(_weather_row_fields(summary))
            diagnostics["weather"]["fixture_count"] += 1
            diagnostics["weather"]["available_count"] += int(bool(summary.get("weather_available")))
        rows.append(row)
    notes.append("Research forecasts: executable models and held-out metrics do not establish a deployable betting edge.")
    return PredictionResult(MODEL_VERSION,rows,dedupe_preserve_order(notes),diagnostics)
