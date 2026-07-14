"""Target-specific snapshot baselines for the live F1 platform.

The full ``packages.f1`` live-race model consumes a causal lap-history trace.
The platform contract currently supplies only the latest state per driver, so
claiming that the package model produced these forecasts would be incorrect.
This module instead exposes honest, deterministic snapshot baselines and uses
the canonical package strategy adapter where its input contract is satisfied.

All available position outputs are uncalibrated conditional assignment
marginals over the target-eligible field. A score-magnitude kernel is
Sinkhorn-balanced so every available driver row and every eligible-position
column sums to one. Participation, classification eligibility, and next-lap
eligibility are deliberately separate: a retired car can remain race-classified
while having no next-lap forecast or strategy recommendation.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]

MODEL_VERSION = "f1_snapshot_target_dispatch_v4"
FEATURES_VERSION = "platform_target_specific_snapshot_v4"

POSITION_SEMANTICS = {
    "race": "conditional_classification_order_given_latest_snapshot_and_observed_participation_status",
    "qualifying": "conditional_qualifying_order_given_observed_snapshot_pace_and_participation_status",
    "next-lap": "conditional_next_lap_relative_order_given_observed_snapshot_pace_and_running_status",
    "strategy": "current_classification_context_only_not_a_finish_forecast",
}
DNF_SEMANTICS = "observed_retired_or_stopped_indicator_not_a_calibrated_future_retirement_probability"
CALIBRATION_STATUS = "uncalibrated_heuristic_not_validated_for_promotion"

TARGET_MODELS: dict[str, dict[str, str]] = {
    "race": {
        "model_version": "live_race_snapshot_joint_heuristic_v4",
        "features_version": "platform_live_race_snapshot_v4",
        "target_definition": POSITION_SEMANTICS["race"],
    },
    "qualifying": {
        "model_version": "qualifying_snapshot_pace_heuristic_v3",
        "features_version": "platform_qualifying_pace_snapshot_v3",
        "target_definition": POSITION_SEMANTICS["qualifying"],
    },
    "next-lap": {
        "model_version": "next_lap_snapshot_pace_heuristic_v3",
        "features_version": "platform_next_lap_pace_snapshot_v3",
        "target_definition": POSITION_SEMANTICS["next-lap"],
    },
    "strategy": {
        "model_version": "live_strategy_legal_mask_heuristic_v3",
        "features_version": "platform_live_strategy_legality_snapshot_v3",
        "target_definition": POSITION_SEMANTICS["strategy"],
    },
}


def predict_from_snapshot(snapshot: JsonObject, *, prediction_kind: str = "race") -> JsonObject:
    prediction_kind = str(prediction_kind).strip().lower().replace("_", "-")
    target = TARGET_MODELS.get(prediction_kind)
    if target is None:
        supported = ", ".join(sorted(TARGET_MODELS))
        raise ValueError(f"unsupported prediction_kind {prediction_kind!r}; expected one of: {supported}")

    drivers = _drivers(snapshot)
    driver_numbers = [int(driver["driver_number"]) for driver in drivers]
    if len(set(driver_numbers)) != len(driver_numbers):
        raise ValueError("snapshot contains duplicate driver_number values")
    generated_at = _utc_now()
    if not drivers:
        return {
            "modelVersion": target["model_version"],
            "featuresVersion": target["features_version"],
            "generatedAt": generated_at,
            "predictionKind": prediction_kind,
            "predictions": [],
            "diagnostics": {
                "driverCount": 0,
                "targetDefinition": target["target_definition"],
                "positionSemantics": POSITION_SEMANTICS[prediction_kind],
                "dnfSemantics": DNF_SEMANTICS,
                "calibrationStatus": CALIBRATION_STATUS,
                "promotionStatus": "not_promoted",
                "strategyPolicyEnabled": False,
                "forecastAvailable": False,
                "unavailableReason": "snapshot_has_no_drivers",
                "provenance": {
                    "modelType": "deterministic_uncalibrated_snapshot_heuristic",
                    "canonicalLiveRaceModelUsed": False,
                    "canonicalLiveRaceUnavailableReason": "snapshot_has_no_drivers",
                    "promotionEligible": False,
                },
            },
        }

    strategy_requested = prediction_kind in {"race", "next-lap", "strategy"}
    if strategy_requested:
        strategy_rows, strategy_enabled, strategy_error = _strategy_policy_rows(drivers)
    else:
        strategy_rows, strategy_enabled, strategy_error = {}, False, None

    scored = _score_drivers(drivers, strategy_rows, prediction_kind=prediction_kind)
    total = len(scored)
    eligible_scored = [
        row
        for row in scored
        if _target_eligibility(row[1], prediction_kind)[0]
    ]
    eligible_total = len(eligible_scored)
    distributions, joint_diagnostics = _joint_position_distributions(
        eligible_scored,
        prediction_kind=prediction_kind,
        strategy_enabled=strategy_enabled,
        total_driver_count=total,
    )
    predictions = []
    for score, driver, policy in scored:
        driver_number = int(driver["driver_number"])
        forecast_available, eligibility_status, unavailable_reason = _target_eligibility(
            driver,
            prediction_kind,
        )
        distribution = distributions.get(driver_number, {})
        position_p10 = _position_percentile(distribution, 0.10) if forecast_available else None
        position_p90 = _position_percentile(distribution, 0.90) if forecast_available else None
        if not forecast_available:
            win_probability = 0.0
            podium_probability = 0.0
            points_probability = 0.0
        else:
            win_probability = distribution.get("1", 0.0)
            podium_probability = sum(
                distribution.get(str(pos), 0.0)
                for pos in range(1, min(3, eligible_total) + 1)
            )
            points_probability = sum(
                distribution.get(str(pos), 0.0)
                for pos in range(1, min(10, eligible_total) + 1)
            )
        expected_position = (
            sum(float(position) * probability for position, probability in distribution.items())
            if forecast_available
            else None
        )
        confidence = (
            _confidence(driver, policy, strategy_enabled, prediction_kind=prediction_kind)
            if forecast_available
            else 0.0
        )
        strategy_payload = _strategy_payload(policy) if strategy_requested else None
        predictions.append(
            {
                "model_version": target["model_version"],
                "prediction_time": generated_at,
                "source_event_sequence": _optional_int(snapshot.get("seq")) or 0,
                "features_version": target["features_version"],
                "driver_number": driver_number,
                "expected_position": round(expected_position, 6) if expected_position is not None else None,
                "position_p10": position_p10,
                "position_p90": position_p90,
                "position_distribution": distribution,
                "win_probability": round(win_probability, 12),
                "podium_probability": round(podium_probability, 12),
                "points_probability": round(points_probability, 12),
                "dnf_probability": _dnf_status_indicator(driver)
                if prediction_kind in {"race", "strategy"}
                else 0.0,
                "dnf_semantics": DNF_SEMANTICS,
                "confidence": round(confidence, 6),
                "strategy": strategy_payload,
                "score": round(score, 6),
                "position_semantics": POSITION_SEMANTICS[prediction_kind],
                "forecast_available": forecast_available,
                "unavailable_reason": unavailable_reason,
                "eligibility_status": eligibility_status,
                "participation_status": driver["participation_status"],
            }
        )

    joint_diagnostics.update(
        {
            "winProbabilitySum": sum(row["win_probability"] for row in predictions),
            "podiumProbabilitySum": sum(row["podium_probability"] for row in predictions),
            "pointsProbabilitySum": sum(row["points_probability"] for row in predictions),
        }
    )
    strategy_available_count = sum(
        1 for row in predictions if row["strategy"] and row["strategy"].get("safeToRecommend") is True
    )
    strategy_unavailable_count = sum(
        1 for row in predictions if row["strategy"] and row["strategy"].get("safeToRecommend") is False
    )
    fallback_strategy_used = any(
        str(policy.get("policy_version") or "").startswith("fallback_strategy_policy")
        for _, _, policy in scored
    )

    return {
        "modelVersion": target["model_version"],
        "featuresVersion": target["features_version"],
        "generatedAt": generated_at,
        "predictionKind": prediction_kind,
        "predictions": predictions,
        "diagnostics": {
            "driverCount": total,
            "targetDefinition": target["target_definition"],
            "positionSemantics": POSITION_SEMANTICS[prediction_kind],
            "dnfSemantics": DNF_SEMANTICS,
            "calibrationStatus": CALIBRATION_STATUS,
            "promotionStatus": "not_promoted",
            "forecastAvailable": bool(eligible_total),
            "forecastAvailableCount": eligible_total,
            "forecastUnavailableCount": total - eligible_total,
            "strategyPolicyEnabled": strategy_enabled,
            "strategyPolicyError": strategy_error,
            "strategyRecommendationAvailableCount": strategy_available_count,
            "strategyRecommendationUnavailableCount": strategy_unavailable_count,
            "strategySafetyContract": "shared_legal_action_mask_or_explicit_unavailable",
            "sourceEventSequence": _optional_int(snapshot.get("seq")) or 0,
            "jointDistribution": joint_diagnostics,
            "provenance": {
                "modelType": "deterministic_uncalibrated_snapshot_heuristic",
                "canonicalPackageComponents": (
                    ["packages.f1.models.live_race.strategy.BaselineStrategyPolicyAdapter"]
                    if strategy_enabled
                    else []
                ),
                "canonicalLiveRaceModelUsed": False,
                "canonicalLiveRaceUnavailableReason": (
                    "run_live_race_prediction_requires_a_causal_lap_history_trace; "
                    "the_platform_request_contains_only_the_latest_driver_state"
                ),
                "fallbackStrategyPolicyUsed": fallback_strategy_used,
                "promotionEligible": False,
            },
        },
    }


def _target_eligibility(
    driver: JsonObject,
    prediction_kind: str,
) -> tuple[bool, str, str | None]:
    """Resolve availability for the requested target, not for the car globally."""

    participation = str(driver.get("participation_status") or "running_or_unknown")
    if prediction_kind == "strategy":
        return False, "target_unavailable", "strategy_has_no_position_forecast"

    if participation in {"dns", "dsq", "withdrawn", "classification_ineligible"}:
        reason_status = "unclassified" if participation == "classification_ineligible" else participation
        return (
            False,
            "target_unavailable",
            f"classification_ineligible:{reason_status}",
        )

    if prediction_kind == "next-lap" and participation != "running_or_unknown":
        return (
            False,
            "target_unavailable",
            f"next_lap_unavailable:{participation}",
        )

    if (
        prediction_kind == "qualifying"
        and participation == "retired_or_stopped"
        and _lap_signal(driver, prefer_best=True) is None
    ):
        return False, "target_unavailable", "qualifying_unavailable:no_valid_lap"

    if participation == "retired_or_stopped":
        return True, "classification_eligible_retired", None
    return True, "classification_eligible", None


def _score_drivers(
    drivers: list[JsonObject],
    strategy_rows: dict[int, JsonObject],
    *,
    prediction_kind: str,
) -> list[tuple[float, JsonObject, JsonObject]]:
    lap_values = [
        value
        for driver in drivers
        if _target_eligibility(driver, prediction_kind)[0]
        for value in [_lap_signal(driver, prefer_best=prediction_kind == "qualifying")]
        if value is not None
    ]
    slow_lap = max(lap_values, default=120.0) + 5.0
    scored: list[tuple[float, JsonObject, JsonObject]] = []
    for driver in drivers:
        number = int(driver["driver_number"])
        policy = strategy_rows.get(number, {})
        if prediction_kind == "race":
            score = _race_score(driver, policy)
        elif prediction_kind == "qualifying":
            lap = _lap_signal(driver, prefer_best=True)
            score = lap if lap is not None else slow_lap + (0.01 * float(driver.get("position") or 20.0))
        elif prediction_kind == "next-lap":
            lap = _lap_signal(driver, prefer_best=False)
            score = (lap if lap is not None else slow_lap) + _next_lap_adjustment(driver, policy)
        else:  # strategy: position is context, not a finishing-order forecast.
            score = float(driver.get("position") or 20.0)
        scored.append((float(score), driver, policy))
    scored.sort(
        key=lambda item: (
            not _target_eligibility(item[1], prediction_kind)[0],
            item[0],
            int(item[1]["driver_number"]),
        )
    )
    return scored


def _known_not_running(driver: JsonObject) -> bool:
    return str(driver.get("participation_status") or "running_or_unknown") != "running_or_unknown"


def _lap_signal(driver: JsonObject, *, prefer_best: bool) -> float | None:
    first = "best_lap_time" if prefer_best else "last_lap_time"
    second = "last_lap_time" if prefer_best else "best_lap_time"
    return _optional_float(driver.get(first)) or _optional_float(driver.get(second))


def _race_score(driver: JsonObject, policy: JsonObject) -> float:
    """Latest-state race baseline; deliberately not presented as a trained model."""

    position = _optional_float(driver.get("position")) or 20.0
    gap = _optional_float(driver.get("gap_to_leader_seconds")) or 0.0
    tyre_used = _optional_float(policy.get("tyre_life_used_ratio")) or 0.0
    pit_urgency = _optional_float(policy.get("pit_urgency")) or 0.0
    return (
        float(position)
        + min(2.5, max(0.0, gap) * 0.02)
        + min(0.8, tyre_used * 0.30)
        + min(1.2, pit_urgency * 0.65)
        + (0.35 if str(policy.get("recommended_action")) == "pit_now" else 0.0)
    )


def _next_lap_adjustment(driver: JsonObject, policy: JsonObject) -> float:
    compound = str(driver.get("compound") or "UNKNOWN").upper()
    tyre_age = max(0.0, _optional_float(driver.get("tyre_age")) or 0.0)
    degradation = {
        "SOFT": 0.050,
        "MEDIUM": 0.040,
        "HARD": 0.030,
        "INTERMEDIATE": 0.060,
        "WET": 0.065,
    }.get(compound, 0.040)
    adjustment = degradation * min(50.0, tyre_age)
    action = str(policy.get("recommended_action") or "")
    if action == "pit_now":
        adjustment += 20.0
    elif action == "pit_next_lap":
        adjustment += 0.2
    return float(adjustment)


def _joint_position_distributions(
    scored: list[tuple[float, JsonObject, JsonObject]],
    *,
    prediction_kind: str,
    strategy_enabled: bool,
    total_driver_count: int,
) -> tuple[dict[int, dict[str, float]], JsonObject]:
    """Return coherent marginals over only the target-eligible field."""

    total = len(scored)
    if not scored:
        return {}, {
            "method": "sinkhorn_balanced_score_position_kernel",
            "iterations": 0,
            "rowMaxAbsError": 0.0,
            "columnMaxAbsError": 0.0,
            "magnitudeSensitive": True,
            "conditionalPositionSemantics": POSITION_SEMANTICS[prediction_kind],
            "eligibleDriverCount": 0,
            "unavailableDriverCount": total_driver_count,
            "classificationPositions": [],
            "groupKernels": [],
            "winProbabilitySum": 0.0,
            "podiumProbabilitySum": 0.0,
            "pointsProbabilitySum": 0.0,
        }

    matrix, iterations, kernel_diagnostics = _score_position_kernel(
        scored,
        prediction_kind=prediction_kind,
        strategy_enabled=strategy_enabled,
    )

    distributions: dict[int, dict[str, float]] = {}
    for row_index, (_, driver, _) in enumerate(scored):
        distributions[int(driver["driver_number"])] = {
            str(position): round(matrix[row_index][position - 1], 12)
            for position in range(1, total + 1)
        }

    row_sums = [sum(row.values()) for row in distributions.values()]
    column_sums = [
        sum(row[str(position)] for row in distributions.values())
        for position in range(1, total + 1)
    ]
    return distributions, {
        "method": "sinkhorn_balanced_score_position_kernel",
        "iterations": iterations,
        "rowMaxAbsError": max(abs(value - 1.0) for value in row_sums),
        "columnMaxAbsError": max(abs(value - 1.0) for value in column_sums),
        "magnitudeSensitive": True,
        "conditionalPositionSemantics": POSITION_SEMANTICS[prediction_kind],
        "eligibleDriverCount": total,
        "unavailableDriverCount": total_driver_count - total,
        "classificationPositions": list(range(1, total + 1)),
        "groupKernels": [
            {
                "name": "target_eligible_field",
                "driverCount": total,
                "classificationPositions": list(range(1, total + 1)),
                **kernel_diagnostics,
            }
        ],
        "winProbabilitySum": column_sums[0],
        "podiumProbabilitySum": sum(column_sums[: min(3, total)]),
        "pointsProbabilitySum": sum(column_sums[: min(10, total)]),
    }


def _score_position_kernel(
    rows: list[tuple[float, JsonObject, JsonObject]],
    *,
    prediction_kind: str,
    strategy_enabled: bool,
) -> tuple[list[list[float]], int, JsonObject]:
    count = len(rows)
    scores = [float(score) for score, _, _ in rows]
    sorted_scores = sorted(scores)
    midpoint = count // 2
    if count % 2:
        center = sorted_scores[midpoint]
    else:
        center = 0.5 * (sorted_scores[midpoint - 1] + sorted_scores[midpoint])
    scale = _score_magnitude_scale(prediction_kind)
    denominator = max(1.0, float(count - 1) / 2.0)
    position_axis = [
        (float(position) - (float(count) + 1.0) / 2.0) / denominator
        for position in range(1, count + 1)
    ]

    matrix: list[list[float]] = []
    clipped_coordinates: list[float] = []
    group_uncertainty = sum(
        _uncertainty(driver, policy, strategy_enabled, prediction_kind=prediction_kind)
        for _, driver, policy in rows
    ) / float(count)
    for score, _, _ in rows:
        coordinate = max(-4.0, min(4.0, (float(score) - center) / scale))
        clipped_coordinates.append(coordinate)
        strength = (2.0 * coordinate) / max(0.75, group_uncertainty)
        matrix.append(
            [
                math.exp(max(-6.0, min(6.0, strength * axis)))
                for axis in position_axis
            ]
        )

    balanced, iterations = _sinkhorn_balance(matrix)
    return balanced, iterations, {
        "scoreScale": scale,
        "scoreCenter": center,
        "scoreSpan": max(scores) - min(scores),
        "maxAbsScaledScore": max(abs(value) for value in clipped_coordinates),
        "groupUncertainty": group_uncertainty,
    }


def _sinkhorn_balance(matrix: list[list[float]]) -> tuple[list[list[float]], int]:
    count = len(matrix)
    if count <= 1:
        return [[1.0]], 0
    iterations = 0
    for iterations in range(1, 1001):
        for row_index in range(count):
            row_sum = sum(matrix[row_index]) or 1.0
            matrix[row_index] = [value / row_sum for value in matrix[row_index]]
        for column_index in range(count):
            column_sum = sum(matrix[row][column_index] for row in range(count)) or 1.0
            for row_index in range(count):
                matrix[row_index][column_index] /= column_sum
        row_error = max(abs(sum(row) - 1.0) for row in matrix)
        column_error = max(
            abs(sum(matrix[row][column] for row in range(count)) - 1.0)
            for column in range(count)
        )
        if max(row_error, column_error) <= 1e-12:
            break
    return matrix, iterations


def _score_magnitude_scale(prediction_kind: str) -> float:
    return {
        "race": 1.50,
        "qualifying": 0.25,
        "next-lap": 0.35,
        "strategy": 1.50,
    }[prediction_kind]


def _drivers(snapshot: JsonObject) -> list[JsonObject]:
    raw = snapshot.get("drivers") or snapshot.get("Drivers") or []
    drivers = [item for item in raw if isinstance(item, dict)]
    normalized = []
    for item in drivers:
        number = _optional_int(item.get("driver_number", item.get("driverNumber")))
        if number is None:
            continue
        result = _session_result_for_driver(snapshot, number)
        current_lap = _optional_float(
            _first_present(
                [item, result],
                ("current_lap", "currentLap", "number_of_laps", "numberOfLaps"),
            )
        )
        compound = _first_present(
            [item],
            ("current_compound", "currentCompound", "compound", "Compound"),
        ) or "UNKNOWN"
        pit_status = _first_present([item, result], ("pit_status", "pitStatus", "status"))
        participation_status = _participation_status_payload(item, result, pit_status)
        track_status = _first_present([item], ("track_status", "trackStatus"))
        if track_status is None:
            track_status = _track_status_from_snapshot(snapshot)
        strategy_context = _strategy_context(
            snapshot,
            item,
            driver_number=number,
            current_lap=current_lap,
            compound=str(compound),
            pit_status=pit_status,
            track_status=str(track_status or ""),
        )
        normalized.append(
            {
                "driver_number": number,
                "driver_id": str(item.get("acronym") or item.get("driver_number") or number),
                "driver_name": str(item.get("full_name") or item.get("acronym") or number),
                "team_name": item.get("team_name"),
                "position": _optional_float(
                    _first_present([item, result], ("position", "Position"))
                ),
                "current_lap": current_lap,
                "last_lap_time": _optional_float(
                    _first_present([item], ("last_lap_time", "lastLapTime"))
                ),
                "best_lap_time": _optional_float(
                    _first_present([item], ("best_lap_time", "bestLapTime"))
                ),
                "compound": compound,
                "tyre_age": _optional_float(
                    _first_present([item], ("tyre_age", "tyreAge"))
                ),
                "track_status": track_status,
                "gap_to_leader_seconds": _gap_seconds(item.get("gap_to_leader") or item.get("interval")),
                "last_speed": _optional_float(item.get("last_speed")),
                "pit_status": pit_status,
                "participation_status": participation_status,
                "is_not_running": participation_status != "running_or_unknown",
                **strategy_context,
            }
        )
    return normalized


def _strategy_context(
    snapshot: JsonObject,
    item: JsonObject,
    *,
    driver_number: int,
    current_lap: float | None,
    compound: str,
    pit_status: Any,
    track_status: str,
) -> JsonObject:
    session_info = snapshot.get("sessionInfo") or snapshot.get("session_info") or {}
    if not isinstance(session_info, dict):
        session_info = {}
    strategy_context = snapshot.get("strategyContext") or snapshot.get("strategy_context") or {}
    if not isinstance(strategy_context, dict):
        strategy_context = {}
    shared = [item, strategy_context, snapshot, session_info]

    total_laps = _optional_int(
        _first_present(shared, ("total_laps", "totalLaps", "race_total_laps", "scheduled_laps"))
    )
    remaining_laps = _optional_int(
        _first_present(shared, ("remaining_laps", "remainingLaps", "laps_remaining"))
    )
    stint_number = _optional_int(
        _first_present([item], ("stint_number", "stintNumber", "stint_id", "Stint"))
    )
    timeline_rows = _driver_strategy_timeline(snapshot, driver_number)
    if stint_number is None and timeline_rows:
        stint_number = max(
            _optional_int(row.get("stint_number", row.get("stintNumber"))) or 0
            for row in timeline_rows
        )

    used_raw = _first_present([item, strategy_context], ("used_compounds", "usedCompounds", "compounds_used"))
    used_compounds = _compound_list(used_raw)
    if not used_compounds and timeline_rows:
        used_compounds = _compound_list(
            [row.get("compound") for row in timeline_rows if row.get("compound")]
        )
    if not used_compounds and stint_number == 1 and str(compound).upper() != "UNKNOWN":
        used_compounds = [str(compound)]

    available_raw = _first_present(
        shared,
        ("available_compounds", "availableCompounds", "allowed_compounds", "allowedCompounds"),
    )
    pit_lane_open = _optional_bool(
        _first_present(shared, ("pit_lane_open", "pitLaneOpen"))
    )
    if pit_lane_open is None:
        pit_lane_open = _pit_lane_state_from_race_control(snapshot)

    is_red = _optional_bool(_first_present(shared, ("is_red", "isRed")))
    if is_red is None:
        is_red = _red_state_from_track_status(track_status)

    is_wet = _optional_bool(
        _first_present(
            shared,
            ("is_wet_track", "isWetTrack", "weather_is_wet", "rain_expected"),
        )
    )
    if is_wet is None:
        is_wet = _wet_state_from_weather(snapshot, compound)

    is_box_lap = _optional_bool(_first_present([item], ("is_box_lap", "isBoxLap")))
    if is_box_lap is None:
        is_box_lap = _box_state_from_pit_status(pit_status)

    return {
        "total_laps": total_laps,
        "remaining_laps": remaining_laps,
        "stint_number": stint_number,
        "used_compounds": used_compounds,
        "available_compounds": _compound_list(available_raw),
        "pit_lane_open": pit_lane_open,
        "is_red": is_red,
        "is_wet_track": is_wet,
        "is_box_lap": is_box_lap,
    }


def _first_present(mappings: list[JsonObject], keys: tuple[str, ...]) -> Any:
    for mapping in mappings:
        if not isinstance(mapping, dict):
            continue
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
    return None


def _session_result_for_driver(snapshot: JsonObject, driver_number: int) -> JsonObject:
    rows = snapshot.get("sessionResults") or snapshot.get("session_results") or []
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        number = _optional_int(row.get("driver_number", row.get("driverNumber")))
        if number == driver_number:
            return row
    return {}


def _driver_strategy_timeline(snapshot: JsonObject, driver_number: int) -> list[JsonObject]:
    rows = snapshot.get("strategyTimeline") or snapshot.get("strategy_timeline") or []
    if not isinstance(rows, list):
        return []
    return [
        row
        for row in rows
        if isinstance(row, dict)
        and _optional_int(row.get("driver_number", row.get("driverNumber"))) == driver_number
    ]


def _compound_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = [part.strip() for part in value.replace("|", ",").split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw = [str(part).strip() for part in value]
    else:
        raw = [str(value).strip()]
    return list(dict.fromkeys(part for part in raw if part))


def _participation_status_payload(
    item: JsonObject,
    result: JsonObject,
    pit_status: Any,
) -> str:
    """Normalize observed participation without deciding target eligibility."""

    running = _optional_bool(
        _first_present([item, result], ("is_running", "isRunning", "is_active", "isActive"))
    )
    explicit_flags = {
        key: _optional_bool(_first_present([item, result], (key,))) is True
        for key in (
            "dnf",
            "dns",
            "dsq",
            "disqualified",
            "excluded",
            "retired",
            "stopped",
            "not_running",
            "withdrawn",
            "unclassified",
        )
    }
    explicit_flags["dns"] = explicit_flags["dns"] or _optional_bool(
        _first_present([item, result], ("did_not_start", "didNotStart"))
    ) is True
    explicit_flags["withdrawn"] = explicit_flags["withdrawn"] or _optional_bool(
        _first_present([item, result], ("excluded",))
    ) is True
    explicit_flags["unclassified"] = explicit_flags["unclassified"] or _optional_bool(
        _first_present([item, result], ("not_classified", "notClassified"))
    ) is True
    classified = _optional_bool(
        _first_present([item, result], ("classified", "is_classified", "isClassified"))
    )
    status_values = [
        pit_status,
        _first_present([item, result], ("status", "classification_status", "classificationStatus")),
    ]
    normalized_statuses = {
        str(value or "").strip().lower().replace("_", " ")
        for value in status_values
        if str(value or "").strip()
    }
    text = " ".join(str(value or "").strip().lower().replace("_", " ") for value in status_values)

    if explicit_flags["dsq"] or explicit_flags["disqualified"] or explicit_flags["excluded"] or any(
        marker in text for marker in ("disqualified", "excluded from classification")
    ) or "dsq" in normalized_statuses:
        return "dsq"
    if explicit_flags["dns"] or any(
        marker in text for marker in ("did not start", "didn't start")
    ) or "dns" in normalized_statuses:
        return "dns"
    if explicit_flags["withdrawn"] or any(
        marker in text for marker in ("withdrawn", "withdrew")
    ):
        return "withdrawn"
    if explicit_flags["unclassified"] or classified is False or any(
        marker in text for marker in ("not classified", "unclassified")
    ) or "nc" in normalized_statuses:
        return "classification_ineligible"
    if (
        explicit_flags["dnf"]
        or explicit_flags["retired"]
        or explicit_flags["stopped"]
        or explicit_flags["not_running"]
        or normalized_statuses
        & {"out", "inactive", "retired", "dnf", "stopped", "not running"}
        or any(
            marker in text
            for marker in ("retired", "did not finish", "stopped", "not running")
        )
    ):
        return "retired_or_stopped"
    if "classified" in normalized_statuses or any(
        "finished" in status or (status.startswith("+") and "lap" in status)
        for status in normalized_statuses
    ):
        return "finished"
    if running is False:
        return "retired_or_stopped"
    return "running_or_unknown"


def _pit_lane_state_from_race_control(snapshot: JsonObject) -> bool | None:
    messages = snapshot.get("raceControl") or snapshot.get("race_control") or []
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        text = " ".join(str(value).lower() for value in message.values())
        if "pit lane closed" in text:
            return False
        if "pit lane open" in text:
            return True
    return None


def _red_state_from_track_status(track_status: str) -> bool | None:
    text = str(track_status or "").strip().lower().replace("_", " ")
    codes = {char for char in text if char.isdigit()}
    if codes:
        return "5" in codes
    if "red" in text:
        return True
    if any(marker in text for marker in ("green", "clear", "yellow", "safety car", "vsc")):
        return False
    return None


def _wet_state_from_weather(snapshot: JsonObject, _compound: str) -> bool | None:
    weather = snapshot.get("weather") or {}
    if not isinstance(weather, dict):
        weather = {}
    rainfall = _optional_float(weather.get("rainfall"))
    if rainfall is not None:
        return rainfall > 0.0
    return None


def _box_state_from_pit_status(pit_status: Any) -> bool | None:
    if pit_status is None or not str(pit_status).strip():
        return None
    text = str(pit_status).strip().lower().replace("_", " ")
    if (
        text == "pit"
        or text.startswith("pit ")
        or any(marker in text for marker in ("box", "pit lane", "in pit", "pitting"))
    ):
        return True
    return False


def _strategy_policy_rows(drivers: list[JsonObject]) -> tuple[dict[int, JsonObject], bool, str | None]:
    rows: dict[int, JsonObject] = {}
    legality_states: dict[int, JsonObject] = {}
    eligible_drivers: list[JsonObject] = []
    for driver in drivers:
        number = int(driver["driver_number"])
        legality_state, missing = _strategy_legality_state(driver)
        if missing:
            reason = "known_not_running" if _known_not_running(driver) else "missing_critical_legality_state"
            rows[number] = _unavailable_strategy_row(
                reason,
                missing_fields=missing,
                legality_state=legality_state,
            )
            continue
        legality_states[number] = legality_state
        eligible_drivers.append(driver)

    if not eligible_drivers:
        return rows, False, None

    adapter_enabled = True
    strategy_error: str | None = None
    try:
        import pandas as pd
        BaselineStrategyPolicyAdapter = _load_strategy_policy_adapter()
    except Exception as exc:
        fallback = _fallback_strategy_rows(eligible_drivers)
        strategy_error = f"{type(exc).__name__}: {exc}"
        adapter_enabled = False
    else:
        frame = pd.DataFrame(
            [
                {
                    "driver_id": driver["driver_id"],
                    "driver_name": driver["driver_name"],
                    "lap_number": legality_states[int(driver["driver_number"])]["lap_number"],
                    "total_laps": legality_states[int(driver["driver_number"])]["total_laps"],
                    "remaining_laps": legality_states[int(driver["driver_number"])]["remaining_laps"],
                    "stint_id": legality_states[int(driver["driver_number"])]["stint_id"],
                    "compound": driver["compound"],
                    "used_compounds": legality_states[int(driver["driver_number"])]["used_compounds"],
                    "available_compounds": legality_states[int(driver["driver_number"])]["available_compounds"],
                    "tyre_age": legality_states[int(driver["driver_number"])]["tyre_age"],
                    "track_status": driver["track_status"],
                    "is_red": legality_states[int(driver["driver_number"])]["is_red"],
                    "is_wet_track": legality_states[int(driver["driver_number"])]["is_wet_track"],
                    "is_box_lap": legality_states[int(driver["driver_number"])]["is_box_lap"],
                    "pit_lane_open": legality_states[int(driver["driver_number"])]["pit_lane_open"],
                    "gap_to_leader_seconds": driver["gap_to_leader_seconds"],
                    "next_lap_mean": driver["last_lap_time"] or driver["best_lap_time"],
                    "position": driver["position"],
                    "driver_number": driver["driver_number"],
                }
                for driver in eligible_drivers
            ]
        )
        try:
            actions = BaselineStrategyPolicyAdapter().evaluate_actions(frame)
        except Exception as exc:
            fallback = _fallback_strategy_rows(eligible_drivers)
            strategy_error = f"{type(exc).__name__}: {exc}"
            adapter_enabled = False
        else:
            fallback = {}
            for idx, driver in enumerate(eligible_drivers):
                row = actions.iloc[idx].to_dict() if idx < len(actions) else {}
                fallback[int(driver["driver_number"])] = _json_safe(row)

    for driver in eligible_drivers:
        number = int(driver["driver_number"])
        policy = fallback.get(number, {})
        rows[number] = _apply_shared_legal_action_mask(policy, legality_states[number])
    return rows, adapter_enabled, strategy_error


def _strategy_legality_state(driver: JsonObject) -> tuple[JsonObject, list[str]]:
    missing: list[str] = []
    if _known_not_running(driver):
        missing.append("known_not_running")
    current_lap = _optional_int(driver.get("current_lap"))
    total_laps = _optional_int(driver.get("total_laps"))
    remaining_laps = _optional_int(driver.get("remaining_laps"))
    tyre_age = _optional_int(driver.get("tyre_age"))
    if current_lap is None or current_lap < 0:
        missing.append("current_lap")
    if total_laps is None or total_laps <= 0:
        missing.append("total_laps")
    if remaining_laps is None or remaining_laps < 0:
        missing.append("remaining_laps")
    if tyre_age is None or tyre_age < 0:
        missing.append("tyre_age")
    stint_number = _optional_int(driver.get("stint_number"))
    if stint_number is None or stint_number < 1:
        missing.append("stint_number")
    compound = _normalize_compound(driver.get("compound"))
    if compound == "UNKNOWN":
        missing.append("current_compound")
    raw_used_compounds = _compound_list(driver.get("used_compounds"))
    used_compounds = [_normalize_compound(value) for value in raw_used_compounds]
    if not used_compounds or "UNKNOWN" in used_compounds:
        missing.append("used_compounds")
    raw_available_compounds = _compound_list(driver.get("available_compounds"))
    available_compounds = [_normalize_compound(value) for value in raw_available_compounds]
    if not available_compounds or "UNKNOWN" in available_compounds:
        missing.append("available_compounds")
    pit_lane_open = _optional_bool(driver.get("pit_lane_open"))
    if pit_lane_open is None:
        missing.append("pit_lane_open")
    is_red = _optional_bool(driver.get("is_red"))
    if is_red is None:
        missing.append("red_flag_state")
    is_wet_track = _optional_bool(driver.get("is_wet_track"))
    if is_wet_track is None:
        missing.append("wet_track_state")
    is_box_lap = _optional_bool(driver.get("is_box_lap"))
    if is_box_lap is None:
        missing.append("box_or_pit_state")
    return {
        "lap_number": current_lap,
        "total_laps": total_laps,
        "remaining_laps": remaining_laps,
        "stint_id": stint_number,
        "compound": compound,
        "tyre_age": tyre_age,
        "used_compounds": used_compounds,
        "available_compounds": available_compounds,
        "pit_lane_open": pit_lane_open,
        "is_red": is_red,
        "is_wet_track": is_wet_track,
        "weather_is_wet": is_wet_track,
        "is_box_lap": is_box_lap,
        "track_status": str(driver.get("track_status") or ""),
        "metadata": {"available_compounds": available_compounds},
    }, missing


def _normalize_compound(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"S", "SOFT", "C4", "C5"}:
        return "SOFT"
    if text in {"M", "MEDIUM", "C3"}:
        return "MEDIUM"
    if text in {"H", "HARD", "C1", "C2"}:
        return "HARD"
    if "INTER" in text:
        return "INTER"
    if "WET" in text:
        return "WET"
    return "UNKNOWN"


def _apply_shared_legal_action_mask(policy: JsonObject, legality_state: JsonObject) -> JsonObject:
    try:
        _, build_legal_action_mask = _load_legal_action_mask_components()
        legal_mask = build_legal_action_mask(legality_state)
    except Exception as exc:
        return _unavailable_strategy_row(
            f"shared_legal_action_mask_unavailable:{type(exc).__name__}:{exc}",
            policy=policy,
            legality_state=legality_state,
        )

    action = str(policy.get("recommended_action") or "").strip().lower()
    try:
        if action not in {"stay_out", "pit_now", "pit_next_lap"}:
            raise ValueError(f"unsupported_policy_action:{action or 'missing'}")
        pace_mode = _explicit_strategy_pace_mode(policy)
        matching_actions = _matching_strategy_actions(
            [*legal_mask.legal_actions, *legal_mask.illegal_actions],
            action_name=action,
            next_compound=policy.get("next_compound"),
            pace_mode=pace_mode,
        )
        if not matching_actions:
            raise ValueError("no_action_space_variant_matches_policy_action")
    except Exception as exc:
        return _unavailable_strategy_row(
            f"policy_action_cannot_be_mapped:{type(exc).__name__}:{exc}",
            policy=policy,
            legal_action_mask=_legal_action_mask_evidence(legal_mask),
            legality_state=legality_state,
        )

    mask_payload = _legal_action_mask_evidence(legal_mask)
    compatible_legal_actions = [
        candidate for candidate in matching_actions if legal_mask.is_legal(candidate)
    ]
    if not compatible_legal_actions:
        illegal_reasons = sorted(
            {legal_mask.reason_for(candidate) for candidate in matching_actions}
        )
        reason = illegal_reasons[0] if len(illegal_reasons) == 1 else "no_compatible_legal_action"
        return _unavailable_strategy_row(
            f"policy_action_illegal:{reason}",
            policy=policy,
            legal_action_mask=mask_payload,
            legality_state=legality_state,
            original_action=action,
        )

    compatible_keys = [candidate.key for candidate in compatible_legal_actions]
    return {
        **policy,
        "strategy_available": True,
        "safe_to_recommend": True,
        "strategy_unavailable_reason": None,
        "missing_legality_fields": [],
        "pace_mode": pace_mode,
        "compatible_legal_action_keys": compatible_keys,
        "legal_action_key": compatible_keys[0] if pace_mode is not None else None,
        "legal_action_mask": mask_payload,
        "legality_state": _strategy_legality_evidence(legality_state),
    }


def _explicit_strategy_pace_mode(policy: JsonObject) -> str | None:
    raw_mode = _first_present(
        [policy],
        ("pace_mode", "paceMode", "action_mode", "actionMode", "strategy_mode", "strategyMode", "mode"),
    )
    if raw_mode is None or not str(raw_mode).strip():
        return None
    pace_mode = str(raw_mode).strip().lower()
    if pace_mode not in {"conservative", "aggressive"}:
        raise ValueError(f"unsupported_pace_mode:{pace_mode}")
    return pace_mode


def _matching_strategy_actions(
    actions: list[Any],
    *,
    action_name: str,
    next_compound: Any,
    pace_mode: str | None,
) -> list[Any]:
    requested_compound = _normalize_compound(next_compound)
    return [
        candidate
        for candidate in actions
        if candidate.action_type == action_name
        and (action_name == "stay_out" or candidate.compound == requested_compound)
        and (pace_mode is None or candidate.mode == pace_mode)
    ]


def _legal_action_mask_evidence(legal_mask: Any) -> JsonObject:
    return {
        "contract": "packages.f1.models.live_race.action_space.build_legal_action_mask",
        "legal_action_count": legal_mask.legal_count,
        "legal_action_keys": [action.key for action in legal_mask.legal_actions],
        "illegal_reasons": sorted(
            {
                legal_mask.reason_for(action)
                for action in legal_mask.illegal_actions
            }
        ),
    }


def _unavailable_strategy_row(
    reason: str,
    *,
    missing_fields: list[str] | None = None,
    policy: JsonObject | None = None,
    legal_action_mask: JsonObject | None = None,
    legality_state: JsonObject | None = None,
    original_action: str | None = None,
) -> JsonObject:
    source = dict(policy or {})
    original = original_action or str(source.get("recommended_action") or "") or None
    return {
        **source,
        "recommended_action": None,
        "recommendation_confidence": 0.0,
        "pit_urgency": _optional_float(source.get("pit_urgency")) or 0.0,
        "strategy_available": False,
        "safe_to_recommend": False,
        "strategy_unavailable_reason": reason,
        "missing_legality_fields": list(missing_fields or []),
        "original_recommended_action": original,
        "pace_mode": None,
        "compatible_legal_action_keys": [],
        "legal_action_key": None,
        "legal_action_mask": legal_action_mask,
        "legality_state": _strategy_legality_evidence(legality_state) if legality_state else None,
        "policy_version": source.get("policy_version") or "strategy_unavailable_v1",
    }


def _strategy_legality_evidence(state: JsonObject | None) -> JsonObject | None:
    if not state:
        return None
    return {
        "lapNumber": state.get("lap_number"),
        "totalLaps": state.get("total_laps"),
        "remainingLaps": state.get("remaining_laps"),
        "tyreAge": state.get("tyre_age"),
        "stintNumber": state.get("stint_id"),
        "currentCompound": state.get("compound"),
        "usedCompounds": list(state.get("used_compounds") or []),
        "availableCompounds": list(state.get("available_compounds") or []),
        "pitLaneOpen": state.get("pit_lane_open"),
        "isRed": state.get("is_red"),
        "isWetTrack": state.get("is_wet_track"),
        "isBoxLap": state.get("is_box_lap"),
    }


def _load_legal_action_mask_components():
    from packages.f1.models.live_race.action_space import StrategyAction, build_legal_action_mask

    return StrategyAction, build_legal_action_mask


def _load_strategy_policy_adapter():
    try:
        from packages.f1.models.live_race.strategy import BaselineStrategyPolicyAdapter

        return BaselineStrategyPolicyAdapter
    except Exception:
        strategy_path = _repo_root() / "packages" / "f1" / "models" / "live_race" / "strategy.py"
        spec = importlib.util.spec_from_file_location("_f1_live_race_strategy", strategy_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load strategy adapter from {strategy_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.BaselineStrategyPolicyAdapter


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "packages" / "f1" / "models" / "live_race" / "strategy.py").exists():
            return parent
    raise RuntimeError("Could not locate repo root containing packages/f1")


def _fallback_strategy_rows(drivers: list[JsonObject]) -> dict[int, JsonObject]:
    rows: dict[int, JsonObject] = {}
    for driver in drivers:
        compound = str(driver.get("compound") or "UNKNOWN").upper()
        tyre_age = _optional_float(driver.get("tyre_age"))
        if tyre_age is None:
            raise ValueError("fallback strategy received driver without tyre_age")
        service_life = _compound_service_life(compound)
        used = min(2.0, max(0.0, tyre_age / max(1.0, service_life)))
        pit_urgency = min(1.0, max(0.0, (used - 0.62) / 0.55))
        if pit_urgency > 0.72:
            action = "pit_now"
            reason = "tyre life beyond fallback pit threshold"
        elif pit_urgency > 0.45:
            action = "pit_next_lap"
            reason = "fallback tyre pressure building"
        else:
            action = "stay_out"
            reason = "fallback tyre state below pit threshold"
        rows[int(driver["driver_number"])] = {
            "recommended_action": action,
            "recommendation_confidence": 0.55 + (0.3 * pit_urgency),
            "pit_urgency": pit_urgency,
            "tyre_life_used_ratio": used,
            "degradation_risk": max(0.0, used - 0.75),
            "track_risk_score": 0.0,
            "next_compound": _next_compound(compound),
            "strategy_reason": reason,
            "policy_version": "fallback_strategy_policy_v1",
        }
    return rows


def _position_percentile(distribution: dict[str, float], percentile: float) -> float | None:
    if not distribution:
        return None
    parsed: list[tuple[float, float]] = []
    for raw_position, raw_probability in distribution.items():
        position = _optional_float(raw_position)
        probability = _optional_float(raw_probability)
        if position is not None and probability is not None and probability > 0:
            parsed.append((position, probability))
    if not parsed:
        return None
    normalizer = sum(probability for _, probability in parsed)
    if normalizer <= 0:
        return None
    target = max(0.0, min(1.0, percentile))
    cumulative = 0.0
    parsed.sort(key=lambda item: item[0])
    for position, probability in parsed:
        cumulative += probability / normalizer
        if cumulative >= target:
            return round(position, 6)
    return round(parsed[-1][0], 6)


def _uncertainty(
    driver: JsonObject,
    policy: JsonObject,
    strategy_enabled: bool,
    *,
    prediction_kind: str,
) -> float:
    base = {
        "race": 1.20 if strategy_enabled else 1.55,
        "qualifying": 1.35,
        "next-lap": 1.20 if strategy_enabled else 1.50,
        "strategy": 0.85,
    }[prediction_kind]
    if driver.get("last_lap_time") is None and driver.get("best_lap_time") is None:
        base += 0.55
    if prediction_kind in {"race", "next-lap"}:
        base += 0.5 * (_optional_float(policy.get("track_risk_score")) or 0.0)
        base += 0.35 * (_optional_float(policy.get("pit_urgency")) or 0.0)
    return max(0.75, min(4.0, base))


def _confidence(
    driver: JsonObject,
    policy: JsonObject,
    strategy_enabled: bool,
    *,
    prediction_kind: str,
) -> float:
    confidence = 0.48 if strategy_enabled else 0.38
    if prediction_kind in {"race", "strategy"} and driver.get("position") is not None:
        confidence += 0.12
    if driver.get("last_lap_time") is not None or driver.get("best_lap_time") is not None:
        confidence += 0.10
    policy_conf = _optional_float(policy.get("recommendation_confidence")) if strategy_enabled else None
    if policy_conf is not None and prediction_kind in {"race", "next-lap", "strategy"}:
        confidence += min(0.12, policy_conf * 0.12)
    return max(0.0, min(0.80, confidence))


def _dnf_status_indicator(driver: JsonObject) -> float:
    return 1.0 if driver.get("participation_status") == "retired_or_stopped" else 0.0


def _strategy_payload(policy: JsonObject) -> JsonObject:
    return {
        "recommendedAction": policy.get("recommended_action"),
        "confidence": _optional_float(policy.get("recommendation_confidence")),
        "pitUrgency": _optional_float(policy.get("pit_urgency")),
        "nextCompound": policy.get("next_compound"),
        "reason": policy.get("strategy_reason"),
        "policyVersion": policy.get("policy_version"),
        "availability": "available" if policy.get("strategy_available") is True else "unavailable",
        "safeToRecommend": policy.get("safe_to_recommend") is True,
        "unavailableReason": policy.get("strategy_unavailable_reason"),
        "missingLegalityFields": list(policy.get("missing_legality_fields") or []),
        "originalRecommendedAction": policy.get("original_recommended_action"),
        "paceMode": policy.get("pace_mode"),
        "compatibleLegalActionKeys": list(policy.get("compatible_legal_action_keys") or []),
        "legalActionKey": policy.get("legal_action_key"),
        "legalActionMask": policy.get("legal_action_mask"),
        "legalityState": policy.get("legality_state"),
    }


def _track_status_from_snapshot(snapshot: JsonObject) -> str:
    messages = snapshot.get("raceControl") or snapshot.get("race_control") or []
    if not isinstance(messages, list) or not messages:
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        text = " ".join(str(value).lower() for value in message.values())
        if "red" in text:
            return "5"
        if "safety" in text or "vsc" in text:
            return "4"
        if "yellow" in text:
            return "2"
        if "green" in text or "clear" in text:
            return "1"
    return ""


def _gap_seconds(value: Any) -> float:
    if value is None:
        return 0.0
    text = str(value).strip().lower()
    if not text or "leader" in text:
        return 0.0
    text = text.replace("+", "").replace("s", "").strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        return 0.0


def _compound_service_life(compound: str) -> float:
    if compound == "SOFT":
        return 16.0
    if compound == "MEDIUM":
        return 22.0
    if compound == "HARD":
        return 28.0
    if "INTER" in compound:
        return 14.0
    if "WET" in compound:
        return 12.0
    return 20.0


def _next_compound(compound: str) -> str:
    if compound == "SOFT":
        return "MEDIUM"
    if compound == "MEDIUM":
        return "HARD"
    if compound == "HARD":
        return "MEDIUM"
    return compound or "UNKNOWN"


def _json_safe(value: Any) -> Any:
    try:
        import pandas as pd
        import numpy as np

        if isinstance(value, np.generic):
            return _json_safe(value.item())
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(float(value) != 0.0)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "open", "green", "wet"}:
        return True
    if text in {"0", "false", "no", "n", "closed", "dry"}:
        return False
    return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
