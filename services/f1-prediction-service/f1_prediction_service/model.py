"""Target-specific snapshot baselines for the live F1 platform.

The full ``packages.f1`` live-race model consumes a causal lap-history trace.
The platform contract currently supplies only the latest state per driver, so
claiming that the package model produced these forecasts would be incorrect.
This module instead exposes honest, deterministic snapshot baselines and uses
the canonical package strategy adapter where its input contract is satisfied.

All position outputs are joint marginals: a Sinkhorn-balanced positive matrix
ensures that every driver row and every finishing-position column sums to one.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]

MODEL_VERSION = "f1_snapshot_target_dispatch_v2"
FEATURES_VERSION = "platform_target_specific_snapshot_v2"

TARGET_MODELS: dict[str, dict[str, str]] = {
    "race": {
        "model_version": "live_race_snapshot_joint_baseline_v2",
        "features_version": "platform_live_race_snapshot_v2",
        "target_definition": "finishing-order marginals conditional on the latest race snapshot",
    },
    "qualifying": {
        "model_version": "qualifying_snapshot_pace_baseline_v1",
        "features_version": "platform_qualifying_pace_snapshot_v1",
        "target_definition": "qualifying-order marginals from observed best-lap pace",
    },
    "next-lap": {
        "model_version": "next_lap_snapshot_pace_baseline_v1",
        "features_version": "platform_next_lap_pace_snapshot_v1",
        "target_definition": "next-lap relative-order marginals from recent pace and tyre state",
    },
    "strategy": {
        "model_version": "live_strategy_policy_context_baseline_v1",
        "features_version": "platform_live_strategy_snapshot_v1",
        "target_definition": "strategy recommendations with current-order context, not a finish forecast",
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
                "strategyPolicyEnabled": False,
                "forecastAvailable": False,
                "unavailableReason": "snapshot_has_no_drivers",
                "provenance": {
                    "modelType": "deterministic_untrained_snapshot_baseline",
                    "canonicalLiveRaceModelUsed": False,
                    "canonicalLiveRaceUnavailableReason": "snapshot_has_no_drivers",
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
    distributions, joint_diagnostics = _joint_position_distributions(
        scored,
        prediction_kind=prediction_kind,
        strategy_enabled=strategy_enabled,
    )
    predictions = []
    for score, driver, policy in scored:
        driver_number = int(driver["driver_number"])
        distribution = distributions[driver_number]
        position_p10 = _position_percentile(distribution, 0.10)
        position_p90 = _position_percentile(distribution, 0.90)
        win_probability = distribution.get("1", 0.0)
        podium_probability = sum(distribution.get(str(pos), 0.0) for pos in range(1, min(3, total) + 1))
        points_probability = sum(distribution.get(str(pos), 0.0) for pos in range(1, min(10, total) + 1))
        expected_position = sum(float(position) * probability for position, probability in distribution.items())
        confidence = _confidence(driver, policy, strategy_enabled, prediction_kind=prediction_kind)
        strategy_payload = _strategy_payload(policy) if strategy_requested else None
        predictions.append(
            {
                "model_version": target["model_version"],
                "prediction_time": generated_at,
                "source_event_sequence": _optional_int(snapshot.get("seq")) or 0,
                "features_version": target["features_version"],
                "driver_number": driver_number,
                "expected_position": round(expected_position, 6),
                "position_p10": position_p10,
                "position_p90": position_p90,
                "position_distribution": distribution,
                "win_probability": round(win_probability, 12),
                "podium_probability": round(podium_probability, 12),
                "points_probability": round(points_probability, 12),
                "dnf_probability": round(_dnf_probability(driver, policy), 6)
                if prediction_kind in {"race", "strategy"}
                else 0.0,
                "confidence": round(confidence, 6),
                "strategy": strategy_payload,
                "score": round(score, 6),
            }
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
            "forecastAvailable": True,
            "strategyPolicyEnabled": strategy_enabled,
            "strategyPolicyError": strategy_error,
            "sourceEventSequence": _optional_int(snapshot.get("seq")) or 0,
            "jointDistribution": joint_diagnostics,
            "provenance": {
                "modelType": "deterministic_untrained_snapshot_baseline",
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
                "fallbackStrategyPolicyUsed": bool(strategy_requested and not strategy_enabled),
            },
        },
    }


def _score_drivers(
    drivers: list[JsonObject],
    strategy_rows: dict[int, JsonObject],
    *,
    prediction_kind: str,
) -> list[tuple[float, JsonObject, JsonObject]]:
    lap_values = [
        value
        for driver in drivers
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
    scored.sort(key=lambda item: (item[0], int(item[1]["driver_number"])))
    return scored


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
) -> tuple[dict[int, dict[str, float]], JsonObject]:
    """Return approximately doubly-stochastic position marginals.

    A strictly-positive rank kernel is alternately normalized over driver rows
    and position columns.  The resulting matrix is a coherent set of assignment
    marginals, unlike independently normalized per-driver curves.
    """

    total = len(scored)
    matrix: list[list[float]] = []
    for rank, (_, driver, policy) in enumerate(scored, start=1):
        sigma = _uncertainty(driver, policy, strategy_enabled, prediction_kind=prediction_kind)
        row = []
        for position in range(1, total + 1):
            distance = float(position - rank)
            row.append(max(1e-15, math.exp(-(distance * distance) / (2.0 * sigma * sigma))))
        matrix.append(row)

    iterations = 0
    for iterations in range(1, 1001):
        for row_index in range(total):
            row_sum = sum(matrix[row_index]) or 1.0
            matrix[row_index] = [value / row_sum for value in matrix[row_index]]
        for column_index in range(total):
            column_sum = sum(matrix[row][column_index] for row in range(total)) or 1.0
            for row_index in range(total):
                matrix[row_index][column_index] /= column_sum
        row_error = max(abs(sum(row) - 1.0) for row in matrix)
        column_error = max(
            abs(sum(matrix[row][column] for row in range(total)) - 1.0)
            for column in range(total)
        )
        if max(row_error, column_error) <= 1e-12:
            break

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
        "method": "sinkhorn_balanced_gaussian_rank_kernel",
        "iterations": iterations,
        "rowMaxAbsError": max(abs(value - 1.0) for value in row_sums),
        "columnMaxAbsError": max(abs(value - 1.0) for value in column_sums),
        "winProbabilitySum": column_sums[0],
        "podiumProbabilitySum": sum(column_sums[: min(3, total)]),
        "pointsProbabilitySum": sum(column_sums[: min(10, total)]),
    }


def _drivers(snapshot: JsonObject) -> list[JsonObject]:
    raw = snapshot.get("drivers") or snapshot.get("Drivers") or []
    drivers = [item for item in raw if isinstance(item, dict)]
    normalized = []
    for item in drivers:
        number = _optional_int(item.get("driver_number", item.get("driverNumber")))
        if number is None:
            continue
        normalized.append(
            {
                "driver_number": number,
                "driver_id": str(item.get("acronym") or item.get("driver_number") or number),
                "driver_name": str(item.get("full_name") or item.get("acronym") or number),
                "team_name": item.get("team_name"),
                "position": _optional_float(item.get("position")),
                "current_lap": _optional_float(item.get("current_lap")),
                "last_lap_time": _optional_float(item.get("last_lap_time")),
                "best_lap_time": _optional_float(item.get("best_lap_time")),
                "compound": item.get("current_compound") or "UNKNOWN",
                "tyre_age": _optional_float(item.get("tyre_age")) or 0.0,
                "track_status": item.get("track_status") or _track_status_from_snapshot(snapshot),
                "gap_to_leader_seconds": _gap_seconds(item.get("gap_to_leader") or item.get("interval")),
                "last_speed": _optional_float(item.get("last_speed")),
                "pit_status": item.get("pit_status"),
            }
        )
    return normalized


def _strategy_policy_rows(drivers: list[JsonObject]) -> tuple[dict[int, JsonObject], bool, str | None]:
    try:
        import pandas as pd
        BaselineStrategyPolicyAdapter = _load_strategy_policy_adapter()
    except Exception as exc:
        return _fallback_strategy_rows(drivers), False, f"{type(exc).__name__}: {exc}"

    frame = pd.DataFrame(
        [
            {
                "driver_id": driver["driver_id"],
                "driver_name": driver["driver_name"],
                "lap_number": driver["current_lap"],
                "compound": driver["compound"],
                "tyre_age": driver["tyre_age"],
                "track_status": driver["track_status"],
                "gap_to_leader_seconds": driver["gap_to_leader_seconds"],
                "next_lap_mean": driver["last_lap_time"] or driver["best_lap_time"],
                "position": driver["position"],
                "driver_number": driver["driver_number"],
            }
            for driver in drivers
        ]
    )
    try:
        actions = BaselineStrategyPolicyAdapter().evaluate_actions(frame)
    except Exception as exc:
        return _fallback_strategy_rows(drivers), False, f"{type(exc).__name__}: {exc}"

    rows: dict[int, JsonObject] = {}
    for idx, driver in enumerate(drivers):
        row = actions.iloc[idx].to_dict() if idx < len(actions) else {}
        rows[int(driver["driver_number"])] = _json_safe(row)
    return rows, True, None


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
        tyre_age = float(driver.get("tyre_age") or 0.0)
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


def _dnf_probability(driver: JsonObject, policy: JsonObject) -> float:
    pit_status = str(driver.get("pit_status") or "").lower()
    if any(term in pit_status for term in ("retired", "dnf", "stopped")):
        return 0.65
    return 0.025 + 0.035 * (_optional_float(policy.get("track_risk_score")) or 0.0)


def _strategy_payload(policy: JsonObject) -> JsonObject:
    return {
        "recommendedAction": policy.get("recommended_action"),
        "confidence": _optional_float(policy.get("recommendation_confidence")),
        "pitUrgency": _optional_float(policy.get("pit_urgency")),
        "nextCompound": policy.get("next_compound"),
        "reason": policy.get("strategy_reason"),
        "policyVersion": policy.get("policy_version"),
    }


def _track_status_from_snapshot(snapshot: JsonObject) -> str:
    messages = snapshot.get("raceControl") or snapshot.get("race_control") or []
    if not isinstance(messages, list) or not messages:
        return "1"
    latest = messages[-1] if isinstance(messages[-1], dict) else {}
    text = " ".join(str(value).lower() for value in latest.values())
    if "red" in text:
        return "5"
    if "safety" in text or "vsc" in text:
        return "4"
    if "yellow" in text:
        return "2"
    return "1"


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
