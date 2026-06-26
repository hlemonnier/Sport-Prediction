"""Model-service mapping from platform snapshots to prediction snapshots."""

from __future__ import annotations

import math
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]

MODEL_VERSION = "packages_f1_live_strategy_v1"
FEATURES_VERSION = "platform_live_snapshot_strategy_v1"


def predict_from_snapshot(snapshot: JsonObject, *, prediction_kind: str = "race") -> JsonObject:
    drivers = _drivers(snapshot)
    generated_at = _utc_now()
    if not drivers:
        return {
            "modelVersion": MODEL_VERSION,
            "featuresVersion": FEATURES_VERSION,
            "generatedAt": generated_at,
            "predictionKind": prediction_kind,
            "predictions": [],
            "diagnostics": {"driverCount": 0, "strategyPolicyEnabled": False},
        }

    strategy_rows, strategy_enabled, strategy_error = _strategy_policy_rows(drivers)
    scored = []
    for driver in drivers:
        number = int(driver["driver_number"])
        policy = strategy_rows.get(number, {})
        score = _driver_score(driver, policy)
        scored.append((score, driver, policy))

    scored.sort(key=lambda item: (item[0], item[1]["driver_number"]))
    predicted_rank = {int(driver["driver_number"]): rank for rank, (_, driver, _) in enumerate(scored, start=1)}
    total = len(scored)
    predictions = []
    for score, driver, policy in scored:
        driver_number = int(driver["driver_number"])
        rank = predicted_rank[driver_number]
        distribution = _position_distribution(rank, total, uncertainty=_uncertainty(driver, policy, strategy_enabled))
        position_p10 = _position_percentile(distribution, 0.10)
        position_p90 = _position_percentile(distribution, 0.90)
        win_probability = distribution.get("1", 0.0)
        podium_probability = sum(distribution.get(str(pos), 0.0) for pos in range(1, min(3, total) + 1))
        points_probability = sum(distribution.get(str(pos), 0.0) for pos in range(1, min(10, total) + 1))
        confidence = _confidence(driver, policy, strategy_enabled)
        predictions.append(
            {
                "model_version": MODEL_VERSION,
                "prediction_time": generated_at,
                "source_event_sequence": _optional_int(snapshot.get("seq")) or 0,
                "features_version": FEATURES_VERSION,
                "driver_number": driver_number,
                "expected_position": round(float(rank), 6),
                "position_p10": position_p10,
                "position_p90": position_p90,
                "position_distribution": distribution,
                "win_probability": round(win_probability, 6),
                "podium_probability": round(podium_probability, 6),
                "points_probability": round(points_probability, 6),
                "dnf_probability": round(_dnf_probability(driver, policy), 6),
                "confidence": round(confidence, 6),
                "strategy": _strategy_payload(policy),
                "score": round(score, 6),
            }
        )

    return {
        "modelVersion": MODEL_VERSION,
        "featuresVersion": FEATURES_VERSION,
        "generatedAt": generated_at,
        "predictionKind": prediction_kind,
        "predictions": predictions,
        "diagnostics": {
            "driverCount": total,
            "strategyPolicyEnabled": strategy_enabled,
            "strategyPolicyError": strategy_error,
            "sourceEventSequence": _optional_int(snapshot.get("seq")) or 0,
        },
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


def _driver_score(driver: JsonObject, policy: JsonObject) -> float:
    position = _optional_float(driver.get("position")) or 20.0
    last_lap = _optional_float(driver.get("last_lap_time"))
    best_lap = _optional_float(driver.get("best_lap_time"))
    lap_signal = last_lap if last_lap is not None else best_lap
    tyre_used = _optional_float(policy.get("tyre_life_used_ratio")) or 0.0
    pit_urgency = _optional_float(policy.get("pit_urgency")) or 0.0
    speed = _optional_float(driver.get("last_speed")) or 0.0
    gap = _optional_float(driver.get("gap_to_leader_seconds")) or 0.0

    score = float(position)
    if lap_signal is not None:
        score += lap_signal * 0.018
    score += min(8.0, max(0.0, gap) * 0.035)
    score += min(2.5, tyre_used * 0.55)
    score += min(2.0, pit_urgency * 0.95)
    if speed > 0:
        score -= min(0.45, speed / 900.0)
    if str(policy.get("recommended_action")) == "pit_now":
        score += 0.45
    return score


def _position_distribution(rank: int, total: int, *, uncertainty: float) -> dict[str, float]:
    sigma = max(0.75, min(4.0, uncertainty))
    weights = {}
    for position in range(1, total + 1):
        distance = float(position - rank)
        weights[position] = math.exp(-(distance * distance) / (2.0 * sigma * sigma))
    normalizer = sum(weights.values()) or 1.0
    return {str(position): round(weight / normalizer, 6) for position, weight in weights.items()}


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


def _uncertainty(driver: JsonObject, policy: JsonObject, strategy_enabled: bool) -> float:
    base = 1.05 if strategy_enabled else 1.45
    if driver.get("last_lap_time") is None and driver.get("best_lap_time") is None:
        base += 0.55
    base += 0.5 * (_optional_float(policy.get("track_risk_score")) or 0.0)
    base += 0.35 * (_optional_float(policy.get("pit_urgency")) or 0.0)
    return base


def _confidence(driver: JsonObject, policy: JsonObject, strategy_enabled: bool) -> float:
    confidence = 0.52 if strategy_enabled else 0.42
    if driver.get("position") is not None:
        confidence += 0.12
    if driver.get("last_lap_time") is not None or driver.get("best_lap_time") is not None:
        confidence += 0.10
    policy_conf = _optional_float(policy.get("recommendation_confidence"))
    if policy_conf is not None:
        confidence += min(0.12, policy_conf * 0.12)
    return max(0.0, min(0.92, confidence))


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
