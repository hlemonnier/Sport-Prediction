"""Prediction service boundary for the live F1 platform."""

from __future__ import annotations

import asyncio
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import environ
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .schemas import DriverState, PredictionSnapshot, SessionSnapshot
from .time import utc_now_iso


class PredictionService(ABC):
    """Interface hiding the existing model stack behind stable platform calls."""

    @abstractmethod
    async def predict_qualifying(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        raise NotImplementedError

    @abstractmethod
    async def predict_race(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        raise NotImplementedError

    @abstractmethod
    async def predict_next_lap(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        raise NotImplementedError

    @abstractmethod
    async def predict_strategy(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        raise NotImplementedError


@dataclass(slots=True)
class RemotePredictionConfig:
    base_url: str
    timeout_seconds: float = 8.0
    fallback_on_error: bool = True


class RemotePredictionService(PredictionService):
    """HTTP client for the separate F1 prediction/model service."""

    def __init__(
        self,
        config: RemotePredictionConfig,
        *,
        fallback: PredictionService | None = None,
        transport: Callable[[Request, float], dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.fallback = fallback or HeuristicPredictionService()
        self._transport = transport

    async def predict_qualifying(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        return await self._predict("qualifying", state)

    async def predict_race(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        return await self._predict("race", state)

    async def predict_next_lap(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        return await self._predict("next-lap", state)

    async def predict_strategy(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        return await self._predict("strategy", state)

    async def _predict(self, kind: str, state: SessionSnapshot) -> list[PredictionSnapshot]:
        try:
            return await asyncio.to_thread(self._predict_sync, kind, state)
        except Exception:
            if not self.config.fallback_on_error:
                raise
            if kind == "qualifying":
                return await self.fallback.predict_qualifying(state)
            if kind == "next-lap":
                return await self.fallback.predict_next_lap(state)
            if kind == "strategy":
                return await self.fallback.predict_strategy(state)
            return await self.fallback.predict_race(state)

    def _predict_sync(self, kind: str, state: SessionSnapshot) -> list[PredictionSnapshot]:
        endpoint = f"{self.config.base_url.rstrip('/')}/api/f1/predict/{kind}"
        payload = json.dumps({"snapshot": state.to_dict()}, separators=(",", ":")).encode("utf-8")
        request = Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            response_payload = (
                self._transport(request, self.config.timeout_seconds)
                if self._transport is not None
                else _urlopen_json(request, timeout=self.config.timeout_seconds)
            )
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"F1 prediction service failed with HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"F1 prediction service failed: {exc.reason}") from exc

        response_kind = response_payload.get("predictionKind", response_payload.get("prediction_kind"))
        if response_kind is not None and str(response_kind).strip().lower().replace("_", "-") != kind:
            raise RuntimeError(
                f"F1 prediction service returned prediction kind {response_kind!r} for requested kind {kind!r}"
            )
        predictions = response_payload.get("predictions")
        if not isinstance(predictions, list):
            raise RuntimeError("F1 prediction service response did not include predictions list")
        prediction_rows = [item for item in predictions if isinstance(item, dict)]
        if len(prediction_rows) != len(predictions):
            raise RuntimeError("F1 prediction service response included a non-object prediction row")
        _validate_joint_prediction_payloads(prediction_rows, state)
        return [_prediction_from_payload(item, state.seq) for item in prediction_rows]


class HeuristicPredictionService(PredictionService):
    """Honest target-specific fallback when the model service is unavailable.

    These are deterministic snapshot baselines, not trained package models.
    Their position matrices are jointly balanced so probability invariants are
    preserved even during a remote-service outage.
    """

    model_versions = {
        "race": "platform_fallback_live_race_joint_baseline_v1",
        "qualifying": "platform_fallback_qualifying_pace_baseline_v1",
        "next-lap": "platform_fallback_next_lap_pace_baseline_v1",
        "strategy": "platform_fallback_strategy_context_baseline_v1",
    }
    feature_versions = {
        "race": "platform_fallback_live_race_snapshot_v1",
        "qualifying": "platform_fallback_qualifying_snapshot_v1",
        "next-lap": "platform_fallback_next_lap_snapshot_v1",
        "strategy": "platform_fallback_strategy_snapshot_v1",
    }

    async def predict_qualifying(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        return self._target_predictions(state, "qualifying")

    async def predict_race(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        return self._target_predictions(state, "race")

    async def predict_next_lap(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        return self._target_predictions(state, "next-lap")

    async def predict_strategy(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        return self._target_predictions(state, "strategy")

    def _target_predictions(self, state: SessionSnapshot, kind: str) -> list[PredictionSnapshot]:
        drivers = list(state.drivers)
        if not drivers:
            return []
        total = len(drivers)
        ordered = sorted(drivers, key=lambda driver: (_fallback_target_score(driver, kind, total), driver.driver_number))
        distributions = _joint_fallback_distributions(ordered, kind=kind)
        snapshots: list[PredictionSnapshot] = []
        for driver in ordered:
            distribution = distributions[driver.driver_number]
            position_p10 = _position_percentile(distribution, 0.10)
            position_p90 = _position_percentile(distribution, 0.90)
            expected_position = sum(float(position) * probability for position, probability in distribution.items())
            snapshots.append(
                PredictionSnapshot(
                    model_version=self.model_versions[kind],
                    prediction_time=utc_now_iso(),
                    source_event_sequence=state.seq,
                    features_version=self.feature_versions[kind],
                    driver_number=driver.driver_number,
                    expected_position=round(expected_position, 6),
                    position_distribution=distribution,
                    win_probability=round(distribution.get("1", 0.0), 12),
                    podium_probability=round(
                        sum(distribution.get(str(position), 0.0) for position in range(1, min(3, total) + 1)),
                        12,
                    ),
                    points_probability=round(
                        sum(distribution.get(str(position), 0.0) for position in range(1, min(10, total) + 1)),
                        12,
                    ),
                    dnf_probability=0.03 if kind in {"race", "strategy"} else 0.0,
                    confidence=0.32 if kind in {"race", "strategy"} else 0.26,
                    position_p10=position_p10,
                    position_p90=position_p90,
                )
            )
        return snapshots


def _fallback_target_score(driver: DriverState, kind: str, total: int) -> float:
    if kind in {"race", "strategy"}:
        return float(driver.position or total)

    lap_signal = driver.best_lap_time if kind == "qualifying" else driver.last_lap_time
    if lap_signal is None:
        lap_signal = driver.last_lap_time if kind == "qualifying" else driver.best_lap_time
    score = float(lap_signal) if lap_signal is not None else 1_000.0 + float(driver.driver_number) / 1_000.0
    if kind == "next-lap":
        compound = str(driver.current_compound or "UNKNOWN").upper()
        deg = {"SOFT": 0.050, "MEDIUM": 0.040, "HARD": 0.030, "INTERMEDIATE": 0.060, "WET": 0.065}.get(
            compound,
            0.040,
        )
        score += deg * max(0.0, float(driver.tyre_age or 0))
    return score


def _joint_fallback_distributions(
    ordered: list[DriverState],
    *,
    kind: str,
) -> dict[int, dict[str, float]]:
    total = len(ordered)
    sigma = 1.50 if kind == "race" else 1.30 if kind in {"qualifying", "next-lap"} else 0.90
    matrix = [
        [
            max(1e-15, math.exp(-((position - rank) ** 2) / (2.0 * sigma * sigma)))
            for position in range(1, total + 1)
        ]
        for rank in range(1, total + 1)
    ]
    for _ in range(1000):
        for row_index in range(total):
            row_sum = sum(matrix[row_index]) or 1.0
            matrix[row_index] = [value / row_sum for value in matrix[row_index]]
        for column_index in range(total):
            column_sum = sum(matrix[row][column_index] for row in range(total)) or 1.0
            for row_index in range(total):
                matrix[row_index][column_index] /= column_sum
        row_error = max(abs(sum(row) - 1.0) for row in matrix)
        if row_error <= 1e-12:
            break
    return {
        driver.driver_number: {
            str(position): round(matrix[row_index][position - 1], 12)
            for position in range(1, total + 1)
        }
        for row_index, driver in enumerate(ordered)
    }


def prediction_service_from_env() -> PredictionService:
    url = environ.get("F1_PLATFORM_PREDICTION_URL")
    if not url:
        return HeuristicPredictionService()
    timeout = _optional_float(environ.get("F1_PLATFORM_PREDICTION_TIMEOUT_SECONDS")) or 8.0
    fallback = environ.get("F1_PLATFORM_PREDICTION_FALLBACK", "1").strip().lower() not in {"0", "false", "no"}
    return RemotePredictionService(
        RemotePredictionConfig(base_url=url, timeout_seconds=timeout, fallback_on_error=fallback),
        fallback=HeuristicPredictionService(),
    )


def _validate_joint_prediction_payloads(
    payloads: list[dict[str, Any]],
    state: SessionSnapshot,
    *,
    tolerance: float = 2e-5,
) -> None:
    """Reject incomplete or incoherent remote position marginals.

    A valid F1 order is an assignment: every driver occupies one position and
    every position is occupied by one driver.  Validating this boundary keeps a
    malformed remote response from silently reaching the UI; the caller can
    then use its configured safe fallback.
    """

    expected_drivers = {int(driver.driver_number) for driver in state.drivers}
    if not expected_drivers:
        if payloads:
            raise RuntimeError("F1 prediction service returned predictions for an empty driver field")
        return
    if len(payloads) != len(expected_drivers):
        raise RuntimeError(
            "F1 prediction service returned an incomplete field: "
            f"expected {len(expected_drivers)} drivers, received {len(payloads)}"
        )

    total = len(expected_drivers)
    expected_positions = {str(position) for position in range(1, total + 1)}
    rows: dict[int, dict[str, float]] = {}
    for payload in payloads:
        driver_number = _required_int(payload.get("driver_number", payload.get("driverNumber")), "driver_number")
        if driver_number in rows:
            raise RuntimeError(f"F1 prediction service returned duplicate driver {driver_number}")
        model_version = payload.get("model_version", payload.get("modelVersion"))
        features_version = payload.get("features_version", payload.get("featuresVersion"))
        if not str(model_version or "").strip() or not str(features_version or "").strip():
            raise RuntimeError(
                f"F1 prediction service driver {driver_number} is missing explicit model/features provenance"
            )
        raw_distribution = payload.get("position_distribution", payload.get("positionDistribution"))
        if not isinstance(raw_distribution, dict):
            raise RuntimeError(f"F1 prediction service driver {driver_number} has no position distribution")
        distribution: dict[str, float] = {}
        for key, raw_probability in raw_distribution.items():
            probability = _optional_float(raw_probability)
            if probability is None or not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise RuntimeError(
                    f"F1 prediction service driver {driver_number} has invalid probability for position {key}"
                )
            distribution[str(key)] = probability
        if set(distribution) != expected_positions:
            raise RuntimeError(
                f"F1 prediction service driver {driver_number} position keys do not cover 1..{total}"
            )
        if abs(sum(distribution.values()) - 1.0) > tolerance:
            raise RuntimeError(f"F1 prediction service driver {driver_number} distribution does not sum to one")

        derived = {
            "win": distribution["1"],
            "podium": sum(distribution[str(position)] for position in range(1, min(3, total) + 1)),
            "points": sum(distribution[str(position)] for position in range(1, min(10, total) + 1)),
        }
        supplied = {
            "win": _optional_float(payload.get("win_probability", payload.get("winProbability"))),
            "podium": _optional_float(payload.get("podium_probability", payload.get("podiumProbability"))),
            "points": _optional_float(payload.get("points_probability", payload.get("pointsProbability"))),
        }
        for name, expected in derived.items():
            if supplied[name] is None or abs(float(supplied[name]) - expected) > tolerance:
                raise RuntimeError(
                    f"F1 prediction service driver {driver_number} {name} probability disagrees with its distribution"
                )
        rows[driver_number] = distribution

    if set(rows) != expected_drivers:
        missing = sorted(expected_drivers - set(rows))
        unexpected = sorted(set(rows) - expected_drivers)
        raise RuntimeError(
            f"F1 prediction service driver field mismatch; missing={missing}, unexpected={unexpected}"
        )

    column_sums = {
        position: sum(distribution[position] for distribution in rows.values())
        for position in expected_positions
    }
    bad_columns = {
        position: value
        for position, value in column_sums.items()
        if abs(value - 1.0) > tolerance
    }
    if bad_columns:
        raise RuntimeError(f"F1 prediction service position columns do not sum to one: {bad_columns}")

    win_sum = sum(distribution["1"] for distribution in rows.values())
    podium_sum = sum(
        sum(distribution[str(position)] for position in range(1, min(3, total) + 1))
        for distribution in rows.values()
    )
    points_sum = sum(
        sum(distribution[str(position)] for position in range(1, min(10, total) + 1))
        for distribution in rows.values()
    )
    if abs(win_sum - 1.0) > tolerance:
        raise RuntimeError("F1 prediction service win probabilities do not sum to one")
    if abs(podium_sum - float(min(3, total))) > tolerance:
        raise RuntimeError("F1 prediction service podium probabilities do not sum to the podium capacity")
    if abs(points_sum - float(min(10, total))) > tolerance:
        raise RuntimeError("F1 prediction service points probabilities do not sum to the points capacity")


def _prediction_from_payload(payload: dict[str, Any], fallback_sequence: int) -> PredictionSnapshot:
    distribution = _distribution_payload(payload.get("position_distribution", payload.get("positionDistribution")))
    position_p10 = _optional_float(
        _first_present(payload, "position_p10", "positionP10", "position_percentile_10", "positionPercentile10")
    )
    position_p90 = _optional_float(
        _first_present(payload, "position_p90", "positionP90", "position_percentile_90", "positionPercentile90")
    )
    if position_p10 is None:
        position_p10 = _position_percentile(distribution, 0.10)
    if position_p90 is None:
        position_p90 = _position_percentile(distribution, 0.90)
    return PredictionSnapshot(
        model_version=str(payload.get("model_version") or payload.get("modelVersion") or "remote_f1_model"),
        prediction_time=str(payload.get("prediction_time") or payload.get("predictionTime") or utc_now_iso()),
        source_event_sequence=_optional_int(payload.get("source_event_sequence", payload.get("sourceEventSequence")))
        or fallback_sequence,
        features_version=str(payload.get("features_version") or payload.get("featuresVersion") or "remote_features"),
        driver_number=_required_int(payload.get("driver_number", payload.get("driverNumber")), "driver_number"),
        expected_position=_optional_float(payload.get("expected_position", payload.get("expectedPosition"))),
        position_distribution=distribution,
        win_probability=_probability(payload.get("win_probability", payload.get("winProbability"))),
        podium_probability=_probability(payload.get("podium_probability", payload.get("podiumProbability"))),
        points_probability=_probability(payload.get("points_probability", payload.get("pointsProbability"))),
        dnf_probability=_probability(payload.get("dnf_probability", payload.get("dnfProbability"))),
        confidence=_probability(payload.get("confidence"), default=0.5),
        position_p10=position_p10,
        position_p90=position_p90,
    )


def _urlopen_json(request: Request, *, timeout: float) -> dict[str, Any]:
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("F1 prediction service response was not a JSON object")
    return payload


def _distribution_payload(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for key, raw in value.items():
        number = _optional_float(raw)
        if number is not None:
            out[str(key)] = round(max(0.0, min(1.0, number)), 12)
    return out


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


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    position_range = payload.get("position_range", payload.get("positionRange"))
    if isinstance(position_range, dict):
        for key in keys:
            short_key = key.replace("position_", "").replace("position", "")
            if key in position_range:
                return position_range[key]
            if short_key in position_range:
                return position_range[short_key]
    return None


def _required_int(value: Any, name: str) -> int:
    parsed = _optional_int(value)
    if parsed is None:
        raise RuntimeError(f"F1 prediction service response missing integer {name}")
    return parsed


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
    if number != number:
        return None
    return number


def _probability(value: Any, *, default: float = 0.0) -> float:
    number = _optional_float(value)
    if number is None:
        return default
    return round(max(0.0, min(1.0, number)), 12)
