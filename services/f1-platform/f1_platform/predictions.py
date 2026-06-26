"""Prediction service boundary for the live F1 platform."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod
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

        predictions = response_payload.get("predictions")
        if not isinstance(predictions, list):
            raise RuntimeError("F1 prediction service response did not include predictions list")
        return [_prediction_from_payload(item, state.seq) for item in predictions if isinstance(item, dict)]


class HeuristicPredictionService(PredictionService):
    """Deterministic placeholder until package F1 models are wired in.

    It is intentionally conservative and versioned so UI/API integration can be
    built without pretending this is a trained production forecast.
    """

    model_version = "heuristic_live_race_v0"
    features_version = "live_state_snapshot_v1"

    async def predict_qualifying(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        return self._position_based_predictions(state)

    async def predict_race(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        return self._position_based_predictions(state)

    async def predict_next_lap(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        return self._position_based_predictions(state)

    async def predict_strategy(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        return self._position_based_predictions(state)

    def _position_based_predictions(self, state: SessionSnapshot) -> list[PredictionSnapshot]:
        drivers = [driver for driver in state.drivers if driver.position is not None]
        if not drivers:
            return []
        total = max(len(drivers), 1)
        snapshots = []
        for driver in drivers:
            rank = float(driver.position or total)
            strength = max(0.0, (total + 1.0 - rank) / total)
            distribution = _distribution_for_driver(driver, total)
            position_p10 = _position_percentile(distribution, 0.10)
            position_p90 = _position_percentile(distribution, 0.90)
            snapshots.append(
                PredictionSnapshot(
                    model_version=self.model_version,
                    prediction_time=utc_now_iso(),
                    source_event_sequence=state.seq,
                    features_version=self.features_version,
                    driver_number=driver.driver_number,
                    expected_position=rank,
                    position_distribution=distribution,
                    win_probability=round(max(0.01, strength**3), 4),
                    podium_probability=round(max(0.03, strength**1.8), 4),
                    points_probability=round(max(0.08, min(0.98, strength + 0.12)), 4),
                    dnf_probability=0.03,
                    confidence=round(min(0.85, 0.35 + state.seq / 250.0), 4),
                    position_p10=position_p10,
                    position_p90=position_p90,
                )
            )
        return snapshots


def _distribution_for_driver(driver: DriverState, total: int) -> dict[str, float]:
    position = driver.position or total
    weights: dict[int, float] = {}
    for place in range(1, total + 1):
        weights[place] = 1.0 / (1.0 + abs(place - position))
    normalizer = sum(weights.values()) or 1.0
    return {str(place): round(weight / normalizer, 4) for place, weight in weights.items()}


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
            out[str(key)] = round(max(0.0, min(1.0, number)), 6)
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
    return round(max(0.0, min(1.0, number)), 6)
