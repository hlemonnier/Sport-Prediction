"""Prediction service boundary for the live F1 platform."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import environ
from pathlib import Path
from typing import Any, Callable, Literal, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .schemas import DriverState, PredictionSnapshot, SessionSnapshot
from .time import utc_now_iso

PredictionKind = Literal["race", "qualifying", "next-lap", "strategy"]
_PREDICTION_KINDS = {"race", "qualifying", "next-lap", "strategy"}
_POSITION_SEMANTICS: dict[PredictionKind, str] = {
    "race": "race_finish_order",
    "qualifying": "qualifying_classification_order",
    "next-lap": "next_lap_pace_order",
    "strategy": "not_applicable",
}


class PredictionService(ABC):
    """Interface hiding the existing model stack behind stable platform calls."""

    async def predict(
        self,
        prediction_kind: PredictionKind | str,
        state: SessionSnapshot,
    ) -> list[PredictionSnapshot]:
        kind = normalize_prediction_kind(prediction_kind)
        if kind == "qualifying":
            return await self.predict_qualifying(state)
        if kind == "next-lap":
            return await self.predict_next_lap(state)
        if kind == "strategy":
            return await self.predict_strategy(state)
        return await self.predict_race(state)

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

    async def _predict(self, kind: PredictionKind, state: SessionSnapshot) -> list[PredictionSnapshot]:
        try:
            return await asyncio.to_thread(self._predict_sync, kind, state)
        except Exception:
            if not self.config.fallback_on_error:
                raise
            return await self.fallback.predict(kind, state)

    def _predict_sync(self, kind: PredictionKind, state: SessionSnapshot) -> list[PredictionSnapshot]:
        endpoint = f"{self.config.base_url.rstrip('/')}/api/f1/predict/{kind}"
        payload = json.dumps(
            {"predictionKind": kind, "snapshot": state.to_dict()},
            separators=(",", ":"),
        ).encode("utf-8")
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
        if kind == "strategy":
            _validate_strategy_prediction_payloads(prediction_rows, state)
        else:
            _validate_joint_prediction_payloads(prediction_rows, state, prediction_kind=kind)
        return [
            _prediction_from_payload(item, state, prediction_kind=kind)
            for item in prediction_rows
        ]


class HeuristicPredictionService(PredictionService):
    """Honest target-specific fallback when the model service is unavailable.

    These are deterministic snapshot baselines, not trained package models.
    Forecast position matrices are jointly balanced; strategy fallback rows
    explicitly carry no recommendation or finish-order probabilities.
    """

    model_versions = {
        "race": "platform_fallback_live_race_joint_baseline_v2",
        "qualifying": "platform_fallback_qualifying_pace_baseline_v2",
        "next-lap": "platform_fallback_next_lap_pace_baseline_v2",
        "strategy": "platform_fallback_strategy_unavailable_v2",
    }
    feature_versions = {
        "race": "platform_fallback_live_race_snapshot_v2",
        "qualifying": "platform_fallback_qualifying_snapshot_v2",
        "next-lap": "platform_fallback_next_lap_snapshot_v2",
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

    def _target_predictions(self, state: SessionSnapshot, kind: PredictionKind) -> list[PredictionSnapshot]:
        drivers = list(state.drivers)
        if not drivers:
            return []
        if kind == "strategy":
            return self._unavailable_strategy_predictions(state, drivers)
        statuses = {
            driver.driver_number: _target_status_for_driver(state, driver, kind)
            for driver in drivers
        }
        eligible = [driver for driver in drivers if statuses[driver.driver_number][0]]
        total = len(eligible)
        ordered = sorted(
            eligible,
            key=lambda driver: (_fallback_target_score(driver, kind, total), driver.driver_number),
        )
        distributions = _joint_fallback_distributions(ordered, kind=kind) if ordered else {}
        snapshots: list[PredictionSnapshot] = []
        for driver in ordered:
            _, eligibility, participation, _ = statuses[driver.driver_number]
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
                    dnf_probability=1.0 if kind == "race" and participation == "retired_or_stopped" else 0.0,
                    confidence=0.32 if kind == "race" else 0.26,
                    position_p10=position_p10,
                    position_p90=position_p90,
                    prediction_kind=kind,
                    position_semantics=position_semantics_for_kind(kind),
                    forecast_available=True,
                    unavailable_reason=None,
                    eligibility_status=eligibility,
                    participation_status=participation,
                )
            )
        for driver in drivers:
            available, eligibility, participation, reason = statuses[driver.driver_number]
            if available:
                continue
            snapshots.append(
                PredictionSnapshot(
                    model_version=self.model_versions[kind],
                    prediction_time=utc_now_iso(),
                    source_event_sequence=state.seq,
                    features_version=self.feature_versions[kind],
                    driver_number=driver.driver_number,
                    expected_position=None,
                    position_distribution={},
                    win_probability=0.0,
                    podium_probability=0.0,
                    points_probability=0.0,
                    dnf_probability=1.0 if kind == "race" and participation == "retired_or_stopped" else 0.0,
                    confidence=0.0,
                    position_p10=None,
                    position_p90=None,
                    prediction_kind=kind,
                    position_semantics=position_semantics_for_kind(kind),
                    forecast_available=False,
                    unavailable_reason=reason,
                    eligibility_status=eligibility,
                    participation_status=participation,
                )
            )
        return snapshots

    def _unavailable_strategy_predictions(
        self,
        state: SessionSnapshot,
        drivers: list[DriverState],
    ) -> list[PredictionSnapshot]:
        """Return an explicit no-recommendation fallback, never a finish forecast."""

        return [
            PredictionSnapshot(
                model_version=self.model_versions["strategy"],
                prediction_time=utc_now_iso(),
                source_event_sequence=state.seq,
                features_version=self.feature_versions["strategy"],
                driver_number=driver.driver_number,
                expected_position=None,
                position_distribution={},
                win_probability=0.0,
                podium_probability=0.0,
                points_probability=0.0,
                dnf_probability=0.0,
                confidence=0.0,
                prediction_kind="strategy",
                position_semantics=position_semantics_for_kind("strategy"),
                strategy=None,
                forecast_available=False,
                unavailable_reason="strategy_service_unavailable",
                eligibility_status="target_unavailable",
                participation_status=_target_status_for_driver(state, driver, "race")[2],
            )
            for driver in drivers
        ]


def _fallback_target_score(driver: DriverState, kind: PredictionKind, total: int) -> float:
    if kind == "race":
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


def _target_status_for_driver(
    state: SessionSnapshot,
    driver: DriverState,
    kind: PredictionKind,
) -> tuple[bool, str, str, str | None]:
    """Return target availability without conflating retirement and exclusion."""

    result = _session_result_for_driver(state, driver.driver_number)
    mappings: list[dict[str, Any]] = [result]
    classified = _optional_bool(
        _mapping_value(result, ("classified", "is_classified", "isClassified"))
    )
    running = _optional_bool(
        _mapping_value(result, ("is_running", "isRunning", "is_active", "isActive"))
    )
    if _truthy_from(mappings, ("dns", "did_not_start", "didNotStart")):
        participation = "dns"
    elif _truthy_from(mappings, ("dsq", "disqualified", "excluded")):
        participation = "dsq"
    elif _truthy_from(mappings, ("withdrawn",)):
        participation = "withdrawn"
    elif classified is False or _truthy_from(
        mappings,
        ("unclassified", "not_classified", "notClassified"),
    ):
        participation = "classification_ineligible"
    else:
        normalized_statuses = [
            str(value or "").strip().lower().replace("_", " ")
            for value in (
                driver.pit_status,
                result.get("status"),
                result.get("classification_status", result.get("classificationStatus")),
            )
            if value is not None
        ]
        status_text = " ".join(normalized_statuses)
        if any(marker in status_text for marker in ("did not start", "dns")):
            participation = "dns"
        elif any(marker in status_text for marker in ("disqualified", "excluded", "dsq")):
            participation = "dsq"
        elif "withdrawn" in status_text:
            participation = "withdrawn"
        elif any(marker in status_text for marker in ("not classified", "unclassified")) or "nc" in normalized_statuses:
            participation = "classification_ineligible"
        elif (
            _truthy_from(mappings, ("dnf", "retired", "stopped", "not_running"))
            or set(normalized_statuses) & {"out", "inactive", "retired", "dnf", "stopped", "not running"}
            or any(
                marker in status_text
                for marker in ("retired", "did not finish", "dnf", "stopped", "not running")
            )
        ):
            participation = "retired_or_stopped"
        elif "finished" in normalized_statuses or any(
            status.startswith("+") and "lap" in status for status in normalized_statuses
        ) or classified is True:
            participation = "finished"
        elif running is False:
            participation = "retired_or_stopped"
        else:
            participation = "running_or_unknown"

    classification_ineligible = participation in {"dns", "dsq", "withdrawn", "classification_ineligible"}
    if classification_ineligible:
        reason_status = "unclassified" if participation == "classification_ineligible" else participation
        return (
            False,
            "target_unavailable",
            participation,
            f"classification_ineligible:{reason_status}",
        )
    if kind == "next-lap" and participation in {"retired_or_stopped", "finished"}:
        return (
            False,
            "target_unavailable",
            participation,
            f"next_lap_unavailable:{participation}",
        )
    if (
        kind == "qualifying"
        and participation in {"retired_or_stopped", "finished"}
        and driver.best_lap_time is None
        and driver.last_lap_time is None
    ):
        return (
            False,
            "target_unavailable",
            participation,
            "qualifying_unavailable:no_valid_lap",
        )
    eligibility = (
        "classification_eligible_retired"
        if participation == "retired_or_stopped"
        else "classification_eligible"
    )
    return True, eligibility, participation, None


def _session_result_for_driver(state: SessionSnapshot, driver_number: int) -> dict[str, Any]:
    for row in state.session_results:
        if not isinstance(row, dict):
            continue
        number = _optional_int(row.get("driver_number", row.get("driverNumber")))
        if number == driver_number:
            return row
    return {}


def _truthy_from(mappings: list[dict[str, Any]], keys: tuple[str, ...]) -> bool:
    for mapping in mappings:
        for key in keys:
            if key in mapping and _optional_bool(mapping.get(key)) is True:
                return True
    return False


def _joint_fallback_distributions(
    ordered: list[DriverState],
    *,
    kind: PredictionKind,
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


def normalize_prediction_kind(value: PredictionKind | str) -> PredictionKind:
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized not in _PREDICTION_KINDS:
        supported = ", ".join(sorted(_PREDICTION_KINDS))
        raise ValueError(f"unsupported prediction kind {value!r}; expected one of: {supported}")
    return cast(PredictionKind, normalized)


def position_semantics_for_kind(value: PredictionKind | str) -> str:
    return _POSITION_SEMANTICS[normalize_prediction_kind(value)]


def prediction_kind_for_session(state: SessionSnapshot) -> PredictionKind:
    """Select the live target from session metadata.

    Qualifying labels take precedence over race-like labels so sprint
    qualifying/shootout sessions can never fall through to a race forecast.
    Unknown or absent metadata keeps the historical race default.
    """

    info = state.session_info or {}
    labels = [
        info.get("session_type"),
        info.get("sessionType"),
        info.get("session_name"),
        info.get("sessionName"),
        info.get("fastf1_session_name"),
        info.get("name"),
    ]
    normalized = [
        str(label).strip().lower().replace("_", "-")
        for label in labels
        if label is not None and str(label).strip()
    ]
    if any(
        label in {"q", "q1", "q2", "q3", "sq"}
        or "qualifying" in label
        or "shootout" in label
        for label in normalized
    ):
        return "qualifying"
    if any(
        label in {"fp", "fp1", "fp2", "fp3"}
        or "practice" in label
        or label.startswith("free practice")
        for label in normalized
    ):
        return "next-lap"
    return "race"


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
    prediction_kind: PredictionKind = "race",
    tolerance: float = 2e-5,
) -> None:
    """Reject target-ineligible rows and incoherent eligible assignments."""

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

    drivers_by_number = {int(driver.driver_number): driver for driver in state.drivers}
    local_status = {
        number: _target_status_for_driver(state, driver, prediction_kind)
        for number, driver in drivers_by_number.items()
    }
    eligible_drivers = {
        number for number, (available, _, _, _) in local_status.items() if available
    }
    eligible_count = len(eligible_drivers)
    expected_positions = {str(position) for position in range(1, eligible_count + 1)}
    rows: dict[int, dict[str, float]] = {}
    availability_by_driver: dict[int, bool] = {}
    event_probabilities_by_driver: dict[int, dict[str, float]] = {}
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
        expected_available, local_eligibility, local_participation, local_reason = local_status.get(
            driver_number,
            (False, "target_unavailable", "unknown", "driver_not_in_local_snapshot"),
        )
        raw_forecast_available = payload.get("forecast_available", payload.get("forecastAvailable"))
        forecast_available = _payload_forecast_available(payload, distribution)
        raw_eligibility = payload.get("eligibility_status", payload.get("eligibilityStatus"))
        eligibility = _normalized_status(raw_eligibility) if raw_eligibility is not None else local_eligibility
        allowed_eligibility = (
            {"classification_eligible", "running_or_unknown"}
            if local_eligibility == "classification_eligible"
            else {"classification_eligible_retired"}
            if local_eligibility == "classification_eligible_retired"
            else {"target_unavailable", "classification_ineligible", "known_not_running"}
        )
        if eligibility not in allowed_eligibility:
            raise RuntimeError(
                f"F1 prediction service driver {driver_number} eligibility {raw_eligibility!r} "
                f"disagrees with local target status {local_eligibility!r}"
            )
        raw_participation = payload.get("participation_status", payload.get("participationStatus"))
        if raw_participation is not None and _normalized_status(raw_participation) != local_participation:
            raise RuntimeError(
                f"F1 prediction service driver {driver_number} participation status disagrees with local state"
            )
        if forecast_available != expected_available:
            raise RuntimeError(
                f"F1 prediction service driver {driver_number} forecast availability disagrees with local "
                f"{prediction_kind} target status ({local_reason or 'eligible'})"
            )
        if not expected_available and (raw_forecast_available is None or raw_eligibility is None):
            raise RuntimeError(
                f"F1 prediction service unavailable driver {driver_number} must explicitly declare "
                "forecast_available=false and eligibility_status"
            )
        if forecast_available:
            if set(distribution) != expected_positions:
                raise RuntimeError(
                    f"F1 prediction service driver {driver_number} position keys do not cover "
                    f"eligible positions 1..{eligible_count}"
                )
            if abs(sum(distribution.values()) - 1.0) > tolerance:
                raise RuntimeError(f"F1 prediction service driver {driver_number} distribution does not sum to one")
        elif distribution:
            raise RuntimeError(
                f"F1 prediction service unavailable driver {driver_number} must have an empty position distribution"
            )
        supplied = {
            "win": _optional_float(payload.get("win_probability", payload.get("winProbability"))),
            "podium": _optional_float(payload.get("podium_probability", payload.get("podiumProbability"))),
            "points": _optional_float(payload.get("points_probability", payload.get("pointsProbability"))),
        }
        for name, probability in supplied.items():
            if probability is None or not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise RuntimeError(
                    f"F1 prediction service driver {driver_number} has invalid {name} probability"
                )
        rows[driver_number] = distribution
        availability_by_driver[driver_number] = forecast_available
        event_probabilities_by_driver[driver_number] = {
            name: float(probability)
            for name, probability in supplied.items()
            if probability is not None
        }
        _validate_remote_prediction_scalars(
            payload,
            driver_number=driver_number,
            distribution=distribution,
            forecast_available=forecast_available,
            tolerance=tolerance,
        )

    if set(rows) != expected_drivers:
        missing = sorted(expected_drivers - set(rows))
        unexpected = sorted(set(rows) - expected_drivers)
        raise RuntimeError(
            f"F1 prediction service driver field mismatch; missing={missing}, unexpected={unexpected}"
        )
    column_sums = {
        position: sum(rows[number][position] for number in eligible_drivers)
        for position in expected_positions
    }
    bad_columns = {
        position: value
        for position, value in column_sums.items()
        if abs(value - 1.0) > tolerance
    }
    if bad_columns:
        raise RuntimeError(f"F1 prediction service position columns do not sum to one: {bad_columns}")

    for driver_number, distribution in rows.items():
        if not availability_by_driver[driver_number]:
            derived = {"win": 0.0, "podium": 0.0, "points": 0.0}
        else:
            derived = {
                "win": distribution["1"] if eligible_count else 0.0,
                "podium": sum(
                    distribution[str(position)]
                    for position in range(1, min(3, eligible_count) + 1)
                ),
                "points": sum(
                    distribution[str(position)]
                    for position in range(1, min(10, eligible_count) + 1)
                ),
            }
        supplied = event_probabilities_by_driver[driver_number]
        for name, expected in derived.items():
            if abs(supplied[name] - expected) > tolerance:
                raise RuntimeError(
                    f"F1 prediction service driver {driver_number} {name} probability disagrees with "
                    f"its eligibility-adjusted classification marginal"
                )

    win_sum = sum(probabilities["win"] for probabilities in event_probabilities_by_driver.values())
    podium_sum = sum(probabilities["podium"] for probabilities in event_probabilities_by_driver.values())
    points_sum = sum(probabilities["points"] for probabilities in event_probabilities_by_driver.values())
    if abs(win_sum - float(min(1, eligible_count))) > tolerance:
        raise RuntimeError(
            "F1 prediction service win probabilities do not sum to the active-driver win capacity"
        )
    if abs(podium_sum - float(min(3, eligible_count))) > tolerance:
        raise RuntimeError(
            "F1 prediction service podium probabilities do not sum to the active-driver podium capacity"
        )
    if abs(points_sum - float(min(10, eligible_count))) > tolerance:
        raise RuntimeError(
            "F1 prediction service points probabilities do not sum to the active-driver points capacity"
        )


def _payload_forecast_available(payload: dict[str, Any], distribution: dict[str, float]) -> bool:
    raw = payload.get("forecast_available", payload.get("forecastAvailable"))
    if raw is None:
        return bool(distribution)
    parsed = _optional_bool(raw)
    if parsed is None:
        raise RuntimeError("F1 prediction service returned an invalid forecast_available flag")
    return parsed


def _validate_remote_prediction_scalars(
    payload: dict[str, Any],
    *,
    driver_number: int,
    distribution: dict[str, float],
    forecast_available: bool,
    tolerance: float,
) -> None:
    expected_raw = payload.get("expected_position", payload.get("expectedPosition"))
    p10_raw = _first_present(
        payload,
        "position_p10",
        "positionP10",
        "position_percentile_10",
        "positionPercentile10",
    )
    p90_raw = _first_present(
        payload,
        "position_p90",
        "positionP90",
        "position_percentile_90",
        "positionPercentile90",
    )
    if not forecast_available:
        if any(value is not None for value in (expected_raw, p10_raw, p90_raw)):
            raise RuntimeError(
                f"F1 prediction service unavailable driver {driver_number} returned position scalars"
            )
        reason = payload.get("unavailable_reason", payload.get("unavailableReason"))
        if not str(reason or "").strip():
            raise RuntimeError(
                f"F1 prediction service unavailable driver {driver_number} is missing an explicit reason"
            )
    else:
        derived_expected = sum(float(position) * probability for position, probability in distribution.items())
        derived_p10 = _position_percentile(distribution, 0.10)
        derived_p90 = _position_percentile(distribution, 0.90)
        for name, raw, expected in (
            ("expected_position", expected_raw, derived_expected),
            ("position_p10", p10_raw, derived_p10),
            ("position_p90", p90_raw, derived_p90),
        ):
            if raw is None:
                continue
            parsed = _optional_float(raw)
            if parsed is None or expected is None or abs(parsed - expected) > tolerance:
                raise RuntimeError(
                    f"F1 prediction service driver {driver_number} {name} disagrees with its distribution"
                )

    for name, raw in (
        ("dnf_probability", payload.get("dnf_probability", payload.get("dnfProbability"))),
        ("confidence", payload.get("confidence")),
    ):
        if raw is None:
            continue
        parsed = _optional_float(raw)
        if parsed is None or not 0.0 <= parsed <= 1.0:
            raise RuntimeError(
                f"F1 prediction service driver {driver_number} has invalid {name}"
            )


def _validate_strategy_prediction_payloads(
    payloads: list[dict[str, Any]],
    state: SessionSnapshot,
) -> None:
    """Validate strategy rows without treating context ranks as finish odds."""

    expected_drivers = {int(driver.driver_number) for driver in state.drivers}
    if not expected_drivers:
        if payloads:
            raise RuntimeError("F1 prediction service returned strategy rows for an empty driver field")
        return
    if len(payloads) != len(expected_drivers):
        raise RuntimeError(
            "F1 prediction service returned an incomplete strategy field: "
            f"expected {len(expected_drivers)} drivers, received {len(payloads)}"
        )

    drivers_by_number = {int(driver.driver_number): driver for driver in state.drivers}
    returned_drivers: set[int] = set()
    for payload in payloads:
        driver_number = _required_int(payload.get("driver_number", payload.get("driverNumber")), "driver_number")
        if driver_number in returned_drivers:
            raise RuntimeError(f"F1 prediction service returned duplicate strategy driver {driver_number}")
        model_version = payload.get("model_version", payload.get("modelVersion"))
        features_version = payload.get("features_version", payload.get("featuresVersion"))
        if not str(model_version or "").strip() or not str(features_version or "").strip():
            raise RuntimeError(
                f"F1 prediction service strategy driver {driver_number} is missing explicit model/features provenance"
            )
        raw_strategy = _raw_strategy_payload(payload)
        if raw_strategy is not None and not isinstance(raw_strategy, dict):
            raise RuntimeError(
                f"F1 prediction service strategy driver {driver_number} has a non-object strategy payload"
            )
        if driver_number not in drivers_by_number:
            raise RuntimeError(f"F1 prediction service returned unexpected strategy driver {driver_number}")
        distribution = payload.get("position_distribution", payload.get("positionDistribution", {}))
        if distribution not in ({}, None):
            raise RuntimeError(
                f"F1 prediction service strategy driver {driver_number} exposed a position forecast"
            )
        for field in ("win_probability", "podium_probability", "points_probability"):
            raw_probability = payload.get(field, payload.get(_snake_to_camel(field)))
            if raw_probability is not None and _optional_float(raw_probability) != 0.0:
                raise RuntimeError(
                    f"F1 prediction service strategy driver {driver_number} exposed {field}"
                )
        raw_forecast_available = payload.get("forecast_available", payload.get("forecastAvailable"))
        raw_eligibility = payload.get("eligibility_status", payload.get("eligibilityStatus"))
        if raw_forecast_available is None or raw_eligibility is None:
            raise RuntimeError(
                f"F1 prediction service strategy driver {driver_number} is missing explicit target availability"
            )
        if _normalized_status(raw_eligibility) != "target_unavailable":
            raise RuntimeError(
                f"F1 prediction service strategy driver {driver_number} has invalid eligibility status"
            )
        raw_participation = payload.get("participation_status", payload.get("participationStatus"))
        local_participation = _target_status_for_driver(state, drivers_by_number[driver_number], "race")[2]
        if raw_participation is not None and _normalized_status(raw_participation) != local_participation:
            raise RuntimeError(
                f"F1 prediction service strategy driver {driver_number} participation status disagrees with local state"
            )
        if _payload_forecast_available(payload, {}):
            raise RuntimeError(
                f"F1 prediction service strategy driver {driver_number} claimed a position forecast"
            )
        _validate_remote_prediction_scalars(
            payload,
            driver_number=driver_number,
            distribution={},
            forecast_available=False,
            tolerance=2e-5,
        )
        _validated_strategy_payload(payload, state, drivers_by_number[driver_number])
        returned_drivers.add(driver_number)

    if returned_drivers != expected_drivers:
        missing = sorted(expected_drivers - returned_drivers)
        unexpected = sorted(returned_drivers - expected_drivers)
        raise RuntimeError(
            f"F1 prediction service strategy driver field mismatch; missing={missing}, unexpected={unexpected}"
        )


def _prediction_from_payload(
    payload: dict[str, Any],
    state: SessionSnapshot,
    *,
    prediction_kind: PredictionKind,
) -> PredictionSnapshot:
    driver_number = _required_int(payload.get("driver_number", payload.get("driverNumber")), "driver_number")
    driver = next((item for item in state.drivers if item.driver_number == driver_number), None)
    if driver is None:
        raise RuntimeError(f"F1 prediction service returned unexpected driver {driver_number}")
    strategy = _validated_strategy_payload(payload, state, driver)
    fallback_sequence = state.seq
    if prediction_kind == "strategy":
        return PredictionSnapshot(
            model_version=str(payload.get("model_version") or payload.get("modelVersion") or "remote_f1_model"),
            prediction_time=str(payload.get("prediction_time") or payload.get("predictionTime") or utc_now_iso()),
            source_event_sequence=_optional_int(
                payload.get("source_event_sequence", payload.get("sourceEventSequence"))
            )
            or fallback_sequence,
            features_version=str(
                payload.get("features_version") or payload.get("featuresVersion") or "remote_features"
            ),
            driver_number=driver_number,
            expected_position=None,
            position_distribution={},
            win_probability=0.0,
            podium_probability=0.0,
            points_probability=0.0,
            dnf_probability=0.0,
            confidence=_probability(payload.get("confidence"), default=0.0),
            position_p10=None,
            position_p90=None,
            prediction_kind=prediction_kind,
            position_semantics=position_semantics_for_kind(prediction_kind),
            strategy=strategy,
            forecast_available=False,
            unavailable_reason=str(
                payload.get("unavailable_reason", payload.get("unavailableReason"))
                or "strategy_is_a_decision_target_not_a_position_forecast"
            ),
            eligibility_status=_normalized_status(
                payload.get("eligibility_status", payload.get("eligibilityStatus"))
                or "target_unavailable"
            ),
            participation_status=_normalized_status(
                payload.get("participation_status", payload.get("participationStatus"))
                or _target_status_for_driver(state, driver, "race")[2]
            ),
        )

    distribution = _distribution_payload(payload.get("position_distribution", payload.get("positionDistribution")))
    forecast_available = _payload_forecast_available(payload, distribution)
    position_p10 = _position_percentile(distribution, 0.10) if forecast_available else None
    position_p90 = _position_percentile(distribution, 0.90) if forecast_available else None
    expected_position = (
        sum(float(position) * probability for position, probability in distribution.items())
        if forecast_available
        else None
    )
    return PredictionSnapshot(
        model_version=str(payload.get("model_version") or payload.get("modelVersion") or "remote_f1_model"),
        prediction_time=str(payload.get("prediction_time") or payload.get("predictionTime") or utc_now_iso()),
        source_event_sequence=_optional_int(payload.get("source_event_sequence", payload.get("sourceEventSequence")))
        or fallback_sequence,
        features_version=str(payload.get("features_version") or payload.get("featuresVersion") or "remote_features"),
        driver_number=driver_number,
        expected_position=round(expected_position, 6) if expected_position is not None else None,
        position_distribution=distribution,
        win_probability=_probability(payload.get("win_probability", payload.get("winProbability"))),
        podium_probability=_probability(payload.get("podium_probability", payload.get("podiumProbability"))),
        points_probability=_probability(payload.get("points_probability", payload.get("pointsProbability"))),
        dnf_probability=_probability(payload.get("dnf_probability", payload.get("dnfProbability"))),
        confidence=_probability(payload.get("confidence"), default=0.5 if forecast_available else 0.0),
        position_p10=position_p10,
        position_p90=position_p90,
        prediction_kind=prediction_kind,
        position_semantics=position_semantics_for_kind(prediction_kind),
        strategy=strategy,
        forecast_available=forecast_available,
        unavailable_reason=(
            None
            if forecast_available
            else str(payload.get("unavailable_reason", payload.get("unavailableReason")) or "target_unavailable")
        ),
        eligibility_status=_normalized_status(
            payload.get("eligibility_status", payload.get("eligibilityStatus"))
            or _target_status_for_driver(state, driver, prediction_kind)[1]
        ),
        participation_status=_normalized_status(
            payload.get("participation_status", payload.get("participationStatus"))
            or _target_status_for_driver(state, driver, prediction_kind)[2]
        ),
    )


def _raw_strategy_payload(payload: dict[str, Any]) -> Any:
    return payload.get("strategy", payload.get("strategy_payload", payload.get("strategyPayload")))


def _validated_strategy_payload(
    payload: dict[str, Any],
    state: SessionSnapshot,
    driver: DriverState,
) -> dict[str, Any] | None:
    value = _raw_strategy_payload(payload)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RuntimeError("F1 prediction service returned a non-object strategy payload")
    strategy = dict(value)
    safe = _optional_bool(strategy.get("safeToRecommend", strategy.get("safe_to_recommend")))
    if safe is not True:
        return _closed_strategy_payload(
            strategy,
            str(
                strategy.get("unavailableReason", strategy.get("strategy_unavailable_reason"))
                or "remote_strategy_not_explicitly_safe"
            ),
        )

    legality_state, missing = _local_strategy_legality_state(state, driver)
    if legality_state is None:
        return _closed_strategy_payload(
            strategy,
            "platform_local_legality_state_incomplete:" + ",".join(missing),
        )

    action_name = str(
        strategy.get("recommendedAction", strategy.get("recommended_action")) or ""
    ).strip().lower()
    next_compound = strategy.get("nextCompound", strategy.get("next_compound"))
    try:
        _, build_legal_action_mask = _load_legal_action_mask_components()
        legal_mask = build_legal_action_mask(legality_state)
    except Exception as exc:
        return _closed_strategy_payload(
            strategy,
            f"platform_local_legal_mask_unavailable:{type(exc).__name__}:{exc}",
        )
    try:
        if action_name not in {"stay_out", "pit_now", "pit_next_lap"}:
            raise ValueError(f"unsupported_strategy_action:{action_name or 'missing'}")
        pace_mode = _explicit_strategy_pace_mode(strategy)
        matching_actions = _matching_strategy_actions(
            [*legal_mask.legal_actions, *legal_mask.illegal_actions],
            action_name=action_name,
            next_compound=next_compound,
            pace_mode=pace_mode,
        )
        if not matching_actions:
            raise ValueError("no_action_space_variant_matches_strategy_action")
    except Exception as exc:
        return _closed_strategy_payload(
            strategy,
            f"platform_local_action_cannot_be_mapped:{type(exc).__name__}:{exc}",
        )
    compatible_legal_actions = [
        candidate for candidate in matching_actions if legal_mask.is_legal(candidate)
    ]
    if not compatible_legal_actions:
        illegal_reasons = sorted(
            {legal_mask.reason_for(candidate) for candidate in matching_actions}
        )
        reason = illegal_reasons[0] if len(illegal_reasons) == 1 else "no_compatible_legal_action"
        return _closed_strategy_payload(
            strategy,
            f"platform_local_action_illegal:{reason}",
        )

    compatible_keys = [candidate.key for candidate in compatible_legal_actions]
    return {
        **strategy,
        "recommendedAction": action_name,
        "availability": "available",
        "safeToRecommend": True,
        "unavailableReason": None,
        "paceMode": pace_mode,
        "compatibleLegalActionKeys": compatible_keys,
        "legalActionKey": compatible_keys[0] if pace_mode is not None else None,
        "legalActionMask": {
            "contract": "packages.f1.models.live_race.action_space.build_legal_action_mask",
            "legalActionCount": legal_mask.legal_count,
            "legalActionKeys": [action.key for action in legal_mask.legal_actions],
        },
        "legalityState": _strategy_legality_evidence(legality_state),
    }


def _explicit_strategy_pace_mode(strategy: dict[str, Any]) -> str | None:
    raw_mode = _mapping_value(
        strategy,
        ("paceMode", "pace_mode", "actionMode", "action_mode", "strategyMode", "strategy_mode", "mode"),
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


def _closed_strategy_payload(strategy: dict[str, Any], reason: str) -> dict[str, Any]:
    original = strategy.get("recommendedAction", strategy.get("recommended_action"))
    return {
        **strategy,
        "recommendedAction": None,
        "confidence": 0.0,
        "availability": "unavailable",
        "safeToRecommend": False,
        "unavailableReason": reason,
        "originalRecommendedAction": original,
        "paceMode": None,
        "compatibleLegalActionKeys": [],
        "legalActionKey": None,
        "legalActionMask": None,
        "legalityState": None,
    }


def _local_strategy_legality_state(
    state: SessionSnapshot,
    driver: DriverState,
) -> tuple[dict[str, Any] | None, list[str]]:
    target_available, _, participation, _ = _target_status_for_driver(state, driver, "race")
    if not target_available or participation in {"retired_or_stopped", "finished"}:
        return None, [f"participation_status:{participation}"]

    info = state.session_info if isinstance(state.session_info, dict) else {}
    current_lap = driver.current_lap
    tyre_age = driver.tyre_age
    stint_number = driver.stint_number
    if stint_number is None:
        driver_segments = [
            segment for segment in state.strategy_timeline if segment.driver_number == driver.driver_number
        ]
        if driver_segments:
            stint_number = max(segment.stint_number for segment in driver_segments)
    total_laps = _optional_int(_mapping_value(info, ("total_laps", "totalLaps", "scheduled_laps")))
    remaining_laps = _optional_int(_mapping_value(info, ("remaining_laps", "remainingLaps", "laps_remaining")))
    compound = _normalize_compound(driver.current_compound)
    used_compounds = _compound_list(
        [
            segment.compound
            for segment in state.strategy_timeline
            if segment.driver_number == driver.driver_number
        ]
    )
    available_compounds = _compound_list(
        _mapping_value(info, ("available_compounds", "availableCompounds", "allowed_compounds"))
    )
    pit_lane_open = _optional_bool(_mapping_value(info, ("pit_lane_open", "pitLaneOpen")))
    if pit_lane_open is None:
        pit_lane_open = _pit_lane_state_from_messages(state.race_control)
    is_red = _optional_bool(_mapping_value(info, ("is_red", "isRed")))
    if is_red is None:
        is_red = _red_state_from_track_status(driver.track_status)
    is_wet = _optional_bool(
        _mapping_value(info, ("is_wet_track", "isWetTrack", "weather_is_wet"))
    )
    if is_wet is None and isinstance(state.weather, dict):
        rainfall = _optional_float(state.weather.get("rainfall"))
        if rainfall is not None:
            is_wet = rainfall > 0.0
    is_box_lap = _box_state_from_pit_status(driver.pit_status)

    missing: list[str] = []
    for field, value in (
        ("current_lap", current_lap),
        ("tyre_age", tyre_age),
        ("stint_number", stint_number),
        ("total_laps", total_laps),
        ("remaining_laps", remaining_laps),
        ("pit_lane_open", pit_lane_open),
        ("red_flag_state", is_red),
        ("wet_track_state", is_wet),
        ("box_or_pit_state", is_box_lap),
    ):
        if value is None:
            missing.append(field)
    if compound == "UNKNOWN":
        missing.append("current_compound")
    if not used_compounds or "UNKNOWN" in used_compounds:
        missing.append("used_compounds")
    if not available_compounds or "UNKNOWN" in available_compounds:
        missing.append("available_compounds")
    if missing:
        return None, missing

    return {
        "lap_number": int(current_lap),
        "total_laps": int(total_laps),
        "remaining_laps": int(remaining_laps),
        "stint_id": int(stint_number),
        "compound": compound,
        "tyre_age": int(tyre_age),
        "used_compounds": used_compounds,
        "available_compounds": available_compounds,
        "pit_lane_open": bool(pit_lane_open),
        "is_red": bool(is_red),
        "is_wet_track": bool(is_wet),
        "weather_is_wet": bool(is_wet),
        "is_box_lap": bool(is_box_lap),
        "track_status": str(driver.track_status or ""),
        "metadata": {"available_compounds": available_compounds},
    }, []


def _strategy_legality_evidence(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "lapNumber": state["lap_number"],
        "totalLaps": state["total_laps"],
        "remainingLaps": state["remaining_laps"],
        "tyreAge": state["tyre_age"],
        "stintNumber": state["stint_id"],
        "currentCompound": state["compound"],
        "usedCompounds": list(state["used_compounds"]),
        "availableCompounds": list(state["available_compounds"]),
        "pitLaneOpen": state["pit_lane_open"],
        "isRed": state["is_red"],
        "isWetTrack": state["is_wet_track"],
        "isBoxLap": state["is_box_lap"],
    }


def _load_legal_action_mask_components():
    try:
        from packages.f1.models.live_race.action_space import StrategyAction, build_legal_action_mask

        return StrategyAction, build_legal_action_mask
    except Exception:
        path = _repo_root() / "packages" / "f1" / "models" / "live_race" / "action_space.py"
        name = "_f1_platform_live_race_action_space"
        module = sys.modules.get(name)
        if module is None:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load legal action mask from {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return module.StrategyAction, module.build_legal_action_mask


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "packages" / "f1" / "models" / "live_race" / "action_space.py").exists():
            return parent
    raise RuntimeError("Could not locate repo root containing packages/f1")


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


def _mapping_value(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _compound_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = [item.strip() for item in value.replace("|", ",").split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw = [str(item).strip() for item in value]
    else:
        raw = [str(value).strip()]
    normalized = [_normalize_compound(item) for item in raw if item]
    return list(dict.fromkeys(item for item in normalized if item != "UNKNOWN"))


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


def _pit_lane_state_from_messages(messages: list[dict[str, Any]]) -> bool | None:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        text = " ".join(str(value).lower() for value in message.values())
        if "pit lane closed" in text:
            return False
        if "pit lane open" in text:
            return True
    return None


def _red_state_from_track_status(value: Any) -> bool | None:
    text = str(value or "").strip().lower().replace("_", " ")
    if not text:
        return None
    if text.isdigit():
        return text == "5"
    if "red" in text:
        return True
    if any(marker in text for marker in ("green", "clear", "yellow", "safety car", "vsc")):
        return False
    return None


def _box_state_from_pit_status(value: Any) -> bool | None:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip().lower().replace("_", " ")
    if text == "pit" or text.startswith("pit ") or any(
        marker in text for marker in ("box", "pit lane", "in pit", "pitting")
    ):
        return True
    return False


def _normalized_status(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _snake_to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


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
    if not math.isfinite(number):
        return None
    return number


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        number = _optional_float(value)
        return None if number is None else number != 0.0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "open", "green", "wet"}:
        return True
    if text in {"0", "false", "no", "n", "closed", "dry"}:
        return False
    return None


def _probability(value: Any, *, default: float = 0.0) -> float:
    number = _optional_float(value)
    if number is None:
        return default
    return round(max(0.0, min(1.0, number)), 12)
