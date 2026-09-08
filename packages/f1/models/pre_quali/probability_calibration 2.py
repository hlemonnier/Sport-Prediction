"""Diagnostic calibration for full-field Qualifying position probabilities.

The input for one event is a square matrix whose rows are drivers and whose
columns are finishing positions.  Such a matrix must be doubly stochastic:
every driver finishes somewhere and every position is occupied exactly once.

Calibration raises every positive probability to ``1 / temperature`` in log
space, then uses a Sinkhorn projection to restore the row and column
constraints.  Temperature selection is deliberately low capacity: a frozen,
bounded grid is scored with event-balanced negative log likelihood on complete
declared calibration events only.

This module is diagnostic infrastructure.  It does not make a promotion claim
and it intentionally keeps outcomes out of the transform API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np


CALIBRATION_MODEL_FAMILY = "qualifying_position_temperature_sinkhorn"
CALIBRATION_SCHEMA_VERSION = 1
DIAGNOSTIC_PROMOTION_STATUS = "diagnostic_only_not_promoted"

# Fixed by design.  Callers cannot tune the search range on an audit set.
TEMPERATURE_GRID: tuple[float, ...] = (0.50, 2.0 / 3.0, 0.80, 1.00, 1.25, 1.50, 2.00)
MIN_TEMPERATURE = min(TEMPERATURE_GRID)
MAX_TEMPERATURE = max(TEMPERATURE_GRID)

_STOCHASTIC_TOLERANCE = 1e-7
_SINKHORN_TOLERANCE = 1e-12
_SINKHORN_MAX_ITERATIONS = 10_000
_LOG_LOSS_FLOOR = 1e-15
_TEMPERATURE_TIE_TOLERANCE = 1e-12
_ECE_BIN_COUNT = 10


@dataclass(frozen=True)
class QualifyingPositionProbabilityMatrix:
    """Uncalibrated full-field position marginals for one event.

    ``driver_ids`` defines row order and ``position_ids`` defines column order.
    Positions must be exactly ``1..field_size``.  Provenance is mandatory so a
    fitted calibration card can identify every prediction snapshot it read.
    """

    event_key: str | int
    driver_ids: tuple[str, ...]
    position_ids: tuple[int, ...]
    probabilities: np.ndarray
    source_model_id: str
    prediction_evidence_id: str


@dataclass(frozen=True)
class QualifyingPositionOutcome:
    """Complete official Qualifying order for one event."""

    event_key: str | int
    driver_ids: tuple[str, ...]
    actual_positions: tuple[int, ...]
    outcome_evidence_id: str


@dataclass(frozen=True)
class CalibrationEventProvenance:
    event_key: str
    field_size: int
    prediction_evidence_id: str
    outcome_evidence_id: str


@dataclass(frozen=True)
class TemperatureCandidateScore:
    temperature: float
    event_balanced_negative_log_likelihood: float


@dataclass(frozen=True)
class QualifyingProbabilityCalibrationCard:
    model_id: str
    model_family: str
    schema_version: int
    promotion_status: str
    source_model_id: str
    selected_temperature: float
    candidate_grid: tuple[float, ...]
    candidate_scores: tuple[TemperatureCandidateScore, ...]
    calibration_event_keys: tuple[str, ...]
    calibration_event_count: int
    event_provenance: tuple[CalibrationEventProvenance, ...]
    calibration_data_sha256: str
    objective: str
    tie_break_policy: str
    transform: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable model-card payload."""

        return asdict(self)


@dataclass(frozen=True)
class CalibratedQualifyingPositionProbabilityMatrix:
    event_key: str
    driver_ids: tuple[str, ...]
    position_ids: tuple[int, ...]
    probabilities: np.ndarray
    source_model_id: str
    prediction_evidence_id: str
    calibration_model_id: str
    temperature: float


@dataclass(frozen=True)
class QualifyingProbabilityAuditMetrics:
    """Event-balanced proper scores and fixed-bin top-k calibration errors."""

    event_keys: tuple[str, ...]
    event_count: int
    driver_count: int
    multiclass_log_loss: float
    normalized_multiclass_brier: float
    top1_ece: float
    top3_ece: float
    top10_ece: float
    ece_bin_count: int
    calibration_model_id: str | None


@dataclass(frozen=True)
class _ValidatedMatrix:
    event_key: str
    driver_ids: tuple[str, ...]
    position_ids: tuple[int, ...]
    probabilities: np.ndarray
    source_model_id: str
    prediction_evidence_id: str


@dataclass(frozen=True)
class _ValidatedOutcome:
    event_key: str
    driver_ids: tuple[str, ...]
    actual_positions: tuple[int, ...]
    outcome_evidence_id: str


def _nonempty_text(value: object, *, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def _event_key(value: object) -> str:
    if isinstance(value, bool):
        raise ValueError("event_key must not be boolean")
    return _nonempty_text(value, field_name="event_key")


def _driver_ids(values: Sequence[object], *, event_key: str) -> tuple[str, ...]:
    drivers = tuple(_nonempty_text(value, field_name="driver_id") for value in values)
    if len(drivers) < 2:
        raise ValueError(f"event {event_key}: a full field requires at least two drivers")
    if len(set(drivers)) != len(drivers):
        raise ValueError(f"event {event_key}: driver_ids must be unique")
    return drivers


def _integral_positions(values: Sequence[object], *, field_name: str, event_key: str) -> tuple[int, ...]:
    positions: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"event {event_key}: {field_name} must contain integer positions")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"event {event_key}: {field_name} must contain integer positions"
            ) from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"event {event_key}: {field_name} must contain integer positions")
        positions.append(int(numeric))
    return tuple(positions)


def _validate_numeric_probability_matrix(
    values: np.ndarray,
    *,
    event_key: str,
    expected_field_size: int | None = None,
) -> np.ndarray:
    try:
        probabilities = np.array(values, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"event {event_key}: probabilities must be numeric") from exc
    if probabilities.ndim != 2 or probabilities.shape[0] != probabilities.shape[1]:
        raise ValueError(f"event {event_key}: probabilities must be a square two-dimensional matrix")
    if expected_field_size is not None and probabilities.shape != (
        expected_field_size,
        expected_field_size,
    ):
        raise ValueError(
            f"event {event_key}: probability shape {probabilities.shape} does not match "
            f"field size {expected_field_size}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError(f"event {event_key}: probabilities must all be finite")
    if (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise ValueError(f"event {event_key}: probabilities must lie in [0, 1]")

    row_sums = probabilities.sum(axis=1)
    column_sums = probabilities.sum(axis=0)
    if not np.allclose(row_sums, 1.0, atol=_STOCHASTIC_TOLERANCE, rtol=0.0):
        raise ValueError(f"event {event_key}: every driver row must sum to one")
    if not np.allclose(column_sums, 1.0, atol=_STOCHASTIC_TOLERANCE, rtol=0.0):
        raise ValueError(f"event {event_key}: every position column must sum to one")
    return probabilities


def _validate_matrix(matrix: QualifyingPositionProbabilityMatrix) -> _ValidatedMatrix:
    key = _event_key(matrix.event_key)
    drivers = _driver_ids(matrix.driver_ids, event_key=key)
    positions = _integral_positions(matrix.position_ids, field_name="position_ids", event_key=key)
    expected_positions = tuple(range(1, len(drivers) + 1))
    if positions != expected_positions:
        raise ValueError(f"event {key}: position_ids must be exactly 1..field_size in column order")
    probabilities = _validate_numeric_probability_matrix(
        matrix.probabilities,
        event_key=key,
        expected_field_size=len(drivers),
    )
    probabilities.setflags(write=False)
    return _ValidatedMatrix(
        event_key=key,
        driver_ids=drivers,
        position_ids=positions,
        probabilities=probabilities,
        source_model_id=_nonempty_text(matrix.source_model_id, field_name="source_model_id"),
        prediction_evidence_id=_nonempty_text(
            matrix.prediction_evidence_id,
            field_name="prediction_evidence_id",
        ),
    )


def _validate_outcome(outcome: QualifyingPositionOutcome) -> _ValidatedOutcome:
    key = _event_key(outcome.event_key)
    drivers = _driver_ids(outcome.driver_ids, event_key=key)
    positions = _integral_positions(
        outcome.actual_positions,
        field_name="actual_positions",
        event_key=key,
    )
    if len(positions) != len(drivers):
        raise ValueError(f"event {key}: every driver must have one actual position")
    if tuple(sorted(positions)) != tuple(range(1, len(drivers) + 1)):
        raise ValueError(f"event {key}: actual_positions must be a complete field permutation")
    return _ValidatedOutcome(
        event_key=key,
        driver_ids=drivers,
        actual_positions=positions,
        outcome_evidence_id=_nonempty_text(
            outcome.outcome_evidence_id,
            field_name="outcome_evidence_id",
        ),
    )


def _aligned_actual_positions(matrix: _ValidatedMatrix, outcome: _ValidatedOutcome) -> np.ndarray:
    if matrix.event_key != outcome.event_key:
        raise ValueError("matrix and outcome event keys do not match")
    if set(matrix.driver_ids) != set(outcome.driver_ids):
        missing = sorted(set(matrix.driver_ids) - set(outcome.driver_ids))
        extra = sorted(set(outcome.driver_ids) - set(matrix.driver_ids))
        raise ValueError(
            f"event {matrix.event_key}: prediction and outcome fields differ; "
            f"missing={missing}, extra={extra}"
        )
    # Length equality is established by ``_validate_outcome`` above.  Avoid
    # ``zip(..., strict=True)`` so the research runtime remains compatible with
    # the repository's supported Python 3.9 environment.
    position_by_driver = dict(zip(outcome.driver_ids, outcome.actual_positions))
    aligned = np.array([position_by_driver[driver] for driver in matrix.driver_ids], dtype=int)
    if tuple(sorted(aligned.tolist())) != matrix.position_ids:
        raise ValueError(f"event {matrix.event_key}: outcome positions do not match prediction columns")
    return aligned


def temperature_scale_and_sinkhorn(probabilities: np.ndarray, *, temperature: float) -> np.ndarray:
    """Apply log-temperature scaling and a deterministic Sinkhorn projection.

    The input must already represent valid full-order marginals.  Temperature
    scaling generally breaks the marginal constraints; alternating row/column
    normalization projects it back onto the doubly stochastic polytope.
    """

    if not np.isfinite(temperature) or not MIN_TEMPERATURE <= float(temperature) <= MAX_TEMPERATURE:
        raise ValueError(
            f"temperature must be finite and bounded in [{MIN_TEMPERATURE}, {MAX_TEMPERATURE}]"
        )
    base = _validate_numeric_probability_matrix(probabilities, event_key="transform")
    positive = base > 0.0
    log_scaled = np.full(base.shape, -np.inf, dtype=float)
    log_scaled[positive] = np.log(base[positive]) / float(temperature)
    finite_values = log_scaled[np.isfinite(log_scaled)]
    if finite_values.size == 0:
        raise ValueError("probability support is empty")

    # A global offset preserves the Sinkhorn result while avoiding overflow.
    scaled = np.zeros(base.shape, dtype=float)
    scaled[positive] = np.exp(log_scaled[positive] - float(finite_values.max()))
    if (scaled.sum(axis=1) <= 0.0).any() or (scaled.sum(axis=0) <= 0.0).any():
        raise ValueError("probability support does not cover every driver and position")

    projected = scaled
    for _ in range(_SINKHORN_MAX_ITERATIONS):
        row_sums = projected.sum(axis=1)
        if (row_sums <= 0.0).any() or not np.isfinite(row_sums).all():
            raise ValueError("Sinkhorn row normalization encountered invalid support")
        projected = projected / row_sums[:, np.newaxis]

        column_sums = projected.sum(axis=0)
        if (column_sums <= 0.0).any() or not np.isfinite(column_sums).all():
            raise ValueError("Sinkhorn column normalization encountered invalid support")
        projected = projected / column_sums[np.newaxis, :]

        error = max(
            float(np.max(np.abs(projected.sum(axis=1) - 1.0))),
            float(np.max(np.abs(projected.sum(axis=0) - 1.0))),
        )
        if error <= _SINKHORN_TOLERANCE:
            result = np.array(projected, copy=True)
            result.setflags(write=False)
            return result
    raise RuntimeError("Sinkhorn projection did not converge")


def _sequence_index(items: Sequence[object], *, collection_name: str) -> Mapping[str, object]:
    indexed: dict[str, object] = {}
    for item in items:
        if not hasattr(item, "event_key"):
            raise TypeError(f"{collection_name} entries must expose event_key")
        key = _event_key(getattr(item, "event_key"))
        if key in indexed:
            raise ValueError(f"{collection_name} contains duplicate event_key {key}")
        indexed[key] = item
    return indexed


def _declared_event_keys(values: Sequence[str | int], *, minimum_count: int) -> tuple[str, ...]:
    keys = tuple(_event_key(value) for value in values)
    if len(set(keys)) != len(keys):
        raise ValueError("declared event keys must be unique")
    if len(keys) < minimum_count:
        raise ValueError(f"at least {minimum_count} independent complete events are required")
    return tuple(sorted(keys))


def _declared_complete_events(
    matrices: Sequence[QualifyingPositionProbabilityMatrix],
    outcomes: Sequence[QualifyingPositionOutcome],
    event_keys: Sequence[str | int],
    *,
    minimum_count: int,
) -> tuple[tuple[str, _ValidatedMatrix, _ValidatedOutcome, np.ndarray], ...]:
    keys = _declared_event_keys(event_keys, minimum_count=minimum_count)
    matrix_index = _sequence_index(matrices, collection_name="matrices")
    outcome_index = _sequence_index(outcomes, collection_name="outcomes")
    prepared: list[tuple[str, _ValidatedMatrix, _ValidatedOutcome, np.ndarray]] = []
    for key in keys:
        if key not in matrix_index:
            raise ValueError(f"declared event {key} is missing a probability matrix")
        if key not in outcome_index:
            raise ValueError(f"declared event {key} is missing a complete outcome")
        matrix = _validate_matrix(matrix_index[key])  # type: ignore[arg-type]
        outcome = _validate_outcome(outcome_index[key])  # type: ignore[arg-type]
        actual_positions = _aligned_actual_positions(matrix, outcome)
        prepared.append((key, matrix, outcome, actual_positions))
    return tuple(prepared)


def _event_negative_log_likelihood(probabilities: np.ndarray, actual_positions: np.ndarray) -> float:
    rows = np.arange(len(actual_positions), dtype=int)
    actual_probabilities = probabilities[rows, actual_positions - 1]
    return float(-np.log(np.clip(actual_probabilities, _LOG_LOSS_FLOOR, 1.0)).mean())


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _calibration_data_payload(
    prepared: Sequence[tuple[str, _ValidatedMatrix, _ValidatedOutcome, np.ndarray]],
) -> list[dict[str, object]]:
    return [
        {
            "event_key": key,
            "driver_ids": list(matrix.driver_ids),
            "position_ids": list(matrix.position_ids),
            "probabilities": matrix.probabilities.tolist(),
            "source_model_id": matrix.source_model_id,
            "prediction_evidence_id": matrix.prediction_evidence_id,
            "actual_positions_aligned": actual_positions.tolist(),
            "outcome_evidence_id": outcome.outcome_evidence_id,
        }
        for key, matrix, outcome, actual_positions in prepared
    ]


@dataclass(frozen=True)
class QualifyingProbabilityCalibrator:
    """Fitted diagnostic calibrator; transformation has no outcome argument."""

    temperature: float
    model_card: QualifyingProbabilityCalibrationCard

    def transform(
        self,
        matrix: QualifyingPositionProbabilityMatrix,
    ) -> CalibratedQualifyingPositionProbabilityMatrix:
        validated = _validate_matrix(matrix)
        if validated.source_model_id != self.model_card.source_model_id:
            raise ValueError(
                "transform source_model_id does not match the model calibrated by this card"
            )
        probabilities = temperature_scale_and_sinkhorn(
            validated.probabilities,
            temperature=self.temperature,
        )
        return CalibratedQualifyingPositionProbabilityMatrix(
            event_key=validated.event_key,
            driver_ids=validated.driver_ids,
            position_ids=validated.position_ids,
            probabilities=probabilities,
            source_model_id=validated.source_model_id,
            prediction_evidence_id=validated.prediction_evidence_id,
            calibration_model_id=self.model_card.model_id,
            temperature=self.temperature,
        )


def fit_qualifying_probability_calibrator(
    matrices: Sequence[QualifyingPositionProbabilityMatrix],
    outcomes: Sequence[QualifyingPositionOutcome],
    *,
    calibration_event_keys: Sequence[str | int],
) -> QualifyingProbabilityCalibrator:
    """Fit the frozen temperature grid on declared complete events only."""

    prepared = _declared_complete_events(
        matrices,
        outcomes,
        calibration_event_keys,
        minimum_count=2,
    )
    source_models = {matrix.source_model_id for _, matrix, _, _ in prepared}
    if len(source_models) != 1:
        raise ValueError("all calibration events must come from one source_model_id")
    source_model_id = next(iter(source_models))

    candidate_scores: list[TemperatureCandidateScore] = []
    for temperature in TEMPERATURE_GRID:
        event_losses = [
            _event_negative_log_likelihood(
                temperature_scale_and_sinkhorn(matrix.probabilities, temperature=temperature),
                actual_positions,
            )
            for _, matrix, _, actual_positions in prepared
        ]
        candidate_scores.append(
            TemperatureCandidateScore(
                temperature=float(temperature),
                event_balanced_negative_log_likelihood=float(np.mean(event_losses)),
            )
        )

    best_score = min(score.event_balanced_negative_log_likelihood for score in candidate_scores)
    tied = [
        score
        for score in candidate_scores
        if score.event_balanced_negative_log_likelihood <= best_score + _TEMPERATURE_TIE_TOLERANCE
    ]
    selected = min(tied, key=lambda score: (abs(score.temperature - 1.0), score.temperature))

    data_sha256 = _canonical_sha256(_calibration_data_payload(prepared))
    model_payload = {
        "family": CALIBRATION_MODEL_FAMILY,
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "source_model_id": source_model_id,
        "temperature": selected.temperature,
        "candidate_grid": TEMPERATURE_GRID,
        "calibration_data_sha256": data_sha256,
    }
    model_id = f"{CALIBRATION_MODEL_FAMILY}_v{CALIBRATION_SCHEMA_VERSION}_{_canonical_sha256(model_payload)[:12]}"
    event_provenance = tuple(
        CalibrationEventProvenance(
            event_key=key,
            field_size=len(matrix.driver_ids),
            prediction_evidence_id=matrix.prediction_evidence_id,
            outcome_evidence_id=outcome.outcome_evidence_id,
        )
        for key, matrix, outcome, _ in prepared
    )
    card = QualifyingProbabilityCalibrationCard(
        model_id=model_id,
        model_family=CALIBRATION_MODEL_FAMILY,
        schema_version=CALIBRATION_SCHEMA_VERSION,
        promotion_status=DIAGNOSTIC_PROMOTION_STATUS,
        source_model_id=source_model_id,
        selected_temperature=selected.temperature,
        candidate_grid=TEMPERATURE_GRID,
        candidate_scores=tuple(candidate_scores),
        calibration_event_keys=tuple(key for key, _, _, _ in prepared),
        calibration_event_count=len(prepared),
        event_provenance=event_provenance,
        calibration_data_sha256=data_sha256,
        objective="mean_of_complete_event_mean_driver_negative_log_likelihood",
        tie_break_policy="within_1e-12_of_best_choose_temperature_closest_to_1_then_lower",
        transform="log_probability_divided_by_temperature_then_sinkhorn_projection",
    )
    return QualifyingProbabilityCalibrator(temperature=selected.temperature, model_card=card)


def _fixed_bin_ece(probabilities: np.ndarray, outcomes: np.ndarray, weights: np.ndarray) -> float:
    edges = np.linspace(0.0, 1.0, _ECE_BIN_COUNT + 1)
    bin_ids = np.minimum(np.searchsorted(edges, probabilities, side="right") - 1, _ECE_BIN_COUNT - 1)
    bin_ids = np.maximum(bin_ids, 0)
    ece = 0.0
    for bin_id in range(_ECE_BIN_COUNT):
        mask = bin_ids == bin_id
        bin_weight = float(weights[mask].sum())
        if bin_weight <= 0.0:
            continue
        predicted = float(np.average(probabilities[mask], weights=weights[mask]))
        observed = float(np.average(outcomes[mask], weights=weights[mask]))
        ece += bin_weight * abs(predicted - observed)
    return float(ece)


def audit_qualifying_position_probabilities(
    matrices: Sequence[QualifyingPositionProbabilityMatrix],
    outcomes: Sequence[QualifyingPositionOutcome],
    *,
    audit_event_keys: Sequence[str | int],
    calibrator: QualifyingProbabilityCalibrator | None = None,
) -> QualifyingProbabilityAuditMetrics:
    """Audit raw or transformed matrices on an explicit complete event block."""

    prepared = _declared_complete_events(
        matrices,
        outcomes,
        audit_event_keys,
        minimum_count=1,
    )
    event_losses: list[float] = []
    event_briers: list[float] = []
    top_k_probabilities: dict[int, list[np.ndarray]] = {1: [], 3: [], 10: []}
    top_k_outcomes: dict[int, list[np.ndarray]] = {1: [], 3: [], 10: []}
    top_k_weights: dict[int, list[np.ndarray]] = {1: [], 3: [], 10: []}
    event_count = len(prepared)
    driver_count = 0

    for _, matrix, _, actual_positions in prepared:
        if calibrator is None:
            probabilities = matrix.probabilities
        else:
            raw = QualifyingPositionProbabilityMatrix(
                event_key=matrix.event_key,
                driver_ids=matrix.driver_ids,
                position_ids=matrix.position_ids,
                probabilities=matrix.probabilities,
                source_model_id=matrix.source_model_id,
                prediction_evidence_id=matrix.prediction_evidence_id,
            )
            probabilities = calibrator.transform(raw).probabilities

        event_losses.append(_event_negative_log_likelihood(probabilities, actual_positions))
        labels = np.eye(len(actual_positions), dtype=float)[actual_positions - 1]
        # Dividing the multiclass Brier score by two maps its theoretical
        # [0, 2] range to [0, 1].
        event_briers.append(float((np.square(probabilities - labels).sum(axis=1) / 2.0).mean()))
        driver_count += len(actual_positions)

        event_driver_weight = 1.0 / (event_count * len(actual_positions))
        for requested_k in (1, 3, 10):
            k = min(requested_k, len(actual_positions))
            top_k_probabilities[requested_k].append(probabilities[:, :k].sum(axis=1))
            top_k_outcomes[requested_k].append((actual_positions <= k).astype(float))
            top_k_weights[requested_k].append(
                np.full(len(actual_positions), event_driver_weight, dtype=float)
            )

    eces: dict[int, float] = {}
    for requested_k in (1, 3, 10):
        eces[requested_k] = _fixed_bin_ece(
            np.concatenate(top_k_probabilities[requested_k]),
            np.concatenate(top_k_outcomes[requested_k]),
            np.concatenate(top_k_weights[requested_k]),
        )
    return QualifyingProbabilityAuditMetrics(
        event_keys=tuple(key for key, _, _, _ in prepared),
        event_count=event_count,
        driver_count=driver_count,
        multiclass_log_loss=float(np.mean(event_losses)),
        normalized_multiclass_brier=float(np.mean(event_briers)),
        top1_ece=eces[1],
        top3_ece=eces[3],
        top10_ece=eces[10],
        ece_bin_count=_ECE_BIN_COUNT,
        calibration_model_id=calibrator.model_card.model_id if calibrator is not None else None,
    )


__all__ = [
    "CALIBRATION_MODEL_FAMILY",
    "CALIBRATION_SCHEMA_VERSION",
    "DIAGNOSTIC_PROMOTION_STATUS",
    "MAX_TEMPERATURE",
    "MIN_TEMPERATURE",
    "TEMPERATURE_GRID",
    "CalibratedQualifyingPositionProbabilityMatrix",
    "CalibrationEventProvenance",
    "QualifyingPositionOutcome",
    "QualifyingPositionProbabilityMatrix",
    "QualifyingProbabilityAuditMetrics",
    "QualifyingProbabilityCalibrationCard",
    "QualifyingProbabilityCalibrator",
    "TemperatureCandidateScore",
    "audit_qualifying_position_probabilities",
    "fit_qualifying_probability_calibrator",
    "temperature_scale_and_sinkhorn",
]
