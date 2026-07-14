from __future__ import annotations

import inspect

import numpy as np
import pytest

from packages.f1.models.pre_quali.probability_calibration import (
    DIAGNOSTIC_PROMOTION_STATUS,
    MAX_TEMPERATURE,
    MIN_TEMPERATURE,
    TEMPERATURE_GRID,
    QualifyingPositionOutcome,
    QualifyingPositionProbabilityMatrix,
    audit_qualifying_position_probabilities,
    fit_qualifying_probability_calibrator,
    temperature_scale_and_sinkhorn,
)


def _matrix(
    event_key: str,
    *,
    field_size: int = 4,
    diagonal_weight: float = 0.65,
    source_model_id: str = "qualifying-joint-samples-v4",
    probabilities: np.ndarray | None = None,
) -> QualifyingPositionProbabilityMatrix:
    drivers = tuple(f"{event_key}-driver-{index}" for index in range(field_size))
    if probabilities is None:
        probabilities = (
            diagonal_weight * np.eye(field_size)
            + (1.0 - diagonal_weight) * np.full((field_size, field_size), 1.0 / field_size)
        )
    return QualifyingPositionProbabilityMatrix(
        event_key=event_key,
        driver_ids=drivers,
        position_ids=tuple(range(1, field_size + 1)),
        probabilities=probabilities,
        source_model_id=source_model_id,
        prediction_evidence_id=f"prediction-snapshot-{event_key}",
    )


def _outcome(
    matrix: QualifyingPositionProbabilityMatrix,
    *,
    positions: tuple[int, ...] | None = None,
) -> QualifyingPositionOutcome:
    return QualifyingPositionOutcome(
        event_key=matrix.event_key,
        driver_ids=matrix.driver_ids,
        actual_positions=positions or matrix.position_ids,
        outcome_evidence_id=f"official-result-{matrix.event_key}",
    )


def _fit_pair():
    matrices = [_matrix("2026-01"), _matrix("2026-02", diagonal_weight=0.55)]
    outcomes = [_outcome(matrix) for matrix in matrices]
    return fit_qualifying_probability_calibrator(
        matrices,
        outcomes,
        calibration_event_keys=["2026-01", "2026-02"],
    ), matrices, outcomes


def test_temperature_transform_is_doubly_stochastic_and_identity_at_one() -> None:
    matrix = _matrix("identity")
    identity = temperature_scale_and_sinkhorn(matrix.probabilities, temperature=1.0)
    sharpened = temperature_scale_and_sinkhorn(matrix.probabilities, temperature=0.5)

    np.testing.assert_allclose(identity, matrix.probabilities, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(sharpened.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(sharpened.sum(axis=0), 1.0, atol=1e-12, rtol=0.0)
    assert (sharpened >= 0.0).all()
    assert not sharpened.flags.writeable


def test_transform_is_deterministic_and_has_no_outcome_parameter() -> None:
    calibrator, matrices, outcomes = _fit_pair()
    first = calibrator.transform(matrices[0])
    second = calibrator.transform(matrices[0])

    assert tuple(inspect.signature(calibrator.transform).parameters) == ("matrix",)
    np.testing.assert_array_equal(first.probabilities, second.probabilities)
    assert first.calibration_model_id == second.calibration_model_id
    with pytest.raises(TypeError):
        calibrator.transform(matrices[0], outcomes[0])  # type: ignore[call-arg]


def test_fit_uses_only_declared_calibration_events() -> None:
    first = _matrix("2026-01", diagonal_weight=0.55)
    second = _matrix("2026-02", diagonal_weight=0.60)
    unrelated = _matrix("2026-99", diagonal_weight=0.98)
    declared_matrices = [first, second]
    declared_outcomes = [_outcome(first), _outcome(second)]
    with_unrelated = fit_qualifying_probability_calibrator(
        [unrelated, second, first],
        [
            _outcome(unrelated, positions=tuple(reversed(unrelated.position_ids))),
            *reversed(declared_outcomes),
        ],
        calibration_event_keys=["2026-02", "2026-01"],
    )
    declared_only = fit_qualifying_probability_calibrator(
        declared_matrices,
        declared_outcomes,
        calibration_event_keys=["2026-01", "2026-02"],
    )

    assert with_unrelated.temperature == declared_only.temperature
    assert with_unrelated.model_card.candidate_scores == declared_only.model_card.candidate_scores
    assert with_unrelated.model_card.calibration_data_sha256 == (
        declared_only.model_card.calibration_data_sha256
    )
    assert with_unrelated.model_card.model_id == declared_only.model_card.model_id
    assert with_unrelated.model_card.calibration_event_keys == ("2026-01", "2026-02")


def test_uniform_probabilities_tie_break_to_identity_temperature() -> None:
    uniform = np.full((4, 4), 0.25)
    matrices = [
        _matrix("2026-01", probabilities=uniform),
        _matrix("2026-02", probabilities=uniform),
    ]
    calibrator = fit_qualifying_probability_calibrator(
        matrices,
        [_outcome(matrix) for matrix in matrices],
        calibration_event_keys=[matrix.event_key for matrix in matrices],
    )

    assert calibrator.temperature == 1.0
    assert 1.0 in TEMPERATURE_GRID
    assert min(TEMPERATURE_GRID) == MIN_TEMPERATURE == 0.5
    assert max(TEMPERATURE_GRID) == MAX_TEMPERATURE == 2.0


def test_fit_fails_closed_with_less_than_two_independent_events() -> None:
    matrix = _matrix("2026-01")
    with pytest.raises(ValueError, match="at least 2 independent complete events"):
        fit_qualifying_probability_calibrator(
            [matrix],
            [_outcome(matrix)],
            calibration_event_keys=["2026-01"],
        )
    with pytest.raises(ValueError, match="unique"):
        fit_qualifying_probability_calibrator(
            [matrix],
            [_outcome(matrix)],
            calibration_event_keys=["2026-01", "2026-01"],
        )


def test_declared_incomplete_or_inconsistent_fields_fail_closed() -> None:
    first = _matrix("2026-01")
    second = _matrix("2026-02")
    incomplete = QualifyingPositionOutcome(
        event_key=second.event_key,
        driver_ids=second.driver_ids[:-1],
        actual_positions=(1, 2, 3),
        outcome_evidence_id="incomplete-result",
    )
    with pytest.raises(ValueError, match="fields differ"):
        fit_qualifying_probability_calibrator(
            [first, second],
            [_outcome(first), incomplete],
            calibration_event_keys=[first.event_key, second.event_key],
        )

    other_model = _matrix("2026-03", source_model_id="different-model-version")
    with pytest.raises(ValueError, match="one source_model_id"):
        fit_qualifying_probability_calibrator(
            [first, other_model],
            [_outcome(first), _outcome(other_model)],
            calibration_event_keys=[first.event_key, other_model.event_key],
        )


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        (np.ones((3, 2)) / 2.0, "square"),
        (np.array([[1.1, -0.1], [0.0, 1.0]]), "lie in"),
        (np.array([[np.nan, 0.0], [0.0, 1.0]]), "finite"),
        (np.array([[0.7, 0.3], [0.7, 0.3]]), "position column"),
        (np.array([[0.9, 0.2], [0.1, 0.8]]), "driver row"),
    ],
)
def test_malformed_probability_matrices_fail(probabilities: np.ndarray, message: str) -> None:
    field_size = probabilities.shape[0]
    matrix = _matrix("malformed", field_size=field_size, probabilities=probabilities)
    with pytest.raises(ValueError, match=message):
        temperature_scale_and_sinkhorn(matrix.probabilities, temperature=1.0)


def test_audit_reports_proper_scores_top_k_ece_and_provenance() -> None:
    calibrator, matrices, outcomes = _fit_pair()
    raw = audit_qualifying_position_probabilities(
        matrices,
        outcomes,
        audit_event_keys=["2026-01", "2026-02"],
    )
    calibrated = audit_qualifying_position_probabilities(
        matrices,
        outcomes,
        audit_event_keys=["2026-02", "2026-01"],
        calibrator=calibrator,
    )

    for metrics in (raw, calibrated):
        assert metrics.event_keys == ("2026-01", "2026-02")
        assert metrics.event_count == 2
        assert metrics.driver_count == 8
        assert np.isfinite(metrics.multiclass_log_loss)
        assert 0.0 <= metrics.normalized_multiclass_brier <= 1.0
        assert 0.0 <= metrics.top1_ece <= 1.0
        assert 0.0 <= metrics.top3_ece <= 1.0
        assert 0.0 <= metrics.top10_ece <= 1.0
        assert metrics.ece_bin_count == 10

    card = calibrator.model_card
    assert raw.calibration_model_id is None
    assert calibrated.calibration_model_id == card.model_id
    assert card.promotion_status == DIAGNOSTIC_PROMOTION_STATUS
    assert card.source_model_id == "qualifying-joint-samples-v4"
    assert card.calibration_event_count == 2
    assert card.calibration_event_keys == ("2026-01", "2026-02")
    assert [item.prediction_evidence_id for item in card.event_provenance] == [
        "prediction-snapshot-2026-01",
        "prediction-snapshot-2026-02",
    ]
    assert [item.outcome_evidence_id for item in card.event_provenance] == [
        "official-result-2026-01",
        "official-result-2026-02",
    ]
    assert len(card.calibration_data_sha256) == 64
    assert card.to_dict()["promotion_status"] == DIAGNOSTIC_PROMOTION_STATUS


def test_transform_rejects_a_different_source_model_version() -> None:
    calibrator, _, _ = _fit_pair()
    with pytest.raises(ValueError, match="source_model_id"):
        calibrator.transform(_matrix("audit", source_model_id="other-model"))
