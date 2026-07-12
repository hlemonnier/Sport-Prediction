from __future__ import annotations

from packages.f1.orchestration.model_runtime import inspect_optional_model_runtime
from packages.f1.orchestration.non_live_validation import (
    EventError,
    evaluate_best_lap_promotion,
    evaluate_qualifying_promotion,
    evaluate_race_promotion,
    paired_event_diagnostics,
    validate_event_partitions,
)


def _events(*, baseline: float = 4.0, candidate: float = 3.0) -> list[EventError]:
    return [
        EventError(f"2025:{round_number:02d}", baseline + round_number / 100, candidate + round_number / 100, "sprint" if round_number % 3 == 0 else "standard")
        for round_number in range(1, 10)
    ]


def test_paired_event_diagnostics_reports_stability_and_gain_concentration() -> None:
    diagnostics = paired_event_diagnostics(_events(), bootstrap_samples=2_000, seed=7)

    assert diagnostics.mean_delta_candidate_minus_baseline == -1.0
    assert diagnostics.ci95_delta[1] < 0.0
    assert diagnostics.probability_of_improvement == 1.0
    assert diagnostics.leave_one_event_out_all_improve is True
    assert diagnostics.largest_positive_gain_share < 0.5
    assert set(diagnostics.stratum_mean_deltas) == {"sprint", "standard"}


def test_event_partitions_must_be_disjoint_and_chronological() -> None:
    clean = validate_event_partitions(
        development=["2022:01", "2022:02"],
        selection=["2023:01"],
        calibration=["2024:01"],
        audit=["2025:01"],
    )
    broken = validate_event_partitions(
        development=["2022:02", "2022:01"],
        selection=["2022:02"],
        calibration=["2024:01"],
        audit=["2025:01"],
    )

    assert clean == ()
    assert "development_events_not_chronological" in broken
    assert "event_partition_overlap:development:selection" in broken


def test_qualifying_gate_requires_both_weekend_strata_and_tail_stability() -> None:
    promoted = evaluate_qualifying_promotion(
        _events(baseline=2.0, candidate=1.7),
        baseline_kendall=0.75,
        candidate_kendall=0.78,
        pole_non_regression=True,
        top3_non_regression=True,
        top10_non_regression=True,
        tail_excluded_delta=-0.1,
        bootstrap_samples=2_000,
    )
    rejected = evaluate_qualifying_promotion(
        _events(baseline=2.0, candidate=1.7),
        baseline_kendall=0.75,
        candidate_kendall=0.75,
        pole_non_regression=True,
        top3_non_regression=True,
        top10_non_regression=True,
        tail_excluded_delta=0.01,
        bootstrap_samples=2_000,
    )

    assert promoted.promoted is True
    assert rejected.promoted is False
    assert "gate_failed:kendall_improvement_at_least_0_02" in rejected.reasons
    assert "gate_failed:tail_excluded_population_improves" in rejected.reasons


def test_race_gate_couples_status_order_coverage_and_legality() -> None:
    promoted = evaluate_race_promotion(
        _events(),
        baseline_kendall=0.52,
        candidate_kendall=0.53,
        baseline_status_brier=0.20,
        candidate_status_brier=0.17,
        baseline_status_log_loss=0.60,
        candidate_status_log_loss=0.54,
        entrant_coverage=1.0,
        all_classifications_legal=True,
        bootstrap_samples=2_000,
    )
    rejected = evaluate_race_promotion(
        _events(),
        baseline_kendall=0.52,
        candidate_kendall=0.53,
        baseline_status_brier=0.20,
        candidate_status_brier=0.21,
        baseline_status_log_loss=0.60,
        candidate_status_log_loss=0.61,
        entrant_coverage=0.95,
        all_classifications_legal=False,
        bootstrap_samples=2_000,
    )

    assert promoted.promoted is True
    assert rejected.promoted is False
    assert "gate_failed:status_brier_improves" in rejected.reasons
    assert "gate_failed:entrant_coverage_complete" in rejected.reasons
    assert "gate_failed:all_classifications_legal" in rejected.reasons


def test_best_lap_gate_checks_interval_calibration_and_width() -> None:
    promoted = evaluate_best_lap_promotion(
        _events(baseline=0.60, candidate=0.50),
        entrant_output_coverage=1.0,
        fastest_driver_non_regression=True,
        top3_non_regression=True,
        interval_coverage=0.84,
        nominal_interval_coverage=0.85,
        baseline_interval_width=1.0,
        candidate_interval_width=1.05,
        bootstrap_samples=2_000,
    )
    rejected = evaluate_best_lap_promotion(
        _events(baseline=0.60, candidate=0.50),
        entrant_output_coverage=1.0,
        fastest_driver_non_regression=True,
        top3_non_regression=True,
        interval_coverage=0.70,
        nominal_interval_coverage=0.85,
        baseline_interval_width=1.0,
        candidate_interval_width=1.2,
        bootstrap_samples=2_000,
    )

    assert promoted.promoted is True
    assert rejected.promoted is False
    assert "gate_failed:interval_coverage_within_5pct_points" in rejected.reasons
    assert "gate_failed:interval_width_inflation_at_most_10pct" in rejected.reasons


def test_optional_runtime_doctor_rejects_unknown_package() -> None:
    try:
        inspect_optional_model_runtime("catboost")
    except ValueError as exc:
        assert "xgboost and lightgbm" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown optional package must fail closed")


# Suggested commit name: test(f1): enforce non-live event-block promotion gates
