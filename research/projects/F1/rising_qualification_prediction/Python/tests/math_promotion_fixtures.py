"""Synthetic positive controls for evidence gates; these are not model evidence."""
from packages.f1.orchestration.evaluation_protocol import EvaluationProtocol
from packages.f1.orchestration.non_live_validation import EventError


def protocol_fixture(*, year=2025, digest="a" * 64):
    audit = tuple(f"{year}:{r:02}" for r in range(1, 10))
    return EvaluationProtocol(
        development=(f"{year-3}:01",), selection=(f"{year-2}:01",),
        calibration=(f"{year-1}:01",), audit=audit,
        candidate_sha256=digest, evaluated_candidate_sha256=digest,
        frozen_at_utc=f"{year-1}-12-31T00:00:00Z",
        audit_prediction_times_utc={key: f"{year}-07-{i+1:02}T12:00:00Z" for i, key in enumerate(audit)},
        development_exposure_reviewed=True, point_in_time_provenance_verified=True,
    )


def paired_fixture(baseline, candidate, *, year=2025):
    return tuple(EventError(f"{year}:{r:02}", baseline, candidate, "sprint" if r % 3 == 0 else "standard") for r in range(1, 10))


def strategy_fixture(candidate_id, *, year=2025, digest="a"*64):
    return {
        "schema_version": "strategy_promotion_evidence_v1", "candidate_model_id": candidate_id,
        "candidate_sha256": digest,
        "ope": {"estimator": "doubly_robust", "held_out": True, "behavior_support_verified": True,
                "uncertainty_unit": "event", "ci95_improvement_vs_baseline": [0.2, 0.8],
                "effective_sample_size": 40, "trajectory_count": 90,
                "event_keys": list(protocol_fixture(year=year).audit)},
        "live_shadow": {"event_keys": [f"{year}:10", f"{year}:11", f"{year}:12"],
                        "prospective": True, "used_for_training_or_selection": False,
                        "illegal_action_rate": 0.0, "decision_count": 200,
                        "started_at_utc": f"{year}-07-20T12:00:00Z",
                        "event_prediction_times_utc": {f"{year}:{r:02}": f"{year}-08-{r:02}T12:00:00Z" for r in (10,11,12)}},
    }
