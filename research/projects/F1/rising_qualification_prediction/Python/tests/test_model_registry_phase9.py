from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from packages.sports_core.paths import find_repo_root
from packages.f1.orchestration.model_registry import (
    CANDIDATE_STATUS,
    FALLBACK_STATUS,
    PRODUCTION_STATUS,
    F1ModelRegistryEntry,
    ModelRegistry,
    PromotionEvidence,
    registry_entry_from_profile,
)
from packages.f1.models.live_race.environment import (
    LEAKAGE_CONTRACT_VERSION,
    REWARD_SEMANTICS,
    TRANSITION_FINGERPRINT_VERSION,
)
from packages.f1.models.live_race.replay_buffer import REPLAY_RECORD_SCHEMA_VERSION
from packages.f1.models.live_race.rl.replay_buffer import REPLAY_DATASET_SCHEMA_VERSION


def _ultimate_metrics(**overrides: float) -> dict[str, float]:
    metrics = {
        "p50_mae": 0.20,
        "p50_rmse": 0.25,
        "p05_pinball": 0.12,
        "p50_pinball": 0.10,
        "p90_pinball": 0.13,
        "interval_coverage": 0.82,
        "fastest_lap_winner_hit_rate": 0.50,
        "top3_fastest_lap_accuracy": 0.67,
    }
    metrics.update(overrides)
    return metrics


def _live_metrics(**overrides: float) -> dict[str, float]:
    metrics = {
        "policy_value": 10.0,
        "illegal_action_rate": 0.02,
        "regret_vs_oracle": 2.0,
    }
    metrics.update(overrides)
    return metrics


def _ultimate_baseline(metrics: dict[str, float] | None = None) -> F1ModelRegistryEntry:
    return F1ModelRegistryEntry(
        model_id="ultimate_lap_time_deterministic_baseline_v1",
        model_family="ultimate_lap_time",
        version="v1",
        training_data_cutoff="2026-06-01",
        feature_schema_version="ultimate_lap_time_telemetry_schema_v2_ideal_targets_normalized",
        artifact_path="artifacts/models/f1/ultimate_lap_time/deterministic_baseline_v1",
        metrics=_ultimate_metrics() if metrics is None else metrics,
        promotion_status=PRODUCTION_STATUS,
        deterministic_fallback=True,
    )


def _ultimate_candidate(**metric_overrides: float) -> F1ModelRegistryEntry:
    return F1ModelRegistryEntry(
        model_id="ultimate_lap_time_distance_tcn_v1",
        model_family="ultimate_lap_time",
        version="v1",
        training_data_cutoff="2026-06-01",
        feature_schema_version="ultimate_lap_time_telemetry_schema_v2_ideal_targets_normalized",
        artifact_path="artifacts/models/f1/ultimate_lap_time/deep/ultimate_lap_time_distance_tcn_v1",
        metrics=_ultimate_metrics(**metric_overrides),
        promotion_status=CANDIDATE_STATUS,
        fallback_model_id="ultimate_lap_time_deterministic_baseline_v1",
    )


def _live_baseline(*, deterministic: bool = True) -> F1ModelRegistryEntry:
    return F1ModelRegistryEntry(
        model_id="deterministic_baseline_v1",
        model_family="live_strategy",
        version="v1",
        training_data_cutoff="2026-06-01",
        feature_schema_version="live_strategy_state_v1_no_future_lap_fields",
        artifact_path="artifacts/models/f1/live_strategy/deterministic_baseline_v1",
        metrics=_live_metrics(),
        promotion_status=PRODUCTION_STATUS,
        deterministic_fallback=deterministic,
    )


def _live_candidate(*, fallback_model_id: str = "deterministic_baseline_v1") -> F1ModelRegistryEntry:
    return F1ModelRegistryEntry(
        model_id="live_strategy_conservative_offline_q_v1",
        model_family="live_strategy",
        version="v1",
        training_data_cutoff="2026-06-01",
        feature_schema_version="live_strategy_state_v1_no_future_lap_fields",
        artifact_path="artifacts/models/f1/live_strategy/rl/live_strategy_conservative_offline_q_v1",
        metrics=_live_metrics(policy_value=12.0, illegal_action_rate=0.0, regret_vs_oracle=1.2),
        promotion_status=CANDIDATE_STATUS,
        fallback_model_id=fallback_model_id,
    )


def _live_evidence(
    *,
    simulator_validation_passed: bool | None,
    locked_priors: bool = True,
    locked_strategy_templates: bool = True,
) -> PromotionEvidence:
    return PromotionEvidence(
        baseline_model_id="deterministic_baseline_v1",
        split_strategy="historical_replay_prefix_walk_forward",
        calibration_report_path="artifacts/reports/f1/live_strategy/rl/calibration_report.json",
        simulator_validation_passed=simulator_validation_passed,
        baseline_comparison_report_path="artifacts/reports/f1/live_strategy/rl/baseline_comparison.json",
        diagnostics={
            "prior_calibration_mode": "locked_replay" if locked_priors else "hand_prior",
            "strategy_template_calibration_mode": (
                "locked_replay" if locked_strategy_templates else "heuristic"
            ),
            "uses_hand_tuned_priors": not locked_priors,
        },
    )


def _materialize_promotion_artifacts(
    root: Path,
    baseline: F1ModelRegistryEntry,
    candidate: F1ModelRegistryEntry,
    evidence: PromotionEvidence,
) -> None:
    for entry in (baseline, candidate):
        artifact_dir = root / entry.artifact_path
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "manifest.json").write_text(json.dumps({"model_id": entry.model_id}), encoding="utf-8")
    assert evidence.calibration_report_path is not None
    calibration_path = root / evidence.calibration_report_path
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(
        json.dumps(
            {
                "candidate_model_id": candidate.model_id,
                "metrics": {"calibration_error": 0.1},
                "prior_calibration": {
                    "filter_mode": "locked_replay",
                    "monte_carlo_mode": "locked_replay",
                    "strategy_template_mode": "locked_replay",
                    "source_id": "locked-replay-test",
                    "source_rows": 500,
                },
            }
        ),
        encoding="utf-8",
    )
    assert evidence.baseline_comparison_report_path is not None
    comparison_path = root / evidence.baseline_comparison_report_path
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(
        json.dumps(
            {
                "candidate_model_id": candidate.model_id,
                "baseline_model_id": baseline.model_id,
                "metric_comparisons": {"policy_value": 2.0},
            }
        ),
        encoding="utf-8",
    )


def test_registry_entry_requires_phase9_fields_and_artifact_scope() -> None:
    payload = {
        "model_id": "candidate_v1",
        "model_family": "ultimate_lap_time",
        "version": "v1",
        "training_data_cutoff": "2026-06-01",
        "feature_schema_version": "schema_v1",
        "artifact_path": "artifacts/models/f1/candidate_v1",
        "metrics": {},
        "promotion_status": CANDIDATE_STATUS,
        "fallback_model_id": None,
    }

    entry = F1ModelRegistryEntry.from_payload(payload)

    assert entry.model_id == "candidate_v1"
    assert entry.artifact_path == "artifacts/models/f1/candidate_v1"
    with pytest.raises(ValueError, match="artifact_path"):
        F1ModelRegistryEntry.from_payload({**payload, "artifact_path": "tmp/candidate_v1"})
    with pytest.raises(ValueError, match="missing required fields"):
        F1ModelRegistryEntry.from_payload({key: value for key, value in payload.items() if key != "metrics"})


def test_ultimate_deep_promotion_fails_for_random_split_missing_calibration_and_leakage() -> None:
    baseline = _ultimate_baseline()
    candidate = _ultimate_candidate(p05_pinball=0.08)
    registry = ModelRegistry([baseline, candidate])

    evaluation = registry.evaluate_promotion(
        candidate.model_id,
        PromotionEvidence(
            baseline_model_id=baseline.model_id,
            split_strategy="random_by_event",
            calibration_report_path=None,
            leakage_issues=("predicted_p50 exactly matches actual lap time",),
            baseline_comparison_report_path="artifacts/reports/f1/ultimate_lap_time/deep/baseline_comparison.json",
        ),
    )

    assert evaluation.promotion_gate_passed is False
    assert "random_split_not_allowed" in evaluation.decision.reasons
    assert "calibration_report_missing" in evaluation.decision.reasons
    assert "candidate_has_leakage_issues" in evaluation.decision.reasons


def test_promotion_requires_baseline_comparison_metrics() -> None:
    baseline = _ultimate_baseline(metrics={})
    candidate = _ultimate_candidate(p05_pinball=0.08)
    registry = ModelRegistry([baseline, candidate])

    evaluation = registry.evaluate_promotion(
        candidate.model_id,
        PromotionEvidence(
            baseline_model_id=baseline.model_id,
            split_strategy="grouped_event_circuit_time_walk_forward",
            calibration_report_path="artifacts/reports/f1/ultimate_lap_time/deep/calibration_report.json",
            baseline_comparison_report_path="artifacts/reports/f1/ultimate_lap_time/deep/baseline_comparison.json",
        ),
    )

    assert evaluation.promotion_gate_passed is False
    assert "baseline_metrics_missing" in evaluation.decision.reasons
    assert "no_valid_baseline_metric_comparisons" in evaluation.decision.reasons


def test_production_replacement_requires_registered_deterministic_fallback() -> None:
    baseline = _live_baseline(deterministic=False)
    candidate = _live_candidate()
    registry = ModelRegistry([baseline, candidate])

    evaluation = registry.evaluate_promotion(
        candidate.model_id,
        _live_evidence(simulator_validation_passed=True),
    )

    assert evaluation.promotion_gate_passed is False
    assert "fallback_model_not_deterministic" in evaluation.decision.reasons


def test_live_promotion_rejects_hand_tuned_filter_and_mc_priors() -> None:
    baseline = _live_baseline()
    candidate = _live_candidate()
    registry = ModelRegistry([baseline, candidate])

    evaluation = registry.evaluate_promotion(
        candidate.model_id,
        _live_evidence(simulator_validation_passed=True, locked_priors=False),
    )

    assert evaluation.promotion_gate_passed is False
    assert "locked_replay_prior_calibration_required" in evaluation.decision.reasons
    assert "hand_tuned_priors_not_promotable" in evaluation.decision.reasons


def test_live_promotion_rejects_uncalibrated_strategy_template_probabilities() -> None:
    baseline = _live_baseline()
    candidate = _live_candidate()
    registry = ModelRegistry([baseline, candidate])

    evaluation = registry.evaluate_promotion(
        candidate.model_id,
        _live_evidence(
            simulator_validation_passed=True,
            locked_priors=True,
            locked_strategy_templates=False,
        ),
    )

    assert evaluation.promotion_gate_passed is False
    assert "locked_replay_strategy_template_calibration_required" in evaluation.decision.reasons


def test_live_strategy_rl_requires_simulator_validation_then_promotes_and_rolls_back(tmp_path: Path) -> None:
    baseline = _live_baseline()
    candidate = _live_candidate()
    registry = ModelRegistry([baseline, candidate])
    missing_evidence = _live_evidence(simulator_validation_passed=None)
    valid_evidence = _live_evidence(simulator_validation_passed=True)
    _materialize_promotion_artifacts(tmp_path, baseline, candidate, valid_evidence)

    missing_simulator = registry.evaluate_promotion(
        candidate.model_id,
        missing_evidence,
        artifact_root=tmp_path,
    )
    promoted = registry.promote_to_production(
        candidate.model_id,
        valid_evidence,
        artifact_root=tmp_path,
    )

    assert missing_simulator.promotion_gate_passed is False
    assert "simulator_validation_missing_or_failed" in missing_simulator.decision.reasons
    assert promoted.promoted is True
    assert promoted.registry.active_model("live_strategy").model_id == candidate.model_id
    assert promoted.registry.fallback_model_for(candidate.model_id).model_id == baseline.model_id
    assert promoted.registry.require(baseline.model_id).promotion_status == FALLBACK_STATUS

    round_tripped = ModelRegistry.from_payload(promoted.registry.to_payload())
    rolled_back = round_tripped.rollback("live_strategy")

    assert round_tripped.active_model("live_strategy").model_id == candidate.model_id
    assert rolled_back.active_model("live_strategy").model_id == baseline.model_id
    assert promoted.evaluation.to_payload()["decision"]["deterministic_fallback_model_id"] == baseline.model_id


def test_phase9_profiles_declare_non_promotion_defaults_and_artifact_paths() -> None:
    repo_root = find_repo_root(__file__)
    profile_paths = (
        repo_root / "configs/f1/profiles/ultimate_lap_time_deep.yaml",
        repo_root / "configs/f1/profiles/live_strategy_rl.yaml",
    )

    profiles = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in profile_paths]
    entries = [registry_entry_from_profile(profile) for profile in profiles]

    assert [entry.promotion_status for entry in entries] == [CANDIDATE_STATUS, CANDIDATE_STATUS]
    assert all(entry.metrics == {} for entry in entries)
    assert all(entry.fallback_model_id for entry in entries)
    assert profiles[0]["fallback"]["deterministic_fallback"] is True
    assert profiles[1]["fallback"]["deterministic_fallback"] is True
    assert profiles[1]["promotion"]["simulator_validation"]["required"] is True
    for profile in profiles:
        assert profile["promotion"]["production_replacement_allowed_by_default"] is False
        assert profile["promotion"]["baseline_comparison"]["required"] is True
        assert profile["promotion"]["leakage_tests"]["required"] is True
        assert str(profile["registry"]["artifact_path"]).startswith("artifacts/")
        assert str(profile["promotion"]["calibration_report_path"]).startswith("artifacts/")
        assert str(profile["promotion"]["baseline_comparison"]["report_path"]).startswith("artifacts/")


def test_live_strategy_rl_profile_uses_phase7_policy_id() -> None:
    repo_root = find_repo_root(__file__)
    phase7_profile = yaml.safe_load((repo_root / "configs/f1/profiles/live_strategy.yaml").read_text(encoding="utf-8"))
    phase9_profile = yaml.safe_load((repo_root / "configs/f1/profiles/live_strategy_rl.yaml").read_text(encoding="utf-8"))

    assert phase9_profile["runtime_family"] == phase7_profile["model_family"]
    assert phase7_profile["rl"]["offline_rl"]["model_id"] == phase9_profile["registry"]["model_id"]
    assert phase7_profile["contracts"]["state_schema"] == LEAKAGE_CONTRACT_VERSION
    assert phase7_profile["contracts"]["transition_schema"] == TRANSITION_FINGERPRINT_VERSION
    assert phase7_profile["contracts"]["replay_record_schema"] == REPLAY_RECORD_SCHEMA_VERSION
    assert phase7_profile["rl"]["replay_dataset"]["dataset_id"] == REPLAY_DATASET_SCHEMA_VERSION
    metadata = phase9_profile["registry"]["metadata"]
    assert metadata["replay_dataset_schema"] == REPLAY_DATASET_SCHEMA_VERSION
    assert metadata["transition_fingerprint_schema"] == TRANSITION_FINGERPRINT_VERSION
    assert metadata["replay_record_schema"] == REPLAY_RECORD_SCHEMA_VERSION
    assert phase7_profile["rl"]["offline_rl"]["reward_semantics"] == REWARD_SEMANTICS
    assert metadata["reward_semantics"] == REWARD_SEMANTICS
    assert phase9_profile["registry"]["deterministic_fallback"] is True
    assert metadata["behavior_cloning_partial_label_rows"] == 9506
    assert metadata["exact_mode_action_keys_supported"] == 0
    assert metadata["offline_q_rows"] == 0
    assert metadata["propensity_ope_rows"] == 0
    assert metadata["strategy_training_readiness_gate_pass"] is False
    assert metadata["current_candidate_trainable"] is False
    replay_audit = "artifacts/backtests/f1/live_strategy/live_strategy_replay_audit_v2_20260714.json"
    assert phase7_profile["artifacts"]["replay_audit"] == replay_audit
    assert phase7_profile["rl"]["replay_dataset"]["current_replay_readiness"]["evidence"] == replay_audit
    assert metadata["replay_audit_artifact"] == replay_audit
