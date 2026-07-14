from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from packages.sports_core.paths import find_repo_root
import run_prequal_telemetry_tcn_sensitivity as sensitivity
from run_prequal_telemetry_tcn_sensitivity import (
    EXPECTED_PROFILE_IDS,
    TCNSensitivityError,
    _load_plan,
    run_matrix,
)


def _plan_path() -> Path:
    return (
        find_repo_root()
        / "configs/f1/prequal_telemetry_tcn_sensitivity_2026.json"
    )


def _mutated_plan(tmp_path: Path, mutation) -> tuple[Path, Path]:
    root = find_repo_root()
    payload = json.loads(_plan_path().read_text(encoding="utf-8"))
    mutation(payload)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, root


def _execution_plan(tmp_path: Path) -> tuple[Path, list[Path], Path]:
    payload = json.loads(_plan_path().read_text(encoding="utf-8"))
    root = Path("/")
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    payload["source_manifest"] = source.relative_to(root).as_posix()
    matrix = tmp_path / "matrix.json"
    payload["matrix_output"] = matrix.relative_to(root).as_posix()
    outputs: list[Path] = []
    for profile in payload["profiles"]:
        output = tmp_path / f"{profile['profile_id']}.json"
        profile["output_path"] = output.relative_to(root).as_posix()
        outputs.append(output)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(payload), encoding="utf-8")
    return plan, outputs, matrix


def _fake_artifact(
    *,
    config,
    generated_at: str,
    profile_design_provenance,
    **_: Any,
) -> dict[str, Any]:
    architecture = {
        "hidden_channels": config.hidden_channels,
        "kernel_size": config.kernel_size,
        "dilations": list(config.dilations),
        "head_hidden_dim": config.head_hidden_dim,
    }
    return {
        "schema_version": "f1_prequal_telemetry_true_tcn_research_v2",
        "generated_at": generated_at,
        "profile_design_provenance": profile_design_provenance,
        "validation_contract": {
            "outer_target_used_for_training": False,
            "outer_target_used_for_early_stopping": False,
            "outer_target_used_for_model_selection": False,
            "prior_outer_results_informed_profile_design": (
                profile_design_provenance[
                    "prior_outer_results_informed_profile_design"
                ]
            ),
            "same_outer_evaluation_targets_seen_before_profile_freeze": (
                profile_design_provenance[
                    "same_outer_evaluation_targets_seen_before_profile_freeze"
                ]
            ),
            "hyperparameters_tuned_on_outer_targets": (
                profile_design_provenance[
                    "hyperparameters_tuned_on_outer_targets"
                ]
            ),
        },
        "training_config": {
            "learning_rate": config.learning_rate,
            "seed": config.seed,
            "telemetry_input_mode": config.telemetry_input_mode,
        },
        "architecture": architecture,
        "capacity": {
            "trainable_scalar_parameter_count": config.hidden_channels * 10,
        },
        "model_input_ablation": {
            "telemetry_input_mode": config.telemetry_input_mode,
        },
        "summary": {"descriptive_only": True},
        "scored_event_keys": [202605],
        "folds": [
            {
                "target_event_key": 202605,
                "inner_selection_and_early_stopping": {
                    "initialization_seed": config.seed,
                },
                "metrics": {
                    "tcn_driver_correction": {
                        "mae_seconds": float(config.learning_rate),
                    }
                },
            }
        ],
    }


def test_checked_in_sensitivity_plan_is_fail_closed_and_complete() -> None:
    root = find_repo_root()
    payload, profiles = _load_plan(_plan_path(), root=root)

    assert payload["decision_policy"][
        "posthoc_profiles_never_selection_evidence"
    ] is True
    assert tuple(profile["profile_id"] for profile in profiles) == EXPECTED_PROFILE_IDS
    roles = {profile["profile_id"]: profile["evidence_role"] for profile in profiles}
    assert roles["d0_control"] == (
        "original_optimization_control_not_selection_evidence"
    )
    assert roles["d1_optimizer_primary"] == (
        "posthoc_exploratory_profile_predeclared_before_durable_matrix_run"
    )
    assert roles["d4_posthoc_lr_1e4"] == (
        "posthoc_hypothesis_after_outer_sensitivity_observed"
    )
    design = {
        profile["profile_id"]: profile["profile_design_provenance"]
        for profile in profiles
    }
    assert design["d0_control"][
        "prior_outer_results_informed_profile_design"
    ] is False
    assert design["d0_control"][
        "hyperparameters_tuned_on_outer_targets"
    ] is False
    assert design["d1_optimizer_primary"][
        "prior_outer_results_informed_profile_design"
    ] is True
    assert design["d1_optimizer_primary"][
        "hyperparameters_tuned_on_outer_targets"
    ] is True


def test_sensitivity_plan_rejects_posthoc_profile_as_selection_evidence(
    tmp_path: Path,
) -> None:
    path, root = _mutated_plan(
        tmp_path,
        lambda payload: payload["decision_policy"].update(
            {"posthoc_profiles_never_selection_evidence": False}
        ),
    )

    with pytest.raises(TCNSensitivityError, match="not fail-closed"):
        _load_plan(path, root=root)


def test_sensitivity_plan_requires_exact_input_ablation_pair(tmp_path: Path) -> None:
    def mutate(payload: dict[str, object]) -> None:
        profiles = payload["profiles"]
        assert isinstance(profiles, list)
        sham = next(
            profile
            for profile in profiles
            if profile["profile_id"] == "d1_zero_telemetry_static_anchor_sham"
        )
        sham["config"]["learning_rate"] = 0.0004

    path, root = _mutated_plan(tmp_path, mutate)

    with pytest.raises(TCNSensitivityError, match="must equal reference"):
        _load_plan(path, root=root)


def test_sensitivity_plan_rejects_outer_target_provenance_contradiction(
    tmp_path: Path,
) -> None:
    def mutate(payload: dict[str, object]) -> None:
        profiles = payload["profiles"]
        assert isinstance(profiles, list)
        reference = next(
            profile
            for profile in profiles
            if profile["profile_id"] == "d1_optimizer_primary"
        )
        reference["profile_design_provenance"][
            "hyperparameters_tuned_on_outer_targets"
        ] = False

    path, root = _mutated_plan(tmp_path, mutate)

    with pytest.raises(
        TCNSensitivityError,
        match="profile design provenance",
    ):
        _load_plan(path, root=root)


def test_run_matrix_publishes_profiles_then_completed_matrix_commit_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, outputs, matrix_path = _execution_plan(tmp_path)
    monkeypatch.setattr(sensitivity, "build_research_artifact", _fake_artifact)

    matrix = run_matrix(
        plan_path=plan,
        root=Path("/"),
        generated_at="2026-07-14T00:00:00Z",
    )

    assert matrix_path.is_file()
    assert all(path.is_file() for path in outputs)
    assert matrix["execution"]["matrix_selects_winner"] is False
    assert matrix["execution"][
        "profile_design_provenance_declared_per_profile"
    ] is True
    assert matrix["execution"]["outer_target_informed_profile_ids"] == list(
        EXPECTED_PROFILE_IDS[1:]
    )
    reference = json.loads(outputs[1].read_text(encoding="utf-8"))
    control = json.loads(outputs[0].read_text(encoding="utf-8"))
    assert control["validation_contract"][
        "hyperparameters_tuned_on_outer_targets"
    ] is False
    assert control["sensitivity_profile"][
        "prior_outer_results_informed_profile_design"
    ] is False
    assert reference["validation_contract"][
        "hyperparameters_tuned_on_outer_targets"
    ] is True
    assert reference["sensitivity_profile"][
        "prior_outer_results_informed_profile_design"
    ] is True
    assert reference["validation_contract"][
        "outer_target_used_for_training"
    ] is False
    assert reference["sensitivity_profile"][
        "completed_matrix_required_for_downstream_consumption"
    ] is True
    assert reference["sensitivity_profile"]["matrix_output_path"] == (
        matrix_path.relative_to(Path("/")).as_posix()
    )


def test_run_matrix_does_not_publish_when_a_later_profile_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, outputs, matrix_path = _execution_plan(tmp_path)
    calls = 0

    def fail_late(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected training failure")
        return _fake_artifact(**kwargs)

    monkeypatch.setattr(sensitivity, "build_research_artifact", fail_late)

    with pytest.raises(RuntimeError, match="injected training failure"):
        run_matrix(plan_path=plan, root=Path("/"))

    assert not matrix_path.exists()
    assert not any(path.exists() for path in outputs)


def test_run_matrix_does_not_publish_when_final_code_guard_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, outputs, matrix_path = _execution_plan(tmp_path)
    monkeypatch.setattr(sensitivity, "build_research_artifact", _fake_artifact)
    calls = 0

    def drifting_manifest(**_: Any) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        return [{"path": "runner.py", "sha256": str(calls), "size_bytes": 1}]

    monkeypatch.setattr(
        sensitivity,
        "_matrix_implementation_manifest",
        drifting_manifest,
    )

    with pytest.raises(TCNSensitivityError, match="implementation changed"):
        run_matrix(plan_path=plan, root=Path("/"))

    assert not matrix_path.exists()
    assert not any(path.exists() for path in outputs)
