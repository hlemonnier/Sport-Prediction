from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_qualifying_probability_calibration_audit as audit_runner
from run_qualifying_probability_calibration_audit import (
    AUDIT_SCHEMA_VERSION,
    IMPLEMENTATION_RELATIVE_PATHS,
    JEFFREYS_PSEUDOCOUNT,
    JEFFREYS_SOURCE_MODEL_SUFFIX,
    _canonical_sha256,
    _frame_digest,
    _jeffreys_smooth_permutation_marginals,
    build_parser,
    run,
    write_exclusive,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_repository(root: Path) -> None:
    for relative_path in IMPLEMENTATION_RELATIVE_PATHS:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(
                f"fixture implementation for {relative_path.as_posix()}\n",
                encoding="utf-8",
            )


def _event_payload(
    event_key: int,
    *,
    reverse_outcome: bool,
    invalid_probability_matrix: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    field_size = 4
    drivers = tuple(f"D{value}" for value in range(1, field_size + 1))
    probabilities = 0.65 * np.eye(field_size) + 0.35 * np.full(
        (field_size, field_size),
        1.0 / field_size,
    )
    if invalid_probability_matrix:
        probabilities[0, 0] += 0.10
    actual_positions = tuple(range(1, field_size + 1))
    if reverse_outcome:
        actual_positions = tuple(reversed(actual_positions))

    rows: list[dict[str, object]] = []
    for row_index, driver in enumerate(drivers):
        row: dict[str, object] = {
            "event_key": event_key,
            "field_size": field_size,
            "driver_id": driver,
            "actual_qualifying_position": actual_positions[row_index],
            "qualifying_model": "shared_qualifying_latent_lap_v4",
            "fastest_driver_probability": float(probabilities[row_index, 0]),
            "fastest_lap_top3_probability": float(probabilities[row_index, :3].sum()),
            "valid_lap_probability_sampled": 1.0,
            "reaches_q2_probability_sampled": 1.0,
            "reaches_q3_probability_sampled": 0.5,
            "expected_qualifying_position": float(
                probabilities[row_index] @ np.arange(1, field_size + 1, dtype=float)
            ),
            "pole_probability": float(probabilities[row_index, 0]),
            "top3_probability": float(probabilities[row_index, :3].sum()),
            "probability_calibration_status": "uncalibrated_joint_latent_samples",
            "position_marginals_calibrated": False,
        }
        for position in range(1, field_size + 1):
            row[f"p_position_{position}"] = float(probabilities[row_index, position - 1])
        rows.append(row)

    marginal_columns = [
        "driver_id",
        "fastest_driver_probability",
        "fastest_lap_top3_probability",
        "valid_lap_probability_sampled",
        "reaches_q2_probability_sampled",
        "reaches_q3_probability_sampled",
        "expected_qualifying_position",
        "pole_probability",
        "top3_probability",
        *[f"p_position_{position}" for position in range(1, field_size + 1)],
        "probability_calibration_status",
        "position_marginals_calibrated",
    ]
    marginal_sha = _frame_digest(pd.DataFrame(rows)[marginal_columns])
    training_manifest = {
        "target_event_key": event_key,
        "training_event_keys": list(range(202601, event_key)),
    }
    model_manifest = {
        "model_name": "shared_qualifying_latent_lap_v4",
        "target_event_key": event_key,
        "training_partition_sha256": _canonical_sha256(training_manifest),
    }
    shared: dict[str, object] = {
        "schema_version": "f1_shared_qualifying_forecast_artifact_v5",
        "event_key": event_key,
        "driver_ids": list(drivers),
        "joint_sample_count": 5000,
        "qualifying_position_marginals_sha256": marginal_sha,
        "model_manifest": model_manifest,
        "model_manifest_sha256": _canonical_sha256(model_manifest),
        "training_partition_manifest": training_manifest,
    }
    shared["artifact_sha256"] = _canonical_sha256(shared)
    for row in rows:
        row["shared_forecast_artifact_sha256"] = shared["artifact_sha256"]
    return rows, shared


def _source_payload(
    *,
    reverse_audit_outcomes: bool = False,
    invalid_probability_event: int | None = None,
) -> dict[str, object]:
    predictions: list[dict[str, object]] = []
    shared_artifacts: list[dict[str, object]] = []
    for round_number in range(1, 10):
        event_key = 202600 + round_number
        rows, shared = _event_payload(
            event_key,
            reverse_outcome=bool(reverse_audit_outcomes and round_number >= 5),
            invalid_probability_matrix=(event_key == invalid_probability_event),
        )
        predictions.extend(reversed(rows) if round_number % 2 else rows)
        shared_artifacts.append(shared)
    return {
        "schema_version": "f1_shared_qualifying_latent_event_block_v5",
        "mode": "qualifying_prediction",
        "target": "official_grand_prix_qualifying_classification",
        "protocol": {
            "event_partitions": {
                "point_fit": [202601, 202602],
                "calibration": [202603, 202604],
                "audit": [202605, 202606, 202607, 202608, 202609],
            },
            "audit_outcomes_reused": False,
        },
        "predictions": predictions,
        "shared_forecast_artifacts": shared_artifacts,
    }


def _write_source(path: Path, payload: dict[str, object]) -> str:
    _prepare_repository(path.parent)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return _sha256(path)


def test_cli_defaults_to_canonical_same_season_source_and_output() -> None:
    args = build_parser().parse_args([])
    assert args.input.name == "shared_latent_same_season_v1.json"
    assert (
        args.output.name
        == "shared_latent_same_season_temperature_sinkhorn_probability_audit.json"
    )


def test_fixed_jeffreys_support_correction_is_positive_and_doubly_stochastic() -> None:
    raw = np.eye(4, dtype=float)
    smoothed = _jeffreys_smooth_permutation_marginals(
        raw,
        sample_count=5000,
        event_key=202601,
    )

    assert np.all(smoothed > 0.0)
    assert np.allclose(smoothed.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
    assert np.allclose(smoothed.sum(axis=0), 1.0, atol=1e-12, rtol=0.0)
    assert smoothed[0, 1] == pytest.approx(
        JEFFREYS_PSEUDOCOUNT / (5000 + 4 * JEFFREYS_PSEUDOCOUNT)
    )
    assert smoothed.flags.writeable is False


@pytest.mark.parametrize("sample_count", [0, -1])
def test_fixed_jeffreys_support_correction_rejects_nonpositive_sample_count(
    sample_count: int,
) -> None:
    with pytest.raises(ValueError, match="joint_sample_count must be positive"):
        _jeffreys_smooth_permutation_marginals(
            np.eye(4, dtype=float),
            sample_count=sample_count,
        )


def test_audit_targets_cannot_change_fitted_temperature_or_model_id(tmp_path: Path) -> None:
    source_a = tmp_path / "source-a.json"
    source_b = tmp_path / "source-b.json"
    source_sha_a = _write_source(source_a, _source_payload())
    source_sha_b = _write_source(
        source_b,
        _source_payload(reverse_audit_outcomes=True),
    )

    result_a = run(
        source_artifact=source_a,
        expected_source_sha256=source_sha_a,
        repository_root=tmp_path,
    )
    result_b = run(
        source_artifact=source_b,
        expected_source_sha256=source_sha_b,
        repository_root=tmp_path,
    )

    card_a = result_a["calibration"]["model_card"]
    card_b = result_b["calibration"]["model_card"]
    assert card_a["selected_temperature"] == card_b["selected_temperature"]
    assert card_a["model_id"] == card_b["model_id"]
    assert card_a["calibration_data_sha256"] == card_b["calibration_data_sha256"]
    assert card_a["calibration_event_keys"] == ("202603", "202604")
    assert result_a["protocol"]["calibration_event_keys"] == [202603, 202604]
    assert result_a["protocol"]["audit_event_keys"] == [
        202605,
        202606,
        202607,
        202608,
        202609,
    ]
    assert result_a["protocol"]["audit_outcomes_used_in_fit"] is False
    assert result_a["promotion_status"] == (
        "diagnostic_only_insufficient_independent_calibration_events"
    )
    assert result_a["promoted"] is False
    assert result_a["source_artifact"]["path"] == "source-a.json"
    assert result_a["implementation_manifest_sha256"] == _canonical_sha256(
        result_a["implementation_manifest"]
    )
    assert [row["path"] for row in result_a["implementation_manifest"]] == [
        path.as_posix() for path in IMPLEMENTATION_RELATIVE_PATHS
    ]
    assert all(
        set(row) == {"path", "sha256", "size_bytes"}
        for row in result_a["implementation_manifest"]
    )
    assert len(result_a["per_event_audit"]) == 5
    assert result_a["aggregate_audit"]["jeffreys_smoothed_uncalibrated"][
        "multiclass_log_loss"
    ] != (
        result_b["aggregate_audit"]["jeffreys_smoothed_uncalibrated"][
            "multiclass_log_loss"
        ]
    )
    support_protocol = result_a["protocol"]["finite_sample_support_correction"]
    assert support_protocol == {
        "method": "symmetric_jeffreys_cell_pseudocount",
        "pseudo_count_per_driver_position_cell": 0.5,
        "status": "fixed_untuned_preprocessing",
        "applied_before_temperature_and_sinkhorn": True,
        "requires_declared_joint_sample_count": True,
        "preserves_permutation_row_and_column_stochasticity": True,
        "calibration_source_model_suffix": JEFFREYS_SOURCE_MODEL_SUFFIX,
    }
    assert card_a["source_model_id"].endswith(f"+{JEFFREYS_SOURCE_MODEL_SUFFIX}")
    for event in result_a["per_event_audit"]:
        correction = event["finite_sample_support_correction"]
        assert correction["joint_sample_count"] == 5000
        assert correction["strictly_positive_support"] is True
        assert correction["preserves_row_and_column_stochasticity"] is True
        assert event["calibration_input_source_model_id"].endswith(
            f"+{JEFFREYS_SOURCE_MODEL_SUFFIX}"
        )
    assert result_a["result_sha256"] != result_b["result_sha256"]

    without_hash = dict(result_a)
    observed_hash = without_hash.pop("result_sha256")
    assert observed_hash == _canonical_sha256(without_hash)
    assert result_a["schema_version"] == AUDIT_SCHEMA_VERSION
    assert result_a["artifact_contract"] == {
        "write_mode": "exclusive_create",
        "result_hash_excludes_only_result_sha256": True,
        "source_path_scope": "repository_relative_only",
        "source_read_mode": (
            "single_immutable_byte_snapshot_with_post_evaluation_digest_check"
        ),
        "source_verified_unchanged_after_evaluation": True,
        "implementation_manifest_verified_unchanged_after_evaluation": True,
    }

    output = tmp_path / "immutable-audit.json"
    assert write_exclusive(result_a, output) == output.resolve()
    assert json.loads(output.read_text(encoding="utf-8"))["result_sha256"] == observed_hash
    with pytest.raises(FileExistsError):
        write_exclusive(result_a, output)


def test_runner_fails_closed_on_source_schema_and_hash_issues(tmp_path: Path) -> None:
    unsupported = _source_payload()
    unsupported["schema_version"] = "unknown-schema"
    unsupported_path = tmp_path / "unsupported.json"
    unsupported_sha = _write_source(unsupported_path, unsupported)
    with pytest.raises(ValueError, match="unsupported Qualifying backtest schema"):
        run(
            source_artifact=unsupported_path,
            expected_source_sha256=unsupported_sha,
            repository_root=tmp_path,
        )

    valid_path = tmp_path / "valid.json"
    _write_source(valid_path, _source_payload())
    with pytest.raises(ValueError, match="source artifact SHA mismatch"):
        run(
            source_artifact=valid_path,
            expected_source_sha256="0" * 64,
            repository_root=tmp_path,
        )

    corrupt = _source_payload()
    corrupt["shared_forecast_artifacts"][2]["artifact_sha256"] = "0" * 64
    corrupt_path = tmp_path / "corrupt.json"
    corrupt_sha = _write_source(corrupt_path, corrupt)
    with pytest.raises(ValueError, match="invalid artifact_sha256"):
        run(
            source_artifact=corrupt_path,
            expected_source_sha256=corrupt_sha,
            repository_root=tmp_path,
        )


def test_runner_fails_closed_on_invalid_full_field_probabilities(tmp_path: Path) -> None:
    source = _source_payload(invalid_probability_event=202605)
    path = tmp_path / "invalid-probability.json"
    source_sha = _write_source(path, source)
    with pytest.raises(ValueError, match="driver row must sum to one"):
        run(
            source_artifact=path,
            expected_source_sha256=source_sha,
            repository_root=tmp_path,
        )


def test_runner_fails_closed_when_declared_fields_are_incomplete(tmp_path: Path) -> None:
    source = _source_payload()
    altered = copy.deepcopy(source)
    audit_rows = [row for row in altered["predictions"] if row["event_key"] == 202605]
    audit_rows[0]["actual_qualifying_position"] = audit_rows[1]["actual_qualifying_position"]
    path = tmp_path / "incomplete-field.json"
    source_sha = _write_source(path, altered)
    with pytest.raises(ValueError, match="actual positions are not a complete permutation"):
        run(
            source_artifact=path,
            expected_source_sha256=source_sha,
            repository_root=tmp_path,
        )


def test_runner_rejects_source_outside_repository(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    _prepare_repository(repository_root)
    source_path = tmp_path / "outside-source.json"
    source_sha = _write_source(source_path, _source_payload())

    with pytest.raises(ValueError, match="source artifact must be stored inside"):
        run(
            source_artifact=source_path,
            expected_source_sha256=source_sha,
            repository_root=repository_root,
        )


def test_runner_fails_closed_when_source_changes_during_audit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.json"
    source_sha = _write_source(source_path, _source_payload())
    original_audit = audit_runner.audit_qualifying_position_probabilities
    source_mutated = False

    def mutate_source_after_first_audit(*args: object, **kwargs: object) -> object:
        nonlocal source_mutated
        result = original_audit(*args, **kwargs)
        if not source_mutated:
            source_path.write_bytes(source_path.read_bytes() + b"\n")
            source_mutated = True
        return result

    monkeypatch.setattr(
        audit_runner,
        "audit_qualifying_position_probabilities",
        mutate_source_after_first_audit,
    )
    with pytest.raises(RuntimeError, match="source artifact changed during"):
        run(
            source_artifact=source_path,
            expected_source_sha256=source_sha,
            repository_root=tmp_path,
        )


def test_runner_fails_closed_when_implementation_changes_during_audit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.json"
    source_sha = _write_source(source_path, _source_payload())
    original_fit = audit_runner.fit_qualifying_probability_calibrator
    implementation_path = tmp_path / IMPLEMENTATION_RELATIVE_PATHS[-1]

    def mutate_implementation_after_fit(*args: object, **kwargs: object) -> object:
        result = original_fit(*args, **kwargs)
        implementation_path.write_bytes(
            implementation_path.read_bytes() + b"# changed during evaluation\n"
        )
        return result

    monkeypatch.setattr(
        audit_runner,
        "fit_qualifying_probability_calibrator",
        mutate_implementation_after_fit,
    )
    with pytest.raises(RuntimeError, match="implementation changed during"):
        run(
            source_artifact=source_path,
            expected_source_sha256=source_sha,
            repository_root=tmp_path,
        )
