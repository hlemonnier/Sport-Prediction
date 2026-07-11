from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import build_f1_evidence_pack as evidence


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _portable(repo: Path, path: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def _file_entry(repo: Path, path: Path) -> dict[str, object]:
    return {
        "path": _portable(repo, path),
        "sha256": evidence._sha256(path),
        "byte_size": path.stat().st_size,
    }


def _file_manifest(repo: Path, path: Path, version: str) -> dict[str, object]:
    files = [_file_entry(repo, path)]
    return {
        "manifest_version": version,
        "file_count": 1,
        "aggregate_sha256": evidence._canonical_json_sha256(files),
        "files": files,
    }


def _make_frozen_run(repo: Path, run_id: str, *, round_number: int = 1) -> Path:
    run_root = repo / "artifacts" / "backtests" / "f1" / "rolling_2026" / "runs" / run_id
    round_root = run_root / "2026" / f"round_{round_number:02d}"
    round_root.mkdir(parents=True)

    source_file = repo / "packages" / "f1" / "frozen_source.py"
    config_file = repo / "configs" / "f1" / "frozen.yaml"
    data_file = repo / "data" / "f1" / "raw" / "weekends" / "2026" / f"round_{round_number:02d}" / "input.csv"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("VALUE = 1\n", encoding="utf-8")
    config_file.write_text("season: 2026\n", encoding="utf-8")
    data_file.write_text("driver_id,value\na,1\n", encoding="utf-8")

    selection_path = run_root / "rolling_backtest_selections.csv"
    summary_path = run_root / "rolling_backtest_summary.csv"
    selection_rows = []
    summary_rows = []
    for target in ("qualifying", "race"):
        cutoff = "post_practice_pre_qualifying" if target == "qualifying" else "post_qualifying_pre_grid"
        selection_rows.append(
            {
                "season": 2026,
                "round": round_number,
                "target": target,
                "information_cutoff": cutoff,
                "market": "top10",
                "selection": "A",
                "model_rank": 1,
                "model_probability_pct": 90.0,
                "result": "hit",
                "train_seasons_used": "2026",
                "rolling_2026_rounds_used": "",
                "experiment_arm": "same_season_walk_forward",
            }
        )
        summary_rows.append(
            {
                "season": 2026,
                "round": round_number,
                "target": target,
                "information_cutoff": cutoff,
                "rows_predicted": 22,
                "rows_actual": 22,
                "field_coverage": 1.0,
                "experiment_arm": "same_season_walk_forward",
            }
        )
    pd.DataFrame(selection_rows).to_csv(selection_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    point_records: list[dict[str, object]] = []
    artifacts: list[dict[str, object]] = []
    for target, filename in (
        ("qualifying", "qualifying_prediction.json"),
        ("race", "postqual_race_prediction.json"),
    ):
        config = {"year": 2026, "round_number": round_number, "mode": target, "train_seasons": [2026]}
        point_in_time = {
            "target": target,
            "target_season": 2026,
            "target_round": round_number,
            "horizon_evidence_complete": True,
            "target_round_allowed_in_training": False,
            "information_cutoff": "post_practice_pre_qualifying" if target == "qualifying" else "post_qualifying_pre_grid",
        }
        path = round_root / filename
        config_hash = evidence._canonical_json_sha256(config)
        _write_json(
            path,
            {
                "schema_version": evidence.ROLLING_BACKTEST_SCHEMA_VERSION,
                "run_id": run_id,
                "config": config,
                "config_sha256": config_hash,
                "point_in_time": point_in_time,
                "rows": [{"driver_id": "a", "rank": 1}],
                "all_prediction_rows": [{"driver_id": "a", "rank": 1}],
                "generated_at": "2026-07-11T12:00:00Z",
            },
        )
        artifacts.append(
            {
                "artifact_type": f"{target}_prediction",
                **_file_entry(repo, path),
                "config_sha256": config_hash,
                "point_in_time": point_in_time,
            }
        )
        point_records.append(point_in_time)
    artifacts.extend(
        [
            {"artifact_type": "rolling_backtest_selections", **_file_entry(repo, selection_path)},
            {"artifact_type": "rolling_backtest_summary", **_file_entry(repo, summary_path)},
        ]
    )

    run_config = {
        "year": 2026,
        "rounds": [round_number],
        "qualifying_information_horizon": "auto",
        "race_information_horizon": "post_qualifying_pre_grid",
    }
    manifest = {
        "schema_version": evidence.ROLLING_BACKTEST_SCHEMA_VERSION,
        "run_id": run_id,
        "started_at": "2026-07-11T11:00:00Z",
        "completed_at": "2026-07-11T12:00:00Z",
        "git": {
            "head_sha": "a" * 40,
            "dirty": False,
            "status_porcelain_sha256": "b" * 64,
        },
        "implementation": _file_manifest(repo, source_file, "source_v1"),
        "configuration_files": _file_manifest(repo, config_file, "config_v1"),
        "run_configuration": {
            "payload": run_config,
            "sha256": evidence._canonical_json_sha256(run_config),
        },
        "runtime": {"python_version": "3.14.0", "packages": {"pandas": "2.3.0"}},
        "training_protocol": {
            "experiment_arm": "same_season_walk_forward",
            "same_season_only": True,
            "train_seasons_used": [2026],
            "transfer_train_seasons": [],
            "current_season_weight_multiplier": 1.0,
            "target_round_allowed_in_training": False,
        },
        "point_in_time_by_round": point_records,
        "input_data_by_round": {str(round_number): _file_manifest(repo, data_file, "data_v1")},
        "artifacts": artifacts,
        "artifact_contract": {
            "immutable_paths": True,
            "write_mode": "exclusive_create",
            "run_root": _portable(repo, run_root),
        },
    }
    manifest_path = run_root / "run_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _make_legacy_layout(repo: Path) -> None:
    docs = repo / "artifacts" / "reports" / "f1" / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    rows = [
        {"season": 2026, "round": 1, "target": "qualifying"},
        {"season": 2026, "round": 1, "target": "race"},
    ]
    pd.DataFrame(rows).to_csv(docs / "F1_2026_Rolling_Backtest_Selections.csv", index=False)
    pd.DataFrame(rows).to_csv(docs / "F1_2026_Rolling_Backtest_Summary.csv", index=False)
    legacy_round = repo / "artifacts" / "backtests" / "f1" / "rolling_2026" / "2026" / "round_01"
    _write_json(legacy_round / "qualifying_prediction.json", {"model_name": "legacy"})
    _write_json(legacy_round / "postqual_race_prediction.json", {"model_name": "legacy"})


def test_discovers_one_complete_frozen_run_and_surfaces_manifest_hash(tmp_path: Path) -> None:
    manifest_path = _make_frozen_run(tmp_path, "run-a")

    source = evidence._discover_evidence_source(tmp_path)

    assert source.run_id == "run-a"
    assert source.frozen_complete is True
    assert source.manifest_path == manifest_path.resolve()
    assert source.manifest_sha256 == evidence._sha256(manifest_path)
    assert set(source.prediction_paths) == {(2026, 1, "qualifying"), (2026, 1, "race")}


def test_automatic_discovery_fails_on_ambiguous_runs_but_run_id_selects(tmp_path: Path) -> None:
    _make_frozen_run(tmp_path, "run-a")
    _make_frozen_run(tmp_path, "run-b")

    with pytest.raises(evidence.EvidenceDiscoveryError, match="exactly one"):
        evidence._discover_evidence_source(tmp_path)

    selected = evidence._discover_evidence_source(tmp_path, run_id="run-b")
    assert selected.run_id == "run-b"


def test_incomplete_run_fails_closed_without_silent_fallback(tmp_path: Path) -> None:
    manifest_path = _make_frozen_run(tmp_path, "run-a")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifacts"][0]["sha256"] = "0" * 64
    _write_json(manifest_path, payload)
    _make_legacy_layout(tmp_path)

    with pytest.raises(evidence.EvidenceDiscoveryError, match="incomplete or invalid|Incomplete or invalid"):
        evidence._discover_evidence_source(tmp_path)


def test_manifest_with_hashed_but_incomplete_csv_schema_is_rejected(tmp_path: Path) -> None:
    manifest_path = _make_frozen_run(tmp_path, "run-a")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection_artifact = next(
        item for item in payload["artifacts"] if item["artifact_type"] == "rolling_backtest_selections"
    )
    selection_path = tmp_path / selection_artifact["path"]
    frame = pd.read_csv(selection_path).drop(columns=["experiment_arm"])
    frame.to_csv(selection_path, index=False)
    selection_artifact["sha256"] = evidence._sha256(selection_path)
    selection_artifact["byte_size"] = selection_path.stat().st_size
    _write_json(manifest_path, payload)

    with pytest.raises(evidence.EvidenceDiscoveryError, match="selection_columns_incomplete"):
        evidence._discover_evidence_source(tmp_path, run_id="run-a")


def test_incomplete_sibling_blocks_automatic_selection_but_not_explicit_run_id(tmp_path: Path) -> None:
    _make_frozen_run(tmp_path, "run-a")
    (tmp_path / "artifacts" / "backtests" / "f1" / "rolling_2026" / "runs" / "partial-run").mkdir()

    with pytest.raises(evidence.EvidenceDiscoveryError, match="missing_run_manifest"):
        evidence._discover_evidence_source(tmp_path)
    assert evidence._discover_evidence_source(tmp_path, run_id="run-a").run_id == "run-a"


def test_frozen_source_is_rechecked_before_pack_write(tmp_path: Path) -> None:
    manifest_path = _make_frozen_run(tmp_path, "run-a")
    source = evidence._discover_evidence_source(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["completed_at"] = "2026-07-11T12:01:00Z"
    _write_json(manifest_path, payload)

    with pytest.raises(evidence.EvidenceDiscoveryError, match="changed after discovery"):
        evidence._assert_source_still_frozen(tmp_path, source)


def test_legacy_layout_requires_explicit_read_only_mode(tmp_path: Path) -> None:
    _make_legacy_layout(tmp_path)

    with pytest.raises(evidence.EvidenceDiscoveryError, match="legacy-read-only"):
        evidence._discover_evidence_source(tmp_path)

    source = evidence._discover_evidence_source(tmp_path, legacy_read_only=True)
    assert source.source_kind == "legacy_read_only_unmanifested"
    assert source.frozen_complete is False
    assert source.manifest_sha256 is None


def test_main_refuses_existing_output_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _make_frozen_run(tmp_path, "run-a")
    output_dir = tmp_path / "already-exists"
    output_dir.mkdir()
    marker = output_dir / "marker.txt"
    marker.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(evidence, "_repo_root", lambda: tmp_path)

    with pytest.raises(FileExistsError):
        evidence.main(["--run-id", "run-a", "--output-dir", str(output_dir)])

    assert marker.read_text(encoding="utf-8") == "preserve"
