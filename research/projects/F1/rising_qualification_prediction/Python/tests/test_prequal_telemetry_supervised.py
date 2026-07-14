from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from build_prequal_telemetry_supervised_dataset import (
    write_immutable_supervised_manifest,
)
from packages.f1.data.providers.telemetry_cache import (
    NORMALIZED_TELEMETRY_CHANNELS,
    TELEMETRY_TENSOR_SCHEMA_VERSION,
    TelemetryHashMismatchError,
    sha256_file,
)
from packages.f1.data.providers.telemetry_supervised import (
    TelemetrySupervisedDatasetError,
    _qualifying_advancement_slots,
    _stage_labels,
    build_prequal_telemetry_supervised_manifest,
    canonical_sha256,
    validate_prequal_telemetry_supervised_manifest,
)


def _write_tensor(path: Path, *, start_seconds: int) -> dict[str, object]:
    bins = 8
    expected_distance = 5000.0
    timestamp_ns = (
        pd.Timestamp("2026-01-01T10:00:00Z").value
        + (start_seconds + np.arange(bins, dtype=np.int64)) * 1_000_000_000
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(TELEMETRY_TENSOR_SCHEMA_VERSION),
        values=np.arange(len(NORMALIZED_TELEMETRY_CHANNELS) * bins, dtype=np.float32).reshape(
            len(NORMALIZED_TELEMETRY_CHANNELS), bins
        ),
        channel_names=np.asarray(NORMALIZED_TELEMETRY_CHANNELS),
        distance_grid_m=np.linspace(0.0, expected_distance, num=bins),
        sample_timestamp_ns=timestamp_ns,
        feature_as_of_ns=np.asarray(int(timestamp_ns[-1]), dtype=np.int64),
        expected_lap_distance_m=np.asarray(expected_distance, dtype=np.float64),
        distance_coverage=np.asarray(1.0, dtype=np.float64),
    )
    return {
        "tensor_schema_version": TELEMETRY_TENSOR_SCHEMA_VERSION,
        "distance_normalized": True,
        "telemetry_shape": [len(NORMALIZED_TELEMETRY_CHANNELS), bins],
        "distance_bins": bins,
        "channels": list(NORMALIZED_TELEMETRY_CHANNELS),
        "expected_lap_distance_m": expected_distance,
        "distance_coverage": 1.0,
        "minimum_distance_coverage": 0.95,
        "feature_as_of": pd.Timestamp(int(timestamp_ns[-1]), tz="UTC").isoformat().replace(
            "+00:00", "Z"
        ),
    }


def _feature_record(
    root: Path,
    path: Path,
    *,
    driver_id: str,
    lap_number: int,
    push_lap_rank: int,
) -> dict[str, object]:
    tensor_metadata = _write_tensor(path, start_seconds=lap_number * 10)
    return {
        "event_key": 202601,
        "driver_id": driver_id,
        "lap_number": lap_number,
        "push_lap_rank": push_lap_rank,
        "lap_time_seconds": 90.0 + lap_number / 10.0,
        "qualifying_start_utc": "2026-01-01T12:00:00Z",
        "telemetry_path": str(path.relative_to(root)),
        "telemetry_sha256": sha256_file(path),
        **tensor_metadata,
    }


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    event_slug = "round_01_test_grand_prix"
    telemetry_event = (
        tmp_path / "data/f1/telemetry/pre_qualifying/2026" / event_slug
    )
    feature_records = [
        _feature_record(
            tmp_path,
            telemetry_event / "features/AAA_lap_001.npz",
            driver_id="AAA",
            lap_number=1,
            push_lap_rank=1,
        ),
        _feature_record(
            tmp_path,
            telemetry_event / "features/AAA_lap_002.npz",
            driver_id="AAA",
            lap_number=2,
            push_lap_rank=2,
        ),
        _feature_record(
            tmp_path,
            telemetry_event / "features/BBB_lap_001.npz",
            driver_id="BBB",
            lap_number=1,
            push_lap_rank=1,
        ),
    ]
    telemetry_manifest = telemetry_event / "telemetry_manifest.json"
    telemetry_manifest.write_text(
        json.dumps(
            {
                "schema_version": "f1_prequal_telemetry_cache_v2",
                "year": 2026,
                "round": 1,
                "event_key": 202601,
                "event_name": "Test Grand Prix",
                "rehearsal_source": "Practice 3",
                "qualifying_start_utc": "2026-01-01T12:00:00Z",
                "feature_records": feature_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    weekend_event = tmp_path / "data/f1/raw/weekends/2026" / event_slug
    weekend_event.mkdir(parents=True)
    laps_path = weekend_event / "04_qualifying_laps.csv"
    pd.DataFrame(
        {
            "Driver": ["AAA", "AAA", "AAA", "BBB"],
            "LapNumber": [10, 11, 12, 9],
            "LapTime": [90.0, 89.0, 90.2, 91.0],
            "Deleted": [False, True, False, False],
            "Sector1Time": [30.0, 29.7, 30.1, "0 days 00:00:30.300"],
            "Sector2Time": [29.5, 29.2, 29.6, 30.2],
            "Sector3Time": [30.5, 30.1, 30.5, 30.5],
        }
    ).to_csv(laps_path, index=False)
    results_path = weekend_event / "04_qualifying_results.csv"
    pd.DataFrame(
        {
            "Abbreviation": ["AAA", "BBB"],
            "Q1": ["0 days 00:01:31", "0 days 00:01:32"],
            "Q2": ["0 days 00:01:30.5", None],
            "Q3": ["0 days 00:01:30", None],
        }
    ).to_csv(results_path, index=False)
    metadata_path = weekend_event / "weekend_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "f1_weekend_snapshot_v2_point_in_time",
                "year": 2026,
                "round_number": 1,
                "event_name": "Test Grand Prix",
                "sessions": [
                    {
                        "session_name": "Qualifying",
                        "scheduled_start_utc": "2026-01-01T12:00:00Z",
                        "available_at": "2026-01-01T14:00:00Z",
                        "captured_at": "2026-01-01T14:00:00Z",
                        "laps_path": str(laps_path.relative_to(tmp_path)),
                        "results_path": str(results_path.relative_to(tmp_path)),
                        "files": {
                            "laps": {"sha256": sha256_file(laps_path)},
                            "results": {"sha256": sha256_file(results_path)},
                        },
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "telemetry_root": tmp_path / "data/f1/telemetry/pre_qualifying",
        "weekends_root": tmp_path / "data/f1/raw/weekends",
        "telemetry_manifest": telemetry_manifest,
        "laps": laps_path,
        "metadata": metadata_path,
        "tamper_tensor": telemetry_event / "features/AAA_lap_001.npz",
    }


def _build(tmp_path: Path, paths: dict[str, Path]) -> dict[str, object]:
    return build_prequal_telemetry_supervised_manifest(
        root=tmp_path,
        telemetry_root=paths["telemetry_root"],
        weekends_root=paths["weekends_root"],
        year=2026,
        generated_at="2026-01-02T00:00:00Z",
    )


def test_builds_one_target_per_driver_event_bag_with_separate_hash_domains(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)
    payload = _build(tmp_path, paths)

    assert payload == _build(tmp_path, paths)
    assert payload["supervised_row_unit"] == "driver_event_bag"
    assert payload["tensor_rows_are_independent_supervised_rows"] is False
    assert payload["audit"] == {
        "event_count": 1,
        "driver_event_bag_count": 2,
        "validated_tensor_count": 3,
        "event_driver_counts": {"202601": 2},
        "stage_label_bag_count": 2,
        "observed_lap_time_target_bag_count": 2,
        "censored_lap_time_target_bag_count": 0,
        "complete_sector_target_bag_count": 2,
        "target_only_driver_event_count": 0,
        "duplicate_driver_event_count": 0,
        "target_objects_per_bag": 1,
        "target_objects_per_tensor": 0,
        "inference_eligible_target_count": 0,
    }
    bags = payload["bags"]
    assert [(bag["event_key"], bag["driver_id"]) for bag in bags] == [
        (202601, "AAA"),
        (202601, "BBB"),
    ]
    aaa, bbb = bags
    assert aaa["feature"]["tensor_count"] == 2
    assert aaa["target"]["lap_time_seconds"] == pytest.approx(90.0)
    assert aaa["target"]["has_legal_qualifying_lap"] is True
    assert aaa["target"]["lap_time_observed"] is True
    assert aaa["target"]["lap_time_target_status"] == "observed_best_legal_lap"
    assert aaa["target"]["stage_reached"] == 3
    assert aaa["target"]["sector1_seconds"] == pytest.approx(30.0)
    assert bbb["feature"]["tensor_count"] == 1
    assert bbb["target"]["stage_reached"] == 1
    assert bbb["target"]["sector1_seconds"] == pytest.approx(30.3)
    assert all(bag["target"]["inference_eligible"] is False for bag in bags)
    assert all("target" not in tensor for bag in bags for tensor in bag["feature"]["tensors"])
    assert len(payload["feature_input_manifest_sha256"]) == 64
    assert len(payload["target_input_manifest_sha256"]) == 64
    feature_paths = {row["path"] for row in payload["feature_input_files"]}
    target_paths = {row["path"] for row in payload["target_input_files"]}
    assert feature_paths.isdisjoint(target_paths)
    validation = validate_prequal_telemetry_supervised_manifest(
        payload, root=tmp_path
    )
    assert validation["bag_set_sha256"] == payload["bag_set_sha256"]
    assert validation["audit"] == payload["audit"]


def test_supervised_validator_rejects_tampered_nested_input(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    payload = _build(tmp_path, paths)
    paths["laps"].write_bytes(paths["laps"].read_bytes() + b"tampered")

    with pytest.raises(TelemetrySupervisedDatasetError, match="SHA-256 mismatch"):
        validate_prequal_telemetry_supervised_manifest(payload, root=tmp_path)


def test_supervised_validator_rejects_missing_nested_input(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    payload = _build(tmp_path, paths)
    paths["tamper_tensor"].unlink()

    with pytest.raises(TelemetrySupervisedDatasetError, match="does not exist"):
        validate_prequal_telemetry_supervised_manifest(payload, root=tmp_path)


def test_supervised_validator_rejects_non_relative_nested_path(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    payload = _build(tmp_path, paths)
    feature_files = payload["feature_input_files"]
    assert isinstance(feature_files, list)
    feature_files[0]["path"] = str((tmp_path / str(feature_files[0]["path"])).resolve())
    payload["feature_input_manifest_sha256"] = canonical_sha256(feature_files)

    with pytest.raises(TelemetrySupervisedDatasetError, match="repository-relative"):
        validate_prequal_telemetry_supervised_manifest(payload, root=tmp_path)


def test_supervised_validator_rejects_rehashed_bag_or_false_audit(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)
    payload = _build(tmp_path, paths)
    bags = payload["bags"]
    assert isinstance(bags, list)
    bags[0]["event_name"] = "Rehashed but false"
    bags[0]["bag_sha256"] = canonical_sha256(
        {key: value for key, value in bags[0].items() if key != "bag_sha256"}
    )
    payload["bag_set_sha256"] = canonical_sha256(
        [bag["bag_sha256"] for bag in bags]
    )

    with pytest.raises(
        TelemetrySupervisedDatasetError, match="disagrees with nested evidence"
    ):
        validate_prequal_telemetry_supervised_manifest(payload, root=tmp_path)

    payload = _build(tmp_path, paths)
    payload["audit"]["validated_tensor_count"] = 999
    with pytest.raises(TelemetrySupervisedDatasetError, match="audit counts"):
        validate_prequal_telemetry_supervised_manifest(payload, root=tmp_path)


def test_duplicate_tensor_record_is_rejected(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    manifest = json.loads(paths["telemetry_manifest"].read_text(encoding="utf-8"))
    manifest["feature_records"].append(dict(manifest["feature_records"][0]))
    paths["telemetry_manifest"].write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(TelemetrySupervisedDatasetError, match="duplicate telemetry record"):
        _build(tmp_path, paths)


def test_telemetry_driver_without_legal_q_lap_is_retained_as_censored_target(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)
    laps = pd.read_csv(paths["laps"])
    laps.loc[laps["Driver"] == "BBB", "Deleted"] = True
    laps.to_csv(paths["laps"], index=False)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    metadata["sessions"][0]["files"]["laps"]["sha256"] = sha256_file(paths["laps"])
    paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")

    payload = _build(tmp_path, paths)

    bbb = next(bag for bag in payload["bags"] if bag["driver_id"] == "BBB")
    assert bbb["target"]["has_legal_qualifying_lap"] is False
    assert bbb["target"]["lap_time_observed"] is False
    assert bbb["target"]["lap_time_target_status"] == "censored_no_legal_lap"
    assert bbb["target"]["lap_time_seconds"] is None
    assert bbb["target"]["lap_number"] is None
    assert bbb["target"]["sector1_seconds"] is None
    assert payload["audit"]["driver_event_bag_count"] == 2
    assert payload["audit"]["observed_lap_time_target_bag_count"] == 1
    assert payload["audit"]["censored_lap_time_target_bag_count"] == 1


def test_event_with_no_legal_q_laps_keeps_all_telemetry_bags_censored(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)
    laps = pd.read_csv(paths["laps"])
    laps["Deleted"] = True
    laps.to_csv(paths["laps"], index=False)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    metadata["sessions"][0]["files"]["laps"]["sha256"] = sha256_file(paths["laps"])
    paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")

    payload = _build(tmp_path, paths)

    assert len(payload["bags"]) == 2
    assert all(bag["target"]["lap_time_observed"] is False for bag in payload["bags"])
    assert payload["audit"]["observed_lap_time_target_bag_count"] == 0
    assert payload["audit"]["censored_lap_time_target_bag_count"] == 2


def test_official_position_recovers_advancement_without_later_stage_time() -> None:
    results = pd.DataFrame(
        {
            "Abbreviation": [f"D{position:02d}" for position in range(1, 23)],
            "Position": list(range(1, 23)),
            "Q1": [90.0] * 22,
            "Q2": [89.0] * 15 + [None] * 7,
            "Q3": [88.0] * 9 + [None] * 13,
        }
    )

    labels = _stage_labels(results)

    # A 22-car field advances positions 1..16 to Q2 and 1..10 to Q3.  D16 and
    # D10 reached the next session despite recording no time in that session.
    assert _qualifying_advancement_slots(20) == (15, 10)
    assert _qualifying_advancement_slots(22) == (16, 10)
    assert labels["D16"]["has_q2_time"] is False
    assert labels["D16"]["reaches_q2"] is True
    assert labels["D16"]["reaches_q2_from_position"] is True
    assert labels["D16"]["stage_reached"] == 2
    assert labels["D10"]["has_q3_time"] is False
    assert labels["D10"]["reaches_q3"] is True
    assert labels["D10"]["reaches_q3_from_position"] is True
    assert labels["D10"]["stage_reached"] == 3
    assert labels["D17"]["reaches_q2"] is False
    with pytest.raises(TelemetrySupervisedDatasetError, match="unsupported field size 21"):
        _qualifying_advancement_slots(21)


def test_sector_targets_are_never_borrowed_from_a_slower_lap(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    laps = pd.read_csv(paths["laps"])
    laps.loc[(laps["Driver"] == "AAA") & (laps["LapNumber"] == 10), "Sector1Time"] = np.nan
    laps.to_csv(paths["laps"], index=False)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    metadata["sessions"][0]["files"]["laps"]["sha256"] = sha256_file(paths["laps"])
    paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")

    payload = _build(tmp_path, paths)

    aaa = next(bag for bag in payload["bags"] if bag["driver_id"] == "AAA")
    assert aaa["target"]["lap_number"] == 10
    assert aaa["target"]["sector1_seconds"] is None


def test_tensor_hash_tampering_is_rejected(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    paths["tamper_tensor"].write_bytes(paths["tamper_tensor"].read_bytes() + b"tampered")

    with pytest.raises(TelemetryHashMismatchError, match="SHA-256 mismatch"):
        _build(tmp_path, paths)


def test_supervised_artifact_is_exclusive_create(tmp_path: Path) -> None:
    output = tmp_path / "artifact/supervised.json"
    evidence = write_immutable_supervised_manifest({"bags": [], "version": 1}, output)

    assert json.loads(output.read_text(encoding="utf-8")) == {"bags": [], "version": 1}
    assert evidence["sha256"] == sha256_file(output)
    with pytest.raises(FileExistsError):
        write_immutable_supervised_manifest({"bags": [], "version": 2}, output)
