from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from packages.f1.data.providers.telemetry_cache import (
    NORMALIZED_TELEMETRY_CHANNELS,
    audit_telemetry_cache_manifests,
    select_representative_push_laps,
    sha256_file,
    validate_cached_telemetry_tensor,
    validate_telemetry_frame,
)
from run_prequal_telemetry_cache import (
    _load_records,
    _manifest_evidence,
    _rehearsal_contract,
    _training_targets,
    _write_lap_tensor,
)


def _telemetry() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2026-07-12T10:00:00Z", periods=12, freq="1s"),
            "Distance": np.linspace(0.0, 5000.0, num=12),
            "Speed": np.linspace(100.0, 300.0, num=12),
            "RPM": np.linspace(8000.0, 12000.0, num=12),
            "nGear": np.linspace(2.0, 8.0, num=12),
            "Throttle": np.linspace(0.0, 100.0, num=12),
            "Brake": np.r_[np.ones(2), np.zeros(10)],
            "DRS": np.r_[np.zeros(6), np.full(6, 12.0)],
        }
    )


def _cache_record(
    tmp_path: Path,
    *,
    event_key: int,
    driver_id: str,
) -> dict[str, object]:
    path = tmp_path / f"{event_key}_{driver_id}.npz"
    metadata = _write_lap_tensor(
        _telemetry(),
        path=path,
        expected_lap_distance_m=5000.0,
        distance_bins=8,
        minimum_distance_coverage=0.95,
    )
    return {
        "event_key": event_key,
        "driver_id": driver_id,
        "lap_number": 1,
        "feature_as_of": "2026-07-12T10:00:11Z",
        "qualifying_start_utc": "2026-07-12T12:00:00Z",
        "telemetry_path": path.name,
        "telemetry_sha256": sha256_file(path),
        **metadata,
    }


def test_push_lap_selection_rejects_deleted_pit_inaccurate_and_flagged_laps() -> None:
    laps = pd.DataFrame(
        {
            "Driver": ["AAA"] * 6 + ["BBB"] * 2,
            "LapNumber": range(1, 9),
            "LapTime": pd.to_timedelta(
                [90.0, 89.0, 88.0, 87.0, 86.0, 85.0, 91.0, 90.5], unit="s"
            ),
            "Deleted": [False, True, False, False, False, False, False, False],
            "IsAccurate": [True, True, False, True, True, True, True, True],
            "PitOutTime": [pd.NaT, pd.NaT, pd.NaT, pd.Timedelta(1, "s"), pd.NaT, pd.NaT, pd.NaT, pd.NaT],
            "PitInTime": [pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.Timedelta(1, "s"), pd.NaT, pd.NaT, pd.NaT],
            "TrackStatus": ["1", "1", "1", "1", "1", "2", "1", "1"],
        }
    )
    selected = select_representative_push_laps(laps, maximum_laps_per_driver=2)
    assert selected.loc[selected["driver_id"] == "AAA", "lap_number"].tolist() == [1]
    assert selected.loc[selected["driver_id"] == "BBB", "lap_number"].tolist() == [8, 7]
    assert selected.loc[selected["driver_id"] == "BBB", "push_lap_rank"].tolist() == [1, 2]


def test_telemetry_validation_requires_finite_physical_data_before_cutoff() -> None:
    telemetry = pd.DataFrame(
        {
            "Date": pd.date_range("2026-07-12T10:00:00Z", periods=3, freq="1s"),
            "Distance": [0.0, 2500.0, 5000.0],
            "Speed": [100.0, 300.0, 120.0],
            "RPM": [8000.0, 12000.0, 9000.0],
            "nGear": [2.0, 8.0, 3.0],
            "Throttle": [50.0, 100.0, 60.0],
            "Brake": [0.0, 0.0, 1.0],
            "DRS": [0.0, 12.0, 0.0],
        }
    )
    summary = validate_telemetry_frame(
        telemetry, qualifying_start_utc="2026-07-12T12:00:00Z"
    )
    assert summary["distance_max_m"] == pytest.approx(5000.0)
    assert summary["feature_as_of"] < summary["qualifying_start_utc"]

    broken = telemetry.copy()
    broken.loc[1, "Speed"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_telemetry_frame(broken, qualifying_start_utc="2026-07-12T12:00:00Z")
    with pytest.raises(ValueError, match="crosses"):
        validate_telemetry_frame(
            telemetry, qualifying_start_utc="2026-07-12T10:00:01Z"
        )


def test_cache_audit_requires_complete_independent_events_and_existing_files(
    tmp_path: Path,
) -> None:
    manifests: list[dict[str, object]] = []
    for event in (202601, 202602):
        for index in range(3):
            manifests.append(_cache_record(tmp_path, event_key=event, driver_id=f"D{index}"))
    ready = audit_telemetry_cache_manifests(
        manifests,
        root=tmp_path,
        minimum_independent_events=2,
        minimum_drivers_per_event=3,
    )
    assert ready.ready_for_deep_model
    assert ready.event_count == 2
    assert ready.driver_event_count == 6
    assert ready.record_count == 6
    assert ready.validated_tensor_count == 6
    assert ready.hash_mismatch_count == 0
    assert ready.invalid_tensor_count == 0
    assert ready.complete_event_keys == ("202601", "202602")
    assert len(ready.validated_cache_sha256) == 64

    reordered = audit_telemetry_cache_manifests(
        reversed(manifests),
        root=tmp_path,
        minimum_independent_events=2,
        minimum_drivers_per_event=3,
    )
    assert reordered.validated_cache_sha256 == ready.validated_cache_sha256

    insufficient = audit_telemetry_cache_manifests(
        manifests,
        root=tmp_path,
        minimum_independent_events=3,
        minimum_drivers_per_event=3,
    )
    assert insufficient.blockers == (
        "insufficient_independent_prequalifying_telemetry_events",
    )

    manifests[0]["qualifying_start_utc"] = "2026-07-12T10:00:11Z"
    (tmp_path / str(manifests[1]["telemetry_path"])).unlink()
    rejected = audit_telemetry_cache_manifests(
        manifests,
        root=tmp_path,
        minimum_independent_events=2,
        minimum_drivers_per_event=3,
    )
    assert not rejected.ready_for_deep_model
    assert set(rejected.blockers) == {
        "insufficient_independent_prequalifying_telemetry_events",
        "telemetry_cutoff_violations",
        "telemetry_files_missing",
        "telemetry_tensor_content_or_shape_invalid",
    }


def test_tensor_cache_is_distance_normalized_timestamped_and_self_describing(
    tmp_path: Path,
) -> None:
    record = _cache_record(tmp_path, event_key=202601, driver_id="AAA")
    validation = validate_cached_telemetry_tensor(record, root=tmp_path)

    assert validation["shape"] == [len(NORMALIZED_TELEMETRY_CHANNELS), 8]
    assert validation["channels"] == list(NORMALIZED_TELEMETRY_CHANNELS)
    assert validation["feature_as_of"] == "2026-07-12T10:00:11Z"
    with np.load(tmp_path / str(record["telemetry_path"]), allow_pickle=False) as payload:
        np.testing.assert_allclose(payload["distance_grid_m"], np.linspace(0.0, 5000.0, 8))
        assert payload["values"].shape == (len(NORMALIZED_TELEMETRY_CHANNELS), 8)
        assert payload["sample_timestamp_ns"].shape == (8,)
        assert int(payload["sample_timestamp_ns"].max()) == int(
            payload["feature_as_of_ns"]
        )


def test_cache_audit_fails_closed_on_hash_and_tensor_shape_corruption(
    tmp_path: Path,
) -> None:
    hash_record = _cache_record(tmp_path, event_key=202601, driver_id="AAA")
    hash_path = tmp_path / str(hash_record["telemetry_path"])
    hash_path.write_bytes(hash_path.read_bytes() + b"tampered")

    shape_record = _cache_record(tmp_path, event_key=202602, driver_id="BBB")
    shape_path = tmp_path / str(shape_record["telemetry_path"])
    np.savez_compressed(shape_path, values=np.zeros((1, 1), dtype=np.float32))
    shape_record["telemetry_sha256"] = sha256_file(shape_path)

    rejected = audit_telemetry_cache_manifests(
        [hash_record, shape_record],
        root=tmp_path,
        minimum_independent_events=1,
        minimum_drivers_per_event=1,
    )

    assert not rejected.ready_for_deep_model
    assert rejected.event_count == 0
    assert rejected.hash_mismatch_count == 1
    assert rejected.invalid_tensor_count == 1
    assert set(rejected.blockers) == {
        "insufficient_independent_prequalifying_telemetry_events",
        "telemetry_hash_mismatches",
        "telemetry_tensor_content_or_shape_invalid",
    }


def test_rehearsal_contract_uses_sprint_qualifying_without_reading_gp_qualifying() -> None:
    event = pd.Series(
        {
            "Session1": "Practice 1",
            "Session1DateUtc": "2026-05-01T10:00:00Z",
            "Session2": "Sprint Qualifying",
            "Session2DateUtc": "2026-05-01T14:00:00Z",
            "Session3": "Sprint",
            "Session3DateUtc": "2026-05-02T10:00:00Z",
            "Session4": "Qualifying",
            "Session4DateUtc": "2026-05-02T14:00:00Z",
            "Session5": "Race",
            "Session5DateUtc": "2026-05-03T14:00:00Z",
        }
    )
    source, cutoff = _rehearsal_contract(event)
    assert source == "Sprint Qualifying"
    assert cutoff.isoformat() == "2026-05-02T14:00:00+00:00"


def test_training_targets_are_separate_and_stage_labels_are_nested() -> None:
    class Session:
        laps = pd.DataFrame(
            {
                "Driver": ["AAA", "AAA", "BBB"],
                "LapTime": pd.to_timedelta([90.0, 89.5, 91.0], unit="s"),
                "Deleted": [False, True, False],
            }
        )
        results = pd.DataFrame(
            {
                "Abbreviation": ["AAA", "BBB"],
                "Q1": [pd.Timedelta(90.0, "s"), pd.Timedelta(91.0, "s")],
                "Q2": [pd.Timedelta(89.8, "s"), pd.NaT],
                "Q3": [pd.NaT, pd.NaT],
            }
        )

    targets = _training_targets(Session(), event_key=202601).set_index("driver_id")
    assert targets.loc["AAA", "lap_time_seconds"] == pytest.approx(90.0)
    assert bool(targets.loc["AAA", "has_q2_time"])
    assert not bool(targets.loc["BBB", "has_q2_time"])
    assert targets["target_available_after_qualifying"].all()


def test_cache_record_loading_is_scoped_to_requested_season(tmp_path: Path) -> None:
    for year in (2025, 2026):
        directory = tmp_path / str(year) / "round_01_test"
        directory.mkdir(parents=True)
        (directory / "telemetry_manifest.json").write_text(
            '{"year": %d, "event_key": %d, "feature_records": [{"driver_id": "D%d"}]}'
            % (year, year * 100 + 1, year),
            encoding="utf-8",
        )

    records = list(_load_records(tmp_path, year=2026))
    evidence = _manifest_evidence(tmp_path, year=2026, root=tmp_path)

    assert records == [{"driver_id": "D2026"}]
    assert len(evidence) == 1
    assert evidence[0]["event_key"] == 202601
    assert evidence[0]["feature_record_count"] == 1
    assert len(evidence[0]["sha256"]) == 64


# Suggested commit name: test(f1-telemetry): enforce causal cache readiness
