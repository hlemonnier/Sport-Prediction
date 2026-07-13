from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from packages.f1.data.providers.telemetry_cache import (
    audit_telemetry_cache_manifests,
    select_representative_push_laps,
    validate_telemetry_frame,
)


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
            path = tmp_path / f"{event}_{index}.npz"
            path.write_bytes(b"evidence")
            manifests.append(
                {
                    "event_key": event,
                    "driver_id": f"D{index}",
                    "feature_as_of": "2026-01-01T10:00:00Z",
                    "qualifying_start_utc": "2026-01-01T12:00:00Z",
                    "telemetry_path": path.name,
                }
            )
    ready = audit_telemetry_cache_manifests(
        manifests,
        root=tmp_path,
        minimum_independent_events=2,
        minimum_drivers_per_event=3,
    )
    assert ready.ready_for_deep_model
    assert ready.event_count == 2

    manifests[0]["feature_as_of"] = "2026-01-01T12:00:00Z"
    (tmp_path / str(manifests[1]["telemetry_path"])).unlink()
    rejected = audit_telemetry_cache_manifests(
        manifests,
        root=tmp_path,
        minimum_independent_events=2,
        minimum_drivers_per_event=3,
    )
    assert not rejected.ready_for_deep_model
    assert set(rejected.blockers) == {
        "telemetry_cutoff_violations",
        "telemetry_files_missing",
    }


# Suggested commit name: test(f1-telemetry): enforce causal cache readiness
