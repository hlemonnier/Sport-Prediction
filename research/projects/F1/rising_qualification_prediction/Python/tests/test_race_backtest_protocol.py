from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from packages.f1.data.providers.local_weekends import _season_entry_list_power_unit

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_race_survival_order_backtest import (  # noqa: E402
    _align_grid_driver_ids_from_qualifying,
    _build_event_rows,
    _fit_binary_terminal_calibrator,
    _rolling_oof_qualifying_prior,
    _same_product_promotion_blockers,
)


def test_signed_qualifying_prior_is_rolling_out_of_event_and_target_blind() -> None:
    history_rows = []
    for event in range(1, 6):
        for driver in range(1, 7):
            history_rows.append(
                {
                    "event_key": 202400 + event,
                    "driver_id": str(driver),
                    "qualy_position": driver,
                    "fp_quali_sim_rank": driver + (0.2 if event % 2 else -0.2),
                    "fp_mean_rank": driver,
                    "fp_quali_sim_evidence_share": 0.8,
                }
            )
    history = pd.DataFrame(history_rows)
    current = pd.DataFrame(
        {
            "driver_id": ["1", "2", "3", "4", "5", "6"],
            "qualy_position": [6, 5, 4, 3, 2, 1],
            "fp_quali_sim_rank": [1, 2, 3, 4, 5, 6],
            "fp_mean_rank": [1, 2, 3, 4, 5, 6],
            "fp_quali_sim_evidence_share": [0.8] * 6,
        }
    )

    first, manifest = _rolling_oof_qualifying_prior(history, current)
    changed_target = current.copy()
    changed_target["qualy_position"] = [1, 2, 3, 4, 5, 6]
    second, _ = _rolling_oof_qualifying_prior(history, changed_target)

    assert first.tolist() == second.tolist()
    assert sorted(first.tolist()) == [1, 2, 3, 4, 5, 6]
    assert manifest["source"].startswith("rolling_out_of_event_ridge")
    assert manifest["strictly_prior_event_keys"] is True
    assert manifest["training_events"] == 5


def test_calibrator_is_fitted_only_from_declared_rows_and_is_monotone() -> None:
    rows = pd.DataFrame(
        {
            "event_key": ["202501"] * 4 + ["202502"] * 4,
            "actual_terminal": [0, 0, 0, 1, 0, 1, 1, 1],
            "predicted_terminal": [0.05, 0.10, 0.20, 0.25, 0.30, 0.50, 0.70, 0.90],
        }
    )
    calibrator = _fit_binary_terminal_calibrator(rows)
    mapped = np.asarray([calibrator.transform(value) for value in np.linspace(0.01, 0.99, 30)])

    assert calibrator.calibration_event_keys == ("202501", "202502")
    assert calibrator.calibration_rows == 8
    assert np.diff(mapped).min() >= -1e-12


def test_calibrator_fails_closed_without_both_classes() -> None:
    with pytest.raises(ValueError, match="both outcome classes"):
        _fit_binary_terminal_calibrator(
            pd.DataFrame(
                {
                    "event_key": ["202501"] * 4,
                    "actual_terminal": [0, 0, 0, 0],
                    "predicted_terminal": [0.1, 0.2, 0.3, 0.4],
                }
            )
        )


def test_power_unit_mapping_is_season_specific_and_causal() -> None:
    assert _season_entry_list_power_unit(2025, "Alpine") == "Renault"
    assert _season_entry_list_power_unit(2026, "Alpine") == "Mercedes"
    assert _season_entry_list_power_unit(2026, "Aston Martin") == "Honda"
    assert _season_entry_list_power_unit(2026, "Cadillac") == "Ferrari"
    assert _season_entry_list_power_unit(2026, "Racing Bulls") == (
        "Red Bull Ford Powertrains"
    )


def test_fia_car_number_maps_to_abbreviation_from_qualifying_only() -> None:
    grid = pd.DataFrame(
        {
            "driver_id": ["63", "12"],
            "grid_car_number": ["63", "12"],
            "grid_official_driver_name": ["George RUSSELL", "Kimi ANTONELLI"],
            "grid_position": [1, 2],
        }
    )
    qualifying = pd.DataFrame(
        {
            "driver_id": ["RUS", "ANT"],
            "car_number": ["63", "12"],
            "driver_abbreviation": ["RUS", "ANT"],
        }
    )
    aligned = _align_grid_driver_ids_from_qualifying(grid, qualifying)

    assert aligned["driver_id"].tolist() == ["RUS", "ANT"]
    assert aligned["grid_provider_driver_id"].tolist() == ["63", "12"]
    assert set(aligned["grid_identity_mapping_source"]) == {
        "pre_race_qualifying_car_number"
    }


def test_race_truth_is_read_only_after_causal_feature_roster_is_frozen(
    tmp_path: Path,
) -> None:
    weekend = tmp_path / "2026" / "round_01_fake"
    weekend.mkdir(parents=True)
    (weekend / "weekend_metadata.json").write_text(
        """{
          "year": 2026,
          "round_number": 1,
          "event_name": "Fake Grand Prix",
          "event_format": "standard",
          "scheduled_event_date": "2026-03-08T04:00:00Z",
          "sessions": []
        }""",
        encoding="utf-8",
    )

    class Provider:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get_qualifying_results(self, *_: object) -> pd.DataFrame:
            self.calls.append("qualifying")
            return pd.DataFrame(
                {
                    "driver_id": ["1", "2"],
                    "position": [1, 2],
                    "team_name": ["CAUSAL_A", "CAUSAL_B"],
                    "power_unit": ["PU_A", "PU_B"],
                    "power_unit_source": ["season_entry_list"] * 2,
                    "car_number": ["1", "2"],
                    "driver_abbreviation": ["AAA", "BBB"],
                }
            )

        def get_fp_features(self, *_: object, **__: object) -> pd.DataFrame:
            self.calls.append("practice")
            return pd.DataFrame(
                {
                    "driver_id": ["1", "2"],
                    "fp_quali_sim_rank": [1.0, 2.0],
                }
            )

        def get_starting_grid(self, *_: object, **__: object) -> pd.DataFrame:
            self.calls.append("grid")
            return pd.DataFrame()

        def get_track_stats(self, *_: object) -> dict[str, float]:
            self.calls.append("track")
            return {}

        def get_race_results(self, *_: object) -> pd.DataFrame:
            self.calls.append("race_truth")
            return pd.DataFrame(
                {
                    "driver_id": ["1", "2"],
                    "position": [1, 2],
                    "team_name": ["LEAKED_A", "LEAKED_B"],
                    "power_unit": ["LEAKED_PU_A", "LEAKED_PU_B"],
                    "race_status_raw": ["Finished", "Engine"],
                    "race_status_evidence_complete": [True, True],
                    "retirement_fraction": [1.0, 0.5],
                    "laps_completed": [60, 30],
                }
            )

    provider = Provider()
    frame, info, _ = _build_event_rows(
        root=Path(__file__).resolve().parents[6],
        provider=provider,  # type: ignore[arg-type]
        weekends_dir=tmp_path,
        year=2026,
        round_number=1,
        history=pd.DataFrame(),
    )

    assert provider.calls[-1] == "race_truth"
    assert frame["team_name"].tolist() == ["CAUSAL_A", "CAUSAL_B"]
    assert frame["power_unit"].tolist() == ["PU_A", "PU_B"]
    assert "race_team_name" not in frame.columns
    assert "race_power_unit" not in frame.columns
    assert "finish_position" not in info["causal_inference_columns"]
    assert "terminal_status" not in info["causal_inference_columns"]
    assert info["race_truth_attached_after_inference_freeze"] is True


def test_post_grid_promotion_fails_closed_without_same_product_protocol() -> None:
    blockers = _same_product_promotion_blockers(
        audit_event_count=9,
        same_product_selection_evidence=False,
        same_product_calibration_evidence=False,
    )

    assert blockers == (
        "missing_same_product_selection_evidence",
        "missing_same_product_calibration_evidence",
    )


# Suggested commit name: test(f1-race): lock rolling OOF and calibration protocols
