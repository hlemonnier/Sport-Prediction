from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from packages.f1.models.live_race.calibration import (
    MonteCarloPriorConfig,
    fit_live_race_calibration_from_locked_replay,
    load_live_race_calibration,
    write_live_race_calibration,
)


def _locked_replay() -> pd.DataFrame:
    regime_cycle = ("1", "1", "2", "2", "4", "4", "1")
    pit_loss_by_status = {"1": 20.0, "2": 15.0, "4": 10.0}
    rows: list[dict[str, object]] = []
    for event_idx, event_key in enumerate((202501, 202502)):
        cumulative = {driver: 0.0 for driver in ("1", "2", "3")}
        for lap in range(1, 43):
            status = regime_cycle[(lap - 1) % len(regime_cycle)]
            for driver_idx, driver in enumerate(("1", "2", "3")):
                lap_time = (
                    89.5
                    + (0.035 * lap)
                    + (0.16 * driver_idx)
                    + (0.025 * np.sin(lap / 2.0 + driver_idx))
                    + (0.05 * event_idx)
                )
                cumulative[driver] += lap_time
                rows.append(
                    {
                        "event_key": event_key,
                        "driver_id": driver,
                        "lap_number": lap,
                        "stint_id": 1,
                        "track_status": status,
                        "lap_time_seconds": lap_time,
                        "race_time_seconds": cumulative[driver],
                        "is_accurate": True,
                        "is_box_lap": False,
                        "observed_pit_loss_seconds": pit_loss_by_status[status]
                        + (0.2 * driver_idx),
                    }
                )
    return pd.DataFrame(rows)


def test_locked_replay_calibrates_filter_and_mc_priors_and_round_trips(tmp_path: Path) -> None:
    replay = _locked_replay()

    bundle = fit_live_race_calibration_from_locked_replay(
        replay,
        source_id="locked-2025-two-events",
        min_clean_filter_rows=30,
        min_transitions_per_regime=5,
        min_pit_rows_per_regime=5,
    )
    path = write_live_race_calibration(bundle, tmp_path / "live_calibration.json")
    restored = load_live_race_calibration(path)

    assert bundle.prior_calibration_ready is True
    assert bundle.promotion_ready is False
    assert bundle.filter_config.calibration_mode == "locked_replay"
    assert bundle.filter_config.calibration_rows >= 30
    assert bundle.monte_carlo_priors.calibration_mode == "locked_replay"
    assert bundle.monte_carlo_priors.transition_rows >= 15
    assert bundle.monte_carlo_priors.pit_rows >= 15
    assert restored.to_payload() == bundle.to_payload()
    assert restored.to_payload()["model_promotion_ready"] is False
    assert restored.filter_config.diagnostics()["uses_hand_tuned_priors"] is False


def test_mc_calibration_fails_closed_without_observed_pit_loss() -> None:
    replay = _locked_replay().drop(columns=["observed_pit_loss_seconds"])

    with pytest.raises(ValueError, match="requires observed_pit_loss_seconds"):
        fit_live_race_calibration_from_locked_replay(
            replay,
            source_id="locked-missing-pit-loss",
            min_clean_filter_rows=30,
            min_transitions_per_regime=5,
            min_pit_rows_per_regime=5,
        )


def test_default_mc_priors_are_explicitly_not_promotable() -> None:
    priors = MonteCarloPriorConfig()

    assert priors.calibration_mode == "hand_prior"
    assert priors.promotion_ready is False
    assert priors.diagnostics()["uses_hand_tuned_priors"] is True
