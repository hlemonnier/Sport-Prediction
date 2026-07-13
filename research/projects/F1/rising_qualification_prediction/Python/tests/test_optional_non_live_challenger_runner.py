from __future__ import annotations

import pandas as pd
import pytest

from run_optional_non_live_challengers import (
    _locked_partitions,
    _quantile_metrics,
    _ranking_metrics,
)


def test_optional_partitions_are_disjoint_and_leave_later_audit_events() -> None:
    keys = [
        *(202400 + value for value in range(1, 4)),
        *(202500 + value for value in range(1, 7)),
        *(202600 + value for value in range(1, 10)),
    ]

    partitions = _locked_partitions(keys, target_year=2026)

    assert partitions["selection"] == tuple(range(202501, 202507))
    assert partitions["calibration"] == tuple(range(202601, 202605))
    assert partitions["audit"] == tuple(range(202605, 202610))
    assert max(partitions["calibration"]) < min(partitions["audit"])


def test_optional_runner_metrics_use_event_blocks_and_exact_quantile_labels() -> None:
    ranking = pd.DataFrame(
        {
            "event_key": [202605, 202605, 202606, 202606],
            "event_format": ["standard", "standard", "sprint", "sprint"],
            "actual_qualifying_position": [1, 2, 1, 2],
            "predicted_rank": [1, 2, 2, 1],
            "baseline_rank": [2, 1, 2, 1],
        }
    )
    quantiles = pd.DataFrame(
        {
            "event_key": [202605, 202605],
            "event_format": ["standard", "standard"],
            "achievable_session_end_lap_time_seconds": [90.0, 91.0],
            "baseline_lap_seconds": [90.2, 91.2],
            "lap_p05": [89.5, 90.5],
            "lap_p50": [90.1, 91.1],
            "lap_p90": [90.5, 91.5],
        }
    )

    rank_rows = _ranking_metrics(ranking)
    quantile_rows = _quantile_metrics(quantiles)

    assert len(rank_rows) == 2
    assert rank_rows[0]["candidate_mae"] == pytest.approx(0.0)
    assert quantile_rows[0]["candidate_mae_seconds"] == pytest.approx(0.1)
    assert quantile_rows[0]["p05_p90_coverage"] == pytest.approx(1.0)
