from __future__ import annotations

import json

import pandas as pd
import pytest

import run_optional_non_live_challengers as optional_runner
from packages.f1.models.grouped_ranking import GroupedRankingFitResult
from packages.f1.models.ultimate_lap_time.tabular_quantile import (
    TabularQuantileBackendUnavailable,
)

from run_optional_non_live_challengers import (
    _align_champion,
    _qualifying_start,
    _load_event_dataset,
    _locked_partitions,
    _paired_diagnostics,
    _quantile_metrics,
    _ranking_metrics,
    _select_candidate_evidence,
    _selection_holdout,
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
            "driver_id": ["a", "b", "a", "b"],
            "actual_qualifying_position": [1, 2, 1, 2],
            "predicted_rank": [1, 2, 2, 1],
            "champion_rank": [1, 2, 1, 2],
            "rehearsal_baseline_rank": [2, 1, 2, 1],
        }
    )
    quantiles = pd.DataFrame(
        {
            "event_key": [202605, 202605, 202605],
            "event_format": ["standard", "standard", "standard"],
            "driver_id": ["a", "b", "c"],
            "achievable_session_end_lap_time_seconds": [90.0, 91.0, None],
            "raw_anchor_lap_seconds": [90.2, 91.2, 92.2],
            "lap_p05": [89.5, 90.5, 91.5],
            "lap_p50": [90.1, 91.1, 92.1],
            "lap_p90": [90.5, 91.5, 92.5],
            "champion_lap_p05": [89.5, 90.5, 91.5],
            "champion_lap_p50": [90.0, 91.0, 92.0],
            "champion_lap_p90": [90.5, 91.5, 92.5],
        }
    )

    rank_rows = _ranking_metrics(ranking)
    quantile_rows = _quantile_metrics(quantiles)

    assert len(rank_rows) == 2
    assert rank_rows[0]["candidate_mae"] == pytest.approx(0.0)
    assert quantile_rows[0]["candidate_mae_seconds"] == pytest.approx(0.1)
    assert quantile_rows[0]["candidate_p05_p90_coverage"] == pytest.approx(1.0)
    assert quantile_rows[0]["rows"] == 3
    assert quantile_rows[0]["target_observed_rate"] == pytest.approx(2 / 3)
    assert quantile_rows[0]["candidate_output_coverage"] == pytest.approx(1.0)


def test_target_season_timestamp_fails_closed_but_selection_can_use_ordinal() -> None:
    metadata = {
        "year": 2026,
        "round_number": 3,
        "sessions": [{"session_type": "qualifying"}],
    }

    with pytest.raises(ValueError, match="selection-only"):
        _qualifying_start(metadata, use_selection_ordinal=False)

    timestamp, semantics = _qualifying_start(
        metadata, use_selection_ordinal=True
    )
    assert timestamp == pd.Timestamp("2026-01-04T00:00:00Z")
    assert semantics == "season_round_ordinal_selection_only"


def test_prior_selection_always_uses_one_round_ordinal_clock() -> None:
    metadata = {
        "year": 2025,
        "round_number": 10,
        "sessions": [
            {
                "session_type": "qualifying",
                "scheduled_start_utc": "2025-06-14T14:00:00Z",
            }
        ],
    }

    timestamp, semantics = _qualifying_start(metadata, use_selection_ordinal=True)

    assert timestamp == pd.Timestamp("2025-01-11T00:00:00Z")
    assert semantics == "season_round_ordinal_selection_only"


def test_target_scheduled_timestamp_year_must_match_metadata() -> None:
    metadata = {
        "year": 2026,
        "round_number": 1,
        "sessions": [
            {
                "session_type": "qualifying",
                "scheduled_start_utc": "2025-03-14T14:00:00Z",
            }
        ],
    }

    with pytest.raises(ValueError, match="does not match weekend metadata year"):
        _qualifying_start(metadata, use_selection_ordinal=False)


@pytest.mark.parametrize(
    ("metadata_year", "metadata_round", "info_event_key", "message"),
    (
        (2025, 1, 202501, "metadata year"),
        (2026, 2, 202602, "metadata round"),
        (2026, 1, 202602, "event_key"),
    ),
)
def test_event_loader_rejects_snapshot_identity_mismatches(
    monkeypatch,
    tmp_path,
    metadata_year: int,
    metadata_round: int,
    info_event_key: int,
    message: str,
) -> None:
    event_dir = tmp_path / "2026" / "round_01"
    event_dir.mkdir(parents=True)
    (event_dir / "weekend_metadata.json").write_text(
        json.dumps(
            {
                "year": metadata_year,
                "round_number": metadata_round,
                "sessions": [
                    {
                        "session_type": "qualifying",
                        "scheduled_start_utc": "2026-03-14T14:00:00Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        optional_runner,
        "_event_frame",
        lambda *args, **kwargs: (
            pd.DataFrame(),
            {"event_key": info_event_key},
            [],
        ),
    )

    with pytest.raises(ValueError, match=message):
        _load_event_dataset(tmp_path, years=(2026,), target_year=2026)


def test_champion_alignment_requires_exact_rows_and_integral_permutation() -> None:
    audit = pd.DataFrame(
        {
            "event_key": [202605, 202605],
            "driver_id": ["a", "b"],
        }
    )
    fractional = pd.DataFrame(
        {
            "event_key": [202605, 202605],
            "driver_id": ["a", "b"],
            "champion_rank": [1.5, 2.0],
        }
    )
    missing = fractional.iloc[:1].assign(champion_rank=1.0)

    with pytest.raises(ValueError, match="must all be integers"):
        _align_champion(audit, fractional, mode="qualifying")
    with pytest.raises(ValueError, match="do not match audit inference rows"):
        _align_champion(audit, missing, mode="qualifying")

    inverted_lap_interval = pd.DataFrame(
        {
            "event_key": [202605, 202605],
            "driver_id": ["a", "b"],
            "champion_lap_p05": [91.0, 90.0],
            "champion_lap_p50": [90.0, 91.0],
            "champion_lap_p90": [92.0, 92.0],
        }
    )
    with pytest.raises(ValueError, match="p05 <= p50 <= p90"):
        _align_champion(audit, inverted_lap_interval, mode="best_lap")


def test_bounded_selection_holdout_and_candidate_choice_use_late_events() -> None:
    selection = pd.DataFrame(
        {
            "event_key": [key for key in range(202501, 202509) for _ in range(2)],
            "event_as_of": [
                f"2025-01-{key - 202500:02d}T00:00:00Z"
                for key in range(202501, 202509)
                for _ in range(2)
            ],
        }
    )

    fit, validation, manifest = _selection_holdout(selection)
    selected = _select_candidate_evidence(
        [
            {"status": "available", "candidate_id": "a", "mae": 0.8, "tie": 0.2},
            {"status": "available", "candidate_id": "b", "mae": 0.6, "tie": 0.3},
        ],
        objective_key="mae",
        tie_break_keys=("tie",),
    )

    assert sorted(fit["event_key"].unique()) == list(range(202501, 202506))
    assert sorted(validation["event_key"].unique()) == list(range(202506, 202509))
    assert manifest["candidate_count_bound"]["ranking_per_backend"] == 2
    assert selected["candidate_id"] == "b"


def test_paired_diagnostics_include_bootstrap_leave_one_out_and_strata() -> None:
    diagnostics = _paired_diagnostics(
        [
            {
                "event_key": 1,
                "stratum": "standard",
                "candidate": 0.8,
                "champion": 1.0,
            },
            {
                "event_key": 2,
                "stratum": "standard",
                "candidate": 0.9,
                "champion": 1.1,
            },
            {
                "event_key": 3,
                "stratum": "sprint",
                "candidate": 0.7,
                "champion": 0.9,
            },
        ],
        candidate_key="candidate",
        champion_key="champion",
        bootstrap_samples=1_000,
        seed=7,
    )

    assert diagnostics["required_standard_and_sprint_present"] is True
    assert diagnostics["all_weekend_strata_improve"] is True
    assert diagnostics["leave_one_event_out_all_improve"] is True
    assert len(diagnostics["ci95_delta"]) == 2


def test_optional_run_consumes_causal_shared_event_contract_end_to_end(
    monkeypatch,
    tmp_path,
) -> None:
    event_keys = [
        *range(202401, 202403),
        *range(202501, 202507),
        *range(202601, 202607),
    ]
    rows = []
    for event_key in event_keys:
        for driver_index, driver in enumerate(("a", "b"), start=1):
            rows.append(
                {
                    "event_key": event_key,
                    "season": event_key // 100,
                    "event_as_of": (
                        f"{event_key // 100}-01-{event_key % 100:02d}T00:00:00Z"
                    ),
                    "event_time_semantics": "test",
                    "event_format": "standard",
                    "driver_id": driver,
                    "team_id": "team",
                    "rehearsal_source": "practice_3",
                    "quality_aware_anchor_seconds": 90.0 + driver_index,
                    "latest_qualifying_rehearsal_rank": driver_index,
                    "qualy_position": driver_index,
                    "achievable_session_end_lap_time_seconds": 89.0 + driver_index,
                    "has_valid_qualifying_lap": 1,
                    "reached_q2": 1,
                    "reached_q3": int(driver_index == 1),
                }
            )
    dataset = pd.DataFrame(rows)
    infos = {
        key: {"event_key": key, "event_time_semantics": "test"}
        for key in event_keys
    }
    monkeypatch.setattr(
        optional_runner,
        "_load_event_dataset",
        lambda *args, **kwargs: (dataset.copy(), infos, (), (), ()),
    )
    monkeypatch.setattr(
        optional_runner,
        "fit_grouped_ranking_challenger",
        lambda *args, **kwargs: GroupedRankingFitResult(
            status="unavailable",
            model=None,
            manifest={"test": True},
            unavailable_reason="test backend disabled",
        ),
    )

    def unavailable_quantile(*args, **kwargs):
        raise TabularQuantileBackendUnavailable("lightgbm", [{"test": True}])

    monkeypatch.setattr(optional_runner, "fit_tabular_quantile_model", unavailable_quantile)
    monkeypatch.setattr(optional_runner, "f1_model_runtime_doctor", lambda: {"test": True})

    audit_keys = range(202605, 202607)
    qualifying_champion = tmp_path / "qualifying_champion.json"
    best_lap_champion = tmp_path / "best_lap_champion.json"
    qualifying_champion.write_text(
        json.dumps(
            {
                "predictions": [
                    {
                        "event_key": event_key,
                        "driver_id": driver,
                        "predicted_qualifying_position": position,
                    }
                    for event_key in audit_keys
                    for position, driver in enumerate(("a", "b"), start=1)
                ]
            }
        ),
        encoding="utf-8",
    )
    best_lap_champion.write_text(
        json.dumps(
            {
                "predictions": [
                    {
                        "event_key": event_key,
                        "driver_id": driver,
                        "lap_p05": 89.0 + position,
                        "lap_p50": 89.5 + position,
                        "lap_p90": 90.0 + position,
                    }
                    for event_key in audit_keys
                    for position, driver in enumerate(("a", "b"), start=1)
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = optional_runner.run(
        weekends_dir=tmp_path,
        target_year=2026,
        qualifying_champion_predictions=qualifying_champion,
        best_lap_champion_predictions=best_lap_champion,
        bootstrap_samples=1_000,
    )

    assert payload["protocol"]["audit_outcomes_used_for_fit"] is False
    assert payload["lightgbm_quantile"]["status"] == "unavailable_selection_failed"
    assert payload["decision"]["promotion_eligible"] is False
    assert payload["protocol_sha256"]
    assert payload["champion_prediction_manifest_sha256"]
    assert set(payload["ranking"]) == {"xgboost_lambdarank", "lightgbm_lambdarank"}
