from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import packages.f1.models.grouped_ranking as grouped_ranking
from packages.f1.models.grouped_ranking import (
    GroupedRankingConfig,
    build_event_pure_pairwise_dataset,
    fit_grouped_ranking_challenger,
    prepare_grouped_ranking_dataset,
    same_season_walk_forward_partitions,
)
from packages.f1.orchestration.model_runtime import inspect_optional_model_runtime


DRIVERS = ("VER", "NOR", "LEC", "PIA")


def _events(season: int, count: int, *, start: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range(start, periods=count, freq="14D", tz="UTC")
    for event_index, event_time in enumerate(dates, start=1):
        event_key = f"{season}-r{event_index}"
        for position, driver in enumerate(DRIVERS, start=1):
            rows.append(
                {
                    "season": season,
                    "event_key": event_key,
                    "event_as_of": event_time,
                    "driver_id": driver,
                    "pace_score": float(5 - position) + 0.02 * event_index,
                    "grid_position": float(position),
                    "team_id": "A" if driver in {"VER", "NOR"} else "B",
                    "position": position,
                }
            )
    return pd.DataFrame(rows)


def _config(backend: str = "sklearn_pairwise") -> GroupedRankingConfig:
    return GroupedRankingConfig(
        feature_columns=("pace_score", "grid_position", "team_id"),
        backend=backend,
        n_estimators=5,
        max_iterations=200,
        minimum_training_events=2,
    )


def test_grouped_dataset_is_chronological_and_pairs_never_cross_events() -> None:
    frame = _events(2026, 3, start="2026-03-01").sample(frac=1.0, random_state=7)
    dataset = prepare_grouped_ranking_dataset(frame, config=_config())
    pairs = build_event_pure_pairwise_dataset(dataset)

    assert dataset.group_sizes == (4, 4, 4)
    assert list(dataset.event_times) == sorted(dataset.event_times)
    assert pairs.cross_event_pair_count == 0
    assert set(pairs.undirected_pairs_by_event.values()) == {6}
    assert set(pairs.pair_rows_by_event.values()) == {12}
    assert pairs.values.shape[0] == 36
    assert np.isclose(sum(pairs.sample_weight), 3.0)


def test_same_season_walk_forward_excludes_prior_season_absolute_pace() -> None:
    history = pd.concat(
        [
            _events(2025, 3, start="2025-03-01"),
            _events(2026, 4, start="2026-03-01"),
        ],
        ignore_index=True,
    )

    folds = same_season_walk_forward_partitions(
        history,
        config=_config(),
        target_season=2026,
        minimum_training_events=2,
    )

    assert len(folds) == 2
    assert [fold.training["event_key"].nunique() for fold in folds] == [2, 3]
    assert all(set(fold.training["season"]) == {2026} for fold in folds)
    assert all(set(fold.validation["season"]) == {2026} for fold in folds)
    assert all(fold.training_max_time < fold.validation_min_time for fold in folds)


def test_sklearn_fit_has_reproducible_manifest_and_separate_selection_history() -> None:
    prior_pace = _events(2025, 2, start="2025-03-01")
    current_pace = _events(2026, 3, start="2026-03-01")
    selection = _events(2024, 2, start="2024-03-01")
    records = pd.concat([prior_pace, current_pace], ignore_index=True)
    config = _config()

    first = fit_grouped_ranking_challenger(
        records,
        config=config,
        target_season=2026,
        selection_records=selection,
    )
    second = fit_grouped_ranking_challenger(
        records,
        config=config,
        target_season=2026,
        selection_records=selection,
    )

    assert first.status == "available"
    assert first.available is True
    assert first.manifest["training_season"] == 2026
    assert first.manifest["season_transfer_policy"] == "same_season_absolute_pace_only"
    assert first.manifest["other_season_rows_excluded_from_fit"] == len(prior_pace)
    assert (
        first.manifest["hyperparameter_selection_history"]["role"]
        == "hyperparameter_selection_only_not_model_fit"
    )
    assert first.manifest["hyperparameter_selection_history"]["status"] == "provided_disjoint"
    assert first.manifest["pairwise_audit"]["cross_event_pair_count"] == 0
    assert first.manifest["config_sha256"] == second.manifest["config_sha256"]
    assert first.manifest["training_data_sha256"] == second.manifest["training_data_sha256"]

    future = _events(2026, 1, start="2026-05-01")
    prediction = first.require_model().predict(future)
    assert sorted(prediction["predicted_rank"].tolist()) == [1, 2, 3, 4]
    np.testing.assert_allclose(
        prediction["ranking_score"],
        second.require_model().predict(future)["ranking_score"],
    )

    empty = first.require_model().predict(future.iloc[0:0])
    assert list(empty.columns) == [
        "event_key",
        "driver_id",
        "ranking_score",
        "predicted_rank",
        "ranking_model",
    ]


def test_selection_history_must_be_disjoint_and_strictly_earlier() -> None:
    current = _events(2026, 3, start="2026-03-01")

    with pytest.raises(ValueError, match="disjoint"):
        fit_grouped_ranking_challenger(
            current,
            config=_config(),
            target_season=2026,
            selection_records=current.iloc[:4].copy(),
        )


def test_explicit_native_backend_returns_unavailable_instead_of_silent_fallback(
    monkeypatch,
) -> None:
    def unavailable(*args, **kwargs):
        raise RuntimeError("native runtime intentionally unavailable")

    monkeypatch.setattr(grouped_ranking, "_fit_backend", unavailable)
    result = fit_grouped_ranking_challenger(
        _events(2026, 3, start="2026-03-01"),
        config=_config("lightgbm_lambdarank"),
        target_season=2026,
    )

    assert result.status == "unavailable"
    assert result.available is False
    assert result.model is None
    assert result.manifest["selected_backend"] is None
    assert [attempt["backend"] for attempt in result.manifest["backend_attempts"]] == [
        "lightgbm_lambdarank"
    ]


@pytest.mark.parametrize(
    ("backend", "package"),
    [
        ("xgboost_lambdarank", "xgboost"),
        ("lightgbm_lambdarank", "lightgbm"),
    ],
)
def test_native_lambdarank_adapter_when_runtime_is_available(
    backend: str, package: str
) -> None:
    runtime = inspect_optional_model_runtime(package)
    if not runtime.available:
        pytest.skip(f"{package} optional runtime unavailable: {runtime.issue}")

    result = fit_grouped_ranking_challenger(
        _events(2026, 3, start="2026-03-01"),
        config=_config(backend),
        target_season=2026,
    )

    assert result.status == "available"
    assert result.manifest["selected_backend"] == backend
    assert result.manifest["group_sizes"] == [4, 4, 4]
    assert result.manifest["backend_attempts"][0]["cross_event_pair_count"] == 0
