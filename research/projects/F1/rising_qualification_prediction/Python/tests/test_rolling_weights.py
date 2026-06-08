from __future__ import annotations

import pandas as pd

from packages.f1.data.schemas.session import PredictionConfig
from packages.f1.orchestration.prediction import _apply_season_sample_weighting
from packages.f1.models.training import EBRankModel


def test_apply_season_sample_weighting_marks_current_year_rows() -> None:
    config = PredictionConfig(
        source="local",
        mode="qualifying",
        year=2026,
        round_number=3,
        train_seasons=[2022, 2023, 2024, 2025, 2026],
        include_standings=False,
        cache_dir=None,
        meeting_name=None,
        country_name=None,
        weekends_dir=None,
        season_weight_year=2026,
        season_weight_multiplier=3.0,
    )
    notes: list[str] = []
    train = pd.DataFrame(
        {
            "event_year": [2025, 2026, 2026],
            "event_key": [202501, 202601, 202602],
            "target": [5.0, 2.0, 1.0],
        }
    )

    out = _apply_season_sample_weighting(train, config, notes)

    assert out["_sample_weight"].tolist() == [1.0, 3.0, 3.0]
    assert any("weighted_rows=2" in note for note in notes)
    assert any("weighted_events=2" in note for note in notes)


def test_eb_rank_model_uses_training_sample_weights() -> None:
    train = pd.DataFrame(
        {
            "driver_id": ["a", "b"],
            "event_year": [2025, 2026],
            "event_round": [1, 1],
            "event_key": [202501, 202601],
            "target": [1.0, 9.0],
            "_sample_weight": [1.0, 3.0],
        }
    )

    model = EBRankModel(decay=0.0, iterations=1)
    model.fit(train)

    assert abs(model.global_mean - 7.0) < 1e-9
