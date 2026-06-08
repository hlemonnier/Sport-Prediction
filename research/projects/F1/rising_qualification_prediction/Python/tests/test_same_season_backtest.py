from __future__ import annotations

from pathlib import Path

import pandas as pd

import run_f1_same_season_backtest as same_season
from packages.f1.orchestration.runtime import parse_train_seasons


class FakeProvider:
    def list_rounds(self, year: int) -> list[dict[str, object]]:
        assert year == 2026
        return [
            {"round_number": 1, "event_name": "Australia"},
            {"round_number": 2, "event_name": "China"},
        ]

    def get_race_results(self, year: int, round_number: int) -> pd.DataFrame:
        return pd.DataFrame({"driver_id": ["a", "b"], "driver_name": ["A", "B"], "position": [1.0, 2.0]})

    def get_qualifying_results(self, year: int, round_number: int) -> pd.DataFrame:
        return pd.DataFrame({"driver_id": ["a", "b"], "driver_name": ["A", "B"], "position": [1.0, 2.0]})

    def get_starting_grid(self, year: int, round_number: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "driver_id": ["a", "b"],
                "driver_name": ["A", "B"],
                "grid_position": [1.0, 2.0],
                "grid_source": ["test_grid", "test_grid"],
            },
        )

    def get_fp_features(self, year: int, round_number: int) -> pd.DataFrame:
        return pd.DataFrame({"driver_id": ["a", "b"], "driver_name": ["A", "B"], "event_pace_index": [1.0, 2.0]})


def test_same_season_training_rounds_stop_before_target() -> None:
    provider = FakeProvider()

    assert same_season._training_rounds_for_target(provider, "race", 2026, 1) == []  # type: ignore[arg-type]
    assert same_season._training_rounds_for_target(provider, "race", 2026, 2) == [1]  # type: ignore[arg-type]


def test_same_season_train_policy_resolves_to_target_year_only() -> None:
    assert parse_train_seasons("auto", 2026, "same_season_walk_forward") == [2026]
    assert parse_train_seasons("auto", 2026, "same_season") == [2026]


def test_same_season_backtest_forces_target_year_training_and_skips_first_round_models(
    monkeypatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "2026" / "round_01_australian_grand_prix").mkdir(parents=True)
    (tmp_path / "2026" / "round_02_chinese_grand_prix").mkdir(parents=True)
    captured_train_seasons: list[tuple[int, str, list[int]]] = []

    monkeypatch.setattr(same_season, "LocalWeekendProvider", lambda weekends_dir: FakeProvider())

    def fake_evaluate_variant(**kwargs: object) -> dict[str, object]:
        spec = kwargs["spec"]
        assert isinstance(spec, dict)
        captured_train_seasons.append((int(kwargs["round_number"]), str(spec["name"]), list(kwargs["train_seasons"])))
        return {
            "mode": kwargs["mode"],
            "round": kwargs["round_number"],
            "event_key": f"{kwargs['mode']}:{kwargs['round_number']}",
            "variant": spec["name"],
            "kind": spec["kind"],
            "model_name": spec["name"],
            "config_signature": {"kind": spec["kind"], "name": spec["name"]},
            "rows": 2,
            "metric_available": True,
            "field_mae": 0.0,
            "mae_on_common": 0.0,
            "field_coverage": 1.0,
            "top10_hit": 1.0,
            "podium_hit_count": 2,
            "winner_hit": 1,
            "evaluation": {"metric_available": True},
        }

    monkeypatch.setattr(same_season, "_evaluate_variant", fake_evaluate_variant)

    payload = same_season.build_same_season_backtest(
        weekends_dir=str(tmp_path),
        years=[2026],
        modes=["race"],
        round_start=None,
        round_end=None,
        f1_model="baseline",
        f1_pl_samples=25,
    )

    first_round_strategic = [
        row for row in payload["rows"] if row["round"] == 1 and row["variant"] == "strategic_race_delta"
    ]
    second_round_strategic = [
        row for row in payload["rows"] if row["round"] == 2 and row["variant"] == "strategic_race_delta"
    ]
    assert first_round_strategic[0]["skip_reason"] == "no_prior_same_season_training_events"
    assert second_round_strategic[0]["metric_available"] is True
    assert all(train_seasons == [2026] for _, _, train_seasons in captured_train_seasons)
    assert any(round_number == 2 and variant == "strategic_race_delta" for round_number, variant, _ in captured_train_seasons)
    assert payload["training_protocol"]["cross_season_training_allowed"] is False
    assert payload["season_summaries"]["2026"]["rounds_seen"] == [1, 2]
