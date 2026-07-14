from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import run_rolling_2026_backtest as rolling


class FakeProvider:
    def list_rounds(self, year: int) -> list[dict[str, object]]:
        assert year == 2026
        return [
            {"round_number": 1, "event_name": "Australia"},
            {"round_number": 9, "event_name": "Britain"},
        ]

    def get_qualifying_results(self, year: int, round_number: int) -> pd.DataFrame:
        return _actual_results()

    def get_race_results(self, year: int, round_number: int) -> pd.DataFrame:
        return _actual_results()


def _actual_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "driver_id": ["a", "b"],
            "driver_name": ["A", "B"],
            "position": [1.0, 2.0],
        }
    )


def _fake_prediction(config: object) -> SimpleNamespace:
    mode = str(getattr(config, "mode"))
    rows = [
        {
            "driver_id": "a",
            "driver_name": "A",
            "rank": 1,
            "pred": 1.0,
            "proba_win": 0.7,
            "proba_top3": 0.9,
            "proba_top10": 1.0,
        },
        {
            "driver_id": "b",
            "driver_name": "B",
            "rank": 2,
            "pred": 2.0,
            "proba_win": 0.3,
            "proba_top3": 0.8,
            "proba_top10": 1.0,
        },
    ]
    extras: dict[str, object] = {
        "all_prediction_rows": rows,
        "training_event_keys": [],
        "training_row_count": 0,
        "target_event_key_excluded_from_training": True,
        "qualifying_information_horizon": {
            "requested_cutoff": str(getattr(config, "qualifying_information_horizon")),
            "resolved_cutoffs": ["before_qualifying"],
            "prediction_as_of": None,
            "weekend_format_versions": ["sprint_2024_plus"],
        },
    }
    if mode == "race":
        extras["race_information_horizon"] = str(getattr(config, "race_information_horizon"))
        extras["prediction_phase"] = {"phase": "post_qualifying_pre_grid"}
    return SimpleNamespace(
        table=pd.DataFrame(rows),
        extras=extras,
        notes=[],
        model_name="fake",
        model_family="test",
        candidate_leaderboard=[],
    )


def _write_completed_round(root: Path, round_number: int) -> None:
    round_dir = root / "2026" / f"round_{round_number:02d}_event"
    round_dir.mkdir(parents=True)
    (round_dir / "04_qualifying_results.csv").write_text(
        "driver_id,position\na,1\nb,2\n",
        encoding="utf-8",
    )
    (round_dir / "05_race_results.csv").write_text(
        "driver_id,position\na,1\nb,2\n",
        encoding="utf-8",
    )


def test_defaults_are_dynamic_same_season_and_single_threaded() -> None:
    args = rolling.build_parser().parse_args([])

    assert args.rounds == "auto"
    assert args.experiment_arm == "same_season_walk_forward"
    assert args.transfer_train_seasons is None
    assert args.base_train_seasons is None
    assert args.current_season_weight_multiplier is None
    assert args.max_threads == 1
    assert args.compare_families == "baseline"
    assert args.f1_model == "baseline"
    assert args.qualifying_model is None
    assert args.race_model is None
    assert args.qualifying_runsim_features == "disabled"
    assert args.race_runsim_features == "enabled"
    assert args.qualifying_information_horizon == "pre_qualifying"
    assert args.race_information_horizon == "post_grid_pre_race"


def test_default_bound_profiles_pass_assertion_before_any_output_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def stop_after_profile_assertion(_path: Path, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("reached_output_setup_after_profile_assertion")

    monkeypatch.setattr(Path, "mkdir", stop_after_profile_assertion)

    with pytest.raises(RuntimeError, match="reached_output_setup_after_profile_assertion"):
        rolling.main(
            [
                "--rounds",
                "1,2,3,4,5,6,7,8,9",
                "--output-dir",
                str(tmp_path / "must_not_be_created"),
                "--run-id",
                "2026_four_mode_rebuild_20260712e",
                "--quiet",
            ]
        )

    assert not (tmp_path / "must_not_be_created").exists()


def test_profile_run_id_aliases_must_be_unambiguous() -> None:
    assert (
        rolling._profile_bound_run_id(
            {"promotion": {"baseline_rebuild_run_id": "run-a", "evidence_run_id": "run-a"}},
            label="qualifying",
        )
        == "run-a"
    )
    assert (
        rolling._profile_bound_run_id(
            {"promotion": {"evidence_run_id": "legacy-run"}},
            label="qualifying",
        )
        == "legacy-run"
    )
    with pytest.raises(ValueError, match="ambiguous promotion run IDs"):
        rolling._profile_bound_run_id(
            {"promotion": {"baseline_rebuild_run_id": "run-a", "evidence_run_id": "run-b"}},
            label="qualifying",
        )


def test_auto_round_discovery_uses_only_completed_local_targets(tmp_path: Path) -> None:
    _write_completed_round(tmp_path, 1)
    _write_completed_round(tmp_path, 9)
    incomplete = tmp_path / "2026" / "round_10_incomplete"
    incomplete.mkdir(parents=True)
    (incomplete / "04_qualifying_results.csv").write_text("driver_id,position\na,1\n", encoding="utf-8")
    (incomplete / "05_race_results.csv").write_text("driver_id,position\n", encoding="utf-8")

    assert rolling._available_local_rounds(str(tmp_path), 2026) == [1, 9]
    assert rolling._resolve_rounds("auto", source="local", weekends_dir=str(tmp_path), year=2026) == [1, 9]
    with pytest.raises(ValueError, match="local-only"):
        rolling._resolve_rounds("auto", source="fastf1", weekends_dir=str(tmp_path), year=2026)


def test_run_id_rejects_parent_directory_alias() -> None:
    with pytest.raises(ValueError, match="run-id"):
        rolling._validated_run_id("..")


def test_cross_regime_training_requires_explicit_separate_arm() -> None:
    primary = rolling._resolve_training_protocol(
        year=2026,
        experiment_arm="same_season_walk_forward",
        transfer_train_seasons=None,
        legacy_base_train_seasons=None,
        current_season_weight_multiplier=None,
    )
    assert primary["train_seasons_used"] == [2026]
    assert primary["current_season_weight_multiplier"] == 1.0
    assert primary["same_season_only"] is True

    with pytest.raises(ValueError, match="forbidden"):
        rolling._resolve_training_protocol(
            year=2026,
            experiment_arm="same_season_walk_forward",
            transfer_train_seasons="2025",
            legacy_base_train_seasons=None,
            current_season_weight_multiplier=None,
        )

    transfer = rolling._resolve_training_protocol(
        year=2026,
        experiment_arm="explicit_transfer",
        transfer_train_seasons="2022,2024,2025",
        legacy_base_train_seasons=None,
        current_season_weight_multiplier=2.0,
    )
    assert transfer["train_seasons_used"] == [2022, 2024, 2025, 2026]
    assert transfer["same_season_only"] is False
    assert transfer["cross_regime_comparability"] == "exploratory_non_primary_transfer_arm"


def test_target_runsim_policy_preserves_legacy_flag_and_allows_overrides() -> None:
    assert rolling._resolve_runsim_disable(legacy_disable=False, target_policy="inherit") is False
    assert rolling._resolve_runsim_disable(legacy_disable=True, target_policy="inherit") is True
    assert rolling._resolve_runsim_disable(legacy_disable=True, target_policy="enabled") is False
    assert rolling._resolve_runsim_disable(legacy_disable=False, target_policy="disabled") is True


def test_main_writes_exclusive_reproducibility_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    weekends = tmp_path / "weekends"
    _write_completed_round(weekends, 1)
    _write_completed_round(weekends, 9)
    captured_configs: list[object] = []

    monkeypatch.setattr(rolling, "_provider", lambda *args, **kwargs: FakeProvider())

    def fake_run_prediction(config: object) -> SimpleNamespace:
        captured_configs.append(config)
        return _fake_prediction(config)

    monkeypatch.setattr(rolling, "run_prediction", fake_run_prediction)
    output = tmp_path / "artifacts"
    qualifying_profile = tmp_path / "pre_quali.yaml"
    race_profile = tmp_path / "pre_race.yaml"
    shared_profile = {
        "source": "local",
        "field_size": 22,
        "training": {
            "protocol": "same_season_walk_forward",
            "seasons": [2026],
        },
        "promotion": {
            "evidence_run_id": "test-run",
            "frozen_rounds": [1, 9],
        },
    }
    qualifying_profile.write_text(
        json.dumps(
            {
                **shared_profile,
                "information_horizon": "pre_qualifying",
                "model": {"requested": "auto", "compare_families": ["baseline"]},
                "features": {"run_simulation_features": False, "standings": False},
            }
        ),
        encoding="utf-8",
    )
    race_profile.write_text(
        json.dumps(
            {
                **shared_profile,
                "information_horizon": "post_qualifying_pre_grid",
                "model": {"requested": "baseline"},
                "features": {"run_simulation_features": True, "standings": False},
            }
        ),
        encoding="utf-8",
    )
    arguments = [
        "--weekends-dir",
        str(weekends),
        "--output-dir",
        str(output),
        "--run-id",
        "test-run",
        "--qualifying-model",
        "auto",
        "--race-model",
        "baseline",
        "--compare-families",
        "baseline",
        "--qualifying-runsim-features",
        "disabled",
        "--race-runsim-features",
        "enabled",
        "--race-information-horizon",
        "post_qualifying_pre_grid",
        "--qualifying-profile",
        str(qualifying_profile),
        "--race-profile",
        str(race_profile),
        "--quiet",
    ]

    rolling.main(arguments)

    run_root = output / "runs" / "test-run"
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert [config.mode for config in captured_configs] == ["qualifying", "race", "qualifying", "race"]
    assert all(config.train_seasons == [2026] for config in captured_configs)
    assert all(config.season_weight_multiplier == 1.0 for config in captured_configs)
    assert [config.f1_model for config in captured_configs] == ["auto", "baseline", "auto", "baseline"]
    assert [config.disable_runsim_features for config in captured_configs] == [True, False, True, False]
    assert manifest["schema_version"] == rolling.ROLLING_BACKTEST_SCHEMA_VERSION
    assert manifest["training_protocol"]["same_season_only"] is True
    assert manifest["training_protocol"]["parallel_rounds"] is False
    assert manifest["runtime"]["resource_limits"]["max_threads"] == 1
    assert manifest["git"]["head_sha"]
    assert manifest["implementation"]["aggregate_sha256"]
    assert manifest["configuration_files"]["aggregate_sha256"]
    assert manifest["run_configuration"]["sha256"]
    assert manifest["run_configuration"]["payload"]["qualifying_model"] == "auto"
    assert manifest["run_configuration"]["payload"]["race_model"] == "baseline"
    assert manifest["run_configuration"]["payload"]["target_disable_runsim_features"] == {
        "qualifying": True,
        "race": False,
    }
    assert set(manifest["input_data_by_round"]) == {"1", "9"}
    assert manifest["input_data_by_round"]["1"]["file_count"] == 2
    assert manifest["input_data_by_round"]["9"]["file_count"] == 4
    assert all(item["horizon_evidence_complete"] for item in manifest["point_in_time_by_round"])
    assert all(item["sha256"] for item in manifest["artifacts"])
    assert manifest["artifact_contract"]["write_mode"] == "exclusive_create"

    with pytest.raises(FileExistsError):
        rolling.main(arguments)
