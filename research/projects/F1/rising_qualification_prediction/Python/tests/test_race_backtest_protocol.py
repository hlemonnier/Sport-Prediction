from __future__ import annotations

import inspect
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
    _MINIMUM_RELATIVE_SELECTION_GAIN,
    _SAME_SEASON_MINIMUM_PRIOR_EVENTS,
    _NO_SAME_HORIZON_ORDER_RESIDUAL_WEIGHT,
    RACE_BACKTEST_SCHEMA_VERSION,
    _apply_selected_position_head,
    _attach_result_sha256,
    _audit_aggregate_payload,
    _align_grid_driver_ids_from_qualifying,
    _blocked_terminal_calibration_lock,
    _build_event_rows,
    _canonicalize_qualifying_driver_identity,
    _canonical_json_sha256,
    _deterministic_rank,
    _event_is_in_partition,
    _fit_binary_terminal_calibrator,
    _legal_grid_baseline,
    _map_driver_ids_from_qualifying,
    _normalize_identity_token,
    _resolve_missing_race_targets,
    _resolve_session_reference,
    _rolling_oof_qualifying_prior,
    _same_season_event_partitions,
    _same_product_promotion_blockers,
    _select_race_policy,
    _set_prediction_order_residual_weight,
    _stable_provisional_grid_positions,
    _strict_prior_history,
    build_parser,
    run,
)
from packages.f1.models.pre_race.joint import SurvivalAwareRaceModel
from packages.f1.models.pre_race.ranking import (
    BradleyTerryOrderRanker,
    ConditionalOrderConfig,
)


def test_order_residual_weight_is_a_score_time_parameter_only() -> None:
    ranker = BradleyTerryOrderRanker(
        ConditionalOrderConfig(
            regularization_c=0.7,
            grid_prior_weight=2.5,
            residual_weight=0.25,
            cold_start_event_k=6.0,
            coefficient_bound=4.0,
            max_iter=321,
            random_state=19,
        )
    )
    model = SurvivalAwareRaceModel(order_model=ranker)
    ranker_identity = id(model.order_model)
    original = model.order_model.config

    _set_prediction_order_residual_weight(model, 0.65)

    assert id(model.order_model) == ranker_identity
    assert model.order_model.config.residual_weight == 0.65
    assert model.order_model.config.regularization_c == original.regularization_c
    assert model.order_model.config.grid_prior_weight == original.grid_prior_weight
    assert model.order_model.config.cold_start_event_k == original.cold_start_event_k
    assert model.order_model.config.coefficient_bound == original.coefficient_bound
    assert model.order_model.config.max_iter == original.max_iter
    assert model.order_model.config.random_state == original.random_state


def test_race_backtest_searches_zero_residual_and_falls_back_to_grid_only() -> None:
    defaults = inspect.signature(run).parameters

    assert 0.0 in defaults["order_residual_candidates"].default
    assert max(defaults["order_residual_candidates"].default) > 0.65
    assert min(defaults["temperature_candidates"].default) < 0.18
    assert _NO_SAME_HORIZON_ORDER_RESIDUAL_WEIGHT == 0.0


def test_same_season_partitions_use_exact_complete_event_order() -> None:
    partitions = _same_season_event_partitions(
        tuple(range(202601, 202610)), target_year=2026
    )

    assert partitions == {
        "development": ["202601", "202602"],
        "selection": ["202603", "202604"],
        "calibration": ["202605", "202606"],
        "audit": ["202607", "202608", "202609"],
    }
    assert _SAME_SEASON_MINIMUM_PRIOR_EVENTS == 2
    with pytest.raises(ValueError, match="outside target year"):
        _same_season_event_partitions(
            (202501, *range(202601, 202608)), target_year=2026
        )
    with pytest.raises(ValueError, match="at least eight complete events"):
        _same_season_event_partitions(
            tuple(range(202601, 202608)), target_year=2026
        )


@pytest.mark.parametrize(
    ("event_count", "selection", "calibration", "audit"),
    (
        (8, range(202603, 202605), range(202605, 202607), range(202607, 202609)),
        (9, range(202603, 202605), range(202605, 202607), range(202607, 202610)),
        (12, range(202603, 202607), range(202607, 202610), range(202610, 202613)),
        (13, range(202603, 202607), range(202607, 202611), range(202611, 202614)),
    ),
)
def test_same_season_partitions_expand_only_when_independent_blocks_exist(
    event_count: int,
    selection: range,
    calibration: range,
    audit: range,
) -> None:
    partitions = _same_season_event_partitions(
        tuple(range(202601, 202601 + event_count)),
        target_year=2026,
    )

    assert partitions["development"] == ["202601", "202602"]
    assert partitions["selection"] == [str(value) for value in selection]
    assert partitions["calibration"] == [str(value) for value in calibration]
    assert partitions["audit"] == [str(value) for value in audit]


def test_mixed_horizons_cannot_pool_events_to_clear_same_horizon_locks() -> None:
    partitions = _same_season_event_partitions(
        tuple(range(202601, 202614)),
        target_year=2026,
    )
    horizons = {
        event_key: ("post_grid_pre_race" if event_key % 2 else "post_qualifying_pre_grid")
        for event_key in range(202601, 202614)
    }
    horizon = "post_grid_pre_race"
    selection_count = sum(
        horizons[int(event_key)] == horizon for event_key in partitions["selection"]
    )
    calibration_count = sum(
        horizons[int(event_key)] == horizon for event_key in partitions["calibration"]
    )
    audit_count = sum(
        horizons[int(event_key)] == horizon for event_key in partitions["audit"]
    )

    blockers = _same_product_promotion_blockers(
        audit_event_count=audit_count,
        same_product_selection_evidence=True,
        same_product_calibration_evidence=True,
        selection_event_count=selection_count,
        calibration_event_count=calibration_count,
    )

    assert blockers == (
        "fewer_than_three_same_horizon_audit_events",
        "fewer_than_four_same_horizon_selection_events",
        "fewer_than_four_same_horizon_calibration_events",
    )


def test_default_race_parser_is_2026_same_season_with_explicit_legacy_opt_in() -> None:
    parser = build_parser()
    defaults = parser.parse_args([])

    assert defaults.years == (2026,)
    assert defaults.evaluation_years == (2026,)
    assert defaults.legacy_cross_season is False
    assert defaults.output.name == "survival_order_same_season_v1.json"

    legacy = parser.parse_args(
        [
            "--legacy-cross-season",
            "--years",
            "2022,2023,2024,2025,2026",
            "--evaluation-years",
            "2024,2025,2026",
        ]
    )
    assert legacy.legacy_cross_season is True
    assert legacy.years == (2022, 2023, 2024, 2025, 2026)


def test_partition_membership_is_event_exact_not_year_wide() -> None:
    partitions = _same_season_event_partitions(
        tuple(range(202601, 202610)), target_year=2026
    )

    assert _event_is_in_partition(202605, partitions, "calibration") is True
    assert _event_is_in_partition(202605, partitions, "audit") is False
    assert _event_is_in_partition(202607, partitions, "audit") is True
    assert _event_is_in_partition(202699, partitions, "audit") is False
    with pytest.raises(ValueError, match="unknown Race event partition"):
        _event_is_in_partition(202607, partitions, "unknown")


def test_same_season_prior_history_is_strictly_earlier_and_excludes_prior_years() -> None:
    history = pd.DataFrame(
        {
            "event_key": [202501, 202601, 202602, 202603, 202604, 202605],
            "driver_id": ["old", "a", "b", "c", "target", "future"],
        }
    )

    same_season = _strict_prior_history(
        history, event_key=202604, same_season_only=True
    )
    legacy = _strict_prior_history(
        history, event_key=202604, same_season_only=False
    )

    assert same_season["event_key"].tolist() == [202601, 202602, 202603]
    assert same_season["event_key"].nunique() == _SAME_SEASON_MINIMUM_PRIOR_EVENTS + 1
    assert legacy["event_key"].tolist() == [202501, 202601, 202602, 202603]
    assert 202604 not in same_season["event_key"].tolist()
    assert 202605 not in same_season["event_key"].tolist()
    assert inspect.signature(run).parameters["same_season_only"].default is True


def test_canonical_manifest_hash_is_order_stable_and_value_sensitive() -> None:
    first = {"b": [2, 3], "a": {"x": 1}}
    reordered = {"a": {"x": 1}, "b": [2, 3]}
    changed = {"a": {"x": 1}, "b": [2, 4]}

    assert _canonical_json_sha256(first) == _canonical_json_sha256(reordered)
    assert _canonical_json_sha256(first) != _canonical_json_sha256(changed)


def test_qualifying_abbreviation_is_the_stable_longitudinal_driver_identity() -> None:
    first, first_lookup = _canonicalize_qualifying_driver_identity(
        pd.DataFrame(
            {
                "driver_id": ["1", "63"],
                "car_number": ["1", "63"],
                "driver_abbreviation": ["VER", "RUS"],
            }
        )
    )
    later, later_lookup = _canonicalize_qualifying_driver_identity(
        pd.DataFrame(
            {
                "driver_id": ["3", "63"],
                "car_number": ["3", "63"],
                "driver_abbreviation": ["VER", "RUS"],
            }
        )
    )

    assert first["driver_id"].tolist() == ["VER", "RUS"]
    assert later["driver_id"].tolist() == ["VER", "RUS"]
    assert first_lookup["1"] == later_lookup["3"] == "VER"
    assert first["provider_driver_id"].tolist() == ["1", "63"]


def test_same_weekend_sources_map_only_through_frozen_qualifying_identity() -> None:
    _, lookup = _canonicalize_qualifying_driver_identity(
        pd.DataFrame(
            {
                "driver_id": ["1", "63"],
                "car_number": ["1", "63"],
                "driver_abbreviation": ["VER", "RUS"],
            }
        )
    )
    practice = _map_driver_ids_from_qualifying(
        pd.DataFrame(
            {
                "driver_id": ["1", "63", "99"],
                "fp_mean_rank": [1.0, 2.0, 3.0],
            }
        ),
        lookup,
        source_name="practice",
        allow_non_roster_rows=True,
    )

    assert practice["driver_id"].tolist() == ["VER", "RUS"]
    assert practice["practice_provider_driver_id"].tolist() == ["1", "63"]
    with pytest.raises(ValueError, match="absent from the causal pre-race roster"):
        _map_driver_ids_from_qualifying(
            pd.DataFrame({"driver_id": ["1", "99"]}),
            lookup,
            source_name="race",
            allow_non_roster_rows=False,
        )


def test_numeric_identity_aliases_are_normalized_before_alignment() -> None:
    canonical, lookup = _canonicalize_qualifying_driver_identity(
        pd.DataFrame(
            {
                "driver_id": [1.0, 63.0],
                "car_number": [1.0, 63.0],
                "driver_abbreviation": ["ver", "rus"],
            }
        )
    )
    mapped = _map_driver_ids_from_qualifying(
        pd.DataFrame({"driver_id": ["1.0", "63"]}),
        lookup,
        source_name="race",
        allow_non_roster_rows=False,
    )

    assert _normalize_identity_token("1.0") == "1"
    assert canonical["provider_driver_id"].tolist() == ["1", "63"]
    assert mapped["driver_id"].tolist() == ["VER", "RUS"]


def test_duplicate_stable_driver_identity_fails_closed() -> None:
    with pytest.raises(ValueError, match="duplicate stable driver identities"):
        _canonicalize_qualifying_driver_identity(
            pd.DataFrame(
                {
                    "driver_id": ["1", "3"],
                    "car_number": ["1", "3"],
                    "driver_abbreviation": ["VER", "VER"],
                }
            )
        )


def test_first_seen_grid_can_authoritatively_complete_a_nonstarter_target() -> None:
    resolved = _resolve_missing_race_targets(
        pd.DataFrame(
            {
                "driver_id": ["RUS", "STR"],
                "race_target_observed": [True, pd.NA],
                "race_target_source": ["provider_race_classification", pd.NA],
                "race_status_raw": ["Finished", pd.NA],
                "race_status_evidence_complete": [True, pd.NA],
                "finish_position": [1.0, np.nan],
                "retirement_fraction": [1.0, np.nan],
                "laps_completed": [60.0, np.nan],
                "grid_evidence_complete": [True, True],
                "grid_publication_pre_race_verified": [True, True],
                "grid_starter_eligible": [1.0, 0.0],
                "grid_baseline_position": [1.0, 2.0],
            }
        ),
        final_capture=True,
    )

    nonstarter = resolved.loc[resolved["driver_id"].eq("STR")].iloc[0]
    assert bool(nonstarter["race_target_observed"]) is True
    assert nonstarter["race_target_source"] == "official_first_seen_grid_nonstarter"
    assert nonstarter["race_status_raw"] == "Did not start"
    assert nonstarter["finish_position"] == 2.0
    assert nonstarter["retirement_fraction"] == 0.0


def test_legacy_metadata_reference_uses_weekend_basename_fallback(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    weekend = root / "data/f1/raw/weekends/2025/round_09_fake"
    weekend.mkdir(parents=True)
    actual = weekend / "01_practice_1_results.csv"
    actual.write_text("DriverNumber,Status\n1,Finished\n", encoding="utf-8")

    resolved = _resolve_session_reference(
        root,
        weekend,
        "data/f1/weekends/2025/round_09_fake/01_practice_1_results.csv",
    )

    assert resolved == actual


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


def test_final_joint_probability_calibration_rejects_zero_shock_platt_claim() -> None:
    rows = pd.DataFrame(
        {
            "event_key": ["202605", "202605", "202606", "202606"],
            "actual_terminal": [0.0, 1.0, 0.0, 1.0],
            "predicted_terminal": [0.10, 0.40, 0.20, 0.60],
        }
    )

    lock = _blocked_terminal_calibration_lock(
        rows,
        information_horizon="post_grid_pre_race",
        simulations_per_event=400,
    )

    assert lock["same_product_calibration_evidence"] is False
    assert lock["calibration_fit_probability_source"] is None
    assert lock["calibration_application_probability_source"] is None
    assert lock["scored_probability_source"] == (
        "empirical_post_shared_shock_joint_samples"
    )
    assert lock["calibration_status"] == (
        "blocked_requires_simulation_in_loop_or_marginal_preserving_copula"
    )
    assert lock["error"] == (
        "zero_shock_platt_rejected_for_final_distribution_mismatch"
    )
    assert lock["event_keys"] == ["202605", "202606"]
    assert lock["raw_terminal_brier"] is not None
    assert lock["raw_terminal_log_loss"] is not None


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


def test_already_canonical_grid_still_retains_identity_provenance() -> None:
    aligned = _align_grid_driver_ids_from_qualifying(
        pd.DataFrame(
            {
                "driver_id": ["VER", "RUS"],
                "grid_position": [1, 2],
            }
        ),
        pd.DataFrame(
            {
                "driver_id": ["VER", "RUS"],
                "car_number": ["3", "63"],
            }
        ),
    )

    assert aligned["grid_provider_driver_id"].tolist() == ["VER", "RUS"]
    assert set(aligned["grid_identity_mapping_source"]) == {
        "provider_driver_id_matches_pre_race_qualifying_canonical_id"
    }


def test_provisional_grid_resolves_provider_ties_by_official_source_order() -> None:
    frame = pd.DataFrame(
        {
            "driver_id": ["a", "b", "c", "d", "e"],
            "qualy_position": [1.0, 2.0, 2.0, 3.0, 5.0],
        },
        index=[10, 20, 30, 40, 50],
    )

    positions = _stable_provisional_grid_positions(frame)

    assert positions.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert positions.index.tolist() == frame.index.tolist()


def test_legal_grid_baseline_ties_preserve_provider_order_not_driver_alphabet() -> None:
    frame = pd.DataFrame(
        {
            "driver_id": ["ZED", "ALP", "MID"],
            "grid_status": ["pit_lane", "pit_lane", "unplaced"],
            "grid_position": [np.nan, np.nan, np.nan],
        },
        index=[20, 10, 30],
    )

    positions = _legal_grid_baseline(frame)

    assert positions.to_dict() == {20: 1.0, 10: 2.0, 30: 3.0}


def test_qualifying_prior_score_ties_preserve_provider_order() -> None:
    ranks = _deterministic_rank(np.asarray([0.5, 0.5, np.nan, np.nan]))

    assert ranks.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_unproven_missing_race_target_fails_closed_after_feature_freeze(
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
                    "driver_id": ["1", "2", "99"],
                    "fp_quali_sim_rank": [1.0, 2.0, 99.0],
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
                    "driver_id": ["1"],
                    "position": [1],
                    "team_name": ["LEAKED_A"],
                    "power_unit": ["LEAKED_PU_A"],
                    "race_status_raw": ["Finished"],
                    "race_status_evidence_complete": [True],
                    "retirement_fraction": [1.0],
                    "laps_completed": [60],
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
    assert frame["driver_id"].tolist() == ["AAA", "BBB"]
    assert frame["provider_driver_id"].tolist() == ["1", "2"]
    assert frame["fp_quali_sim_rank"].tolist() == [1.0, 2.0]
    assert frame.loc[frame["driver_id"].eq("AAA"), "race_provider_driver_id"].iloc[0] == "1"
    assert pd.isna(
        frame.loc[frame["driver_id"].eq("BBB"), "race_provider_driver_id"].iloc[0]
    )
    assert pd.isna(
        frame.loc[frame["driver_id"].eq("BBB"), "race_status_raw"].iloc[0]
    )
    assert (
        frame.loc[frame["driver_id"].eq("BBB"), "race_target_observed"].iloc[0]
        is False
    )
    assert info["race_target_coverage_complete"] is False
    assert info["race_target_missing_driver_ids"] == ["BBB"]
    assert info["qualifying_snapshot_provenance"]["first_seen_verified"] is False
    assert "retrospective" in info["driver_identity_contract"]["canonical_source"]
    assert "race_team_name" not in frame.columns
    assert "race_power_unit" not in frame.columns
    assert "finish_position" not in info["causal_inference_columns"]
    assert "terminal_status" not in info["causal_inference_columns"]
    assert info["race_truth_attached_after_inference_freeze"] is True
    assert info["driver_identity_contract"]["race_truth_allowed_to_repair_identity"] is False


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


def test_race_policy_retains_grid_when_challenger_does_not_clear_selection_gate() -> None:
    challenger = [
        {
            "model_id": "survival_aware_joint",
            "information_horizon": "post_grid_pre_race",
            "plackett_luce_temperature": 0.18,
            "order_residual_weight": 0.65,
            "mean_position_mae": 2.91,
            "event_count": 2,
            "event_keys": ["202603", "202604"],
        }
    ]
    baseline = {
        "model_id": "legal_grid_baseline",
        "information_horizon": "post_grid_pre_race",
        "mean_position_mae": 2.73,
        "event_count": 2,
        "event_keys": ["202603", "202604"],
    }

    selected = _select_race_policy(
        challenger,
        baseline_row=baseline,
        diagnostic_temperature=0.25,
    )

    assert selected["selected_model_id"] == "legal_grid_baseline"
    assert selected["challenger_selected"] is False
    assert selected["relative_challenger_gain"] < 0.0
    assert (
        selected["minimum_relative_selection_gain"]
        == _MINIMUM_RELATIVE_SELECTION_GAIN
    )


def test_rejected_challenger_cannot_leak_through_public_position_alias() -> None:
    scored = pd.DataFrame(
        {
            "driver_id": ["A", "B", "C"],
            "grid_baseline_position": [1, 2, 3],
            "candidate_predicted_position": [2, 1, 3],
        }
    )

    retained = _apply_selected_position_head(
        scored,
        selected_model_id="legal_grid_baseline",
    )

    assert retained["candidate_predicted_position"].tolist() == [2, 1, 3]
    assert retained["selected_predicted_position"].tolist() == [1, 2, 3]
    assert retained["predicted_position"].tolist() == [1, 2, 3]
    assert retained["predicted_position"].equals(
        retained["selected_predicted_position"]
    )
    assert retained["selected_model_id"].unique().tolist() == [
        "legal_grid_baseline"
    ]
    assert retained["selected_position_source"].unique().tolist() == [
        "grid_baseline_position"
    ]


def test_selected_challenger_is_public_only_after_selection_gate() -> None:
    scored = pd.DataFrame(
        {
            "driver_id": ["A", "B", "C"],
            "grid_baseline_position": [1, 2, 3],
            "candidate_predicted_position": [2, 1, 3],
        }
    )

    selected = _apply_selected_position_head(
        scored,
        selected_model_id="survival_aware_joint",
    )

    assert selected["candidate_predicted_position"].tolist() == [2, 1, 3]
    assert selected["selected_predicted_position"].tolist() == [2, 1, 3]
    assert selected["predicted_position"].tolist() == [2, 1, 3]
    assert selected["selected_position_source"].unique().tolist() == [
        "candidate_predicted_position"
    ]


def _aggregate_event(
    event_key: int,
    *,
    baseline_mae: float,
    candidate_mae: float,
    selected_mae: float,
) -> dict[str, object]:
    return {
        "event_key": event_key,
        "year": event_key // 100,
        "information_horizon": "post_grid_pre_race",
        "baseline_mae": baseline_mae,
        "candidate_mae": candidate_mae,
        "selected_mae": selected_mae,
        "baseline_kendall": 0.7,
        "candidate_kendall": 0.6,
        "selected_kendall": 0.7,
        "baseline_status_brier": 0.2,
        "candidate_status_brier": 0.3,
        "selected_status_brier": 0.2,
        "baseline_status_log_loss": 0.4,
        "candidate_status_log_loss": 0.5,
        "selected_status_log_loss": 0.4,
        "candidate_status_terminal_ece": 0.1,
        "candidate_retirement_fraction_mae": 0.25,
        "global_meal_mae": candidate_mae,
        "dns_constrained_meal_mae": candidate_mae,
        "dns_constrained_minus_global_meal_mae": 0.0,
    }


def test_audit_aggregate_excludes_selection_and_calibration_events() -> None:
    events = [
        _aggregate_event(
            202606,
            baseline_mae=9.0,
            candidate_mae=1.0,
            selected_mae=9.0,
        ),
        _aggregate_event(
            202607,
            baseline_mae=2.0,
            candidate_mae=3.0,
            selected_mae=2.0,
        ),
        _aggregate_event(
            202608,
            baseline_mae=4.0,
            candidate_mae=6.0,
            selected_mae=4.0,
        ),
    ]

    audit = _audit_aggregate_payload(
        events,
        audit_event_keys=[202607, 202608],
    )

    assert audit["partition_role"] == "audit"
    assert audit["event_keys"] == [202607, 202608]
    assert audit["events"] == 2
    assert audit["baseline_mean_mae"] == pytest.approx(3.0)
    assert audit["candidate_mean_mae"] == pytest.approx(4.5)
    assert audit["selected_mean_mae"] == pytest.approx(3.0)
    assert audit["by_year"]["2026"]["events"] == 2
    assert audit["by_horizon"]["post_grid_pre_race"]["events"] == 2


def test_v8_result_hash_binds_every_field_except_itself() -> None:
    payload = {
        "schema_version": RACE_BACKTEST_SCHEMA_VERSION,
        "events": [{"event_key": 202607, "selected_mae": 2.0}],
        "predictions": [{"driver_id": "A", "predicted_position": 1}],
    }

    finalized = _attach_result_sha256(payload)
    without_hash = dict(finalized)
    observed = without_hash.pop("result_sha256")

    assert RACE_BACKTEST_SCHEMA_VERSION.endswith("_v8")
    assert observed == _canonical_json_sha256(without_hash)
    with pytest.raises(ValueError, match="must not exist before finalization"):
        _attach_result_sha256(finalized)


def test_race_policy_selects_challenger_only_after_material_same_event_gain() -> None:
    challenger = [
        {
            "model_id": "survival_aware_joint",
            "information_horizon": "post_grid_pre_race",
            "plackett_luce_temperature": 0.25,
            "order_residual_weight": 0.25,
            "mean_position_mae": 2.50,
            "event_count": 2,
            "event_keys": ["202603", "202604"],
        }
    ]
    baseline = {
        "model_id": "legal_grid_baseline",
        "information_horizon": "post_grid_pre_race",
        "mean_position_mae": 3.00,
        "event_count": 2,
        "event_keys": ["202603", "202604"],
    }

    selected = _select_race_policy(
        challenger,
        baseline_row=baseline,
        diagnostic_temperature=0.25,
    )

    assert selected["selected_model_id"] == "survival_aware_joint"
    assert selected["challenger_selected"] is True
    assert selected["relative_challenger_gain"] == pytest.approx(1.0 / 6.0)


def test_race_promotion_records_selection_rejection_separately() -> None:
    blockers = _same_product_promotion_blockers(
        audit_event_count=3,
        same_product_selection_evidence=True,
        same_product_calibration_evidence=True,
        challenger_selected_on_selection=False,
    )

    assert blockers == ("challenger_not_selected_on_same_season_selection",)


def test_race_promotion_requires_four_independent_events_per_lock() -> None:
    blockers = _same_product_promotion_blockers(
        audit_event_count=3,
        same_product_selection_evidence=True,
        same_product_calibration_evidence=True,
        selection_event_count=1,
        calibration_event_count=1,
    )

    assert blockers == (
        "fewer_than_four_same_horizon_selection_events",
        "fewer_than_four_same_horizon_calibration_events",
    )


# Suggested commit name: test(f1-race): lock rolling OOF and calibration protocols
