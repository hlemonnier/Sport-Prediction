from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rqp.config import PredictionConfig
from rqp.live_runner import (
    _finalize_output_mapping,
    _mc_position_distribution,
    _strategy_template_probabilities,
    _write_trace,
    run_live_race_prediction,
)
from rqp.live_sources import LiveSourceResult, _build_race_time_seconds, load_live_observations
from rqp.live_state_space import (
    BaselineModel,
    FilterConfig,
    apply_track_gating,
    build_event_lap_baseline,
    compound_deg_prior,
    initialize_filter_state,
    parse_track_status,
    predict_state,
    reset_filter_state,
    update_state,
)


def _base_config(**overrides: object) -> PredictionConfig:
    base = {
        "source": "local",
        "mode": "race",
        "year": 2026,
        "round_number": 1,
        "train_seasons": [2022, 2023, 2024],
        "include_standings": True,
        "cache_dir": None,
        "meeting_name": None,
        "country_name": None,
        "weekends_dir": None,
        "enable_dl_candidates": False,
        "compare_families": ["ml"],
        "dl_device": "cpu",
        "dl_arch": "mlp_tabular_v1",
        "dl_hyperparams": {},
        "dl_seed": 42,
        "disable_runsim_features": False,
        "f1_model": "auto",
        "f1_listwise": "off",
        "f1_pl_samples": 2000,
        "f1_pl_temperature": 1.0,
        "f1_listwise_seed": 42,
        "shadow_eval": True,
        "f1_mode": "live",
        "f1_live_source": "local",
        "f1_live_model": "ssm_v1",
        "f1_live_horizon_laps": 10,
        "f1_live_seed": 42,
        "f1_live_cache_dir": None,
        "f1_live_replay_path": None,
    }
    base.update(overrides)
    return PredictionConfig(**base)


def _sample_live_observations(include_race_time: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for driver_idx, driver_id in enumerate(["1", "2"], start=1):
        cumulative = 0.0
        for lap in range(1, 4):
            lap_time = 90.0 + (0.7 * lap) + (0.4 * driver_idx)
            cumulative += lap_time
            rows.append(
                {
                    "event_key": 202601,
                    "session": "Race",
                    "driver_id": driver_id,
                    "driver_name": f"D{driver_id}",
                    "lap_number": lap,
                    "stint_id": 1,
                    "compound": "MEDIUM",
                    "tyre_age": lap - 1,
                    "is_box_lap": False,
                    "is_accurate": True,
                    "track_status": "1",
                    "lap_time_seconds": lap_time,
                    "timestamp": float(lap),
                    "race_time_seconds": cumulative if include_race_time else float("nan"),
                    "gap_to_leader_seconds": float(driver_idx - 1),
                    "source": "local",
                }
            )
    return pd.DataFrame(rows)


def test_track_status_parser_handles_multicodes_and_nan() -> None:
    flags = parse_track_status("267")
    assert flags.is_yellow is True
    assert flags.is_sc_vsc is True
    assert flags.is_red is False
    assert flags.is_greenish is False

    nan_flags = parse_track_status(np.nan)
    assert nan_flags.codes == set()
    assert nan_flags.is_greenish is False

    none_flags = parse_track_status(None)
    assert none_flags.codes == set()


def test_gating_rules_match_locked_spec() -> None:
    base_r = 0.2

    sc_gate = apply_track_gating(base_r, parse_track_status("4"))
    assert sc_gate.skip_update is True
    assert sc_gate.mode == "skip_sc_vsc_red"

    red_gate = apply_track_gating(base_r, parse_track_status("5"))
    assert red_gate.skip_update is True

    yellow_gate = apply_track_gating(base_r, parse_track_status("2"))
    assert yellow_gate.skip_update is False
    assert abs(yellow_gate.r_effective - (base_r * 9.0)) < 1e-9

    green_gate = apply_track_gating(base_r, parse_track_status("1"))
    assert green_gate.skip_update is False
    assert abs(green_gate.r_effective - base_r) < 1e-9

    ambiguous_gate = apply_track_gating(base_r, parse_track_status(""))
    assert ambiguous_gate.skip_update is False
    assert abs(ambiguous_gate.r_effective - (base_r * 4.0)) < 1e-9


def test_baseline_uses_winsorization_min_obs_and_interpolation() -> None:
    rows: list[dict[str, object]] = []
    for lap, base in [(1, 90.0), (2, 91.0), (3, 92.0)]:
        for idx in range(8):
            rows.append(
                {
                    "lap_number": lap,
                    "lap_time_seconds": base + (0.02 * idx),
                    "is_box_lap": False,
                    "is_accurate": True,
                    "track_status": "1",
                }
            )

    frame = pd.DataFrame(rows)
    # Make lap 2 invalid for min_clean_obs_per_lap, forcing interpolation.
    frame.loc[(frame["lap_number"] == 2) & (frame.index >= 13), "is_accurate"] = False
    # Add an extreme outlier for winsorization robustness.
    frame.loc[(frame["lap_number"] == 1) & (frame.index == 0), "lap_time_seconds"] = 130.0

    baseline = build_event_lap_baseline(frame, min_clean_obs_per_lap=8)

    b1 = baseline.value_at(1)
    b2 = baseline.value_at(2)
    b3 = baseline.value_at(3)
    assert np.isfinite(b1)
    assert np.isfinite(b2)
    assert np.isfinite(b3)
    assert abs(b2 - ((b1 + b3) / 2.0)) < 1e-9
    assert np.isfinite(baseline.value_at(10))


def test_pit_reset_reanchors_state_and_deg_prior() -> None:
    cfg = FilterConfig()
    state = initialize_filter_state("SOFT", cfg)

    for observation in [1.2, 1.4]:
        mean_pred, cov_pred = predict_state(state, cfg)
        mean_post, cov_post, _ = update_state(
            mean_pred=mean_pred,
            cov_pred=cov_pred,
            observation=observation,
            r_effective=cfg.r_obs,
            cfg=cfg,
        )
        state.mean = mean_post
        state.cov = cov_post
        state.last_stint_id = 1
        state.tyre_age += 1
        state.assimilated_laps += 1

    pre_reset_pace = float(state.mean[0])
    reset = reset_filter_state(state, compound="MEDIUM", cfg=cfg)

    assert float(reset.mean[0]) < (pre_reset_pace - 0.5)
    assert abs(float(reset.mean[1]) - compound_deg_prior("MEDIUM")) < 1e-6
    assert reset.tyre_age == 0


def test_strategy_template_probabilities_shift_with_urgency() -> None:
    early = _strategy_template_probabilities(
        compound="HARD",
        tyre_age=3,
        deg_rate=0.01,
        horizon=12,
    )
    urgent = _strategy_template_probabilities(
        compound="SOFT",
        tyre_age=18,
        deg_rate=0.06,
        horizon=12,
    )
    assert abs(sum(early.values()) - 1.0) < 1e-9
    assert abs(sum(urgent.values()) - 1.0) < 1e-9
    assert float(early["hold_track_position"]) > float(urgent["hold_track_position"])
    assert float(urgent["two_stop_balanced"] + urgent["two_stop_aggressive"]) > float(
        early["two_stop_balanced"] + early["two_stop_aggressive"]
    )


def test_horizon_distribution_probabilities_sum_to_one() -> None:
    cfg = FilterConfig()
    baseline = BaselineModel(by_lap={1: 90.0, 2: 90.2}, intercept=90.0, slope=0.05)
    snapshot = pd.DataFrame(
        {
            "driver_id": ["1", "2", "3", "4"],
            "lap_last": [2, 2, 2, 2],
            "race_time_seconds": [180.0, 180.8, 181.6, 182.4],
            "next_lap_mean": [90.3, 90.6, 90.9, 91.2],
        }
    )
    states = {driver_id: initialize_filter_state("MEDIUM", cfg) for driver_id in snapshot["driver_id"]}

    out, summary = _mc_position_distribution(
        snapshot=snapshot,
        states=states,
        baseline=baseline,
        cfg=cfg,
        horizon_laps=10,
        seed=123,
    )
    assert bool(summary["position_dist_enabled"]) is True
    assert abs(float(summary["sum_p_win"]) - 1.0) < 0.02
    assert (out["p_win_H"] >= 0.0).all()
    assert (out["p_win_H"] <= 1.0).all()


def test_horizon_distribution_emits_rollout_strategy_summary() -> None:
    cfg = FilterConfig()
    baseline = BaselineModel(by_lap={1: 90.0}, intercept=90.0, slope=0.0)
    snapshot = pd.DataFrame(
        {
            "driver_id": ["1", "2", "3"],
            "lap_last": [15, 15, 15],
            "race_time_seconds": [1350.0, 1352.0, 1354.0],
            "next_lap_mean": [90.0, 90.2, 90.4],
            "tyre_age": [18, 19, 20],
            "compound": ["SOFT", "SOFT", "SOFT"],
            "track_status": ["1", "1", "1"],
        }
    )
    states = {driver_id: initialize_filter_state("SOFT", cfg) for driver_id in snapshot["driver_id"]}
    for state in states.values():
        state.mean[1] = 0.06

    _, summary = _mc_position_distribution(
        snapshot=snapshot,
        states=states,
        baseline=baseline,
        cfg=cfg,
        horizon_laps=12,
        seed=123,
    )

    assert bool(summary["rollout_strategy_enabled"]) is True
    assert summary["rollout_regime_initial"] == "green"
    assert 0.0 <= float(summary["rollout_sc_vsc_share"]) <= 1.0
    assert 0.0 <= float(summary["rollout_yellow_share"]) <= 1.0
    assert int(summary["rollout_pit_events_total"]) > 0
    assert float(summary["rollout_pit_events_mean"]) > 0.0
    strategy_mix = summary.get("rollout_strategy_mix")
    assert isinstance(strategy_mix, dict)
    assert abs(sum(float(value) for value in strategy_mix.values()) - 1.0) < 1e-9


def test_position_ranking_uses_lap_count_before_total_time() -> None:
    cfg = FilterConfig()
    baseline = BaselineModel(by_lap={1: 90.0}, intercept=90.0, slope=0.0)
    snapshot = pd.DataFrame(
        {
            "driver_id": ["lead", "lapped"],
            "lap_last": [50, 49],
            "race_time_seconds": [5000.0, 4900.0],
            "next_lap_mean": [90.0, 90.0],
        }
    )
    states = {driver_id: initialize_filter_state("MEDIUM", cfg) for driver_id in snapshot["driver_id"]}

    out, summary = _mc_position_distribution(
        snapshot=snapshot,
        states=states,
        baseline=baseline,
        cfg=cfg,
        horizon_laps=5,
        seed=321,
    )

    assert bool(summary["position_dist_enabled"]) is True
    win_map = out.set_index("driver_id")["p_win_H"].to_dict()
    assert float(win_map["lead"]) == 1.0
    assert float(win_map["lapped"]) == 0.0


def test_position_distribution_penalizes_partial_missing_race_times() -> None:
    cfg = FilterConfig()
    baseline = BaselineModel(by_lap={1: 90.0}, intercept=90.0, slope=0.05)
    snapshot = pd.DataFrame(
        {
            "driver_id": ["1", "2", "3"],
            "lap_last": [8, 8, 8],
            "race_time_seconds": [720.0, 721.0, float("nan")],
            "next_lap_mean": [90.0, 90.2, 90.1],
        }
    )
    states = {driver_id: initialize_filter_state("MEDIUM", cfg) for driver_id in snapshot["driver_id"]}

    out, summary = _mc_position_distribution(
        snapshot=snapshot,
        states=states,
        baseline=baseline,
        cfg=cfg,
        horizon_laps=6,
        seed=123,
    )

    assert bool(summary["position_dist_enabled"]) is True
    assert int(summary["invalid_race_time_count"]) == 1
    assert float(out.loc[out["driver_id"] == "3", "p_win_H"].iloc[0]) == 0.0


def test_build_race_time_seconds_preserves_unknowns_without_zero_fill() -> None:
    work = pd.DataFrame(
        {
            "driver_id": ["1", "1", "2", "2", "2"],
            "lap_number": [1, 2, 1, 2, 3],
            "timestamp": [1.0, 2.0, 1.0, 2.0, 3.0],
            "lap_time_seconds": [float("nan"), float("nan"), 90.0, float("nan"), 91.0],
        }
    )

    out = _build_race_time_seconds(work)
    ordered = work.assign(race_time_seconds=out).sort_values(
        ["driver_id", "lap_number", "timestamp"], kind="mergesort"
    )

    driver_1 = ordered[ordered["driver_id"] == "1"]["race_time_seconds"].tolist()
    driver_2 = ordered[ordered["driver_id"] == "2"]["race_time_seconds"].tolist()

    assert all(pd.isna(value) for value in driver_1)
    assert driver_2 == [90.0, 90.0, 181.0]


def test_mc_cpu_guard_reduces_samples_and_logs_reason() -> None:
    cfg = FilterConfig()
    baseline = BaselineModel(by_lap={1: 90.0}, intercept=90.0, slope=0.05)
    driver_ids = [str(i) for i in range(300)]
    snapshot = pd.DataFrame(
        {
            "driver_id": driver_ids,
            "lap_last": [1] * len(driver_ids),
            "race_time_seconds": [180.0 + (0.3 * i) for i in range(len(driver_ids))],
            "next_lap_mean": [90.0 + (0.01 * i) for i in range(len(driver_ids))],
        }
    )
    states = {driver_id: initialize_filter_state("MEDIUM", cfg) for driver_id in driver_ids}

    _, summary = _mc_position_distribution(
        snapshot=snapshot,
        states=states,
        baseline=baseline,
        cfg=cfg,
        horizon_laps=4,
        seed=123,
    )

    assert summary["mc_samples_requested"] == 1000
    assert int(summary["mc_samples_effective"]) < 1000
    assert isinstance(summary["mc_samples_reduction_reason"], str)
    assert "reduced" in str(summary["mc_samples_reduction_reason"])


def test_fallback_mapping_when_position_distribution_disabled() -> None:
    cfg = FilterConfig()
    baseline = BaselineModel(by_lap={1: 90.0}, intercept=90.0, slope=0.0)
    snapshot = pd.DataFrame(
        {
            "driver_id": ["1", "2"],
            "lap_last": [1, 1],
            "race_time_seconds": [float("nan"), float("nan")],
            "next_lap_mean": [91.2, 90.9],
        }
    )
    states = {"1": initialize_filter_state("MEDIUM", cfg), "2": initialize_filter_state("MEDIUM", cfg)}

    no_dist, dist_summary = _mc_position_distribution(
        snapshot=snapshot,
        states=states,
        baseline=baseline,
        cfg=cfg,
        horizon_laps=10,
        seed=123,
    )
    mapped = _finalize_output_mapping(no_dist, dist_summary, horizon_laps=10)

    assert bool(dist_summary["position_dist_enabled"]) is False
    assert dist_summary["position_dist_disabled_reason"] == "missing_race_time_seconds"
    assert mapped["proba_top10"].isna().all()
    assert mapped["proba_top3"].isna().all()
    assert list(mapped["driver_id"]) == ["2", "1"]


def test_live_summary_includes_disable_reason_when_no_position_dist(monkeypatch) -> None:
    frame = _sample_live_observations(include_race_time=False)

    def _fake_loader(config: PredictionConfig) -> LiveSourceResult:
        _ = config
        return LiveSourceResult(frame=frame.copy(), source_used="local", notes=[])

    monkeypatch.setattr("rqp.live_runner.load_live_observations", _fake_loader)
    monkeypatch.setattr(
        "rqp.live_runner._write_trace",
        lambda trace, config, event_key: {
            "trace_path": "/tmp/fake_trace.parquet",
            "trace_path_jsonl": "/tmp/fake_trace.jsonl",
            "trace_format_effective": "parquet",
        },
    )

    result = run_live_race_prediction(_base_config())
    assert bool(result.summary["position_dist_enabled"]) is False
    assert result.summary["position_dist_disabled_reason"] == "missing_race_time_seconds"
    assert result.snapshot["proba_top10"].isna().all()
    assert result.snapshot["proba_top3"].isna().all()


def test_live_replay_predictions_are_truncation_invariant_through_lap(monkeypatch) -> None:
    full_frame = _sample_live_observations(include_race_time=True)
    truncated_frame = full_frame[pd.to_numeric(full_frame["lap_number"], errors="coerce") <= 2].copy()
    frames = [full_frame, truncated_frame]

    def _fake_loader(config: PredictionConfig) -> LiveSourceResult:
        _ = config
        return LiveSourceResult(frame=frames.pop(0).copy(), source_used="local", notes=[])

    monkeypatch.setattr("rqp.live_runner.load_live_observations", _fake_loader)
    monkeypatch.setattr(
        "rqp.live_runner._write_trace",
        lambda trace, config, event_key: {
            "trace_path": "/tmp/fake_trace.parquet",
            "trace_path_jsonl": "/tmp/fake_trace.jsonl",
            "trace_format_effective": "parquet",
        },
    )

    full = run_live_race_prediction(_base_config())
    truncated = run_live_race_prediction(_base_config())

    full_prefix = full.trace[pd.to_numeric(full.trace["lap_number"], errors="coerce") <= 2]
    full_prefix = full_prefix.sort_values(["driver_id", "lap_number"]).reset_index(drop=True)
    truncated_trace = truncated.trace.sort_values(["driver_id", "lap_number"]).reset_index(drop=True)
    cols = ["driver_id", "lap_number", "baseline_lap", "one_step_pred_mean", "next_lap_mean"]

    pd.testing.assert_frame_equal(
        full_prefix[cols],
        truncated_trace[cols],
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_trace_writer_falls_back_to_jsonl_when_parquet_unavailable(monkeypatch, tmp_path: Path) -> None:
    trace = pd.DataFrame(
        [
            {"driver_id": "1", "lap_number": 1, "next_lap_mean": 90.1},
            {"driver_id": "2", "lap_number": 1, "next_lap_mean": 90.4},
        ]
    )
    config = _base_config()

    monkeypatch.setattr("rqp.live_runner._resolve_artifacts_dir", lambda cfg: tmp_path)

    def _raise_no_parquet(self, path, index=False):  # noqa: ARG001
        raise ImportError("pyarrow missing")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _raise_no_parquet)

    meta = _write_trace(trace, config=config, event_key=202601)
    assert meta["trace_format_effective"] == "jsonl"
    assert meta["trace_path"].endswith(".jsonl")
    assert Path(meta["trace_path"]).exists()
    assert meta["trace_path_jsonl"] is not None


def test_openf1_live_source_is_explicitly_rejected() -> None:
    result = load_live_observations(_base_config(f1_live_source="openf1"))
    assert result.frame.empty
    assert any("not supported" in note.lower() for note in result.notes)


def test_cli_help_exposes_horizon_b_flags_and_live_race_phase() -> None:
    project_python_dir = Path(__file__).resolve().parents[1]

    prediction_help = subprocess.run(
        [sys.executable, str(project_python_dir / "run_prediction.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "--f1_mode" in prediction_help
    assert "--f1_live_source" in prediction_help
    assert "--f1_live_model" in prediction_help
    assert "--f1_live_horizon_laps" in prediction_help
    assert "--f1_live_seed" in prediction_help
    assert "--f1_live_cache_dir" in prediction_help
    assert "--f1_live_replay_path" in prediction_help

    weekend_help = subprocess.run(
        [sys.executable, str(project_python_dir / "run_live_weekend_pipeline.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "live-race" in weekend_help
    assert "--f1-mode" in weekend_help
    assert "--f1-live-source" in weekend_help
