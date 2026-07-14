from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_experiment as experiment_runner
import run_live_weekend_pipeline as weekend_runner
from run_experiment import build_parser
from packages.f1.orchestration.prediction import _pl_gumbel_listwise


def _sample_frame() -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.DataFrame(
        {
            "event_key": [202501] * 5 + [202502] * 5,
            "event_year": [2025] * 10,
            "event_round": [1] * 5 + [2] * 5,
            "event_name_norm": ["alpha"] * 5 + ["beta"] * 5,
            "driver_id": [f"d{i}" for i in range(10)],
        }
    )
    preds = pd.Series(
        [1.0, 2.0, 3.5, 5.0, 6.0, 1.2, 2.3, 3.1, 3.8, 6.1],
        index=frame.index,
        dtype=float,
    )
    return frame, preds


def test_pl_listwise_probabilities_are_valid_and_normalized_per_event() -> None:
    frame, preds = _sample_frame()
    out = _pl_gumbel_listwise(
        frame=frame,
        preds=preds,
        mode="qualifying",
        samples=2000,
        temperature=1.0,
        seed=42,
    )
    assert ((out["p_win"] >= 0.0) & (out["p_win"] <= 1.0)).all()
    assert ((out["p_top3"] >= 0.0) & (out["p_top3"] <= 1.0)).all()
    assert ((out["p_top10"] >= 0.0) & (out["p_top10"] <= 1.0)).all()
    assert (out["p_top3"] <= (out["p_top10"] + 1e-9)).all()

    for event_key, event_rows in out.groupby(frame["event_key"], sort=False):
        _ = event_key
        assert abs(float(event_rows["p_win"].sum()) - 1.0) < 0.02
        assert abs(float(event_rows["p_top3"].sum()) - 3.0) < 0.02
        assert abs(float(event_rows["p_top10"].sum()) - float(len(event_rows))) < 0.02


def test_pl_listwise_is_deterministic_with_fixed_seed() -> None:
    frame, preds = _sample_frame()
    out_a = _pl_gumbel_listwise(
        frame=frame,
        preds=preds,
        mode="qualifying",
        samples=2000,
        temperature=1.0,
        seed=123,
    )
    out_b = _pl_gumbel_listwise(
        frame=frame,
        preds=preds,
        mode="qualifying",
        samples=2000,
        temperature=1.0,
        seed=123,
    )
    cols = ["p_win", "p_top3", "p_top10", "exp_pos", "pos_p10", "pos_p50", "pos_p90"]
    assert np.allclose(out_a[cols].to_numpy(dtype=float), out_b[cols].to_numpy(dtype=float))


def test_cli_help_exposes_horizon_a_flags() -> None:
    stdout = build_parser("prediction").format_help()
    assert "--f1_model" in stdout
    assert "--f1_listwise" in stdout
    assert "--f1_pl_samples" in stdout
    assert "--f1_pl_temperature" in stdout
    assert "--f1_listwise_seed" in stdout
    assert "--shadow_eval" in stdout


def test_weekend_phase_forwards_explicit_model_listwise_and_horizon_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = {
        "name": "test-weekend-controls",
        "workflow": "weekend_phase",
        "defaults": {
            "phase": "pre-qualifying",
            "source": "local",
            "year": 2026,
            "round_number": 1,
            "weekends_dir": "data/f1/raw/weekends",
        },
        "training": {"train_seasons": [2026]},
        "experiments": {"compare_families": ["baseline"]},
    }
    args = build_parser("profile").parse_args(
        [
            "--profile",
            "unused.yaml",
            "--output-dir",
            str(tmp_path),
            "--f1_model",
            "baseline",
            "--f1_listwise",
            "off",
            "--f1_pl_samples",
            "321",
            "--f1_pl_temperature",
            "0.7",
            "--f1_listwise_seed",
            "17",
            "--qualifying-information-horizon",
            "pre_qualifying",
            "--race-information-horizon",
            "post_fp_pre_qualifying",
        ]
    )
    captured: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = list(cmd)
        output_path = Path(cmd[cmd.index("--output-path") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"rows": [], "notes": []}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(experiment_runner.subprocess, "run", fake_run)

    payload = experiment_runner._run_weekend_phase(profile, args)
    nested_args = weekend_runner.build_parser().parse_args(captured["cmd"][2:])

    assert nested_args.f1_model == "baseline"
    assert nested_args.f1_listwise == "off"
    assert nested_args.f1_pl_samples == 321
    assert nested_args.f1_pl_temperature == pytest.approx(0.7)
    assert nested_args.f1_listwise_seed == 17
    assert nested_args.qualifying_information_horizon == "pre_qualifying"
    assert nested_args.race_information_horizon == "post_fp_pre_qualifying"
    assert payload["config"]["f1_listwise"] == "off"


def test_prequal_weekend_builds_prediction_configs_with_explicit_off_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = []

    def capture_config(config):
        captured.append(config)
        return {"config": {"mode": config.mode}}

    monkeypatch.setattr(weekend_runner, "_prediction_payload", capture_config)

    weekend_runner._run_pre_qualifying(
        output_dir=tmp_path,
        source="local",
        year=2026,
        round_number=1,
        train_seasons=[2026],
        include_standings=False,
        cache_dir=None,
        weekends_dir="data/f1/raw/weekends",
        meeting_name=None,
        country_name=None,
        enable_dl_candidates=False,
        compare_families=["baseline"],
        dl_device="cpu",
        dl_arch="mlp_tabular_v1",
        dl_hyperparams={},
        dl_seed=42,
        disable_runsim_features=True,
        disable_circuit_features=True,
        weather_config={},
        prediction_as_of=None,
        f1_model="baseline",
        f1_listwise="off",
        f1_pl_samples=321,
        f1_pl_temperature=0.7,
        f1_listwise_seed=17,
        qualifying_information_horizon="pre_qualifying",
        race_information_horizon="post_fp_pre_qualifying",
    )

    assert [config.mode for config in captured] == ["qualifying", "race"]
    assert all(config.f1_model == "baseline" for config in captured)
    assert all(config.f1_listwise == "off" for config in captured)
    assert all(config.f1_pl_samples == 321 for config in captured)
    assert all(config.f1_pl_temperature == pytest.approx(0.7) for config in captured)
    assert all(config.f1_listwise_seed == 17 for config in captured)
    assert captured[0].qualifying_information_horizon == "pre_qualifying"
    assert captured[1].race_information_horizon == "post_fp_pre_qualifying"


def test_prequal_phase_rejects_future_information_horizons() -> None:
    with pytest.raises(ValueError, match="later than the 'pre-qualifying' phase boundary"):
        weekend_runner._validate_phase_information_horizons(
            phase="pre-qualifying",
            year=2026,
            qualifying_information_horizon="post_qualifying",
            race_information_horizon="post_fp_pre_qualifying",
        )

    with pytest.raises(ValueError, match="later than the 'pre-qualifying' phase boundary"):
        weekend_runner._validate_phase_information_horizons(
            phase="pre-qualifying",
            year=2026,
            qualifying_information_horizon="pre_qualifying",
            race_information_horizon="post_grid_pre_race",
        )


def test_phase_horizon_validation_allows_earlier_ablation_cutoffs() -> None:
    weekend_runner._validate_phase_information_horizons(
        phase="post-qualifying",
        year=2026,
        qualifying_information_horizon="pre_qualifying",
        race_information_horizon="post_fp_pre_qualifying",
    )


def test_phase_horizon_validation_respects_sprint_session_order_by_era() -> None:
    weekend_runner._validate_phase_information_horizons(
        phase="pre-qualifying",
        year=2026,
        qualifying_information_horizon="post_sprint",
        race_information_horizon="post_fp_pre_qualifying",
    )

    with pytest.raises(ValueError, match="sprint_2023"):
        weekend_runner._validate_phase_information_horizons(
            phase="pre-qualifying",
            year=2023,
            qualifying_information_horizon="post_sprint",
            race_information_horizon="post_fp_pre_qualifying",
        )
