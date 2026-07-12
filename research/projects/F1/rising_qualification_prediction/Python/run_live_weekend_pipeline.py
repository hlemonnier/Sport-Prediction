#!/usr/bin/env python3
"""Operational weekend pipeline for in-season F1 predictions."""

from __future__ import annotations
import repo_bootstrap  # noqa: F401

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from packages.f1 import PredictionConfig, run_prediction
from packages.f1.orchestration.backtest import evaluate_prediction_rows
from packages.f1.data.providers import BaseProvider, FastF1Provider, LocalWeekendProvider, OpenF1Provider
from packages.f1.orchestration.runtime import parse_compare_families, parse_json_object, parse_train_seasons


def default_output_dir() -> str:
    project_root = Path(__file__).resolve().parents[5]
    return str(project_root / "artifacts" / "predictions" / "f1" / "live")


def _build_provider(
    source: str,
    cache_dir: Optional[str],
    weekends_dir: Optional[str],
    round_number: int,
    meeting_name: Optional[str],
    country_name: Optional[str],
) -> BaseProvider:
    if source == "fastf1":
        return FastF1Provider(cache_dir=cache_dir)
    if source == "openf1":
        return OpenF1Provider(
            cache_dir=cache_dir,
            target_round=round_number,
            meeting_name=meeting_name,
            country_name=country_name,
        )
    return LocalWeekendProvider(weekends_dir=weekends_dir)


def _prediction_payload(config: PredictionConfig) -> dict[str, Any]:
    result = run_prediction(config)
    rows: list[dict[str, Any]]
    if result.table.empty:
        rows = []
    else:
        rows = json.loads(result.table.to_json(orient="records"))
    payload = {
        "version": result.version,
        "sport": "F1",
        "project": "Rising Qualification Prediction",
        "config": asdict(config),
        "rows": rows,
        "notes": result.notes,
        "model_name": result.model_name,
        "model_family": result.model_family,
        "device_used": result.device_used,
        "dl_available": result.dl_available,
        "candidate_leaderboard": result.candidate_leaderboard,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if isinstance(result.extras, dict) and result.extras:
        payload.update(result.extras)
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _payload_prediction_rows(payload: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    all_rows = payload.get("all_prediction_rows")
    if isinstance(all_rows, list):
        return [row for row in all_rows if isinstance(row, dict)]
    rows = payload.get("rows")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _round_dir(base_output_dir: str, year: int, round_number: int) -> Path:
    return Path(base_output_dir) / str(year) / f"round_{int(round_number):02d}"


def _run_qualifying_prediction(
    *,
    source: str,
    year: int,
    round_number: int,
    train_seasons: list[int],
    cache_dir: Optional[str],
    weekends_dir: Optional[str],
    meeting_name: Optional[str],
    country_name: Optional[str],
    enable_dl_candidates: bool,
    compare_families: list[str],
    dl_device: str,
    dl_arch: str,
    dl_hyperparams: dict[str, Any],
    dl_seed: int,
    disable_runsim_features: bool,
    disable_circuit_features: bool,
    weather_config: dict[str, Any],
    qualifying_information_horizon: str,
    prediction_as_of: Optional[str],
) -> dict[str, Any]:
    config = PredictionConfig(
        source=source,
        mode="qualifying",
        year=year,
        round_number=round_number,
        train_seasons=train_seasons,
        include_standings=False,
        cache_dir=cache_dir,
        meeting_name=meeting_name,
        country_name=country_name,
        weekends_dir=weekends_dir,
        enable_dl_candidates=enable_dl_candidates,
        compare_families=compare_families,
        dl_device=dl_device,
        dl_arch=dl_arch,
        dl_hyperparams=dl_hyperparams,
        dl_seed=dl_seed,
        disable_runsim_features=disable_runsim_features,
        disable_circuit_features=disable_circuit_features,
        qualifying_information_horizon=qualifying_information_horizon,
        prediction_as_of=prediction_as_of,
        **weather_config,
    )
    return _prediction_payload(config)


def _run_race_prediction(
    *,
    source: str,
    year: int,
    round_number: int,
    train_seasons: list[int],
    include_standings: bool,
    cache_dir: Optional[str],
    weekends_dir: Optional[str],
    meeting_name: Optional[str],
    country_name: Optional[str],
    enable_dl_candidates: bool,
    compare_families: list[str],
    dl_device: str,
    dl_arch: str,
    dl_hyperparams: dict[str, Any],
    dl_seed: int,
    disable_runsim_features: bool,
    disable_circuit_features: bool,
    weather_config: dict[str, Any],
    qualifying_information_horizon: str,
    race_information_horizon: str,
    prediction_as_of: Optional[str],
    f1_mode: str = "offline",
    f1_live_source: str = "auto",
    f1_live_model: str = "ssm_v1",
    f1_live_horizon_laps: int = 10,
    f1_live_seed: int = 42,
    f1_live_cache_dir: Optional[str] = None,
    f1_live_replay_path: Optional[str] = None,
    f1_live_calibration_path: Optional[str] = None,
    f1_live_replay_cutoff_lap: Optional[int] = None,
    f1_live_replay_cutoff_time_seconds: Optional[float] = None,
    f1_live_next_lap_ssm_weight: float = 0.0,
) -> dict[str, Any]:
    config = PredictionConfig(
        source=source,
        mode="race",
        year=year,
        round_number=round_number,
        train_seasons=train_seasons,
        include_standings=include_standings,
        cache_dir=cache_dir,
        meeting_name=meeting_name,
        country_name=country_name,
        weekends_dir=weekends_dir,
        enable_dl_candidates=enable_dl_candidates,
        compare_families=compare_families,
        dl_device=dl_device,
        dl_arch=dl_arch,
        dl_hyperparams=dl_hyperparams,
        dl_seed=dl_seed,
        disable_runsim_features=disable_runsim_features,
        disable_circuit_features=disable_circuit_features,
        qualifying_information_horizon=qualifying_information_horizon,
        race_information_horizon=race_information_horizon,
        prediction_as_of=prediction_as_of,
        **weather_config,
        f1_mode=f1_mode,
        f1_live_source=f1_live_source,
        f1_live_model=f1_live_model,
        f1_live_horizon_laps=f1_live_horizon_laps,
        f1_live_seed=f1_live_seed,
        f1_live_cache_dir=f1_live_cache_dir,
        f1_live_replay_path=f1_live_replay_path,
        f1_live_calibration_path=f1_live_calibration_path,
        f1_live_replay_cutoff_lap=f1_live_replay_cutoff_lap,
        f1_live_replay_cutoff_time_seconds=f1_live_replay_cutoff_time_seconds,
        f1_live_next_lap_ssm_weight=f1_live_next_lap_ssm_weight,
    )
    return _prediction_payload(config)


def _run_pre_qualifying(
    output_dir: Path,
    *,
    source: str,
    year: int,
    round_number: int,
    train_seasons: list[int],
    include_standings: bool,
    cache_dir: Optional[str],
    weekends_dir: Optional[str],
    meeting_name: Optional[str],
    country_name: Optional[str],
    enable_dl_candidates: bool,
    compare_families: list[str],
    dl_device: str,
    dl_arch: str,
    dl_hyperparams: dict[str, Any],
    dl_seed: int,
    disable_runsim_features: bool,
    disable_circuit_features: bool,
    weather_config: dict[str, Any],
    prediction_as_of: Optional[str],
) -> dict[str, str]:
    qualifying_payload = _run_qualifying_prediction(
        source=source,
        year=year,
        round_number=round_number,
        train_seasons=train_seasons,
        cache_dir=cache_dir,
        weekends_dir=weekends_dir,
        meeting_name=meeting_name,
        country_name=country_name,
        enable_dl_candidates=enable_dl_candidates,
        compare_families=compare_families,
        dl_device=dl_device,
        dl_arch=dl_arch,
        dl_hyperparams=dl_hyperparams,
        dl_seed=dl_seed,
        disable_runsim_features=disable_runsim_features,
        disable_circuit_features=disable_circuit_features,
        weather_config=weather_config,
        qualifying_information_horizon="pre_qualifying",
        prediction_as_of=prediction_as_of,
    )
    qualifying_path = output_dir / "prequal_qualifying_prediction.json"
    _write_json(qualifying_path, qualifying_payload)

    race_payload = _run_race_prediction(
        source=source,
        year=year,
        round_number=round_number,
        train_seasons=train_seasons,
        include_standings=include_standings,
        cache_dir=cache_dir,
        weekends_dir=weekends_dir,
        meeting_name=meeting_name,
        country_name=country_name,
        enable_dl_candidates=enable_dl_candidates,
        compare_families=compare_families,
        dl_device=dl_device,
        dl_arch=dl_arch,
        dl_hyperparams=dl_hyperparams,
        dl_seed=dl_seed,
        disable_runsim_features=disable_runsim_features,
        disable_circuit_features=disable_circuit_features,
        weather_config=weather_config,
        qualifying_information_horizon="pre_qualifying",
        race_information_horizon="post_fp_pre_qualifying",
        prediction_as_of=prediction_as_of,
        f1_mode="offline",
    )
    race_path = output_dir / "prequal_race_prediction.json"
    _write_json(race_path, race_payload)
    return {
        "qualifying_prediction": str(qualifying_path),
        "race_prediction": str(race_path),
    }


def _run_post_qualifying(
    output_dir: Path,
    provider: BaseProvider,
    *,
    source: str,
    year: int,
    round_number: int,
    train_seasons: list[int],
    include_standings: bool,
    cache_dir: Optional[str],
    weekends_dir: Optional[str],
    meeting_name: Optional[str],
    country_name: Optional[str],
    enable_dl_candidates: bool,
    compare_families: list[str],
    dl_device: str,
    dl_arch: str,
    dl_hyperparams: dict[str, Any],
    dl_seed: int,
    disable_runsim_features: bool,
    disable_circuit_features: bool,
    weather_config: dict[str, Any],
    prediction_as_of: Optional[str],
) -> dict[str, Any]:
    race_payload = _run_race_prediction(
        source=source,
        year=year,
        round_number=round_number,
        train_seasons=train_seasons,
        include_standings=include_standings,
        cache_dir=cache_dir,
        weekends_dir=weekends_dir,
        meeting_name=meeting_name,
        country_name=country_name,
        enable_dl_candidates=enable_dl_candidates,
        compare_families=compare_families,
        dl_device=dl_device,
        dl_arch=dl_arch,
        dl_hyperparams=dl_hyperparams,
        dl_seed=dl_seed,
        disable_runsim_features=disable_runsim_features,
        disable_circuit_features=disable_circuit_features,
        weather_config=weather_config,
        qualifying_information_horizon="post_qualifying",
        race_information_horizon="post_qualifying_pre_grid",
        prediction_as_of=prediction_as_of,
        f1_mode="offline",
    )
    race_path = output_dir / "postqual_race_prediction.json"
    _write_json(race_path, race_payload)

    prequal_qualifying = _load_json(output_dir / "prequal_qualifying_prediction.json")
    actual_qualifying = provider.get_qualifying_results(year, round_number)
    qual_eval = evaluate_prediction_rows(
        predicted_rows=_payload_prediction_rows(prequal_qualifying),
        actual_results=actual_qualifying,
        actual_position_col="position",
        include_podium_and_winner=True,
    )
    eval_payload = {
        "phase": "post-qualifying",
        "year": year,
        "round_number": round_number,
        "qualifying_eval_prequal_vs_actual": qual_eval,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    eval_path = output_dir / "postqual_evaluation.json"
    _write_json(eval_path, eval_payload)
    return {
        "race_prediction": str(race_path),
        "evaluation": str(eval_path),
    }


def _run_post_race(
    output_dir: Path,
    provider: BaseProvider,
    *,
    year: int,
    round_number: int,
) -> dict[str, Any]:
    prequal_qualifying = _load_json(output_dir / "prequal_qualifying_prediction.json")
    prequal_race = _load_json(output_dir / "prequal_race_prediction.json")
    postqual_race = _load_json(output_dir / "postqual_race_prediction.json")

    actual_qualifying = provider.get_qualifying_results(year, round_number)
    actual_race = provider.get_race_results(year, round_number)

    qualifying_eval = evaluate_prediction_rows(
        predicted_rows=_payload_prediction_rows(prequal_qualifying),
        actual_results=actual_qualifying,
        actual_position_col="position",
        include_podium_and_winner=True,
    )
    race_eval_prequal = evaluate_prediction_rows(
        predicted_rows=_payload_prediction_rows(prequal_race),
        actual_results=actual_race,
        actual_position_col="position",
        include_podium_and_winner=True,
    )
    race_eval_postqual = evaluate_prediction_rows(
        predicted_rows=_payload_prediction_rows(postqual_race),
        actual_results=actual_race,
        actual_position_col="position",
        include_podium_and_winner=True,
    )

    delta_mae = None
    delta_top10 = None
    if race_eval_prequal.get("available") and race_eval_postqual.get("available"):
        pre_mae = race_eval_prequal.get("field_mae")
        post_mae = race_eval_postqual.get("field_mae")
        pre_top10 = race_eval_prequal.get("top10_hit")
        post_top10 = race_eval_postqual.get("top10_hit")
        if pre_mae is not None and post_mae is not None:
            delta_mae = float(post_mae) - float(pre_mae)
        if pre_top10 is not None and post_top10 is not None:
            delta_top10 = float(post_top10) - float(pre_top10)

    payload = {
        "phase": "post-race",
        "year": year,
        "round_number": round_number,
        "qualifying_eval_prequal_vs_actual": qualifying_eval,
        "race_eval_prequal_vs_actual": race_eval_prequal,
        "race_eval_postqual_vs_actual": race_eval_postqual,
        "race_eval_delta_postqual_minus_prequal": {
            "field_mae": delta_mae,
            "top10_hit": delta_top10,
        },
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    output_path = output_dir / "postrace_evaluation.json"
    _write_json(output_path, payload)
    return {"evaluation": str(output_path)}


def _run_live_race(
    output_dir: Path,
    *,
    source: str,
    year: int,
    round_number: int,
    train_seasons: list[int],
    include_standings: bool,
    cache_dir: Optional[str],
    weekends_dir: Optional[str],
    meeting_name: Optional[str],
    country_name: Optional[str],
    enable_dl_candidates: bool,
    compare_families: list[str],
    dl_device: str,
    dl_arch: str,
    dl_hyperparams: dict[str, Any],
    dl_seed: int,
    disable_runsim_features: bool,
    disable_circuit_features: bool,
    weather_config: dict[str, Any],
    f1_live_source: str,
    f1_live_model: str,
    f1_live_horizon_laps: int,
    f1_live_seed: int,
    f1_live_cache_dir: Optional[str],
    f1_live_replay_path: Optional[str],
    f1_live_calibration_path: Optional[str],
    f1_live_replay_cutoff_lap: Optional[int],
    f1_live_replay_cutoff_time_seconds: Optional[float],
    f1_live_next_lap_ssm_weight: float,
    prediction_as_of: Optional[str],
) -> dict[str, Any]:
    race_payload = _run_race_prediction(
        source=source,
        year=year,
        round_number=round_number,
        train_seasons=train_seasons,
        include_standings=include_standings,
        cache_dir=cache_dir,
        weekends_dir=weekends_dir,
        meeting_name=meeting_name,
        country_name=country_name,
        enable_dl_candidates=enable_dl_candidates,
        compare_families=compare_families,
        dl_device=dl_device,
        dl_arch=dl_arch,
        dl_hyperparams=dl_hyperparams,
        dl_seed=dl_seed,
        disable_runsim_features=disable_runsim_features,
        disable_circuit_features=disable_circuit_features,
        weather_config=weather_config,
        qualifying_information_horizon="pre_race",
        race_information_horizon="post_grid_pre_race",
        prediction_as_of=prediction_as_of,
        f1_mode="live",
        f1_live_source=f1_live_source,
        f1_live_model=f1_live_model,
        f1_live_horizon_laps=f1_live_horizon_laps,
        f1_live_seed=f1_live_seed,
        f1_live_cache_dir=f1_live_cache_dir,
        f1_live_replay_path=f1_live_replay_path,
        f1_live_calibration_path=f1_live_calibration_path,
        f1_live_replay_cutoff_lap=f1_live_replay_cutoff_lap,
        f1_live_replay_cutoff_time_seconds=f1_live_replay_cutoff_time_seconds,
        f1_live_next_lap_ssm_weight=f1_live_next_lap_ssm_weight,
    )
    snapshot_path = output_dir / "live_race_snapshot.json"
    _write_json(snapshot_path, race_payload)

    live_summary = race_payload.get("live_summary")
    summary_path = output_dir / "live_race_summary.json"
    if isinstance(live_summary, dict):
        _write_json(summary_path, live_summary)
    else:
        _write_json(
            summary_path,
            {
                "available": False,
                "reason": "live_summary_missing",
                "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
        )

    return {
        "snapshot": str(snapshot_path),
        "summary": str(summary_path),
        "trace_path": race_payload.get("trace_path"),
        "trace_path_jsonl": race_payload.get("trace_path_jsonl"),
        "trace_format_effective": race_payload.get("trace_format_effective"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live weekend pipeline: pre-qualifying, post-qualifying, post-race, live-race.",
    )
    parser.add_argument(
        "--phase",
        choices=["pre-qualifying", "post-qualifying", "post-race", "live-race", "full"],
        required=True,
    )
    parser.add_argument("--source", choices=["fastf1", "openf1", "local"], required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", dest="round_number", type=int, required=True)
    parser.add_argument("--train-seasons", default="auto")
    parser.add_argument(
        "--train-policy",
        choices=["same_season", "same_season_walk_forward", "strict_transfer", "rolling", "frozen_preseason", "legacy_auto"],
        default="same_season_walk_forward",
        help=(
            "auto train-season policy. same_season_walk_forward is the default for the 2026 reset; "
            "cross-regime transfer policies must be requested explicitly and evaluated separately."
        ),
    )
    parser.add_argument("--include-standings", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--weekends-dir", default="data/f1/raw/weekends")
    parser.add_argument(
        "--prediction-as-of",
        default=None,
        help=(
            "Optional ISO-8601 point-in-time cutoff. Local snapshots must carry an available_at or "
            "first_published_at timestamp no later than this value; mutable retrospective providers fail closed."
        ),
    )
    parser.add_argument("--meeting-name", default=None)
    parser.add_argument("--country-name", default=None)
    parser.add_argument("--enable-dl-candidates", action="store_true")
    parser.add_argument("--compare-families", default="ml")
    parser.add_argument("--dl-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dl-arch", default="mlp_tabular_v1")
    parser.add_argument("--dl-hyperparams", default="{}")
    parser.add_argument("--dl-seed", type=int, default=42)
    parser.add_argument("--disable-runsim-features", action="store_true")
    parser.add_argument(
        "--disable-circuit-features",
        dest="disable_circuit_features",
        action="store_true",
        default=None,
        help="Disable predictive circuit/track priors. This remains the default unless explicitly enabled.",
    )
    parser.add_argument(
        "--enable-circuit-features",
        dest="disable_circuit_features",
        action="store_false",
        help="Enable circuit-card and historical track priors for research/live weekend runs.",
    )
    parser.add_argument("--weather", choices=["on", "off"], default="off")
    parser.add_argument("--weather-provider", default="open_meteo")
    parser.add_argument("--weather-latitude", type=float, default=None)
    parser.add_argument("--weather-longitude", type=float, default=None)
    parser.add_argument("--weather-timezone", default=None)
    parser.add_argument("--weather-start", default=None)
    parser.add_argument("--weather-end", default=None)
    parser.add_argument("--weather-cache-dir", default=None)
    parser.add_argument("--f1-mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--f1-live-source", choices=["auto", "local", "fastf1"], default="auto")
    parser.add_argument("--f1-live-model", choices=["ssm_v1"], default="ssm_v1")
    parser.add_argument("--f1-live-horizon-laps", type=int, default=10)
    parser.add_argument("--f1-live-seed", type=int, default=42)
    parser.add_argument("--f1-live-cache-dir", default=None)
    parser.add_argument("--f1-live-replay-path", default=None)
    parser.add_argument("--f1-live-calibration-path", default=None)
    parser.add_argument("--f1-live-replay-cutoff-lap", type=int, default=None)
    parser.add_argument(
        "--f1-live-replay-cutoff-time-seconds",
        type=float,
        default=None,
        help="Global session timestamp cutoff for causal replay; preferred over lap-only truncation.",
    )
    parser.add_argument(
        "--f1-live-next-lap-ssm-weight",
        type=float,
        default=0.0,
        help="Causal SSM weight in the next-lap point blend; 0.0 is the evidence-gated naive fallback.",
    )
    parser.add_argument("--output-dir", default=default_output_dir())
    parser.add_argument("--output-format", choices=["text", "json"], default="text")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    train_seasons = parse_train_seasons(args.train_seasons, args.year, args.train_policy)
    compare_families = parse_compare_families(args.compare_families)
    dl_hyperparams = parse_json_object(args.dl_hyperparams, "--dl-hyperparams")
    disable_circuit_features = (
        True if args.disable_circuit_features is None else bool(args.disable_circuit_features)
    )
    weather_config = {
        "weather_enabled": str(args.weather).strip().lower() == "on",
        "weather_provider": str(args.weather_provider),
        "weather_latitude": args.weather_latitude,
        "weather_longitude": args.weather_longitude,
        "weather_timezone": args.weather_timezone,
        "weather_start": args.weather_start,
        "weather_end": args.weather_end,
        "weather_cache_dir": args.weather_cache_dir,
    }
    output_dir = _round_dir(args.output_dir, args.year, args.round_number)
    output_dir.mkdir(parents=True, exist_ok=True)

    provider = _build_provider(
        source=args.source,
        cache_dir=args.cache_dir,
        weekends_dir=args.weekends_dir,
        round_number=args.round_number,
        meeting_name=args.meeting_name,
        country_name=args.country_name,
    )

    executed: list[str] = []
    artifacts: dict[str, Any] = {}

    if args.phase in {"pre-qualifying", "full"}:
        artifacts["pre_qualifying"] = _run_pre_qualifying(
            output_dir=output_dir,
            source=args.source,
            year=args.year,
            round_number=args.round_number,
            train_seasons=train_seasons,
            include_standings=args.include_standings,
            cache_dir=args.cache_dir,
            weekends_dir=args.weekends_dir,
            meeting_name=args.meeting_name,
            country_name=args.country_name,
            enable_dl_candidates=args.enable_dl_candidates,
            compare_families=compare_families,
            dl_device=args.dl_device,
            dl_arch=args.dl_arch,
            dl_hyperparams=dl_hyperparams,
            dl_seed=args.dl_seed,
            disable_runsim_features=args.disable_runsim_features,
            disable_circuit_features=disable_circuit_features,
            weather_config=weather_config,
            prediction_as_of=args.prediction_as_of,
        )
        executed.append("pre-qualifying")

    if args.phase in {"post-qualifying", "full"}:
        artifacts["post_qualifying"] = _run_post_qualifying(
            output_dir=output_dir,
            provider=provider,
            source=args.source,
            year=args.year,
            round_number=args.round_number,
            train_seasons=train_seasons,
            include_standings=args.include_standings,
            cache_dir=args.cache_dir,
            weekends_dir=args.weekends_dir,
            meeting_name=args.meeting_name,
            country_name=args.country_name,
            enable_dl_candidates=args.enable_dl_candidates,
            compare_families=compare_families,
            dl_device=args.dl_device,
            dl_arch=args.dl_arch,
            dl_hyperparams=dl_hyperparams,
            dl_seed=args.dl_seed,
            disable_runsim_features=args.disable_runsim_features,
            disable_circuit_features=disable_circuit_features,
            weather_config=weather_config,
            prediction_as_of=args.prediction_as_of,
        )
        executed.append("post-qualifying")

    if args.phase in {"post-race", "full"}:
        artifacts["post_race"] = _run_post_race(
            output_dir=output_dir,
            provider=provider,
            year=args.year,
            round_number=args.round_number,
        )
        executed.append("post-race")

    if args.phase == "live-race":
        artifacts["live_race"] = _run_live_race(
            output_dir=output_dir,
            source=args.source,
            year=args.year,
            round_number=args.round_number,
            train_seasons=train_seasons,
            include_standings=args.include_standings,
            cache_dir=args.cache_dir,
            weekends_dir=args.weekends_dir,
            meeting_name=args.meeting_name,
            country_name=args.country_name,
            enable_dl_candidates=args.enable_dl_candidates,
            compare_families=compare_families,
            dl_device=args.dl_device,
            dl_arch=args.dl_arch,
            dl_hyperparams=dl_hyperparams,
            dl_seed=args.dl_seed,
            disable_runsim_features=args.disable_runsim_features,
            disable_circuit_features=disable_circuit_features,
            weather_config=weather_config,
            f1_live_source=args.f1_live_source,
            f1_live_model=args.f1_live_model,
            f1_live_horizon_laps=args.f1_live_horizon_laps,
            f1_live_seed=args.f1_live_seed,
            f1_live_cache_dir=args.f1_live_cache_dir,
            f1_live_replay_path=args.f1_live_replay_path,
            f1_live_calibration_path=args.f1_live_calibration_path,
            f1_live_replay_cutoff_lap=args.f1_live_replay_cutoff_lap,
            f1_live_replay_cutoff_time_seconds=args.f1_live_replay_cutoff_time_seconds,
            f1_live_next_lap_ssm_weight=args.f1_live_next_lap_ssm_weight,
            prediction_as_of=args.prediction_as_of,
        )
        executed.append("live-race")

    live_snapshot_payload: Optional[dict[str, Any]] = None
    if args.phase == "live-race":
        live_artifacts = artifacts.get("live_race")
        if isinstance(live_artifacts, dict):
            snapshot_path = live_artifacts.get("snapshot")
            if isinstance(snapshot_path, str) and snapshot_path:
                live_snapshot_payload = _load_json(Path(snapshot_path))

    f1_mode_effective = "live" if args.phase == "live-race" else str(args.f1_mode)
    phase_information_contracts = {
        "pre-qualifying": {
            "session_cutoff": "pre_qualifying",
            "race_horizon": "post_fp_pre_qualifying",
        },
        "post-qualifying": {
            "session_cutoff": "post_qualifying",
            "race_horizon": "post_qualifying_pre_grid",
        },
        "post-race": {
            "prediction_generated": False,
            "outcomes_role": "evaluation_only",
        },
        "live-race": {
            "session_cutoff": "pre_race",
            "race_horizon": "post_grid_pre_race",
        },
    }

    payload = {
        "sport": "F1",
        "project": "Rising Qualification Prediction",
        "source": args.source,
        "year": args.year,
        "round_number": args.round_number,
        "phase_requested": args.phase,
        "phases_executed": executed,
        "train_seasons": train_seasons,
        "train_policy": args.train_policy,
        "prediction_as_of": args.prediction_as_of,
        "phase_information_contract": (
            {
                phase_name: phase_information_contracts[phase_name]
                for phase_name in executed
                if phase_name in phase_information_contracts
            }
            if args.phase == "full"
            else phase_information_contracts.get(args.phase)
        ),
        "enable_dl_candidates": bool(args.enable_dl_candidates),
        "compare_families": compare_families,
        "dl_device": args.dl_device,
        "dl_arch": args.dl_arch,
        "dl_seed": int(args.dl_seed),
        "disable_runsim_features": bool(args.disable_runsim_features),
        "disable_circuit_features": bool(disable_circuit_features),
        "circuit_feature_state": "quarantined" if disable_circuit_features else "enabled_research_only",
        "weather": weather_config,
        "f1_mode": str(args.f1_mode),
        "f1_mode_effective": f1_mode_effective,
        "f1_live_source": str(args.f1_live_source),
        "f1_live_model": str(args.f1_live_model),
        "f1_live_horizon_laps": int(args.f1_live_horizon_laps),
        "f1_live_seed": int(args.f1_live_seed),
        "f1_live_cache_dir": args.f1_live_cache_dir,
        "f1_live_replay_path": args.f1_live_replay_path,
        "f1_live_calibration_path": args.f1_live_calibration_path,
        "f1_live_replay_cutoff_lap": args.f1_live_replay_cutoff_lap,
        "f1_live_replay_cutoff_time_seconds": args.f1_live_replay_cutoff_time_seconds,
        "f1_live_next_lap_ssm_weight": float(args.f1_live_next_lap_ssm_weight),
        "output_dir": str(output_dir),
        "artifacts": artifacts,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if isinstance(live_snapshot_payload, dict):
        for key in [
            "version",
            "rows",
            "notes",
            "model_name",
            "model_family",
            "device_used",
            "dl_available",
            "candidate_leaderboard",
            "live_summary",
            "trace_path",
            "trace_path_jsonl",
            "trace_format_effective",
            "prediction_phase",
            "race_input_evidence",
            "circuit_feature_state",
            "weather",
        ]:
            if key in live_snapshot_payload:
                payload[key] = live_snapshot_payload[key]

    if args.output_path:
        _write_json(Path(args.output_path), payload)

    if args.quiet:
        return

    if args.output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("=" * 72)
    print("F1 live weekend pipeline")
    print("=" * 72)
    print(f"Phase requested: {args.phase}")
    print(f"Phases executed: {', '.join(executed) if executed else 'none'}")
    print(f"Source: {args.source} | Year: {args.year} | Round: {args.round_number}")
    print(f"Train policy: {args.train_policy}")
    print(f"Train seasons: {', '.join(str(y) for y in train_seasons)}")
    print(f"Output directory: {output_dir}")
    print("\nArtifacts:")
    for phase_name, phase_artifacts in artifacts.items():
        print(f"- {phase_name}")
        if isinstance(phase_artifacts, dict):
            for key, value in phase_artifacts.items():
                print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
