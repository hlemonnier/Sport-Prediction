#!/usr/bin/env python3
"""Operational weekend pipeline for in-season F1 predictions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from rqp import PredictionConfig, run_prediction
from rqp.providers import BaseProvider, FastF1Provider, LocalWeekendProvider, OpenF1Provider


def parse_train_seasons(value: str, target_year: int, policy: str) -> list[int]:
    if value.lower() not in {"auto", "default"}:
        return sorted({int(x.strip()) for x in value.split(",") if x.strip()})

    if policy == "strict_transfer":
        # Example 2026 target: train on 2022-2024 + 2026 rounds already completed.
        seasons = [target_year - 4, target_year - 3, target_year - 2, target_year]
    elif policy == "rolling":
        # Example 2026 target: train on 2023-2025 + 2026 rounds already completed.
        seasons = [target_year - 3, target_year - 2, target_year - 1, target_year]
    elif policy == "frozen_preseason":
        # Static historical block only, no in-season adaptation.
        seasons = [target_year - 4, target_year - 3, target_year - 2]
    else:
        # Backward-compatible fallback.
        seasons = [target_year - 2, target_year - 1, target_year]

    return sorted({int(y) for y in seasons if int(y) > 0})


def default_output_dir() -> str:
    project_root = Path(__file__).resolve().parents[5]
    return str(project_root / "data" / "f1" / "live" / "2026" / "pipeline_runs")


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
    return {
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


def _normalize_name_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    return " ".join(text.split())


def _actual_name_column(frame: pd.DataFrame) -> Optional[str]:
    for col in ["driver_name", "driver_id", "Abbreviation", "Driver"]:
        if col in frame.columns:
            return col
    return None


def _evaluate_prediction_rows(
    predicted_rows: list[dict[str, Any]],
    actual_results: pd.DataFrame,
    actual_position_col: str,
) -> dict[str, Any]:
    if not predicted_rows:
        return {"available": False, "reason": "prediction_rows_unavailable"}
    if actual_results is None or actual_results.empty:
        return {"available": False, "reason": "actual_results_unavailable"}

    pred = pd.DataFrame(predicted_rows).copy()
    if pred.empty or "driver_name" not in pred.columns:
        return {"available": False, "reason": "prediction_driver_name_unavailable"}
    pred["driver_key"] = pred["driver_name"].map(_normalize_name_key)
    pred = pred[pred["driver_key"] != ""]
    if pred.empty:
        return {"available": False, "reason": "prediction_driver_key_unavailable"}
    if "rank" in pred.columns:
        pred["pred_rank"] = pd.to_numeric(pred["rank"], errors="coerce")
    else:
        pred["pred_rank"] = pd.Series(range(1, len(pred) + 1), index=pred.index, dtype=float)
    pred = pred.dropna(subset=["pred_rank"])
    pred["pred_rank"] = pred["pred_rank"].astype(float)

    actual = actual_results.copy()
    if actual_position_col not in actual.columns:
        return {"available": False, "reason": "actual_position_unavailable"}
    name_col = _actual_name_column(actual)
    if name_col is None:
        return {"available": False, "reason": "actual_driver_name_unavailable"}
    actual["driver_key"] = actual[name_col].map(_normalize_name_key)
    actual["actual_rank"] = pd.to_numeric(actual[actual_position_col], errors="coerce")
    actual = actual[(actual["driver_key"] != "") & actual["actual_rank"].notna()]
    if actual.empty:
        return {"available": False, "reason": "actual_clean_unavailable"}

    pred_unique = pred.sort_values("pred_rank", kind="mergesort").drop_duplicates(subset=["driver_key"], keep="first")
    actual_unique = actual.sort_values("actual_rank", kind="mergesort").drop_duplicates(
        subset=["driver_key"],
        keep="first",
    )
    merged = pred_unique.merge(actual_unique[["driver_key", "actual_rank"]], on="driver_key", how="inner")
    merged = merged.dropna(subset=["pred_rank", "actual_rank"])

    mae = float((merged["pred_rank"] - merged["actual_rank"]).abs().mean()) if not merged.empty else None
    predicted_top10 = set(pred_unique.sort_values("pred_rank").head(10)["driver_key"].tolist())
    actual_top10 = set(actual_unique[actual_unique["actual_rank"] <= 10]["driver_key"].tolist())
    top10_hit = None
    if actual_top10:
        top10_hit = float(len(predicted_top10.intersection(actual_top10)) / float(min(10, len(actual_top10))))

    predicted_top3 = set(pred_unique.sort_values("pred_rank").head(3)["driver_key"].tolist())
    actual_top3 = set(actual_unique[actual_unique["actual_rank"] <= 3]["driver_key"].tolist())
    top3_hit = float(len(predicted_top3.intersection(actual_top3))) if actual_top3 else None

    winner_pred_key = pred_unique.sort_values("pred_rank").head(1)["driver_key"].tolist()
    winner_actual_key = actual_unique.sort_values("actual_rank").head(1)["driver_key"].tolist()
    winner_hit = None
    if winner_pred_key and winner_actual_key:
        winner_hit = bool(winner_pred_key[0] == winner_actual_key[0])

    return {
        "available": True,
        "rows_predicted": int(len(pred_unique)),
        "rows_actual": int(len(actual_unique)),
        "rows_common": int(len(merged)),
        "mae_on_common": mae,
        "top10_hit": top10_hit,
        "podium_hit_count": top3_hit,
        "winner_hit": winner_hit,
    }


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
    )
    race_path = output_dir / "postqual_race_prediction.json"
    _write_json(race_path, race_payload)

    prequal_qualifying = _load_json(output_dir / "prequal_qualifying_prediction.json")
    actual_qualifying = provider.get_qualifying_results(year, round_number)
    qual_eval = _evaluate_prediction_rows(
        predicted_rows=prequal_qualifying.get("rows", []) if prequal_qualifying else [],
        actual_results=actual_qualifying,
        actual_position_col="position",
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

    qualifying_eval = _evaluate_prediction_rows(
        predicted_rows=prequal_qualifying.get("rows", []) if prequal_qualifying else [],
        actual_results=actual_qualifying,
        actual_position_col="position",
    )
    race_eval_prequal = _evaluate_prediction_rows(
        predicted_rows=prequal_race.get("rows", []) if prequal_race else [],
        actual_results=actual_race,
        actual_position_col="position",
    )
    race_eval_postqual = _evaluate_prediction_rows(
        predicted_rows=postqual_race.get("rows", []) if postqual_race else [],
        actual_results=actual_race,
        actual_position_col="position",
    )

    delta_mae = None
    delta_top10 = None
    if race_eval_prequal.get("available") and race_eval_postqual.get("available"):
        pre_mae = race_eval_prequal.get("mae_on_common")
        post_mae = race_eval_postqual.get("mae_on_common")
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
            "mae_on_common": delta_mae,
            "top10_hit": delta_top10,
        },
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    output_path = output_dir / "postrace_evaluation.json"
    _write_json(output_path, payload)
    return {"evaluation": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live weekend pipeline: pre-qualifying, post-qualifying, post-race.",
    )
    parser.add_argument("--phase", choices=["pre-qualifying", "post-qualifying", "post-race", "full"], required=True)
    parser.add_argument("--source", choices=["fastf1", "openf1", "local"], required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", dest="round_number", type=int, required=True)
    parser.add_argument("--train-seasons", default="auto")
    parser.add_argument(
        "--train-policy",
        choices=["strict_transfer", "rolling", "frozen_preseason", "legacy_auto"],
        default="strict_transfer",
        help=(
            "auto train-season policy. "
            "strict_transfer excludes target_year-1 (ex: excludes 2025 for 2026)."
        ),
    )
    parser.add_argument("--include-standings", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--weekends-dir", default="data/f1/raw/weekends")
    parser.add_argument("--meeting-name", default=None)
    parser.add_argument("--country-name", default=None)
    parser.add_argument("--enable-dl-candidates", action="store_true")
    parser.add_argument("--compare-families", default="ml")
    parser.add_argument("--dl-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dl-arch", default="mlp_tabular_v1")
    parser.add_argument("--dl-hyperparams", default="{}")
    parser.add_argument("--dl-seed", type=int, default=42)
    parser.add_argument("--disable-runsim-features", action="store_true")
    parser.add_argument("--output-dir", default=default_output_dir())
    parser.add_argument("--output-format", choices=["text", "json"], default="text")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    train_seasons = parse_train_seasons(args.train_seasons, args.year, args.train_policy)
    compare_families = [part.strip().lower() for part in args.compare_families.split(",") if part.strip()]
    if not compare_families:
        compare_families = ["ml"]
    try:
        dl_hyperparams = json.loads(args.dl_hyperparams)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --dl-hyperparams JSON: {exc}") from exc
    if not isinstance(dl_hyperparams, dict):
        raise SystemExit("Invalid --dl-hyperparams: expected JSON object.")
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
        "enable_dl_candidates": bool(args.enable_dl_candidates),
        "compare_families": compare_families,
        "dl_device": args.dl_device,
        "dl_arch": args.dl_arch,
        "dl_seed": int(args.dl_seed),
        "disable_runsim_features": bool(args.disable_runsim_features),
        "output_dir": str(output_dir),
        "artifacts": artifacts,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

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
