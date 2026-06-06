#!/usr/bin/env python3
"""Strict paired baseline ladder for F1 prediction validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from rqp import PredictionConfig, run_prediction
from rqp.evaluation import evaluate_prediction_rows
from rqp.providers import LocalWeekendProvider
from rqp.runtime import parse_train_seasons


def _race_ladder_specs() -> list[dict[str, Any]]:
    return [
        {"name": "grid_only", "kind": "baseline"},
        {"name": "grid_plus_fp_race_pace_residual", "kind": "baseline"},
        {
            "name": "grid_delta_unconstrained",
            "kind": "model",
            "disable_circuit_features": True,
            "race_delta_constraint_mode": "unconstrained",
        },
        {
            "name": "grid_delta_constrained",
            "kind": "model",
            "disable_circuit_features": True,
            "race_delta_constraint_mode": "constrained",
        },
        {
            "name": "current_full_model",
            "kind": "model",
            "disable_circuit_features": True,
            "race_delta_constraint_mode": "constrained",
        },
        {
            "name": "current_full_model_plus_circuit_cards",
            "kind": "model",
            "disable_circuit_features": False,
            "race_delta_constraint_mode": "constrained",
        },
    ]


def _qualifying_ladder_specs() -> list[dict[str, Any]]:
    return [
        {"name": "fp_weighted_rank_baseline", "kind": "baseline"},
        {"name": "fp_plus_driver_team_rolling_priors", "kind": "model", "f1_model": "baseline"},
        {"name": "current_full_model", "kind": "model", "disable_circuit_features": True},
        {"name": "current_full_model_plus_circuit_cards", "kind": "model", "disable_circuit_features": False},
    ]


def _records(result: object) -> list[dict[str, Any]]:
    extras = getattr(result, "extras", {})
    rows = extras.get("all_prediction_rows") if isinstance(extras, dict) else None
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    table = getattr(result, "table", pd.DataFrame())
    if table.empty:
        return []
    return json.loads(table.to_json(orient="records"))


def _prediction_config(
    *,
    mode: str,
    year: int,
    round_number: int,
    train_seasons: list[int],
    weekends_dir: str,
    f1_model: str,
    f1_pl_samples: int,
    disable_circuit_features: bool,
    race_delta_constraint_mode: str,
) -> PredictionConfig:
    return PredictionConfig(
        source="local",
        mode=mode,
        year=int(year),
        round_number=int(round_number),
        train_seasons=train_seasons,
        include_standings=False,
        cache_dir=None,
        meeting_name=None,
        country_name=None,
        weekends_dir=weekends_dir,
        f1_model=f1_model,
        f1_pl_samples=int(f1_pl_samples),
        disable_circuit_features=bool(disable_circuit_features),
        race_delta_constraint_mode=str(race_delta_constraint_mode),
        shadow_eval=True,
    )


def _rank_rows(frame: pd.DataFrame, *, score_col: str) -> list[dict[str, Any]]:
    if frame.empty or score_col not in frame.columns:
        return []
    work = frame.copy()
    work["pred"] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.dropna(subset=["pred"]).sort_values("pred", kind="mergesort").copy()
    work["rank"] = range(1, len(work) + 1)
    if "driver_name" not in work.columns:
        work["driver_name"] = work.get("driver_id", pd.Series(index=work.index, dtype=object)).astype(str)
    return json.loads(work[["rank", "driver_name", "driver_id", "pred"]].to_json(orient="records"))


def _grid_baseline_frame(provider: LocalWeekendProvider, year: int, round_number: int) -> pd.DataFrame:
    grid = provider.get_starting_grid(year, round_number)
    if grid is not None and not grid.empty and "grid_position" in grid.columns:
        work = grid.copy()
        work["grid_position"] = pd.to_numeric(work["grid_position"], errors="coerce")
        work = work.dropna(subset=["grid_position"])
        if not work.empty:
            if "driver_name" not in work.columns:
                work["driver_name"] = work.get("driver_id", pd.Series(index=work.index, dtype=object)).astype(str)
            work["grid_source"] = work.get("grid_source", "pre_race_official_grid")
            return work

    qualy = provider.get_qualifying_results(year, round_number)
    if qualy.empty or "position" not in qualy.columns:
        return pd.DataFrame()
    work = qualy.copy()
    work["grid_position"] = pd.to_numeric(work["position"], errors="coerce")
    work = work.dropna(subset=["grid_position"])
    if "driver_name" not in work.columns:
        work["driver_name"] = work.get("driver_id", pd.Series(index=work.index, dtype=object)).astype(str)
    work["grid_source"] = "qualifying_fallback"
    return work


def _score_grid_only(provider: LocalWeekendProvider, year: int, round_number: int) -> list[dict[str, Any]]:
    return _rank_rows(_grid_baseline_frame(provider, year, round_number), score_col="grid_position")


def _score_grid_plus_fp_residual(provider: LocalWeekendProvider, year: int, round_number: int) -> list[dict[str, Any]]:
    grid = _grid_baseline_frame(provider, year, round_number)
    fp = provider.get_fp_features(year, round_number)
    if grid.empty or fp.empty:
        return []
    grid_rows = pd.DataFrame(_score_grid_only(provider, year, round_number))
    if grid_rows.empty:
        return []
    pace_col = next(
        (col for col in ["event_pace_index", "fp_race_sim_delta", "fp_weighted_delta", "fp_mean_delta"] if col in fp.columns),
        None,
    )
    if pace_col is None:
        return []
    work = grid_rows[["driver_id", "pred"]].rename(columns={"pred": "grid_score"}).merge(
        fp[["driver_id", "driver_name", pace_col]],
        on="driver_id",
        how="inner",
    )
    pace_rank = pd.to_numeric(work[pace_col], errors="coerce").rank(method="average", ascending=True)
    work["score"] = pd.to_numeric(work["grid_score"], errors="coerce") + (0.25 * pace_rank)
    return _rank_rows(work, score_col="score")


def _baseline_rows(provider: LocalWeekendProvider, mode: str, variant: str, year: int, round_number: int) -> list[dict[str, Any]]:
    if mode == "race":
        if variant == "grid_only":
            return _score_grid_only(provider, year, round_number)
        if variant == "grid_plus_fp_race_pace_residual":
            return _score_grid_plus_fp_residual(provider, year, round_number)
        return []
    fp = provider.get_fp_features(year, round_number)
    if fp.empty:
        return []
    score_col = next(
        (col for col in ["event_pace_index", "fp_weighted_delta", "fp_quali_sim_delta", "fp_mean_delta"] if col in fp.columns),
        None,
    )
    if score_col is None:
        return []
    return _rank_rows(fp, score_col=score_col)


def _evaluate_variant(
    *,
    provider: LocalWeekendProvider,
    mode: str,
    spec: dict[str, Any],
    year: int,
    round_number: int,
    train_seasons: list[int],
    weekends_dir: str,
    f1_model: str,
    f1_pl_samples: int,
) -> dict[str, Any]:
    actual = provider.get_race_results(year, round_number) if mode == "race" else provider.get_qualifying_results(year, round_number)
    if spec["kind"] == "baseline":
        rows = _baseline_rows(provider, mode, str(spec["name"]), year, round_number)
        model_name = str(spec["name"])
    else:
        config = _prediction_config(
            mode=mode,
            year=year,
            round_number=round_number,
            train_seasons=train_seasons,
            weekends_dir=weekends_dir,
            f1_model=str(spec.get("f1_model", f1_model)),
            f1_pl_samples=int(f1_pl_samples),
            disable_circuit_features=bool(spec.get("disable_circuit_features", False)),
            race_delta_constraint_mode=str(spec.get("race_delta_constraint_mode", "constrained")),
        )
        result = run_prediction(config)
        rows = _records(result)
        model_name = getattr(result, "model_name", str(spec["name"]))
    evaluation = evaluate_prediction_rows(rows, actual, "position", include_podium_and_winner=True)
    return {
        "mode": mode,
        "round": int(round_number),
        "variant": str(spec["name"]),
        "kind": str(spec["kind"]),
        "model_name": model_name,
        "rows": int(len(rows)),
        "metric_available": bool(evaluation.get("metric_available", False)),
        "field_mae": evaluation.get("field_mae"),
        "mae_on_common": evaluation.get("mae_on_common"),
        "field_coverage": evaluation.get("field_coverage"),
        "top10_hit": evaluation.get("top10_hit"),
        "podium_hit_count": evaluation.get("podium_hit_count"),
        "winner_hit": evaluation.get("winner_hit"),
        "evaluation": evaluation,
    }


def _summarize_variant_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    frame["event_key"] = frame["mode"].astype(str) + ":" + frame["round"].astype(str)
    variants = sorted(frame["variant"].dropna().astype(str).unique().tolist())
    event_sets = {
        variant: set(frame[(frame["variant"] == variant) & (frame["metric_available"] == True)]["event_key"].tolist())
        for variant in variants
    }
    common_events = set.intersection(*event_sets.values()) if event_sets else set()
    summary: dict[str, Any] = {
        "available": True,
        "variants": variants,
        "common_event_count": int(len(common_events)),
        "variant_event_counts": {variant: int(len(events)) for variant, events in event_sets.items()},
        "variant_metrics": {},
    }
    common_frame = frame[frame["event_key"].isin(common_events)].copy()
    for variant in variants:
        part = common_frame[common_frame["variant"] == variant]
        mae = pd.to_numeric(part["field_mae"], errors="coerce")
        top10 = pd.to_numeric(part["top10_hit"], errors="coerce")
        summary["variant_metrics"][variant] = {
            "events": int(len(part)),
            "field_mae_avg": float(mae.mean()) if mae.notna().any() else None,
            "top10_hit_avg": float(top10.mean()) if top10.notna().any() else None,
        }
    return summary


def _summarize_paired_ladder(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"available": False, "reason": "no_rows"}
    frame = pd.DataFrame(rows)
    modes = sorted(frame["mode"].dropna().astype(str).unique().tolist())
    if len(modes) == 1:
        return _summarize_variant_rows(rows)
    mode_summaries = {
        mode: _summarize_variant_rows(frame[frame["mode"].astype(str) == mode].to_dict(orient="records"))
        for mode in modes
    }
    return {
        "available": True,
        "modes": modes,
        "mode_summaries": mode_summaries,
        "common_event_count_by_mode": {
            mode: int(summary.get("common_event_count", 0))
            for mode, summary in mode_summaries.items()
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paired F1 baseline ladder validation.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--round-start", type=int, default=1)
    parser.add_argument("--round-end", type=int, default=24)
    parser.add_argument("--modes", default="qualifying,race")
    parser.add_argument("--weekends-dir", default="data/f1/raw/weekends")
    parser.add_argument("--train-seasons", default="auto")
    parser.add_argument("--train-policy", choices=["strict_transfer", "rolling", "frozen_preseason", "legacy_auto"], default="legacy_auto")
    parser.add_argument("--f1-model", choices=["auto", "baseline", "xgb_rank", "eb_rank", "lgbm_rank"], default="auto")
    parser.add_argument("--f1-pl-samples", type=int, default=300)
    parser.add_argument("--output-path", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    provider = LocalWeekendProvider(weekends_dir=args.weekends_dir)
    train_seasons = parse_train_seasons(args.train_seasons, int(args.year), args.train_policy)
    modes = [part.strip().lower() for part in str(args.modes).split(",") if part.strip()]
    rows: list[dict[str, Any]] = []
    for mode in modes:
        specs = _race_ladder_specs() if mode == "race" else _qualifying_ladder_specs()
        for round_number in range(int(args.round_start), int(args.round_end) + 1):
            for spec in specs:
                rows.append(
                    _evaluate_variant(
                        provider=provider,
                        mode=mode,
                        spec=spec,
                        year=int(args.year),
                        round_number=int(round_number),
                        train_seasons=train_seasons,
                        weekends_dir=str(args.weekends_dir),
                        f1_model=str(args.f1_model),
                        f1_pl_samples=int(args.f1_pl_samples),
                    )
                )
    payload = {
        "workflow": "f1_baseline_ladder",
        "config": {**vars(args), "train_seasons_effective": train_seasons},
        "summary": _summarize_paired_ladder(rows),
        "rows": rows,
    }
    if args.output_path:
        path = Path(args.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
