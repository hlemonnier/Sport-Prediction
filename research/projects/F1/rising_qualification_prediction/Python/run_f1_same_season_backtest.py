#!/usr/bin/env python3
"""Same-season walk-forward F1 backtest.

Each target event trains only on prior rounds from the same season. Race
predictions are post-qualifying/pre-race: current-round grid or qualifying can
be used, but the target race result is never part of the training window.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from rqp.providers import LocalWeekendProvider

from run_f1_baseline_ladder import (
    _evaluate_variant,
    _qualifying_ladder_specs,
    _race_ladder_specs,
    _summarize_paired_ladder,
)


ROUND_DIR_RE = re.compile(r"round_(\d+)", re.IGNORECASE)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def default_weekends_dir() -> str:
    return str(_project_root() / "data" / "f1" / "raw" / "weekends")


def default_output_path() -> str:
    return str(_project_root() / "outputs" / "f1" / "reviews" / "same_season_walk_forward_backtest.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _available_years(weekends_dir: str) -> list[int]:
    root = Path(weekends_dir).expanduser()
    if not root.is_absolute():
        root = _project_root() / root
    if not root.exists():
        return []
    years: list[int] = []
    for candidate in root.iterdir():
        if candidate.is_dir() and candidate.name.isdigit():
            years.append(int(candidate.name))
    return sorted(years)


def _round_number_from_path(path: Path) -> Optional[int]:
    match = ROUND_DIR_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def _available_rounds(weekends_dir: str, year: int) -> list[int]:
    root = Path(weekends_dir).expanduser()
    if not root.is_absolute():
        root = _project_root() / root
    year_dir = root / str(year)
    if not year_dir.exists():
        return []
    rounds = [_round_number_from_path(path) for path in year_dir.iterdir() if path.is_dir()]
    return sorted({int(round_number) for round_number in rounds if round_number is not None})


def _resolve_years(value: str, weekends_dir: str) -> list[int]:
    if str(value).strip().lower() in {"auto", "all", "available"}:
        return _available_years(weekends_dir)
    return sorted(set(_parse_int_csv(value)))


def _filter_rounds(rounds: list[int], round_start: Optional[int], round_end: Optional[int]) -> list[int]:
    output = rounds
    if round_start is not None:
        output = [round_number for round_number in output if round_number >= int(round_start)]
    if round_end is not None:
        output = [round_number for round_number in output if round_number <= int(round_end)]
    return output


def _event_name(provider: LocalWeekendProvider, year: int, round_number: int) -> str:
    try:
        rounds = provider.list_rounds(year)
    except Exception:
        return f"Round {round_number}"
    for item in rounds:
        try:
            if int(item.get("round_number", 0)) == int(round_number):
                return str(item.get("event_name") or item.get("meeting_name") or item.get("country_name") or f"Round {round_number}")
        except Exception:
            continue
    return f"Round {round_number}"


def _actual_results(provider: LocalWeekendProvider, mode: str, year: int, round_number: int) -> pd.DataFrame:
    if mode == "race":
        return provider.get_race_results(year, round_number)
    return provider.get_qualifying_results(year, round_number)


def _training_rounds_for_target(provider: LocalWeekendProvider, mode: str, year: int, target_round: int) -> list[int]:
    try:
        rounds = provider.list_rounds(year)
    except Exception:
        return []
    output: list[int] = []
    for item in sorted(rounds, key=lambda row: int(row.get("round_number", 0))):
        round_number = int(item.get("round_number", 0))
        if round_number <= 0 or round_number >= int(target_round):
            continue
        try:
            actual = _actual_results(provider, mode, year, round_number)
        except Exception:
            continue
        if not actual.empty:
            output.append(round_number)
    return output


def _race_current_input_sources(provider: LocalWeekendProvider, year: int, round_number: int) -> list[str]:
    try:
        grid = provider.get_starting_grid(year, round_number)
    except Exception:
        grid = pd.DataFrame()
    if not grid.empty and "grid_source" in grid.columns:
        return sorted(set(grid["grid_source"].dropna().astype(str).tolist()))
    if not grid.empty:
        return ["starting_grid"]
    try:
        qualifying = provider.get_qualifying_results(year, round_number)
    except Exception:
        qualifying = pd.DataFrame()
    if not qualifying.empty:
        return ["qualifying_fallback"]
    return []


def _information_cutoff(mode: str) -> str:
    if mode == "race":
        return "post-qualifying_pre-race"
    return "post-practice_pre-qualifying"


def _current_event_inputs(provider: LocalWeekendProvider, mode: str, year: int, round_number: int) -> dict[str, Any]:
    if mode == "race":
        return {
            "allowed": True,
            "description": "current_round_starting_grid_or_qualifying_fallback",
            "sources": _race_current_input_sources(provider, year, round_number),
        }
    return {
        "allowed": True,
        "description": "current_round_practice_features",
        "sources": ["free_practice_features"],
    }


def _config_signature_for_skip(spec: dict[str, Any], mode: str, f1_model: str) -> dict[str, Any]:
    kind = str(spec["kind"])
    effective_model = str(spec.get("f1_model", f1_model))
    return {
        "kind": kind,
        "f1_model": effective_model if kind != "baseline" else str(spec["name"]),
        "disable_circuit_features": bool(spec.get("disable_circuit_features", False)) if kind != "baseline" else None,
        "race_delta_constraint_mode": str(spec.get("race_delta_constraint_mode", "constrained")) if mode == "race" and kind != "baseline" else None,
    }


def _evaluation_cache_key(
    *,
    spec: dict[str, Any],
    mode: str,
    year: int,
    round_number: int,
    weekends_dir: str,
    f1_model: str,
    f1_pl_samples: int,
) -> Optional[str]:
    if spec["kind"] != "model":
        return None
    return json.dumps(
        {
            "mode": mode,
            "year": int(year),
            "round_number": int(round_number),
            "train_seasons": [int(year)],
            "weekends_dir": weekends_dir,
            "f1_pl_samples": int(f1_pl_samples),
            "config_signature": _config_signature_for_skip(spec, mode, f1_model),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _retarget_cached_row(row: dict[str, Any], spec: dict[str, Any], mode: str, f1_model: str) -> dict[str, Any]:
    retargeted = copy.deepcopy(row)
    retargeted["variant"] = str(spec["name"])
    retargeted["kind"] = str(spec["kind"])
    retargeted["config_signature"] = _config_signature_for_skip(spec, mode, f1_model)
    return retargeted


def _skip_row(
    *,
    mode: str,
    year: int,
    round_number: int,
    event_name: str,
    spec: dict[str, Any],
    reason: str,
    f1_model: str,
    training_rounds_used: list[int],
    current_event_inputs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "season": int(year),
        "year": int(year),
        "mode": mode,
        "round": int(round_number),
        "event_name": event_name,
        "event_key": f"{int(year)}:{mode}:{int(round_number)}",
        "variant": str(spec["name"]),
        "kind": str(spec["kind"]),
        "model_name": None,
        "config_signature": _config_signature_for_skip(spec, mode, f1_model),
        "rows": 0,
        "metric_available": False,
        "field_mae": None,
        "mae_on_common": None,
        "field_coverage": None,
        "top10_hit": None,
        "podium_hit_count": None,
        "winner_hit": None,
        "evaluation": {"metric_available": False, "reason": reason},
        "skip_reason": reason,
        "train_policy_effective": "same_season_walk_forward",
        "train_seasons_effective": [int(year)],
        "same_season_only": True,
        "training_rounds_used": training_rounds_used,
        "training_event_count": int(len(training_rounds_used)),
        "target_round_allowed_in_training": False,
        "information_cutoff": _information_cutoff(mode),
        "current_event_inputs": current_event_inputs,
    }


def _enrich_row(
    row: dict[str, Any],
    *,
    mode: str,
    year: int,
    round_number: int,
    event_name: str,
    training_rounds_used: list[int],
    current_event_inputs: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(row)
    enriched["season"] = int(year)
    enriched["year"] = int(year)
    enriched["event_name"] = event_name
    enriched["event_key"] = f"{int(year)}:{mode}:{int(round_number)}"
    enriched["train_policy_effective"] = "same_season_walk_forward"
    enriched["train_seasons_effective"] = [int(year)]
    enriched["same_season_only"] = True
    enriched["training_rounds_used"] = training_rounds_used
    enriched["training_event_count"] = int(len(training_rounds_used))
    enriched["target_round_allowed_in_training"] = False
    enriched["information_cutoff"] = _information_cutoff(mode)
    enriched["current_event_inputs"] = current_event_inputs
    return enriched


def _season_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    summaries: dict[str, Any] = {}
    for season in sorted(frame["season"].dropna().astype(int).unique().tolist()):
        season_rows = frame[frame["season"].astype(int) == int(season)].to_dict(orient="records")
        season_frame = pd.DataFrame(season_rows)
        summaries[str(season)] = {
            "rounds_seen": sorted(season_frame["round"].dropna().astype(int).unique().tolist()),
            "rows": int(len(season_rows)),
            "metric_available_rows": int(season_frame["metric_available"].fillna(False).astype(bool).sum()),
            "skipped_rows": int(season_frame.get("skip_reason", pd.Series(index=season_frame.index)).notna().sum()),
            "summary": _summarize_paired_ladder(season_rows),
        }
    return summaries


def _evaluate_task(task: dict[str, Any]) -> dict[str, Any]:
    provider = LocalWeekendProvider(weekends_dir=str(task["weekends_dir"]))
    spec = task["spec"]
    assert isinstance(spec, dict)
    try:
        row = _evaluate_variant(
            provider=provider,
            mode=str(task["mode"]),
            spec=spec,
            year=int(task["year"]),
            round_number=int(task["round_number"]),
            train_seasons=[int(task["year"])],
            weekends_dir=str(task["weekends_dir"]),
            f1_model=str(task["f1_model"]),
            f1_pl_samples=int(task["f1_pl_samples"]),
        )
        return {"order": int(task["order"]), "row": row, "error": None}
    except Exception as exc:
        return {
            "order": int(task["order"]),
            "row": None,
            "error": f"evaluation_error:{type(exc).__name__}:{exc}",
        }


def _run_tasks(tasks: list[dict[str, Any]], max_workers: int) -> list[dict[str, Any]]:
    if int(max_workers) <= 1:
        return [_evaluate_task(task) for task in tasks]
    worker_count = min(int(max_workers), len(tasks))
    if worker_count <= 1:
        return [_evaluate_task(task) for task in tasks]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(_evaluate_task, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _finalize_task_result(
    *,
    result: dict[str, Any],
    task_by_order: dict[int, dict[str, Any]],
    aliases_by_order: dict[int, list[dict[str, Any]]],
    rows_by_order: dict[int, dict[str, Any]],
) -> None:
    order = int(result["order"])
    task = task_by_order[order]
    error = result.get("error")
    row = result.get("row")
    targets = [task] + aliases_by_order.get(order, [])
    for target in targets:
        spec = target["spec"]
        assert isinstance(spec, dict)
        if error or not isinstance(row, dict):
            rows_by_order[int(target["order"])] = _skip_row(
                mode=str(target["mode"]),
                year=int(target["year"]),
                round_number=int(target["round_number"]),
                event_name=str(target["event_name"]),
                spec=spec,
                reason=str(error or "evaluation_error:missing_row"),
                f1_model=str(target["f1_model"]),
                training_rounds_used=list(target["training_rounds_used"]),
                current_event_inputs=dict(target["current_event_inputs"]),
            )
            continue
        target_row = row if int(target["order"]) == order else _retarget_cached_row(row, spec, str(target["mode"]), str(target["f1_model"]))
        rows_by_order[int(target["order"])] = _enrich_row(
            target_row,
            mode=str(target["mode"]),
            year=int(target["year"]),
            round_number=int(target["round_number"]),
            event_name=str(target["event_name"]),
            training_rounds_used=list(target["training_rounds_used"]),
            current_event_inputs=dict(target["current_event_inputs"]),
        )


def build_same_season_backtest(
    *,
    weekends_dir: str,
    years: list[int],
    modes: list[str],
    round_start: Optional[int],
    round_end: Optional[int],
    f1_model: str,
    f1_pl_samples: int,
    max_workers: int = 1,
) -> dict[str, Any]:
    provider = LocalWeekendProvider(weekends_dir=weekends_dir)
    rows_by_order: dict[int, dict[str, Any]] = {}
    coverage: dict[str, Any] = {}
    task_by_order: dict[int, dict[str, Any]] = {}
    aliases_by_order: dict[int, list[dict[str, Any]]] = {}
    first_order_by_cache_key: dict[str, int] = {}
    order = 0

    for year in years:
        available_rounds = _available_rounds(weekends_dir, year)
        target_rounds = _filter_rounds(available_rounds, round_start, round_end)
        coverage[str(year)] = {
            "available_rounds": available_rounds,
            "target_rounds": target_rounds,
        }
        for round_number in target_rounds:
            event_name = _event_name(provider, year, round_number)
            for mode in modes:
                specs = _race_ladder_specs() if mode == "race" else _qualifying_ladder_specs()
                training_rounds_used = _training_rounds_for_target(provider, mode, year, round_number)
                current_event_inputs = _current_event_inputs(provider, mode, year, round_number)
                for spec in specs:
                    order += 1
                    if spec["kind"] == "model" and not training_rounds_used:
                        rows_by_order[order] = (
                            _skip_row(
                                mode=mode,
                                year=year,
                                round_number=round_number,
                                event_name=event_name,
                                spec=spec,
                                reason="no_prior_same_season_training_events",
                                f1_model=f1_model,
                                training_rounds_used=training_rounds_used,
                                current_event_inputs=current_event_inputs,
                            )
                        )
                        continue
                    cache_key = _evaluation_cache_key(
                        spec=spec,
                        mode=mode,
                        year=year,
                        round_number=round_number,
                        weekends_dir=weekends_dir,
                        f1_model=f1_model,
                        f1_pl_samples=f1_pl_samples,
                    )
                    task = {
                        "order": order,
                        "weekends_dir": weekends_dir,
                        "mode": mode,
                        "year": int(year),
                        "round_number": int(round_number),
                        "event_name": event_name,
                        "spec": spec,
                        "training_rounds_used": training_rounds_used,
                        "current_event_inputs": current_event_inputs,
                        "f1_model": f1_model,
                        "f1_pl_samples": int(f1_pl_samples),
                    }
                    if cache_key is not None and cache_key in first_order_by_cache_key:
                        aliases_by_order.setdefault(first_order_by_cache_key[cache_key], []).append(task)
                        continue
                    if cache_key is not None:
                        first_order_by_cache_key[cache_key] = order
                    task_by_order[order] = task

    for result in _run_tasks(list(task_by_order.values()), max_workers=max_workers):
        _finalize_task_result(
            result=result,
            task_by_order=task_by_order,
            aliases_by_order=aliases_by_order,
            rows_by_order=rows_by_order,
        )

    rows = [rows_by_order[index] for index in sorted(rows_by_order)]

    return {
        "workflow": "f1_same_season_walk_forward_backtest",
        "generated_at": _utc_now(),
        "training_protocol": {
            "name": "same_season_walk_forward",
            "rule": "For target season Y and round R, model variants receive train_seasons=[Y]; rqp.data.build_training_data excludes rounds >= R in the same season.",
            "race_information_cutoff": "post-qualifying/pre-race; current-round grid or qualifying fallback is allowed, target race result is not.",
            "qualifying_information_cutoff": "post-practice/pre-qualifying; current-round practice features are allowed, target qualifying result is not.",
            "cross_season_training_allowed": False,
        },
        "config": {
            "weekends_dir": weekends_dir,
            "years_effective": years,
            "modes": modes,
            "round_start": round_start,
            "round_end": round_end,
            "f1_model": f1_model,
            "f1_pl_samples": int(f1_pl_samples),
            "max_workers": int(max_workers),
        },
        "data_coverage": coverage,
        "summary": _summarize_paired_ladder(rows),
        "season_summaries": _season_summaries(rows),
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run same-season walk-forward F1 baseline ladder validation.")
    parser.add_argument("--weekends-dir", default=default_weekends_dir())
    parser.add_argument("--years", default="auto", help="Comma-separated seasons, or auto/all for local data coverage.")
    parser.add_argument("--round-start", type=int, default=None)
    parser.add_argument("--round-end", type=int, default=None)
    parser.add_argument("--modes", default="qualifying,race")
    parser.add_argument("--f1-model", choices=["auto", "baseline", "xgb_rank", "eb_rank", "lgbm_rank"], default="baseline")
    parser.add_argument("--f1-pl-samples", type=int, default=300)
    parser.add_argument("--max-workers", type=int, default=min(4, max(1, os.cpu_count() or 1)))
    parser.add_argument("--output-path", default=default_output_path())
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    years = _resolve_years(args.years, args.weekends_dir)
    modes = [part.strip().lower() for part in str(args.modes).split(",") if part.strip()]
    payload = build_same_season_backtest(
        weekends_dir=str(args.weekends_dir),
        years=years,
        modes=modes,
        round_start=args.round_start,
        round_end=args.round_end,
        f1_model=str(args.f1_model),
        f1_pl_samples=int(args.f1_pl_samples),
        max_workers=int(args.max_workers),
    )
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
