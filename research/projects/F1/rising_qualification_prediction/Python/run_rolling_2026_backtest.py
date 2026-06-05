#!/usr/bin/env python3
"""Walk-forward 2026 F1 backtest with weighted current-season learning."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from rqp import PredictionConfig, run_prediction
from rqp.evaluation import evaluate_prediction_rows
from rqp.providers import FastF1Provider, LocalWeekendProvider, OpenF1Provider
from rqp.runtime import parse_compare_families, parse_json_object


def default_output_dir() -> str:
    project_root = Path(__file__).resolve().parents[5]
    return str(project_root / "outputs" / "f1" / "rolling_2026")


def default_weekends_dir() -> str:
    project_root = Path(__file__).resolve().parents[5]
    return str(project_root / "data" / "f1" / "raw" / "weekends")


def default_selection_csv() -> str:
    project_root = Path(__file__).resolve().parents[5]
    return str(project_root / "outputs" / "f1" / "docs" / "F1_2026_Rolling_Backtest_Selections.csv")


def default_summary_csv() -> str:
    project_root = Path(__file__).resolve().parents[5]
    return str(project_root / "outputs" / "f1" / "docs" / "F1_2026_Rolling_Backtest_Summary.csv")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _normalize_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def _provider(source: str, cache_dir: Optional[str], weekends_dir: Optional[str], round_number: int) -> object:
    if source == "fastf1":
        return FastF1Provider(cache_dir=cache_dir)
    if source == "openf1":
        return OpenF1Provider(cache_dir=cache_dir, target_round=round_number)
    return LocalWeekendProvider(weekends_dir=weekends_dir)


def _actual_position_lookup(actual: pd.DataFrame) -> dict[str, float]:
    if actual.empty or "position" not in actual.columns:
        return {}
    frame = actual.copy()
    frame["position"] = pd.to_numeric(frame["position"], errors="coerce")
    frame = frame.dropna(subset=["position"])
    lookup: dict[str, float] = {}
    for _, row in frame.iterrows():
        position = float(row["position"])
        for col in ["driver_id", "driver_name", "Abbreviation", "Driver"]:
            if col not in frame.columns:
                continue
            key = _normalize_key(row.get(col))
            if key and key not in lookup:
                lookup[key] = position
    return lookup


def _actual_position_for_row(row: dict[str, Any], lookup: dict[str, float]) -> Optional[float]:
    for col in ["driver_id", "driver_name"]:
        key = _normalize_key(row.get(col))
        if key in lookup:
            return lookup[key]
    return None


def _event_name(provider: object, year: int, round_number: int) -> str:
    try:
        rounds = provider.list_rounds(year)  # type: ignore[attr-defined]
    except Exception:
        return f"Round {round_number}"
    for item in rounds:
        try:
            if int(item.get("round_number", 0)) == round_number:
                return str(item.get("event_name") or item.get("meeting_name") or item.get("country_name") or f"Round {round_number}")
        except Exception:
            continue
    return f"Round {round_number}"


def _prediction_rows(result: object) -> list[dict[str, Any]]:
    extras = getattr(result, "extras", None)
    if isinstance(extras, dict) and isinstance(extras.get("all_prediction_rows"), list):
        return list(extras["all_prediction_rows"])
    table = getattr(result, "table", pd.DataFrame())
    if isinstance(table, pd.DataFrame) and not table.empty:
        return json.loads(table.to_json(orient="records"))
    return []


def _selection_rows(
    *,
    year: int,
    round_number: int,
    race_name: str,
    target: str,
    information_cutoff: str,
    result: object,
    actual: pd.DataFrame,
    base_train_seasons: list[int],
    train_seasons_used: list[int],
    current_year_weight_multiplier: float,
) -> list[dict[str, Any]]:
    rows = _prediction_rows(result)
    rows = sorted(rows, key=lambda r: float(r.get("rank", 9999)))
    lookup = _actual_position_lookup(actual)
    output: list[dict[str, Any]] = []
    markets = [
        ("winner", 1, "proba_win"),
        ("podium", 3, "proba_top3"),
        ("top10", 10, "proba_top10"),
    ]
    for market, cutoff_rank, probability_col in markets:
        for row in rows[:cutoff_rank]:
            actual_position = _actual_position_for_row(row, lookup)
            hit = None
            if actual_position is not None:
                hit = actual_position <= cutoff_rank
            probability = row.get(probability_col)
            output.append(
                {
                    "season": year,
                    "round": round_number,
                    "grand_prix": race_name,
                    "target": target,
                    "information_cutoff": information_cutoff,
                    "market": market,
                    "selection": row.get("driver_name") or row.get("driver_id"),
                    "driver_id": row.get("driver_id"),
                    "model_rank": int(row.get("rank")) if pd.notna(row.get("rank")) else None,
                    "model_probability_pct": round(float(probability) * 100.0, 3) if probability is not None and pd.notna(probability) else None,
                    "prediction_score": row.get("pred"),
                    "actual_position": actual_position,
                    "result": "hit" if hit is True else "miss" if hit is False else "unmatched",
                    "model_name": getattr(result, "model_name", None),
                    "model_family": getattr(result, "model_family", None),
                    "base_train_seasons": ",".join(str(y) for y in base_train_seasons),
                    "train_seasons_used": ",".join(str(y) for y in train_seasons_used),
                    "rolling_2026_rounds_used": ",".join(str(r) for r in range(1, round_number)),
                    "current_year_weight_multiplier": float(current_year_weight_multiplier),
                }
            )
    return output


def _summary_row(
    *,
    year: int,
    round_number: int,
    race_name: str,
    target: str,
    information_cutoff: str,
    result: object,
    actual: pd.DataFrame,
    base_train_seasons: list[int],
    train_seasons_used: list[int],
    current_year_weight_multiplier: float,
) -> dict[str, Any]:
    rows = _prediction_rows(result)
    evaluation = evaluate_prediction_rows(
        predicted_rows=rows,
        actual_results=actual,
        actual_position_col="position",
        include_podium_and_winner=True,
    )
    top_driver = rows[0] if rows else {}
    leaderboard = getattr(result, "candidate_leaderboard", []) or []
    top_candidate = leaderboard[0] if leaderboard else {}
    return {
        "season": year,
        "round": round_number,
        "grand_prix": race_name,
        "target": target,
        "information_cutoff": information_cutoff,
        "rows_predicted": evaluation.get("rows_predicted"),
        "rows_actual": evaluation.get("rows_actual"),
        "rows_common": evaluation.get("rows_common"),
        "mae_on_common": evaluation.get("mae_on_common"),
        "top10_hit_pct": round(float(evaluation["top10_hit"]) * 100.0, 2) if evaluation.get("top10_hit") is not None else None,
        "podium_hit_count": evaluation.get("podium_hit_count"),
        "winner_hit": evaluation.get("winner_hit"),
        "top_model_selection": top_driver.get("driver_name") or top_driver.get("driver_id"),
        "top_model_probability_pct": round(float(top_driver.get("proba_win")) * 100.0, 3) if top_driver.get("proba_win") is not None and pd.notna(top_driver.get("proba_win")) else None,
        "model_name": getattr(result, "model_name", None),
        "model_family": getattr(result, "model_family", None),
        "cv_top_candidate": top_candidate.get("name"),
        "cv_top_composite": top_candidate.get("composite"),
        "base_train_seasons": ",".join(str(y) for y in base_train_seasons),
        "train_seasons_used": ",".join(str(y) for y in train_seasons_used),
        "rolling_2026_rounds_used": ",".join(str(r) for r in range(1, round_number)),
        "current_year_weight_multiplier": float(current_year_weight_multiplier),
    }


def _write_prediction_artifact(path: Path, config: PredictionConfig, result: object) -> None:
    table = getattr(result, "table", pd.DataFrame())
    payload = {
        "sport": "F1",
        "project": "Rising Qualification Prediction",
        "config": asdict(config),
        "rows": json.loads(table.to_json(orient="records")) if isinstance(table, pd.DataFrame) and not table.empty else [],
        "all_prediction_rows": _prediction_rows(result),
        "notes": getattr(result, "notes", []),
        "model_name": getattr(result, "model_name", None),
        "model_family": getattr(result, "model_family", None),
        "candidate_leaderboard": getattr(result, "candidate_leaderboard", []),
        "generated_at": _utc_now(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rolling 2026 backtest: train on 2022-2025 plus prior 2026 rounds with weighted current-season data.",
    )
    parser.add_argument("--source", choices=["local", "fastf1", "openf1"], default="local")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--rounds", default="1,2,3,4,5")
    parser.add_argument("--base-train-seasons", default="2022,2023,2024,2025")
    parser.add_argument("--current-season-weight-multiplier", type=float, default=3.0)
    parser.add_argument("--include-standings", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--weekends-dir", default=default_weekends_dir())
    parser.add_argument("--compare-families", default="ml")
    parser.add_argument("--enable-dl-candidates", action="store_true")
    parser.add_argument("--dl-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dl-arch", default="mlp_tabular_v1")
    parser.add_argument("--dl-hyperparams", default="{}")
    parser.add_argument("--dl-seed", type=int, default=42)
    parser.add_argument("--disable-runsim-features", action="store_true")
    parser.add_argument("--f1-model", default="auto")
    parser.add_argument("--output-dir", default=default_output_dir())
    parser.add_argument("--selection-csv", default=default_selection_csv())
    parser.add_argument("--summary-csv", default=default_summary_csv())
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    rounds = _parse_int_list(args.rounds)
    base_train_seasons = _parse_int_list(args.base_train_seasons)
    train_seasons_used = sorted(set(base_train_seasons + [int(args.year)]))
    compare_families = parse_compare_families(args.compare_families)
    dl_hyperparams = parse_json_object(args.dl_hyperparams, "--dl-hyperparams")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for round_number in rounds:
        provider = _provider(args.source, args.cache_dir, args.weekends_dir, round_number)
        race_name = _event_name(provider, args.year, round_number)
        actual_qualifying = provider.get_qualifying_results(args.year, round_number)  # type: ignore[attr-defined]
        actual_race = provider.get_race_results(args.year, round_number)  # type: ignore[attr-defined]

        common_config = {
            "source": args.source,
            "year": args.year,
            "round_number": round_number,
            "train_seasons": train_seasons_used,
            "include_standings": bool(args.include_standings),
            "cache_dir": args.cache_dir,
            "meeting_name": None,
            "country_name": None,
            "weekends_dir": args.weekends_dir,
            "enable_dl_candidates": bool(args.enable_dl_candidates),
            "compare_families": compare_families,
            "dl_device": args.dl_device,
            "dl_arch": args.dl_arch,
            "dl_hyperparams": dl_hyperparams,
            "dl_seed": int(args.dl_seed),
            "disable_runsim_features": bool(args.disable_runsim_features),
            "f1_model": args.f1_model,
            "season_weight_year": int(args.year),
            "season_weight_multiplier": float(args.current_season_weight_multiplier),
        }

        qualifying_config = PredictionConfig(mode="qualifying", **common_config)
        qualifying_result = run_prediction(qualifying_config)
        _write_prediction_artifact(
            output_dir / str(args.year) / f"round_{round_number:02d}" / "qualifying_prediction.json",
            qualifying_config,
            qualifying_result,
        )
        selection_rows.extend(
            _selection_rows(
                year=args.year,
                round_number=round_number,
                race_name=race_name,
                target="qualifying",
                information_cutoff="post-practice",
                result=qualifying_result,
                actual=actual_qualifying,
                base_train_seasons=base_train_seasons,
                train_seasons_used=train_seasons_used,
                current_year_weight_multiplier=args.current_season_weight_multiplier,
            )
        )
        summary_rows.append(
            _summary_row(
                year=args.year,
                round_number=round_number,
                race_name=race_name,
                target="qualifying",
                information_cutoff="post-practice",
                result=qualifying_result,
                actual=actual_qualifying,
                base_train_seasons=base_train_seasons,
                train_seasons_used=train_seasons_used,
                current_year_weight_multiplier=args.current_season_weight_multiplier,
            )
        )

        race_config = PredictionConfig(mode="race", **common_config)
        race_result = run_prediction(race_config)
        _write_prediction_artifact(
            output_dir / str(args.year) / f"round_{round_number:02d}" / "postqual_race_prediction.json",
            race_config,
            race_result,
        )
        selection_rows.extend(
            _selection_rows(
                year=args.year,
                round_number=round_number,
                race_name=race_name,
                target="race",
                information_cutoff="post-qualifying",
                result=race_result,
                actual=actual_race,
                base_train_seasons=base_train_seasons,
                train_seasons_used=train_seasons_used,
                current_year_weight_multiplier=args.current_season_weight_multiplier,
            )
        )
        summary_rows.append(
            _summary_row(
                year=args.year,
                round_number=round_number,
                race_name=race_name,
                target="race",
                information_cutoff="post-qualifying",
                result=race_result,
                actual=actual_race,
                base_train_seasons=base_train_seasons,
                train_seasons_used=train_seasons_used,
                current_year_weight_multiplier=args.current_season_weight_multiplier,
            )
        )

    selection_path = Path(args.selection_csv)
    summary_path = Path(args.summary_csv)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selection_rows).to_csv(selection_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    if not args.quiet:
        print(json.dumps(
            {
                "sport": "F1",
                "year": args.year,
                "rounds": rounds,
                "base_train_seasons": base_train_seasons,
                "train_seasons_used": train_seasons_used,
                "current_year_weight_multiplier": float(args.current_season_weight_multiplier),
                "selection_csv": str(selection_path),
                "summary_csv": str(summary_path),
                "artifact_dir": str(output_dir),
                "selection_rows": len(selection_rows),
                "summary_rows": len(summary_rows),
                "generated_at": _utc_now(),
            },
            ensure_ascii=False,
            indent=2,
        ))


if __name__ == "__main__":
    main()
