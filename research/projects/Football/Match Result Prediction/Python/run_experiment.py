#!/usr/bin/env python3
"""Canonical experiment runner for football match result prediction."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional, Sequence

from mrp import PredictionConfig, run_prediction

SPORT = "Football"
PROJECT_NAME = "Match Result Prediction"
ENTRYPOINT = "run_experiment.py"
SCHEMA_VERSION = "1.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_train_seasons(value: str, target_season: int) -> list[int]:
    if value.lower() in {"auto", "default"}:
        return [target_season - 2, target_season - 1, target_season]
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def format_rows(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    columns = sorted({key for row in rows for key in row.keys()})
    widths = {
        column: max(len(column), max(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)
    lines = [header, separator]
    for row in rows:
        lines.append(
            " | ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns)
        )
    return "\n".join(lines)


def _detect_primary_model(rows: list[dict[str, str]]) -> Optional[str]:
    if not rows:
        return None
    model_name = rows[0].get("primary_model")
    return model_name if model_name else None


def _build_payload(config: PredictionConfig) -> tuple[dict[str, object], list[dict[str, str]]]:
    result = run_prediction(config)
    model_name = _detect_primary_model(result.rows) or "dixon_coles"
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "version": result.version,
        "sport": SPORT,
        "project": PROJECT_NAME,
        "entrypoint": ENTRYPOINT,
        "workflow": "single_prediction",
        "config": asdict(config),
        "rows": result.rows,
        "notes": result.notes,
        "model_name": model_name,
        "model_family": "ml",
        "device_used": None,
        "dl_available": False,
        "candidate_leaderboard": [],
        "diagnostics": result.diagnostics,
        "generated_at": _utc_now(),
    }
    return payload, result.rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Match Result Prediction (football)")
    parser.add_argument("--mode", choices=["match_result", "scoreline"], required=True)
    parser.add_argument("--league", required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", dest="round_number", type=int, required=True)
    parser.add_argument("--data-source", default="placeholder")
    parser.add_argument("--train-seasons", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--football_model", choices=["dixon", "gbdt", "hybrid"], default="dixon")
    parser.add_argument("--football_calibration", choices=["off", "auto", "platt", "isotonic"], default="auto")
    parser.add_argument("--shadow_eval", choices=["on", "off"], default="on")
    parser.add_argument("--output-format", choices=["text", "json"], default="text")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    config = PredictionConfig(
        league=args.league,
        season=args.season,
        round_number=args.round_number,
        mode=args.mode,
        data_source=args.data_source,
        train_seasons=parse_train_seasons(args.train_seasons, args.season),
        cache_dir=args.cache_dir,
        football_model=args.football_model,
        football_calibration=args.football_calibration,
        shadow_eval=(str(args.shadow_eval).strip().lower() == "on"),
    )

    payload, rows = _build_payload(config)

    if args.output_format == "json":
        if args.output_path:
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        if not args.quiet:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.quiet:
        return

    print("=" * 72)
    print(
        f"Mode: {config.mode} | League: {config.league} | Season: {config.season} | Round: {config.round_number}"
    )
    print(f"Model version: {payload['version']}")
    print("=" * 72)
    if not rows:
        print("Aucune prediction disponible.")
    else:
        print(format_rows(rows))
    notes = payload.get("notes")
    if isinstance(notes, list) and notes:
        print("\nNotes:")
        for note in notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()
