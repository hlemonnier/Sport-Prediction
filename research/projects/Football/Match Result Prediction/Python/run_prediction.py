#!/usr/bin/env python3
"""Entry point for football match result prediction (stub)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone

from mrp import PredictionConfig, run_prediction


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Match Result Prediction (football)")
    parser.add_argument("--mode", choices=["match_result", "scoreline"], required=True)
    parser.add_argument("--league", required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", dest="round_number", type=int, required=True)
    parser.add_argument("--data-source", default="placeholder")
    parser.add_argument("--train-seasons", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-format", choices=["text", "json"], default="text")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    config = PredictionConfig(
        league=args.league,
        season=args.season,
        round_number=args.round_number,
        mode=args.mode,
        data_source=args.data_source,
        train_seasons=parse_train_seasons(args.train_seasons, args.season),
        cache_dir=args.cache_dir,
    )

    result = run_prediction(config)

    if args.output_format == "json":
        payload = {
            "version": result.version,
            "sport": "Football",
            "project": "Match Result Prediction",
            "config": asdict(config),
            "rows": result.rows,
            "notes": result.notes,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
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
    print(f"Model version: {result.version}")
    print("=" * 72)
    if not result.rows:
        print("Aucune prediction disponible.")
    else:
        print(format_rows(result.rows))
    if result.notes:
        print("\nNotes:")
        for note in result.notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()
