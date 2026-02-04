#!/usr/bin/env python3
"""Entry point for race/qualifying prediction."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone

from rqp import PredictionConfig, run_prediction


def parse_train_seasons(value: str, target_year: int) -> list[int]:
    if value.lower() in {"auto", "default"}:
        return [target_year - 2, target_year - 1, target_year]
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rising Qualification Prediction (FastF1 / OpenF1)"
    )
    parser.add_argument("--mode", choices=["qualifying", "race"], required=True)
    parser.add_argument("--source", choices=["fastf1", "openf1"], required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", dest="round_number", type=int, required=True)
    parser.add_argument("--train-seasons", default="auto")
    parser.add_argument("--include-standings", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--meeting-name", default=None)
    parser.add_argument("--country-name", default=None)
    parser.add_argument("--output-format", choices=["text", "json"], default="text")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    config = PredictionConfig(
        source=args.source,
        mode=args.mode,
        year=args.year,
        round_number=args.round_number,
        train_seasons=parse_train_seasons(args.train_seasons, args.year),
        include_standings=args.include_standings,
        cache_dir=args.cache_dir,
        meeting_name=args.meeting_name,
        country_name=args.country_name,
    )

    result = run_prediction(config)

    if args.output_format == "json":
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
        f"Mode: {config.mode} | Source: {config.source} | Year: {config.year} | Round: {config.round_number}"
    )
    print(f"Model version: {result.version}")
    print("=" * 72)
    if result.table.empty:
        print("Aucune prediction disponible.")
    else:
        print(result.table.to_string(index=False))
    if result.notes:
        print("\nNotes:")
        for note in result.notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()
