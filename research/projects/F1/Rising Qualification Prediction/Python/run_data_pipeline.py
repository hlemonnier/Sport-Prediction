#!/usr/bin/env python3
"""Build historical F1 datasets from FastF1/OpenF1."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from rqp.pipeline import PipelineConfig, run_pipeline


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_years(value: str) -> list[int]:
    years = []
    for item in parse_csv_list(value):
        years.append(int(item))
    return sorted(set(years))


def default_output_dir() -> str:
    project_root = Path(__file__).resolve().parents[5]
    return str(project_root / "data" / "f1")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="F1 data pipeline (OpenF1 + FastF1)",
    )
    parser.add_argument("--sources", default="fastf1,openf1")
    parser.add_argument("--years", required=True, help="Ex: 2023,2024,2025")
    parser.add_argument("--output-dir", default=default_output_dir())
    parser.add_argument("--cache-dir", default=".cache/f1")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--output-format", choices=["text", "json"], default="text")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = PipelineConfig(
        sources=parse_csv_list(args.sources),
        years=parse_years(args.years),
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        max_rounds=args.max_rounds,
    )
    result = run_pipeline(config)

    payload = {
        "sport": "F1",
        "project": "Rising Qualification Prediction",
        "config": {
            "sources": config.sources,
            "years": config.years,
            "output_dir": config.output_dir,
            "cache_dir": config.cache_dir,
            "max_rounds": config.max_rounds,
        },
        "rows": int(len(result.dataset)),
        "events": int(result.coverage.shape[0]),
        "dataset_csv_path": result.dataset_csv_path,
        "dataset_parquet_path": result.dataset_parquet_path,
        "coverage_csv_path": result.coverage_csv_path,
        "coverage_parquet_path": result.coverage_parquet_path,
        "notes": result.notes,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    if args.quiet:
        return

    if args.output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("=" * 72)
    print("F1 data pipeline")
    print("=" * 72)
    print(f"Sources: {', '.join(config.sources)}")
    print(f"Years: {', '.join(str(y) for y in config.years)}")
    print(f"Events ingested: {payload['events']}")
    print(f"Rows written: {payload['rows']}")
    print(f"Dataset CSV: {result.dataset_csv_path}")
    if result.dataset_parquet_path:
        print(f"Dataset Parquet: {result.dataset_parquet_path}")
    print(f"Coverage CSV: {result.coverage_csv_path}")
    if result.coverage_parquet_path:
        print(f"Coverage Parquet: {result.coverage_parquet_path}")
    if result.notes:
        print("\nNotes:")
        for note in result.notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()
