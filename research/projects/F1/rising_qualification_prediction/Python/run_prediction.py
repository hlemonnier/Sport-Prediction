#!/usr/bin/env python3
"""Entry point for race/qualifying prediction."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone

from rqp import PredictionConfig, run_prediction
from rqp.runtime import parse_compare_families, parse_json_object, parse_train_seasons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rising Qualification Prediction (FastF1 / OpenF1 / local offline)"
    )
    parser.add_argument("--mode", choices=["qualifying", "race"], required=True)
    parser.add_argument("--source", choices=["fastf1", "openf1", "local"], required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", dest="round_number", type=int, required=True)
    parser.add_argument("--train-seasons", default="auto")
    parser.add_argument(
        "--train-policy",
        choices=["strict_transfer", "rolling", "frozen_preseason", "legacy_auto"],
        default="legacy_auto",
        help="Policy used only when --train-seasons=auto.",
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
    parser.add_argument("--output-format", choices=["text", "json"], default="text")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    dl_hyperparams = parse_json_object(args.dl_hyperparams, "--dl-hyperparams")

    config = PredictionConfig(
        source=args.source,
        mode=args.mode,
        year=args.year,
        round_number=args.round_number,
        train_seasons=parse_train_seasons(args.train_seasons, args.year, args.train_policy),
        include_standings=args.include_standings,
        cache_dir=args.cache_dir,
        meeting_name=args.meeting_name,
        country_name=args.country_name,
        weekends_dir=args.weekends_dir,
        enable_dl_candidates=args.enable_dl_candidates,
        compare_families=parse_compare_families(args.compare_families),
        dl_device=args.dl_device,
        dl_arch=args.dl_arch,
        dl_hyperparams=dl_hyperparams,
        dl_seed=args.dl_seed,
        disable_runsim_features=args.disable_runsim_features,
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
            "model_name": result.model_name,
            "model_family": result.model_family,
            "device_used": result.device_used,
            "dl_available": result.dl_available,
            "candidate_leaderboard": result.candidate_leaderboard,
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
    print(f"Model selected: {result.model_name} [{result.model_family}]")
    if result.device_used:
        print(f"Device: {result.device_used}")
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
