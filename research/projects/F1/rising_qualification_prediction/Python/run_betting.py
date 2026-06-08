#!/usr/bin/env python3
"""Build F1 betting recommendations from model predictions and market odds."""

from __future__ import annotations
import repo_bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from packages.f1.betting import (
    BettingConfig,
    build_betting_recommendations,
    build_betting_report,
    load_odds_frame,
    load_prediction_frame,
)


def _pct(value: float) -> float:
    return float(value) / 100.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="F1 model-driven betting recommendation engine.")
    parser.add_argument("--predictions", required=True, help="Prediction JSON from run_experiment.py.")
    parser.add_argument("--odds", required=True, help="CSV or JSON odds file with market, driver_name, decimal_odds.")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--fractional-kelly", type=float, default=0.25)
    parser.add_argument("--min-edge-pct", type=float, default=3.0)
    parser.add_argument("--min-expected-roi-pct", type=float, default=2.0)
    parser.add_argument("--min-probability-pct", type=float, default=2.0)
    parser.add_argument("--max-bet-pct", type=float, default=1.0)
    parser.add_argument("--max-market-pct", type=float, default=3.0)
    parser.add_argument("--max-total-pct", type=float, default=5.0)
    parser.add_argument("--min-stake", type=float, default=0.0)
    parser.add_argument("--fair-market-min-selections", type=int, default=10)
    parser.add_argument("--fair-market-overround-min", type=float, default=0.90)
    parser.add_argument("--fair-market-overround-max", type=float, default=1.35)
    parser.add_argument(
        "--allow-uncalibrated-research-bets",
        action="store_true",
        help="Disable probability invariant gating; use only for research diagnostics, not user-facing betting.",
    )
    parser.add_argument("--output-format", choices=["text", "json", "csv"], default="text")
    parser.add_argument("--output-path", default=None)
    return parser


def _print_text(report: dict[str, object]) -> None:
    summary = report.get("summary", {})
    print("=" * 72)
    print("F1 paper betting candidates")
    print("=" * 72)
    readiness = report.get("readiness_status")
    if readiness:
        print(f"Readiness: {readiness}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    rows = report.get("recommendations", [])
    if not isinstance(rows, list) or not rows:
        print("\nNo odds rows available.")
        return
    bets = [row for row in rows if isinstance(row, dict) and row.get("status") == "bet"]
    if not bets:
        print("\nNo bets passed the edge, Kelly, and exposure filters.")
        return
    print("\nPaper candidates:")
    for row in bets:
        print(
            "- "
            f"{row.get('market')} | {row.get('driver_name')} @ {row.get('decimal_odds')} "
            f"| p={float(row.get('model_probability', 0.0)):.3f} "
            f"| edge={float(row.get('probability_edge', 0.0)):.3f} "
            f"| stake={float(row.get('stake', 0.0)):.2f}"
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = BettingConfig(
        bankroll=float(args.bankroll),
        fractional_kelly=float(args.fractional_kelly),
        min_edge=_pct(args.min_edge_pct),
        min_expected_roi=_pct(args.min_expected_roi_pct),
        min_probability=_pct(args.min_probability_pct),
        max_bet_fraction=_pct(args.max_bet_pct),
        max_market_fraction=_pct(args.max_market_pct),
        max_total_fraction=_pct(args.max_total_pct),
        min_stake=float(args.min_stake),
        require_probability_gate=not bool(args.allow_uncalibrated_research_bets),
        require_oof_probability_audit=not bool(args.allow_uncalibrated_research_bets),
        fair_market_min_selection_count=int(args.fair_market_min_selections),
        fair_market_overround_min=float(args.fair_market_overround_min),
        fair_market_overround_max=float(args.fair_market_overround_max),
    )
    predictions = load_prediction_frame(args.predictions)
    odds = load_odds_frame(args.odds)
    recommendations = build_betting_recommendations(predictions, odds, config)
    report = build_betting_report(recommendations, config)
    report["inputs"] = {
        "predictions": str(Path(args.predictions)),
        "odds": str(Path(args.odds)),
    }

    if args.output_format == "csv":
        if args.output_path is None:
            print(recommendations.to_csv(index=False))
        else:
            Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
            recommendations.to_csv(args.output_path, index=False)
        return

    if args.output_format == "json":
        text = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output_path is None:
            print(text)
        else:
            Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_path).write_text(text, encoding="utf-8")
        return

    if args.output_path is not None:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_text(report)


if __name__ == "__main__":
    main()
