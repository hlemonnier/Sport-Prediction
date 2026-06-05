#!/usr/bin/env python3
"""Append hash-chained F1 forward-test betting records before market close."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from rqp.betting import (
    BettingConfig,
    build_betting_recommendations,
    build_betting_report,
    load_odds_frame,
    load_prediction_frame,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _record_hash(record: dict[str, Any]) -> str:
    payload = {k: v for k, v in record.items() if k != "record_hash"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _previous_hash(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    previous = ""
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                previous = str(payload.get("record_hash") or previous)
    return previous


def _pct(value: float) -> float:
    return float(value) / 100.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Log an immutable F1 forward-test record with predictions, odds, stake recommendations, and hash chain.",
    )
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--information-cutoff", required=True, choices=["pre-practice", "post-practice", "post-qualifying", "live"])
    parser.add_argument("--market-close-utc", required=True, help="ISO UTC timestamp. Logging after this fails unless --allow-after-close is set.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--odds", required=True)
    parser.add_argument("--log-path", default="outputs/f1/forward_test/f1_forward_bet_log.jsonl")
    parser.add_argument("--selection-logged-at-utc", default=None)
    parser.add_argument("--allow-after-close", action="store_true")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--fractional-kelly", type=float, default=0.25)
    parser.add_argument("--min-edge-pct", type=float, default=3.0)
    parser.add_argument("--min-expected-roi-pct", type=float, default=2.0)
    parser.add_argument("--min-probability-pct", type=float, default=2.0)
    parser.add_argument("--max-bet-pct", type=float, default=1.0)
    parser.add_argument("--max-market-pct", type=float, default=3.0)
    parser.add_argument("--max-total-pct", type=float, default=5.0)
    parser.add_argument("--min-stake", type=float, default=0.0)
    parser.add_argument("--output-path", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    logged_at = _parse_utc(args.selection_logged_at_utc or _utc_now())
    market_close = _parse_utc(args.market_close_utc)
    if logged_at > market_close and not args.allow_after_close:
        raise SystemExit(
            "Refusing to append forward-test record after market close. "
            "Use --allow-after-close only for dry-run/backfill records, not evidence logs."
        )

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
    )
    predictions = load_prediction_frame(args.predictions)
    odds = load_odds_frame(args.odds)
    recommendations = build_betting_recommendations(predictions, odds, config)
    report = build_betting_report(recommendations, config)

    log_path = Path(args.log_path)
    previous_hash = _previous_hash(log_path)
    record = {
        "schema_version": "f1_forward_bet_log_v1",
        "event_id": args.event_id,
        "information_cutoff": args.information_cutoff,
        "selection_logged_at_utc": logged_at.isoformat().replace("+00:00", "Z"),
        "market_close_utc": market_close.isoformat().replace("+00:00", "Z"),
        "pre_market_logged": logged_at <= market_close,
        "predictions_path": str(Path(args.predictions)),
        "predictions_sha256": _sha256_file(args.predictions),
        "odds_path": str(Path(args.odds)),
        "odds_sha256": _sha256_file(args.odds),
        "previous_record_hash": previous_hash,
        "betting_report": report,
    }
    record["record_hash"] = _record_hash(record)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"log_path": str(log_path), "record_hash": record["record_hash"]}, indent=2))


if __name__ == "__main__":
    main()
