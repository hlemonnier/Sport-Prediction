#!/usr/bin/env python3
"""Settle hash-chained F1 forward-test betting logs with explicit result data."""

from __future__ import annotations
import repo_bootstrap  # noqa: F401

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from packages.f1.betting import load_settlement_frame, settle_forward_bet_log


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute realized F1 paper-bet P&L only from hash-valid forward logs "
            "and explicit settlement rows."
        ),
    )
    parser.add_argument("--log-path", default="artifacts/predictions/f1/forward_test/f1_forward_bet_log.jsonl")
    parser.add_argument("--settlements", required=True, help="CSV/JSON result rows with event_id, market, selection, and won/result/finish_position.")
    parser.add_argument("--settled-at-utc", default=None)
    parser.add_argument("--allow-after-close-records", action="store_true")
    parser.add_argument("--skip-hash-verification", action="store_true")
    parser.add_argument("--output-path", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    settlements = load_settlement_frame(args.settlements)
    report = settle_forward_bet_log(
        log_path=args.log_path,
        settlements=settlements,
        settlement_source_path=args.settlements,
        settled_at_utc=args.settled_at_utc or _utc_now(),
        require_pre_market=not bool(args.allow_after_close_records),
        verify_hash_chain=not bool(args.skip_hash_verification),
    )

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
