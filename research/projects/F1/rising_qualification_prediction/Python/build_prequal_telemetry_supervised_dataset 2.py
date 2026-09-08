#!/usr/bin/env python3
"""Build an immutable driver-event supervised telemetry manifest.

This command only joins already-captured pre-Qualifying tensors to separate
post-Qualifying truth.  It never fetches data and never trains a model.
"""

from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from packages.f1.data.providers.telemetry_cache import sha256_file
from packages.f1.data.providers.telemetry_supervised import (
    build_prequal_telemetry_supervised_manifest,
)
from packages.sports_core.paths import find_repo_root


def write_immutable_supervised_manifest(
    payload: Mapping[str, Any],
    output: Path,
) -> dict[str, object]:
    """Write once with exclusive-create semantics and return file evidence."""

    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "size_bytes": int(destination.stat().st_size),
    }


def _parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Join validated pre-Qualifying telemetry tensors to one legal "
            "Qualifying target per driver-event."
        )
    )
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument(
        "--telemetry-root",
        type=Path,
        default=root / "data/f1/telemetry/pre_qualifying",
    )
    parser.add_argument(
        "--weekends-root",
        type=Path,
        default=root / "data/f1/raw/weekends",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination must not already exist; existing evidence is never overwritten.",
    )
    parser.add_argument(
        "--generated-at",
        default=None,
        help="Optional fixed ISO timestamp for reproducible rehearsal builds.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    root = find_repo_root()
    args = _parser(root).parse_args(argv)
    payload = build_prequal_telemetry_supervised_manifest(
        root=root,
        telemetry_root=args.telemetry_root,
        weekends_root=args.weekends_root,
        year=args.year,
        generated_at=args.generated_at,
    )
    output = args.output or (
        root / f"data/f1/derived/prequal_telemetry_supervised_{int(args.year)}.json"
    )
    evidence = write_immutable_supervised_manifest(payload, output)
    print(
        json.dumps(
            {
                "artifact": evidence,
                "audit": payload["audit"],
                "bag_set_sha256": payload["bag_set_sha256"],
                "feature_input_manifest_sha256": payload[
                    "feature_input_manifest_sha256"
                ],
                "target_input_manifest_sha256": payload[
                    "target_input_manifest_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
