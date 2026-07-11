#!/usr/bin/env python3
"""Paired full-field circuit-card ablation for local F1 prediction artifacts."""

from __future__ import annotations
import repo_bootstrap  # noqa: F401

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from packages.f1 import PredictionConfig, run_prediction
from packages.f1.orchestration.backtest import evaluate_prediction_rows
from packages.f1.data.providers import LocalWeekendProvider
from packages.f1.orchestration.runtime import parse_train_seasons


def _records(result) -> list[dict[str, Any]]:
    rows = result.extras.get("all_prediction_rows") if isinstance(result.extras, dict) else None
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    if result.table.empty:
        return []
    return json.loads(result.table.to_json(orient="records"))


def _actual(provider: LocalWeekendProvider, mode: str, year: int, round_number: int) -> pd.DataFrame:
    if mode == "race":
        return provider.get_race_results(year, round_number)
    return provider.get_qualifying_results(year, round_number)


def _metric_delta(with_cards: dict[str, Any], without_cards: dict[str, Any], key: str) -> Optional[float]:
    left = with_cards.get(key)
    right = without_cards.get(key)
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _bootstrap_ci(values: list[float], *, samples: int, seed: int) -> dict[str, Any]:
    clean = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    if clean.size == 0:
        return {"available": False, "reason": "no_values"}
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(clean, size=(int(max(1, samples)), clean.size), replace=True).mean(axis=1)
    return {
        "available": True,
        "mean": float(clean.mean()),
        "median": float(np.median(clean)),
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
        "n_events": int(clean.size),
        "bootstrap_samples": int(max(1, samples)),
    }


def _two_sided_sign_p_value(positive: int, negative: int) -> Optional[float]:
    trials = int(positive) + int(negative)
    if trials <= 0:
        return None
    observed = min(int(positive), int(negative))
    tail = sum(math.comb(trials, k) for k in range(0, observed + 1)) / float(2**trials)
    return float(min(1.0, 2.0 * tail))


def _sign_summary(values: list[float], *, improvement_when: str) -> dict[str, Any]:
    clean = [float(v) for v in values if np.isfinite(float(v)) and abs(float(v)) > 1e-12]
    if improvement_when == "negative":
        improved = sum(1 for v in clean if v < 0.0)
        degraded = sum(1 for v in clean if v > 0.0)
    else:
        improved = sum(1 for v in clean if v > 0.0)
        degraded = sum(1 for v in clean if v < 0.0)
    return {
        "available": bool(clean),
        "improved_events": int(improved),
        "degraded_events": int(degraded),
        "ties": int(len(values) - len(clean)),
        "two_sided_p_value": _two_sided_sign_p_value(improved, degraded),
    }


def _decision_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    if not bool(summary.get("available", False)):
        return {"state": "insufficient_evidence", "reason": summary.get("reason", "summary_unavailable")}
    mae_delta = summary.get("paired_deltas", {}).get("mae_with_minus_without", {})
    top10_delta = summary.get("paired_deltas", {}).get("top10_with_minus_without", {})
    mae_mean = mae_delta.get("mean")
    mae_low = mae_delta.get("ci95_low")
    mae_high = mae_delta.get("ci95_high")
    top10_mean = top10_delta.get("mean")
    near_zero_mae = (
        isinstance(mae_mean, (int, float))
        and isinstance(mae_low, (int, float))
        and isinstance(mae_high, (int, float))
        and -0.05 <= float(mae_mean) <= 0.05
        and float(mae_low) <= 0.0 <= float(mae_high)
    )
    no_top10_gain = isinstance(top10_mean, (int, float)) and float(top10_mean) <= 0.0
    if near_zero_mae and no_top10_gain:
        return {
            "state": "quarantine",
            "reason": "paired_fullfield_ablation_zero_or_uncertain_effect",
            "action": "Do not market circuit cards as proven alpha until auto-model or stricter holdout ablation clears the kill criteria.",
        }
    if isinstance(mae_high, (int, float)) and float(mae_high) < -0.05:
        return {
            "state": "keep",
            "reason": "paired_mae_ci_supports_card_improvement",
        }
    return {
        "state": "research_only",
        "reason": "effect_not_strong_enough_for_user_facing_claims",
    }


def _summarize(rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    if not rows:
        return {"events": 0, "available": False, "reason": "no_completed_events"}
    frame = pd.DataFrame(rows)
    out: dict[str, Any] = {
        "available": True,
        "events": int(len(frame)),
        "with_cards": {
            "mae_avg": float(pd.to_numeric(frame["mae_with_cards"], errors="coerce").mean()),
            "top10_hit_avg": float(pd.to_numeric(frame["top10_with_cards"], errors="coerce").mean()),
        },
        "without_cards": {
            "mae_avg": float(pd.to_numeric(frame["mae_without_cards"], errors="coerce").mean()),
            "top10_hit_avg": float(pd.to_numeric(frame["top10_without_cards"], errors="coerce").mean()),
        },
    }
    mae_delta = pd.to_numeric(frame["mae_delta_with_minus_without"], errors="coerce").dropna().tolist()
    top10_delta = pd.to_numeric(frame["top10_delta_with_minus_without"], errors="coerce").dropna().tolist()
    out["paired_deltas"] = {
        "mae_with_minus_without": _bootstrap_ci(mae_delta, samples=bootstrap_samples, seed=seed),
        "top10_with_minus_without": _bootstrap_ci(top10_delta, samples=bootstrap_samples, seed=seed + 17),
    }
    out["sign_tests"] = {
        "mae": _sign_summary(mae_delta, improvement_when="negative"),
        "top10": _sign_summary(top10_delta, improvement_when="positive"),
    }
    out["circuit_card_decision"] = _decision_from_summary(out)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paired circuit-card ablation with bootstrap CIs.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--round-start", type=int, default=6)
    parser.add_argument("--round-end", type=int, default=24)
    parser.add_argument("--modes", default="qualifying,race")
    parser.add_argument("--source", choices=["local"], default="local")
    parser.add_argument("--train-seasons", default="auto")
    parser.add_argument(
        "--train-policy",
        choices=["same_season", "same_season_walk_forward", "strict_transfer", "rolling", "frozen_preseason", "legacy_auto"],
        default="legacy_auto",
    )
    parser.add_argument("--weekends-dir", default="data/f1/raw/weekends")
    parser.add_argument(
        "--f1-model",
        choices=["auto", "baseline", "strategic_baseline", "xgb_rank", "eb_rank", "lgbm_rank"],
        default="auto",
    )
    parser.add_argument("--f1-pl-samples", type=int, default=300)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.round_end < args.round_start:
        raise SystemExit("--round-end must be >= --round-start")

    provider = LocalWeekendProvider(weekends_dir=args.weekends_dir)
    modes = [part.strip().lower() for part in str(args.modes).split(",") if part.strip()]
    train_seasons = parse_train_seasons(args.train_seasons, args.year, args.train_policy)
    event_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for mode in modes:
        if mode not in {"qualifying", "race"}:
            skipped.append({"mode": mode, "reason": "unsupported_mode"})
            continue
        for rnd in range(int(args.round_start), int(args.round_end) + 1):
            base_config = dict(
                source="local",
                mode=mode,
                year=int(args.year),
                round_number=int(rnd),
                train_seasons=train_seasons,
                include_standings=False,
                cache_dir=None,
                meeting_name=None,
                country_name=None,
                weekends_dir=args.weekends_dir,
                f1_model=str(args.f1_model),
                f1_pl_samples=int(args.f1_pl_samples),
                shadow_eval=True,
            )
            with_config = PredictionConfig(**base_config, disable_circuit_features=False)
            without_config = PredictionConfig(**base_config, disable_circuit_features=True)
            with_result = run_prediction(with_config)
            without_result = run_prediction(without_config)
            actual = _actual(provider, mode, int(args.year), int(rnd))
            with_eval = evaluate_prediction_rows(_records(with_result), actual, "position")
            without_eval = evaluate_prediction_rows(_records(without_result), actual, "position")
            if (
                not bool(with_eval.get("available"))
                or not bool(without_eval.get("available"))
                or not bool(with_eval.get("metric_available", False))
                or not bool(without_eval.get("metric_available", False))
            ):
                skipped.append(
                    {
                        "mode": mode,
                        "round": int(rnd),
                        "with_cards": with_eval,
                        "without_cards": without_eval,
                    }
                )
                continue
            event_rows.append(
                {
                    "mode": mode,
                    "round": int(rnd),
                    "rows_common_with_cards": int(with_eval.get("rows_common", 0)),
                    "rows_common_without_cards": int(without_eval.get("rows_common", 0)),
                    "field_coverage_with_cards": with_eval.get("field_coverage"),
                    "field_coverage_without_cards": without_eval.get("field_coverage"),
                    "mae_on_common_with_cards": with_eval.get("mae_on_common"),
                    "mae_on_common_without_cards": without_eval.get("mae_on_common"),
                    "mae_with_cards": with_eval.get("field_mae"),
                    "mae_without_cards": without_eval.get("field_mae"),
                    "mae_delta_with_minus_without": _metric_delta(with_eval, without_eval, "field_mae"),
                    "top10_with_cards": with_eval.get("top10_hit"),
                    "top10_without_cards": without_eval.get("top10_hit"),
                    "top10_delta_with_minus_without": _metric_delta(with_eval, without_eval, "top10_hit"),
                    "model_with_cards": with_result.model_name,
                    "model_without_cards": without_result.model_name,
                }
            )

    by_mode = {
        mode: _summarize(
            [row for row in event_rows if row["mode"] == mode],
            bootstrap_samples=int(args.bootstrap_samples),
            seed=int(args.seed),
        )
        for mode in modes
    }
    payload = {
        "workflow": "f1_circuit_card_ablation",
        "config": {
            **vars(args),
            "train_seasons_effective": train_seasons,
        },
        "interpretation": {
            "mae_delta_with_minus_without": "negative means circuit cards improved MAE",
            "top10_delta_with_minus_without": "positive means circuit cards improved top10 hit rate",
            "unit": "paired event",
        },
        "summary": _summarize(event_rows, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed)),
        "by_mode": by_mode,
        "events": event_rows,
        "skipped": skipped,
    }
    if args.output_path:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.quiet:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
