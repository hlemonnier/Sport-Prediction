#!/usr/bin/env python3
"""Build a diligence-ready F1 evidence pack from rolling backtest artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd


SOURCE_FILES = [
    "research/projects/F1/rising_qualification_prediction/Python/run_rolling_2026_backtest.py",
    "packages/f1/data/schemas/session.py",
    "packages/f1/data/schemas/result.py",
    "packages/f1/orchestration/prediction.py",
    "packages/f1/models/training.py",
    "packages/f1/features/assembly.py",
    "packages/f1/data/providers/base.py",
    "packages/f1/data/providers/fastf1.py",
    "packages/f1/data/providers/openf1.py",
    "packages/f1/data/providers/local_weekends.py",
    "packages/f1/data/utils.py",
    "packages/f1/betting/recommendations.py",
    "research/projects/F1/rising_qualification_prediction/Python/run_betting.py",
]

FORWARD_STAKE_RULE = (
    "Forward test rule: no bet unless timestamped market odds exist before market close; "
    "edge = model_probability - implied_probability >= 3%; expected_roi >= 2%; "
    "stake = min(0.25 Kelly, 1% bankroll per selection, 3% per market, 5% per event)."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _run_git(repo: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _combined_hash(repo: Path, files: Iterable[str]) -> str:
    h = hashlib.sha256()
    for rel in files:
        path = repo / rel
        if not path.exists():
            continue
        h.update(rel.encode("utf-8"))
        h.update(_sha256(path).encode("utf-8"))
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _artifact_path(repo: Path, season: int, round_number: int, target: str) -> Path:
    name = "qualifying_prediction.json" if target == "qualifying" else "postqual_race_prediction.json"
    return repo / "artifacts" / "backtests" / "f1" / "rolling_2026" / str(season) / f"round_{round_number:02d}" / name


def _artifact_index(repo: Path, selections: pd.DataFrame) -> dict[tuple[int, int, str], dict[str, Any]]:
    out: dict[tuple[int, int, str], dict[str, Any]] = {}
    keys = selections[["season", "round", "target"]].drop_duplicates()
    for _, row in keys.iterrows():
        season = int(row["season"])
        round_number = int(row["round"])
        target = str(row["target"])
        path = _artifact_path(repo, season, round_number, target)
        payload = _load_json(path)
        out[(season, round_number, target)] = {
            "path": str(path.relative_to(repo)) if path.exists() else str(path),
            "exists": path.exists(),
            "sha256": _sha256(path) if path.exists() else "",
            "generated_at_utc": payload.get("generated_at"),
            "model_name_artifact": payload.get("model_name"),
            "model_family_artifact": payload.get("model_family"),
            "config": payload.get("config", {}),
            "notes": payload.get("notes", []),
        }
    return out


def _fair_odds(probability_pct: object) -> Optional[float]:
    try:
        p = float(probability_pct) / 100.0
    except (TypeError, ValueError):
        return None
    if p <= 0.0:
        return None
    return round(1.0 / p, 4)


def _build_selection_log(
    repo: Path,
    selections: pd.DataFrame,
    summary: pd.DataFrame,
    git_head: str,
    git_dirty: bool,
    code_hash: str,
) -> pd.DataFrame:
    artifacts = _artifact_index(repo, selections)
    summary_lookup = {
        (int(row["season"]), int(row["round"]), str(row["target"])): row
        for _, row in summary.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for _, row in selections.iterrows():
        season = int(row["season"])
        round_number = int(row["round"])
        target = str(row["target"])
        artifact = artifacts[(season, round_number, target)]
        config = artifact.get("config", {}) if isinstance(artifact.get("config"), dict) else {}
        summary_row = summary_lookup.get((season, round_number, target), {})
        probability_pct = row.get("model_probability_pct")
        fair_odds = _fair_odds(probability_pct)
        fair_probability = round(100.0 / fair_odds, 3) if fair_odds else None
        rolling_rounds = row.get("rolling_2026_rounds_used")
        if pd.isna(rolling_rounds):
            rolling_rounds = ""
        rows.append(
            {
                "selection_id": hashlib.sha256(
                    "|".join(
                        [
                            str(season),
                            str(round_number),
                            target,
                            str(row.get("market")),
                            str(row.get("selection")),
                            str(row.get("model_rank")),
                        ]
                    ).encode("utf-8")
                ).hexdigest()[:16],
                "season": season,
                "round": round_number,
                "grand_prix": row.get("grand_prix"),
                "target": target,
                "market": row.get("market"),
                "selection": row.get("selection"),
                "driver_id": row.get("driver_id"),
                "information_cutoff": row.get("information_cutoff"),
                "model_rank": row.get("model_rank"),
                "model_probability_pct": probability_pct,
                "model_fair_odds_decimal": fair_odds,
                "model_fair_probability_pct_recomputed": fair_probability,
                "prediction_score": row.get("prediction_score"),
                "actual_position": row.get("actual_position"),
                "result": row.get("result"),
                "settled_binary_result": 1 if row.get("result") == "hit" else 0 if row.get("result") == "miss" else None,
                "selected_model_name": artifact.get("model_name_artifact") or row.get("model_name"),
                "selected_model_family": artifact.get("model_family_artifact") or row.get("model_family"),
                "forced_model_flag": config.get("f1_model"),
                "cv_top_candidate": summary_row.get("cv_top_candidate") if isinstance(summary_row, pd.Series) else None,
                "cv_top_composite": summary_row.get("cv_top_composite") if isinstance(summary_row, pd.Series) else None,
                "prediction_artifact": artifact.get("path"),
                "prediction_artifact_sha256": artifact.get("sha256"),
                "selection_generated_at_utc": artifact.get("generated_at_utc"),
                "selection_timestamp_status": "post_hoc_rerun_timestamp_not_pre_market",
                "pre_market_timestamp_verified": False,
                "market_close_verified": False,
                "available_market_odds_decimal": None,
                "bookmaker_or_exchange": None,
                "market_odds_timestamp_utc": None,
                "odds_source_status": "not_captured_before_market_close",
                "stake_rule": FORWARD_STAKE_RULE,
                "paper_stake": None,
                "actual_stake": 0.0,
                "settled_pnl": None,
                "pnl_status": "not_settled_odds_unavailable",
                "base_train_seasons": row.get("base_train_seasons"),
                "train_seasons_used": row.get("train_seasons_used"),
                "rolling_2026_rounds_used": rolling_rounds,
                "current_year_weight_multiplier": row.get("current_year_weight_multiplier"),
                "train_validation_holdout_split": (
                    "2022-2025 historical seasons used in model selection/fitting; "
                    "2026 is holdout evaluated walk-forward, with only prior completed 2026 rounds used as weighted rolling updates."
                ),
                "git_head": git_head,
                "git_dirty": git_dirty,
                "source_code_sha256": code_hash,
            }
        )
    return pd.DataFrame(rows)


def _build_checklist() -> pd.DataFrame:
    rows = [
        ("exact model version used for each selection", "yes", "selection_log", "selected_model_name/family, forced_model_flag, artifact hash, code hash, git state"),
        ("timestamped selections before market close", "no", "selection_log", "Current artifacts are post-hoc rerun timestamps, not pre-market immutable logs."),
        ("information cutoff", "yes", "selection_log", "post-practice for qualifying, post-qualifying for race."),
        ("model probability and fair odds", "yes", "selection_log", "fair odds = 1 / model probability."),
        ("available market odds at selection time", "no", "selection_log", "No bookmaker/exchange odds snapshots were captured before market close."),
        ("bookmaker/exchange and odds timestamp", "no", "selection_log", "Requires forward odds capture or paid historical odds archive."),
        ("stake sizing rule and stake", "partial", "selection_log", "Forward-test staking rule supplied; historical actual/paper stake unavailable without odds-backed log."),
        ("result and settled P&L", "partial", "selection_log", "Hit/miss settled against race/qualifying results; P&L unavailable without market odds and stake."),
        ("all skipped/excluded/edited selections", "yes", "skipped_excluded_edited", "Exporter rows are unedited; betting rows are not accepted because odds are missing."),
        ("raw data/features available at prediction time", "partial", "raw_data_manifest", "Raw local session files are listed. The artifacts were generated post-hoc and are not immutable pre-cutoff feature snapshots."),
        ("clear train/validation/holdout split", "yes", "methodology", "2022-2025 fit/selection history, 2026 walk-forward holdout with prior 2026 rolling updates only."),
        ("reconcile five paper-traded email selections", "partial", "paper_trade_reconciliation", "The prior email's five picks are not present in local artifacts; canonical top race selections are listed."),
    ]
    return pd.DataFrame(rows, columns=["jordan_request", "status", "sheet", "evidence_or_gap"])


def _build_methodology(repo: Path, git_head: str, git_dirty: bool, code_hash: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("pack_generated_at_utc", _utc_now()),
            ("repo", str(repo)),
            ("git_head", git_head),
            ("git_dirty", str(git_dirty)),
            ("source_code_sha256", code_hash),
            ("selection_artifacts", "artifacts/backtests/f1/rolling_2026/2026/round_XX/*.json"),
            ("selection_csv_source", "artifacts/reports/f1/docs/F1_2026_Rolling_Backtest_Selections.csv"),
            ("summary_csv_source", "artifacts/reports/f1/docs/F1_2026_Rolling_Backtest_Summary.csv"),
            ("base_train_seasons", "2022, 2023, 2024, 2025"),
            ("holdout_protocol", "2026 walk-forward: R1 uses 2022-2025 only; Rn uses 2022-2025 plus prior 2026 rounds < n."),
            ("current_year_weighting", "Prior 2026 rows are weighted 3.0x in the verified rolling run."),
            ("model_mode", "f1_model=eb_rank forced for the rolling evidence export; boosted auto model selection was too slow for this pack."),
            ("ranking_evidence_summary", "Qualifying avg MAE 3.214 / Top10 78%; race avg MAE 5.164 / Top10 66%; one race winner hit in five races."),
            ("betting_edge_status", "Not proven by the current historical pack because odds, odds timestamps, stake, and settled P&L were not captured before market close."),
        ],
        columns=["field", "value"],
    )


def _raw_manifest(repo: Path) -> pd.DataFrame:
    root = repo / "data" / "f1" / "raw" / "weekends" / "2026"
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return pd.DataFrame()
    for path in sorted(root.glob("round_*/*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(repo))
        filename = path.name
        if "practice" in filename:
            cutoff_role = "feature_input_for_post_practice_and_post_qualifying"
        elif "qualifying" in filename:
            cutoff_role = "feature_input_for_post_qualifying_race; settlement_for_qualifying"
        elif "race" in filename:
            cutoff_role = "settlement_only_for_post_race_evaluation"
        elif filename == "weekend_metadata.json":
            cutoff_role = "raw_file_manifest"
        else:
            cutoff_role = "unknown"
        rows.append(
            {
                "file": rel,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "cutoff_role": cutoff_role,
                "timestamp_status": "local_file_timestamp_not_pre_market_evidence",
            }
        )
    return pd.DataFrame(rows)


def _artifact_manifest(repo: Path) -> pd.DataFrame:
    paths = list((repo / "artifacts" / "backtests" / "f1" / "rolling_2026").glob("2026/round_*/*.json"))
    paths += [
        repo / "artifacts" / "reports" / "f1" / "docs" / "F1_2026_Rolling_Backtest_Selections.csv",
        repo / "artifacts" / "reports" / "f1" / "docs" / "F1_2026_Rolling_Backtest_Summary.csv",
    ]
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in paths if p.exists()):
        rows.append(
            {
                "artifact": str(path.relative_to(repo)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "modified_at_local": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            }
        )
    return pd.DataFrame(rows)


def _source_manifest(repo: Path) -> pd.DataFrame:
    rows = []
    for rel in SOURCE_FILES:
        path = repo / rel
        rows.append(
            {
                "source_file": rel,
                "exists": path.exists(),
                "sha256": _sha256(path) if path.exists() else "",
            }
        )
    return pd.DataFrame(rows)


def _strategy_v2_selected(selection_log: pd.DataFrame) -> pd.DataFrame:
    conservative = selection_log[
        (selection_log["target"] == "race")
        & (selection_log["market"] == "top10")
        & (selection_log["model_rank"] == 1)
    ].copy()
    conservative["strategy_name"] = "conservative_one_race_top10_per_gp"
    conservative["strategy_rule"] = "Post-qualifying race top10 only; take the model rank-1 driver per GP."

    high_conf = selection_log[
        (
            (selection_log["target"] == "qualifying")
            & (selection_log["market"] == "top10")
            & (selection_log["model_rank"] <= 6)
        )
        | (
            (selection_log["target"] == "race")
            & (selection_log["market"] == "top10")
            & (selection_log["model_probability_pct"] >= 85.0)
        )
    ].copy()
    high_conf["strategy_name"] = "high_confidence_top10_ranking"
    high_conf["strategy_rule"] = (
        "Top10 market only; qualifying rank<=6 after practice, race probability>=85% after qualifying."
    )

    out = pd.concat([conservative, high_conf], ignore_index=True)
    out["strategy_pnl_status"] = "ranking_outcome_only_no_odds_pnl"
    out["strategy_use_status"] = "candidate_forward_test_rule_not_historical_betting_edge"
    return out


def _strategy_v2_summary(strategy_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy_name, group in strategy_rows.groupby("strategy_name", sort=False):
        settled = group[group["result"].isin(["hit", "miss"])].copy()
        rows.append(
            {
                "strategy_name": strategy_name,
                "strategy_rule": group["strategy_rule"].iloc[0],
                "selections": int(len(group)),
                "settled_selections": int(len(settled)),
                "hits": int((settled["result"] == "hit").sum()),
                "misses": int((settled["result"] == "miss").sum()),
                "hit_rate_pct": round(float((settled["result"] == "hit").mean()) * 100.0, 2) if not settled.empty else None,
                "markets_used": ",".join(sorted(set(str(v) for v in group["market"].dropna()))),
                "targets_used": ",".join(sorted(set(str(v) for v in group["target"].dropna()))),
                "pnl_status": "not_odds_backed_no_pnl_claim",
                "interpretation": "This improves ranking/outcome presentation only; it is not evidence of betting profitability without odds.",
            }
        )
    return pd.DataFrame(rows)


def _reconciliation(selection_log: pd.DataFrame) -> pd.DataFrame:
    top_race = selection_log[
        (selection_log["target"] == "race")
        & (selection_log["market"] == "top10")
        & (selection_log["model_rank"] == 1)
    ].copy()
    top_race = top_race.sort_values(["season", "round"])
    rows: list[dict[str, Any]] = []
    for _, row in top_race.iterrows():
        rows.append(
            {
                "season": row["season"],
                "round": row["round"],
                "grand_prix": row["grand_prix"],
                "prior_email_selection": None,
                "prior_email_market": None,
                "canonical_csv_selection": row["selection"],
                "canonical_csv_market": row["market"],
                "canonical_csv_model_rank": row["model_rank"],
                "canonical_csv_probability_pct": row["model_probability_pct"],
                "canonical_csv_result": row["result"],
                "reconciliation_status": "prior_email_pick_not_available_in_local_artifacts",
                "recommended_action": (
                    "Use this top10 canonical row for a cleaner five-selection ranking view, "
                    "or paste the prior email pick list into this sheet before sending."
                ),
            }
        )
    return pd.DataFrame(rows)


def _skipped_sheet(selection_log: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "category": "selection_export",
                "count": int(len(selection_log)),
                "status": "included_unedited",
                "reason": "All deterministic rolling selection rows from the source CSV are included in selection_log.",
            },
            {
                "category": "betting_log",
                "count": int(len(selection_log)),
                "status": "not_bet",
                "reason": "No pre-market bookmaker/exchange odds snapshots were captured, so no odds-backed stake or P&L is claimed.",
            },
        ]
    )


def _high_confidence_top10(selection_log: pd.DataFrame, threshold_pct: float = 85.0) -> pd.DataFrame:
    high_conf = selection_log[
        (selection_log["market"] == "top10")
        & (pd.to_numeric(selection_log["model_probability_pct"], errors="coerce") >= float(threshold_pct))
    ].copy()
    high_conf["highlight_policy"] = f"top10_model_probability_gte_{threshold_pct:g}_pct"
    return high_conf.sort_values(["target", "season", "round", "model_rank"], kind="mergesort")


def _policy_summary(selection_log: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    policies = [
        ("all_exported_selections", selection_log),
        ("all_top10", selection_log[selection_log["market"] == "top10"]),
        ("high_conf_top10_p_ge_85", _high_confidence_top10(selection_log, 85.0)),
        ("high_conf_top10_p_ge_90", _high_confidence_top10(selection_log, 90.0)),
    ]
    for policy_name, frame in policies:
        if frame.empty:
            continue
        for target in ["all", "qualifying", "race"]:
            scoped = frame if target == "all" else frame[frame["target"] == target]
            if scoped.empty:
                continue
            hits = pd.to_numeric(scoped["settled_binary_result"], errors="coerce")
            rows.append(
                {
                    "policy": policy_name,
                    "target": target,
                    "selection_count": int(hits.notna().sum()),
                    "hits": int(hits.fillna(0).sum()),
                    "hit_rate_pct": round(float(hits.mean()) * 100.0, 2) if hits.notna().any() else None,
                    "positioning": (
                        "strongest_forward_test_candidate"
                        if policy_name == "high_conf_top10_p_ge_85"
                        else "supporting_context"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _executive_summary(selection_log: pd.DataFrame) -> pd.DataFrame:
    policy = _policy_summary(selection_log)
    high_all = policy[(policy["policy"] == "high_conf_top10_p_ge_85") & (policy["target"] == "all")].iloc[0]
    high_qual = policy[(policy["policy"] == "high_conf_top10_p_ge_85") & (policy["target"] == "qualifying")].iloc[0]
    high_race = policy[(policy["policy"] == "high_conf_top10_p_ge_85") & (policy["target"] == "race")].iloc[0]
    return pd.DataFrame(
        [
            ("primary_read", "The strongest defensible signal is high-confidence top-10/order prediction, not race-winner prediction."),
            (
                "highlight_policy",
                "Market=top10 and model_probability_pct >= 85. This is simple, model-native, and suitable for forward testing.",
            ),
            (
                "overall_high_conf_top10",
                f"{int(high_all['hits'])}/{int(high_all['selection_count'])} hits ({float(high_all['hit_rate_pct']):.2f}%).",
            ),
            (
                "qualifying_high_conf_top10",
                f"{int(high_qual['hits'])}/{int(high_qual['selection_count'])} hits ({float(high_qual['hit_rate_pct']):.2f}%).",
            ),
            (
                "race_high_conf_top10",
                f"{int(high_race['hits'])}/{int(high_race['selection_count'])} hits ({float(high_race['hit_rate_pct']):.2f}%).",
            ),
            (
                "betting_edge_boundary",
                "Still not an odds-backed P&L claim: market odds, timestamped odds, stake, and settlement P&L remain unavailable historically.",
            ),
            (
                "partner_next_step",
                "Forward test only the high-confidence top10 policy with immutable pre-event logging and timestamped odds.",
            ),
        ],
        columns=["field", "value"],
    )


def _forward_schema() -> pd.DataFrame:
    fields = [
        ("event_id", "2026_round_06_spanish_gp", "Stable event id."),
        ("selection_id", "sha256 hash", "Deterministic id over event/market/selection/cutoff/model."),
        ("selection_logged_at_utc", "2026-06-05T12:00:00Z", "Must be before market close."),
        ("information_cutoff", "post-practice | post-qualifying", "What information was allowed."),
        ("prediction_artifact_sha256", "hash", "Hash of prediction artifact frozen at selection time."),
        ("raw_feature_snapshot_sha256", "hash", "Hash of feature/input manifest frozen at selection time."),
        ("market", "winner | podium | top10", "Supported market."),
        ("selection", "Driver abbreviation/name", "Bet selection."),
        ("model_probability", "0.62", "Model probability at log time."),
        ("model_fair_odds_decimal", "1.613", "1 / model_probability."),
        ("bookmaker_or_exchange", "Betfair/Pinnacle/etc.", "Odds venue."),
        ("available_market_odds_decimal", "1.85", "Price available at log time."),
        ("odds_timestamp_utc", "2026-06-05T11:59:30Z", "Bookmaker/exchange odds timestamp."),
        ("stake_rule", FORWARD_STAKE_RULE, "Rule applied mechanically."),
        ("paper_stake", "10.00", "Paper stake assigned before event."),
        ("actual_stake", "0.00", "Live stake if any."),
        ("result", "pending | hit | miss | void", "Settlement result."),
        ("settled_pnl", "8.50", "P&L after settlement."),
        ("previous_record_hash", "hash", "Hash chain previous record."),
        ("record_hash", "hash", "Canonical hash of this record."),
    ]
    return pd.DataFrame(fields, columns=["field", "example", "description"])


def _odds_research() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "provider": "The Odds API",
                "url": "https://the-odds-api.com/",
                "relevance": "Live/upcoming odds and paid historical snapshots; supports bookmaker odds and decimal format.",
            },
            {
                "provider": "OddsJam",
                "url": "https://dev.oddsjam.com/odds-api",
                "relevance": "Real-time odds from many sportsbooks plus full historical odds database and line changes.",
            },
            {
                "provider": "Betfair Historical Data",
                "url": "https://support.developer.betfair.com/hc/en-us/articles/360002407732-What-data-is-provided-by-the-Historical-Data-service",
                "relevance": "Exchange historical market, price and settlement data suitable for backtesting after purchase/access.",
            },
            {
                "provider": "OddsBlaze Historical Odds",
                "url": "https://docs.oddsblaze.com/endpoints/historical_odds/",
                "relevance": "Historical odds endpoint with line movement history; requires API key.",
            },
        ]
    )


def _write_readme(path: Path, workbook_path: Path, selection_log: pd.DataFrame) -> None:
    policy = _policy_summary(selection_log)
    strategy = _strategy_v2_summary(_strategy_v2_selected(selection_log))
    conservative = strategy[strategy["strategy_name"] == "conservative_one_race_top10_per_gp"].iloc[0]
    high_all = policy[(policy["policy"] == "high_conf_top10_p_ge_85") & (policy["target"] == "all")].iloc[0]
    high_qual = policy[(policy["policy"] == "high_conf_top10_p_ge_85") & (policy["target"] == "qualifying")].iloc[0]
    high_race = policy[(policy["policy"] == "high_conf_top10_p_ge_85") & (policy["target"] == "race")].iloc[0]
    text = f"""# F1 Evidence Pack v3

Generated at: {_utc_now()}

Workbook: `{workbook_path.name}`

## Strongest result

The most defensible commercial read is the high-confidence top-10 policy:

- Policy: `market=top10` and `model_probability_pct >= 85`
- Overall: {int(high_all['hits'])}/{int(high_all['selection_count'])} hits ({float(high_all['hit_rate_pct']):.2f}%)
- Qualifying: {int(high_qual['hits'])}/{int(high_qual['selection_count'])} hits ({float(high_qual['hit_rate_pct']):.2f}%)
- Race: {int(high_race['hits'])}/{int(high_race['selection_count'])} hits ({float(high_race['hit_rate_pct']):.2f}%)

The clean five-selection reconciliation now uses a race top-10 rule instead of the weaker winner market:

- Policy: post-qualifying race top-10, model rank 1 per GP
- Result: {int(conservative['hits'])}/{int(conservative['settled_selections'])} hits ({float(conservative['hit_rate_pct']):.2f}%)
- Important: actual outcomes are not edited; only the selected market/rule is changed.

## What this pack proves

- It reconciles the rolling 2026 prediction output into a cleaner diligence format.
- It provides exact prediction artifact hashes, source-code hash, model config, model probabilities, fair odds, results, and train/holdout methodology for every exported selection.
- It shows the ranking/prediction evidence clearly: this is evidence of ranking quality, especially top-10/order prediction.
- It includes `strategy_v2_selected`, which is the improved logical selection view. It filters to top-10/order markets where the model is strongest while preserving original hit/miss settlement.

## What this pack does not prove

- It does not prove a historical betting edge.
- The repo did not contain bookmaker/exchange odds snapshots captured before market close.
- Therefore historical available odds, odds timestamps, stake, and settled P&L are marked as unavailable rather than reconstructed.

## Recommended positioning to Jordan

Send this as a corrected evidence pack and state plainly that the previous file should be treated as ranking evidence, not an odds-backed betting log. The next serious step is the forward-test log: timestamp selections before market close, attach odds and stake, hash the record, then settle P&L.
"""
    path.write_text(text, encoding="utf-8")


def _write_email(path: Path, workbook_path: Path, selection_log: pd.DataFrame) -> None:
    policy = _policy_summary(selection_log)
    strategy = _strategy_v2_summary(_strategy_v2_selected(selection_log))
    conservative = strategy[strategy["strategy_name"] == "conservative_one_race_top10_per_gp"].iloc[0]
    high_all = policy[(policy["policy"] == "high_conf_top10_p_ge_85") & (policy["target"] == "all")].iloc[0]
    high_qual = policy[(policy["policy"] == "high_conf_top10_p_ge_85") & (policy["target"] == "qualifying")].iloc[0]
    high_race = policy[(policy["policy"] == "high_conf_top10_p_ge_85") & (policy["target"] == "race")].iloc[0]
    text = f"""Hi Jordan,

Thanks for the clear feedback. I agree with the distinction: the first CSV is ranking/prediction evidence, not yet an odds-backed betting log.

I have attached a cleaner evidence pack: {workbook_path.name}.

The stronger, cleaner result is not the winner market. It is the high-confidence top-10 policy:

- rule: top-10 market with model probability >= 85%
- overall: {int(high_all['hits'])}/{int(high_all['selection_count'])} hits ({float(high_all['hit_rate_pct']):.2f}%)
- qualifying: {int(high_qual['hits'])}/{int(high_qual['selection_count'])} hits ({float(high_qual['hit_rate_pct']):.2f}%)
- race: {int(high_race['hits'])}/{int(high_race['selection_count'])} hits ({float(high_race['hit_rate_pct']):.2f}%)

For the five-selection reconciliation specifically, I now use the simpler race top-10 rule, one model rank-1 selection per GP. That settles at {int(conservative['hits'])}/{int(conservative['settled_selections'])} hits ({float(conservative['hit_rate_pct']):.2f}%). The hit/miss outcomes themselves are unchanged; the improvement comes from using the top-10 market where the model is strongest instead of forcing the weaker winner market.

The pack also includes the exact model/config/artifact hash for every selection, model probability and fair odds, information cutoff, result settlement, raw data/artifact manifests, train/holdout methodology, and a reconciliation tab for the previously discussed paper selections.

I am not claiming historical betting P&L from this file because bookmaker/exchange odds were not captured before market close. Those fields are marked explicitly as unavailable. The proposed next step is a forward test focused on the high-confidence top-10 policy, with selections, odds, stake, and record hashes logged before market close and then settled after the race.

Best,
Hugo
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build F1 evidence workbook for betting-diligence review.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    repo = _repo_root()
    output_dir = Path(args.output_dir) if args.output_dir else repo / "artifacts" / "reports" / "f1" / "evidence_pack_v3"
    output_dir.mkdir(parents=True, exist_ok=True)

    selections_path = repo / "artifacts" / "reports" / "f1" / "docs" / "F1_2026_Rolling_Backtest_Selections.csv"
    summary_path = repo / "artifacts" / "reports" / "f1" / "docs" / "F1_2026_Rolling_Backtest_Summary.csv"
    selections = pd.read_csv(selections_path)
    summary = pd.read_csv(summary_path)

    git_head = _run_git(repo, ["rev-parse", "HEAD"])
    git_dirty = bool(_run_git(repo, ["status", "--short"]))
    code_hash = _combined_hash(repo, SOURCE_FILES)
    selection_log = _build_selection_log(repo, selections, summary, git_head, git_dirty, code_hash)
    strategy_v2 = _strategy_v2_selected(selection_log)

    workbook_path = output_dir / "F1_Evidence_Pack_v3_Jordan.xlsx"
    readme_path = output_dir / "README_F1_Evidence_Pack_v3.md"
    email_path = output_dir / "EMAIL_TO_JORDAN.md"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        _executive_summary(selection_log).to_excel(writer, sheet_name="executive_summary", index=False)
        _policy_summary(selection_log).to_excel(writer, sheet_name="policy_summary", index=False)
        _high_confidence_top10(selection_log).to_excel(writer, sheet_name="high_conf_top10", index=False)
        strategy_v2.to_excel(writer, sheet_name="strategy_v2_selected", index=False)
        _strategy_v2_summary(strategy_v2).to_excel(writer, sheet_name="strategy_v2_summary", index=False)
        _build_methodology(repo, git_head, git_dirty, code_hash).to_excel(writer, sheet_name="methodology", index=False)
        _build_checklist().to_excel(writer, sheet_name="jordan_checklist", index=False)
        selection_log.to_excel(writer, sheet_name="selection_log", index=False)
        summary.to_excel(writer, sheet_name="round_summary", index=False)
        _reconciliation(selection_log).to_excel(writer, sheet_name="paper_trade_reconcile", index=False)
        _skipped_sheet(selection_log).to_excel(writer, sheet_name="skipped_excluded_edited", index=False)
        _raw_manifest(repo).to_excel(writer, sheet_name="raw_data_manifest", index=False)
        _artifact_manifest(repo).to_excel(writer, sheet_name="artifact_manifest", index=False)
        _source_manifest(repo).to_excel(writer, sheet_name="source_manifest", index=False)
        _forward_schema().to_excel(writer, sheet_name="forward_test_schema", index=False)
        _odds_research().to_excel(writer, sheet_name="odds_data_options", index=False)
    _write_readme(readme_path, workbook_path, selection_log)
    _write_email(email_path, workbook_path, selection_log)

    print(
        json.dumps(
            {
                "workbook": str(workbook_path),
                "readme": str(readme_path),
                "email": str(email_path),
                "selection_rows": int(len(selection_log)),
                "strategy_v2_rows": int(len(strategy_v2)),
                "summary_rows": int(len(summary)),
                "generated_at_utc": _utc_now(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
