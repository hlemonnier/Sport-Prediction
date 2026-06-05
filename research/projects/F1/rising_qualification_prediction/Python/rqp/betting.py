"""Betting portfolio construction from F1 prediction probabilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Optional

import pandas as pd


MARKET_ALIASES = {
    "win": "winner",
    "winner": "winner",
    "race_winner": "winner",
    "outright": "winner",
    "podium": "podium",
    "top_3": "podium",
    "top3": "podium",
    "top_three": "podium",
    "top_10": "top10",
    "top10": "top10",
    "points": "top10",
    "finish_top10": "top10",
}

MARKET_PROBABILITY_COLUMNS = {
    "winner": ["proba_win", "p_win"],
    "podium": ["proba_top3", "p_top3"],
    "top10": ["proba_top10", "p_top10"],
}


@dataclass
class BettingConfig:
    bankroll: float = 1000.0
    fractional_kelly: float = 0.25
    min_edge: float = 0.03
    min_expected_roi: float = 0.02
    min_probability: float = 0.02
    max_bet_fraction: float = 0.01
    max_market_fraction: float = 0.03
    max_total_fraction: float = 0.05
    min_stake: float = 0.0
    require_probability_gate: bool = True
    require_oof_probability_audit: bool = True
    probability_sum_tolerance: float = 0.10
    fair_market_min_selection_count: int = 10
    fair_market_overround_min: float = 0.90
    fair_market_overround_max: float = 1.35


def _to_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def normalize_market(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return MARKET_ALIASES.get(text, text)


def normalize_participant(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    return " ".join(text.split())


def _records_from_json_payload(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("all_prediction_rows", "rows", "recommendations", "odds"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def load_prediction_frame(path: str | Path) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    frame = pd.DataFrame(_records_from_json_payload(payload))
    if isinstance(payload, dict):
        audit = payload.get("probability_audit")
        if isinstance(audit, dict):
            frame.attrs["probability_audit"] = audit
    return frame


def load_odds_frame(path: str | Path) -> pd.DataFrame:
    odds_path = Path(path)
    if odds_path.suffix.lower() == ".csv":
        return pd.read_csv(odds_path)
    payload = json.loads(odds_path.read_text(encoding="utf-8"))
    return pd.DataFrame(_records_from_json_payload(payload))


def _driver_key_frame(frame: pd.DataFrame) -> pd.Series:
    if "driver_name" in frame.columns:
        return frame["driver_name"].map(normalize_participant)
    if "driver_id" in frame.columns:
        return frame["driver_id"].map(normalize_participant)
    return pd.Series("", index=frame.index, dtype=str)


def _prediction_lookup_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for column in ("driver_name", "driver_id"):
        if column not in predictions.columns:
            continue
        part = predictions.copy()
        part["driver_key"] = part[column].map(normalize_participant)
        part = part[part["driver_key"] != ""]
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=list(predictions.columns) + ["driver_key"])
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=["driver_key"], keep="first")


def _standardize_odds_frame(odds: pd.DataFrame) -> pd.DataFrame:
    out = odds.copy()
    if "decimal_odds" not in out.columns and "odds" in out.columns:
        out = out.rename(columns={"odds": "decimal_odds"})
    if "driver_name" not in out.columns and "participant" in out.columns:
        out = out.rename(columns={"participant": "driver_name"})
    if "driver_name" not in out.columns and "selection" in out.columns:
        out = out.rename(columns={"selection": "driver_name"})
    if "market" not in out.columns:
        out["market"] = "winner"
    return out


def _probability_gate(predictions: pd.DataFrame, config: BettingConfig) -> tuple[bool, str]:
    if predictions.empty:
        return False, "predictions_empty"
    key = _driver_key_frame(predictions)
    frame = predictions[key != ""].copy()
    if frame.empty:
        return False, "prediction_driver_keys_missing"
    frame["_driver_key"] = key.loc[frame.index]
    frame = frame.drop_duplicates(subset=["_driver_key"], keep="first")
    n = int(len(frame))
    if n <= 0:
        return False, "prediction_field_empty"
    tolerance = float(max(config.probability_sum_tolerance, 0.0))
    checks = [
        ("proba_win", min(1.0, float(n))),
        ("proba_top3", min(3.0, float(n))),
        ("proba_top10", min(10.0, float(n))),
    ]
    for column, expected in checks:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        total = float(values.fillna(0.0).sum())
        if abs(total - expected) > tolerance:
            return False, f"{column}_sum_{total:.3f}_expected_{expected:.3f}"
    if {"proba_win", "proba_top3"}.issubset(frame.columns):
        win = pd.to_numeric(frame["proba_win"], errors="coerce")
        top3 = pd.to_numeric(frame["proba_top3"], errors="coerce")
        if ((win > top3 + 1e-9) & win.notna() & top3.notna()).any():
            return False, "win_gt_top3"
    if {"proba_top3", "proba_top10"}.issubset(frame.columns):
        top3 = pd.to_numeric(frame["proba_top3"], errors="coerce")
        top10 = pd.to_numeric(frame["proba_top10"], errors="coerce")
        if ((top3 > top10 + 1e-9) & top3.notna() & top10.notna()).any():
            return False, "top3_gt_top10"
    return True, "passed"


def _probability_audit_gate(predictions: pd.DataFrame, config: BettingConfig) -> tuple[bool, str]:
    if not bool(config.require_oof_probability_audit):
        return True, "disabled"
    audit = predictions.attrs.get("probability_audit") if hasattr(predictions, "attrs") else None
    if not isinstance(audit, dict) or not audit:
        return False, "probability_audit_missing"
    if str(audit.get("source", "")).strip().lower() != "walk_forward_oof":
        return False, f"probability_audit_source_{audit.get('source', 'missing')}"
    if not bool(audit.get("available", False)):
        return False, f"probability_audit_unavailable_{audit.get('reason', 'unknown')}"
    if not bool(audit.get("passed", False)):
        return False, f"probability_audit_failed_{audit.get('reason', 'unknown')}"
    metrics = audit.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        return False, "probability_audit_metrics_missing"
    for market in ("win", "top3", "top10"):
        metric = metrics.get(market)
        if not isinstance(metric, dict) or not bool(metric.get("available", False)):
            return False, f"probability_audit_{market}_missing"
        if "passed" in metric and not bool(metric.get("passed", False)):
            return False, f"probability_audit_{market}_failed"
    return True, "passed"


def _probability_for_market(row: pd.Series, market: str) -> float:
    for column in MARKET_PROBABILITY_COLUMNS.get(market, []):
        if column in row.index:
            value = _to_float(row.get(column))
            if math.isfinite(value):
                return max(0.0, min(1.0, value))
    return float("nan")


def _reject_reason(row: pd.Series, config: BettingConfig) -> str:
    if not bool(row.get("matched_prediction", False)):
        return "no_prediction_match"
    if str(row.get("market", "")) not in MARKET_PROBABILITY_COLUMNS:
        return "unsupported_market"
    if not math.isfinite(_to_float(row.get("decimal_odds"))) or _to_float(row.get("decimal_odds")) <= 1.0:
        return "invalid_decimal_odds"
    if not math.isfinite(_to_float(row.get("model_probability"))):
        return "missing_model_probability"
    if bool(config.require_probability_gate) and not bool(row.get("probability_gate_passed", False)):
        return "probability_gate_failed"
    if bool(config.require_oof_probability_audit) and not bool(row.get("probability_audit_passed", False)):
        return "probability_audit_failed"
    if _to_float(row.get("model_probability")) < config.min_probability:
        return "probability_below_min"
    if _to_float(row.get("edge_used")) < config.min_edge:
        return "edge_below_min"
    if _to_float(row.get("expected_roi")) < config.min_expected_roi:
        return "expected_roi_below_min"
    if _to_float(row.get("kelly_fraction_raw")) <= 0.0:
        return "kelly_non_positive"
    return ""


def build_betting_recommendations(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    config: Optional[BettingConfig] = None,
) -> pd.DataFrame:
    cfg = config or BettingConfig()
    if odds.empty:
        return pd.DataFrame()

    pred = _prediction_lookup_frame(predictions)
    probability_gate_passed, probability_gate_reason = _probability_gate(predictions, cfg)
    probability_audit_passed, probability_audit_reason = _probability_audit_gate(predictions, cfg)

    prices = _standardize_odds_frame(odds)
    prices["market"] = prices["market"].map(normalize_market)
    prices["driver_key"] = _driver_key_frame(prices)
    if "decimal_odds" in prices.columns:
        prices["decimal_odds"] = pd.to_numeric(prices["decimal_odds"], errors="coerce")
    else:
        prices["decimal_odds"] = pd.Series(float("nan"), index=prices.index, dtype=float)
    prices = prices[prices["driver_key"] != ""].copy()

    merged = prices.merge(
        pred,
        on="driver_key",
        how="left",
        suffixes=("", "_prediction"),
    )
    if "rank" in merged.columns:
        merged["matched_prediction"] = merged["rank"].notna()
    elif "driver_name_prediction" in merged.columns:
        merged["matched_prediction"] = merged["driver_name_prediction"].notna()
    else:
        merged["matched_prediction"] = False
    if "driver_name_prediction" in merged.columns:
        merged["driver_name"] = merged["driver_name"].where(
            merged["driver_name"].notna(),
            merged["driver_name_prediction"],
        )

    merged["model_probability"] = merged.apply(
        lambda row: _probability_for_market(row, str(row.get("market", ""))),
        axis=1,
    )
    merged["probability_gate_passed"] = bool(probability_gate_passed)
    merged["probability_gate_reason"] = str(probability_gate_reason)
    merged["probability_audit_passed"] = bool(probability_audit_passed)
    merged["probability_audit_reason"] = str(probability_audit_reason)
    merged["implied_probability_raw"] = 1.0 / merged["decimal_odds"]
    group_cols = ["market"]
    if "bookmaker" in merged.columns:
        group_cols.append("bookmaker")
    merged["market_overround"] = merged.groupby(group_cols)["implied_probability_raw"].transform("sum")
    merged["market_selection_count"] = merged.groupby(group_cols)["implied_probability_raw"].transform("count")
    merged["fair_market_probability"] = merged["implied_probability_raw"] / merged["market_overround"].where(
        merged["market_overround"] > 0.0,
    )
    merged["probability_edge"] = merged["model_probability"] - merged["implied_probability_raw"]
    merged["fair_probability_edge"] = merged["model_probability"] - merged["fair_market_probability"]
    merged["fair_edge_available"] = (
        (merged["market_selection_count"] >= int(max(cfg.fair_market_min_selection_count, 2)))
        & (merged["market_overround"] >= float(max(cfg.fair_market_overround_min, 0.0)))
        & (merged["market_overround"] <= float(max(cfg.fair_market_overround_max, cfg.fair_market_overround_min)))
    )
    merged["edge_used"] = merged["probability_edge"]
    merged.loc[merged["fair_edge_available"], "edge_used"] = merged.loc[
        merged["fair_edge_available"],
        "fair_probability_edge",
    ]
    merged["edge_source"] = merged["fair_edge_available"].map(lambda available: "fair_market" if available else "raw_odds")
    merged["expected_roi"] = (merged["model_probability"] * merged["decimal_odds"]) - 1.0
    merged["kelly_fraction_raw"] = merged["expected_roi"] / (merged["decimal_odds"] - 1.0)
    merged["kelly_fraction_raw"] = merged["kelly_fraction_raw"].clip(lower=0.0)
    merged["target_stake_fraction"] = (
        merged["kelly_fraction_raw"] * float(max(cfg.fractional_kelly, 0.0))
    ).clip(lower=0.0, upper=float(max(cfg.max_bet_fraction, 0.0)))
    merged["target_stake"] = merged["target_stake_fraction"] * float(max(cfg.bankroll, 0.0))
    merged["reject_reason"] = merged.apply(lambda row: _reject_reason(row, cfg), axis=1)
    merged["status"] = merged["reject_reason"].map(lambda reason: "candidate" if not reason else "skip")
    merged["stake"] = 0.0
    merged["stake_fraction"] = 0.0

    total_cap = float(max(cfg.bankroll, 0.0)) * float(max(cfg.max_total_fraction, 0.0))
    market_cap = float(max(cfg.bankroll, 0.0)) * float(max(cfg.max_market_fraction, 0.0))
    total_allocated = 0.0
    market_allocated: dict[str, float] = {}

    candidate_order = merged[merged["status"] == "candidate"].sort_values(
        ["expected_roi", "edge_used", "probability_edge", "model_probability"],
        ascending=[False, False, False, False],
        kind="mergesort",
    )
    for idx, row in candidate_order.iterrows():
        market = str(row.get("market", "unknown"))
        remaining_total = max(0.0, total_cap - total_allocated)
        remaining_market = max(0.0, market_cap - market_allocated.get(market, 0.0))
        stake = min(float(row["target_stake"]), remaining_total, remaining_market)
        if stake < float(max(cfg.min_stake, 0.0)):
            merged.loc[idx, "status"] = "skip"
            merged.loc[idx, "reject_reason"] = "exposure_or_min_stake"
            continue
        merged.loc[idx, "status"] = "bet"
        merged.loc[idx, "reject_reason"] = ""
        merged.loc[idx, "stake"] = stake
        merged.loc[idx, "stake_fraction"] = stake / float(cfg.bankroll) if cfg.bankroll > 0.0 else 0.0
        total_allocated += stake
        market_allocated[market] = market_allocated.get(market, 0.0) + stake

    output_cols = [
        "status",
        "market",
        "driver_name",
        "bookmaker",
        "decimal_odds",
        "model_probability",
        "implied_probability_raw",
        "fair_market_probability",
        "probability_edge",
        "fair_probability_edge",
        "edge_used",
        "edge_source",
        "fair_edge_available",
        "probability_gate_passed",
        "probability_gate_reason",
        "probability_audit_passed",
        "probability_audit_reason",
        "expected_roi",
        "kelly_fraction_raw",
        "stake_fraction",
        "stake",
        "reject_reason",
    ]
    ordered_cols = [col for col in output_cols if col in merged.columns]
    remaining_cols = [col for col in merged.columns if col not in ordered_cols]
    return merged[ordered_cols + remaining_cols].sort_values(
        ["status", "stake", "expected_roi"],
        ascending=[True, False, False],
        kind="mergesort",
    )


def recommendations_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records"))


def build_betting_report(recommendations: pd.DataFrame, config: BettingConfig) -> dict[str, Any]:
    bets = recommendations[recommendations["status"] == "bet"] if not recommendations.empty else pd.DataFrame()
    stake_total = float(bets["stake"].sum()) if not bets.empty else 0.0
    expected_profit = float((bets["stake"] * bets["expected_roi"]).sum()) if not bets.empty else 0.0
    return {
        "workflow": "f1_betting_recommendations",
        "config": asdict(config),
        "summary": {
            "bets": int(len(bets)),
            "stake_total": stake_total,
            "stake_fraction_total": stake_total / float(config.bankroll) if config.bankroll > 0.0 else 0.0,
            "expected_profit": expected_profit,
            "expected_roi_on_staked": expected_profit / stake_total if stake_total > 0.0 else None,
            "max_loss": stake_total,
        },
        "recommendations": recommendations_to_records(recommendations),
    }
