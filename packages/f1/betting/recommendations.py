"""Betting portfolio construction from F1 prediction probabilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Optional

import pandas as pd

from packages.f1.data.schemas.driver import driver_identity_signature, resolve_driver_matches


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

PROBABILITY_AUDIT_SCHEMA_VERSION = "pl_gumbel_probability_audit_v4_disjoint_calibration"
REQUIRED_PROBABILITY_AUDIT_FIELDS = {
    "schema_version",
    "probability_layer",
    "score_layer",
    "same_probability_layer_as_production",
    "evaluation_disjoint_from_temperature_fit",
    "samples",
    "event_total_audit",
    "metrics",
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
    require_odds_timestamp: bool = True
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
    for key in ("all_prediction_rows", "rows", "recommendations", "odds", "settlements", "results"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def forward_record_hash(record: dict[str, Any]) -> str:
    """Canonical hash for immutable forward-test betting records."""

    payload = {k: v for k, v in record.items() if k != "record_hash"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_forward_bet_log(path: str | Path, *, verify_hash_chain: bool = True) -> list[dict[str, Any]]:
    log_path = Path(path)
    records: list[dict[str, Any]] = []
    previous_hash = ""
    with open(log_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid forward log JSON at line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Invalid forward log record at line {line_number}: expected object")
            if verify_hash_chain:
                stored_hash = str(record.get("record_hash") or "")
                computed_hash = forward_record_hash(record)
                if not stored_hash or stored_hash != computed_hash:
                    raise ValueError(f"Invalid forward log record hash at line {line_number}")
                stored_previous = str(record.get("previous_record_hash") or "")
                if stored_previous != previous_hash:
                    raise ValueError(f"Invalid forward log hash chain at line {line_number}")
                previous_hash = stored_hash
            records.append(record)
    return records


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


def load_settlement_frame(path: str | Path) -> pd.DataFrame:
    settlement_path = Path(path)
    if settlement_path.suffix.lower() == ".csv":
        return pd.read_csv(settlement_path)
    payload = json.loads(settlement_path.read_text(encoding="utf-8"))
    return pd.DataFrame(_records_from_json_payload(payload))


def _driver_key_frame(frame: pd.DataFrame) -> pd.Series:
    if "driver_name" in frame.columns:
        return frame["driver_name"].map(normalize_participant)
    if "driver_id" in frame.columns:
        return frame["driver_id"].map(normalize_participant)
    return pd.Series("", index=frame.index, dtype=str)


def _ranked_prediction_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    pred = predictions.copy()
    if "pred_rank" not in pred.columns:
        if "rank" in pred.columns:
            pred["pred_rank"] = pd.to_numeric(pred["rank"], errors="coerce")
        else:
            pred["pred_rank"] = pd.Series(range(1, len(pred) + 1), index=pred.index, dtype=float)
    pred["pred_rank"] = pd.to_numeric(pred["pred_rank"], errors="coerce")
    pred = pred[pred["pred_rank"].notna()].copy()
    pred["_identity_signature"] = pred.apply(driver_identity_signature, axis=1)
    return pred


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


def _attach_prediction_matches(prices: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    pred = _ranked_prediction_frame(predictions)
    out = prices.copy()
    out["_identity_signature"] = out.apply(driver_identity_signature, axis=1)
    matchable_prices = out[out["_identity_signature"] != ""].drop_duplicates(subset=["_identity_signature"]).copy()
    if pred.empty or matchable_prices.empty:
        out["matched_prediction"] = False
        out["_prediction_index"] = pd.NA
        return out, {
            "available": False,
            "reason": "identity_frames_empty",
            "matched_count": 0,
            "prices": int(len(prices)),
            "predictions": int(len(predictions)),
        }

    matches, diagnostics = resolve_driver_matches(pred, matchable_prices)
    signature_to_prediction: dict[str, Any] = {}
    matched_alias_by_signature: dict[str, str] = {}
    for _, row in matches.iterrows():
        actual_index = row.get("actual_index")
        pred_index = row.get("pred_index")
        if actual_index not in matchable_prices.index:
            continue
        signature = str(matchable_prices.loc[actual_index, "_identity_signature"])
        signature_to_prediction[signature] = pred_index
        matched_alias_by_signature[signature] = str(row.get("matched_alias") or "")

    out["_prediction_index"] = out["_identity_signature"].map(signature_to_prediction)
    out["matched_alias"] = out["_identity_signature"].map(matched_alias_by_signature).fillna("")
    out["matched_prediction"] = out["_prediction_index"].notna()

    for column in pred.columns:
        if column in {"_identity_signature"}:
            continue
        target = f"{column}_prediction" if column in out.columns else column
        values: list[Any] = []
        for pred_index in out["_prediction_index"].tolist():
            if pred_index in pred.index:
                values.append(pred.loc[pred_index, column])
            else:
                values.append(pd.NA)
        out[target] = values

    diagnostics = dict(diagnostics)
    diagnostics.update(
        {
            "available": True,
            "matched_count": int(out["matched_prediction"].sum()),
            "prices": int(len(prices)),
            "predictions": int(len(pred)),
        }
    )
    return out, diagnostics


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
    timestamp_aliases = ["odds_timestamp_utc", "odds_timestamp", "timestamp_utc", "timestamp", "captured_at"]
    close_aliases = ["market_close_utc", "market_close", "close_time_utc", "event_start_utc"]
    for alias in timestamp_aliases:
        if alias in out.columns and alias != "odds_timestamp_utc":
            out = out.rename(columns={alias: "odds_timestamp_utc"})
            break
    for alias in close_aliases:
        if alias in out.columns and alias != "market_close_utc":
            out = out.rename(columns={alias: "market_close_utc"})
            break
    if "odds_timestamp_utc" in out.columns:
        out["odds_timestamp_utc"] = pd.to_datetime(out["odds_timestamp_utc"], errors="coerce", utc=True)
    if "market_close_utc" in out.columns:
        out["market_close_utc"] = pd.to_datetime(out["market_close_utc"], errors="coerce", utc=True)
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
    missing = sorted(REQUIRED_PROBABILITY_AUDIT_FIELDS - set(audit))
    if missing:
        return False, f"probability_audit_stale_schema_missing_{','.join(missing)}"
    if str(audit.get("schema_version") or "") != PROBABILITY_AUDIT_SCHEMA_VERSION:
        return False, f"probability_audit_schema_{audit.get('schema_version', 'missing')}"
    if str(audit.get("probability_layer") or "") != "pl_gumbel":
        return False, f"probability_audit_layer_{audit.get('probability_layer', 'missing')}"
    if not bool(audit.get("same_probability_layer_as_production", False)):
        return False, "probability_audit_not_production_layer"
    if not bool(audit.get("evaluation_disjoint_from_temperature_fit", False)):
        return False, "probability_audit_temperature_fit_not_disjoint"
    total_audit = audit.get("event_total_audit")
    if not isinstance(total_audit, dict) or not bool(total_audit.get("passed", False)):
        return False, "probability_audit_event_totals_failed"
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
    if bool(config.require_odds_timestamp):
        odds_ts = row.get("odds_timestamp_utc")
        if pd.isna(odds_ts):
            return "odds_timestamp_missing"
        close_ts = row.get("market_close_utc")
        if pd.notna(close_ts) and odds_ts > close_ts:
            return "odds_after_market_close"
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

    probability_gate_passed, probability_gate_reason = _probability_gate(predictions, cfg)
    probability_audit_passed, probability_audit_reason = _probability_audit_gate(predictions, cfg)

    prices = _standardize_odds_frame(odds)
    prices["market"] = prices["market"].map(normalize_market)
    prices["driver_key"] = _driver_key_frame(prices)
    if "decimal_odds" in prices.columns:
        prices["decimal_odds"] = pd.to_numeric(prices["decimal_odds"], errors="coerce")
    else:
        prices["decimal_odds"] = pd.Series(float("nan"), index=prices.index, dtype=float)
    prices = prices[prices.apply(driver_identity_signature, axis=1) != ""].copy()

    merged, identity_diagnostics = _attach_prediction_matches(prices, predictions)
    merged["identity_match_diagnostics"] = json.dumps(identity_diagnostics, sort_keys=True)
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
        "odds_timestamp_utc",
        "market_close_utc",
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
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def build_betting_report(recommendations: pd.DataFrame, config: BettingConfig) -> dict[str, Any]:
    bets = recommendations[recommendations["status"] == "bet"] if not recommendations.empty else pd.DataFrame()
    stake_total = float(bets["stake"].sum()) if not bets.empty else 0.0
    expected_profit = float((bets["stake"] * bets["expected_roi"]).sum()) if not bets.empty else 0.0
    readiness_status = "research_only_blocked"
    return {
        "workflow": "f1_betting_recommendations",
        "readiness_status": readiness_status,
        "readiness_reason": "model probabilities require market-calibrated proof and settled forward evidence before live betting",
        "stake_label": "paper_candidate",
        "config": asdict(config),
        "summary": {
            "bets": int(len(bets)),
            "paper_candidates": int(len(bets)),
            "stake_total": stake_total,
            "stake_fraction_total": stake_total / float(config.bankroll) if config.bankroll > 0.0 else 0.0,
            "expected_profit": expected_profit,
            "model_implied_expected_profit": expected_profit,
            "expected_roi_on_staked": expected_profit / stake_total if stake_total > 0.0 else None,
            "max_loss": stake_total,
        },
        "recommendations": recommendations_to_records(recommendations),
    }


def _boolean_outcome(value: object) -> Optional[bool]:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return bool(value)
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return bool(float(numeric) > 0.0)
    text = str(value).strip().lower()
    if text in {"win", "won", "winner", "hit", "true", "yes", "y", "settled_win"}:
        return True
    if text in {"loss", "lost", "lose", "miss", "false", "no", "n", "settled_loss"}:
        return False
    return None


def _standardize_settlement_frame(settlements: pd.DataFrame) -> pd.DataFrame:
    out = settlements.copy()
    if out.empty:
        return out
    if "driver_name" not in out.columns and "participant" in out.columns:
        out = out.rename(columns={"participant": "driver_name"})
    if "driver_name" not in out.columns and "selection" in out.columns:
        out = out.rename(columns={"selection": "driver_name"})
    if "market" not in out.columns:
        out["market"] = "winner"
    out["market"] = out["market"].map(normalize_market)
    out["driver_key"] = _driver_key_frame(out)
    out["_identity_signature"] = out.apply(driver_identity_signature, axis=1)
    if "event_id" in out.columns:
        out["event_id"] = out["event_id"].astype(str)
    else:
        out["event_id"] = ""

    outcome: pd.Series = pd.Series([None] * len(out), index=out.index, dtype=object)
    for col in ("won", "settled_won", "hit", "settled_binary_result", "result", "settlement_result"):
        if col in out.columns:
            parsed = out[col].map(_boolean_outcome)
            outcome = outcome.where(parsed.isna(), parsed)
    if outcome.isna().any() and "finish_position" in out.columns:
        finish = pd.to_numeric(out["finish_position"], errors="coerce")
        inferred = pd.Series([None] * len(out), index=out.index, dtype=object)
        inferred.loc[(out["market"] == "winner") & finish.notna()] = finish.loc[
            (out["market"] == "winner") & finish.notna()
        ].le(1.0)
        inferred.loc[(out["market"] == "podium") & finish.notna()] = finish.loc[
            (out["market"] == "podium") & finish.notna()
        ].le(3.0)
        inferred.loc[(out["market"] == "top10") & finish.notna()] = finish.loc[
            (out["market"] == "top10") & finish.notna()
        ].le(10.0)
        outcome = outcome.where(inferred.isna(), inferred)
    out["settled_won"] = outcome
    return out


def _settlement_candidates_for_recommendation(
    settlement_frame: pd.DataFrame,
    recommendation: dict[str, Any],
    *,
    market: str,
    event_id: str,
) -> pd.DataFrame:
    pool = settlement_frame[
        (settlement_frame["market"] == market)
        & ((settlement_frame["event_id"] == event_id) | (settlement_frame["event_id"] == ""))
    ].copy()
    if pool.empty:
        return pool
    recommendation_frame = pd.DataFrame([{**recommendation, "pred_rank": 1.0}])
    matches, _ = resolve_driver_matches(recommendation_frame, pool)
    if not matches.empty:
        actual_indices = [idx for idx in matches["actual_index"].tolist() if idx in pool.index]
        if actual_indices:
            return pool.loc[actual_indices].copy()
    driver_key = normalize_participant(
        recommendation.get("driver_name") or recommendation.get("selection") or recommendation.get("participant")
    )
    if not driver_key:
        return pool.iloc[0:0].copy()
    return pool[pool["driver_key"] == driver_key].copy()


def settle_forward_bet_log(
    *,
    log_path: str | Path,
    settlements: pd.DataFrame,
    settlement_source_path: Optional[str | Path] = None,
    settled_at_utc: Optional[str] = None,
    require_pre_market: bool = True,
    verify_hash_chain: bool = True,
) -> dict[str, Any]:
    """Evaluate realized P&L only from predeclared forward logs plus settlement data."""

    records = load_forward_bet_log(log_path, verify_hash_chain=verify_hash_chain)
    settlement_frame = _standardize_settlement_frame(settlements)
    settlement_source_sha256 = _sha256_file(settlement_source_path) if settlement_source_path is not None else None

    settled_rows: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    total_bets = 0
    for record_index, record in enumerate(records):
        pre_market = bool(record.get("pre_market_logged", False))
        if require_pre_market and not pre_market:
            skipped_records.append(
                {
                    "record_index": int(record_index),
                    "record_hash": record.get("record_hash"),
                    "reason": "record_not_pre_market",
                },
            )
            continue
        report = record.get("betting_report")
        recommendations = report.get("recommendations") if isinstance(report, dict) else None
        if not isinstance(recommendations, list):
            continue
        event_id = str(record.get("event_id") or "")
        for recommendation in recommendations:
            if not isinstance(recommendation, dict):
                continue
            if str(recommendation.get("status") or "") != "bet":
                continue
            total_bets += 1
            market = normalize_market(recommendation.get("market"))
            candidates = _settlement_candidates_for_recommendation(
                settlement_frame,
                recommendation,
                market=market,
                event_id=event_id,
            )
            candidates = candidates[candidates["settled_won"].notna()]
            if candidates.empty:
                settled_rows.append(
                    {
                        "event_id": event_id,
                        "market": market,
                        "driver_name": recommendation.get("driver_name"),
                        "status": "unsettled",
                        "stake": float(_to_float(recommendation.get("stake"), default=0.0)),
                        "pnl": None,
                        "record_hash": record.get("record_hash"),
                        "selection_logged_at_utc": record.get("selection_logged_at_utc"),
                        "market_close_utc": record.get("market_close_utc"),
                    },
                )
                continue
            settlement = candidates.iloc[0]
            won = bool(settlement["settled_won"])
            stake = float(max(0.0, _to_float(recommendation.get("stake"), default=0.0)))
            odds = float(_to_float(recommendation.get("decimal_odds"), default=float("nan")))
            pnl = (stake * (odds - 1.0)) if won and math.isfinite(odds) else -stake
            settled_rows.append(
                {
                    "event_id": event_id,
                    "market": market,
                    "driver_name": recommendation.get("driver_name"),
                    "status": "settled",
                    "settled_won": won,
                    "stake": stake,
                    "decimal_odds": odds if math.isfinite(odds) else None,
                    "pnl": float(pnl),
                    "record_hash": record.get("record_hash"),
                    "selection_logged_at_utc": record.get("selection_logged_at_utc"),
                    "market_close_utc": record.get("market_close_utc"),
                    "settled_at_utc": settled_at_utc,
                    "settlement_source_sha256": settlement_source_sha256,
                },
            )

    settled = [row for row in settled_rows if row.get("status") == "settled"]
    unsettled = [row for row in settled_rows if row.get("status") == "unsettled"]
    stake_settled = float(sum(float(row.get("stake") or 0.0) for row in settled))
    pnl_settled = float(sum(float(row.get("pnl") or 0.0) for row in settled))
    return {
        "workflow": "f1_forward_bet_settlement",
        "pnl_policy": "settlement_only_from_hash_valid_pre_market_forward_logs",
        "log_path": str(log_path),
        "settlement_source_path": str(settlement_source_path) if settlement_source_path is not None else None,
        "settlement_source_sha256": settlement_source_sha256,
        "settled_at_utc": settled_at_utc,
        "hash_chain_verified": bool(verify_hash_chain),
        "require_pre_market": bool(require_pre_market),
        "summary": {
            "records": int(len(records)),
            "records_skipped": int(len(skipped_records)),
            "bets_logged": int(total_bets),
            "bets_settled": int(len(settled)),
            "bets_unsettled": int(len(unsettled)),
            "stake_settled": stake_settled,
            "pnl_settled": pnl_settled,
            "roi_on_settled": (pnl_settled / stake_settled) if stake_settled > 0.0 else None,
        },
        "skipped_records": skipped_records,
        "settlements": settled_rows,
    }
