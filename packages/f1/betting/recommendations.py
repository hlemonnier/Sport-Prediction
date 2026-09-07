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

from packages.f1.data.schemas.driver import driver_identity_signature, resolve_driver_matches, row_identity_aliases


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
MARKET_POSITION_COUNTS = {"winner": 1, "podium": 3, "top10": 10}
ALLOCATION_METHOD = "capped_independent_binary_fractional_kelly_heuristic"

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

    def __post_init__(self) -> None:
        for key in ("bankroll", "min_edge", "min_expected_roi", "min_stake", "probability_sum_tolerance",
                    "fair_market_overround_min", "fair_market_overround_max"):
            if not math.isfinite(float(getattr(self, key))):
                raise ValueError(f"{key} must be finite")
        for key in ("fractional_kelly", "min_probability", "max_bet_fraction", "max_market_fraction", "max_total_fraction"):
            value = float(getattr(self, key))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{key} must be a finite fraction between zero and one")
        if self.bankroll < 0 or self.min_stake < 0 or self.probability_sum_tolerance < 0:
            raise ValueError("bankroll, minimum stake and probability tolerance must be non-negative")
        if not 0.0 < self.fair_market_overround_min <= self.fair_market_overround_max:
            raise ValueError("overround bounds must be positive and ordered")


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


def _event_identity_token(value: object) -> str:
    numeric = _to_float(value)
    if math.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


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
    pred = predictions.reset_index(drop=True).copy()
    if "pred_rank" not in pred.columns:
        if "rank" in pred.columns:
            pred["pred_rank"] = pd.to_numeric(pred["rank"], errors="coerce")
        else:
            pred["pred_rank"] = pd.Series(range(1, len(pred) + 1), index=pred.index, dtype=float)
    pred["pred_rank"] = pd.to_numeric(pred["pred_rank"], errors="coerce")
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


def _prediction_integrity(predictions: pd.DataFrame) -> tuple[bool, str, dict[str, pd.Series]]:
    """Validate identities and every supplied probability alias before sums.

    Disabling an empirical probability gate never permits invalid probability
    domains, conflicting aliases, missing values or ambiguous driver rows.
    """

    if predictions.empty:
        return False, "predictions_empty", {}
    if not predictions.columns.is_unique:
        return False, "prediction_duplicate_columns", {}
    if not predictions.index.is_unique:
        return False, "prediction_duplicate_row_index", {}
    seen_aliases: set[tuple[int, str]] = set()
    for _, row in predictions.iterrows():
        aliases = {alias for alias in row_identity_aliases(row) if alias[0] <= 4}
        if not aliases:
            return False, "prediction_driver_keys_missing", {}
        if len({alias for priority, alias in aliases if priority == 0}) > 1:
            return False, "prediction_driver_identity_conflict", {}
        if seen_aliases.intersection(aliases):
            return False, "prediction_duplicate_driver_identity", {}
        seen_aliases.update(aliases)
    for column in ("event_id", "event_key", "year", "round_number"):
        if column in predictions and predictions[column].dropna().map(_event_identity_token).nunique() > 1:
            return False, "prediction_multiple_events_unsupported", {}
    probabilities: dict[str, pd.Series] = {}
    for market, aliases in MARKET_PROBABILITY_COLUMNS.items():
        present = [column for column in aliases if column in predictions]
        for column in present:
            values = pd.to_numeric(predictions[column], errors="coerce")
            if values.isna().any() or (~values.between(0.0, 1.0)).any():
                return False, f"{column}_invalid_or_missing_probability", {}
            if market in probabilities and not values.sub(probabilities[market]).abs().le(1e-12).all():
                return False, f"{market}_probability_alias_conflict", {}
            probabilities[market] = values
    if not probabilities:
        return False, "prediction_probabilities_missing", {}
    return True, "passed", probabilities


def _probability_gate(predictions: pd.DataFrame, config: BettingConfig) -> tuple[bool, str]:
    valid, reason, probabilities = _prediction_integrity(predictions)
    if not valid:
        return False, reason
    n = int(len(predictions))
    tolerance = float(max(config.probability_sum_tolerance, 0.0))
    for market, values in probabilities.items():
        expected = float(min(MARKET_POSITION_COUNTS[market], n))
        total = float(values.sum())
        if abs(total - expected) > tolerance:
            return False, f"{MARKET_PROBABILITY_COLUMNS[market][0]}_sum_{total:.3f}_expected_{expected:.3f}"
    for lower, higher in (("winner", "podium"), ("podium", "top10"), ("winner", "top10")):
        if lower in probabilities and higher in probabilities:
            if (probabilities[lower] > probabilities[higher] + 1e-12).any():
                return False, f"{lower}_gt_{higher}"
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
    values: list[float] = []
    for column in MARKET_PROBABILITY_COLUMNS.get(market, []):
        if column in row.index:
            value = _to_float(row.get(column))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                return float("nan")
            values.append(value)
    if not values or max(values) - min(values) > 1e-12:
        return float("nan")
    return values[0]


def _reject_reason(row: pd.Series, config: BettingConfig) -> str:
    if not bool(row.get("prediction_integrity_passed", False)):
        return "prediction_integrity_failed"
    if not bool(row.get("matched_prediction", False)):
        return "no_prediction_match"
    if bool(row.get("duplicate_odds_selection", False)):
        return "duplicate_odds_selection"
    if not bool(row.get("prediction_event_match", True)):
        return "prediction_event_mismatch"
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


def _attach_fair_market_probabilities(
    merged: pd.DataFrame,
    *,
    prediction_count: int,
    probability_gate_passed: bool,
    config: BettingConfig,
) -> pd.DataFrame:
    """Remove proportional margin only from complete, unique event markets.

    A top-K book contains K winning selections, so its fair marginals sum to
    min(K, field size). Sum(implied)/K is the comparable overround ratio. This
    proportional marginal correction does not infer a joint ranking law.
    """

    out = merged.copy()
    out["fair_market_probability"] = float("nan")
    out["fair_edge_available"] = False
    out["fair_edge_unavailable_reason"] = "incomplete_market"
    out["duplicate_odds_selection"] = False
    out["market_overround"] = float("nan")
    out["market_implied_probability_sum"] = float("nan")
    out["market_selection_count"] = 0
    out["market_probability_mass"] = float("nan")
    groups = [column for column in ("event_id", "event_key", "year", "round_number", "market", "bookmaker") if column in out]
    for _, indices in out.groupby(groups, dropna=False, sort=False).groups.items():
        group = out.loc[indices]
        market = str(group["market"].iloc[0])
        mass = float(min(MARKET_POSITION_COUNTS.get(market, 0), prediction_count))
        indices_matched = group.loc[group["matched_prediction"], "_prediction_index"]
        repeated = indices_matched.duplicated(keep=False)
        out.loc[indices_matched.index[repeated], "duplicate_odds_selection"] = True
        implied = pd.to_numeric(group["implied_probability_raw"], errors="coerce")
        implied_sum = float(implied.sum())
        ratio = implied_sum / mass if mass > 0.0 else float("nan")
        out.loc[indices, "market_implied_probability_sum"] = implied_sum
        out.loc[indices, "market_selection_count"] = len(group)
        out.loc[indices, "market_probability_mass"] = mass
        out.loc[indices, "market_overround"] = ratio
        reason = ""
        if market not in MARKET_POSITION_COUNTS:
            reason = "unsupported_market"
        elif not probability_gate_passed:
            reason = "prediction_field_not_probability_certified"
        elif repeated.any():
            reason = "duplicate_market_selection"
        elif not group["matched_prediction"].all() or len(group) != prediction_count or indices_matched.nunique() != prediction_count:
            reason = "incomplete_or_unmatched_market_field"
        elif not group["prediction_event_match"].all():
            reason = "prediction_event_mismatch"
        elif len(group) < max(2, int(config.fair_market_min_selection_count)):
            reason = "market_below_minimum_selection_count"
        elif implied.isna().any() or (~implied.between(0.0, 1.0, inclusive="neither")).any():
            reason = "invalid_market_odds"
        elif not float(config.fair_market_overround_min) <= ratio <= float(config.fair_market_overround_max):
            reason = "overround_ratio_outside_configured_bounds"
        else:
            fair = implied / ratio
            if not fair.between(0.0, 1.0).all():
                reason = "proportional_fair_marginals_outside_probability_domain"
            else:
                out.loc[indices, "fair_market_probability"] = fair
                out.loc[indices, "fair_edge_available"] = True
        out.loc[indices, "fair_edge_unavailable_reason"] = reason
    return out


def build_betting_recommendations(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    config: Optional[BettingConfig] = None,
) -> pd.DataFrame:
    cfg = config or BettingConfig()
    if odds.empty:
        return pd.DataFrame()

    probability_gate_passed, probability_gate_reason = _probability_gate(predictions, cfg)
    prediction_integrity_passed, prediction_integrity_reason, _ = _prediction_integrity(predictions)
    probability_audit_passed, probability_audit_reason = _probability_audit_gate(predictions, cfg)

    prices = _standardize_odds_frame(odds)
    # Model probabilities must originate in the prediction frame. A quote
    # input carrying a similarly named column cannot override those values.
    probability_columns = {column for aliases in MARKET_PROBABILITY_COLUMNS.values() for column in aliases}
    prices = prices.rename(columns={column: f"odds_input_{column}" for column in probability_columns if column in prices})
    prices["market"] = prices["market"].map(normalize_market)
    prices["driver_key"] = _driver_key_frame(prices)
    if "decimal_odds" in prices.columns:
        prices["decimal_odds"] = pd.to_numeric(prices["decimal_odds"], errors="coerce")
    else:
        prices["decimal_odds"] = pd.Series(float("nan"), index=prices.index, dtype=float)
    prices = prices.reset_index(drop=True)
    # Invalid prediction frames still produce explicit skipped quote rows;
    # don't pass duplicate columns into identity matching or pandas indexing.
    match_predictions = predictions if predictions.columns.is_unique else predictions.iloc[:, ~predictions.columns.duplicated()]

    merged, identity_diagnostics = _attach_prediction_matches(prices, match_predictions)
    merged["prediction_integrity_passed"] = bool(prediction_integrity_passed)
    merged["prediction_integrity_reason"] = prediction_integrity_reason
    merged["prediction_event_match"] = True
    for column in ("event_id", "event_key", "year", "round_number"):
        if column in prices:
            prediction_column = f"{column}_prediction"
            if prediction_column in merged:
                merged["prediction_event_match"] &= merged[column].map(_event_identity_token).eq(merged[prediction_column].map(_event_identity_token))
            elif prices[column].dropna().map(_event_identity_token).nunique() > 1:
                merged["prediction_event_match"] = False
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
    merged = _attach_fair_market_probabilities(merged, prediction_count=len(predictions),
        probability_gate_passed=probability_gate_passed, config=cfg)
    merged["probability_edge"] = merged["model_probability"] - merged["implied_probability_raw"]
    merged["fair_probability_edge"] = merged["model_probability"] - merged["fair_market_probability"]
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
    merged["allocation_method"] = ALLOCATION_METHOD
    merged["joint_outcome_dependence_modeled"] = False
    merged["portfolio_log_growth_optimal"] = False

    total_cap = float(max(cfg.bankroll, 0.0)) * float(max(cfg.max_total_fraction, 0.0))
    market_cap = float(max(cfg.bankroll, 0.0)) * float(max(cfg.max_market_fraction, 0.0))
    total_allocated = 0.0
    market_allocated: dict[str, float] = {}
    selection_allocated: dict[tuple[str, object], float] = {}
    selection_cap = float(max(cfg.bankroll, 0.0)) * float(max(cfg.max_bet_fraction, 0.0))

    candidate_order = merged[merged["status"] == "candidate"].sort_values(
        ["expected_roi", "edge_used", "probability_edge", "model_probability"],
        ascending=[False, False, False, False],
        kind="mergesort",
    )
    for idx, row in candidate_order.iterrows():
        market = str(row.get("market", "unknown"))
        remaining_total = max(0.0, total_cap - total_allocated)
        remaining_market = max(0.0, market_cap - market_allocated.get(market, 0.0))
        selection = (market, row["_prediction_index"])
        remaining_selection = max(0.0, selection_cap - selection_allocated.get(selection, 0.0))
        stake = min(float(row["target_stake"]), remaining_total, remaining_market, remaining_selection)
        if stake <= 0.0 or stake < float(max(cfg.min_stake, 0.0)):
            merged.loc[idx, "status"] = "skip"
            merged.loc[idx, "reject_reason"] = "exposure_or_min_stake"
            continue
        merged.loc[idx, "status"] = "bet"
        merged.loc[idx, "reject_reason"] = ""
        merged.loc[idx, "stake"] = stake
        merged.loc[idx, "stake_fraction"] = stake / float(cfg.bankroll) if cfg.bankroll > 0.0 else 0.0
        total_allocated += stake
        market_allocated[market] = market_allocated.get(market, 0.0) + stake
        selection_allocated[selection] = selection_allocated.get(selection, 0.0) + stake

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
        "allocation_method": ALLOCATION_METHOD,
        "joint_outcome_dependence_modeled": False,
        "portfolio_log_growth_optimal": False,
        "allocation_limitations": [
            "winner_podium_top10_payoffs_share_the_same_race_outcome",
            "exposure_caps_do_not_make_independent_binary_kelly_a_joint_portfolio_optimizer",
            "joint_scenario_rank_samples_and_matched_probability_marginals_required_for_joint_optimization",
        ],
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
