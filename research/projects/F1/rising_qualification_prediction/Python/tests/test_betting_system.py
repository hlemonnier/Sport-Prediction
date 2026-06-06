from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_betting import build_parser as build_betting_parser
from run_forward_bet_settlement import build_parser as build_settlement_parser
from rqp.betting import (
    BettingConfig,
    build_betting_report,
    build_betting_recommendations,
    forward_record_hash,
    load_prediction_frame,
    settle_forward_bet_log,
)


def _prediction_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rank": 1,
                "driver_name": "Driver A",
                "driver_id": "a",
                "proba_win": 0.42,
                "proba_top3": 0.78,
                "proba_top10": 0.97,
            },
            {
                "rank": 2,
                "driver_name": "Driver B",
                "driver_id": "b",
                "proba_win": 0.18,
                "proba_top3": 0.55,
                "proba_top10": 0.90,
            },
        ],
    )


def test_betting_recommendations_use_edge_kelly_and_exposure_caps() -> None:
    odds = pd.DataFrame(
        [
            {"market": "winner", "driver_name": "Driver A", "decimal_odds": 3.40, "bookmaker": "book"},
            {"market": "podium", "driver_name": "Driver A", "decimal_odds": 1.80, "bookmaker": "book"},
            {"market": "winner", "driver_name": "Driver B", "decimal_odds": 3.00, "bookmaker": "book"},
        ],
    )
    config = BettingConfig(
        bankroll=1000.0,
        fractional_kelly=0.25,
        min_edge=0.03,
        min_expected_roi=0.02,
        max_bet_fraction=0.02,
        max_market_fraction=0.03,
        max_total_fraction=0.04,
        require_probability_gate=False,
        require_oof_probability_audit=False,
        require_odds_timestamp=False,
    )

    recommendations = build_betting_recommendations(_prediction_rows(), odds, config)
    bets = recommendations[recommendations["status"] == "bet"]

    assert not bets.empty
    assert float(bets["stake"].sum()) <= 40.0 + 1e-9
    assert float(bets.groupby("market")["stake"].sum().max()) <= 30.0 + 1e-9
    assert (bets["expected_roi"] > 0.0).all()
    assert (bets["probability_edge"] >= config.min_edge).all()


def test_betting_recommendations_skip_unsupported_or_negative_edges() -> None:
    odds = pd.DataFrame(
        [
            {"market": "winner", "driver_name": "Driver B", "decimal_odds": 3.00},
            {"market": "fastest_lap", "driver_name": "Driver A", "decimal_odds": 8.00},
        ],
    )

    recommendations = build_betting_recommendations(
        _prediction_rows(),
        odds,
        BettingConfig(require_probability_gate=False, require_oof_probability_audit=False, require_odds_timestamp=False),
    )

    assert set(recommendations["status"]) == {"skip"}
    assert "unsupported_market" in set(recommendations["reject_reason"])
    assert "edge_below_min" in set(recommendations["reject_reason"])


def test_prediction_loader_prefers_full_field_rows(tmp_path: Path) -> None:
    payload = {
        "rows": [{"driver_name": "Top 10 Only"}],
        "all_prediction_rows": [
            {"driver_name": "Driver A", "driver_id": "a"},
            {"driver_name": "Driver B", "driver_id": "b"},
        ],
    }
    path = tmp_path / "prediction.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_prediction_frame(path)

    assert loaded["driver_name"].tolist() == ["Driver A", "Driver B"]


def test_betting_requires_timestamped_pre_close_odds_by_default() -> None:
    predictions = _prediction_rows()
    missing_timestamp = pd.DataFrame(
        [{"market": "winner", "driver_name": "Driver A", "decimal_odds": 3.40, "bookmaker": "book"}],
    )
    after_close = pd.DataFrame(
        [
            {
                "market": "winner",
                "driver_name": "Driver A",
                "decimal_odds": 3.40,
                "bookmaker": "book",
                "odds_timestamp_utc": "2026-05-24T14:01:00Z",
                "market_close_utc": "2026-05-24T14:00:00Z",
            }
        ],
    )
    before_close = after_close.copy()
    before_close["odds_timestamp_utc"] = "2026-05-24T13:55:00Z"

    config = BettingConfig(
        require_probability_gate=False,
        require_oof_probability_audit=False,
        min_edge=0.01,
        min_expected_roi=0.01,
    )

    missing = build_betting_recommendations(predictions, missing_timestamp, config).iloc[0]
    late = build_betting_recommendations(predictions, after_close, config).iloc[0]
    timely = build_betting_recommendations(predictions, before_close, config).iloc[0]

    assert missing["reject_reason"] == "odds_timestamp_missing"
    assert late["reject_reason"] == "odds_after_market_close"
    assert timely["status"] == "bet"


def test_betting_matches_full_name_odds_to_prediction_code_identity() -> None:
    predictions = pd.DataFrame(
        [
            {
                "rank": 1,
                "driver_id": "HAM",
                "proba_win": 0.62,
                "proba_top3": 0.90,
                "proba_top10": 0.99,
            }
        ],
    )
    odds = pd.DataFrame(
        [
            {
                "market": "winner",
                "selection": "Lewis Hamilton",
                "decimal_odds": 2.10,
                "bookmaker": "book",
            }
        ],
    )

    recommendations = build_betting_recommendations(
        predictions,
        odds,
        BettingConfig(
            require_probability_gate=False,
            require_oof_probability_audit=False,
            require_odds_timestamp=False,
            min_edge=0.01,
            min_expected_roi=0.01,
        ),
    )
    row = recommendations.iloc[0]

    assert bool(row["matched_prediction"]) is True
    assert row["matched_alias"] == "known:lewis_hamilton"
    assert float(row["model_probability"]) == 0.62
    assert row["status"] == "bet"


def test_betting_gate_blocks_stale_probability_audit_schema() -> None:
    predictions = _prediction_rows()
    predictions.attrs["probability_audit"] = {
        "available": True,
        "passed": True,
        "source": "walk_forward_oof",
        "metrics": {"win": {"available": True}, "top3": {"available": True}, "top10": {"available": True}},
    }
    odds = pd.DataFrame(
        [{"market": "winner", "driver_name": "Driver A", "decimal_odds": 3.40, "bookmaker": "book"}],
    )

    recommendations = build_betting_recommendations(
        predictions,
        odds,
        BettingConfig(
            require_probability_gate=False,
            require_oof_probability_audit=True,
            require_odds_timestamp=False,
            min_edge=0.01,
            min_expected_roi=0.01,
        ),
    )
    row = recommendations.iloc[0]

    assert bool(row["probability_audit_passed"]) is False
    assert "stale_schema" in row["probability_audit_reason"]
    assert row["reject_reason"] == "probability_audit_failed"


def test_betting_report_labels_research_only_without_market_calibration() -> None:
    odds = pd.DataFrame(
        [{"market": "winner", "driver_name": "Driver A", "decimal_odds": 3.40, "bookmaker": "book"}],
    )
    recommendations = build_betting_recommendations(
        _prediction_rows(),
        odds,
        BettingConfig(require_probability_gate=False, require_oof_probability_audit=False, require_odds_timestamp=False),
    )
    report = build_betting_report(recommendations, BettingConfig())

    assert report["readiness_status"] == "research_only_blocked"
    assert report["stake_label"] == "paper_candidate"
    assert "model_implied_expected_profit" in report["summary"]


def test_betting_cli_help() -> None:
    stdout = build_betting_parser().format_help()
    assert "--predictions" in stdout
    assert "--odds" in stdout
    assert "--fractional-kelly" in stdout
    assert "--max-total-pct" in stdout


def _forward_record(event_id: str, *, pre_market: bool, previous_hash: str = "") -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": "f1_forward_bet_log_v1",
        "event_id": event_id,
        "information_cutoff": "post-qualifying",
        "selection_logged_at_utc": "2026-05-24T13:55:00Z",
        "market_close_utc": "2026-05-24T14:00:00Z",
        "pre_market_logged": bool(pre_market),
        "predictions_sha256": "prediction-hash",
        "odds_sha256": "odds-hash",
        "previous_record_hash": previous_hash,
        "betting_report": {
            "workflow": "f1_betting_recommendations",
            "summary": {"bets": 1, "stake_total": 20.0},
            "recommendations": [
                {
                    "status": "bet",
                    "market": "top10",
                    "driver_name": "Driver A",
                    "decimal_odds": 1.50,
                    "stake": 20.0,
                }
            ],
        },
    }
    record["record_hash"] = forward_record_hash(record)
    return record


def test_forward_bet_settlement_uses_only_hash_valid_pre_market_records(tmp_path: Path) -> None:
    log_path = tmp_path / "forward.jsonl"
    first = _forward_record("2026-monaco", pre_market=True)
    second = _forward_record("2026-spain", pre_market=False, previous_hash=str(first["record_hash"]))
    log_path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in [first, second]) + "\n",
        encoding="utf-8",
    )
    settlements = pd.DataFrame(
        [
            {"event_id": "2026-monaco", "market": "top10", "selection": "Driver A", "won": True},
            {"event_id": "2026-spain", "market": "top10", "selection": "Driver A", "won": True},
        ],
    )

    report = settle_forward_bet_log(
        log_path=log_path,
        settlements=settlements,
        settlement_source_path=None,
        settled_at_utc="2026-05-24T17:00:00Z",
    )

    assert report["pnl_policy"] == "settlement_only_from_hash_valid_pre_market_forward_logs"
    assert report["summary"]["records"] == 2
    assert report["summary"]["records_skipped"] == 1
    assert report["summary"]["bets_logged"] == 1
    assert report["summary"]["bets_settled"] == 1
    assert report["summary"]["pnl_settled"] == 10.0
    assert report["skipped_records"][0]["reason"] == "record_not_pre_market"


def test_forward_bet_settlement_matches_code_to_full_name_settlement(tmp_path: Path) -> None:
    log_path = tmp_path / "forward.jsonl"
    record = {
        "schema_version": "f1_forward_bet_log_v1",
        "event_id": "2026-monaco",
        "information_cutoff": "post-qualifying",
        "selection_logged_at_utc": "2026-05-24T13:55:00Z",
        "market_close_utc": "2026-05-24T14:00:00Z",
        "pre_market_logged": True,
        "predictions_sha256": "prediction-hash",
        "odds_sha256": "odds-hash",
        "previous_record_hash": "",
        "betting_report": {
            "workflow": "f1_betting_recommendations",
            "summary": {"bets": 1, "stake_total": 20.0},
            "recommendations": [
                {
                    "status": "bet",
                    "market": "winner",
                    "driver_id": "HAM",
                    "decimal_odds": 2.00,
                    "stake": 20.0,
                }
            ],
        },
    }
    record["record_hash"] = forward_record_hash(record)
    log_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    settlements = pd.DataFrame(
        [{"event_id": "2026-monaco", "market": "winner", "selection": "Lewis Hamilton", "won": True}],
    )

    report = settle_forward_bet_log(log_path=log_path, settlements=settlements)

    assert report["summary"]["bets_settled"] == 1
    assert report["summary"]["pnl_settled"] == 20.0


def test_forward_bet_settlement_refuses_tampered_hash_chain(tmp_path: Path) -> None:
    log_path = tmp_path / "forward.jsonl"
    record = _forward_record("2026-monaco", pre_market=True)
    report = record["betting_report"]
    assert isinstance(report, dict)
    recommendations = report["recommendations"]
    assert isinstance(recommendations, list)
    recommendations[0]["stake"] = 25.0
    log_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

    settlements = pd.DataFrame(
        [{"event_id": "2026-monaco", "market": "top10", "selection": "Driver A", "won": True}],
    )

    try:
        settle_forward_bet_log(log_path=log_path, settlements=settlements)
    except ValueError as exc:
        assert "record hash" in str(exc)
    else:
        raise AssertionError("tampered forward log should not settle")


def test_forward_bet_settlement_cli_help() -> None:
    stdout = build_settlement_parser().format_help()
    assert "--log-path" in stdout
    assert "--settlements" in stdout
    assert "--allow-after-close-records" in stdout
