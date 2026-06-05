from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from rqp.betting import (
    BettingConfig,
    build_betting_recommendations,
    load_prediction_frame,
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
        BettingConfig(require_probability_gate=False, require_oof_probability_audit=False),
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


def test_betting_cli_help() -> None:
    project_python_dir = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(project_python_dir / "run_betting.py"), "--help"]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stdout = completed.stdout
    assert "--predictions" in stdout
    assert "--odds" in stdout
    assert "--fractional-kelly" in stdout
    assert "--max-total-pct" in stdout
