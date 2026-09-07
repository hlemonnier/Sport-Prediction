"""Probability domains, complete top-K books, and allocation semantics."""

import numpy as np
import pandas as pd
import pytest

from packages.f1.betting import BettingConfig, build_betting_recommendations, build_betting_report
from packages.f1.betting.recommendations import _probability_for_market, _probability_gate


def _predictions(n=20):
    return pd.DataFrame({"driver_id": [f"entrant_{i}" for i in range(n)],
        "driver_name": [f"Entrant {i}" for i in range(n)],
        "proba_win": [1./n]*n, "proba_top3": [min(3., n)/n]*n,
        "proba_top10": [min(10., n)/n]*n})


def _config(**kwargs):
    return BettingConfig(require_oof_probability_audit=False, require_odds_timestamp=False,
                         fair_market_min_selection_count=2, **kwargs)


def _quotes(predictions, market="winner", overround=1.1, bookmaker="book"):
    column = {"winner": "proba_win", "podium": "proba_top3", "top10": "proba_top10"}[market]
    return pd.DataFrame({"driver_name": predictions["driver_name"], "market": market,
                         "bookmaker": bookmaker, "decimal_odds": 1. / (predictions[column] * overround)})


@pytest.mark.parametrize("values", [[1.5, -0.5], [float("nan"), 1.], [float("inf"), 0.],
                                   [float("nan"), float("nan")], ["bad", 1.]])
def test_probability_domain_errors_cannot_pass_a_conservation_gate(values):
    predictions = _predictions(2)
    predictions["proba_win"] = values
    passed, reason = _probability_gate(predictions, _config())
    assert passed is False
    assert "invalid_or_missing_probability" in reason


@pytest.mark.parametrize("payload", [{"proba_win": 1.5}, {"proba_win": -0.5},
    {"proba_win": np.nan, "p_win": 0.2}, {"proba_win": 0.3, "p_win": 0.2}])
def test_market_probability_never_clips_or_selectively_ignores_invalid_aliases(payload):
    assert np.isnan(_probability_for_market(pd.Series(payload), "winner"))


def test_alias_only_probabilities_receive_the_same_checks_and_nesting():
    predictions = _predictions(4).rename(columns={"proba_win": "p_win", "proba_top3": "p_top3", "proba_top10": "p_top10"})
    assert _probability_gate(predictions, _config()) == (True, "passed")
    predictions["p_win"] = [0.9, 0.05, 0.03, 0.02]
    assert _probability_gate(predictions, _config())[0] is False
    predictions["p_win"] = [0.2]*4
    assert "sum" in _probability_gate(predictions, _config())[1]


def test_missing_probability_families_and_conflicting_aliases_fail_closed():
    predictions = _predictions(4)
    assert _probability_gate(predictions[["driver_id"]], _config())[1] == "prediction_probabilities_missing"
    predictions["p_win"] = [0.1, 0.2, 0.3, 0.4]
    assert _probability_gate(predictions, _config())[1] == "winner_probability_alias_conflict"


@pytest.mark.parametrize("kind", ["duplicate_name", "duplicate_index", "known_alias_duplicate", "known_identity_conflict", "missing_identity", "multiple_events"])
def test_ambiguous_prediction_identity_or_event_is_rejected(kind):
    predictions = _predictions(2)
    if kind == "duplicate_name":
        predictions["driver_name"] = ["Same Driver"]*2
    elif kind == "duplicate_index":
        predictions.index = [0, 0]
    elif kind == "known_alias_duplicate":
        predictions["driver_id"] = ["HAM", "44"]
    elif kind == "known_identity_conflict":
        predictions.loc[0, ["driver_id", "driver_name"]] = ["HAM", "Max Verstappen"]
    elif kind == "missing_identity":
        predictions.loc[0, ["driver_id", "driver_name"]] = [None, None]
    else:
        predictions["event_key"] = [202601, 202602]
    assert _probability_gate(predictions, _config())[0] is False


def test_invalid_probability_integrity_cannot_be_disabled_for_paper_bets():
    predictions = _predictions(2)
    predictions["proba_win"] = [1.5, -0.5]
    recommendations = build_betting_recommendations(predictions, _quotes(_predictions(2), overround=0.5),
        _config(require_probability_gate=False))
    assert set(recommendations["status"]) == {"skip"}
    assert set(recommendations["reject_reason"]) == {"prediction_integrity_failed"}
    assert recommendations["model_probability"].isna().all()


def test_quote_columns_cannot_override_prediction_probabilities():
    predictions = _predictions(2)
    odds = _quotes(predictions).assign(proba_win=0.99, p_win=0.99)
    result = build_betting_recommendations(predictions, odds, _config())
    assert result["model_probability"].tolist() == [0.5, 0.5]
    assert result["odds_input_proba_win"].tolist() == [0.99, 0.99]


@pytest.mark.parametrize("market,mass", [("winner", 1.), ("podium", 3.), ("top10", 10.)])
@pytest.mark.parametrize("bookmaker", ["book", None])
def test_complete_top_k_book_fair_marginals_have_correct_mass(market, mass, bookmaker):
    predictions = _predictions()
    result = build_betting_recommendations(predictions, _quotes(predictions, market, bookmaker=bookmaker), _config())
    assert result["fair_edge_available"].all()
    assert result["fair_market_probability"].sum() == pytest.approx(mass)
    np.testing.assert_allclose(result["market_overround"], 1.1)
    np.testing.assert_allclose(result["market_implied_probability_sum"], 1.1*mass)
    np.testing.assert_allclose(result["fair_market_probability"], mass/20.)


@pytest.mark.parametrize("kind", ["missing_driver", "duplicate_driver", "unmatched_driver", "missing_identity", "invalid_odds"])
def test_incomplete_or_ambiguous_books_never_claim_fair_probabilities(kind):
    predictions = _predictions()
    odds = _quotes(predictions, "podium")
    if kind == "missing_driver":
        odds = odds.iloc[:-1]
    elif kind == "duplicate_driver":
        odds.loc[19, "driver_name"] = odds.loc[0, "driver_name"]
    elif kind == "unmatched_driver":
        odds.loc[19, "driver_name"] = "Unknown Entrant"
    elif kind == "missing_identity":
        odds.loc[19, "driver_name"] = None
    else:
        odds.loc[19, "decimal_odds"] = np.nan
    result = build_betting_recommendations(predictions, odds, _config())
    assert not result["fair_edge_available"].any()
    assert result["fair_market_probability"].isna().all()
    assert set(result["edge_source"]) == {"raw_odds"}
    if kind == "duplicate_driver":
        assert result["duplicate_odds_selection"].sum() == 2
        assert set(result.loc[result["duplicate_odds_selection"], "reject_reason"]) == {"duplicate_odds_selection"}


def test_fair_books_are_not_pooled_across_bookmakers_or_events():
    predictions = _predictions().assign(event_key=202601)
    first = _quotes(predictions).iloc[:10].assign(bookmaker="first", event_key=202601)
    second = _quotes(predictions).iloc[10:].assign(bookmaker="second", event_key=202601)
    result = build_betting_recommendations(predictions, pd.concat([first, second]), _config())
    assert not result["fair_edge_available"].any()
    wrong_event = build_betting_recommendations(predictions, _quotes(predictions).assign(event_key=202602), _config())
    assert not wrong_event["fair_edge_available"].any()
    assert set(wrong_event["reject_reason"]) == {"prediction_event_mismatch"}
    equivalent_numeric_id = build_betting_recommendations(predictions, _quotes(predictions).assign(event_key=202601.), _config())
    assert equivalent_numeric_id["fair_edge_available"].all()


def test_duplicate_economic_selection_across_books_shares_one_stake_cap():
    predictions = _predictions(2)
    odds = pd.DataFrame({"driver_name": ["Entrant 0", "Entrant 0"], "market": ["winner"]*2,
                         "decimal_odds": [4., 3.], "bookmaker": ["best", "worse"]})
    config = _config(max_bet_fraction=0.01, max_total_fraction=0.5, max_market_fraction=0.5)
    result = build_betting_recommendations(predictions, odds, config)
    assert result["stake"].sum() == 10.
    assert result.set_index("bookmaker").loc["best", "stake"] == 10.
    assert result.set_index("bookmaker").loc["worse", "status"] == "skip"
    report = build_betting_report(result, config)
    assert report["allocation_method"] == "capped_independent_binary_fractional_kelly_heuristic"
    assert report["joint_outcome_dependence_modeled"] is False
    assert report["portfolio_log_growth_optimal"] is False
    assert report["readiness_status"] == "research_only_blocked"


def test_no_zero_stake_row_is_marked_as_a_bet():
    predictions = _predictions(2)
    result = build_betting_recommendations(predictions, _quotes(predictions, overround=0.5),
                                           _config(max_total_fraction=0.))
    assert set(result["status"]) == {"skip"}
    assert result["stake"].sum() == 0.


@pytest.mark.parametrize("kwargs", [{"fractional_kelly": np.nan}, {"max_bet_fraction": 1.1},
    {"probability_sum_tolerance": np.nan}, {"bankroll": -1.}])
def test_invalid_allocation_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        BettingConfig(**kwargs)
