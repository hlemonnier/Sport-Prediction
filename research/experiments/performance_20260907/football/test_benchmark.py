"""Independent contracts for the research runner's timing and scoring math."""
from dataclasses import replace
from datetime import date, timedelta
import math
import json
import numpy as np
import pytest
from research.experiments.performance_20260907.football import benchmark as b

SPEC = json.loads((b.LANE / "spec.json").read_text())


def row(name, day, home, away, label):
    goals = [(1, 0), (0, 0), (0, 1)][label]
    m = b.MatchRecord(name, b.utc_midnight(day), day.year, "epl", None, home, away,
                      *goals, None, None, result_available_at=b.utc_midnight(day+timedelta(days=1)))
    return b.Row(day, m, label)


def test_elo_features_are_unchanged_by_future_or_same_day_results():
    day = date(2023, 1, 1)
    rows = [row("first", day, "A", "B", 0), row("same_day", day, "A", "C", 2),
            row("tomorrow", day+timedelta(days=1), "A", "B", 1)]
    before = b.causal_elo_features(rows, SPEC["elo"])
    changed = b.causal_elo_features([replace(rows[0], label=2), rows[1], replace(rows[2], label=0)], SPEC["elo"])
    assert before["first"] == changed["first"] == [0., 0.]
    assert before["same_day"] == changed["same_day"] == [0., 0.]
    assert before["tomorrow"] != changed["tomorrow"]
    changed_future = b.causal_elo_features(rows[:-1]+[replace(rows[-1], label=0)], SPEC["elo"])
    assert before == changed_future
    assert before == b.causal_elo_features(list(reversed(rows)), SPEC["elo"])


def test_local_day_result_availability_handles_dst_calendar_day():
    spring = date(2024, 3, 31)
    autumn = date(2024, 10, 27)
    assert (b.utc_midnight(spring+timedelta(days=1))-b.utc_midnight(spring)).total_seconds() == 23*3600
    assert (b.utc_midnight(autumn+timedelta(days=1))-b.utc_midnight(autumn)).total_seconds() == 25*3600


def test_uniform_logloss_brier_and_ece_have_declared_scale():
    records = [{"label": y, "probabilities": {"uniform": [1/3]*3}} for y in [0, 1, 2]]
    result = b.metrics(records, "uniform")
    assert result["log_loss"] == pytest.approx(math.log(3))
    assert result["brier_sum_classes"] == pytest.approx(2/3)
    assert result["accuracy"] == pytest.approx(1/3)
    assert result["top_label_ece_10_bins"] == pytest.approx(0)


def test_odds_benchmark_and_invalid_probability_contracts():
    assert b.normalized_odds(["2", "4", "4"]) == pytest.approx((.5, .25, .25))
    for odds in (["", "4", "4"], ["NaN", "4", "4"], ["1", "4", "4"]):
        assert b.normalized_odds(odds) is None
    for p in ([[.2, .2, .2]], [[1., 0., 0.]], [[np.nan, .5, .5]]):
        with pytest.raises(ValueError):
            b.validate_probabilities(p)


@pytest.mark.parametrize("block_days", [1, 28])
def test_paired_bootstrap_preserves_exact_constant_logloss_difference(block_days):
    records = [{"season": season, "day": f"{season}-08-{day:02}", "label": 0,
                "probabilities": {"dc_equal": [.25, .25, .5], "candidate": [.5, .25, .25]}}
               for season in [2023, 2024] for day in [1, 2, 2, 29]]
    result = b.paired_uncertainty(records, "candidate", SPEC, block_days)
    assert result["difference_candidate_minus_dc_equal"] == pytest.approx(-math.log(2))
    assert result["percentile_95_interval"] == pytest.approx([-math.log(2)]*2)
    assert result["bootstrap_fraction_difference_below_zero"] == 1


def test_frozen_design_has_disjoint_chronology_and_three_candidates():
    assert max(SPEC["warmup_seasons"]) < min(SPEC["validation_seasons"])
    assert max(SPEC["validation_seasons"]) < min(SPEC["test_seasons"])
    assert len(SPEC["performance_mechanisms"]) == 3
    assert "closing_market" not in SPEC["selectable_models_in_tie_order"]
