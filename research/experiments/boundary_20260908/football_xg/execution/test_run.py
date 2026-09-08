"""Selection/promotion contracts, immutable output and target-free issuance."""
from datetime import datetime
import copy

import pytest

from research.experiments.boundary_20260908.football_xg.execution import run


def test_current_targets_are_absent_from_forecast_input_and_source_unchanged():
    row = {"match_id": "E0:2023:A:B", "label": 2, "goals": [0, 9],
           "result_available_at": "2024-01-02T00:00:00", "log_xg_means": {"90": [.1,.2]}}
    frozen = copy.deepcopy(row)
    before = run.forecast_input(row)
    row.update(label=0, goals=[999, -999], result_available_at="poison")
    assert run.forecast_input(row) == before
    before["log_xg_means"]["90"][0] = 999
    assert row["log_xg_means"] == frozen["log_xg_means"]


def test_correction_training_excludes_unresolved_and_current_targets():
    base = {"forecast_cutoff_utc": "2023-01-01T00:00:00", "result_available_at": "2023-01-02T00:00:00"}
    rows = [{**base, "match_id": "past"},
            {**base, "match_id": "unresolved", "result_available_at": "2023-01-09T00:00:01"},
            {**base, "match_id": "current", "forecast_cutoff_utc": "2023-01-09T00:00:00"},
            {**base, "match_id": "too_old", "forecast_cutoff_utc": "2010-01-01T00:00:00"}]
    assert [r["match_id"] for r in run.previous.prior_training_rows(rows, datetime(2023,1,9))] == ["past"]


def example_summaries():
    metrics = {name: {"log_loss": 1., "brier_sum_classes": .6, "top_label_ece_10_bins": .02} for name in run.REFS}
    metrics["candidate"] = {"log_loss": .97, "brier_sum_classes": .59, "top_label_ece_10_bins": .02}
    summary = {"metrics": metrics, "paired": {ref: {"28": {"percentile_95_interval": [-.04,-.01]}} for ref in run.REFS}}
    return summary, {"E0": copy.deepcopy(summary), "SP1": copy.deepcopy(summary)}, {"2024": copy.deepcopy(summary), "2025": copy.deepcopy(summary)}


def test_substantial_gate_cannot_be_won_against_only_weaker_reference():
    pooled, leagues, seasons = example_summaries()
    assert all(run.transfer_gate(pooled, leagues, seasons, pooled, leagues, "candidate").values())
    pooled["metrics"][run.SHOT]["log_loss"] = .975
    gates = run.transfer_gate(pooled, leagues, seasons, pooled, leagues, "candidate")
    assert not gates["pooled_two_percent_and_negative_28day_ci"]


def test_substantial_gate_rejects_ci_crossing_zero_or_delay_league_failure():
    pooled, leagues, seasons = example_summaries()
    sensitivity_leagues = copy.deepcopy(leagues)
    pooled["paired"]["dc_180"]["28"]["percentile_95_interval"][1] = .001
    sensitivity_leagues["SP1"]["metrics"]["candidate"]["log_loss"] = 1.01
    gates = run.transfer_gate(pooled, leagues, seasons, pooled, sensitivity_leagues, "candidate")
    assert not gates["pooled_two_percent_and_negative_28day_ci"]
    assert not gates["fourteen_day_each_league_improves_shot"]


def test_outputs_cannot_replace_frozen_attempt(tmp_path):
    path = tmp_path / "result.json"
    run.write_new(path, {"first": True})
    original = path.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        run.write_new(path, {"second": True})
    assert path.read_bytes() == original
