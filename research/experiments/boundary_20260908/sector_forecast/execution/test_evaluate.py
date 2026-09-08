import numpy as np
import pandas as pd
import pytest

from research.experiments.boundary_20260908.sector_forecast.execution import evaluate as e


def cohort():
    rows = []
    for event in ["202301", "202302", "202303", "202304"]:
        for n, (target, error) in enumerate([(0, 0), (0, 0), (0, 0), (1, 8)]):
            rows.append({"event_key": event, "ledger_id": f"{event}:{n}", "driver": "1",
                         "sector": 1 + n % 2, "checkpoint_ms": 1000 + n,
                         "outcome_status": e.MATCHED, "target_id": f"{event}:1:{target}",
                         "target_time_seconds": 10. + target, "y_true": 90., "ref": 90. + error,
                         "candidate": 90. + error / 2})
    return pd.DataFrame(rows)


def test_joint_target_weighting_does_not_count_sector_as_another_outcome():
    summary = e.summarize(cohort(), ["ref", "candidate"])
    assert summary["models"]["ref"][e.METRICS[0]] == 2
    assert summary["models"]["ref"][e.METRICS[1]] == 4
    assert summary["resolved_event_targets"] == 8


def test_both_uncertainty_series_use_their_own_event_errors():
    summary = e.summarize(cohort(), ["ref", "candidate"])
    compare = e.compare(summary, "candidate", ["ref"], resamples=100)
    assert compare["ref"][e.METRICS[0]]["three_event_percentile_95_interval"] == [-1, -1]
    assert compare["ref"][e.METRICS[1]]["three_event_percentile_95_interval"] == [-2, -2]


@pytest.mark.parametrize("mutation", ["event", "identity", "infinity", "missing_label", "label_conflict", "time_conflict", "split_target", "earlier_time", "missing_prediction"])
def test_invalid_cohorts_fail_instead_of_silently_losing_rows(mutation):
    frame = cohort()
    if mutation == "event": frame.loc[0, "event_key"] = None
    if mutation == "identity": frame.loc[0, "ledger_id"] = frame.loc[1, "ledger_id"]
    if mutation == "infinity": frame.loc[0, "y_true"] = np.inf
    if mutation == "missing_label": frame.loc[0, "y_true"] = np.nan
    if mutation == "label_conflict": frame.loc[0, "y_true"] = 99
    if mutation == "time_conflict": frame.loc[0, "target_time_seconds"] = 99
    if mutation == "split_target": frame.loc[0, "target_id"] += ":S1"
    if mutation == "earlier_time": frame.loc[0, "target_time_seconds"] = .5
    if mutation == "missing_prediction": frame.loc[0, "candidate"] = np.nan
    with pytest.raises(ValueError): e.summarize(frame, ["ref", "candidate"])


def test_unmatched_outcomes_are_counted_and_not_assigned_an_error():
    frame = cohort()
    frame.loc[0, ["outcome_status", "target_id", "target_time_seconds", "y_true"]] = [
        "unmatched_in_terminal_recorded_archive", None, np.nan, np.nan]
    result = e.summarize(frame, ["ref"])
    assert (result["issued"], result["resolved_issued"], result["unmatched_issued"]) == (16, 15, 1)


def test_zero_reference_error_cannot_pass_relative_gain_requirement():
    frame = cohort(); frame["perfect"] = 90.
    comparison = e.compare(e.summarize(frame, ["perfect", "candidate"]), "candidate", ["perfect"], resamples=100)
    assert comparison["perfect"][e.METRICS[0]]["relative_gain_fraction"] is None


def test_equal_predictions_have_zero_uncertainty():
    summary = e.summarize(cohort(), ["ref"])
    for metric in e.compare(summary, "ref", ["ref"], resamples=100)["ref"].values():
        assert metric["event_percentile_95_interval"] == [0, 0]
        assert metric["three_event_percentile_95_interval"] == [0, 0]
        assert metric["leave_one_event_out_max_difference"] == 0


def test_selection_cannot_substitute_a_different_year_or_event_cohort():
    frame = cohort()
    spec = {"discovery_events": [202301], "split": {"selection_years": [2023]}}
    with pytest.raises(ValueError, match="exact declared"):
        e.selection_report(frame, ["candidate"], ["ref"], spec)
