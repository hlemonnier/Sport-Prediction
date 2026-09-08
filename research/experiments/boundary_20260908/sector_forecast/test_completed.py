"""Synthetic streaming, timing, attribution and sector-identity regressions."""
from copy import deepcopy
import json
import math

import pytest

from research.experiments.boundary_20260908.sector_forecast import completed as c


def line(seconds, payload):
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}" + json.dumps(payload) + "\n"


def patch(seconds, driver="1", **values):
    return line(seconds, {"Lines": {driver: values}})


def opening():
    return patch(1, NumberOfLaps=0, InPit=False, PitOut=False)


def lap(start=1, new_counter=1, *, last="1:25.000", s3="20.000"):
    return (patch(start + 30, Sectors={"0": {"Value": "30.000"}, "1": {"Value": ""}, "2": {"Value": ""}})
            + patch(start + 65, Sectors={"1": {"Value": "35.000"}})
            + patch(start + 85, NumberOfLaps=new_counter, Sectors={"2": {"Value": s3}}, LastLapTime={"Value": last}))


def paths(tmp_path, timing, *, session=None, track=None, tyres=None):
    content = {"TimingData": timing,
               "SessionStatus": line(0, {"Status": "Started"}) if session is None else session,
               "TrackStatus": line(0, {"Status": "1"}) if track is None else track,
               "TimingAppData": line(0, {"Lines": {"1": {"Stints": {"0": {"Compound": "SOFT", "New": "true"}}}}}) if tyres is None else tyres}
    result = {}
    for key, value in content.items():
        result[key] = tmp_path / f"{key}.jsonStream"
        result[key].write_text(value)
    return result


def replay(inputs):
    stats = {}
    rows = list(c.iter_completed(inputs, 202201, stats))
    return rows, stats


def last_completion(rows):
    return [r for r in rows if r["record_kind"] == "unit_counter_completion"][-1]


def test_unit_boundary_saves_old_epoch_sources_and_exact_sector_equation(tmp_path):
    rows, stats = replay(paths(tmp_path, opening() + lap()))
    row = last_completion(rows)
    assert row["observed_valid_completed"] is True
    assert row["previous_counter"] == 0 and row["new_counter"] == 1
    assert row["sectors_seconds"] == [30., 35., 20.]
    assert row["full_lap_seconds"] == 85. and row["sum_residual_seconds"] == 0
    assert row["available_ms"] == 86000
    assert [s["packet_sequence"] for s in row["sector_sources"]] == [1, 2, 3]
    assert row["full_lap_source"]["packet_sequence"] == 3
    assert all(row["support"][k] for k in ("ordered_prepacket_s1_s2", "unit_counter_advance", "sector_sum_coherent"))
    assert not row["canonical_lap_number_certified"] and not row["canonical_is_accurate_certified"]
    assert not row["canonical_tyre_life_certified"] and not row["historical_client_receipt_certified"]
    assert stats["stream_input_valid"] and stats["counts"]["observed_valid_completed"] == 1
    assert json.loads(json.dumps(rows, allow_nan=False)) == rows


def test_raw_initialization_is_retained_but_never_certified_complete(tmp_path):
    rows, _ = replay(paths(tmp_path, patch(10, NumberOfLaps=1, InPit=False, Sectors={"2": {"Value": "20"}})))
    assert len(rows) == 1 and rows[0]["record_kind"] == "counter_initialization"
    assert not rows[0]["observed_valid_completed"]
    assert "unknown_previous_counter" in rows[0]["ambiguity_reasons"]


def test_future_appends_and_payload_poison_do_not_rewrite_prior_records(tmp_path):
    inputs = paths(tmp_path, opening() + lap())
    before, _ = replay(inputs)
    with inputs["TimingData"].open("a") as f:
        f.write(lap(86, 2, last="9:59.000", s3="534"))
    with inputs["TimingAppData"].open("a") as f:
        f.write(line(200, {"Lines": {"1": {"Stints": {"9": {"Compound": "POISON", "TotalLaps": 999}}}}}))
    after, _ = replay(inputs)
    assert before == after[:len(before)]
    assert before[-1]["raw_tyre_support"][-1]["fields"]["Compound"]["value"] == "SOFT"
    for row in before:
        for source in row["sector_sources"]:
            if source:
                assert source["available_ms"] <= row["available_ms"]
        for group in row["raw_control_observations"].values():
            assert all(s["available_ms"] < row["available_ms"] for s in group)


def test_final_csv_or_target_objects_are_never_accessed(tmp_path):
    class PoisonPath:
        def __fspath__(self):
            raise AssertionError("final CSV accessed")
    inputs = paths(tmp_path, opening() + lap())
    inputs["canonical_csv"] = inputs["target_csv"] = PoisonPath()
    assert last_completion(replay(inputs)[0])["observed_valid_completed"]


def test_each_timing_prefix_emits_identical_already_closed_rows(tmp_path):
    timing = opening() + lap() + lap(86, 2) + lap(171, 3)
    inputs = paths(tmp_path, timing)
    full, _ = replay(inputs)
    raw = timing.splitlines(keepends=True)
    for cut in range(1, len(raw) + 1):
        inputs["TimingData"].write_text("".join(raw[:cut]))
        prefix, _ = replay(inputs)
        assert prefix == [r for r in full if r["packet_sequence"] < cut]


def test_late_s3_adds_ambiguous_diagnostic_without_upgrading_closed_history(tmp_path):
    timing = opening() + lap(s3=None)
    inputs = paths(tmp_path, timing)
    before, _ = replay(inputs)
    with inputs["TimingData"].open("a") as f:
        f.write(patch(86.031, Sectors={"2": {"Value": "20.000"}}))
    after, _ = replay(inputs)
    assert after[:len(before)] == before
    old, new = after[-2:]
    assert old["sectors_seconds"][2] is None and old["sum_residual_seconds"] is None
    assert new["sectors_seconds"] == [30., 35., 20.] and new["sum_residual_seconds"] == 0
    assert new["revision_of"] == old["completion_id"]
    assert new["record_kind"] == "late_completion_update" and new["available_ms"] == 86031
    assert not old["observed_valid_completed"] and not new["observed_valid_completed"]


def test_later_lastlap_revision_never_overwrites_prior_valid_value(tmp_path):
    inputs = paths(tmp_path, opening() + lap())
    iterator = c.iter_completed(inputs, 202201)
    first, completion = next(iterator), next(iterator)
    saved = deepcopy(completion)
    list(iterator)
    with inputs["TimingData"].open("a") as f:
        f.write(patch(87, LastLapTime={"Value": "1:26.000"}))
    rows, _ = replay(inputs)
    assert completion == saved == rows[-2]
    assert rows[-1]["full_lap_seconds"] == 86 and rows[-1]["revision_of"] == completion["completion_id"]
    assert rows[-1]["sum_residual_seconds"] == -1 and not rows[-1]["observed_valid_completed"]


def test_s3_received_before_counter_is_not_substituted_for_missing_boundary_s3(tmp_path):
    timing = opening() + lap(s3=None)
    lines = timing.splitlines(keepends=True)
    timing = "".join(lines[:-1]) + patch(85, Sectors={"2": {"Value": "20"}}) + lines[-1]
    rows, _ = replay(paths(tmp_path, timing))
    assert any(r["record_kind"] == "unassigned_completion_update" for r in rows)
    closed = last_completion(rows)
    assert closed["sectors_seconds"][2] is None and not closed["observed_valid_completed"]


def test_neutralization_between_driver_updates_contaminates_whole_epoch(tmp_path):
    track = line(0, {"Status": "1"}) + line(40, {"Status": "4"}) + line(50, {"Status": "1"})
    row = last_completion(replay(paths(tmp_path, opening() + lap(), track=track))[0])
    assert not row["observed_valid_completed"] and "neutralization_observed" in row["quality_reasons"]
    assert [s["value"] for s in row["raw_control_observations"]["TrackStatus"]] == ["1", "4", "1"]


def test_equal_clock_control_changes_are_not_read_at_completion(tmp_path):
    track = line(0, {"Status": "1"}) + line(86, {"Status": "4"})
    row = last_completion(replay(paths(tmp_path, opening() + lap(), track=track))[0])
    assert row["observed_valid_completed"]
    assert [s["value"] for s in row["raw_control_observations"]["TrackStatus"]] == ["1"]


def test_pit_transition_cannot_be_erased_by_later_false_state(tmp_path):
    timing = opening() + patch(31, Sectors={"0": {"Value": "30"}})
    timing += patch(45, InPit=True) + patch(50, InPit=False)
    timing += patch(66, Sectors={"1": {"Value": "35"}})
    timing += patch(86, NumberOfLaps=1, Sectors={"2": {"Value": "20"}}, LastLapTime={"Value": "1:25"})
    row = last_completion(replay(paths(tmp_path, timing))[0])
    assert "pit_observed" in row["quality_reasons"] and not row["observed_valid_completed"]


def test_fresh_pitout_contaminates_epoch_but_is_not_carried_forever(tmp_path):
    timing = opening() + patch(5, PitOut=True) + lap() + lap(86, 2)
    rows, _ = replay(paths(tmp_path, timing))
    completions = [r for r in rows if r["record_kind"] == "unit_counter_completion"]
    assert "pit_out_observed" in completions[0]["quality_reasons"]
    assert not completions[0]["observed_valid_completed"] and completions[1]["observed_valid_completed"]


@pytest.mark.parametrize("field,value,reason", [("InPit", 0, "pit_state_unknown"),
    ("InPit", "false", "pit_state_unknown"), ("Retired", True, "retirement_or_stoppage_observed")])
def test_pit_boolean_state_is_explicit(field, value, reason, tmp_path):
    timing = opening() + patch(5, **{field: value}) + lap()
    row = last_completion(replay(paths(tmp_path, timing))[0])
    assert reason in row["quality_reasons"] and not row["observed_valid_completed"]


def test_sector_repeat_is_safe_but_revision_and_clear_are_not(tmp_path):
    base = opening() + patch(31, Sectors={"0": {"Value": "30"}})
    tail = patch(66, Sectors={"1": {"Value": "35"}}) + patch(86, NumberOfLaps=1, Sectors={"2": {"Value": "20"}}, LastLapTime={"Value": "1:25"})
    repeated = last_completion(replay(paths(tmp_path, base + patch(40, Sectors={"0": {"Value": "30"}}) + tail))[0])
    assert repeated["observed_valid_completed"] and repeated["support"]["sector_repeat_counts"] == {"1": 1}
    revised = last_completion(replay(paths(tmp_path, base + patch(40, Sectors={"0": {"Value": "31"}}) + tail))[0])
    assert "sector_revision_within_epoch" in revised["ambiguity_reasons"]
    cleared = last_completion(replay(paths(tmp_path, base + patch(35, Sectors={"0": {"Value": ""}})
                                      + patch(40, Sectors={"0": {"Value": "30"}}) + tail))[0])
    assert "sector_cleared_after_positive" in cleared["ambiguity_reasons"]
    assert not revised["observed_valid_completed"] and not cleared["observed_valid_completed"]


def test_sector_values_in_boundary_packet_cannot_repair_prior_missing_support(tmp_path):
    text = opening() + patch(86, NumberOfLaps=1, Sectors={str(i): {"Value": str(v)} for i, v in enumerate([30, 35, 20])}, LastLapTime={"Value": "1:25"})
    row = last_completion(replay(paths(tmp_path, text))[0])
    assert row["sectors_seconds"] == [None, None, 20.]
    assert "s1_s2_in_counter_boundary_packet" in row["ambiguity_reasons"]
    assert not row["observed_valid_completed"]


def test_samepacket_s1_s2_cannot_supply_ordered_prior_pair(tmp_path):
    text = opening() + patch(31, Sectors={"0": {"Value": "30"}, "1": {"Value": "35"}})
    text += patch(86, NumberOfLaps=1, Sectors={"2": {"Value": "20"}}, LastLapTime={"Value": "1:25"})
    row = last_completion(replay(paths(tmp_path, text))[0])
    assert "no_ordered_prepacket_s1_s2" in row["ambiguity_reasons"]


def test_five_second_boundary_window_marks_sector_attribution_ambiguous(tmp_path):
    text = opening() + patch(6, Sectors={"0": {"Value": "30"}})
    text += patch(66, Sectors={"1": {"Value": "35"}})
    text += patch(86, NumberOfLaps=1, Sectors={"2": {"Value": "20"}}, LastLapTime={"Value": "1:25"})
    row = last_completion(replay(paths(tmp_path, text))[0])
    assert "sector_within_five_seconds_of_boundary" in row["ambiguity_reasons"]


def test_counter_reversal_quarantines_until_exceeding_highwater(tmp_path):
    text = opening() + lap() + lap(86, 0) + lap(171, 1) + lap(256, 2) + lap(341, 3)
    rows, _ = replay(paths(tmp_path, text))
    assert rows[2]["record_kind"] == "counter_reversal"
    assert all(not r["observed_valid_completed"] for r in rows[2:5])
    assert rows[-1]["observed_valid_completed"]


def test_counter_jump_never_guesses_missing_lap_association(tmp_path):
    rows, _ = replay(paths(tmp_path, opening() + lap(new_counter=2) + lap(86, 3) + lap(171, 4)))
    assert rows[1]["record_kind"] == "counter_jump" and not rows[1]["observed_valid_completed"]
    assert not rows[2]["observed_valid_completed"] and rows[3]["observed_valid_completed"]


def test_boolean_counter_is_invalid_and_cannot_advance_epoch(tmp_path):
    row = last_completion(replay(paths(tmp_path, opening() + patch(5, NumberOfLaps=True) + lap()))[0])
    assert row["previous_counter"] == 0 and "invalid_counter" in row["ambiguity_reasons"]


@pytest.mark.parametrize("last,expected", [("1:25.003", True), ("1:24.997", True), ("1:25.004", False)])
def test_fixed_millisecond_sum_tolerance(last, expected, tmp_path):
    row = last_completion(replay(paths(tmp_path, opening() + lap(last=last)))[0])
    assert row["observed_valid_completed"] is expected


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "-1:90", "0:90", "", None, True])
def test_invalid_raw_durations_cannot_make_valid_history(value, tmp_path):
    row = last_completion(replay(paths(tmp_path, opening() + lap(last=value)))[0])
    assert row["full_lap_seconds"] is None and not row["observed_valid_completed"]


def test_regressed_sector_or_control_timestamps_are_never_certified(tmp_path):
    timing = opening() + patch(31, Sectors={"0": {"Value": "30"}})
    timing += patch(30, Sectors={"1": {"Value": "35"}})
    timing += patch(86, NumberOfLaps=1, Sectors={"2": {"Value": "20"}}, LastLapTime={"Value": "1:25"})
    row = last_completion(replay(paths(tmp_path, timing))[0])
    assert "raw_timestamp_regression" in row["ambiguity_reasons"]
    session = line(0, {"Status": "Started"}) + line(20, {"Status": "Finished"}) + line(15, {"Status": "Started"})
    row = last_completion(replay(paths(tmp_path, opening() + lap(), session=session))[0])
    assert "regressed_auxiliary_timestamp" in row["ambiguity_reasons"] and not row["observed_valid_completed"]


def test_auxiliary_future_parse_failures_do_not_backdate_into_rows_but_invalidate_input(tmp_path):
    inputs = paths(tmp_path, opening() + lap())
    before, stats = replay(inputs)
    assert stats["stream_input_valid"]
    with inputs["TrackStatus"].open("a") as f:
        f.write("00:03:20.000{broken}\n")
    after, stats = replay(inputs)
    assert after == before and not stats["stream_input_valid"]
    with inputs["TimingAppData"].open("a") as f:
        f.write("bad clock and malformed payload\n")
    after, stats = replay(inputs)
    assert after == before and not stats["stream_input_valid"]
    assert stats["auxiliary"]["TimingAppData"]["unplaced_timestamp"]


def test_sparse_tyre_updates_preserve_only_received_fields_and_times(tmp_path):
    tyres = line(0, {"Lines": {"1": {"Stints": {"0": {"Compound": "SOFT", "New": "true"}}}}})
    tyres += line(40, {"Lines": {"1": {"Stints": {"0": {"TotalLaps": 3}}}}})
    tyres += line(86, {"Lines": {"1": {"Stints": {"1": {"Compound": "HARD"}}}}})
    row = last_completion(replay(paths(tmp_path, opening() + lap(), tyres=tyres))[0])
    final = row["raw_tyre_support"][-1]
    assert final["highest_observed_stint_index"] == 0
    assert final["fields"]["Compound"]["value"] == "SOFT" and final["fields"]["TotalLaps"]["value"] == 3
    assert final["fields"]["Compound"]["available_ms"] == 0 and final["fields"]["TotalLaps"]["available_ms"] == 40000
    assert not final["active_stint_certified"] and not final["canonical_tyre_life_certified"]


def test_completion_time_equality_requires_consumer_exclusion(tmp_path):
    rows, _ = replay(paths(tmp_path, opening() + lap()))
    eligible = [r for r in rows if r["observed_valid_completed"]]
    assert [r for r in eligible if r["available_ms"] < 86000] == []
    assert len([r for r in eligible if r["available_ms"] < 86001]) == 1
