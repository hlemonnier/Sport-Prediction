"""Synthetic mathematical and causal tests; no event data or fitting required."""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import numpy as np
import pytest

from research.experiments.boundary_20260908.sector_forecast.baselines import (
    GAUSSIAN_NAME, PACE_REFERENCE, predict_baselines,
)
from research.experiments.boundary_20260908.sector_forecast.execution_v2.features import (
    FEATURES, encode,
)


class Forbidden(Mapping):
    """Fail on *reading* a forbidden payload, not just converting its value."""

    def __init__(self, values, forbidden):
        self.values = values
        self.forbidden = set(forbidden)

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError(f"Unavailable payload was read: {key}")
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


def source(value, clock, sequence, **extras):
    return {"value": value, "available_ms": clock, "sequence": sequence,
            "recorded_ms": clock, "timestamp_regressed": False, **extras}


def sample(stage=2, n=12, contaminated=False):
    history = []
    for i in range(n):
        clock = 100_000 + i * 100_000
        sectors = [20 + .2 * i, 30 + .3 * i + (i % 2) * .1, 40 + .4 * i]
        lap = sum(sectors) + (i % 2) * .001
        row = {
            "event_key": 202201, "driver": "1", "completion_id": f"completion-{i}",
            "observed_valid_completed": True, "available_ms": clock,
            "packet_sequence": 10 * i + 3, "sectors_seconds": sectors,
            "full_lap_seconds": lap,
            "sector_sources": [source(v, clock - (2 - j) * 1000, 10 * i + j + 1)
                               for j, v in enumerate(sectors)],
            "full_lap_source": source(lap, clock, 10 * i + 3),
        }
        history.append(row)
    checkpoint = {
        "event_key": 202201, "driver": "1", "packet_sequence": 1000,
        "sector": stage, "sector_seconds": 22.8 if stage == 1 else 33.7,
        "checkpoint_ms": 2_000_000, "counter_seen": 18, "local_epoch": 19,
        "candidate_checkpoint": True,
        "received_driver_fields": {
            "Sector1": source(22.8, 2_000_000 if stage == 1 else 1_970_000, 1000 if stage == 1 else 998),
            "Sector2": source(33.7, 2_000_000, 1000),
            "Sector3": source(49., 1_450_000, 990),
            "Position": source("4", 1_990_000, 999),
            "SpeedI1": source("280.5", 1_970_000, 998),
            "SpeedI2": source(291.2, 2_000_000, 1000),
            "SpeedFL": source(299., 1_450_000, 990),
            "SpeedST": source(302., 1_990_000, 999),
        },
        "received_tyre_candidate": {
            "highest_observed_stint_index": 2,
            "fields": {"Compound": source("MEDIUM", 1_500_000, 1),
                       "New": source("false", 1_500_000, 1),
                       "StartLaps": source(2, 1_500_000, 1),
                       "TotalLaps": source(8, 1_900_000, 2),
                       "LapNumber": source(18, 1_900_000, 2)},
        },
    }
    context = {
        "event_key": 202201, "driver": "1", "packet_sequence": 1000, "sector": stage,
        "context_key": [202201, "1", 1000, stage], "checkpoint_ms": 2_000_000,
        "counter_seen": 18, "local_epoch": 19, "raw_epoch_start_ms": 1_500_000,
        "known_pit_contamination": contaminated, "known_neutralization_contamination": False,
        "known_asof_contamination": contaminated, "unknown_coverage": False,
        "current_in_pit": False, "fresh_pit_out_true_in_epoch": contaminated,
        "strictly_prior_control_snapshot": {"TrackStatus": source("1", 1_950_000, 7, processed_parse_gaps=0)},
    }
    prefix = [22.8] if stage == 1 else [22.8, 33.7]
    baseline = predict_baselines(history, prefix, known_asof_contamination=contaminated)
    return checkpoint, history, context, baseline


def same(left, right):
    assert tuple(left) == tuple(right) == FEATURES
    np.testing.assert_equal(list(left.values()), list(right.values()))


def test_contract_order_count_and_published_dependencies_remain_bound():
    path = Path(__file__).with_name("feature_contract.json")
    contract = json.loads(path.read_text())
    assert len(FEATURES) == len(set(FEATURES)) == contract["feature_count"] == 75
    assert list(FEATURES) == contract["ordered_features"]
    root = Path(__file__).resolve().parents[5]
    for relative, expected_hash in contract["source_files"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected_hash


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("contaminated", [False, True])
def test_exact_output_schema_types_and_missingness(stage, contaminated):
    case = sample(stage=stage, contaminated=contaminated)
    result = encode(*case)
    assert tuple(result) == FEATURES
    assert all(type(value) is float and not math.isinf(value) for value in result.values())
    assert math.isnan(result["s2_over_b4"]) == (stage == 1)
    assert result["known_asof_contamination"] == float(contaminated)
    assert result["compound_medium"] == 1
    assert sum(result["compound_" + category] for category in
               ("soft", "medium", "hard", "intermediate", "wet", "unknown")) == 1
    assert result["reported_tyre_new"] == 0  # "false" is not truthy True.


@pytest.mark.parametrize("stage", [1, 2])
def test_stage_excludes_later_sectors_and_all_final_labels(stage):
    checkpoint, history, context, baseline = sample(stage=stage)
    expected = encode(checkpoint, history, context, baseline)
    fields = checkpoint["received_driver_fields"]
    checkpoint["received_driver_fields"] = Forbidden(fields, [f"Sector{s}" for s in range(stage + 1, 4)])
    checkpoint = Forbidden({**checkpoint, "target_seconds": 0., "IsAccurate": False,
                            "TyreLife": -100, "year": 9999},
                           ["target_seconds", "IsAccurate", "TyreLife", "year"])
    history = [Forbidden({**row, "canonical_lap_number": 9999, "target": 0},
                         ["canonical_lap_number", "target"]) for row in history]
    same(expected, encode(checkpoint, history, context, baseline))


@pytest.mark.parametrize("kind", ["future", "equal", "other_driver", "other_event", "invalid"])
def test_disallowed_history_is_filtered_before_reading_any_payload(kind):
    checkpoint, history, context, baseline = sample()
    expected = encode(checkpoint, history, context, baseline)
    extra = dict(history[-1])
    if kind in ("future", "equal"):
        extra["available_ms"] = checkpoint["checkpoint_ms"] + (1 if kind == "future" else 0)
    elif kind == "other_driver":
        extra["driver"] = "99"
    elif kind == "other_event":
        extra["event_key"] = 202301
    else:
        extra["observed_valid_completed"] = False
    extra = Forbidden(extra, ["sectors_seconds", "full_lap_seconds", "sector_sources",
                              "full_lap_source", "completion_id", "packet_sequence"])
    same(expected, encode(checkpoint, [*history, extra], context, baseline))


@pytest.mark.parametrize("field", ["sector_sources", "full_lap_source"])
def test_completion_component_clocks_checked_before_durations(field):
    checkpoint, history, context, baseline = sample()
    row = dict(history[0])
    future_source = source(1., row["available_ms"] + 1, row["packet_sequence"])
    row[field] = [future_source] if field == "sector_sources" else future_source
    history[0] = Forbidden(row, ["sectors_seconds", "full_lap_seconds"])
    with pytest.raises(ValueError, match="component is unavailable"):
        encode(checkpoint, history, context, baseline)


def test_history_chronology_is_deterministic_and_inputs_are_immutable():
    case = sample()
    before = deepcopy(case)
    expected = encode(*case)
    assert case == before
    checkpoint, history, context, baseline = case
    shuffled = history[::2] + history[1::2]
    same(expected, encode(checkpoint, shuffled, context, baseline))
    assert case == before


def test_history_minimum_and_duplicate_support_fail_closed():
    checkpoint, history, context, baseline = sample()
    with pytest.raises(ValueError, match="at least three"):
        encode(checkpoint, history[:2], context, baseline)
    with pytest.raises(ValueError, match="duplicate completion"):
        encode(checkpoint, [*history, history[0]], context, baseline)


@pytest.mark.parametrize("field,value", [
    ("event_key", 202301), ("driver", "2"), ("packet_sequence", 1001),
    ("sector", 1), ("checkpoint_ms", 2_000_001), ("counter_seen", 17),
    ("local_epoch", 18), ("context_key", [202201, "1", 1000, 1]),
])
def test_exact_context_pairing(field, value):
    checkpoint, history, context, baseline = sample()
    context[field] = value
    with pytest.raises(ValueError, match="context"):
        encode(checkpoint, history, context, baseline)


@pytest.mark.parametrize("change", ["future_clock", "future_sequence", "old_epoch", "old_packet", "mismatch", "regressed"])
def test_current_prefix_requires_valid_asof_association(change):
    checkpoint, history, context, baseline = sample()
    field = checkpoint["received_driver_fields"]["Sector2"]
    if change == "future_clock":
        field["available_ms"] += 1
    elif change == "future_sequence":
        field["sequence"] += 1
    elif change == "old_epoch":
        field["available_ms"] = field["recorded_ms"] = context["raw_epoch_start_ms"] - 1
    elif change == "old_packet":
        field["sequence"] -= 1
    elif change == "mismatch":
        field["value"] += 1
    else:
        field["timestamp_regressed"] = True
    with pytest.raises(ValueError, match="prefix|current sector|checkpoint sector"):
        encode(checkpoint, history, context, baseline)


def test_same_clock_earlier_s1_packet_remains_available():
    checkpoint, history, context, baseline = sample()
    checkpoint["received_driver_fields"]["Sector1"].update(available_ms=2_000_000, recorded_ms=2_000_000)
    assert encode(checkpoint, history, context, baseline)["s1_over_b4"] > 0


@pytest.mark.parametrize("location", ["tyre", "speed", "track"])
@pytest.mark.parametrize("invalidity", ["future", "regressed", "bad_clock", "same_aux_clock"])
def test_optional_unavailable_sources_never_expose_values(location, invalidity):
    checkpoint, history, context, baseline = sample()
    clock = checkpoint["checkpoint_ms"] + 1
    field = source(123, clock, checkpoint["packet_sequence"])
    if invalidity == "regressed":
        field = source(123, clock - 2, 1, timestamp_regressed=True)
    elif invalidity == "bad_clock":
        field["available_ms"] = "invalid clock"
    elif invalidity == "same_aux_clock":
        if location == "speed":
            # Own-packet equality is intentionally allowed; a future sequence is not.
            field = source(123, clock - 1, checkpoint["packet_sequence"] + 1)
        else:
            field = source(123, clock - 1, 1)
    field = Forbidden(field, ["value"])
    if location == "tyre":
        checkpoint["received_tyre_candidate"]["fields"] = {"Compound": field}
    elif location == "speed":
        checkpoint["received_driver_fields"]["SpeedST"] = field
    else:
        context["strictly_prior_control_snapshot"]["TrackStatus"] = field
    result = encode(checkpoint, history, context, baseline)
    if location == "tyre":
        assert result["compound_unknown"] == 1
        assert math.isnan(result["reported_stint_index"])
    elif location == "speed":
        assert result["speed_st_missing"] == 1
        assert math.isnan(result["speed_st_kmh"])
    else:
        assert result["track_state_unknown"] == 1
        assert math.isnan(result["current_track_yellow"])


def test_independent_rolling_equations_preserve_rounding_residual_and_units():
    checkpoint, history, context, baseline = sample(n=12)
    result = encode(checkpoint, history, context, baseline)
    scale = baseline["points"][PACE_REFERENCE]
    for n in (3, 5, 10):
        rows = history[-n:]
        full = [row["full_lap_seconds"] for row in rows]
        prefixes = [sum(row["sectors_seconds"][:2]) for row in rows]
        remainder = [lap - prefix for lap, prefix in zip(full, prefixes)]
        for label, array in (("full", full), ("remainder", remainder)):
            center = median(array)
            deviation = median(abs(value - center) for value in array)
            slope = median((array[j] - array[i]) / (j - i)
                           for j in range(len(array)) for i in range(j))
            assert result[f"history{n}_{label}_median_over_b4"] == pytest.approx(center / scale)
            assert result[f"history{n}_{label}_mad_over_b4"] == pytest.approx(deviation / scale)
            assert result[f"history{n}_{label}_trend_over_b4"] == pytest.approx(slope / scale)
        center = median(prefixes)
        prefix_mad = median(abs(value - center) for value in prefixes)
        assert result[f"history{n}_prefix_deviation_over_mad"] == pytest.approx((22.8 + 33.7 - center) / max(prefix_mad, .1))
    assert result["last_completion_age_seconds"] == 800
    assert result["history10_span_seconds"] == 900
    assert result["epoch_age_seconds"] == 500
    assert result["track_status_age_seconds"] == 50
    assert result["compound_report_age_seconds"] == 500
    assert result["tyre_latest_report_age_seconds"] == 100


def test_prefix_deviation_uses_fixed_seconds_floor_for_identical_history():
    checkpoint, history, context, _ = sample(stage=1, n=3)
    for row in history:
        row["sectors_seconds"] = [20., 30., 40.]
        row["full_lap_seconds"] = 90.
    baseline = predict_baselines(history, [22.8], known_asof_contamination=False)
    result = encode(checkpoint, history, context, baseline)
    assert result["history3_prefix_deviation_over_mad"] == pytest.approx(28.)
    assert result["history3_full_trend_over_b4"] == 0.
    assert result["history3_full_mad_over_b4"] == 0.


def test_same_window_remainder_preserves_received_rounding_residual():
    checkpoint, history, context, _ = sample(stage=2, n=3)
    for row in history:
        row["full_lap_seconds"] = sum(row["sectors_seconds"]) + .002
    baseline = predict_baselines(history, [22.8, 33.7], known_asof_contamination=False)
    result = encode(checkpoint, history, context, baseline)
    scale = baseline["points"][PACE_REFERENCE]
    difference = result["history3_remainder_median_over_b4"] - result["history5_s3_median_over_b4"]
    assert difference * scale == pytest.approx(.002, abs=1e-12)


def test_history_older_than_last_ten_only_changes_support_count():
    checkpoint, history, context, baseline = sample(n=12)
    result = encode(checkpoint, history, context, baseline)
    recent = history[-10:]
    recent_baseline = predict_baselines(recent, [22.8, 33.7], known_asof_contamination=False)
    comparison = encode(checkpoint, recent, context, recent_baseline)
    result["history_count"] = comparison["history_count"]
    same(result, comparison)


def test_scale_equivariance_except_declared_seconds_and_support():
    checkpoint, history, context, baseline = sample()
    expected = encode(checkpoint, history, context, baseline)
    factor = 2.
    for row in history:
        row["sectors_seconds"] = [factor * value for value in row["sectors_seconds"]]
        # The doubled rounding discrepancy stays below the published .003s bound.
        row["full_lap_seconds"] *= factor
    checkpoint["sector_seconds"] *= factor
    for key in ("Sector1", "Sector2"):
        checkpoint["received_driver_fields"][key]["value"] *= factor
    baseline["points"] = {key: factor * value for key, value in baseline["points"].items()}
    scaled = encode(checkpoint, history, context, baseline)
    expected["b4_lap_seconds"] *= factor
    for key in FEATURES:
        assert scaled[key] == pytest.approx(expected[key], nan_ok=True)


@pytest.mark.parametrize("value,expected", [(True, 1.), (False, 0.), ("true", 1.), ("false", 0.),
                                           ("False", math.nan), (1, math.nan), (0, math.nan), (None, math.nan)])
def test_reported_new_uses_exact_native_boolean_contract(value, expected):
    checkpoint, history, context, baseline = sample()
    checkpoint["received_tyre_candidate"]["fields"]["New"]["value"] = value
    result = encode(checkpoint, history, context, baseline)
    assert result["reported_tyre_new"] == pytest.approx(expected, nan_ok=True)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1, True, "n/a", ["SOFT"]])
def test_unsupported_optional_metadata_never_becomes_infinity(value):
    checkpoint, history, context, baseline = sample()
    checkpoint["received_driver_fields"]["Position"]["value"] = value
    checkpoint["received_driver_fields"]["SpeedST"]["value"] = value
    tyre = checkpoint["received_tyre_candidate"]
    tyre["highest_observed_stint_index"] = value
    for key in ("Compound", "StartLaps", "TotalLaps"):
        tyre["fields"][key]["value"] = value
    result = encode(checkpoint, history, context, baseline)
    assert result["compound_unknown"] == result["speed_st_missing"] == 1
    assert all(not math.isinf(number) for number in result.values())
    for key in ("position", "speed_st_kmh", "reported_stint_index", "reported_start_laps", "reported_total_laps"):
        assert math.isnan(result[key])


@pytest.mark.parametrize("change", ["history_count", "prefix_sector_count", "known_asof_contamination", "reference_branch", "point"])
def test_misbound_baselines_fail_closed(change):
    checkpoint, history, context, baseline = sample()
    if change == "point":
        baseline["points"][PACE_REFERENCE] = float("inf")
    else:
        baseline["diagnostics"][change] = {"history_count": 11, "prefix_sector_count": 1,
            "known_asof_contamination": True, "reference_branch": "whole_lap"}[change]
    with pytest.raises(ValueError, match="baseline"):
        encode(checkpoint, history, context, baseline)


def test_gaussian_fallback_is_observed_status_not_inferred_from_point_equality():
    checkpoint, history, context, baseline = sample()
    baseline["points"][GAUSSIAN_NAME] = baseline["points"][PACE_REFERENCE]
    assert encode(checkpoint, history, context, baseline)["gaussian_fallback"] == 0
    baseline["diagnostics"]["gaussian"]["status"] = "fallback_invalid_conditional_remainder"
    assert encode(checkpoint, history, context, baseline)["gaussian_fallback"] == 1


def test_identifiers_are_pairing_only_not_numeric_features():
    checkpoint, history, context, baseline = sample()
    expected = encode(checkpoint, history, context, baseline)
    for row in (checkpoint, context, *history):
        row["event_key"] = 999999
        row["driver"] = "new-private-identity"
    context["context_key"][:2] = [999999, "new-private-identity"]
    same(expected, encode(checkpoint, history, context, baseline))


@pytest.mark.parametrize("field", ["known_pit_contamination", "known_neutralization_contamination",
                                    "known_asof_contamination", "unknown_coverage", "fresh_pit_out_true_in_epoch"])
def test_context_flags_require_explicit_boolean(field):
    checkpoint, history, context, baseline = sample()
    context[field] = "false"
    with pytest.raises(ValueError, match="explicit boolean"):
        encode(checkpoint, history, context, baseline)


def test_inconsistent_contamination_union_rejected():
    checkpoint, history, context, baseline = sample()
    context["known_pit_contamination"] = True
    with pytest.raises(ValueError, match="union"):
        encode(checkpoint, history, context, baseline)


def test_nonrepresentable_prefix_total_rejected_without_infinite_feature():
    checkpoint, history, context, baseline = sample()
    checkpoint["sector_seconds"] = 1e308
    for field in ("Sector1", "Sector2"):
        checkpoint["received_driver_fields"][field]["value"] = 1e308
    with pytest.raises(ValueError, match="prefix total"):
        encode(checkpoint, history, context, baseline)
