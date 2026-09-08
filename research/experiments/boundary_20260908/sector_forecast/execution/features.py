"""Fixed numeric sector-prefix features, without outcomes or later-sector reads."""
from collections.abc import Mapping
from numbers import Integral, Real
import math

import numpy as np

from research.experiments.boundary_20260908.sector_forecast.baselines import (
    GAUSSIAN_NAME, PACE_REFERENCE, REFERENCE_NAMES, _templates,
)

FEATURES = (
    "stage", "observed_counter", "position", "b4_lap_seconds", "s1_over_b4", "s2_over_b4",
    "prefix_over_b4", "history_count", "last_completion_age_seconds", "history10_span_seconds",
    *(f"history{n}_{name}" for n in (3, 5, 10) for name in (
        "full_median_over_b4", "full_mad_over_b4", "full_trend_over_b4", "prefix_median_over_b4",
        "remainder_median_over_b4", "remainder_mad_over_b4", "remainder_trend_over_b4", "prefix_deviation_over_mad")),
    *(f"history5_s{s}_{stat}_over_b4" for s in (1, 2, 3) for stat in ("median", "mad")),
    "last_template_delta_over_b4", "joint_median5_delta_over_b4", "weighted_median10_delta_over_b4", "gaussian_delta_over_b4",
    "known_pit_contamination", "known_neutralization_contamination", "known_asof_contamination", "unknown_coverage",
    "epoch_age_seconds", "current_in_pit", "fresh_pit_out_true", "current_track_yellow", "track_status_age_seconds", "track_state_unknown",
    "compound_soft", "compound_medium", "compound_hard", "compound_intermediate", "compound_wet", "compound_unknown",
    "reported_tyre_new", "reported_stint_index", "reported_start_laps", "reported_total_laps", "tyre_latest_report_age_seconds", "compound_report_age_seconds",
    *(f"speed_{name}_{stat}" for name in ("i1", "i2", "fl", "st") for stat in ("kmh", "missing")),
    "gaussian_fallback",
)
NAN = float("nan")
PREFIX_MAD_FLOOR_SECONDS = .1


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _clock(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative millisecond clock")
    return float(value)


def _number(value):
    """Optional native numeric/string metadata; booleans are never numbers."""
    if isinstance(value, (bool, np.bool_)) or value is None:
        return NAN
    try:
        value = float(value)
        return value if math.isfinite(value) else NAN
    except (TypeError, ValueError, OverflowError):
        return NAN


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} requires numeric positive seconds")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} requires finite positive seconds")
    return value


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} requires an explicit boolean")
    return value


def _identity(row):
    event = _integer(row["event_key"], "event_key")
    driver = row["driver"]
    if not isinstance(driver, str) or not driver:
        raise ValueError("driver identity must be a nonempty string")
    sequence = _integer(row["packet_sequence"], "packet_sequence")
    stage = _integer(row["sector"], "sector")
    if stage not in (1, 2):
        raise ValueError("only an S1 or S1+S2 prefix may be encoded")
    return event, driver, sequence, stage


def _available(source, instant, *, sequence=None, strictly=False):
    """Check source metadata before a possibly poisoned/unavailable value."""
    if not isinstance(source, Mapping):
        return False
    raw_stamp = source.get("available_ms")
    stamp = _number(raw_stamp) if isinstance(raw_stamp, Real) else NAN
    if not math.isfinite(stamp) or stamp < 0 or (stamp >= instant if strictly else stamp > instant):
        return False
    if sequence is not None:
        source_sequence = source.get("sequence", source.get("packet_sequence"))
        if (isinstance(source_sequence, (bool, np.bool_)) or not isinstance(source_sequence, Integral)
                or not 0 <= source_sequence <= sequence):
            return False
    if source.get("timestamp_regressed", False) is not False:
        return False
    raw_clock = source.get("recorded_ms", stamp)
    raw = _number(raw_clock) if isinstance(raw_clock, Real) else NAN
    return math.isfinite(raw) and 0 <= raw <= stamp


def _field(fields, key, instant, *, sequence=None, strictly=False):
    source = fields.get(key)
    return source if _available(source, instant, sequence=sequence, strictly=strictly) else None


def _history(records, event, driver, instant):
    selected = []
    for row in records:
        # Metadata filtering precedes all duration/source-value access. Appended
        # future rows or other drivers cannot affect earlier features.
        if row["event_key"] != event or row["driver"] != driver or row["observed_valid_completed"] is not True:
            continue
        available = _clock(row["available_ms"], "history available_ms")
        if available >= instant:
            continue
        sequence = _integer(row["packet_sequence"], "history packet_sequence")
        identity = row["completion_id"]
        if not isinstance(identity, str) or not identity:
            raise ValueError("eligible history requires completion identity")
        for source in [*(row.get("sector_sources") or []), row.get("full_lap_source")]:
            if source is not None and not _available(source, available, sequence=sequence):
                raise ValueError("historical component is unavailable at its completion clock")
        selected.append((available, sequence, identity, row))
    selected.sort(key=lambda item: item[:3])
    if len(selected) < 3:
        raise ValueError("at least three strictly earlier own valid completions are required")
    if len({r[2] for r in selected}) != len(selected):
        raise ValueError("duplicate completion identities cannot count twice")
    ordered = [item[3] for item in selected]
    sectors, laps = _templates(ordered)
    return ordered, sectors, laps, np.asarray([item[0] for item in selected], dtype=float)


def _median(values):
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    return float(values[n // 2]) if n % 2 else float(.5 * values[n // 2 - 1] + .5 * values[n // 2])


def _mad(values):
    return _median(np.abs(np.asarray(values, dtype=float) - _median(values)))


def _trend(values):
    values = np.asarray(values, dtype=float)
    return _median([(values[j] - values[i]) / (j - i) for i in range(len(values)) for j in range(i + 1, len(values))])


def _divide(numerator, denominator):
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        value = float(np.divide(numerator, denominator))
    return value if math.isfinite(value) else NAN


def encode(checkpoint, history, context, baseline_result):
    """Return the 75 frozen columns, in FEATURES order, without mutating inputs."""
    event, driver, sequence, stage = _identity(checkpoint)
    if checkpoint["candidate_checkpoint"] is not True:
        raise ValueError("encoder requires a frozen pilot candidate checkpoint")
    if _identity(context) != (event, driver, sequence, stage):
        raise ValueError("context identity does not match checkpoint")
    if "context_key" in context and context["context_key"] != [event, driver, sequence, stage]:
        raise ValueError("context_key does not match checkpoint")
    instant = _clock(checkpoint["checkpoint_ms"], "checkpoint_ms")
    if _clock(context["checkpoint_ms"], "context checkpoint_ms") != instant:
        raise ValueError("context clock does not match checkpoint")
    for field in ("counter_seen", "local_epoch"):
        if context[field] != checkpoint[field]:
            raise ValueError(f"context {field} does not match checkpoint")
    raw_counter = checkpoint["counter_seen"]
    counter = NAN if raw_counter is None else float(_integer(raw_counter, "observed counter"))
    epoch_start = context.get("raw_epoch_start_ms")
    if epoch_start is not None:
        epoch_start = _clock(epoch_start, "raw epoch start")
        if epoch_start > instant:
            raise ValueError("epoch start cannot follow checkpoint")
    fields = checkpoint["received_driver_fields"]
    prefix = []
    # Deliberately no iteration over received_driver_fields: later-sector slots
    # may hold unavailable/poisoned values that this stage must never inspect.
    for sector in range(1, stage + 1):
        source = _field(fields, f"Sector{sector}", instant, sequence=sequence)
        if source is None:
            raise ValueError("current prefix source is missing or unavailable")
        if epoch_start is not None and source["available_ms"] < epoch_start:
            raise ValueError("current prefix source predates observed counter epoch")
        value = _positive(source["value"], "current sector")
        if sector == stage:
            if source.get("sequence", source.get("packet_sequence")) != sequence:
                raise ValueError("current sector does not originate in checkpoint packet")
            if value != _positive(checkpoint["sector_seconds"], "checkpoint sector"):
                raise ValueError("checkpoint sector value disagrees with received source")
        prefix.append(value)
    try:
        prefix_total = math.fsum(prefix)
    except OverflowError as exc:
        raise ValueError("current prefix total must be finite") from exc
    if not math.isfinite(prefix_total):
        raise ValueError("current prefix total must be finite")
    ordered, sectors, laps, clocks = _history(history, event, driver, instant)
    pit = _boolean(context["known_pit_contamination"], "pit contamination")
    neutralized = _boolean(context["known_neutralization_contamination"], "neutralization contamination")
    contaminated = _boolean(context["known_asof_contamination"], "as-of contamination")
    if contaminated != (pit or neutralized):
        raise ValueError("context contamination union is inconsistent")
    unknown = _boolean(context["unknown_coverage"], "unknown coverage")
    fresh_out = _boolean(context["fresh_pit_out_true_in_epoch"], "fresh pitout")
    diagnostics = baseline_result["diagnostics"]
    if (_integer(diagnostics["history_count"], "baseline history count") != len(ordered)
            or _integer(diagnostics["prefix_sector_count"], "baseline prefix count") != stage
            or _boolean(diagnostics["known_asof_contamination"], "baseline contamination") != contaminated):
        raise ValueError("baseline diagnostics do not match encoder inputs")
    expected_branch = "whole_lap" if contaminated else "observed_prefix_plus_remainder"
    if diagnostics["reference_branch"] != expected_branch:
        raise ValueError("baseline branch is inconsistent with received contamination")
    points = {name: _positive(baseline_result["points"][name], "baseline point")
              for name in (*REFERENCE_NAMES, GAUSSIAN_NAME)}
    scale = points[PACE_REFERENCE]
    values = {name: NAN for name in FEATURES}
    values.update(stage=float(stage), observed_counter=counter, b4_lap_seconds=scale,
                  s1_over_b4=_divide(prefix[0], scale),
                  s2_over_b4=_divide(prefix[1], scale) if stage == 2 else NAN,
                  prefix_over_b4=_divide(prefix_total, scale), history_count=float(len(ordered)),
                  last_completion_age_seconds=(instant - clocks[-1]) / 1000,
                  history10_span_seconds=(clocks[-1] - clocks[-min(10, len(clocks))]) / 1000)
    position = _field(fields, "Position", instant, sequence=sequence)
    if position is not None:
        number = _number(position.get("value"))
        values["position"] = number if number > 0 and number.is_integer() else NAN
    historical_prefix = sectors[:, :stage].sum(1)
    remainder = laps - historical_prefix
    if not np.isfinite(remainder).all() or not (remainder > 0).all():
        raise ValueError("historical remainder must be positive and finite")
    for n in (3, 5, 10):
        full, past_prefix, remaining = laps[-n:], historical_prefix[-n:], remainder[-n:]
        names = {"full_median_over_b4": _median(full), "full_mad_over_b4": _mad(full),
                 "full_trend_over_b4": _trend(full), "prefix_median_over_b4": _median(past_prefix),
                 "remainder_median_over_b4": _median(remaining), "remainder_mad_over_b4": _mad(remaining),
                 "remainder_trend_over_b4": _trend(remaining)}
        for name, value in names.items():
            values[f"history{n}_{name}"] = _divide(value, scale)
        values[f"history{n}_prefix_deviation_over_mad"] = _divide(prefix_total - _median(past_prefix), max(_mad(past_prefix), PREFIX_MAD_FLOOR_SECONDS))
    for sector in (1, 2, 3):
        observed = sectors[-5:, sector - 1]
        values[f"history5_s{sector}_median_over_b4"] = _divide(_median(observed), scale)
        values[f"history5_s{sector}_mad_over_b4"] = _divide(_mad(observed), scale)
    for feature, reference in zip(("last_template_delta_over_b4", "joint_median5_delta_over_b4",
                                   "weighted_median10_delta_over_b4", "gaussian_delta_over_b4"),
                                  (*REFERENCE_NAMES[:3], GAUSSIAN_NAME)):
        values[feature] = _divide(points[reference] - scale, scale)
    values.update(known_pit_contamination=float(pit), known_neutralization_contamination=float(neutralized),
                  known_asof_contamination=float(contaminated), unknown_coverage=float(unknown),
                  epoch_age_seconds=(instant - epoch_start) / 1000 if epoch_start is not None else NAN,
                  current_in_pit=float(context["current_in_pit"]) if type(context["current_in_pit"]) is bool else NAN,
                  fresh_pit_out_true=float(fresh_out))
    track = context.get("strictly_prior_control_snapshot", {}).get("TrackStatus")
    track_known = _available(track, instant, strictly=True)
    if track_known and track.get("processed_parse_gaps", 0) != 0:
        track_known = False
    track_code = track.get("value") if track_known else None
    track_known = track_known and track_code in ("1", "2", "4", "5", "6", "7")
    values["track_state_unknown"] = float(not track_known)
    if track_known:
        values["current_track_yellow"] = float(track_code == "2")
        values["track_status_age_seconds"] = (instant - track["available_ms"]) / 1000
    tyre = checkpoint.get("received_tyre_candidate", {})
    tyre_fields = tyre.get("fields", {})
    received = {key: source for key in ("Compound", "New", "StartLaps", "TotalLaps", "LapNumber")
                if (source := _field(tyre_fields, key, instant, strictly=True)) is not None}
    compound_source = received.get("Compound")
    compound = compound_source.get("value") if compound_source is not None else None
    compound = compound if isinstance(compound, str) else None
    categories = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")
    for category in categories:
        values["compound_" + category.lower()] = float(compound == category)
    values["compound_unknown"] = float(compound not in categories)
    if compound_source is not None:
        values["compound_report_age_seconds"] = (instant - compound_source["available_ms"]) / 1000
    if received:
        index = _number(tyre.get("highest_observed_stint_index"))
        values["reported_stint_index"] = index if index >= 0 and index.is_integer() else NAN
        values["tyre_latest_report_age_seconds"] = min((instant - s["available_ms"]) / 1000 for s in received.values())
    if "New" in received:
        new = received["New"].get("value")
        if type(new) is bool:
            values["reported_tyre_new"] = float(new)
        elif isinstance(new, str) and new in ("true", "false"):
            values["reported_tyre_new"] = float(new == "true")
    for key, feature in (("StartLaps", "reported_start_laps"), ("TotalLaps", "reported_total_laps")):
        if key in received:
            number = _number(received[key].get("value"))
            values[feature] = number if number >= 0 else NAN
    for key in ("I1", "I2", "FL", "ST"):
        source = _field(fields, "Speed" + key, instant, sequence=sequence)
        speed = _number(source.get("value")) if source is not None else NAN
        speed = speed if speed >= 0 else NAN
        values[f"speed_{key.lower()}_kmh"] = speed
        values[f"speed_{key.lower()}_missing"] = float(not math.isfinite(speed))
    gaussian_status = diagnostics["gaussian"]["status"]
    if not isinstance(gaussian_status, str):
        raise ValueError("Gaussian status must be explicit")
    values["gaussian_fallback"] = float(gaussian_status.startswith("fallback_"))
    # Missing/unsupported optional metadata and nonrepresentable transforms
    # remain missing. No infinities or implicit object/string columns escape.
    return {name: float(value) if math.isfinite(value) else NAN for name, value in values.items()}
