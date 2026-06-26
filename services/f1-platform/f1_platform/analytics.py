"""Derived analytics for reduced F1 session snapshots.

The live reducer keeps a compact current state. This module turns that state
into durable, query-oriented analytics without depending on pandas or FastF1 in
the near-live path.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import exp, isfinite, sqrt
from statistics import median
from typing import Any

from .schemas import DriverState, LapPoint, SessionSnapshot, StintSegment
from .time import utc_now_iso

TYRE_DEGRADATION_ANALYTIC_NAME = "tyre_degradation_v1"
WEATHER_EVOLUTION_ANALYTIC_NAME = "weather_evolution_v1"
PACE_ANALYSIS_ANALYTIC_NAME = "pace_analysis_v1"
BATTLE_DASHBOARD_ANALYTIC_NAME = "battle_dashboard_v1"


@dataclass(slots=True)
class _LapSample:
    driver_number: int
    lap: int
    lap_time: float
    compound: str
    tyre_age: int
    field_lap_median: float
    driver_baseline: float = 0.0
    adjusted_pace: float = 0.0


def build_projection_analytics(snapshot: SessionSnapshot) -> dict[str, dict[str, Any]]:
    """Build the derived analytics payloads persisted by the projection store."""

    return {
        "projection_summary": {
            "version": 1,
            "generatedAt": utc_now_iso(),
            "sessionMetadata": 1 if snapshot.session_info else 0,
            "drivers": len(snapshot.drivers),
            "laps": len(snapshot.lap_chart),
            "stints": len(snapshot.strategy_timeline),
            "pitStops": len(snapshot.pit_stops),
            "overtakes": len(snapshot.overtakes),
            "sessionResults": len(snapshot.session_results),
            "customMicroSectors": len(snapshot.custom_micro_sectors),
            "weatherSamples": len(snapshot.weather_samples),
            "predictions": len(snapshot.predictions),
        },
        TYRE_DEGRADATION_ANALYTIC_NAME: build_tyre_degradation_analytics(snapshot),
        WEATHER_EVOLUTION_ANALYTIC_NAME: build_weather_evolution_analytics(snapshot),
        PACE_ANALYSIS_ANALYTIC_NAME: build_pace_analysis_analytics(snapshot),
        BATTLE_DASHBOARD_ANALYTIC_NAME: build_battle_dashboard_analytics(snapshot),
    }


def build_pace_analysis_analytics(snapshot: SessionSnapshot) -> dict[str, Any]:
    """Summarize reduced lap pace without pretending it is a telemetry model."""

    generated_at = utc_now_iso()
    lap_times_by_driver: dict[int, list[LapPoint]] = {}
    for point in snapshot.lap_chart:
        lap_time = _finite_float(point.value)
        if lap_time is not None and lap_time > 0:
            lap_times_by_driver.setdefault(point.driver_number, []).append(point)

    if not lap_times_by_driver:
        return {
            "version": 1,
            "generatedAt": generated_at,
            "status": "no_lap_data",
            "driverCount": 0,
            "fieldSeries": [],
            "drivers": [],
        }

    drivers_by_number = {driver.driver_number: driver for driver in snapshot.drivers}
    rows: list[dict[str, Any]] = []
    for driver_number, points in lap_times_by_driver.items():
        points.sort(key=lambda point: point.lap)
        values = [float(point.value) for point in points]
        recent_values = values[-3:]
        driver = drivers_by_number.get(driver_number)
        rows.append(
            {
                "driverNumber": driver_number,
                "acronym": driver.acronym if driver else None,
                "teamName": driver.team_name if driver else None,
                "position": driver.position if driver else None,
                "compound": driver.current_compound if driver else None,
                "lapCount": len(values),
                "firstLap": points[0].lap,
                "lastLap": points[-1].lap,
                "bestLapTime": _round(min(values)),
                "lastLapTime": _round(values[-1]),
                "averageLapTime": _round(sum(values) / len(values)),
                "medianLapTime": _round(median(values)),
                "rollingMedianLast3": _round(median(recent_values)),
                "consistencyStdSeconds": _round(_stddev(values)),
                "trendLastVsFirst": _round(values[-1] - values[0]) if len(values) >= 2 else None,
                "lapTimes": [{"lap": point.lap, "lapTime": _round(point.value)} for point in points[-20:]],
            }
        )

    rows.sort(
        key=lambda row: (
            row["medianLapTime"] if row["medianLapTime"] is not None else 1_000_000,
            row["position"] if row["position"] is not None else 10_000,
            row["driverNumber"],
        )
    )

    return {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ok",
        "method": "reduced lap chart pace summary; telemetry comparison remains FastF1 post-session",
        "driverCount": len(rows),
        "fieldSeries": _field_pace_series(snapshot.lap_chart),
        "drivers": rows,
    }


def build_battle_dashboard_analytics(snapshot: SessionSnapshot) -> dict[str, Any]:
    """Identify adjacent-driver battles and DRS-train candidates from reduced state."""

    generated_at = utc_now_iso()
    ordered = [
        driver
        for driver in sorted(
            snapshot.drivers,
            key=lambda item: (
                item.position if item.position is not None else 10_000,
                item.driver_number,
            ),
        )
        if driver.position is not None
    ]
    if len(ordered) < 2:
        return {
            "version": 1,
            "generatedAt": generated_at,
            "status": "insufficient_driver_order",
            "battleCount": 0,
            "activeOvertakeWindows": 0,
            "battles": [],
            "drsTrains": [],
        }

    recent_pace = _recent_pace_by_driver(snapshot.lap_chart)
    battles = []
    for index in range(1, len(ordered)):
        ahead = ordered[index - 1]
        chaser = ordered[index]
        gap_seconds = _adjacent_gap_seconds(ahead, chaser)
        pace_delta = _pace_delta_seconds(recent_pace, ahead.driver_number, chaser.driver_number)
        probability = _overtake_window_probability(chaser, ahead, gap_seconds, pace_delta)
        window_state = _battle_window_state(gap_seconds, probability)
        battles.append(
            {
                "ahead": _driver_ref(ahead),
                "chaser": _driver_ref(chaser),
                "gapSeconds": _round(gap_seconds),
                "recentPaceDeltaSeconds": _round(pace_delta),
                "chaserDrsActive": _drs_active(chaser.drs),
                "tyreAgeDelta": _optional_tyre_age_delta(chaser, ahead),
                "overtakeWindowProbability": probability,
                "windowState": window_state,
                "reason": _battle_reason(gap_seconds, pace_delta, chaser),
            }
        )

    battles.sort(key=lambda item: item["overtakeWindowProbability"], reverse=True)
    return {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ok",
        "method": "adjacent order gaps, recent reduced lap pace, DRS state and tyre-age heuristic",
        "battleCount": len(battles),
        "activeOvertakeWindows": sum(1 for item in battles if item["windowState"] == "active"),
        "battles": battles,
        "drsTrains": _drs_trains(ordered),
    }


def _field_pace_series(points: list[LapPoint]) -> list[dict[str, Any]]:
    by_lap: dict[int, list[float]] = {}
    for point in points:
        lap_time = _finite_float(point.value)
        if lap_time is not None and lap_time > 0:
            by_lap.setdefault(point.lap, []).append(lap_time)

    series: list[dict[str, Any]] = []
    for lap, values in sorted(by_lap.items()):
        series.append(
            {
                "lap": lap,
                "sampleCount": len(values),
                "medianLapTime": _round(median(values)),
                "spreadSeconds": _round(max(values) - min(values)) if values else None,
            }
        )
    return series[-120:]


def _recent_pace_by_driver(points: list[LapPoint]) -> dict[int, float]:
    by_driver: dict[int, list[LapPoint]] = {}
    for point in points:
        lap_time = _finite_float(point.value)
        if lap_time is not None and lap_time > 0:
            by_driver.setdefault(point.driver_number, []).append(point)

    recent: dict[int, float] = {}
    for driver_number, driver_points in by_driver.items():
        driver_points.sort(key=lambda point: point.lap)
        values = [float(point.value) for point in driver_points[-3:]]
        if values:
            recent[driver_number] = median(values)
    return recent


def _adjacent_gap_seconds(ahead: DriverState, chaser: DriverState) -> float | None:
    interval_gap = _gap_seconds(chaser.interval)
    if interval_gap is not None:
        return interval_gap
    chaser_leader_gap = _gap_seconds(chaser.gap_to_leader)
    ahead_leader_gap = _gap_seconds(ahead.gap_to_leader)
    if chaser_leader_gap is None or ahead_leader_gap is None:
        return None
    return max(0.0, chaser_leader_gap - ahead_leader_gap)


def _pace_delta_seconds(recent_pace: dict[int, float], ahead_number: int, chaser_number: int) -> float | None:
    ahead_pace = recent_pace.get(ahead_number)
    chaser_pace = recent_pace.get(chaser_number)
    if ahead_pace is None or chaser_pace is None:
        return None
    return chaser_pace - ahead_pace


def _overtake_window_probability(
    chaser: DriverState,
    ahead: DriverState,
    gap_seconds: float | None,
    pace_delta: float | None,
) -> float:
    if gap_seconds is None:
        return 0.05
    gap_score = max(0.0, min(1.0, (2.5 - gap_seconds) / 2.5))
    pace_score = 0.0 if pace_delta is None else max(0.0, min(1.0, (-pace_delta + 0.05) / 0.55))
    tyre_delta = _optional_tyre_age_delta(chaser, ahead)
    tyre_score = 0.0 if tyre_delta is None else max(0.0, min(1.0, tyre_delta / 10.0))
    drs_bonus = 0.15 if gap_seconds <= 1.0 or _drs_active(chaser.drs) else 0.0
    probability = 0.04 + 0.46 * gap_score + 0.24 * pace_score + 0.11 * tyre_score + drs_bonus
    if gap_seconds > 4.0 and pace_score <= 0.05:
        probability *= 0.35
    return _round(max(0.01, min(0.95, probability))) or 0.01


def _battle_window_state(gap_seconds: float | None, probability: float) -> str:
    if gap_seconds is not None and gap_seconds <= 1.0:
        return "active"
    if probability >= 0.5:
        return "active"
    if probability >= 0.25:
        return "building"
    return "distant"


def _battle_reason(gap_seconds: float | None, pace_delta: float | None, chaser: DriverState) -> str:
    pieces: list[str] = []
    if gap_seconds is not None:
        pieces.append(f"{gap_seconds:.3f}s gap")
    if pace_delta is not None:
        if pace_delta < -0.05:
            pieces.append(f"chaser faster by {abs(pace_delta):.3f}s")
        elif pace_delta > 0.05:
            pieces.append(f"chaser slower by {pace_delta:.3f}s")
        else:
            pieces.append("matched recent pace")
    if _drs_active(chaser.drs):
        pieces.append("DRS active")
    return ", ".join(pieces) if pieces else "limited reduced-state evidence"


def _drs_trains(ordered: list[DriverState]) -> list[dict[str, Any]]:
    trains: list[list[DriverState]] = []
    current: list[DriverState] = []
    for index, driver in enumerate(ordered):
        if index == 0:
            current = [driver]
            continue
        gap = _adjacent_gap_seconds(ordered[index - 1], driver)
        if gap is not None and gap <= 1.2:
            current.append(driver)
        else:
            if len(current) >= 2:
                trains.append(current)
            current = [driver]
    if len(current) >= 2:
        trains.append(current)

    return [
        {
            "size": len(train),
            "leader": _driver_ref(train[0]),
            "drivers": [_driver_ref(driver) for driver in train],
            "maxAdjacentGapSeconds": _round(
                max(
                    _adjacent_gap_seconds(train[index - 1], train[index]) or 0.0
                    for index in range(1, len(train))
                )
            ),
        }
        for train in trains
    ]


def _driver_ref(driver: DriverState) -> dict[str, Any]:
    return {
        "driverNumber": driver.driver_number,
        "acronym": driver.acronym,
        "teamName": driver.team_name,
        "position": driver.position,
        "compound": driver.current_compound,
        "tyreAge": driver.tyre_age,
    }


def _optional_tyre_age_delta(chaser: DriverState, ahead: DriverState) -> int | None:
    if chaser.tyre_age is None or ahead.tyre_age is None:
        return None
    return ahead.tyre_age - chaser.tyre_age


def _drs_active(drs: int | None) -> bool:
    return drs is not None and drs >= 10


def _gap_seconds(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or "leader" in text:
        return 0.0 if "leader" in text else None
    text = text.replace("+", "").replace("s", "").strip()
    try:
        number = float(text)
    except ValueError:
        return None
    return number if isfinite(number) else None


def build_weather_evolution_analytics(snapshot: SessionSnapshot) -> dict[str, Any]:
    """Summarize weather and track-temperature evolution from retained samples."""

    generated_at = utc_now_iso()
    samples = [_weather_sample_payload(sample, index) for index, sample in enumerate(snapshot.weather_samples)]
    if not samples:
        return {
            "version": 1,
            "generatedAt": generated_at,
            "status": "no_weather_samples",
            "sampleCount": 0,
            "latest": None,
            "trackTemperatureDelta": None,
            "airTemperatureDelta": None,
            "windSpeedDelta": None,
            "rainfallDetected": False,
            "maxRainfall": None,
            "series": [],
        }

    latest = samples[-1]
    first_track = _first_numeric(samples, "trackTemperature")
    latest_track = _last_numeric(samples, "trackTemperature")
    first_air = _first_numeric(samples, "airTemperature")
    latest_air = _last_numeric(samples, "airTemperature")
    first_wind = _first_numeric(samples, "windSpeed")
    latest_wind = _last_numeric(samples, "windSpeed")
    rainfall_values = [value for value in (_finite_float(sample.get("rainfall")) for sample in samples) if value is not None]
    return {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ok",
        "sampleCount": len(samples),
        "latest": latest,
        "trackTemperatureDelta": _delta(latest_track, first_track),
        "airTemperatureDelta": _delta(latest_air, first_air),
        "windSpeedDelta": _delta(latest_wind, first_wind),
        "rainfallDetected": any(value > 0 for value in rainfall_values),
        "maxRainfall": _round(max(rainfall_values)) if rainfall_values else None,
        "series": samples[-120:],
    }


def build_tyre_degradation_analytics(snapshot: SessionSnapshot) -> dict[str, Any]:
    """Estimate tyre degradation from reduced lap/stint state.

    The model deliberately avoids raw lap-time-vs-tyre-age regression. It:
    joins lap points to stints, derives tyre age from stint start, filters common
    dirty laps, removes field-level lap median pace, then removes each driver's
    baseline delta before fitting compound-level trends.
    """

    generated_at = utc_now_iso()
    filters = {
        "invalid_lap_time": 0,
        "no_matching_stint": 0,
        "pit_boundary": 0,
        "race_status": 0,
        "field_anomaly": 0,
    }
    if not snapshot.lap_chart or not snapshot.strategy_timeline:
        return _empty_tyre_degradation(
            generated_at,
            status="insufficient_lap_or_stint_data",
            filters=filters,
        )

    stints_by_driver = _stints_by_driver(snapshot.strategy_timeline)
    flagged_laps = _flagged_laps(snapshot.race_control)
    field_lap_medians = _field_lap_medians(snapshot.lap_chart)
    candidate_samples: list[_LapSample] = []

    for point in snapshot.lap_chart:
        lap_time = _finite_float(point.value)
        if lap_time is None or lap_time <= 0.0:
            filters["invalid_lap_time"] += 1
            continue
        stint = _matching_stint(stints_by_driver.get(point.driver_number, []), point.lap)
        if stint is None:
            filters["no_matching_stint"] += 1
            continue
        if point.lap == stint.start_lap or (stint.end_lap is not None and point.lap == stint.end_lap):
            filters["pit_boundary"] += 1
            continue
        if point.lap in flagged_laps:
            filters["race_status"] += 1
            continue
        field_median = field_lap_medians.get(point.lap)
        if field_median is None:
            filters["field_anomaly"] += 1
            continue
        anomaly_limit = max(5.0, field_median * 0.075)
        if abs(lap_time - field_median) > anomaly_limit:
            filters["field_anomaly"] += 1
            continue

        tyre_age = stint.tyre_age_start + max(0, point.lap - stint.start_lap)
        candidate_samples.append(
            _LapSample(
                driver_number=point.driver_number,
                lap=point.lap,
                lap_time=lap_time,
                compound=stint.compound.upper(),
                tyre_age=tyre_age,
                field_lap_median=field_median,
            )
        )

    if not candidate_samples:
        return _empty_tyre_degradation(
            generated_at,
            status="no_clean_laps_after_filters",
            filters=filters,
        )

    driver_baselines = _driver_baselines(candidate_samples)
    for sample in candidate_samples:
        sample.driver_baseline = driver_baselines.get(sample.driver_number, 0.0)
        sample.adjusted_pace = sample.lap_time - sample.field_lap_median - sample.driver_baseline

    compounds = _compound_summaries(candidate_samples)
    return {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ok" if any(item["cleanLapCount"] >= 3 for item in compounds) else "low_sample",
        "method": "field-lap-median and driver-baseline adjusted compound trend",
        "adjustments": [
            "lap_stint_join",
            "stint_start_tyre_age",
            "pit_boundary_filter",
            "race_status_filter_when_lap_available",
            "field_lap_median_fuel_track_adjustment",
            "driver_baseline_adjustment",
            "compound_level_linear_trend",
        ],
        "sampleCount": len(candidate_samples),
        "excludedCount": sum(filters.values()),
        "filters": filters,
        "compounds": compounds,
        "compoundCrossovers": _compound_crossovers(compounds),
        "cleanLapSample": [_sample_payload(sample) for sample in candidate_samples[:120]],
    }


def _empty_tyre_degradation(generated_at: str, *, status: str, filters: dict[str, int]) -> dict[str, Any]:
    return {
        "version": 1,
        "generatedAt": generated_at,
        "status": status,
        "method": "field-lap-median and driver-baseline adjusted compound trend",
        "adjustments": [],
        "sampleCount": 0,
        "excludedCount": sum(filters.values()),
        "filters": filters,
        "compounds": [],
        "compoundCrossovers": [],
        "cleanLapSample": [],
    }


def _stints_by_driver(stints: list[StintSegment]) -> dict[int, list[StintSegment]]:
    by_driver: dict[int, list[StintSegment]] = {}
    for stint in stints:
        by_driver.setdefault(stint.driver_number, []).append(stint)
    for driver_stints in by_driver.values():
        driver_stints.sort(key=lambda item: (item.start_lap, item.stint_number))
    return by_driver


def _field_lap_medians(points: list[LapPoint]) -> dict[int, float]:
    by_lap: dict[int, list[float]] = {}
    for point in points:
        lap_time = _finite_float(point.value)
        if lap_time is not None and lap_time > 0:
            by_lap.setdefault(point.lap, []).append(lap_time)
    return {lap: median(values) for lap, values in by_lap.items() if values}


def _matching_stint(stints: list[StintSegment], lap: int) -> StintSegment | None:
    for stint in stints:
        if lap < stint.start_lap:
            continue
        if stint.end_lap is not None and lap > stint.end_lap:
            continue
        return stint
    return None


def _flagged_laps(messages: list[dict[str, Any]]) -> set[int]:
    flagged: set[int] = set()
    dirty_terms = ("yellow", "red", "safety", "virtual safety", "vsc", "sc deployed", "rain")
    for message in messages:
        text = " ".join(str(value).lower() for value in message.values() if value is not None)
        if not any(term in text for term in dirty_terms):
            continue
        lap_number = _finite_float(message.get("lap_number", message.get("lap")))
        if lap_number is not None:
            flagged.add(int(lap_number))
    return flagged


def _driver_baselines(samples: list[_LapSample]) -> dict[int, float]:
    by_driver: dict[int, list[float]] = {}
    for sample in samples:
        by_driver.setdefault(sample.driver_number, []).append(sample.lap_time - sample.field_lap_median)
    return {driver: median(values) for driver, values in by_driver.items() if values}


def _compound_summaries(samples: list[_LapSample]) -> list[dict[str, Any]]:
    by_compound: dict[str, list[_LapSample]] = {}
    for sample in samples:
        by_compound.setdefault(sample.compound, []).append(sample)

    summaries: list[dict[str, Any]] = []
    for compound, compound_samples in sorted(by_compound.items()):
        x_values = [sample.tyre_age for sample in compound_samples]
        y_values = [sample.adjusted_pace for sample in compound_samples]
        fit = _linear_fit(x_values, y_values)
        by_age = _by_age_summary(compound_samples)
        slope = fit["slopeSecondsPerTyreLap"]
        slope_ci = fit["slopeConfidenceInterval95"]
        summaries.append(
            {
                "compound": compound,
                "cleanLapCount": len(compound_samples),
                "driverCount": len({sample.driver_number for sample in compound_samples}),
                "minTyreAge": min(x_values),
                "maxTyreAge": max(x_values),
                "medianAdjustedPace": _round(median(y_values)),
                "slopeSecondsPerTyreLap": slope,
                "slopeConfidenceInterval95": slope_ci,
                "projectedLossNext5Laps": _round(slope * 5.0) if slope is not None else None,
                "projectedLossNext10Laps": _round(slope * 10.0) if slope is not None else None,
                "tyreCliffProbability": _cliff_probability(slope, fit["residualStdSeconds"], len(compound_samples)),
                "fit": fit,
                "byTyreAge": by_age,
            }
        )
    return summaries


def _by_age_summary(samples: list[_LapSample]) -> list[dict[str, Any]]:
    by_age: dict[int, list[_LapSample]] = {}
    for sample in samples:
        by_age.setdefault(sample.tyre_age, []).append(sample)

    rows: list[dict[str, Any]] = []
    for tyre_age, age_samples in sorted(by_age.items()):
        adjusted = [sample.adjusted_pace for sample in age_samples]
        raw = [sample.lap_time for sample in age_samples]
        stderr = _stderr(adjusted)
        rows.append(
            {
                "tyreAge": tyre_age,
                "sampleCount": len(age_samples),
                "rawLapTimeMedian": _round(median(raw)),
                "adjustedPaceMean": _round(sum(adjusted) / len(adjusted)),
                "adjustedPaceMedian": _round(median(adjusted)),
                "confidenceInterval95": {
                    "lower": _round((sum(adjusted) / len(adjusted)) - 1.96 * stderr),
                    "upper": _round((sum(adjusted) / len(adjusted)) + 1.96 * stderr),
                },
            }
        )
    return rows


def _compound_crossovers(compounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    crossovers: list[dict[str, Any]] = []
    fit_by_compound = {compound["compound"]: compound["fit"] for compound in compounds}
    for first, second in combinations(sorted(fit_by_compound), 2):
        fit_a = fit_by_compound[first]
        fit_b = fit_by_compound[second]
        slope_a = fit_a.get("slopeSecondsPerTyreLap")
        slope_b = fit_b.get("slopeSecondsPerTyreLap")
        intercept_a = fit_a.get("interceptSeconds")
        intercept_b = fit_b.get("interceptSeconds")
        if slope_a is None or slope_b is None or intercept_a is None or intercept_b is None:
            continue
        denominator = slope_a - slope_b
        if abs(denominator) < 1e-9:
            continue
        tyre_age = (intercept_b - intercept_a) / denominator
        if 0.0 <= tyre_age <= 90.0:
            crossovers.append(
                {
                    "compoundA": first,
                    "compoundB": second,
                    "tyreAge": _round(tyre_age),
                    "caveat": "reduced-state estimate; validate with FastF1 post-session telemetry",
                }
            )
    return crossovers


def _linear_fit(x_values: list[int], y_values: list[float]) -> dict[str, Any]:
    n = len(x_values)
    if n < 2 or len(set(x_values)) < 2:
        return {
            "status": "insufficient_variation",
            "sampleCount": n,
            "slopeSecondsPerTyreLap": None,
            "interceptSeconds": None,
            "slopeConfidenceInterval95": None,
            "residualStdSeconds": None,
            "rSquared": None,
        }

    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n
    sxx = sum((x - mean_x) ** 2 for x in x_values)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    if sxx <= 0.0:
        return {
            "status": "insufficient_variation",
            "sampleCount": n,
            "slopeSecondsPerTyreLap": None,
            "interceptSeconds": None,
            "slopeConfidenceInterval95": None,
            "residualStdSeconds": None,
            "rSquared": None,
        }

    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(x_values, y_values)]
    sse = sum(residual * residual for residual in residuals)
    sst = sum((y - mean_y) ** 2 for y in y_values)
    residual_std = sqrt(sse / max(1, n - 2))
    slope_ci = 1.96 * residual_std / sqrt(sxx) if n > 2 else None
    return {
        "status": "ok",
        "sampleCount": n,
        "slopeSecondsPerTyreLap": _round(slope),
        "interceptSeconds": _round(intercept),
        "slopeConfidenceInterval95": {
            "lower": _round(slope - slope_ci),
            "upper": _round(slope + slope_ci),
        }
        if slope_ci is not None
        else None,
        "residualStdSeconds": _round(residual_std),
        "rSquared": _round(1.0 - sse / sst) if sst > 0 else None,
    }


def _cliff_probability(slope: float | None, residual_std: float | None, sample_count: int) -> float | None:
    if slope is None:
        return None
    noise = max(0.25, residual_std or 0.5)
    sample_weight = min(1.0, sample_count / 20.0)
    score = ((max(0.0, slope) * 10.0) - 0.8) / noise
    return _round((1.0 / (1.0 + exp(-score))) * sample_weight)


def _sample_payload(sample: _LapSample) -> dict[str, Any]:
    return {
        "driverNumber": sample.driver_number,
        "lap": sample.lap,
        "compound": sample.compound,
        "tyreAge": sample.tyre_age,
        "lapTime": _round(sample.lap_time),
        "fieldLapMedian": _round(sample.field_lap_median),
        "driverBaseline": _round(sample.driver_baseline),
        "adjustedPace": _round(sample.adjusted_pace),
    }


def _weather_sample_payload(sample: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "index": index,
        "eventTime": sample.get("event_time") or sample.get("date"),
        "airTemperature": _round(_finite_float(sample.get("air_temperature"))),
        "trackTemperature": _round(_finite_float(sample.get("track_temperature"))),
        "rainfall": _round(_finite_float(sample.get("rainfall"))),
        "windSpeed": _round(_finite_float(sample.get("wind_speed"))),
        "sourceId": sample.get("source_id"),
    }


def _first_numeric(samples: list[dict[str, Any]], key: str) -> float | None:
    for sample in samples:
        value = _finite_float(sample.get(key))
        if value is not None:
            return value
    return None


def _last_numeric(samples: list[dict[str, Any]], key: str) -> float | None:
    for sample in reversed(samples):
        value = _finite_float(sample.get(key))
        if value is not None:
            return value
    return None


def _delta(latest: float | None, first: float | None) -> float | None:
    if latest is None or first is None:
        return None
    return _round(latest - first)


def _stderr(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return sqrt(variance) / sqrt(len(values))


def _stddev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return sqrt(variance)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(number):
        return None
    return number


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)
