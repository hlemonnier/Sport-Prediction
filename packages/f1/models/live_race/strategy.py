"""Deterministic telemetry and strategy adapters for live-race F1 modeling.

The adapters in this module are intentionally self-contained: they use only
features already present in local replay/live-state frames and never fetch
external telemetry.  They are not meant to be the final optimizer.  They are a
typed, explainable baseline that can score simple pit decisions during a race.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol

import numpy as np
import pandas as pd


class TelemetryFeatureAdapter(Protocol):
    """Build deterministic lap-level features from already-loaded race data."""

    def build_lap_features(self, laps: pd.DataFrame) -> pd.DataFrame:
        ...


class StrategyPolicyAdapter(Protocol):
    """Score live strategy actions for each row in a state frame."""

    def evaluate_actions(self, state_frame: pd.DataFrame) -> pd.DataFrame:
        ...


@dataclass(frozen=True)
class CompoundProfile:
    normalized: str
    service_life_laps: float
    deg_prior_seconds_per_lap: float


@dataclass(frozen=True)
class StrategyPolicyConfig:
    """Parameters for the deterministic strategy baseline."""

    default_horizon_laps: int = 12
    default_pit_loss_seconds: float = 21.0
    sc_vsc_pit_loss_seconds: float = 11.0
    yellow_pit_loss_seconds: float = 15.5
    minimum_laps_after_stop: int = 2
    mandatory_stop_window_laps: int = 10


@dataclass(frozen=True)
class TrackFlags:
    is_red: bool
    is_sc_vsc: bool
    is_yellow: bool
    is_greenish: bool


TELEMETRY_FEATURE_COLUMNS = [
    "lap_number",
    "stint_id",
    "tyre_age",
    "compound_normalized",
    "compound_service_life_laps",
    "compound_deg_prior",
    "tyre_life_used_ratio",
    "is_clean_lap",
    "is_greenish",
    "is_yellow",
    "is_sc_vsc",
    "is_red",
    "track_risk_score",
    "lap_time_delta_to_event_median",
    "rolling_clean_pace_delta_3",
    "estimated_deg_slope_5",
    "race_time_seconds",
    "gap_to_leader_seconds",
]


STRATEGY_POLICY_COLUMNS = [
    "recommended_action",
    "recommendation_confidence",
    "score_stay_out",
    "score_pit_next_lap",
    "score_pit_now",
    "pit_urgency",
    "tyre_life_used_ratio",
    "degradation_risk",
    "pace_risk",
    "track_risk_score",
    "pit_loss_estimate_seconds",
    "next_compound",
    "strategy_reason",
    "policy_version",
]


ACTION_ORDER = ("stay_out", "pit_next_lap", "pit_now")
POLICY_VERSION = "deterministic_baseline_v1"


def _empty_frame(index: pd.Index, columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(index=index, columns=list(columns))


def _first_available(frame: pd.DataFrame, columns: Iterable[str]) -> Optional[str]:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def _numeric_series(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    default: float = float("nan"),
) -> pd.Series:
    column = _first_available(frame, columns)
    if column is None:
        return pd.Series(default, index=frame.index, dtype=float)
    series = frame[column]
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() > 0:
        return numeric.astype(float)
    try:
        timed = pd.to_timedelta(series, errors="coerce")
        if timed.notna().sum() > 0:
            return timed.dt.total_seconds()
    except Exception:
        pass
    return pd.Series(default, index=frame.index, dtype=float)


def _text_series(frame: pd.DataFrame, columns: Iterable[str], *, default: str = "") -> pd.Series:
    column = _first_available(frame, columns)
    if column is None:
        return pd.Series(default, index=frame.index, dtype=object)
    return frame[column].fillna(default).astype(str)


def _bool_series(frame: pd.DataFrame, columns: Iterable[str], *, default: bool = False) -> pd.Series:
    column = _first_available(frame, columns)
    if column is None:
        return pd.Series(default, index=frame.index, dtype=bool)
    series = frame[column]
    if series.dtype == bool:
        return series.fillna(default).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(float(default)).astype(float) != 0.0
    text = series.fillna(str(default)).astype(str).str.strip().str.lower()
    truthy = {"1", "true", "yes", "y", "t"}
    falsy = {"0", "false", "no", "n", "f", ""}
    return text.map(lambda value: True if value in truthy else False if value in falsy else default).astype(bool)


def _clip01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(lower=0.0, upper=1.0)


def _parse_track_flags(value: object) -> TrackFlags:
    if value is None:
        codes: set[str] = set()
    else:
        try:
            if pd.isna(value):
                codes = set()
            else:
                codes = {ch for ch in str(value).strip() if ch.isdigit()}
        except Exception:
            codes = set()
    is_red = "5" in codes
    is_sc_vsc = any(code in codes for code in {"4", "6", "7"})
    is_yellow = "2" in codes
    is_greenish = ("1" in codes) and (not is_sc_vsc) and (not is_red)
    return TrackFlags(
        is_red=bool(is_red),
        is_sc_vsc=bool(is_sc_vsc),
        is_yellow=bool(is_yellow),
        is_greenish=bool(is_greenish),
    )


def _track_flags_frame(frame: pd.DataFrame) -> pd.DataFrame:
    status = _text_series(frame, ["track_status", "TrackStatus"], default="")
    parsed = status.map(_parse_track_flags)
    out = pd.DataFrame(index=frame.index)
    out["is_red"] = parsed.map(lambda item: item.is_red).astype(bool)
    out["is_sc_vsc"] = parsed.map(lambda item: item.is_sc_vsc).astype(bool)
    out["is_yellow"] = parsed.map(lambda item: item.is_yellow).astype(bool)
    out["is_greenish"] = parsed.map(lambda item: item.is_greenish).astype(bool)

    for column in ["is_red", "is_sc_vsc", "is_yellow", "is_greenish"]:
        if column in frame.columns:
            out[column] = _bool_series(frame, [column], default=False)
    return out


def _track_risk_score(flags: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.20, index=flags.index, dtype=float)
    score.loc[flags["is_greenish"].astype(bool)] = 0.0
    score.loc[flags["is_yellow"].astype(bool)] = 0.45
    score.loc[flags["is_sc_vsc"].astype(bool)] = 0.85
    score.loc[flags["is_red"].astype(bool)] = 1.0
    return score.clip(lower=0.0, upper=1.0)


def _compound_profile(value: object) -> CompoundProfile:
    text = str(value or "").strip().upper()
    if text in {"SOFT", "C4", "C5"}:
        return CompoundProfile("SOFT", service_life_laps=16.0, deg_prior_seconds_per_lap=0.050)
    if text in {"MEDIUM", "C3"}:
        return CompoundProfile("MEDIUM", service_life_laps=22.0, deg_prior_seconds_per_lap=0.040)
    if text in {"HARD", "C1", "C2"}:
        return CompoundProfile("HARD", service_life_laps=28.0, deg_prior_seconds_per_lap=0.030)
    if "INTER" in text:
        return CompoundProfile("INTER", service_life_laps=14.0, deg_prior_seconds_per_lap=0.060)
    if "WET" in text:
        return CompoundProfile("WET", service_life_laps=12.0, deg_prior_seconds_per_lap=0.065)
    return CompoundProfile("UNKNOWN", service_life_laps=20.0, deg_prior_seconds_per_lap=0.040)


def _compound_profiles(frame: pd.DataFrame) -> pd.Series:
    compounds = _text_series(frame, ["compound", "Compound"], default="UNKNOWN")
    return compounds.map(_compound_profile)


def _infer_tyre_age(frame: pd.DataFrame) -> pd.Series:
    observed = _numeric_series(frame, ["tyre_age", "TyreAge", "TyreLife", "tyre_life_raw"], default=float("nan"))
    if observed.notna().all():
        return observed.clip(lower=0.0)

    driver_id = _text_series(frame, ["driver_id", "DriverNumber", "Driver"], default="driver")
    stint_id = _numeric_series(frame, ["stint_id", "Stint"], default=1.0).fillna(1.0)
    lap_number = _numeric_series(frame, ["lap_number", "LapNumber"], default=float("nan"))
    timestamp = _numeric_series(frame, ["timestamp", "Time", "LapStartTime"], default=float("nan"))
    order_frame = pd.DataFrame(
        {
            "driver_id": driver_id,
            "stint_id": stint_id,
            "lap_number": lap_number,
            "timestamp": timestamp,
        },
        index=frame.index,
    ).sort_values(["driver_id", "stint_id", "lap_number", "timestamp"], kind="mergesort")

    inferred = pd.Series(index=frame.index, dtype=float)
    for _, idx in order_frame.groupby(["driver_id", "stint_id"], sort=False).groups.items():
        inferred.loc[idx] = np.arange(len(idx), dtype=float)
    return observed.fillna(inferred).fillna(0.0).clip(lower=0.0)


def _rolling_mean_by_driver(
    frame: pd.DataFrame,
    values: pd.Series,
    *,
    window: int,
) -> pd.Series:
    driver_id = _text_series(frame, ["driver_id", "DriverNumber", "Driver"], default="driver")
    lap_number = _numeric_series(frame, ["lap_number", "LapNumber"], default=float("nan"))
    timestamp = _numeric_series(frame, ["timestamp", "Time", "LapStartTime"], default=float("nan"))
    order_frame = pd.DataFrame(
        {
            "driver_id": driver_id,
            "lap_number": lap_number,
            "timestamp": timestamp,
            "value": values,
        },
        index=frame.index,
    ).sort_values(["driver_id", "lap_number", "timestamp"], kind="mergesort")

    out = pd.Series(index=frame.index, dtype=float)
    for _, idx in order_frame.groupby("driver_id", sort=False).groups.items():
        out.loc[idx] = order_frame.loc[idx, "value"].rolling(window=window, min_periods=1).mean().to_numpy()
    return out


def _rolling_slope_by_driver(
    frame: pd.DataFrame,
    x: pd.Series,
    y: pd.Series,
    *,
    window: int,
) -> pd.Series:
    driver_id = _text_series(frame, ["driver_id", "DriverNumber", "Driver"], default="driver")
    lap_number = _numeric_series(frame, ["lap_number", "LapNumber"], default=float("nan"))
    timestamp = _numeric_series(frame, ["timestamp", "Time", "LapStartTime"], default=float("nan"))
    order_frame = pd.DataFrame(
        {
            "driver_id": driver_id,
            "lap_number": lap_number,
            "timestamp": timestamp,
            "x": x,
            "y": y,
        },
        index=frame.index,
    ).sort_values(["driver_id", "lap_number", "timestamp"], kind="mergesort")

    out = pd.Series(index=frame.index, dtype=float)
    for _, idx in order_frame.groupby("driver_id", sort=False).groups.items():
        subset = order_frame.loc[idx]
        slopes: list[float] = []
        xs = subset["x"].to_numpy(dtype=float)
        ys = subset["y"].to_numpy(dtype=float)
        for pos in range(len(subset)):
            start = max(0, pos - int(window) + 1)
            x_window = xs[start : pos + 1]
            y_window = ys[start : pos + 1]
            mask = np.isfinite(x_window) & np.isfinite(y_window)
            if int(mask.sum()) < 2 or float(np.ptp(x_window[mask])) <= 1e-9:
                slopes.append(float("nan"))
                continue
            coef = np.polyfit(x_window[mask], y_window[mask], deg=1)
            slopes.append(float(coef[0]))
        out.loc[idx] = slopes
    return out


def _remaining_laps(frame: pd.DataFrame, config: StrategyPolicyConfig) -> pd.Series:
    explicit = _numeric_series(frame, ["laps_remaining", "remaining_laps"], default=float("nan"))
    total_laps = _numeric_series(frame, ["race_total_laps", "scheduled_laps", "total_laps"], default=float("nan"))
    current_lap = _numeric_series(frame, ["lap_number", "lap_last", "LapNumber"], default=float("nan"))
    derived = total_laps - current_lap
    fallback = pd.Series(float(config.default_horizon_laps), index=frame.index, dtype=float)
    return explicit.fillna(derived).fillna(fallback).clip(lower=0.0)


def _next_compound(current: str, remaining_laps: float) -> str:
    compound = str(current or "UNKNOWN").upper()
    remaining = float(remaining_laps) if np.isfinite(float(remaining_laps)) else 0.0
    if compound == "SOFT":
        return "HARD" if remaining > 18.0 else "MEDIUM"
    if compound == "MEDIUM":
        return "HARD" if remaining > 16.0 else "SOFT"
    if compound == "HARD":
        return "MEDIUM" if remaining > 12.0 else "SOFT"
    if compound in {"INTER", "WET"}:
        return compound
    if remaining > 20.0:
        return "HARD"
    if remaining > 10.0:
        return "MEDIUM"
    return "SOFT"


def _softmax_confidence(scores: np.ndarray) -> float:
    if scores.size == 0 or not np.isfinite(scores).any():
        return float("nan")
    safe = np.where(np.isfinite(scores), scores, -1e9)
    shifted = safe - float(np.max(safe))
    exp_scores = np.exp(np.clip(shifted, -60.0, 60.0))
    total = float(np.sum(exp_scores))
    if total <= 0.0:
        return float("nan")
    return float(np.max(exp_scores / total))


@dataclass
class BaselineTelemetryFeatureAdapter:
    """Build explainable lap features for the live strategy baseline."""

    rolling_window_laps: int = 3
    degradation_window_laps: int = 5

    def __post_init__(self) -> None:
        if int(self.rolling_window_laps) <= 0:
            raise ValueError("rolling_window_laps must be positive")
        if int(self.degradation_window_laps) <= 1:
            raise ValueError("degradation_window_laps must be greater than 1")

    def build_lap_features(self, laps: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(laps, pd.DataFrame):
            raise TypeError("laps must be a pandas DataFrame")
        if laps.empty:
            return _empty_frame(laps.index, TELEMETRY_FEATURE_COLUMNS)

        work = laps.copy()
        lap_number = _numeric_series(work, ["lap_number", "LapNumber"], default=float("nan"))
        stint_id = _numeric_series(work, ["stint_id", "Stint"], default=1.0).fillna(1.0)
        tyre_age = _infer_tyre_age(work)
        lap_time = _numeric_series(work, ["lap_time_seconds", "LapTime", "lap_time", "duration"], default=float("nan"))
        race_time = _numeric_series(work, ["race_time_seconds", "RaceTimeSeconds"], default=float("nan"))
        gap_to_leader = _numeric_series(work, ["gap_to_leader_seconds", "gap_to_leader"], default=float("nan"))
        is_box_lap = _bool_series(work, ["is_box_lap", "IsBoxLap"], default=False)
        is_accurate = _bool_series(work, ["is_accurate", "IsAccurate"], default=True)
        flags = _track_flags_frame(work)
        profiles = _compound_profiles(work)

        clean_lap = (
            (~is_box_lap)
            & is_accurate
            & flags["is_greenish"].astype(bool)
            & lap_number.notna()
            & lap_time.notna()
            & (lap_time > 0.0)
        )

        event_lap_median = lap_time.loc[clean_lap].groupby(lap_number.loc[clean_lap]).median()
        baseline_for_lap = lap_number.map(event_lap_median)
        lap_delta = lap_time - baseline_for_lap
        clean_delta = lap_delta.where(clean_lap)
        rolling_delta = _rolling_mean_by_driver(
            work,
            clean_delta,
            window=int(self.rolling_window_laps),
        )
        deg_slope = _rolling_slope_by_driver(
            work,
            tyre_age.where(clean_lap),
            clean_delta,
            window=int(self.degradation_window_laps),
        )

        service_life = profiles.map(lambda profile: profile.service_life_laps).astype(float)
        deg_prior = profiles.map(lambda profile: profile.deg_prior_seconds_per_lap).astype(float)
        normalized = profiles.map(lambda profile: profile.normalized).astype(str)

        out = pd.DataFrame(index=work.index)
        out["lap_number"] = lap_number
        out["stint_id"] = stint_id
        out["tyre_age"] = tyre_age
        out["compound_normalized"] = normalized
        out["compound_service_life_laps"] = service_life
        out["compound_deg_prior"] = deg_prior
        out["tyre_life_used_ratio"] = (tyre_age / service_life.replace(0.0, np.nan)).clip(lower=0.0, upper=2.0)
        out["is_clean_lap"] = clean_lap.astype(bool)
        out["is_greenish"] = flags["is_greenish"].astype(bool)
        out["is_yellow"] = flags["is_yellow"].astype(bool)
        out["is_sc_vsc"] = flags["is_sc_vsc"].astype(bool)
        out["is_red"] = flags["is_red"].astype(bool)
        out["track_risk_score"] = _track_risk_score(flags)
        out["lap_time_delta_to_event_median"] = lap_delta
        out["rolling_clean_pace_delta_3"] = rolling_delta
        out["estimated_deg_slope_5"] = deg_slope.fillna(deg_prior)
        out["race_time_seconds"] = race_time
        out["gap_to_leader_seconds"] = gap_to_leader
        return out[TELEMETRY_FEATURE_COLUMNS]


@dataclass
class BaselineStrategyPolicyAdapter:
    """Score pit actions from a live state frame using deterministic heuristics."""

    config: StrategyPolicyConfig = field(default_factory=StrategyPolicyConfig)

    def __post_init__(self) -> None:
        if int(self.config.default_horizon_laps) <= 0:
            raise ValueError("default_horizon_laps must be positive")
        if float(self.config.default_pit_loss_seconds) <= 0.0:
            raise ValueError("default_pit_loss_seconds must be positive")

    def evaluate_actions(self, state_frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(state_frame, pd.DataFrame):
            raise TypeError("state_frame must be a pandas DataFrame")
        if state_frame.empty:
            return _empty_frame(state_frame.index, STRATEGY_POLICY_COLUMNS)

        work = state_frame.copy()
        profiles = _compound_profiles(work)
        normalized = profiles.map(lambda profile: profile.normalized).astype(str)
        service_life = profiles.map(lambda profile: profile.service_life_laps).astype(float)
        deg_prior = profiles.map(lambda profile: profile.deg_prior_seconds_per_lap).astype(float)

        tyre_age = _numeric_series(work, ["tyre_age", "TyreAge", "TyreLife", "tyre_life_raw"], default=0.0).fillna(0.0)
        tyre_age = tyre_age.clip(lower=0.0)
        tyre_life_used = (tyre_age / service_life.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        tyre_life_used = tyre_life_used.clip(lower=0.0, upper=2.0)

        deg_rate = _numeric_series(
            work,
            ["deg_rate_mean", "estimated_deg_slope_5", "compound_deg_prior"],
            default=float("nan"),
        ).fillna(deg_prior)
        degradation_risk = ((deg_rate - deg_prior) / (deg_prior + 0.02)).clip(lower=0.0, upper=1.0)

        pace_penalty = _numeric_series(
            work,
            ["pace_penalty_mean", "rolling_clean_pace_delta_3"],
            default=0.0,
        ).fillna(0.0)
        next_lap_mean = _numeric_series(work, ["next_lap_mean"], default=float("nan"))
        if next_lap_mean.notna().any():
            field_next_lap = float(next_lap_mean.median(skipna=True))
            next_lap_pressure = ((next_lap_mean - field_next_lap) / 2.5).clip(lower=0.0, upper=1.0).fillna(0.0)
        else:
            next_lap_pressure = pd.Series(0.0, index=work.index, dtype=float)
        pace_risk = ((pace_penalty.clip(lower=0.0) / 2.5) + next_lap_pressure).clip(lower=0.0, upper=1.0)

        flags = _track_flags_frame(work)
        track_risk = _track_risk_score(flags)
        remaining_laps = _remaining_laps(work, self.config)
        stint_id = _numeric_series(work, ["stint_id", "Stint"], default=1.0).fillna(1.0)

        explicit_pit_loss = _numeric_series(work, ["pit_loss_estimate_seconds", "pit_loss_seconds"], default=float("nan"))
        pit_loss = pd.Series(float(self.config.default_pit_loss_seconds), index=work.index, dtype=float)
        pit_loss.loc[flags["is_yellow"].astype(bool)] = float(self.config.yellow_pit_loss_seconds)
        pit_loss.loc[flags["is_sc_vsc"].astype(bool)] = float(self.config.sc_vsc_pit_loss_seconds)
        pit_loss = explicit_pit_loss.fillna(pit_loss)
        pit_advantage = (
            (float(self.config.default_pit_loss_seconds) - pit_loss) / 10.0
        ).clip(lower=-0.4, upper=1.2)

        age_pressure = ((tyre_life_used - 0.70) / 0.35).clip(lower=0.0, upper=1.0)
        dry_compound = normalized.isin({"SOFT", "MEDIUM", "HARD"})
        mandatory_window = pd.Series(
            float(self.config.mandatory_stop_window_laps),
            index=work.index,
            dtype=float,
        ).clip(lower=4.0)
        mandatory_pressure = ((mandatory_window - remaining_laps) / mandatory_window).clip(lower=0.0, upper=1.0)
        mandatory_pressure = mandatory_pressure.where(dry_compound & (stint_id <= 1.0), 0.0)

        late_stop_suppression = (remaining_laps <= float(self.config.minimum_laps_after_stop)).astype(float)
        is_box_lap = _bool_series(work, ["is_box_lap", "IsBoxLap"], default=False).astype(float)
        is_red = flags["is_red"].astype(float)
        is_yellow = flags["is_yellow"].astype(float)

        score_stay_out = (
            1.00
            + (1.15 * (1.0 - age_pressure))
            + (0.50 * (1.0 - degradation_risk))
            - (0.55 * pit_advantage)
            - (0.65 * mandatory_pressure)
            - (0.35 * track_risk)
            + (0.50 * is_red)
        )
        score_pit_now = (
            0.45
            + (1.55 * age_pressure)
            + (1.10 * degradation_risk)
            + (0.70 * pace_risk)
            + (0.90 * pit_advantage)
            + (0.80 * mandatory_pressure)
            - (1.40 * late_stop_suppression)
            - (2.00 * is_red)
            - (1.20 * is_box_lap)
        )
        score_pit_next = (
            0.65
            + (1.25 * age_pressure)
            + (0.85 * degradation_risk)
            + (0.50 * pace_risk)
            + (0.25 * is_yellow)
            + (0.55 * mandatory_pressure)
            - (0.35 * pit_advantage)
            - (1.20 * late_stop_suppression)
            - (1.50 * is_red)
        )

        score_frame = pd.DataFrame(
            {
                "stay_out": score_stay_out,
                "pit_next_lap": score_pit_next,
                "pit_now": score_pit_now,
            },
            index=work.index,
        )

        recommended: list[str] = []
        confidence: list[float] = []
        reasons: list[str] = []
        for idx, row in score_frame.iterrows():
            scores = {action: float(row[action]) for action in ACTION_ORDER}
            action = max(ACTION_ORDER, key=lambda item: scores[item])
            recommended.append(action)
            confidence.append(_softmax_confidence(np.asarray([scores[item] for item in ACTION_ORDER], dtype=float)))

            if action == "pit_now" and bool(flags.loc[idx, "is_sc_vsc"]):
                reasons.append("neutralized track window reduces pit loss")
            elif action == "pit_now":
                reasons.append("tyre/degradation pressure exceeds stay-out value")
            elif action == "pit_next_lap":
                reasons.append("pit pressure building; prepare next-lap stop")
            else:
                reasons.append("tyre state below pit threshold")

        out = pd.DataFrame(index=work.index)
        out["recommended_action"] = recommended
        out["recommendation_confidence"] = confidence
        out["score_stay_out"] = score_stay_out
        out["score_pit_next_lap"] = score_pit_next
        out["score_pit_now"] = score_pit_now
        out["pit_urgency"] = _clip01(
            (0.45 * age_pressure)
            + (0.25 * degradation_risk)
            + (0.15 * pace_risk)
            + (0.15 * pit_advantage.clip(lower=0.0))
        )
        out["tyre_life_used_ratio"] = tyre_life_used
        out["degradation_risk"] = degradation_risk
        out["pace_risk"] = pace_risk
        out["track_risk_score"] = track_risk
        out["pit_loss_estimate_seconds"] = pit_loss
        out["next_compound"] = [
            _next_compound(compound, remaining)
            for compound, remaining in zip(normalized.tolist(), remaining_laps.tolist())
        ]
        out["strategy_reason"] = reasons
        out["policy_version"] = POLICY_VERSION
        return out[STRATEGY_POLICY_COLUMNS]


@dataclass
class NoopTelemetryFeatureAdapter(BaselineTelemetryFeatureAdapter):
    """Backward-compatible default telemetry adapter with real baseline features."""

    pass


@dataclass
class NoopStrategyPolicyAdapter(BaselineStrategyPolicyAdapter):
    """Backward-compatible default strategy adapter with real baseline scoring."""

    pass


__all__ = [
    "BaselineStrategyPolicyAdapter",
    "BaselineTelemetryFeatureAdapter",
    "CompoundProfile",
    "NoopStrategyPolicyAdapter",
    "NoopTelemetryFeatureAdapter",
    "StrategyPolicyAdapter",
    "StrategyPolicyConfig",
    "TelemetryFeatureAdapter",
]
