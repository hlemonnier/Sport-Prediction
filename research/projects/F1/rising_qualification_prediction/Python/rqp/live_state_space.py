"""Core state-space utilities for live lap-by-lap F1 modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class TrackStatusFlags:
    codes: set[str]
    is_red: bool
    is_sc_vsc: bool
    is_yellow: bool
    is_greenish: bool


@dataclass
class BaselineModel:
    by_lap: dict[int, float]
    intercept: float
    slope: float

    def value_at(self, lap_number: object) -> float:
        lap = _safe_lap_number(lap_number)
        if lap is None:
            return float(self.intercept)
        if lap in self.by_lap:
            return float(self.by_lap[lap])
        return float(self.intercept + (self.slope * float(lap)))


@dataclass
class FilterConfig:
    phi: float = 0.90
    q_pace: float = 0.10
    q_deg: float = 0.02
    r_obs: float = 0.16
    huber_k: float = 2.5


@dataclass
class FilterState:
    mean: np.ndarray
    cov: np.ndarray
    last_stint_id: Optional[int] = None
    tyre_age: int = 0
    assimilated_laps: int = 0


@dataclass
class GateDecision:
    mode: str
    skip_update: bool
    r_effective: float
    note: Optional[str] = None


def _safe_lap_number(value: object) -> Optional[int]:
    try:
        if value is None or pd.isna(value):
            return None
        lap = int(float(value))
        if lap <= 0:
            return None
        return lap
    except Exception:
        return None


def _safe_float(value: object) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def parse_track_status(value: object) -> TrackStatusFlags:
    if value is None:
        codes: set[str] = set()
    else:
        try:
            if pd.isna(value):
                codes = set()
            else:
                text = str(value).strip()
                codes = {ch for ch in text if ch.isdigit()}
        except Exception:
            codes = set()

    is_red = "5" in codes
    is_sc_vsc = any(code in codes for code in {"4", "6", "7"})
    is_yellow = "2" in codes
    is_greenish = ("1" in codes) and (not is_sc_vsc) and (not is_red)

    return TrackStatusFlags(
        codes=codes,
        is_red=is_red,
        is_sc_vsc=is_sc_vsc,
        is_yellow=is_yellow,
        is_greenish=is_greenish,
    )


def apply_track_gating(base_r: float, flags: TrackStatusFlags) -> GateDecision:
    r = max(float(base_r), 1e-6)
    if flags.is_red or flags.is_sc_vsc:
        return GateDecision(
            mode="skip_sc_vsc_red",
            skip_update=True,
            r_effective=r,
            note="SC/VSC/Red regime: update skipped",
        )
    if flags.is_yellow:
        return GateDecision(
            mode="yellow_inflate",
            skip_update=False,
            r_effective=(r * 9.0),
            note="Yellow regime: observation variance inflated x9",
        )
    if flags.is_greenish:
        return GateDecision(
            mode="green_nominal",
            skip_update=False,
            r_effective=r,
        )
    return GateDecision(
        mode="ambiguous_inflate",
        skip_update=False,
        r_effective=(r * 4.0),
        note="Ambiguous track status: observation variance inflated x4",
    )


def _huber_weights(residual: np.ndarray, scale: float, k: float) -> np.ndarray:
    if scale <= 1e-9:
        return np.ones_like(residual, dtype=float)
    z = np.abs(residual) / (scale * max(k, 1e-6))
    w = np.ones_like(z, dtype=float)
    mask = z > 1.0
    w[mask] = 1.0 / z[mask]
    return np.clip(w, 1e-3, 1.0)


def _fit_huber_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size == 0 or y.size == 0:
        return 0.0, 0.0
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    beta = coef.astype(float)

    for _ in range(6):
        pred = X @ beta
        residual = y - pred
        mad = np.median(np.abs(residual - np.median(residual)))
        scale = 1.4826 * mad if mad > 1e-9 else max(float(np.std(residual)), 1e-3)
        w = _huber_weights(residual, scale=scale, k=1.345)
        WX = X * np.sqrt(w)[:, None]
        Wy = y * np.sqrt(w)
        beta, *_ = np.linalg.lstsq(WX, Wy, rcond=None)

    intercept = float(beta[0])
    slope = float(beta[1]) if beta.size > 1 else 0.0
    return intercept, slope


def _winsorized_median(values: pd.Series, low_q: float = 0.05, high_q: float = 0.95) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    low = float(numeric.quantile(low_q))
    high = float(numeric.quantile(high_q))
    clipped = numeric.clip(lower=low, upper=high)
    return float(clipped.median())


def build_event_lap_baseline(
    observations: pd.DataFrame,
    *,
    min_clean_obs_per_lap: int = 8,
) -> BaselineModel:
    frame = observations.copy()
    if frame.empty:
        return BaselineModel(by_lap={}, intercept=0.0, slope=0.0)

    frame["lap_number"] = pd.to_numeric(frame.get("lap_number"), errors="coerce")
    frame["lap_time_seconds"] = pd.to_numeric(frame.get("lap_time_seconds"), errors="coerce")
    frame["is_box_lap"] = frame.get("is_box_lap", False).astype(bool)
    frame["is_accurate"] = frame.get("is_accurate", False).astype(bool)

    flags = frame.get("track_status", pd.Series(index=frame.index, dtype=object)).apply(parse_track_status)
    is_greenish = flags.apply(lambda item: bool(item.is_greenish))

    clean = frame[
        frame["lap_number"].notna()
        & frame["lap_time_seconds"].notna()
        & (frame["lap_time_seconds"] > 0.0)
        & (~frame["is_box_lap"])
        & frame["is_accurate"]
        & is_greenish
    ].copy()

    clean["lap_number"] = clean["lap_number"].astype(int)

    baseline_by_lap: dict[int, float] = {}
    if not clean.empty:
        for lap, group in clean.groupby("lap_number", sort=True):
            if len(group) < int(min_clean_obs_per_lap):
                continue
            median = _winsorized_median(group["lap_time_seconds"], low_q=0.05, high_q=0.95)
            if np.isfinite(median):
                baseline_by_lap[int(lap)] = float(median)

    fit_source = clean if not clean.empty else frame[frame["lap_time_seconds"].notna() & (frame["lap_time_seconds"] > 0.0)]
    if fit_source.empty:
        intercept = 0.0
        slope = 0.0
    else:
        x = pd.to_numeric(fit_source["lap_number"], errors="coerce").dropna().to_numpy(dtype=float)
        y = pd.to_numeric(fit_source["lap_time_seconds"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(x) > 1 and len(y) == len(x):
            intercept, slope = _fit_huber_line(x, y)
        elif len(y) > 0:
            intercept, slope = float(np.nanmedian(y)), 0.0
        else:
            intercept, slope = 0.0, 0.0

    if baseline_by_lap:
        lap_min = min(baseline_by_lap)
        lap_max = max(baseline_by_lap)
        full = pd.Series(index=range(lap_min, lap_max + 1), dtype=float)
        for lap, value in baseline_by_lap.items():
            full.loc[int(lap)] = float(value)
        full = full.interpolate(method="linear", limit_direction="both")
        for lap, value in full.items():
            if np.isfinite(value):
                baseline_by_lap[int(lap)] = float(value)

    if not baseline_by_lap and np.isfinite(intercept):
        laps = (
            pd.to_numeric(frame["lap_number"], errors="coerce")
            .dropna()
            .astype(int)
            .sort_values()
            .unique()
            .tolist()
        )
        for lap in laps:
            baseline_by_lap[int(lap)] = float(intercept + (slope * float(lap)))

    return BaselineModel(
        by_lap=baseline_by_lap,
        intercept=float(intercept),
        slope=float(slope),
    )


def compound_deg_prior(compound: object) -> float:
    text = str(compound or "").strip().upper()
    if text in {"SOFT", "C5", "C4"}:
        return 0.050
    if text in {"MEDIUM", "C3"}:
        return 0.040
    if text in {"HARD", "C2", "C1"}:
        return 0.030
    if "INTER" in text:
        return 0.060
    if "WET" in text:
        return 0.065
    return 0.040


def initialize_filter_state(compound: object, cfg: FilterConfig) -> FilterState:
    _ = cfg
    prior_deg = compound_deg_prior(compound)
    mean = np.asarray([0.0, prior_deg], dtype=float)
    cov = np.asarray([[0.36, 0.0], [0.0, 0.04]], dtype=float)
    return FilterState(mean=mean, cov=cov)


def reset_filter_state(state: FilterState, compound: object, cfg: FilterConfig) -> FilterState:
    prior = initialize_filter_state(compound=compound, cfg=cfg)
    prior.last_stint_id = state.last_stint_id
    prior.tyre_age = 0
    prior.assimilated_laps = state.assimilated_laps
    return prior


def _transition_matrix(phi: float) -> np.ndarray:
    return np.asarray([[1.0, 1.0], [0.0, float(phi)]], dtype=float)


def _process_cov(q_pace: float, q_deg: float) -> np.ndarray:
    return np.asarray([[float(q_pace) ** 2, 0.0], [0.0, float(q_deg) ** 2]], dtype=float)


def predict_state(state: FilterState, cfg: FilterConfig) -> tuple[np.ndarray, np.ndarray]:
    A = _transition_matrix(cfg.phi)
    Q = _process_cov(cfg.q_pace, cfg.q_deg)
    mean_pred = A @ state.mean
    cov_pred = (A @ state.cov @ A.T) + Q
    return mean_pred, cov_pred


def update_state(
    *,
    mean_pred: np.ndarray,
    cov_pred: np.ndarray,
    observation: float,
    r_effective: float,
    cfg: FilterConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    H = np.asarray([1.0, 0.0], dtype=float)
    R = max(float(r_effective), 1e-6)

    innovation = float(observation - (H @ mean_pred))
    s_base = float(H @ cov_pred @ H.T) + R
    robust_applied = False
    robust_scale = 1.0

    threshold = float(cfg.huber_k) * float(np.sqrt(max(s_base, 1e-9)))
    if threshold > 0.0 and abs(innovation) > threshold:
        robust_applied = True
        robust_scale = float(abs(innovation) / threshold)
        R = R * (robust_scale**2)

    S = float(H @ cov_pred @ H.T) + R
    if S <= 1e-12:
        return mean_pred, cov_pred, {
            "innovation": innovation,
            "S": S,
            "robust_applied": robust_applied,
            "robust_scale": robust_scale,
            "skip_reason": "singular_observation_variance",
        }

    K = cov_pred @ H / S
    mean_post = mean_pred + (K * innovation)
    cov_post = cov_pred - np.outer(K, H @ cov_pred)
    cov_post = 0.5 * (cov_post + cov_post.T)

    return mean_post, cov_post, {
        "innovation": innovation,
        "S": S,
        "robust_applied": robust_applied,
        "robust_scale": robust_scale,
    }


def next_lap_distribution(
    state: FilterState,
    baseline_next: float,
    cfg: FilterConfig,
    r_effective: Optional[float] = None,
) -> tuple[float, float]:
    mean_pred, cov_pred = predict_state(state, cfg)
    H = np.asarray([1.0, 0.0], dtype=float)
    obs_var = float(H @ cov_pred @ H.T) + float(r_effective if r_effective is not None else cfg.r_obs)
    obs_var = max(obs_var, 1e-6)
    mean_time = float(baseline_next + (H @ mean_pred))
    std_time = float(np.sqrt(obs_var))
    return mean_time, std_time


def lap_one_step_prediction(
    state: FilterState,
    baseline_current: float,
    cfg: FilterConfig,
    r_effective: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    mean_pred, cov_pred = predict_state(state, cfg)
    H = np.asarray([1.0, 0.0], dtype=float)
    obs_var = float(H @ cov_pred @ H.T) + float(r_effective if r_effective is not None else cfg.r_obs)
    obs_var = max(obs_var, 1e-6)
    pred_mean = float(baseline_current + (H @ mean_pred))
    pred_std = float(np.sqrt(obs_var))
    return mean_pred, cov_pred, pred_mean, pred_std


def as_float(value: object, fallback: float = float("nan")) -> float:
    numeric = _safe_float(value)
    if np.isfinite(numeric):
        return float(numeric)
    return float(fallback)
