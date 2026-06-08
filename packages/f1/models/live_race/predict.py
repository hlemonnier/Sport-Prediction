"""Live race runner for Horizon B v1 state-space modeling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from packages.sports_core.paths import find_repo_root

from packages.f1.data.schemas.session import PredictionConfig
from packages.f1.models.live_race.evaluate import evaluate_live_replay
from packages.f1.models.live_race.strategy import (
    NoopStrategyPolicyAdapter,
    NoopTelemetryFeatureAdapter,
    StrategyPolicyAdapter,
    TelemetryFeatureAdapter,
)
from packages.f1.models.live_race.sources import load_live_observations
from packages.f1.models.live_race.state import (
    BaselineModel,
    FilterConfig,
    FilterState,
    apply_track_gating,
    as_float,
    build_event_lap_baseline,
    initialize_filter_state,
    lap_one_step_prediction,
    next_lap_distribution,
    parse_track_status,
    reset_filter_state,
    update_state,
)


@dataclass
class LiveRunResult:
    snapshot: pd.DataFrame
    trace: pd.DataFrame
    summary: dict[str, Any]
    notes: list[str]


@dataclass(frozen=True)
class StrategyTemplate:
    name: str
    first_pit_age_mean: Optional[float]
    first_pit_age_std: float
    second_stint_interval_mean: Optional[float] = None
    second_stint_interval_std: float = 0.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stable_hash(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def _event_seed(event_key: int, base_seed: int) -> int:
    value = (int(base_seed) + _stable_hash(str(int(event_key)))) % (2**32 - 1)
    return value if value > 0 else 1


def _resolve_artifacts_dir(config: PredictionConfig) -> Path:
    project_root = find_repo_root(__file__)
    return project_root / "artifacts" / "predictions" / "f1" / "live" / "traces" / str(int(config.year)) / f"round_{int(config.round_number):02d}"


def _write_trace(trace: pd.DataFrame, config: PredictionConfig, event_key: int) -> dict[str, Any]:
    out_dir = _resolve_artifacts_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"live_trace_{int(event_key)}_{stamp}"

    parquet_path = out_dir / f"{stem}.parquet"
    jsonl_path = out_dir / f"{stem}.jsonl"

    trace_path: Optional[str] = None
    trace_path_jsonl: Optional[str] = None
    effective = "jsonl"

    try:
        trace.to_parquet(parquet_path, index=False)
        trace_path = str(parquet_path)
        effective = "parquet"
    except Exception:
        trace_path = None

    try:
        with open(jsonl_path, "w", encoding="utf-8") as handle:
            for record in trace.to_dict(orient="records"):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        trace_path_jsonl = str(jsonl_path)
    except Exception:
        trace_path_jsonl = None

    if trace_path is None and trace_path_jsonl is not None:
        trace_path = trace_path_jsonl
        effective = "jsonl"

    if trace_path is None:
        trace_path = ""
        effective = "none"

    return {
        "trace_path": trace_path,
        "trace_path_jsonl": trace_path_jsonl,
        "trace_format_effective": effective,
    }


def _sigmoid(value: float) -> float:
    clipped = float(np.clip(value, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-clipped)))


TELEMETRY_STRATEGY_FEED_COLUMNS = (
    "compound_service_life_laps",
    "compound_deg_prior",
    "tyre_life_used_ratio",
    "track_risk_score",
    "rolling_clean_pace_delta_3",
    "estimated_deg_slope_5",
)


def _json_safe_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _normalized_compound(compound: object) -> str:
    text = str(compound or "").strip().upper()
    if text in {"SOFT", "C4", "C5"}:
        return "SOFT"
    if text in {"MEDIUM", "C3"}:
        return "MEDIUM"
    if text in {"HARD", "C1", "C2"}:
        return "HARD"
    if "INTER" in text:
        return "INTER"
    if "WET" in text:
        return "WET"
    return "UNKNOWN"


def _compound_service_life(compound: object) -> float:
    key = _normalized_compound(compound)
    if key == "SOFT":
        return 16.0
    if key == "MEDIUM":
        return 22.0
    if key == "HARD":
        return 28.0
    if key == "INTER":
        return 14.0
    if key == "WET":
        return 12.0
    return 20.0


def _strategy_templates(compound: object) -> list[StrategyTemplate]:
    service_life = _compound_service_life(compound)
    return [
        StrategyTemplate(
            name="hold_track_position",
            first_pit_age_mean=None,
            first_pit_age_std=0.0,
        ),
        StrategyTemplate(
            name="one_stop_conservative",
            first_pit_age_mean=service_life + 4.0,
            first_pit_age_std=2.4,
        ),
        StrategyTemplate(
            name="two_stop_balanced",
            first_pit_age_mean=max(8.0, service_life - 1.5),
            first_pit_age_std=2.0,
            second_stint_interval_mean=max(10.0, service_life - 4.0),
            second_stint_interval_std=2.0,
        ),
        StrategyTemplate(
            name="two_stop_aggressive",
            first_pit_age_mean=max(6.0, service_life - 5.0),
            first_pit_age_std=1.6,
            second_stint_interval_mean=max(8.0, service_life - 6.5),
            second_stint_interval_std=1.8,
        ),
    ]


def _strategy_template_probabilities(
    *,
    compound: object,
    tyre_age: int,
    deg_rate: float,
    horizon: int,
) -> dict[str, float]:
    templates = _strategy_templates(compound)
    urgency_input = (
        ((max(0, int(tyre_age)) + (0.35 * float(max(1, int(horizon))))) - _compound_service_life(compound)) / 2.8
        + (10.0 * max(0.0, float(deg_rate) - 0.03))
    )
    urgency = _sigmoid(float(urgency_input))

    raw = {
        "hold_track_position": 0.58 * (1.0 - urgency) + 0.04,
        "one_stop_conservative": 0.26 + (0.08 * urgency),
        "two_stop_balanced": 0.10 + (0.30 * urgency),
        "two_stop_aggressive": 0.02 + (0.18 * (urgency**1.3)),
    }
    template_names = {template.name for template in templates}
    clipped = {
        name: max(1e-6, float(value))
        for name, value in raw.items()
        if name in template_names
    }
    total = float(sum(clipped.values()))
    if total <= 0.0:
        uniform = 1.0 / float(max(1, len(templates)))
        return {template.name: uniform for template in templates}
    return {name: float(value / total) for name, value in clipped.items()}


def _sample_strategy_template(
    *,
    compound: object,
    tyre_age: int,
    deg_rate: float,
    horizon: int,
    rng: np.random.Generator,
) -> StrategyTemplate:
    templates = _strategy_templates(compound)
    probabilities = _strategy_template_probabilities(
        compound=compound,
        tyre_age=tyre_age,
        deg_rate=deg_rate,
        horizon=horizon,
    )
    names = [template.name for template in templates]
    probs = np.asarray([probabilities.get(name, 0.0) for name in names], dtype=float)
    probs_sum = float(np.sum(probs))
    if probs_sum <= 0.0:
        probs = np.full(len(names), 1.0 / float(max(1, len(names))), dtype=float)
    else:
        probs = probs / probs_sum
    selected_name = str(rng.choice(names, p=probs))
    for template in templates:
        if template.name == selected_name:
            return template
    return templates[0]


def _sample_planned_pit_steps(
    *,
    template: StrategyTemplate,
    tyre_age: int,
    horizon: int,
    rng: np.random.Generator,
) -> list[int]:
    if template.first_pit_age_mean is None:
        return []

    current_age = max(0, int(tyre_age))
    horizon_laps = max(1, int(horizon))

    first_target_age = float(
        rng.normal(
            loc=float(template.first_pit_age_mean),
            scale=max(0.1, float(template.first_pit_age_std)),
        )
    )
    first_step = int(np.rint(first_target_age - float(current_age)))
    if first_step < 1:
        first_step = 1
    if first_step > horizon_laps:
        return []

    steps = [first_step]
    if template.second_stint_interval_mean is not None:
        next_interval = float(
            rng.normal(
                loc=float(template.second_stint_interval_mean),
                scale=max(0.1, float(template.second_stint_interval_std)),
            )
        )
        second_step = first_step + max(6, int(np.rint(next_interval)))
        if 1 <= second_step <= horizon_laps:
            steps.append(second_step)

    return sorted({int(step) for step in steps if 1 <= int(step) <= horizon_laps})


def _sample_next_compound(current_compound: object, rng: np.random.Generator) -> str:
    current = _normalized_compound(current_compound)
    if current == "SOFT":
        choices = ["MEDIUM", "HARD", "SOFT"]
        probs = [0.55, 0.35, 0.10]
    elif current == "MEDIUM":
        choices = ["HARD", "SOFT", "MEDIUM"]
        probs = [0.50, 0.35, 0.15]
    elif current == "HARD":
        choices = ["MEDIUM", "SOFT", "HARD"]
        probs = [0.60, 0.25, 0.15]
    elif current in {"INTER", "WET"}:
        choices = [current]
        probs = [1.0]
    else:
        choices = ["MEDIUM", "HARD", "SOFT"]
        probs = [0.50, 0.30, 0.20]
    return str(rng.choice(choices, p=np.asarray(probs, dtype=float)))


def _infer_rollout_regime(snapshot: pd.DataFrame) -> str:
    if snapshot.empty:
        return "green"

    if "is_sc_vsc" in snapshot.columns:
        sc_share = pd.to_numeric(snapshot["is_sc_vsc"], errors="coerce").fillna(0.0).mean()
        if float(sc_share) >= 0.5:
            return "sc_vsc"
    if "is_yellow" in snapshot.columns:
        yellow_share = pd.to_numeric(snapshot["is_yellow"], errors="coerce").fillna(0.0).mean()
        if float(yellow_share) >= 0.5:
            return "yellow"

    if "track_status" in snapshot.columns:
        counts = {"green": 0, "yellow": 0, "sc_vsc": 0}
        for value in snapshot["track_status"].tolist():
            flags = parse_track_status(value)
            if flags.is_sc_vsc:
                counts["sc_vsc"] += 1
            elif flags.is_yellow:
                counts["yellow"] += 1
            elif flags.is_greenish:
                counts["green"] += 1
        if counts["sc_vsc"] > max(counts["yellow"], counts["green"]):
            return "sc_vsc"
        if counts["yellow"] > counts["green"]:
            return "yellow"

    return "green"


def _advance_rollout_regime(current: str, rng: np.random.Generator) -> str:
    regime = str(current or "green")
    u = float(rng.random())
    if regime == "sc_vsc":
        if u < 0.38:
            return "green"
        if u < 0.52:
            return "yellow"
        return "sc_vsc"
    if regime == "yellow":
        if u < 0.22:
            return "sc_vsc"
        if u < 0.47:
            return "green"
        return "yellow"
    if u < 0.035:
        return "sc_vsc"
    if u < 0.085:
        return "yellow"
    return "green"


def _regime_lap_factors(regime: str) -> tuple[float, float, float]:
    if regime == "sc_vsc":
        return 11.0, 0.30, 1.25
    if regime == "yellow":
        return 4.0, 0.70, 1.10
    return 0.0, 1.0, 1.0


def _sample_pit_loss_seconds(regime: str, rng: np.random.Generator) -> float:
    mean_loss = 21.0
    if regime == "yellow":
        mean_loss = 15.5
    elif regime == "sc_vsc":
        mean_loss = 11.0
    sampled = float(rng.normal(loc=mean_loss, scale=1.4))
    return max(5.0, sampled)


def _mc_observability_payload(
    *,
    positions: np.ndarray,
    driver_ids: list[str],
    race_time_start: np.ndarray,
    top_drivers: Optional[int],
    max_position: Optional[int],
) -> dict[str, Any]:
    if positions.size == 0 or not driver_ids:
        return {
            "position_probabilities": [],
            "pairwise_ahead_probabilities": [],
            "observability_top_drivers": 0,
            "observability_max_position": 0,
        }

    total_drivers = len(driver_ids)
    if top_drivers is None:
        selected_count = int(total_drivers)
    else:
        selected_count = max(2, min(total_drivers, int(top_drivers)))
    if max_position is None:
        max_position_effective = int(total_drivers)
    else:
        max_position_effective = max(1, min(total_drivers, int(max_position)))

    # Use current race-time ordering to focus on front-running drivers.
    order = np.argsort(race_time_start, kind="mergesort")
    selected = order[:selected_count]
    selected_ids = [str(driver_ids[idx]) for idx in selected]
    effective_samples = int(positions.shape[0])

    position_probabilities: list[dict[str, Any]] = []
    for idx in selected:
        driver_id = str(driver_ids[idx])
        sampled = np.clip(np.rint(positions[:, idx]).astype(int), 1, total_drivers)
        counts = np.bincount(sampled, minlength=total_drivers + 1)
        probabilities = counts[1 : max_position_effective + 1].astype(float) / float(max(1, effective_samples))
        for position, probability in enumerate(probabilities, start=1):
            position_probabilities.append(
                {
                    "driver_id": driver_id,
                    "position": int(position),
                    "probability": float(probability),
                }
            )

    pairwise_ahead_probabilities: list[dict[str, Any]] = []
    for left_idx in selected:
        for right_idx in selected:
            if int(left_idx) == int(right_idx):
                continue
            left_driver = str(driver_ids[left_idx])
            right_driver = str(driver_ids[right_idx])
            prob = float(np.mean(positions[:, left_idx] < positions[:, right_idx]))
            pairwise_ahead_probabilities.append(
                {
                    "driver_a": left_driver,
                    "driver_b": right_driver,
                    "probability_a_ahead_b": prob,
                }
            )

    return {
        "position_probabilities": position_probabilities,
        "pairwise_ahead_probabilities": pairwise_ahead_probabilities,
        "observability_top_drivers": int(selected_count),
        "observability_max_position": int(max_position_effective),
        "observability_driver_ids": selected_ids,
    }


def _mc_position_distribution(
    snapshot: pd.DataFrame,
    states: dict[str, FilterState],
    baseline: BaselineModel,
    cfg: FilterConfig,
    horizon_laps: int,
    seed: int,
    requested_samples: int = 1000,
    max_mc_work: int = 250000,
    emit_observability: bool = False,
    observability_top_drivers: Optional[int] = None,
    observability_max_position: Optional[int] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    requested = max(50, int(requested_samples))
    max_work_limit = max(1000, int(max_mc_work))
    observability_stub = {
        "position_probabilities": [],
        "pairwise_ahead_probabilities": [],
        "observability_top_drivers": 0,
        "observability_max_position": 0,
        "observability_driver_ids": [],
    }

    if snapshot.empty:
        out = snapshot.copy()
        out["position_dist_enabled"] = False
        summary = {
            "position_dist_enabled": False,
            "position_dist_disabled_reason": "empty_snapshot",
            "mc_samples_requested": int(requested),
            "mc_samples_effective": 0,
            "mc_samples_reduction_reason": None,
            "max_mc_work": int(max_work_limit),
        }
        if emit_observability:
            summary.update(observability_stub)
        return out, summary

    out = snapshot.copy()
    out["position_dist_enabled"] = False

    if "race_time_seconds" in out.columns:
        race_time = pd.to_numeric(out["race_time_seconds"], errors="coerce")
    else:
        race_time = pd.Series(index=out.index, data=np.nan, dtype=float)
    race_time_array = race_time.to_numpy(dtype=float)
    valid_race_time_mask = np.isfinite(race_time_array)
    valid_race_time_count = int(valid_race_time_mask.sum())
    if valid_race_time_count == 0:
        summary = {
            "position_dist_enabled": False,
            "position_dist_disabled_reason": "missing_race_time_seconds",
            "mc_samples_requested": int(requested),
            "mc_samples_effective": 0,
            "mc_samples_reduction_reason": None,
            "max_mc_work": int(max_work_limit),
        }
        if emit_observability:
            summary.update(observability_stub)
        return out, summary
    invalid_race_time_count = int(len(race_time_array) - valid_race_time_count)
    race_time_start = race_time_array.astype(float, copy=True)
    invalid_race_time_penalty_seconds: Optional[float] = None
    if invalid_race_time_count > 0:
        max_valid_time = float(np.max(race_time_array[valid_race_time_mask]))
        invalid_race_time_penalty_seconds = max(600.0, float(max(1, int(horizon_laps))) * 30.0)
        invalid_indices = np.flatnonzero(~valid_race_time_mask)
        for offset, idx in enumerate(invalid_indices, start=1):
            race_time_start[idx] = max_valid_time + invalid_race_time_penalty_seconds + float(offset)

    driver_ids = out["driver_id"].astype(str).tolist()
    if len(driver_ids) <= 1:
        summary = {
            "position_dist_enabled": False,
            "position_dist_disabled_reason": "insufficient_driver_count",
            "mc_samples_requested": int(requested),
            "mc_samples_effective": 0,
            "mc_samples_reduction_reason": None,
            "max_mc_work": int(max_work_limit),
        }
        if emit_observability:
            summary.update(observability_stub)
        return out, summary

    work = len(driver_ids) * int(horizon_laps) * requested
    reduction_reason: Optional[str] = None
    if work > max_work_limit and len(driver_ids) > 0 and horizon_laps > 0:
        effective_samples = max(50, int(max_work_limit // (len(driver_ids) * int(horizon_laps))))
        reduction_reason = (
            f"work={work} exceeded max_mc_work={max_work_limit}; "
            f"mc_samples reduced from {requested} to {effective_samples}"
        )
    else:
        effective_samples = requested

    rng = np.random.default_rng(int(seed))
    positions = np.zeros((effective_samples, len(driver_ids)), dtype=float)
    if "lap_last" in out.columns:
        lap_last = pd.to_numeric(out["lap_last"], errors="coerce")
    else:
        lap_last = pd.Series(index=out.index, data=0.0, dtype=float)
    base_laps = lap_last.fillna(0).astype(int).to_numpy(dtype=int)
    if "tyre_age" in out.columns:
        tyre_age_series = pd.to_numeric(out["tyre_age"], errors="coerce").fillna(0.0)
    else:
        tyre_age_series = pd.Series(
            [float(states.get(driver_id).tyre_age) if states.get(driver_id) is not None else 0.0 for driver_id in driver_ids],
            index=out.index,
            dtype=float,
        )
    tyre_age_start = tyre_age_series.clip(lower=0.0).astype(int).to_numpy(dtype=int)
    if "compound" in out.columns:
        compound_start = out["compound"].fillna("UNKNOWN").astype(str).tolist()
    else:
        compound_start = ["UNKNOWN"] * len(driver_ids)
    regime_start = _infer_rollout_regime(out)
    horizon = int(horizon_laps)

    A = np.asarray([[1.0, 1.0], [0.0, float(cfg.phi)]], dtype=float)
    Q = np.asarray([[float(cfg.q_pace) ** 2, 0.0], [0.0, float(cfg.q_deg) ** 2]], dtype=float)
    pit_events_total = 0
    regime_sc_vsc_steps = 0
    regime_yellow_steps = 0
    strategy_counts: dict[str, int] = {}
    strategy_assignments_total = 0

    for sample_idx in range(effective_samples):
        final_laps = base_laps + horizon
        final_times = race_time_start.astype(float, copy=True)
        means: list[Optional[np.ndarray]] = []
        covs: list[Optional[np.ndarray]] = []
        compounds = list(compound_start)
        tyre_age = tyre_age_start.astype(int, copy=True)
        strategy_templates: list[Optional[StrategyTemplate]] = []
        planned_pit_steps: list[Optional[list[int]]] = []
        planned_pit_index: list[Optional[int]] = []
        regime = str(regime_start)

        for driver_idx, driver_id in enumerate(driver_ids):
            state = states.get(driver_id)
            if state is None:
                means.append(None)
                covs.append(None)
                strategy_templates.append(None)
                planned_pit_steps.append(None)
                planned_pit_index.append(None)
                final_times[driver_idx] = np.inf
                continue
            mean_init = state.mean.astype(float).copy()
            cov_init = state.cov.astype(float).copy()
            means.append(mean_init)
            covs.append(cov_init)

            strategy = _sample_strategy_template(
                compound=compounds[driver_idx],
                tyre_age=int(tyre_age[driver_idx]),
                deg_rate=float(mean_init[1]) if mean_init.size > 1 else 0.0,
                horizon=horizon,
                rng=rng,
            )
            strategy_templates.append(strategy)
            planned_steps = _sample_planned_pit_steps(
                template=strategy,
                tyre_age=int(tyre_age[driver_idx]),
                horizon=horizon,
                rng=rng,
            )
            planned_pit_steps.append(planned_steps)
            planned_pit_index.append(0)
            strategy_counts[strategy.name] = int(strategy_counts.get(strategy.name, 0) + 1)
            strategy_assignments_total += 1

        for step in range(1, horizon + 1):
            regime = _advance_rollout_regime(regime, rng)
            regime_offset, regime_pace_scale, regime_noise_scale = _regime_lap_factors(regime)
            if regime == "sc_vsc":
                regime_sc_vsc_steps += 1
            elif regime == "yellow":
                regime_yellow_steps += 1

            for driver_idx, driver_id in enumerate(driver_ids):
                mean = means[driver_idx]
                cov = covs[driver_idx]
                if mean is None or cov is None:
                    continue

                pit_now = False
                target_steps = planned_pit_steps[driver_idx]
                pointer = planned_pit_index[driver_idx]
                if target_steps is not None and pointer is not None and pointer < len(target_steps):
                    target_step = int(target_steps[pointer])
                    if int(step) == target_step:
                        pit_now = True
                    elif regime == "sc_vsc" and int(step) < target_step and (target_step - int(step)) <= 2:
                        pull_forward_probability = 0.55 if (target_step - int(step)) == 1 else 0.35
                        if float(rng.random()) < pull_forward_probability:
                            pit_now = True

                if pit_now:
                    pit_events_total += 1
                    final_times[driver_idx] += _sample_pit_loss_seconds(regime, rng)
                    compounds[driver_idx] = _sample_next_compound(compounds[driver_idx], rng)
                    pit_prior = initialize_filter_state(compounds[driver_idx], cfg)
                    mean = pit_prior.mean.astype(float).copy()
                    cov = pit_prior.cov.astype(float).copy()
                    tyre_age[driver_idx] = 0
                    if pointer is not None:
                        planned_pit_index[driver_idx] = int(pointer) + 1

                mean_pred = A @ mean
                cov_pred = (A @ cov @ A.T) + Q
                cov_pred = 0.5 * (cov_pred + cov_pred.T)
                cov_pred += np.eye(2, dtype=float) * 1e-9

                try:
                    sampled_state = rng.multivariate_normal(mean=mean_pred, cov=cov_pred)
                except Exception:
                    sampled_state = mean_pred

                lap_number = int(base_laps[driver_idx]) + step
                lap_baseline = baseline.value_at(lap_number)
                noise_sigma = np.sqrt(max(float(cfg.r_obs), 1e-6)) * float(regime_noise_scale)
                lap_noise = float(rng.normal(0.0, noise_sigma))
                lap_time = float(
                    lap_baseline
                    + regime_offset
                    + (regime_pace_scale * float(sampled_state[0]))
                    + lap_noise
                )
                final_times[driver_idx] += max(0.1, lap_time)

                means[driver_idx] = sampled_state
                covs[driver_idx] = cov_pred
                tyre_age[driver_idx] = int(tyre_age[driver_idx]) + 1

        # Race order is lap-count first, then cumulative time within the same lap.
        order = np.lexsort((final_times, -final_laps))
        sampled_positions = np.empty(len(driver_ids), dtype=float)
        sampled_positions[order] = np.arange(1, len(driver_ids) + 1, dtype=float)
        positions[sample_idx, :] = sampled_positions

    p_win = (positions == 1).mean(axis=0)
    p_top3 = (positions <= 3).mean(axis=0)
    p_top10 = (positions <= 10).mean(axis=0)
    exp_pos = positions.mean(axis=0)
    pos_p10 = np.percentile(positions, 10, axis=0)
    pos_p50 = np.percentile(positions, 50, axis=0)
    pos_p90 = np.percentile(positions, 90, axis=0)

    out["exp_pos_H"] = exp_pos
    out["p_win_H"] = np.clip(p_win, 0.0, 1.0)
    out["p_top3_H"] = np.clip(p_top3, 0.0, 1.0)
    out["p_top10_H"] = np.clip(p_top10, 0.0, 1.0)
    out["pos_p10_H"] = pos_p10
    out["pos_p50_H"] = pos_p50
    out["pos_p90_H"] = pos_p90
    out["position_dist_enabled"] = True

    sum_p_win = float(np.sum(out["p_win_H"].to_numpy(dtype=float)))
    strategy_mix = {
        name: float(count / max(1, strategy_assignments_total))
        for name, count in sorted(strategy_counts.items(), key=lambda item: item[0])
    }

    horizon_steps_total = max(1, int(effective_samples) * int(horizon))

    summary: dict[str, Any] = {
        "position_dist_enabled": True,
        "position_dist_disabled_reason": None,
        "mc_samples_requested": int(requested),
        "mc_samples_effective": int(effective_samples),
        "mc_samples_reduction_reason": reduction_reason,
        "max_mc_work": int(max_work_limit),
        "sum_p_win": sum_p_win,
        "invalid_race_time_count": int(invalid_race_time_count),
        "invalid_race_time_penalty_seconds": invalid_race_time_penalty_seconds,
        "rollout_strategy_enabled": True,
        "rollout_regime_initial": str(regime_start),
        "rollout_sc_vsc_share": float(regime_sc_vsc_steps / horizon_steps_total),
        "rollout_yellow_share": float(regime_yellow_steps / horizon_steps_total),
        "rollout_pit_events_total": int(pit_events_total),
        "rollout_pit_events_mean": float(pit_events_total / max(1, int(effective_samples))),
        "rollout_strategy_mix": strategy_mix,
        "rollout_strategy_assignments": int(strategy_assignments_total),
    }
    if emit_observability:
        summary.update(
            _mc_observability_payload(
                positions=positions,
                driver_ids=driver_ids,
                race_time_start=race_time_start,
                top_drivers=observability_top_drivers,
                max_position=observability_max_position,
            )
        )
    return out, summary


def _build_snapshot(trace: pd.DataFrame) -> pd.DataFrame:
    if trace.empty:
        return pd.DataFrame()
    snapshot = (
        trace.sort_values(["driver_id", "lap_number", "timestamp"], kind="mergesort")
        .groupby("driver_id", sort=False)
        .tail(1)
        .copy()
    )
    snapshot = snapshot.reset_index(drop=True)
    base_columns = [
        "driver_id",
        "driver_name",
        "lap_number",
        "stint_id",
        "compound",
        "tyre_age",
        "race_status",
        "source",
        "track_status",
        "is_sc_vsc",
        "is_yellow",
        "race_time_seconds",
        "gap_to_leader_seconds",
        "pace_penalty_mean",
        "pace_penalty_std",
        "deg_rate_mean",
        "deg_rate_std",
        "next_lap_mean",
        "next_lap_std",
        "next_lap_pi90_low",
        "next_lap_pi90_high",
    ]
    telemetry_columns = [
        col
        for col in trace.columns
        if col.startswith("telemetry_") or col in TELEMETRY_STRATEGY_FEED_COLUMNS
    ]
    snapshot = snapshot[[col for col in [*base_columns, *telemetry_columns] if col in snapshot.columns]]
    snapshot = snapshot.rename(columns={"lap_number": "lap_last"})
    return snapshot


def _finalize_output_mapping(snapshot: pd.DataFrame, summary: dict[str, Any], horizon_laps: int) -> pd.DataFrame:
    out = snapshot.copy()
    out["horizon_laps"] = int(horizon_laps)

    if bool(summary.get("position_dist_enabled", False)):
        out["pred"] = pd.to_numeric(out["exp_pos_H"], errors="coerce")
        out["rank"] = out["pred"].rank(method="first", ascending=True).astype(int)
        out["proba_top10"] = pd.to_numeric(out["p_top10_H"], errors="coerce")
        out["proba_top3"] = pd.to_numeric(out["p_top3_H"], errors="coerce")
    else:
        out["pred"] = pd.to_numeric(out["next_lap_mean"], errors="coerce")
        out["rank"] = out["pred"].rank(method="first", ascending=True).astype(int)
        out["proba_top10"] = float("nan")
        out["proba_top3"] = float("nan")

    out = out.sort_values("pred", ascending=True, kind="mergesort").reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out


def run_live_race_prediction(
    config: PredictionConfig,
    *,
    telemetry_adapter: Optional[TelemetryFeatureAdapter] = None,
    strategy_adapter: Optional[StrategyPolicyAdapter] = None,
) -> LiveRunResult:
    notes: list[str] = []

    telemetry = telemetry_adapter or NoopTelemetryFeatureAdapter()
    strategy = strategy_adapter or NoopStrategyPolicyAdapter()

    source_result = load_live_observations(config)
    notes.extend(source_result.notes)
    observations = source_result.frame
    if observations.empty:
        summary = {
            "available": False,
            "reason": "live_observations_unavailable",
            "source_used": source_result.source_used,
            "generated_at": _utc_now(),
        }
        return LiveRunResult(snapshot=pd.DataFrame(), trace=pd.DataFrame(), summary=summary, notes=notes)

    observations = observations.copy()
    observations = observations.sort_values(["driver_id", "lap_number", "timestamp"], kind="mergesort")
    cutoff_lap = getattr(config, "f1_live_replay_cutoff_lap", None)
    if cutoff_lap is not None:
        cutoff_value = int(cutoff_lap)
        lap_filter = pd.to_numeric(observations.get("lap_number"), errors="coerce") <= float(cutoff_value)
        observations = observations.loc[lap_filter].copy()
        notes.append(f"Live replay cutoff active: using observations through lap {cutoff_value}.")
        if observations.empty:
            summary = {
                "available": False,
                "reason": "live_observations_empty_after_cutoff",
                "source_used": source_result.source_used,
                "cutoff_lap": cutoff_value,
                "generated_at": _utc_now(),
            }
            return LiveRunResult(snapshot=pd.DataFrame(), trace=pd.DataFrame(), summary=summary, notes=notes)

    lap_number_numeric = pd.to_numeric(observations.get("lap_number"), errors="coerce")
    baseline_cache: dict[int, BaselineModel] = {}

    def _baseline_before_lap(lap_number: int) -> BaselineModel:
        lap_key = int(lap_number)
        if lap_key not in baseline_cache:
            prior_observations = observations.loc[lap_number_numeric < float(lap_key)]
            baseline_cache[lap_key] = build_event_lap_baseline(prior_observations, min_clean_obs_per_lap=8)
        return baseline_cache[lap_key]

    filter_cfg = FilterConfig()

    telemetry_trace_columns: list[str] = []
    telemetry_strategy_columns: list[str] = []
    telemetry_disabled_reason: Optional[str] = None
    try:
        telemetry_features = telemetry.build_lap_features(observations)
        if not isinstance(telemetry_features, pd.DataFrame):
            telemetry_disabled_reason = "telemetry_adapter_returned_non_dataframe"
        else:
            telemetry_features = telemetry_features.reindex(observations.index)
            for column in telemetry_features.columns:
                trace_column = f"telemetry_{column}"
                observations[trace_column] = telemetry_features[column]
                telemetry_trace_columns.append(trace_column)
            for column in TELEMETRY_STRATEGY_FEED_COLUMNS:
                if column in telemetry_features.columns and column not in observations.columns:
                    observations[column] = telemetry_features[column]
                    telemetry_strategy_columns.append(column)
            if telemetry_trace_columns:
                notes.append(f"Live telemetry adapter produced {len(telemetry_trace_columns)} feature columns.")
    except Exception as exc:
        telemetry_disabled_reason = f"{type(exc).__name__}: {exc}"
        notes.append(f"Live telemetry adapter unavailable: {telemetry_disabled_reason}.")

    event_key = int(pd.to_numeric(observations.get("event_key"), errors="coerce").dropna().iloc[0])
    seed = _event_seed(event_key=event_key, base_seed=int(config.f1_live_seed))

    states: dict[str, FilterState] = {}
    trace_rows: list[dict[str, Any]] = []

    for driver_id, driver_laps in observations.groupby("driver_id", sort=False):
        driver_frame = driver_laps.sort_values(["lap_number", "timestamp"], kind="mergesort")
        if driver_frame.empty:
            continue

        first_compound = driver_frame.iloc[0].get("compound")
        state = initialize_filter_state(first_compound, filter_cfg)

        for _, row in driver_frame.iterrows():
            lap_number = int(as_float(row.get("lap_number"), 0.0))
            stint_id = int(as_float(row.get("stint_id"), 1.0))
            compound = str(row.get("compound") or "UNKNOWN")
            is_box_lap = bool(row.get("is_box_lap", False))
            is_accurate = bool(row.get("is_accurate", False))
            lap_time = as_float(row.get("lap_time_seconds"))
            race_time_seconds = as_float(row.get("race_time_seconds"))

            flags = parse_track_status(row.get("track_status"))

            reset_applied = False
            stint_transition = (state.last_stint_id is None) or (state.last_stint_id != stint_id)
            if stint_transition and (not is_box_lap):
                state = reset_filter_state(state, compound=compound, cfg=filter_cfg)
                reset_applied = True

            lap_baseline_model = _baseline_before_lap(lap_number)
            baseline_current = lap_baseline_model.value_at(lap_number)
            mean_pred, cov_pred, one_step_pred_mean, one_step_pred_std = lap_one_step_prediction(
                state=state,
                baseline_current=baseline_current,
                cfg=filter_cfg,
            )

            gate = apply_track_gating(filter_cfg.r_obs, flags)
            can_assimilate = (
                (not is_box_lap)
                and bool(is_accurate)
                and np.isfinite(lap_time)
                and float(lap_time) > 0.0
            )

            if (not can_assimilate) or gate.skip_update:
                mean_post = mean_pred
                cov_post = cov_pred
                update_meta: dict[str, Any] = {
                    "innovation": float("nan"),
                    "S": float("nan"),
                    "robust_applied": False,
                    "robust_scale": 1.0,
                    "skip_reason": "not_assimilable" if not can_assimilate else "track_status_skip",
                }
            else:
                observation = float(lap_time - baseline_current)
                mean_post, cov_post, update_meta = update_state(
                    mean_pred=mean_pred,
                    cov_pred=cov_pred,
                    observation=observation,
                    r_effective=gate.r_effective,
                    cfg=filter_cfg,
                )

            state.mean = mean_post
            state.cov = cov_post
            state.last_stint_id = stint_id

            if can_assimilate and (not gate.skip_update):
                if not reset_applied:
                    state.tyre_age += 1
                else:
                    state.tyre_age = 0
                state.assimilated_laps += 1

            baseline_next = lap_baseline_model.value_at(lap_number + 1)
            next_lap_mean, next_lap_std = next_lap_distribution(
                state=state,
                baseline_next=baseline_next,
                cfg=filter_cfg,
            )

            trace_row: dict[str, Any] = {
                "event_key": event_key,
                "source": source_result.source_used,
                "driver_id": str(driver_id),
                "driver_name": str(row.get("driver_name") or driver_id),
                "lap_number": lap_number,
                "stint_id": stint_id,
                "compound": compound,
                "tyre_age": int(state.tyre_age),
                "is_box_lap": bool(is_box_lap),
                "is_accurate": bool(is_accurate),
                "track_status": str(row.get("track_status") or ""),
                "is_red": bool(flags.is_red),
                "is_sc_vsc": bool(flags.is_sc_vsc),
                "is_yellow": bool(flags.is_yellow),
                "is_greenish": bool(flags.is_greenish),
                "gate_mode": gate.mode,
                "gate_skip_update": bool(gate.skip_update),
                "gate_note": gate.note,
                "r_effective": float(gate.r_effective),
                "reset_applied": bool(reset_applied),
                "baseline_lap": float(baseline_current),
                "lap_time_seconds": lap_time,
                "one_step_pred_mean": float(one_step_pred_mean),
                "one_step_pred_std": float(one_step_pred_std),
                "one_step_error": float(lap_time - one_step_pred_mean) if np.isfinite(lap_time) else float("nan"),
                "pace_penalty_mean": float(state.mean[0]),
                "pace_penalty_std": float(np.sqrt(max(state.cov[0, 0], 1e-9))),
                "deg_rate_mean": float(state.mean[1]),
                "deg_rate_std": float(np.sqrt(max(state.cov[1, 1], 1e-9))),
                "innovation": float(update_meta.get("innovation", float("nan"))),
                "innovation_var": float(update_meta.get("S", float("nan"))),
                "robust_applied": bool(update_meta.get("robust_applied", False)),
                "robust_scale": float(update_meta.get("robust_scale", 1.0)),
                "skip_reason": update_meta.get("skip_reason"),
                "next_lap_mean": float(next_lap_mean),
                "next_lap_std": float(next_lap_std),
                "next_lap_pi90_low": float(next_lap_mean - (1.645 * next_lap_std)),
                "next_lap_pi90_high": float(next_lap_mean + (1.645 * next_lap_std)),
                "race_time_seconds": race_time_seconds,
                "gap_to_leader_seconds": as_float(row.get("gap_to_leader_seconds")),
                "timestamp": as_float(row.get("timestamp")),
                "eval_included": bool(can_assimilate and (not gate.skip_update)),
                "assim_laps_driver": int(state.assimilated_laps),
                "race_status": "running",
            }
            for column in [*telemetry_trace_columns, *telemetry_strategy_columns]:
                trace_row[column] = _json_safe_scalar(row.get(column))
            trace_rows.append(trace_row)

        states[str(driver_id)] = state

    trace = pd.DataFrame(trace_rows)
    snapshot = _build_snapshot(trace)
    baseline = build_event_lap_baseline(observations, min_clean_obs_per_lap=8)

    horizon = max(1, int(config.f1_live_horizon_laps))
    snapshot_with_dist, dist_summary = _mc_position_distribution(
        snapshot=snapshot,
        states=states,
        baseline=baseline,
        cfg=filter_cfg,
        horizon_laps=horizon,
        seed=seed,
    )

    strategy_policy_columns: list[str] = []
    strategy_policy_disabled_reason: Optional[str] = None
    try:
        strategy_actions = strategy.evaluate_actions(snapshot_with_dist)
        if not isinstance(strategy_actions, pd.DataFrame):
            strategy_policy_disabled_reason = "strategy_adapter_returned_non_dataframe"
        elif strategy_actions.empty and not snapshot_with_dist.empty:
            strategy_policy_disabled_reason = "strategy_adapter_returned_empty_actions"
        else:
            strategy_actions = strategy_actions.reindex(snapshot_with_dist.index)
            for column in strategy_actions.columns:
                snapshot_with_dist[column] = strategy_actions[column]
                strategy_policy_columns.append(column)
    except Exception as exc:
        strategy_policy_disabled_reason = f"{type(exc).__name__}: {exc}"
        notes.append(f"Live strategy policy unavailable: {strategy_policy_disabled_reason}.")

    snapshot_final = _finalize_output_mapping(snapshot_with_dist, dist_summary, horizon_laps=horizon)

    replay_eval = evaluate_live_replay(trace, warmup_laps=3)
    trace_meta = _write_trace(trace, config=config, event_key=event_key)

    live_summary: dict[str, Any] = {
        "available": True,
        "source_used": source_result.source_used,
        "event_key": int(event_key),
        "year": int(config.year),
        "round_number": int(config.round_number),
        "f1_live_model": str(config.f1_live_model),
        "f1_live_source": str(config.f1_live_source),
        "horizon_laps": int(horizon),
        "replay_cutoff_lap": int(cutoff_lap) if cutoff_lap is not None else None,
        "drivers_processed": int(snapshot_final["driver_id"].nunique()) if not snapshot_final.empty else 0,
        "laps_processed": int(pd.to_numeric(trace.get("lap_number"), errors="coerce").max())
        if not trace.empty
        else 0,
        "position_dist_enabled": bool(dist_summary.get("position_dist_enabled", False)),
        "position_dist_disabled_reason": dist_summary.get("position_dist_disabled_reason"),
        "mc_samples_requested": int(dist_summary.get("mc_samples_requested", 1000)),
        "mc_samples_effective": int(dist_summary.get("mc_samples_effective", 0)),
        "mc_samples_reduction_reason": dist_summary.get("mc_samples_reduction_reason"),
        "max_mc_work": int(dist_summary.get("max_mc_work", 250000)),
        "sum_p_win": dist_summary.get("sum_p_win"),
        "rollout_strategy_enabled": bool(dist_summary.get("rollout_strategy_enabled", False)),
        "rollout_regime_initial": dist_summary.get("rollout_regime_initial"),
        "rollout_sc_vsc_share": dist_summary.get("rollout_sc_vsc_share"),
        "rollout_yellow_share": dist_summary.get("rollout_yellow_share"),
        "rollout_pit_events_total": int(dist_summary.get("rollout_pit_events_total", 0)),
        "rollout_pit_events_mean": dist_summary.get("rollout_pit_events_mean"),
        "rollout_strategy_mix": dist_summary.get("rollout_strategy_mix"),
        "rollout_strategy_assignments": int(dist_summary.get("rollout_strategy_assignments", 0)),
        "telemetry_features_enabled": bool(telemetry_trace_columns),
        "telemetry_features_disabled_reason": telemetry_disabled_reason,
        "telemetry_feature_columns": telemetry_trace_columns,
        "telemetry_strategy_feed_columns": telemetry_strategy_columns,
        "strategy_policy_enabled": bool(strategy_policy_columns),
        "strategy_policy_disabled_reason": strategy_policy_disabled_reason,
        "strategy_policy_columns": strategy_policy_columns,
        "strategy_action_counts": (
            snapshot_final["recommended_action"].value_counts(dropna=False).to_dict()
            if "recommended_action" in snapshot_final.columns
            else {}
        ),
        "strategy_mean_pit_urgency": (
            float(pd.to_numeric(snapshot_final["pit_urgency"], errors="coerce").mean())
            if "pit_urgency" in snapshot_final.columns
            else None
        ),
        "invalid_race_time_count": int(dist_summary.get("invalid_race_time_count", 0)),
        "invalid_race_time_penalty_seconds": dist_summary.get("invalid_race_time_penalty_seconds"),
        "seed_effective": int(seed),
        "trace_records": int(len(trace)),
        "replay_eval": replay_eval,
        "generated_at": _utc_now(),
        **trace_meta,
    }

    if not bool(live_summary.get("position_dist_enabled", False)):
        notes.append(
            f"Position distribution disabled: {live_summary.get('position_dist_disabled_reason') or 'unknown_reason'}."
        )
    else:
        sum_p_win = live_summary.get("sum_p_win")
        if isinstance(sum_p_win, (int, float)):
            notes.append(f"Live position distribution enabled: sum(p_win_H)={float(sum_p_win):.4f}.")
        pit_events_mean = live_summary.get("rollout_pit_events_mean")
        if isinstance(pit_events_mean, (int, float)):
            notes.append(f"Rollout pit events/sample={float(pit_events_mean):.2f}.")
        strategy_mix = live_summary.get("rollout_strategy_mix")
        if isinstance(strategy_mix, dict) and strategy_mix:
            lead_name, lead_share = max(strategy_mix.items(), key=lambda item: float(item[1]))
            notes.append(f"Rollout dominant strategy={lead_name} ({100.0 * float(lead_share):.1f}%).")
    if bool(live_summary.get("strategy_policy_enabled", False)):
        action_counts = live_summary.get("strategy_action_counts")
        if isinstance(action_counts, dict) and action_counts:
            lead_action, lead_count = max(action_counts.items(), key=lambda item: int(item[1]))
            notes.append(f"Live strategy policy action={lead_action} for {int(lead_count)} driver(s).")

    if replay_eval.get("available"):
        model_metrics = replay_eval.get("model", {})
        mae = model_metrics.get("mae") if isinstance(model_metrics, dict) else None
        rmse = model_metrics.get("rmse") if isinstance(model_metrics, dict) else None
        if isinstance(mae, (float, int)) and isinstance(rmse, (float, int)):
            notes.append(f"Replay one-step metrics: MAE={float(mae):.3f}, RMSE={float(rmse):.3f}.")
        if bool(replay_eval.get("baseline_arima_unavailable", False)):
            notes.append("ARIMA baseline unavailable: fallback to naive last-lap baseline.")

    return LiveRunResult(
        snapshot=snapshot_final,
        trace=trace,
        summary=live_summary,
        notes=notes,
    )
