#!/usr/bin/env python3
"""Evaluate Horizon A vs Horizon B on early live-race lap snapshots."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency for plot artifact
    plt = None

from rqp.live_runner import (
    _build_snapshot,
    _event_seed,
    _finalize_output_mapping,
)
from rqp.live_state_space import FilterConfig, FilterState, build_event_lap_baseline, parse_track_status
from rqp.providers import LocalWeekendProvider


NEVER_BEFORE_FINISH = "Never before finish"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _resolve_path(project_root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _parse_lap_cutoffs(raw: str) -> list[int]:
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    out = sorted({max(1, int(value)) for value in values})
    if not out:
        raise SystemExit("No lap cutoffs provided.")
    return out


def _parse_pct_cutoffs(raw: str) -> list[int]:
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    out: set[int] = set()
    for value in values:
        try:
            pct = int(round(float(value)))
        except Exception as exc:
            raise SystemExit(f"Invalid distance cutoff '{value}': {exc}") from exc
        pct = max(1, min(100, pct))
        out.add(int(pct))
    parsed = sorted(out)
    if not parsed:
        raise SystemExit("No distance cutoffs provided.")
    return parsed


def _parse_rounds(raw: str, available: Iterable[int]) -> list[int]:
    available_set = sorted({int(value) for value in available})
    if str(raw).strip().lower() == "all":
        return available_set
    requested = sorted({int(part.strip()) for part in str(raw).split(",") if part.strip()})
    return [value for value in requested if value in available_set]


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _trace_path_from_payload(payload: dict[str, Any], project_root: Path, payload_path: Path) -> Optional[Path]:
    candidates: list[Path] = []
    direct = payload.get("trace_path")
    if isinstance(direct, str) and direct.strip():
        candidates.append(Path(direct.strip()).expanduser())

    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        live_race = artifacts.get("live_race")
        if isinstance(live_race, dict):
            nested = live_race.get("trace_path")
            if isinstance(nested, str) and nested.strip():
                candidates.append(Path(nested.strip()).expanduser())

    for raw_path in candidates:
        if raw_path.is_absolute() and raw_path.exists():
            return raw_path
        rel_candidates = [
            (payload_path.parent / raw_path).resolve(),
            (project_root / raw_path).resolve(),
            (project_root / raw_path.name).resolve(),
        ]
        for rel in rel_candidates:
            if rel.exists():
                return rel
    return None


def _fallback_trace_path(project_root: Path, year: int, round_number: int) -> Optional[Path]:
    directory = project_root / "data" / "f1" / "live" / "artifacts" / str(int(year)) / f"round_{int(round_number):02d}"
    if not directory.exists():
        return None
    parquet_paths = sorted(directory.glob("live_trace_*.parquet"))
    if parquet_paths:
        return parquet_paths[-1]
    jsonl_paths = sorted(directory.glob("live_trace_*.jsonl"))
    if jsonl_paths:
        return jsonl_paths[-1]
    return None


def _read_trace(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise SystemExit(f"Unsupported trace format: {path}")


def _normalize_name_key(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip().lower()
    if not text:
        return ""
    return " ".join(text.split())


def _safe_float(value: object, default: float) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(number):
        return float(default)
    return number


def _safe_int(value: object, default: int) -> int:
    try:
        number = int(float(value))
    except Exception:
        return int(default)
    return int(number)


def _safe_optional_float(value: object) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    if not np.isfinite(number):
        return None
    return float(number)


def _total_laps_completed(trace: pd.DataFrame) -> Optional[int]:
    if trace.empty:
        return None
    laps = pd.to_numeric(trace.get("lap_number"), errors="coerce").dropna()
    laps = laps[laps > 0.0]
    if laps.empty:
        return None
    return int(max(1, int(np.floor(float(laps.max())))))


def _lap_cutoff_from_pct(total_laps_completed: int, pct_cutoff: int) -> int:
    total = max(1, int(total_laps_completed))
    pct = max(1.0, min(100.0, float(pct_cutoff)))
    lap = int(np.ceil((pct / 100.0) * float(total)))
    return int(min(total, max(1, lap)))


def _build_cutoff_plan(
    *,
    cutoff_mode: str,
    total_laps_completed: int,
    pct_cutoffs: list[int],
    lap_cutoffs: list[int],
) -> list[dict[str, Any]]:
    total = max(1, int(total_laps_completed))
    rows: list[dict[str, Any]] = []

    if cutoff_mode == "distance_pct":
        for pct in pct_cutoffs:
            lap_cutoff = _lap_cutoff_from_pct(total, pct)
            rows.append(
                {
                    "cutoff_mode": "distance_pct",
                    "cutoff_label": f"{int(pct)}%",
                    "cutoff_sort_value": float(pct),
                    "cutoff_pct_requested": int(pct),
                    "lap_cutoff": int(lap_cutoff),
                    "cutoff_pct_realized": float((100.0 * float(lap_cutoff)) / float(total)),
                }
            )
        return rows

    for lap in lap_cutoffs:
        lap_cutoff = int(min(total, max(1, int(lap))))
        rows.append(
            {
                "cutoff_mode": "lap",
                "cutoff_label": f"Lap {int(lap)}",
                "cutoff_sort_value": float(lap),
                "cutoff_pct_requested": None,
                "lap_cutoff": int(lap_cutoff),
                "cutoff_pct_realized": float((100.0 * float(lap_cutoff)) / float(total)),
            }
        )
    return rows


def _is_sc_vsc_red(track_status: object) -> bool:
    flags = parse_track_status(track_status)
    return bool(flags.is_sc_vsc or flags.is_red)


def _build_chaos_profile(
    trace: pd.DataFrame,
    *,
    total_laps_completed: int,
    clean_max_chaos_fraction: float,
    chaotic_min_chaos_fraction: float,
) -> dict[str, Any]:
    total = max(1, int(total_laps_completed))
    if trace.empty:
        return {
            "total_laps_completed": int(total),
            "sc_vsc_laps": 0,
            "chaos_fraction": 0.0,
            "has_sc_vsc_or_red": False,
            "chaos_segment": "clean",
        }

    frame = trace.copy()
    frame["lap_number"] = pd.to_numeric(frame.get("lap_number"), errors="coerce")
    frame = frame[frame["lap_number"].notna() & (frame["lap_number"] > 0.0)].copy()
    if frame.empty:
        return {
            "total_laps_completed": int(total),
            "sc_vsc_laps": 0,
            "chaos_fraction": 0.0,
            "has_sc_vsc_or_red": False,
            "chaos_segment": "clean",
        }

    frame["lap_number"] = frame["lap_number"].astype(int)
    lap_incidents = frame.groupby("lap_number", sort=True)["track_status"].apply(
        lambda values: any(_is_sc_vsc_red(value) for value in values),
    )
    sc_vsc_laps = int(lap_incidents.sum())
    chaos_fraction = float(sc_vsc_laps / float(total))
    if chaos_fraction <= float(clean_max_chaos_fraction):
        segment = "clean"
    elif chaos_fraction >= float(chaotic_min_chaos_fraction):
        segment = "chaotic"
    else:
        segment = "intermediate"

    return {
        "total_laps_completed": int(total),
        "sc_vsc_laps": int(sc_vsc_laps),
        "chaos_fraction": float(chaos_fraction),
        "has_sc_vsc_or_red": bool(sc_vsc_laps > 0),
        "chaos_segment": segment,
    }


def _build_states_from_trace(trace_cutoff: pd.DataFrame) -> dict[str, FilterState]:
    if trace_cutoff.empty:
        return {}
    tails = (
        trace_cutoff.sort_values(["driver_id", "lap_number", "timestamp"], kind="mergesort")
        .groupby("driver_id", sort=False)
        .tail(1)
        .copy()
    )
    states: dict[str, FilterState] = {}
    for _, row in tails.iterrows():
        driver_id = str(row.get("driver_id") or "").strip()
        if not driver_id:
            continue
        pace_mean = _safe_float(row.get("pace_penalty_mean"), 0.0)
        deg_mean = _safe_float(row.get("deg_rate_mean"), 0.0)
        pace_std = max(1e-3, _safe_float(row.get("pace_penalty_std"), 0.6))
        deg_std = max(1e-3, _safe_float(row.get("deg_rate_std"), 0.1))
        covariance = np.asarray(
            [
                [pace_std**2, 0.0],
                [0.0, deg_std**2],
            ],
            dtype=float,
        )
        states[driver_id] = FilterState(
            mean=np.asarray([pace_mean, deg_mean], dtype=float),
            cov=covariance,
            last_stint_id=_safe_int(row.get("stint_id"), 1),
            tyre_age=max(0, _safe_int(row.get("tyre_age"), 0)),
            assimilated_laps=max(0, _safe_int(row.get("assim_laps_driver"), 0)),
        )
    return states


def _mc_position_distribution_with_samples(
    snapshot: pd.DataFrame,
    *,
    states: dict[str, FilterState],
    baseline: Any,
    cfg: FilterConfig,
    horizon_laps: int,
    seed: int,
    requested_samples: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if snapshot.empty:
        out = snapshot.copy()
        out["position_dist_enabled"] = False
        return out, {
            "position_dist_enabled": False,
            "position_dist_disabled_reason": "empty_snapshot",
            "mc_samples_requested": int(requested_samples),
            "mc_samples_effective": 0,
            "mc_samples_reduction_reason": None,
            "max_mc_work": 250000,
        }

    out = snapshot.copy()
    out["position_dist_enabled"] = False

    race_time = pd.to_numeric(out.get("race_time_seconds"), errors="coerce")
    race_time_array = race_time.to_numpy(dtype=float)
    valid_race_time_mask = np.isfinite(race_time_array)
    valid_race_time_count = int(valid_race_time_mask.sum())
    if valid_race_time_count == 0:
        return out, {
            "position_dist_enabled": False,
            "position_dist_disabled_reason": "missing_race_time_seconds",
            "mc_samples_requested": int(requested_samples),
            "mc_samples_effective": 0,
            "mc_samples_reduction_reason": None,
            "max_mc_work": 250000,
        }

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
        return out, {
            "position_dist_enabled": False,
            "position_dist_disabled_reason": "insufficient_driver_count",
            "mc_samples_requested": int(requested_samples),
            "mc_samples_effective": 0,
            "mc_samples_reduction_reason": None,
            "max_mc_work": 250000,
        }

    max_mc_work = 250000
    horizon = max(1, int(horizon_laps))
    requested = max(50, int(requested_samples))
    work = len(driver_ids) * horizon * requested
    reduction_reason: Optional[str] = None
    if work > max_mc_work:
        effective_samples = max(100, int(max_mc_work // max(1, len(driver_ids) * horizon)))
        reduction_reason = (
            f"work={work} exceeded max_mc_work={max_mc_work}; "
            f"mc_samples reduced from {requested} to {effective_samples}"
        )
    else:
        effective_samples = requested

    rng = np.random.default_rng(int(seed))
    positions = np.zeros((effective_samples, len(driver_ids)), dtype=float)
    lap_last = pd.to_numeric(out.get("lap_last"), errors="coerce").fillna(0).astype(int)
    base_laps = lap_last.to_numpy(dtype=int)

    A = np.asarray([[1.0, 1.0], [0.0, float(cfg.phi)]], dtype=float)
    Q = np.asarray([[float(cfg.q_pace) ** 2, 0.0], [0.0, float(cfg.q_deg) ** 2]], dtype=float)

    for sample_idx in range(effective_samples):
        final_laps = base_laps + horizon
        final_times = np.full(len(driver_ids), np.inf, dtype=float)
        for driver_idx, driver_id in enumerate(driver_ids):
            state = states.get(driver_id)
            if state is None:
                continue

            mean = state.mean.astype(float).copy()
            cov = state.cov.astype(float).copy()
            base_lap = int(base_laps[driver_idx])
            total_time = float(race_time_start[driver_idx])

            for step in range(1, horizon + 1):
                mean_pred = A @ mean
                cov_pred = (A @ cov @ A.T) + Q
                cov_pred = 0.5 * (cov_pred + cov_pred.T)
                cov_pred += np.eye(2, dtype=float) * 1e-9

                try:
                    sampled_state = rng.multivariate_normal(mean=mean_pred, cov=cov_pred)
                except Exception:
                    sampled_state = mean_pred

                lap_number = base_lap + step
                lap_baseline = baseline.value_at(lap_number)
                lap_noise = rng.normal(0.0, np.sqrt(max(float(cfg.r_obs), 1e-6)))
                lap_time = float(lap_baseline + sampled_state[0] + lap_noise)
                total_time += max(0.1, lap_time)

                mean = sampled_state
                cov = cov_pred

            final_times[driver_idx] = total_time

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

    return out, {
        "position_dist_enabled": True,
        "position_dist_disabled_reason": None,
        "mc_samples_requested": int(requested),
        "mc_samples_effective": int(effective_samples),
        "mc_samples_reduction_reason": reduction_reason,
        "max_mc_work": int(max_mc_work),
        "sum_p_win": float(np.sum(out["p_win_H"].to_numpy(dtype=float))),
        "invalid_race_time_count": int(invalid_race_time_count),
        "invalid_race_time_penalty_seconds": invalid_race_time_penalty_seconds,
    }


def _predict_snapshot_from_trace(
    trace: pd.DataFrame,
    *,
    lap_cutoff: int,
    horizon_laps: int,
    base_seed: int,
    mc_samples: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if trace.empty:
        return pd.DataFrame(), {
            "position_dist_enabled": False,
            "position_dist_disabled_reason": "empty_trace",
        }

    work = trace.copy()
    work["lap_number"] = pd.to_numeric(work.get("lap_number"), errors="coerce")
    work = work[work["lap_number"].notna()].copy()
    work = work[work["lap_number"] <= int(lap_cutoff)].copy()
    if work.empty:
        return pd.DataFrame(), {
            "position_dist_enabled": False,
            "position_dist_disabled_reason": "no_rows_at_or_before_cutoff",
        }

    event_values = pd.to_numeric(work.get("event_key"), errors="coerce").dropna()
    event_key = int(event_values.iloc[0]) if not event_values.empty else 0
    seed = _event_seed(event_key=event_key, base_seed=int(base_seed)) if event_key > 0 else int(base_seed)

    baseline = build_event_lap_baseline(work, min_clean_obs_per_lap=8)
    states = _build_states_from_trace(work)
    snapshot = _build_snapshot(work)
    cfg = FilterConfig()

    snapshot_with_dist, dist_summary = _mc_position_distribution_with_samples(
        snapshot=snapshot,
        states=states,
        baseline=baseline,
        cfg=cfg,
        horizon_laps=max(1, int(horizon_laps)),
        seed=seed,
        requested_samples=max(50, int(mc_samples)),
    )
    snapshot_final = _finalize_output_mapping(
        snapshot_with_dist,
        dist_summary,
        horizon_laps=max(1, int(horizon_laps)),
    )
    return snapshot_final, dist_summary


def _evaluate_prediction(pred_rows: list[dict[str, Any]], actual_results: pd.DataFrame) -> dict[str, Any]:
    if not pred_rows:
        return {"available": False, "reason": "prediction_rows_unavailable"}
    if actual_results is None or actual_results.empty:
        return {"available": False, "reason": "actual_results_unavailable"}

    pred = pd.DataFrame(pred_rows).copy()
    if pred.empty or "driver_name" not in pred.columns:
        return {"available": False, "reason": "prediction_driver_name_unavailable"}
    pred["driver_key"] = pred["driver_name"].map(_normalize_name_key)
    pred = pred[pred["driver_key"] != ""].copy()
    if pred.empty:
        return {"available": False, "reason": "prediction_driver_key_unavailable"}

    if "rank" in pred.columns:
        pred["pred_rank"] = pd.to_numeric(pred["rank"], errors="coerce")
    else:
        pred["pred_rank"] = pd.Series(range(1, len(pred) + 1), index=pred.index, dtype=float)
    pred = pred[pred["pred_rank"].notna()].copy()
    pred["pred_rank"] = pred["pred_rank"].astype(float)
    pred_unique = pred.sort_values("pred_rank", kind="mergesort").drop_duplicates(subset=["driver_key"], keep="first")

    actual = actual_results.copy()
    if "position" not in actual.columns:
        return {"available": False, "reason": "actual_position_unavailable"}
    if "driver_name" not in actual.columns:
        return {"available": False, "reason": "actual_driver_name_unavailable"}
    actual["driver_key"] = actual["driver_name"].map(_normalize_name_key)
    actual["actual_rank"] = pd.to_numeric(actual["position"], errors="coerce")
    actual = actual[(actual["driver_key"] != "") & actual["actual_rank"].notna()].copy()
    actual_unique = actual.sort_values("actual_rank", kind="mergesort").drop_duplicates(
        subset=["driver_key"],
        keep="first",
    )
    if actual_unique.empty:
        return {"available": False, "reason": "actual_clean_unavailable"}

    merged = pred_unique.merge(actual_unique[["driver_key", "actual_rank"]], on="driver_key", how="inner")
    merged = merged.dropna(subset=["pred_rank", "actual_rank"]).copy()

    mae = float((merged["pred_rank"] - merged["actual_rank"]).abs().mean()) if not merged.empty else None
    rmse = float(np.sqrt(((merged["pred_rank"] - merged["actual_rank"]) ** 2).mean())) if not merged.empty else None
    spearman = None
    if len(merged) >= 2:
        value = merged[["pred_rank", "actual_rank"]].corr(method="spearman").iloc[0, 1]
        if pd.notna(value):
            spearman = float(value)

    predicted_top3 = set(pred_unique.head(3)["driver_key"].tolist())
    predicted_top10 = set(pred_unique.head(10)["driver_key"].tolist())
    actual_top3 = set(actual_unique[actual_unique["actual_rank"] <= 3]["driver_key"].tolist())
    actual_top10 = set(actual_unique[actual_unique["actual_rank"] <= 10]["driver_key"].tolist())

    top3_hit = None
    if actual_top3:
        top3_hit = float(len(predicted_top3.intersection(actual_top3)) / float(min(3, len(actual_top3))))
    top10_hit = None
    if actual_top10:
        top10_hit = float(len(predicted_top10.intersection(actual_top10)) / float(min(10, len(actual_top10))))

    pred_top10 = pred_unique.head(10).merge(actual_unique[["driver_key", "actual_rank"]], on="driver_key", how="inner")
    pred_top10_mae = (
        float((pred_top10["pred_rank"] - pred_top10["actual_rank"]).abs().mean()) if not pred_top10.empty else None
    )

    return {
        "available": True,
        "rows_predicted": int(len(pred_unique)),
        "rows_actual": int(len(actual_unique)),
        "rows_common": int(len(merged)),
        "mae": mae,
        "rmse": rmse,
        "spearman": spearman,
        "top3_hit": top3_hit,
        "top10_hit": top10_hit,
        "pred_top10_mae": pred_top10_mae,
    }


def _b_on_a_top10_mae(a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]], actual_results: pd.DataFrame) -> Optional[float]:
    a = pd.DataFrame(a_rows).copy()
    b = pd.DataFrame(b_rows).copy()
    if a.empty or b.empty or actual_results is None or actual_results.empty:
        return None
    if "driver_name" not in a.columns or "driver_name" not in b.columns:
        return None

    a["driver_key"] = a["driver_name"].map(_normalize_name_key)
    b["driver_key"] = b["driver_name"].map(_normalize_name_key)
    a["rank"] = pd.to_numeric(a.get("rank"), errors="coerce")
    b["rank"] = pd.to_numeric(b.get("rank"), errors="coerce")
    a = a[(a["driver_key"] != "") & a["rank"].notna()].copy()
    b = b[(b["driver_key"] != "") & b["rank"].notna()].copy()
    if a.empty or b.empty:
        return None

    a_top10 = set(a.sort_values("rank", kind="mergesort").head(10)["driver_key"].tolist())
    if not a_top10:
        return None

    actual = actual_results.copy()
    actual["driver_key"] = actual["driver_name"].map(_normalize_name_key)
    actual["actual_rank"] = pd.to_numeric(actual.get("position"), errors="coerce")
    actual = actual[(actual["driver_key"] != "") & actual["actual_rank"].notna()].copy()
    if actual.empty:
        return None

    b_subset = b[b["driver_key"].isin(a_top10)][["driver_key", "rank"]].rename(columns={"rank": "pred_rank"})
    merged = b_subset.merge(actual[["driver_key", "actual_rank"]], on="driver_key", how="inner")
    if merged.empty:
        return None
    return float((merged["pred_rank"] - merged["actual_rank"]).abs().mean())


def _weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str) -> Optional[float]:
    if value_col not in frame.columns or weight_col not in frame.columns:
        return None
    values = pd.to_numeric(frame[value_col], errors="coerce")
    weights = pd.to_numeric(frame[weight_col], errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0.0)
    if not bool(mask.any()):
        return None
    weighted_sum = float((values[mask] * weights[mask]).sum())
    total_weight = float(weights[mask].sum())
    if total_weight <= 0.0:
        return None
    return float(weighted_sum / total_weight)


def _mean(frame: pd.DataFrame, value_col: str) -> Optional[float]:
    if value_col not in frame.columns:
        return None
    values = pd.to_numeric(frame[value_col], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _summarize_cutoff(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "rounds_total": 0,
        }

    first = frame.iloc[0]
    cutoff_pct_requested = first.get("cutoff_pct_requested")
    lap_cutoff = first.get("lap_cutoff")

    summary = {
        "cutoff_mode": str(first.get("cutoff_mode") or ""),
        "cutoff_label": str(first.get("cutoff_label") or ""),
        "cutoff_sort_value": _safe_optional_float(first.get("cutoff_sort_value")),
        "cutoff_pct_requested": int(cutoff_pct_requested) if pd.notna(cutoff_pct_requested) else None,
        "lap_cutoff": int(lap_cutoff) if pd.notna(lap_cutoff) else None,
        "cutoff_pct_realized_mean": _mean(frame, "cutoff_pct_realized"),
        "rounds_total": int(frame["round"].nunique()),
        "A_weighted_mae": _weighted_mean(frame, "A_mae", "A_matched"),
        "B_weighted_mae": _weighted_mean(frame, "B_mae", "B_matched"),
        "A_weighted_rmse": _weighted_mean(frame, "A_rmse", "A_matched"),
        "B_weighted_rmse": _weighted_mean(frame, "B_rmse", "B_matched"),
        "A_weighted_spearman": _weighted_mean(frame, "A_spearman", "A_matched"),
        "B_weighted_spearman": _weighted_mean(frame, "B_spearman", "B_matched"),
        "A_mean_top10_hit": _mean(frame, "A_top10_hit"),
        "B_mean_top10_hit": _mean(frame, "B_top10_hit"),
        "A_mean_top3_hit": _mean(frame, "A_top3_hit"),
        "B_mean_top3_hit": _mean(frame, "B_top3_hit"),
        "A_mean_pred_top10_mae": _mean(frame, "A_pred_top10_mae"),
        "B_mean_pred_top10_mae": _mean(frame, "B_pred_top10_mae"),
        "B_mean_on_A_top10_mae": _mean(frame, "B_on_A_top10_mae"),
        "A_better_rounds_mae": int((frame["A_mae"] < frame["B_mae"]).sum()),
        "B_better_rounds_mae": int((frame["B_mae"] < frame["A_mae"]).sum()),
        "A_better_rounds_top10_hit": int((frame["A_top10_hit"] > frame["B_top10_hit"]).sum()),
        "B_better_rounds_top10_hit": int((frame["B_top10_hit"] > frame["A_top10_hit"]).sum()),
        "rounds_clean": int((frame["chaos_segment"] == "clean").sum()) if "chaos_segment" in frame.columns else None,
        "rounds_chaotic": int((frame["chaos_segment"] == "chaotic").sum()) if "chaos_segment" in frame.columns else None,
    }

    a_w_mae = summary.get("A_weighted_mae")
    b_w_mae = summary.get("B_weighted_mae")
    a_top10 = summary.get("A_mean_top10_hit")
    b_top10 = summary.get("B_mean_top10_hit")
    summary["B_minus_A_weighted_mae"] = (float(b_w_mae) - float(a_w_mae)) if a_w_mae is not None and b_w_mae is not None else None
    summary["A_minus_B_weighted_mae"] = (float(a_w_mae) - float(b_w_mae)) if a_w_mae is not None and b_w_mae is not None else None
    summary["B_minus_A_mean_top10_hit"] = (float(b_top10) - float(a_top10)) if a_top10 is not None and b_top10 is not None else None
    return summary


def _render_overtake_plot(summary_frame: pd.DataFrame, output_path: Path, *, cutoff_mode: str) -> Optional[str]:
    if plt is None or summary_frame.empty:
        return None

    work = summary_frame.copy().sort_values("cutoff_sort_value", kind="mergesort")
    x_values = pd.to_numeric(work["cutoff_sort_value"], errors="coerce")
    mae_gain = pd.to_numeric(work["A_minus_B_weighted_mae"], errors="coerce")
    top10_gain = pd.to_numeric(work["B_minus_A_mean_top10_hit"], errors="coerce")
    labels = work.get("cutoff_label", pd.Series(index=work.index, dtype=object)).astype(str).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

    axes[0].plot(x_values, mae_gain, marker="o", linewidth=2.0, color="#0b7285")
    axes[0].axhline(0.0, color="#6c757d", linewidth=1.0, linestyle="--")
    axes[0].set_title("MAE Gain Vs Horizon A")
    axes[0].set_xlabel("Distance Cutoff (%)" if cutoff_mode == "distance_pct" else "Lap Cutoff")
    axes[0].set_ylabel("A - B Weighted MAE")
    axes[0].grid(alpha=0.25, linewidth=0.7)

    axes[1].plot(x_values, top10_gain, marker="o", linewidth=2.0, color="#2b8a3e")
    axes[1].axhline(0.0, color="#6c757d", linewidth=1.0, linestyle="--")
    axes[1].set_title("Top-10 Hit Gain Vs Horizon A")
    axes[1].set_xlabel("Distance Cutoff (%)" if cutoff_mode == "distance_pct" else "Lap Cutoff")
    axes[1].set_ylabel("B - A Mean Top10 Hit")
    axes[1].grid(alpha=0.25, linewidth=0.7)

    if labels and len(labels) <= 16:
        ticks = x_values.to_numpy(dtype=float)
        for axis in axes:
            axis.set_xticks(ticks)
            axis.set_xticklabels(labels, rotation=35, ha="right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _crossover_metric_specs(epsilon_rank: float, epsilon_score: float) -> list[dict[str, Any]]:
    return [
        {"metric": "mae", "a_col": "A_mae", "b_col": "B_mae", "higher_is_better": False, "epsilon": float(epsilon_rank)},
        {"metric": "rmse", "a_col": "A_rmse", "b_col": "B_rmse", "higher_is_better": False, "epsilon": float(epsilon_rank)},
        {
            "metric": "pred_top10_mae",
            "a_col": "A_pred_top10_mae",
            "b_col": "B_pred_top10_mae",
            "higher_is_better": False,
            "epsilon": float(epsilon_rank),
        },
        {"metric": "spearman", "a_col": "A_spearman", "b_col": "B_spearman", "higher_is_better": True, "epsilon": float(epsilon_score)},
        {"metric": "top3_hit", "a_col": "A_top3_hit", "b_col": "B_top3_hit", "higher_is_better": True, "epsilon": float(epsilon_score)},
        {"metric": "top10_hit", "a_col": "A_top10_hit", "b_col": "B_top10_hit", "higher_is_better": True, "epsilon": float(epsilon_score)},
    ]


def _is_crossover(a_value: float, b_value: float, *, higher_is_better: bool, epsilon: float) -> bool:
    if higher_is_better:
        return bool(b_value >= (a_value + float(epsilon)))
    return bool(b_value <= (a_value - float(epsilon)))


def _build_crossover_per_round(
    per_round_frame: pd.DataFrame,
    *,
    metric_specs: list[dict[str, Any]],
) -> pd.DataFrame:
    if per_round_frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for round_number, round_frame in per_round_frame.groupby("round", sort=True):
        ordered = round_frame.sort_values("cutoff_sort_value", kind="mergesort").reset_index(drop=True)
        if ordered.empty:
            continue
        round_meta = ordered.iloc[0]
        for spec in metric_specs:
            metric = str(spec["metric"])
            a_col = str(spec["a_col"])
            b_col = str(spec["b_col"])
            higher_is_better = bool(spec["higher_is_better"])
            epsilon = float(spec["epsilon"])

            crossover_row: Optional[pd.Series] = None
            a_reference: Optional[float] = None
            b_at_crossover: Optional[float] = None
            for _, row in ordered.iterrows():
                a_value = _safe_optional_float(row.get(a_col))
                b_value = _safe_optional_float(row.get(b_col))
                if a_value is None or b_value is None:
                    continue
                if a_reference is None:
                    a_reference = float(a_value)
                if _is_crossover(a_value, b_value, higher_is_better=higher_is_better, epsilon=epsilon):
                    crossover_row = row
                    b_at_crossover = float(b_value)
                    break

            found = crossover_row is not None
            if found:
                bucket = str(crossover_row.get("cutoff_label") or "")
                cutoff_sort_value = _safe_optional_float(crossover_row.get("cutoff_sort_value"))
                cutoff_lap = _safe_optional_float(crossover_row.get("lap_cutoff"))
                cutoff_pct_realized = _safe_optional_float(crossover_row.get("cutoff_pct_realized"))
            else:
                bucket = NEVER_BEFORE_FINISH
                cutoff_sort_value = None
                cutoff_lap = None
                cutoff_pct_realized = None

            rows.append(
                {
                    "round": int(round_number),
                    "metric": metric,
                    "higher_is_better": bool(higher_is_better),
                    "epsilon": float(epsilon),
                    "crossover_found": bool(found),
                    "crossover_bucket": bucket,
                    "crossover_cutoff_label": None if not found else bucket,
                    "crossover_cutoff_sort_value": cutoff_sort_value,
                    "crossover_lap_cutoff": int(cutoff_lap) if cutoff_lap is not None else None,
                    "crossover_pct_realized": cutoff_pct_realized,
                    "a_reference": a_reference,
                    "b_at_crossover": b_at_crossover,
                    "cutoff_mode": str(round_meta.get("cutoff_mode") or ""),
                    "total_laps_completed": _safe_optional_float(round_meta.get("total_laps_completed")),
                    "sc_vsc_laps": _safe_optional_float(round_meta.get("sc_vsc_laps")),
                    "chaos_fraction": _safe_optional_float(round_meta.get("chaos_fraction")),
                    "chaos_segment": str(round_meta.get("chaos_segment") or "unknown"),
                    "chaos_has_incident": bool(round_meta.get("chaos_has_incident", False)),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["metric", "chaos_segment", "round"], kind="mergesort").reset_index(drop=True)


def _crossover_distribution(crossover_frame: pd.DataFrame) -> pd.DataFrame:
    if crossover_frame.empty:
        return pd.DataFrame()

    expanded = pd.concat(
        [
            crossover_frame.assign(chaos_segment="all"),
            crossover_frame.copy(),
        ],
        ignore_index=True,
    )
    grouped = (
        expanded.groupby(["metric", "chaos_segment", "crossover_bucket"], dropna=False, as_index=False)
        .size()
        .rename(columns={"size": "rounds"})
    )
    totals = (
        expanded.groupby(["metric", "chaos_segment"], dropna=False, as_index=False)
        .size()
        .rename(columns={"size": "rounds_total"})
    )
    out = grouped.merge(totals, on=["metric", "chaos_segment"], how="left")
    out["share"] = out["rounds"] / out["rounds_total"].replace(0, np.nan)

    def _bucket_order(value: object) -> float:
        label = str(value or "").strip().lower()
        if label == NEVER_BEFORE_FINISH.lower():
            return 1_000_000.0
        pct_match = re.match(r"^(\d+(?:\.\d+)?)%$", label)
        if pct_match:
            return float(pct_match.group(1))
        lap_match = re.match(r"^lap\s+(\d+(?:\.\d+)?)$", label)
        if lap_match:
            return float(lap_match.group(1))
        return 999_999.0

    out["bucket_order"] = out["crossover_bucket"].map(_bucket_order)
    out = out.sort_values(["metric", "chaos_segment", "bucket_order"], kind="mergesort").reset_index(drop=True)
    return out.drop(columns=["bucket_order"])


def _crossover_survival_curve(crossover_frame: pd.DataFrame, per_round_frame: pd.DataFrame) -> pd.DataFrame:
    if crossover_frame.empty or per_round_frame.empty:
        return pd.DataFrame()

    cutoff_defs = (
        per_round_frame[["cutoff_sort_value", "cutoff_label"]]
        .dropna(subset=["cutoff_sort_value"])
        .drop_duplicates()
        .sort_values("cutoff_sort_value", kind="mergesort")
        .reset_index(drop=True)
    )
    if cutoff_defs.empty:
        return pd.DataFrame()

    expanded = pd.concat(
        [
            crossover_frame.assign(chaos_segment="all"),
            crossover_frame.copy(),
        ],
        ignore_index=True,
    )

    rows: list[dict[str, Any]] = []
    for (metric, segment), group in expanded.groupby(["metric", "chaos_segment"], sort=True):
        total_rounds = int(len(group))
        crossed_values = pd.to_numeric(
            group.loc[group["crossover_found"], "crossover_cutoff_sort_value"],
            errors="coerce",
        ).dropna()
        for _, cutoff in cutoff_defs.iterrows():
            cutoff_value = float(cutoff["cutoff_sort_value"])
            crossed_rounds = int((crossed_values <= cutoff_value).sum())
            rows.append(
                {
                    "metric": str(metric),
                    "chaos_segment": str(segment),
                    "cutoff_label": str(cutoff["cutoff_label"]),
                    "cutoff_sort_value": float(cutoff_value),
                    "crossed_rounds": int(crossed_rounds),
                    "rounds_total": int(total_rounds),
                    "crossed_share": float(crossed_rounds / total_rounds) if total_rounds > 0 else None,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["metric", "chaos_segment", "cutoff_sort_value"],
        kind="mergesort",
    ).reset_index(drop=True)


def _crossover_overview(crossover_frame: pd.DataFrame) -> pd.DataFrame:
    if crossover_frame.empty:
        return pd.DataFrame()

    expanded = pd.concat(
        [
            crossover_frame.assign(chaos_segment="all"),
            crossover_frame.copy(),
        ],
        ignore_index=True,
    )
    rows: list[dict[str, Any]] = []
    for (metric, segment), group in expanded.groupby(["metric", "chaos_segment"], sort=True):
        total_rounds = int(len(group))
        crossed = group[group["crossover_found"]].copy()
        cutoff_median = pd.to_numeric(crossed.get("crossover_cutoff_sort_value"), errors="coerce").dropna()
        lap_median = pd.to_numeric(crossed.get("crossover_lap_cutoff"), errors="coerce").dropna()
        rows.append(
            {
                "metric": str(metric),
                "chaos_segment": str(segment),
                "rounds_total": int(total_rounds),
                "crossed_rounds": int(len(crossed)),
                "crossed_share": float(len(crossed) / total_rounds) if total_rounds > 0 else None,
                "median_crossover_sort_value": float(cutoff_median.median()) if not cutoff_median.empty else None,
                "median_crossover_lap": float(lap_median.median()) if not lap_median.empty else None,
            }
        )
    return pd.DataFrame(rows).sort_values(["metric", "chaos_segment"], kind="mergesort").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Horizon A vs Horizon B with lap or distance-percentage cutoffs.",
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--horizon-a-dir",
        default="data/f1/compare_2025_afterfix/horizon_a",
        help="Folder containing Horizon A artifacts (rXX.json).",
    )
    parser.add_argument(
        "--horizon-b-dir",
        default="data/f1/compare_2025_afterfix/horizon_b",
        help="Folder containing Horizon B artifacts (rXX.json) with trace_path.",
    )
    parser.add_argument(
        "--weekends-dir",
        default="data/f1/raw/weekends",
        help="Local weekends root for actual race results.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/f1/compare_2025_afterfix_lap_snapshots",
        help="Output directory for CSV/JSON/plot artifacts.",
    )
    parser.add_argument(
        "--rounds",
        default="all",
        help="Comma-separated rounds or 'all' (default).",
    )
    parser.add_argument(
        "--cutoff-mode",
        choices=["distance_pct", "lap"],
        default="distance_pct",
        help="Cutoff mode: compare by percent distance completed or by absolute laps.",
    )
    parser.add_argument(
        "--distance-cutoffs",
        default="5,10,20,30,40,50,60,70,80,90,100",
        help="Comma-separated distance-percentage cutoffs (used when --cutoff-mode=distance_pct).",
    )
    parser.add_argument(
        "--lap-cutoffs",
        default="5,10,20,30",
        help="Comma-separated lap cutoffs (used when --cutoff-mode=lap).",
    )
    parser.add_argument(
        "--horizon-laps",
        type=int,
        default=10,
        help="Future horizon laps for B position distribution.",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=250,
        help="Requested Monte Carlo samples per snapshot (default: 250).",
    )
    parser.add_argument(
        "--clean-max-chaos-fraction",
        type=float,
        default=0.02,
        help="Max chaos_fraction for classifying a round as clean.",
    )
    parser.add_argument(
        "--chaotic-min-chaos-fraction",
        type=float,
        default=0.05,
        help="Min chaos_fraction for classifying a round as chaotic.",
    )
    parser.add_argument(
        "--epsilon-rank",
        type=float,
        default=0.10,
        help="Crossover margin epsilon for lower-is-better metrics (MAE/RMSE/rank errors).",
    )
    parser.add_argument(
        "--epsilon-score",
        type=float,
        default=0.02,
        help="Crossover margin epsilon for higher-is-better metrics (Spearman/TopK hit).",
    )
    parser.add_argument("--f1-live-seed", type=int, default=42)
    parser.add_argument("--output-format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    project_root = _project_root()
    horizon_a_dir = _resolve_path(project_root, args.horizon_a_dir)
    horizon_b_dir = _resolve_path(project_root, args.horizon_b_dir)
    weekends_dir = _resolve_path(project_root, args.weekends_dir)
    output_dir = _resolve_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not horizon_a_dir.exists():
        raise SystemExit(f"Horizon A directory not found: {horizon_a_dir}")
    if not horizon_b_dir.exists():
        raise SystemExit(f"Horizon B directory not found: {horizon_b_dir}")
    if not weekends_dir.exists():
        raise SystemExit(f"Weekends directory not found: {weekends_dir}")

    if float(args.clean_max_chaos_fraction) > float(args.chaotic_min_chaos_fraction):
        raise SystemExit("--clean-max-chaos-fraction must be <= --chaotic-min-chaos-fraction.")

    pct_cutoffs = _parse_pct_cutoffs(args.distance_cutoffs)
    lap_cutoffs = _parse_lap_cutoffs(args.lap_cutoffs)
    round_pattern = re.compile(r"^r(\d+)\.json$", re.IGNORECASE)
    available_rounds = []
    for path in sorted(horizon_a_dir.glob("r*.json")):
        match = round_pattern.match(path.name)
        if not match:
            continue
        available_rounds.append(int(match.group(1)))
    rounds = _parse_rounds(args.rounds, available_rounds)
    if not rounds:
        raise SystemExit("No rounds selected after filtering.")

    provider = LocalWeekendProvider(weekends_dir=str(weekends_dir))

    per_round_rows: list[dict[str, Any]] = []
    issues: list[str] = []

    for round_number in rounds:
        a_path = horizon_a_dir / f"r{int(round_number):02d}.json"
        b_path = horizon_b_dir / f"r{int(round_number):02d}.json"

        a_payload = _load_json(a_path)
        if not a_payload:
            issues.append(f"Round {round_number:02d}: missing/invalid A artifact ({a_path}).")
            continue
        b_payload = _load_json(b_path)
        if not b_payload:
            issues.append(f"Round {round_number:02d}: missing/invalid B artifact ({b_path}).")
            continue

        a_rows = a_payload.get("rows")
        if not isinstance(a_rows, list) or not a_rows:
            issues.append(f"Round {round_number:02d}: A rows unavailable.")
            continue

        actual_results = provider.get_race_results(args.year, int(round_number))
        if actual_results.empty:
            issues.append(f"Round {round_number:02d}: race results unavailable.")
            continue

        a_eval = _evaluate_prediction(a_rows, actual_results)
        if not a_eval.get("available"):
            issues.append(f"Round {round_number:02d}: A evaluation unavailable ({a_eval.get('reason')}).")
            continue

        trace_path = _trace_path_from_payload(b_payload, project_root=project_root, payload_path=b_path)
        if trace_path is None:
            trace_path = _fallback_trace_path(project_root=project_root, year=args.year, round_number=int(round_number))
        if trace_path is None or not trace_path.exists():
            issues.append(f"Round {round_number:02d}: trace path unavailable.")
            continue

        try:
            trace = _read_trace(trace_path)
        except Exception as exc:
            issues.append(f"Round {round_number:02d}: trace read failed ({trace_path}): {exc}")
            continue

        total_laps_completed = _total_laps_completed(trace)
        if total_laps_completed is None:
            issues.append(f"Round {round_number:02d}: unable to infer total_laps_completed from trace.")
            continue

        chaos_profile = _build_chaos_profile(
            trace,
            total_laps_completed=total_laps_completed,
            clean_max_chaos_fraction=float(args.clean_max_chaos_fraction),
            chaotic_min_chaos_fraction=float(args.chaotic_min_chaos_fraction),
        )
        cutoff_plan = _build_cutoff_plan(
            cutoff_mode=args.cutoff_mode,
            total_laps_completed=total_laps_completed,
            pct_cutoffs=pct_cutoffs,
            lap_cutoffs=lap_cutoffs,
        )
        for cutoff in cutoff_plan:
            lap_cutoff = int(cutoff["lap_cutoff"])
            cutoff_label = str(cutoff["cutoff_label"])
            snapshot, dist_summary = _predict_snapshot_from_trace(
                trace,
                lap_cutoff=lap_cutoff,
                horizon_laps=args.horizon_laps,
                base_seed=args.f1_live_seed,
                mc_samples=args.mc_samples,
            )
            if snapshot.empty:
                issues.append(f"Round {round_number:02d} {cutoff_label}: snapshot unavailable.")
                continue

            b_rows = json.loads(snapshot.to_json(orient="records"))
            b_eval = _evaluate_prediction(b_rows, actual_results)
            if not b_eval.get("available"):
                issues.append(
                    f"Round {round_number:02d} {cutoff_label}: B evaluation unavailable ({b_eval.get('reason')}).",
                )
                continue

            row = {
                "round": int(round_number),
                "cutoff_mode": str(cutoff["cutoff_mode"]),
                "cutoff_label": str(cutoff["cutoff_label"]),
                "cutoff_sort_value": float(cutoff["cutoff_sort_value"]),
                "cutoff_pct_requested": cutoff["cutoff_pct_requested"],
                "lap_cutoff": int(cutoff["lap_cutoff"]),
                "cutoff_pct_realized": float(cutoff["cutoff_pct_realized"]),
                "total_laps_completed": int(chaos_profile["total_laps_completed"]),
                "sc_vsc_laps": int(chaos_profile["sc_vsc_laps"]),
                "chaos_fraction": float(chaos_profile["chaos_fraction"]),
                "chaos_segment": str(chaos_profile["chaos_segment"]),
                "chaos_has_incident": bool(chaos_profile["has_sc_vsc_or_red"]),
                "A_rows": int(a_eval["rows_predicted"]),
                "B_rows": int(b_eval["rows_predicted"]),
                "A_matched": int(a_eval["rows_common"]),
                "B_matched": int(b_eval["rows_common"]),
                "A_mae": a_eval["mae"],
                "B_mae": b_eval["mae"],
                "A_rmse": a_eval["rmse"],
                "B_rmse": b_eval["rmse"],
                "A_spearman": a_eval["spearman"],
                "B_spearman": b_eval["spearman"],
                "A_top3_hit": a_eval["top3_hit"],
                "B_top3_hit": b_eval["top3_hit"],
                "A_top10_hit": a_eval["top10_hit"],
                "B_top10_hit": b_eval["top10_hit"],
                "A_pred_top10_mae": a_eval["pred_top10_mae"],
                "B_pred_top10_mae": b_eval["pred_top10_mae"],
                "B_on_A_top10_mae": _b_on_a_top10_mae(a_rows, b_rows, actual_results),
                "B_position_dist_enabled": bool(dist_summary.get("position_dist_enabled", False)),
                "B_invalid_race_time_count": dist_summary.get("invalid_race_time_count"),
                "trace_path": str(trace_path),
            }
            per_round_rows.append(row)

    per_round_frame = pd.DataFrame(per_round_rows)
    if per_round_frame.empty:
        raise SystemExit("No evaluable round/lap rows produced.")

    per_round_frame = per_round_frame.sort_values(["cutoff_sort_value", "round"], kind="mergesort").reset_index(drop=True)

    summaries: list[dict[str, Any]] = []
    for cutoff_value in sorted(per_round_frame["cutoff_sort_value"].dropna().astype(float).unique().tolist()):
        cutoff_frame = per_round_frame[per_round_frame["cutoff_sort_value"] == float(cutoff_value)].copy()
        summaries.append(_summarize_cutoff(cutoff_frame))
    summary_frame = pd.DataFrame(summaries).sort_values("cutoff_sort_value", kind="mergesort").reset_index(drop=True)

    metric_specs = _crossover_metric_specs(
        epsilon_rank=float(args.epsilon_rank),
        epsilon_score=float(args.epsilon_score),
    )
    crossover_per_round_frame = _build_crossover_per_round(
        per_round_frame,
        metric_specs=metric_specs,
    )
    crossover_distribution_frame = _crossover_distribution(crossover_per_round_frame)
    crossover_survival_frame = _crossover_survival_curve(crossover_per_round_frame, per_round_frame)
    crossover_overview_frame = _crossover_overview(crossover_per_round_frame)

    per_round_path = output_dir / "horizon_a_vs_b_lap_snapshots_per_round.csv"
    summary_csv_path = output_dir / "horizon_a_vs_b_lap_snapshots_summary.csv"
    summary_json_path = output_dir / "horizon_a_vs_b_lap_snapshots_summary.json"
    curve_csv_path = output_dir / "horizon_a_vs_b_lap_overtake_curve.csv"
    crossover_per_round_path = output_dir / "horizon_a_vs_b_crossover_per_round.csv"
    crossover_distribution_path = output_dir / "horizon_a_vs_b_crossover_distribution.csv"
    crossover_survival_path = output_dir / "horizon_a_vs_b_crossover_survival.csv"
    crossover_overview_path = output_dir / "horizon_a_vs_b_crossover_overview.csv"
    plot_path = output_dir / "horizon_a_vs_b_lap_overtake.png"

    per_round_frame.to_csv(per_round_path, index=False)
    summary_frame.to_csv(summary_csv_path, index=False)
    crossover_per_round_frame.to_csv(crossover_per_round_path, index=False)
    crossover_distribution_frame.to_csv(crossover_distribution_path, index=False)
    crossover_survival_frame.to_csv(crossover_survival_path, index=False)
    crossover_overview_frame.to_csv(crossover_overview_path, index=False)

    curve_frame = summary_frame[
        [
            "cutoff_mode",
            "cutoff_label",
            "cutoff_sort_value",
            "cutoff_pct_requested",
            "lap_cutoff",
            "cutoff_pct_realized_mean",
            "A_weighted_mae",
            "B_weighted_mae",
            "A_minus_B_weighted_mae",
            "A_mean_top10_hit",
            "B_mean_top10_hit",
            "B_minus_A_mean_top10_hit",
        ]
    ].copy()
    curve_frame.to_csv(curve_csv_path, index=False)

    plot_written = _render_overtake_plot(summary_frame, plot_path, cutoff_mode=args.cutoff_mode)

    round_chaos = (
        per_round_frame.sort_values(["round", "cutoff_sort_value"], kind="mergesort")
        .drop_duplicates(subset=["round"], keep="first")
        .copy()
    )
    chaos_segment_counts = {
        str(label): int(count)
        for label, count in round_chaos["chaos_segment"].value_counts(dropna=False).to_dict().items()
    }

    summary_payload = {
        "year": int(args.year),
        "cutoff_mode": str(args.cutoff_mode),
        "distance_cutoffs": [int(value) for value in pct_cutoffs],
        "lap_cutoffs": [int(value) for value in lap_cutoffs],
        "horizon_laps": int(args.horizon_laps),
        "mc_samples": int(args.mc_samples),
        "chaos_thresholds": {
            "clean_max_fraction": float(args.clean_max_chaos_fraction),
            "chaotic_min_fraction": float(args.chaotic_min_chaos_fraction),
        },
        "crossover_epsilons": {
            "rank_error": float(args.epsilon_rank),
            "score": float(args.epsilon_score),
        },
        "rounds_requested": [int(value) for value in rounds],
        "rounds_with_output": sorted({int(value) for value in per_round_frame["round"].unique().tolist()}),
        "rows_total": int(len(per_round_frame)),
        "rounds_chaos": json.loads(
            round_chaos[
                [
                    "round",
                    "total_laps_completed",
                    "sc_vsc_laps",
                    "chaos_fraction",
                    "chaos_segment",
                    "chaos_has_incident",
                ]
            ].to_json(orient="records")
        ),
        "chaos_segment_counts": chaos_segment_counts,
        "by_cutoff": json.loads(summary_frame.to_json(orient="records")),
        "by_lap_cutoff": json.loads(summary_frame.to_json(orient="records")),
        "crossover_summary": json.loads(crossover_overview_frame.to_json(orient="records")),
        "artifacts": {
            "per_round_csv": str(per_round_path),
            "summary_csv": str(summary_csv_path),
            "summary_json": str(summary_json_path),
            "curve_csv": str(curve_csv_path),
            "crossover_per_round_csv": str(crossover_per_round_path),
            "crossover_distribution_csv": str(crossover_distribution_path),
            "crossover_survival_csv": str(crossover_survival_path),
            "crossover_overview_csv": str(crossover_overview_path),
            "overtake_plot": plot_written,
        },
        "issues": issues,
        "generated_at": _utc_now(),
    }
    summary_json_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_format == "json":
        print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
        return

    print("=" * 72)
    print("Horizon A vs B lap snapshot comparison")
    print("=" * 72)
    print(f"Year: {args.year}")
    print(f"Rounds evaluated: {len(summary_payload['rounds_with_output'])}/{len(rounds)}")
    print(f"Cutoff mode: {args.cutoff_mode}")
    if args.cutoff_mode == "distance_pct":
        print(f"Distance cutoffs (%): {', '.join(str(value) for value in pct_cutoffs)}")
    else:
        print(f"Lap cutoffs: {', '.join(str(value) for value in lap_cutoffs)}")
    print(
        "Chaos thresholds: "
        f"clean<= {float(args.clean_max_chaos_fraction):.3f}, "
        f"chaotic>= {float(args.chaotic_min_chaos_fraction):.3f}"
    )
    print(f"Chaos segments: {chaos_segment_counts}")
    print(f"Per-round CSV: {per_round_path}")
    print(f"Summary CSV:   {summary_csv_path}")
    print(f"Summary JSON:  {summary_json_path}")
    print(f"Curve CSV:     {curve_csv_path}")
    print(f"Crossover per-round CSV:  {crossover_per_round_path}")
    print(f"Crossover distribution:   {crossover_distribution_path}")
    print(f"Crossover survival:       {crossover_survival_path}")
    print(f"Crossover overview:       {crossover_overview_path}")
    print(f"Overtake plot: {plot_written or 'not generated (matplotlib unavailable)'}")

    mae_all = crossover_overview_frame[
        (crossover_overview_frame["metric"] == "mae") & (crossover_overview_frame["chaos_segment"] == "all")
    ]
    if not mae_all.empty:
        row = mae_all.iloc[0]
        crossed_rounds = int(row.get("crossed_rounds", 0))
        rounds_total = int(row.get("rounds_total", 0))
        median_cutoff = _safe_optional_float(row.get("median_crossover_sort_value"))
        if median_cutoff is not None:
            if args.cutoff_mode == "distance_pct":
                print(f"MAE crossover median: {median_cutoff:.1f}% ({crossed_rounds}/{rounds_total} rounds crossed)")
            else:
                print(f"MAE crossover median: lap {median_cutoff:.1f} ({crossed_rounds}/{rounds_total} rounds crossed)")
        else:
            print(f"MAE crossover median: unavailable ({crossed_rounds}/{rounds_total} rounds crossed)")

    if issues:
        print(f"Issues: {len(issues)}")
        for issue in issues[:10]:
            print(f" - {issue}")
        if len(issues) > 10:
            print(f" - ... ({len(issues) - 10} more)")
    else:
        print("Issues: none")


if __name__ == "__main__":
    main()
