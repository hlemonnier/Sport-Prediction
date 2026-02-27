#!/usr/bin/env python3
"""Evaluate Horizon A vs Horizon B on early live-race lap snapshots."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
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
    _mc_position_distribution,
    _sample_pit_loss_seconds,
    _strategy_template_probabilities,
)
from rqp.live_state_space import FilterConfig, FilterState, build_event_lap_baseline, parse_track_status
from rqp.providers import LocalWeekendProvider


NEVER_BEFORE_FINISH = "Never before finish"
Z_SCORE_50 = 0.6744897501960817
Z_SCORE_90 = 1.6448536269514722


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
    return _mc_position_distribution(
        snapshot=snapshot,
        states=states,
        baseline=baseline,
        cfg=cfg,
        horizon_laps=horizon_laps,
        seed=seed,
        requested_samples=requested_samples,
        max_mc_work=250000,
        emit_observability=True,
        observability_top_drivers=10,
        observability_max_position=20,
    )


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


def _safe_bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        out = numeric.fillna(1.0 if default else 0.0) > 0.0
        return out.astype(bool)
    text = series.fillna(str(default)).astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y"})


def _safe_numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(index=frame.index, data=np.nan, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _lap_regime_series(frame: pd.DataFrame) -> pd.Series:
    is_red = _safe_bool_series(frame, "is_red", default=False)
    is_sc_vsc = _safe_bool_series(frame, "is_sc_vsc", default=False)
    is_yellow = _safe_bool_series(frame, "is_yellow", default=False)
    regimes = np.where(
        is_red.to_numpy(),
        "red",
        np.where(is_sc_vsc.to_numpy(), "sc_vsc", np.where(is_yellow.to_numpy(), "yellow", "green")),
    )
    return pd.Series(regimes, index=frame.index, dtype=object)


def _select_focus_driver_ids(trace: pd.DataFrame, limit: int = 3) -> list[str]:
    if trace.empty:
        return []
    frame = trace.copy()
    frame["lap_number"] = _safe_numeric_series(frame, "lap_number")
    frame["timestamp"] = _safe_numeric_series(frame, "timestamp")
    frame["race_time_seconds"] = _safe_numeric_series(frame, "race_time_seconds")
    tails = (
        frame.sort_values(["driver_id", "lap_number", "timestamp"], kind="mergesort")
        .groupby("driver_id", sort=False)
        .tail(1)
        .copy()
    )
    if tails.empty:
        return []
    tails["lap_number"] = pd.to_numeric(tails["lap_number"], errors="coerce").fillna(-1.0)
    tails["race_time_seconds"] = pd.to_numeric(tails["race_time_seconds"], errors="coerce")
    tails["race_time_order"] = tails["race_time_seconds"].fillna(tails["race_time_seconds"].max(skipna=True) + 9999.0)
    ordered = tails.sort_values(["lap_number", "race_time_order"], ascending=[False, True], kind="mergesort")
    ids = [str(value) for value in ordered["driver_id"].tolist()[: max(1, int(limit))]]
    return ids


def _gaussian_crps(errors: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma_clipped = np.clip(sigma.astype(float), 1e-6, None)
    z = errors.astype(float) / sigma_clipped
    phi = np.exp(-0.5 * (z**2)) / np.sqrt(2.0 * np.pi)
    erf_values = np.vectorize(math.erf, otypes=[float])(z / np.sqrt(2.0))
    Phi = 0.5 * (1.0 + erf_values)
    return sigma_clipped * (z * ((2.0 * Phi) - 1.0) + (2.0 * phi) - (1.0 / np.sqrt(np.pi)))


def _build_round_observability(
    *,
    round_number: int,
    trace: pd.DataFrame,
    horizon_laps: int,
    pit_window_laps: int,
) -> dict[str, pd.DataFrame]:
    if trace.empty:
        return {}

    frame = trace.copy()
    frame["round"] = int(round_number)
    frame["lap_number"] = _safe_numeric_series(frame, "lap_number")
    frame["timestamp"] = _safe_numeric_series(frame, "timestamp")
    frame["baseline_lap"] = _safe_numeric_series(frame, "baseline_lap")
    frame["lap_time_seconds"] = _safe_numeric_series(frame, "lap_time_seconds")
    frame["one_step_pred_mean"] = _safe_numeric_series(frame, "one_step_pred_mean")
    frame["one_step_pred_std"] = _safe_numeric_series(frame, "one_step_pred_std")
    frame["innovation"] = _safe_numeric_series(frame, "innovation")
    frame["innovation_var"] = _safe_numeric_series(frame, "innovation_var")
    frame["race_time_seconds"] = _safe_numeric_series(frame, "race_time_seconds")
    frame["tyre_age"] = _safe_numeric_series(frame, "tyre_age")
    frame["deg_rate_mean"] = _safe_numeric_series(frame, "deg_rate_mean")
    frame["pace_penalty_mean"] = _safe_numeric_series(frame, "pace_penalty_mean")
    frame["pace_penalty_std"] = _safe_numeric_series(frame, "pace_penalty_std")
    frame["deg_rate_std"] = _safe_numeric_series(frame, "deg_rate_std")

    frame["eval_included"] = _safe_bool_series(frame, "eval_included")
    frame["is_box_lap"] = _safe_bool_series(frame, "is_box_lap")
    frame["is_accurate"] = _safe_bool_series(frame, "is_accurate")
    frame["gate_skip_update"] = _safe_bool_series(frame, "gate_skip_update")
    frame["robust_applied"] = _safe_bool_series(frame, "robust_applied")
    frame["reset_applied"] = _safe_bool_series(frame, "reset_applied")
    frame["lap_regime"] = _lap_regime_series(frame)

    frame["driver_id"] = frame.get("driver_id", pd.Series(index=frame.index, dtype=object)).astype(str)
    frame["driver_name"] = frame.get("driver_name", frame["driver_id"]).astype(str)
    frame = frame[frame["driver_id"].str.strip() != ""].copy()
    if frame.empty:
        return {}

    # A1) Lap availability per driver.
    lap_availability = (
        frame.groupby(["round", "driver_id", "driver_name"], sort=True)
        .agg(
            laps_observed=("lap_number", "size"),
            laps_assimilated=("eval_included", "sum"),
            laps_box=("is_box_lap", "sum"),
            laps_inaccurate=("is_accurate", lambda values: int((~values).sum())),
            laps_sc_vsc_skip=("gate_skip_update", "sum"),
            pit_resets=("reset_applied", "sum"),
        )
        .reset_index()
    )
    lap_availability["laps_skipped"] = lap_availability["laps_observed"] - lap_availability["laps_assimilated"]

    # A2) Track-status composition by round and by driver.
    driver_track = (
        frame.groupby(["round", "driver_id", "driver_name", "lap_regime"], dropna=False)
        .size()
        .rename("laps")
        .reset_index()
    )
    driver_total = (
        driver_track.groupby(["round", "driver_id", "driver_name"], as_index=False)["laps"]
        .sum()
        .rename(columns={"laps": "laps_total"})
    )
    driver_track = driver_track.merge(driver_total, on=["round", "driver_id", "driver_name"], how="left")
    driver_track["share"] = driver_track["laps"] / driver_track["laps_total"].replace(0, np.nan)
    driver_track["scope"] = "driver"

    lap_scope = frame[frame["lap_number"].notna()].copy()
    lap_scope["lap_number"] = lap_scope["lap_number"].astype(int)
    lap_regime_round = (
        lap_scope.groupby(["round", "lap_number"], sort=True)["lap_regime"]
        .apply(
            lambda values: (
                "red"
                if "red" in set(values)
                else "sc_vsc"
                if "sc_vsc" in set(values)
                else "yellow"
                if "yellow" in set(values)
                else "green"
            )
        )
        .rename("lap_regime")
        .reset_index()
    )
    round_track = (
        lap_regime_round.groupby(["round", "lap_regime"], dropna=False)
        .size()
        .rename("laps")
        .reset_index()
    )
    round_total = round_track.groupby(["round"], as_index=False)["laps"].sum().rename(columns={"laps": "laps_total"})
    round_track = round_track.merge(round_total, on=["round"], how="left")
    round_track["share"] = round_track["laps"] / round_track["laps_total"].replace(0, np.nan)
    round_track["driver_id"] = "ALL"
    round_track["driver_name"] = "ALL"
    round_track["scope"] = "round"
    track_status_composition = pd.concat(
        [
            round_track[["round", "scope", "driver_id", "driver_name", "lap_regime", "laps", "laps_total", "share"]],
            driver_track[["round", "scope", "driver_id", "driver_name", "lap_regime", "laps", "laps_total", "share"]],
        ],
        ignore_index=True,
    )

    # A3) Baseline quality and residual spread.
    baseline_curve = (
        frame[frame["lap_number"].notna() & frame["baseline_lap"].notna()]
        .assign(lap_number=lambda df: df["lap_number"].astype(int))
        .groupby(["round", "lap_number"], as_index=False)
        .agg(
            event_lap_baseline_t=("baseline_lap", "median"),
            baseline_obs=("baseline_lap", "count"),
        )
        .sort_values(["round", "lap_number"], kind="mergesort")
    )
    residual_source = frame[frame["lap_time_seconds"].notna() & frame["baseline_lap"].notna() & frame["lap_number"].notna()].copy()
    residual_source["lap_number"] = residual_source["lap_number"].astype(int)
    residual_source["lap_minus_baseline"] = residual_source["lap_time_seconds"] - residual_source["baseline_lap"]
    baseline_residual_distribution = (
        residual_source.groupby(["round", "lap_number"], as_index=False)
        .agg(
            residual_mean=("lap_minus_baseline", "mean"),
            residual_median=("lap_minus_baseline", "median"),
            residual_std=("lap_minus_baseline", "std"),
            residual_p10=("lap_minus_baseline", lambda values: float(np.nanpercentile(values, 10.0))),
            residual_p90=("lap_minus_baseline", lambda values: float(np.nanpercentile(values, 90.0))),
            residual_obs=("lap_minus_baseline", "count"),
        )
        .sort_values(["round", "lap_number"], kind="mergesort")
    )

    # A4) Race-time sanity summary + focused curves.
    race_time_summary_rows: list[dict[str, Any]] = []
    race_time_curve_rows: list[dict[str, Any]] = []
    focus_ids = set(_select_focus_driver_ids(frame, limit=3))
    for (driver_id, driver_name), driver_frame in frame.groupby(["driver_id", "driver_name"], sort=False):
        ordered = driver_frame.sort_values(["lap_number", "timestamp"], kind="mergesort").copy()
        race_times = pd.to_numeric(ordered["race_time_seconds"], errors="coerce")
        deltas = race_times.diff()
        non_monotone = int((deltas < -1e-6).sum())
        nonpositive = int((race_times <= 0.0).sum())
        race_time_summary_rows.append(
            {
                "round": int(round_number),
                "driver_id": str(driver_id),
                "driver_name": str(driver_name),
                "laps_observed": int(len(ordered)),
                "race_time_non_monotone_steps": int(non_monotone),
                "race_time_nonpositive_count": int(nonpositive),
                "race_time_monotone": bool(non_monotone == 0),
                "focus_driver": bool(driver_id in focus_ids),
            }
        )
        if driver_id in focus_ids:
            for _, row in ordered.iterrows():
                race_time_curve_rows.append(
                    {
                        "round": int(round_number),
                        "driver_id": str(driver_id),
                        "driver_name": str(driver_name),
                        "lap_number": _safe_optional_float(row.get("lap_number")),
                        "race_time_seconds": _safe_optional_float(row.get("race_time_seconds")),
                    }
                )
    race_time_summary = pd.DataFrame(race_time_summary_rows)
    race_time_curve = pd.DataFrame(race_time_curve_rows)

    # B5) Posterior state trajectories on focus drivers.
    state_trajectories = frame[frame["driver_id"].isin(focus_ids)][
        [
            "round",
            "driver_id",
            "driver_name",
            "lap_number",
            "pace_penalty_mean",
            "pace_penalty_std",
            "deg_rate_mean",
            "deg_rate_std",
            "reset_applied",
            "stint_id",
            "tyre_age",
        ]
    ].copy()
    state_trajectories = state_trajectories.sort_values(["driver_id", "lap_number"], kind="mergesort")

    # B6) Innovation diagnostics.
    innovation_rows = frame[
        [
            "round",
            "driver_id",
            "driver_name",
            "lap_number",
            "innovation",
            "innovation_var",
            "eval_included",
            "lap_regime",
        ]
    ].copy()
    innovation_rows["innovation_std"] = np.sqrt(np.clip(pd.to_numeric(innovation_rows["innovation_var"], errors="coerce"), 1e-9, None))
    innovation_rows["z_innovation"] = innovation_rows["innovation"] / innovation_rows["innovation_std"]

    green_innov = innovation_rows[
        (innovation_rows["lap_regime"] == "green")
        & innovation_rows["z_innovation"].notna()
        & np.isfinite(innovation_rows["z_innovation"])
    ]["z_innovation"].to_numpy(dtype=float)
    bins = np.linspace(-5.0, 5.0, 41)
    hist_counts, hist_edges = np.histogram(green_innov, bins=bins)
    innovation_hist = pd.DataFrame(
        {
            "round": int(round_number),
            "bin_left": hist_edges[:-1],
            "bin_right": hist_edges[1:],
            "count": hist_counts.astype(int),
        }
    )

    # B7) Update behavior mix.
    behaviors = np.where(
        frame["eval_included"] & (~frame["robust_applied"]) & (frame.get("gate_mode", pd.Series(index=frame.index)).fillna("") != "yellow_inflate"),
        "assimilated_normal",
        np.where(
            frame["eval_included"] & frame["robust_applied"],
            "clipped_robust",
            np.where(
                frame.get("gate_mode", pd.Series(index=frame.index)).fillna("") == "skip_sc_vsc_red",
                "skipped_sc_vsc",
                np.where(
                    frame.get("gate_mode", pd.Series(index=frame.index)).fillna("") == "yellow_inflate",
                    "yellow_heavy_inflated_r",
                    "skipped_other",
                ),
            ),
        ),
    )
    update_behavior_source = frame[["round", "driver_id", "driver_name"]].copy()
    update_behavior_source["update_behavior"] = behaviors
    update_behavior = (
        update_behavior_source.groupby(["round", "driver_id", "driver_name", "update_behavior"], as_index=False)
        .size()
        .rename(columns={"size": "laps"})
    )
    update_total = (
        update_behavior.groupby(["round", "driver_id", "driver_name"], as_index=False)["laps"]
        .sum()
        .rename(columns={"laps": "laps_total"})
    )
    update_behavior = update_behavior.merge(update_total, on=["round", "driver_id", "driver_name"], how="left")
    update_behavior["share"] = update_behavior["laps"] / update_behavior["laps_total"].replace(0, np.nan)

    # B8) Reset sanity around pit resets.
    pit_reset_rows: list[dict[str, Any]] = []
    for (driver_id, driver_name), driver_frame in frame.groupby(["driver_id", "driver_name"], sort=False):
        ordered = driver_frame.sort_values(["lap_number", "timestamp"], kind="mergesort").reset_index(drop=True)
        reset_idx = ordered.index[ordered["reset_applied"]].tolist()
        for idx in reset_idx:
            prev_row = ordered.iloc[idx - 1] if idx > 0 else None
            current_row = ordered.iloc[idx]
            next_rows = ordered.iloc[idx + 1 : idx + 4]
            pre_pace = _safe_optional_float(prev_row.get("pace_penalty_mean")) if prev_row is not None else None
            post_pace = _safe_optional_float(current_row.get("pace_penalty_mean"))
            ramp_3lap = None
            if not next_rows.empty and post_pace is not None:
                last_next = _safe_optional_float(next_rows.iloc[-1].get("pace_penalty_mean"))
                if last_next is not None:
                    ramp_3lap = float(last_next - post_pace)
            pit_reset_rows.append(
                {
                    "round": int(round_number),
                    "driver_id": str(driver_id),
                    "driver_name": str(driver_name),
                    "lap_number": _safe_optional_float(current_row.get("lap_number")),
                    "pre_pace_penalty": pre_pace,
                    "post_pace_penalty": post_pace,
                    "pace_drop_on_reset": (pre_pace - post_pace) if pre_pace is not None and post_pace is not None else None,
                    "pace_ramp_next_3laps": ramp_3lap,
                }
            )
    pit_reset_sanity = pd.DataFrame(pit_reset_rows)

    # C9-C11) One-step forecasting quality, coverage, and proper scoring.
    one_step_rows = frame[
        [
            "round",
            "driver_id",
            "driver_name",
            "lap_number",
            "baseline_lap",
            "lap_time_seconds",
            "one_step_pred_mean",
            "one_step_pred_std",
            "eval_included",
        ]
    ].copy()
    one_step_rows = one_step_rows[
        one_step_rows["lap_time_seconds"].notna()
        & one_step_rows["one_step_pred_mean"].notna()
        & one_step_rows["one_step_pred_std"].notna()
    ].copy()
    one_step_rows["predicted_residual"] = one_step_rows["one_step_pred_mean"] - one_step_rows["baseline_lap"]
    one_step_rows["actual_residual"] = one_step_rows["lap_time_seconds"] - one_step_rows["baseline_lap"]
    one_step_rows["forecast_error"] = one_step_rows["lap_time_seconds"] - one_step_rows["one_step_pred_mean"]
    one_step_rows["abs_forecast_error"] = one_step_rows["forecast_error"].abs()

    interval_rows = one_step_rows.copy()
    interval_rows["sigma"] = np.clip(pd.to_numeric(interval_rows["one_step_pred_std"], errors="coerce"), 1e-6, None)
    interval_rows["inside_50"] = interval_rows["forecast_error"].abs() <= (Z_SCORE_50 * interval_rows["sigma"])
    interval_rows["inside_90"] = interval_rows["forecast_error"].abs() <= (Z_SCORE_90 * interval_rows["sigma"])

    interval_coverage = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "round": int(round_number),
                        "scope": "round",
                        "lap_number": None,
                        "coverage_50": float(interval_rows["inside_50"].mean()) if not interval_rows.empty else None,
                        "coverage_90": float(interval_rows["inside_90"].mean()) if not interval_rows.empty else None,
                        "rows": int(len(interval_rows)),
                    }
                ]
            ),
            interval_rows.groupby("lap_number", as_index=False)
            .agg(
                coverage_50=("inside_50", "mean"),
                coverage_90=("inside_90", "mean"),
                rows=("inside_50", "size"),
            )
            .assign(round=int(round_number), scope="lap")[["round", "scope", "lap_number", "coverage_50", "coverage_90", "rows"]],
        ],
        ignore_index=True,
    )

    scoring_rows = interval_rows.copy()
    scoring_rows["sigma"] = np.clip(pd.to_numeric(scoring_rows["sigma"], errors="coerce"), 1e-6, None)
    scoring_rows["nll_like"] = 0.5 * np.log(2.0 * np.pi * (scoring_rows["sigma"] ** 2)) + 0.5 * (
        (scoring_rows["forecast_error"] / scoring_rows["sigma"]) ** 2
    )
    scoring_rows["crps_like"] = _gaussian_crps(
        scoring_rows["forecast_error"].to_numpy(dtype=float),
        scoring_rows["sigma"].to_numpy(dtype=float),
    )
    proper_scoring = pd.DataFrame(
        [
            {
                "round": int(round_number),
                "rows": int(len(scoring_rows)),
                "mae": float(scoring_rows["abs_forecast_error"].mean()) if not scoring_rows.empty else None,
                "nll_like": float(scoring_rows["nll_like"].mean()) if not scoring_rows.empty else None,
                "crps_like": float(scoring_rows["crps_like"].mean()) if not scoring_rows.empty else None,
            }
        ]
    )

    # E16-E19) Strategy/pit diagnostics from strategy posterior proxy.
    pit_hazard_rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        compound = str(row.get("compound") or "UNKNOWN")
        tyre_age = max(0, _safe_int(row.get("tyre_age"), 0))
        deg_rate = _safe_float(row.get("deg_rate_mean"), 0.0)
        probs = _strategy_template_probabilities(
            compound=compound,
            tyre_age=tyre_age,
            deg_rate=deg_rate,
            horizon=max(1, int(horizon_laps)),
        )
        p_hold = float(probs.get("hold_track_position", 0.0))
        p_one = float(probs.get("one_stop_conservative", 0.0))
        p_two_bal = float(probs.get("two_stop_balanced", 0.0))
        p_two_agg = float(probs.get("two_stop_aggressive", 0.0))
        hazard = float(np.clip(0.03 + (0.28 * p_one) + (0.52 * p_two_bal) + (0.74 * p_two_agg), 0.0, 0.98))
        pit_hazard_rows.append(
            {
                "round": int(round_number),
                "driver_id": str(row.get("driver_id")),
                "driver_name": str(row.get("driver_name")),
                "lap_number": _safe_optional_float(row.get("lap_number")),
                "tyre_age": tyre_age,
                "deg_rate_mean": deg_rate,
                "pit_hazard": hazard,
                "pit_event": bool(row.get("reset_applied", False)),
                "p_hold_track_position": p_hold,
                "p_one_stop_conservative": p_one,
                "p_two_stop_balanced": p_two_bal,
                "p_two_stop_aggressive": p_two_agg,
            }
        )
    pit_hazard_curve = pd.DataFrame(pit_hazard_rows)
    strategy_posterior = pit_hazard_curve[
        [
            "round",
            "driver_id",
            "driver_name",
            "lap_number",
            "tyre_age",
            "deg_rate_mean",
            "p_hold_track_position",
            "p_one_stop_conservative",
            "p_two_stop_balanced",
            "p_two_stop_aggressive",
        ]
    ].copy()

    pit_calibration_rows: list[dict[str, Any]] = []
    if not pit_hazard_curve.empty:
        for driver_id, driver_frame in pit_hazard_curve.groupby("driver_id", sort=False):
            ordered = driver_frame.sort_values("lap_number", kind="mergesort").reset_index(drop=True)
            laps = pd.to_numeric(ordered["lap_number"], errors="coerce").fillna(-1).astype(int).to_numpy(dtype=int)
            pit_events = ordered["pit_event"].astype(bool).to_numpy(dtype=bool)
            hazards = pd.to_numeric(ordered["pit_hazard"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            for idx, lap in enumerate(laps):
                if lap <= 0:
                    continue
                window_end = lap + max(1, int(pit_window_laps))
                future_mask = (laps > lap) & (laps <= window_end)
                event = bool(pit_events[future_mask].any())
                pred = float(1.0 - ((1.0 - np.clip(hazards[idx], 0.0, 0.999)) ** max(1, int(pit_window_laps))))
                pit_calibration_rows.append(
                    {
                        "round": int(round_number),
                        "driver_id": str(driver_id),
                        "lap_number": int(lap),
                        "pred_pit_within_window": pred,
                        "event_pit_within_window": float(event),
                        "window_laps": int(max(1, int(pit_window_laps))),
                    }
                )
    pit_calibration_samples = pd.DataFrame(pit_calibration_rows)
    if pit_calibration_samples.empty:
        time_to_pit_calibration = pd.DataFrame()
    else:
        pit_calibration_samples["bucket"] = np.clip(
            np.floor(pit_calibration_samples["pred_pit_within_window"] * 10.0) / 10.0,
            0.0,
            0.9,
        )
        time_to_pit_calibration = (
            pit_calibration_samples.groupby(["round", "window_laps", "bucket"], as_index=False)
            .agg(
                predicted_mean=("pred_pit_within_window", "mean"),
                observed_rate=("event_pit_within_window", "mean"),
                rows=("event_pit_within_window", "size"),
            )
            .sort_values(["round", "bucket"], kind="mergesort")
        )

    round_health = pd.DataFrame(
        [
            {
                "round": int(round_number),
                "pit_reset_count": int(frame["reset_applied"].sum()),
                "skip_share": float((~frame["eval_included"]).mean()),
                "invalid_race_time_count": int((~np.isfinite(frame["race_time_seconds"])).sum()),
                "non_monotone_drivers": int((race_time_summary["race_time_monotone"] == False).sum())
                if not race_time_summary.empty
                else 0,
                "laps_total_driver_rows": int(len(frame)),
            }
        ]
    )

    return {
        "a1_lap_availability": lap_availability,
        "a2_track_status_composition": track_status_composition,
        "a3_baseline_curve": baseline_curve,
        "a3_residual_distribution": baseline_residual_distribution,
        "a4_race_time_sanity_summary": race_time_summary,
        "a4_race_time_sanity_curve": race_time_curve,
        "b5_state_trajectories": state_trajectories,
        "b6_innovation_rows": innovation_rows,
        "b6_innovation_hist": innovation_hist,
        "b7_update_behavior": update_behavior,
        "b8_pit_reset_sanity": pit_reset_sanity,
        "c9_one_step_rows": one_step_rows,
        "c10_interval_coverage": interval_coverage,
        "c11_proper_scoring": proper_scoring,
        "e16_pit_hazard_curve": pit_hazard_curve,
        "e17_time_to_pit_calibration": time_to_pit_calibration,
        "e17_time_to_pit_calibration_samples": pit_calibration_samples,
        "e19_strategy_posterior": strategy_posterior,
        "round_health": round_health,
    }


def _build_cutoff_observability(
    *,
    round_number: int,
    cutoff: dict[str, Any],
    snapshot: pd.DataFrame,
    dist_summary: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    cutoff_common = {
        "round": int(round_number),
        "cutoff_mode": str(cutoff.get("cutoff_mode") or ""),
        "cutoff_label": str(cutoff.get("cutoff_label") or ""),
        "cutoff_sort_value": _safe_optional_float(cutoff.get("cutoff_sort_value")),
        "lap_cutoff": _safe_optional_float(cutoff.get("lap_cutoff")),
        "cutoff_pct_realized": _safe_optional_float(cutoff.get("cutoff_pct_realized")),
    }

    ranked = snapshot.copy()
    if ranked.empty:
        return {
            "d12_position_distribution_top5": pd.DataFrame(),
            "d13_pairwise_ahead_top10": pd.DataFrame(),
            "d14_ranking_curve": pd.DataFrame(),
            "d15_mc_health": pd.DataFrame([cutoff_common]),
        }

    ranked["rank"] = pd.to_numeric(ranked.get("rank"), errors="coerce")
    ranked = ranked[ranked["rank"].notna()].copy()
    ranked = ranked.sort_values("rank", kind="mergesort")
    top5_ids = set(ranked.head(5)["driver_id"].astype(str).tolist())
    top10_ids = set(ranked.head(10)["driver_id"].astype(str).tolist())
    id_to_name = {str(row["driver_id"]): str(row.get("driver_name") or row["driver_id"]) for _, row in ranked.iterrows()}

    raw_position_probs = dist_summary.get("position_probabilities")
    if isinstance(raw_position_probs, list) and raw_position_probs:
        position_df = pd.DataFrame(raw_position_probs).copy()
        position_df["driver_id"] = position_df.get("driver_id", "").astype(str)
        position_df["position"] = pd.to_numeric(position_df.get("position"), errors="coerce")
        position_df["probability"] = pd.to_numeric(position_df.get("probability"), errors="coerce")
        position_df = position_df[
            position_df["driver_id"].isin(top5_ids)
            & position_df["position"].notna()
            & (position_df["position"] <= 10)
            & position_df["probability"].notna()
        ].copy()
        position_df["driver_name"] = position_df["driver_id"].map(id_to_name)
        for key, value in cutoff_common.items():
            position_df[key] = value
        position_dist_top5 = position_df[
            [
                "round",
                "cutoff_mode",
                "cutoff_label",
                "cutoff_sort_value",
                "lap_cutoff",
                "cutoff_pct_realized",
                "driver_id",
                "driver_name",
                "position",
                "probability",
            ]
        ].copy()
    else:
        position_dist_top5 = pd.DataFrame()

    raw_pairwise = dist_summary.get("pairwise_ahead_probabilities")
    if isinstance(raw_pairwise, list) and raw_pairwise:
        pairwise_df = pd.DataFrame(raw_pairwise).copy()
        pairwise_df["driver_a"] = pairwise_df.get("driver_a", "").astype(str)
        pairwise_df["driver_b"] = pairwise_df.get("driver_b", "").astype(str)
        pairwise_df["probability_a_ahead_b"] = pd.to_numeric(pairwise_df.get("probability_a_ahead_b"), errors="coerce")
        pairwise_df = pairwise_df[
            pairwise_df["driver_a"].isin(top10_ids)
            & pairwise_df["driver_b"].isin(top10_ids)
            & pairwise_df["probability_a_ahead_b"].notna()
        ].copy()
        pairwise_df["driver_a_name"] = pairwise_df["driver_a"].map(id_to_name)
        pairwise_df["driver_b_name"] = pairwise_df["driver_b"].map(id_to_name)
        for key, value in cutoff_common.items():
            pairwise_df[key] = value
        pairwise_ahead_top10 = pairwise_df[
            [
                "round",
                "cutoff_mode",
                "cutoff_label",
                "cutoff_sort_value",
                "lap_cutoff",
                "cutoff_pct_realized",
                "driver_a",
                "driver_a_name",
                "driver_b",
                "driver_b_name",
                "probability_a_ahead_b",
            ]
        ].copy()
    else:
        pairwise_ahead_top10 = pd.DataFrame()

    ranking_curve = ranked[
        [
            "driver_id",
            "driver_name",
            "rank",
            "exp_pos_H",
            "p_win_H",
            "p_top3_H",
            "p_top10_H",
        ]
    ].copy()
    for key, value in cutoff_common.items():
        ranking_curve[key] = value
    ranking_curve = ranking_curve[
        [
            "round",
            "cutoff_mode",
            "cutoff_label",
            "cutoff_sort_value",
            "lap_cutoff",
            "cutoff_pct_realized",
            "driver_id",
            "driver_name",
            "rank",
            "exp_pos_H",
            "p_win_H",
            "p_top3_H",
            "p_top10_H",
        ]
    ]

    p_win = pd.to_numeric(ranked.get("p_win_H"), errors="coerce")
    mc_samples_effective = _safe_optional_float(dist_summary.get("mc_samples_effective"))
    if mc_samples_effective is not None and mc_samples_effective > 0 and p_win.notna().any():
        p_win_valid = p_win.dropna()
        p_win_se = np.sqrt(np.clip(p_win_valid * (1.0 - p_win_valid), 0.0, 1.0) / float(mc_samples_effective))
        mean_p_win_se = float(p_win_se.mean())
        max_p_win_se = float(p_win_se.max())
    else:
        mean_p_win_se = None
        max_p_win_se = None

    d15_mc_health = pd.DataFrame(
        [
            {
                **cutoff_common,
                "mc_samples_requested": _safe_optional_float(dist_summary.get("mc_samples_requested")),
                "mc_samples_effective": mc_samples_effective,
                "mc_samples_reduction_reason": dist_summary.get("mc_samples_reduction_reason"),
                "sum_p_win": _safe_optional_float(dist_summary.get("sum_p_win")),
                "mean_p_win_se": mean_p_win_se,
                "max_p_win_se": max_p_win_se,
            }
        ]
    )

    return {
        "d12_position_distribution_top5": position_dist_top5,
        "d13_pairwise_ahead_top10": pairwise_ahead_top10,
        "d14_ranking_curve": ranking_curve,
        "d15_mc_health": d15_mc_health,
    }


def _build_ranking_stability(ranking_curve: pd.DataFrame) -> pd.DataFrame:
    if ranking_curve.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for round_number, group in ranking_curve.groupby("round", sort=True):
        ordered = group.sort_values("cutoff_sort_value", kind="mergesort")
        if ordered.empty:
            continue
        cutoff_values = sorted(
            [float(value) for value in pd.to_numeric(ordered["cutoff_sort_value"], errors="coerce").dropna().unique().tolist()]
        )
        if not cutoff_values:
            continue
        final_cutoff = cutoff_values[-1]
        final_rows = ordered[pd.to_numeric(ordered["cutoff_sort_value"], errors="coerce") == final_cutoff]
        final_top3 = set(final_rows.nsmallest(3, "exp_pos_H")["driver_id"].astype(str).tolist())
        for cutoff_value in cutoff_values:
            cutoff_rows = ordered[pd.to_numeric(ordered["cutoff_sort_value"], errors="coerce") == cutoff_value]
            top3 = set(cutoff_rows.nsmallest(3, "exp_pos_H")["driver_id"].astype(str).tolist())
            inter = len(top3.intersection(final_top3))
            union = len(top3.union(final_top3))
            rows.append(
                {
                    "round": int(round_number),
                    "cutoff_sort_value": float(cutoff_value),
                    "cutoff_label": str(cutoff_rows.iloc[0].get("cutoff_label") or ""),
                    "top3_overlap_count": int(inter),
                    "top3_jaccard_vs_final": float(inter / union) if union > 0 else None,
                    "top3_matches_final": bool(inter == 3),
                }
            )
    return pd.DataFrame(rows).sort_values(["round", "cutoff_sort_value"], kind="mergesort").reset_index(drop=True)


def _sample_pit_loss_distribution(seed: int, samples_per_regime: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    for regime in ["green", "yellow", "sc_vsc"]:
        for sample_idx in range(max(100, int(samples_per_regime))):
            rows.append(
                {
                    "regime": regime,
                    "sample_idx": int(sample_idx),
                    "pit_loss_seconds": float(_sample_pit_loss_seconds(regime, rng)),
                }
            )
    return pd.DataFrame(rows)


def _build_metric_curve_frame(summary_frame: pd.DataFrame) -> pd.DataFrame:
    if summary_frame.empty:
        return pd.DataFrame()
    metric_map = [
        ("mae", "A_weighted_mae", "B_weighted_mae", False),
        ("rmse", "A_weighted_rmse", "B_weighted_rmse", False),
        ("spearman", "A_weighted_spearman", "B_weighted_spearman", True),
        ("top10_hit", "A_mean_top10_hit", "B_mean_top10_hit", True),
        ("top3_hit", "A_mean_top3_hit", "B_mean_top3_hit", True),
        ("pred_top10_mae", "A_mean_pred_top10_mae", "B_mean_pred_top10_mae", False),
    ]
    rows: list[dict[str, Any]] = []
    for _, row in summary_frame.iterrows():
        for metric, a_col, b_col, higher_is_better in metric_map:
            rows.append(
                {
                    "cutoff_mode": str(row.get("cutoff_mode") or ""),
                    "cutoff_label": str(row.get("cutoff_label") or ""),
                    "cutoff_sort_value": _safe_optional_float(row.get("cutoff_sort_value")),
                    "metric": metric,
                    "higher_is_better": bool(higher_is_better),
                    "A_value": _safe_optional_float(row.get(a_col)),
                    "B_value": _safe_optional_float(row.get(b_col)),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["B_minus_A"] = pd.to_numeric(out["B_value"], errors="coerce") - pd.to_numeric(out["A_value"], errors="coerce")
    out["oriented_gain_B"] = np.where(
        out["higher_is_better"],
        out["B_minus_A"],
        -out["B_minus_A"],
    )
    return out.sort_values(["metric", "cutoff_sort_value"], kind="mergesort").reset_index(drop=True)


def _build_crossover_heatmap_frame(
    per_round_frame: pd.DataFrame,
    *,
    metric_specs: list[dict[str, Any]],
) -> pd.DataFrame:
    if per_round_frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in per_round_frame.iterrows():
        for spec in metric_specs:
            metric = str(spec["metric"])
            a_value = _safe_optional_float(row.get(spec["a_col"]))
            b_value = _safe_optional_float(row.get(spec["b_col"]))
            epsilon = float(spec["epsilon"])
            higher = bool(spec["higher_is_better"])
            winner = "unknown"
            encoded = np.nan
            if a_value is not None and b_value is not None:
                if higher:
                    if b_value >= a_value + epsilon:
                        winner, encoded = "B", 1.0
                    elif a_value >= b_value + epsilon:
                        winner, encoded = "A", -1.0
                    else:
                        winner, encoded = "Tie", 0.0
                else:
                    if b_value <= a_value - epsilon:
                        winner, encoded = "B", 1.0
                    elif a_value <= b_value - epsilon:
                        winner, encoded = "A", -1.0
                    else:
                        winner, encoded = "Tie", 0.0
            rows.append(
                {
                    "round": int(row.get("round")),
                    "cutoff_label": str(row.get("cutoff_label") or ""),
                    "cutoff_sort_value": _safe_optional_float(row.get("cutoff_sort_value")),
                    "metric": metric,
                    "winner": winner,
                    "winner_encoded": encoded,
                }
            )
    return pd.DataFrame(rows).sort_values(["metric", "round", "cutoff_sort_value"], kind="mergesort").reset_index(drop=True)


def _build_round_outliers(
    *,
    crossover_per_round: pd.DataFrame,
    round_health: pd.DataFrame,
    metric: str,
    top_n: int = 3,
) -> pd.DataFrame:
    if crossover_per_round.empty:
        return pd.DataFrame()
    metric_rows = crossover_per_round[crossover_per_round["metric"] == str(metric)].copy()
    if metric_rows.empty:
        return pd.DataFrame()
    metric_rows["sort_value_effective"] = pd.to_numeric(metric_rows["crossover_cutoff_sort_value"], errors="coerce")
    metric_rows["sort_value_effective"] = metric_rows["sort_value_effective"].fillna(1_000_000.0)
    ranked = metric_rows.sort_values(["sort_value_effective", "round"], ascending=[False, True], kind="mergesort")
    outliers = ranked.head(max(1, int(top_n))).copy()
    outliers["never_before_finish"] = ~outliers["crossover_found"].astype(bool)
    if not round_health.empty:
        outliers = outliers.merge(round_health, on="round", how="left")
    return outliers


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, ignore_index=True)


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


def _render_baseline_quality_plot(
    baseline_curve: pd.DataFrame,
    baseline_residual_distribution: pd.DataFrame,
    output_path: Path,
) -> Optional[str]:
    if plt is None or baseline_curve.empty or baseline_residual_distribution.empty:
        return None
    base = (
        baseline_curve.groupby("lap_number", as_index=False)["event_lap_baseline_t"]
        .median()
        .sort_values("lap_number", kind="mergesort")
    )
    resid = (
        baseline_residual_distribution.groupby("lap_number", as_index=False)
        .agg(
            residual_median=("residual_median", "median"),
            residual_p10=("residual_p10", "median"),
            residual_p90=("residual_p90", "median"),
        )
        .sort_values("lap_number", kind="mergesort")
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    axes[0].plot(base["lap_number"], base["event_lap_baseline_t"], color="#0b7285", linewidth=2.0)
    axes[0].set_title("Event Lap Baseline (Median)")
    axes[0].set_xlabel("Lap")
    axes[0].set_ylabel("Baseline seconds")
    axes[0].grid(alpha=0.25)

    axes[1].plot(resid["lap_number"], resid["residual_median"], color="#1c7ed6", linewidth=2.0, label="Median")
    axes[1].fill_between(
        resid["lap_number"],
        resid["residual_p10"],
        resid["residual_p90"],
        color="#74c0fc",
        alpha=0.25,
        label="P10-P90",
    )
    axes[1].axhline(0.0, color="#6c757d", linewidth=1.0, linestyle="--")
    axes[1].set_title("Lap Time - Baseline Distribution")
    axes[1].set_xlabel("Lap")
    axes[1].set_ylabel("Seconds")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _render_innovation_histogram_plot(innovation_hist: pd.DataFrame, output_path: Path) -> Optional[str]:
    if plt is None or innovation_hist.empty:
        return None
    grouped = innovation_hist.groupby(["bin_left", "bin_right"], as_index=False)["count"].sum()
    centers = 0.5 * (grouped["bin_left"] + grouped["bin_right"])
    widths = grouped["bin_right"] - grouped["bin_left"]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.bar(centers, grouped["count"], width=widths, align="center", color="#495057", alpha=0.85)
    ax.set_title("Standardized Innovation Histogram (Green Laps)")
    ax.set_xlabel("z innovation")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _render_one_step_scatter_plot(one_step_rows: pd.DataFrame, output_path: Path) -> Optional[str]:
    if plt is None or one_step_rows.empty:
        return None
    frame = one_step_rows.copy()
    frame = frame[
        frame["predicted_residual"].notna() & frame["actual_residual"].notna() & frame["lap_number"].notna()
    ].copy()
    if frame.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    axes[0].scatter(
        frame["predicted_residual"],
        frame["actual_residual"],
        s=10,
        alpha=0.35,
        color="#2b8a3e",
        edgecolors="none",
    )
    lim_min = float(min(frame["predicted_residual"].min(), frame["actual_residual"].min()))
    lim_max = float(max(frame["predicted_residual"].max(), frame["actual_residual"].max()))
    axes[0].plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", color="#6c757d", linewidth=1.0)
    axes[0].set_title("One-Step Residual Forecast vs Actual")
    axes[0].set_xlabel("Predicted residual")
    axes[0].set_ylabel("Actual residual")
    axes[0].grid(alpha=0.2)

    error_by_lap = frame.groupby("lap_number", as_index=False)["abs_forecast_error"].mean()
    axes[1].plot(error_by_lap["lap_number"], error_by_lap["abs_forecast_error"], color="#d9480f", linewidth=2.0)
    axes[1].set_title("Absolute Error by Lap")
    axes[1].set_xlabel("Lap")
    axes[1].set_ylabel("Mean abs error (s)")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _render_metric_curves_plot(metric_curve_frame: pd.DataFrame, output_path: Path, *, cutoff_mode: str) -> Optional[str]:
    if plt is None or metric_curve_frame.empty:
        return None
    metrics = ["mae", "rmse", "spearman", "top10_hit", "top3_hit", "pred_top10_mae"]
    frame = metric_curve_frame.copy()
    frame["cutoff_sort_value"] = pd.to_numeric(frame["cutoff_sort_value"], errors="coerce")
    frame = frame[frame["cutoff_sort_value"].notna()].copy()
    if frame.empty:
        return None

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), dpi=150, sharex=True)
    for idx, metric in enumerate(metrics):
        ax = axes.flat[idx]
        subset = frame[frame["metric"] == metric].sort_values("cutoff_sort_value", kind="mergesort")
        if subset.empty:
            ax.set_visible(False)
            continue
        x = subset["cutoff_sort_value"].to_numpy(dtype=float)
        ax.plot(x, subset["A_value"], marker="o", linewidth=1.8, color="#495057", label="A")
        ax.plot(x, subset["B_value"], marker="o", linewidth=1.8, color="#1c7ed6", label="B")
        ax.set_title(metric.replace("_", " ").upper())
        ax.grid(alpha=0.2)
        if idx in {0, 1}:
            ax.legend(loc="best")
    xlabel = "Distance cutoff (%)" if cutoff_mode == "distance_pct" else "Lap cutoff"
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _render_delta_plot(metric_curve_frame: pd.DataFrame, output_path: Path, *, cutoff_mode: str) -> Optional[str]:
    if plt is None or metric_curve_frame.empty:
        return None
    metrics = ["mae", "spearman", "top10_hit", "top3_hit", "pred_top10_mae"]
    frame = metric_curve_frame.copy()
    frame["cutoff_sort_value"] = pd.to_numeric(frame["cutoff_sort_value"], errors="coerce")
    frame = frame[frame["cutoff_sort_value"].notna()].copy()
    if frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    for metric in metrics:
        subset = frame[frame["metric"] == metric].sort_values("cutoff_sort_value", kind="mergesort")
        if subset.empty:
            continue
        ax.plot(
            subset["cutoff_sort_value"],
            subset["oriented_gain_B"],
            marker="o",
            linewidth=1.7,
            label=metric,
        )
    ax.axhline(0.0, color="#6c757d", linewidth=1.0, linestyle="--")
    ax.set_title("Delta by Cutoff (B gain, oriented)")
    ax.set_xlabel("Distance cutoff (%)" if cutoff_mode == "distance_pct" else "Lap cutoff")
    ax.set_ylabel("Oriented gain (higher is better)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _render_heatmap_plot(
    heatmap_frame: pd.DataFrame,
    output_path: Path,
    *,
    metric: str,
    cutoff_mode: str,
) -> Optional[str]:
    if plt is None or heatmap_frame.empty:
        return None
    frame = heatmap_frame[heatmap_frame["metric"] == metric].copy()
    if frame.empty:
        return None
    frame["cutoff_sort_value"] = pd.to_numeric(frame["cutoff_sort_value"], errors="coerce")
    frame = frame[frame["cutoff_sort_value"].notna()].copy()
    if frame.empty:
        return None
    matrix = frame.pivot_table(
        index="round",
        columns="cutoff_sort_value",
        values="winner_encoded",
        aggfunc="first",
    ).sort_index(axis=0).sort_index(axis=1)
    if matrix.empty:
        return None
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    im = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=-1.0, vmax=1.0)
    ax.set_title(f"Crossover Heatmap ({metric})")
    ax.set_xlabel("Distance cutoff (%)" if cutoff_mode == "distance_pct" else "Lap cutoff")
    ax.set_ylabel("Round")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([f"{value:g}" for value in matrix.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([str(int(value)) for value in matrix.index])
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("-1=A better, +1=B better")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _render_pit_hazard_plot(pit_hazard_curve: pd.DataFrame, output_path: Path) -> Optional[str]:
    if plt is None or pit_hazard_curve.empty:
        return None
    frame = pit_hazard_curve.copy()
    frame["tyre_age"] = pd.to_numeric(frame["tyre_age"], errors="coerce")
    frame["pit_hazard"] = pd.to_numeric(frame["pit_hazard"], errors="coerce")
    frame["pit_event"] = pd.to_numeric(frame["pit_event"], errors="coerce").fillna(0.0)
    frame = frame[frame["tyre_age"].notna() & frame["pit_hazard"].notna()].copy()
    if frame.empty:
        return None
    grouped = frame.groupby("tyre_age", as_index=False).agg(
        pit_hazard_mean=("pit_hazard", "mean"),
        pit_event_rate=("pit_event", "mean"),
    )
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(grouped["tyre_age"], grouped["pit_hazard_mean"], linewidth=2.0, color="#c92a2a", label="Pit hazard")
    ax.plot(grouped["tyre_age"], grouped["pit_event_rate"], linewidth=2.0, color="#0b7285", label="Observed pit rate")
    ax.set_title("Pit Hazard vs Tyre Age")
    ax.set_xlabel("Tyre age")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _render_chaos_scatter_plot(round_chaos: pd.DataFrame, outliers: pd.DataFrame, output_path: Path) -> Optional[str]:
    if plt is None or round_chaos.empty:
        return None
    frame = round_chaos.copy()
    frame["chaos_fraction"] = pd.to_numeric(frame["chaos_fraction"], errors="coerce")
    frame = frame[frame["chaos_fraction"].notna()].copy()
    if frame.empty:
        return None
    marker_map: dict[int, float] = {}
    if not outliers.empty:
        for _, row in outliers.iterrows():
            round_number = _safe_int(row.get("round"), -1)
            marker_map[round_number] = float(_safe_optional_float(row.get("sort_value_effective")) or 1_000_000.0)
    y_values = [marker_map.get(int(value), np.nan) for value in frame["round"].tolist()]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.scatter(frame["chaos_fraction"], y_values, color="#5f3dc4", alpha=0.75)
    ax.set_title("Chaos Fraction vs Crossover Timing (Outliers)")
    ax.set_xlabel("Chaos fraction")
    ax.set_ylabel("Crossover cutoff sort value (MAE)")
    ax.grid(alpha=0.2)
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
        default="outputs/f1/compare_2025_afterfix_fullrace/horizon_a",
        help="Folder containing Horizon A artifacts (rXX.json).",
    )
    parser.add_argument(
        "--horizon-b-dir",
        default="outputs/f1/compare_2025_afterfix_fullrace/horizon_b",
        help="Folder containing Horizon B artifacts (rXX.json) with trace_path.",
    )
    parser.add_argument(
        "--weekends-dir",
        default="data/f1/raw/weekends",
        help="Local weekends root for actual race results.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/f1/compare_2025_afterfix_lap_snapshots",
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
        default=1000,
        help="Requested Monte Carlo samples per snapshot (default: 1000).",
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
    parser.add_argument(
        "--pit-window-laps",
        type=int,
        default=3,
        help="Window size for pit-within-W-laps calibration diagnostics.",
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
    observability_parts: dict[str, list[pd.DataFrame]] = {
        "a1_lap_availability": [],
        "a2_track_status_composition": [],
        "a3_baseline_curve": [],
        "a3_residual_distribution": [],
        "a4_race_time_sanity_summary": [],
        "a4_race_time_sanity_curve": [],
        "b5_state_trajectories": [],
        "b6_innovation_rows": [],
        "b6_innovation_hist": [],
        "b7_update_behavior": [],
        "b8_pit_reset_sanity": [],
        "c9_one_step_rows": [],
        "c10_interval_coverage": [],
        "c11_proper_scoring": [],
        "d12_position_distribution_top5": [],
        "d13_pairwise_ahead_top10": [],
        "d14_ranking_curve": [],
        "d15_mc_health": [],
        "e16_pit_hazard_curve": [],
        "e17_time_to_pit_calibration": [],
        "e17_time_to_pit_calibration_samples": [],
        "e19_strategy_posterior": [],
        "round_health": [],
    }

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
        round_obs = _build_round_observability(
            round_number=int(round_number),
            trace=trace,
            horizon_laps=max(1, int(args.horizon_laps)),
            pit_window_laps=max(1, int(args.pit_window_laps)),
        )
        for key, frame in round_obs.items():
            if frame is None or frame.empty:
                continue
            payload = frame.copy()
            if key == "round_health":
                payload["chaos_fraction"] = float(chaos_profile["chaos_fraction"])
                payload["chaos_segment"] = str(chaos_profile["chaos_segment"])
                payload["sc_vsc_laps"] = int(chaos_profile["sc_vsc_laps"])
            observability_parts.setdefault(key, []).append(payload)

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
                "B_mc_samples_requested": dist_summary.get("mc_samples_requested"),
                "B_mc_samples_effective": dist_summary.get("mc_samples_effective"),
                "B_sum_p_win": dist_summary.get("sum_p_win"),
                "trace_path": str(trace_path),
            }
            per_round_rows.append(row)
            cutoff_obs = _build_cutoff_observability(
                round_number=int(round_number),
                cutoff=cutoff,
                snapshot=snapshot,
                dist_summary=dist_summary,
            )
            for key, frame in cutoff_obs.items():
                if frame is None or frame.empty:
                    continue
                observability_parts.setdefault(key, []).append(frame.copy())

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

    round_chaos = (
        per_round_frame.sort_values(["round", "cutoff_sort_value"], kind="mergesort")
        .drop_duplicates(subset=["round"], keep="first")
        .copy()
    )
    chaos_segment_counts = {
        str(label): int(count)
        for label, count in round_chaos["chaos_segment"].value_counts(dropna=False).to_dict().items()
    }

    obs_frames = {key: _concat_frames(parts) for key, parts in observability_parts.items()}
    d14_ranking_stability = _build_ranking_stability(obs_frames.get("d14_ranking_curve", pd.DataFrame()))
    e18_pit_loss_distribution = _sample_pit_loss_distribution(seed=int(args.f1_live_seed), samples_per_regime=2500)
    metric_curve_frame = _build_metric_curve_frame(summary_frame)
    g26_crossover_heatmap = _build_crossover_heatmap_frame(per_round_frame, metric_specs=metric_specs)
    f21_crossover_by_segment = crossover_distribution_frame[
        crossover_distribution_frame["chaos_segment"].astype(str).str.lower() != "all"
    ].copy()
    f22_crossover_survival_selected = crossover_survival_frame[
        crossover_survival_frame["metric"].isin(["mae", "spearman", "top10_hit", "top3_hit"])
    ].copy()
    f23_round_outliers = _build_round_outliers(
        crossover_per_round=crossover_per_round_frame,
        round_health=obs_frames.get("round_health", pd.DataFrame()),
        metric="mae",
        top_n=3,
    )
    f20_chaos_index = round_chaos[
        ["round", "total_laps_completed", "sc_vsc_laps", "chaos_fraction", "chaos_segment", "chaos_has_incident"]
    ].copy()

    per_round_path = output_dir / "horizon_a_vs_b_lap_snapshots_per_round.csv"
    summary_csv_path = output_dir / "horizon_a_vs_b_lap_snapshots_summary.csv"
    summary_json_path = output_dir / "horizon_a_vs_b_lap_snapshots_summary.json"
    curve_csv_path = output_dir / "horizon_a_vs_b_lap_overtake_curve.csv"
    crossover_per_round_path = output_dir / "horizon_a_vs_b_crossover_per_round.csv"
    crossover_distribution_path = output_dir / "horizon_a_vs_b_crossover_distribution.csv"
    crossover_survival_path = output_dir / "horizon_a_vs_b_crossover_survival.csv"
    crossover_overview_path = output_dir / "horizon_a_vs_b_crossover_overview.csv"
    plot_path = output_dir / "horizon_a_vs_b_lap_overtake.png"
    observability_dir = output_dir / "observability"
    observability_dir.mkdir(parents=True, exist_ok=True)

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

    obs_csv_targets: dict[str, tuple[pd.DataFrame, Path]] = {
        "a1_lap_availability_csv": (obs_frames.get("a1_lap_availability", pd.DataFrame()), observability_dir / "a1_lap_availability.csv"),
        "a2_track_status_composition_csv": (
            obs_frames.get("a2_track_status_composition", pd.DataFrame()),
            observability_dir / "a2_track_status_composition.csv",
        ),
        "a3_baseline_curve_csv": (obs_frames.get("a3_baseline_curve", pd.DataFrame()), observability_dir / "a3_baseline_curve.csv"),
        "a3_residual_distribution_csv": (
            obs_frames.get("a3_residual_distribution", pd.DataFrame()),
            observability_dir / "a3_residual_distribution.csv",
        ),
        "a4_race_time_sanity_summary_csv": (
            obs_frames.get("a4_race_time_sanity_summary", pd.DataFrame()),
            observability_dir / "a4_race_time_sanity_summary.csv",
        ),
        "a4_race_time_sanity_curve_csv": (
            obs_frames.get("a4_race_time_sanity_curve", pd.DataFrame()),
            observability_dir / "a4_race_time_sanity_curve.csv",
        ),
        "b5_state_trajectories_csv": (
            obs_frames.get("b5_state_trajectories", pd.DataFrame()),
            observability_dir / "b5_state_trajectories.csv",
        ),
        "b6_innovation_rows_csv": (
            obs_frames.get("b6_innovation_rows", pd.DataFrame()),
            observability_dir / "b6_innovation_rows.csv",
        ),
        "b6_innovation_hist_csv": (
            obs_frames.get("b6_innovation_hist", pd.DataFrame()),
            observability_dir / "b6_innovation_hist.csv",
        ),
        "b7_update_behavior_csv": (
            obs_frames.get("b7_update_behavior", pd.DataFrame()),
            observability_dir / "b7_update_behavior.csv",
        ),
        "b8_pit_reset_sanity_csv": (
            obs_frames.get("b8_pit_reset_sanity", pd.DataFrame()),
            observability_dir / "b8_pit_reset_sanity.csv",
        ),
        "c9_one_step_rows_csv": (
            obs_frames.get("c9_one_step_rows", pd.DataFrame()),
            observability_dir / "c9_one_step_rows.csv",
        ),
        "c10_interval_coverage_csv": (
            obs_frames.get("c10_interval_coverage", pd.DataFrame()),
            observability_dir / "c10_interval_coverage.csv",
        ),
        "c11_proper_scoring_csv": (
            obs_frames.get("c11_proper_scoring", pd.DataFrame()),
            observability_dir / "c11_proper_scoring.csv",
        ),
        "d12_position_distribution_top5_csv": (
            obs_frames.get("d12_position_distribution_top5", pd.DataFrame()),
            observability_dir / "d12_position_distribution_top5.csv",
        ),
        "d13_pairwise_ahead_top10_csv": (
            obs_frames.get("d13_pairwise_ahead_top10", pd.DataFrame()),
            observability_dir / "d13_pairwise_ahead_top10.csv",
        ),
        "d14_ranking_curve_csv": (
            obs_frames.get("d14_ranking_curve", pd.DataFrame()),
            observability_dir / "d14_ranking_curve.csv",
        ),
        "d14_ranking_stability_csv": (
            d14_ranking_stability,
            observability_dir / "d14_ranking_stability.csv",
        ),
        "d15_mc_health_csv": (
            obs_frames.get("d15_mc_health", pd.DataFrame()),
            observability_dir / "d15_mc_health.csv",
        ),
        "e16_pit_hazard_curve_csv": (
            obs_frames.get("e16_pit_hazard_curve", pd.DataFrame()),
            observability_dir / "e16_pit_hazard_curve.csv",
        ),
        "e17_time_to_pit_calibration_csv": (
            obs_frames.get("e17_time_to_pit_calibration", pd.DataFrame()),
            observability_dir / "e17_time_to_pit_calibration.csv",
        ),
        "e17_time_to_pit_calibration_samples_csv": (
            obs_frames.get("e17_time_to_pit_calibration_samples", pd.DataFrame()),
            observability_dir / "e17_time_to_pit_calibration_samples.csv",
        ),
        "e18_pit_loss_distribution_csv": (
            e18_pit_loss_distribution,
            observability_dir / "e18_pit_loss_distribution.csv",
        ),
        "e19_strategy_posterior_csv": (
            obs_frames.get("e19_strategy_posterior", pd.DataFrame()),
            observability_dir / "e19_strategy_posterior.csv",
        ),
        "f20_chaos_index_csv": (f20_chaos_index, observability_dir / "f20_chaos_index.csv"),
        "f21_crossover_by_segment_csv": (f21_crossover_by_segment, observability_dir / "f21_crossover_by_segment.csv"),
        "f22_crossover_survival_csv": (
            f22_crossover_survival_selected,
            observability_dir / "f22_crossover_survival_selected.csv",
        ),
        "f23_round_outliers_csv": (f23_round_outliers, observability_dir / "f23_round_outliers.csv"),
        "g24_metric_curves_csv": (metric_curve_frame, observability_dir / "g24_metric_curves.csv"),
        "g25_delta_by_cutoff_csv": (metric_curve_frame, observability_dir / "g25_delta_by_cutoff.csv"),
        "g26_crossover_heatmap_csv": (g26_crossover_heatmap, observability_dir / "g26_crossover_heatmap.csv"),
    }
    observability_artifacts: dict[str, Optional[str]] = {}
    for artifact_name, (artifact_frame, artifact_path) in obs_csv_targets.items():
        if artifact_frame is None or artifact_frame.empty:
            observability_artifacts[artifact_name] = None
            continue
        artifact_frame.to_csv(artifact_path, index=False)
        observability_artifacts[artifact_name] = str(artifact_path)

    observability_plot_targets = {
        "a3_baseline_quality_plot": _render_baseline_quality_plot(
            obs_frames.get("a3_baseline_curve", pd.DataFrame()),
            obs_frames.get("a3_residual_distribution", pd.DataFrame()),
            observability_dir / "a3_baseline_quality.png",
        ),
        "b6_innovation_histogram_plot": _render_innovation_histogram_plot(
            obs_frames.get("b6_innovation_hist", pd.DataFrame()),
            observability_dir / "b6_innovation_histogram.png",
        ),
        "c9_one_step_scatter_plot": _render_one_step_scatter_plot(
            obs_frames.get("c9_one_step_rows", pd.DataFrame()),
            observability_dir / "c9_one_step_scatter.png",
        ),
        "e16_pit_hazard_plot": _render_pit_hazard_plot(
            obs_frames.get("e16_pit_hazard_curve", pd.DataFrame()),
            observability_dir / "e16_pit_hazard_plot.png",
        ),
        "f20_chaos_scatter_plot": _render_chaos_scatter_plot(
            f20_chaos_index,
            f23_round_outliers,
            observability_dir / "f20_chaos_vs_crossover.png",
        ),
        "g24_metric_curves_plot": _render_metric_curves_plot(
            metric_curve_frame,
            observability_dir / "g24_metric_curves.png",
            cutoff_mode=args.cutoff_mode,
        ),
        "g25_delta_plot": _render_delta_plot(
            metric_curve_frame,
            observability_dir / "g25_delta_plot.png",
            cutoff_mode=args.cutoff_mode,
        ),
        "g26_heatmap_mae_plot": _render_heatmap_plot(
            g26_crossover_heatmap,
            observability_dir / "g26_heatmap_mae.png",
            metric="mae",
            cutoff_mode=args.cutoff_mode,
        ),
    }
    observability_artifacts.update({name: path for name, path in observability_plot_targets.items()})

    summary_payload = {
        "year": int(args.year),
        "cutoff_mode": str(args.cutoff_mode),
        "distance_cutoffs": [int(value) for value in pct_cutoffs],
        "lap_cutoffs": [int(value) for value in lap_cutoffs],
        "horizon_laps": int(args.horizon_laps),
        "mc_samples": int(args.mc_samples),
        "pit_window_laps": int(max(1, int(args.pit_window_laps))),
        "common_random_numbers": True,
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
        "observability": {
            "available_blocks": sorted([name for name, value in observability_artifacts.items() if value]),
            "artifact_count": int(sum(1 for value in observability_artifacts.values() if value)),
            "artifacts": observability_artifacts,
        },
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
            "observability_dir": str(observability_dir),
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
    print(f"Observability dir: {observability_dir}")
    print(
        "Observability artifacts written: "
        f"{int(sum(1 for value in observability_artifacts.values() if value))}"
    )

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
