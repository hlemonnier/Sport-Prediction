"""Calibration helpers for the Phase 5 live-race simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from packages.sports_core.paths import find_repo_root

from packages.f1.models.live_race.environment import StrategyState, infer_observed_action
from packages.f1.models.live_race.simulator import LiveRaceSimulator, RaceSimulatorConfig


@dataclass(frozen=True)
class SimulatorCalibrationResult:
    metrics: dict[str, object]
    rows: list[dict[str, object]] = field(default_factory=list)
    report_path: Optional[str] = None

    def to_payload(self) -> dict[str, object]:
        return {"metrics": self.metrics, "rows": self.rows, "report_path": self.report_path}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_report_dir() -> Path:
    return find_repo_root(__file__) / "artifacts" / "reports" / "f1"


def _finite(value: object, default: float = float("nan")) -> float:
    try:
        if value is None or not np.isfinite(float(value)):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _actual_elapsed(row_t: pd.Series, row_t1: pd.Series) -> float:
    race_t = _finite(row_t.get("race_time_seconds"), float("nan"))
    race_t1 = _finite(row_t1.get("race_time_seconds"), float("nan"))
    if np.isfinite(race_t) and np.isfinite(race_t1) and race_t1 >= race_t:
        return float(race_t1 - race_t)
    lap_time = _finite(row_t1.get("lap_time_seconds"), float("nan"))
    if np.isfinite(lap_time):
        return float(lap_time)
    lap_time = _finite(row_t.get("next_actual_lap_time_seconds"), float("nan"))
    return float(lap_time)


def one_step_lap_time_calibration(
    laps: pd.DataFrame,
    *,
    simulator: Optional[LiveRaceSimulator] = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not isinstance(laps, pd.DataFrame):
        raise TypeError("laps must be a pandas DataFrame")
    if laps.empty:
        return {"available": False, "reason": "empty_laps", "rows": 0}, []

    sim = simulator or LiveRaceSimulator(config=RaceSimulatorConfig())
    frame = laps.copy()
    lap_col = "lap_number" if "lap_number" in frame.columns else "LapNumber"
    driver_col = "driver_id" if "driver_id" in frame.columns else None
    sort_cols = [col for col in (driver_col, lap_col, "timestamp") if col and col in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols, kind="mergesort")

    rows: list[dict[str, object]] = []
    groups: Iterable[tuple[object, pd.DataFrame]]
    groups = frame.groupby(driver_col, sort=False) if driver_col else [(None, frame)]
    for _, group in groups:
        row_list = [row for _, row in group.iterrows()]
        for idx in range(len(row_list) - 1):
            row_t = row_list[idx]
            row_t1 = row_list[idx + 1]
            actual = _actual_elapsed(row_t, row_t1)
            if not np.isfinite(actual):
                continue
            state = StrategyState.from_mapping(row_t)
            action = infer_observed_action(row_t, row_t1)
            transition = sim.step(state, action)
            predicted = _finite(transition.reward_t.components.get("race_time_delta_seconds"), float("nan"))
            if not np.isfinite(predicted):
                continue
            rows.append(
                {
                    "driver_id": state.driver_id,
                    "lap_number": int(state.lap_number),
                    "action_key": action.key,
                    "actual_elapsed_seconds": float(actual),
                    "predicted_elapsed_seconds": float(predicted),
                    "error_seconds": float(predicted - actual),
                    "track_status": state.track_status,
                    "is_pit_action": bool(action.is_pit_action),
                    "predicted_pit_loss_seconds": transition.reward_t.components.get("pit_loss"),
                }
            )

    if not rows:
        return {"available": False, "reason": "no_comparable_one_step_rows", "rows": 0}, rows
    errors = np.asarray([float(row["error_seconds"]) for row in rows], dtype=float)
    abs_errors = np.abs(errors)
    return (
        {
            "available": True,
            "rows": int(len(rows)),
            "mae": float(np.mean(abs_errors)),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "crps_deterministic_point_forecast": float(np.mean(abs_errors)),
            "bias_seconds": float(np.mean(errors)),
        },
        rows,
    )


def pit_loss_calibration(rows: Iterable[dict[str, object]], laps: pd.DataFrame) -> dict[str, object]:
    explicit_columns = [column for column in ("pit_loss_seconds", "pit_loss_estimate_seconds", "observed_pit_loss_seconds") if column in laps.columns]
    if not explicit_columns:
        predicted = [float(row["predicted_pit_loss_seconds"]) for row in rows if row.get("is_pit_action") and np.isfinite(_finite(row.get("predicted_pit_loss_seconds")))]
        return {
            "available": False,
            "reason": "no_explicit_observed_pit_loss_column",
            "pit_action_rows": int(len(predicted)),
            "predicted_pit_loss_mean": float(np.mean(predicted)) if predicted else None,
        }
    observed = pd.to_numeric(laps[explicit_columns[0]], errors="coerce").dropna().to_numpy(dtype=float)
    predicted = np.asarray(
        [
            float(row["predicted_pit_loss_seconds"])
            for row in rows
            if row.get("is_pit_action") and np.isfinite(_finite(row.get("predicted_pit_loss_seconds")))
        ],
        dtype=float,
    )
    n = min(int(observed.size), int(predicted.size))
    if n == 0:
        return {"available": False, "reason": "no_matched_pit_loss_rows", "rows": 0}
    err = predicted[:n] - observed[:n]
    return {
        "available": True,
        "rows": int(n),
        "mae": float(np.mean(np.abs(err))),
        "bias_seconds": float(np.mean(err)),
        "observed_mean": float(np.mean(observed[:n])),
        "predicted_mean": float(np.mean(predicted[:n])),
    }


def track_status_calibration(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    frame = pd.DataFrame(list(rows))
    if frame.empty or "track_status" not in frame.columns:
        return {"available": False, "reason": "no_one_step_rows", "groups": {}}
    groups: dict[str, object] = {}
    for status, group in frame.groupby("track_status", dropna=False, sort=True):
        errors = pd.to_numeric(group["error_seconds"], errors="coerce").dropna().to_numpy(dtype=float)
        if errors.size == 0:
            continue
        groups[str(status)] = {
            "rows": int(errors.size),
            "mae": float(np.mean(np.abs(errors))),
            "bias_seconds": float(np.mean(errors)),
        }
    return {"available": bool(groups), "groups": groups}


def final_order_proxy_metrics(
    laps: pd.DataFrame,
    *,
    simulator: Optional[LiveRaceSimulator] = None,
) -> dict[str, object]:
    if not isinstance(laps, pd.DataFrame) or laps.empty or "driver_id" not in laps.columns:
        return {"available": False, "reason": "requires_multi_driver_laps", "proxy_only": True}
    actual_col = next(
        (column for column in ("final_position", "classified_position", "finish_position", "result_position") if column in laps.columns),
        None,
    )
    if actual_col is None:
        return {"available": False, "reason": "missing_actual_final_position", "proxy_only": True}

    sim = simulator or LiveRaceSimulator(config=RaceSimulatorConfig())
    finals: list[dict[str, object]] = []
    for driver_id, group in laps.groupby("driver_id", sort=False):
        transitions = sim.replay_race(group, driver_id=str(driver_id))
        if transitions:
            final_state = transitions[-1].state_t1
            predicted_time = _finite(final_state.race_time_seconds, float("nan"))
        else:
            predicted_time = float("nan")
        actual = pd.to_numeric(group[actual_col], errors="coerce").dropna()
        if np.isfinite(predicted_time) and not actual.empty:
            finals.append({"driver_id": str(driver_id), "predicted_time": predicted_time, "actual_position": float(actual.iloc[-1])})

    if len(finals) < 2:
        return {
            "available": False,
            "reason": "insufficient_multi_driver_final_rows",
            "rows": int(len(finals)),
            "proxy_only": True,
        }
    frame = pd.DataFrame(finals)
    frame["predicted_position_proxy"] = frame["predicted_time"].rank(method="first", ascending=True)
    spearman = frame[["predicted_position_proxy", "actual_position"]].corr(method="spearman").iloc[0, 1]
    return {
        "available": bool(np.isfinite(float(spearman))),
        "rows": int(len(frame)),
        "spearman_rank_corr": float(spearman) if np.isfinite(float(spearman)) else None,
        "proxy_only": True,
        "note": "final order is ranked by simulated single-car race-time proxy, not a counterfactual multi-car order",
    }


def build_simulator_calibration_report(
    laps: pd.DataFrame,
    *,
    simulator: Optional[LiveRaceSimulator] = None,
    write_report: bool = False,
    report_dir: Optional[Path] = None,
) -> SimulatorCalibrationResult:
    sim = simulator or LiveRaceSimulator(config=RaceSimulatorConfig())
    one_step_metrics, rows = one_step_lap_time_calibration(laps, simulator=sim)
    metrics = {
        "one_step_lap_time": one_step_metrics,
        "pit_loss": pit_loss_calibration(rows, laps),
        "track_status": track_status_calibration(rows),
        "final_order_proxy": final_order_proxy_metrics(laps, simulator=sim),
        "simulator_id": sim.model_id,
        "scenario_id": sim.scenario.scenario_id,
        "seed": int(sim.config.seed),
    }
    result = SimulatorCalibrationResult(metrics=metrics, rows=rows)
    if write_report:
        path = write_simulator_calibration_report(result, report_dir=report_dir)
        result = SimulatorCalibrationResult(metrics=metrics, rows=rows, report_path=str(path))
    return result


def write_simulator_calibration_report(
    result: SimulatorCalibrationResult,
    *,
    report_dir: Optional[Path] = None,
    filename: Optional[str] = None,
) -> Path:
    output_dir = report_dir or _default_report_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (filename or f"live_race_simulator_calibration_{_utc_stamp()}.json")
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result.to_payload(), handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return output


__all__ = [
    "SimulatorCalibrationResult",
    "build_simulator_calibration_report",
    "final_order_proxy_metrics",
    "one_step_lap_time_calibration",
    "pit_loss_calibration",
    "track_status_calibration",
    "write_simulator_calibration_report",
]
