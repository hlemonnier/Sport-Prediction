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

from packages.f1.models.live_race.action_space import ACTION_STAY_OUT, StrategyAction
from packages.f1.models.live_race.environment import build_replay_transitions
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


def _actual_elapsed(state_t: object, state_t1: object) -> float:
    race_t = _finite(getattr(state_t, "race_time_seconds", None), float("nan"))
    race_t1 = _finite(getattr(state_t1, "race_time_seconds", None), float("nan"))
    if np.isfinite(race_t) and np.isfinite(race_t1) and race_t1 >= race_t:
        return float(race_t1 - race_t)
    return float("nan")


def _observed_pit_loss(rows: Iterable[pd.Series]) -> float:
    """Return an explicitly observed pit loss aligned to this transition.

    ``pit_loss_estimate_seconds`` is deliberately excluded: it is a model input,
    not ground truth, and comparing the simulator against it would make the
    calibration circular.
    """

    for row in rows:
        for column in ("observed_pit_loss_seconds", "pit_loss_seconds"):
            value = _finite(row.get(column), float("nan"))
            if np.isfinite(value):
                return float(value)
    return float("nan")


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
    source_rows = {str(index): row for index, row in frame.iterrows()}
    rows: list[dict[str, object]] = []
    observed_transitions = build_replay_transitions(
        frame,
        action_space=sim.action_space,
        action_mask_config=sim.config.action_mask,
    )
    invalid_simulator_actions = 0
    missing_elapsed = 0
    for observed in observed_transitions:
        actual = _actual_elapsed(observed.state_t, observed.state_t1)
        if not np.isfinite(actual):
            missing_elapsed += 1
            continue
        elapsed_laps = max(1, int(observed.metadata.get("elapsed_laps", 1)))
        actions = [observed.action_t]
        actions.extend(
            StrategyAction(ACTION_STAY_OUT)
            for _ in range(elapsed_laps - 1)
        )
        simulated = sim.simulate_action_sequence(
            observed.state_t,
            actions,
            stop_on_done=True,
        )
        if len(simulated) != elapsed_laps or any(
            not transition.is_action_legal() for transition in simulated
        ):
            invalid_simulator_actions += 1
            continue
        predicted_parts = [
            _finite(
                transition.reward_t.components.get("race_time_delta_seconds"),
                float("nan"),
            )
            for transition in simulated
        ]
        if not predicted_parts or not np.isfinite(predicted_parts).all():
            continue
        predicted = float(np.sum(predicted_parts))
        source_ids = [
            observed.metadata.get("row_t_index"),
            observed.metadata.get("pit_in_row_index"),
            *tuple(observed.metadata.get("pit_out_row_indices", ())),
            observed.metadata.get("row_t1_index"),
        ]
        matched_source_rows = [
            source_rows[str(source_id)]
            for source_id in dict.fromkeys(source_ids)
            if source_id is not None and str(source_id) in source_rows
        ]
        predicted_pit_loss = float(
            np.sum(
                [
                    _finite(transition.reward_t.components.get("pit_loss"), 0.0)
                    for transition in simulated
                ]
            )
        )
        rows.append(
            {
                "driver_id": observed.state_t.driver_id,
                "lap_number": int(observed.state_t.lap_number),
                "next_lap_number": int(observed.state_t1.lap_number),
                "elapsed_laps": int(elapsed_laps),
                "transition_kind": observed.metadata.get("transition_kind"),
                "calibration_matching": "semi_markov_decision_transition",
                "action_key": observed.action_t.key,
                "actual_elapsed_seconds": float(actual),
                "predicted_elapsed_seconds": float(predicted),
                "error_seconds": float(predicted - actual),
                "track_status": observed.state_t.track_status,
                "is_pit_action": bool(observed.action_t.is_pit_action),
                "predicted_pit_loss_seconds": predicted_pit_loss,
                "observed_pit_loss_seconds": _observed_pit_loss(matched_source_rows),
            }
        )

    if not rows:
        return {
            "available": False,
            "reason": "no_comparable_semi_markov_rows",
            "rows": 0,
            "replay_transition_count": int(len(observed_transitions)),
            "invalid_simulator_action_count": int(invalid_simulator_actions),
            "missing_elapsed_count": int(missing_elapsed),
            "matching": "semi_markov_decision_transition",
        }, rows
    errors = np.asarray([float(row["error_seconds"]) for row in rows], dtype=float)
    abs_errors = np.abs(errors)
    return (
        {
            "available": True,
            "rows": int(len(rows)),
            "replay_transition_count": int(len(observed_transitions)),
            "invalid_simulator_action_count": int(invalid_simulator_actions),
            "missing_elapsed_count": int(missing_elapsed),
            "matching": "semi_markov_decision_transition",
            "mae": float(np.mean(abs_errors)),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "crps_deterministic_point_forecast": float(np.mean(abs_errors)),
            "bias_seconds": float(np.mean(errors)),
        },
        rows,
    )


def pit_loss_calibration(rows: Iterable[dict[str, object]], laps: pd.DataFrame) -> dict[str, object]:
    explicit_columns = [
        column
        for column in ("observed_pit_loss_seconds", "pit_loss_seconds")
        if column in laps.columns
    ]
    if not explicit_columns:
        predicted = [float(row["predicted_pit_loss_seconds"]) for row in rows if row.get("is_pit_action") and np.isfinite(_finite(row.get("predicted_pit_loss_seconds")))]
        return {
            "available": False,
            "reason": "no_explicit_observed_pit_loss_column",
            "pit_action_rows": int(len(predicted)),
            "predicted_pit_loss_mean": float(np.mean(predicted)) if predicted else None,
        }
    matched = [
        (
            float(row["predicted_pit_loss_seconds"]),
            float(row["observed_pit_loss_seconds"]),
        )
        for row in rows
        if row.get("is_pit_action")
        and np.isfinite(_finite(row.get("predicted_pit_loss_seconds")))
        and np.isfinite(_finite(row.get("observed_pit_loss_seconds")))
    ]
    if not matched:
        return {"available": False, "reason": "no_matched_pit_loss_rows", "rows": 0}
    predicted = np.asarray([pair[0] for pair in matched], dtype=float)
    observed = np.asarray([pair[1] for pair in matched], dtype=float)
    err = predicted - observed
    return {
        "available": True,
        "rows": int(len(matched)),
        "mae": float(np.mean(np.abs(err))),
        "bias_seconds": float(np.mean(err)),
        "observed_mean": float(np.mean(observed)),
        "predicted_mean": float(np.mean(predicted)),
        "matching": "transition_aligned_explicit_observed_pit_loss",
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
