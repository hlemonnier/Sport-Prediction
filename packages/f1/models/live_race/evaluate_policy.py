"""Validation harness for live race strategy policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Protocol

import numpy as np
import pandas as pd

from packages.sports_core.paths import find_repo_root

from packages.f1.models.live_race.action_space import StrategyAction
from packages.f1.models.live_race.environment import (
    StrategyState,
    StrategyTransition,
    assert_replay_prefix_invariant,
)

if TYPE_CHECKING:
    from packages.f1.models.live_race.planner import DeterministicStrategyPlanner


class PolicyLike(Protocol):
    def select_action(self, state: StrategyState) -> StrategyAction:
        ...


@dataclass(frozen=True)
class PolicyEvaluationConfig:
    report_dir: Optional[Path] = None
    write_report: bool = False
    fail_closed_on_missing_metrics: bool = True
    replay_invariance_cutoff_lap: Optional[int] = None


@dataclass(frozen=True)
class PolicyEvaluationResult:
    metrics: dict[str, object]
    action_rows: list[dict[str, object]] = field(default_factory=list)
    transition_errors: list[dict[str, object]] = field(default_factory=list)
    report_path: Optional[str] = None

    def to_payload(self) -> dict[str, object]:
        return {
            "metrics": self.metrics,
            "action_rows": self.action_rows,
            "transition_errors": self.transition_errors,
            "report_path": self.report_path,
        }


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_report_dir() -> Path:
    return find_repo_root(__file__) / "artifacts" / "reports" / "f1"


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _select_action(
    policy: PolicyLike | "DeterministicStrategyPlanner" | Callable[[StrategyState], StrategyAction] | None,
    state: StrategyState,
    fallback: StrategyAction,
) -> tuple[StrategyAction, Optional[str]]:
    if policy is None:
        return fallback, None
    try:
        if hasattr(policy, "select_action"):
            action = policy.select_action(state)  # type: ignore[union-attr]
        elif hasattr(policy, "plan"):
            action = policy.plan(state).action  # type: ignore[union-attr]
        else:
            action = policy(state)  # type: ignore[misc]
        if not isinstance(action, StrategyAction):
            action = StrategyAction.from_key(action)
        return action, None
    except Exception as exc:
        return fallback, f"{type(exc).__name__}: {exc}"


def _action_distribution(actions: list[StrategyAction]) -> dict[str, object]:
    if not actions:
        return {"counts": {}, "proportions": {}, "by_type": {}}
    counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    for action in actions:
        counts[action.key] = int(counts.get(action.key, 0) + 1)
        type_counts[action.action_type] = int(type_counts.get(action.action_type, 0) + 1)
    total = float(len(actions))
    return {
        "counts": counts,
        "proportions": {key: float(value / total) for key, value in counts.items()},
        "by_type": {key: int(value) for key, value in type_counts.items()},
    }


def _transition_consistency(
    transitions: list[StrategyTransition],
) -> tuple[int, list[dict[str, object]]]:
    errors: list[dict[str, object]] = []
    for idx, transition in enumerate(transitions):
        validation_errors = transition.validate()
        if not transition.legal_action_mask.legal_actions:
            validation_errors.append("no_legal_actions")
        if validation_errors:
            errors.append(
                {
                    "index": int(idx),
                    "driver_id": transition.state_t.driver_id,
                    "lap_number": int(transition.state_t.lap_number),
                    "errors": validation_errors,
                }
            )
    return len(errors), errors


def _replay_invariance_metric(
    transitions: list[StrategyTransition],
    comparison_transitions: Optional[list[StrategyTransition]],
    cutoff_lap: Optional[int],
) -> dict[str, object]:
    metadata_ok = all(
        int(transition.state_t.metadata.get("available_through_lap", transition.state_t.lap_number))
        <= int(transition.state_t.lap_number)
        for transition in transitions
    )
    if comparison_transitions is None:
        return {
            "available": False,
            "metadata_available_through_lap_ok": bool(metadata_ok),
            "prefix_invariant": None,
            "cutoff_lap": cutoff_lap,
        }
    if cutoff_lap is None:
        left_laps = [int(item.state_t.lap_number) for item in transitions]
        right_laps = [int(item.state_t.lap_number) for item in comparison_transitions]
        cutoff_lap = min(max(left_laps or [0]), max(right_laps or [0]))
    invariant = assert_replay_prefix_invariant(transitions, comparison_transitions, cutoff_lap=int(cutoff_lap))
    return {
        "available": True,
        "metadata_available_through_lap_ok": bool(metadata_ok),
        "prefix_invariant": bool(invariant),
        "cutoff_lap": int(cutoff_lap),
    }


def evaluate_strategy_policy(
    transitions: Iterable[StrategyTransition],
    *,
    policy: PolicyLike | "DeterministicStrategyPlanner" | Callable[[StrategyState], StrategyAction] | None = None,
    oracle_planner: Optional["DeterministicStrategyPlanner"] = None,
    comparison_transitions: Optional[Iterable[StrategyTransition]] = None,
    config: PolicyEvaluationConfig | None = None,
) -> PolicyEvaluationResult:
    """Evaluate a policy against replay transitions and optional DP oracle."""

    cfg = config or PolicyEvaluationConfig()
    transition_list = list(transitions)
    comparison_list = list(comparison_transitions) if comparison_transitions is not None else None
    rows: list[dict[str, object]] = []
    selected_actions: list[StrategyAction] = []
    illegal_count = 0
    policy_errors = 0
    realized_values: list[float] = []
    planner_values: list[float] = []
    oracle_values: list[float] = []
    regrets: list[float] = []

    for idx, transition in enumerate(transition_list):
        action, error = _select_action(policy, transition.state_t, transition.action_t)
        selected_actions.append(action)
        legal = transition.legal_action_mask.is_legal(action)
        if not legal:
            illegal_count += 1
        if error is not None:
            policy_errors += 1

        realized_value = float(transition.reward_t.value) if action == transition.action_t else float("nan")
        if np.isfinite(realized_value):
            realized_values.append(realized_value)

        planner_value = float("nan")
        oracle_value = float("nan")
        regret = float("nan")
        if oracle_planner is not None:
            try:
                planner_value = float(oracle_planner.value_action(transition.state_t, action))
                oracle_result = oracle_planner.plan(transition.state_t)
                oracle_value = float(oracle_result.value)
                regret = max(0.0, float(oracle_value - planner_value))
                planner_values.append(planner_value)
                oracle_values.append(oracle_value)
                regrets.append(regret)
            except Exception:
                planner_value = float("nan")

        rows.append(
            {
                "index": int(idx),
                "driver_id": transition.state_t.driver_id,
                "lap_number": int(transition.state_t.lap_number),
                "selected_action": action.key,
                "replay_action": transition.action_t.key,
                "is_legal": bool(legal),
                "illegal_reason": None if legal else transition.legal_action_mask.reason_for(action),
                "policy_error": error,
                "realized_replay_value": realized_value,
                "planner_policy_value": planner_value,
                "oracle_value": oracle_value,
                "regret_vs_oracle": regret,
            }
        )

    consistency_error_count, transition_errors = _transition_consistency(transition_list)
    invariance = _replay_invariance_metric(
        transition_list,
        comparison_list,
        cfg.replay_invariance_cutoff_lap,
    )
    total = len(transition_list)
    action_dist = _action_distribution(selected_actions)

    missing_metrics: list[str] = []
    if total == 0:
        missing_metrics.append("policy_value")
        missing_metrics.append("illegal_action_rate")
    if policy is None:
        missing_metrics.append("candidate_policy")
    if policy_errors > 0:
        missing_metrics.append("policy_execution_errors")
    if oracle_planner is None:
        missing_metrics.append("counterfactual_policy_value")
        missing_metrics.append("regret_vs_oracle")
    elif len(planner_values) != total or len(regrets) != total:
        missing_metrics.append("counterfactual_policy_value")
        missing_metrics.append("regret_vs_oracle")
    if not bool(invariance.get("available", False)):
        missing_metrics.append("replay_prefix_invariance")
    missing_metrics = list(dict.fromkeys(missing_metrics))
    if cfg.fail_closed_on_missing_metrics and missing_metrics:
        promotion_gate_pass = False
    else:
        promotion_gate_pass = bool(
            total > 0
            and policy is not None
            and policy_errors == 0
            and illegal_count == 0
            and consistency_error_count == 0
            and oracle_planner is not None
            and len(planner_values) == total
            and len(regrets) == total
            and bool(invariance.get("metadata_available_through_lap_ok", False))
            and bool(invariance.get("available", False))
            and invariance.get("prefix_invariant") is True
        )

    metrics: dict[str, object] = {
        "available": bool(total > 0),
        "rows": int(total),
        "policy_value": {
            "realized_replay_mean": _mean_or_none(realized_values),
            "planner_estimated_mean": _mean_or_none(planner_values),
            "method": "realized_when_policy_matches_replay_plus_optional_dp_counterfactual",
        },
        "illegal_action_rate": float(illegal_count / total) if total else None,
        "illegal_action_count": int(illegal_count),
        "policy_error_count": int(policy_errors),
        "counterfactual_value_coverage": float(len(planner_values) / total) if total else None,
        "action_distribution": action_dist,
        "transition_consistency": {
            "error_count": int(consistency_error_count),
            "ok": bool(consistency_error_count == 0),
        },
        "no_leakage_replay_invariance": invariance,
        "regret_vs_oracle": {
            "available": bool(oracle_planner is not None and regrets),
            "mean": _mean_or_none(regrets),
            "max": _max_or_none(regrets),
            "oracle_value_mean": _mean_or_none(oracle_values),
            "policy_value_mean": _mean_or_none(planner_values),
        },
        "missing_metrics": missing_metrics,
        "promotion_gate_pass": bool(promotion_gate_pass),
    }

    report_path: Optional[str] = None
    result = PolicyEvaluationResult(metrics=metrics, action_rows=rows, transition_errors=transition_errors)
    if cfg.write_report:
        output = write_policy_evaluation_report(result, report_dir=cfg.report_dir)
        report_path = str(output)
        result = PolicyEvaluationResult(
            metrics=metrics,
            action_rows=rows,
            transition_errors=transition_errors,
            report_path=report_path,
        )
    return result


def _mean_or_none(values: Iterable[float]) -> Optional[float]:
    arr = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=float)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _max_or_none(values: Iterable[float]) -> Optional[float]:
    arr = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=float)
    if arr.size == 0:
        return None
    return float(np.max(arr))


def write_policy_evaluation_report(
    result: PolicyEvaluationResult,
    *,
    report_dir: Optional[Path] = None,
    filename: Optional[str] = None,
) -> Path:
    output_dir = report_dir or _default_report_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (filename or f"live_strategy_policy_eval_{_utc_stamp()}.json")
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result.to_payload(), handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    return output


__all__ = [
    "PolicyEvaluationConfig",
    "PolicyEvaluationResult",
    "evaluate_strategy_policy",
    "write_policy_evaluation_report",
]
