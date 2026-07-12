"""Conservative tabular offline RL for live F1 race strategy.

This is deliberately small-data and fail-closed.  It can learn bounded Q-values
from replay/synthetic transitions, but it does not claim strategy improvement
unless evaluated through a supplied locked simulator/evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from packages.f1.models.live_race.action_space import ACTION_STAY_OUT, StrategyAction
from packages.f1.models.live_race.environment import StrategyState
from packages.f1.models.live_race.rl.behavior_cloning import MaskedBehaviorCloningPolicy
from packages.f1.models.live_race.rl.replay_buffer import (
    RLReplayDataset,
    StrategyActionIndex,
    bucket_state_features,
    state_to_feature_vector,
)


@dataclass(frozen=True)
class ConservativeOfflineRLConfig:
    """Config for deterministic conservative fitted Q-learning."""

    bucket_precision: int = 2
    iterations: int = 60
    learning_rate: float = 0.30
    discount: float = 0.92
    conservative_penalty: float = 0.40
    ood_action_penalty: float = 8.0
    reward_clip: tuple[float, float] = (-500.0, 100.0)
    value_clip: tuple[float, float] = (-800.0, 200.0)
    fallback_action_key: str = f"{ACTION_STAY_OUT}:conservative"


@dataclass
class ConservativeOfflineRLPolicy:
    """Legal-mask-aware conservative Q policy."""

    action_index: StrategyActionIndex
    q_table: dict[tuple[float, ...], np.ndarray]
    state_action_counts: dict[tuple[float, ...], np.ndarray]
    global_action_values: np.ndarray
    config: ConservativeOfflineRLConfig = field(default_factory=ConservativeOfflineRLConfig)
    behavior_policy: Optional[MaskedBehaviorCloningPolicy] = None
    training_diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def diagnostics(self) -> dict[str, object]:
        return {
            **self.training_diagnostics,
            "policy_family": "conservative_tabular_offline_q",
            "legal_mask_aware": True,
            "historical_accuracy_is_not_policy_value": True,
            "locked_simulator_evaluation_required": True,
            "promotion_gate_pass": False,
        }

    def _bucket_for(self, state_features: np.ndarray) -> tuple[float, ...]:
        return bucket_state_features(state_features, precision=int(self.config.bucket_precision))

    def action_values_for_features(self, state_features: np.ndarray) -> np.ndarray:
        key = self._bucket_for(np.asarray(state_features, dtype=float))
        values = self.q_table.get(key)
        if values is None:
            values = np.asarray(self.global_action_values, dtype=float)
        return np.clip(values, float(self.config.value_clip[0]), float(self.config.value_clip[1]))

    def score_actions(self, state_features: np.ndarray, legal_action_mask: np.ndarray) -> np.ndarray:
        legal = np.asarray(legal_action_mask, dtype=bool)
        if legal.size != self.action_index.size:
            raise ValueError("legal mask length does not match action index")
        key = self._bucket_for(np.asarray(state_features, dtype=float))
        values = self.action_values_for_features(state_features).copy()
        counts = self.state_action_counts.get(key)
        if counts is None:
            counts = np.zeros(self.action_index.size, dtype=float)
        unseen_legal = legal & (counts <= 0.0)
        values[unseen_legal] -= float(self.config.ood_action_penalty)
        values = np.clip(values, float(self.config.value_clip[0]), float(self.config.value_clip[1]))
        values[~legal] = -np.inf
        return values

    def predict_index(self, state_features: np.ndarray, legal_action_mask: np.ndarray) -> int:
        legal = np.asarray(legal_action_mask, dtype=bool)
        scores = self.score_actions(state_features, legal)
        if np.isfinite(scores).any():
            return int(np.argmax(scores))
        if self.behavior_policy is not None:
            return self.behavior_policy.predict_index(state_features, legal)
        fallback = self.action_index.index_for(self.config.fallback_action_key, default=0)
        if 0 <= fallback < legal.size and bool(legal[fallback]):
            return int(fallback)
        legal_indices = np.flatnonzero(legal)
        return int(legal_indices[0]) if legal_indices.size else 0

    def select_action(self, state: StrategyState) -> StrategyAction:
        features = state_to_feature_vector(state)
        legal = self.action_index.legal_mask_for_state(state)
        return self.action_index.action_for(self.predict_index(features, legal))

    def value(self, state: StrategyState, action: StrategyAction | None = None) -> float:
        features = state_to_feature_vector(state)
        legal = self.action_index.legal_mask_for_state(state)
        if action is not None:
            idx = self.action_index.index_for(action)
            if idx < 0 or idx >= self.action_index.size:
                return float(self.config.value_clip[0])
            return float(self.action_values_for_features(features)[idx])
        scores = self.score_actions(features, legal)
        finite = scores[np.isfinite(scores)]
        if finite.size == 0:
            return float(self.config.value_clip[0])
        return float(np.max(finite))


@dataclass(frozen=True)
class OfflineRLEvaluationResult:
    """Evaluation payload for locked simulator/offline policy comparisons."""

    available: bool
    metrics: dict[str, object]
    comparison_payloads: dict[str, object] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        return {
            "available": bool(self.available),
            "metrics": self.metrics,
            "comparison_payloads": self.comparison_payloads,
            "diagnostics": self.diagnostics,
        }


def fit_conservative_offline_q(
    dataset: RLReplayDataset,
    *,
    config: ConservativeOfflineRLConfig | None = None,
    behavior_policy: Optional[MaskedBehaviorCloningPolicy] = None,
) -> ConservativeOfflineRLPolicy:
    """Fit bounded conservative fitted Q-iteration from replay transitions."""

    cfg = config or ConservativeOfflineRLConfig()
    examples = dataset.learning_examples()
    action_count = dataset.action_index.size
    reward_low, reward_high = float(cfg.reward_clip[0]), float(cfg.reward_clip[1])
    value_low, value_high = float(cfg.value_clip[0]), float(cfg.value_clip[1])

    observed_rewards = np.asarray(
        [np.clip(example.reward, reward_low, reward_high) for example in examples],
        dtype=float,
    )
    if observed_rewards.size:
        pessimistic_default = float(
            np.clip(np.min(observed_rewards) - abs(cfg.ood_action_penalty), value_low, value_high)
        )
    else:
        pessimistic_default = value_low

    global_sums = np.zeros(action_count, dtype=float)
    global_counts = np.zeros(action_count, dtype=float)
    for example in examples:
        idx = int(example.action_index)
        global_sums[idx] += float(np.clip(example.reward, reward_low, reward_high))
        global_counts[idx] += 1.0
    global_values = np.full(action_count, pessimistic_default, dtype=float)
    observed = global_counts > 0.0
    global_values[observed] = global_sums[observed] / global_counts[observed]
    global_values = np.clip(global_values, value_low, value_high)

    q_table: dict[tuple[float, ...], np.ndarray] = {}
    state_action_counts: dict[tuple[float, ...], np.ndarray] = {}
    for example in examples:
        key = bucket_state_features(example.state_features, precision=int(cfg.bucket_precision))
        next_key = bucket_state_features(example.next_state_features, precision=int(cfg.bucket_precision))
        q_table.setdefault(key, global_values.copy())
        q_table.setdefault(next_key, global_values.copy())
        state_action_counts.setdefault(key, np.zeros(action_count, dtype=float))[int(example.action_index)] += 1.0
        state_action_counts.setdefault(next_key, np.zeros(action_count, dtype=float))

    learning_rate = float(np.clip(cfg.learning_rate, 0.0, 1.0))
    discount = float(np.clip(cfg.discount, 0.0, 1.0))
    for _ in range(max(0, int(cfg.iterations))):
        for example in examples:
            key = bucket_state_features(example.state_features, precision=int(cfg.bucket_precision))
            next_key = bucket_state_features(example.next_state_features, precision=int(cfg.bucket_precision))
            action_idx = int(example.action_index)
            q_values = q_table.setdefault(key, global_values.copy())
            next_values = q_table.setdefault(next_key, global_values.copy())

            if example.done or not example.next_legal_action_mask.any():
                next_value = 0.0
            else:
                next_counts = state_action_counts.get(next_key, np.zeros(action_count, dtype=float))
                next_scores = np.asarray(next_values, dtype=float).copy()
                next_scores[example.next_legal_action_mask & (next_counts <= 0.0)] -= float(cfg.ood_action_penalty)
                next_scores[~example.next_legal_action_mask] = -np.inf
                finite = next_scores[np.isfinite(next_scores)]
                next_value = float(np.max(finite)) if finite.size else value_low

            reward = float(np.clip(example.reward, reward_low, reward_high))
            target = float(np.clip(reward + (discount * next_value), value_low, value_high))
            q_values[action_idx] = float(q_values[action_idx] + (learning_rate * (target - q_values[action_idx])))

            legal_indices = np.flatnonzero(example.legal_action_mask)
            for idx in legal_indices:
                if int(idx) == action_idx:
                    continue
                count = state_action_counts.get(key, np.zeros(action_count, dtype=float))[int(idx)]
                penalty = float(cfg.conservative_penalty) / float(np.sqrt(count + 1.0))
                q_values[int(idx)] -= learning_rate * penalty

            q_table[key] = np.clip(q_values, value_low, value_high)

    all_values = np.concatenate([values for values in q_table.values()]) if q_table else global_values
    return ConservativeOfflineRLPolicy(
        action_index=dataset.action_index,
        q_table=q_table,
        state_action_counts=state_action_counts,
        global_action_values=global_values,
        config=cfg,
        behavior_policy=behavior_policy,
        training_diagnostics={
            "training_rows": int(len(examples)),
            "excluded_ood_rows": int(dataset.rows - len(examples)),
            "state_buckets": int(len(q_table)),
            "iterations": int(max(0, cfg.iterations)),
            "discount": float(discount),
            "conservative_penalty": float(cfg.conservative_penalty),
            "ood_action_penalty": float(cfg.ood_action_penalty),
            "reward_clip": (reward_low, reward_high),
            "value_clip": (value_low, value_high),
            "value_min": float(np.min(all_values)) if all_values.size else None,
            "value_max": float(np.max(all_values)) if all_values.size else None,
            "warm_started_from_behavior_cloning": bool(behavior_policy is not None),
        },
    )


def evaluate_offline_rl_policy(
    policy: ConservativeOfflineRLPolicy,
    *,
    simulator: object | None = None,
    locked_evaluator: Optional[Callable[..., object]] = None,
    behavior_cloning_policy: object | None = None,
    dp_mpc_policy: object | None = None,
    evaluator_kwargs: Optional[dict[str, object]] = None,
) -> OfflineRLEvaluationResult:
    """Evaluate offline RL only through a supplied locked simulator setting."""

    if simulator is None:
        return _unavailable_result(
            "locked_simulator_required",
            "offline RL is not evaluated from historical action accuracy alone",
        )

    evaluator = locked_evaluator or _simulator_evaluator(simulator)
    if evaluator is None:
        return _unavailable_result(
            "locked_policy_evaluator_unavailable",
            "simulator did not expose evaluate_policy/run_policy and no locked_evaluator was supplied",
        )

    kwargs = dict(evaluator_kwargs or {})
    comparison_payloads: dict[str, object] = {}
    offline_payload = _normalise_payload(_call_evaluator(evaluator, simulator, policy, kwargs))
    comparison_payloads["offline_rl"] = offline_payload

    if behavior_cloning_policy is not None:
        comparison_payloads["behavior_cloning"] = _normalise_payload(
            _call_evaluator(evaluator, simulator, behavior_cloning_policy, kwargs)
        )
    if dp_mpc_policy is not None:
        comparison_payloads["dp_mpc"] = _normalise_payload(_call_evaluator(evaluator, simulator, dp_mpc_policy, kwargs))

    offline_value = _extract_policy_value(offline_payload)
    deltas: dict[str, Optional[float]] = {}
    comparison_values: dict[str, Optional[float]] = {}
    for key in ("behavior_cloning", "dp_mpc"):
        if key in comparison_payloads:
            value = _extract_policy_value(comparison_payloads[key])
            comparison_values[key] = value
            deltas[f"policy_value_delta_vs_{key}"] = None if offline_value is None or value is None else float(
                offline_value - value
            )
    required_comparisons = ("behavior_cloning", "dp_mpc")
    missing_comparisons = tuple(key for key in required_comparisons if key not in comparison_payloads)
    comparison_gate_pass = bool(
        offline_value is not None
        and not missing_comparisons
        and all(value is not None and float(offline_value) > float(value) for value in comparison_values.values())
    )
    locked_evidence_issues = _locked_evaluation_issues(offline_payload)
    offline_payload_gate_pass = bool(
        _payload_gate(offline_payload)
        and offline_value is not None
        and not locked_evidence_issues
    )

    return OfflineRLEvaluationResult(
        available=True,
        metrics={
            "available": True,
            "evaluation_setting": "locked_simulator",
            "policy_value": offline_value,
            "comparison_deltas": deltas,
            "comparison_values": comparison_values,
            "comparison_gate_pass": bool(comparison_gate_pass),
            "missing_comparisons_for_promotion": list(missing_comparisons),
            "locked_evaluation_issues": locked_evidence_issues,
            "historical_accuracy_used_for_promotion": False,
            "promotion_gate_pass": bool(offline_payload_gate_pass and comparison_gate_pass),
        },
        comparison_payloads=comparison_payloads,
        diagnostics={
            **policy.diagnostics,
            "locked_simulator_supplied": True,
            "comparison_keys": sorted(comparison_payloads.keys()),
            "offline_payload_gate_pass": bool(offline_payload_gate_pass),
        },
    )


def _unavailable_result(reason: str, note: str) -> OfflineRLEvaluationResult:
    return OfflineRLEvaluationResult(
        available=False,
        metrics={
            "available": False,
            "policy_value": None,
            "missing_metrics": ["locked_simulator_policy_value"],
            "reason": reason,
            "promotion_gate_pass": False,
        },
        diagnostics={
            "locked_simulator_evaluation_required": True,
            "historical_accuracy_used_for_promotion": False,
            "note": note,
        },
    )


def _simulator_evaluator(simulator: object) -> Optional[Callable[..., object]]:
    if hasattr(simulator, "evaluate_policy"):
        return getattr(simulator, "evaluate_policy")
    if hasattr(simulator, "run_policy"):
        return getattr(simulator, "run_policy")
    return None


def _call_evaluator(
    evaluator: Callable[..., object],
    simulator: object,
    policy: object,
    kwargs: dict[str, object],
) -> object:
    try:
        return evaluator(policy=policy, **kwargs)
    except TypeError:
        try:
            return evaluator(policy, **kwargs)
        except TypeError:
            return evaluator(simulator, policy, **kwargs)


def _normalise_payload(payload: object) -> dict[str, object]:
    if hasattr(payload, "to_payload"):
        payload = payload.to_payload()  # type: ignore[assignment, union-attr]
    if isinstance(payload, dict):
        return dict(payload)
    return {"result": payload}


def _extract_policy_value(payload: dict[str, object]) -> Optional[float]:
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        for key in ("mean_return", "policy_value", "simulator_value", "value"):
            value = metrics.get(key)
            numeric = _numeric_or_none(value)
            if numeric is not None:
                return numeric
            if isinstance(value, dict):
                for nested in ("simulator_mean", "mean", "planner_estimated_mean", "policy_value_mean"):
                    numeric = _numeric_or_none(value.get(nested))
                    if numeric is not None:
                        return numeric
    for key in ("mean_return", "policy_value", "simulator_value", "value"):
        numeric = _numeric_or_none(payload.get(key))
        if numeric is not None:
            return numeric
    return None


def _numeric_or_none(value: object) -> Optional[float]:
    try:
        numeric = float(value)  # type: ignore[arg-type]
        if np.isfinite(numeric):
            return numeric
    except Exception:
        return None
    return None


def _payload_gate(payload: dict[str, object]) -> bool:
    metrics = payload.get("metrics")
    if isinstance(metrics, dict) and "promotion_gate_pass" in metrics:
        return bool(metrics.get("promotion_gate_pass"))
    if "promotion_gate_pass" in payload:
        return bool(payload.get("promotion_gate_pass"))
    return False


def _locked_evaluation_issues(payload: dict[str, object]) -> list[str]:
    """Require provenance and safety evidence, not a self-declared pass bit."""

    metrics = payload.get("metrics")
    evidence = metrics if isinstance(metrics, dict) else payload
    issues: list[str] = []
    if evidence.get("evaluation_locked") is not True:
        issues.append("evaluation_not_locked")
    for key in ("simulator_artifact_hash", "calibration_report_hash"):
        value = str(evidence.get(key) or "").strip().lower()
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            issues.append(f"{key}_missing_or_invalid")
    if not str(evidence.get("evaluation_split_id") or "").strip():
        issues.append("evaluation_split_id_missing")
    if evidence.get("causal_event_time_replay") is not True:
        issues.append("causal_event_time_replay_unproven")
    if _numeric_or_none(evidence.get("illegal_action_rate")) != 0.0:
        issues.append("illegal_action_rate_nonzero_or_missing")
    policy_errors = _numeric_or_none(evidence.get("policy_error_count"))
    if policy_errors != 0.0:
        issues.append("policy_error_count_nonzero_or_missing")
    support = _numeric_or_none(evidence.get("support_coverage"))
    if support is None or support < 0.90:
        issues.append("offline_support_coverage_below_0_90_or_missing")
    return issues


__all__ = [
    "ConservativeOfflineRLConfig",
    "ConservativeOfflineRLPolicy",
    "OfflineRLEvaluationResult",
    "evaluate_offline_rl_policy",
    "fit_conservative_offline_q",
]
