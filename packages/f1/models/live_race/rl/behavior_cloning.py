"""Masked behavior cloning for live F1 strategy warm starts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from packages.f1.models.live_race.action_space import ACTION_STAY_OUT, StrategyAction
from packages.f1.models.live_race.environment import StrategyState
from packages.f1.models.live_race.rl.replay_buffer import (
    RLReplayDataset,
    RLReplayExample,
    StrategyActionIndex,
    bucket_state_features,
    state_to_feature_vector,
)


@dataclass(frozen=True)
class BehaviorCloningConfig:
    """Config for deterministic tabular behavior cloning."""

    bucket_precision: int = 2
    smoothing: float = 0.25
    min_bucket_count: int = 1
    fallback_action_key: str = f"{ACTION_STAY_OUT}:conservative"


@dataclass
class TrivialLegalActionBaseline:
    """Frequency baseline that always picks the most common legal replay action."""

    action_index: StrategyActionIndex
    action_counts: np.ndarray
    fallback_action_index: int = 0

    @classmethod
    def fit(cls, dataset: RLReplayDataset) -> "TrivialLegalActionBaseline":
        counts = np.zeros(dataset.action_index.size, dtype=float)
        for example in dataset.behavior_cloning_examples():
            targets = np.flatnonzero(example.behavior_cloning_action_mask)
            if targets.size:
                counts[targets] += float(example.weight or 1.0) / float(targets.size)
        fallback = dataset.action_index.index_for(f"{ACTION_STAY_OUT}:conservative", default=0)
        return cls(action_index=dataset.action_index, action_counts=counts, fallback_action_index=fallback)

    def predict_index(self, legal_action_mask: np.ndarray) -> int:
        legal = np.asarray(legal_action_mask, dtype=bool)
        if legal.size != self.action_index.size:
            raise ValueError("legal mask length does not match action index")
        scores = np.asarray(self.action_counts, dtype=float).copy()
        scores[~legal] = -np.inf
        if np.isfinite(scores).any():
            return int(np.argmax(scores))
        if 0 <= self.fallback_action_index < legal.size and bool(legal[self.fallback_action_index]):
            return int(self.fallback_action_index)
        legal_indices = np.flatnonzero(legal)
        return int(legal_indices[0]) if legal_indices.size else 0

    def select_action(self, state: StrategyState) -> StrategyAction:
        legal = self.action_index.legal_mask_for_state(state)
        if not legal.any():
            raise ValueError("no constraint-legal live-strategy action")
        return self.action_index.action_for(self.predict_index(legal))


@dataclass
class MaskedBehaviorCloningPolicy:
    """Legal-mask-aware tabular behavior cloning policy.

    This is intentionally labeled as a warm start.  Historical action matching
    is not a strategy-optimizer promotion signal.
    """

    action_index: StrategyActionIndex
    global_action_counts: np.ndarray
    bucket_action_counts: dict[tuple[float, ...], np.ndarray]
    config: BehaviorCloningConfig = field(default_factory=BehaviorCloningConfig)
    training_diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def diagnostics(self) -> dict[str, object]:
        return {
            **self.training_diagnostics,
            "policy_family": "masked_behavior_cloning",
            "role": "warm_start_only",
            "not_promoted_strategy_optimizer": True,
            "promotion_gate_pass": False,
        }

    def _bucket_for(self, state_features: np.ndarray) -> tuple[float, ...]:
        return bucket_state_features(state_features, precision=int(self.config.bucket_precision))

    def score_actions(self, state_features: np.ndarray, legal_action_mask: np.ndarray) -> np.ndarray:
        legal = np.asarray(legal_action_mask, dtype=bool)
        if legal.size != self.action_index.size:
            raise ValueError("legal mask length does not match action index")

        bucket = self._bucket_for(np.asarray(state_features, dtype=float))
        counts = self.bucket_action_counts.get(bucket)
        if counts is None or float(np.sum(counts)) < float(self.config.min_bucket_count):
            scores = np.asarray(self.global_action_counts, dtype=float).copy()
        else:
            scores = np.asarray(counts, dtype=float) + (
                float(self.config.smoothing) * np.asarray(self.global_action_counts, dtype=float)
            )
        scores[~legal] = -np.inf
        return scores

    def predict_index(self, state_features: np.ndarray, legal_action_mask: np.ndarray) -> int:
        legal = np.asarray(legal_action_mask, dtype=bool)
        scores = self.score_actions(state_features, legal)
        if np.isfinite(scores).any():
            return int(np.argmax(scores))

        fallback = self.action_index.index_for(self.config.fallback_action_key, default=0)
        if 0 <= fallback < legal.size and bool(legal[fallback]):
            return int(fallback)
        legal_indices = np.flatnonzero(legal)
        return int(legal_indices[0]) if legal_indices.size else 0

    def select_action(self, state: StrategyState) -> StrategyAction:
        features = state_to_feature_vector(state)
        legal = self.action_index.legal_mask_for_state(state)
        if not legal.any():
            raise ValueError("no constraint-legal live-strategy action")
        return self.action_index.action_for(self.predict_index(features, legal))


def fit_behavior_cloning(
    dataset: RLReplayDataset,
    *,
    config: BehaviorCloningConfig | None = None,
) -> MaskedBehaviorCloningPolicy:
    """Fit a deterministic masked BC warm-start policy from replay actions."""

    cfg = config or BehaviorCloningConfig()
    global_counts = np.zeros(dataset.action_index.size, dtype=float)
    bucket_counts: dict[tuple[float, ...], np.ndarray] = {}
    learning_examples = dataset.behavior_cloning_examples()
    for example in learning_examples:
        weight = float(example.weight or 1.0)
        targets = np.flatnonzero(example.behavior_cloning_action_mask)
        if not targets.size:
            continue
        target_mass = weight / float(targets.size)
        global_counts[targets] += target_mass
        bucket = bucket_state_features(example.state_features, precision=int(cfg.bucket_precision))
        if bucket not in bucket_counts:
            bucket_counts[bucket] = np.zeros(dataset.action_index.size, dtype=float)
        bucket_counts[bucket][targets] += target_mass

    fallback = dataset.action_index.index_for(cfg.fallback_action_key, default=0)
    if not np.any(global_counts) and 0 <= fallback < global_counts.size:
        global_counts[fallback] = 1.0

    return MaskedBehaviorCloningPolicy(
        action_index=dataset.action_index,
        global_action_counts=global_counts,
        bucket_action_counts=bucket_counts,
        config=cfg,
        training_diagnostics={
            "training_rows": int(len(learning_examples)),
            "excluded_behavior_cloning_rows": int(dataset.rows - len(learning_examples)),
            "state_buckets": int(len(bucket_counts)),
            "legal_mask_aware": True,
            "target": "observed_team_action_partial_label",
            "partial_label_semantics": (
                "target_mass_is_uniform_over_modes_when_action_mode_is_unobserved"
            ),
            "action_support": dataset.action_support_diagnostics()["behavior_cloning"],
            "diagnostic_note": "behavior cloning is a warm start only, not a promoted strategy optimizer",
        },
    )


def evaluate_behavior_cloning(
    policy: MaskedBehaviorCloningPolicy,
    dataset: RLReplayDataset,
    *,
    baseline: Optional[TrivialLegalActionBaseline] = None,
) -> dict[str, object]:
    """Evaluate historical action selection/timing against a trivial baseline."""

    examples = dataset.behavior_cloning_examples()
    trivial = baseline or TrivialLegalActionBaseline.fit(dataset)
    # Historical target masks are labels, not legal-action evidence.  Score the
    # classifier over the full action space and use the partial label only for
    # correctness, avoiding both target leakage and fabricated legality.
    candidate_mask = np.ones(dataset.action_index.size, dtype=bool)
    policy_predictions = [
        policy.predict_index(example.state_features, candidate_mask)
        for example in examples
    ]
    baseline_predictions = [trivial.predict_index(candidate_mask) for _ in examples]
    policy_metrics = _classification_metrics(examples, policy_predictions, dataset.action_index)
    baseline_metrics = _classification_metrics(examples, baseline_predictions, dataset.action_index)

    return {
        "model": "masked_behavior_cloning_partial_label_warm_start_v2",
        "available": bool(examples),
        "rows": int(len(examples)),
        "metrics": policy_metrics,
        "trivial_baseline": {
            "model": "most_frequent_partial_label_action",
            "metrics": baseline_metrics,
        },
        "delta_vs_trivial_baseline": {
            "action_selection_accuracy": _delta(
                policy_metrics.get("action_selection_accuracy"),
                baseline_metrics.get("action_selection_accuracy"),
            ),
            "pit_decision_f1": _delta(policy_metrics.get("pit_decision_f1"), baseline_metrics.get("pit_decision_f1")),
            "pit_timing_lap_mae": _reverse_delta(
                policy_metrics.get("pit_timing_lap_mae"),
                baseline_metrics.get("pit_timing_lap_mae"),
            ),
        },
        "diagnostics": {
            **policy.diagnostics,
            "historical_accuracy_is_not_policy_value": True,
            "promotion_gate_pass": False,
        },
    }


def _delta(left: object, right: object) -> Optional[float]:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _reverse_delta(left: object, right: object) -> Optional[float]:
    delta = _delta(left, right)
    return None if delta is None else -float(delta)


def _classification_metrics(
    examples: tuple[RLReplayExample, ...],
    predictions: list[int],
    action_index: StrategyActionIndex,
) -> dict[str, object]:
    total = len(examples)
    if total == 0:
        return {
            "action_selection_accuracy": None,
            "action_type_accuracy": None,
            "pit_decision_accuracy": None,
            "pit_decision_precision": None,
            "pit_decision_recall": None,
            "pit_decision_f1": None,
            "pit_timing_lap_mae": None,
            "illegal_prediction_rate": None,
            "legal_mask_certified_rows": 0,
            "legal_mask_unknown_rows": 0,
        }

    action_correct = 0
    type_correct = 0
    pit_correct = 0
    illegal = 0
    certified_mask_rows = 0
    unknown_mask_rows = 0
    tp = fp = fn = 0
    for example, pred_idx in zip(examples, predictions):
        actual_idx = int(example.action_index)
        pred_idx = int(pred_idx)
        action_correct += int(
            0 <= pred_idx < example.behavior_cloning_action_mask.size
            and bool(example.behavior_cloning_action_mask[pred_idx])
        )
        actual_action = action_index.action_for(actual_idx)
        pred_action = action_index.action_for(pred_idx)
        type_correct += int(pred_action.action_type == actual_action.action_type)
        actual_pit = bool(actual_action.is_pit_action)
        pred_pit = bool(pred_action.is_pit_action)
        pit_correct += int(actual_pit == pred_pit)
        tp += int(actual_pit and pred_pit)
        fp += int((not actual_pit) and pred_pit)
        fn += int(actual_pit and not pred_pit)
        transition_metadata = example.metadata.get("transition_metadata", {})
        full_mask_certified = bool(
            str(example.source).strip().lower().startswith(
                ("synthetic", "simulator", "self_play")
            )
            or (
                isinstance(transition_metadata, dict)
                and transition_metadata.get("full_legal_action_mask_certified") is True
            )
        )
        if full_mask_certified:
            certified_mask_rows += 1
            if pred_idx >= example.legal_action_mask.size or not bool(
                example.legal_action_mask[pred_idx]
            ):
                illegal += 1
        else:
            unknown_mask_rows += 1

    actual_positive = tp + fn
    predicted_positive = tp + fp
    precision = _safe_ratio(tp, predicted_positive)
    recall = _safe_ratio(tp, actual_positive)
    if precision is None and actual_positive > 0:
        precision = 0.0
    if recall is None and predicted_positive > 0:
        recall = 0.0
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))
    return {
        "action_selection_accuracy": float(action_correct / total),
        "action_type_accuracy": float(type_correct / total),
        "pit_decision_accuracy": float(pit_correct / total),
        "pit_decision_precision": precision,
        "pit_decision_recall": recall,
        "pit_decision_f1": f1,
        "pit_timing_lap_mae": _pit_lap_mae(examples, predictions, action_index),
        # A fail-closed mask with unknown tyre inventory or pit-lane status is
        # not evidence that a historical prediction was illegal.  Report a
        # rate only on rows whose complete mask is certified.
        "illegal_prediction_rate": (
            float(illegal / certified_mask_rows) if certified_mask_rows else None
        ),
        "legal_mask_certified_rows": int(certified_mask_rows),
        "legal_mask_unknown_rows": int(unknown_mask_rows),
    }


def _safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _pit_lap_mae(
    examples: tuple[RLReplayExample, ...],
    predictions: list[int],
    action_index: StrategyActionIndex,
) -> Optional[float]:
    actual_by_episode: dict[str, list[int]] = {}
    pred_by_episode: dict[str, list[int]] = {}
    for example, pred_idx in zip(examples, predictions):
        actual = action_index.action_for(int(example.action_index))
        pred = action_index.action_for(int(pred_idx))
        key = example.episode_key or f"{example.source}:{example.driver_id}"
        if actual.is_pit_action:
            actual_by_episode.setdefault(key, []).append(int(example.lap_number))
        if pred.is_pit_action:
            pred_by_episode.setdefault(key, []).append(int(example.lap_number))

    errors: list[float] = []
    for key, actual_laps in actual_by_episode.items():
        predicted_laps = list(pred_by_episode.get(key, []))
        if not predicted_laps:
            continue
        used: set[int] = set()
        for actual_lap in actual_laps:
            candidates = [
                (abs(actual_lap - pred_lap), idx)
                for idx, pred_lap in enumerate(predicted_laps)
                if idx not in used
            ]
            if not candidates:
                break
            error, idx = min(candidates, key=lambda item: (item[0], item[1]))
            used.add(idx)
            errors.append(float(error))
    if not errors:
        return None
    return float(np.mean(errors))


__all__ = [
    "BehaviorCloningConfig",
    "MaskedBehaviorCloningPolicy",
    "TrivialLegalActionBaseline",
    "evaluate_behavior_cloning",
    "fit_behavior_cloning",
]
