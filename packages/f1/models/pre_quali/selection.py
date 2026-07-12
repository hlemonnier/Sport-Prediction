"""Frozen event-block selector for the Qualifying challenger.

This policy intentionally refuses per-round model hopping.  Baseline and
challenger evidence must cover the same complete events, meet a minimum sample,
and accumulate a configurable number of genuinely new events before another
switch is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class QualifyingModelEvidence:
    model_id: str
    mean_absolute_position_error: float
    event_keys: tuple[int, ...]
    promotion_gates_passed: bool = False

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if not np.isfinite(self.mean_absolute_position_error):
            raise ValueError("mean_absolute_position_error must be finite")
        normalized = tuple(sorted({int(value) for value in self.event_keys}))
        if normalized != self.event_keys:
            raise ValueError("event_keys must be unique and sorted")


@dataclass(frozen=True)
class FrozenSelectorConfig:
    baseline_model_id: str = "qualifying_rehearsal_rank_baseline_v1"
    challenger_model_id: str = "qualifying_pairwise_logistic_residual_v1"
    minimum_evidence_events: int = 8
    freeze_for_new_events: int = 4
    minimum_mae_improvement: float = 0.15

    def __post_init__(self) -> None:
        if self.minimum_evidence_events < 2:
            raise ValueError("minimum_evidence_events must be at least two")
        if self.freeze_for_new_events < 1:
            raise ValueError("freeze_for_new_events must be positive")
        if self.minimum_mae_improvement < 0.0:
            raise ValueError("minimum_mae_improvement must be non-negative")


@dataclass(frozen=True)
class QualifyingSelectionState:
    selected_model_id: str
    observed_event_keys: tuple[int, ...]
    selection_event_keys: tuple[int, ...]
    decision: str
    baseline_mae: float
    challenger_mae: float

    @property
    def new_events_since_selection(self) -> int:
        return len(set(self.observed_event_keys) - set(self.selection_event_keys))


def select_frozen_qualifying_model(
    evidence: Sequence[QualifyingModelEvidence],
    *,
    config: FrozenSelectorConfig = FrozenSelectorConfig(),
    previous_state: QualifyingSelectionState | None = None,
) -> QualifyingSelectionState:
    """Select only from matched event-block evidence and enforce a freeze."""

    by_model: Mapping[str, QualifyingModelEvidence] = {item.model_id: item for item in evidence}
    if len(by_model) != len(evidence):
        raise ValueError("each model_id may appear only once in evidence")
    if config.baseline_model_id not in by_model or config.challenger_model_id not in by_model:
        raise ValueError("matched baseline and challenger evidence are both required")
    baseline = by_model[config.baseline_model_id]
    challenger = by_model[config.challenger_model_id]
    if baseline.event_keys != challenger.event_keys:
        raise ValueError("baseline and challenger must be scored on identical event blocks")
    observed = baseline.event_keys

    if previous_state is not None:
        if not set(previous_state.observed_event_keys).issubset(observed):
            raise ValueError("selector evidence may not discard previously observed events")
        if previous_state.selected_model_id not in {
            config.baseline_model_id,
            config.challenger_model_id,
        }:
            raise ValueError("previous_state contains a model outside the frozen selector")

    if len(observed) < config.minimum_evidence_events:
        selected = config.baseline_model_id
        decision = "baseline_retained_insufficient_event_evidence"
    else:
        improvement = baseline.mean_absolute_position_error - challenger.mean_absolute_position_error
        proposed = (
            config.challenger_model_id
            if improvement >= config.minimum_mae_improvement
            and challenger.promotion_gates_passed
            else config.baseline_model_id
        )
        if previous_state is not None and proposed != previous_state.selected_model_id:
            new_events = len(set(observed) - set(previous_state.selection_event_keys))
            if new_events < config.freeze_for_new_events:
                selected = previous_state.selected_model_id
                decision = "previous_selection_retained_during_freeze_window"
            else:
                selected = proposed
                decision = "selection_changed_after_freeze_and_matched_evidence"
        else:
            selected = proposed
            decision = (
                "challenger_selected_on_matched_event_evidence"
                if selected == config.challenger_model_id
                else (
                    "baseline_retained_challenger_promotion_gates_failed"
                    if improvement >= config.minimum_mae_improvement
                    and not challenger.promotion_gates_passed
                    else "baseline_retained_no_material_challenger_gain"
                )
            )

    if previous_state is None or selected != previous_state.selected_model_id:
        selection_events = observed
    else:
        selection_events = previous_state.selection_event_keys
    return QualifyingSelectionState(
        selected_model_id=selected,
        observed_event_keys=observed,
        selection_event_keys=selection_events,
        decision=decision,
        baseline_mae=float(baseline.mean_absolute_position_error),
        challenger_mae=float(challenger.mean_absolute_position_error),
    )


__all__ = [
    "FrozenSelectorConfig",
    "QualifyingModelEvidence",
    "QualifyingSelectionState",
    "select_frozen_qualifying_model",
]
