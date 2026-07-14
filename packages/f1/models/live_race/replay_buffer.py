"""Replay-buffer records for live race strategy learning/evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

from packages.f1.models.live_race.environment import (
    REWARD_SEMANTICS,
    TRANSITION_FINGERPRINT_VERSION,
    StrategyTransition,
    assert_replay_prefix_invariant,
    transition_prefix_fingerprint,
)


_SYNTHETIC_SOURCE_PREFIXES = ("synthetic", "simulator", "self_play")
REPLAY_RECORD_SCHEMA_VERSION = (
    "live_strategy_replay_record_v7_full_current_next_mask_input_and_feasibility_evidence"
)


def _is_explicit_synthetic_source(
    record_source: object,
    transition_metadata: dict[str, object] | None,
) -> bool:
    metadata = transition_metadata or {}
    tokens = (
        str(record_source or "").strip().lower(),
        str(metadata.get("source") or "").strip().lower(),
    )
    return any(
        token.startswith(prefix)
        for token in tokens
        for prefix in _SYNTHETIC_SOURCE_PREFIXES
        if token
    )


def _reported_policy_learning_eligibility(record: "ReplayBufferRecord") -> bool:
    metadata = record.transition.metadata or {}
    if "policy_training_eligible" in metadata:
        return metadata.get("policy_training_eligible") is True
    if "policy_learning_eligible" in metadata:
        return metadata.get("policy_learning_eligible") is True
    return _is_explicit_synthetic_source(record.source, metadata)


def _reported_propensity_ope_eligibility(record: "ReplayBufferRecord") -> bool:
    metadata = record.transition.metadata or {}
    raw_probability = metadata.get(
        "behavior_action_probability",
        record.transition.state_t.metadata.get("behavior_action_probability"),
    )
    try:
        probability = float(raw_probability)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(probability) or not 0.0 < probability <= 1.0:
        return False
    if "propensity_ope_eligible" in metadata:
        return metadata.get("propensity_ope_eligible") is True
    # Legacy records used policy-learning eligibility for the stricter
    # training-plus-propensity contract. Preserve that interpretation only
    # when the v3 training field is absent.
    if (
        "policy_training_eligible" not in metadata
        and "policy_learning_eligible" in metadata
    ):
        return metadata.get("policy_learning_eligible") is True
    return False


@dataclass(frozen=True)
class ReplayBufferRecord:
    transition: StrategyTransition
    record_id: str
    source: str = "unknown"
    split_key: Optional[str] = None
    weight: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_transition(
        cls,
        transition: StrategyTransition,
        *,
        source: str = "unknown",
        split_key: Optional[str] = None,
        weight: float = 1.0,
        metadata: Optional[dict[str, object]] = None,
    ) -> "ReplayBufferRecord":
        record_id = transition.fingerprint()
        return cls(
            transition=transition,
            record_id=record_id,
            source=str(source),
            split_key=split_key,
            weight=float(weight),
            metadata=dict(metadata or {}),
        )

    def validate(self, *, require_legal_action: bool = True) -> list[str]:
        errors = self.transition.validate()
        if require_legal_action and not self.transition.is_action_legal():
            errors.append("action_not_legal_under_mask")
        if not np.isfinite(float(self.weight)) or float(self.weight) <= 0.0:
            errors.append("non_positive_record_weight")
        return errors

    def to_payload(self) -> dict[str, object]:
        return {
            "record_schema_version": REPLAY_RECORD_SCHEMA_VERSION,
            "transition_fingerprint_version": TRANSITION_FINGERPRINT_VERSION,
            "reward_semantics": REWARD_SEMANTICS,
            "record_id": self.record_id,
            "source": self.source,
            "split_key": self.split_key,
            "weight": float(self.weight),
            "transition": self.transition.to_payload(),
            "metadata": self.metadata,
        }


class LiveStrategyReplayBuffer:
    """Append-only replay buffer with strict transition validation by default."""

    def __init__(self, records: Optional[Iterable[ReplayBufferRecord | StrategyTransition]] = None) -> None:
        self._records: list[ReplayBufferRecord] = []
        if records is not None:
            self.extend(records)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[ReplayBufferRecord]:
        return iter(self._records)

    def __getitem__(self, index: int) -> ReplayBufferRecord:
        return self._records[index]

    @property
    def records(self) -> tuple[ReplayBufferRecord, ...]:
        return tuple(self._records)

    @property
    def transitions(self) -> tuple[StrategyTransition, ...]:
        return tuple(record.transition for record in self._records)

    def append(
        self,
        record: ReplayBufferRecord | StrategyTransition,
        *,
        strict: bool = True,
        require_legal_action: bool = True,
    ) -> None:
        item = record if isinstance(record, ReplayBufferRecord) else ReplayBufferRecord.from_transition(record)
        errors = item.validate(require_legal_action=require_legal_action)
        if strict and errors:
            raise ValueError(f"invalid replay record {item.record_id}: {', '.join(errors)}")
        self._records.append(item)

    def extend(
        self,
        records: Iterable[ReplayBufferRecord | StrategyTransition],
        *,
        strict: bool = True,
        require_legal_action: bool = True,
    ) -> None:
        for record in records:
            self.append(record, strict=strict, require_legal_action=require_legal_action)

    def sample(
        self,
        n: int,
        *,
        seed: int = 42,
        replace: bool = False,
    ) -> tuple[ReplayBufferRecord, ...]:
        if n <= 0 or not self._records:
            return ()
        rng = np.random.default_rng(int(seed))
        size = int(n) if replace else min(int(n), len(self._records))
        idx = rng.choice(len(self._records), size=size, replace=bool(replace))
        return tuple(self._records[int(item)] for item in np.atleast_1d(idx))

    def to_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for record in self._records:
            transition = record.transition
            rows.append(
                {
                    "record_id": record.record_id,
                    "source": record.source,
                    "split_key": record.split_key,
                    "weight": float(record.weight),
                    "driver_id": transition.state_t.driver_id,
                    "lap_number": int(transition.state_t.lap_number),
                    "next_lap_number": int(transition.state_t1.lap_number),
                    "action_key": transition.action_t.key,
                    "action_type": transition.action_t.action_type,
                    "action_compound": transition.action_t.compound,
                    "action_mode": transition.action_t.mode,
                    "reward": float(transition.reward_t.value),
                    "done": bool(transition.done),
                    "elapsed_laps": int(
                        transition.metadata.get(
                            "elapsed_laps",
                            max(1, transition.state_t1.lap_number - transition.state_t.lap_number),
                        )
                    ),
                    "transition_kind": transition.metadata.get("transition_kind", "unknown"),
                    "is_action_legal": bool(transition.is_action_legal()),
                    "action_legality_status": transition.metadata.get("action_legality_status", "unspecified"),
                    "behavior_action_support_status": transition.metadata.get(
                        "behavior_action_support_status",
                        "unspecified",
                    ),
                    "behavior_action_probability": transition.metadata.get(
                        "behavior_action_probability",
                        transition.state_t.metadata.get(
                            "behavior_action_probability"
                        ),
                    ),
                    "reward_observation_status": transition.metadata.get(
                        "reward_observation_status",
                        "unspecified",
                    ),
                    "reward_observation_blockers": tuple(
                        transition.metadata.get(
                            "reward_observation_blockers",
                            (),
                        )
                    ),
                    "policy_training_eligible": _reported_policy_learning_eligibility(
                        record
                    ),
                    "propensity_ope_eligible": _reported_propensity_ope_eligibility(
                        record
                    ),
                    # Compatibility alias for older reporting code.
                    "policy_learning_eligible": _reported_policy_learning_eligibility(record),
                    "legal_action_count": int(transition.legal_action_mask.legal_count),
                    "state_fingerprint": transition.state_t.fingerprint(),
                    "transition_fingerprint": transition.fingerprint(),
                }
            )
        return pd.DataFrame(rows)

    def prefix_fingerprint(self, *, cutoff_lap: int) -> str:
        return transition_prefix_fingerprint(self.transitions, cutoff_lap=int(cutoff_lap))

    def prefix_invariant_with(self, other: "LiveStrategyReplayBuffer", *, cutoff_lap: int) -> bool:
        return assert_replay_prefix_invariant(
            self.transitions,
            other.transitions,
            cutoff_lap=int(cutoff_lap),
        )

    def write_jsonl(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for record in self._records:
                handle.write(json.dumps(record.to_payload(), sort_keys=True, ensure_ascii=True) + "\n")
        return output


def replay_buffer_from_transitions(
    transitions: Sequence[StrategyTransition],
    *,
    source: str = "lap_replay",
    split_key: Optional[str] = None,
    strict: bool = True,
    require_legal_action: bool = True,
) -> LiveStrategyReplayBuffer:
    buffer = LiveStrategyReplayBuffer()
    for transition in transitions:
        buffer.append(
            ReplayBufferRecord.from_transition(
                transition,
                source=source,
                split_key=split_key,
            ),
            strict=strict,
            require_legal_action=require_legal_action,
        )
    return buffer


__all__ = [
    "LiveStrategyReplayBuffer",
    "REPLAY_RECORD_SCHEMA_VERSION",
    "ReplayBufferRecord",
    "replay_buffer_from_transitions",
]
