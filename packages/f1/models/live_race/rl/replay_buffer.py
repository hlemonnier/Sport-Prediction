"""RL-ready adapters for live race strategy replay buffers.

The source of truth remains ``StrategyTransition``.  This module only converts
those typed replay records into deterministic numpy arrays plus auditable
metadata for behavior cloning and conservative offline RL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from packages.f1.models.live_race.action_space import (
    KNOWN_COMPOUNDS,
    STRATEGY_MODES,
    LegalActionMask,
    StrategyAction,
    build_action_space,
    build_legal_action_mask,
    normalize_compound,
)
from packages.f1.models.live_race.environment import StrategyState, StrategyTransition
from packages.f1.models.live_race.replay_buffer import LiveStrategyReplayBuffer, ReplayBufferRecord


COMPOUND_FEATURES: tuple[str, ...] = (*KNOWN_COMPOUNDS, "UNKNOWN")
DEFAULT_STATE_FEATURE_NAMES: tuple[str, ...] = (
    "lap_fraction",
    "remaining_fraction",
    "stint_fraction",
    "tyre_age_fraction",
    "used_compound_count_fraction",
    "race_time_minutes_fraction",
    "gap_to_leader_fraction",
    "position_fraction",
    "is_red",
    "is_sc_vsc",
    "is_yellow",
    "is_greenish",
    "pace_penalty_mean_fraction",
    "pace_penalty_std_fraction",
    "deg_rate_mean_fraction",
    "deg_rate_std_fraction",
    "next_lap_mean_fraction",
    "next_lap_std_fraction",
    "pit_loss_estimate_fraction",
    "circuit_overtaking_difficulty",
    "circuit_tyre_degradation",
    "circuit_safety_car_probability",
    "circuit_strategy_variance",
    "track_overtake_propensity",
    "track_chaos_index",
    *tuple(f"compound_{compound}" for compound in COMPOUND_FEATURES),
    *tuple(f"used_compound_{compound}" for compound in KNOWN_COMPOUNDS),
)


def _finite(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        numeric = float(value)
        if not np.isfinite(numeric):
            return float(default)
        return numeric
    except Exception:
        return float(default)


def _fraction(value: object, scale: float, *, default: float = 0.0, lower: float = -1.0, upper: float = 1.0) -> float:
    if scale <= 0.0:
        return float(default)
    return float(np.clip(_finite(value, default) / float(scale), lower, upper))


def _unit_interval(value: object, default: float = 0.5) -> float:
    return float(np.clip(_finite(value, default), 0.0, 1.0))


def state_to_feature_vector(
    state: StrategyState,
    *,
    feature_names: Sequence[str] = DEFAULT_STATE_FEATURE_NAMES,
) -> np.ndarray:
    """Convert a no-leakage ``StrategyState`` into bounded numeric features."""

    total_laps = max(1.0, _finite(state.total_laps, float(state.lap_number + (state.remaining_laps or 0) or 1)))
    remaining = _finite(state.remaining_laps, max(0.0, total_laps - float(state.lap_number)))
    used = set(normalize_compound(item) for item in state.used_compounds)
    compound = normalize_compound(state.compound)

    base: dict[str, float] = {
        "lap_fraction": float(np.clip(float(state.lap_number) / total_laps, 0.0, 1.25)),
        "remaining_fraction": float(np.clip(remaining / total_laps, 0.0, 1.25)),
        "stint_fraction": _fraction(state.stint_id, 8.0, lower=0.0, upper=1.0),
        "tyre_age_fraction": _fraction(state.tyre_age, 40.0, lower=0.0, upper=1.5),
        "used_compound_count_fraction": _fraction(len(used), 5.0, lower=0.0, upper=1.0),
        "race_time_minutes_fraction": _fraction(state.race_time_seconds, 7200.0, lower=0.0, upper=1.5),
        "gap_to_leader_fraction": _fraction(state.gap_to_leader_seconds, 120.0),
        "position_fraction": _fraction(state.position, 20.0, lower=0.0, upper=1.5),
        "is_red": float(bool(state.is_red)),
        "is_sc_vsc": float(bool(state.is_sc_vsc)),
        "is_yellow": float(bool(state.is_yellow)),
        "is_greenish": float(bool(state.is_greenish)),
        "pace_penalty_mean_fraction": _fraction(state.pace_penalty_mean, 10.0),
        "pace_penalty_std_fraction": _fraction(state.pace_penalty_std, 10.0, lower=0.0, upper=1.0),
        "deg_rate_mean_fraction": _fraction(state.deg_rate_mean, 0.5, lower=0.0, upper=1.5),
        "deg_rate_std_fraction": _fraction(state.deg_rate_std, 0.5, lower=0.0, upper=1.0),
        "next_lap_mean_fraction": _fraction(state.next_lap_mean, 180.0, lower=0.0, upper=1.5),
        "next_lap_std_fraction": _fraction(state.next_lap_std, 20.0, lower=0.0, upper=1.0),
        "pit_loss_estimate_fraction": _fraction(state.pit_loss_estimate_seconds, 40.0, lower=0.0, upper=1.5),
        "circuit_overtaking_difficulty": _unit_interval(state.circuit_overtaking_difficulty),
        "circuit_tyre_degradation": _unit_interval(state.circuit_tyre_degradation),
        "circuit_safety_car_probability": _unit_interval(state.circuit_safety_car_probability),
        "circuit_strategy_variance": _unit_interval(state.circuit_strategy_variance),
        "track_overtake_propensity": _unit_interval(state.track_overtake_propensity),
        "track_chaos_index": _unit_interval(state.track_chaos_index),
    }
    for item in COMPOUND_FEATURES:
        base[f"compound_{item}"] = float(compound == item)
    for item in KNOWN_COMPOUNDS:
        base[f"used_compound_{item}"] = float(item in used)

    vector = np.asarray([base.get(str(name), 0.0) for name in feature_names], dtype=float)
    return np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)


def bucket_state_features(
    features: Sequence[float] | np.ndarray,
    *,
    precision: int = 2,
) -> tuple[float, ...]:
    """Return a stable coarse tabular key for small-data BC/offline RL."""

    arr = np.asarray(features, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    return tuple(float(value) for value in np.round(arr, decimals=int(precision)).tolist())


@dataclass(frozen=True)
class StrategyActionIndex:
    """Stable action-index mapping shared by BC and offline RL policies."""

    actions: tuple[StrategyAction, ...] = field(default_factory=build_action_space)

    def __post_init__(self) -> None:
        actions = tuple(self.actions or build_action_space())
        keys = [action.key for action in actions]
        if len(set(keys)) != len(keys):
            raise ValueError("action index contains duplicate action keys")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "_key_to_index", {action.key: idx for idx, action in enumerate(actions)})

    @classmethod
    def from_action_space(cls, action_space: Optional[Sequence[StrategyAction]] = None) -> "StrategyActionIndex":
        return cls(actions=tuple(action_space or build_action_space()))

    @property
    def size(self) -> int:
        return len(self.actions)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(action.key for action in self.actions)

    def index_for(self, action: StrategyAction | str, *, default: int = -1) -> int:
        key = action.key if isinstance(action, StrategyAction) else str(action)
        return int(getattr(self, "_key_to_index").get(key, default))

    def action_for(self, index: int) -> StrategyAction:
        return self.actions[int(index)]

    def legal_mask_from(self, mask: LegalActionMask) -> np.ndarray:
        aligned = np.zeros(self.size, dtype=bool)
        for action, legal in zip(mask.actions, mask.mask.tolist()):
            idx = self.index_for(action)
            if idx >= 0:
                aligned[idx] = bool(legal)
        return aligned

    def legal_mask_for_state(self, state: StrategyState) -> np.ndarray:
        return self.legal_mask_from(build_legal_action_mask(state, action_space=self.actions))


@dataclass(frozen=True)
class RLReplayExample:
    """Single model-ready live-strategy transition."""

    record_id: str
    state_features: np.ndarray
    next_state_features: np.ndarray
    action_index: int
    action_key: str
    legal_action_mask: np.ndarray
    next_legal_action_mask: np.ndarray
    reward: float
    done: bool
    weight: float = 1.0
    source: str = "unknown"
    split_key: Optional[str] = None
    state_fingerprint: str = ""
    next_state_fingerprint: str = ""
    lap_number: int = 0
    driver_id: str = ""
    episode_key: str = ""
    metadata: dict[str, object] = field(default_factory=dict)
    ood: bool = False
    ood_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        state = np.asarray(self.state_features, dtype=float)
        next_state = np.asarray(self.next_state_features, dtype=float)
        legal = np.asarray(self.legal_action_mask, dtype=bool)
        next_legal = np.asarray(self.next_legal_action_mask, dtype=bool)
        if state.ndim != 1 or next_state.ndim != 1:
            raise ValueError("state feature arrays must be one-dimensional")
        if state.shape != next_state.shape:
            raise ValueError("state and next-state feature arrays must have matching shape")
        if legal.ndim != 1 or next_legal.ndim != 1:
            raise ValueError("legal masks must be one-dimensional")
        if legal.shape != next_legal.shape:
            raise ValueError("legal and next-legal masks must have matching shape")
        object.__setattr__(self, "state_features", np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0))
        object.__setattr__(self, "next_state_features", np.nan_to_num(next_state, nan=0.0, posinf=1.0, neginf=-1.0))
        object.__setattr__(self, "legal_action_mask", legal)
        object.__setattr__(self, "next_legal_action_mask", next_legal)
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "weight", max(0.0, float(self.weight)))

    @property
    def is_valid_for_learning(self) -> bool:
        return bool(not self.ood and self.action_index >= 0 and self.legal_action_mask[int(self.action_index)])

    def to_payload(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "action_index": int(self.action_index),
            "action_key": self.action_key,
            "legal_action_mask": [bool(value) for value in self.legal_action_mask.tolist()],
            "reward": float(self.reward),
            "done": bool(self.done),
            "weight": float(self.weight),
            "source": self.source,
            "split_key": self.split_key,
            "state_fingerprint": self.state_fingerprint,
            "next_state_fingerprint": self.next_state_fingerprint,
            "lap_number": int(self.lap_number),
            "driver_id": self.driver_id,
            "episode_key": self.episode_key,
            "ood": bool(self.ood),
            "ood_reasons": list(self.ood_reasons),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class RLReplayDataset:
    """Small deterministic replay dataset consumed by Phase 7 learners."""

    examples: tuple[RLReplayExample, ...]
    action_index: StrategyActionIndex
    feature_names: tuple[str, ...] = DEFAULT_STATE_FEATURE_NAMES
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return len(self.examples)

    @property
    def states(self) -> np.ndarray:
        if not self.examples:
            return np.empty((0, len(self.feature_names)), dtype=float)
        return np.vstack([example.state_features for example in self.examples]).astype(float)

    @property
    def next_states(self) -> np.ndarray:
        if not self.examples:
            return np.empty((0, len(self.feature_names)), dtype=float)
        return np.vstack([example.next_state_features for example in self.examples]).astype(float)

    @property
    def actions(self) -> np.ndarray:
        return np.asarray([example.action_index for example in self.examples], dtype=int)

    @property
    def rewards(self) -> np.ndarray:
        return np.asarray([example.reward for example in self.examples], dtype=float)

    @property
    def dones(self) -> np.ndarray:
        return np.asarray([example.done for example in self.examples], dtype=bool)

    @property
    def legal_action_masks(self) -> np.ndarray:
        if not self.examples:
            return np.empty((0, self.action_index.size), dtype=bool)
        return np.vstack([example.legal_action_mask for example in self.examples]).astype(bool)

    def learning_examples(self, *, include_ood: bool = False) -> tuple[RLReplayExample, ...]:
        if include_ood:
            return tuple(example for example in self.examples if example.action_index >= 0)
        return tuple(example for example in self.examples if example.is_valid_for_learning)

    def diagnostics(self) -> dict[str, object]:
        action_counts = {key: 0 for key in self.action_index.keys}
        source_counts: dict[str, int] = {}
        split_counts: dict[str, int] = {}
        ood_reasons: dict[str, int] = {}
        for example in self.examples:
            if example.action_index >= 0:
                action_counts[example.action_key] = int(action_counts.get(example.action_key, 0) + 1)
            source_counts[example.source] = int(source_counts.get(example.source, 0) + 1)
            split = str(example.split_key or "unknown")
            split_counts[split] = int(split_counts.get(split, 0) + 1)
            for reason in example.ood_reasons:
                ood_reasons[reason] = int(ood_reasons.get(reason, 0) + 1)
        return {
            "rows": int(self.rows),
            "learning_rows": int(len(self.learning_examples())),
            "ood_rows": int(sum(1 for example in self.examples if example.ood)),
            "feature_count": int(len(self.feature_names)),
            "action_count": int(self.action_index.size),
            "action_counts": action_counts,
            "source_counts": source_counts,
            "split_counts": split_counts,
            "ood_reasons": ood_reasons,
            "metadata": self.metadata,
        }


def _iter_replay_records(
    records: LiveStrategyReplayBuffer
    | Iterable[ReplayBufferRecord | StrategyTransition]
    | ReplayBufferRecord
    | StrategyTransition,
) -> Iterable[ReplayBufferRecord]:
    if isinstance(records, LiveStrategyReplayBuffer):
        yield from records.records
        return
    if isinstance(records, ReplayBufferRecord):
        yield records
        return
    if isinstance(records, StrategyTransition):
        yield ReplayBufferRecord.from_transition(records)
        return
    for item in records:
        if isinstance(item, ReplayBufferRecord):
            yield item
        elif isinstance(item, StrategyTransition):
            yield ReplayBufferRecord.from_transition(item)
        else:
            raise TypeError(f"unsupported replay item {type(item)!r}")


def _episode_key(record: ReplayBufferRecord) -> str:
    transition = record.transition
    event = transition.state_t.event_key if transition.state_t.event_key is not None else "event"
    return f"{record.source}:{event}:{transition.state_t.driver_id}"


def _ood_reasons_for(
    record: ReplayBufferRecord,
    *,
    action_index: int,
    legal_mask: np.ndarray,
    reward: float,
    unclipped_reward: float,
    state_features: np.ndarray,
    next_state_features: np.ndarray,
) -> tuple[str, ...]:
    reasons: list[str] = []
    for error in record.validate(require_legal_action=False):
        reasons.append(f"record_validation:{error}")
    if action_index < 0:
        reasons.append("action_not_in_action_index")
    elif action_index >= legal_mask.size or not bool(legal_mask[action_index]):
        reasons.append("observed_action_illegal_under_mask")
    if reward != unclipped_reward:
        reasons.append("reward_clipped")
    if not np.isfinite(state_features).all():
        reasons.append("state_features_non_finite")
    if not np.isfinite(next_state_features).all():
        reasons.append("next_state_features_non_finite")
    if not legal_mask.any():
        reasons.append("no_legal_actions_after_alignment")
    return tuple(dict.fromkeys(reasons))


def build_rl_replay_dataset(
    records: LiveStrategyReplayBuffer
    | Iterable[ReplayBufferRecord | StrategyTransition]
    | ReplayBufferRecord
    | StrategyTransition,
    *,
    action_index: StrategyActionIndex | None = None,
    action_space: Optional[Sequence[StrategyAction]] = None,
    feature_names: Sequence[str] = DEFAULT_STATE_FEATURE_NAMES,
    reward_clip: tuple[float, float] = (-500.0, 100.0),
    strict: bool = False,
    metadata: Optional[Mapping[str, object]] = None,
) -> RLReplayDataset:
    """Convert strategy replay records into legal-mask-aware RL examples."""

    mapping = action_index or StrategyActionIndex.from_action_space(action_space)
    names = tuple(feature_names)
    examples: list[RLReplayExample] = []
    low, high = float(reward_clip[0]), float(reward_clip[1])
    for record in _iter_replay_records(records):
        transition = record.transition
        state_features = state_to_feature_vector(transition.state_t, feature_names=names)
        next_state_features = state_to_feature_vector(transition.state_t1, feature_names=names)
        action_idx = mapping.index_for(transition.action_t)
        legal_mask = mapping.legal_mask_from(transition.legal_action_mask)
        next_legal_mask = mapping.legal_mask_for_state(transition.state_t1)
        raw_reward = _finite(transition.reward_t.value, 0.0)
        reward = float(np.clip(raw_reward, low, high))
        reasons = _ood_reasons_for(
            record,
            action_index=action_idx,
            legal_mask=legal_mask,
            reward=reward,
            unclipped_reward=raw_reward,
            state_features=state_features,
            next_state_features=next_state_features,
        )
        if strict and reasons:
            raise ValueError(f"RL replay record {record.record_id} is not learnable: {', '.join(reasons)}")
        merged_metadata = {
            "record_metadata": dict(record.metadata or {}),
            "transition_metadata": dict(transition.metadata or {}),
            "reward_components": dict(transition.reward_t.components or {}),
        }
        example = RLReplayExample(
            record_id=record.record_id,
            state_features=state_features,
            next_state_features=next_state_features,
            action_index=action_idx,
            action_key=transition.action_t.key,
            legal_action_mask=legal_mask,
            next_legal_action_mask=next_legal_mask,
            reward=reward,
            done=bool(transition.done),
            weight=float(record.weight),
            source=record.source,
            split_key=record.split_key,
            state_fingerprint=transition.state_t.fingerprint(),
            next_state_fingerprint=transition.state_t1.fingerprint(),
            lap_number=int(transition.state_t.lap_number),
            driver_id=transition.state_t.driver_id,
            episode_key=_episode_key(record),
            metadata=merged_metadata,
            ood=bool(reasons),
            ood_reasons=reasons,
        )
        examples.append(example)

    return RLReplayDataset(
        examples=tuple(examples),
        action_index=mapping,
        feature_names=names,
        metadata={
            "dataset_builder": "live_strategy_rl_replay_v1",
            "reward_clip": (low, high),
            **dict(metadata or {}),
        },
    )


__all__ = [
    "COMPOUND_FEATURES",
    "DEFAULT_STATE_FEATURE_NAMES",
    "RLReplayDataset",
    "RLReplayExample",
    "StrategyActionIndex",
    "bucket_state_features",
    "build_rl_replay_dataset",
    "state_to_feature_vector",
]

