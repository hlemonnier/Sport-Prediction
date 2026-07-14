"""RL-ready adapters for live race strategy replay buffers.

The source of truth remains ``StrategyTransition``.  This module only converts
those typed replay records into deterministic numpy arrays plus auditable
metadata for behavior cloning and conservative offline RL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from packages.f1.models.live_race.action_space import (
    ACTION_STAY_OUT,
    ActionMaskConfig,
    KNOWN_COMPOUNDS,
    PIT_ACTION_TYPES,
    STRATEGY_MODES,
    LegalActionMask,
    StrategyAction,
    build_action_space,
    build_legal_action_mask,
    normalize_compound,
)
from packages.f1.models.live_race.environment import (
    REWARD_SEMANTICS,
    StrategyState,
    StrategyTransition,
    legal_action_mask_input_evidence,
)
from packages.f1.models.live_race.replay_buffer import (
    LiveStrategyReplayBuffer,
    ReplayBufferRecord,
    _is_explicit_synthetic_source,
)


COMPOUND_FEATURES: tuple[str, ...] = (*KNOWN_COMPOUNDS, "UNKNOWN")
REPLAY_DATASET_SCHEMA_VERSION = (
    "live_strategy_rl_replay_v7_full_current_next_mask_input_and_feasibility_evidence"
)
MIN_STRATEGY_SUPPORTED_ACTION_KEYS = 2
DEFAULT_STATE_FEATURE_NAMES: tuple[str, ...] = (
    "lap_fraction",
    "lap_number_per_100",
    "remaining_fraction",
    "race_horizon_known",
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


def _canonical_payload_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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

    total_known = bool(state.total_laps is not None and int(state.total_laps) > 0)
    remaining_known = bool(
        state.remaining_laps is not None and int(state.remaining_laps) >= 0
    )
    horizon_known = bool(total_known or remaining_known)
    if total_known:
        total_laps = max(1.0, _finite(state.total_laps, 1.0))
    elif remaining_known:
        total_laps = max(
            1.0,
            float(state.lap_number) + _finite(state.remaining_laps, 0.0),
        )
    else:
        total_laps = 1.0
    remaining = (
        _finite(
            state.remaining_laps,
            max(0.0, total_laps - float(state.lap_number)),
        )
        if horizon_known
        else 0.0
    )
    used = set(normalize_compound(item) for item in state.used_compounds)
    compound = normalize_compound(state.compound)

    base: dict[str, float] = {
        "lap_fraction": (
            float(np.clip(float(state.lap_number) / total_laps, 0.0, 1.25))
            if horizon_known
            else 0.0
        ),
        "lap_number_per_100": float(
            np.clip(float(state.lap_number) / 100.0, 0.0, 1.25)
        ),
        "remaining_fraction": (
            float(np.clip(remaining / total_laps, 0.0, 1.25))
            if horizon_known
            else 0.0
        ),
        "race_horizon_known": float(horizon_known),
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
    action_mask_config: ActionMaskConfig = field(default_factory=ActionMaskConfig)

    def __post_init__(self) -> None:
        actions = tuple(self.actions or build_action_space())
        keys = [action.key for action in actions]
        if len(set(keys)) != len(keys):
            raise ValueError("action index contains duplicate action keys")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "_key_to_index", {action.key: idx for idx, action in enumerate(actions)})

    @classmethod
    def from_action_space(
        cls,
        action_space: Optional[Sequence[StrategyAction]] = None,
        *,
        action_mask_config: ActionMaskConfig | None = None,
    ) -> "StrategyActionIndex":
        return cls(
            actions=tuple(action_space or build_action_space()),
            action_mask_config=action_mask_config or ActionMaskConfig(),
        )

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
        for action, legal in zip(
            mask.actions,
            mask.constraint_legal_mask.tolist(),
        ):
            idx = self.index_for(action)
            if idx >= 0:
                aligned[idx] = bool(legal)
        return aligned

    def legal_mask_for_state(self, state: StrategyState) -> np.ndarray:
        return self.legal_mask_from(
            build_legal_action_mask(
                state,
                action_space=self.actions,
                config=self.action_mask_config,
            )
        )


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
    elapsed_laps: int = 1
    driver_id: str = ""
    episode_key: str = ""
    metadata: dict[str, object] = field(default_factory=dict)
    behavior_cloning_action_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    behavior_cloning_eligible: bool = False
    behavior_cloning_ineligible_reasons: tuple[str, ...] = ()
    behavior_cloning_label_kind: str = "ineligible"
    behavior_cloning_mode_observed: bool = False
    ood: bool = False
    ood_reasons: tuple[str, ...] = ()
    behavior_action_probability: Optional[float] = None
    propensity_ope_eligible: bool = False
    propensity_ope_ineligible_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        state = np.asarray(self.state_features, dtype=float)
        next_state = np.asarray(self.next_state_features, dtype=float)
        legal = np.asarray(self.legal_action_mask, dtype=bool)
        next_legal = np.asarray(self.next_legal_action_mask, dtype=bool)
        behavior_cloning_mask = np.asarray(self.behavior_cloning_action_mask, dtype=bool)
        if state.ndim != 1 or next_state.ndim != 1:
            raise ValueError("state feature arrays must be one-dimensional")
        if state.shape != next_state.shape:
            raise ValueError("state and next-state feature arrays must have matching shape")
        if legal.ndim != 1 or next_legal.ndim != 1:
            raise ValueError("legal masks must be one-dimensional")
        if legal.shape != next_legal.shape:
            raise ValueError("legal and next-legal masks must have matching shape")
        if behavior_cloning_mask.size == 0:
            behavior_cloning_mask = np.zeros_like(legal, dtype=bool)
            if (
                self.behavior_cloning_eligible
                and 0 <= int(self.action_index) < behavior_cloning_mask.size
            ):
                behavior_cloning_mask[int(self.action_index)] = True
        if behavior_cloning_mask.ndim != 1 or behavior_cloning_mask.shape != legal.shape:
            raise ValueError("behavior-cloning action mask must match the legal action mask")
        object.__setattr__(self, "state_features", np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0))
        object.__setattr__(self, "next_state_features", np.nan_to_num(next_state, nan=0.0, posinf=1.0, neginf=-1.0))
        object.__setattr__(self, "legal_action_mask", legal)
        object.__setattr__(self, "next_legal_action_mask", next_legal)
        object.__setattr__(self, "behavior_cloning_action_mask", behavior_cloning_mask)
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "weight", max(0.0, float(self.weight)))
        object.__setattr__(self, "elapsed_laps", max(1, int(self.elapsed_laps)))
        raw_probability = self.behavior_action_probability
        probability: Optional[float]
        try:
            candidate = float(raw_probability) if raw_probability is not None else float("nan")
        except (TypeError, ValueError):
            candidate = float("nan")
        probability = float(candidate) if np.isfinite(candidate) else None
        object.__setattr__(self, "behavior_action_probability", probability)

    @property
    def is_valid_for_behavior_cloning(self) -> bool:
        return bool(
            self.behavior_cloning_eligible
            and not self.behavior_cloning_ineligible_reasons
            and self.action_index >= 0
            and self.action_index < self.behavior_cloning_action_mask.size
            and self.behavior_cloning_action_mask[int(self.action_index)]
        )

    @property
    def is_valid_for_learning(self) -> bool:
        return bool(
            not self.ood
            and 0 <= self.action_index < self.legal_action_mask.size
            and self.legal_action_mask[int(self.action_index)]
        )

    @property
    def is_valid_for_propensity_ope(self) -> bool:
        probability = self.behavior_action_probability
        return bool(
            self.is_valid_for_learning
            and self.propensity_ope_eligible
            and not self.propensity_ope_ineligible_reasons
            and probability is not None
            and np.isfinite(float(probability))
            and 0.0 < float(probability) <= 1.0
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "action_index": int(self.action_index),
            "action_key": self.action_key,
            "legal_action_mask": [bool(value) for value in self.legal_action_mask.tolist()],
            "next_legal_action_mask": [
                bool(value) for value in self.next_legal_action_mask.tolist()
            ],
            "behavior_cloning_action_mask": [
                bool(value) for value in self.behavior_cloning_action_mask.tolist()
            ],
            "reward": float(self.reward),
            "done": bool(self.done),
            "weight": float(self.weight),
            "source": self.source,
            "split_key": self.split_key,
            "state_fingerprint": self.state_fingerprint,
            "next_state_fingerprint": self.next_state_fingerprint,
            "lap_number": int(self.lap_number),
            "elapsed_laps": int(self.elapsed_laps),
            "driver_id": self.driver_id,
            "episode_key": self.episode_key,
            "behavior_cloning_eligible": bool(self.behavior_cloning_eligible),
            "behavior_cloning_ineligible_reasons": list(
                self.behavior_cloning_ineligible_reasons
            ),
            "behavior_cloning_label_kind": self.behavior_cloning_label_kind,
            "behavior_cloning_mode_observed": bool(
                self.behavior_cloning_mode_observed
            ),
            "ood": bool(self.ood),
            "ood_reasons": list(self.ood_reasons),
            "behavior_action_probability": self.behavior_action_probability,
            "propensity_ope_eligible": bool(self.propensity_ope_eligible),
            "propensity_ope_ineligible_reasons": list(
                self.propensity_ope_ineligible_reasons
            ),
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
    def elapsed_laps(self) -> np.ndarray:
        """Semi-Markov duration for each decision transition."""

        return np.asarray([example.elapsed_laps for example in self.examples], dtype=int)

    @property
    def legal_action_masks(self) -> np.ndarray:
        if not self.examples:
            return np.empty((0, self.action_index.size), dtype=bool)
        return np.vstack([example.legal_action_mask for example in self.examples]).astype(bool)

    @property
    def behavior_action_probabilities(self) -> np.ndarray:
        return np.asarray(
            [
                float(example.behavior_action_probability)
                if example.behavior_action_probability is not None
                else np.nan
                for example in self.examples
            ],
            dtype=float,
        )

    def learning_examples(self, *, include_ood: bool = False) -> tuple[RLReplayExample, ...]:
        """Backward-compatible alias for offline-Q learning examples."""

        return self.offline_q_examples(include_ood=include_ood)

    def behavior_cloning_examples(self) -> tuple[RLReplayExample, ...]:
        """Causal observed-action labels; counterfactual legality is not required."""

        return tuple(
            example
            for example in self.examples
            if example.is_valid_for_behavior_cloning
        )

    def offline_q_examples(self, *, include_ood: bool = False) -> tuple[RLReplayExample, ...]:
        """Reward transitions with a certified logged-action legal mask."""

        if include_ood:
            return tuple(example for example in self.examples if example.action_index >= 0)
        return tuple(example for example in self.examples if example.is_valid_for_learning)

    def propensity_ope_examples(self) -> tuple[RLReplayExample, ...]:
        """Rows carrying both training validity and positive logged propensity."""

        return tuple(
            example
            for example in self.examples
            if example.is_valid_for_propensity_ope
        )

    def action_support_diagnostics(self) -> dict[str, object]:
        """Report fail-closed action support for BC, offline-Q, and OPE separately."""

        return {
            "minimum_supported_action_keys": int(MIN_STRATEGY_SUPPORTED_ACTION_KEYS),
            "behavior_cloning": _action_support_payload(
                self.behavior_cloning_examples(),
                action_index=self.action_index,
                use_behavior_cloning_labels=True,
            ),
            "offline_q": _action_support_payload(
                self.offline_q_examples(),
                action_index=self.action_index,
            ),
            "propensity_ope": _action_support_payload(
                self.propensity_ope_examples(),
                action_index=self.action_index,
            ),
        }

    def diagnostics(self) -> dict[str, object]:
        action_counts = {key: 0 for key in self.action_index.keys}
        representative_action_counts = {key: 0 for key in self.action_index.keys}
        source_counts: dict[str, int] = {}
        split_counts: dict[str, int] = {}
        ood_reasons: dict[str, int] = {}
        propensity_ope_ineligible_reasons: dict[str, int] = {}
        behavior_cloning_ineligible_reasons: dict[str, int] = {}
        behavior_cloning_label_kinds: dict[str, int] = {}
        for example in self.examples:
            if example.action_index >= 0:
                representative_action_counts[example.action_key] = int(
                    representative_action_counts.get(example.action_key, 0) + 1
                )
                if example.behavior_cloning_mode_observed:
                    action_counts[example.action_key] = int(
                        action_counts.get(example.action_key, 0) + 1
                    )
            source_counts[example.source] = int(source_counts.get(example.source, 0) + 1)
            split = str(example.split_key or "unknown")
            split_counts[split] = int(split_counts.get(split, 0) + 1)
            for reason in example.ood_reasons:
                ood_reasons[reason] = int(ood_reasons.get(reason, 0) + 1)
            behavior_cloning_label_kinds[example.behavior_cloning_label_kind] = int(
                behavior_cloning_label_kinds.get(
                    example.behavior_cloning_label_kind,
                    0,
                )
                + 1
            )
            for reason in example.behavior_cloning_ineligible_reasons:
                behavior_cloning_ineligible_reasons[reason] = int(
                    behavior_cloning_ineligible_reasons.get(reason, 0) + 1
                )
            for reason in example.propensity_ope_ineligible_reasons:
                propensity_ope_ineligible_reasons[reason] = int(
                    propensity_ope_ineligible_reasons.get(reason, 0) + 1
                )
        action_support = self.action_support_diagnostics()
        return {
            "rows": int(self.rows),
            "behavior_cloning_rows": int(len(self.behavior_cloning_examples())),
            "learning_rows": int(len(self.offline_q_examples())),
            "offline_q_rows": int(len(self.offline_q_examples())),
            "propensity_ope_rows": int(len(self.propensity_ope_examples())),
            "ood_rows": int(sum(1 for example in self.examples if example.ood)),
            "feature_count": int(len(self.feature_names)),
            "action_count": int(self.action_index.size),
            "action_counts": action_counts,
            "representative_action_counts_not_mode_evidence": (
                representative_action_counts
            ),
            "source_counts": source_counts,
            "split_counts": split_counts,
            "ood_reasons": ood_reasons,
            "behavior_cloning_label_kinds": behavior_cloning_label_kinds,
            "behavior_cloning_ineligible_reasons": (
                behavior_cloning_ineligible_reasons
            ),
            "propensity_ope_ineligible_reasons": (
                propensity_ope_ineligible_reasons
            ),
            "action_support": action_support,
            "strategy_training_readiness_gate_pass": bool(
                action_support["offline_q"]["gate_pass"]
            ),
            "metadata": self.metadata,
        }


def _action_support_payload(
    examples: Sequence[RLReplayExample],
    *,
    action_index: StrategyActionIndex,
    use_behavior_cloning_labels: bool = False,
) -> dict[str, object]:
    exact_counts = {key: 0 for key in action_index.keys}
    compatible_counts = {key: 0 for key in action_index.keys}
    family_counts: dict[str, int] = {}
    exact_label_rows = 0
    coarsened_label_rows = 0
    for example in examples:
        family = _action_family_key(
            action_index.action_for(int(example.action_index))
        )
        family_counts[family] = int(family_counts.get(family, 0) + 1)
        if use_behavior_cloning_labels:
            compatible_indices = np.flatnonzero(
                example.behavior_cloning_action_mask
            )
            for idx in compatible_indices:
                key = action_index.action_for(int(idx)).key
                compatible_counts[key] = int(compatible_counts.get(key, 0) + 1)
            if example.behavior_cloning_label_kind == "exact":
                exact_label_rows += 1
                exact_counts[example.action_key] = int(
                    exact_counts.get(example.action_key, 0) + 1
                )
            elif example.behavior_cloning_label_kind == "coarsened_missing_mode":
                coarsened_label_rows += 1
        elif 0 <= int(example.action_index) < action_index.size:
            exact_label_rows += 1
            exact_counts[example.action_key] = int(
                exact_counts.get(example.action_key, 0) + 1
            )
            compatible_counts[example.action_key] = int(
                compatible_counts.get(example.action_key, 0) + 1
            )
    compatible_supported_keys = sorted(
        key for key, count in compatible_counts.items() if int(count) > 0
    )
    exact_supported_keys = sorted(
        key for key, count in exact_counts.items() if int(count) > 0
    )
    supported_keys = compatible_supported_keys
    supported_actions = [
        action_index.action_for(action_index.index_for(key)) for key in supported_keys
    ]
    exact_supported_actions = [
        action_index.action_for(action_index.index_for(key))
        for key in exact_supported_keys
    ]
    supported_types = sorted({action.action_type for action in supported_actions})
    supported_modes = sorted({action.mode for action in exact_supported_actions})
    compatible_modes = sorted({action.mode for action in supported_actions})
    supported_compounds = sorted(
        {str(action.compound) for action in supported_actions if action.compound is not None}
    )
    has_stay_out = any(action.action_type == ACTION_STAY_OUT for action in supported_actions)
    has_pit_action = any(action.action_type in PIT_ACTION_TYPES for action in supported_actions)
    total_families = {
        _action_family_key(action) for action in action_index.actions
    }
    supported_families = sorted(family_counts)
    blockers: list[str] = []
    if len(supported_keys) < int(MIN_STRATEGY_SUPPORTED_ACTION_KEYS):
        blockers.append("fewer_than_two_supported_action_keys")
    if not has_stay_out:
        blockers.append("stay_out_action_support_missing")
    if not has_pit_action:
        blockers.append("pit_action_support_missing")
    return {
        "rows": int(len(examples)),
        "exact_label_rows": int(exact_label_rows),
        "coarsened_label_rows": int(coarsened_label_rows),
        "pace_mode_evidence_rows": int(exact_label_rows),
        "supported_action_count": int(len(supported_keys)),
        "total_action_count": int(action_index.size),
        "supported_action_fraction": float(
            len(supported_keys) / max(1, int(action_index.size))
        ),
        "supported_action_keys": supported_keys,
        "compatible_action_key_count": int(len(compatible_supported_keys)),
        "compatible_action_keys": compatible_supported_keys,
        "exact_action_key_count": int(len(exact_supported_keys)),
        "exact_action_keys": exact_supported_keys,
        "supported_action_family_count": int(len(supported_families)),
        "total_action_family_count": int(len(total_families)),
        "supported_action_family_fraction": float(
            len(supported_families) / max(1, len(total_families))
        ),
        "supported_action_families": supported_families,
        "supported_action_types": supported_types,
        "supported_modes": supported_modes,
        "compatible_modes_not_observed": compatible_modes,
        "supported_compounds": supported_compounds,
        "action_counts": exact_counts,
        "compatible_action_counts": compatible_counts,
        "action_family_counts": family_counts,
        "has_stay_out_support": bool(has_stay_out),
        "has_pit_action_support": bool(has_pit_action),
        "gate_pass": bool(not blockers),
        "blockers": blockers,
    }


def _action_family_key(action: StrategyAction) -> str:
    if action.action_type == ACTION_STAY_OUT:
        return ACTION_STAY_OUT
    return f"{action.action_type}:{action.compound}"


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


def _training_ood_reasons_for(
    record: ReplayBufferRecord,
    *,
    action_index: int,
    legal_mask: np.ndarray,
    stored_legal_mask: np.ndarray,
    legal_mask_constraint_feasible: bool,
    stored_legal_mask_constraint_feasible: bool,
    legal_mask_operational_fallback_applied: bool,
    stored_legal_mask_operational_fallback_applied: bool,
    next_legal_mask: np.ndarray,
    next_legal_mask_constraint_feasible: bool,
    next_legal_mask_operational_fallback_applied: bool,
    next_legal_mask_payload: Mapping[str, object],
    reward: float,
    unclipped_reward: float,
    state_features: np.ndarray,
    next_state_features: np.ndarray,
) -> tuple[str, ...]:
    reasons: list[str] = []
    for error in record.validate(require_legal_action=False):
        reasons.append(f"record_validation:{error}")
    transition_metadata = record.transition.metadata or {}
    explicit_synthetic = _is_explicit_synthetic_source(record.source, transition_metadata)
    if (
        not explicit_synthetic
        and transition_metadata.get("observed_action_mode_known") is not True
    ):
        reasons.append("observed_action_mode_evidence_missing")
    current_mask_input_evidence = legal_action_mask_input_evidence(
        record.transition.state_t
    )
    if not np.array_equal(
        np.asarray(stored_legal_mask, dtype=bool),
        np.asarray(legal_mask, dtype=bool),
    ):
        reasons.append("legal_action_mask_evidence_mismatch")
    if (
        bool(stored_legal_mask_constraint_feasible)
        != bool(legal_mask_constraint_feasible)
        or bool(stored_legal_mask_operational_fallback_applied)
        != bool(legal_mask_operational_fallback_applied)
    ):
        reasons.append("legal_action_mask_feasibility_evidence_mismatch")
    if not legal_mask_constraint_feasible:
        reasons.append("legal_action_mask_constraint_infeasible")
    if legal_mask_operational_fallback_applied:
        reasons.append(
            "legal_action_mask_operational_fallback_not_training_eligible"
        )
    if not record.transition.done and not next_legal_mask_constraint_feasible:
        reasons.append("next_legal_action_mask_constraint_infeasible")
    if (
        not record.transition.done
        and next_legal_mask_operational_fallback_applied
    ):
        reasons.append(
            "next_legal_action_mask_operational_fallback_not_training_eligible"
        )
    if not explicit_synthetic:
        if (
            transition_metadata.get("full_legal_action_mask_certified")
            is not True
            or current_mask_input_evidence["certified"] is not True
        ):
            reasons.append("full_legal_action_mask_not_certified")
            reasons.extend(
                f"legal_action_mask_input_unknown:{blocker}"
                for blocker in current_mask_input_evidence["blockers"]
            )
        stored_current_input_evidence = transition_metadata.get(
            "legal_action_mask_input_evidence"
        )
        if not isinstance(stored_current_input_evidence, Mapping):
            reasons.append("legal_action_mask_input_evidence_missing")
        elif _canonical_payload_digest(stored_current_input_evidence) != (
            _canonical_payload_digest(current_mask_input_evidence)
        ):
            reasons.append("legal_action_mask_input_evidence_mismatch")
    if not explicit_synthetic and not record.transition.done:
        next_mask_input_evidence = legal_action_mask_input_evidence(
            record.transition.state_t1
        )
        if (
            transition_metadata.get("next_full_legal_action_mask_certified")
            is not True
            or next_mask_input_evidence["certified"] is not True
        ):
            reasons.append("next_full_legal_action_mask_not_certified")
            reasons.extend(
                f"next_legal_action_mask_input_unknown:{blocker}"
                for blocker in next_mask_input_evidence["blockers"]
            )
        stored_next_input_evidence = transition_metadata.get(
            "next_legal_action_mask_input_evidence"
        )
        if not isinstance(stored_next_input_evidence, Mapping):
            reasons.append("next_legal_action_mask_input_evidence_missing")
        elif _canonical_payload_digest(stored_next_input_evidence) != (
            _canonical_payload_digest(next_mask_input_evidence)
        ):
            reasons.append("next_legal_action_mask_input_evidence_mismatch")
        stored_next_mask = transition_metadata.get("next_legal_action_mask")
        if not isinstance(stored_next_mask, Mapping):
            reasons.append("next_legal_action_mask_evidence_missing")
        elif _canonical_payload_digest(stored_next_mask) != (
            _canonical_payload_digest(next_legal_mask_payload)
        ):
            reasons.append("next_legal_action_mask_evidence_mismatch")
        stored_next_mask_fingerprint = transition_metadata.get(
            "next_legal_action_mask_fingerprint"
        )
        expected_next_mask_fingerprint = _canonical_payload_digest(
            {"next_legal_action_mask": next_legal_mask_payload}
        )
        if stored_next_mask_fingerprint is None:
            reasons.append("next_legal_action_mask_fingerprint_missing")
        elif str(stored_next_mask_fingerprint) != expected_next_mask_fingerprint:
            reasons.append("next_legal_action_mask_fingerprint_mismatch")
    boundary_status = transition_metadata.get(
        "causal_transition_boundary_status"
    )
    if not explicit_synthetic and boundary_status is None:
        reasons.append("causal_transition_boundary_evidence_missing")
    elif boundary_status != "valid" and boundary_status is not None:
        reasons.append("causal_transition_boundary_invalid")
        for blocker in transition_metadata.get(
            "causal_transition_boundary_blockers", ()
        ):
            reasons.append(f"causal_transition_boundary_blocker:{blocker}")
    legality_status = transition_metadata.get("action_legality_status")
    if not explicit_synthetic and legality_status is None:
        reasons.append("observed_action_legality_evidence_missing")
    elif legality_status == "unknown":
        reasons.append("observed_action_legality_unknown")
    reward_observation_status = transition_metadata.get(
        "reward_observation_status"
    )
    if not explicit_synthetic and reward_observation_status is None:
        reasons.append("reward_observation_evidence_missing")
    elif (
        not explicit_synthetic
        and reward_observation_status != "observed_required_components"
    ):
        reasons.append("reward_observation_not_certified")
        for blocker in transition_metadata.get(
            "reward_observation_blockers",
            (),
        ):
            reasons.append(f"reward_observation_blocker:{blocker}")
    eligibility = transition_metadata.get("policy_training_eligible")
    if eligibility is None:
        eligibility = transition_metadata.get("policy_learning_eligible")
    if not explicit_synthetic and eligibility is None:
        reasons.append("policy_training_eligibility_evidence_missing")
    elif eligibility is not None and not (
        isinstance(eligibility, bool) and eligibility
    ):
        reasons.append("policy_training_declared_ineligible")
        blockers = transition_metadata.get(
            "policy_training_blockers",
            transition_metadata.get("policy_learning_blockers", ()),
        )
        for blocker in blockers:
            reasons.append(f"policy_training_blocker:{blocker}")
    if action_index < 0:
        reasons.append("action_not_in_action_index")
    elif action_index >= legal_mask.size or not bool(legal_mask[action_index]):
        if transition_metadata.get("action_legality_status") == "known_illegal":
            reasons.append("observed_action_illegal_under_mask")
        elif transition_metadata.get("action_legality_status") == "known_legal":
            reasons.append("observed_action_legality_mask_conflict")
    if reward != unclipped_reward:
        reasons.append("reward_clipped")
    if not np.isfinite(state_features).all():
        reasons.append("state_features_non_finite")
    if not np.isfinite(next_state_features).all():
        reasons.append("next_state_features_non_finite")
    if not legal_mask.any():
        reasons.append("no_legal_actions_after_alignment")
    if not record.transition.done and not next_legal_mask.any():
        reasons.append("no_next_legal_actions_after_alignment")
    return tuple(dict.fromkeys(reasons))


def _behavior_cloning_label_for(
    record: ReplayBufferRecord,
    *,
    action_index: StrategyActionIndex,
    observed_action_index: int,
    legal_action_mask: LegalActionMask,
    state_features: np.ndarray,
    next_state_features: np.ndarray,
) -> tuple[np.ndarray, str, tuple[str, ...]]:
    """Build an observed-action target without fabricating an unobserved mode."""

    mask = np.zeros(action_index.size, dtype=bool)
    reasons: list[str] = []
    for error in record.validate(require_legal_action=False):
        reasons.append(f"record_validation:{error}")
    if not legal_action_mask.constraint_feasible:
        reasons.append("legal_action_mask_constraint_infeasible")
    if legal_action_mask.operational_fallback_applied:
        reasons.append(
            "legal_action_mask_operational_fallback_not_behavior_cloning_eligible"
        )
    transition = record.transition
    metadata = transition.metadata or {}
    explicit_synthetic = _is_explicit_synthetic_source(record.source, metadata)
    boundary_status = metadata.get("causal_transition_boundary_status")
    if not explicit_synthetic and boundary_status is None:
        reasons.append("causal_transition_boundary_evidence_missing")
    elif boundary_status not in (None, "valid"):
        reasons.append("causal_transition_boundary_invalid")
    if observed_action_index < 0:
        reasons.append("action_not_in_action_index")
    if not np.isfinite(state_features).all():
        reasons.append("state_features_non_finite")
    if not np.isfinite(next_state_features).all():
        reasons.append("next_state_features_non_finite")

    mode_observed = bool(
        explicit_synthetic or metadata.get("observed_action_mode_known") is True
    )
    if not reasons and observed_action_index >= 0:
        observed = transition.action_t
        if mode_observed:
            mask[int(observed_action_index)] = True
            label_kind = "exact"
        else:
            for idx, candidate in enumerate(action_index.actions):
                if (
                    candidate.action_type == observed.action_type
                    and candidate.compound == observed.compound
                ):
                    mask[idx] = True
            label_kind = "coarsened_missing_mode"
            if int(np.sum(mask)) < 2:
                reasons.append("coarsened_mode_label_has_no_alternative_mode")
    else:
        label_kind = "ineligible"
    if not mask.any() and not reasons:
        reasons.append("behavior_cloning_label_mask_empty")
    if reasons:
        label_kind = "ineligible"
    return mask, label_kind, tuple(dict.fromkeys(reasons))


def _propensity_ope_ineligible_reasons_for(
    record: ReplayBufferRecord,
    *,
    training_reasons: Sequence[str],
) -> tuple[str, ...]:
    """Keep propensity requirements out of BC/offline-Q training validity."""

    reasons: list[str] = []
    metadata = record.transition.metadata or {}
    support_status = metadata.get("behavior_action_support_status")
    if support_status is None:
        reasons.append("behavior_action_support_evidence_missing")
    elif support_status == "unknown":
        reasons.append("behavior_action_support_unknown")
    elif support_status == "zero_support":
        reasons.append("behavior_action_zero_support")
    elif support_status == "invalid":
        reasons.append("behavior_action_support_invalid")

    raw_probability = metadata.get(
        "behavior_action_probability",
        record.transition.state_t.metadata.get("behavior_action_probability"),
    )
    try:
        probability = float(raw_probability)
    except (TypeError, ValueError):
        probability = float("nan")
    if raw_probability is None:
        reasons.append("behavior_action_probability_missing")
    elif not np.isfinite(probability) or probability > 1.0:
        reasons.append("behavior_action_probability_invalid")
    elif probability <= 0.0:
        reasons.append("behavior_action_probability_zero_support")

    eligibility = metadata.get("propensity_ope_eligible")
    if eligibility is None and "policy_training_eligible" not in metadata:
        # A legacy policy-learning flag represented the old stricter
        # legality-plus-positive-propensity contract.
        eligibility = metadata.get("policy_learning_eligible")
    if eligibility is None:
        reasons.append("propensity_ope_eligibility_evidence_missing")
    elif not (isinstance(eligibility, bool) and eligibility):
        reasons.append("propensity_ope_declared_ineligible")
        blockers = metadata.get("propensity_ope_blockers", ())
        for blocker in blockers:
            reasons.append(f"propensity_ope_blocker:{blocker}")
    if training_reasons:
        reasons.append("policy_training_ineligible")
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
        legal_mask_contract = build_legal_action_mask(
            transition.state_t,
            action_space=mapping.actions,
            config=mapping.action_mask_config,
        )
        legal_mask = mapping.legal_mask_from(legal_mask_contract)
        stored_legal_mask = mapping.legal_mask_from(
            transition.legal_action_mask
        )
        next_legal_mask_contract = build_legal_action_mask(
            transition.state_t1,
            action_space=mapping.actions,
            config=mapping.action_mask_config,
        )
        next_legal_mask = mapping.legal_mask_from(next_legal_mask_contract)
        next_legal_mask_payload = next_legal_mask_contract.to_payload()
        raw_reward = _finite(transition.reward_t.value, 0.0)
        reward = float(np.clip(raw_reward, low, high))
        (
            behavior_cloning_mask,
            behavior_cloning_label_kind,
            behavior_cloning_reasons,
        ) = _behavior_cloning_label_for(
            record,
            action_index=mapping,
            observed_action_index=action_idx,
            legal_action_mask=legal_mask_contract,
            state_features=state_features,
            next_state_features=next_state_features,
        )
        reasons = _training_ood_reasons_for(
            record,
            action_index=action_idx,
            legal_mask=legal_mask,
            stored_legal_mask=stored_legal_mask,
            legal_mask_constraint_feasible=bool(
                legal_mask_contract.constraint_feasible
            ),
            stored_legal_mask_constraint_feasible=bool(
                transition.legal_action_mask.constraint_feasible
            ),
            legal_mask_operational_fallback_applied=bool(
                legal_mask_contract.operational_fallback_applied
            ),
            stored_legal_mask_operational_fallback_applied=bool(
                transition.legal_action_mask.operational_fallback_applied
            ),
            next_legal_mask=next_legal_mask,
            next_legal_mask_constraint_feasible=bool(
                next_legal_mask_contract.constraint_feasible
            ),
            next_legal_mask_operational_fallback_applied=bool(
                next_legal_mask_contract.operational_fallback_applied
            ),
            next_legal_mask_payload=next_legal_mask_payload,
            reward=reward,
            unclipped_reward=raw_reward,
            state_features=state_features,
            next_state_features=next_state_features,
        )
        propensity_ope_reasons = _propensity_ope_ineligible_reasons_for(
            record,
            training_reasons=reasons,
        )
        if strict and reasons:
            raise ValueError(f"RL replay record {record.record_id} is not learnable: {', '.join(reasons)}")
        merged_metadata = {
            "record_metadata": dict(record.metadata or {}),
            "transition_metadata": dict(transition.metadata or {}),
            "reward_components": dict(transition.reward_t.components or {}),
        }
        raw_behavior_probability = transition.metadata.get(
            "behavior_action_probability",
            transition.state_t.metadata.get("behavior_action_probability"),
        )
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
            elapsed_laps=int(
                transition.metadata.get(
                    "elapsed_laps",
                    max(1, transition.state_t1.lap_number - transition.state_t.lap_number),
                )
            ),
            driver_id=transition.state_t.driver_id,
            episode_key=_episode_key(record),
            metadata=merged_metadata,
            behavior_cloning_action_mask=behavior_cloning_mask,
            behavior_cloning_eligible=not behavior_cloning_reasons,
            behavior_cloning_ineligible_reasons=behavior_cloning_reasons,
            behavior_cloning_label_kind=behavior_cloning_label_kind,
            behavior_cloning_mode_observed=bool(
                behavior_cloning_label_kind == "exact"
            ),
            ood=bool(reasons),
            ood_reasons=reasons,
            behavior_action_probability=raw_behavior_probability,
            propensity_ope_eligible=not propensity_ope_reasons,
            propensity_ope_ineligible_reasons=propensity_ope_reasons,
        )
        examples.append(example)

    return RLReplayDataset(
        examples=tuple(examples),
        action_index=mapping,
        feature_names=names,
        metadata={
            "dataset_builder": REPLAY_DATASET_SCHEMA_VERSION,
            "transition_duration_semantics": "elapsed_laps",
            "reward_duration_semantics": REWARD_SEMANTICS,
            "bellman_discount_semantics": (
                "aggregated_multi_lap_rewards_require_gamma_one_unless_per_lap_rewards_exist"
            ),
            "training_eligibility_semantics": (
                "causal_transition_exact_observed_action_mode_all_legality_driving_inputs_certified_for_current_and_nonterminal_next_masks_and_reward"
            ),
            "behavior_cloning_eligibility_semantics": (
                "causal_observed_action_family_with_partial_label_over_unobserved_pace_mode"
            ),
            "propensity_ope_eligibility_semantics": (
                "training_eligible_plus_positive_logged_action_probability"
            ),
            "reward_clip": (low, high),
            **dict(metadata or {}),
        },
    )


__all__ = [
    "COMPOUND_FEATURES",
    "DEFAULT_STATE_FEATURE_NAMES",
    "MIN_STRATEGY_SUPPORTED_ACTION_KEYS",
    "REPLAY_DATASET_SCHEMA_VERSION",
    "RLReplayDataset",
    "RLReplayExample",
    "StrategyActionIndex",
    "bucket_state_features",
    "build_rl_replay_dataset",
    "state_to_feature_vector",
]
