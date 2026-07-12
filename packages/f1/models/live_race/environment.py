"""Typed live-race strategy environment contracts.

This module is not the Phase 5 simulator.  It provides the stable replay and
transition records that the simulator, DP/MPC planner, and masked policies share.
All state builders intentionally use only fields available through the current
lap row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from packages.f1.models.live_race.action_space import (
    ACTION_PIT_NEXT_LAP,
    ACTION_PIT_NOW,
    ACTION_STAY_OUT,
    ActionMaskConfig,
    LegalActionMask,
    StrategyAction,
    build_legal_action_mask,
    normalize_compound,
)
from packages.f1.models.live_race.state import parse_track_status


LEAKAGE_CONTRACT_VERSION = "live_strategy_state_v1_no_future_lap_fields"

FUTURE_ONLY_COLUMNS: tuple[str, ...] = (
    "classified_position",
    "final_position",
    "finish_position",
    "race_result",
    "result_position",
    "points_finish",
    "future_lap_time_seconds",
    "next_actual_lap_time_seconds",
)


def _json_safe(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _stable_digest(payload: Mapping[str, object]) -> str:
    text = json.dumps(_json_safe(dict(payload)), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: object, default: int = 0) -> int:
    numeric = _safe_float(value, float(default))
    if not np.isfinite(numeric):
        return int(default)
    return int(numeric)


def _safe_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return bool(default)
    try:
        if pd.isna(value):
            return bool(default)
    except Exception:
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(float(value) != 0.0)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f", ""}:
        return False
    return bool(default)


def _first(row: Mapping[str, object], keys: Iterable[str], default: object = None) -> object:
    for key in keys:
        if key in row:
            value = row.get(key)
            try:
                if value is not None and not pd.isna(value):
                    return value
            except Exception:
                if value is not None:
                    return value
    return default


def _tuple_compounds(value: object, fallback: object = None) -> tuple[str, ...]:
    raw: list[object]
    if value is None:
        raw = [] if fallback is None else [fallback]
    elif isinstance(value, str):
        raw = [item.strip() for item in value.replace("|", ",").split(",")]
    elif isinstance(value, Sequence):
        raw = list(value)
    else:
        raw = [value]
    compounds = tuple(
        compound for compound in (normalize_compound(item) for item in raw) if compound != "UNKNOWN"
    )
    return tuple(dict.fromkeys(compounds))


def _row_payload(row: Mapping[str, object] | pd.Series) -> dict[str, object]:
    if isinstance(row, pd.Series):
        return row.to_dict()
    return dict(row)


@dataclass(frozen=True)
class StrategyState:
    """Information available to a live strategy policy through lap ``t``."""

    event_key: Optional[int]
    driver_id: str
    lap_number: int
    total_laps: Optional[int]
    remaining_laps: Optional[int]
    stint_id: int
    compound: str
    tyre_age: int
    used_compounds: tuple[str, ...]
    race_time_seconds: Optional[float] = None
    gap_to_leader_seconds: Optional[float] = None
    position: Optional[int] = None
    track_status: str = ""
    is_red: bool = False
    is_sc_vsc: bool = False
    is_yellow: bool = False
    is_greenish: bool = False
    pace_penalty_mean: float = 0.0
    pace_penalty_std: float = 0.0
    deg_rate_mean: float = 0.04
    deg_rate_std: float = 0.0
    next_lap_mean: Optional[float] = None
    next_lap_std: Optional[float] = None
    pit_loss_estimate_seconds: Optional[float] = None
    circuit_overtaking_difficulty: Optional[float] = None
    circuit_tyre_degradation: Optional[float] = None
    circuit_safety_car_probability: Optional[float] = None
    circuit_strategy_variance: Optional[float] = None
    track_overtake_propensity: Optional[float] = None
    track_chaos_index: Optional[float] = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lap_number = max(0, int(self.lap_number))
        total_laps = None if self.total_laps is None else max(0, int(self.total_laps))
        remaining = self.remaining_laps
        if remaining is None and total_laps is not None:
            remaining = max(0, int(total_laps) - int(lap_number))
        elif remaining is not None:
            remaining = max(0, int(remaining))

        compound = normalize_compound(self.compound)
        if compound == "UNKNOWN":
            compound = "UNKNOWN"
        used = tuple(dict.fromkeys(_tuple_compounds(self.used_compounds, fallback=compound)))
        metadata = dict(self.metadata or {})
        metadata.setdefault("available_through_lap", lap_number)
        metadata.setdefault("leakage_contract_version", LEAKAGE_CONTRACT_VERSION)

        object.__setattr__(self, "lap_number", lap_number)
        object.__setattr__(self, "total_laps", total_laps)
        object.__setattr__(self, "remaining_laps", remaining)
        object.__setattr__(self, "stint_id", max(0, int(self.stint_id)))
        object.__setattr__(self, "compound", compound)
        object.__setattr__(self, "tyre_age", max(0, int(self.tyre_age)))
        object.__setattr__(self, "used_compounds", used)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_mapping(cls, row: Mapping[str, object] | pd.Series, *, metadata: Optional[Mapping[str, object]] = None) -> "StrategyState":
        payload = _row_payload(row)
        lap_number = _safe_int(_first(payload, ("lap_number", "lap_last", "LapNumber"), 0), 0)
        total_laps_raw = _first(payload, ("total_laps", "race_total_laps", "scheduled_laps"), None)
        total_laps = None if total_laps_raw is None else _safe_int(total_laps_raw, 0)
        remaining_raw = _first(payload, ("remaining_laps", "laps_remaining"), None)
        remaining = None if remaining_raw is None else _safe_int(remaining_raw, 0)

        track_status = str(_first(payload, ("track_status", "TrackStatus"), "") or "")
        flags = parse_track_status(track_status)
        compound = normalize_compound(_first(payload, ("compound", "Compound", "compound_normalized"), "UNKNOWN"))
        used_compounds = _tuple_compounds(
            _first(payload, ("used_compounds", "compounds_used"), None),
            fallback=compound,
        )
        meta = dict(metadata or {})
        meta.update(
            {
                "source_row_index": payload.get("_source_row_index"),
                "available_through_lap": lap_number,
                "ignored_future_columns": tuple(column for column in FUTURE_ONLY_COLUMNS if column in payload),
            }
        )
        available = _first(payload, ("available_compounds", "allowed_compounds"), None)
        if available is not None:
            meta["available_compounds"] = _tuple_compounds(available)
        for state_key, aliases in {
            "pit_lane_open": ("pit_lane_open",),
            "is_box_lap": ("is_box_lap", "box_lap"),
            "is_wet_track": ("is_wet_track",),
            "weather_is_wet": ("weather_is_wet",),
            "rain_expected": ("rain_expected",),
            "track_status_is_wet": ("track_status_is_wet",),
        }.items():
            raw_value = _first(payload, aliases, None)
            if raw_value is not None:
                meta[state_key] = _safe_bool(raw_value, False)

        event_key_raw = _first(payload, ("event_key", "meeting_key"), None)
        event_key = None if event_key_raw is None else _safe_int(event_key_raw, 0)
        position_raw = _first(payload, ("position", "rank", "running_position"), None)
        position = None if position_raw is None else _safe_int(position_raw, 0)

        return cls(
            event_key=event_key,
            driver_id=str(_first(payload, ("driver_id", "Driver", "DriverNumber"), "driver")),
            lap_number=lap_number,
            total_laps=total_laps,
            remaining_laps=remaining,
            stint_id=_safe_int(_first(payload, ("stint_id", "Stint"), 1), 1),
            compound=compound,
            tyre_age=_safe_int(_first(payload, ("tyre_age", "TyreAge", "TyreLife", "tyre_life_raw"), 0), 0),
            used_compounds=used_compounds,
            race_time_seconds=_none_if_nan(_safe_float(_first(payload, ("race_time_seconds", "RaceTimeSeconds"), None))),
            gap_to_leader_seconds=_none_if_nan(
                _safe_float(_first(payload, ("gap_to_leader_seconds", "gap_to_leader"), None))
            ),
            position=position,
            track_status=track_status,
            is_red=_safe_bool(_first(payload, ("is_red",), flags.is_red), flags.is_red),
            is_sc_vsc=_safe_bool(_first(payload, ("is_sc_vsc",), flags.is_sc_vsc), flags.is_sc_vsc),
            is_yellow=_safe_bool(_first(payload, ("is_yellow",), flags.is_yellow), flags.is_yellow),
            is_greenish=_safe_bool(_first(payload, ("is_greenish",), flags.is_greenish), flags.is_greenish),
            pace_penalty_mean=_safe_float(_first(payload, ("pace_penalty_mean", "rolling_clean_pace_delta_3"), 0.0), 0.0),
            pace_penalty_std=_safe_float(_first(payload, ("pace_penalty_std",), 0.0), 0.0),
            deg_rate_mean=_safe_float(_first(payload, ("deg_rate_mean", "estimated_deg_slope_5", "compound_deg_prior"), 0.04), 0.04),
            deg_rate_std=_safe_float(_first(payload, ("deg_rate_std",), 0.0), 0.0),
            next_lap_mean=_none_if_nan(_safe_float(_first(payload, ("next_lap_mean",), None))),
            next_lap_std=_none_if_nan(_safe_float(_first(payload, ("next_lap_std",), None))),
            pit_loss_estimate_seconds=_none_if_nan(
                _safe_float(_first(payload, ("pit_loss_estimate_seconds", "pit_loss_seconds"), None))
            ),
            circuit_overtaking_difficulty=_none_if_nan(
                _safe_float(_first(payload, ("circuit_overtaking_difficulty",), None))
            ),
            circuit_tyre_degradation=_none_if_nan(_safe_float(_first(payload, ("circuit_tyre_degradation",), None))),
            circuit_safety_car_probability=_none_if_nan(
                _safe_float(_first(payload, ("circuit_safety_car_probability",), None))
            ),
            circuit_strategy_variance=_none_if_nan(_safe_float(_first(payload, ("circuit_strategy_variance",), None))),
            track_overtake_propensity=_none_if_nan(_safe_float(_first(payload, ("track_overtake_propensity",), None))),
            track_chaos_index=_none_if_nan(_safe_float(_first(payload, ("track_chaos_index",), None))),
            metadata=meta,
        )

    def as_feature_dict(self) -> dict[str, object]:
        return {
            "event_key": self.event_key,
            "driver_id": self.driver_id,
            "lap_number": self.lap_number,
            "total_laps": self.total_laps,
            "remaining_laps": self.remaining_laps,
            "stint_id": self.stint_id,
            "compound": self.compound,
            "tyre_age": self.tyre_age,
            "used_compounds": list(self.used_compounds),
            "race_time_seconds": self.race_time_seconds,
            "gap_to_leader_seconds": self.gap_to_leader_seconds,
            "position": self.position,
            "track_status": self.track_status,
            "is_red": self.is_red,
            "is_sc_vsc": self.is_sc_vsc,
            "is_yellow": self.is_yellow,
            "is_greenish": self.is_greenish,
            "pace_penalty_mean": self.pace_penalty_mean,
            "pace_penalty_std": self.pace_penalty_std,
            "deg_rate_mean": self.deg_rate_mean,
            "deg_rate_std": self.deg_rate_std,
            "next_lap_mean": self.next_lap_mean,
            "next_lap_std": self.next_lap_std,
            "pit_loss_estimate_seconds": self.pit_loss_estimate_seconds,
            "circuit_overtaking_difficulty": self.circuit_overtaking_difficulty,
            "circuit_tyre_degradation": self.circuit_tyre_degradation,
            "circuit_safety_car_probability": self.circuit_safety_car_probability,
            "circuit_strategy_variance": self.circuit_strategy_variance,
            "track_overtake_propensity": self.track_overtake_propensity,
            "track_chaos_index": self.track_chaos_index,
            "available_through_lap": self.metadata.get("available_through_lap"),
            "leakage_contract_version": self.metadata.get("leakage_contract_version"),
        }

    def fingerprint(self) -> str:
        return _stable_digest(self.as_feature_dict())


def _none_if_nan(value: float) -> Optional[float]:
    if value is None:
        return None
    try:
        if not np.isfinite(float(value)):
            return None
    except Exception:
        return None
    return float(value)


@dataclass(frozen=True)
class StrategyReward:
    """Scalar policy reward plus auditable components."""

    value: float
    components: dict[str, float] = field(default_factory=dict)
    note: Optional[str] = None

    def to_payload(self) -> dict[str, object]:
        return {
            "value": float(self.value),
            "components": {str(key): float(value) for key, value in self.components.items()},
            "note": self.note,
        }


@dataclass(frozen=True)
class RewardConfig:
    race_time_weight: float = 1.0
    position_gain_weight: float = 2.0
    pit_action_penalty: float = 0.0
    illegal_action_penalty: float = 1000.0


def compute_transition_reward(
    state_t: StrategyState,
    action_t: StrategyAction,
    state_t1: StrategyState,
    *,
    legal_action_mask: Optional[LegalActionMask] = None,
    config: RewardConfig | None = None,
) -> StrategyReward:
    """Reward lower elapsed race time and position gains under the same replay."""

    cfg = config or RewardConfig()
    race_time_delta = 0.0
    if state_t.race_time_seconds is not None and state_t1.race_time_seconds is not None:
        race_time_delta = max(0.0, float(state_t1.race_time_seconds) - float(state_t.race_time_seconds))
    elif state_t1.next_lap_mean is not None:
        race_time_delta = max(0.0, float(state_t1.next_lap_mean))
    elif state_t.next_lap_mean is not None:
        race_time_delta = max(0.0, float(state_t.next_lap_mean))

    position_gain = 0.0
    if state_t.position is not None and state_t1.position is not None:
        position_gain = float(state_t.position - state_t1.position)

    pit_penalty = float(cfg.pit_action_penalty) if action_t.is_pit_action else 0.0
    illegal_penalty = 0.0
    if legal_action_mask is not None and not legal_action_mask.is_legal(action_t):
        illegal_penalty = float(cfg.illegal_action_penalty)

    value = (
        -(float(cfg.race_time_weight) * race_time_delta)
        + (float(cfg.position_gain_weight) * position_gain)
        - pit_penalty
        - illegal_penalty
    )
    return StrategyReward(
        value=float(value),
        components={
            "race_time_delta_seconds": float(race_time_delta),
            "position_gain": float(position_gain),
            "pit_action_penalty": float(pit_penalty),
            "illegal_action_penalty": float(illegal_penalty),
        },
    )


@dataclass(frozen=True)
class StrategyTransition:
    """Replay/environment transition consumed by planners and policies."""

    state_t: StrategyState
    action_t: StrategyAction
    reward_t: StrategyReward
    state_t1: StrategyState
    done: bool
    legal_action_mask: LegalActionMask
    metadata: dict[str, object] = field(default_factory=dict)

    def is_action_legal(self) -> bool:
        return self.legal_action_mask.is_legal(self.action_t)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.state_t1.lap_number < self.state_t.lap_number:
            errors.append("state_t1_lap_before_state_t")
        if self.state_t1.driver_id != self.state_t.driver_id:
            errors.append("driver_id_changed")
        if len(self.legal_action_mask.actions) != len(self.legal_action_mask.mask):
            errors.append("legal_action_mask_shape_mismatch")
        available_lap = _safe_int(self.state_t.metadata.get("available_through_lap"), self.state_t.lap_number)
        if available_lap > self.state_t.lap_number:
            errors.append("state_t_available_through_future_lap")
        ignored = self.state_t.metadata.get("ignored_future_columns")
        if ignored is None:
            errors.append("missing_ignored_future_columns_marker")
        return errors

    def to_payload(self) -> dict[str, object]:
        return {
            "state_t": self.state_t.as_feature_dict(),
            "action_t": self.action_t.to_payload(),
            "reward_t": self.reward_t.to_payload(),
            "state_t1": self.state_t1.as_feature_dict(),
            "done": bool(self.done),
            "legal_action_mask": self.legal_action_mask.to_payload(),
            "metadata": _json_safe(self.metadata),
            "transition_fingerprint": self.fingerprint(),
        }

    def fingerprint(self) -> str:
        return _stable_digest(
            {
                "state_t": self.state_t.as_feature_dict(),
                "action_t": self.action_t.to_payload(),
                "state_t1_lap": self.state_t1.lap_number,
                "done": bool(self.done),
                "legal_keys": [action.key for action in self.legal_action_mask.legal_actions],
            }
        )


def infer_observed_action(row_t: Mapping[str, object] | pd.Series, row_t1: Mapping[str, object] | pd.Series) -> StrategyAction:
    """Infer a replay action from explicit action fields or stint/compound change."""

    current = _row_payload(row_t)
    nxt = _row_payload(row_t1)
    explicit = _first(current, ("observed_action", "action_t", "recommended_action"), None)
    explicit_compound = _first(current, ("action_compound", "next_compound", "pit_compound"), None)
    explicit_mode = _first(current, ("action_mode", "strategy_mode"), "conservative")
    if explicit is not None:
        text = str(explicit).strip()
        if text in {ACTION_STAY_OUT, "stay", "none"}:
            return StrategyAction(ACTION_STAY_OUT, mode=explicit_mode)
        if text in {ACTION_PIT_NOW, "pit", "box"}:
            compound = explicit_compound or _first(nxt, ("compound", "Compound"), None)
            return StrategyAction(ACTION_PIT_NOW, compound=compound, mode=explicit_mode)
        if text in {ACTION_PIT_NEXT_LAP, "pit_next", "box_next"}:
            compound = explicit_compound or _first(nxt, ("compound", "Compound"), None)
            return StrategyAction(ACTION_PIT_NEXT_LAP, compound=compound, mode=explicit_mode)
        return StrategyAction.from_key(text)

    current_stint = _safe_int(_first(current, ("stint_id", "Stint"), 1), 1)
    next_stint = _safe_int(_first(nxt, ("stint_id", "Stint"), current_stint), current_stint)
    current_compound = normalize_compound(_first(current, ("compound", "Compound"), "UNKNOWN"))
    next_compound = normalize_compound(_first(nxt, ("compound", "Compound"), current_compound))
    next_is_box = _safe_bool(_first(nxt, ("is_box_lap", "IsBoxLap"), False), False)
    if next_stint > current_stint or (next_compound != current_compound and next_compound != "UNKNOWN") or next_is_box:
        return StrategyAction(ACTION_PIT_NOW, compound=next_compound if next_compound != "UNKNOWN" else current_compound)
    return StrategyAction(ACTION_STAY_OUT)


def build_replay_transitions(
    laps: pd.DataFrame,
    *,
    action_space: Optional[Sequence[StrategyAction]] = None,
    action_mask_config: ActionMaskConfig | None = None,
    reward_config: RewardConfig | None = None,
) -> list[StrategyTransition]:
    """Build no-leakage replay transitions from lap-level rows.

    Future-only result columns may exist in ``laps``; they are explicitly ignored
    by ``StrategyState.from_mapping`` and only listed in metadata for auditing.
    """

    if not isinstance(laps, pd.DataFrame):
        raise TypeError("laps must be a pandas DataFrame")
    if laps.empty:
        return []
    frame = laps.copy()
    frame["_source_row_index"] = frame.index.astype(str)
    lap_col = "lap_number" if "lap_number" in frame.columns else "lap_last"
    sort_cols = [col for col in ("driver_id", lap_col, "timestamp") if col in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols, kind="mergesort")

    transitions: list[StrategyTransition] = []
    group_col = "driver_id" if "driver_id" in frame.columns else None
    grouped = frame.groupby(group_col, sort=False) if group_col else [(None, frame)]
    for _, group in grouped:
        rows = [row for _, row in group.iterrows()]
        for idx in range(len(rows) - 1):
            row_t = rows[idx]
            row_t1 = rows[idx + 1]
            state_t = StrategyState.from_mapping(row_t)
            state_t1 = StrategyState.from_mapping(row_t1)
            action = infer_observed_action(row_t, row_t1)
            legal_mask = build_legal_action_mask(
                state_t,
                action_space=action_space,
                config=action_mask_config,
            )
            reward = compute_transition_reward(
                state_t,
                action,
                state_t1,
                legal_action_mask=legal_mask,
                config=reward_config,
            )
            done = bool(state_t1.remaining_laps == 0) if state_t1.remaining_laps is not None else idx == len(rows) - 2
            transition = StrategyTransition(
                state_t=state_t,
                action_t=action,
                reward_t=reward,
                state_t1=state_t1,
                done=done,
                legal_action_mask=legal_mask,
                metadata={
                    "source": "lap_replay",
                    "row_t_index": row_t.get("_source_row_index"),
                    "row_t1_index": row_t1.get("_source_row_index"),
                    "leakage_contract_version": LEAKAGE_CONTRACT_VERSION,
                },
            )
            transitions.append(transition)
    return transitions


def transition_prefix_fingerprint(transitions: Iterable[StrategyTransition], *, cutoff_lap: int) -> str:
    payload = [
        transition.fingerprint()
        for transition in transitions
        if int(transition.state_t.lap_number) <= int(cutoff_lap)
    ]
    return _stable_digest({"cutoff_lap": int(cutoff_lap), "transitions": payload})


def assert_replay_prefix_invariant(
    reference: Iterable[StrategyTransition],
    candidate: Iterable[StrategyTransition],
    *,
    cutoff_lap: int,
) -> bool:
    return transition_prefix_fingerprint(reference, cutoff_lap=cutoff_lap) == transition_prefix_fingerprint(
        candidate,
        cutoff_lap=cutoff_lap,
    )


__all__ = [
    "FUTURE_ONLY_COLUMNS",
    "LEAKAGE_CONTRACT_VERSION",
    "RewardConfig",
    "StrategyReward",
    "StrategyState",
    "StrategyTransition",
    "assert_replay_prefix_invariant",
    "build_replay_transitions",
    "compute_transition_reward",
    "infer_observed_action",
    "transition_prefix_fingerprint",
]
