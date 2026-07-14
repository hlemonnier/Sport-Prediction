"""Legal action space for live F1 race strategy policies.

The action space is deliberately discrete.  It is the shared contract for the
deterministic baseline, DP/MPC challengers, and later masked RL policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


ACTION_STAY_OUT = "stay_out"
ACTION_PIT_NOW = "pit_now"
ACTION_PIT_NEXT_LAP = "pit_next_lap"

ACTION_TYPES: tuple[str, ...] = (ACTION_STAY_OUT, ACTION_PIT_NOW, ACTION_PIT_NEXT_LAP)
PIT_ACTION_TYPES: tuple[str, ...] = (ACTION_PIT_NOW, ACTION_PIT_NEXT_LAP)
DRY_COMPOUNDS: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD")
WET_COMPOUNDS: tuple[str, ...] = ("INTER", "WET")
KNOWN_COMPOUNDS: tuple[str, ...] = (*DRY_COMPOUNDS, *WET_COMPOUNDS)
STRATEGY_MODES: tuple[str, ...] = ("conservative", "aggressive")
LEGAL_ACTION_MASK_SCHEMA_VERSION = (
    "live_strategy_legal_action_mask_v2_constraint_feasibility_separated_from_operational_fallback"
)


def normalize_compound(value: object) -> str:
    """Normalize public/FastF1 compound labels into the live strategy contract."""

    text = str(value or "").strip().upper()
    if text in {"S", "SOFT", "C4", "C5"}:
        return "SOFT"
    if text in {"M", "MEDIUM", "C3"}:
        return "MEDIUM"
    if text in {"H", "HARD", "C1", "C2"}:
        return "HARD"
    if "INTER" in text:
        return "INTER"
    if "WET" in text:
        return "WET"
    return "UNKNOWN"


def normalize_mode(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in STRATEGY_MODES:
        return text
    if text in {"attack", "push", "risk_on"}:
        return "aggressive"
    return "conservative"


def is_dry_compound(value: object) -> bool:
    return normalize_compound(value) in DRY_COMPOUNDS


def is_wet_compound(value: object) -> bool:
    return normalize_compound(value) in WET_COMPOUNDS


def _state_value(state: Any, key: str, default: Any = None) -> Any:
    if isinstance(state, Mapping):
        return state.get(key, default)
    if hasattr(state, key):
        return getattr(state, key)
    metadata = getattr(state, "metadata", None)
    if isinstance(metadata, Mapping) and key in metadata:
        return metadata.get(key, default)
    return default


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


def _as_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw = [item.strip() for item in value.replace("|", ",").split(",")]
    elif isinstance(value, Sequence):
        raw = [str(item).strip() for item in value]
    else:
        raw = [str(value).strip()]
    return tuple(item for item in raw if item)


@dataclass(frozen=True)
class StrategyAction:
    """One discrete live-strategy action.

    ``compound`` is intentionally absent for stay-out actions.  For pit actions
    it is mandatory and normalized to the shared compound vocabulary.
    """

    action_type: str
    compound: Optional[str] = None
    mode: str = "conservative"

    def __post_init__(self) -> None:
        action_type = str(self.action_type or "").strip().lower()
        if action_type not in ACTION_TYPES:
            raise ValueError(f"unsupported action_type={self.action_type!r}")
        mode = normalize_mode(self.mode)
        compound = None
        if action_type in PIT_ACTION_TYPES:
            compound = normalize_compound(self.compound)
            if compound == "UNKNOWN":
                raise ValueError("pit actions require a known compound")
        object.__setattr__(self, "action_type", action_type)
        object.__setattr__(self, "compound", compound)
        object.__setattr__(self, "mode", mode)

    @property
    def key(self) -> str:
        if self.action_type == ACTION_STAY_OUT:
            return f"{self.action_type}:{self.mode}"
        return f"{self.action_type}:{self.compound}:{self.mode}"

    @property
    def is_pit_action(self) -> bool:
        return self.action_type in PIT_ACTION_TYPES

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": LEGAL_ACTION_MASK_SCHEMA_VERSION,
            "action_type": self.action_type,
            "compound": self.compound,
            "mode": self.mode,
            "key": self.key,
        }

    @classmethod
    def from_key(cls, value: object) -> "StrategyAction":
        text = str(value or "").strip()
        if not text:
            return cls(ACTION_STAY_OUT)
        if ":" in text:
            parts = text.split(":")
        else:
            parts = text.split("_")
        action_type = "_".join(parts[:2]) if len(parts) >= 2 and parts[0] == "pit" else parts[0]
        if action_type == "stay":
            action_type = ACTION_STAY_OUT
        if action_type in {ACTION_STAY_OUT, "stay"}:
            mode = parts[-1] if parts[-1] in STRATEGY_MODES else "conservative"
            return cls(ACTION_STAY_OUT, mode=mode)
        if action_type == "pit":
            action_type = ACTION_PIT_NOW
        if action_type not in PIT_ACTION_TYPES:
            if str(value).startswith("pit_next"):
                action_type = ACTION_PIT_NEXT_LAP
            elif str(value).startswith("pit_now"):
                action_type = ACTION_PIT_NOW
        compound = None
        for part in parts:
            normalized = normalize_compound(part)
            if normalized != "UNKNOWN":
                compound = normalized
                break
        mode = "aggressive" if "aggressive" in parts else "conservative"
        return cls(action_type, compound=compound, mode=mode)


@dataclass(frozen=True)
class ActionMaskConfig:
    """Business rules for converting state into legal action masks."""

    allow_pit_next_lap: bool = True
    min_laps_after_stop: int = 2
    dry_compounds: tuple[str, ...] = DRY_COMPOUNDS
    wet_compounds: tuple[str, ...] = WET_COMPOUNDS
    # Unknown tyre inventory is not permission to fit every dry compound.
    # Research callers may opt into an explicit default, but production and
    # OPE fail closed by default.
    default_available_compounds: tuple[str, ...] = ()
    allow_wet_compounds_when_declared: bool = True
    allow_same_compound_pit: bool = True
    enforce_dry_mandatory_change: bool = True
    mandatory_stop_window_laps: int = 2
    pit_lane_closed_on_red: bool = True
    pit_lane_closed_on_box_lap: bool = True
    aggressive_disallowed_under_red: bool = True


@dataclass(frozen=True)
class LegalActionMask:
    actions: tuple[StrategyAction, ...]
    mask: np.ndarray
    reasons: tuple[str, ...]
    constraint_feasible: bool = True
    operational_fallback_applied: bool = False

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 1:
            raise ValueError("mask must be one-dimensional")
        if mask.size != len(self.actions):
            raise ValueError("mask length must match actions")
        reasons = tuple(self.reasons)
        if len(reasons) != len(self.actions):
            raise ValueError("reasons length must match actions")
        feasible = bool(self.constraint_feasible)
        fallback_applied = bool(self.operational_fallback_applied)
        if feasible and fallback_applied:
            raise ValueError(
                "an operational fallback cannot be constraint-feasible"
            )
        if feasible and not bool(mask.any()):
            raise ValueError("a constraint-feasible mask must contain an action")
        if fallback_applied and not bool(mask.any()):
            raise ValueError("an applied operational fallback must be selectable")
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "constraint_feasible", feasible)
        object.__setattr__(
            self,
            "operational_fallback_applied",
            fallback_applied,
        )

    @property
    def selectable_actions(self) -> tuple[StrategyAction, ...]:
        """Actions exposed to runtime code solely to avoid an all-false tensor."""

        return tuple(
            action
            for action, selectable in zip(self.actions, self.mask.tolist())
            if bool(selectable)
        )

    @property
    def constraint_legal_mask(self) -> np.ndarray:
        if not self.constraint_feasible:
            return np.zeros_like(self.mask, dtype=bool)
        return self.mask.copy()

    @property
    def legal_actions(self) -> tuple[StrategyAction, ...]:
        if not self.constraint_feasible:
            return ()
        return self.selectable_actions

    @property
    def illegal_actions(self) -> tuple[StrategyAction, ...]:
        legal = set(self.legal_actions)
        return tuple(action for action in self.actions if action not in legal)

    @property
    def legal_count(self) -> int:
        return int(len(self.legal_actions))

    @property
    def selectable_count(self) -> int:
        return int(np.sum(self.mask))

    def is_legal(self, action: StrategyAction) -> bool:
        if not self.constraint_feasible:
            return False
        for candidate, legal in zip(self.actions, self.mask.tolist()):
            if candidate == action:
                return bool(legal)
        return False

    def is_selectable(self, action: StrategyAction) -> bool:
        for candidate, selectable in zip(self.actions, self.mask.tolist()):
            if candidate == action:
                return bool(selectable)
        return False

    def reason_for(self, action: StrategyAction) -> str:
        for candidate, reason in zip(self.actions, self.reasons):
            if candidate == action:
                return str(reason)
        return "action_not_in_space"

    def to_payload(self) -> dict[str, object]:
        return {
            "actions": [action.to_payload() for action in self.actions],
            "mask": [bool(value) for value in self.mask.tolist()],
            "mask_semantics": (
                "runtime_selectability_with_explicit_operational_fallback;_constraint_legal_mask_is_authoritative_for_policy_and_learning"
            ),
            "constraint_legal_mask": [
                bool(value) for value in self.constraint_legal_mask.tolist()
            ],
            "reasons": list(self.reasons),
            "constraint_feasible": bool(self.constraint_feasible),
            "operational_fallback_applied": bool(
                self.operational_fallback_applied
            ),
            "legal_action_keys": [action.key for action in self.legal_actions],
            "selectable_action_keys": [
                action.key for action in self.selectable_actions
            ],
        }


def build_action_space(
    *,
    compounds: Sequence[str] = KNOWN_COMPOUNDS,
    modes: Sequence[str] = STRATEGY_MODES,
    include_pit_next_lap: bool = True,
) -> tuple[StrategyAction, ...]:
    normalized_compounds = tuple(
        compound for compound in (normalize_compound(item) for item in compounds) if compound != "UNKNOWN"
    )
    normalized_modes = tuple(dict.fromkeys(normalize_mode(item) for item in modes))

    actions: list[StrategyAction] = []
    for mode in normalized_modes:
        actions.append(StrategyAction(ACTION_STAY_OUT, mode=mode))
    pit_types = [ACTION_PIT_NOW]
    if include_pit_next_lap:
        pit_types.append(ACTION_PIT_NEXT_LAP)
    for action_type in pit_types:
        for compound in normalized_compounds:
            for mode in normalized_modes:
                actions.append(StrategyAction(action_type, compound=compound, mode=mode))
    return tuple(actions)


def _available_compounds(state: Any, config: ActionMaskConfig) -> tuple[str, ...]:
    explicit = _state_value(state, "available_compounds", None)
    if explicit is None:
        metadata = _state_value(state, "metadata", {})
        if isinstance(metadata, Mapping):
            explicit = metadata.get("available_compounds")
    if explicit is not None:
        compounds = tuple(
            compound for compound in (normalize_compound(item) for item in _as_tuple(explicit)) if compound != "UNKNOWN"
        )
        return tuple(dict.fromkeys(compounds))

    current = normalize_compound(_state_value(state, "compound", None))
    is_wet = any(
        _safe_bool(_state_value(state, key, False), False)
        for key in ("is_wet_track", "weather_is_wet", "rain_expected", "track_status_is_wet")
    )
    declared_wet = current in WET_COMPOUNDS or is_wet
    base = list(config.default_available_compounds)
    if declared_wet and config.allow_wet_compounds_when_declared:
        base.extend(config.wet_compounds)
    return tuple(dict.fromkeys(normalize_compound(item) for item in base if normalize_compound(item) != "UNKNOWN"))


def _remaining_laps(state: Any) -> float:
    explicit = _safe_float(_state_value(state, "remaining_laps", None), float("nan"))
    if np.isfinite(explicit):
        return max(0.0, float(explicit))
    total = _safe_float(_state_value(state, "total_laps", None), float("nan"))
    lap = _safe_float(_state_value(state, "lap_number", _state_value(state, "lap_last", None)), float("nan"))
    if np.isfinite(total) and np.isfinite(lap):
        return max(0.0, float(total - lap))
    return float("inf")


def _used_compounds(state: Any) -> tuple[str, ...]:
    used = _state_value(state, "used_compounds", None)
    if used is None:
        metadata = _state_value(state, "metadata", {})
        if isinstance(metadata, Mapping):
            used = metadata.get("used_compounds")
    compounds = tuple(
        compound for compound in (normalize_compound(item) for item in _as_tuple(used)) if compound != "UNKNOWN"
    )
    current = normalize_compound(_state_value(state, "compound", None))
    if current != "UNKNOWN" and current not in compounds:
        compounds = (*compounds, current)
    return tuple(dict.fromkeys(compounds))


def _mandatory_change_required(state: Any, config: ActionMaskConfig) -> bool:
    explicit = _state_value(state, "mandatory_compound_change_required", None)
    if explicit is not None:
        return _safe_bool(explicit, False)
    if not config.enforce_dry_mandatory_change:
        return False
    current = normalize_compound(_state_value(state, "compound", None))
    used = set(_used_compounds(state))
    if any(compound in WET_COMPOUNDS for compound in used):
        return False
    if current not in DRY_COMPOUNDS:
        return False
    dry_used = {compound for compound in used if compound in DRY_COMPOUNDS}
    return len(dry_used) < 2


def _forced_pit_compound(state: Any) -> Optional[str]:
    metadata = _state_value(state, "metadata", {})
    value = None
    if isinstance(metadata, Mapping):
        value = metadata.get("forced_pit_next_compound")
    value = _state_value(state, "forced_pit_next_compound", value)
    compound = normalize_compound(value)
    return compound if compound != "UNKNOWN" else None


def _compound_satisfies_mandatory_change(
    compound: object,
    *,
    used: set[str],
) -> bool:
    normalized = normalize_compound(compound)
    if normalized in WET_COMPOUNDS:
        return True
    return bool(normalized in DRY_COMPOUNDS and normalized not in used)


def build_legal_action_mask(
    state: Any,
    *,
    action_space: Optional[Sequence[StrategyAction]] = None,
    config: ActionMaskConfig | None = None,
) -> LegalActionMask:
    """Return a legal mask for ``state`` without using future race information."""

    cfg = config or ActionMaskConfig()
    actions = tuple(action_space or build_action_space(include_pit_next_lap=cfg.allow_pit_next_lap))
    remaining = _remaining_laps(state)
    available = set(_available_compounds(state, cfg))
    current = normalize_compound(_state_value(state, "compound", None))
    used = set(_used_compounds(state))
    mandatory_change = _mandatory_change_required(state, cfg)
    forced_pit = _forced_pit_compound(state)

    is_red = _safe_bool(_state_value(state, "is_red", False), False)
    is_box_lap = _safe_bool(_state_value(state, "is_box_lap", False), False)
    pit_lane_open = _safe_bool(_state_value(state, "pit_lane_open", None), False)

    mask: list[bool] = []
    reasons: list[str] = []
    for action in actions:
        legal = True
        reason = "legal"

        if action.mode == "aggressive" and is_red and cfg.aggressive_disallowed_under_red:
            legal = False
            reason = "aggressive_mode_illegal_under_red"

        if forced_pit is not None:
            if action.action_type != ACTION_PIT_NOW or action.compound != forced_pit:
                legal = False
                reason = "forced_pit_next_lap_commitment"

        if legal and action.action_type == ACTION_STAY_OUT:
            if mandatory_change and remaining <= float(
                cfg.min_laps_after_stop + 1
            ):
                legal = False
                reason = (
                    "mandatory_change_requires_pit_now"
                    if remaining > float(cfg.min_laps_after_stop)
                    else "mandatory_change_deadline_infeasible"
                )
            mask.append(bool(legal))
            reasons.append(reason)
            continue

        if legal and action.action_type in PIT_ACTION_TYPES:
            if not pit_lane_open:
                legal = False
                reason = "pit_lane_closed"
            elif cfg.pit_lane_closed_on_red and is_red:
                legal = False
                reason = "pit_lane_closed_red_flag"
            elif cfg.pit_lane_closed_on_box_lap and is_box_lap:
                legal = False
                reason = "already_on_box_lap"
            elif action.action_type == ACTION_PIT_NEXT_LAP and not cfg.allow_pit_next_lap:
                legal = False
                reason = "pit_next_lap_disabled"
            elif action.action_type == ACTION_PIT_NOW and remaining <= float(cfg.min_laps_after_stop):
                legal = False
                reason = "too_late_to_pit_now"
            elif action.action_type == ACTION_PIT_NEXT_LAP and remaining <= float(cfg.min_laps_after_stop + 1):
                legal = False
                reason = "too_late_to_pit_next_lap"
            elif action.compound not in available:
                legal = False
                reason = "compound_not_available"
            elif action.compound == current and not cfg.allow_same_compound_pit:
                legal = False
                reason = "same_compound_pit_disabled"
            elif (
                mandatory_change
                and action.action_type == ACTION_PIT_NOW
                and remaining <= float(cfg.min_laps_after_stop + 1)
                and not _compound_satisfies_mandatory_change(
                    action.compound,
                    used=used,
                )
            ):
                legal = False
                reason = "pit_action_cannot_satisfy_mandatory_change"
            elif (
                mandatory_change
                and action.compound == current
                and current in DRY_COMPOUNDS
                and remaining <= float(cfg.mandatory_stop_window_laps)
            ):
                legal = False
                reason = "same_dry_compound_misses_mandatory_change"
            elif mandatory_change and action.compound in DRY_COMPOUNDS and action.compound not in used:
                reason = "legal_satisfies_mandatory_dry_change"

        mask.append(bool(legal))
        reasons.append(reason)

    # The environment should never hand an RL policy an all-false mask.  If the
    # state is already impossible, allow the safest no-op so evaluation can fail
    # with diagnostics instead of crashing downstream.
    constraint_feasible = bool(any(mask))
    operational_fallback_applied = False
    if not constraint_feasible:
        for idx, action in enumerate(actions):
            if action.action_type == ACTION_STAY_OUT and action.mode == "conservative":
                mask[idx] = True
                reasons[idx] = "fallback_safe_noop_after_all_actions_masked"
                operational_fallback_applied = True
                break

    return LegalActionMask(
        actions=actions,
        mask=np.asarray(mask, dtype=bool),
        reasons=tuple(reasons),
        constraint_feasible=constraint_feasible,
        operational_fallback_applied=operational_fallback_applied,
    )


__all__ = [
    "ACTION_PIT_NEXT_LAP",
    "ACTION_PIT_NOW",
    "ACTION_STAY_OUT",
    "ACTION_TYPES",
    "ActionMaskConfig",
    "DRY_COMPOUNDS",
    "KNOWN_COMPOUNDS",
    "LEGAL_ACTION_MASK_SCHEMA_VERSION",
    "LegalActionMask",
    "PIT_ACTION_TYPES",
    "STRATEGY_MODES",
    "StrategyAction",
    "WET_COMPOUNDS",
    "build_action_space",
    "build_legal_action_mask",
    "is_dry_compound",
    "is_wet_compound",
    "normalize_compound",
    "normalize_mode",
]
