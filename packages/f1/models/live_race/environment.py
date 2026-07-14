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
    STRATEGY_MODES,
    StrategyAction,
    build_legal_action_mask,
    normalize_compound,
    normalize_mode,
)
from packages.f1.models.live_race.state import parse_track_status


LEAKAGE_CONTRACT_VERSION = (
    "live_strategy_state_v6_full_legal_mask_input_and_feasibility_evidence"
)
TRANSITION_FINGERPRINT_VERSION = (
    "live_strategy_transition_fingerprint_v8_sporting_deadlines_derived_from_action_timing"
)
LEGAL_ACTION_MASK_EVIDENCE_VERSION = (
    "live_strategy_legal_action_mask_inputs_v2_override_and_feasibility_fail_closed"
)
REWARD_SEMANTICS = (
    "undiscounted_interval_total_composite_reward_"
    "negative_elapsed_time_plus_position_gain_shaping_minus_action_penalties"
)

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

PIT_IN_COLUMNS: tuple[str, ...] = ("is_pit_in_lap", "PitInTime", "pit_in_time", "pit_in")
PIT_OUT_COLUMNS: tuple[str, ...] = ("is_pit_out_lap", "PitOutTime", "pit_out_time", "pit_out")


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


def _is_observed(value: object, *, allow_empty_string: bool = False) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, str) and not allow_empty_string and not value.strip():
        return False
    return True


def _has_payload_key(payload: Mapping[str, object], aliases: Sequence[str]) -> bool:
    return any(alias in payload for alias in aliases)


def _finite_at_least(value: object, lower: float) -> bool:
    numeric = _safe_float(value)
    return bool(np.isfinite(numeric) and numeric >= float(lower))


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


def _strict_bool_evidence(value: object) -> tuple[bool, bool]:
    """Return ``(parsed_value, is_unambiguous_boolean_evidence)``."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value), True
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric) and numeric in {0.0, 1.0}:
            return bool(numeric), True
        return False, False
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "t"}:
        return True, True
    if text in {"0", "false", "no", "n", "f"}:
        return False, True
    return False, False


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


def _observed_action_mode_evidence(
    row: Mapping[str, object] | pd.Series,
) -> tuple[str, bool, str]:
    """Return a representative mode plus whether the source actually observed it.

    ``StrategyAction`` requires a concrete mode, but historical lap timing does
    not reveal whether the team was executing the conservative or aggressive
    pace sub-contract.  A conservative representative is therefore an encoding
    detail only; it must never be promoted into exact mode evidence.
    """

    payload = _row_payload(row)
    raw_mode = _first(payload, ("action_mode", "strategy_mode"), None)
    if raw_mode is not None:
        token = str(raw_mode).strip().lower()
        accepted = set(STRATEGY_MODES) | {
            "attack",
            "push",
            "risk_on",
        }
        if token in accepted:
            return normalize_mode(raw_mode), True, "explicit_action_mode_field"

    explicit = _first(
        payload,
        ("observed_action", "action_t", "recommended_action"),
        None,
    )
    if explicit is not None:
        parts = str(explicit).strip().lower().replace("_", ":").split(":")
        for mode in STRATEGY_MODES:
            if mode in parts:
                return mode, True, "explicit_action_key"

    return "conservative", False, "missing"


def _observed_action_label_metadata(
    row: Mapping[str, object] | pd.Series,
    action: StrategyAction,
) -> dict[str, object]:
    representative_mode, mode_known, mode_source = _observed_action_mode_evidence(row)
    return {
        "observed_action_mode_known": bool(mode_known),
        "observed_action_mode_source": mode_source,
        "observed_action_label_status": (
            "exact" if mode_known else "coarsened_missing_mode"
        ),
        "observed_action_representative_mode": action.mode,
        "observed_action_representative_only": bool(not mode_known),
        "observed_action_type": action.action_type,
        "observed_action_compound": action.compound,
        # Guard against a future caller changing the default representative
        # without updating the label contract.
        "observed_action_mode_encoding_consistent": bool(
            mode_known or action.mode == representative_mode
        ),
    }


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
        lap_number_raw = _first(payload, ("lap_number", "lap_last", "LapNumber"), None)
        lap_number = _safe_int(lap_number_raw, 0)
        lap_number_known = _finite_at_least(lap_number_raw, 0.0)
        total_laps_raw = _first(payload, ("total_laps", "race_total_laps", "scheduled_laps"), None)
        total_laps = None if total_laps_raw is None else _safe_int(total_laps_raw, 0)
        total_laps_known = _finite_at_least(total_laps_raw, 1.0)
        remaining_raw = _first(payload, ("remaining_laps", "laps_remaining"), None)
        remaining = None if remaining_raw is None else _safe_int(remaining_raw, 0)
        remaining_laps_known = _finite_at_least(remaining_raw, 0.0)
        race_horizon_known = bool(
            remaining_laps_known or (lap_number_known and total_laps_known)
        )

        track_status_raw = _first(payload, ("track_status", "TrackStatus"), None)
        track_status = str(track_status_raw or "")
        flags = parse_track_status(track_status)
        compound_raw = _first(
            payload,
            ("compound", "Compound", "compound_normalized"),
            None,
        )
        compound = normalize_compound(compound_raw)
        compound_claim = _first(payload, ("current_compound_known",), None)
        current_compound_known = bool(
            (True if compound_claim is None else _safe_bool(compound_claim, False))
            and _is_observed(compound_raw)
            and compound != "UNKNOWN"
        )
        used_compounds_raw = _first(
            payload,
            ("used_compounds", "compounds_used"),
            None,
        )
        used_compounds = _tuple_compounds(
            used_compounds_raw,
            fallback=compound,
        )
        used_compounds_claim = _first(payload, ("used_compounds_known",), None)
        used_compounds_known = bool(
            (
                _is_observed(used_compounds_raw)
                if used_compounds_claim is None
                else _safe_bool(used_compounds_claim, False)
            )
            and _is_observed(used_compounds_raw)
            and current_compound_known
            and compound in used_compounds
        )
        explicit_is_red = _first(payload, ("is_red",), None)
        red_value_observed = bool(
            _is_observed(track_status_raw) or _is_observed(explicit_is_red)
        )
        red_claim = _first(
            payload,
            ("red_flag_known", "track_status_known", "is_red_known"),
            None,
        )
        red_flag_known = bool(
            red_value_observed
            and (True if red_claim is None else _safe_bool(red_claim, False))
        )
        meta = dict(metadata or {})
        meta.update(
            {
                "source_row_index": payload.get("_source_row_index"),
                "available_through_lap": lap_number,
                "ignored_future_columns": tuple(column for column in FUTURE_ONLY_COLUMNS if column in payload),
                "lap_number_known": bool(lap_number_known),
                "race_horizon_known": bool(race_horizon_known),
                "current_compound_known": bool(current_compound_known),
                "used_compounds_known": bool(used_compounds_known),
                "red_flag_known": bool(red_flag_known),
            }
        )
        available = _first(payload, ("available_compounds", "allowed_compounds"), None)
        available_compounds = _tuple_compounds(available)
        if available is not None:
            meta["available_compounds"] = available_compounds
        inventory_known = _first(payload, ("compound_inventory_known",), None)
        inventory_claimed = (
            _safe_bool(inventory_known, False) if inventory_known is not None else available is not None
        )
        meta["compound_inventory_known"] = bool(
            inventory_claimed and available_compounds
        )

        pit_lane_open = _first(payload, ("pit_lane_open",), None)
        pit_lane_known = _first(payload, ("pit_lane_open_known",), None)
        pit_lane_claimed = (
            _safe_bool(pit_lane_known, False) if pit_lane_known is not None else pit_lane_open is not None
        )
        meta["pit_lane_open_known"] = bool(
            pit_lane_claimed and pit_lane_open is not None
        )

        box_lap_aliases = ("is_box_lap", "box_lap")
        box_lap_raw = _first(payload, box_lap_aliases, None)
        box_lap_claim = _first(
            payload,
            ("box_lap_status_known", "is_box_lap_known"),
            None,
        )
        box_lap_observed = bool(
            _has_payload_key(payload, box_lap_aliases)
            and _is_observed(box_lap_raw)
        )
        meta["box_lap_status_known"] = bool(
            box_lap_observed
            and (
                True
                if box_lap_claim is None
                else _safe_bool(box_lap_claim, False)
            )
        )

        forced_pit_aliases = ("forced_pit_next_compound",)
        forced_pit_raw = _first(payload, forced_pit_aliases, None)
        forced_pit_claim = _first(
            payload,
            ("forced_pit_commitment_known",),
            None,
        )
        forced_pit_key_present = _has_payload_key(payload, forced_pit_aliases)
        forced_pit_absent = not _is_observed(forced_pit_raw)
        forced_pit_compound = normalize_compound(forced_pit_raw)
        forced_pit_value_valid = bool(
            forced_pit_absent or forced_pit_compound != "UNKNOWN"
        )
        forced_pit_claimed = (
            bool(_safe_bool(forced_pit_claim, False))
            if forced_pit_claim is not None
            else bool(forced_pit_key_present and not forced_pit_absent)
        )
        meta["forced_pit_commitment_known"] = bool(
            forced_pit_claimed
            and forced_pit_key_present
            and forced_pit_value_valid
        )
        if forced_pit_key_present:
            meta["forced_pit_next_compound"] = (
                None if forced_pit_absent else forced_pit_compound
            )

        mandatory_change_aliases = ("mandatory_compound_change_required",)
        mandatory_change_raw = _first(payload, mandatory_change_aliases, None)
        mandatory_change_key_present = _has_payload_key(
            payload,
            mandatory_change_aliases,
        )
        mandatory_change_claim = _first(
            payload,
            (
                "mandatory_compound_change_override_known",
                "mandatory_compound_change_required_known",
            ),
            None,
        )
        claim_value, claim_valid = _strict_bool_evidence(
            mandatory_change_claim
        )
        if not mandatory_change_key_present:
            mandatory_change_override_known = bool(
                mandatory_change_claim is None
                or (claim_valid and claim_value)
            )
            mandatory_change_override_source = (
                "derived_from_compound_history"
                if mandatory_change_override_known
                else "override_presence_unknown"
            )
        else:
            parsed_override, override_valid = _strict_bool_evidence(
                mandatory_change_raw
            )
            mandatory_change_override_known = bool(
                override_valid
                and (
                    mandatory_change_claim is None
                    or (claim_valid and claim_value)
                )
            )
            mandatory_change_override_source = (
                "explicit_known"
                if mandatory_change_override_known
                else "explicit_missing_or_invalid"
            )
            if override_valid:
                meta["mandatory_compound_change_required"] = bool(
                    parsed_override
                )
        meta["mandatory_compound_change_override_known"] = bool(
            mandatory_change_override_known
        )
        meta["mandatory_compound_change_override_source"] = (
            mandatory_change_override_source
        )

        support_known = _first(payload, ("behavior_action_support_known",), None)
        propensity = _first(payload, ("behavior_action_probability", "behavior_propensity"), None)
        propensity_value = _safe_float(propensity)
        propensity_finite = bool(np.isfinite(propensity_value))
        propensity_positive = bool(propensity_finite and 0.0 < propensity_value <= 1.0)
        support_claimed = (
            _safe_bool(support_known, False)
            if support_known is not None
            else propensity is not None
        )
        meta["behavior_action_support_known"] = bool(support_claimed and propensity_positive)
        if propensity is not None:
            meta["behavior_action_probability"] = (
                float(propensity_value) if propensity_finite else None
            )
        if propensity is None:
            meta["behavior_action_support_reason"] = "propensity_missing"
        elif not propensity_finite or propensity_value > 1.0:
            meta["behavior_action_support_reason"] = "propensity_invalid"
        elif propensity_value <= 0.0:
            meta["behavior_action_support_reason"] = "zero_support"
        elif not support_claimed:
            meta["behavior_action_support_reason"] = "propensity_not_certified"
        else:
            meta["behavior_action_support_reason"] = "known_positive_support"

        for state_key, aliases in {
            "pit_lane_open": ("pit_lane_open",),
            "is_box_lap": ("is_box_lap", "box_lap"),
            "is_pit_in_lap": PIT_IN_COLUMNS,
            "is_pit_out_lap": PIT_OUT_COLUMNS,
            "pit_in_signal_known": ("pit_in_signal_known",),
            "pit_out_signal_known": ("pit_out_signal_known",),
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
            "pit_lane_open": self.metadata.get("pit_lane_open"),
            "pit_lane_open_known": self.metadata.get("pit_lane_open_known", False),
            "available_compounds": list(self.metadata.get("available_compounds", ())),
            "compound_inventory_known": self.metadata.get("compound_inventory_known", False),
            "lap_number_known": self.metadata.get("lap_number_known", False),
            "race_horizon_known": self.metadata.get("race_horizon_known", False),
            "current_compound_known": self.metadata.get(
                "current_compound_known",
                False,
            ),
            "used_compounds_known": self.metadata.get(
                "used_compounds_known",
                False,
            ),
            "red_flag_known": self.metadata.get("red_flag_known", False),
            "is_box_lap": self.metadata.get("is_box_lap"),
            "box_lap_status_known": self.metadata.get(
                "box_lap_status_known",
                False,
            ),
            "forced_pit_next_compound": self.metadata.get(
                "forced_pit_next_compound"
            ),
            "forced_pit_commitment_known": self.metadata.get(
                "forced_pit_commitment_known",
                False,
            ),
            "mandatory_compound_change_required": self.metadata.get(
                "mandatory_compound_change_required"
            ),
            "mandatory_compound_change_override_known": self.metadata.get(
                "mandatory_compound_change_override_known",
                False,
            ),
            "mandatory_compound_change_override_source": self.metadata.get(
                "mandatory_compound_change_override_source"
            ),
            "is_pit_in_lap": self.metadata.get("is_pit_in_lap", False),
            "is_pit_out_lap": self.metadata.get("is_pit_out_lap", False),
        }

    def fingerprint(self) -> str:
        return _stable_digest(self.as_feature_dict())


def legal_action_mask_input_evidence(state: StrategyState) -> dict[str, object]:
    """Certify every state input that can change ``build_legal_action_mask``.

    The action-mask implementation has fail-safe defaults for ordinary online
    planning, but those defaults are not historical evidence.  Offline-Q may
    bootstrap through ``max_a Q(s', a)`` only when every legality-driving input
    is observed.  The configuration and action space are bound separately by
    the stored/recomputed mask payload.
    """

    metadata = state.metadata or {}
    available_compounds = tuple(
        item
        for item in _tuple_compounds(metadata.get("available_compounds"))
        if item != "UNKNOWN"
    )
    used_compounds = tuple(state.used_compounds)
    input_status = {
        "race_horizon": bool(
            metadata.get("race_horizon_known") is True
            and state.remaining_laps is not None
            and int(state.remaining_laps) >= 0
        ),
        "current_compound": bool(
            metadata.get("current_compound_known") is True
            and state.compound != "UNKNOWN"
        ),
        "used_compound_history": bool(
            metadata.get("used_compounds_known") is True
            and state.compound != "UNKNOWN"
            and state.compound in used_compounds
        ),
        "red_flag_status": bool(metadata.get("red_flag_known") is True),
        "box_lap_status": bool(
            metadata.get("box_lap_status_known") is True
            and "is_box_lap" in metadata
            and _is_observed(metadata.get("is_box_lap"))
        ),
        "pit_lane_status": bool(
            metadata.get("pit_lane_open_known") is True
            and "pit_lane_open" in metadata
            and _is_observed(metadata.get("pit_lane_open"))
        ),
        "compound_inventory": bool(
            metadata.get("compound_inventory_known") is True
            and bool(available_compounds)
        ),
        "forced_pit_commitment": bool(
            metadata.get("forced_pit_commitment_known") is True
            and "forced_pit_next_compound" in metadata
            and (
                metadata.get("forced_pit_next_compound") is None
                or normalize_compound(
                    metadata.get("forced_pit_next_compound")
                )
                != "UNKNOWN"
            )
        ),
        "mandatory_compound_change_override": bool(
            metadata.get("mandatory_compound_change_override_known") is True
        ),
    }
    blockers = tuple(
        key for key, known in input_status.items() if not bool(known)
    )
    return {
        "schema_version": LEGAL_ACTION_MASK_EVIDENCE_VERSION,
        "inputs_known": input_status,
        "certified": bool(not blockers),
        "blockers": blockers,
    }


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
            "transition_fingerprint_version": TRANSITION_FINGERPRINT_VERSION,
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
                "fingerprint_version": TRANSITION_FINGERPRINT_VERSION,
                "state_t": self.state_t.as_feature_dict(),
                "action_t": self.action_t.to_payload(),
                # Reward values and components are model inputs for BC/offline
                # RL.  They must be part of the replay identity: otherwise a
                # reward-function change can retain the same record id and
                # falsely pass prefix-invariance checks.
                "reward_t": self.reward_t.to_payload(),
                "state_t1": self.state_t1.as_feature_dict(),
                "done": bool(self.done),
                "legal_action_mask": self.legal_action_mask.to_payload(),
                "policy_evidence": {
                    key: self.metadata.get(key)
                    for key in (
                        "action_legality_status",
                        "action_legality_unknown_reasons",
                        "full_legal_action_mask_certified",
                        "legal_action_mask_input_evidence",
                        "legal_action_mask_constraint_feasible",
                        "legal_action_mask_operational_fallback_applied",
                        "next_legal_action_mask_required",
                        "next_full_legal_action_mask_certified",
                        "next_legal_action_mask_input_evidence",
                        "next_legal_action_mask_constraint_feasible",
                        "next_legal_action_mask_operational_fallback_applied",
                        "next_legal_action_mask",
                        "next_legal_action_mask_fingerprint",
                        "observed_action_label_status",
                        "observed_action_mode_known",
                        "observed_action_mode_source",
                        "observed_action_representative_only",
                        "reward_observation_status",
                        "reward_observation_blockers",
                        "reward_observation_components_known",
                        "reward_race_time_weight",
                        "reward_position_gain_weight",
                        "causal_transition_boundary_status",
                        "causal_transition_boundary_blockers",
                        "behavior_action_support_status",
                        "behavior_action_probability",
                        "policy_training_eligible",
                        "policy_training_blockers",
                        "propensity_ope_eligible",
                        "propensity_ope_blockers",
                        # Retained aliases keep older consumers explicit while
                        # the current fingerprint records their training-only
                        # meaning.
                        "policy_learning_eligible",
                        "policy_learning_blockers",
                        "pit_boundary_support_status",
                    )
                },
                "elapsed_laps": self.metadata.get(
                    "elapsed_laps",
                    max(1, int(self.state_t1.lap_number) - int(self.state_t.lap_number)),
                ),
                "raw_lap_delta": self.metadata.get(
                    "raw_lap_delta",
                    int(self.state_t1.lap_number) - int(self.state_t.lap_number),
                ),
            }
        )


def _row_event_flag(row: Mapping[str, object] | pd.Series, keys: Sequence[str]) -> bool:
    payload = _row_payload(row)
    for key in keys:
        if key not in payload:
            continue
        value = payload.get(key)
        try:
            if value is None or pd.isna(value):
                continue
        except Exception:
            if value is None:
                continue
        if str(key).startswith("is_") or isinstance(value, (bool, np.bool_)):
            return _safe_bool(value, False)
        text = str(value).strip().lower()
        if text in {"", "0", "false", "no", "n", "f"}:
            return False
        if text in {"1", "true", "yes", "y", "t"}:
            return True
        return True
    return False


def _reward_observation_evidence(
    state_t: StrategyState,
    state_t1: StrategyState,
    *,
    config: RewardConfig,
) -> dict[str, object]:
    """Certify that every weighted replay reward component is observed."""

    blockers: list[str] = []
    race_time_known = bool(
        state_t.race_time_seconds is not None
        and state_t1.race_time_seconds is not None
        and np.isfinite(float(state_t.race_time_seconds))
        and np.isfinite(float(state_t1.race_time_seconds))
    )
    race_time_nondecreasing = bool(
        race_time_known
        and float(state_t1.race_time_seconds) >= float(state_t.race_time_seconds)
    )
    if float(config.race_time_weight) != 0.0:
        if not race_time_known:
            blockers.append("elapsed_race_time_endpoints_unobserved")
        elif not race_time_nondecreasing:
            blockers.append("elapsed_race_time_endpoints_decrease")

    position_known = bool(
        state_t.position is not None
        and state_t1.position is not None
        and int(state_t.position) > 0
        and int(state_t1.position) > 0
    )
    if float(config.position_gain_weight) != 0.0 and not position_known:
        blockers.append("running_position_endpoints_unobserved")

    return {
        "reward_observation_status": (
            "observed_required_components" if not blockers else "incomplete"
        ),
        "reward_observation_blockers": tuple(blockers),
        "reward_observation_components_known": {
            "elapsed_race_time": bool(race_time_known and race_time_nondecreasing),
            "running_position": bool(position_known),
            "pit_action_penalty": True,
            "illegal_action_penalty": True,
        },
        "reward_race_time_weight": float(config.race_time_weight),
        "reward_position_gain_weight": float(config.position_gain_weight),
    }


def _transition_evidence(
    state_t: StrategyState,
    state_t1: StrategyState,
    action: StrategyAction,
    legal_mask: LegalActionMask,
    next_legal_mask: LegalActionMask,
    done: bool,
    transition_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Return causal-boundary, legality and propensity evidence separately."""

    unknown_legality: list[str] = []
    if action.is_pit_action:
        if not _safe_bool(state_t.metadata.get("pit_lane_open_known"), False):
            unknown_legality.append("pit_lane_status_unknown")
        if not _safe_bool(state_t.metadata.get("compound_inventory_known"), False):
            unknown_legality.append("compound_inventory_unknown")

    legality_known = not unknown_legality
    if not legality_known:
        legality_status = "unknown"
    elif not legal_mask.constraint_feasible:
        legality_status = "constraint_infeasible"
    elif legal_mask.is_legal(action):
        legality_status = "known_legal"
    else:
        legality_status = "known_illegal"

    elapsed_laps = max(
        1,
        int(transition_metadata.get("elapsed_laps") or 1),
    )
    raw_lap_delta = int(
        transition_metadata.get("raw_lap_delta", elapsed_laps)
    )
    transition_kind = str(
        transition_metadata.get("transition_kind") or "unknown"
    )
    boundary_blockers: list[str] = []
    if transition_kind == "adjacent_running_laps":
        if raw_lap_delta != 1:
            boundary_blockers.append("adjacent_transition_is_not_one_lap")
        if action.is_pit_action:
            boundary_blockers.append(
                "pit_action_missing_physical_pit_boundary_evidence"
            )
    elif transition_kind == "pit_stop_semi_markov":
        if not action.is_pit_action:
            boundary_blockers.append("pit_boundary_has_non_pit_action")
        if transition_metadata.get("semi_markov") is not True:
            boundary_blockers.append("pit_boundary_missing_semi_markov_marker")
        if transition_metadata.get("pit_out_observed") is not True:
            boundary_blockers.append("pit_boundary_missing_pit_out_observation")
        if transition_metadata.get("end_is_first_post_pit_running_lap") is not True:
            boundary_blockers.append(
                "pit_boundary_end_is_not_first_post_pit_running_lap"
            )
        if (
            str(transition_metadata.get("pit_boundary_support_status") or "")
            != "pit_in_and_pit_out_observed"
        ):
            boundary_blockers.append("pit_boundary_support_not_observed")
        if raw_lap_delta < 2:
            boundary_blockers.append("pit_boundary_duration_is_not_semi_markov")
    else:
        boundary_blockers.append("transition_kind_unknown")
    boundary_status = "valid" if not boundary_blockers else "invalid"

    behavior_support_known = _safe_bool(
        state_t.metadata.get("behavior_action_support_known"),
        False,
    )
    behavior_support_reason = str(
        state_t.metadata.get("behavior_action_support_reason", "propensity_missing")
    )
    behavior_action_probability = state_t.metadata.get(
        "behavior_action_probability"
    )
    if behavior_support_known:
        behavior_support_status = "known_positive"
    elif behavior_support_reason == "zero_support":
        behavior_support_status = "zero_support"
    elif behavior_support_reason == "propensity_invalid":
        behavior_support_status = "invalid"
    else:
        behavior_support_status = "unknown"
    legal_mask_input_evidence = legal_action_mask_input_evidence(state_t)
    full_legal_action_mask_certified = bool(
        legal_mask_input_evidence["certified"]
    )
    next_legal_action_mask_required = bool(not done)
    next_legal_mask_input_evidence = legal_action_mask_input_evidence(state_t1)
    next_full_legal_action_mask_certified = bool(
        next_legal_mask_input_evidence["certified"]
    )
    next_legal_action_mask_payload = next_legal_mask.to_payload()
    next_legal_action_mask_fingerprint = _stable_digest(
        {"next_legal_action_mask": next_legal_action_mask_payload}
    )
    action_mode_known = _safe_bool(
        transition_metadata.get("observed_action_mode_known"),
        False,
    )
    training_blockers = [*boundary_blockers, *unknown_legality]
    if legality_status == "known_illegal":
        training_blockers.append("observed_action_illegal_under_known_mask")
    if not legal_mask.constraint_feasible:
        training_blockers.append("legal_action_mask_constraint_infeasible")
    if legal_mask.operational_fallback_applied:
        training_blockers.append(
            "legal_action_mask_operational_fallback_not_training_eligible"
        )
    if not full_legal_action_mask_certified:
        training_blockers.append("full_legal_action_mask_not_certified")
        training_blockers.extend(
            f"legal_action_mask_input_unknown:{blocker}"
            for blocker in legal_mask_input_evidence["blockers"]
        )
    if (
        next_legal_action_mask_required
        and not next_full_legal_action_mask_certified
    ):
        training_blockers.append(
            "next_full_legal_action_mask_not_certified"
        )
        training_blockers.extend(
            f"next_legal_action_mask_input_unknown:{blocker}"
            for blocker in next_legal_mask_input_evidence["blockers"]
        )
    if next_legal_action_mask_required and not next_legal_mask.constraint_feasible:
        training_blockers.append(
            "next_legal_action_mask_constraint_infeasible"
        )
    if (
        next_legal_action_mask_required
        and next_legal_mask.operational_fallback_applied
    ):
        training_blockers.append(
            "next_legal_action_mask_operational_fallback_not_training_eligible"
        )
    if not action_mode_known:
        training_blockers.append("observed_action_mode_unknown")
    reward_observation_status = str(
        transition_metadata.get("reward_observation_status") or "missing"
    )
    if reward_observation_status != "observed_required_components":
        training_blockers.append("reward_observation_not_certified")
        training_blockers.extend(
            str(blocker)
            for blocker in transition_metadata.get(
                "reward_observation_blockers",
                ("reward_observation_evidence_missing",),
            )
        )

    policy_training_eligible = bool(
        boundary_status == "valid"
        and legality_status == "known_legal"
        and full_legal_action_mask_certified
        and legal_mask.constraint_feasible
        and not legal_mask.operational_fallback_applied
        and (
            not next_legal_action_mask_required
            or (
                next_full_legal_action_mask_certified
                and next_legal_mask.constraint_feasible
                and not next_legal_mask.operational_fallback_applied
            )
        )
        and action_mode_known
        and reward_observation_status == "observed_required_components"
    )
    propensity_ope_blockers = list(training_blockers)
    if behavior_support_status == "zero_support":
        propensity_ope_blockers.append("behavior_action_zero_support")
    elif behavior_support_status == "invalid":
        propensity_ope_blockers.append("behavior_action_support_invalid")
    elif not behavior_support_known:
        propensity_ope_blockers.append("behavior_action_support_unknown")
    propensity_ope_eligible = bool(
        policy_training_eligible and behavior_support_known
    )
    return {
        "action_legality_known": bool(legality_known),
        "action_legality_status": legality_status,
        "action_legality_unknown_reasons": tuple(unknown_legality),
        "full_legal_action_mask_certified": bool(
            full_legal_action_mask_certified
        ),
        "legal_action_mask_input_evidence": legal_mask_input_evidence,
        "legal_action_mask_constraint_feasible": bool(
            legal_mask.constraint_feasible
        ),
        "legal_action_mask_operational_fallback_applied": bool(
            legal_mask.operational_fallback_applied
        ),
        "next_legal_action_mask_required": bool(
            next_legal_action_mask_required
        ),
        "next_full_legal_action_mask_certified": bool(
            next_full_legal_action_mask_certified
        ),
        "next_legal_action_mask_input_evidence": (
            next_legal_mask_input_evidence
        ),
        "next_legal_action_mask_constraint_feasible": bool(
            next_legal_mask.constraint_feasible
        ),
        "next_legal_action_mask_operational_fallback_applied": bool(
            next_legal_mask.operational_fallback_applied
        ),
        "next_legal_action_mask": next_legal_action_mask_payload,
        "next_legal_action_mask_fingerprint": (
            next_legal_action_mask_fingerprint
        ),
        "causal_transition_boundary_status": boundary_status,
        "causal_transition_boundary_blockers": tuple(boundary_blockers),
        "behavior_action_support_known": bool(behavior_support_known),
        "behavior_action_support_status": behavior_support_status,
        "behavior_action_probability": behavior_action_probability,
        "reward_observation_status": reward_observation_status,
        "reward_observation_blockers": tuple(
            transition_metadata.get("reward_observation_blockers", ())
        ),
        "reward_observation_components_known": dict(
            transition_metadata.get("reward_observation_components_known", {})
        ),
        "policy_training_eligible": bool(policy_training_eligible),
        "policy_training_blockers": tuple(dict.fromkeys(training_blockers)),
        "propensity_ope_eligible": propensity_ope_eligible,
        "propensity_ope_blockers": tuple(
            dict.fromkeys(propensity_ope_blockers)
        ),
        # Backward-compatible aliases. Propensity is an evaluation requirement,
        # not a prerequisite for behavior cloning or conservative offline-Q.
        "policy_learning_eligible": bool(policy_training_eligible),
        "policy_learning_blockers": tuple(dict.fromkeys(training_blockers)),
    }


def infer_observed_action(row_t: Mapping[str, object] | pd.Series, row_t1: Mapping[str, object] | pd.Series) -> StrategyAction:
    """Infer a replay action from explicit action fields or stint/compound change."""

    current = _row_payload(row_t)
    nxt = _row_payload(row_t1)
    explicit = _first(current, ("observed_action", "action_t", "recommended_action"), None)
    explicit_compound = _first(current, ("action_compound", "next_compound", "pit_compound"), None)
    explicit_mode, _, _ = _observed_action_mode_evidence(current)
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
        parsed = StrategyAction.from_key(text)
        if _first(current, ("action_mode", "strategy_mode"), None) is not None:
            return StrategyAction(
                parsed.action_type,
                compound=parsed.compound,
                mode=explicit_mode,
            )
        return parsed

    current_stint = _safe_int(_first(current, ("stint_id", "Stint"), 1), 1)
    next_stint = _safe_int(_first(nxt, ("stint_id", "Stint"), current_stint), current_stint)
    current_compound = normalize_compound(_first(current, ("compound", "Compound"), "UNKNOWN"))
    next_compound = normalize_compound(_first(nxt, ("compound", "Compound"), "UNKNOWN"))
    compound_changed = (
        current_compound != "UNKNOWN"
        and next_compound != "UNKNOWN"
        and next_compound != current_compound
    )
    if next_stint > current_stint or compound_changed:
        if next_compound == "UNKNOWN":
            raise ValueError("cannot infer a pit action without an observed post-stop compound")
        return StrategyAction(
            ACTION_PIT_NOW,
            compound=next_compound,
            mode=explicit_mode,
        )
    return StrategyAction(ACTION_STAY_OUT, mode=explicit_mode)


def _build_transition(
    row_t: pd.Series,
    row_t1: pd.Series,
    *,
    action: StrategyAction,
    done: bool,
    action_space: Optional[Sequence[StrategyAction]],
    action_mask_config: ActionMaskConfig | None,
    reward_config: RewardConfig | None,
    metadata: Mapping[str, object],
) -> StrategyTransition:
    state_t = StrategyState.from_mapping(row_t)
    state_t1 = StrategyState.from_mapping(row_t1)
    legal_mask = build_legal_action_mask(
        state_t,
        action_space=action_space,
        config=action_mask_config,
    )
    next_legal_mask = build_legal_action_mask(
        state_t1,
        action_space=action_space,
        config=action_mask_config,
    )
    raw_lap_delta = int(state_t1.lap_number) - int(state_t.lap_number)
    elapsed_laps = max(1, raw_lap_delta)
    effective_reward_config = reward_config or RewardConfig()
    transition_metadata = {
        "source": "lap_replay",
        "row_t_index": row_t.get("_source_row_index"),
        "row_t1_index": row_t1.get("_source_row_index"),
        "elapsed_laps": int(elapsed_laps),
        "raw_lap_delta": int(raw_lap_delta),
        "leakage_contract_version": LEAKAGE_CONTRACT_VERSION,
        **dict(metadata),
    }
    transition_metadata.update(
        _reward_observation_evidence(
            state_t,
            state_t1,
            config=effective_reward_config,
        )
    )
    evidence = _transition_evidence(
        state_t,
        state_t1,
        action,
        legal_mask,
        next_legal_mask,
        bool(done),
        transition_metadata,
    )
    reward = compute_transition_reward(
        state_t,
        action,
        state_t1,
        # A fail-closed mask is not proof that a historically observed action
        # was illegal.  Preserve unknown legality as an eligibility blocker
        # without fabricating an illegal-action reward penalty.
        legal_action_mask=(
            legal_mask if evidence["action_legality_status"] != "unknown" else None
        ),
        config=effective_reward_config,
    )
    return StrategyTransition(
        state_t=state_t,
        action_t=action,
        reward_t=reward,
        state_t1=state_t1,
        done=bool(done),
        legal_action_mask=legal_mask,
        metadata={
            **transition_metadata,
            **evidence,
        },
    )


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
    event_col = "event_key" if "event_key" in frame.columns else (
        "meeting_key" if "meeting_key" in frame.columns else None
    )
    lap_col = "lap_number" if "lap_number" in frame.columns else "lap_last"
    sort_cols = [
        col
        for col in (event_col, "driver_id", lap_col, "timestamp")
        if col and col in frame.columns
    ]
    if sort_cols:
        frame = frame.sort_values(sort_cols, kind="mergesort")

    transitions: list[StrategyTransition] = []
    group_cols = [
        col
        for col in (event_col, "driver_id")
        if col and col in frame.columns
    ]
    grouped = frame.groupby(group_cols, sort=False) if group_cols else [(None, frame)]
    for _, group in grouped:
        rows = [row for _, row in group.iterrows()]
        pit_episodes: dict[int, tuple[int, int, int]] = {}
        pit_rows: set[int] = set()
        for pit_in_idx, row in enumerate(rows):
            if not _row_event_flag(row, PIT_IN_COLUMNS) or pit_in_idx in pit_rows:
                continue
            pit_out_idx: Optional[int] = None
            for candidate_idx in range(pit_in_idx, len(rows)):
                if _row_event_flag(rows[candidate_idx], PIT_OUT_COLUMNS):
                    pit_out_idx = candidate_idx
                    break
                if candidate_idx > pit_in_idx and _row_event_flag(
                    rows[candidate_idx], PIT_IN_COLUMNS
                ):
                    break
            if pit_out_idx is None:
                # A pit-in marker alone does not identify the first running lap
                # after the physical stop.  Drop the label and isolate the
                # boundary row so adjacent inference cannot invent one.
                pit_rows.add(pit_in_idx)
                continue
            first_running_idx: Optional[int] = None
            for candidate_idx in range(pit_out_idx + 1, len(rows)):
                candidate = rows[candidate_idx]
                if not _row_event_flag(candidate, PIT_IN_COLUMNS) and not _row_event_flag(candidate, PIT_OUT_COLUMNS):
                    first_running_idx = candidate_idx
                    break
            if first_running_idx is None:
                pit_rows.update(range(pit_in_idx, len(rows)))
                continue
            pit_rows.update(range(pit_in_idx, first_running_idx))
            pre_action_idx = pit_in_idx - 1
            if pre_action_idx >= 0 and pre_action_idx not in pit_rows:
                pit_episodes[pre_action_idx] = (
                    pit_in_idx,
                    pit_out_idx,
                    first_running_idx,
                )
        pit_rows.update(
            row_idx
            for row_idx, row in enumerate(rows)
            if _row_event_flag(row, PIT_IN_COLUMNS) or _row_event_flag(row, PIT_OUT_COLUMNS)
        )

        idx = 0
        while idx < len(rows) - 1:
            if idx in pit_episodes:
                pit_in_idx, pit_out_idx, first_running_idx = pit_episodes[idx]
                row_t = rows[idx]
                row_t1 = rows[first_running_idx]
                post_compound = normalize_compound(_first(_row_payload(row_t1), ("compound", "Compound"), None))
                if post_compound != "UNKNOWN":
                    representative_mode, _, _ = _observed_action_mode_evidence(
                        row_t
                    )
                    action = StrategyAction(
                        ACTION_PIT_NOW,
                        compound=post_compound,
                        mode=representative_mode,
                    )
                    pit_out_indices = [pit_out_idx]
                    state_t1_preview = StrategyState.from_mapping(row_t1)
                    done = (
                        bool(state_t1_preview.remaining_laps == 0)
                        if state_t1_preview.remaining_laps is not None
                        else first_running_idx == len(rows) - 1
                    )
                    transitions.append(
                        _build_transition(
                            row_t,
                            row_t1,
                            action=action,
                            done=done,
                            action_space=action_space,
                            action_mask_config=action_mask_config,
                            reward_config=reward_config,
                            metadata={
                                "transition_kind": "pit_stop_semi_markov",
                                "semi_markov": True,
                                "pit_in_row_index": rows[pit_in_idx].get("_source_row_index"),
                                "pit_in_lap": _safe_int(
                                    _first(_row_payload(rows[pit_in_idx]), ("lap_number", "lap_last"), 0),
                                    0,
                                ),
                                "pit_out_row_indices": tuple(
                                    rows[candidate_idx].get("_source_row_index")
                                    for candidate_idx in pit_out_indices
                                ),
                                "pit_out_observed": True,
                                "pit_boundary_support_status": "pit_in_and_pit_out_observed",
                                "first_post_pit_running_lap": int(state_t1_preview.lap_number),
                                "end_is_first_post_pit_running_lap": True,
                                **_observed_action_label_metadata(row_t, action),
                            },
                        )
                    )
                # Rows inside a physical stop are outcomes of the same decision,
                # never additional action-labelled examples.  If the compound is
                # unknown, drop the unsupported decision instead of inventing it.
                idx = first_running_idx
                continue

            if idx in pit_rows or (idx + 1) in pit_rows:
                idx += 1
                continue

            row_t = rows[idx]
            row_t1 = rows[idx + 1]
            try:
                action = infer_observed_action(row_t, row_t1)
            except ValueError:
                idx += 1
                continue
            state_t1_preview = StrategyState.from_mapping(row_t1)
            done = (
                bool(state_t1_preview.remaining_laps == 0)
                if state_t1_preview.remaining_laps is not None
                else idx == len(rows) - 2
            )
            transitions.append(
                _build_transition(
                    row_t,
                    row_t1,
                    action=action,
                    done=done,
                    action_space=action_space,
                    action_mask_config=action_mask_config,
                    reward_config=reward_config,
                    metadata={
                        "transition_kind": "adjacent_running_laps",
                        "semi_markov": False,
                        **_observed_action_label_metadata(row_t, action),
                    },
                )
            )
            idx += 1
    return transitions


def transition_prefix_fingerprint(transitions: Iterable[StrategyTransition], *, cutoff_lap: int) -> str:
    payload = [
        transition.fingerprint()
        for transition in transitions
        if int(transition.state_t.lap_number) <= int(cutoff_lap)
        and int(transition.state_t1.lap_number) <= int(cutoff_lap)
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
    "LEGAL_ACTION_MASK_EVIDENCE_VERSION",
    "LEAKAGE_CONTRACT_VERSION",
    "REWARD_SEMANTICS",
    "TRANSITION_FINGERPRINT_VERSION",
    "RewardConfig",
    "StrategyReward",
    "StrategyState",
    "StrategyTransition",
    "assert_replay_prefix_invariant",
    "build_replay_transitions",
    "compute_transition_reward",
    "infer_observed_action",
    "legal_action_mask_input_evidence",
    "transition_prefix_fingerprint",
]
