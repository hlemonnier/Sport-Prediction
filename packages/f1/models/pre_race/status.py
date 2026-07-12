"""Reason-coded terminal status taxonomy for Race Final Position."""

from __future__ import annotations

from enum import Enum

import pandas as pd


class TerminalStatus(str, Enum):
    """Mutually exclusive status classes used by the joint race model."""

    DNS_WITHDRAWAL = "dns_withdrawal"
    MECHANICAL_POWER_UNIT = "mechanical_power_unit"
    COLLISION_INCIDENT = "collision_incident"
    NON_CLASSIFIED = "non_classified"
    CLASSIFIED_FINISH = "classified_finish"


TERMINAL_STATUSES: tuple[TerminalStatus, ...] = tuple(TerminalStatus)


def reason_code_terminal_status(value: object) -> TerminalStatus | None:
    """Map provider result text to the stable terminal taxonomy.

    Unknown or missing text returns ``None`` rather than guessing a target.
    """

    if isinstance(value, TerminalStatus):
        return value
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        # Non-scalar objects are not valid targets, but converting them to text
        # below still fails closed instead of invoking ambiguous truthiness.
        pass
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not text or text in {"nan", "none", "null", "unknown"}:
        return None
    direct = {status.value: status for status in TerminalStatus}
    if text in direct:
        return direct[text]

    if any(
        token in text
        for token in (
            "withdraw",
            "withdrew",
            "did_not_start",
            "dns",
            "not_started",
            "illness",
            "injured",
        )
    ):
        return TerminalStatus.DNS_WITHDRAWAL
    if any(
        token in text
        for token in (
            "engine",
            "power_unit",
            "gearbox",
            "hydraulic",
            "electrical",
            "mechanical",
            "transmission",
            "brake",
            "suspension",
            "oil",
            "water_pressure",
            "fuel_pressure",
            "overheating",
            "clutch",
            "differential",
            "driveshaft",
            "halfshaft",
            "turbo",
            "exhaust",
            "radiator",
            "electronics",
            "technical",
            "steering",
            "fire",
            "vibration",
            "fuel",
            "ers",
            "cooling",
            "power_loss",
            "water",
            "pump",
        )
    ):
        return TerminalStatus.MECHANICAL_POWER_UNIT
    if any(
        token in text
        for token in (
            "collision",
            "accident",
            "incident",
            "damage",
            "puncture",
            "spun",
            "crash",
            "front_wing",
            "undertray",
        )
    ):
        return TerminalStatus.COLLISION_INCIDENT
    if any(
        token in text
        for token in (
            "not_classified",
            "non_classified",
            "dnf",
            "retired",
            "disqualified",
            "excluded",
            "underweight",
        )
    ):
        return TerminalStatus.NON_CLASSIFIED
    if (text.startswith("+") and "lap" in text) or any(
        token in text
        for token in (
            "finished",
            "classified",
            "lapping",
            "lapped",
            "lap_behind",
            "+1_lap",
            "+2_laps",
            "+3_laps",
        )
    ):
        return TerminalStatus.CLASSIFIED_FINISH
    return None


def add_reason_coded_terminal_targets(
    frame: pd.DataFrame,
    *,
    raw_status_col: str = "race_status_raw",
    output_col: str = "terminal_status",
    strict: bool = True,
) -> pd.DataFrame:
    """Attach model targets while refusing to guess unknown result statuses."""

    if raw_status_col not in frame.columns:
        raise ValueError(f"missing raw terminal-status evidence: {raw_status_col}")
    out = frame.copy()
    encoded = out[raw_status_col].map(reason_code_terminal_status)
    unknown = encoded.isna()
    if strict and unknown.any():
        examples = sorted(out.loc[unknown, raw_status_col].astype(str).unique().tolist())[:5]
        raise ValueError(
            "unrecognized terminal status evidence; target construction failed closed: "
            f"{examples}"
        )
    out[output_col] = encoded.map(lambda value: value.value if isinstance(value, TerminalStatus) else None)
    out["terminal_status_evidence_complete"] = ~unknown
    return out


__all__ = [
    "TERMINAL_STATUSES",
    "TerminalStatus",
    "add_reason_coded_terminal_targets",
    "reason_code_terminal_status",
]
