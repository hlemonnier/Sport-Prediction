"""Time-based features shared by sport models."""

from __future__ import annotations

from datetime import datetime, timezone


def days_between(reference: datetime | None, target: datetime | None) -> float | None:
    if reference is None or target is None:
        return None
    return (target - reference).total_seconds() / 86400.0


def utc_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
