"""Validation helpers for tabular sport data contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def require_fields(row: Mapping[str, Any], fields: Iterable[str]) -> list[str]:
    """Return missing fields from a row-like mapping."""

    missing: list[str] = []
    for field in fields:
        value = row.get(field)
        if value is None or str(value).strip() == "":
            missing.append(field)
    return missing


def validate_probability(value: object, *, name: str = "probability") -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return probability
