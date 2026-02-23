"""Shared runtime helpers for F1 CLIs."""

from __future__ import annotations

import json


def parse_train_seasons(value: str, target_year: int, train_policy: str) -> list[int]:
    if value.lower() not in {"auto", "default"}:
        return sorted({int(x.strip()) for x in value.split(",") if x.strip()})

    if train_policy == "strict_transfer":
        seasons = [target_year - 4, target_year - 3, target_year - 2, target_year]
    elif train_policy == "rolling":
        seasons = [target_year - 3, target_year - 2, target_year - 1, target_year]
    elif train_policy == "frozen_preseason":
        seasons = [target_year - 4, target_year - 3, target_year - 2]
    else:
        seasons = [target_year - 2, target_year - 1, target_year]
    return sorted({int(y) for y in seasons if int(y) > 0})


def parse_compare_families(value: str) -> list[str]:
    families = [part.strip().lower() for part in str(value).split(",") if part.strip()]
    if not families:
        return ["ml"]
    allowed = {"ml", "dl", "baseline"}
    return [f for f in families if f in allowed] or ["ml"]


def parse_json_object(value: str, arg_name: str) -> dict[str, object]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid {arg_name} JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid {arg_name}: expected JSON object.")
    return payload
