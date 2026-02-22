"""Utility helpers for football prediction."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .constants import AWAY_WIN_CLASS, DATA_DIR_COMPONENTS, DRAW_CLASS, HOME_WIN_CLASS

_NULL_STRINGS = {"", "na", "n/a", "nan", "none", "null"}


def normalize_text(value: Any) -> str:
    """Lowercase text normalization used for resilient matching."""
    if value is None:
        return ""
    return str(value).strip().lower()


def normalize_key(value: Any) -> str:
    return (
        normalize_text(value)
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace(".", "_")
    )


def canonical_team_id(value: Any) -> str:
    return normalize_text(value)


def _value_is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return normalize_text(value) in _NULL_STRINGS
    return False


def first_non_empty(row: Mapping[str, Any], candidate_keys: Sequence[str]) -> Any | None:
    if not row:
        return None
    normalized = {normalize_key(key): value for key, value in row.items() if key is not None}
    for key in candidate_keys:
        value = normalized.get(normalize_key(key))
        if _value_is_missing(value):
            continue
        if isinstance(value, str):
            return value.strip()
        return value
    return None


def parse_int(value: Any) -> int | None:
    if _value_is_missing(value):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value)
    text = str(value).strip()
    if normalize_text(text) in _NULL_STRINGS:
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return None


def parse_float(value: Any) -> float | None:
    if _value_is_missing(value):
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if normalize_text(text) in _NULL_STRINGS:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    if math.isnan(result):
        return None
    return result


def parse_datetime(value: Any) -> datetime | None:
    if _value_is_missing(value):
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    text = str(value).strip()
    if normalize_text(text) in _NULL_STRINGS:
        return None

    normalized = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        else:
            return None

    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def format_probability(value: float) -> str:
    return f"{clamp(value, 0.0, 1.0):.4f}"


def format_decimal(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def datetime_to_iso(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.strftime("%Y-%m-%d")


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    items = list(values)
    if not items:
        return default
    return sum(items) / len(items)


def outcome_class(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return HOME_WIN_CLASS
    if home_goals < away_goals:
        return AWAY_WIN_CLASS
    return DRAW_CLASS


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def discover_data_directory(data_source: str | None) -> tuple[Path | None, list[Path]]:
    """Find the local data/football directory from explicit and implicit roots."""
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add_candidate(path: Path) -> None:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(resolved)

    normalized_source = normalize_text(data_source)
    if normalized_source and normalized_source not in {"placeholder", "default", "local"}:
        source_path = Path(str(data_source)).expanduser()
        if source_path.is_file():
            add_candidate(source_path.parent)
        else:
            if source_path.name.lower() != DATA_DIR_COMPONENTS[-1]:
                add_candidate(source_path.joinpath(*DATA_DIR_COMPONENTS))
            add_candidate(source_path)

    search_roots = [Path.cwd(), *Path.cwd().parents]
    module_path = Path(__file__).resolve()
    search_roots.extend([module_path.parent, *module_path.parents])
    for root in search_roots:
        add_candidate(root.joinpath(*DATA_DIR_COMPONENTS))

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate, candidates
    return None, candidates


def record_sort_key(record: Any) -> tuple[datetime, int, int, str]:
    date_value = getattr(record, "date", None) or datetime.max
    season_value = getattr(record, "season", None)
    round_value = getattr(record, "round_number", None)
    match_id = str(getattr(record, "match_id", ""))
    return (
        date_value,
        season_value if isinstance(season_value, int) else 9999,
        round_value if isinstance(round_value, int) else 9999,
        match_id,
    )
