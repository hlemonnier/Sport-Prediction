"""Driver identity resolution shared by evaluation and betting workflows."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Optional
import unicodedata

import pandas as pd


DRIVER_NUMBER_COLUMNS = (
    "driver_number",
    "DriverNumber",
    "driverNo",
    "racing_number",
    "permanent_number",
    "PermanentNumber",
    "number",
)
ABBREVIATION_COLUMNS = (
    "abbreviation",
    "Abbreviation",
    "driver_abbreviation",
    "driver_code",
    "DriverCode",
    "Tla",
    "tla",
)
PROVIDER_ID_COLUMNS = (
    "driver_id",
    "DriverId",
    "driver_uuid",
    "openf1_driver_id",
    "fastf1_driver_id",
)
FULL_NAME_COLUMNS = (
    "driver_name",
    "Driver",
    "FullName",
    "full_name",
    "driver_full_name",
    "BroadcastName",
    "broadcast_name",
    "selection",
    "participant",
)
FIRST_NAME_COLUMNS = ("FirstName", "first_name", "given_name")
LAST_NAME_COLUMNS = ("LastName", "last_name", "family_name", "surname")


_KNOWN_DRIVER_ALIAS_GROUPS = {
    "max_verstappen": {"1", "33", "ver", "max verstappen", "verstappen"},
    "yuki_tsunoda": {"22", "tsu", "yuki tsunoda", "tsunoda"},
    "charles_leclerc": {"16", "lec", "charles leclerc", "leclerc"},
    "lewis_hamilton": {"44", "ham", "lewis hamilton", "hamilton"},
    "george_russell": {"63", "rus", "george russell", "russell"},
    "andrea_kimi_antonelli": {"12", "ant", "kimi antonelli", "andrea kimi antonelli", "antonelli"},
    "lando_norris": {"4", "nor", "lando norris", "norris"},
    "oscar_piastri": {"81", "pia", "oscar piastri", "piastri"},
    "fernando_alonso": {"14", "alo", "fernando alonso", "alonso"},
    "lance_stroll": {"18", "str", "lance stroll", "stroll"},
    "pierre_gasly": {"10", "gas", "pierre gasly", "gasly"},
    "franco_colapinto": {"43", "col", "franco colapinto", "colapinto"},
    "esteban_ocon": {"31", "oco", "esteban ocon", "ocon"},
    "oliver_bearman": {"87", "bea", "ollie bearman", "oliver bearman", "bearman"},
    "isack_hadjar": {"6", "had", "isack hadjar", "hadjar"},
    "liam_lawson": {"30", "law", "liam lawson", "lawson"},
    "alex_albon": {"23", "alb", "alex albon", "alexander albon", "albon"},
    "carlos_sainz": {"55", "sai", "carlos sainz", "sainz"},
    "nico_hulkenberg": {"27", "hul", "nico hulkenberg", "nico hulkenberg", "hulkenberg"},
    "gabriel_bortoleto": {"5", "bor", "gabriel bortoleto", "bortoleto"},
    "valtteri_bottas": {"77", "bot", "valtteri bottas", "bottas"},
    "zhou_guanyu": {"24", "zho", "zhou guanyu", "guanyu zhou", "zhou"},
    "sergio_perez": {"11", "per", "sergio perez", "perez"},
    "daniel_ricciardo": {"3", "ric", "daniel ricciardo", "ricciardo"},
    "kevin_magnussen": {"20", "mag", "kevin magnussen", "magnussen"},
    "logan_sargeant": {"2", "sar", "logan sargeant", "sargeant"},
}
KNOWN_DRIVER_ALIASES = {
    alias: canonical
    for canonical, aliases in _KNOWN_DRIVER_ALIAS_GROUPS.items()
    for alias in aliases
}


def missing(value: object) -> bool:
    if value is None or pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def normalize_text(value: object) -> str:
    if missing(value):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = " ".join(text.split())
    if not text:
        return ""
    return text


def numeric_token(value: object) -> str:
    if missing(value):
        return ""
    text = str(value).strip()
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(numeric) and float(numeric).is_integer():
        number = int(numeric)
        if 0 < number < 1000:
            return str(number)
    return ""


def _add_known_aliases(aliases: set[tuple[int, str]], row: pd.Series) -> None:
    columns = (
        *DRIVER_NUMBER_COLUMNS,
        *PROVIDER_ID_COLUMNS,
        *ABBREVIATION_COLUMNS,
        *FULL_NAME_COLUMNS,
        *FIRST_NAME_COLUMNS,
        *LAST_NAME_COLUMNS,
    )
    for col in columns:
        if col not in row.index:
            continue
        tokens = {normalize_text(row.get(col)), numeric_token(row.get(col))}
        for token in tokens:
            canonical = KNOWN_DRIVER_ALIASES.get(token)
            if canonical:
                aliases.add((0, f"known:{canonical}"))


def row_identity_aliases(row: pd.Series) -> set[tuple[int, str]]:
    aliases: set[tuple[int, str]] = set()
    _add_known_aliases(aliases, row)

    for col in DRIVER_NUMBER_COLUMNS:
        if col in row.index:
            token = numeric_token(row.get(col))
            if token:
                aliases.add((1, f"number:{token}"))

    for col in PROVIDER_ID_COLUMNS:
        if col not in row.index:
            continue
        value = row.get(col)
        token = numeric_token(value)
        if token:
            aliases.add((1, f"number:{token}"))
            continue
        clean = normalize_text(value)
        if clean:
            aliases.add((3, f"id:{clean}"))
            if re.fullmatch(r"[a-z]{2,4}", clean):
                aliases.add((2, f"abbr:{clean}"))

    for col in ABBREVIATION_COLUMNS:
        if col in row.index:
            clean = normalize_text(row.get(col))
            if clean and len(clean) <= 5 and " " not in clean:
                aliases.add((2, f"abbr:{clean}"))

    first = next((normalize_text(row.get(col)) for col in FIRST_NAME_COLUMNS if col in row.index), "")
    last = next((normalize_text(row.get(col)) for col in LAST_NAME_COLUMNS if col in row.index), "")
    if first and last:
        aliases.add((4, f"name:{first} {last}"))
        aliases.add((5, f"surname:{last}"))

    for col in FULL_NAME_COLUMNS:
        if col not in row.index:
            continue
        name = normalize_text(row.get(col))
        if not name:
            continue
        if len(name) <= 5 and " " not in name and col in {"Driver", "driver_id", "selection", "participant"}:
            aliases.add((2, f"abbr:{name}"))
        aliases.add((4, f"name:{name}"))
        parts = name.split()
        if len(parts) >= 2:
            aliases.add((5, f"surname:{parts[-1]}"))

    return aliases


def driver_identity_signature(row: pd.Series) -> str:
    aliases = sorted(alias for _, alias in row_identity_aliases(row))
    return "|".join(aliases)


def driver_key_column(frame: pd.DataFrame) -> Optional[str]:
    for col in [
        *PROVIDER_ID_COLUMNS,
        *DRIVER_NUMBER_COLUMNS,
        *ABBREVIATION_COLUMNS,
        *FULL_NAME_COLUMNS,
        *FIRST_NAME_COLUMNS,
        *LAST_NAME_COLUMNS,
    ]:
        if col in frame.columns:
            return col
    return None


def resolve_driver_matches(pred: pd.DataFrame, actual: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    pred_aliases = {idx: row_identity_aliases(row) for idx, row in pred.iterrows()}
    actual_aliases = {idx: row_identity_aliases(row) for idx, row in actual.iterrows()}
    pred_by_alias: dict[tuple[int, str], set[Any]] = defaultdict(set)
    actual_by_alias: dict[tuple[int, str], set[Any]] = defaultdict(set)
    for idx, aliases in pred_aliases.items():
        for alias in aliases:
            pred_by_alias[alias].add(idx)
    for idx, aliases in actual_aliases.items():
        for alias in aliases:
            actual_by_alias[alias].add(idx)

    matches: list[dict[str, Any]] = []
    used_actual: set[Any] = set()
    ambiguous_predictions = 0
    rank_col = "pred_rank" if "pred_rank" in pred.columns else None
    pred_order = pred.sort_values(rank_col, kind="mergesort").index if rank_col else pred.index
    for pred_idx in pred_order:
        aliases = pred_aliases.get(pred_idx, set())
        matched_actual: Optional[Any] = None
        matched_alias = ""
        ambiguous = False
        for priority in sorted({priority for priority, _ in aliases}):
            candidates: set[Any] = set()
            candidate_aliases: list[tuple[int, str]] = []
            for alias in aliases:
                if alias[0] != priority:
                    continue
                if len(pred_by_alias.get(alias, set())) != 1 or len(actual_by_alias.get(alias, set())) != 1:
                    if alias in actual_by_alias:
                        ambiguous = True
                    continue
                actual_idx = next(iter(actual_by_alias[alias]))
                if actual_idx in used_actual:
                    continue
                candidates.add(actual_idx)
                candidate_aliases.append(alias)
            if len(candidates) == 1:
                matched_actual = next(iter(candidates))
                matched_alias = candidate_aliases[0][1]
                break
            if len(candidates) > 1:
                ambiguous = True
        if matched_actual is None:
            ambiguous_predictions += int(ambiguous)
            continue
        used_actual.add(matched_actual)
        matches.append(
            {
                "pred_index": pred_idx,
                "actual_index": matched_actual,
                "matched_alias": matched_alias,
            }
        )

    return pd.DataFrame(matches), {
        "ambiguous_prediction_count": int(ambiguous_predictions),
        "unmatched_prediction_count": int(len(pred) - len(matches)),
        "unmatched_actual_count": int(len(actual) - len(matches)),
    }

