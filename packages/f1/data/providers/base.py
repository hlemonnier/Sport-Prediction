"""Data providers for FastF1 and OpenF1."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from packages.sports_core.paths import find_repo_root

from packages.f1.data.constants import POINTS_TABLE
from packages.f1.data.utils import (
    complete_classification_positions,
    first_available,
    merge_fp_frames,
    normalize_event_name,
)
from packages.f1.domain import (
    PredictionTarget,
    Session,
    SessionCutoff,
    WeekendFormat,
    build_weekend_contract,
    canonicalize_session_sequence,
    infer_weekend_contract,
    parse_session_cutoff,
)

try:
    import fastf1
except Exception:  # pragma: no cover - optional dependency
    fastf1 = None


def _parse_grid_position_status(value: object) -> tuple[float, str]:
    if value is None:
        return float("nan"), "missing"
    try:
        if pd.isna(value):
            return float("nan"), "missing"
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return float("nan"), "missing"
    lowered = text.lower().replace("-", " ").replace("_", " ")
    compact = re.sub(r"\s+", "", lowered)
    if compact in {"pl", "pitlane", "pit"} or "pit lane" in lowered:
        return float("nan"), "pit_lane"
    if compact in {"dns", "dnq", "wd", "withdrawn"} or "didnotstart" in compact:
        return float("nan"), "dns"
    if compact in {"dsq", "dq", "disqualified"}:
        return float("nan"), "disqualified"
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return float("nan"), "non_numeric"
    position = float(numeric)
    if position <= 0.0:
        return float("nan"), "pit_lane"
    return position, "grid"


def _assign_pit_lane_grid_positions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "grid_position" not in frame.columns or "grid_status" not in frame.columns:
        return frame
    out = frame.copy()
    valid_grid = pd.to_numeric(out["grid_position"], errors="coerce")
    pit_lane = out["grid_status"].astype(str).str.lower().eq("pit_lane")
    if pit_lane.any():
        max_grid = valid_grid.max(skipna=True)
        if pd.notna(max_grid):
            pit_positions = range(int(max_grid) + 1, int(max_grid) + 1 + int(pit_lane.sum()))
            out.loc[pit_lane, "grid_position"] = list(pit_positions)
    return out


def _standardize_grid_columns(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    driver_col = first_available(
        frame,
        ["driver_id", "DriverNumber", "driver_number", "Abbreviation", "Driver", "DriverId", "FullName"],
    )
    grid_col = first_available(
        frame,
        ["grid_position", "GridPosition", "Grid", "StartingGridPosition", "starting_grid_position", "grid"],
    )
    if driver_col is None or grid_col is None:
        return pd.DataFrame()
    out = pd.DataFrame()
    raw_driver_id = frame[driver_col]
    out["driver_id"] = raw_driver_id.where(raw_driver_id.notna(), "").astype(str).str.strip()
    invalid_driver_id = out["driver_id"].str.lower().isin({"", "nan", "none", "null", "<na>"})
    out.loc[invalid_driver_id, "driver_id"] = ""
    parsed = frame[grid_col].apply(_parse_grid_position_status)
    out["grid_position"] = parsed.map(lambda item: item[0])
    out["grid_status"] = parsed.map(lambda item: item[1])
    out = _assign_pit_lane_grid_positions(out)
    out["grid_source"] = str(source)
    out = out[out["driver_id"] != ""]
    return out


PACE_EVIDENCE_SESSIONS = frozenset(
    {Session.FP1, Session.FP2, Session.FP3, Session.SPRINT_QUALIFYING, Session.SPRINT},
)


def _prediction_target(value: str | PredictionTarget) -> PredictionTarget:
    if isinstance(value, PredictionTarget):
        return value
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "qualifying": PredictionTarget.GRAND_PRIX_QUALIFYING,
        "pre_quali": PredictionTarget.GRAND_PRIX_QUALIFYING,
        "grand_prix_qualifying": PredictionTarget.GRAND_PRIX_QUALIFYING,
        "race": PredictionTarget.RACE,
        "pre_race": PredictionTarget.RACE,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported prediction target for pace sessions: {value!r}") from exc


def _contract_for_provider_sessions(
    year: int,
    session_names: List[str],
    *,
    event_format_hint: Optional[str] = None,
):
    hint = str(event_format_hint or "").strip().lower()
    if "sprint" in hint or "alternative" in hint:
        if year in {2021, 2022}:
            weekend_format = WeekendFormat.SPRINT_2021_2022
        elif year == 2023:
            weekend_format = WeekendFormat.SPRINT_2023
        elif year >= 2024:
            weekend_format = WeekendFormat.SPRINT_2024_PLUS
        else:
            raise ValueError(f"Sprint format hint is unsupported for season {year}")
        return build_weekend_contract(year, weekend_format)
    return infer_weekend_contract(year, session_names)


def _eligible_pace_session_indices(
    *,
    year: int,
    session_names: List[str],
    prediction_target: str | PredictionTarget,
    session_cutoff: str | SessionCutoff | None,
    event_format_hint: Optional[str] = None,
) -> tuple[List[int], str, str]:
    if not session_names:
        return [], "before_weekend", WeekendFormat.STANDARD.value
    contract = _contract_for_provider_sessions(
        year,
        session_names,
        event_format_hint=event_format_hint,
    )
    target = _prediction_target(prediction_target)
    cutoff = parse_session_cutoff(contract, session_cutoff, target=target)
    eligible = set(contract.eligible_sessions(target, cutoff)).intersection(PACE_EVIDENCE_SESSIONS)
    canonical = canonicalize_session_sequence(year, session_names)
    indices = [index for index, session in enumerate(canonical) if session in eligible]
    return indices, cutoff.label, contract.format.value


class BaseProvider:
    def list_rounds(self, year: int) -> List[Dict[str, object]]:
        raise NotImplementedError

    def get_fp_features(
        self,
        year: int,
        round_number: int,
        *,
        session_cutoff: str | SessionCutoff | None = None,
        prediction_target: str | PredictionTarget = "qualifying",
        prediction_as_of: Optional[str] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def get_qualifying_results(self, year: int, round_number: int) -> pd.DataFrame:
        raise NotImplementedError

    def get_race_results(self, year: int, round_number: int) -> pd.DataFrame:
        raise NotImplementedError

    def get_starting_grid(
        self,
        year: int,
        round_number: int,
        *,
        prediction_as_of: Optional[str] = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def get_standings(self, year: int, round_number: int) -> Optional[pd.DataFrame]:
        return None

    def get_track_stats(self, year: int, round_number: int) -> Optional[dict[str, float]]:
        return None
