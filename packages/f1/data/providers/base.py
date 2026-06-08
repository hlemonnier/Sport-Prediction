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
from packages.f1.data.utils import first_available, merge_fp_frames, normalize_event_name

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
    out["driver_id"] = frame[driver_col].astype(str).str.strip()
    parsed = frame[grid_col].apply(_parse_grid_position_status)
    out["grid_position"] = parsed.map(lambda item: item[0])
    out["grid_status"] = parsed.map(lambda item: item[1])
    out = _assign_pit_lane_grid_positions(out)
    out["grid_source"] = str(source)
    out = out[out["driver_id"] != ""]
    return out


class BaseProvider:
    def list_rounds(self, year: int) -> List[Dict[str, object]]:
        raise NotImplementedError

    def get_fp_features(self, year: int, round_number: int) -> pd.DataFrame:
        raise NotImplementedError

    def get_qualifying_results(self, year: int, round_number: int) -> pd.DataFrame:
        raise NotImplementedError

    def get_race_results(self, year: int, round_number: int) -> pd.DataFrame:
        raise NotImplementedError

    def get_starting_grid(self, year: int, round_number: int) -> pd.DataFrame:
        return pd.DataFrame()

    def get_standings(self, year: int, round_number: int) -> Optional[pd.DataFrame]:
        return None

    def get_track_stats(self, year: int, round_number: int) -> Optional[dict[str, float]]:
        return None
