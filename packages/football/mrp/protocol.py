"""Explicit disjoint chronological populations for football model assessment."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from typing import Any
from .data import MatchRecord, match_available_at
from .utils import record_sort_key

MIN_FIT = 45
MIN_CALIBRATION = 30
MIN_SELECTION = 30
MIN_TEST = 30


def split_chronological_tail(matches: list[MatchRecord], minimum_tail: int) -> tuple[list[MatchRecord], list[MatchRecord]]:
    """Keep whole UTC matchdays together and purge unavailable prefix results."""
    rows = sorted(matches, key=record_sort_key)
    if not rows or any(m.date is None for m in rows):
        return [], []
    index = max(0, len(rows) - minimum_tail)
    if index == len(rows):
        return rows, []
    day = rows[index].date.date()
    while index > 0 and rows[index-1].date.date() == day:
        index -= 1
    tail = rows[index:]
    cutoff = min(m.date for m in tail)
    prefix = [m for m in rows[:index] if match_available_at(m) is not None and match_available_at(m) <= cutoff]
    return prefix, tail


@dataclass(frozen=True)
class ChronologicalPopulations:
    fit: list[MatchRecord]
    calibration: list[MatchRecord]
    selection: list[MatchRecord]
    test: list[MatchRecord]
    excluded_ids: list[str]

    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": "disjoint_chronological_fit_calibration_selection_test_v1",
            "evaluation_status": "held_out" if self.test else "insufficient_history",
            "parameter_policy": "Frozen fit-prefix parameters are used for evaluation and fixtures; completed history may update form features.",
            "availability_policy": "explicit result publication time, otherwise following UTC day",
            "excluded_boundary_ids": self.excluded_ids,
        }
        for name in ("fit", "calibration", "selection", "test"):
            rows = getattr(self,name)
            ids = [m.match_id for m in rows]
            content = json.dumps([asdict(m) for m in rows], default=lambda value:value.isoformat(), sort_keys=True,separators=(",",":"))
            result[name] = {"sample_size":len(rows),"match_ids":ids,
                "data_sha256":hashlib.sha256(content.encode()).hexdigest(),
                "population_sha256":hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest(),
                "first_kickoff":min((m.date.isoformat() for m in rows),default=None),
                "last_kickoff":max((m.date.isoformat() for m in rows),default=None),
                "latest_result_available_at":max((match_available_at(m).isoformat() for m in rows),default=None)}
        return result


def chronological_populations(matches: list[MatchRecord]) -> ChronologicalPopulations:
    rows = sorted(matches,key=record_sort_key)
    if any(m.date is None or match_available_at(m) is None for m in rows):
        raise ValueError("Chronological fitting requires known kickoff/result availability.")
    if len({m.match_id for m in rows}) != len(rows):
        raise ValueError("Match identities must be unique before chronological splitting.")
    n = len(rows)
    prefix, test = split_chronological_tail(rows,max(MIN_TEST,math.ceil(.2*n)))
    prefix, selection = split_chronological_tail(prefix,max(MIN_SELECTION,math.ceil(.15*n)))
    fit, calibration = split_chronological_tail(prefix,max(MIN_CALIBRATION,math.ceil(.15*n)))
    if len(fit)>=MIN_FIT and len(calibration)>=MIN_CALIBRATION and len(selection)>=MIN_SELECTION and len(test)>=MIN_TEST:
        groups = [fit,calibration,selection,test]
    else:
        fit,calibration=split_chronological_tail(rows,max(MIN_CALIBRATION,math.ceil(.2*n)))
        if len(fit)>=MIN_FIT and len(calibration)>=MIN_CALIBRATION:
            groups=[fit,calibration,[],[]]
        else:
            groups=[rows,[],[],[]]
    used={m.match_id for group in groups for m in group}
    return ChronologicalPopulations(*groups,[m.match_id for m in rows if m.match_id not in used])
