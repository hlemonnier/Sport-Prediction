"""Causal observed-lap features for the frozen September 2026 frontier.

Port of the frozen research encoder. Other-driver state is snapshotted before
an entire timestamp batch. Missing stints are propagated forwards only; pit,
compound and observed-stint transitions reset local histories. The target is
the next eligible clean completed lap, which may skip numbered laps.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FEATURES = [
    "level_gap",
    "own_last_delta",
    "own_robust_trend",
    "common_increment",
    "common_support",
    "log_stint_clean_count",
    "tyre_age",
    "clean_lap_gap",
    "wet_compound",
]
LEVELS = [(a, c) for a in (0.4, 0.7) for c in (1.0, 2.0)]
KEYS = ["event_key", "driver_id", "issued_after_lap_number", "issued_at_timestamp"]
FEATURE_PREFIX = "x_"
SECTORS = ["Sector1Time", "Sector2Time", "Sector3Time"]
SPEEDS = ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]


def numeric(value):
    try:
        out = float(value)
        return out if np.isfinite(out) else np.nan
    except (ValueError, TypeError):
        return np.nan


def truth(value):
    return str(value).lower().strip() in {"true", "1", "1.0"}


def driver_key(value):
    number = numeric(value)
    return (
        str(int(number))
        if np.isfinite(number) and number.is_integer()
        else str(value).strip()
    )


def level_name(a, c):
    return f"robust_level_a{a:g}_c{c:g}"


@dataclass
class DriverState:
    provider_stint: float = np.nan
    compound: str = ""
    last_lap: int = 0
    last_timestamp: float = -np.inf
    last_clean: float = np.nan
    last_clean_lap: int = 0
    total_clean: int = 0
    stint_clean: int = 0
    stint_generation: int = 0
    last_delta: float = 0.0
    deltas: deque = field(default_factory=lambda: deque(maxlen=4))
    levels: dict = field(default_factory=dict)
    common_record: tuple | None = None


def finite(value, default=0.0):
    value = numeric(value)
    return float(value) if np.isfinite(value) else float(default)


def _base_features(frame, event_key):
    """Emit causal base features after three eligible observations."""
    work = frame.copy()
    work["Time"] = pd.to_numeric(work["Time"], errors="coerce")
    if not np.isfinite(work.Time).all():
        raise ValueError("every observation requires a finite global timestamp")
    work["_driver"] = work.DriverNumber.map(driver_key)
    work = work.sort_values(["Time", "_driver", "LapNumber"], kind="mergesort")
    states, emitted = {}, []
    for timestamp, batch in work.groupby("Time", sort=False):
        common_before = {
            key: state.common_record
            for key, state in states.items()
            if state.common_record is not None
        }
        for row in batch.to_dict("records"):
            driver = row["_driver"]
            state = states.setdefault(driver, DriverState())
            lap_float = numeric(row["LapNumber"])
            if (
                not np.isfinite(lap_float)
                or lap_float != int(lap_float)
                or lap_float <= state.last_lap
                or timestamp <= state.last_timestamp
            ):
                raise ValueError(
                    f"invalid driver/lap chronology: {event_key}/{driver}/{lap_float}"
                )
            lap = int(lap_float)
            compound = str(row.get("Compound", "UNKNOWN")).upper()
            provider_stint = numeric(row.get("Stint"))
            pit_out = np.isfinite(numeric(row.get("PitOutTime")))
            pit_in = np.isfinite(numeric(row.get("PitInTime")))
            reset = (
                state.last_lap == 0
                or pit_out
                or (compound != state.compound and state.compound != "")
                or (
                    np.isfinite(provider_stint)
                    and np.isfinite(state.provider_stint)
                    and provider_stint != state.provider_stint
                )
            )
            if reset:
                state.stint_generation += 1
                state.stint_clean = 0
                state.levels.clear()
                state.deltas.clear()
                state.last_delta = 0.0
                state.common_record = None
                if not np.isfinite(provider_stint):
                    state.provider_stint = np.nan
            if np.isfinite(provider_stint):
                state.provider_stint = provider_stint
            state.compound = compound
            state.last_lap, state.last_timestamp = lap, float(timestamp)
            y = numeric(row.get("LapTime"))
            status = str(row.get("TrackStatus", ""))
            eligible = (
                np.isfinite(y)
                and y > 0
                and truth(row.get("IsAccurate", False))
                and not (pit_in or pit_out)
                and not any(code in status for code in "4567")
            )
            if not eligible:
                continue
            gap = lap - state.last_clean_lap if state.stint_clean else 1
            if state.stint_clean:
                delta = (y - state.last_clean) / gap
                state.last_delta = float(np.clip(delta, -1.0, 1.0))
                state.deltas.append(state.last_delta)
                # Only successive clean laps provide a local common increment.
                state.common_record = (
                    (float(timestamp), lap, state.last_delta) if gap == 1 else None
                )
            else:
                state.common_record = None
            for a, c in LEVELS:
                old = state.levels.get((a, c), y)
                state.levels[(a, c)] = old + a * float(np.clip(y - old, -c, c))
            state.last_clean, state.last_clean_lap = y, lap
            state.total_clean += 1
            state.stint_clean += 1
            own_trend = (
                float(np.median(state.deltas)) if len(state.deltas) >= 2 else 0.0
            )
            others = [
                record
                for key, record in common_before.items()
                if key != driver
                and 0 < timestamp - record[0] <= 180.0
                and abs(lap - record[1]) <= 1
            ]
            common = (
                float(np.median([record[2] for record in others]))
                if len(others) >= 3
                else 0.0
            )
            if state.total_clean < 3:
                continue
            tyre_age = numeric(row.get("TyreLife"))
            features = [
                state.levels[(0.7, 2.0)] - y,
                state.last_delta,
                own_trend,
                common,
                min(len(others), 10) / 10.0,
                np.log1p(state.stint_clean),
                min(max(tyre_age, 0.0), 100.0)
                if np.isfinite(tyre_age)
                else float(state.stint_clean),
                float(gap),
                float(compound in {"INTERMEDIATE", "WET"}),
            ]
            issuance = dict(
                event_key=int(event_key),
                driver_id=driver,
                issued_after_lap_number=lap,
                issued_at_timestamp=float(timestamp),
                forecast_naive_seconds=y,
                stint_generation=state.stint_generation,
                compound=compound,
                stint_clean_count=state.stint_clean,
                common_other_drivers=len(others),
                common_evidence_max_timestamp=max(
                    (record[0] for record in others), default=np.nan
                ),
            )
            issuance.update(zip(FEATURES, features))
            emitted.append(issuance)
    return pd.DataFrame(emitted)


REQUIRED = [
    "DriverNumber",
    "LapNumber",
    "Time",
    "LapTime",
    "IsAccurate",
    "Stint",
    "Compound",
    "TyreLife",
    "PitInTime",
    "PitOutTime",
    "TrackStatus",
    *SECTORS,
    *SPEEDS,
    "Position",
    "FreshTyre",
]


def validate_schema(raw):
    missing = sorted(set(REQUIRED) - set(raw))
    if missing:
        raise ValueError("Observed-input schema is incomplete: " + ", ".join(missing))
    for column in [
        "Time",
        "LapTime",
        "PitInTime",
        "PitOutTime",
        *SECTORS,
        *SPEEDS,
        "Position",
        "TyreLife",
    ]:
        if not pd.api.types.is_numeric_dtype(raw[column]):
            raise TypeError(
                column
                + " must have explicit numeric seconds or native speed/position units"
            )


def observed_features(raw, event_key):
    validate_schema(raw)
    issued = _base_features(raw, event_key)
    if issued.empty:
        return issued
    work = raw.copy()
    work["Time"] = pd.to_numeric(work.Time)
    work["_driver"] = work.DriverNumber.map(driver_key)
    work = work.sort_values(["Time", "_driver", "LapNumber"], kind="mergesort")
    states, extra = {}, []
    issuance_keys = set(
        zip(
            issued.driver_id, issued.issued_after_lap_number, issued.issued_at_timestamp
        )
    )
    for timestamp, batch in work.groupby("Time", sort=False):
        peers = {d: s["peer"] for d, s in states.items() if s["peer"] is not None}
        for row in batch.to_dict("records"):
            d, lap = row["_driver"], int(row["LapNumber"])
            s = states.setdefault(
                d,
                {
                    "stint": np.nan,
                    "compound": "",
                    "history": deque(maxlen=12),
                    "ewma": {},
                    "peer": None,
                    "generation": 0,
                },
            )
            stint, compound = (
                numeric(row.get("Stint")),
                str(row.get("Compound", "UNKNOWN")).upper(),
            )
            pit_out = np.isfinite(numeric(row.get("PitOutTime")))
            pit_in = np.isfinite(numeric(row.get("PitInTime")))
            reset = (
                s["generation"] == 0
                or pit_out
                or (s["compound"] and compound != s["compound"])
                or (
                    np.isfinite(stint)
                    and np.isfinite(s["stint"])
                    and stint != s["stint"]
                )
            )
            if reset:
                s["history"].clear()
                s["ewma"].clear()
                s["peer"] = None
                s["generation"] += 1
                if not np.isfinite(stint):
                    s["stint"] = np.nan
            if np.isfinite(stint):
                s["stint"] = stint
            s["compound"] = compound
            y = numeric(row.get("LapTime"))
            eligible = (
                np.isfinite(y)
                and y > 0
                and truth(row.get("IsAccurate", False))
                and not (pit_in or pit_out)
                and not any(c in str(row.get("TrackStatus", "")) for c in "4567")
            )
            if not eligible:
                continue
            previous = s["history"][-1] if s["history"] else None
            delta = (y - previous["y"]) / (lap - previous["lap"]) if previous else 0.0
            s["peer"] = {
                "time": timestamp,
                "lap": lap,
                "y": y,
                "delta": float(np.clip(delta, -5, 5)),
            }
            item = {
                "lap": lap,
                "y": y,
                **{k: numeric(row.get(k)) for k in SECTORS + SPEEDS},
            }
            s["history"].append(item)
            for alpha in [0.25, 0.5, 0.75]:
                old = s["ewma"].get(alpha, y)
                s["ewma"][alpha] = y if abs(y - old) > 2 else old + alpha * (y - old)
            if (d, lap, timestamp) not in issuance_keys:
                continue
            h = list(s["history"])
            yy = np.array([i["y"] for i in h])
            ll = np.array([i["lap"] for i in h])
            features = {
                "observed_lap": lap,
                "position": finite(row.get("Position"), 11),
                "lap_duration": y,
                "fresh_tyre": float(truth(row.get("FreshTyre"))),
                "history_count": len(h),
            }
            for k in range(1, 7):
                features[f"lag_{k}_gap"] = (
                    float(np.clip(yy[-k - 1] - y, -15, 15)) if len(yy) > k else 0.0
                )
                features[f"lag_{k}_lap_distance"] = (
                    float(lap - ll[-k - 1]) if len(yy) > k else 0.0
                )
            experts = {}
            for k in [2, 3, 5, 8]:
                v, l = yy[-k:], ll[-k:]
                med = float(np.median(v))
                avg = float(v.mean())
                slope = (
                    float(
                        np.dot(l - l.mean(), v - v.mean())
                        / np.square(l - l.mean()).sum()
                    )
                    if len(v) >= 2
                    else 0.0
                )
                slope = float(np.clip(slope, -1, 1))
                features.update(
                    {
                        f"mean_{k}_gap": float(np.clip(avg - y, -15, 15)),
                        f"median_{k}_gap": float(np.clip(med - y, -15, 15)),
                        f"mad_{k}": float(np.median(np.abs(v - med))),
                        f"trend_{k}": slope,
                        f"range_{k}": float(np.clip(v.max() - v.min(), 0, 20)),
                    }
                )
                experts[f"mean_{k}"] = y + float(np.clip(avg - y, -3, 3))
                experts[f"median_{k}"] = y + float(np.clip(med - y, -3, 3))
                experts[f"trend_{k}"] = y + float(
                    np.clip(avg + slope * (lap + 1 - l.mean()) - y, -3, 3)
                )
            for alpha, value in s["ewma"].items():
                features[f"reset_ewma_{alpha}_gap"] = value - y
                experts[f"reset_ewma_{alpha}"] = value
            for col in SECTORS + SPEEDS:
                current = finite(row.get(col))
                vals = np.array([v[col] for v in h[-5:]])
                vals = vals[np.isfinite(vals)]
                features[col] = current
                features[col + "_missing"] = float(
                    not np.isfinite(numeric(row.get(col)))
                )
                features[col + "_median_gap"] = (
                    float(np.clip(np.median(vals) - current, -30, 30))
                    if len(vals)
                    else 0.0
                )
            for c in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
                features["compound_" + c] = float(compound == c)
            others = [
                p
                for driver, p in peers.items()
                if driver != d
                and 0 < timestamp - p["time"] <= 180
                and abs(lap - p["lap"]) <= 1
            ]
            features["peer_count"] = len(others)
            features["peer_delta_mean"] = (
                float(np.mean([p["delta"] for p in others])) if others else 0.0
            )
            features["peer_delta_median"] = (
                float(np.median([p["delta"] for p in others])) if others else 0.0
            )
            features["peer_delta_mad"] = (
                float(
                    np.median(
                        np.abs(
                            np.array([p["delta"] for p in others])
                            - features["peer_delta_median"]
                        )
                    )
                )
                if others
                else 0.0
            )
            features["relative_field_pace"] = (
                float(np.clip(y - np.median([p["y"] for p in others]), -15, 15))
                if others
                else 0.0
            )
            extra.append(
                dict(
                    event_key=event_key,
                    driver_id=d,
                    issued_after_lap_number=lap,
                    issued_at_timestamp=float(timestamp),
                    **{FEATURE_PREFIX + k: float(v) for k, v in features.items()},
                    **{"expert_" + k: float(v) for k, v in experts.items()},
                )
            )
    extra = pd.DataFrame(extra)
    enriched = issued.merge(extra, on=KEYS, validate="one_to_one")
    assert len(enriched) == len(issued)
    return enriched
