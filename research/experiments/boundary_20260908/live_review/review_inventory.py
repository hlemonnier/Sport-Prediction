"""Read only 2022–23 streams; inspect checkpoint support without fitting models.

The bundled HGB was fitted on these seasons. Its errors here are descriptive
training-era headroom, never independent validation or estimated attainable gain.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from packages.f1.models.live_race.next_lap import (
    MODEL_SHA256, forecast_laps, snapshot_lap_forecasts,
)
from packages.f1.models.live_race.next_lap_features import (
    REQUIRED, SECTORS, SPEEDS, driver_key, numeric, truth,
)

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(row):
    y = numeric(row.get("LapTime"))
    return (np.isfinite(y) and y > 0 and truth(row.get("IsAccurate"))
            and not any(np.isfinite(numeric(row.get(c))) for c in ("PitInTime", "PitOutTime"))
            and not any(c in str(row.get("TrackStatus", "")) for c in "4567"))


def cohort(row):
    if clean(row):
        return "eligible"
    if np.isfinite(numeric(row.get("PitOutTime"))):
        return "pit_out"
    if np.isfinite(numeric(row.get("PitInTime"))):
        return "pit_in"
    if any(c in str(row.get("TrackStatus", "")) for c in "4567"):
        return "neutralized"
    return "other_ineligible"


def inventory_event(raw, key):
    predictions = forecast_laps(raw, key)
    if predictions.status != "available":
        raise ValueError((key, predictions.status, predictions.reason))
    lookup = {(r.driver_id, r.issued_after_lap_number, r.issued_at_timestamp): r
              for r in predictions.frame.itertuples()}
    latest, waiting, matched = {}, defaultdict(list), []
    counts, missing = Counter(), defaultdict(Counter)
    rows = raw.assign(_driver=raw.DriverNumber.map(driver_key)).sort_values(
        ["Time", "_driver", "LapNumber"], kind="mergesort")
    for row in rows.to_dict("records"):
        d, lap, timestamp = row["_driver"], int(row["LapNumber"]), float(row["Time"])
        category = cohort(row)
        if category == "eligible":
            for checkpoint in waiting.pop(d, []):
                assert checkpoint["time"] < timestamp and checkpoint["lap"] < lap
                matched.append({**checkpoint, "target_lap": lap,
                                "target_time": timestamp,
                                "error": abs(checkpoint["point"] - float(row["LapTime"]))})
            if (d, lap, timestamp) in lookup:
                latest[d] = lookup[d, lap, timestamp]
        if d not in latest:
            counts["no_pending_point"] += 1
            continue
        counts["issued"] += 1
        counts[category] += 1
        for field in ["LapTime", "Stint", "Compound", "TyreLife", "FreshTyre", "Position", *SECTORS, *SPEEDS]:
            missing[category][field] += int(pd.isna(row.get(field)))
        previous = latest[d]
        waiting[d].append(dict(event_key=key, driver=d, lap=lap, time=timestamp,
                               category=category, point=float(previous.forecast_seconds),
                               point_lap=int(previous.issued_after_lap_number),
                               point_age_laps=lap-int(previous.issued_after_lap_number)))
    return matched, dict(counts), {k: dict(v) for k, v in missing.items()}, sum(map(len, waiting.values()))


def main():
    if (HERE / "inventory.json").exists():
        raise FileExistsError("review evidence is immutable; choose a fresh path")
    events, frames = [], []
    missing_totals = defaultdict(Counter)
    totals = Counter()
    for year in [2022, 2023]:
        for path in sorted((ROOT / "data/f1/raw/weekends" / str(year)).glob("*/*_race_laps.csv")):
            key = year*100 + int(path.parent.name.split("_")[1])
            raw = pd.read_csv(path)
            assert not set(REQUIRED) - set(raw)
            matched, counts, missing, unmatched = inventory_event(raw, key)
            frames.extend(matched)
            totals.update(counts)
            for name, fields in missing.items():
                missing_totals[name].update(fields)
            events.append(dict(event_key=key, path=str(path.relative_to(ROOT)), sha256=sha(path),
                               raw_rows=len(raw), columns=raw.columns.tolist(), counts=counts,
                               unmatched_checkpoints=unmatched,
                               pit_timestamps_after_completion=int(sum(
                                   (pd.to_numeric(raw[c], errors="coerce") > raw.Time).sum()
                                   for c in ["PitInTime", "PitOutTime"]))))
    f = pd.DataFrame(frames)
    per_event = []
    for key, group in f.groupby("event_key"):
        gated = group.category.ne("eligible")
        per_event.append(dict(event_key=int(key), checkpoint_mae=float(group.error.mean()),
                              checkpoints=len(group), ineligible_checkpoints=int(gated.sum()),
                              eligible_mae=float(group.loc[~gated, "error"].mean()),
                              ineligible_mae=float(group.loc[gated, "error"].mean()),
                              perfect_ineligible_expert_maximum_reduction_seconds=float(
                                  group.loc[gated, "error"].sum()/len(group))))
    unique = f.groupby(["event_key", "driver", "target_lap"]).size()
    result = dict(
        scope="2022_and_2023_only_no_fitting_no_later_scores",
        baseline_model_sha256=MODEL_SHA256,
        baseline_error_role="descriptive_in_sample_headroom_not_generalization_or_attainable_gain",
        input_events=events, totals=dict(totals),
        matched_checkpoints=len(f), distinct_driver_targets=len(unique),
        maximum_checkpoint_multiplicity_per_target=int(unique.max()),
        repeated_target_groups=int(unique.gt(1).sum()),
        matched_cohorts={str(k): dict(rows=len(g), mae=float(g.error.mean()),
                                    maximum_point_age_laps=int(g.point_age_laps.max()))
                         for k, g in f.groupby("category")},
        missing_counts_by_issued_cohort={k: dict(v) for k, v in missing_totals.items()},
        per_event=per_event,
        script_sha256=sha(Path(__file__)),
    )
    result["event_balanced_descriptive_mae"] = float(np.mean([e["checkpoint_mae"] for e in per_event]))
    result["perfect_ineligible_expert_upper_bound_event_balanced_reduction_seconds"] = float(np.mean(
        [e["perfect_ineligible_expert_maximum_reduction_seconds"] for e in per_event]))
    # Actual snapshot adapter: the next ineligible observation retains both
    # point value and original issuance metadata for the chosen driver.
    path = ROOT / events[0]["path"]
    raw = pd.read_csv(path)
    _, initial_matched = None, None
    found = False
    for _, row in raw.sort_values("Time").iterrows():
        if clean(row):
            continue
        previous = raw.loc[raw.Time.lt(row.Time)]
        if previous.empty:
            continue
        def snapshot(frame, as_of):
            return dict(sessionInfo={"session_type": "race"},
                        lapObservations=frame.to_dict("records"),
                        lapObservationsAsOfTimeSeconds=float(as_of))
        before, _ = snapshot_lap_forecasts(snapshot(previous, previous.Time.max()))
        d = driver_key(row.DriverNumber)
        if d not in before:
            continue
        through = raw.loc[raw.Time.le(row.Time)]
        after, _ = snapshot_lap_forecasts(snapshot(through, row.Time))
        assert before[d] == after[d]
        result["production_carry_probe"] = dict(event_key=events[0]["event_key"], driver=d,
                                               ineligible_lap=int(row.LapNumber),
                                               previous_issuance_lap=before[d]["issued_after_lap"],
                                               exact_payload_unchanged=True)
        found = True
        break
    assert found
    (HERE/"inventory.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"input_events", "per_event", "missing_counts_by_issued_cohort"}}, indent=2))


if __name__ == "__main__":
    main()
