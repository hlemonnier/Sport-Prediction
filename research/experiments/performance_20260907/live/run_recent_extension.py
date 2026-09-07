"""Acquire at most four recent official races for the already frozen ridge.

No model fitting or parameter search occurs here. All acquired events and
failures are reported, and original caches/artifacts are never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import fastf1
import numpy as np
import pandas as pd

from research.experiments.performance_20260907.live.run_experiment import (
    HERE, ROOT, SPEC_PATH, diagnostics, ridge_predict, sha, stream_event,
)


def normalized(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")


def native_laps_to_seconds(laps):
    columns=["DriverNumber","Driver","LapNumber","Time","LapTime","Stint","Compound","TyreLife","TrackStatus","IsAccurate","PitInTime","PitOutTime"]
    raw=pd.DataFrame(laps[columns]).copy()
    for column in ["Time","LapTime","PitInTime","PitOutTime"]:
        if not pd.api.types.is_timedelta64_dtype(raw[column]):
            raise TypeError(f"native {column} must be timedelta, never unlabelled numeric nanoseconds")
        raw[column]=raw[column].dt.total_seconds()
    return raw


def bounded_call(function, label, attempts):
    for attempt in [1, 2]:
        try:
            value=function()
            attempts.append(dict(label=label, attempt=attempt, status="success"))
            return value
        except Exception as exc:
            attempts.append(dict(label=label, attempt=attempt, status="failed", exception=f"{type(exc).__name__}: {exc}"))
            print(json.dumps(attempts[-1]),flush=True)
            if attempt==1:
                time.sleep(2)
    return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=ROOT/"artifacts/research/performance_20260907/live/recent_extension")
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data/f1/performance_20260907/live_recent")
    args=parser.parse_args()
    out,data=args.output_dir,args.data_dir
    out.mkdir(parents=True,exist_ok=True);data.mkdir(parents=True,exist_ok=True)
    if (out/"results.json").exists() or (out/"frozen_before_acquisition.json").exists():
        raise FileExistsError("refusing to overwrite an extension acquisition or result")
    base=ROOT/"artifacts/research/performance_20260907/live/corrected_cycle_1"
    baseline_result=json.loads((base/"results.json").read_text())
    selected=json.loads((base/"selected_models.json").read_text())
    model=selected["final_ridge_model"]
    frozen_at=datetime.now(timezone.utc)
    freeze=dict(frozen_at_utc=frozen_at.isoformat(),selected_models_sha256=sha(base/"selected_models.json"),
        model_coefficients_sha256=hashlib.sha256(json.dumps(model,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        model=model,specification_sha256=sha(SPEC_PATH),feature_runner_sha256=sha(HERE/"run_experiment.py"),
        extension_runner_sha256=sha(__file__),base_results_sha256=sha(base/"results.json"),
        model_fit_years=[2022,2023],selection_year=2023,refit_permitted=False,maximum_absent_events=4,
        event_selection="earliest four official Race session dates after latest cached race date, absent by normalized event name, at least six hours before freeze",
        evidence_role="historical_2026_transfer_held_aside_in_this_cycle_not_prospective",
        load_flags=dict(laps=True,telemetry=False,weather=False,messages=False),maximum_attempts_per_acquisition=2)
    assert freeze["selected_models_sha256"]==baseline_result["selection_artifact_sha256"]
    assert freeze["feature_runner_sha256"]==baseline_result["runner_sha256"]
    assert freeze["specification_sha256"]==baseline_result["specification_sha256"]
    write_json(out/"frozen_before_acquisition.json",freeze)
    cache=data/"isolated_fastf1_cache";cache.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))
    attempts=[]

    def get_schedule():
        schedule=fastf1.get_event_schedule(2026,include_testing=False,backend="f1timing")
        if schedule is None or schedule.empty:
            raise RuntimeError("official F1 timing schedule unavailable")
        return schedule

    schedule=bounded_call(get_schedule,"official_2026_f1timing_schedule",attempts)
    if schedule is None:
        write_json(out/"results.json",dict(status="blocked_schedule_unavailable",freeze=freeze,acquisition_attempts=attempts,events=[],no_evaluation_performed=True))
        return
    schedule_rows=[]
    for _,event in schedule.iterrows():
        for number in range(1,6):
            if str(event.get(f"Session{number}"))=="Race":
                timestamp=pd.Timestamp(event[f"Session{number}DateUtc"])
                if pd.isna(timestamp):continue
                timestamp=timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
                schedule_rows.append(dict(round_number=int(event.RoundNumber),event_name=str(event.EventName),country=str(event.Country),race_start_utc=timestamp.isoformat()))
    write_json(out/"official_schedule_dates.json",dict(backend="f1timing",source="https://livetiming.formula1.com/static/2026/Index.json",events=schedule_rows))
    cached_names={normalized(" ".join(path.parts[-2].split("_")[2:])) for path in (ROOT/"data/f1/raw/weekends/2026").glob("*/*_race_laps.csv")}
    cached_dates=[pd.Timestamp(item["race_start_utc"]) for item in schedule_rows if normalized(item["event_name"]) in cached_names]
    if not cached_dates:raise ValueError("official schedule names cannot be matched to cached race names")
    latest_cached=max(cached_dates)
    eligible=[item for item in schedule_rows if normalized(item["event_name"]) not in cached_names
              and latest_cached<pd.Timestamp(item["race_start_utc"])<pd.Timestamp(frozen_at)-pd.Timedelta(hours=6)]
    chosen=sorted(eligible,key=lambda item:item["race_start_utc"])[:4]
    write_json(out/"chosen_events_before_lap_acquisition.json",dict(latest_cached_official_race_start_utc=latest_cached.isoformat(),chosen=chosen,all_absent_past_date_candidates=eligible))
    frames=[];events=[]
    for item in chosen:
        print(json.dumps({"acquiring":item}),flush=True)

        def fetch():
            session=schedule.get_event_by_round(item["round_number"]).get_session("Race")
            session.load(laps=True,telemetry=False,weather=False,messages=False)
            if session.laps is None or session.laps.empty:raise RuntimeError("race laps unavailable")
            statuses=session.session_status.Status.astype(str).tolist()
            if not set(statuses).intersection({"Finished","Finalised"}):
                raise RuntimeError("completed-session status not established")
            raw=native_laps_to_seconds(session.laps)
            return raw,statuses

        acquired=bounded_call(fetch,item["event_name"],attempts)
        if acquired is None:
            events.append({**item,"status":"acquisition_failed"})
            continue
        raw,statuses=acquired
        slug=re.sub(r"[^a-z0-9]+","_",item["event_name"].lower()).strip("_")
        path=data/f"2026_round_{item['round_number']:02d}_{slug}_race_laps.csv"
        if path.exists():raise FileExistsError(f"new acquisition would overwrite {path}")
        raw.to_csv(path,index=False)
        event_key=202600+item["round_number"]
        try:
            emitted,matched=stream_event(raw,event_key)
            if matched.empty:raise ValueError("no matched eligible forecasts")
            matched["prediction_ridge_correction"]=ridge_predict(matched,model)
            matched["event_name"]=item["event_name"]
            # Frozen coefficients and all raw features must be prefix invariant.
            cutoff=float(np.quantile(raw.Time,.6))
            prefix,_=stream_event(raw,event_key,end_at=cutoff)
            expected=emitted.loc[emitted.issued_at_timestamp<=cutoff].reset_index(drop=True)
            pd.testing.assert_frame_equal(expected,prefix.reset_index(drop=True),check_exact=True)
            np.testing.assert_array_equal(ridge_predict(expected,model),ridge_predict(prefix,model))
            event_result=diagnostics(matched,"prediction_ridge_correction")
            event_result.pop("ci95_delta_seconds")
            event_result.pop("bootstrap_fraction_improving")
            frames.append(matched)
            events.append({**item,"event_key":event_key,"status":"evaluated","data_path":path.relative_to(ROOT).as_posix(),
                           "data_sha256":sha(path),"raw_rows":len(raw),"issuances":len(emitted),"session_status_values":sorted(set(statuses)),
                           "prefix_invariance_passed":True,"metrics":event_result})
        except Exception as exc:
            events.append({**item,"event_key":event_key,"status":"evaluation_failed","data_path":path.relative_to(ROOT).as_posix(),
                           "data_sha256":sha(path),"exception":f"{type(exc).__name__}: {exc}"})
    result=dict(status="completed" if frames else "no_events_evaluated",freeze=freeze,events=events,acquisition_attempts=attempts,
                official_schedule_sha256=sha(out/"official_schedule_dates.json"),chosen_events_sha256=sha(out/"chosen_events_before_lap_acquisition.json"))
    if frames:
        scored=pd.concat(frames,ignore_index=True)
        scored.to_csv(out/"matched_forecasts.csv.gz",index=False)
        result["aggregate"]=diagnostics(scored,"prediction_ridge_correction")
        if result["aggregate"]["events"]<2:
            for field in ["ci95_delta_seconds","bootstrap_fraction_improving","loo_mean_deltas","loo_all_improve"]:
                result["aggregate"].pop(field,None)
        result["matched_forecasts_sha256"]=sha(out/"matched_forecasts.csv.gz")
        result["uncertainty_note"]="At most four events: event-bootstrap interval is descriptive and underpowered, not prospective validation."
    assert sha(base/"selected_models.json")==freeze["selected_models_sha256"]
    assert sha(SPEC_PATH)==freeze["specification_sha256"]
    assert sha(HERE/"run_experiment.py")==freeze["feature_runner_sha256"]
    assert sha(__file__)==freeze["extension_runner_sha256"]
    write_json(out/"results.json",result)
    print(json.dumps({"output":str(out/"results.json"),"status":result["status"],"events":events,"aggregate":result.get("aggregate")},indent=2),flush=True)


if __name__=="__main__":main()
