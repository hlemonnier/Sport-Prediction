"""Read one cached public TimingData response; no network or model fitting."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import sqlite3
from time import perf_counter

import numpy as np
import pandas as pd
import fastf1
from fastf1.utils import to_timedelta

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/"artifacts/research/boundary_20260908/sector_feasibility"
CACHE = ROOT/"data/f1/cache/fastf1_http_cache.sqlite"
SESSION = "2026/2026-03-08_Australian_Grand_Prix/2026-03-08_Race"
NATIVE = ROOT/"data/f1/cache"/SESSION/"_extended_timing_data.ff1pkl"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def seconds(value):
    try:
        parsed = to_timedelta(value) if isinstance(value, str) else value
        return float(parsed.total_seconds())
    except (AttributeError, TypeError, ValueError):
        return float("nan")


def run():
    started = perf_counter()
    connection = sqlite3.connect(f"file:{CACHE}?mode=ro", uri=True)
    races, sample, cache_key = [], None, None
    for key, blob in connection.execute("SELECT key,value FROM responses"):
        response = pickle.loads(blob)
        url = response.get("url", "")
        if url.endswith("/TimingData.jsonStream") and "_Race/" in url:
            races.append(dict(url=url, bytes=len(response["_content"])))
            if SESSION in url:
                sample, cache_key = response, key
    connection.close()
    assert sample is not None and sample["status_code"] == 200
    body = sample["_content"]
    packets, sectors, finish_updates, state = [], [], [], {}
    backwards, sector_count = 0, Counter()
    for sequence, line in enumerate(body.decode("utf-8-sig").splitlines()):
        time, payload = seconds(line[:12]), json.loads(line[12:])
        packets.append(time)
        for driver, patch in payload.get("Lines", {}).items():
            s = state.setdefault(driver, dict(completed=None))
            previous = s["completed"]
            current = patch.get("NumberOfLaps", previous)
            count_advanced = previous is not None and current is not None and current > previous
            if current is not None and previous is not None and current < previous:
                backwards += 1
            if current is not None and (previous is None or current > previous):
                finish_updates.append(dict(driver=driver, completed=int(current), time=time, sequence=sequence))
            if current is not None: s["completed"] = current
            values = patch.get("Sectors", {})
            if not isinstance(values, dict): continue
            for sector, update in values.items():
                if not isinstance(update, dict) or not update.get("Value"): continue
                duration = seconds(update["Value"])
                if not np.isfinite(duration) or duration <= 0: continue
                sector_count[sector] += 1
                sectors.append(dict(driver=driver, sector=int(sector)+1, time=time, seconds=duration,
                                    sequence=sequence, completed_seen=current, counter_advanced_in_packet=count_advanced))
    raw = pd.DataFrame(sectors)
    with NATIVE.open("rb") as handle: native = pickle.load(handle)["data"][0]
    native_rows, backdating_examples, timestamp_deltas = [], [], []
    for row in native.to_dict("records"):
        for sn in [1, 2]:
            duration, time = seconds(row[f"Sector{sn}Time"]), seconds(row[f"Sector{sn}SessionTime"])
            if not np.isfinite(duration+time): continue
            choices = raw.loc[raw.driver.eq(row["Driver"]) & raw.sector.eq(sn)
                              & np.isclose(raw.seconds, duration, atol=1e-9, rtol=0)
                              & raw.time.sub(time).abs().lt(10)]
            native_rows.append(dict(driver=row["Driver"], lap=int(row["NumberOfLaps"]), sector=sn,
                                    matched=not choices.empty))
            if choices.empty: continue
            observed = choices.iloc[np.argmin(choices.time.sub(time).abs().to_numpy())]
            delta = float(observed.time-time)
            timestamp_deltas.append(delta)
            if delta > .1 and len(backdating_examples) < 8:
                backdating_examples.append(dict(driver=row["Driver"], lap=int(row["NumberOfLaps"]), sector=sn,
                    duration=duration, processed_sector_time=time, raw_sector_packet_time=float(observed.time),
                    backdated_seconds=delta, raw_packet_sequence=int(observed.sequence)))
    s12 = raw.loc[raw.sector.isin([1, 2])].copy()
    labelled = s12.dropna(subset=["completed_seen"])
    first = labelled.drop_duplicates(["driver", "completed_seen", "sector"], keep="first")
    finishes = pd.DataFrame(finish_updates)
    lead_times, unresolved, leads_by_sector, capped_outcomes = [], [], {1:[], 2:[]}, []
    for row in first.to_dict("records"):
        # Coverage diagnostic only: future finish data never changes issuance.
        later = finishes.loc[finishes.driver.eq(row["driver"]) & finishes.time.gt(row["time"])
                             & finishes.completed.gt(row["completed_seen"])]
        if not later.empty:
            lead = float(later.time.min()-row["time"])
            lead_times.append(lead)
            leads_by_sector[row["sector"]].append(lead)
            capped_outcomes.append(min(lead, 180.0))
        else:
            unresolved.append({k:row[k] for k in ["driver", "sector", "time", "completed_seen"]})
            capped_outcomes.append(180.0 if max(packets) >= row["time"]+180.0 else float("nan"))
    def quantiles(values):
        return np.quantile(values, [0,.5,.9,.99,1]).tolist() if len(values) else []
    site = Path(fastf1.__file__).parent
    result = dict(status="feasible_raw_packet_replay_processed_timestamps_not_receipt_evidence",
        source_sha256=sha(Path(__file__).read_bytes()), fastf1_version=fastf1.__version__,
        primary_source_hashes={str(p):sha(p.read_bytes()) for p in [site/"_api.py", site/"core.py", site/"req.py"]},
        sample=dict(url=sample["url"], raw_bytes=len(body), raw_sha256=sha(body), cache_key=cache_key,
                    cache_path=str(CACHE.relative_to(ROOT)), cache_response_created_at=str(sample["created_at"]),
                    native_path=str(NATIVE.relative_to(ROOT)), native_sha256=sha(NATIVE.read_bytes()),
                    packet_count=len(packets), unique_drivers=len(state),
                    timestamp_min_max=[min(packets), max(packets)], nonmonotone_packet_steps=int((np.diff(packets)<0).sum())),
        cache_inventory=dict(race_responses=len(races), race_streams=races,
                             race_stream_size_min_median_max_bytes=np.quantile([r["bytes"] for r in races], [0,.5,1]).tolist()),
        sector_value_updates={str(int(k)+1):v for k,v in sector_count.items()},
        sector12=dict(value_updates=len(s12), missing_seen_lap_count=int(s12.completed_seen.isna().sum()),
                      updates_coincident_with_counter_advance=int(s12.counter_advanced_in_packet.sum()),
                      first_observed_driver_counter_sector_keys=len(first),
                      repeat_updates_within_same_observed_counter=int(len(labelled)-len(first)),
                      backwards_counter_updates=backwards,
                      first_update_time_to_next_counter_advance_quantiles_seconds=quantiles(lead_times),
                      first_update_time_to_next_counter_advance_by_sector={str(k):quantiles(v) for k,v in leads_by_sector.items()},
                      first_updates_with_subsequent_counter_advance=len(lead_times),
                      first_updates_without_subsequent_counter_advance=len(first)-len(lead_times),
                      unresolved_updates=unresolved,
                      capped180_outcomes_observed=int(np.isfinite(capped_outcomes).sum()),
                      capped180_outcomes_at_cap=int((np.asarray(capped_outcomes) == 180.0).sum()),
                      capped180_right_censored_outcomes=int((~np.isfinite(capped_outcomes)).sum()),
                      first_updates_with_at_least_180s_archive_followup=int(first.time.le(max(packets)-180).sum())),
        processed_cache=dict(lap_rows=len(native), sector12_finite_rows=len(native_rows),
                              sector12_matched_raw_value_within10s=sum(r["matched"] for r in native_rows),
                              raw_minus_processed_time_quantiles_seconds=quantiles(timestamp_deltas),
                              raw_after_processed_timestamp_rows=int((np.asarray(timestamp_deltas)>1e-9).sum()),
                              backdating_examples=backdating_examples),
        probe_wall_seconds=perf_counter()-started,
        limits=["Single2026session is a coverage/cost probe, not a score or discovery selection.",
                "Raw stream timestamps establish recorded packet ordering, not historical client receipt time.",
                "First driver/counter/sector counts are a feasibility diagnostic, not a verified causal lap-alignment parser.",
                "No source receipt times or labels were recovered from future lap endpoints for issuance counts.",
                "Native timestamp matching uses final data only to demonstrate postprocessing; it cannot define model issuance."])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"probe.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ["primary_source_hashes","cache_inventory"]}, indent=2))


if __name__ == "__main__": run()
