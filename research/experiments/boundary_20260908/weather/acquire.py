"""Acquire public weather streams from FastF1's configured fallback mirror.

Numeric missing fields remain missing; this avoids FastF1 weather_data's default
zero imputation for omitted fields. No result labels are fetched or scored.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import requests
from fastf1 import _api

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / "artifacts/research/boundary_20260908/weather"
DATA = ROOT / "data/f1/boundary_20260908/weather"
CHANNELS = ["AirTemp","Humidity","Pressure","Rainfall","TrackTemp","WindDirection","WindSpeed"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def slug(value):
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    return re.sub("[^a-z0-9]+", "_", normalized).strip("_")


def legacy_slug(value):
    return re.sub("[^a-z0-9]+", "_", value.lower()).strip("_")


def parse(raw):
    rows = []
    for line in raw.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        time = pd.to_timedelta(line[:12]).total_seconds()
        entry = json.loads(line[12:])
        if not isinstance(entry, dict) or not np.isfinite(time):
            raise ValueError("Invalid weather record")
        row = {"Time":float(time)}
        for key in CHANNELS:
            value = entry.get(key)
            if key == "Rainfall":
                if value in ["0",0,False]:
                    number = 0.
                elif value in ["1",1,True]:
                    number = 1.
                else:
                    number = np.nan
            else:
                try:
                    number = float(value)
                    if not np.isfinite(number):
                        number = np.nan
                except (TypeError,ValueError):
                    number = np.nan
            row[key] = number
        rows.append(row)
    data = pd.DataFrame(rows, columns=["Time"]+CHANNELS)
    if data.empty or data.Time.duplicated().any() or not data.Time.is_monotonic_increasing:
        raise ValueError("Weather stream must be nonempty and chronologically unique")
    return data


def fetch(url, path, fallback_url=None):
    metadata_path = path.with_name(path.name+".source.json")
    if path.exists():
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            assert metadata["sha256"] == sha(path)
        else:
            # The first failed attempt fetched this exact mirror index before
            # the2023index404; no other metadata-less source is accepted.
            assert path.name == "2022_Index.json"
            metadata = {"url":url,"sha256":sha(path),"prior_attempt":"failed_acquisition_attempt_1.json"}
            write(metadata_path,metadata)
        return path.read_bytes(), "local_reuse", metadata["url"]
    failures = []
    for candidate in [u for u in [url,fallback_url] if u]:
        try:
            response = requests.get(candidate, headers=_api.headers, timeout=20)
            response.raise_for_status()
        except requests.RequestException as exc:
            failures.append({"url":candidate,"error":str(exc)})
            continue
        path.write_bytes(response.content)
        write(metadata_path,{"url":candidate,"sha256":sha(path),
            "downloaded_at":datetime.now(timezone.utc).isoformat(),"earlier_url_failures":failures})
        return response.content, "downloaded", candidate
    raise RuntimeError("Public source attempts failed: "+json.dumps(failures))


def acquire_one(item):
    event = item["event_key"]
    raw_path = DATA / f"{event}_WeatherData.jsonStream"
    csv_path = DATA / f"{event}_weather.csv"
    try:
        content, status, actual_url = fetch(item["url"], raw_path,item["fallback_url"])
        frame = parse(content)
        if csv_path.exists():
            existing = pd.read_csv(csv_path)
            pd.testing.assert_frame_equal(existing, frame, check_exact=False, atol=1e-12, rtol=0)
        else:
            frame.to_csv(csv_path, index=False)
        result = {**item,"status":status,"actual_download_url":actual_url,"raw_path":str(raw_path.relative_to(ROOT)),
            "raw_sha256":sha(raw_path),"csv_path":str(csv_path.relative_to(ROOT)),
            "csv_sha256":sha(csv_path),"rows":len(frame),"min_time":float(frame.Time.min()),
            "max_time":float(frame.Time.max()),"missing_by_channel":frame[CHANNELS].isna().sum().to_dict(),
            "observed_rain_rows":int(frame.Rainfall.eq(1).sum())}
        print("weather",event,len(frame),result["observed_rain_rows"],flush=True)
        return result
    except Exception as exc:
        print("weather_failure",event,type(exc).__name__,str(exc),flush=True)
        return {**item,"status":"failed","error_type":type(exc).__name__,"error":str(exc)}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    DATA.mkdir(parents=True,exist_ok=True)
    if (OUT / "input_manifest.json").exists():
        raise FileExistsError("Acquisition manifest already frozen")
    source = ROOT / "artifacts/research/frontier_20260907/live/selection.json"
    discovery = json.loads(source.read_text())["input_manifest"]
    index_by_year, index_sources = {}, []
    for year in [2022,2023]:
        url = _api.base_url_mirror+f"/static/{year}/Index.json"
        path = DATA / f"{year}_Index.json"
        raw, status, actual_url = fetch(url,path,_api.base_url+f"/static/{year}/Index.json")
        index = json.loads(raw.decode("utf-8-sig"))
        assert index["Year"] == year
        index_by_year[year] = index["Meetings"]
        index_sources.append({"year":year,"url":actual_url,"path":str(path.relative_to(ROOT)),"sha256":sha(path),"status":status})
    mappings = []
    for original in discovery:
        event = int(original["event_key"])
        path = ROOT / original["path"]
        assert sha(path) == original["sha256"]
        name = re.sub(r"^round_\d+_", "", path.parent.name)
        meetings = [m for m in index_by_year[event//100]
                    if name in {slug(m["Name"]),legacy_slug(m["Name"])}]
        if len(meetings) != 1:
            raise ValueError(f"Ambiguous meeting match for{event}:{name}")
        meeting = meetings[0]
        races = [s for s in meeting["Sessions"] if s.get("Name") == "Race"]
        if len(races) != 1:
            raise ValueError(f"Ambiguous race session for{event}")
        session = races[0]
        assert session["Path"].startswith(str(event//100)+"/")
        mappings.append({"event_key":event,"meeting_name":meeting["Name"],
            "mapping_method":"transliterated_slug" if slug(meeting["Name"]) == name else "legacy_ascii_regex_slug",
            "meeting_key":meeting["Key"],"provider_meeting_number":meeting.get("Number"),
            "session_key":session["Key"],"session_start_local":session["StartDate"],
            "session_gmt_offset":session["GmtOffset"],"session_path":session["Path"],
            "url":_api.base_url_mirror+"/static/"+session["Path"]+"WeatherData.jsonStream",
            "fallback_url":_api.base_url+"/static/"+session["Path"]+"WeatherData.jsonStream",
            "original_laps_path":original["path"],"original_laps_sha256":original["sha256"]})
    attempt = {"started_at":datetime.now(timezone.utc).isoformat(),"source_sha256":sha(__file__),
        "original_selection_sha256":sha(source),"index_sources":index_sources,"mappings":mappings,
        "scope":"2022-23 discovery weather acquisition only; no fitting or label scoring",
        "mirror_provenance":"Configured fallback _api.base_url_mirror of installed FastF1; original upstream Index returned403, mirror returned200.",
        "decoder":"12-character recorded session-clock prefix plus JSON; absent/invalid fields stay missing"}
    write(OUT / "acquisition_attempt.json",attempt)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(acquire_one,mappings))
    write(OUT / "input_manifest.json",{**attempt,"finished_at":datetime.now(timezone.utc).isoformat(),
        "events":results,"failed_events":[r["event_key"] for r in results if r["status"] == "failed"],
        "acquisition_only_no_performance_claim":True})
    print(json.dumps({"events":len(results),"failed_events":[r["event_key"] for r in results if r["status"] == "failed"]}))


if __name__ == "__main__":
    main()
