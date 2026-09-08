"""Bounded public Understat league pages; no authentication or prediction fitting."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data/football/boundary_20260908/understat"
OUT = ROOT / "artifacts/research/boundary_20260908/football_xg_data"
LEAGUES = {"EPL": "E0", "La_liga": "SP1", "Serie_A": "I1"}
YEARS = (2022, 2023, 2024, 2025)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def canonical_inputs():
    paths = []
    base = ROOT / "data/football/performance_20260907"
    for league, code in LEAGUES.items():
        for year in YEARS:
            name = ("" if code == "E0" else "transfer_")+f"{code}_{year}_{year+1}.csv"
            paths.append(base / name)
    paths += [base / "source_manifest.json", base / "transfer_source_manifest.json"]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def fetch(league, year):
    assert league in LEAGUES and year in YEARS
    url = f"https://understat.com/league/{league}/{year}"
    path = DATA / f"{league}_{year}.html"
    receipt = path.with_suffix(".receipt.json")
    if path.exists() or receipt.exists():
        saved = json.loads(receipt.read_text())
        assert saved["requested_url"] == url and saved["sha256"] == sha(path)
        return saved
    with requests.get(url, timeout=45, stream=True, headers={"User-Agent": "SportPredictionResearch/1.0"}) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        payload = response.raw.read(5_000_001)
        if len(payload) > 5_000_000 or b"datesData" not in payload:
            raise ValueError("Unexpected or oversized public league page")
        if not response.url.startswith("https://understat.com/"):
            raise ValueError("Unexpected non-primary redirect")
        path.write_bytes(payload)
        saved = {"league": league, "canonical_league": LEAGUES[league], "season_start_year": year,
            "requested_url": url, "actual_url": response.url, "http_status": response.status_code,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "sha256": sha(path),
            "bytes": len(payload), "content_type": response.headers.get("Content-Type"),
            "last_modified": response.headers.get("Last-Modified"), "etag": response.headers.get("ETag"),
            "path": str(path.relative_to(ROOT)), "receipt_path": str(receipt.relative_to(ROOT)),
            "acquisition_source_sha256": sha(__file__)}
        write(receipt, saved)
        return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Fetch at most the twelve preregistered public pages")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "acquisition_lock.json"
    expected = {"source_sha256": sha(__file__), "canonical_inputs": canonical_inputs(),
        "pages": [{"league": league, "season_start_year": year} for league in LEAGUES for year in YEARS],
        "purpose": "Completed-match xG feasibility and exact date/team/score joins only; no fitting or predictive scores",
        "timing_limit": "Historical publication and revision times are unverified; acquisition time is not original availability"}
    if lock.exists():
        existing = json.loads(lock.read_text())
        assert {k: existing[k] for k in expected} == expected
    else:
        write(lock, {**expected, "frozen_at_utc": datetime.now(timezone.utc).isoformat()})
    records, failures = [], []
    requests_to_make = [(league, year) for league in LEAGUES for year in YEARS] if args.all else [("EPL", 2022)]
    for league, year in requests_to_make:
        try:
            record = fetch(league, year); records.append(record)
            print(json.dumps(record), flush=True)
        except Exception as exc:
            failures.append({"league": league, "season_start_year": year, "error": repr(exc)})
            print("failed", league, year, repr(exc), flush=True)
    assert canonical_inputs() == expected["canonical_inputs"] and sha(__file__) == expected["source_sha256"]
    filename = "acquisition_manifest.json" if args.all else "probe_manifest.json"
    output = OUT / filename
    if output.exists():
        raise FileExistsError("immutable acquisition attempt already exists")
    write(output, {"lock_sha256": sha(lock), "pages": records, "failures": failures,
                   "finished_at_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
