"""Use the current first-party endpoint explicitly linked by public league.min.js."""
import argparse
from datetime import datetime, timezone
import json
from urllib.parse import quote

import requests

from research.experiments.boundary_20260908.football_xg_data import acquire


def fetch(league, year):
    public_name = league.replace("_", " ")
    url = f"https://understat.com/getLeagueData/{quote(public_name, safe='')}/{year}"
    path = acquire.DATA / f"{league}_{year}_league_data.json"
    receipt_path = path.with_suffix(".receipt.json")
    if path.exists() or receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt["requested_url"] == url and receipt["sha256"] == acquire.sha(path)
        return receipt
    response = requests.get(url, timeout=45, headers={"User-Agent": "SportPredictionResearch/1.0"})
    response.raise_for_status()
    if len(response.content) > 5_000_000 or not response.url.startswith("https://understat.com/"):
        raise ValueError("Unexpected public league-data response")
    data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get("dates"), list) or not data["dates"]:
        raise ValueError("Missing dates match list")
    path.write_bytes(response.content)
    receipt = {"league": league, "canonical_league": acquire.LEAGUES[league], "season_start_year": year,
        "requested_url": url, "actual_url": response.url, "http_status": response.status_code,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "sha256": acquire.sha(path),
        "bytes": len(response.content), "content_type": response.headers.get("Content-Type"),
        "last_modified": response.headers.get("Last-Modified"), "etag": response.headers.get("ETag"),
        "path": str(path.relative_to(acquire.ROOT)), "receipt_path": str(receipt_path.relative_to(acquire.ROOT)),
        "matches_listed": len(data["dates"]), "acquisition_source_sha256": acquire.sha(__file__)}
    acquire.write(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    prior = json.loads((acquire.OUT / "acquisition_lock.json").read_text())
    assert acquire.canonical_inputs() == prior["canonical_inputs"]
    assert acquire.sha(acquire.__file__) == prior["source_sha256"]
    schema_manifest = json.loads((acquire.OUT / "schema_manifest.json").read_text())
    for item in schema_manifest["files"]:
        assert acquire.sha(acquire.ROOT / item["path"]) == item["sha256"]
    loader = (acquire.DATA / "schema_league.min.js").read_text()
    assert 'url:"getLeagueData/"+league+"/"+season' in loader and "datesData=data.dates" in loader
    lock = acquire.OUT / "matchlist_acquisition_lock.json"
    expected = {"source_sha256": acquire.sha(__file__), "schema_manifest_sha256": acquire.sha(acquire.OUT / "schema_manifest.json"),
        "original_acquisition_lock_sha256": acquire.sha(acquire.OUT / "acquisition_lock.json"),
        "canonical_inputs": prior["canonical_inputs"], "pages": prior["pages"],
        "endpoint_contract": "Unauthenticated GET from current public first-party JavaScript; use selected league display name and season-start label",
        "fitting_or_performance_scoring": False}
    if lock.exists():
        saved = json.loads(lock.read_text()); assert {k: saved[k] for k in expected} == expected
    else:
        acquire.write(lock, {**expected, "frozen_at_utc": datetime.now(timezone.utc).isoformat()})
    records, failures = [], []
    items = [(league, year) for league in acquire.LEAGUES for year in acquire.YEARS] if args.all else [("EPL", 2022)]
    for league, year in items:
        try:
            item = fetch(league, year); records.append(item)
            print(json.dumps(item), flush=True)
        except Exception as exc:
            failure = {"league": league, "season_start_year": year, "error": repr(exc)}
            failures.append(failure); print(json.dumps(failure), flush=True)
    assert acquire.canonical_inputs() == expected["canonical_inputs"] and acquire.sha(__file__) == expected["source_sha256"]
    output = acquire.OUT / ("matchlist_manifest.json" if args.all else "matchlist_probe_manifest.json")
    if output.exists():
        raise FileExistsError("preserve immutable acquisition attempt")
    acquire.write(output, {"lock_sha256": acquire.sha(lock), "pages": records, "failures": failures,
                          "finished_at_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
