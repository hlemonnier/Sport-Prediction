"""Capture the exact public jQuery AJAX request, with its standard routing header."""
from datetime import datetime, timezone
import json
from urllib.parse import quote

import requests

from research.experiments.boundary_20260908.football_xg_data import acquire


def main():
    original = json.loads((acquire.OUT / "acquisition_lock.json").read_text())
    assert acquire.canonical_inputs() == original["canonical_inputs"]
    assert acquire.sha(acquire.__file__) == original["source_sha256"]
    schema = json.loads((acquire.OUT / "schema_manifest.json").read_text())
    for item in schema["files"]:
        assert acquire.sha(acquire.ROOT / item["path"]) == item["sha256"]
    loader = (acquire.DATA / "schema_league.min.js").read_text()
    assert 'url:"getLeagueData/"+league+"/"+season' in loader and "datesData=data.dates" in loader
    lock = acquire.OUT / "public_ajax_lock.json"
    if lock.exists():
        raise FileExistsError("immutable public AJAX attempt already exists")
    source_sha = acquire.sha(__file__)
    acquire.write(lock, {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_sha, "canonical_inputs": original["canonical_inputs"],
        "original_acquisition_lock_sha256": acquire.sha(acquire.OUT / "acquisition_lock.json"),
        "schema_manifest_sha256": acquire.sha(acquire.OUT / "schema_manifest.json"),
        "pages": original["pages"],
        "request_contract": "Public first-party GET as used by the league page's jQuery AJAX call; standard X-Requested-With:XMLHttpRequest routing header and public page Referer, no credentials or cookies",
        "purpose": "Match-list acquisition and joins only; no fitted model or performance scores"})
    records, failures = [], []
    for league in acquire.LEAGUES:
        for year in acquire.YEARS:
            url = f"https://understat.com/getLeagueData/{quote(league.replace('_', ' '), safe='')}/{year}"
            page = f"https://understat.com/league/{league}/{year}"
            path = acquire.DATA / f"{league}_{year}_public_ajax.json"
            receipt_path = path.with_suffix(".receipt.json")
            if path.exists() or receipt_path.exists():
                raise FileExistsError(path)
            try:
                response = requests.get(url, timeout=45, headers={"User-Agent": "SportPredictionResearch/1.0",
                    "X-Requested-With": "XMLHttpRequest", "Referer": page})
                response.raise_for_status()
                if len(response.content) > 5_000_000 or not response.url.startswith("https://understat.com/"):
                    raise ValueError("Unexpected public response")
                data = response.json()
                if not isinstance(data, dict) or not isinstance(data.get("dates"), list) or not data["dates"]:
                    raise ValueError("Missing public dates match list")
                path.write_bytes(response.content)
                record = {"league": league, "canonical_league": acquire.LEAGUES[league], "season_start_year": year,
                    "requested_url": url, "actual_url": response.url, "public_page_url": page,
                    "http_status": response.status_code, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "sha256": acquire.sha(path), "bytes": len(response.content),
                    "content_type": response.headers.get("Content-Type"), "last_modified": response.headers.get("Last-Modified"),
                    "etag": response.headers.get("ETag"), "path": str(path.relative_to(acquire.ROOT)),
                    "receipt_path": str(receipt_path.relative_to(acquire.ROOT)), "matches_listed": len(data["dates"]),
                    "acquisition_source_sha256": source_sha}
                acquire.write(receipt_path, record)
                record["receipt_sha256"] = acquire.sha(receipt_path)
                records.append(record)
                print(json.dumps({k: record[k] for k in ("league", "season_start_year", "matches_listed", "bytes", "sha256")}), flush=True)
            except Exception as exc:
                failed = {"league": league, "season_start_year": year, "error": repr(exc)}
                failures.append(failed); print(json.dumps(failed), flush=True)
            acquire.write(acquire.OUT / "public_ajax_progress.json", {"records": records, "failures": failures})
    assert acquire.canonical_inputs() == original["canonical_inputs"] and acquire.sha(__file__) == source_sha
    acquire.write(acquire.OUT / "public_ajax_manifest.json", {"lock_sha256": acquire.sha(lock), "pages": records,
        "failures": failures, "finished_at_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
