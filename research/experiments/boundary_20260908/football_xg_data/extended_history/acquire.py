"""Fifteen additional public 2017-2021 league-season lists; preserve the first12."""
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import quote

import requests

from research.experiments.boundary_20260908.football_xg_data import acquire as base

HERE = Path(__file__).resolve().parent
DATA = base.DATA / "extended_history"
OUT = base.OUT / "extended_history"
YEARS = (2017, 2018, 2019, 2020, 2021)


def canonical_inputs():
    folder = base.ROOT / "data/football/performance_20260907"
    paths = [folder / (("" if code == "E0" else "transfer_")+f"{code}_{year}_{year+1}.csv")
             for code in base.LEAGUES.values() for year in YEARS]
    paths += [folder / "source_manifest.json", folder / "transfer_source_manifest.json"]
    return {str(p.relative_to(base.ROOT)): base.sha(p) for p in paths}


def main():
    DATA.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    lock_path = OUT / "acquisition_lock.json"
    if lock_path.exists():
        raise FileExistsError("immutable extended acquisition already exists")
    before = canonical_inputs()
    first = {str(p.relative_to(base.ROOT)): base.sha(p) for p in (
        base.OUT / "public_ajax_manifest.json", base.OUT / "join_quality.json", base.OUT / "verification.json",
        base.DATA / "joined_matches.csv", base.HERE / "timing_review.json")}
    source_sha = base.sha(__file__)
    base.write(lock_path, {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": source_sha,
        "canonical_inputs": before, "first_twelve_immutable": first,
        "scope": "Exactly15additional public league-season lists,2017-2021; no2026season, no model fits or predictive scores",
        "request_contract": "Same unauthenticated first-party jQuery XMLHttpRequest routing header as verified first12"})
    records, failures = [], []
    for league, code in base.LEAGUES.items():
        for year in YEARS:
            url = f"https://understat.com/getLeagueData/{quote(league.replace('_', ' '), safe='')}/{year}"
            path = DATA / f"{league}_{year}_public_ajax.json"
            receipt_path = path.with_suffix(".receipt.json")
            if path.exists() or receipt_path.exists():
                raise FileExistsError(path)
            try:
                response = requests.get(url, timeout=45, headers={"User-Agent": "SportPredictionResearch/1.0",
                    "X-Requested-With": "XMLHttpRequest", "Referer": f"https://understat.com/league/{league}/{year}"})
                response.raise_for_status()
                if len(response.content) > 5_000_000 or not response.url.startswith("https://understat.com/"):
                    raise ValueError("Unexpected public response")
                payload = response.json()
                if not isinstance(payload, dict) or not isinstance(payload.get("dates"), list) or not payload["dates"]:
                    raise ValueError("Missing public match list")
                path.write_bytes(response.content)
                record = {"league": league, "canonical_league": code, "season_start_year": year,
                    "requested_url": url, "actual_url": response.url, "http_status": response.status_code,
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "sha256": base.sha(path),
                    "bytes": len(response.content), "content_type": response.headers.get("Content-Type"),
                    "last_modified": response.headers.get("Last-Modified"), "etag": response.headers.get("ETag"),
                    "path": str(path.relative_to(base.ROOT)), "receipt_path": str(receipt_path.relative_to(base.ROOT)),
                    "matches_listed": len(payload["dates"]), "acquisition_source_sha256": source_sha}
                base.write(receipt_path, record); record["receipt_sha256"] = base.sha(receipt_path)
                records.append(record)
                print(json.dumps({k: record[k] for k in ("league", "season_start_year", "matches_listed", "bytes", "sha256")}), flush=True)
            except Exception as exc:
                failure = {"league": league, "season_start_year": year, "error": repr(exc)}
                failures.append(failure); print(json.dumps(failure), flush=True)
            base.write(OUT / "progress.json", {"records": records, "failures": failures})
    assert before == canonical_inputs() and base.sha(__file__) == source_sha
    assert all(base.sha(base.ROOT / p) == h for p, h in first.items())
    base.write(OUT / "acquisition_manifest.json", {"lock_sha256": base.sha(lock_path), "pages": records,
        "failures": failures, "finished_at_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
