"""Capture the public page and its observed first-party loader after schema drift."""
from datetime import datetime, timezone
import json
import requests

from research.experiments.boundary_20260908.football_xg_data import acquire


def main():
    entries = []
    for name, url in [
        ("schema_EPL_2022.html", "https://understat.com/league/EPL/2022"),
        ("schema_league.min.js", "https://understat.com/js/league.min.js?t=1765269520"),
    ]:
        path = acquire.DATA / name
        if path.exists():
            raise FileExistsError(path)
        response = requests.get(url, timeout=45, headers={"User-Agent": "SportPredictionResearch/1.0"})
        response.raise_for_status()
        assert response.url.startswith("https://understat.com/") and len(response.content) < 5_000_000
        path.write_bytes(response.content)
        entries.append({"requested_url": url, "actual_url": response.url,
            "path": str(path.relative_to(acquire.ROOT)), "sha256": acquire.sha(path),
            "http_status": response.status_code, "bytes": len(response.content),
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "content_type": response.headers.get("Content-Type")})
    acquire.write(acquire.OUT / "schema_manifest.json", {"source_sha256": acquire.sha(__file__), "files": entries})
    print(json.dumps(entries, indent=2))


if __name__ == "__main__":
    main()
