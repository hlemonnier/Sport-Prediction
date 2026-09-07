"""Acquire only the predeclared primary Spanish/Italian transfer inputs."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[4]
LANE = Path(__file__).resolve().parent
DATA = ROOT / "data/football/performance_20260907"
OUT = ROOT / "artifacts/research/performance_20260907/football"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec_path = LANE / "transfer_spec.json"
    spec = json.loads(spec_path.read_text())
    lock_path = OUT / "transfer_design_lock.json"
    if lock_path.exists():
        assert json.loads(lock_path.read_text())["spec_sha256"] == sha(spec_path)
    else:
        lock_path.write_text(json.dumps({"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "spec_sha256": sha(spec_path), "spec": spec, "transfer_metrics_inspected": False}, indent=2)+"\n")
    manifest_path = DATA / "transfer_source_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": []}
    entries = {r["name"]: r for r in manifest["files"]}
    requests = [(f"transfer_{league}_{year}_{year+1}.csv",
                 f"https://football-data.co.uk/mmz4281/{str(year)[2:]}{str(year+1)[2:]}/{league}.csv")
                for league in spec["leagues"] for year in spec["input_season_start_years"]]
    requests += [("transfer_spain_source.html", "https://football-data.co.uk/spainm.php"),
                 ("transfer_italy_source.html", "https://football-data.co.uk/italym.php")]
    for name, url in requests:
        path = DATA / name
        if path.exists():
            if name not in entries or sha(path) != entries[name]["sha256"]:
                raise ValueError("Unmanifested or changed transfer input")
        else:
            request = Request(url, headers={"User-Agent": "FootballPredictionResearch/1.0"})
            with urlopen(request, timeout=45) as response:
                payload = response.read(4_000_001)
                if len(payload) > 4_000_000 or (name.endswith(".csv") and b"FTHG" not in payload[:3000]):
                    raise ValueError("Unexpected source response")
                path.write_bytes(payload)
                entries[name] = {"name": name, "url": url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "sha256": sha(path), "bytes": len(payload), "last_modified": response.headers.get("Last-Modified"),
                    "etag": response.headers.get("ETag")}
        # Retain all previously manifested entries if an interrupted cached rerun resumes.
        manifest_path.write_text(json.dumps({"source": "Football-Data.co.uk", "transfer_spec_sha256": sha(spec_path),
            "usage": "Local league match prediction research; no open redistribution license assumed",
            "shared_schema_sha256": sha(DATA / "notes.txt"), "files": list(entries.values())}, indent=2)+"\n")
        print(json.dumps({"name": name, "bytes": entries[name]["bytes"], "sha256": entries[name]["sha256"]}), flush=True)


if __name__ == "__main__":
    main()
