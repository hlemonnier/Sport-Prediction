"""Modest, cached retrieval of primary public EPL CSVs and source documentation."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[4]
LANE = Path(__file__).resolve().parent
DATA = ROOT / "data/football/performance_20260907"
ARTIFACTS = ROOT / "artifacts/research/performance_20260907/football"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    spec_path = LANE / "spec.json"
    spec = json.loads(spec_path.read_text())
    args.data_dir.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    lock = ARTIFACTS / "design_lock.json"
    if lock.exists():
        assert json.loads(lock.read_text())["spec_sha256"] == sha(spec_path), "Frozen spec changed"
    else:
        lock.write_text(json.dumps({"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "spec_sha256": sha(spec_path), "spec": spec,
            "metrics_inspected": False}, indent=2) + "\n")
    manifest_path = args.data_dir / "source_manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": []}
    old_by_name = {item["name"]: item for item in old["files"]}
    requests = [(f"E0_{year}_{year+1}.csv",
                 f"https://football-data.co.uk/mmz4281/{str(year)[2:]}{str(year+1)[2:]}/E0.csv")
                for year in spec["input_season_start_years"]]
    requests += [("notes.txt", "https://football-data.co.uk/notes.txt"),
                 ("data_source.html", "https://football-data.co.uk/data.php"),
                 ("england_source.html", "https://football-data.co.uk/englandm.php"),
                 ("disclaimer.html", "https://football-data.co.uk/disclaimer.php")]
    files = []
    for name, url in requests:
        dest = args.data_dir / name
        if dest.exists():
            if name not in old_by_name or sha(dest) != old_by_name[name]["sha256"]:
                raise ValueError(f"Unmanifested or changed cached input: {dest}")
            item = old_by_name[name]
        else:
            request = Request(url, headers={"User-Agent": "FootballPredictionResearch/1.0 (public historical EPL match prediction)"})
            with urlopen(request, timeout=45) as response:
                payload = response.read(4_000_001)
                if len(payload) > 4_000_000:
                    raise ValueError("Unexpectedly large source response")
                if name.endswith(".csv") and b"FTHG" not in payload[:3000]:
                    raise ValueError("Expected CSV header absent")
                dest.write_bytes(payload)
                item = {"name": name, "url": url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                        "sha256": sha(dest), "bytes": len(payload),
                        "last_modified": response.headers.get("Last-Modified"), "etag": response.headers.get("ETag")}
        files.append(item)
        manifest_path.write_text(json.dumps({"source": "Football-Data.co.uk", "spec_sha256": sha(spec_path),
            "usage": "Local league match prediction research; free availability is not an open redistribution license",
            "files": files}, indent=2) + "\n")
        print(json.dumps({"file": name, "bytes": item["bytes"], "sha256": item["sha256"]}), flush=True)


if __name__ == "__main__":
    main()
