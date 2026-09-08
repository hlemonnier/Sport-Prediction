"""Raw code co-occurrence only; no lap tables, forecast clocks or targets."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from fastf1 import _api

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "artifacts/research/boundary_20260908/telemetry_pilot"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    destination = OUT / "code_diagnostics.json"
    if destination.exists():
        raise FileExistsError("Code diagnostics are immutable")
    manifest_path = OUT / "acquisition.json"
    assert sha(manifest_path) == "9d39fc16e23db3a3c9066b94e2add265d6a68e639d73417b3d3a813c348ca0e4"
    manifest = json.loads(manifest_path.read_text())
    result = {"status": "passed", "acquisition_sha256": sha(manifest_path),
        "source_sha256": sha(__file__), "streams": [], "models_fitted": 0,
        "predictive_scores_computed": 0, "lap_tables_opened": 0, "network_calls": 0}
    for stream in manifest["streams"]:
        path = ROOT / stream["decoded_body_path"]
        assert sha(path) == stream["decoded_body_sha256"]
        counts, drivers = Counter(), {}
        for record in path.read_text(encoding="utf-8-sig").splitlines():
            clock = record[:12]
            for entry in _api.parse(record[12:].strip(), zipped=True)["Entries"]:
                for driver, car in entry["Cars"].items():
                    c = car["Channels"]
                    counts["rows"] += 1
                    if c["4"] == 104:
                        counts["throttle104"] += 1
                        counts["throttle104_and_brake104"] += c["5"] == 104
                        counts["throttle104_and_speed_zero"] += c["2"] == 0
                        counts["throttle104_and_rpm_zero"] += c["0"] == 0
                        counts["throttle104_and_speed_above_30"] += c["2"] > 30
                        counts["throttle104_and_rpm_above_3000"] += c["0"] > 3000
                        d = drivers.setdefault(driver, {"throttle104": 0,
                            "first_recorded_clock": clock, "last_recorded_clock": clock})
                        d["throttle104"] += 1
                        d["last_recorded_clock"] = clock
                    if c["5"] == 104:
                        counts["brake104"] += 1
                        counts["brake104_and_speed_above_30"] += c["2"] > 30
                    if not 0 <= c["3"] <= 8:
                        counts["gear_outside_0_to_8"] += 1
                        counts["invalid_gear_and_throttle104"] += c["4"] == 104
                        counts["invalid_gear_and_speed_above_30"] += c["2"] > 30
        row = {"event_key": stream["event_key"], "raw_sha256": stream["decoded_body_sha256"],
            "cooccurrence_counts": dict(counts), "throttle104_by_driver": drivers}
        result["streams"].append(row)
        print(json.dumps({"event_key": row["event_key"], "counts": dict(counts)}), flush=True)
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["interpretation"] = "Co-occurrence is observed code behavior, not proof of sentinel semantics or physical pressure/throttle percent. No raw value was changed or discarded."
    result["supplemental_preflight_fix"] = "The first supplemental reader attempt failed at the first BOM-prefixed record before producing counts. Its text reader was corrected to utf-8-sig. The frozen parser and completed independent verifier already handled the BOM correctly; raw inputs and their hashes are unchanged."
    with destination.open("x") as file:
        json.dump(result,file,indent=2,sort_keys=True,allow_nan=False)
        file.write("\n")
    print(json.dumps({"path": str(destination.relative_to(ROOT)), "sha256": sha(destination)}))


if __name__ == "__main__":
    main()
