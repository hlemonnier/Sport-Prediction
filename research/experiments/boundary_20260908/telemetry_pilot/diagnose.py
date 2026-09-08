"""Close quality diagnostics for the two acquired streams without model targets."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from . import packets

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/research/boundary_20260908/telemetry_pilot"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    destination = OUT / "diagnostics.json"
    if destination.exists():
        raise FileExistsError("Closed pilot diagnostics are immutable")
    manifest_path = OUT / "acquisition.json"
    manifest = json.loads(manifest_path.read_text())
    if [x["event_key"] for x in manifest["streams"]] != [202201, 202301]:
        raise ValueError("Unexpected pilot population")
    for path, expected in manifest["source_files"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Acquisition/parser source drift: " + path)
    inputs = {}
    for row in manifest["streams"]:
        if row["status"] != "downloaded_unparsed":
            raise ValueError("Both pilot streams must be downloaded before diagnosis")
        path = ROOT / row["decoded_body_path"]
        if sha(path) != row["decoded_body_sha256"] or path.stat().st_size != row["decoded_body_bytes"]:
            raise ValueError("Raw body drift")
        inputs[row["decoded_body_path"]] = row["decoded_body_sha256"]
    lock = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "inputs": inputs,
        "acquisition_sha256": sha(manifest_path), "source_files": {
            str(Path(__file__).relative_to(ROOT)): sha(__file__),
            str(Path(packets.__file__).relative_to(ROOT)): sha(packets.__file__)},
        "models_fitted": 0, "predictive_scores_computed": 0}
    with (OUT / "diagnostics_lock.json").open("x") as stream:
        json.dump(lock, stream, indent=2, sort_keys=True)
        stream.write("\n")
    result = {**lock, "streams": [], "status": "passed"}
    for row in manifest["streams"]:
        try:
            diagnostic = packets.summarize(ROOT / row["decoded_body_path"])
            result["streams"].append({"event_key": row["event_key"], **diagnostic})
            print(json.dumps({"event_key": row["event_key"], "counts": diagnostic["counts"],
                              "packet_validation": diagnostic["packet_validation"]}), flush=True)
        except packets.PacketError as exc:
            result["status"] = "failed"
            result["streams"].append({"event_key": row["event_key"], "error": str(exc)})
            print(json.dumps(result["streams"][-1]), flush=True)
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "path": str(destination.relative_to(ROOT)),
                      "sha256": sha(destination)}))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
