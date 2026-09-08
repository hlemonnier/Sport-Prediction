"""Bounded, immutable two-event CarData.z acquisition; no historical modelling."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import zlib

import requests

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SPEC = HERE / "specification.json"
OUT = ROOT / "artifacts/research/boundary_20260908/telemetry_pilot"
DATA = ROOT / "data/f1/boundary_20260908/telemetry_pilot"
ALLOWED_BASE_URLS = ["https://livetiming-mirror.fastf1.dev", "https://livetiming.formula1.com"]
BODY_LIMIT = 64 * 1024 * 1024


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def decode_body(path, encoding, cap):
    """Enforce a second cap on HTTP content decoding, before packet parsing."""
    if type(cap) is not int or cap <= 0:
        raise ValueError("Decoded body cap must be a positive integer")
    if encoding in ("", "identity"):
        if path.stat().st_size > cap:
            raise ValueError("Decoded HTTP body exceeds cap")
        return path
    if encoding not in ("gzip", "deflate"):
        raise ValueError(f"Unsupported content encoding: {encoding}")
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS if encoding == "gzip" else zlib.MAX_WBITS)
    body = decoder.decompress(path.read_bytes(), cap + 1)
    if len(body) > cap or decoder.unconsumed_tail:
        raise ValueError("Decoded HTTP body exceeds cap")
    if not decoder.eof or decoder.unused_data:
        raise ValueError("Incomplete HTTP encoding or trailing compressed data")
    destination = path.with_suffix(".decoded.jsonStream")
    with destination.open("xb") as stream:
        stream.write(body)
    return destination


def validate_spec():
    spec = json.loads(SPEC.read_text())
    binding = spec["parent_session_manifest"]
    if sha(ROOT / binding["path"]) != binding["sha256"]:
        raise ValueError("Parent manifest drift")
    parents = {x["event_key"]: x for x in json.loads((ROOT / binding["path"]).read_text())["sessions"]}
    if [x["event_key"] for x in spec["sessions"]] != [202201, 202301]:
        raise ValueError("Only the two fixed Bahrain pilot sessions are authorized")
    for session in spec["sessions"]:
        if session != parents[session["event_key"]]:
            raise ValueError("Session mapping drift")
    if (spec["stream"] != "CarData.z" or spec["max_parallel_requests"] != 2
            or spec["max_requests_per_event"] != 2 or spec["follow_redirects"] is not False
            or spec["base_urls"] != ALLOWED_BASE_URLS):
        raise ValueError("Acquisition scope drift")
    expected_limits = {"max_http_body_bytes": BODY_LIMIT, "max_decoded_body_bytes": BODY_LIMIT,
        "max_request_elapsed_seconds": 150, "request_timeout_connect_seconds": 15,
        "request_timeout_read_seconds": 30, "minimum_free_disk_bytes": 512 * 1024 * 1024}
    if any(type(spec.get(key)) is not int or spec[key] != value for key, value in expected_limits.items()):
        raise ValueError("Acquisition limit drift")
    if spec.get("elapsed_deadline_semantics") != "response_acceptance_checked_before_and_after_each_read1_not_hard_transport_cancellation":
        raise ValueError("Acquisition deadline contract drift")
    if shutil.disk_usage(ROOT).free < spec["minimum_free_disk_bytes"]:
        raise ValueError("Insufficient disk headroom")
    return spec


def fetch(session, spec):
    event = session["event_key"]
    # Refuse all existing event outputs before opening any connection. This also
    # covers a prior failed connection with a receipt but no response-body file.
    existing = list(DATA.glob(f"{event}_CarData.z.*"))
    if existing:
        raise FileExistsError(f"Existing event acquisition artifacts: {existing[0]}; inspect before a new attempt")
    result = {"event_key": event, "session_path": session["session_path"],
              "stream": "CarData.z", "status": "unavailable", "attempts": [],
              "historical_client_receipt_certified": False}
    for ordinal, base in enumerate(spec["base_urls"], 1):
        url = base + "/static/" + session["session_path"] + "CarData.z.jsonStream"
        path = DATA / f"{event}_CarData.z.attempt{ordinal}.httpbody"
        if path.exists():
            raise FileExistsError(f"Unreceipted or previous body: {path}; inspect before a new attempt")
        attempt = {"requested_url": url, "request_started_at_utc": now(), "wire_body_complete": False}
        started, response = time.monotonic(), None
        try:
            response = requests.get(url, headers={"Connection": "close", "TE": "identity",
                "User-Agent": "BestHTTP", "Accept-Encoding": "identity"},
                timeout=(spec["request_timeout_connect_seconds"], spec["request_timeout_read_seconds"]),
                allow_redirects=False, stream=True)
            attempt.update(actual_url=response.url, http_status=response.status_code,
                headers_received_at_utc=now(), response_headers={k: v for k, v in response.headers.items()
                if k.lower() in {"date", "etag", "last-modified", "content-type", "content-encoding", "content-length", "location"}})
            count, cap = 0, spec["max_http_body_bytes"]
            with path.open("xb") as stream:
                while True:
                    if time.monotonic() - started > spec["max_request_elapsed_seconds"]:
                        raise TimeoutError("Total request duration exceeded; preserved body is partial")
                    # read1 does not wait to fill an entire 64KiB buffer from
                    # arbitrarily many small socket reads. It is still subject
                    # to the declared socket timeout, not hard cancellation.
                    chunk = response.raw.read1(min(65536, cap + 1 - count), decode_content=False)
                    if time.monotonic() - started > spec["max_request_elapsed_seconds"]:
                        raise TimeoutError("Response acceptance deadline exceeded after read; preserved body is partial")
                    if not chunk:
                        break
                    stream.write(chunk)
                    count += len(chunk)
                    if count > cap:
                        raise ValueError("HTTP body exceeds cap; preserved body is partial")
            expected = response.headers.get("Content-Length")
            if expected is not None and int(expected) != count:
                raise ValueError("HTTP content length mismatch")
            attempt["wire_body_complete"] = True
            if response.status_code == 200:
                decoded = decode_body(path, response.headers.get("Content-Encoding", "").lower(),
                                      spec["max_decoded_body_bytes"])
                if decoded.stat().st_size == 0:
                    raise ValueError("Empty successful response")
                result.update(status="downloaded_unparsed", decoded_body_path=rel(decoded),
                    decoded_body_sha256=sha(decoded), decoded_body_bytes=decoded.stat().st_size)
        except Exception as exc:
            attempt.update(error_type=type(exc).__name__, error=str(exc))
        finally:
            if response is not None:
                response.close()
            attempt.update(body_finished_at_utc=now(), wall_seconds=time.monotonic()-started)
            if path.exists():
                attempt.update(wire_body_path=rel(path), wire_body_sha256=sha(path), wire_body_bytes=path.stat().st_size)
            result["attempts"].append(attempt)
            write(DATA / f"{event}_CarData.z.attempt{ordinal}.receipt.json", attempt)
        if result["status"] == "downloaded_unparsed":
            break
    result["completed_at_utc"] = now()
    write(DATA / f"{event}_CarData.z.source.json", result)
    print(json.dumps({k: result.get(k) for k in ("event_key", "status", "decoded_body_bytes")}), flush=True)
    return result


def main(dry_run=False):
    spec = validate_spec()
    sources = {rel(path): sha(path) for path in [Path(__file__), SPEC, HERE / "packets.py", HERE / "test_packets.py",
        HERE / "test_acquire.py", HERE / "README.md"]}
    summary = {"scope": spec["purpose"], "event_keys": [202201, 202301], "max_http_requests": 4,
               "max_parallel_http_requests": 2, "source_files": sources, "models_fitted": 0,
               "predictive_scores_computed": 0}
    if dry_run:
        print(json.dumps(summary, sort_keys=True))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    write(OUT / "acquisition_lock.json", {**summary, "frozen_at_utc": now(), "specification_sha256": sha(SPEC)})
    with ThreadPoolExecutor(max_workers=2) as executor:
        rows = list(executor.map(lambda s: fetch(s, spec), spec["sessions"]))
    for path, expected in sources.items():
        if sha(ROOT / path) != expected:
            raise ValueError("Source changed during acquisition")
    result = {**summary, "sessions": spec["sessions"], "streams": rows, "completed_at_utc": now(),
              "acquisition_lock_sha256": sha(OUT / "acquisition_lock.json"),
              "status": "all_stream_attempts_recorded", "downloaded": sum(r["status"] == "downloaded_unparsed" for r in rows)}
    write(OUT / "acquisition.json", result)
    print(json.dumps({"path": rel(OUT / "acquisition.json"), "sha256": sha(OUT / "acquisition.json"), "downloaded": result["downloaded"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args().dry_run)
