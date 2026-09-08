"""2022-only raw token corpus, frozen hierarchical CPC schedule and replay.

No lap table, outcome, stint boundary or source UTC enters this module.
Suggested commit: research(f1-live): freeze causal raw-token CPC sampling
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys

import numpy as np
import torch

from . import encoder, tokens

ROOT = tokens.ROOT
EVENT_KEYS = tuple(range(202201, 202223))
MAX_RSS_BYTES = 3*1024**3
LRU_DRIVERS = 8
DISTANCE_NS = 180*tokens.NS_PER_SECOND
FIELDS = {"values": ("<f4", (34,)), "availability_ns": ("<i8", ()),
          "packet_sequence": ("<i8", ()), "previous_availability_ns": ("<i8", ())}


class ResourceLimitError(RuntimeError): pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False);stream.write("\n")


def _read(path):
    def invalid(value): raise ValueError("Nonstandard JSON constant: "+value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def _now(): return datetime.now(timezone.utc).isoformat()


def _record(path, **kwargs):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size, **kwargs}


def _check(binding):
    path = Path(binding["path"])
    if path.stat().st_size != binding["bytes"] or sha(path) != binding["sha256"]:
        raise ValueError("Corpus/schedule input hash or byte count changed: "+str(path))
    return path


def _rss():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = int(raw if sys.platform == "darwin" else raw*1024)
    if peak > MAX_RSS_BYTES: raise ResourceLimitError(f"Peak resident memory {peak} exceeds {MAX_RSS_BYTES} bytes")
    return peak


def source_files():
    result = tokens.dependency_bindings()
    for path in (Path(__file__), Path(tokens.__file__), Path(encoder.__file__)):
        result[str(path.resolve().relative_to(ROOT))] = sha(path)
    return result


def _progress(**value): print(json.dumps(value, allow_nan=False), flush=True)


def _open(binding):
    shape = tuple(binding["shape"]);dtype = np.dtype(binding["dtype"])
    if int(np.prod(shape))*dtype.itemsize != binding["bytes"]: raise ValueError("Array byte shape mismatch")
    if not int(np.prod(shape)):
        value = np.empty(shape, dtype=dtype);value.setflags(write=False);return value
    return np.memmap(binding["path"], dtype=dtype, mode="r", shape=shape)


def _close_maps(arrays):
    for value in arrays.values():
        mmap = getattr(value, "_mmap", None)
        if mmap is not None: mmap.close()


def _binary(path, values, dtype):
    array = np.asarray(values, dtype=dtype)
    with Path(path).open("xb") as stream: stream.write(array.tobytes(order="C"))
    return _record(path, dtype=np.dtype(dtype).str, shape=list(array.shape))


def negative_ids(availability, endpoint):
    """Sorted same-driver IDs >=180s away, excluding ALL three positive IDs."""
    endpoint = tokens._integer(endpoint, "endpoint", minimum=0)
    if endpoint+max(encoder.HORIZONS) >= len(availability): raise ValueError("Endpoint lacks all three future packet targets")
    now = int(availability[endpoint])
    before = int(np.searchsorted(availability, now-DISTANCE_NS, side="right"))
    after = int(np.searchsorted(availability, now+DISTANCE_NS, side="left"))
    candidates = np.concatenate((np.arange(before, dtype=np.int64), np.arange(after, len(availability), dtype=np.int64)))
    return candidates[~np.isin(candidates, endpoint+np.asarray(encoder.HORIZONS))]


def eligible_endpoints(availability):
    """Vectorized eligibility on physical prefixes; no token measurements read."""
    time = np.asarray(availability)
    if time.dtype.kind not in "iu" or time.ndim != 1 or (time < 0).any() or np.any(time[1:] < time[:-1]):
        raise ValueError("Availability must be ordered nonnegative integer nanoseconds")
    if len(time) and int(time[-1]) > np.iinfo(np.int64).max-DISTANCE_NS-1: raise ValueError("Availability arithmetic would overflow")
    n = len(time);ids = np.arange(max(0, n-max(encoder.HORIZONS)), dtype=np.int64)
    instant = time[ids].astype(np.int64, copy=False)
    # Query cutoff is endpoint clock+1ns. Same-clock packets after endpoint are
    # not part of its physical prefix; the right edge is explicitly ids+1.
    start = np.searchsorted(time, instant+1-30*tokens.NS_PER_SECOND, side="left")
    support = (ids-start+1 >= 6) & (instant-time[start] >= 24*tokens.NS_PER_SECOND)
    before = np.searchsorted(time, instant-DISTANCE_NS, side="right")
    after = np.searchsorted(time, instant+DISTANCE_NS, side="left")
    count = before+(n-after)
    for horizon in encoder.HORIZONS:
        count -= np.abs(time[ids+horizon]-instant) >= DISTANCE_NS
    eligible = ids[support & (count >= encoder.NEGATIVES)]
    return eligible, {"packets": n, "endpoints_with_all_positive_packets": len(ids),
        "endpoints_with_support_and_positives": int(support.sum()),
        "endpoints_with_negatives_and_positives": int((count >= encoder.NEGATIVES).sum()),
        "eligible_endpoints": len(eligible), "ineligible_endpoints": n-len(eligible)}


def _input_events(events):
    events = list(events)
    # Reject any selection/later event before reading hashes or raw contents.
    keys = [tokens._integer(row["event_key"], "event_key") for row in events]
    if keys != list(EVENT_KEYS): raise ValueError("CPC corpus requires exactly the22 ascending2022 events")
    result = []
    for row in events:
        if row["status"] not in ("downloaded_unparsed", "reused_verified_pilot"):
            raise ValueError("Every training event must have an available raw stream")
        path = (ROOT/row["decoded_body_path"]).resolve()
        item = {"event_key": row["event_key"], "path": str(path),
            "sha256": row["decoded_body_sha256"], "bytes": row["decoded_body_bytes"], "status": row["status"]}
        _check(item);result.append(item)
    return result


def build_corpus(events, out, *, design_lock_path, design_sha256):
    """Stream exactly22 training races into per-driver disk arrays, exclusively."""
    design = Path(design_lock_path).resolve()
    if sha(design) != design_sha256: raise ValueError("CPC corpus design lock differs")
    source = source_files();inputs = _input_events(events)
    out = Path(out).resolve();out.mkdir(parents=True, exist_ok=False)
    started = _now();manifest_events = [];drivers_total = eligible_drivers = 0
    try:
        for item in inputs:
            _rss();event_dir = out/str(item["event_key"]);event_dir.mkdir()
            handles = {};counts = {};driver_dirs = {};statistics = {}
            try:
                for row in tokens.iter_packet_tokens(item["path"], stats=statistics):
                    driver = row.driver_id
                    if driver not in handles:
                        # Driver text never becomes a path component.
                        directory = event_dir/hashlib.sha256(driver.encode()).hexdigest()[:16];directory.mkdir()
                        driver_dirs[driver] = directory;counts[driver] = 0
                        handles[driver] = {name: (directory/(name+".bin")).open("xb") for name in FIELDS}
                    values = {"values": row.values, "availability_ns": row.availability_ns,
                        "packet_sequence": row.packet_sequence,
                        "previous_availability_ns": -1 if row.previous_availability_ns is None else row.previous_availability_ns}
                    for name, (dtype, _) in FIELDS.items(): handles[driver][name].write(np.asarray(values[name], dtype=dtype).tobytes())
                    counts[driver] += 1
            finally:
                for streams in handles.values():
                    for stream in streams.values(): stream.close()
            if statistics.get("complete") is not True or statistics.get("stream_input_valid") is not True:
                raise ValueError("Training raw stream did not pass exhaustive integrity validation")
            _check(item);drivers = []
            for driver in sorted(counts):
                n = counts[driver];directory = driver_dirs[driver]
                arrays = {name: _record(directory/(name+".bin"), dtype=dtype, shape=[n, *tail]) for name, (dtype, tail) in FIELDS.items()}
                available = _open(arrays["availability_ns"])
                try: eligible, diagnostic = eligible_endpoints(available)
                finally: _close_maps({"availability": available})
                arrays["eligible_endpoints"] = _binary(directory/"eligible_endpoints.bin", eligible, "<i8")
                drivers.append({"driver_id": driver, "arrays": arrays, **diagnostic})
            count_eligible = sum(row["eligible_endpoints"] > 0 for row in drivers)
            record = {"event_key": item["event_key"], "source": item, "drivers": drivers,
                "driver_count": len(drivers), "eligible_driver_count": count_eligible,
                "ineligible_driver_count": len(drivers)-count_eligible, "raw_validation": statistics}
            _write(event_dir/"event.json", record)
            manifest_events.append(record);drivers_total += len(drivers);eligible_drivers += count_eligible
            peak = _rss();_progress(stage="corpus_event", event_key=item["event_key"], drivers=len(drivers), eligible_drivers=count_eligible, peak_rss_bytes=peak)
            if not count_eligible: raise ValueError(f"Training event {item['event_key']} has no eligible endpoint")
        if source_files() != source or sha(design) != design_sha256: raise ValueError("CPC source/design changed during corpus build")
        target = out/"corpus.json"
        _write(target, {"schema": "causal_2022_token_corpus_v1", "status": "closed", "started_at_utc": started,
            "completed_at_utc": _now(), "design_lock": _record(design), "source_files": source,
            "token_names": list(tokens.TOKEN_NAMES), "events": manifest_events, "driver_count": drivers_total,
            "eligible_driver_count": eligible_drivers, "ineligible_driver_count": drivers_total-eligible_drivers,
            "years": [2022], "lap_labels_read": False, "model_fits": 0, "peak_rss_bytes": _rss()})
        return target
    except Exception as exc:
        _write(out/"corpus_failure.json", {"failed_at_utc": _now(), "failure": type(exc).__name__+": "+str(exc),
            "completed_events": [r["event_key"] for r in manifest_events], "existing_outputs_preserved": True})
        raise


def verify_corpus(path):
    corpus = _read(path)
    if corpus["schema"] != "causal_2022_token_corpus_v1" or corpus["status"] != "closed" or corpus["years"] != [2022]: raise ValueError("Wrong corpus contract")
    if corpus["lap_labels_read"] is not False or corpus["model_fits"] != 0: raise ValueError("Corpus must precede models and contain no lap labels")
    if corpus["source_files"] != source_files() or corpus["token_names"] != list(tokens.TOKEN_NAMES): raise ValueError("Corpus source/token schema changed")
    _check(corpus["design_lock"])
    if [r["event_key"] for r in corpus["events"]] != list(EVENT_KEYS): raise ValueError("Corpus event population changed")
    for event in corpus["events"]:
        _check(event["source"])
        names = [r["driver_id"] for r in event["drivers"]]
        if names != sorted(set(names)) or not event["eligible_driver_count"]: raise ValueError("Corpus driver order or support differs")
        for driver in event["drivers"]:
            for array in driver["arrays"].values(): _check(array)
    return corpus


class _DriverCache:
    def __init__(self, corpus):
        self.streams = [(event["event_key"], driver) for event in corpus["events"] for driver in event["drivers"]]
        self.cache = OrderedDict()

    def get(self, index):
        index = tokens._integer(index, "stream_index", minimum=0)
        if index >= len(self.streams): raise ValueError("Invalid scheduled driver stream")
        if index not in self.cache:
            if len(self.cache) >= LRU_DRIVERS: _close_maps(self.cache.popitem(last=False)[1])
            self.cache[index] = {name: _open(value) for name, value in self.streams[index][1]["arrays"].items()}
        self.cache.move_to_end(index)
        return self.cache[index]

    def close(self):
        for arrays in self.cache.values(): _close_maps(arrays)
        self.cache.clear()


def build_schedule(corpus, out):
    corpus_path = Path(corpus).resolve();manifest = verify_corpus(corpus_path);cache = _DriverCache(manifest)
    out = Path(out).resolve();out.mkdir(parents=True, exist_ok=False)
    shape = (encoder.TRAIN_STEPS, encoder.BATCH_SIZE)
    arrays = {"stream_index": np.empty(shape, dtype="<i4"), "endpoint": np.empty(shape, dtype="<i8"),
        "positives": np.empty((*shape, len(encoder.HORIZONS)), dtype="<i8"),
        "negatives": np.empty((*shape, encoder.NEGATIVES), dtype="<i8")}
    event_streams = []
    for event in manifest["events"]:
        eligible = [i for i, (key, driver) in enumerate(cache.streams) if key == event["event_key"] and driver["eligible_endpoints"] > 0]
        if not eligible: raise ValueError("Every fixed event requires an eligible driver")
        event_streams.append(eligible)
    rng = np.random.Generator(np.random.PCG64(tokens.SEED))
    try:
        for step in range(encoder.TRAIN_STEPS):
            for row in range(encoder.BATCH_SIZE):
                drivers = event_streams[int(rng.integers(len(event_streams)))];stream = drivers[int(rng.integers(len(drivers)))]
                mapped = cache.get(stream);eligible = mapped["eligible_endpoints"]
                endpoint = int(eligible[int(rng.integers(len(eligible)))])
                candidates = negative_ids(mapped["availability_ns"], endpoint)
                if len(candidates) < encoder.NEGATIVES: raise ValueError("Stored endpoint negative support differs")
                arrays["stream_index"][step, row] = stream;arrays["endpoint"][step, row] = endpoint
                arrays["positives"][step, row] = endpoint+np.asarray(encoder.HORIZONS)
                arrays["negatives"][step, row] = rng.choice(candidates, encoder.NEGATIVES, replace=False)
            _rss()
        files = {name: _binary(out/(name+".bin"), value, value.dtype) for name, value in arrays.items()}
        frequencies = np.bincount(arrays["stream_index"].ravel(), minlength=len(cache.streams))
        draws = [{"event_key": event, "driver_id": driver["driver_id"], "draws": int(frequencies[i])}
                 for i, (event, driver) in enumerate(cache.streams)]
        target = out/"schedule.json"
        _write(target, {"schema": "fixed_cpc_hierarchical_schedule_v1", "status": "closed", "closed_at_utc": _now(),
            "corpus": _record(corpus_path), "source_files": source_files(), "seed": tokens.SEED, "rng": "PCG64",
            "steps": encoder.TRAIN_STEPS, "batch_size": encoder.BATCH_SIZE, "horizons": list(encoder.HORIZONS),
            "negatives": encoder.NEGATIVES, "arrays": files, "order": "step_then_batch_row_event_then_driver_then_endpoint_then_negatives",
            "driver_draw_counts": draws,
            "event_draw_counts": [{"event_key": event, "draws": sum(row["draws"] for row in draws if row["event_key"] == event)} for event in EVENT_KEYS],
            "draw_frequency_policy": "descriptive_only_no_balance_rejection_or_resampling",
            "model_fits_before_schedule_close": 0, "peak_rss_bytes": _rss()})
        return target
    except Exception as exc:
        _write(out/"schedule_failure.json", {"failed_at_utc": _now(), "failure": type(exc).__name__+": "+str(exc), "existing_outputs_preserved": True})
        raise
    finally: cache.close()


class BatchProvider:
    """Replay identical saved identities to both encoders, with at most8 maps."""
    def __init__(self, corpus, schedule):
        self.corpus_path = Path(corpus).resolve();manifest = verify_corpus(self.corpus_path)
        self.schedule = _read(schedule)
        wanted = {"schema": "fixed_cpc_hierarchical_schedule_v1", "status": "closed", "seed": tokens.SEED,
            "rng": "PCG64", "steps": encoder.TRAIN_STEPS, "batch_size": encoder.BATCH_SIZE,
            "horizons": list(encoder.HORIZONS), "negatives": encoder.NEGATIVES, "model_fits_before_schedule_close": 0}
        if any(self.schedule.get(k) != v for k, v in wanted.items()) or self.schedule["source_files"] != source_files(): raise ValueError("Wrong frozen CPC schedule")
        if _check(self.schedule["corpus"]) != self.corpus_path: raise ValueError("Schedule is bound to a different corpus")
        self.cache = _DriverCache(manifest);self.closed = False
        self.arrays = {}
        try:
            for name, binding in self.schedule["arrays"].items(): _check(binding);self.arrays[name] = _open(binding)
            if set(self.arrays) != {"stream_index", "endpoint", "positives", "negatives"}: raise ValueError("Incomplete schedule arrays")
            shape = (encoder.TRAIN_STEPS, encoder.BATCH_SIZE)
            for name, tail in (("stream_index", ()), ("endpoint", ()), ("positives", (3,)), ("negatives", (31,))):
                if self.arrays[name].shape != (*shape, *tail) or self.arrays[name].dtype.kind not in "iu": raise ValueError("Schedule array shape/dtype differs")
            _rss()
        except Exception:
            self.close();raise

    def __call__(self, step):
        if self.closed: raise ValueError("CPC batch provider is closed")
        step = tokens._integer(step, "step", minimum=0)
        if step >= encoder.TRAIN_STEPS: raise ValueError("CPC step outside frozen schedule")
        _rss();contexts = np.zeros((encoder.BATCH_SIZE, 128, 34), dtype=np.float32);contexts[:, :, 33] = 1.
        positives = np.empty((encoder.BATCH_SIZE, 3, 34), dtype=np.float32)
        negatives = np.empty((encoder.BATCH_SIZE, 31, 34), dtype=np.float32);keys = []
        for row in range(encoder.BATCH_SIZE):
            stream = int(self.arrays["stream_index"][step, row]);endpoint = int(self.arrays["endpoint"][step, row]);mapped = self.cache.get(stream)
            event, driver = self.cache.streams[stream];times = mapped["availability_ns"]
            pos, neg = self.arrays["positives"][step, row], self.arrays["negatives"][step, row]
            eligible = mapped["eligible_endpoints"];where = int(np.searchsorted(eligible, endpoint))
            if where >= len(eligible) or eligible[where] != endpoint: raise ValueError("Scheduled endpoint is ineligible")
            if not np.array_equal(pos, endpoint+np.asarray(encoder.HORIZONS)): raise ValueError("Scheduled positive packet IDs differ")
            if len(set(map(int, neg))) != 31 or np.any(neg < 0) or np.any(neg >= len(times)) or np.isin(neg, pos).any(): raise ValueError("Invalid or positive-overlapping negative identities")
            instant = int(times[endpoint])
            if np.any(np.abs(times[neg]-instant) < DISTANCE_NS): raise ValueError("Negative token is within180 seconds")
            left = int(np.searchsorted(times, instant+1-DISTANCE_NS, side="left"));left = max(left, endpoint-127)
            # The explicit physical index bound excludes future equal-clock rows.
            count = endpoint-left+1;contexts[row, -count:] = mapped["values"][left:endpoint+1]
            positives[row] = mapped["values"][pos];negatives[row] = mapped["values"][neg]
            keys.append(tokens.permutation_key(event, driver["driver_id"], int(mapped["packet_sequence"][endpoint])))
        batch = encoder.CPCBatch(torch.from_numpy(contexts), torch.from_numpy(positives), torch.from_numpy(negatives), tuple(keys))
        encoder.validate_batch(batch, training=True);_rss();return batch

    def close(self):
        if not getattr(self, "closed", True):
            self.cache.close();_close_maps(self.arrays);self.closed = True

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
