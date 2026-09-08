"""Synthetic corpus/schedule identity, temporal support and bounded-I/O tests."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from . import corpus as c, encoder, tokens
from .test_tokens import channels, packet, token

NS = 1_000_000_000


def inputs(tmp_path, *, small=False):
    times = [i*1000 for i in range(4)] if small else [(i//3)*5000 for i in range(240)]
    bodies = []
    for i, ms in enumerate(times):
        entries = [{"Cars": {"1": {"Channels": channels(100+i%200)}, "2": {"Channels": channels(200+i%100)}}}]
        if i == 0: entries[0]["Cars"]["unavailable-driver"] = {"Channels": {}}
        bodies.append(packet(ms, entries))
    raw = tmp_path/"synthetic.jsonStream";raw.write_bytes(b"".join(bodies))
    events = [{"event_key": event, "status": "downloaded_unparsed", "decoded_body_path": str(raw),
        "decoded_body_sha256": c.sha(raw), "decoded_body_bytes": raw.stat().st_size} for event in c.EVENT_KEYS]
    design = tmp_path/"design.json";design.write_text('{"synthetic_only":true}\n')
    return events, design


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-corpus");events, design = inputs(root)
    corpus = c.build_corpus(events, root/"corpus", design_lock_path=design, design_sha256=c.sha(design))
    schedule = c.build_schedule(corpus, root/"schedule")
    return root, corpus, schedule


def test_vectorized_eligibility_matches_bruteforce_with_equal_clock_prefixes():
    time = np.array([(i//3)*5*NS for i in range(240)], dtype=np.int64)
    actual, diagnostic = c.eligible_endpoints(time)
    expected = []
    for i in range(len(time)-32):
        cutoff = int(time[i])+1
        recent = [j for j in range(i+1) if int(time[j]) >= cutoff-30*NS]
        positive = {i+4, i+16, i+32}
        negative = [j for j in range(len(time)) if abs(int(time[j])-int(time[i])) >= 180*NS and j not in positive]
        if len(recent) >= 6 and time[i]-time[recent[0]] >= 24*NS and len(negative) >= 31:
            expected.append(i)
    np.testing.assert_array_equal(actual, expected)
    assert diagnostic["packets"] == 240 and diagnostic["eligible_endpoints"] == len(expected)


def test_negative_ids_exclude_all_positive_horizons_when_clock_gaps_are_large():
    time = np.arange(160, dtype=np.int64)*5*NS
    time[10:] += 1000*NS
    i = 5;positive = {i+4, i+16, i+32}
    expected = [j for j in range(len(time)) if abs(int(time[j])-int(time[i])) >= 180*NS and j not in positive]
    actual = c.negative_ids(time, i)
    np.testing.assert_array_equal(actual, expected)
    assert i+16 not in actual and i+32 not in actual
    assert np.all(np.diff(actual) > 0)


def test_support_excludes_30second_boundary_at_endpoint_plus_one_nanosecond():
    time = np.r_[np.array([0, 6, 12, 18, 24, 30])*NS, np.arange(36, 1200, 6)*NS].astype(np.int64)
    eligible, _ = c.eligible_endpoints(time)
    assert 5 not in eligible  # Six rows in [0,30], but left edge is1ns.
    rows = [token(i, int(a), previous=int(time[i-1]) if i else None) for i, a in enumerate(time)]
    assert not tokens.context_from_tokens(rows[:6], cutoff_ns=int(time[5])+1, driver_id="1").supported


@pytest.mark.parametrize("bad", [np.array([0., 1.]), np.array([1, 0]), np.array([-1, 1]), np.array([np.iinfo(np.int64).max])])
def test_invalid_clock_arrays_fail_without_numerical_wrapping(bad):
    with pytest.raises(ValueError): c.eligible_endpoints(bad)


def test_2023_is_rejected_before_any_raw_file_read(tmp_path, monkeypatch):
    events, design = inputs(tmp_path);events[-1]["event_key"] = 202301
    monkeypatch.setattr(tokens, "iter_packet_tokens", lambda *a, **kw: pytest.fail("Selection raw stream read"))
    with pytest.raises(ValueError, match="exactly the22"):
        c.build_corpus(events, tmp_path/"corpus", design_lock_path=design, design_sha256=c.sha(design))
    assert not (tmp_path/"corpus").exists()


def test_wrong_design_hash_precedes_raw_reads(tmp_path, monkeypatch):
    events, design = inputs(tmp_path)
    monkeypatch.setattr(tokens, "iter_packet_tokens", lambda *a, **kw: pytest.fail("Unfrozen raw input read"))
    with pytest.raises(ValueError, match="design lock"):
        c.build_corpus(events, tmp_path/"corpus", design_lock_path=design, design_sha256="0"*64)


def test_corpus_retains_every_driver_reports_no_support_and_uses_only_disk_arrays(prepared):
    _, corpus, _ = prepared;manifest = c.verify_corpus(corpus)
    assert [r["event_key"] for r in manifest["events"]] == list(c.EVENT_KEYS)
    assert manifest["driver_count"] == 66 and manifest["eligible_driver_count"] == 44
    assert manifest["ineligible_driver_count"] == 22
    assert manifest["lap_labels_read"] is False and manifest["model_fits"] == 0
    for event in manifest["events"]:
        assert event["ineligible_driver_count"] == 1
        assert [r["driver_id"] for r in event["drivers"]] == ["1", "2", "unavailable-driver"]
        for driver in event["drivers"]:
            for array in driver["arrays"].values():
                assert Path(array["path"]).exists()
                assert array["bytes"] == int(np.prod(array["shape"]))*np.dtype(array["dtype"]).itemsize


def test_unsupported_event_fails_without_reweighting_or_erasing_artifacts(tmp_path):
    events, design = inputs(tmp_path, small=True)
    with pytest.raises(ValueError, match="no eligible endpoint"):
        c.build_corpus(events, tmp_path/"corpus", design_lock_path=design, design_sha256=c.sha(design))
    failure = c._read(tmp_path/"corpus/corpus_failure.json")
    assert failure["completed_events"] == [202201]
    assert (tmp_path/"corpus/202201/event.json").exists()
    assert not (tmp_path/"corpus/corpus.json").exists()


def test_schedule_exact_pcg64_hierarchical_first_batch_and_unique_negatives(prepared):
    _, corpus, schedule = prepared;manifest = c.verify_corpus(corpus);saved = c._read(schedule)
    assert saved["steps"] == 800 and saved["batch_size"] == 32 and saved["seed"] == 20260908
    cache = c._DriverCache(manifest);arrays = {name: c._open(binding) for name, binding in saved["arrays"].items()}
    event_streams = [[i for i, (key, driver) in enumerate(cache.streams) if key == event and driver["eligible_endpoints"] > 0] for event in c.EVENT_KEYS]
    rng = np.random.default_rng(20260908)
    try:
        for row in range(32):
            drivers = event_streams[int(rng.integers(22))];stream = drivers[int(rng.integers(len(drivers)))];mapped = cache.get(stream)
            endpoint = int(mapped["eligible_endpoints"][int(rng.integers(len(mapped["eligible_endpoints"])))])
            negatives = rng.choice(c.negative_ids(mapped["availability_ns"], endpoint), 31, replace=False)
            assert arrays["stream_index"][0, row] == stream and arrays["endpoint"][0, row] == endpoint
            np.testing.assert_array_equal(arrays["positives"][0, row], endpoint+np.array([4,16,32]))
            np.testing.assert_array_equal(arrays["negatives"][0, row], negatives)
        assert len(np.unique(arrays["stream_index"]//3)) == 22
    finally: cache.close();c._close_maps(arrays)


def test_schedule_reproduction_is_exact_and_existing_outputs_immutable(prepared):
    root, corpus, schedule = prepared
    reproduced = c.build_schedule(corpus, root/"second_schedule")
    one, two = c._read(schedule), c._read(reproduced)
    for name in one["arrays"]: assert one["arrays"][name]["sha256"] == two["arrays"][name]["sha256"]
    with pytest.raises(FileExistsError): c.build_schedule(corpus, root/"schedule")


def test_provider_exact_token_context_positive_negative_replay_and_identical_repeated_calls(prepared):
    _, corpus, schedule = prepared
    with c.BatchProvider(corpus, schedule) as provider:
        a, b = provider(0), provider(0)
        assert a.context_keys == b.context_keys
        for name in ("contexts", "positives", "negatives"): torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0)
        found_equal_future = False
        for row in range(32):
            stream = int(provider.arrays["stream_index"][0, row]);i = int(provider.arrays["endpoint"][0, row]);mapped = provider.cache.get(stream)
            event, driver = provider.cache.streams[stream];time = mapped["availability_ns"]
            all_tokens = [tokens.PacketToken(driver["driver_id"], int(time[j]), int(mapped["packet_sequence"][j]),
                None if mapped["previous_availability_ns"][j] == -1 else int(mapped["previous_availability_ns"][j]), mapped["values"][j].copy()) for j in range(i+1)]
            expected = tokens.context_from_tokens(all_tokens, cutoff_ns=int(time[i])+1, driver_id=driver["driver_id"])
            assert expected.supported
            np.testing.assert_array_equal(a.contexts[row].numpy(), expected.values)
            np.testing.assert_array_equal(a.positives[row].numpy(), mapped["values"][i+np.array([4,16,32])])
            assert a.context_keys[row] == tokens.permutation_key(event, driver["driver_id"], int(mapped["packet_sequence"][i]))
            if time[i+1] == time[i]:
                found_equal_future = True
                np.testing.assert_array_equal(a.contexts[row, -1].numpy(), mapped["values"][i])
                assert not np.array_equal(a.contexts[row, -1].numpy(), mapped["values"][i+1])
        assert found_equal_future
        assert len(provider.cache.cache) <= 8
    with pytest.raises(ValueError, match="closed"): provider(0)


def test_lru_evicts_and_closes_old_driver_mapping(prepared):
    _, corpus, _ = prepared;cache = c._DriverCache(c.verify_corpus(corpus))
    first = cache.get(0)["values"]
    for index in range(1, 12): cache.get(index)
    assert len(cache.cache) == 8 and first._mmap.closed
    cache.close();assert len(cache.cache) == 0


def test_resource_limit_stops_before_provider_work(prepared, monkeypatch):
    _, corpus, schedule = prepared
    with c.BatchProvider(corpus, schedule) as provider:
        monkeypatch.setattr(c, "MAX_RSS_BYTES", 1)
        with pytest.raises(c.ResourceLimitError): provider(0)


def test_hashed_schedule_corruption_is_rejected_before_replay(prepared, tmp_path):
    _, corpus, schedule = prepared;manifest = c._read(schedule)
    source = Path(manifest["arrays"]["negatives"]["path"])
    corrupt = tmp_path/"negative.bin";corrupt.write_bytes(source.read_bytes()+b"bad")
    manifest["arrays"]["negatives"]["path"] = str(corrupt)
    changed = tmp_path/"schedule.json";c._write(changed, manifest)
    with pytest.raises(ValueError, match="hash or byte"): c.BatchProvider(corpus, changed)


def test_scheduled_positive_cannot_be_reused_as_negative_even_with_updated_array_hash(prepared, tmp_path):
    _, corpus, schedule = prepared;manifest = c._read(schedule)
    value = np.array(c._open(manifest["arrays"]["negatives"]), copy=True)
    positive = c._open(manifest["arrays"]["positives"])
    value[0, 0, 0] = positive[0, 0, 0]
    c._close_maps({"positive": positive})
    manifest["arrays"]["negatives"] = c._binary(tmp_path/"changed.bin", value, "<i8")
    changed = tmp_path/"schedule.json";c._write(changed, manifest)
    with c.BatchProvider(corpus, changed) as provider:
        with pytest.raises(ValueError, match="positive-overlapping"): provider(0)


def test_provider_checks_step_bounds_and_integral_identity(prepared):
    _, corpus, schedule = prepared
    with c.BatchProvider(corpus, schedule) as provider:
        for step in (-1, 800, True, 0.):
            with pytest.raises(ValueError): provider(step)
