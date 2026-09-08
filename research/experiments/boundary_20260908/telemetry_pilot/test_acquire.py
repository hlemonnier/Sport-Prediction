"""Synthetic transport/immutability checks; never makes a network request."""
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zlib

import pytest
import requests
from requests.structures import CaseInsensitiveDict

from research.experiments.boundary_20260908.telemetry_pilot import acquire as a


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network access forbidden in acquisition tests")
    monkeypatch.setattr(a.requests, "get", forbidden)


def encoded(body, encoding):
    return gzip.compress(body) if encoding == "gzip" else zlib.compress(body)


@pytest.mark.parametrize("encoding", ["gzip", "deflate"])
def test_http_decompression_matches_original_and_exact_cap(tmp_path, encoding):
    body = b"raw archival stream\n"*10
    path = tmp_path / "attempt.httpbody"
    path.write_bytes(encoded(body, encoding))
    result = a.decode_body(path, encoding, len(body))
    assert result.read_bytes() == body
    assert path.read_bytes() == encoded(body, encoding)


@pytest.mark.parametrize("encoding", ["gzip", "deflate"])
def test_http_decompression_bomb_cap(tmp_path, encoding):
    path = tmp_path / "attempt.httpbody"
    path.write_bytes(encoded(b"x"*100000, encoding))
    with pytest.raises(ValueError, match="exceeds cap"):
        a.decode_body(path, encoding, 100)
    assert not path.with_suffix(".decoded.jsonStream").exists()


@pytest.mark.parametrize("encoding", ["gzip", "deflate"])
@pytest.mark.parametrize("mutation", ["truncate", "trailing", "second_member", "corrupt"])
def test_bad_http_encoding_fails_without_publishing_decoded_body(tmp_path, encoding, mutation):
    compressed = encoded(b"body", encoding)
    altered = {"truncate": compressed[:-1], "trailing": compressed+b"extra",
        "second_member": compressed+compressed, "corrupt": b"not compressed"}[mutation]
    path = tmp_path / "attempt.httpbody"
    path.write_bytes(altered)
    with pytest.raises((ValueError, zlib.error)):
        a.decode_body(path, encoding, 100)
    assert not path.with_suffix(".decoded.jsonStream").exists()


@pytest.mark.parametrize("encoding", ["", "identity"])
def test_identity_encoding_enforces_independent_decoded_cap(tmp_path, encoding):
    path = tmp_path / "body"
    path.write_bytes(b"1234")
    assert a.decode_body(path, encoding, 4) == path
    with pytest.raises(ValueError, match="exceeds cap"):
        a.decode_body(path, encoding, 3)


@pytest.mark.parametrize("encoding", ["br", "gzip, deflate", "raw-deflate"])
def test_unknown_http_encoding_rejected(tmp_path, encoding):
    path = tmp_path / "body"
    path.write_bytes(b"body")
    with pytest.raises(ValueError, match="Unsupported"):
        a.decode_body(path, encoding, 100)


@pytest.mark.parametrize("cap", [True, 0, -1, 1.0, float("inf")])
def test_invalid_decoded_cap_rejected(tmp_path, cap):
    path = tmp_path / "body"
    path.write_bytes(b"body")
    with pytest.raises(ValueError, match="positive integer"):
        a.decode_body(path, "identity", cap)


def test_exclusive_json_and_decoded_writes_preserve_existing_bytes(tmp_path):
    receipt = tmp_path / "receipt.json"
    a.write(receipt, {"original": True})
    previous = receipt.read_bytes()
    with pytest.raises(FileExistsError):
        a.write(receipt, {"new": True})
    assert receipt.read_bytes() == previous
    path = tmp_path / "body.httpbody"
    path.write_bytes(gzip.compress(b"decoded"))
    destination = path.with_suffix(".decoded.jsonStream")
    destination.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        a.decode_body(path, "gzip", 100)
    assert destination.read_bytes() == b"existing"


class Raw:
    def __init__(self, body, on_read=lambda: None):
        self.buffer = io.BytesIO(body)
        self.on_read = on_read
        self.calls = []

    def read1(self, count, *, decode_content):
        assert decode_content is False
        self.calls.append(count)
        self.on_read()
        return self.buffer.read(count)


class Response:
    def __init__(self, body=b"stream", status=200, headers=None, on_read=lambda: None):
        self.status_code, self.url = status, "https://synthetic.invalid/requested"
        self.headers = CaseInsensitiveDict(headers or {})
        self.raw = Raw(body, on_read)
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def scope(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(a, "ROOT", tmp_path)
    monkeypatch.setattr(a, "DATA", data)
    clock = [0.0]
    monkeypatch.setattr(a.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(a, "now", lambda: "2026-09-08T00:00:00+00:00")
    spec = {"base_urls": a.ALLOWED_BASE_URLS, "request_timeout_connect_seconds": 15,
        "request_timeout_read_seconds": 30, "max_http_body_bytes": 64,
        "max_decoded_body_bytes": 64, "max_request_elapsed_seconds": 150}
    return SimpleNamespace(data=data, clock=clock, spec=spec,
        session={"event_key": 202201, "session_path": "2022/fixed/Race/"})


def transport(monkeypatch, replies):
    calls, remaining = [], iter(replies)
    def get(url, **kwargs):
        calls.append((url, kwargs))
        response = next(remaining)
        if isinstance(response, Exception):
            raise response
        response.url = url
        return response
    monkeypatch.setattr(a.requests, "get", get)
    return calls


def test_success_is_unparsed_bound_and_stops_before_official_fallback(scope, monkeypatch):
    response = Response(b"stream", headers={"Content-Length": "6"})
    calls = transport(monkeypatch, [response])
    result = a.fetch(scope.session, scope.spec)
    assert result["status"] == "downloaded_unparsed" and len(calls) == 1 and response.closed
    assert result["decoded_body_bytes"] == 6
    raw = a.ROOT / result["decoded_body_path"]
    assert raw.read_bytes() == b"stream" and result["decoded_body_sha256"] == a.sha(raw)
    assert result["attempts"][0]["wire_body_complete"] is True
    url, kwargs = calls[0]
    assert url == a.ALLOWED_BASE_URLS[0]+"/static/2022/fixed/Race/CarData.z.jsonStream"
    assert kwargs["allow_redirects"] is False and kwargs["stream"] is True
    assert kwargs["timeout"] == (15, 30) and kwargs["headers"]["Accept-Encoding"] == "identity"
    assert json.loads((scope.data / "202201_CarData.z.source.json").read_text()) == result


@pytest.mark.parametrize("first", [403, 429, 500, 302])
def test_error_body_preserved_and_exactly_one_fallback(scope, monkeypatch, first):
    responses = [Response(b"denied", first, {"Location": "https://not-followed.invalid"}), Response(b"ok")]
    calls = transport(monkeypatch, responses)
    result = a.fetch(scope.session, scope.spec)
    assert result["status"] == "downloaded_unparsed" and len(calls) == 2
    assert calls[1][0].startswith(a.ALLOWED_BASE_URLS[1]+"/static/")
    failed = result["attempts"][0]
    assert failed["http_status"] == first and failed["wire_body_complete"] is True
    assert (a.ROOT / failed["wire_body_path"]).read_bytes() == b"denied"
    assert all(response.closed for response in responses)


def test_connection_failure_receipted_then_fallback(scope, monkeypatch):
    calls = transport(monkeypatch, [requests.ConnectionError("synthetic refusal"), Response(b"ok")])
    result = a.fetch(scope.session, scope.spec)
    assert len(calls) == 2 and result["status"] == "downloaded_unparsed"
    failed = result["attempts"][0]
    assert failed["error_type"] == "ConnectionError" and "wire_body_path" not in failed
    assert (scope.data / "202201_CarData.z.attempt1.receipt.json").exists()


def test_body_cap_failure_keeps_partial_sentinel_and_fallback(scope, monkeypatch):
    first = Response(b"x"*100)
    transport(monkeypatch, [first, Response(b"ok")])
    result = a.fetch(scope.session, scope.spec)
    failed = result["attempts"][0]
    assert failed["wire_body_bytes"] == scope.spec["max_http_body_bytes"]+1
    assert failed["wire_body_complete"] is False and "exceeds cap" in failed["error"]
    assert first.raw.calls == [65]
    assert result["status"] == "downloaded_unparsed"


@pytest.mark.parametrize("headers,body", [({"Content-Length": "7"}, b"short"),
    ({"Content-Length": "nonsense"}, b"body"), ({}, b""),
    ({"Content-Encoding": "br"}, b"body")])
def test_invalid_success_never_published_as_complete_download(scope, monkeypatch, headers, body):
    calls = transport(monkeypatch, [Response(body, headers=headers), Response(b"denied", 403)])
    result = a.fetch(scope.session, scope.spec)
    assert len(calls) == 2 and result["status"] == "unavailable"
    assert "error_type" in result["attempts"][0] and "decoded_body_path" not in result


def test_http_gzip_success_binds_both_wire_and_decoded_bodies(scope, monkeypatch):
    compressed = gzip.compress(b"stream")
    transport(monkeypatch, [Response(compressed, headers={"Content-Encoding": "gzip", "Content-Length": str(len(compressed))})])
    result = a.fetch(scope.session, scope.spec)
    assert result["status"] == "downloaded_unparsed"
    assert (a.ROOT / result["decoded_body_path"]).read_bytes() == b"stream"
    assert (a.ROOT / result["attempts"][0]["wire_body_path"]).read_bytes() == compressed


def test_deadline_after_read_rejects_late_complete_body(scope, monkeypatch):
    def late():
        scope.clock[0] += 151
    transport(monkeypatch, [Response(b"late", on_read=late), Response(b"ok")])
    result = a.fetch(scope.session, scope.spec)
    failed = result["attempts"][0]
    assert failed["error_type"] == "TimeoutError" and failed["wire_body_complete"] is False
    assert failed["wire_body_bytes"] == 0 and failed["wall_seconds"] == 151
    assert result["status"] == "downloaded_unparsed"


def test_deadline_after_eof_read_cannot_certify_body_complete(scope, monkeypatch):
    steps = iter([100, 151])
    def clock_steps():
        scope.clock[0] = next(steps)
    transport(monkeypatch, [Response(b"late EOF", on_read=clock_steps), Response(b"ok")])
    result = a.fetch(scope.session, scope.spec)
    failed = result["attempts"][0]
    assert failed["error_type"] == "TimeoutError" and failed["wire_body_complete"] is False
    assert failed["wire_body_bytes"] == len(b"late EOF")


@pytest.mark.parametrize("suffix", ["attempt1.httpbody", "attempt1.receipt.json", "attempt1.decoded.jsonStream", "source.json"])
def test_existing_artifacts_block_before_any_request(scope, monkeypatch, suffix):
    path = scope.data / f"202201_CarData.z.{suffix}"
    path.write_bytes(b"original")
    calls = transport(monkeypatch, [])
    with pytest.raises(FileExistsError, match="Existing event"):
        a.fetch(scope.session, scope.spec)
    assert calls == [] and path.read_bytes() == b"original"


@pytest.fixture
def validated_spec(tmp_path, monkeypatch):
    spec = json.loads(a.SPEC.read_text())
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({"sessions": spec["sessions"]}))
    spec["parent_session_manifest"] = {"path": "parent.json", "sha256": a.sha(parent)}
    path = tmp_path / "specification.json"
    path.write_text(json.dumps(spec))
    monkeypatch.setattr(a, "ROOT", tmp_path)
    monkeypatch.setattr(a, "SPEC", path)
    monkeypatch.setattr(a.shutil, "disk_usage", lambda _: SimpleNamespace(free=2**30))
    return spec, path, parent


def test_fixed_spec_accepts_exact_closed_parent_mapping(validated_spec):
    spec, _, _ = validated_spec
    assert a.validate_spec() == spec


@pytest.mark.parametrize("key,value", [
    ("base_urls", ["https://wrong.invalid", "https://livetiming.formula1.com"]),
    ("base_urls", list(reversed(a.ALLOWED_BASE_URLS))), ("max_parallel_requests", 3),
    ("max_requests_per_event", 3), ("follow_redirects", True), ("stream", "TimingData"),
    ("max_http_body_bytes", 2**30), ("max_decoded_body_bytes", 0),
    ("max_request_elapsed_seconds", 150.0), ("request_timeout_read_seconds", True),
    ("minimum_free_disk_bytes", 1), ("elapsed_deadline_semantics", "hard_cancel")])
def test_changed_scope_or_resource_contract_rejected(validated_spec, key, value):
    spec, path, _ = validated_spec
    altered = deepcopy(spec)
    altered[key] = value
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError):
        a.validate_spec()


def test_parent_mapping_bytes_or_session_drift_rejected(validated_spec):
    spec, path, parent = validated_spec
    spec["sessions"][0]["session_path"] = "different/"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="mapping drift"):
        a.validate_spec()
    parent.write_text("{}")
    with pytest.raises(ValueError, match="manifest drift"):
        a.validate_spec()


def test_insufficient_disk_fails_before_requests(validated_spec, monkeypatch):
    monkeypatch.setattr(a.shutil, "disk_usage", lambda _: SimpleNamespace(free=2**20))
    with pytest.raises(ValueError, match="disk headroom"):
        a.validate_spec()
