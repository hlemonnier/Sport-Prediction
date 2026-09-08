# Raw telemetry availability pilot

The existing global next-eligible-clean-lap HGB uses completed laps, sectors and four speed traps. This pilot checks whether raw CarData channels provide usable additional information at that same issuance horizon. Other repository projects already use telemetry; the proposed novelty is its causal use for this specific target and issuance.

Exactly the 2022 and 2023 Bahrain race CarData.z streams are requested, using the session mappings in the existing hashed 44-race acquisition. The acquisition-only specification is frozen before the first request. Each event permits one mirror attempt and one official-source fallback, at most two requests concurrently, without redirects. Wire bodies and decoded bodies are capped at 64 MiB each, with exclusive writes and UTC acquisition receipts. HTTP success remains `downloaded_unparsed` until the separate strict parser completes. Acquisition timestamps are current download receipts, not original live client receipts.

The 150-second elapsed limit is a response-acceptance deadline checked before and after each bounded `read1`, not hard transport cancellation. An in-progress body read may overrun it by one 30-second socket-read timeout; DNS, connection/TLS and library scheduling are not certified wall-clock bounds. A late read cannot make a response complete. Body-cap detection can preserve one extra sentinel byte in an explicitly partial failed body. Existing event bodies, receipts, source records or decoded outputs prevent a new connection for that event. Identity, gzip and zlib-wrapped HTTP deflate decoding enforce the independent decoded-body cap; malformed, truncated or trailing compressed data fail explicitly.

The parser preserves physical stream order. A packet's availability is the cumulative maximum of archive-prefix milliseconds. At a forecast cutoff only strictly earlier packets qualify; equal-clock and later packets are excluded before their payload is decoded. Bundled measurements arrive together. Missing channels remain missing; zero and false remain valid observed values. Source Utc is diagnostic only. No full-session origin offset, lap assignment, interpolation, lap CSV, outcome, fitted feature or score is used in this pilot.

This source does not identify fuel, tyre energy or battery state directly. Any later model must keep the original issuance population and incumbent fallback, use availability-window features, and test fixed zero- and two-second additional delivery lags. Historical archive clocks cannot establish actual live delivery latency. This pilot authorizes neither model fitting nor later-season acquisition; predictive selection requires its own reviewed and frozen protocol.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python -m pytest -q -p no:cacheprovider research/experiments/boundary_20260908/telemetry_pilot/test_packets.py research/experiments/boundary_20260908/telemetry_pilot/test_acquire.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python -m research.experiments.boundary_20260908.telemetry_pilot.acquire --dry-run
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python -m research.experiments.boundary_20260908.telemetry_pilot.acquire
```

Raw streams remain local under `data/f1/boundary_20260908/telemetry_pilot/`; closed receipts and later quality diagnostics are under the matching `artifacts/research` directory. Completed attempts cannot be overwritten. The parser tests include poisoned future packets, time regressions, bounded decompression, missingness and bundled-entry preservation.

Acquisition tests use synthetic HTTP responses only: no requests, provider data, lap tables, labels or model fitting. The acquisition lock binds this README and both test files as well as the parser, acquisition implementation and specification. Later diagnostics use a separate source lock.

Primary implementation reference: installed FastF1 3.8.3 `_api.car_data` and its raw-DEFLATE decoder. The pilot deliberately avoids the processed API's zero filling and the full-session origin calculation. The [FastF1 source](https://github.com/theOehrly/Fast-F1/blob/master/fastf1/_api.py) documents the raw channel mapping; repository experiments, rather than that source, must establish predictive value.

Suggested commit: `research(f1-live): validate raw telemetry packet availability pilot`.
