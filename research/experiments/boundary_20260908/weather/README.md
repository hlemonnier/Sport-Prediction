The weather lane adds new observed information to the historical research data.
The original 44 discovery races had no adjacent weather CSVs. `acquire.py`
obtained their public race WeatherData streams from the upstream F1 archive or
FastF1's configured fallback mirror, retaining raw bytes and exact source URLs.

All **44 races and 7,425 weather observations** were acquired. No prediction
gain is implied by acquisition. The separately frozen model experiment lives
under `model/`; independent source and clock checks live under `quality/`.

Each decoded time is the stream's recorded session-clock value. No scheduled
start or GMT offset is added. Missing or invalid channels remain missing rather
than becoming fabricated zero measurements or dry conditions. Meeting identity
uses a unique normalized name match, including the repository's older São Paulo
directory spelling; provider calendar numbering is preserved as metadata rather
than used to relabel the repository's races.

The first attempt found the 2023 index absent on the mirror; the primary index
was available. The second exposed the legacy São Paulo slug. Both failures
occurred before weather fitting or scoring and are recorded separately. The
completed manifest binds each source, raw and decoded hash and acquisition
time. File acquisition time is not historical publication time.

Reproduce checks from the repository root with `PYTHONPATH=.`:

```sh
.venv-f1/bin/python -m pytest -q research/experiments/boundary_20260908/weather/test_acquire.py
```

Completed acquisition refuses to overwrite its manifest. The model must use
strictly prior weather observations and its predeclared receipt-lag assumption;
historical stream timestamps do not establish client receipt timestamps.

Suggested commit: `research(f1-live): acquire and validate historical weather inputs`.
