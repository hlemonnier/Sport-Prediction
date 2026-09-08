# Race-order gap feasibility pilot

Status: **source and synthetic tests only; historical payload analysis has not
run**. This is a conditional next information audit while the independent CPC
experiment executes. No CPC source is changed.

The [specification](specification.json) fixes Bahrain 2022 and 2023, their already
acquired TimingData streams, and all 853 and 874 original issuances. Both zero-
and two-second cutoffs will be inspected, producing 3,454 retained snapshots.
Only original issuance identity and clock metadata are used from the closed
issuance files. No target table, fitted model, prediction score, new download or
additional race is part of the pilot.

`pilot.py` uses a local binary header cursor with physical packet sequence and
cumulative archive clock. It treats every driver's patch in one packet
atomically and requires packet availability strictly before the cutoff. Fields
absent from a patch retain their original observation timestamps; this does not
invent a new measurement or interpolate a value. Explicit nulls and blank values
clear the previous observation. Metadata-only objects do not refresh its age.
Only the next exact 12-byte ASCII timestamp header is read before admission;
payload reading, UTF-8/JSON decoding and size validation are deferred. Tests
establish prefix invariance even for future invalid UTF-8 or oversize payloads.
An invalid timestamp has no certified temporal placement: consumption stops
permanently at that physical boundary once reached, its availability stays null,
and subsequent snapshots carry an uncertainty flag. A separate inventory pass
may continue through the full file only after both snapshot files close. It
cannot feed information back into forecasts. The original pre-review
[specification](specification_before_header_cursor.json) is retained; the revised
contract removes the generic text decoder dependency before historical analysis.

The parser separates seconds, leader markers, leader lap counters, categorical
lap deficits, clears, unavailable markers and invalid values. Position means
race rank. Rank changes invalidate earlier rival intervals until another update;
duplicate reported ranks and contradictory leader categories are flagged.
Every field retains its original and cumulative timestamps, sequence and age.
Readiness uses a fixed ten-second age **at the admission cutoff**, with parallel
2/5/10/30-second diagnostics. The runner also reports age at actual issuance;
with the two-second latency assumption, a ten-second cutoff age is twelve
seconds at issuance. These flags do not remove issuances or form a performance
advancement gate.

Race-order interval is **not physical nearest-car distance** and cannot locate
lapped traffic. FastF1 identifies the fields in its [primary timing parser](https://raw.githubusercontent.com/theOehrly/Fast-F1/v3.8.3/fastf1/_api.py);
this pilot does not use its processed forward-filled timing frame or derived
driver-ahead distance. Raw field coverage remains unknown until the locked pilot
is authorized and executed. Archive clocks and the original canonical issuance
times remain retrospective availability proxies rather than certified historical
client receipt times.

An independent review must approve the exact immediate source/specification
files before `freeze`. The `run` command requires that lock and rechecks every
input hash. Outputs and failed attempts are retained without replacement.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python -m pytest -q --import-mode=importlib -p no:cacheprovider research/experiments/boundary_20260908/rival_gap_pilot/test_pilot.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python -m research.experiments.boundary_20260908.rival_gap_pilot.pilot freeze --review PATH_TO_APPROVED_REVIEW
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python -m research.experiments.boundary_20260908.rival_gap_pilot.pilot run
```

Suggested commit: `research(f1-live): prepare causal rival-gap feasibility pilot`.
