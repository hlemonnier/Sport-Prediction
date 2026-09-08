# Race-order gap feasibility: Bahrain 2022 and 2023

The corrected, target-free pilot found usable race-order gap observations at the
fixed ten-second cutoff-age threshold for **1,633 of 1,727 original issuances
(94.56%)** under both latency settings in aggregate. This establishes input
feasibility in two races at one circuit. **No model was fitted or scored, no
predictive gain was established, and no production model was changed.**

The first locked run stopped at a leading UTF-8 byte-order mark in both acquired
streams. Its 3,454 zero-ready snapshots are preserved as an ingestion failure;
they are not evidence that gap data are unavailable. A separately reviewed and
locked sibling accepts exactly one optional BOM at file byte zero. The
executable change contains only this handling and its separate output path.
Midstream or repeated BOMs remain invalid. The
[v1 result](evidence/boundary_rival_gap_pilot_20260908/v1_result.json),
[v2 amendment](evidence/boundary_rival_gap_pilot_20260908/v2_amendment_specification.json)
and [v2 result](evidence/boundary_rival_gap_pilot_20260908/v2_result.json) retain
the full sequence.

All 853 Bahrain 2022 and 874 Bahrain 2023 original issuances are retained at both
zero- and two-second latency assumptions, giving 3,454 diagnostic rows. The
two-second setting is primary; zero seconds is a fixed sensitivity. Both use
the same input files, original issuance identities and clocks, parser categories,
freshness thresholds and readiness conditions declared before payload analysis.

| Race | Latency | Original issuances | Ready: age ≤2s | ≤5s | ≤10s | ≤30s | Ready at 10s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bahrain 2022 | 2s | 853 | 0 | 806 | 807 | 807 | 94.61% |
| Bahrain 2022 | 0s | 853 | 798 | 798 | 802 | 802 | 94.02% |
| Bahrain 2023 | 2s | 874 | 1 | 823 | 826 | 826 | 94.51% |
| Bahrain 2023 | 0s | 874 | 826 | 826 | 831 | 831 | 95.08% |

Age is measured **at the admission cutoff**, not at actual issuance. Under the
two-second assumption, ten seconds at cutoff can therefore mean twelve seconds
at issuance; each saved field reports both ages. The 2/5/10/30-second thresholds
are fixed descriptive diagnostics. No observed coverage threshold selects a
model or authorizes promotion.

The runner reads only a bounded binary timestamp header before admission. It
uses physical packet sequence and cumulative maximum archive timestamps, and
admits a whole packet only when its availability is strictly earlier than the
cutoff. Payload reading, UTF-8/JSON decoding and size validation happen after
that comparison. Invalid timestamp headers stop forecasting consumption without
assigning them a guessed clock. Complete-file inventory is a separate pass after
both snapshot files close.

Both corrected full-stream inventories are structurally clean: **69,772 packets
in 2022 and 65,627 in 2023**, with zero invalid timestamp headers, timestamp
regressions or payload errors. Across all saved snapshots there are zero unknown
ranks, duplicate reported ranks, intervals predating the latest rank change,
leader/rank category inconsistencies or source uncertainty flags. All three
fields have a recorded observation at every issuance; observed invalid or stale
values remain explicitly distinct from never-observed fields.

The frozen category parser still reports nonnumeric values it does not
recognize. Full-stream `GapToLeader` updates include 20 such values in 2022 and
23 in 2023; the interval field has three in each race. These are field-category
observations, not JSON or timestamp failures. At saved issuances,
`GapToLeader` is classified invalid on 66/853 and 89/874 rows respectively, at
each latency. The parser does not change those categories after inspection.
No update was classified into its declared lap-deficit category; this does not
establish the absence of lapped cars or certify unrecognized strings' meanings.

Independent pre-lock review reran **69 v1 tests** and **78 v2 tests** and approved
the exact source maps. V2 regressions cover leading/no-BOM parity, midstream and
repeated BOM rejection, future invalid UTF-8 and oversize payload deferral,
strict equal-time exclusion, explicit clears, rank changes and unplaceable
timestamps. A separate post-run verification by the implementer checked all
3,454 saved rows against the original identities and order, source/input hashes,
strict clock inequalities, age arithmetic, readiness conditions and aggregate
counts. That verifier reads saved diagnostics and metadata; it does not
independently replay raw feed payloads. Its
[receipt](evidence/boundary_rival_gap_pilot_20260908/v2_snapshot_verification.json)
is separate from the
[independent source review](evidence/boundary_rival_gap_pilot_20260908/v2_review.json).

`Position` means race rank. `IntervalToPositionAhead` and `GapToLeader` are
race-order gaps, not physical nearest-car distances; asynchronous ranks do not
certify an ahead-driver identity or locate lapped traffic. The field mapping is
documented in the version-pinned
[FastF1 timing parser](https://raw.githubusercontent.com/theOehrly/Fast-F1/v3.8.3/fastf1/_api.py).
This pilot does not use FastF1's processed gap forward-fill/interpolation.
Archive and original canonical issuance clocks remain retrospective availability
proxies, not certified historical client receipt times. Only Bahrain 2022/2023
were replayed; coverage across circuits, wet races or the other discovery
sessions is untested. Existing seasons are already exposed.

The reproducible source is
[rival_gap_pilot_v2](../../research/experiments/boundary_20260908/rival_gap_pilot_v2/README.md),
with the unchanged [v1 source](../../research/experiments/boundary_20260908/rival_gap_pilot/README.md)
retained. The
[publication manifest](evidence/boundary_rival_gap_pilot_20260908/publication_manifest.json)
binds byte-identical compact receipts and source hashes. Raw provider streams,
issuance JSONL files and diagnostic JSONL files are not copied into this
publication; reproducing the replay requires the hash-bound local inputs.

Suggested commit: `research(f1-live): publish bounded rival-gap feasibility evidence`.
