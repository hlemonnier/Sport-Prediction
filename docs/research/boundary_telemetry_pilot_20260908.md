# Raw telemetry pilot at the existing forecast horizon

The two fixed Bahrain archives passed complete structural and clock validation. This establishes a usable raw-data route for a new next-eligible-clean-lap experiment; **no telemetry candidate has been fitted or scored**, and the globally integrated HGB remains unchanged.

| Pilot race | Archived packets | Bundled sample entries | Raw car rows | Drivers | Largest packet-clock gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022 Bahrain | 9,626 | 36,656 | 733,120 | 20 | 3.20 s |
| 2023 Bahrain | 9,486 | 36,065 | 721,300 | 20 | 12.48 s |

Every raw car row contains all six inspected channel keys. Neither stream contains a regressing packet clock, duplicate packet clock, invalid source UTC, missing car object or missing channel object. The 1,454,420 car rows are correlated telemetry records, not independent predictive validation examples. Packets bundle up to 14 and 17 sample entries respectively; those entries become available together at the packet clock.

Numeric completeness does not imply physical validity. Throttle code 104 occurs in **72,279 rows (9.86%)** in 2022 and **100,975 rows (14.00%)** in 2023. The 2023 stream also contains 63 gear values above eight, reaching 49. Raw brake values are exclusively 0, 100 and 104 in these pilots. The [FastF1 implementation](https://github.com/theOehrly/Fast-F1/blob/main/fastf1/_api.py) treats brake as a Boolean channel; its raw numeric magnitude is not measured braking pressure. Future feature code must preserve and explicitly handle unexpected values, with missingness and data-quality controls, rather than silently convert them into physical state estimates. The meaning of code 104 is not established by these frequency counts.

The acquisition requested only `CarData.z` for 202201 and 202301, with the session mappings and source code locked beforehand. Three HTTP attempts sufficed: the 2022 mirror succeeded; the 2023 mirror returned 404 and the official source succeeded. Both successful bodies and the failed response are preserved with current acquisition receipts. The successful streams total **16,584,058 bytes**. No later-season stream, lap CSV or model outcome was used by this pilot.

The strict parser preserves physical order and uses cumulative-max archive-prefix milliseconds. Queries accept only packets strictly before the cutoff and stop before decoding a future packet. Synthetic checks cover poisoned future payloads, regressions, exact decompression limits, duplicates, missing/null/zero/false distinctions and atomic bundle preservation. **128 synthetic tests passed** before acquisition: 72 packet tests and 56 transport tests. Independent parser review additionally checked installed-FastF1 decoding equivalence and cutoff replay cases. A separate complete replay using the installed FastF1 decoder reproduced all packet, channel and driver diagnostics exactly and verified 21 unique file bindings.

This is an archive-availability experiment. Source UTC is diagnostic only; no full-session origin estimate or future interpolation is used. Download receipts are not historical client receipts. Complete structural validation alone does not establish availability at each original forecast, real delivery latency, tyre energy, battery state, fuel load or predictive gain. A subsequent experiment must retain every original issuance and target, use the exact incumbent fallback, and compare measured values with a matched support/missingness-only control at fixed delivery lags before any later-season evaluation.

Closed acquisition SHA256: `9d39fc16e23db3a3c9066b94e2add265d6a68e639d73417b3d3a813c348ca0e4`. Closed diagnostic SHA256: `9f1650bd43d6868874835398debebe12a1727c7bc18df8ae6aa1ceaa5ea92a4f`. The [source protocol](../../research/experiments/boundary_20260908/telemetry_pilot/README.md) records the bounded acquisition and source-time contract. Raw data remain local; the [publication manifest](evidence/boundary_telemetry_pilot_20260908/publication_manifest.json) binds the compact evidence copies and source files.

Suggested commit: `research(f1-live): validate raw telemetry availability on two races`.
