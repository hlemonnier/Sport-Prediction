# Weather parser and clock alignment review

The completed 2022–2023 acquisition passes the independent parser, identity and time-alignment review. The weather-model lane can use the data with the declared 60-second availability lag. That lag is an assumption about delivery, not verified historical client receipt latency.

All 44 race streams were checked against their raw/CSV hashes, source receipts and provider meeting plus Race-session identities. Their 7,425 records each contain all seven weather fields, with no invalid or missing numeric values and no partial updates. Rainfall is explicitly encoded as string `"0"` in 6,869 records and `"1"` in 556; the parser preserves missing or invalid rainfall as missing, rather than converting it to false. Ten regression tests passed, including missing-field handling, binary rainfall, BOM decoding and rejection of duplicate or reversed timestamps.

The timestamp contract is direct: `WeatherData.jsonStream` prefixes and exported race-lap `Time` values use the same recorded session clock. Do not add or subtract scheduled race start, actual green-flag time or GMT offset. Installed FastF1 3.8.3 documents stream zero separately as `Session.t0_date`, and its weather parser converts each 12-character prefix directly to a session-time delta. Its lap `Time` is the completed-lap session timestamp, with postprocessing to improve timing consistency.

This is supported by source inspection and actual native-cache comparisons:

- `fastf1/_api.py:1611` defines weather timestamp semantics and channel units; `fetch_page` at line 1744 splits the 12-character prefix.
- `fastf1/core.py:1352` defines stream zero; line 1342 distinguishes actual session start. `_api.py:125` describes lap timestamp postprocessing, and line 151 identifies the completed-lap time.
- Three 2026 native cache examples—Australia, Japan and Miami—exactly reproduce all 472 exported weather rows. All 3,148 comparable cached/exported lap timestamps agree with zero difference. Five lap rows have no finite native timing counterpart and are explicitly counted as noncomparable, not declared verified. No 2026 forecasting metrics were read or scored.

For all 38,370 original eligible matched discovery checkpoints, a strict weather-time `< issuance-time - 60 seconds` join finds a prior reading. Reading ages have minimum/median/99th percentile/maximum **60.001 / 89.853 / 119.39931 / 120.012 seconds**. None exceeds 180 seconds. The streams contain one update gap as large as 480.072 seconds, but no checkpoint in this population is affected by stale weather at that gap. Typical update cadence is 60 seconds.

Observed numeric ranges agree with the channel units: air temperature 11.1–37.2 °C, track temperature 15.6–67.0 °C, humidity 5–94%, pressure 778.5–1019 mbar, wind direction 0–359 degrees and wind speed 0–10.1 m/s. These are source-quality checks, not predictive findings.

One acquisition blocker was found and fixed by the acquisition owner: the historical path slug `s_o_paulo` did not equal the transliterated provider-name slug `sao_paulo`. The final code accepts either uniquely matching name form and records the mapping method. Provider calendar meeting numbers differ from canonical race-round numbers for 17 later 2023 events after the cancelled calendar entry; matching by unique meeting name/key and Race session is correct. Forcing numerical round equality would misalign those events.

Historical receipt time is the remaining limit. The downloaded archive contains stream timestamps, not when a specific live client received the readings. The new `.source.json` download timestamps establish today's acquisition provenance, not original availability. A zero-lag sensitivity run cannot establish production causality. If later streams contain partial updates, carry-forward logic must retain each channel's own observation timestamp and staleness; never backfill or treat an omitted channel as newly observed zero. No partial-update reconstruction was required in these 44 streams.

Canonical quality artifact: `artifacts/research/boundary_20260908/weather/quality/review.json`

SHA256: `7d298751fd0662c9d4be7c3e47b09e17423fd55c9490158ced8117b5b51e0b19`

Reviewed acquisition-manifest SHA256: `78f76a325c6724a1a3b69ce4183d83a36480c11bbe3aa6b1d79c784ef9ab9ec5`

```sh
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python research/experiments/boundary_20260908/weather/quality/review.py
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python -m pytest -q -p no:cacheprovider research/experiments/boundary_20260908/weather/quality/test_parser.py
```

No model fitting, forecasting-error scoring or raw-data modification was performed by this review. Suggested commit: `test(f1): verify weather parsing and session clock alignment`
