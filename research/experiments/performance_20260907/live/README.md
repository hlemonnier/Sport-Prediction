The frozen ridge correction improved event-balanced next-eligible-lap MAE from
0.510496s to 0.496580s across the 48 races in 2024–2025: a 2.726% reduction,
with paired event-bootstrap delta CI [−0.020599, −0.008162]s. It improved 39 races;
every leave-one-event-out mean remained improving. Separate 2024 and 2025 CIs
also excluded zero. The mechanism and ridge penalty were selected on 2023,
then its coefficients were refitted once on 2022–2023 before transfer scoring.

| Frozen mechanism | Selected parameter | 2024–25 MAE | Delta vs naïve | Events improved |
|---|---|---:|---:|---:|
| Naïve last eligible lap | — | 0.510496s | — | — |
| Robust local level | alpha 0.7, innovation clip 2s | 0.575845s | +0.065348s | 38/48 |
| Robust local trend | shrinkage 0.25 | 0.515324s | +0.004828s | 5/48 |
| Other-car common increment | shrinkage 0.25 | 0.506843s | −0.003653s | 26/48 |
| Ridge correction | penalty 100 | 0.496580s | −0.013917s | 39/48 |

The clipped local level won many ordinary races but adapted too slowly after
large eligible-lap changes. At its selected settings it can move only 1.4s per
observation. In 2024 Japan, some initial eligible observations anchored it
tens of seconds above subsequent pace; event MAE rose from 0.5433s to 2.4452s.
This failure is included in all results. No event-wide “clean pace” mask removes
those observations or changes the population for a challenger.

The already exposed nine-race 2026 diagnostic block gives ridge MAE 0.541081s
versus naïve 0.549184s, a 1.475% reduction. Its CI [−0.022039, +0.006396]s crosses
zero. Six races improved. These historical comparisons are research evidence,
not prospective validation or a production promotion.

Inventory: 101 canonical race streams, 111,717 raw rows, 92,304 issuances and
90,348 matched targets. The 2024–25 transfer block contains 43,834 matched rows;
2026 contains 8,144. The 2026 target identities, naïve predictions and outcomes
match the retained v10 replay exactly. On that population, the existing SSM's
row MAE is 0.694288s versus naïve 0.558880s. Its deterioration is concentrated
in same-stint forecasts (0.647620s versus 0.507737s); neither method handles
cross-stint targets well (approximately 2.06s row MAE for both).

Raw completed-lap timestamps determine observation order. Same-timestamp cars
cannot use each other's observations. Stint transitions use only available
pit-out, compound and forward provider state; the 446 missing Stint values in
2025 Miami are never backward-filled. Eligibility is decided at observation
arrival. The next eligible target is matched later without rebuilding its
forecast. Recorded accuracy and status flags are assumed available on receipt;
these CSVs do not establish the provider's original publication latency.

Five regression tests cover future poisoning, timestamp ties, missing-stint
resets, target-independent issuance and feature immutability. Six real-stream
prefix cases compare both truncation and poisoned futures across every feature
and selected mechanism, including the ridge predictor. The final verification
also recomputes scores and serialized-model predictions from the exported CSV,
and verifies all 101 input hashes.

Canonical results are in
`artifacts/research/performance_20260907/live/corrected_cycle_1/results.json`.
Its neighboring `selected_models.json`, `matched_forecasts.csv.gz` and
`verification.json` contain coefficients, row-level predictions and hashes.
The first execution in the parent artifact directory is preserved and marked
invalid: an output/input column-name collision corrupted its ridge transfer
predictions. Fixing names changed no model, grid, partition or loss.

Run from the repository root with the existing Python environment:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv-f1/bin/python -m pytest -q -p no:cacheprovider research/experiments/performance_20260907/live/test_experiment.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv-f1/bin/python research/experiments/performance_20260907/live/run_experiment.py --output-dir artifacts/research/performance_20260907/live/replication_01
```

The runner refuses to overwrite completed results. Suggested commit:
`research(f1-live): evaluate frozen causal next-lap challengers`.

The completed recent-2026 extension is in
`artifacts/research/performance_20260907/live/recent_extension_final/results.json`.
It used the same coefficients fitted on 2022–2023, frozen before acquisition;
their SHA256 is
`b6e52e72f2a3fc5307e5754ae84130efbfdcfb9e34f05f62cacb9ef06416fdb1`.
The official FastF1 `f1timing` schedule selected the first four completed,
uncached races after the cached British Grand Prix: Belgium (July 19), Hungary
(July 26), Netherlands (August 23) and Italy (September 6). Dates and event
selection were saved before loading laps. No coefficients, features, eligibility
rules or candidate choices changed in this extension.

| Recent event | Matched targets | Naïve MAE | Frozen ridge MAE | Delta |
|---|---:|---:|---:|---:|
| Belgium | 640 | 0.464472s | 0.459923s | −0.004549s |
| Hungary | 1,229 | 0.563044s | 0.553844s | −0.009200s |
| Netherlands | 1,059 | 0.508437s | 0.529677s | +0.021240s |
| Italy | 851 | 0.435137s | 0.423053s | −0.012085s |

Across 3,779 matched targets, event-balanced MAE was 0.492773s for naïve and
0.491624s for ridge: a 0.233% reduction. The paired event-bootstrap delta CI
was [−0.010642, +0.013630]s, and leave-one-event-out mean improvements were
unstable. This small extension weakens confidence in a current-season gain;
it does not establish robust transfer or justify promotion. These races were
historical data held aside in this cycle, not prospective predictions.

All four acquisitions and evaluations succeeded in the final run. The isolated
data directory is `data/f1/performance_20260907/live_recent_final/`; it reuses the
separate extension cache through a symlink and does not alter existing raw
weekend files. Only laps were requested, with telemetry, weather and messages
disabled. Native FastF1 duration fields were converted explicitly to seconds.
Provider warnings about timing alignment and corrected tyre stints remain
subject to the unchanged eligibility rules; no event-specific exclusions were
added. Every event passed prefix invariance. Independent verification replayed
all four saved raw streams, checked raw/model/source hashes and recomputed
predictions, event metrics and the aggregate bootstrap interval. Its record is
the neighboring `verification.json`.

Earlier attempts are preserved separately. `recent_extension/` failed to fetch
the schedule in the restricted sandbox; that failure did not establish provider
unavailability. The approved network context subsequently fetched the schedule
and Belgium in `recent_extension_approved/`, but a relative custom data path
failed during metadata serialization. `recent_extension_verified/` then hit
the existing-CSV guard; no saved data was overwritten. The final approved run
used fresh absolute output/data directories and completed normally. The model,
specification and runner hashes were unchanged across these attempts.

For a separate reproduction, custom directories must be **fresh absolute
paths inside this repository**. Network access is required. For example:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv-f1/bin/python -m pytest -q -p no:cacheprovider research/experiments/performance_20260907/live/test_experiment.py research/experiments/performance_20260907/live/test_recent_extension.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv-f1/bin/python research/experiments/performance_20260907/live/run_recent_extension.py --output-dir "$PWD/artifacts/research/performance_20260907/live/recent_extension_replication_01" --data-dir "$PWD/data/f1/performance_20260907/live_recent_replication_01"
```

This acquisition runner selects events relative to its actual execution time,
so later reruns may select different events as the official schedule and local
cache inventory change. The saved four CSVs, schedule and frozen event list
support exact replay of this extension. The final results SHA256 is
`bfad09dd5d60aadafe279bd0df649fc23a44b0852802d9803cfdd0f215acef93`.
Suggested extension commit:
`research(f1-live): test frozen ridge on recent 2026 races`.
