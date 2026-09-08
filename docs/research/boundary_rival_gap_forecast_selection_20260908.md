# Rival-gap forecast screen: small improvement, advancement failed

The fixed gap-content candidate reduced 2023 event-balanced MAE from **0.512080723s to 0.510266933s**, a **0.35420%** improvement over the original-feature HGB. Its improvement over the quality/category control was **0.19869%**. Both missed the predeclared 1% selection threshold, and the event-bootstrap confidence interval against the control crossed zero. **The advancement decision is false.** No later-season fitting, transfer evaluation or global model replacement followed this result.

The first independent historical replay passed without changing sources or refitting. This is a reproducible retrospective result on previously exposed seasons, not a new substantial gain or evidence of prospective live performance.

## Fixed comparison and complete population

Exactly two new HGB models were fitted on all **18,363 matched 2022 issuances** across 22 races. Each keeps the original 80 features and appends 45 gap coordinates. The candidate adds 12 content coordinates and 33 quality/category coordinates; the control replaces all 12 content coordinates with zero, including missing values, and retains the same quality/category inputs. The control therefore contains structural gap information and is stronger than a pure missingness indicator.

Content covers race rank, numeric leader and preceding-rank gaps, lap deficit, threshold indicators, and recent interval changes/trend/range. Histories use only explicit numeric updates before each cutoff; rank changes, invalidating updates and regressed timestamps reset the applicable history. There is no interpolation or inference about the physically nearest car. The original model objective, event weights `N/(E*n_event)`, 150 iterations, 15 leaves, learning rate 0.06, minimum leaf size 80, L2=10, residual target clipping at ±5 seconds and prediction correction clipping at ±3 seconds are unchanged.

The saved baseline was fitted on **2022 only** and was reused exactly. It has the same configuration as the global predictor; it is not the deployed parameter asset, whose fitting years include 2023. Testing that deployed asset on its own training rows would be invalid.

All **39,220** original 2022–2023 issuances were retained at both latency settings. Selection retained **20,432** 2023 forecasts at each lag and scored the same **20,007 matched** targets across all 22 races; **425 unmatched** forecasts remained recorded with unavailable outcomes. The primary lag is 2 seconds, and the fixed 0-second sensitivity uses the same fitted coefficients. Forecast times and next-eligible-lap targets were preserved.

| Year | Original issuances per lag | Supported at 2s | Supported at 0s |
|---|---:|---:|---:|
| 2022 | 18,788 | 17,673 | 17,687 |
| 2023 | 20,432 | 19,202 | 19,169 |

The 2-second selection population contains 1,230 unsupported forecasts, including 1,181 matched rows. Both new models use the exact baseline forecast on those rows. All unsupported training rows also remain in the training population. Coverage is a source-readiness measurement, not an accuracy result.

## All fixed scores

Lower MAE is better. Event MAE averages the 22 race MAEs equally; row MAE averages all matched forecast errors. Units are seconds.

| Model | 2s event MAE | 2s row MAE | 0s event MAE | 0s row MAE |
|---|---:|---:|---:|---:|
| `base_hgb` | 0.512080723 | 0.532032646 | 0.512080723 | 0.532032646 |
| `gap_hgb` | 0.510266933 | 0.530047307 | 0.510826869 | 0.530512265 |
| `quality_hgb` | 0.511282808 | 0.531295516 | 0.511452137 | 0.531425687 |

Each paired delta below is candidate minus reference; negative values indicate improvement. Confidence intervals are 95% percentile intervals from 20,000 event resamples or 20,000 circular blocks of three consecutive events, seed 20260907. Delta, interval and leave-one-event-out columns are in milliseconds.

| Lag | Reference | Delta ms | Relative reduction | Event CI ms | Block3 CI ms | Worst LOO delta ms | Events improved |
|---|---|---:|---:|---|---|---:|---:|
| 2s | `base_hgb` | -1.813790 | 0.35420% | [-3.052652, -0.640517] | [-3.017411, -0.684042] | -1.491460 | 14/22 |
| 2s | `quality_hgb` | -1.015875 | 0.19869% | [-2.232261, +0.040574] | [-2.151079, -0.150504] | -0.571933 | 15/22 |
| 0s | `base_hgb` | -1.253854 | 0.24485% | [-2.709756, +0.103882] | [-2.559237, -0.041923] | -0.839596 | 15/22 |
| 0s | `quality_hgb` | -0.625268 | 0.12225% | [-1.967046, +0.529906] | [-1.849749, +0.347019] | -0.131192 | 13/22 |

Against the baseline, both primary uncertainty intervals exclude zero and all leave-one-event-out deltas remain negative. Against the stronger control, the event interval reaches **+0.040574ms**. Five of eight primary checks pass; the two minimum-gain checks and that control event-CI check fail. Both fixed zero-lag point-gain checks pass, although the zero-lag event intervals cross zero. The complete advancement rule therefore fails.

The evidence supports a small retrospective reduction in error under the fixed candidate. It does not establish a sufficiently large or transferable incremental contribution from numeric gaps and their dynamics. The exposed 2023 sample was not used to relax thresholds, select another latency, retune features or fit a replacement.

## Independent verification and chronology

The source lock bound 44 files and 150 input records before historical feature construction. All 44 races were processed from already available raw TimingData bodies. Features closed before either supervised fit; both full 2023 forecast ledgers closed before this runner attached the external selection labels. Earlier experiments had already exposed these seasons and labels.

| Closure | UTC on 2026-09-08 |
|---|---|
| `design_lock.json` | 07:33:03.477809+00:00 |
| `data_lock.json` | 07:34:52.716806+00:00 |
| `fit_lock.json` | 07:35:28.998685+00:00 |
| `selection_issuance_lock.json` | 07:36:20.897808+00:00 |
| `selection_lock.json` | 07:36:34.090661+00:00 |

The independent verifier checked **294 hash bindings**, replayed all **122,592 saved HGB forecasts** across three models and two lags, and confirmed exact baseline reuse and fallback values. It independently recomputed all event/row scores, four paired comparisons and every advancement decision without fitting models. A separate Decimal implementation reconstructed **24 predeclared raw-prefix snapshots**, including four unsupported snapshots; the maximum finite-coordinate discrepancy was **2.220e-15**. Both the saved pilot provenance and interval-history source identities matched. The fixed sample was selected from original issuance order, not from targets, support or scores.

Raw-prefix replay covers that sample; the full feature ledgers are hash-bound and the complete forecast replay is exhaustive. Parsing/support primitives are shared with the frozen pilot, while numerical history assembly is separate. The verifier does not retrain the optimizer. The execution review, inherited statistical implementation and late-failure checks are also bound.

CI was exercised in a source-only temporary copy without data or artifacts: **1,193 passed, 2 optional historical tests skipped**. The new lane contributes 150 forecast contracts and 11 independent prefix/cross-implementation tests. These are distinct from the historical saved-forecast replay.

## Global integration and scope

The verified 80-feature HGB remains the global next-lap point model: `frontier_live_hgb_l15_i150_20260907`. Its global integration is already in commit `1f2608088aea0b066ca9c646ed727a2077695bae`, and the current portable asset SHA256 is `d8bd93f1b04ecf290eb821f02397253b77248cf3c1943b05e1c540d4942e7e36`. The current change adds research and verification code, results and CI coverage. It does not promote this failed candidate.

Archived TimingData observation clocks remain proxies for prospective receipt. A future successful gap model would need a separately verified live adapter retaining field timestamps, explicit clears and rank/interval history at issuance. The current latest-state gap projection is insufficient for that feature contract. [OpenF1 documents](https://openf1.org/docs/#intervals) an approximately four-second update frequency for race intervals; this does not certify equivalence to the archived stream or authenticated live access. The [FastF1 timing implementation](https://raw.githubusercontent.com/theOehrly/Fast-F1/v3.8.3/fastf1/_api.py) also distinguishes timing streams and derived lap timing. No claimed gain depends on treating these different clocks as interchangeable.

## Reproducibility

The [fixed specification](../../research/experiments/boundary_20260908/rival_gap_forecast/specification.json), [execution README](../../research/experiments/boundary_20260908/rival_gap_forecast/README.md), [verification protocol](../../research/experiments/boundary_20260908/rival_gap_verification/protocol.json) and [compact evidence manifest](evidence/boundary_rival_gap_forecast_20260908/publication_manifest.json) describe the closed run. The compact evidence copies are byte-exact; raw streams, full JSONL ledgers, labels and pickle model assets remain local and are identified by their hashes. Replaying all forecasts requires those omitted assets and the recorded environment.

| Artifact | SHA256 |
|---|---|
| `design_lock.json` | `cf656765fad43c8c95e831f2abb3c6b4edf621b8a92b3605ed4fdd3f26d4d49b` |
| `data_lock.json` | `9d6ca6d5fd1fb7109b3ca92472934509a1629146e9d866a18bd547745af0ef74` |
| `selection.json` | `947952fd0fa2322415a7ccf934a104c424f61b8eab15d05f60a9367fc69d95ab` |
| `verification/result.json` | `38b4aefb9a5bbddb532c06aa88d9bc79824f5115ebd62cb80aa89ffb67f622bc` |

Suggested commit: `research(f1-live): publish verified rival-gap selection results`.
