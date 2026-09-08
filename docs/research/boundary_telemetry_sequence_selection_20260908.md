# Causal telemetry sequence experiment: selection failed

The ordered CPC sequence candidate did not improve the original-feature HGB in the 2023 selection screen. On the same 20,007 matched forecasts across all 22 races in 2023, primary event-balanced MAE increased from **0.512080723s to 0.514094207s**: **+0.002013484s (+0.39320% error)**. It also trailed every other primary comparator. No later-season sequence data were acquired, no transfer evaluation was run, and no model was promoted.

This is a closed, verified retrospective result. The historical seasons were already exposed by previous research. The experiment preserves the original next-eligible-lap issuance horizon; it does not substitute a later sector checkpoint.

## Complete common population and mechanism

All 39,220 original 2022–2023 issuances were embedded at both receipt lags. The supervised fit used all 18,363 matched 2022 issuances. Selection retained all **20,432** original 2023 issuances at each lag, including **425 unmatched** forecasts; MAE uses only the **20,007 matched** targets. The primary 2-second lag had four unsupported issuances (all matched), and the 0-second sensitivity had three. Every unsupported new forecast exactly preserved the original baseline.

The new information was a causal ordered sequence of at most 128 reduced driver packets within 180 seconds, with 34 numeric coordinates per packet. The 2022 corpus contained 4,423,260 reduced packets across 440 event/driver streams; every stream had an eligible endpoint. Its fixed sampler chose an event uniformly, then an eligible driver uniformly, then an eligible endpoint uniformly. Every contrastive row used future packet offsets 4/16/32 and 31 distinct same-driver/race negatives at least 180 seconds away, excluding all three positive IDs. Sampling used raw packets without lap targets or stint boundaries.

Two 15,048-parameter causal encoders each trained for exactly 800 steps of 32 examples, using the same initialization and precomputed schedule. The ordered encoder retained packet order; its separately trained control shuffled older packet-content bundles while preserving timing slots and the newest packet. A third encoder retained the untrained initial weights. Each produced 16 coordinates appended to the same 80 original plus 90 telemetry features. Three new HGB regressors were fitted on the identical complete matched 2022 population. The three older references were reused bit for bit.

## All six fixed forecasts

Lower MAE is better. Event MAE averages the 22 race MAEs equally; row MAE weights every matched forecast equally. Units are seconds. Every table entry uses the same 20,007 matched rows.

| Model | 2s event MAE | 2s row MAE | 0s event MAE | 0s row MAE |
|---|---:|---:|---:|---:|
| `base_hgb` | 0.512080723 | 0.532032646 | 0.512080723 | 0.532032646 |
| `telemetry_hgb` | 0.514001578 | 0.534058984 | 0.513621878 | 0.533610761 |
| `quality_hgb` | 0.513613563 | 0.533641967 | 0.513604737 | 0.533611632 |
| `ordered_hgb` | 0.514094207 | 0.534231321 | 0.513430287 | 0.533538292 |
| `permuted_hgb` | 0.513561575 | 0.533458327 | 0.513147767 | 0.532958068 |
| `random_hgb` | 0.513051512 | 0.532862233 | 0.512629483 | 0.532478230 |

`base_hgb` uses the current original-feature model's configuration, fitted only on 2022 for this selection comparison. Its saved forecasts come from the preceding closed telemetry experiment. It is not the deployed parameter asset, which was trained on 2022–2023 and would be inappropriate for testing on its own 2023 training rows. `telemetry_hgb` adds the 90 summaries; `quality_hgb` removes their content coordinates but retains telemetry-quality measurements. The remaining three use ordered, order-destroyed and untrained-random sequence representations respectively.

## Paired primary comparisons

Each delta is **ordered candidate minus reference**, so positive values mean deterioration. Relative reduction is positive only for improvement. Delta, confidence interval and leave-one-event-out columns are in **milliseconds**. Intervals are 95% percentile intervals from 20,000 event resamples or 20,000 circular blocks of three consecutive events, with fixed seed 20260907. They describe this exposed historical sample.

| Reference | Delta ms | Relative reduction | Event CI ms | Block3 CI ms | Worst LOO delta ms | Events improved |
|---|---:|---:|---|---|---:|---:|
| `base_hgb` | +2.013484 | -0.39320% | [+0.052697, +4.356856] | [+0.055979, +4.129927] | +2.316818 | 9/22 |
| `telemetry_hgb` | +0.092629 | -0.01802% | [-0.833070, +0.973643] | [-0.591563, +0.768998] | +0.355177 | 9/22 |
| `quality_hgb` | +0.480644 | -0.09358% | [-1.456161, +2.360943] | [-1.182871, +2.191391] | +1.061919 | 10/22 |
| `permuted_hgb` | +0.532632 | -0.10371% | [-0.897758, +1.934803] | [-0.484631, +1.647013] | +0.933311 | 9/22 |
| `random_hgb` | +1.042695 | -0.20323% | [-0.035334, +2.269079] | [-0.107191, +2.302568] | +1.328808 | 8/22 |

All **20 primary checks failed**: for each of the five references, the ordered candidate failed the minimum 1% event-MAE reduction, negative event-CI upper bound, negative block-CI upper bound and negative delta under every leave-one-event-out calculation. The slightly positive lower CI bounds against the baseline indicate deterioration under the declared resampling scheme; they do not remove prior research exposure or establish prospective significance.

## Fixed 0-second latency sensitivity

The 0-second predictions reuse the same fitted coefficients; they were neither refitted nor used to select a different model. Units and signs match the preceding table.

| Reference | Delta ms | Relative reduction | Event CI ms | Block3 CI ms | Worst LOO delta ms | Positive-gain gate |
|---|---:|---:|---|---|---:|---|
| `base_hgb` | +1.349564 | -0.26355% | [-0.708328, +3.790742] | [-0.662532, +3.466842] | +1.636739 | fail |
| `telemetry_hgb` | -0.191591 | +0.03730% | [-1.461412, +0.960176] | [-1.479342, +0.891700] | +0.164267 | pass |
| `quality_hgb` | -0.174450 | +0.03397% | [-2.269774, +1.900168] | [-1.719481, +1.669918] | +0.433316 | pass |
| `permuted_hgb` | +0.282520 | -0.05506% | [-1.491971, +1.988749] | [-1.211256, +1.790478] | +0.765416 | fail |
| `random_hgb` | +0.800804 | -0.15622% | [-0.564490, +2.172189] | [-0.927249, +2.342857] | +1.190205 | fail |

Only the two older telemetry controls passed the sensitivity point-gain condition. The baseline, permuted and random comparisons failed. Therefore the combined fixed advancement decision is **false**.

## Training loss is not forecasting accuracy

| Encoder | Steps | Mean loss, first 50 steps | Mean loss, final 50 steps | Fit time seconds |
|---|---:|---:|---:|---:|
| ordered | 800 | 3.014245 | 2.623487 | 60.707 |
| permuted | 800 | 3.054538 | 2.769543 | 134.016 |

The contrastive objective improved during training, but the 2023 lap forecasts worsened. These losses measure the declared future-packet discrimination task. They are not lap-time errors, evidence of calibrated probabilities, or a substitute for the downstream comparison. The ordered representation also failed to beat its order-destroyed and random controls; this experiment does not establish predictive value from the learned temporal representation.

## Independent replay and chronology

The first historical verifier execution passed without refitting or source changes. It checked all **25,600** schedule rows, **76,800** positive identities and **793,600** negative identities, including exact PCG64 draws and physical-prefix eligibility. It verified shared initialization, the two 800-entry training traces, all original populations and unmatched retention, exact older references and fallback values, and **245,184** direct saved-HGB forecast values across both lags. It independently recomputed all ten paired comparisons, event/row metrics and every advancement check.

The verifier reconstructed the 24 predeclared raw-prefix snapshots from four fixed races, two lags and first/middle/last issuance positions. A separate NumPy forward pass checked all three encoder outputs; maximum absolute discrepancy was **4.68601707e-07**, inside the predeclared tolerance. The verifier bound **2,615** consumed files. Its author also authored tokens/corpus; the NumPy forward and reused independent statistical routines have separate authorship.

The initial freeze correctly failed when the final failed-attempt guard was added during testing. It created no design lock and performed no historical construction or fitting. Both prior source snapshots, the exact amendment, failed receipt and earlier review remain preserved. Scientific settings did not change. The successful run used the separate `execution_v2` directory.

All times below are UTC on 2026-09-08. The protocol was explicitly hash-bound by the execution review at **06:17:50.104478**, before the design lock at **06:18:45.459710**, corpus start at **06:19:06.550192** and pretraining start at **06:23:48.769165**. The complete verifier implementation was reviewed at **06:30:25.315333**, after SSL but before selection. The verification protocol and full implementation review are distinct milestones.

| Successful execution closure | UTC |
|---|---|
| design | 06:18:45.459710+00:00 |
| corpus | 06:23:23.544035+00:00 |
| pretraining | 06:27:06.699123+00:00 |
| embeddings | 06:50:10.632711+00:00 |
| supervised_fit | 06:51:17.870486+00:00 |
| selection_forecasts | 06:52:25.355799+00:00 |
| selection | 06:52:29.883894+00:00 |

The local source-only test receipt records **243 passing tests**, with no historical data or artifacts present. The publication helper adds **9 byte-parity/refusal tests**. These are test results for software behavior, not additional predictive experiments or proof of a remote CI execution.

## Evidence and practical limits

- [Exact selection results](evidence/boundary_telemetry_sequence_20260908/execution_v2/selection.json).
- [Independent verification](evidence/boundary_telemetry_sequence_20260908/verification/result.json).
- [Publication manifest](evidence/boundary_telemetry_sequence_20260908/publication_manifest.json) and [byte-copy checks](evidence/boundary_telemetry_sequence_20260908/publication_check.json).
- [Preserved source amendment](evidence/boundary_telemetry_sequence_20260908/prior_freeze/freeze_source_before_guard/amendment.json) and [local source-only CI receipt](evidence/boundary_telemetry_sequence_20260908/reviews/ci_source_only.json).

The compact package contains **29 byte-identical evidence copies**, totaling **2,851,733 bytes**. Raw streams, binary token/schedule arrays, embedding NPZ files, serialized models and per-issuance feature/forecast/label ledgers remain local; the included closures retain their hashes. Full numerical replay requires these omitted hash-matched artifacts.

The neural replay is a predeclared sample, not a full raw-prefix replay of every embedding. Packet sequence IDs are reconstructed and reported; they cannot be compared with a sequence-ID column that was never stored in the NPZ. Recorded optimizer traces are checked but training is not independently reproduced. Archive timestamps remain availability proxies, and historical exposure prevents interpreting 2023 as a pristine holdout. No later acquisition or transfer test was run after the selection gate failed.

| Evidence | SHA256 |
|---|---|
| Successful design | `0081cb2e946e590e76be704238f98084ee8c8a7cbe47ac0f27a75c2617a1e8a7` |
| Selection | `326a7aa20ce7a41284a80df36f96ea2531399b64f7a7b5f23ae81f733d381c00` |
| Independent verification | `b5a0202bb46da8c8da23ad78ea1c36ad706ade3e7654e920e3ff14c12e5ea2a1` |
| Publication manifest | `e0b078b46eaaedaa4692d128bbc69be3e114bfaf19a8be59b579a415df4d9a19` |

Suggested commit: `research(f1-live): publish verified causal sequence selection failure`.
