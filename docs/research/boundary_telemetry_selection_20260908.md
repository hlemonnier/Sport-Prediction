# Original-issuance telemetry: closed selection result

The fixed 90-feature telemetry augmentation **failed the 2023 selection gate**.
At the primary two-second lag, it increased event-balanced mean absolute error
by **0.375108% versus the original HGB** and **0.075546% versus the matched
quality control**. No later telemetry evaluation, production promotion, or new
predictive gain follows from this experiment. The existing global model remains
unchanged by this research result.

The [frozen specification](../../research/experiments/boundary_20260908/telemetry/specification.json)
kept the original next-eligible-lap issuance, target, 80 predictors and naive
anchor. Exactly three models were fitted on the same **18,363 matched 2022
rows**, including six rows without primary-lag telemetry support. The baseline
was reconstructed using 2022 alone; it was not the deployed model trained on
2022–2023. Both augmented fits retained all rows and the original event weights.
The control used the same 170-column layout, with the 27 driving measurements
zeroed and all 63 telemetry quality indicators retained. HGB settings, training
residual clipping at ±5 seconds, and forecast correction clipping at ±3 seconds
were fixed before fitting. Zero-second sensitivity used the same coefficients.

Selection retained all **22 races**, **20,432 issuances**, **20,007 matched
outcomes**, and **425 unmatched rows**. Across both discovery years, all 39,220
issuances, 38,370 matched outcomes and 850 unmatched rows were preserved. Four
selection rows at two seconds and three at zero seconds used the exact baseline
fallback; no unsupported or unmatched issuance was removed.

| Model | Two-second event MAE (s) | Zero-second event MAE (s) |
|---|---:|---:|
| Original 80-feature HGB | 0.512080723334 | 0.512080723334 |
| Matched quality control | 0.513613563242 | 0.513604736610 |
| Full telemetry augmentation | 0.514001578289 | 0.513621877798 |

The following paired differences are **telemetry minus reference**; positive
values mean worse accuracy. Intervals resample whole races with equal event
weight, using 20,000 independent-event resamples and 20,000 circular
block-bootstrap resamples with blocks of three consecutive events, seed 20260907.

| Lag / reference | MAE difference (s) | Event 95% interval (s) | Block-3 95% interval (s) |
|---|---:|---:|---:|
| 2 s / original HGB | +0.001920855 | [+0.000020846, +0.004067020] | [+0.000047394, +0.003868632] |
| 2 s / quality control | +0.000388015 | [−0.001712387, +0.002261326] | [−0.001410013, +0.002101282] |
| 0 s / original HGB | +0.001541154 | [−0.000340389, +0.003598620] | [−0.000267347, +0.003353408] |
| 0 s / quality control | +0.000017141 | [−0.002125140, +0.001921532] | [−0.001750940, +0.001730372] |

Primary telemetry improved only 8 of 22 races against each reference. The frozen
gate required at least 1% improvement against **both** references, negative upper
interval endpoints, improvement under every leave-one-event-out comparison,
and positive zero-second sensitivity against both references. Every gate check
was false. The [selection result](evidence/boundary_telemetry_selection_20260908/selection.json)
and [decision lock](evidence/boundary_telemetry_selection_20260908/selection_lock.json)
preserve the full outcomes. This is evidence against this fixed augmentation,
not a claim that telemetry contains no useful predictive information.

The independent verification protocol was frozen before selection results. Its
first execution stopped before event replay because it requested
`fit_lock['selection_labels_read']`; the frozen runner's field was
`external_selection_labels_attached`. The
[failed receipt](evidence/boundary_telemetry_selection_20260908/verification/result.json)
is preserved. A sibling verifier changed only that key, with an exact-source-diff
regression and strict false/true/missing-field checks. Its **28 tests passed**.
The [amendment](evidence/boundary_telemetry_selection_20260908/verification/protocol_v2.json)
did not alter forecasts, samples, metrics, tolerances, gates, or experiment source.

The [successful verification](evidence/boundary_telemetry_selection_20260908/verification/result_v2.json)
checked 455 file bindings, reconstructed every original issuance and matched
target, and replayed all three saved models at both lags with **bitwise identical
forecasts**. It independently reproduced event MAE, both bootstrap procedures,
leave-one-event-out results and the failed decision. A sample fixed before
results covered 24 snapshots and 2,160 coordinates across four races; separate
packet-weighted measurement calculations agreed within 1.819 × 10⁻¹².
The verifier performed no fits. Preparation and selection each closed on their
first attempt; the preserved failure concerns the verifier's metadata schema.

This remains retrospective research on seasons already exposed in earlier
experiments. Bootstrap intervals do not correct the entire research search.
Archive-prefix timestamps are availability proxies, not certified historical
client receipt. The original historical issuance helper internally constructs
and discards matching information; external labels and the matched cache were
attached only after the relevant forecast closure. Independent raw measurement
reconstruction covered the declared sample; exhaustive 44-stream integrity is
supported by the closed validation receipts and checked hashes.

The [earlier discovery publication](boundary_telemetry_discovery_20260908.md)
contains the acquisition and pre-fit design evidence. This separate publication
adds byte-exact outcome and verification receipts, mapped by its
[hash manifest](evidence/boundary_telemetry_selection_20260908/publication_manifest.json).
Raw provider bodies, per-issuance feature/forecast tables and serialized models
remain local. Copied receipts retain their original paths and hashes; the
publication manifest maps the included copies. Full replay requires the omitted,
hash-matched local inputs and models. This compact publication is not a standalone
model-replay dataset.

Suggested commit: `research(f1-live): publish verified telemetry selection failure`.
