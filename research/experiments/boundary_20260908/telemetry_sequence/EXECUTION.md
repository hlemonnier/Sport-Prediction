# Controlled telemetry representation experiment

The [specification](specification.json) extends the preceding synthetic prototype
into a fixed empirical comparison. It was written before new historical token
construction or representation fitting. The original proposal, token encoder,
causal neural network and their tests remain hash-bound to the published bytes.

Self-supervision uses only the 22 races from 2022. The sampler chooses a race,
driver and supported packet endpoint in that order, uniformly within each level.
It preserves the physical endpoint prefix even when later packets have equal
archive clocks. Three future packet observations supply positives; 31 distinct
same-driver/race negatives must be at least 180 seconds away and exclude all
three positive identities. Every batch identity is saved before training. Both
trained encoders replay that exact schedule from identical initial parameters;
only the context permutation differs. The random control uses the initial state.

The contrastive loss adapts the predictive representation approach in
[van den Oord et al.](https://arxiv.org/html/1807.03748v2). Our restricted negative
proposal depends on driver, race and endpoint time. Consequently, we do not
interpret `log(32) - loss` as a measured mutual-information bound. The loss is a
training objective; the decisive outcome is subsequent lap-time accuracy.

All three frozen representations add 16 coordinates to the existing 170-column
input. Exactly three supervised models use the same 18,363 original matched
2022 rows. No unsupported row is removed, and unsupported forecasts retain the
exact 80-feature baseline. Existing 2023 reference forecasts are reused without
refitting. Both-lag candidate forecasts close before the new evaluation stage
loads external 2023 labels. Those labels and seasons were exposed by earlier
research; this procedure preserves execution order but does not create a fresh
holdout or erase prior selection.

Only the ordered model is eligible. It must lower event-balanced MAE by at least
1% against all five alternatives, with negative event and three-event-block
interval upper bounds, negative leave-one-event-out differences and positive
zero-lag sensitivity against each. A pass permits the separately gated later
stage. That policy is fixed now: retain the 2022 neural representations, refit
the five augmented supervised models on all 2022–2023 matched rows, and reuse
the current portable baseline without refitting it.
The ordered candidate must beat each of its five references by 10% over all
48 races in 2024–2025 and 5% over all 13 declared 2026 races, including the
specified uncertainty, year and sensitivity checks. A failed screen ends this
fixed experiment without later telemetry acquisition or parameter changes.

Execution writes separate design, corpus/schedule, encoder, embedding, fitting,
forecast and selection closures under
`artifacts/research/boundary_20260908/telemetry_sequence`. Every stage verifies
its consumed inputs and reviewed sources. Historical replay reuses the closed
parent ledgers; it does not regenerate or overwrite the preceding experiment.
One CPU thread, serial fits, a shared one-hour pretraining limit and a 3 GiB
resident-memory limit constrain this experiment without changing its population.

Suggested commit: `research(f1-live): execute controlled causal telemetry representation test`.
