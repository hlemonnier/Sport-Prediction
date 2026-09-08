# Causal race-order gap prediction screen

This experiment tests whether race-order gap magnitudes and recent changes
improve next-eligible-lap forecasts on the original issuance horizon. It follows
the [target-free two-race pilot](../../../../docs/research/boundary_rival_gap_pilot_20260908.md)
and the [information rationale](../rival_gap_rationale.md). Pilot coverage is not
evidence of predictive utility.

The [fixed specification](specification.json) declares 12 content coordinates
and 33 measurement-quality/category coordinates. The content includes current
rank and gaps, threshold indicators, changes over at least five and fifteen
seconds, and a thirty-second trend and range. A history contains only explicit
numeric updates since the most recent observed rank change or invalidating
interval update. Equal-clock updates have one final physical-sequence value.
No interpolation, future rows, physical-nearest-car reconstruction or inferred
fuel/tyre state enters the features.

Missing content stays missing. The stronger control replaces all 12 content
coordinates, including their missing values, with zero while retaining the
same 33 quality and category coordinates. This tests the incremental value of
magnitudes and dynamics beyond those structural observations. Both models add
45 columns to the original 80, using exactly the original HGB configuration.

Exactly two new supervised fits use all 18,363 matched 2022 issuances. Their
event weights are \(N/(E n_e)\), so each of the 22 events has the same aggregate
training weight. The residual target is clipped at five seconds, and the
predicted correction at three seconds around the original naive anchor. These
are unchanged reference settings. Unsupported gap observations remain in the
training population; at prediction time they retain the base forecast exactly.

The base reference is the previously fitted **2022-only** original-feature HGB,
whose closed 2023 forecasts are reused. It is the same model configuration as
the current global predictor, not its deployed 2022–2023 parameter asset. Using
the latter on 2023 would evaluate it on its own training rows.

All 39,220 original 2022–2023 issuances are retained at both fixed latency
settings. Selection scores the same 20,007 matched 2023 targets and records all
425 unmatched forecasts without assigning them invented outcomes. The two new
models and both complete forecast ledgers close before the evaluation step
attaches external 2023 labels. Earlier research exposed these seasons already;
the execution order does not create a pristine holdout.

Only the gap-content model may advance. It must improve event-balanced MAE by
at least 1% against both references, with negative event-bootstrap and
three-event-block upper interval bounds, negative differences after leaving
out any one event, and positive improvement at zero-second latency. The
predeclared later-stage thresholds are 10% over 2024–2025 and 5% over the 13
declared 2026 races, against both references and with the additional year,
uncertainty and latency checks in the specification. A failed selection closes
this fixed experiment.

No successful historical result is claimed by this protocol. The runner must
close reviewed sources, dependencies, original inventories and input hashes
before constructing the 44-race features. Ambiguous gap observations trigger
the frozen fallback; changed input hashes fail execution. The global predictor
changes only after predictive validation and a separately verified causal live
adapter.

Suggested commit: `research(f1-live): test causal rival-gap magnitude and dynamics`.
