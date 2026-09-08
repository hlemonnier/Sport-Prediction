# Verified full-history football default

Suggested commit: `feat(football): activate the verified full-history default`.

The automatic Dixon default now fits a separate equal-weight goal model on all
history admitted at the fixture cutoff and uses identity calibration. The previous
default forecast used the older assessment fit prefix and its automatic
calibrator. Historical assessment remains on its original disjoint populations.

This is a verified improvement to the operational default, not a new best research
model. Both training scope and calibration policy change together; the comparison
does not identify their separate causal contributions. F1's previously integrated
frontier model is unchanged.

## Fixed historical result

The comparison contains all 2,280 original fixtures: 380 per country-season across
England, Italy and Spain in 2024/25 and 2025/26. It retains the same 36 fit blocks,
1,095-local-day admitted windows, source snapshots, cutoff convention, fixture
order and resumed Fiorentina–Inter identity. No rows were selected or removed.

| Metric | Previous default | Full-history default | Change |
|---|---:|---:|---:|
| 1X2 mean log loss | 1.028381677475 | 0.993084869788 | 3.4323% lower |
| Joint-score mean log loss | 2.898580601234 | 2.840058212851 | 2.0190% lower |
| Class-sum Brier score | 0.615575562572 | 0.592925922309 | 0.022649640263 lower |
| Top-1 accuracy | 49.2105% | 51.4474% | 51 more correct outcomes |
| Top-label ECE, 10 bins | 0.028784206345 | 0.011320817222 | 0.017463389123 lower |

All ten fixed numerical gates passed. Every country improves in 1X2 and joint-score
log loss; every country-season improves in 1X2 log loss and class-sum Brier score.

| Country | Previous 1X2 loss | New 1X2 loss | Previous joint loss | New joint loss |
|---|---:|---:|---:|---:|
| England | 1.0545700330 | 1.0144268242 | 3.0034423212 | 2.9421457391 |
| Italy | 1.0239160111 | 0.9860296422 | 2.8349401060 | 2.7807600941 |
| Spain | 1.0066589884 | 0.9787981430 | 2.8573593765 | 2.7972688053 |

Paired 28-day calendar-block resampling, stratified by country-season with 4,000
draws and seed 20260908, gives these 95% percentile intervals for candidate minus
reference loss:

- 1X2: **−0.035296807687**, interval **[−0.045945337271, −0.025066141261]**.
- Joint score: **−0.058522388383**, interval **[−0.074585203875, −0.041341519280]**.

Both losses use the same sampled blocks. The published evaluation includes the
predeclared 1-day and 7-day sensitivity intervals and all country-season metrics.
These dates and the old 1X2 comparison were already exposed. This is retrospective
evidence, not prospective validation or evidence of a bookmaker-relative edge.

## Complete forecast and independent verification

The actual previous caller reconstructed exactly 36 prefix goal models and made
36 original automatic calibration-policy calls. All 36 complete full-history
states were restored from their pinned artifacts. There were no candidate goal
fits, learned candidate calibrators, duplicate parity fits or new model searches.

All 2,280 reference and candidate probability vectors reproduced the old vectors.
Both policies' complete emitted finite score matrices closed before external
scoring labels attached. Support depends only on goal rates, rho and selected
1X2 probabilities; no observed goal count expands a scoring grid. No row had zero
or out-of-support observed mass. Joint-score improvement was measured from the
distinct saved kernels, not inferred from 1X2 improvement.

Independent arithmetic reconstructed raw CSV membership, all assessment partitions,
goal rates, calibrated probabilities, support, every matrix, expectations, modal
scores, rendered caller outputs, external targets, metrics and paired gates.
Maximum complete-output replay discrepancy was **3.55e-15** over 4,560 forecasts.
The replay performed zero fits.

The first verifier stopped on a metadata schema omission: its expected block
lineage lacked `goal_state_sha256`, `source_artifact` and `source_artifact_sha256`.
The original failed attempt remains immutable. A separately frozen corrective
replay independently derives exactly those three fields from the pinned source
artifacts, preserves all original numerical checks, and permits only the exact
recorded bookkeeping failure. That replay passed. Forecasts, labels, scores,
parameters, populations and acceptance thresholds were unchanged.

## Global integration

`packages.football.mrp.prediction.run_prediction` applies the new policy only for
the automatic Dixon default without temporal decay. The legacy CLI and API
command builder reach that same caller. Explicit calibration, decay, GBDT and
hybrid settings retain their previous behavior.

`fixture_model` records full-history model state and fit lineage;
`assessment_model` retains the original protocol and assessment fit. Existing
held-out metrics continue to describe the assessment model. `run_assessment_prediction`
preserves the old caller for reproducible comparisons. Scoreline, expected goals
and 1X2 fields continue to come from one coherent emitted distribution.

Synthetic integration tests exercise fresh fitting, default routing through the
real CLI/API argument path, explicit settings and unavailable-future-result
poisoning. A separate complete-population integration replay checks the global
default's actual admission and fit inputs and all saved fixture outputs using
verified states at the fit seams, with zero additional historical optimizer fits.
That replay passed for all 2,280 fixtures with **zero matrix discrepancy**. Its
first adapter attempt encountered an omitted legacy diagnostic ID list; the
successful attempt restores that list from verified lineage without changing
learned parameters. Both attempt receipts are retained in the evidence package.

The initial local boundary-contract suite passed **2,007 tests**, with two optional
historical-asset tests skipped. The final complete tracked-source copy, without
local data or artifacts, passed **2,008 tests** with five skips. It includes all
**16** default/API/CLI and league-metadata integration cases. F1's existing
integration separately passed all **55 tests**. Independent
historical replay also passed again after the canonical caller changed, using
the immutable original source snapshot.

The first remote research CI run exposed a case-sensitive legacy CLI path and an
optional raw-provider test that lacked an availability guard. Both were corrected.
The default caller also preserves the provider's support for omitted or mixed-case
league metadata: only copied inputs to the fixture model receive the selected
request league; historical assessment records remain unchanged. A final full
historical integration replay still has zero matrix discrepancy.

## Reproduction and exact evidence

The compact evidence package is
[evidence/football_default_deployment_20260908](evidence/football_default_deployment_20260908/manifest.json).
Its manifest lists byte hashes for published receipts and local larger artifacts.
Raw CSVs and complete historical forecast ledgers remain local and hash-bound.

- [Original fixed protocol](../../research/experiments/boundary_20260908/football_deployment_policy/specification.json).
- [Execution instructions](../../research/experiments/boundary_20260908/football_deployment_policy/EXECUTION.md).
- [Narrow corrective replay](../../research/experiments/boundary_20260908/football_deployment_replay_v2/README.md).
- [Global caller replay](../../research/experiments/boundary_20260908/football_deployment_integration/run.py).

Run synthetic contracts with `PYTHONPATH=.` and a single BLAS/OpenMP thread:

```sh
python -m pytest -q --import-mode=importlib -p no:cacheprovider \
  packages/football/tests \
  research/experiments/boundary_20260908/football_deployment_policy \
  research/experiments/boundary_20260908/football_deployment_replay_v2
```

With the closed local historical artifacts, the independent corrected replay is
available through its read-only `verify()` function in a fresh Python interpreter.
The original execution directory is terminal and must not be rerun or overwritten.
