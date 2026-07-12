# F1 Non-Live Mode Improvement Implementation Plan

Created: 2026-07-13 Europe/Paris.

This document is the execution contract for improving the three non-live F1
modes. It intentionally challenges the current implementation instead of
treating existing code as a design constraint. The retained baseline remains
available until a challenger passes every declared gate.

## Scope and target products

The work covers exactly these prediction products:

1. **Qualifying Prediction** — the official Grand Prix Qualifying
   classification, as a joint full-field ranking, using only pre-Qualifying
   evidence.
2. **Race Final Position** — the official terminal Race classification and
   terminal status, at an explicitly named pre-Race information horizon.
3. **Best Estimated Lap Time** — each entrant's best valid achievable Grand
   Prix Qualifying lap, with calibrated uncertainty and valid-lap probability.

Live Race Intelligence is intentionally excluded from this workstream. Its
counterfactual strategy policy and reinforcement-learning contract remain a
separate problem.

## Non-negotiable information and validation rules

- Every feature must have an `as_of` boundary earlier than the prediction
  horizon. No post-session target, result, corrected grid, or retrospectively
  enriched field may enter inference.
- Evaluation splits are complete event blocks in chronological order. Driver
  rows are never randomly split across train, calibration, and evaluation.
- The final outputs for Qualifying and Race are valid permutations, not twenty
  independent rounded regressions.
- Older seasons may supply weak, regulation-aware priors for invariant effects;
  their absolute car pace is never pooled into the current season unchanged.
- Model selection, calibration, and final audit use disjoint events.
- Every run records data, feature, implementation, configuration, and protocol
  hashes. A metric without immutable provenance is diagnostic only.
- A challenger that fails promotion remains useful negative evidence but does
  not replace the simple causal baseline.

The historical protocol is:

- 2022-2024: feature development and weak-prior estimation;
- 2025: transfer and hyperparameter validation;
- already inspected 2026 rounds 1-9: retrospective audit only;
- future 2026 rounds: locked prospective evaluation.

## Shared latent qualifying-lap engine

Qualifying Prediction and Best Estimated Lap Time must be two projections of
one latent pace process rather than two unrelated models. For driver `i` and
event `e`:

\[
T_{i,e}=B^*_{i,e}+\Delta_{session,e}+u_{team,e}+u_{driver,e}
+f(x_{i,e})+\epsilon_{i,e}.
\]

`B*` is a quality-aware pre-Qualifying anchor, `Delta_session` is the expected
rehearsal-to-Qualifying shift, the team and driver effects are partially
pooled, and `f(x)` is a regularized residual model. The shared engine emits
valid-lap probability and lap-time samples. Best Estimated Lap summarizes each
driver's samples; Qualifying jointly samples all drivers and sorts each field.

### Quality-aware rehearsal evidence

For every entrant, build the following causal features and provenance:

- official classified rehearsal best;
- best valid clean lap;
- best deleted or track-limits lap, flagged as potential evidence only;
- fastest compatible-sector reconstruction, never presented as a valid lap;
- valid-best minus potential gap;
- best-two and best-three lap spread, push-lap count, and lap recency;
- session progress and estimated track evolution at lap timestamp;
- compound, tyre age, fresh-tyre flag, and tyre-evidence completeness;
- track status, flags, obstruction/traffic indicators, and lap accuracy;
- speed-trap evidence where it exists before the cutoff;
- teammate-relative and field-relative pace;
- source, source age, coverage, imputation, and anchor-quality flags.

The entrant fallback order is:

1. official classified rehearsal best;
2. best valid clean rehearsal lap;
3. deleted/potential evidence with an explicit penalty and wider uncertainty;
4. an earlier practice or Sprint source;
5. a teammate/team partial-pooled prior with the widest uncertainty.

Deleted laps and compatible-sector reconstructions may improve the estimate of
potential, but must never be relabelled as valid target observations.

## Mode 1 — Qualifying Prediction

### Statistical contract

The first challenger is a small regularized Bradley-Terry pairwise residual
ranker. It learns which driver should finish ahead conditional on the shared
latent pace features and the rehearsal baseline. Pairwise probabilities are
aggregated into full-field scores and sorted deterministically. The first
version caps movement around the causal rehearsal ranking to three positions;
the cap is an explicit hyperparameter validated on event blocks.

Classification survival is represented separately:

\[
P(valid\ classification)=P(valid\ lap)P(Q2\mid Q1)P(Q3\mid Q2).
\]

The implementation therefore emits:

- probability of at least one valid classified lap;
- Q2 and Q3 advancement probabilities;
- a joint point ranking from latent samples or pairwise scores;
- optional position marginals only after calibration passes its own gate.

Standard and Sprint weekends receive explicit rehearsal-source interactions.
A selector is frozen for a minimum evidence window; it may not switch models
round by round on tiny validation samples.

### Promotion gates

- mean absolute position error improves by at least 0.15-0.20 positions;
- mean Kendall tau-b improves by at least 0.02;
- pole, top-three, and top-ten metrics do not regress materially;
- standard and Sprint-weekend strata both improve;
- improvement survives removal of the four largest classification shocks;
- event-bootstrap upper confidence bound for challenger-minus-baseline MAE is
  below zero;
- no single event supplies more than 50% of total improvement.

## Mode 2 — Race Final Position

### Snapshot contract

The current post-Qualifying/pre-grid proxy remains a separate named product.
Add an immutable `post_grid_pre_race` snapshot containing only information
first known before the Race:

- official grid revision and publication/as-of timestamp;
- applied grid penalties and reasons;
- withdrawals and DNS evidence;
- pit-lane starts;
- starter/eligibility status;
- complete evidence provenance and unresolved-state flags.

Historical final results must not be used to reconstruct a supposedly
pre-Race grid. If first-seen evidence is absent, the final-grid product fails
closed while the explicitly labelled Qualifying proxy may still run.

### Survival-aware classification model

Factor terminal status from conditional running order:

\[
P(\pi,z,\tau\mid X,G)=P(z,\tau\mid X,G)
P(\pi_{running}\mid z,\tau,X,G),
\]

where `z` is reason-coded terminal status and `tau` is retirement time/lap.

The first terminal model is a regularized discrete-time or beta-binomial
hazard with partial pooling and recency weighting. It distinguishes at least:

- DNS/withdrawal;
- mechanical or power-unit retirement;
- collision/incident retirement;
- non-classified finish;
- classified finish.

Features include team/PU reliability history, driver incident history,
current-weekend stoppage or missed-practice evidence, circuit, weather, and
starting-grid state. It reports Brier score, log loss, calibration, and
reason-specific recall rather than relying on AUC alone.

The conditional-order model is a regularized Bradley-Terry or Plackett-Luce
ranker with:

- strong signed grid prior;
- dynamic team and driver strength;
- signed Qualifying surprise rather than its absolute value;
- teammate-relative long-run pace;
- compound/tyre-age pace and degradation adjusted for fuel and track evolution;
- longest clean stint, evidence share, and uncertainty;
- circuit mobility/overtaking prior and Sprint race pace where legal.

Monte Carlo samples terminal status/time and then conditional running order.
The point classification is reconciled with an assignment solver minimizing
expected absolute position loss under FIA classification constraints.

### Promotion gates

- complete-field MAE improves by at least 5%;
- status Brier score and log loss beat the causal rolling-hazard baseline;
- Kendall tau-b does not worsen by more than 0.02;
- entrant coverage is 100% and every emitted classification is legal;
- event-bootstrap upper bound for MAE difference is below zero and posterior
  probability of improvement is at least 0.95;
- leave-one-event-out results are directionally stable and no event contributes
  more than 50% of total gain.

## Mode 3 — Best Estimated Lap Time

### Statistical contract

Fix scalar cleaning before fitting any challenger: a single finite observed
value is valid evidence, not an empty sample. The mode then uses the shared
quality-aware anchor and a hurdle/stage mixture:

\[
F_i(y)=P(no\ valid\ lap)+\sum_kP(K_i=k\mid X)F_i(y\mid K_i=k,X),
\]

where `K` is the deepest reached Qualifying stage. The forecast emits:

- probability of no valid classified Qualifying lap;
- stage probabilities;
- conditional `p05`, `p50`, and `p90` achievable valid-lap time;
- unconditional availability/provenance status;
- fastest-driver and top-three probabilities from joint field samples.

Split the target process into event-level fastest Qualifying time and each
driver's gap to that time. The first learned residual model is robust and
hierarchical (Huber/Student-t behavior with team/driver shrinkage). A quantile
challenger is allowed only through an explicit model allowlist. Prediction
intervals use rolling event-block conformal calibration.

Deep temporal models remain blocked until the repository contains a stable
telemetry cache, categorical embeddings, and multiple representative push laps
per entrant. A generic raw-lap Ridge or static team correction is not accepted
as sufficient feature engineering.

### Promotion gates

- observed-target MAE improves by at least 5%;
- entrant output coverage is 100%, with missing targets scored separately;
- fastest-driver and top-three identification do not regress materially;
- empirical interval coverage is within five percentage points of the nominal
  85%;
- interval width does not inflate by more than 10%;
- leave-one-event-out results are stable and no event contributes more than 50%
  of total gain.

## Model/tooling policy

The initial production-capable challengers deliberately use small,
regularized, inspectable models: pairwise logistic ranking, robust Huber
residuals, partial-pooled hazards, Monte Carlo, and assignment. XGBoost
LambdaRank or LightGBM LambdaRank may be added behind an optional dependency
boundary only after the event-block baseline is reproducible. A runtime doctor
must report missing OpenMP support clearly and the sklearn fallback must remain
functional.

Neural networks are not the default for the available event count. RL is not a
pre-Race prediction method and remains confined to Live Race Intelligence.

## Required artifacts and tests

Each mode must add:

- unit tests for feature causality, fallback order, finite-value handling, and
  legal permutations;
- synthetic tests for deleted-lap potential, missing entrants, DNS, pit-lane
  starts, ties, and all-terminal edge cases;
- chronological event-block backtests against the retained baseline;
- per-round prediction-versus-reality tables;
- event-bootstrap intervals, leave-one-event-out stability, and gain
  concentration diagnostics;
- an immutable run manifest and a plain-language decision record.

## Execution checklist

- [x] Push the clean four-mode baseline and retained evidence.
- [x] Freeze this implementation and promotion contract in the repository.
- [ ] Build shared quality-aware rehearsal feature extraction.
- [ ] Build shared latent lap sampler and stage/valid-lap interfaces.
- [ ] Add the Qualifying pairwise residual challenger and joint outputs.
- [ ] Add Best-Lap robust residual, stage mixture, and conformal calibration.
- [ ] Add immutable `post_grid_pre_race` evidence snapshots.
- [ ] Add Race terminal hazard and reason-coded status model.
- [ ] Add Race conditional-order ranker, Monte Carlo, and assignment output.
- [ ] Add optional LTR runtime diagnostics and deterministic fallback.
- [ ] Run complete event-block backtests and negative controls.
- [ ] Update per-round prediction-versus-reality evidence and audit decision.
- [ ] Run all affected test suites, commit focused changes, and push.

Suggested commit name: `docs(f1): freeze non-live mode improvement plan`
