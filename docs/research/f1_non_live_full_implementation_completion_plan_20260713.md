# F1 Non-Live Full Implementation Completion Plan

Created: 2026-07-13 Europe/Paris.

This document converts the original non-live recommendation into a
requirement-by-requirement implementation contract. A checked item requires
working code, causal data boundaries, tests that exercise the integrated path,
and chronological evidence. Passing isolated unit tests is not sufficient.

## Shared Qualifying and Best-Lap engine

- [ ] One fitted latent lap-time model owns the event shift, team/driver
  shrinkage, regularized residual, valid-lap hurdle, stage mixture, and
  uncertainty distribution.
- [ ] Best Estimated Lap and Qualifying consume the same fitted model and the
  same joint samples; Qualifying may not construct independent rank noise.
- [ ] Valid-lap and Q1/Q2/Q3 outcomes alter the simulated official
  classification, rather than being merged only as diagnostics.
- [ ] Pole, top-three, position, valid-lap, and stage probabilities are derived
  from coherent field simulations and explicitly calibrated or fail closed.
- [ ] Rehearsal evidence includes valid, deleted-potential, compatible-sector,
  dispersion, push-lap count, progress, track status, interruption, compound,
  tyre age/freshness, speed trap, teammate/field-relative, and provenance
  fields. Deleted laps remain potential evidence only.
- [ ] The entrant roster is captured before Qualifying and never recovered
  from a target-session result file.

## Best Estimated Lap

- [ ] The valid-lap hurdle and Q1/Q2/Q3 probabilities are driver-conditioned,
  trained on causal features, and use nested labels.
- [ ] Stage-conditional lap distributions are learned from event-block history;
  hard-coded Q1/Q2/Q3 time penalties are not the final implementation.
- [ ] Huber selection is executed reproducibly on the declared selection block,
  rather than represented by a hard-coded outcome.
- [ ] Interval names match their actual quantiles. P05/P90 denotes an
  asymmetric interval with 85% nominal mass; an 85% central interval would be
  P07.5/P92.5.
- [ ] Quantile challenger and event-block conformal calibration are evaluated
  chronologically with complete entrant coverage.

## Qualifying Prediction

- [ ] The capped Bradley-Terry challenger remains an explicit comparator, but
  the primary challenger is the shared latent/stage simulation.
- [ ] Classification output is a legal full-field permutation under invalid
  laps and stage elimination.
- [ ] Standard and Sprint weekends are modeled explicitly.
- [ ] LambdaRank is evaluated only after its optional runtime is healthy and
  only on event-grouped chronological folds.

## Race Final Position

- [ ] `post_qualifying_pre_grid` and `post_grid_pre_race` remain distinct
  products with immutable first-seen evidence and fail-closed semantics.
- [ ] A capture command stores publication/as-of time, revision identity,
  penalties, pit-lane starts, withdrawals, DNS, eligibility, and provenance.
- [ ] Backtests use real first-seen final-grid snapshots when available and
  never reconstruct them from post-race results.
- [ ] Terminal risk is a driver-conditioned discrete-time hazard with partial
  pooling for team, power unit, driver incident, circuit, weather, missed
  practice, and current-weekend stoppages.
- [ ] Terminal cause and retirement-lap distributions are evaluated separately;
  coarse provider labels remain explicitly coarse rather than fabricated.
- [ ] Conditional order uses signed Qualifying surprise, long-run pace,
  compound/tyre-age pace, adjusted degradation, clean-stint length, evidence
  uncertainty, overtaking mobility, and Sprint pace where causally available.
- [ ] Joint simulations model relevant shared shocks and produce FIA-style
  classification probabilities plus minimum-expected-absolute-loss assignment.

## Optional and deferred model families

- [ ] Runtime diagnostics are clean for XGBoost and LightGBM, or the run records
  an explicit external dependency blocker while retaining sklearn fallback.
- [ ] Grouped LambdaRank and LightGBM quantile challengers have reproducible
  configurations and chronological evidence.
- [ ] Telemetry TCN remains fail closed until a timestamped, distance-normalized
  pre-Qualifying telemetry cache has enough independent events. The ingestion,
  dataset audit, and promotion gate must exist before any deep-model claim.
- [ ] RL remains confined to Live Race Intelligence and is not used for passive
  Qualifying, Race-position, or Best-Lap forecasting.

## Evidence and promotion

- [ ] Selection, calibration, and final audit events are disjoint.
- [ ] No target-round or post-cutoff file is read by inference construction.
- [ ] Per-round and per-driver prediction-versus-reality artifacts are saved
  with data, implementation, configuration, and protocol hashes.
- [ ] Promotion gates from the original recommendation are enforced exactly;
  failed challengers remain shadow/diagnostic implementations.
- [ ] Research, platform, and prediction-service suites pass from their correct
  working directories, and the final branch is clean and synchronized.

Suggested commit name: `docs(f1): define full non-live implementation contract`
