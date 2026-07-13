# F1 Non-Live Full Implementation Completion Plan

Created: 2026-07-13 Europe/Paris.

Status: final evidence regeneration is in progress. The checked count is only
final after the cutoff-safe Qualifying/Best-Lap artifacts, Race v4 artifact,
and cross-service suites have all been verified.

This document converts the original non-live recommendation into a
requirement-by-requirement implementation contract. A checked item requires
working code, causal data boundaries, tests that exercise the integrated path,
and chronological evidence. Passing isolated unit tests is not sufficient.

## Shared Qualifying and Best-Lap engine

- [x] One fitted latent lap-time model owns the event shift, the selected
  team/driver residual path, valid-lap hurdle, stage mixture, and uncertainty
  distribution. The frozen selector may disable the residual path when it
  fails its selection-block gate.
- [x] Best Estimated Lap and Qualifying consume the same fitted model and the
  same joint samples; Qualifying may not construct independent rank noise.
- [x] Valid-lap and Q1/Q2/Q3 outcomes alter the simulated official
  classification, rather than being merged only as diagnostics.
- [x] Pole, top-three, position, valid-lap, and stage probabilities are derived
  from coherent field simulations and explicitly calibrated or fail closed.
- [x] Rehearsal evidence includes valid, deleted-potential, compatible-sector,
  dispersion, push-lap count, progress, track status, interruption, compound,
  tyre age/freshness, speed trap, teammate/field-relative, and provenance
  fields. Deleted laps remain potential evidence only.
- [x] The entrant roster is captured before Qualifying and never recovered
  from a target-session result file.

## Best Estimated Lap

- [x] The valid-lap hurdle and Q1/Q2/Q3 probabilities are driver-conditioned,
  trained on causal features, and use nested labels.
- [x] Stage-conditional lap distributions are learned from event-block history;
  hard-coded Q1/Q2/Q3 time penalties are not the final implementation.
- [x] Huber selection is executed reproducibly on the declared selection block,
  rather than represented by a hard-coded outcome.
- [x] Interval names match their actual quantiles. P05/P90 denotes an
  asymmetric interval with 85% nominal mass; an 85% central interval would be
  P07.5/P92.5.
- [x] Quantile challenger and event-block conformal calibration are evaluated
  chronologically with complete entrant coverage.

## Qualifying Prediction

- [x] The capped Bradley-Terry challenger remains an explicit comparator, but
  the primary challenger is the shared latent/stage simulation.
- [x] Classification output is a legal full-field permutation under invalid
  laps and stage elimination.
- [x] Standard and Sprint weekends are modeled explicitly.
- [x] LambdaRank is evaluated only after its optional runtime is healthy and
  only on event-grouped chronological folds.

## Race Final Position

- [x] `post_qualifying_pre_grid` and `post_grid_pre_race` remain distinct
  products. Verified point-in-time evidence is used where available;
  retrospective inputs are labeled and block promotion.
- [x] A capture command stores publication/as-of time, revision identity,
  penalties, pit-lane starts, withdrawals, DNS, eligibility, and provenance.
- [x] Backtests use official FIA final-grid documents with verified pre-race
  publication timestamps when available and never reconstruct grids from
  post-race results. Retrospective local capture is not described as
  contemporaneous first-seen capture.
- [x] Terminal risk is a driver-conditioned discrete-time hazard with partial
  pooling for team, power unit, driver incident, and circuit effects, plus
  regularized causal weather, missed-practice, and current-weekend-stoppage
  covariates.
- [x] Terminal cause and retirement-lap distributions are evaluated separately;
  coarse provider labels remain explicitly coarse rather than fabricated.
- [x] Conditional order uses signed Qualifying surprise, long-run pace,
  compound/tyre-age pace, adjusted degradation, clean-stint length, evidence
  uncertainty, overtaking mobility, and Sprint pace where causally available.
- [x] Joint simulations model relevant shared shocks and produce FIA-style
  classification probabilities plus minimum-expected-absolute-loss assignment.

## Optional and deferred model families

- [x] Runtime diagnostics are clean for XGBoost and LightGBM, or the run records
  an explicit external dependency blocker while retaining sklearn fallback.
- [x] Grouped LambdaRank and LightGBM quantile challengers have reproducible
  configurations and chronological evidence.
- [x] Telemetry TCN remains fail closed until a timestamped, distance-normalized
  pre-Qualifying telemetry cache has enough independent events. The ingestion,
  dataset audit, and promotion gate must exist before any deep-model claim.
- [x] RL remains confined to Live Race Intelligence and is not used for passive
  Qualifying, Race-position, or Best-Lap forecasting.

## Evidence and promotion

- [x] Selection, calibration, and final audit events are disjoint.
- [x] No target-round or post-cutoff file is opened by inference construction.
  Evaluation truth is opened only after a matching event forecast artifact is
  frozen, and later events receive earlier labels sequentially.
- [x] Per-round and per-driver prediction-versus-reality artifacts are saved
  with file-level data and selected-implementation SHA-256 inventories,
  embedded configuration/protocol metadata, mode-specific model/output
  hashes, and a registered whole-artifact SHA-256.
- [x] Promotion gates from the original recommendation are enforced exactly;
  failed challengers remain shadow/diagnostic implementations.
- [ ] Research, platform, and prediction-service suites pass from their correct
  working directories, and the final branch is clean and synchronized.

Suggested commit name: `docs(f1): define full non-live implementation contract`
