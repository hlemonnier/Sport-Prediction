# F1 Non-Live Challenger Results

Finalized: 2026-07-13 Europe/Paris. Evaluation cutoff: completed 2026 rounds
1-9. These are retrospective research results on the exact new challenger
paths, not production claims.

## Executive decision

| Mode | Retained baseline | New challenger result | Decision |
| --- | --- | --- | --- |
| Qualifying Prediction | Latest target-aligned valid rehearsal rank | MAE 2.1414 vs 1.8687; Kendall 0.7518 vs 0.7729 on 2026 | Reject point challenger; retain baseline and keep stage probabilities uncalibrated |
| Race Final Position | Grand Prix Qualifying order proxy at `post_qualifying_pre_grid` | MAE 3.7677 vs 3.7374; terminal Brier 0.1765 vs 0.1828 | Reject joint point challenger; retain proxy baseline and keep improved terminal model diagnostic |
| Best Estimated Lap | Valid-rehearsal shift | MAE 0.4179 s vs 0.5240 s, a 20.24% gain; 195/195 observed targets covered | Keep quality-aware challenger diagnostic because bootstrap CI still crosses zero by 0.0065 s |

No retained policy is silently replaced. The code, datasets, backtests, and
promotion gates are now in place so future rounds can accumulate genuinely
prospective evidence.

## What was implemented

### Shared Qualifying and Best-Lap evidence

`packages/f1/features/qualifying_lap.py` now creates one causal row per entrant
and keeps the following concepts separate:

- official/valid clean lap;
- deleted or track-limits potential evidence;
- compatible-sector potential;
- credible-potential checks and potential penalties;
- best-two/best-three consistency, push-lap count and progress;
- track evolution, tyre, track-status and speed-trap evidence;
- teammate/field-relative evidence;
- earlier-session and team fallback provenance;
- quality anchor versus latent potential-adjusted anchor.

The Miami failure is explicitly fixed. Alonso's valid but unrepresentative
101.311 s lap remains recorded as valid evidence, while his credible 92.490 s
deleted lap informs a 92.439 s latent anchor. Stroll's 129.082 s deleted outlier
is rejected and falls back to the teammate latent anchor instead of becoming a
129 s forecast.

### Qualifying Prediction

- event-pure Bradley-Terry/logistic pairwise training;
- equal event weighting and explicit feature allowlist;
- strong rehearsal-rank prior with a configurable ±3 movement cap;
- exact assignment to a legal full-field permutation;
- standard/Sprint source interactions;
- distinct `P(valid lap)`, `P(Q2|valid)` and `P(Q3|Q2)` models;
- joint samples and position marginals marked uncalibrated;
- frozen selector that requires matched events and explicit promotion gates.

The 2025 transfer arm showed a small MAE gain, 3.3583 to 3.3083. That gain did
not transfer to 2026: the challenger worsened MAE by 0.2727 positions. This is
exactly why selection is frozen across event blocks rather than switched after
one encouraging season.

### Race Final Position

- immutable `post_grid_pre_race` snapshot contract for revisions, penalties,
  pit-lane starts, withdrawals, DNS and starter eligibility;
- explicit retention of the separate `post_qualifying_pre_grid` proxy;
- reason-coded, partial-pooled terminal hazard with recency and causal cutoffs;
- conditional Bradley-Terry order with grid, signed Qualifying surprise,
  strength, long-run, degradation, evidence and mobility interfaces;
- joint Monte Carlo status/distance/order sampling;
- exact minimum-expected-absolute-loss assignment to a legal permutation;
- binary and multiclass Brier/log-loss, calibration bins and reason recall;
- raw race status, completed laps and retirement fraction preserved by the
  local provider.

A critical mathematical bug found during the audit was fixed: classified
finishers initially received independent Beta distance draws, which randomized
lapped distance and sent a grid-P2 car to P16. Classified distance is now tied
to conditional pace, defaults to zero deficit without causal evidence, and
uses the same order shock when an explicit deficit exists. Terminal cars retain
hazard distance, so a late retiree can still classify ahead of a genuinely
lapped finisher.

The corrected 2025 selector arm was essentially neutral. The frozen 2026 arm
also remained neutral on order while slightly improving status calibration.
That is useful decomposition evidence, but not enough to replace the grid
baseline.

### Best Estimated Lap

- full quality-aware entrant roster with earlier-session fallbacks;
- latent potential-adjusted location model;
- event-balanced valid-lap hurdle and stage-mixture interface;
- robust hierarchical Huber residual model retained as a diagnostic;
- Huber on/off selection frozen on 2025 before the 2026 audit;
- recency-weighted weak 2022-2025 invariant-feature priors;
- rolling event-block conformal intervals;
- full per-round and per-driver prediction-versus-reality evidence.

The Huber residual was rejected on 2025 (1.2344 s versus 1.2183 s for the
quality-location model), so the 2026 run used the frozen location model. The
quality-aware representation itself produced the 20.24% 2026 MAE gain.

## 2026 prediction versus reality by round

Every underlying artifact also contains driver-level predicted and actual
values. The tables below report event-level error on the complete scored
population.

### Qualifying Prediction

| Rd | Event | Baseline MAE | Pairwise MAE | Baseline Kendall | Pairwise Kendall |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Australia | 2.636 | 2.455 | 0.671 | 0.714 |
| 2 | China | 0.909 | 1.545 | 0.896 | 0.844 |
| 3 | Japan | 1.727 | 1.727 | 0.801 | 0.784 |
| 4 | Miami | 2.909 | 3.455 | 0.610 | 0.576 |
| 5 | Canada | 2.000 | 1.727 | 0.758 | 0.784 |
| 6 | Monaco | 2.545 | 3.000 | 0.697 | 0.662 |
| 7 | Barcelona | 1.727 | 1.727 | 0.801 | 0.801 |
| 8 | Austria | 0.909 | 1.636 | 0.887 | 0.827 |
| 9 | Britain | 1.455 | 2.000 | 0.835 | 0.775 |
| **Mean** |  | **1.869** | **2.141** | **0.773** | **0.752** |

Paired event-bootstrap challenger-minus-baseline MAE: `+0.2727`, 95% CI
`[+0.0303, +0.5051]`, improvement probability `1.01%`. The frozen selector
retains `qualifying_rehearsal_rank_baseline_v1`.

The separate stage models were evaluated on 678 2025-2026 entrant rows:

| Output | Brier | Log loss | Mean probability | Observed rate |
| --- | ---: | ---: | ---: | ---: |
| Valid classified lap | 0.0168 | 0.0936 | 0.9881 | 0.9838 |
| Reaches Q2 | 0.1505 | 0.4654 | 0.7144 | 0.7301 |
| Reaches Q3 | 0.1661 | 0.5142 | 0.4463 | 0.4779 |

These probabilities remain explicitly uncalibrated until a disjoint
calibration block exists.

### Race Final Position

| Rd | Event | Grid-proxy MAE | Joint MAE | Rolling-status Brier | Hazard Brier |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Australia | 4.545 | 4.455 | 0.1834 | 0.1788 |
| 2 | China | 4.182 | 4.273 | 0.2487 | 0.2407 |
| 3 | Japan | 2.000 | 2.000 | 0.0853 | 0.0876 |
| 4 | Miami | 3.455 | 3.727 | 0.1504 | 0.1492 |
| 5 | Canada | 5.273 | 5.182 | 0.2154 | 0.2064 |
| 6 | Monaco | 5.091 | 5.091 | 0.2475 | 0.2346 |
| 7 | Barcelona | 3.091 | 3.273 | 0.2468 | 0.2295 |
| 8 | Austria | 1.818 | 1.727 | 0.1500 | 0.1423 |
| 9 | Britain | 4.182 | 4.182 | 0.1179 | 0.1194 |
| **Mean** |  | **3.737** | **3.768** | **0.1828** | **0.1765** |

Mean Kendall is 0.5257 for both policies. Status log loss improves from 0.5617
to 0.5389, but the joint position MAE does not. Paired position delta is
`+0.0303`, 95% CI `[-0.0404, +0.1111]`, improvement probability `26.16%`.
The full joint challenger is rejected. The terminal component remains a
diagnostic until status-specific uncertainty is promoted separately.

This is still a `post_qualifying_pre_grid` evaluation. No local 2026 event has
an immutable first-seen final-grid snapshot, so the code deliberately makes no
historical `post_grid_pre_race` accuracy claim.

### Best Estimated Lap

| Rd | Rehearsal | Baseline MAE (s) | Quality-aware MAE (s) | Fastest hit | Interval coverage |
| ---: | --- | ---: | ---: | --- | ---: |
| 1 | FP3 | 0.700 | 0.300 | yes | 68.4% |
| 2 | Sprint Qualifying | 0.404 | 0.510 | no | 61.9% |
| 3 | FP3 | 0.303 | 0.295 | yes | 100.0% |
| 4 | Sprint Qualifying | 1.089 | 0.586 | no | 90.5% |
| 5 | Sprint Qualifying | 0.355 | 0.254 | yes | 95.0% |
| 6 | FP3 | 0.532 | 0.533 | yes | 81.8% |
| 7 | FP3 | 0.613 | 0.543 | yes | 77.3% |
| 8 | FP3 | 0.225 | 0.222 | yes | 100.0% |
| 9 | Sprint Qualifying | 0.494 | 0.518 | no | 90.9% |
| **Mean** |  | **0.524** | **0.418** | **6/9** | **84.62%** |

The quality-aware model scores all 195 observed targets. Its event-mean gain is
20.24%, fastest-driver hit remains 6/9, top-three overlap is unchanged, and
interval width increases only 3.06%. Leave-one-event-out deltas are all
negative and the largest event supplies 46.40% of positive gain.

Paired event-bootstrap delta is `-0.1061 s`, 95% CI
`[-0.2424, +0.0065]`, improvement probability `96.29%`. Every declared gate
passes except the requirement that the upper confidence bound be below zero.
The challenger therefore remains diagnostic until more prospective rounds
remove that last uncertainty.

## Evidence artifacts

- Qualifying: `artifacts/backtests/f1/qualifying/quality_aware_pairwise_v1_20260713.json`
- Race: `artifacts/backtests/f1/race_final_position/survival_order_v2_20260713.json`
- Best Lap: `artifacts/backtests/f1/best_estimated_lap/2026_walk_forward_quality_aware_huber_v2_20260713.json`

Each artifact records input and implementation hashes, exact protocol,
per-event metrics, and per-driver predictions. They are local immutable run
outputs; this report is the versioned decision record.

Suggested commit name: `docs(f1): record non-live challenger decisions`
