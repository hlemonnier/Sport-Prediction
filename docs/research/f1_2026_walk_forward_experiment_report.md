# F1 2026 Point-in-Time Walk-Forward Experiment Report

Date: 2026-07-11. Scope: completed 2026 rounds 1-9, ending with the British
Grand Prix. This report promotes only the policy choices supported by the
frozen local snapshots and leaves probability, live-strategy, and deep-model
claims closed.

## Frozen Protocol

- Each round is predicted from earlier completed 2026 rounds only. No 2025
  observations are silently pooled across the 2026 regulation change.
- Qualifying uses only sessions completed before Grand Prix qualifying. Race
  uses the `post_qualifying_pre_grid` horizon.
- Every evaluated field has exactly 22 predicted, 22 actual, and 22 matched
  drivers, with zero missing or unexpected entries. Round 1 is necessarily
  heuristic because no earlier 2026 event exists.
- The active roster comes from the latest eligible session plus two-seat team
  continuity. This carried two temporarily absent cars in rounds 1 and 4,
  excluded one superseded reserve in round 3, and excluded six reserves in each
  of rounds 7 and 8. Unclassified result rows receive stable tail ranks.
- Runs execute sequentially with one thread and immutable per-run manifests.
- Differences use paired event-level bootstrap resampling with seed `20260711`.
  The primary MAE comparison uses 500,000 resamples of the nine events.

## Locked Runs

| Run | Purpose | Decision |
| --- | --- | --- |
| `2026_primary_baseline_20260711b` | Frozen pace/grid baselines | Comparator |
| `2026_primary_auto_20260711b` | Automatic ML/baseline selection with run-simulation features | Qualifying gain; race rejected |
| `2026_ablation_no_runsim_20260711b` | Automatic selection without run-simulation features | Retained for qualifying |
| `2026_ablation_with_standings_20260711b` | Target-specific policy with standings | Rejected: exactly zero metric or selection change |
| `2026_promoted_policy_20260711c` | Canonical target-specific retained policy | Evidence-pack source |

## Qualifying Result

| Metric | Frozen baseline | Retained policy | Change |
| --- | ---: | ---: | ---: |
| Full-field MAE | 2.8586 | 2.2323 | -0.6263 |
| Kendall tau-b | 0.6508 | 0.7287 | +0.0779 |
| Top-3 hit rate | 0.5185 | 0.6667 | +0.1481 |
| Top-5 hit rate | 0.7556 | 0.7556 | 0.0000 |
| Top-10 hit percentage | 80.00% | 87.78% | +7.78 pp |
| Exact-position rate | 0.1313 | 0.1616 | +0.0303 |
| Position-interval coverage | 0.8889 | 0.8131 | -0.0758 |

The paired qualifying MAE delta is -0.6263 positions, with a 95% event
bootstrap interval of [-1.1313, -0.1818] and bootstrap probability of
improvement 0.999246. Kendall tau-b improves by 0.0779 with interval
[0.0221, 0.1414], and top-3 hit rate improves by 0.1481 with interval
[0.0370, 0.2593]. The selector is retained for ranking. Its interval coverage
regression is statistically adverse, so its uncertainty intervals are not
promoted.

The retained selector chose simple pace baselines on seven trained events and
a ridge/pace blend only on round 9. This is evidence for adaptive selection,
not evidence that a complex model is universally superior.

## Race Result

The automatic candidate is rejected. Mean full-field MAE worsened from 3.7374
to 3.8182 (+0.0808), Kendall tau-b fell from 0.5257 to 0.5200, and interval
coverage fell from 0.6414 to 0.6162. The paired MAE interval for candidate
minus baseline is [-0.0202, 0.1818], with only 0.0467 bootstrap probability of
improvement. The retained policy therefore remains the
post-qualifying grid-only point-ranking baseline.

## Ablations and Non-Promotions

- Run-simulation features do not improve qualifying: the full feature set is
  0.0101 MAE worse than the ablation, with interval [0.0000, 0.0303]. They stay
  available as quality/provenance diagnostics but are disabled for the
  promoted qualifying policy.
- Standings cause exactly zero prediction, selection, or metric change in the
  tested rounds and are excluded.
- Circuit identity is not tested as a promoted feature because the nine-event
  slice has no repeated 2026 circuit from which to estimate a causal within-
  regime effect.
- Weather is not promoted because immutable historical forecast snapshots are
  absent; retrospective observations would not reproduce a pre-event forecast.
- Historical transfer, DL, and RL are not promoted. The 2026 regime change and
  short sample make complexity or cross-season pooling a weaker prior than the
  measured same-season policies.
- The temporal probability audit correctly remains unavailable: the rolling
  histories do not yet supply separate, sufficiently sized selection,
  calibration, and final-audit blocks. Probability columns are diagnostic and
  must not be marketed as calibrated.

## Retained Runtime Policy

- Qualifying: `model=auto`, run-simulation disabled, standings/circuit/weather
  disabled, same-season walk-forward, `pre_qualifying` cutoff.
- Race: `model=baseline`, run-simulation retained for the existing diagnostic
  probability layer, standings/circuit/weather disabled,
  `post_qualifying_pre_grid` cutoff.
- Live race and ultimate-lap paths remain experimental until their own locked
  causal replay or grouped temporal reports pass.

## Independent Module Audits

The round-9 live-race replay was cut at global session time 5,200 seconds. A
run from the full 1,113-row race file and a run from a physically truncated
411-row prefix produced identical 22-driver prediction payload hashes. The
causal cutoff works. The model does not earn promotion: one-step MAE is 0.4481
seconds versus 0.3759 for naive last-lap persistence, a -0.0722-second gain
(that is, a regression), and the filter, transition, and strategy-template
priors remain hand tuned. Evidence:
`artifacts/reports/f1/live_race_round09_global_prefix_invariance.json`.

The ultimate-lap deterministic baseline was evaluated on 21 drivers with a
valid compatible-sector qualifying target at Silverstone. With only earlier
circuits, unseen-circuit MAE is 10.5461 seconds and interval coverage is zero,
which proves that raw cross-circuit lap-time transfer is invalid. Adding only
causally available Silverstone FP1, Sprint Qualifying, and Sprint evidence
reduces MAE to 1.1656 seconds, gives Spearman 0.9091 and identifies the fastest
driver, but the interval is an uninformative 10 seconds wide and the audit is a
single event without a locked comparator. It remains a research lower-bound
baseline, not an achievable-lap forecast or promoted model. Evidence:
`artifacts/backtests/f1/ultimate_lap_time_round09_post_sprint_pre_q_20260711.json`.

Suggested commit name: `feat(f1): promote target-specific 2026 walk-forward policies`
