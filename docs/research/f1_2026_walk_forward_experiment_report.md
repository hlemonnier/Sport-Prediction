# F1 2026 Qualifying and Race Walk-Forward Report

Finalized: 2026-07-12 Europe/Paris. Evaluation-data cutoff: completed 2026
rounds 1-9 through the British Grand Prix on 2026-07-05. The full mathematical,
logical, implementation, Live, and Best Lap review is in
[`f1_four_mode_math_logic_implementation_audit_20260711.md`](f1_four_mode_math_logic_implementation_audit_20260711.md).

## Protocol

- Each round uses only strictly earlier completed 2026 rounds for model or
  policy selection. There is no silent pre-2026 pooling across the regulation
  change.
- Qualifying uses only sessions completed before Grand Prix Qualifying.
- Race uses the `post_qualifying_pre_grid` horizon. The frozen data do not
  contain immutable final-grid-as-of snapshots, so Qualifying is explicitly a
  proxy rather than claimed as the official final grid.
- Every evaluated event has 22 predicted, 22 actual, and 22 matched drivers.
- Rounds execute in ascending order with one thread.
- The canonical run records actual CSV access by inference/evaluation role,
  actual training event keys, bound profile hashes, and start/end drift guards.

## Canonical Run

`2026_four_mode_rebuild_20260712e` is the canonical Qualifying/Race evidence
run. It uses:

- Qualifying: `model=auto`, candidate family `baseline` only, target-specific
  run-simulation features disabled;
- Race: `model=baseline`, run-simulation diagnostics retained;
- standings, circuit identity, and weather disabled;
- same-season walk-forward with one sequential training job.

The “auto” label for Qualifying means chronological selection among transparent
causal baseline experts; it does not enable ML in this run.

## Qualifying

The retained causal expert selector first competes against the latest
target-aligned qualifying rehearsal: FP3 on a standard weekend and Sprint
Qualifying on an alternative-format weekend. The source is event-consistent,
and missing driver observations are explicitly flagged.

| Metric | Current causal expert |
| --- | ---: |
| Full-field MAE | 1.8788 |
| Kendall tau-b | 0.7749 |
| Top-10 hit percentage | 91.11% |
| Exact-position rate | 0.2475 |
| Position-interval coverage | 0.7020 |

Earlier legacy, fixed-rehearsal, and ML comparator runs no longer match the
current source/configuration hashes and are quarantined. These numbers support
the current absolute walk-forward point-policy measurement, not a current
paired edge claim.

Decision:

- retain the causal expert point ranking;
- keep ML shadow-only;
- do not promote probability marginals or intervals because no disjoint
  selection/calibration/final-audit blocks exist and interval coverage is poor.

## Race Final Position

The current post-Qualifying proxy baseline has full-field MAE 3.7374 and
Kendall tau-b 0.5257. This is not a resolved-final-grid result: immutable final
grid snapshots are unavailable. The old automatic-ML comparison is stale and
is not used as current evidence.

An independent same-season causal challenger run also rejects every residual
or reliability adjustment. Team residual shrinkage is the closest at MAE
3.7980, still 0.0606 worse than baseline with paired 95% interval
[-0.1111, 0.2424]. Driver residual, reliability hazard, hierarchical residual,
rank-normalized hierarchy, and the combined running/reliability policy have MAE
between 3.8990 and 4.0000. None is promoted. Evidence:
`artifacts/backtests/f1/race_final_position/2026_walk_forward_causal_residual_challengers_v2_20260712e.json`.

The new starting-grid resolver handles qualifying source, provisional/final
phase, penalties, withdrawals, pit-lane starts, and actual start separately.
That implementation correction does not retroactively create missing snapshot
evidence. A final-grid horizon must be evaluated again after first-seen grid
snapshots are captured.

Race win, podium, points, and position probabilities remain diagnostic and
unpromoted. Terminal-status learning/evaluation is not implemented in the
current research provider path; the contract remains a target, not a delivered
model.

## Rejected Complexity

- Run-simulation features are not needed by the retained Qualifying point
  policy.
- Standings previously produced zero selection or metric change and remain
  excluded.
- Circuit identity has no repeated-circuit 2026 evidence in rounds 1-9.
- Retrospective weather observations are not immutable forecast snapshots.
- Historical transfer, DL, and RL are not justified for these two offline
  classification targets by the current sample.

Suggested commit name: `feat(f1): retain causal 2026 qualifying and race policies`
