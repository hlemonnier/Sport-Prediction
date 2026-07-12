# F1 Four-Mode Mathematical, Logical, and Implementation Audit

Finalized: 2026-07-12 Europe/Paris. Primary evaluation-data cutoff: completed
2026 rounds 1-9 through 2026-07-05.

> 2026-07-13 challenger addendum: the non-live feature/model implementation
> and new locked results are recorded in
> `docs/research/f1_non_live_challenger_results_20260713.md`. Qualifying and
> Race baselines remain retained. The quality-aware Best-Lap challenger is
> materially better but remains diagnostic because its event-bootstrap upper
> confidence bound is still slightly above zero.

## Executive Decision

The F1 product has exactly four user-facing modes:

1. **Qualifying Prediction**
2. **Race Final Position**
3. **Best Estimated Lap Time**
4. **Live Race Intelligence**

The previous module layout did not enforce those four targets cleanly. It mixed
qualifying classification with the race grid, a theoretical sector floor with
an achievable lap forecast, ordinary live forecasts with counterfactual
strategy decisions, and research outputs with promotion claims. Those
assumptions are no longer accepted.

The retained 2026 evidence is:

| Mode or subcontract | Retained output | Current evidence | Decision |
| --- | --- | ---: | --- |
| Qualifying | Full-field point ranking | MAE 1.8788 positions; Kendall 0.7749 | Retain transparent causal expert selector; do not promote probabilities or intervals |
| Race final position | Full-field point ranking | MAE 3.7374; Kendall 0.5257 | Retain post-Qualifying proxy baseline; reject tested causal challengers and do not claim final-grid or terminal-status validation |
| Best estimated lap | Achievable per-driver qualifying-lap point estimate | Conditional MAE 0.5133 s; 191/196 union rows scored | Retain the rehearsal-shift estimate only on its observed-target population; full mode, intervals, and telemetry DL remain unpromoted |
| Live next lap | Next representative-lap point estimate | Evaluated blend MAE 0.5379 s vs naive 0.5492 s | Reject the blend because its event-bootstrap interval crosses zero; use the causal naive baseline as the research-runner default, not a deployed service claim |
| Live pit/compound/pace | Constrained action | No locked policy-value or live-shadow evidence | RL remains offline research only |

This is deliberately not a claim that all four modes are production-ready. It
is a target-correct architecture with measured point policies where evidence
exists and explicit blockers where it does not.

## Weekend and Information-Time Boundary

The code must follow the weekend as information actually arrives. Under the
[FIA 2026 Sporting Regulations, Issue 07](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_07_-_2026-06-25.pdf):

- on an alternative-format weekend, Sprint Qualifying sets the Sprint grid;
- Grand Prix Qualifying remains a separate session and sets the Race grid;
- penalties, withdrawals, unclassified drivers, pit-lane starts, provisional
  grid publication, and final grid publication can change the starting order;
- the final Race grid is published one hour before the formation lap;
- in a dry Race, absent intermediate/wet use, at least two dry specifications
  must be used and one must be a mandatory Race specification.

Consequences for the model contracts:

- Qualifying classification is never relabelled as the final Race grid.
- The Race model consumes the best grid evidence available at its named
  horizon. Before the final grid, qualifying is an explicit proxy, not truth.
- Historical data recorded after the Race cannot be treated as a pre-Race
  snapshot merely because a field describes the starting grid.
- Live strategy actions are invalid unless the current tyre inventory, weather
  state, pit state, race-control state, and mandatory-tyre obligation are known.

The new resolver in `packages/f1/domain/starting_grid.py` represents this as an
event-sourced, as-of object. It keeps qualifying classification, provisional
grid, final grid, pit-lane starters, withdrawals, and actual starters distinct.
Unresolved evidence fails closed.

## Cross-Mode Rules

Every mode now obeys the following invariants:

- one explicit estimand, output unit, and legal information horizon;
- no target or post-event field in inference input;
- complete-field evaluation for classification modes;
- ascending event walk-forward validation, never random row splitting;
- 2026 same-season evidence is the primary arm; earlier regulation regimes are
  not silently pooled;
- a simple causal baseline is mandatory before ML, DL, or RL;
- point accuracy, probability calibration, interval calibration, and decision
  value are separately promoted;
- a green test suite proves implementation behavior, not predictive edge;
- an artifact is evidence only when input, implementation, configuration, and
  protocol hashes are retained.

## Mode 1 — Qualifying Prediction

### Intent and target

For each entered driver, predict the official Grand Prix Qualifying
classification at session completion using only evidence available before
Qualifying begins. The primary output is one full-field permutation. A position
distribution may also be emitted, but it requires a separate calibration gate.

On a standard weekend, the last target-aligned rehearsal is FP3. On an
alternative-format weekend, Sprint Qualifying is the closest qualifying-style
rehearsal. The selection is event-consistent: the same source is used for every
driver, and missing driver observations are explicitly imputed and flagged.

### Mathematics

The minimum credible baseline is a causal rank:

\[
r_{i,e}=\operatorname{rank}\left(\text{latest completed qualifying-style
pace}_{i,e}\right).
\]

A learned full-field model should optimize a ranking likelihood, not pretend
that 22 independent position regressions form a valid permutation. For scores
\(s_i=f(x_i)\), a Plackett–Luce model is:

\[
P(\pi\mid X)=\prod_{k=1}^{N}
\frac{\exp(s_{\pi_k})}{\sum_{j=k}^{N}\exp(s_{\pi_j})}.
\]

Listwise or pairwise learning-to-rank is mathematically appropriate only when
rows are grouped by event; the [XGBoost learning-to-rank contract](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html)
also treats each ordered set as a query group. Calibration must then map the
joint score process to position marginals on event-disjoint calibration races.

### Rejected assumptions in the prior implementation

- The existing aggregate pace score was assumed to be the best default without
  directly competing against the last target-aligned rehearsal.
- A `baseline` experiment flag silently enabled ML candidates, invalidating the
  claimed baseline-only arm.
- Derived rehearsal columns could survive a pre-FP scrub.
- Missing nested qualifying context could return before the requested Race
  horizon scrub, leaving retrospective grid/qualifying fields in place.
- Emitted probability and interval fields were easy to confuse with calibrated
  products.

### Implemented policy

- `packages/f1/features/assembly.py` builds event-consistent rehearsal rank,
  delta, source, coverage, and imputation features.
- `packages/f1/models/training.py` makes baseline-only mean baseline-only and
  prioritizes the rehearsal comparator.
- `packages/f1/orchestration/prediction.py` carries the evidence provenance,
  scrubs it at earlier horizons, and uses it in the transparent fallback.
- The model registry exposes Qualifying Prediction as one of exactly four
  contracts in `packages/f1/orchestration/contracts.py`.

### 2026 result and decision

Across nine complete 22-driver fields, the chronological causal expert policy
has mean field MAE 1.8788 and mean Kendall tau-b 0.7749. Those are absolute
walk-forward measurements under the current code. Earlier comparison runs for
the 2.2323 legacy policy, 1.8788 fixed-rehearsal baseline, and ML arm no longer
match the current implementation/configuration hashes and are quarantined.
Therefore this audit makes no current paired-improvement or complexity-edge
claim.

Top-10 hit rate is 91.11%, but event-mean interval coverage is only 0.7020.
Therefore:

- retain the causal point ranking;
- retain simple expert selection as the current research point policy;
- do not promote position probabilities or intervals;
- do not add circuit, weather, standings, DL, or cross-season transfer until a
  predeclared event-disjoint ablation beats this baseline.

## Mode 2 — Race Final Position

### Intent and target

Predict the official terminal Race classification and terminal status—not the
order at the chequered flag alone. The point output is a full-field
classification permutation. The probabilistic output is a joint position and
status distribution over classified finish, DNF, non-classified, and DSQ.

The mode has multiple legal horizons. A prediction after Qualifying but before
the final grid is not the same product as a prediction after the final grid.
Every output must state the horizon and its grid evidence.

### Mathematics

The non-negotiable baseline is resolved starting-order persistence:

\[
\hat\pi_e^{\text{base}}=G_e(t),
\]

where \(G_e(t)\) is the event-sourced starting-order object known at time
\(t\). If the final grid is unresolved, the system may use qualifying context
but must label it as a proxy.

A credible learned model couples survival and ranking. One useful factorization
is:

\[
P(\pi,z\mid X,G)=P(z\mid X,G)
\;P(\pi\mid z,X,G),
\]

where \(z_i\) is terminal status, modeled through a competing-risk hazard, and
the conditional permutation is modeled through a survival-aware listwise
ranker or an event-driven Monte Carlo simulator. Independent driver position
regressions are insufficient because they can assign multiple cars to the same
position and cannot conserve podium or points probability mass.

### Rejected assumptions in the prior implementation

- Qualifying classification and starting grid were treated as interchangeable.
- 2021-22 Sprint weekends could label Friday Qualifying as the Grand Prix grid,
  although the Sprint classification set that grid in those formats.
- The local frozen 2026 data did not contain a point-in-time final-grid snapshot,
  yet evaluation labels could suggest official-grid completeness.
- The deployed snapshot service converted rank ordering into heuristic event
  probabilities without magnitude-sensitive score gaps.
- Retired drivers could receive non-zero win, podium, or points probability.

### Implemented policy

- `packages/f1/domain/starting_grid.py` resolves qualifying source, phase,
  adjustments, withdrawals, pit-lane starters, and actual start separately.
- Empty or ambiguous grid evidence fails closed; source and as-of provenance
  travel with the resolution.
- Historical 2021-22 Qualifying is explicitly a pre-Sprint proxy, never the GP
  grid.
- `services/f1-prediction-service/f1_prediction_service/model.py` produces a
  magnitude-sensitive, balanced joint position matrix for classification-
  eligible cars; DNS/DSQ/withdrawn/unclassified rows are explicitly unavailable
  instead of receiving an invented tail rank.
- `services/f1-platform/f1_platform/predictions.py` validates row and column
  mass, the active/non-runner partition, and active-field win/podium/points
  capacities before accepting remote output.

### 2026 result and decision

The frozen post-Qualifying/pre-final-grid evaluation has mean field MAE 3.7374
and Kendall tau-b 0.5257 for an explicit Qualifying-order proxy. The older
automatic-ML comparison is stale under the current code and is not current
evidence.

A separate causal challenger backtest tested team and driver places-gained
shrinkage, a reliability hazard, hierarchical residuals, rank-normalized
residuals, and a running-residual/reliability combination. Every candidate used
only earlier 2026 rounds and emitted a complete permutation. The closest,
team-residual shrinkage, has MAE 3.7980: 0.0606 worse than the baseline, with
paired 95% interval [-0.1111, 0.2424]. Reliability improves winner hit rate
from 0.6667 to 0.7778 but worsens the primary full-field MAE to 3.8990. No
challenger is retained.

This evidence does **not** certify the most useful pre-Race final-grid horizon,
because no immutable final-grid-as-of snapshots exist in the local rounds.
Therefore:

- retain the post-Qualifying proxy point baseline for this exact horizon;
- reject the tested causal residual/reliability challengers;
- keep the new joint service output explicitly uncalibrated and unpromoted;
- require first-seen official grid, penalty, withdrawal, and actual-start
  snapshots before claiming final-grid evaluation;
- next test a survival-aware listwise model and calibrated event simulator only
  after the stronger data contract exists.

## Mode 3 — Best Estimated Lap Time

### Intent and target

The user-facing question is: for each driver, what best valid representative
lap should be achievable by the end of Grand Prix Qualifying? The primary
output is \(p05,p50,p90\) in seconds, with a separately ranked fastest-driver
output.

The old name “ultimate lap” hid two incompatible estimands:

1. a **theoretical compatible-sector lower bound**; and
2. an **achievable session-end best-lap distribution**.

The first is diagnostic. It is not an expected lap. Combining arbitrary sector
minima from different fuel, tyre, session, or car states is forbidden. The
second is the actual product target.

### Mathematics

Let \(B_{i,e}\) be the best valid target-aligned rehearsal lap for driver \(i\)
in event \(e\). For rehearsal source \(s\), the retained causal baseline learns
only from strictly earlier events:

\[
\delta_{s,e}=\operatorname{median}_{e'<e}
\operatorname{median}_{i}\left(Y_{i,e'}-B_{i,e'}\right),
\qquad
\hat Y^{50}_{i,e}=B_{i,e}+\delta_{s,e}.
\]

Prequential residual quantiles produce diagnostic interval offsets. If there
is no earlier same-source history, the point estimate uses zero shift and the
interval is explicitly unavailable rather than invented.

The theoretical lower bound is separately defined as

\[
L_{i,e,c}=\sum_{q=1}^{3}\min \{S_{i,e,c,q}\},
\]

only within one compatible session/compound/car-state group \(c\). No equality
between \(L\) and achievable \(Y\) is assumed.

### Rejected assumptions in the prior implementation

- Theoretical sector minima were presented close to an achievable lap target.
- Deep inference required target labels, so it was not genuine inference.
- Unlabelled training rows and degenerate quantiles could pass too far through
  the pipeline.
- A fixed 10-second interval on one event looked numerically safe but was not a
  calibrated forecast.
- A telemetry TCN was treated as a natural upgrade although the local cache has
  no 2026 car telemetry with the required distance-normalized contract.

### Implemented policy

- `packages/f1/models/ultimate_lap_time/achievable.py` implements the strictly
  causal source-specific rehearsal-shift baseline.
- Training rejects unlabelled rows; inference rejects any target/outcome field.
- Deep training additionally requires rehearsal telemetry from a different
  session with `feature_as_of < target_as_of` for the separate GP Qualifying
  session-end label; same-Q-lap reconstruction is rejected.
- Deep inference accepts genuine unlabelled input and enforces positive ordered
  sector/lap outputs with lap p50 equal to the sum of sector p50s.
- Evaluation blocks degenerate quantiles and keeps theoretical and achievable
  target semantics explicit.

### 2026 result and decision

Across nine rounds, the causal rehearsal roster contains 192 inference rows,
the valid-Q-lap target population contains 195 rows, and their union contains
196 rows. Exactly 191 rows are scoreable (97.45% union coverage). Conditional
on those matched observed targets:

- event-mean p50 MAE: 0.5133 seconds;
- raw rehearsal event-mean MAE: 0.7639 seconds;
- paired improvement: 0.2505 seconds;
- event-mean Spearman: 0.9313;
- fastest-driver hit rate: 6/9;
- top-three overlap: 0.5926;
- diagnostic interval coverage: 0.7748 on 151 rows.

The paired event-bootstrap improvement interval is [-0.4366, -0.0685] seconds,
with improvement probability 0.997445 (200,000 event resamples, seed
`20260712`). Retain the conditional point estimate, but do not promote the full
mode until it models no-valid-Q-lap/non-participation outcomes and covers the
causal roster. Intervals remain unpromoted. Do not train the telemetry TCN until
2026 distance-normalized rehearsal telemetry with separate pre-Q feature and Q
target timestamps exists and the simple baseline is beaten under grouped
temporal validation.

## Mode 4 — Live Race Intelligence

Live is one product surface with two mathematically different internal layers.
Calling all of it RL would be a category error.

### Forecasting layer

The forecasting subcontracts are:

- next representative lap time;
- clean-lap stint degradation;
- joint running order at a named future checkpoint;
- terminal classified/DNF/NC/DSQ status.

These are supervised, probabilistic, state-space, ranking, or survival tasks.
They are evaluated against observed future state. RL is neither necessary nor
appropriate merely because the data arrive sequentially.

For next-lap pace, the evaluated candidate is:

\[
\hat y_{i,l+1}=w_e\mu^{SSM}_{i,l+1}+(1-w_e)y^{clean}_{i,l},
\]

where \(w_e\in\{0,0.05,\ldots,1\}\) is selected using mean MAE from strictly
earlier events only. The global event timestamp—not each driver's lap number—
defines the replay prefix.

The corrected nine-event walk-forward scores the exact frozen forecast emitted
after the current eligible lap against that driver's next eligible lap. It gives:

- emitted blend event-mean MAE: 0.5379 seconds;
- naive last-clean-lap MAE: 0.5492 seconds;
- pure state-space MAE: 0.6755 seconds;
- paired blend-minus-naive delta: -0.0112 seconds;
- 95% interval: [-0.0255, +0.0010]; improvement probability 0.96081.

The pure state-space model is rejected, and the blend also fails because the
event-level interval crosses zero. The research-selected next-event weight is
0.20 SSM, but the research runner fails closed to 0.00 SSM and 1.00 naive:
\(\hat y_{i,l+1}=y^{clean}_{i,l}\) whenever a prior eligible lap exists; the
cold-start path may use the causal SSM until that baseline exists. No interval,
degradation, order, or terminal-status probability is promoted.

Timing MAE is conditional on another eligible lap occurring: 8,144 of 8,331
eligible issuances were matched (97.76%), while 187 had no later eligible lap.
Those are reported as coverage outcomes rather than silently treated as timing
targets. The seconds-valued Live next-lap and Best-Lap
research runners are not yet wired to the platform prediction API: its current
`next-lap` kind is a relative-order distribution, not lap time in seconds, and
Race sessions route to the Race-kind output. No deployment claim is made for
either seconds-valued mode.

### Decision layer and RL boundary

The decision subcontracts are:

- pit now, pit next opportunity, or stay out;
- next legal compound;
- bounded pace/conservation target.

For state \(s_t\), the legal mask defines \(A_{legal}(s_t)\). A strategy policy
solves a constrained partially observed control problem:

\[
a_t^*=\arg\max_{a\in A_{legal}(s_t)}
\mathbb E\left[U(\text{finish},\text{points},\text{risk})\mid s_t,a\right].
\]

RL is eligible here because actions affect later state and rewards. It is still
not automatically the best method. Constrained dynamic programming and MPC are
mandatory comparators. The recent F1 strategy literature follows the same
discipline: [Towards Learning-Based Formula 1 Race Strategies](https://arxiv.org/abs/2512.21570)
compares an RL policy to a mixed-integer nonlinear optimization benchmark;
[Explainable Reinforcement Learning for Formula One Race Strategy](https://arxiv.org/abs/2501.04068)
trains and tests inside a simulator; and [Learning-based Multi-agent Race
Strategies in Formula 1](https://arxiv.org/abs/2602.23056) explicitly models
energy, tyre degradation, aerodynamic interaction, pit decisions, and opponent
responses.

Therefore a credible RL promotion requires all of the following:

1. point-in-time state and first-seen race-control data;
2. legal action masks with zero illegal-action rate;
3. a holdout-calibrated simulator for pace, tyres, fuel/energy, traffic, pit
   loss, flags, weather, reliability, and interaction;
4. simple rule, DP, MPC, and oracle-regret comparisons;
5. support-checked off-policy evaluation with uncertainty;
6. locked causal replay followed by live shadow mode.

Enforcement is currently partial and deliberately fail-closed. A self-declared
`promotion_ready` boolean no longer passes the offline evaluator: policy
exceptions, missing oracle/value evidence, illegal actions, missing hashes,
insufficient support, and unlocked simulator/calibration reports all block it.
The registry does **not yet** accept and validate OPE uncertainty or live-shadow
reports as first-class evidence. Those two requirements remain explicit
promotion blockers, so no RL policy can honestly be called shadow-tested or
promotion-ready today.

### Rejected assumptions and production repairs

- The earlier one-step comparison used mismatched model and baseline rows and
  fitted ARIMA in-sample. Evaluation now uses matched populations and causal
  expanding-origin baselines.
- The superseded v2 Live artifact scored a same-row nowcast rather than freezing
  the emitted forecast until the same driver's next eligible lap. The v3
  protocol fixes issuance/target alignment; its wider event interval rejects
  the apparent blend gain and fails the research runner closed to the naive
  baseline.
- Missing timestamps could be replaced by lap number. They now remain unknown,
  and any fallback is explicitly labelled.
- Per-driver lap cutoffs allowed later physical observations from lapped cars.
  Replays now use one global event time.
- DP terminal tyre penalties were applied at a finite planning horizon rather
  than actual Race end; branch limits could remove action types. Both are fixed.
- Remote inference held the platform lock and stale responses could overwrite a
  newer state. Inference now runs outside the lock and is rejected if reducer
  identity, sequence, or prediction kind changed.
- Session routing conflated qualifying, practice, and race predictions.
  Payloads now carry `prediction_kind` and `position_semantics`.
- Strategy recommendations survived missing legality state. They now fail
  closed for unknown compounds, unknown track state, existing pit state, or a
  mask-illegal action.
- The shared RL/DP/MPC mask formerly invented all dry compounds and an open pit
  lane when those inputs were absent. Missing inventory or pit-lane state now
  masks every pit action, mapping preserves box/wet state, and INTER/WET use
  correctly waives the two-dry-specification terminal penalty.
- DNS, DSQ, withdrawn, and unclassified rows could retain win/podium/points
  probability or receive an invented tail rank. They are now explicitly
  unavailable. Retired/stopped cars remain classification-eligible for Race,
  while next-lap output is unavailable once a car is no longer running.

## Promotion Matrix

| Surface | Point output | Probability/interval | Policy/action |
| --- | --- | --- | --- |
| Qualifying | Retained research policy | Not promoted | Not applicable |
| Race final position | Retained baseline | Not promoted | Not applicable |
| Best estimated lap | Conditional research policy retained; full mode blocked | Not promoted | Not applicable |
| Live next lap | Naive causal research-runner default; SSM blend rejected | Not promoted | Not applicable |
| Live degradation/order/status | Not promotion-certified | Not promoted | Not applicable |
| Live pit/compound/pace | Forecast inputs only | Not a forecast promotion | Offline RL/DP/MPC research; no shadow evidence |

## What Must Happen Next

The highest-return next work is data and evaluation, not a larger network:

1. Capture immutable first-seen official final grids, penalties, withdrawals,
   tyre inventories, weather, Race Control, pit state, and actual starters.
2. Add 2026 distance-normalized car telemetry before any Best Lap TCN run.
3. Accumulate enough completed 2026 events for event-disjoint selection,
   calibration, and final-audit blocks.
4. For Race Final Position, compare a survival-aware listwise ranker and an
   event-driven simulator against the final-grid baseline—not against a weaker
   invented comparator.
5. For Live, calibrate the simulator component by component before training or
   valuing an RL policy; then require a locked MPC/RL comparison and shadow run.

Until those gates pass, more ML/DL/RL complexity would increase confidence
faster than accuracy. The current redesign chooses the strongest measured
simple policy and makes every unsupported claim visibly fail closed.

## Reproducible Evidence

- Qualifying and Race:
  `artifacts/backtests/f1/rolling_2026/runs/2026_four_mode_rebuild_20260712e/`
- Race causal challenger audit:
  `artifacts/backtests/f1/race_final_position/2026_walk_forward_causal_residual_challengers_v2_20260712e.json`
- Best Estimated Lap:
  `artifacts/backtests/f1/best_estimated_lap/2026_walk_forward_rehearsal_shift_v2_20260712e.json`
- Live next lap:
  `artifacts/backtests/f1/live_next_lap/2026_walk_forward_ssm_naive_emitted_forecast_v3_20260712e.json`
- Live global-prefix invariance:
  `artifacts/reports/f1/live_race_round09_global_prefix_invariance_v2_20260711a.json`
  (historical diagnostic; current global-event-time behavior is also exercised
  by the Live backtest and dedicated regression tests)

Suggested commit name: `feat(f1): rebuild four modes around causal evidence contracts`
