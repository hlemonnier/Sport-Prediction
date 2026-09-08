# F1 four-mode model research v2: final evidence package

Frozen on 2026-07-14 from 2026 rounds 1-9. A race weekend is the independent
validation unit; driver rows and telemetry laps are correlated observations,
not additional independent experiments.

This report answers four different questions separately:

1. Was a mathematically coherent candidate implemented and actually run?
2. Did it beat the retained baseline on the within-run selection evidence?
3. Did it clear the stronger promotion gates on disjoint within-run evidence?
4. Was it deployed to the runtime path?

Those states are not interchangeable. In this package, “rejected” means the
candidate was evaluated but the selected public output remains the baseline.
“Diagnostic” means the result is informative but not valid promotion evidence.
“Blocked” means a required data contract is absent; it does not mean the idea
was abandoned.

Every forecast is frozen before its target is opened within the current replay.
However, earlier R7-R9 outcomes informed later implementation repairs and
parameter-profile design. Those partitions are therefore post-development
replay diagnostics, not pristine prospective promotion evidence. This
distinction does not invalidate the measured numbers; it prevents us from
overstating what they prove.

## Executive result

| Mode | What was implemented and tested | Selected output | Promotion result | Runtime consequence |
| --- | --- | --- | --- | --- |
| Qualifying | Same-season latent lap model, official stage/no-time logic, legal joint sampling, exact minimum-expected-absolute-loss assignment, position-probability calibration | Public: rehearsal-rank baseline. Research-selection winner: shared latent v4 plus MEAL | Point model not promoted: R7-R9 replay ties baseline. Probabilities not promoted: log loss/ECE improve but Brier worsens and only two calibration weekends exist | Existing service heuristic remains separate |
| Race final position | Main survival/order model plus a second final-grid/prior-race-state ablation covering R1-R9, bound to nine pinned official FIA PDF files | Legal final-grid baseline | Neither challenger promoted. Main survival loses materially; FIA-byte-certified ablation essentially ties and has only post-development descriptive evidence | Existing service heuristic remains separate |
| Best estimated lap | Achievable session-end point target, calibrated intervals, same-season residual models, temporal sequence model, true bounded TCN and parameter/sham matrix | Retained Best-Lap baseline and its intervals | Point and intervals not promoted; TCN is parameter-sensitive and inconclusive | No new production endpoint |
| Live intelligence | Supervised next-lap filter, legal heuristic, simulator/DP/MPC, state v6 plus transition/replay/dataset v8 and legal-mask v3 evidence, partial-label BC and separate offline-Q/OPE gates | Naive next-lap forecast plus deterministic legal strategy | Blend CI crosses zero; rolling BC ties the trivial classifier; 0 rows qualify for offline-Q or OPE | No learned-policy promotion |

The important conclusion is not “the models failed.” The implementations are
now substantially more honest and reproducible. Qualifying improved sharply
versus its earlier broken implementation, the TCN was trained instead of being
deferred, and the Race provenance objection was converted into a certified
alternative experiment. What did not happen was promotion without evidence.

## 1. Qualifying prediction

### Target, horizon, and mathematics

The target is the complete official Grand Prix Qualifying classification. The
forecast is frozen before the target result is opened. Current-season practice
or rehearsal evidence may be used only if it is available at the target-aligned
pre-Qualifying cutoff; target Qualifying rows cannot repair the input roster.

The same-season protocol is:

- R1-R2: structural point fit;
- R3-R4: model, residual, and point-head selection;
- R5-R6: interval and probability-calibration diagnostics;
- R7-R9: mechanically held-out post-development replay block (named `audit`
  in the artifact for partition compatibility).

For a joint sampled legal classification with sampled positions `P[d,s]`, the
selected point head solves

`min over permutations pi of sum_d E_s[abs(pi_d - P[d,s])]`.

This is an exact Hungarian assignment. It is better aligned to the scoring
loss than simply sorting median lap time, while still guaranteeing a legal
permutation.

Three material implementation assumptions were corrected:

1. official stage advancement is not equivalent to having a recorded later-stage time;
2. penalties can move a driver outside a cutoff even when a later-stage time proves participation;
3. 30% shared variance requires `sqrt(.30) Z_shared + sqrt(.70) Z_driver`, not `.30 Z_shared + sqrt(1-.30^2) Z_driver`.

The old shared challenger reported all-round MAE 2.9899. The repaired research
challenger is 1.8586 versus 1.8687 for the baseline across all nine rounds when
the unavoidable R1-R2 cold start is scored by baseline substitution. Because
the challenger did not pass promotion, the public selected output is the
baseline and scores 1.8687. On the R7-R9 post-development replay, research
challenger, baseline, and public output are all 1.3636. That is a major
implementation improvement, but not an independent prospective edge.

### Prediction versus reality by round

Candidate is genuinely unavailable in R1-R2; those cells are not treated as
zero error.

| Rd | Event | Role | Baseline MAE | Research candidate MAE | Research candidate top 3 | Public top 3 | Actual top 3 |
| ---: | --- | --- | ---: | ---: | --- | --- | --- |
| 1 | Australia | fit | 2.6364 | n/a | n/a | RUS / HAM / LEC | RUS / ANT / HAD |
| 2 | China | fit | 0.9091 | n/a | n/a | RUS / ANT / NOR | ANT / RUS / HAM |
| 3 | Japan | selection | 1.7273 | 1.6364 | ANT / RUS / LEC | ANT / RUS / LEC | ANT / RUS / PIA |
| 4 | Miami | selection | 2.9091 | 2.5455 | NOR / ANT / PIA | NOR / ANT / PIA | ANT / VER / LEC |
| 5 | Canada | calibration diagnostic | 2.0000 | 2.1818 | RUS / ANT / PIA | RUS / ANT / NOR | RUS / ANT / NOR |
| 6 | Monaco | calibration diagnostic | 2.5455 | 2.7273 | ANT / LEC / HAM | ANT / LEC / HAM | ANT / VER / HAM |
| 7 | Barcelona | replay diagnostic | 1.7273 | 1.7273 | RUS / NOR / LEC | RUS / PIA / LEC | RUS / HAM / ANT |
| 8 | Austria | replay diagnostic | 0.9091 | 1.0000 | RUS / ANT / HAM | RUS / ANT / HAM | RUS / LEC / HAM |
| 9 | Britain | replay diagnostic | 1.4545 | 1.3636 | ANT / HAM / VER | HAM / ANT / VER | ANT / LEC / HAM |

The full 198 driver-event rows are in
`docs/research/evidence/f1_qualifying_prediction_vs_reality_2026.csv`.

### Why position probabilities were not promoted

Finite Monte Carlo counts can produce exact zero cells. A fixed Jeffreys
pseudocount of 0.5 is therefore applied before temperature scaling. For field
size `N` and sample count `S`, `(count + 0.5) / (S + 0.5*N)` preserves the row
and column sums of the doubly stochastic position matrix while giving every
cell positive support.

Temperature 0.8 is fitted on R5-R6 only. On R7-R9 it changes:

- multiclass log loss: 2.847352 to 2.834999, better by 0.012354;
- top-10 ECE: 0.270163 to 0.256601, better by 0.013562;
- normalized Brier: 0.467761 to 0.467988, worse by 0.000226.

So calibration improves two diagnostics and worsens another. More importantly,
two independent calibration weekends are not enough for a promotion-grade
claim. The probability artifact exists and is code/source bound; the output is
kept as a diagnostic rather than discarded.

## 2. Race final-position prediction

### Main survival/order model

The main challenger factorizes the outcome into terminal-status dynamics and
conditional running order:

`P(terminal status, retirement time | causal evidence)`

times

`P(running order | survives, legal grid, causal evidence)`.

It uses a partial-pooled discrete-time hazard, conditional Bradley-Terry order,
shared team/power-unit/weather/incident shocks, and an exact legal assignment.
R1-R2 are development; R3-R4 selection; R5-R6 calibration monitoring; R7-R9
are the post-development replay block.

The research challenger's selected parameter pair has order-residual weight 0 and Plackett-Luce
temperature 0.08. On R3-R4 it scores 2.7727 versus 2.7273 for the grid, a 1.67%
loss rather than the required 5% gain. Across R3-R9 it scores 4.0260 versus
3.5844. On the replay block it scores 3.7879 versus 3.0909, 22.55% worse. Replay Kendall is
0.52092 versus 0.62771; terminal Brier is 0.17361 versus 0.16410; terminal log
loss is 0.52987 versus 0.50822. This is a current-replay empirical loss and is
sufficient reason to retain the baseline. It does not reject the model family
forever; formal promotion remains unevaluated because the prospective horizon
and calibration gates are not yet available.

| Rd | Event | Role | Grid MAE | Survival candidate MAE | Candidate top 3 | Actual top 3 |
| ---: | --- | --- | ---: | ---: | --- | --- |
| 3 | Japan | selection | 2.0000 | 2.2727 | ANT / RUS / PIA | ANT / PIA / LEC |
| 4 | Miami | selection | 3.4545 | 3.5455 | ANT / VER / LEC | ANT / NOR / PIA |
| 5 | Canada | calibration | 5.2727 | 5.3636 | RUS / ANT / NOR | ANT / HAM / VER |
| 6 | Monaco | calibration | 5.0909 | 5.6364 | ANT / VER / HAM | ANT / HAM / GAS |
| 7 | Barcelona | replay | 3.0909 | 3.2727 | RUS / HAM / ANT | HAM / RUS / NOR |
| 8 | Austria | replay | 1.8182 | 3.7273 | RUS / LEC / HAM | RUS / VER / ANT |
| 9 | Britain | replay | 4.3636 | 4.3636 | ANT / LEC / HAM | LEC / RUS / HAM |

The 154 R3-R9 survival rows are preserved separately in
`docs/research/evidence/f1_race_survival_prediction_vs_reality_2026.csv`.

### Certified alternative: all R1-R9

Historical provider Qualifying/practice state is mutable. A retrospective API
call cannot prove exactly what its first response would have been at an old
cutoff. That specific “first seen provider snapshot” cannot be manufactured
afterward.

The work did not stop there. A second ablation removes those unverifiable
current-event inputs. It uses only:

- a certified FIA final-grid document whose authoritative publication time is before race start;
- FIA driver identity;
- strictly prior race result and prior race team state.

Every one of R1-R9 now has certified post-grid/pre-race evidence. This is not a
JSON field certifying itself: the artifact pins the official FIA URL, repository
path, and SHA-256 for each of nine PDF files, reopens each PDF, parses its FIA
document metadata and complete grid, and cross-checks the parsed rows and
publication time against the capture record. For each event, the PDF header's
local date/time is interpreted with a pinned venue IANA timezone and converted
to UTC. The claimed publication time must be between the parsed PDF timestamp
and 300 seconds afterward, and both timestamps must precede race start. All
nine events pass: the lag is zero seconds for R1-R8 and 60 seconds for Britain.
The candidate is a
recency-weighted, partially pooled prior-race residual adjustment around the
legal grid. The four fixed profiles are descriptive follow-up profiles, not
retrospective promotion selection.

| Rd | Event | Role | Grid MAE | Certified candidate MAE | Candidate top 3 | Actual top 3 |
| ---: | --- | --- | ---: | ---: | --- | --- |
| 1 | Australia | development | 4.5455 | 4.5455 | RUS / ANT / HAD | RUS / ANT / LEC |
| 2 | China | development | 3.8182 | 3.8182 | ANT / RUS / HAM | ANT / RUS / HAM |
| 3 | Japan | selection | 2.0000 | 2.0000 | ANT / RUS / LEC | ANT / PIA / LEC |
| 4 | Miami | selection | 3.4545 | 3.4545 | ANT / VER / LEC | ANT / NOR / PIA |
| 5 | Canada | calibration | 5.2727 | 5.2727 | RUS / ANT / NOR | ANT / HAM / VER |
| 6 | Monaco | calibration | 5.0909 | 5.0909 | ANT / VER / HAM | ANT / HAM / GAS |
| 7 | Barcelona | replay | 3.0909 | 3.0909 | RUS / HAM / ANT | HAM / RUS / NOR |
| 8 | Austria | replay | 1.8182 | 1.9091 | RUS / LEC / HAM | RUS / VER / ANT |
| 9 | Britain | replay | 4.3636 | 4.3636 | ANT / LEC / HAM | LEC / RUS / HAM |

Across all nine events the certified candidate is 3.7273 versus 3.7172; replay
is 3.1212 versus 3.0909. This proves the alternative was implemented and
tested. It also shows that prior-race residual adjustment adds no useful edge
yet. Formal promotion remains unavailable because there are only two selection
and two calibration weekends, and this fixed profile matrix is post-development
descriptive evidence rather than a pristine prospective selection study.

The public all-round Race export now has exactly 198 rows in
`docs/research/evidence/f1_race_prediction_vs_reality_2026.csv`. It is explicitly
the certified grid/prior-state product; it is not silently mixed with the main
survival experiment.

## 3. Best estimated lap time

The user-facing target is the fastest valid representative Qualifying lap a
driver achieves by session end. It is not the sum of independently fastest
sectors, which is a theoretical lower bound and may not be driveable.

R1-R2 fit the point engine, R3-R6 calibrate intervals, and R7-R9 form a
target-isolated but post-development replay diagnostic.
Intervals target P05-P90, nominal mass 85%, and must pass coverage and width
gates at pooled, event-balanced, per-event, and weekend-format levels.

| Rd | Role | Baseline MAE (s) | Candidate MAE (s) | Candidate fastest | Actual fastest | Candidate coverage | Candidate width (s) |
| ---: | --- | ---: | ---: | --- | --- | ---: | ---: |
| 1 | fit | 0.699526 | 0.699526 | RUS | RUS | 21.05% | 0.600 |
| 2 | fit | 0.404333 | 0.404333 | RUS | ANT | 57.14% | 0.600 |
| 3 | calibration | 0.303299 | 0.303299 | ANT | ANT | 100.00% | 1.579 |
| 4 | calibration | 1.089261 | 0.649825 | NOR | ANT | 90.48% | 2.268 |
| 5 | calibration | 0.465479 | 0.438819 | RUS | RUS | 95.00% | 2.268 |
| 6 | calibration | 0.534214 | 0.534214 | ANT | ANT | 81.82% | 1.579 |
| 7 | replay | 0.393050 | 0.398388 | RUS | RUS | 90.91% | 1.834 |
| 8 | replay | 0.221589 | 0.219061 | RUS | RUS | 100.00% | 1.834 |
| 9 | replay | 0.499500 | 0.516307 | HAM | ANT | 81.82% | 1.834 |

Replay point MAE is 0.377919 s versus the retained baseline at 0.371380 s, a
1.76% loss. Candidate interval coverage is 90.91% versus 86.36%, but candidate
width is 1.8338 s versus 1.3950 s, 31.45% wider. Higher coverage purchased by a
much wider interval is not a better calibrated product; neither point nor
interval is promoted.

The raw rehearsal comparator is 0.805303 s on the replay block. A nested same-season Ridge
model reaches 0.484785 s, a real 39.8% improvement over that weak comparator,
but still loses the retained 0.371380 s champion. This is useful feature
engineering, not a basis for replacing the current output. Its forecasts are
mechanically frozen before target reads within the run, but the result is also
post-development and explicitly ineligible for formal prospective promotion.

The full 198 driver-event rows are in
`docs/research/evidence/f1_best_lap_prediction_vs_reality_2026.csv`.

### Telemetry sequence model

The arbitrary “20 events before training” rule was removed. The validated
source pack contains 9 independent weekends, 191 driver-event bags, and 550
correlated tensors. Integrity readiness and promotion power are now separate.

The first event-blocked sequence model uses 96 ordered features: six telemetry
channels by eight distance segments by mean and mean absolute gradient. Raw
event-balanced MAE is 0.820070 s. A causal source-shift baseline reaches
0.787567 s. Nested temporal Ridge selects zero correction in every fold, so its
output is also 0.787567 s. That is a trained negative incremental result, not a
refusal to run the model.

### True TCN and parameter audit

A true bounded residual TCN was trained with deterministic CPU execution. The
reference architecture has 274 trainable parameters, hidden width 4, kernel 3,
dilations 1 and 2, two static anchor features, Huber loss, bounded 1.5 s output,
and strictly expanding event folds. The primary post-development optimizer
profile uses learning rate 0.0005, weight decay 0.0001, seed 20260714, at most
200 epochs, and nested prior-event early stopping.

Its provenance is now explicit. D0 is the only pre-sensitivity control. D1,
D2, D3 and both D1/D4 shams or follow-ups were designed after results on these
same outer evaluation targets had already informed development, so their
hyperparameters are marked outer-target-informed and their design evidence is
promotion-ineligible. This is not current-fold target leakage: fitting, early
stopping, and inner selection for each outer event still use strictly earlier
events. It does mean the attractive D1 number is descriptive post-development
evidence, not an untouched validation result.

The 200-epoch setting is a ceiling, not an instruction to force 200 updates.
Prior-event early stopping freezes D1 at one epoch in all five outer folds;
the lower-capacity D2 reaches 36 epochs in one fold and one epoch in the other
four. That is evidence that extra optimization is not stable across weekends.
Forcing longer training after inspecting the same five outer targets would make
the number look more tuned, not make the experiment more independent.

Across R5-R9:

- source-shift baseline: 0.430620 s;
- direct reference TCN: 0.429903 s, better by 0.000717 s (0.17%);
- locked selected policy: 0.430620 s because every inner fold selected zero correction;
- fixed-seed repeat: 0.432008 s, worse than source shift;
- mean absolute event difference between reference and seed repeat: 0.002294 s;
- reference telemetry minus equal-architecture zero-telemetry sham: -0.002267 s on average.

The last number looks positive but is dominated by R7: the five event deltas
are approximately -0.000140, -0.000387, -0.011295, +0.000515, and -0.000027 s.
The apparent telemetry contribution is therefore neither stable nor
confirmatory. The reference gain of 0.000717 s is also smaller than the
0.002294 s seed variation.

| Profile | Key change | Event-balanced TCN MAE (s) | Delta vs source shift (s) |
| --- | --- | ---: | ---: |
| D0 control | learning rate 0.003, 60 epochs | 0.430631 | +0.000010 |
| D1 reference | learning rate 0.0005, seed 20260714 | 0.429903 | -0.000717 |
| D2 lower capacity | width 2, broad receptive field | 0.434371 | +0.003751 |
| D3 seed repeat | D1 with seed 20260715 | 0.432008 | +0.001387 |
| D1 sham | D1 with telemetry zeroed, static anchors retained | 0.432170 | +0.001549 |
| D4 posthoc | learning rate 0.0001 | 0.429787 | -0.000834 |
| D4 posthoc sham | D4 with telemetry zeroed | 0.429839 | -0.000782 |

D1-D3 were also post-development designs informed by earlier exposure to these
outer targets; D4 was created after the durable sensitivity was observed. All
are explicitly non-promotion evidence. The correct interpretation is
`evaluated_parameter_sensitive_inconclusive`, not “TCN rejected forever” and
not “TCN works.” The next serious test is a frozen configuration on an
independent future season/regime block. The current Best-Lap output remains the
retained baseline.

## 4. Live race intelligence and RL

RL is appropriate for pit, compound, and pace decisions because an action
changes future state and reward. It is usually not the right primary model for
passive Qualifying classification, final-position forecasting, Best-Lap
estimation, next-lap prediction, degradation, or terminal-status probability.
Those are supervised, state-estimation, ranking, or survival problems that can
feed an RL policy and simulator.

### Supervised live forecast

The frozen next-lap artifact contains 8,144 matched emitted forecasts:

- adaptive blend MAE: 0.537946 s;
- naive MAE: 0.549184 s;
- pure SSM MAE: 0.675550 s;
- blend-minus-naive event mean: -0.011238 s;
- event-bootstrap 95% CI: [-0.025452, +0.001032];
- bootstrap probability of improvement: 0.960432 from 500,000 event resamples, seed 20260711.

The blend is promising, but the interval crosses zero. The naive forecast is
therefore retained and the blend remains shadow evidence.

### State v6 / transition and replay v8 / legal-mask v3 RL evidence contract

The replay and learning contracts now distinguish three different evidence
levels that were previously conflated:

- behavior cloning: a causal observed physical action family; an unobserved
  conservative/aggressive pace mode remains a two-class partial label;
- offline Q-learning: exact observed pace mode, valid causal boundary, complete
  certified current and nonterminal-next legal-action masks, known-legal action,
  and every weighted reward component observed;
- propensity OPE: offline-Q eligibility plus a finite logged behavior
  probability in `(0, 1]`.

A row can train a supervised behavior model without being valid for Q-learning,
WIS, or doubly robust OPE. Missing propensity no longer incorrectly destroys
behavior-cloning or offline-Q rows, while synthetic or historical rows without
a propensity cannot enter OPE.

The full mask certificate now covers every input that can alter legal support:
race horizon, current compound, complete used-compound history, red status,
box-lap status, pit-lane status, nonempty available-compound inventory, and an
explicit forced-pit commitment state including known-none. It also requires
`mandatory_compound_change_required` to be explicitly known or certifiably
derived. Sporting deadlines are now derived from execution timing rather than
a tunable two-lap preference. `pit_now` puts the requested compound on the next
lap and therefore needs one remaining lap; `pit_next_lap` first completes one
lap and needs two. With a mandatory compound change outstanding, `stay_out`
remains legal with two laps left because a final-lap `pit_now` is still
possible, but is blocked with one. A pit that does not itself satisfy the rule
needs one additional lap beyond its execution delay. When tyre inventory is
known and contains no reachable satisfying compound, every completion-dependent
action is infeasible; missing inventory remains unknown evidence rather than a
false claim of an empty inventory. Terminal states expose no constraint-legal
action. These rules are required for both `s_t` and every nonterminal
`s_(t+1)`. The replay loader independently
rederives both masks and evidence payloads and rejects any stored mismatch. That
next-state requirement matters mathematically because a Bellman target uses
`max_a Q(s_(t+1), a)`; an uncertified next mask would let learning bootstrap
through an action that was never known to be legal.

Constraint legality and operational safety are separate concepts. If no
compliant action exists, the authoritative constraint-legal mask stays all
false; the system may expose a separately tagged conservative safety no-op to
avoid a runtime crash, but that no-op is not relabeled as legal. A simulator
may advance it only as an explicitly penalized emergency transition. It is
excluded from behavior cloning, Q-learning, OPE, and learned-policy selection.
This avoids the
previous mathematical error where an impossible state could silently create a
fake legal action and contaminate Bellman targets.

Physical pit stops are represented as one semi-Markov transition from the last
pre-stop running lap to the first non-pit post-stop lap. Pit-in-only events,
unknown compounds, duplicated laps, cross-event/driver adjacency, and illegal
mask/action combinations fail closed. Raw lap adjacency and the numeric
behavior propensity, reward-observation evidence, and full transition payload
are fingerprinted. Historical pit-lane messages are included as nine explicit
race-control inputs. `PIT LANE ENTRY OPEN/CLOSED` becomes causal only when the
message lap is strictly less than the state lap; same-lap and future messages
cannot alter the action mask, and pit-exit messages are not misread as pit-entry
permission.

The reward stored for a multi-lap transition is an undiscounted aggregate.
Consequently, `gamma ** elapsed_laps` would be mathematically inconsistent
unless per-lap rewards were also stored. The current finite-horizon contract
sets `gamma = 1`; any multi-lap replay with another discount is rejected.

The frozen replay audit now measures the blocker instead of describing it
abstractly. Across nine races it builds 9,506 causal transitions:

- behavior-cloning partial-label rows: 9,506;
- offline-Q rows: 0;
- propensity-OPE rows: 0;
- physical actions: 9,191 stay-out, 163 HARD pits, 93 SOFT pits, and 59 MEDIUM pits;
- compatible full action keys: 8 of 22 across four physical families;
- exact pace-mode keys: 0 of 22 because the historical feed never observed the
  conservative/aggressive mode;
- current-state mask blockers: race horizon 9,506, certified tyre-set inventory
  9,506, explicit forced-pit commitment state 9,506, and pit-lane status 7,857;
- nonterminal-next mask blockers: race horizon 9,336, certified tyre-set
  inventory 9,336, explicit forced-pit commitment state 9,336, and pit-lane
  status 7,685;
- fully observed composite rewards: 9,489; the remaining 17 lack one running-position endpoint;
- nonzero position-gain rewards: 1,424;
- states whose expanding history contains multiple used compounds: 5,939.

The source adapter now preserves FastF1 running position, physical `TyreLife`
when present, every completed-lap tyre-age fallback, expanding used-compound
history, observed race-clock endpoints, and an explicit known/unknown race
horizon. This repaired real state and reward information, but it did not invent
tyre-set inventory, pace-mode labels, or logged propensities.

Rolling-origin BC is trained only on strictly earlier races. It reaches 96.7473%
action-family accuracy, exactly the same as the most-frequent partial-label
baseline, and both score pit F1 = 0. The headline accuracy is therefore class
imbalance, not strategy intelligence. BC remains available as a warm start,
but it is not called a policy optimizer and is not promoted. No offline-Q or OPE
model is fitted on zero valid rows. The deterministic legal policy remains the
runtime fallback while legal inventory/mode/propensity evidence, calibrated
counterfactual scenarios, support-aware WIS/DR, and a prospective live shadow
are gathered.

## Evidence products and reproducibility

The notebook
`research/projects/F1/rising_qualification_prediction/Jupyter/f1_model_research_v2.ipynb`
uses eleven explicit paths and eleven exact SHA-256 plus schema assertions. It
contains no artifact glob or “latest file” discovery. Every artifact then
passes a recursive closure check over code, configuration, input manifests,
nested JSON references, and declared aggregate-manifest digests. A fresh
project-venv kernel executed all 9 code cells in order with no error outputs.

The final register reports 11/11 provenance closures passing: 111 direct
implementation declarations, 1,007 direct input declarations, 4,140 transitive
file-reference checks, 158 nested JSON references, and 41 aggregate-manifest
digests. These are validation checks summed across artifacts, not claims of
4,140 unique files.

Driver-level exports:

- Qualifying: 198 rows, R1-R9;
- Race certified grid/prior-state: 198 rows, R1-R9;
- Race main survival diagnostic: 154 rows, R3-R9;
- Best estimated lap: 198 rows, R1-R9.

The three user-facing non-live exports therefore contain 594 driver-event rows.
Round summaries, long-form audit comparisons, the JSON summary, immutable
artifact register, and comparison chart are under `docs/research/evidence/` and
`docs/research/assets/`.

The bounded Data Analytics manifest and snapshot were updated to the final
Race/TCN/Live decisions and passed artifact validation with status `ready`, two
datasets, and two canonical sources. The already-selected report surface was
not rendered a second time.

| Artifact | Schema | SHA-256 |
| --- | --- | --- |
| `qualifying/shared_latent_same_season_v2_20260713.json` | `f1_shared_qualifying_latent_event_block_v6` | `9a14686c8d1dcf44489d8af64955e4bd06e154bb5164ecedb79fec5faeae70da` |
| `qualifying/shared_latent_same_season_probability_audit_v4_20260714.json` | `f1_qualifying_probability_temperature_sinkhorn_audit_v3` | `082387f0afe451c1a5d1b9c0634e0545330e886ee8f0161aebb3bf8428c27a20` |
| `race_final_position/survival_order_same_season_v8_selector_safe_20260713.json` | `f1_race_survival_order_event_block_v8` | `92c0179d7312367ecda253c0a8b6427721267efe572b3051a9cfc9ed25d69639` |
| `race_final_position/certified_grid_prior_state_ablation_v1_20260714.json` | `f1_race_certified_grid_prior_ablation_v1` | `b85f0312eead971e8f1ea3669247d89091d556352ba21eca0b057f1ef2500dbd` |
| `best_estimated_lap/same_season_event_gated_v18_tcn_matrix_bound_20260714.json` | `f1_best_estimated_lap_shared_latent_v11` | `dcc0a2bc5d89aa1a89400598eb38f7dfca5334712b839f4fd46442d1c2029133` |
| `research/same_season_latent_residual_frozen_audit_v2_20260713.json` | `f1_same_season_latent_residual_research_v3` | `757c57d15facd1a789437db783899717da2b81707a41a3d250e9392c5d18b479` |
| `telemetry/prequal_telemetry_residual_sequence_v6_repository_bound_2026.json` | `f1_prequal_telemetry_residual_research_v4` | `1743e0a5a492f05ec0507092b192332e33f22102a585af45a1a051472c87e4f8` |
| `telemetry/prequal_telemetry_true_tcn_research_v2_2026.json` | `f1_prequal_telemetry_true_tcn_research_v2` | `9c98244bd4c854b86850bda00a8dfd2454a845a2cbf83b1944c21e3bff09f999` |
| `telemetry/prequal_telemetry_tcn_sensitivity_matrix_v1_2026.json` | `f1_prequal_telemetry_tcn_sensitivity_matrix_v1` | `d148d512376bc7c14479ff48f07c73181fe7cf033b5a2eea8936d0067ef35197` |
| `live_next_lap/2026_walk_forward_ssm_naive_emitted_forecast_v7_20260714.json` | `f1_live_next_lap_walk_forward_emitted_forecast_v3` | `13de7a53e2b5eda07a92acb2bb8dea5fbcff435565676796b052fa8831c84c53` |
| `live_strategy/live_strategy_replay_audit_v2_20260714.json` | `f1_live_strategy_replay_audit_v1` | `004a5028eedd335247548db7fe18187c24165eeed2f521704a2524d0845d044e` |

The hardened certified Race artifact additionally records payload result SHA
`df01978d428a9af9bd91d43d76145d902ed73e1c125e2a24810bdbcc3d85e818`
and input-manifest SHA
`d96ea9644505beb165b041a6de44dfa1907f6e7acca8a026a7315ba80b52a65b`.

### Decision rule going forward

No current result justifies silently replacing a public baseline. The next
high-value work is not blind hyperparameter search on the same nine events. It
is to freeze the current candidates and gather independent prospective event
blocks with first-seen snapshots, certified legality/inventory/propensity for
pit actions, and enough selection/calibration weekends to measure stability.
The implemented candidates and negative results stay available as explicit
research branches; they are not deleted because they lost one gate.

Suggested commit name: `docs(f1): publish corrected live feasibility evidence`
