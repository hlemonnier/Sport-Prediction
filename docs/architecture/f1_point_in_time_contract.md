# F1 Point-in-Time Prediction Contract

Status: authoritative research contract as of 2026-07-11. A model run that
cannot prove its target, cutoff, source availability, and complete field is not
eligible for promotion.

## System Boundaries

The repository contains four different prediction problems. They share data
contracts, but they do not share a label or an information set.

| User-facing mode | Target | Earliest useful same-weekend evidence | Forbidden evidence |
| --- | --- | --- | --- |
| Qualifying Prediction | official Grand Prix qualifying classification | completed sessions before Grand Prix qualifying | Q classification, final grid, race data |
| Race Final Position | official race classification and terminal status | named pre-FP, pre-Q, post-Q, or final-grid horizon | any session or grid revision published after the declared horizon |
| Best Estimated Lap Time | best valid representative qualifying lap achieved by session end | target-aligned rehearsal laps available by the declared cutoff | Q target laps; theoretical sector floors relabelled as achievable estimates |
| Live Race Intelligence | named next-lap/degradation/order/status forecast or legal pit/compound/pace decision | observations whose event timestamp is at or before replay time | later laps, global events learned after replay time, or any action outside the legal mask |

Qualifying Prediction output can become provisional Race Final Position context
before Q. It is not
an official grid. Qualifying classification, provisional grid, final grid,
pit-lane start, DNS, and actual start order remain separate states.

Live forecasts and decisions are different estimands. Forecasting observable
future state does not use an RL objective. RL is eligible only for the
counterfactual pit, compound, and pace decision subcontracts after the legal
mask, calibrated simulator, MPC baseline, locked OPE, and shadow gates pass.

## Season-Aware Weekend DAGs

The executable contract is `packages/f1/domain/weekend.py`.

| Era | Canonical order | Sprint grid source | Grand Prix grid source |
| --- | --- | --- | --- |
| Standard | FP1 -> FP2 -> FP3 -> Q -> Race | n/a | Q classification, then later grid decisions |
| 2021-2022 Sprint | FP1 -> Q -> FP2 -> Sprint -> Race | Q | Sprint classification, then later grid decisions |
| 2023 Sprint | FP1 -> Q -> SQ -> Sprint -> Race | SQ | Q classification, then later grid decisions |
| 2024+ Sprint | FP1 -> SQ -> Sprint -> Q -> Race | SQ | Q classification, then later grid decisions |

For 2026 the field contract is 22 eligible cars. Six are eliminated after Q1
and Q2 (and after SQ1 and SQ2 on Sprint weekends), leaving ten in Q3/SQ3. The
current Sprint format has separate SQ-to-Sprint and Q-to-Race parc-ferme
windows; setup freedom reopens after the Sprint and before Q.

Regulatory anchors: [FIA 2026 Sporting Regulations, Issue 07](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_07_-_2026-06-25.pdf),
[FIA regulations archive](https://www.fia.com/regulation/category/110), and the
[official 2026 Sprint calendar](https://www.formula1.com/en/latest/article/formula-1-and-fia-announce-2026-sprint-calendar.3PyLPAazrBNe8kQIS3wOfY.3PyLPAazrBNe8kQIS3wOfY).

## Availability Matrix

Every feature row carries a requested session cutoff. Providers resolve it
against the actual weekend format and only admit completed sessions with a
classification. `auto` means the latest whole-session boundary before the
target; it never means every file currently on disk.

| Artifact horizon | Same-event evidence allowed | Grid semantics |
| --- | --- | --- |
| `pre_weekend` / `pre_fp_provisional` | no current-weekend session | predicted grid only |
| `post_fp1` | FP1 | predicted grid only |
| `post_fp2` | FP1, FP2 on a standard weekend; invalid on current Sprint format | predicted grid only |
| `post_fp3_pre_q` | FP1-FP3 on a standard weekend | predicted grid only |
| `post_sprint_pre_q` | FP1, SQ, Sprint on a 2024+ Sprint weekend | predicted grid only |
| `post_qualifying` | all target-safe pre-Q evidence plus Q classification | qualifying fallback, not a final grid |
| `post_grid_pre_race` | pre-race evidence plus a separately published official grid | penalty-adjusted official grid |
| `live_race(t)` | pre-race evidence plus globally timestamped observations through `t` | actual start state plus causal live updates |

A named cutoff that does not exist in the weekend format fails closed. A
mutable retrospective API cannot claim arbitrary `prediction_as_of` replay
unless an immutable first-seen snapshot proves what was available then.

## Practice and Session Taxonomy

The shared provider contract excludes inaccurate, deleted, pit-in, pit-out,
yellow, Safety Car, VSC, and red-flag laps from clean pace. Counts remain in the
artifact so low-quality sessions are observable.

Qualifying-simulation and race-simulation labels are heuristics, not ground
truth. They must be accompanied by evidence counts/shares and may not be
invented by falling back to arbitrary laps. Long-run slope is labelled raw: it
still combines tyre degradation, fuel burn, traffic, driver management, and
track evolution. No universal fuel correction is assumed.

## Roster and Target Contract

The target classification owns the training roster. Current pre-qualifying
practice uses the latest completed eligible session roster, then carries the
most recent earlier participant only when a constructor's two-car seat count
is incomplete. This keeps a car that missed the latest session while excluding
superseded reserve drivers. Practice is left-joined so target drivers without
representative laps retain explicit missingness. Current race scoring starts
from the qualifying or official-grid roster. For training/evaluation target
normalization only, authoritative classification rows with no numeric position
receive stable tail ranks in source order. That does not authorize a serving
forecast: DNS, DSQ, withdrawn, and unclassified participants are explicitly
unavailable, while retired or stopped cars may remain eligible for an official
Race classification. A complete-field evaluation requires exact one-to-one
roster equality: predictions may neither omit target drivers nor add reserves
or other non-target drivers.

## Frozen Baselines and Evaluation

The 2026 primary arm is same-season walk-forward: for round `r`, training may
use only completed 2026 events before `r`. 2025 is not silently pooled into the
new-regulation regime. Older-season transfer is a separate, predeclared
ablation whose result must stand on its own.

Each candidate is compared against frozen, named baselines on identical event
blocks and cutoffs. At minimum report:

- full-field MAE/RMSE, exact-position rate, top-3/top-5/top-10 accuracy, and
  Kendall tau-b;
- winner/top-3/top-10 log loss, Brier score, expected calibration error, and
  probability-sum diagnostics;
- interval coverage and width when uncertainty intervals are emitted;
- event-level bootstrap or paired confidence intervals, plus worst-event and
  regime slices;
- separate chronological event blocks for model selection, probability
  calibration, and final untouched audit.

Green unit tests are necessary but do not establish predictive edge. A new
feature, model family, circuit prior, weather adjustment, CNN, DL model, or RL
policy stays experimental unless its locked, immutable artifact beats the
frozen baseline under predeclared gates without a material calibration or tail
risk regression.

The current target-specific decisions and rejected experiments are recorded in
[`docs/research/f1_2026_walk_forward_experiment_report.md`](../research/f1_2026_walk_forward_experiment_report.md).

## 2026 Calendar State

The local snapshot and official calendar must agree before a locked run. As of
2026-07-11, the British Grand Prix on 2026-07-05 is the latest completed event
and official round 9; Belgium on 2026-07-17 to 2026-07-19 is next. Bahrain and
Saudi Arabia did not take place in April, so preseason numbering must not be
used as ground truth. Sources: [official 2026 calendar](https://www.formula1.com/en/racing/2026),
[British Grand Prix result page](https://www.formula1.com/en/racing/2026/great-britain),
[FIA final British race classification](https://www.fia.com/system/files/decision-document/2026_british_grand_prix_-_final_race_classification.pdf), and
[FIA calendar update](https://www.fia.com/news/bahrain-and-saudi-arabian-grands-prix-will-not-take-place-april).

Suggested commit name: `feat(f1): enforce season-aware point-in-time prediction contracts`
