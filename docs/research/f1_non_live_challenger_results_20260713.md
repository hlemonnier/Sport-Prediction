# F1 Non-Live Full Implementation and Evidence Report

Finalized: 2026-07-13 Europe/Paris. Audit population: completed 2026 rounds
1-9. This is a chronological research decision record, not a claim that every
new model is better than its retained baseline.

## Bottom line

The recommendation is implemented across Qualifying Prediction, Race Final
Position, and Best Estimated Lap. Optional LambdaRank/quantile paths and the
telemetry deep-model gate are also implemented. RL remains confined to Live
Race Intelligence.

Implementation completeness and model promotion are different outcomes:

| Mode | Locked comparison | Evidence decision |
| --- | --- | --- |
| Qualifying Prediction | Shared latent challenger MAE 2.9899 vs baseline 1.8687 over rounds 1-9 | Rejected; point and uncalibrated probability outputs are not promoted |
| Race Final Position | Stable-identity candidate MAE 4.3737 vs grid baseline 3.7172 over rounds 1-9 | Diagnostic only; worse on all four primary aggregates and blocked by missing same-product post-grid selection/calibration plus retrospective Qualifying evidence |
| Best Estimated Lap | Audit-five event MAE 0.4470 s vs baseline 0.4090 s | Point, interval, and deep paths remain diagnostic |
| Optional ranking/quantile | LightGBM/XGBoost paths ran with healthy native runtimes | Fail-closed diagnostic evidence only |
| Live Race Intelligence | RL remains the action/strategy tool | Outside this non-live rerun; no passive forecast was converted to RL |

No production policy is silently replaced. A completed challenger that loses
or lacks point-in-time evidence remains shadow/diagnostic.

## What is now implemented

### One shared Qualifying and Best-Lap engine

The two modes now consume the same event model and the same joint samples. The
engine owns the quality-aware rehearsal anchor, common event/session shift,
the frozen-selected team/driver residual path, valid-lap hurdle, nested
Q1/Q2/Q3 stages, and uncertainty distribution. The selector disables the
residual path when it fails selection-block evidence. Best Lap reads marginal
seconds; Qualifying applies valid/stage outcomes to every joint field draw and
converts those draws into legal official classifications.

The pre-boundary reference artifacts match on driver IDs, joint-sample count
and hash, model and model-manifest hashes, Best-Lap and Qualifying output
hashes, position-marginal hash, and shared-artifact hash for all nine events.
The cutoff-safe regeneration must reproduce those nine integrity fields per
round before this report is finalized.

The causal rehearsal record separates valid clean pace from deleted potential
and includes compatible sectors, lap dispersion, push-lap count, session
progress, track status/interruption, compound, tyre life/freshness, speed trap,
teammate/field-relative pace, missingness, uncertainty, and source provenance.
Deleted laps can influence latent potential but can never become legal targets.

Absolute target-season pace is fitted from target-season evidence. Older
seasons are restricted to weak invariant transition/reliability priors; old
constructor pace is not transferred as current pace.

### Race as survival times conditional order

The Race model implements

`P(status, retirement time | causal evidence) * P(running order | survival, evidence)`

rather than one final-position regressor. It contains:

- driver-conditioned discrete-time terminal hazards with team, power unit,
  driver-incident, circuit, weather, missed-practice and current-weekend inputs;
- cause and retirement-fraction distributions with explicit coarse-label
  handling;
- a regularized Bradley-Terry conditional order using the legal grid, signed
  Qualifying surprise, teammate/field-relative long-run pace, compound and
  tyre-age pace, adjusted degradation, representative stint length, evidence
  uncertainty, circuit mobility, and Sprint pace;
- shared team, power-unit, weather, incident and pace shocks;
- FIA-style complete classifications and exact minimum-expected-absolute-loss
  assignment.

`post_qualifying_pre_grid` and `post_grid_pre_race` are separate products. The
capture command stores document publication and local capture times, revisions,
document/raw hashes, penalties, pit-lane starts, withdrawals, DNS/eligibility
and source provenance. All nine 2026 evaluations use official FIA final-grid
documents whose publication timestamps predate the Race; rounds 2, 4, 5 and 7
contain one pit-lane starter. The local documents were captured retrospectively
on July 13, so this report does not call them contemporaneous first-seen
captures.

The final audit also fixed three silent integrity problems:

- car numbers are event-specific, so longitudinal `driver_id` is now the FIA
  abbreviation while car/provider IDs remain artifact provenance;
- an entrant absent from a provider Race file is no longer invented as a DNS;
  2025 round 9 is excluded because STR lacks authoritative nonstarter evidence;
- all 1,210 examined input files resolve through the same legacy path rules as
  the provider and enter the hashed input manifest.

Historical provider Qualifying/practice snapshots are retrospective captures,
not verified first-seen files. Their capture semantics are recorded explicitly
and block promotion. The official 2026 final grids do have verified pre-race
publication times. This distinction is intentional.

### Optional and deep paths

OpenMP, XGBoost 3.3.0, and LightGBM 4.6.0 load successfully. Event-grouped
LambdaRank and LightGBM quantile challengers run only on chronological blocks
with exact champion-artifact alignment. Telemetry v2 validates 550 fixed-shape,
distance-normalized tensors across 9 independent events and 191 driver-events,
with zero cutoff, missing-file, SHA, shape, schema, or content failures. The
declared minimum is 20 events, so the TCN remains blocked.

## Prediction versus reality by round

Every artifact contains full driver-level rows. These tables give the complete
event-level comparison and the most interpretable predicted-versus-actual
leaders.

### Qualifying Prediction

| Rd | Event | Base MAE | Shared MAE | Base Kendall | Shared Kendall | Predicted top 3 | Actual top 3 |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | Australia | 2.636 | 2.545 | .671 | .680 | RUS / HAM / LEC | RUS / ANT / HAD |
| 2 | China | .909 | .818 | .896 | .905 | RUS / ANT / HAM | ANT / RUS / HAM |
| 3 | Japan | 1.727 | 3.909 | .801 | .550 | HAM / ANT / HAD | ANT / RUS / PIA |
| 4 | Miami | 2.909 | 3.000 | .610 | .602 | LEC / ANT / PIA | ANT / VER / LEC |
| 5 | Canada | 2.000 | 3.182 | .758 | .593 | RUS / LEC / HAM | RUS / ANT / NOR |
| 6 | Monaco | 2.545 | 2.727 | .697 | .662 | LEC / HAM / ANT | ANT / VER / HAM |
| 7 | Barcelona | 1.727 | 4.818 | .801 | .411 | PIA / LEC / HAD | RUS / HAM / ANT |
| 8 | Austria | .909 | 4.182 | .887 | .463 | HAM / LEC / HAD | RUS / LEC / HAM |
| 9 | Britain | 1.455 | 1.727 | .835 | .801 | HAM / ANT / NOR | ANT / LEC / HAM |
| **Mean** |  | **1.869** | **2.990** | **.773** | **.630** |  |  |

Across all nine rounds, pole hit is 66.67% vs 22.22%, top-three overlap
59.26% vs 51.85%, and top-ten overlap 90.00% vs 84.44%. On the locked audit
rounds 5-9, challenger-minus-baseline MAE is `+1.6000`, 95% CI
`[+0.4182, +2.7818]`, with 0% improvement probability. The rejection is
decisive.

Stage diagnostics cover 198 driver-rounds:

| Output | Brier | Log loss | Mean probability | Observed |
| --- | ---: | ---: | ---: | ---: |
| Valid classified lap | .031925 | .269176 | .912116 | .984848 |
| Reaches Q2 | .174811 | .531151 | .724751 | .722222 |
| Reaches Q3 | .157185 | .487714 | .454545 | .444444 |

The joint position matrix is legal, but it is explicitly uncalibrated and
fail-closed.

### Best Estimated Lap

| Rd | Role / rehearsal | Candidate MAE (s) | Baseline MAE (s) | Predicted fastest | Actual fastest | Predicted top 3 | Actual top 3 |
| ---: | --- | ---: | ---: | --- | --- | --- | --- |
| 1 | Fit / FP3 | .555760 | .299796 | RUS 77.912 | RUS 78.518 | RUS / HAM / LEC | RUS / ANT / HAD |
| 2 | Fit / Sprint Q | .456453 | .491958 | RUS 91.061 | ANT 92.064 | RUS / ANT / NOR | ANT / RUS / HAM |
| 3 | Calibration / FP3 | .285753 | .294106 | ANT 88.700 | ANT 88.778 | ANT / RUS / HAM | ANT / RUS / PIA |
| 4 | Calibration / Sprint Q | .621836 | .567835 | NOR 87.440 | ANT 87.798 | NOR / ANT / PIA | ANT / VER / LEC |
| 5 | Audit / Sprint Q | .393693 | .282077 | RUS 72.492 | RUS 72.578 | RUS / ANT / HAM | RUS / ANT / NOR |
| 6 | Audit / FP3 | .486679 | .532485 | ANT 72.046 | ANT 72.051 | ANT / LEC / HAM | ANT / VER / HAM |
| 7 | Audit / FP3 | .556935 | .534999 | RUS 75.074 | RUS 74.679 | RUS / LEC / PIA | RUS / HAM / ANT |
| 8 | Audit / FP3 | .269948 | .223575 | RUS 66.357 | RUS 66.113 | RUS / HAM / ANT | RUS / LEC / HAM |
| 9 | Audit / Sprint Q | .527984 | .472047 | ANT 87.900 | ANT 88.111 | ANT / HAM / VER | ANT / LEC / HAM |

The promotion aggregate is the untouched audit block, rounds 5-9: event-mean
MAE `0.447048 s` vs `0.409037 s` baseline, or 9.29% worse. Row-weighted MAE is
`0.487791 s`; Spearman is `.930226`; fastest hit is 100% vs 80%; top-three
overlap is 60.00% vs 66.67%. P05/P90 coverage is 97.27% with `2.714819 s`
width, versus `1.999628 s` baseline width. The interval is too wide and
overcovers the nominal 85% target.

Paired delta is `+0.038011 s`, CI `[-0.008935, +0.080631]`, improvement
probability 5.50%. Point, interval, and deep promotion all fail closed.

### Race Final Position

<!-- RACE_V4_START -->
| Rd | Event | Base MAE | Candidate MAE | Base Kendall | Candidate Kendall | Base/Candidate Brier | Predicted top 3 | Actual top 3 |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | Australia | 4.545 | 4.545 | .385 | .385 | .1885 / .1850 | RUS / ANT / HAD | RUS / ANT / LEC |
| 2 | China | 3.818 | 5.000 | .489 | .342 | .2535 / .2666 | LEC / HAM / ANT | ANT / RUS / HAM |
| 3 | Japan | 2.000 | 2.091 | .784 | .758 | .0851 / .0755 | ANT / RUS / NOR | ANT / PIA / LEC |
| 4 | Miami | 3.455 | 3.727 | .541 | .558 | .1502 / .1508 | ANT / LEC / NOR | ANT / NOR / PIA |
| 5 | Canada | 5.273 | 5.182 | .359 | .359 | .2169 / .2047 | RUS / ANT / HAM | ANT / HAM / VER |
| 6 | Monaco | 5.091 | 7.273 | .316 | -.022 | .2494 / .4163 | LEC / HAM / NOR | ANT / HAM / GAS |
| 7 | Barcelona | 3.091 | 4.091 | .619 | .472 | .2411 / .2319 | RUS / PIA / ANT | HAM / RUS / NOR |
| 8 | Austria | 1.818 | 2.909 | .801 | .628 | .1400 / .1680 | LEC / RUS / NOR | RUS / VER / ANT |
| 9 | Britain | 4.364 | 4.545 | .463 | .489 | .1177 / .1181 | ANT / HAM / LEC | LEC / RUS / HAM |
| **Mean** |  | **3.717** | **4.374** | **.529** | **.441** | **.1825 / .2019** |  |  |

Candidate MAE is 17.66% worse. Terminal log loss also worsens from `.562626`
to `.604372`; retirement-fraction MAE is `.256683` and terminal ECE is
`.136525`. Canada is the only round with a lower candidate position MAE;
Australia ties and the other seven rounds regress.

All nine point outputs are legal 22-driver permutations, use an official FIA
final-grid document, have complete target coverage, and attach Race truth only
after inference freeze. The artifact uses stable FIA abbreviations for
longitudinal identity while retaining provider/car-number provenance. It
content-verifies all 1,210 input files and records 56 event-model fits plus 192
safe cache hits.

Promotion is not evaluated: the `post_grid_pre_race` product lacks same-product
selection and calibration blocks, and the historical Qualifying inputs are
retrospective rather than verified first-seen snapshots. Independently of
those evidence blockers, the candidate loses all four aggregate comparisons:
position MAE, Kendall, terminal Brier, and terminal log loss. Production is
therefore unchanged.
<!-- RACE_V4_END -->

## Optional challenger evidence

- LightGBM LambdaRank: MAE `2.290909`, Kendall `.745455` vs the rejected shared
  champion `3.327273` / `.586147`, but the retained rehearsal baseline is still
  better at `1.727273`. CI versus the shared champion crosses zero.
- XGBoost LambdaRank: MAE `3.127273`, Kendall `.600000`; Sprint performance
  worsens and the paired CI crosses zero.
- LightGBM raw quantile: MAE `.507133 s` vs shared champion `.487791 s`, 70.00%
  coverage vs 97.27%, and `1.595980 s` width vs `2.714819 s`. It is raw,
  non-conformal diagnostic output.

Overall decision: `fail_closed_diagnostic_evidence_only`; production remains
unchanged.

## Evidence register

| Mode | Local immutable artifact | Schema | SHA-256 |
| --- | --- | --- | --- |
| Qualifying | `artifacts/backtests/f1/qualifying/shared_latent_v6_20260713.json` | `f1_shared_qualifying_latent_event_block_v3` | pending cutoff-safe regeneration |
| Best Lap | `artifacts/backtests/f1/best_estimated_lap/shared_latent_v5_20260713.json` | `f1_best_estimated_lap_shared_latent_v4` | pending cutoff-safe regeneration |
| Race | `artifacts/backtests/f1/race_final_position/survival_order_v4_20260713.json` | `f1_race_survival_order_event_block_v3` | `2e8c8448af407b4d86ec38e34121c3f427556fdb5092ceb7ecbb2f3ed71d6c6e` |
| Optional | `artifacts/backtests/f1/optional_models/2026_event_block_challengers_v3_20260713.json` | `f1_optional_non_live_event_block_evidence_v2` | `bd9edafff0b70297e7a547426389d0e7a9ed946f37c49d791fc634d361a46f6a` |
| Telemetry audit | `artifacts/backtests/f1/telemetry/prequal_cache_audit_v2_20260713.json` | `f1_prequal_telemetry_cache_audit_v2` | `56a994a43cc025b0b63198bd6951eb6ecfe814d7c123194fdaf89e77183fc733` |

The large run files remain local and ignored by repository policy. Their exact
hashes and decisions are versioned here. The Qualifying and Best-Lap artifacts
contain nine round-level records and 198 driver-level prediction/reality rows,
SHA-256 inventories for accessed inputs and selected implementation files, and
embedded run, partition, and protocol metadata. Per-event shared forecasts hash
the fitted model, training partition, joint samples, and mode outputs. Race v4
additionally records separate canonical input-manifest, implementation,
configuration, and protocol hashes. The evidence register pins every complete
artifact by its whole-file SHA-256.

## Validation

Final full-suite counts are inserted after the cutoff-safe shared-model reruns.
Focused Race protocol, survival, grid and capture validation currently passes
81 tests. Race v4 completed in `8391.57 s` wall time under the bounded
single-process policy.

Primary technical references: [FastF1](https://docs.fastf1.dev/core.html),
[OpenF1](https://openf1.org/docs/),
[XGBoost learning-to-rank](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html),
and [LightGBM parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html).

Suggested commit name: `docs(f1): record final non-live evidence decisions`
