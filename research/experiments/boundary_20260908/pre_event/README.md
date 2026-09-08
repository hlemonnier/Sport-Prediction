# Dynamic hierarchical qualifying correction — negative result

The frozen season-reset model did **not** beat the strongest verified qualifying comparator. It should remain research evidence; no production change or promotion is justified by this experiment.

One mechanism, twelve predefined configurations: Huber regression of practice-to-qualifying rank residuals, exponentially weighted within the current season, with separate penalized driver and team effects. It resets each January. Sixteen eligible 2023 events selected an eight-event half-life, penalty 0.5 and correction strength 1. The selection was serialized before the runner opened a 2024 qualifying target. Six 2023 sprint weekends lacked a causal, target-aligned FP3/Sprint Qualifying rehearsal and were excluded. All 24 events in each of 2024 and 2025 and the nine locally available 2026 events were scored with complete qualifying rosters. All these dates were previously exposed; the transfer results are retrospective.

## Results

MAE is mean absolute **rank-position** error, first averaged over the complete event roster and then equally over events. Every comparator uses the exact same target rows. The baseline and Q1/Q2 comparator forecasts are the immutable outputs from the preceding `performance_20260907` experiment.

| Period | Events / drivers | Rehearsal baseline | Fixed blend Q1 | Prior ridge Q2 | Dynamic candidate |
|---|---:|---:|---:|---:|---:|
| 2024 | 24 / 479 | 3.230044 | 2.980263 | 2.763377 | 2.880044 |
| 2025 | 24 / 480 | 3.366667 | 3.200000 | 3.054167 | 3.129167 |
| 2024–2025 | 48 / 959 | 3.298355 | 3.090132 | 2.908772 | 3.004605 |
| Exposed 2026 | 9 / 198 | 1.868687 | 1.848485 | 2.010101 | 2.050505 |

Against Q2, the predefined primary comparator, pooled 2024–2025 deterioration is **+0.095833 positions**. The paired-event bootstrap 95% interval is `[+0.006250, +0.187500]`; the circular three-event block interval is `[-0.004167, +0.189583]`. It wins 17 events, ties nine and loses 22; every leave-one-event-out pooled delta remains positive (`+0.080851` to `+0.112766`).

In 2026 the candidate is **+0.202020 positions** worse than Q1, the strongest preceding 2026 point result, and +0.040404 worse than Q2. The paired-event interval against Q1 is approximately `[0, +0.464646]`; its three-event block interval is `[+0.070707, +0.353535]`. It wins two of nine events against Q1, ties one and loses six. The predefined material-success screen—at least 0.15 positions better than Q2 historically and no 2026 regression against Q1—fails both requirements. Bootstrap intervals are descriptive resampling intervals, not prospective confirmation or posterior probabilities.

The candidate improves over the rehearsal baseline by 0.293750 positions historically, but this does not establish a new best model. Resetting state did not fix contemporary overcorrection. Selection chose the longest memory and weakest regularization available, which is consistent with information loss from discarding prior seasons; that interpretation is a hypothesis, not a separate tested result.

## Reproduction and verification

Run from the repository root, with a new output directory to preserve this result:

```sh
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python research/experiments/boundary_20260908/pre_event/run_experiment.py --output artifacts/research/boundary_20260908/pre_event/reproduction
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python -m pytest -q -p no:cacheprovider research/experiments/boundary_20260908/pre_event/test_experiment.py
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python research/experiments/boundary_20260908/pre_event/verify_results.py
```

Four regression tests passed. The separate verifier passed 86 implementation hashes, 428 input hashes, two output hashes, exact prior forecast/target pairing, all 73 pre-target forecast hashes, 468 legal rank permutations and 1,404 event metric values across 1,477 rows. It reconstructed selection and all paired summary statistics. All 234 fitted histories contain only strictly earlier events of the target season. No variant failed. The largest recorded objective-gradient component was `3.493436e-6`; some IRLS fits reached the fixed 40-iteration limit, so the evidence does not claim that every fit met the coefficient stopping tolerance.

The first execution stopped before fitting because five files in the previous experiment's broad source manifest had subsequently changed during the production integration. The retry compares the unchanged old prediction bytes and unchanged target bytes, explicitly records those five historical/current hash pairs, and binds the sources actually loaded by the new run. It does not claim the old broad source closure remains current. Both execution logs are retained; no old artifact was modified.

Canonical result: `artifacts/research/boundary_20260908/pre_event/cycle1/results.json`

SHA256: `11f5ea871774c20ac0318b15dbf14ceb3468d8b9153fa1c793e2002c73fc8e0e`

Additional evidence is adjacent: `predictions.csv`, `selection_lock.json`, `verification.json`. The result contains the exact frozen specification, every tried configuration and 2023 score, exclusions, fit coefficients and diagnostics, and source/input/output hashes. The verifier checks recorded forecast bindings and chronology; confirming execution ordering also requires reading the runner. No probability forecast is issued by this rank-only experiment, so probability scoring is not applicable.

## Next hypothesis, not yet evaluated

The next larger opportunity is to change the measurement of practice pace, rather than fit another regression on its single rank summary: infer relative driver pace from multiple clean laps matched by session clock and tyre compound, with common track-evolution effects and partial pooling between team and driver. Pooling may preserve useful historical information while a measurement-error model reduces the weight of a rehearsal whose apparent rank rests on sparse or incomparable laps. Fuel-related effects must remain uncertainty unless independently observed; they cannot be identified merely by renaming a residual.

A bounded follow-up would freeze one such measurement model and a comparator before scoring, use 2023 alone for tuning, and retain the complete roster with an explicit Q2/Q1 fallback when common support is absent. This is motivated by the failed rank-only correction and has not been evaluated. Acquiring causal FP3/Sprint Qualifying inputs for later 2026 rounds is needed before measuring contemporary transfer beyond the already exposed nine-event cache; the recently downloaded race-only inputs do not supply that horizon.

Suggested commit title: `research(f1): record dynamic qualifying correction experiment`
