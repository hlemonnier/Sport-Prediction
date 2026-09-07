# Pre-event performance cycle 1 — 2026-09-07

The learned rank and lap corrections improve the older chronological replays,
but both regress on the exposed 2026 events. They are research candidates; no
model, serving default, maturity record, or previously frozen evidence changed.

`specification.json` froze four mechanisms before the first experiment target
read. There was no hyperparameter search, per-event oracle, or winner selection.
Fixed parameters were retained after every result. Each forecast is hashed before
the current target is opened. Learned models see only earlier complete events,
with preprocessing learned from the same training rows. All driver rows in an
event share total weight one. The fixed blend and ridge head emit complete rank
permutations; this cycle emits no probabilities or calibrated intervals.

## Results

MAE is an equally weighted mean across events. Lower is better. Rank MAE is in
positions; lap MAE is in seconds. These are retrospective chronological research
results, not prospective confirmation.

| Point method | 2024, 24 events | 2025, 24 events | 2024–25 pooled, 48 events | Exposed 2026, 9 events |
|---|---:|---:|---:|---:|
| Latest rehearsal rank baseline | 3.230044 | 3.366667 | 3.298355 | 1.868687 |
| Q1 fixed practice/history blend | 2.980263 | 3.200000 | 3.090132 | 1.848485 |
| Q2 ridge rank residual | 2.763377 | 3.054167 | 2.908772 | 2.010101 |
| Same-season source-shift lap baseline | 2.674027 | 1.246308 | 1.960168 | 0.513577 |
| L1 fractional source shift | 2.588292 | 1.236985 | 1.912639 | 0.523419 |
| L2 Huber context residual | 1.740858 | 1.152458 | 1.446658 | 0.595635 |

| Challenger, pooled 2024–25 | Mean delta | Paired event bootstrap 95% interval | Three-event circular block interval | Worst leave-one-event-out delta | Events improved |
|---|---:|---|---|---:|---:|
| Q1 | -0.208224 | [-0.318640, -0.118531] | [-0.312390, -0.124890] | -0.170101 | 29/48 |
| Q2 | -0.389583 | [-0.570833, -0.245833] | [-0.581250, -0.243750] | -0.325532 | 38/48 |
| L1 | -0.047529 | [-0.097904, -0.003475] | [-0.102728, 0.002556] | -0.032227 | 24/48 |
| L2 | -0.513510 | [-1.071047, -0.080729] | [-1.221599, -0.036025] | -0.336095 | 37/48 |

All intervals use 20,000 draws and seed 20260907. Bootstrap frequencies in the
JSON are descriptive resampling frequencies, not posterior probabilities.
They do not adjust for broader historical research exposure or multiple models.

The lap headline is strongly affected by weather transitions: three wet
rehearsals have baseline/candidate L2 MAE 10.410650/3.572380 seconds, versus
1.396802/1.304943 seconds over the other 45 events. This grouping was inspected
after scoring; it is a diagnosis, not a newly selected model or test population.
The smaller 2025 L2 gain is -0.093850 seconds with block interval
[-0.175758, -0.011695].

Transfer to 2026 is unproven and presently unfavorable for the learned models.
Q2's 2026 delta is +0.141414 positions, event interval [0.000000, 0.323232],
block interval [-0.010101, 0.333333]. L2's delta is +0.082058 seconds, event
interval [-0.092731, 0.265788], block interval [-0.031555, 0.219801]. The fixed
rank blend's small 2026 delta, -0.020202, has an interval spanning zero. The
2026 rehearsal baseline is already substantially stronger than the older-season
baseline. Applying older-season corrections unchanged does not establish a
current performance improvement.

## Population and limitations

- Local inventory: 2022=22, 2023=22, 2024=24, 2025=24, 2026=9 weekends.
- Scored: 92 complete inference/target rosters. Nine 2022–23 sprint weekends
  have no causal FP3/Sprint Qualifying before GP qualifying and are explicitly
  unsupported. Their post-qualifying FP2 sessions are never read as inputs.
- Primary 2024–25: 959 qualifying roster rows, including one 19-entry source
  event; 936 common lap-scoring rows. Eleven targets and thirteen valid-clean
  rehearsal anchors are missing, with one overlap. All methods use the same
  finite lap population; this is conditional lap performance, not a full-field
  no-valid-lap probability model.
- Exposed 2026: 198 qualifying rows, 191 matched lap rows; three target and six
  rehearsal absences overlap on two rows. All rows remain in the prediction CSV.
- No roster is inferred from the current target. Same roster equality is checked
  only after the forecast has been frozen. No roster mismatch exclusions occurred.
- The lap comparator is the existing analytic source-shift equation, expanding
  within each season. It is not the prior report's frozen R7–R9 fitted object;
  these lap MAEs should not be compared to that object's differently conditioned
  audit summary.
- Snapshot capture is retrospective. Strict event order establishes the
  computation's chronology; it does not certify that all historical raw bytes
  were available at the original forecast time.

Further work should explain the 2026 transfer failure before fitting stronger
models: predeclare a same-season adaptation test, preserve all four results
above, and evaluate subsequent unexposed events. Covering old sprint formats
requires a separately defined FP1-only model; additional downloads alone do not
create a pre-qualifying FP3 session. No such extra candidate was tried here.

## Reproduction and integrity

Run from the repository root, using a new output directory for a repeat:

```sh
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python research/experiments/performance_20260907/pre_event/run_experiment.py --output artifacts/research/performance_20260907/pre_event/cycle1_reproduction
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python research/experiments/performance_20260907/pre_event/verify_results.py --directory artifacts/research/performance_20260907/pre_event/cycle1_reproduction
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python -m pytest -q -p no:cacheprovider research/experiments/performance_20260907/pre_event/test_experiment.py
```

Canonical new results: `artifacts/research/performance_20260907/pre_event/cycle1/`.
`results.json` SHA256: `b3477a1e90f167dd42693e9045a5526700c1e5e726f9aaf01baf86c6af27adba`.
Specification SHA256: `d315482d2e8ec772b861dbc942ed3eb4e96f23c413f8b00f6fa33f5eef0ff434`.

The initial failed execution log is preserved beside the results directory.
Older metadata contained paths from a moved checkout; the research-only adapter
now resolves an existing file of the same name in the same event directory.
Generated PyTorch module names without physical source files are excluded from
source-file hashing. These execution repairs changed no candidate parameter.

Verification passed: 84 loaded implementation files and 545 inputs hash-matched;
92 frozen forecast hashes recomputed; 276 legal rank permutations and 552 event
MAEs independently checked. Five experiment tests passed. Final fitting had no
errors or convergence warnings. Early insufficient-history fallbacks are recorded
for both learned candidates. Detailed event results, coefficients, training event
keys, baseline error groups, all four candidates, and exclusions are in the JSON.

Suggested commit: `research(f1): benchmark causal pre-event corrections across seasons`.
