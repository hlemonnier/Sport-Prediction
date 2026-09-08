# Practice information and pairwise race ranking — selection failed

Neither new candidate passed the frozen selection gate against strong same-horizon references. The pairwise model improves over qualifying order, but loses to the previously selected Huber model and its own otherwise-identical model without practice features. No candidate was evaluated on 2024–2026, and production is unchanged.

The study retains the earlier race experiment's post-qualifying, provisional-grid horizon and exact conditional cohort. There are 21 training events in 2022 and 20 selection events in 2023, with 820 drivers across both years. Prior exclusions are fixed; no additional exclusion depends on a new candidate's result. The baseline uses qualifying order, not later grid penalties. The complete classification/roster requirements mean these results are not universal all-race estimates.

| 2023 selection model | Event-balanced position MAE |
| --- | ---: |
| Qualifying order | 3.680 |
| Previously selected Huber residual | **3.235** |
| Previous shallow boosted residual | 3.470 |
| Pairwise model with practice — selected new candidate | 3.330 |
| Same pairwise model without practice | 3.265 |
| New HGB with practice | 3.505 |
| Same new HGB without practice | 3.495 |

The selected pairwise candidate is **9.511% better than qualifying order**, but **2.937% worse than Huber** and **1.991% worse than its matched ablation**. Its candidate-minus-Huber difference is +0.095 positions, with paired event 95% interval [-0.070, +0.265] and circular three-event interval [-0.065, +0.280]. Against its ablation, the difference is +0.065 positions, with block interval [-0.100, +0.275]. Against qualifying alone, the block interval is [-0.715, +0.015]. Reporting only the qualifying comparison would overstate the new evidence.

Selection required at least 1% improvement against qualifying, prior Huber, prior boosting and the corresponding no-practice ablation, plus negative differences after every event omission. It failed. The HGB candidate also loses to Huber and its ablation. This was two new candidate configurations with two matched diagnostic ablations; the ablations were not eligible for selection. No post-result parameter search occurred.

The new inputs summarize completed free-practice sessions strictly before Grand Prix qualifying. Long runs require at least five accurate, all-clear, non-pit laps with known compound/stint/tyre age. Each lap is matched to one nearest-clock lap per independent peer driver, on the same compound, within 300 seconds and three tyre-age laps; at least three peers are required. Deltas are percentages of session median clean lap time. Ten aggregate features describe relative pace, support, dispersion, raw tyre-age slope, team context and practice best-lap ranks. Of 400 selection drivers, 337 have at least two matched long-run laps. Unavailable measurements retain explicit missingness and zero prior adjustment.

This is observed relative pace, not inferred fuel load or identified tyre degradation. Whole completed practice sessions are available at the declared horizon, so symmetric within-session matching is causal at that cutoff. Historical provider corrections and original publication times remain unverified. The negative ablation result does not prove that genuine fuel-adjusted practice pace has no value.

The pairwise model fits a regularized Bradley–Terry regression: `P(i before j) = sigmoid((x_j - x_i) beta)`, with each event contributing total pair weight one. Scaling uses only earlier events; the qualifying coefficient is constrained nonnegative. The optimizer solves the event-mean binary loss plus its declared ridge penalty, and the final projected gradient must be below 1e-6. Stable scalar-score sorting gives a complete permutation. Correlated within-event pairs are not counted as independent evidence; uncertainty resamples events. No joint probability calibration is claimed. The HGB fits normalized finishing-minus-qualifying rank residuals with equal event total weight. Both methods update only after completed earlier races.

Verification:

- **9 tests passed**, covering independent peers, quality filters, feature units and missingness, phase exclusion, current-target independence, analytic gradients, pair weights, ranking orientation, permutation equivariance and event/block aggregation.
- Independent pre-fit review reconstructed all 12 inherited features and all roster/target/status fields exactly across 41 events / 820 drivers; numerical gradient error was 1.50e-11.
- The verifier refit all 80 model/ablation fits, reconstructed all feature rows and **2,800 predictions**, reproduced all selection metrics, and checked **1,200 exact prior prediction/target bindings**, 228 input hashes and eight source hashes.
- Source, specification, original selection and frozen locks remain unchanged after scores.

One bookkeeping defect is preserved explicitly: the runner's output glob hashed its still-open stdout log before printing the final summary. The initial verifier consequently failed that log hash. The amended verifier proves that the original 738 bytes remain an exact line prefix of the closed 2,772-byte log and records both hashes. Every input, source, feature and prediction hash remains exact; no model or result changed. The original failed verification log and attempt record remain available. This experiment failed selection independently of that logging defect.

Artifacts are in `artifacts/research/boundary_20260908/race_practice/`:

- Selection SHA256: `d89525a0aad7e63eddb858d406daf47307ede406013fbb3111360470df823658`.
- Frozen selection binding SHA256: `25e5e3ee99b5f643d5ebb70457866e19350db3536a6f03c00200e7a0aa611e79`.
- Verification SHA256: `9382492293b4380130dd32bd22bdfe8d55806ca879fb8286f6f591da745c64c8`.

Reproduce the checks using local frozen inputs:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD" .venv-f1/bin/python -m pytest -q --import-mode=importlib research/experiments/boundary_20260908/race_practice/test_experiment.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD" .venv-f1/bin/python research/experiments/boundary_20260908/race_practice/verification/verify_closed_artifacts.py
```

The original runner refuses to overwrite its frozen attempt. Its transfer branch is not used after failed selection. Any next experiment needs a separate protocol and output directory.

Primary motivation: [dynamic BTL rankings](https://www.jmlr.org/beta/papers/v24/21-1179.html) and [time-varying Plackett–Luce rankings](https://arxiv.org/abs/2101.04040). This custom regression does not implement their spectral or score-driven state estimators or inherit their statistical guarantees.

Suggested commit: `research(f1): test practice long-run signals with pairwise race ranking`.
