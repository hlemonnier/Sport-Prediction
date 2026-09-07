# Race-order performance research — 7 September 2026

The 2023-selected Huber residual model does not establish a stable improvement. On the final 43-event 2024–2025 cohort, MAE changes from 3.100122 to 3.084088 positions; paired delta is −0.016034, with event-bootstrap 95% interval [−0.130233, +0.097674]. It improves 2024 but worsens 2025 and the exposed 2026 diagnostic. Retain the baseline.

| Model selected within family on 2023 | 2024–2025 MAE | Delta to baseline | Event 95% interval |
|---|---:|---:|---|
| baseline | 3.100122 | +0.000000 | [+0.000000, +0.000000] |
| shrunken_movement_1 | 3.141983 | +0.041860 | [+0.004651, +0.079070] |
| ridge_movement_1 | 3.067687 | -0.032436 | [-0.148715, +0.081640] |
| huber_movement_1 | 3.084088 | -0.016034 | [-0.130233, +0.097674] |
| boosted_absolute_movement_1 | 3.032925 | -0.067197 | [-0.148592, +0.014198] |

The baseline is the complete qualifying order at the post-qualifying, provisional-grid horizon. It does not use the eventual race-table starting grid, penalties unavailable at that horizon, or any current race covariates. The existing published 2026 post-grid baseline is a different information product; its MAE is not used as this experiment comparator.

Features are normalized qualifying rank, within-team qualifying context, decayed/shrunk prior driver and constructor form, historical failure rates and past circuit movement. Form updates use strictly previous races; constructor and driver form reset by season. Ridge, Huber and shallow absolute-error boosting predict rank residuals, then a stable sort yields a complete permutation. Each training event receives total weight one. Knobs and the preferred family are selected only on 2023; later fitting is expanding and prequential with those choices fixed.

Coverage is conditional: 92 of 101 cached events enter the study (43 of 48 in 2024–2025, 859 drivers). Nine events fail raw qualifying completeness, race roster equality or complete target permutation checks. Some exclusions depend on target availability, so the reported errors are not universal all-race estimates. The source metadata do not prove original publication timestamps or revisions. Race bootstrap intervals treat events as exchangeable; serial/team dependence is not removed.

Execution history is retained: v1 had an overly broad filename match that confused Sprint Qualifying/Race files with Grand Prix sessions and used incomplete provider DriverId fields. The adapter was corrected to exact Grand Prix filenames and qualifying abbreviations, without changing model families or hyperparameters. v2 completed fitting but failed output serialization when given a relative output directory. v3 is the final amended retrospective run; it must not be described as pristine holdout evidence. Earlier outputs are superseded, not additional independent experiments.

Eight focused tests pass, including all model forecasts under current/future target poisoning, event-prefix invariance, roster equivariance and Grand Prix session filtering. No fit warnings occurred. The result JSON records every tested configuration, paired event outcomes, exclusion reason, exact input and implementation hashes.

Run from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv-f1/bin/python research/experiments/performance_20260907/race/run.py --output artifacts/research/performance_20260907/race/my_reproduction
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv-f1/bin/python -m pytest -q research/experiments/performance_20260907/race/test_run.py
```

Suggested commit: `research(f1): test causal race-order residual challengers`
