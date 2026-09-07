# First football performance cycle — 7 September 2026

The validation-selected 50/50 Dixon–Coles/Elo mixture improved historical test log loss by **0.00995349 (0.981%)** versus equal-weight Dixon–Coles. Both test seasons improved. This is a research candidate; production models and promotion status are unchanged.

| Model | Validation log loss (760 matches) | Test log loss (760) | Test Brier, sum over classes | Test accuracy |
|---|---:|---:|---:|---:|
| Equal-weight Dixon–Coles, baseline | 0.973417 | 1.014427 | 0.608484 | 49.47% |
| Dixon–Coles, 365-day half-life | 0.969398 | 1.005261 | 0.602423 | 50.39% |
| Dixon–Coles, 180-day half-life | 0.972201 | 1.003406 | 0.601425 | 49.87% |
| **DC365/Elo50, selected on validation** | **0.959599** | **1.004473** | **0.601888** | **50.13%** |
| Elo component, descriptive only | 0.959942 | 1.009537 | 0.605504 | 50.13% |
| Smoothed league frequency | 1.055944 | 1.083686 | 0.656679 | 41.71% |
| Closing market, later information | — | 0.989250 | 0.591462 | 52.50% |

The 180-day model happens to have the lowest pooled test loss; it was **not** substituted for the locked validation winner. It deteriorated against the baseline in 2025/26 (1.028756 versus 1.026810), while the selected mixture improved in both years:

| Test season | Equal-weight DC | Selected mixture | Difference |
|---|---:|---:|---:|
| 2024/25 | 1.002044 | 0.986700 | −0.015344 |
| 2025/26 | 1.026810 | 1.022247 | −0.004563 |

For selected-minus-baseline paired log loss, the 95% percentile interval is **[−0.017567, −0.002303]** when resampling match-day clusters and **[−0.016736, −0.002762]** with 28-calendar-day blocks. Both use 4,000 paired resamples, stratified by season with equal season weights. There are 223 match-day clusters and only 22 longer blocks; these intervals quantify variation within these two seasons, not transfer to independently sampled future seasons. No multiple-comparison correction or prospective validation is claimed. The selected model's top-label ECE changed from 0.035148 to 0.035345, so this run does not demonstrate improved calibration.

## Data and timing

Nine complete EPL seasons, 2017/18–2025/26, provide 3,420 matches. The runner checks each season has 20 teams, 380 unique fixtures and 38 appearances per team; goals must agree with the result label. Files are acquired directly from [Football-Data's England archive](https://football-data.co.uk/englandm.php). Exact URLs, retrieval times, source modification headers and SHA256 hashes are in `data/football/performance_20260907/source_manifest.json`.

[The provider's schema](https://football-data.co.uk/notes.txt) distinguishes final scores, result labels and pre-closing/closing odds. Only date, teams and **historically available final results** enter causal models; same-match shots, cards, half-time scores and odds do not. Date is interpreted as the English local calendar date. Forecasts are made at 00:00 Europe/London; a result becomes usable at the next local midnight. Daylight-saving transitions therefore create 23- or 25-hour calendar days. This is a declared historical availability assumption: the downloaded files do not archive original result publication times or subsequent revisions.

Average closing odds (`AvgCH`, `AvgCD`, `AvgCA`) are converted to normalized reciprocal probabilities. All 760 test matches have this benchmark, so the population is identical. Closing prices have later information than the model's day-start cutoff. They are never features, mixture components or selection candidates. [The data source describes its odds collection schedule and the changed handling of unreliable Pinnacle quotes from July 2025](https://football-data.co.uk/data.php).

The source describes the files as free for league match prediction and retains copyright; this local research use does not assume an unrestricted redistribution license. Source pages, schema and [liability disclaimer](https://football-data.co.uk/disclaimer.php) are saved with hashes. No paid service, betting execution or external publication was used.

## Frozen experiment

`spec.json` was serialized to `design_lock.json` before acquisition or any model metrics. Warmup is 2017/18–2021/22; validation is 2022/23–2023/24; test is 2024/25–2025/26. Test seasons were held aside **during this research cycle**. They are historical, and are not described as universally unseen or prospective.

Every causal model refits at the first forecast day of a calendar two-month bucket, using the same prior 1,095-calendar-day match IDs. The test is prequential: already completed test matches may inform later test predictions under the frozen schedule. Future and same-day outcomes never enter a forecast. Elo rating updates occur atomically after all fixtures of a local day. Selection and hyperparameters remain frozen throughout test.

There are exactly three candidate mechanisms beyond the equal-weight baseline: 365-day exponential likelihood weighting, 180-day weighting, and a fixed 50/50 DC365/Elo probability mixture. Production joint Dixon–Coles likelihood, feasibility constraints, regularization and tail-controlled outcome calculation are reused unchanged. Elo uses initial rating 1,500, K=20, home advantage 60 points and the standard base-10 scale of 400. Its multinomial outcome mapping learns from causal rating difference and absolute difference on the identical rolling fit population (L2 logistic C=1). No grid search, retrospective model substitution or calibration tuning occurred. Returning teams keep their old rating; unseen promoted clubs start neutral. These simple choices and the absence of lower-league/player information remain modelling limits.

The validation phase alone chooses minimum pooled natural-log loss among baseline and three candidates. It serializes `selection_lock.json` before the test phase may load test labels. Every tried model, including the nonselectable component and descriptive baselines, is recorded. Code, spec, source manifest and validation artifact hashes must still agree when test starts.

All 72 joint likelihood fits converged; maximum independently checked KKT residual was 5.036013e−6, below the production 1e−5 threshold. Seven focused tests verify day causality, future perturbation invariance, DST, probability contracts, metric scales and paired resampling. Artifact verification independently reconstructs all 1,520 scored forecasts from stored fitted parameters; maximum probability error is 1.11e−16. A complete fresh replay produced exactly identical validation/test prediction rows, metrics and selected model; `reproducibility.json` records that check. The replay repeats the same experiment and adds no candidate or independent statistical evidence.

## Run and inspect

From the repository root with the existing Python environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python research/experiments/performance_20260907/football/acquire.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python -m pytest -p no:cacheprovider research/experiments/performance_20260907/football/test_benchmark.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python research/experiments/performance_20260907/football/replay.py --run-name my_reproduction
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python research/experiments/performance_20260907/football/validate_evidence.py
```

The original cycle used `benchmark.py --phase validation`, then `--phase test`, then `--phase report`. Completed validation/test artifacts refuse silent overwrite. `replay.py` runs all three phases into a fresh directory under `artifacts/research/performance_20260907/football/replays/`; it neither changes models nor acquires additional data. Fits are serial with one BLAS thread and complete in seconds in the existing environment.

Evidence is in `artifacts/research/performance_20260907/football/evidence.json`; validation/test JSON includes every prediction, fit membership and fitted parameter. `verification.json` contains reconstruction results. This first cycle supports extending the fixed mixture to other leagues or a forward collection period under a new frozen protocol. It does not establish a market edge, recommend bets, or justify production promotion.

Suggested commit: `research: benchmark causal football models on frozen EPL seasons`
