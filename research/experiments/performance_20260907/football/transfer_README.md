# Frozen football model transfer to Spain and Italy

The EPL-selected **DC365/Elo50** model improved pooled historical log loss in both new leagues, with unchanged model settings. The 95% paired intervals include zero in each league and when pooled. This is encouraging directional evidence, insufficient to establish transportability or justify promotion.

| League, 2024/25–2025/26 | Matches | Equal DC loss | Fixed mixture loss | Relative reduction | Candidate-minus-baseline 95% interval |
|---|---:|---:|---:|---:|---:|
| Spain La Liga | 760 | 0.978798143 | 0.972121601 | 0.6821% | [−0.014653743, +0.001855985] |
| Italy Serie A | 760 | 0.986029642 | 0.983639797 | 0.2424% | [−0.008876079, +0.004284109] |
| Pooled transfer | 1,520 | 0.982413893 | 0.977880699 | 0.4614% | [−0.009604590, +0.000771562] |

| League/season | Equal DC loss | Fixed mixture loss | Difference |
|---|---:|---:|---:|
| Spain 2024/25 | 0.973598792 | 0.970052498 | −0.003546293 |
| Spain 2025/26 | 0.983997494 | 0.974190704 | −0.009806790 |
| Italy 2024/25 | 0.976557332 | 0.970689914 | −0.005867418 |
| Italy 2025/26 | 0.995501952 | 0.996589679 | +0.001087726 |

Pooled Brier loss improves from 0.585147 to 0.581848 and accuracy from 52.43% to 52.63%. Ten-bin top-label ECE worsens from 0.009082 to 0.020244. The gain is not uniform across seasons or metrics. No transfer result was used to retune, replace or reselect the candidate.

## Design and provenance

`transfer_spec.json` was frozen before acquisition/scoring, SHA256 `9e2dd99685014ad4d8bbeed500b14c39177a6c80d4e47c5102af61d48f47e2b5`. The wrapper verifies the original EPL spec, selected model and source fingerprints against the frozen EPL evidence. It never modifies `benchmark.py`, `spec.json`, `acquire.py`, production modules or existing EPL artifacts.

Eighteen CSVs come directly from the primary [Spain archive](https://football-data.co.uk/spainm.php) and [Italy archive](https://football-data.co.uk/italym.php), using SP1 and I1. Each league has nine complete seasons (3,420 fixtures), 2017/18–2025/26. The seven seasons through 2023/24 provide initial history; only 2024/25 and 2025/26 are scored. League identity, unique home/away fixtures, all 380 final scores and 38 appearances for each of 20 teams are verified for every season. Inputs and source-page snapshots have exact URLs, timestamps and hashes in `data/football/performance_20260907/transfer_source_manifest.json`.

The model remains the arithmetic 50/50 mixture selected on EPL validation: joint Dixon–Coles with 365-day exponential weights plus causal Elo multinomial probabilities. All Elo parameters, regularization, 1,095-day rolling fit window, two-month refit cadence, score probability support and no-calibration policy remain unchanged. The equal-weight DC baseline shares every fit-match ID. Only those two forecast policies are evaluated; no 180-day alternative or additional selection runs in transfer. The wrapper passes the original specification to the unchanged prediction function, temporarily configuring its timezone and required DC component fits inside a context that restores both afterward.

Timing uses the same **local calendar day** rule: forecasts at 00:00 Europe/Madrid or Europe/Rome, result availability at the next local midnight, and atomic Elo updates after all same-day forecasts. The models update causally with completed prior test matches under the fixed schedule. Odds and same-match statistics are not parsed into transfer inputs. Historical first-publication times and later corrections remain unarchived assumptions; these are retrospective, not prospective estimates.

Intervals use 4,000 paired bootstrap resamples of nonoverlapping 28-calendar-day blocks, separately within each league-season. Spain has 22 blocks, Italy 21; pooled uncertainty uses four league-season strata. This measures conditional variability within the sampled seasons and does not establish independent future-season generalization.

## Resumed matches and timing sensitivity

A targeted check identified two Italian fixtures whose CSV date is their completion day. [Inter's official announcement](https://www.inter.it/en/news/rescheduled-continuation-fiorentina-inter-6-february-2025) identifies Fiorentina–Inter on 6 February 2025 as a resumption after its earlier interruption. [Roma's official team news](https://www.asroma.com/en/news/71046/team-news-starting-xi-for-rearranged-udinese-match) similarly identifies Udinese–Roma on 25 April 2024 as a resumption. The latter is in warmup; the former is a test row.

Their final scores enter history only after completion, so this source-date convention does not expose those final results early. However, the forecast on resumption day is not a forecast before the original kickoff, and the model does not condition on the known partial match. `transfer_timing_notes.json` recorded this distinction **before transfer metrics**. This was a targeted check, not an exhaustive audit of every historical stoppage.

The primary Italy population remains all 760 matches. The additional predeclared sensitivity excludes the one identified resumed test fixture **only from scoring**, without refitting or changing predictions: 759 matches, baseline loss 0.985397851, mixture loss 0.982900915, difference −0.002496936, interval [−0.008999131, +0.004180852]. The interpretation is unchanged.

## Verification and reproduction

Five transfer tests passed in 2.02 seconds, covering parameter inheritance, timezone/context restoration, wrong-league and score rejection, complete schedules and next-local-midnight availability. All 48 joint fits converged: maximum KKT residual 4.940629e−6. For each league, artifact verification checks chronology, membership and causal Elo features and independently reconstructs all fixed-mixture and DC probabilities from saved parameters. Maximum reconstruction error is 1.11e−16 over 1,520 transfer forecasts.

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python research/experiments/performance_20260907/football/transfer_acquire.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python -m pytest -p no:cacheprovider research/experiments/performance_20260907/football/transfer_test.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python research/experiments/performance_20260907/football/transfer_run.py --replay-name my_reproduction
```

The original run used `transfer_run.py` without the replay flag. Existing output files refuse silent overwrite. Fits run serially with one numerical-library thread. Replays create a fresh `transfer_replay_*` directory and add no independent statistical evidence.

Machine-readable evidence: `artifacts/research/performance_20260907/football/transfer_evidence.json`. Per-league `transfer_SP1.json` and `transfer_I1.json` retain every prediction, fit row ID, model parameter and diagnostic. Core keys are `leagues.{SP1,I1}.overall`, `.by_season`, `.verification`, `.convergence`, and `pooled_1520_match_transfer`. Italy also contains `pre_metric_known_resumption_exclusion_sensitivity`. Exact original-file preservation and all new file hashes are recorded separately.

Suggested commit: `research: test frozen football blend across Spain and Italy`
