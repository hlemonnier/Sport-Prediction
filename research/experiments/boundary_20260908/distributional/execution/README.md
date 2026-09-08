# Conditional quantile forecasts — verified partial gain, gate failed

The frozen seven-leaf, 100-iteration quantile HGB improves 2024–2025 weighted interval score (WIS) by **7.917% against global empirical quantiles** and **5.165% against conditional empirical quantiles**. Both paired three-event intervals exclude zero. The improvement transports to each evaluated year, including 2026, but does **not** reach the prespecified 10% requirement against both references. The comparative coverage gate also narrowly fails. No production promotion or point-MAE gain is claimed.

The original published draft, specification, paused status and design lock remain immutable in the parent source/artifact directories. This execution directory corrects the draft input-manifest reader, adds tests and independent verification, and executes the original three settings and gates unchanged. All dates had already been exposed in earlier research; these results are retrospective, not prospective confirmation.

## Results

Selection used 20,007 original eligible issuances across 22 events in 2023. Distributional learners saw only 13,364 residuals from three expanding 2022 crossfit folds. The smallest candidate was selected and locked before final fitting and transfer scoring.

| 2023 selection forecast | Event-balanced WIS |
| --- | ---: |
| Global empirical reference | 0.3541337383 |
| Conditional empirical reference | 0.3484546323 |
| HGB, 7 leaves / 100 iterations — selected | 0.3351732798 |
| HGB, 15 leaves / 100 iterations | 0.3364098321 |
| HGB, 15 leaves / 200 iterations | 0.3388761823 |

The selected model passed the 1% improvement screen against both references and the nominal-coverage screen. Its final distributional fit used 33,371 past residuals across 38 events in 2022–2023. No 2024–2026 labels were used in that fit, and no interval updates occurred during transfer. The immutable production HGB provides every transfer point forecast.

| Population | Events / issuances | Global WIS | Conditional WIS | Selected WIS | Gain vs global / conditional |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2024 | 24 / 22,171 | 0.3092848588 | 0.3013667114 | 0.2857088905 | 7.623% / 5.196% |
| 2025 | 24 / 21,663 | 0.2750404328 | 0.2660006807 | 0.2523565829 | 8.247% / 5.129% |
| 2024–2025 | 48 / 43,834 | 0.2921626458 | 0.2836836961 | 0.2690327367 | 7.917% / 5.165% |
| 2026, exposed | 13 / 11,923 | 0.2896314738 | 0.2857589060 | 0.2781673777 | 3.958% / 2.657% |
| 2026 rounds 10–13, nested | 4 / 3,779 | 0.2600701056 | 0.2582336848 | 0.2555716737 | 1.730% / 1.031% |

| Paired WIS difference, selected minus reference | Difference | Event 95% interval | Three-event-block 95% interval |
| --- | ---: | --- | --- |
| 2024–2025 vs global | -0.0231299091 | [-0.0351797258, -0.0131564654] | [-0.0351474974, -0.0125764877] |
| 2024–2025 vs conditional | -0.0146509594 | [-0.0229186432, -0.0076652205] | [-0.0227522777, -0.0071835603] |
| 2026 vs global | -0.0114640961 | [-0.0236392287, -0.0029432215] | [-0.0217376406, -0.0028241796] |
| 2026 vs conditional | -0.0075915283 | [-0.0175419810, -0.0000675364] | [-0.0154107133, -0.0006774122] |

The selected model improves 45 of 48 historical events against global and 43 against conditional quantiles; every leave-one-event-out historical aggregate remains negative. The recent four-event intervals cross zero against both references. Resampling uses 20,000 paired draws and within-year circular three-event blocks; these are conditional historical intervals without adjustment for repeated research or selection uncertainty.

## Coverage and unchanged point predictions

| Year | Selected 80% / 90% / 95% coverage | Conditional 80% / 90% / 95% coverage |
| --- | --- | --- |
| 2024 | 77.973% / 88.553% / 93.948% | 80.849% / 89.902% / 94.487% |
| 2025 | 79.829% / 89.690% / 94.986% | 82.968% / 91.321% / 95.401% |
| 2026 | 77.275% / 87.880% / 93.662% | 79.315% / 89.124% / 94.178% |

Every year passes the frozen nominal-minus-five-percentage-point requirement. The **relative coverage rule fails** at 80% in 2025: coverage declines by 3.138587 percentage points against the conditional reference, exceeding the allowed three-point decline by 0.138587 points. The candidate is closer to nominal 80% coverage than that reference, but the frozen comparative gate remains failed; it was not relaxed after scoring.

Historical mean interval widths at 50% / 80% / 90% / 95% are 0.540772 / 1.204291 / 1.920913 / 2.871417 seconds, versus conditional-reference widths 0.521183 / 1.248486 / 1.973294 / 3.151807. Full coverage, widths and individual-quantile losses are preserved for each year/event in the evidence.

All nine-quantile outputs are finite and ordered. The final projection corrected raw quantile crossings in 8,046 of 55,757 transfer rows. This is a declared forecast operation shared with the references; independent quantile regressions themselves are not guaranteed coherent. The median is exactly the original HGB point in every row. Event-balanced point MAE remains **0.4590892772 seconds** in 2024–2025 and **0.4755415818 seconds** in 2026, with maximum point difference **0.0 seconds**. This experiment supplies evidence about predictive quantiles, not improved median accuracy, a density, or strategy value.

## Mathematical contract and verification

Each non-median residual quantile minimizes event-weighted pinball loss with the frozen HGB settings. Lower residual quantiles are capped at zero and sorted; upper residual quantiles are floored at zero and sorted; the median residual is zero. Targets are untrimmed. No interval-width cap or fitted density is used.

The global reference uses weighted inverse-CDF quantiles of the same past residuals. The conditional reference uses wet compound, early stint and training-only volatility quartiles, with each cell CDF shrunk toward the global CDF by weight `n_eff / (n_eff + 200)`. It computes **quantiles of the mixture CDF**, not averages of quantiles. These references preserve the same fixed point and use identical evaluation rows.

For four central intervals, `WIS = [0.5*abs(y-m) + sum(alpha/2 * IS_alpha)] / 4.5`, exactly `2/9 * sum(pinball_loss)` on the fixed nine-quantile grid. It is a proper finite-quantile score, not an exact continuous CRPS calculation. The [WIS paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008618) grounds this scoring rule. [NGBoost](https://proceedings.mlr.press/v119/duan20a.html) motivates distributional forecasting; this implementation is custom quantile HGB and does not implement NGBoost or natural-gradient updates.

**11 tests passed in 2.00 seconds** before fitting. An initial test-only mistake indexed the 90th rather than 75th percentile in the CDF-mixture counterexample; it was fixed before source freezing. The independent verifier passed: 105 input hashes; exact expanding-fold and training-population reconstruction; four deterministic base-model refits; all 24 selection quantile models and the eight final quantile models replayed; global/conditional empirical fits rebuilt; WIS recomputed from interval penalties; coverage, widths, both bootstrap methods and all gates independently recomputed. Forecast replay and median-preservation errors were zero. No fitting failure or post-fit source edit occurred. Selection including final refit took 202.9 seconds; transfer took 1.5 seconds, with one CPU thread and sequential fits.

Remaining limits include transporting residual distributions from earlier, smaller base fits to the later frozen point model; retrospective provider accuracy/eligibility flags and publication latency; repeated research exposure; small recent-event sample; and fixed-point medians being estimates, not known true conditional medians.

## Reproduction and artifact closure

All completed output is under `artifacts/research/boundary_20260908/distributional/execution/`. Compact publication evidence comprises `execution_lock.json`, `pre_fit_tests.json`, `selection_attempt.json`, `selection.json`, `selection_lock.json`, `fit_lock.json`, `pre_transfer_verification.json`, `transfer_attempt.json`, `results.json`, `verification.json`, and `evidence.json`. Pickled training/forecast tables and fitted models are local replay dependencies; their hashes are preserved, and they need not be published as source/findings evidence.

- Original specification SHA256: `84c97707c6b8ada670b4a4b12df9cf33dee3e6a6087f85cb15565166a45b0067`.
- Execution source fingerprint: `d5cdcc5d2201fc49f0d09186083510053ec8959b7382aad811bf4521f4f05c5a`.
- Execution lock SHA256: `1099f130137712f368d7cb48955175a9a776649f3ef63d6fbbd886991f005254`.
- Results SHA256: `219d4ed8b9001a076257111891953c8eda7daa8dc11d00497b3e1efd2a4b0095`.
- Verification SHA256: `1fff8f05b37285012d6f7d706f1fdc47f3bd7b02dc83d9b0e1f08ba4a02a9fd3`.
- Evidence SHA256: `eb48cbd67a79e9fa4df6bc3c422555092dbbf1793903f10d0f147e98c4d063d8`.

Run focused tests from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python -m pytest -q -p no:cacheprovider research/experiments/boundary_20260908/distributional/execution/test_math.py
```

Replay using the same frozen source and local inherited inputs, without overwriting completed results:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python - <<'PY'
import json
import shutil
from research.experiments.boundary_20260908.distributional.execution import run, verify
original = run.OUT
run.OUT = original / 'replay_clean'
run.OUT.mkdir(exist_ok=False)
shutil.copy2(original / 'execution_lock.json', run.OUT / 'execution_lock.json')
run.discover()
if json.loads((run.OUT / 'selection.json').read_text())['advancement_passed']:
    run.transfer()
verify.main()
PY
```

The locked inherited cached inputs must be available locally; this README does not claim a fresh public-data reconstruction independent of those provider artifacts. The replay enforces their recorded hashes.

Suggested commit: `research(f1-live): verify conditional quantile gains against empirical baselines`.
