# Football boundary experiment — completed, no promotion

The frozen six-candidate experiment found a small improvement over the strongest previous research model, not the substantial gain it targeted. The selected 90-day shot-strength correction with penalty 0.1 achieved pooled log loss **0.9841272141** on 2,280 matches. This is **4.303%** below the controlled production comparator, **0.265%** below the prior DC365/Elo50 blend, and **0.836%** below DC180. The 28-day paired interval against the prior blend crosses zero. The predeclared requirement of at least 2% improvement against both strong research references **failed**.

These are retrospective results on previously exposed dates. No production file, previous experiment, provider input or Git state was changed by this experiment.

## Selection and evaluation

The specification was frozen before new candidate scores, SHA256 `74e11bdeeb2ec5951daef1ae97c55ea5ad852ab6133110c003edabf47fef58a8`. Pooled historical prequential features from England, Spain and Italy train the correction; EPL 2022/23–2023/24 alone selects among six candidates. The prior blend remains eligible as the identity fallback. All 2024/25–2025/26 evaluation dates and prior aggregate results were already inspected in the previous cycle. These countries are part of historical meta-training, so the evaluation tests chronological country/year stability, not zero-shot country transfer.

| Selection candidate | EPL selection log loss, 760 matches |
| --- | ---: |
| Shot strength 90d, penalty 0.1 — selected | 0.9581581185 |
| Shot strength 90d, penalty 0.01 | 0.9585640460 |
| Prior DC365/Elo50 identity fallback | 0.9595989050 |
| Shot strength 365d, penalty 0.1 | 0.9615498796 |
| Log-probability residual only, penalty 0.1 | 0.9616659759 |
| Log-probability residual only, penalty 0.01 | 0.9624334019 |
| Shot strength 365d, penalty 0.01 | 0.9633519216 |

The selected correction was locked before computing its new evaluation scores. No alternative candidate was subsequently evaluated on those dates.

| Evaluation model | Log loss | Class-sum Brier | Accuracy | Top-label ECE, 10 bins |
| --- | ---: | ---: | ---: | ---: |
| Selected shot correction | 0.9841272141 | 0.5869165325 | 52.105% | 0.0228483856 |
| Strong prior DC365/Elo50 | 0.9867449115 | 0.5885281970 | 51.798% | 0.0091553347 |
| DC180 | 0.9924257346 | 0.5921102468 | 51.228% | 0.0138967957 |
| DC365 | 0.9901843502 | 0.5908215215 | 52.193% | 0.0116717382 |
| Equal-weight DC | 0.9930848698 | 0.5929259223 | 51.447% | 0.0113208172 |
| Controlled production DC+auto | 1.0283816775 | 0.6155755626 | 49.211% | 0.0287842063 |

The selected correction improves log loss, Brier and accuracy against the prior blend, but **worsens top-label ECE**. It should not be presented as an unqualified calibration improvement.

| Reference | Candidate minus reference log loss | Paired 28-day 95% interval |
| --- | ---: | --- |
| Production DC+auto | -0.0442544633 | [-0.0558200136, -0.0323752617] |
| Prior DC365/Elo50 | -0.0026176974 | [-0.0052358920, +0.0001833751] |
| DC180 | -0.0082985205 | [-0.0135112648, -0.0030656616] |

Intervals use 4,000 seeded paired fixed-calendar-block resamples, stratified by league-season. One-day and seven-day versions also cross zero against the prior blend. They describe conditional historical uncertainty; they do not adjust for research reuse, selection across previous cycles, cross-stratum common shocks or prospective deployment changes.

| League/year | Matches | Selected log loss | Relative improvement vs prior blend |
| --- | ---: | ---: | ---: |
| EPL, both years | 760 | 1.0020804387 | +0.2382% |
| Spain, both years | 760 | 0.9705333950 | +0.1634% |
| Italy, both years | 760 | 0.9797678087 | +0.3936% |
| EPL 2024/25 | 380 | 0.9861297214 | +0.0578% |
| EPL 2025/26 | 380 | 1.0180311560 | +0.4124% |
| Spain 2024/25 | 380 | 0.9666167875 | +0.3542% |
| Spain 2025/26 | 380 | 0.9744500025 | -0.0266% |
| Italy 2024/25 | 380 | 0.9637076276 | +0.7193% |
| Italy 2025/26 | 380 | 0.9958279898 | +0.0764% |

All three full-league intervals against the prior blend cross zero. Excluding the known resumed Fiorentina–Inter fixture leaves 2,279 matches and a candidate-minus-blend difference of -0.0026284960, with 28-day interval [-0.0052462135, +0.0001745023]. The conclusion is unchanged. This exclusion affects scoring only; it does not invent an alternative fixture history.

## Mathematical and timing contract

Each shot stream has a ridge Poisson GLM with team attack, opponent defence, a home indicator and an intercept. Separate models estimate shots and shots on target, on prior 1,095-day match histories. Exponential temporal weights have a frozen half-life of 90 or 365 days. The means are explanatory features, not xG estimates or a claim that observed shots are conditionally independent or equidispersed.

The correction is `softmax(log(p_DC365_Elo50) + XW)`, where `X` contains an intercept, standardized log odds, DC/Elo disagreement, country indicators and optional log predicted shot means. It minimizes mean outcome negative log likelihood plus `penalty / 2 * ||W||²`; the penalty includes the correction intercept. Zero weights reproduce the previous blend exactly. Scalers and corrections fit only to available, strictly earlier prequential forecasts in a trailing 1,460-day window. Coefficients are refit at each inherited base-model cutoff. Earlier evaluation outcomes can enter later refits after availability; this is a prequential evaluation.

Forecasts occur at each league's local midnight. Final results and shot statistics are assumed available at the next local midnight. Same-day result updates are forbidden. Base goal models and shot strengths refit on the first fixture day in each calendar two-month bucket, with the same history and timing as the previous benchmark. Original publication/revision timestamps are unavailable; this availability contract is an explicit assumption. Known resumed fixtures retain the source's completion date, which is not a pre-original-kickoff forecast.

The production comparator reproduces the actual default equal-weight DC fit plus automatic held-out calibration, using `chronological_populations` and the same input history/cadence. Its output was checked against `run_prediction`, with only data loading and population selectors injected for the regression. Production's fit/calibration partition leaves less recent data in its DC fit than the research references; the larger production gain therefore must not substitute for the much smaller gain against the strong prior blend.

All 10,260 cached input rows have finite, nonnegative integer HS/AS/HST/AST fields. One provider anomaly, Newcastle–West Ham 2021/22, reports away shots 8 and away shots on target 9. It is retained unchanged and recorded in the evidence. Separate count GLMs remain mathematically defined; no result-dependent repair or exclusion was made.

## Reproducibility and verification

Source fingerprint: `1928a5cf5e70e340dba20389b3ada4a5a3b4cf7abb05b0380943d0532a1902a2`.

The original public-provider files and manifests remain under `data/football/performance_20260907/`. The runner verifies their existing SHA256 values. All new output is under `artifacts/research/boundary_20260908/football/`:

- `design_lock.json`, `selection_attempt.json`, `selection_lock.json`, `evaluation_attempt.json`: timestamped specification, source, runtime and selection locks.
- `features_{selection,evaluation}_{E0,SP1,I1}.json`: prequential features, fitted parameters, precise fit membership and source/input hashes.
- `selection.json`, `evaluation.json`: complete forecast populations, every tried selection model and selected evaluation forecasts.
- `verification.json`, `evidence.json`, `regression_test_results.json`: compact findings and verification.
- `selection.log`, `evaluation.log`, `verification.log`: execution logs.

Nine regressions passed in 2.22 seconds. Verification reconstructed 504 shot models and 108 correction models/scalers over 7,980 prequential rows and 126 base cutoffs. All 7,600 checked inherited comparator vectors exactly matched the previous frozen evidence; serialized correction replay also had maximum absolute error 0.0. All 504 goal fits converged; maximum KKT residual was 5.4193e-6. Maximum correction objective gradient norm was 5.4731e-7. The selection run took about 86 seconds and evaluation about 27 seconds, on one CPU thread; no fitting failures or post-launch source edits occurred.

Run tests from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python -m pytest -q -p no:cacheprovider research/experiments/boundary_20260908/football/test_math.py
```

To reproduce all fits without overwriting the completed output, use a new output subdirectory and the same frozen source and cached inputs:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python - <<'PY'
from pathlib import Path
import shutil
import sys
from research.experiments.boundary_20260908.football import run, verify
original = run.OUT
run.OUT = original / "replay_clean"
run.OUT.mkdir(exist_ok=False)
shutil.copy2(original / "design_lock.json", run.OUT / "design_lock.json")
for phase in ("selection", "evaluation"):
    sys.argv = ["run", "--phase", phase]
    run.main()
verify.main()
PY
```

The source uses unchanged prior DC/Elo code and current production probability math. [Wheatcroft's match-statistics forecasting paper](https://arxiv.org/abs/2001.09097) motivates using forecastable shot statistics; [Kull et al.'s multiclass calibration work](https://papers.nips.cc/paper_files/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html) motivates log-probability transformations. This experiment is an identity-regularized residual model, not a claimed replication of either paper.

Suggested commit: `research: evaluate causal shot-strength football corrections`.
