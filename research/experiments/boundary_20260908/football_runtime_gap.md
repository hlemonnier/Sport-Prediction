# Football runtime gap: post hoc operational audit

**A material runtime-performance opportunity exists, but no previously approved football model is merely waiting for a configuration switch.** The global football path still uses equal-weight Dixon–Coles with automatic calibration and frozen evaluation-fit parameters. Previously closed research forecasts outperform this controlled runtime on 2,280 matches. This note preserves that distinction; it neither changes production nor converts an older failed research gate into a success.

Audit time: 2026-09-08T09:10:11.420922+00:00. Inspected HEAD: `8d2a1e21a578981142adfc06c95bd3c928d7d2ec`. Scope: source inspection and arithmetic on already closed forecasts only; **zero new fits, zero new forecasts, zero newly acquired data**. The comparison below was chosen after inspecting the earlier results and is explicitly **post hoc**, not a new predeclared test, promotion decision, or achievement of the ongoing substantial-improvement objective.

## Current global path and the concrete gap

- `apps/api/src/index.js:794–807` exposes `dixon`, `gbdt`, and `hybrid`, defaulting to `dixon` and calibration `auto`; lines 1458–1474 pass those settings to the football runner.
- `research/projects/football/Match Result Prediction/Python/run_experiment.py:90–93,117–128` preserves those defaults. Its `mrp/__init__.py:12–26` resolves the authoritative package under `packages/football/mrp`.
- `packages/football/mrp/config.py:18–21` defaults to no goal-strength decay. `constants.py:3` identifies version `0.4.0`. `prediction.py:62–66` accepts only the three runtime model names and maps unknown names back to Dixon–Coles. A shot or Elo name is therefore not a usable integration switch.
- `prediction.py:168–173` fits Dixon–Coles on `chronological_populations(history).fit`; `protocol.py:69–74` reserves the more recent approximately 15%, 15%, and 20% for calibration, selection, and test. The fixture predictions at `prediction.py:279–287` continue using that oldest fit prefix. This is an explicit policy, not a newly discovered target leak or an accidental invocation of the wrong function.
- Runtime `hybrid` is DC plus GBDT (`prediction.py:221–226`), not the research DC/Elo blend. The optional CLI half-life still operates on the truncated fit population; `--goal-strength-half-life-days 365 --football_calibration off` does **not** reproduce the evaluated full-history DC365/Elo50 policy.

The first closed EPL transfer block is `2024-08-16:64188d62dff2`. It has **1,130** causally admitted historical matches. The runtime comparator's DC model fits **558**, ending at **2023-01-18 00:00 UTC**, while the forecast block starts **2024-08-16**. Its held-out partitions contain 170 calibration, 172 selection, and 230 test matches. The full-history equal-DC research comparator uses all 1,130 admitted matches. This example is directly recorded in `artifacts/research/boundary_20260908/football/features_evaluation_E0.json`, first `fits` entry; its hash is also bound by the closed evaluation artifact.

`research/experiments/boundary_20260908/football/models.py:19–33` reconstructs the current default fit/calibration policy. The existing regression `football/test_math.py:95–113`, `test_comparator_matches_real_production_prediction_api`, compares it to the actual canonical prediction API while replacing only data loading and population selectors. The audit verified all **19** source hashes in the closed evaluation, including all 12 bound `packages/football/mrp` files, still match the current files. Thus this is a relevant controlled comparator, not a claim about measurements from a deployed service.

## Previously closed 2,280-match results

The population is the original complete England, Spain, and Italy 2024/25–2025/26 cohort: 380 matches per country-season, with the known resumed fixture retained. These are historically reused dates, not prospective data. The comparison controls the input history window and research refit cadence; it does not measure every possible live caller's history selection or scheduling.

NLL is mean negative natural-log probability assigned to the observed H/D/A result. Brier is the sum over all three classes, averaged over fixtures. Lower is better for both.

| Existing policy | NLL | Class-sum Brier | NLL reduction versus controlled runtime |
| --- | ---: | ---: | ---: |
| Controlled runtime DC+auto | 1.0283816775 | 0.6155755626 | 0.000000% |
| Full-history equal DC, uncalibrated | 0.9930848698 | 0.5929259223 | 3.432267% |
| Full-history DC365 | 0.9901843502 | 0.5908215215 | 3.714314% |
| Full-history DC180 | 0.9924257346 | 0.5921102468 | 3.496362% |
| Previously selected DC365/Elo50 | 0.9867449115 | 0.5885281970 | 4.048766% |
| Previously selected shot90/ridge0.1 | 0.9841272141 | 0.5869165325 | 4.303311% |

The table does not select a new winner. Every value reuses the earlier frozen forecasts. It does not attribute the gap uniquely to calibration, fit size, or recency: those aspects differ jointly between the runtime comparator and full-history reference.

## New arithmetic on old forecasts: equal DC versus controlled runtime

The audited pair is fixed here as existing `dc_equal` minus existing `production_default_dc_auto`. On all 2,280 rows, the NLL difference is **−0.0352968076872463**, a **3.432267%** reduction. Its 95% paired percentile interval is **[−0.0459453372712680, −0.0250661412611010]**.

This calculation reuses the original uncertainty settings: 4,000 draws, seed 20260908, nonoverlapping 28-calendar-day blocks anchored at each league-season's first observed date, separate resampling within each league-season, and original fixture-count weights across strata. The six stratum block counts are E0:2024=11, E0:2025=11, I1:2024=11, I1:2025=10, SP1:2024=11, SP1:2025=11: **65 observed blocks**. Empty blocks are not sampled. All 4,000 sampled differences are negative; this frequency is not a posterior probability, a multiple-comparison adjustment, or a future-season guarantee.

The following subgroup comparisons are descriptive; no subgroup confidence claim or new subgroup gate is introduced. `2024` denotes season 2024/25 and `2025` denotes 2025/26.

| Population | Matches | Equal-DC NLL | Runtime NLL | NLL delta | Equal-DC Brier | Runtime Brier | Brier delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pooled | 2280 | 0.993084870 | 1.028381677 | -0.035296808 | 0.592925922 | 0.615575563 | -0.022649640 |
| E0 | 760 | 1.014426824 | 1.054570033 | -0.040143209 | 0.608483684 | 0.635000388 | -0.026516704 |
| SP1 | 760 | 0.978798143 | 1.006658988 | -0.027860845 | 0.581301359 | 0.597949016 | -0.016647657 |
| I1 | 760 | 0.986029642 | 1.023916011 | -0.037886369 | 0.588992723 | 0.613777284 | -0.024784560 |
| E0:2024 | 380 | 1.002044130 | 1.063314539 | -0.061270409 | 0.599685257 | 0.638770274 | -0.039085017 |
| E0:2025 | 380 | 1.026809518 | 1.045825527 | -0.019016008 | 0.617282111 | 0.631230502 | -0.013948391 |
| SP1:2024 | 380 | 0.973598792 | 1.009106923 | -0.035508131 | 0.577366685 | 0.598045847 | -0.020679163 |
| SP1:2025 | 380 | 0.983997494 | 1.004211054 | -0.020213560 | 0.585236034 | 0.597852185 | -0.012616150 |
| I1:2024 | 380 | 0.976557332 | 1.008010557 | -0.031453225 | 0.583387816 | 0.604699765 | -0.021311949 |
| I1:2025 | 380 | 0.995501952 | 1.039821465 | -0.044319513 | 0.594597631 | 0.622854803 | -0.028257172 |

Both losses improve in every country and all six country-seasons. This supports a focused assessment of the operational fitting policy; it does not establish why that policy underperforms. The canonical runtime's disjoint historical assessment remains useful and should not be replaced by in-sample evaluation merely to fit newer production parameters.

## Why the older research remains research

The shot correction's earlier frozen target required at least 2% pooled NLL improvement over **both** strong references, DC365/Elo50 and DC180, plus improvement over the current default. It achieved only **0.265%** over DC365/Elo50 and **0.836%** over DC180. Its paired 28-day interval against the blend was **[−0.005235891990316751, +0.0001833751043361011]**, crossing zero. The earlier `substantial_gain_target_met` remains **false**. Its 4.303% advantage over the weaker runtime must not replace that failed criterion. Its pooled top-label ECE, 0.0228484, also exceeds the blend's 0.0091553; this was not an unqualified calibration improvement.

The original EPL-selected DC365/Elo50 transfer to Spain and Italy had a pooled 1,520-match interval **[−0.009604590, +0.000771562]** versus equal-weight DC; both individual-country intervals also crossed zero. Italy 2025/26 deteriorated slightly against equal DC. The frozen [transfer report](../performance_20260907/football/transfer_README.md) and [shot report](football/README.md) explicitly did not promote the models. No newly evaluated Elo-Odds reference, past-market candidate, or later experiment is included in this audit's numerical comparison.

The existing policies are executable 1X2 research forecasters, not production-ready parameter assets. In particular, the shot policy needs past HS/AS/HST/AST, causal shot GLMs, and prequential correction training; the runtime `MatchRecord` at `packages/football/mrp/data.py:61–77` does not retain those shot fields. The goal-only DC/Elo route would need its historical state, trained outcome mapping, cadence, and history policy implemented explicitly.

The current runtime also emits a reconciled joint score distribution, expected goals, and most likely scoreline (`prediction.py:281–287`; `score_distribution.py:39–58`). These 1X2 NLL/Brier results do **not** validate scoreline log loss, total-goal calibration, exact-score probabilities, or betting returns. A mathematically coherent region reconciliation alone cannot establish those empirical properties.

## Concrete next assessment

1. Freeze a separate operational policy comparison, preserving the exact current default as a reference and keeping every historical forecast's information cutoff intact. Start with the already evaluated full-history equal-DC policy; do not conflate its integration with selecting a new research mechanism.
2. Demonstrate caller-to-model parity for the local-midnight information contract, admitted history, refit cadence, and saved full-history reference predictions. Keep historical evaluation forecasts frozen while assessing a separate model fitted for the actual next fixture. Setting `shadow_eval=False` alone does not change the current fit partition.
3. If attributing the gain is necessary, predeclare a small matched assessment separating training-history recency from calibration policy before obtaining new scores. A calibrator intended for newer fitted models requires its own causal validation; the old fitted calibrator cannot simply be assumed transferable to refitted probabilities.
4. Decide advancement on the frozen operational comparison and intended product scope. Assess 1X2 and joint-score outputs separately, specify unavailable-history behavior, then wire an eligible policy through configuration, canonical prediction, API and CLI with parity regressions. Keep old failed research decisions and this post hoc audit unchanged.

This is an actionable runtime assessment opportunity, not a request to keep searching for nearby research hyperparameters or a claim that the existing production baseline is already optimal. No gate was changed and no integration is performed by this note.

## Bindings and arithmetic reproduction

The original verified evaluation SHA256 is `1c1d037ffbdf00a693728254e630792315ee7db8d8f79cbdb3545d34df66a795`; ordered population digest is `13e5eb0e6763266167dc003d982756d933a873bec7a16017c2765256f03505ed`. The audit checked the evaluation hash against its prior passed verification and all 19 embedded source hashes against current bytes. The authoritative complete prior source map remains `evaluation.json.source_files`; the following additional audit bindings pin the concrete runtime paths, statistics implementations and partition evidence inspected here.

```json
{
  "apps/api/src/index.js": "734e6275f414a7ef314b2a77d46a8db6c793afd27c9d26970b050cb8be55bb99",
  "artifacts/research/boundary_20260908/football/evaluation.json": "1c1d037ffbdf00a693728254e630792315ee7db8d8f79cbdb3545d34df66a795",
  "artifacts/research/boundary_20260908/football/features_evaluation_E0.json": "d499ff3b88e6c3c1e6d71de5f817cf115c28814a177d27474b932f61eed986e0",
  "artifacts/research/boundary_20260908/football/verification.json": "3a6328942dd4fe0708aa3539d2f89464106f11fc727abbbf7c15bfbbef89356a",
  "packages/football/maturity.json": "243e064cd7f1a791ebf50b01258d0c2ee54c3b64707ab3fd019e48657517bb06",
  "packages/football/mrp/config.py": "5c637788e5c44a0602521b777e9755c1d6cf5e743091b94c41de548cde4fee7b",
  "packages/football/mrp/constants.py": "2c32ada9148c9fdf1a57844559ce77ba7530ed1d3cb5d58afa33335b7a4ea28b",
  "packages/football/mrp/data.py": "c4742cd2744b46817e34d3e7767f175179ee4919d4a0d61ae5770b254138fb33",
  "packages/football/mrp/prediction.py": "c5085b32c8a33efa485cac87ce2c2b279a23cff97b0fe04fdeb10f383cb90918",
  "packages/football/mrp/protocol.py": "61df7a8735bd523fece0e75a1558e41c94978b356515ad40b833d5a1c24c9906",
  "packages/football/mrp/score_distribution.py": "904c747f36658050112600fa31ee2392f289fab96ec292a66a2552f9dcfb0bfa",
  "research/experiments/boundary_20260908/football/models.py": "fe66fd4d399efe8249d1887820884cd576541e06769d3289109e11d64d01c971",
  "research/experiments/boundary_20260908/football/run.py": "d735042fb251e324012c625ab5e51ce265c3498bf53fc175eca43f45eca81083",
  "research/experiments/boundary_20260908/football/test_math.py": "9f6a09d4562797bcdcd836bcac4ffa8e337bd430480e28029d835b3a7d24d474",
  "research/experiments/boundary_20260908/football_past_market/independent.py": "3b6705833412783daf337aa61e0d2c0a2e7abadece26947a73399666454874f8",
  "research/experiments/performance_20260907/football/benchmark.py": "4a6a06578f920eedac0c92c9f8b55fce1ba3395ce7102f6362d2f765770070eb",
  "research/projects/football/Match Result Prediction/Python/mrp/__init__.py": "bb67482296b5bf67bebe7b423d3670171d7821212e80189dcfeefbf9ebd6e117",
  "research/projects/football/Match Result Prediction/Python/run_experiment.py": "41048a0fd60054708c3ac562b126b8aac5e0cf18dbad2d47e7d98e8df3806766"
}
```

The paired calculation used the already tested independent arithmetic in `football_past_market/independent.py`, without importing that experiment's operational fitting/data modules. It was cross-checked against the original `football.run.uncertainty` helper: differences, interval endpoints and bootstrap frequency agree within 2e-15. That legacy helper reserves `dc_equal` as the reference slot, so the equal-DC candidate must first be copied to a distinct in-memory alias for the cross-check; otherwise it would compare the reference with itself. No frozen helper or stored prediction was modified. All six models' pooled NLL/Brier/accuracy/ECE also reproduced the saved table within 2e-15. No fresh model test suite or historical replay was run.

Minimal reproduction from the repository root, using only closed predictions:

```sh
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python - <<'PYCODE'
import hashlib, json
from pathlib import Path
from research.experiments.boundary_20260908.football_past_market import independent as i
p = Path('artifacts/research/boundary_20260908/football/evaluation.json')
assert hashlib.sha256(p.read_bytes()).hexdigest() == '1c1d037ffbdf00a693728254e630792315ee7db8d8f79cbdb3545d34df66a795'
a = json.loads(p.read_text())
rows = a['predictions']
assert len(rows) == len({r['match_id'] for r in rows}) == 2280
spec = {'uncertainty': {'resamples': 4000, 'seed': 20260908}}
print(i.paired_uncertainty(rows, 'dc_equal', 'production_default_dc_auto', 28, spec))
for label, subset in [('pooled', rows)] + [
    (f'{league}:{season}', [r for r in rows if r['league'] == league and r['season'] == season])
    for league in ('E0', 'SP1', 'I1') for season in (2024, 2025)
]:
    print(label, {name: {k: v for k, v in i.metrics(subset, name).items()
                         if k in ('n', 'log_loss', 'brier_sum_classes')}
                  for name in ('dc_equal', 'production_default_dc_auto')})
PYCODE
```

Suggested commit: `docs(football): audit the deployment-fit performance gap`.
