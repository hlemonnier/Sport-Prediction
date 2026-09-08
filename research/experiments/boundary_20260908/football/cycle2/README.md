# Nonlinear residual cycle — completed, rejected on selection

All three frozen nonlinear challengers performed worse than the existing shot correction on the 760 EPL selection matches. The experiment retained that correction exactly. No losing tree model was evaluated on the transfer dates, and this cycle establishes **no additional predictive gain**.

| Model | Selection log loss |
| --- | ---: |
| Existing 90-day shot correction, penalty 0.1 — retained | 0.9581581184934563 |
| Earlier DC365/Elo50 blend | 0.9595989049822483 |
| Depth 2, 25 trees | 0.9612071929840409 |
| Depth 2, 75 trees | 0.9634880236771010 |
| Depth 3, 75 trees | 0.9651112052518959 |

This is an observed generalization failure. Every ensemble decreased its penalized training objective. Across the twelve historical selection refits, mean objective decreases were 0.00754, 0.01568 and 0.02575 as complexity increased, while selection loss worsened. Optimizer failure cannot explain away this result.

The retained model's evaluation is the unchanged cycle1 result: 2,280 matches, log loss 0.9841272141. Its gain is 4.303% against the controlled production comparator, 0.265% against the earlier blend and 0.836% against DC180. The 28-day interval against the earlier blend crosses zero. Against the retained cycle1 model, this cycle has exactly zero difference. The requirement of a 2% gain against all three strong research references therefore fails. These numbers are repeated incumbent results, not a new tree-model transfer test.

## Mathematical implementation

Let `z_i = log(p_prior_i) + sum_t f_t(x_i)`, where the prior is the frozen DC365/Elo50 probability and each tree contributes a three-component leaf vector. The probability is `softmax(z_i)`. Zero trees reproduce the prior exactly.

The stagewise objective is mean negative log likelihood plus `lambda / (2*n)` times the sum of squared **actual added** leaf vectors. For each new tree, partitions are selected by fitting a shallow regression tree to the negative multinomial log-loss gradient `one_hot(y) - p`. Within a fixed leaf, the unshrunk vector `w` minimizes:

`sum_i_in_leaf [logsumexp(z_i+w) - (z_i+w)[y_i]] + lambda/2 * ||w||²`.

Its exact gradient is `sum_i(p_i - one_hot(y_i)) + lambda*w`; its full Hessian is `sum_i(diag(p_i) - p_i p_i.T) + lambda*I`. With frozen `lambda=25`, the Hessian is positive definite, including the otherwise unidentified common-logit direction. Newton updates use an Armijo line search and a maximum gradient norm of 1e-6. This uses the complete cross-class Hessian, not three independent binary fits.

The fitted leaf vector is multiplied by learning rate 0.05 before adding it. Convexity of the fixed-partition leaf objective ensures that shrinking a descent solution toward zero also descends. The code checks the ensemble's penalized objective after every added tree. Tree partition selection remains greedy; the experiment makes no global optimality or global convergence claim.

All candidates use at least 100 training rows per leaf. Features are the prior log odds, DC/Elo disagreement, country indicators, 90-day predicted shot and shot-target means, prior confidence, and DC180/prior disagreement. The original incumbent correction is a comparator, not an input feature. Inputs are rounded to float32 for tree routing, matching scikit-learn; double-precision thresholds are preserved. A nextafter regression checks threshold-adjacent external float64 inputs.

## Timing, selection and evidence

The design reuses cycle1's immutable prequential forecasts. Each tree model trains only on strictly prior, available rows in the trailing 1,460 days and refits at the same inherited two-month league cutoff. EPL 2022/23–2023/24 selects among the three candidates and two identity fallbacks. The frozen fallback retains the best previous model. No transfer-based selection occurs.

All 2024/25–2025/26 dates and earlier cycle metrics were already exposed. The repeated incumbent transfer summary is retrospective. Historical meta-training includes all three countries; this is not a country holdout. Source statistics assume availability by the next local midnight; publication timestamps are unavailable. Known resumed fixture and provider count anomalies remain unchanged.

Specification SHA256: `120107763852b21dce64dd0055429391123b33ce740e0829a2ac662e4689e62d`.

Source fingerprint: `eb0ce9bc7ad6407482177272e1481da4af9263bfd2570ae4df1ccb6a7e39cb19`.

Evidence SHA256: `f40b8c2957b108eff792f5885363ae661a4d51f6f5c3cf9944f284c4e01f5552`.

All output is under `artifacts/research/boundary_20260908/football/cycle2/`: the design/source/runtime locks, complete `selection.json`, incumbent-only `evaluation.json`, `verification.json`, compact `evidence.json` and execution logs. Earlier cycle artifacts were not changed.

Verification replayed 36 ensembles, 2,100 trees and all 10,133 fitted leaf gradients. Maximum leaf gradient was 9.9636134e-7, within the frozen 1e-6 tolerance. Forecast replay had maximum absolute error 0.0. Exact training membership, availability, all parent/source hashes and stage objective monotonicity passed. Selection took approximately 38 seconds, with serial fits and one CPU thread per fit. The incumbent-only evaluation export took less than one second.

Thirteen combined regressions passed in 2.60 seconds:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python -m pytest -q -p no:cacheprovider research/experiments/boundary_20260908/football/test_math.py research/experiments/boundary_20260908/football/cycle2/test_model.py
```

To reproduce this cycle without overwriting its completed artifacts:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python - <<'PY'
import shutil
import sys
from research.experiments.boundary_20260908.football.cycle2 import run, verify
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

Suggested commit: `research: reject nonlinear football residuals on chronological selection`.
