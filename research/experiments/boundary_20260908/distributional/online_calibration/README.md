# Online quantile offsets — verified, rejected at selection gate

The best frozen policy improves 2023 WIS by **0.789% over the static selected quantile model**, below the required **1%** advancement screen. Its paired three-event interval excludes zero and coverage passes, but the frozen practical threshold remains failed. **No online-calibrated transfer forecasts were generated.** The previous static distributional evidence and production model remain unchanged.

The policy uses a 0.01-second update step, reciprocal-second shrinkage 0.2, and one offset state per race shared across drivers. All four predeclared policies were evaluated on the same 20,007 original eligible issuances across 22 events in 2023. Static quantiles were trained only on expanding 2022 out-of-fold residuals. During this sequential evaluation, 2023 labels can update offsets only after their target timestamps have passed; this is explicitly an online forecasting comparison.

| Policy | Quantile-HGB base WIS | Conditional-empirical base WIS |
| --- | ---: | ---: |
| Static, no offsets | 0.3351732798 | 0.3484546323 |
| Step 0.01 s, shared race — selected candidate | 0.3325295887 | 0.3454667243 |
| Step 0.01 s, separate compounds | 0.3329918655 | 0.3461343263 |
| Step 0.03 s, shared race — best conditional comparator | 0.3329253282 | 0.3452355874 |
| Step 0.03 s, separate compounds | 0.3332799022 | 0.3456989925 |

The additional global empirical reference has WIS 0.3541337383. Against the five frozen reference roles, the selected policy gains 6.101% over global empirical, 4.570% over static conditional empirical, **0.789% over static quantile HGB**, 3.745% over its matched online conditional reference, and 3.680% over the independently best conditional-online policy. This comparison prevents attributing the value of newly observed outcomes entirely to the learned quantile model.

Against the static quantile model, selected-minus-reference WIS is **-0.0026436911 seconds**, with event-bootstrap interval **[-0.0046271691, -0.0010764776]** and three-event-block interval **[-0.0047496609, -0.0010651338]**. Seventeen of 22 events improve; every leave-one-event-out aggregate is negative. These are conditional historical intervals for a selected policy, without adjustment for four-policy selection or repeated research. They do not override the frozen advancement threshold.

Selected 50% / 80% / 90% / 95% coverage is **49.587% / 79.759% / 90.062% / 94.416%**. The original HGB point remains exactly the median in every row, with unchanged event-balanced point MAE **0.5120807233 seconds**. No point-accuracy improvement or conformal-coverage guarantee is claimed.

## Update and timing contract

For each non-median quantile level `tau`, the issued forecast is the static quantile plus its current contextual offset, followed by the frozen coherent-quantile projection around the unchanged median. The actual resulting quantile is stored with that issuance. At a later outcome arrival, use the pinball feedback `tau - 1{Y <= q_issued}`; do not compare the delayed outcome against an updated quantile.

The state update is `theta <- (1 - eta*lambda)*theta + eta*mean(feedback)` for all labels resolving at the same timestamp in the same context. Here `eta` has units seconds and `lambda` has units reciprocal seconds. The four frozen settings use eta 0.01/0.03 and either a race-wide or race-and-issuance-compound state; lambda is always 0.2. The median offset remains zero. Sorting and median constraints mean this is a custom delayed quantile-feedback method, not a claim of unconstrained gradient descent or convergence in raw offsets.

Every race starts with zero offsets. All forecasts at a timestamp are issued before any outcome at that same timestamp is assimilated; every recorded feedback timestamp is therefore strictly earlier than its forecast. This also conservatively excludes own-car feedback arriving exactly at issuance. Same-time label batches use their mean gradient once, with no arbitrary within-batch ordering. Compound context is recorded at original issuance, never chosen using the target. No row is removed based on future stint equality, target value or future availability.

The inherited population itself is the previously matched retrospective next-eligible-lap dataset. Provider eligibility/accuracy flags, original publication latency and previously exposed research dates remain limitations. This code does not claim reconstruction of unavailable historical client receipt timestamps.

[Adaptive Conformal Inference](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html) and its [later adaptation work](https://jmlr.org/beta/papers/v25/22-1218.html) motivate outcome-driven updates under shift. This implementation uses custom regularized delayed pinball feedback; it is not either published conformal algorithm and inherits no stated coverage guarantee.

## Verification and immutable evidence

Eight pre-score tests passed in **1.62 seconds**. An independent pre-score reviewer checked timing, frozen-issued feedback, context assignment, simultaneous batches, comparator selection and update units. That review found a product-only parameter check that admitted two negative values. Explicit finite-positive checks and a regression were added **before source locking and any scores**; the four positive settings and gates were unchanged.

The independent verifier uses an issuance-driven heap, separate from the scoring runner's prebuilt action schedule. It replayed **all eight online policies on all 20,007 selection rows**, reproduced every quantile and feedback timestamp exactly, checked strictly prior feedback and unchanged medians, recomputed WIS from interval penalties, both bootstrap intervals, selections and gates, and confirmed that no transfer artifacts exist. Maximum forecast/point replay difference is **0.0 seconds**. There were no static model refits; causal state updates were replayed in 6.14 seconds with one CPU thread. No source changes followed scoring.

Artifacts are in `artifacts/research/boundary_20260908/distributional/online_calibration/`. Compact source/findings publication can retain all JSON locks, tests/review records, `selection.json`, `results.json`, `verification.json` and `evidence.json`. The pickled forecast and feedback audit matrices are local replay artifacts with recorded hashes.

- Specification SHA256: `9099de60861c0c99a5c9f9d51a603ba495b3bf23e309eef4801e69e2a39d9a39`.
- Source fingerprint: `81ebb8e6c6c1a9819982a9e338ecc4e81150ee350014820474405d230be24516`.
- Execution lock SHA256: `bcb4e3d34763498ffb19e6dbdd405c67176396cbe60da3f67d245ea63f26b64c`.
- Selection SHA256: `d07700e776d5d86d0104998089b6d56ac930f2a752e5193a3f4c7019248dbe4c`.
- Verification SHA256: `44a8d33d515a748c0a303d06a3935ec3b191882c24e71bddd236dbd0ab705608`.
- Evidence SHA256: `8a142c25f2ffd9bcaba98463e6a6ba1703134d12914adcbc53cc71930bb92e89`.

Run tests from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python -m pytest -q -p no:cacheprovider research/experiments/boundary_20260908/distributional/online_calibration/test_model.py
```

Replay with the same frozen source and inherited local artifacts, preserving completed output:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python - <<'PY'
import json
import shutil
from research.experiments.boundary_20260908.distributional.online_calibration import run, verify
original = run.OUT
run.OUT = original / 'replay_clean'
run.OUT.mkdir(exist_ok=False)
for name in ('design_lock.json', 'execution_lock.json'):
    shutil.copy2(original / name, run.OUT / name)
run.discover()
if json.loads((run.OUT / 'selection.json').read_text())['advancement_passed']:
    run.transfer()
verify.main()
PY
```

Suggested commit: `research(f1-live): evaluate causal online quantile calibration against matched references`.
