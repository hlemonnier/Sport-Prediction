# Same-checkpoint sector forecasting experiment

This execution tests whether raw S1/S2 prefixes improve forecasts of the next same-driver recorded eligible clean completed lap. It uses 2022 for fitting and 2023 for selection. The target may skip the current numbered lap; unmatched outcomes and unsupported checkpoints remain explicit. This is a later issuance than the existing HGB's lap-end forecast, with a separate comparison population.

The frozen candidate family is one regularized Gaussian conditional-remainder model and two residual HGB models with seven/fifteen maximum leaves. Each must beat four fixed references at the same checkpoint. The 75-column encoder uses only recorded prefix information and strictly earlier raw-valid completed histories. Final eligibility and canonical labels are attached separately after target-free ledgers close.

Selection requires at least 2% target-balanced event-MAE improvement against every reference, a negative upper three-event bootstrap bound and no checkpoint-MAE deterioration. Only a passing result opens the declared 2024–2026 transfer scope. The complete substantial-gain gates, all 44 discovery events and all 61 transfer events are in [specification.json](specification.json). Exact model formulas and limitations are in [mathematical_review.md](mathematical_review.md).

This v2 execution corrects an input-availability restriction discovered in the preserved, unscored [first attempt](../execution/). At the 2022 Hungarian GP, a genuine S1 correction from 30.526 to 30.486 seconds arrived in the same packet as S2. V2 requires a prior positive S1 observation in the same epoch, then uses the latest atomic-packet value at its actual source clock. The earlier forecast remains immutable. Candidate models, numerical feature definitions and validation gates are unchanged; the exact amendment and original failure fingerprints are in the specification.

Status at source preparation: independent pre-fit review in progress; no historical HGB model has been fitted or scored in this execution. Closed run artifacts and the subsequent publication record the outcome. This source snapshot and the original failed attempt remain preserved after execution.

The runner freezes tested source and input hashes, builds all target-free discovery ledgers, labels/fits only 2022, closes every 2023 model forecast, then attaches 2023 targets and scores. Run from the repository root with one numerical thread:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python -m research.experiments.boundary_20260908.sector_forecast.execution_v2.run freeze
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python -m research.experiments.boundary_20260908.sector_forecast.execution_v2.run build
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python -m research.experiments.boundary_20260908.sector_forecast.execution_v2.run select
```

The commands preserve completed outputs and failed attempts. Raw local inputs must match the acquisition manifest. A new output directory can be specified with `--out`; do not overwrite a frozen experiment or reuse its artifacts after changing runtime source. Source review and synthetic tests do not establish a forecasting gain.

Suggested commit: `research(f1-live): execute frozen same-checkpoint sector forecast comparison`.
