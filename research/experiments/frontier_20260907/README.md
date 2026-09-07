**Live forecasting frontier — 7 September 2026**

The 2023-selected boosting residual model clears the predeclared research gate: 10.07% lower event MAE in 2024–2025 and 10.58% in the 13 available 2026 races. See the [full report](../../../docs/research/frontier_live_performance_20260907.md) for exact populations, uncertainty, input recovery, all competitors and limitations.

Canonical artifacts are under `artifacts/research/frontier_20260907/live/`:

- `selection.json` retains every one of the 36 candidates and the selection-period scores.
- `fit_lock.json` and `fitted_models.pkl` freeze the selected configurations and 2022–2023 fits.
- `corrected_input_contract/results.json` is the final transfer evidence; its neighboring verification and forecast files support reconstruction.
- `candidate/model.pkl` contains only the preferred fitted research model plus its input and provenance contract. Its `manifest.json`, `verification.json`, and observed-prefix CSV demonstrate inference without a future target.
- `attribution/results.json` contains post-discovery diagnostics, with no reselection.

The first `live/results.json` is preserved but is not canonical: the four recent exports omitted observed features required by this new head. `input_recovery/manifest.json` records the offline recovery from unchanged original cache payloads. This used no new fitting or tuning. The corrected run is amended retrospective evidence, not a prospective result.

The scientific source in `live_frontier.py` remains frozen. `feature_encoder.py` copies its exact feature expressions into an inference-only interface with two changes: it rejects incomplete schemas, and it returns issuances even when no future target has arrived. Equivalence is verified. `candidate.py` exports and runs the selected model; it does not modify production defaults.

Run the focused tests from the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv-f1/bin/python -m pytest -q research/experiments/frontier_20260907
```

Use the existing frozen bundle on an observed prefix. Timing columns must already be numeric seconds; sector, speed, position and fresh-tyre fields must be present. This example uses only observations through session-relative time 7,800 seconds:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv-f1/bin/python research/experiments/frontier_20260907/candidate.py predict --laps data/f1/frontier_20260907/recent_full_observations/2026_round_13_italian_grand_prix_race_laps.csv --event-key 202613 --cutoff 7800 --output artifacts/research/frontier_20260907/live/candidate/my_prefix_predictions_7800.csv
```

To inspect and reproduce the existing evidence, without refitting or reading network data:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv-f1/bin/python research/experiments/frontier_20260907/verify_live_frontier.py
MPLCONFIGDIR=/private/tmp/sport-frontier-matplotlib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-f1/bin/python research/experiments/frontier_20260907/summarize_frontier.py
```

Fresh full-experiment execution uses, in order, `live_frontier.py discover`, `live_frontier.py transfer`, `recover_recent_inputs.py`, `corrected_transfer.py`, `verify_live_frontier.py`, `ablate_live_frontier.py`, and `candidate.py export`. Runners intentionally use fixed isolated artifact locations and refuse to overwrite completed experiments; a fresh reproduction requires an output-free research copy with the same raw inputs and cached observations. The initial transfer stage retains the input-contract failure as part of the execution record. Provider revisions can change inputs, so validate hashes before equating a replay with the original cycle.

Runtime: Python 3.12, NumPy, pandas, SciPy/scikit-learn, PyTorch, FastF1, matplotlib and pytest in the existing `.venv-f1` environment. Model fitting and verification use one numerical-library thread. The pickle files are locally generated research artifacts and must be used with compatible Python/scikit-learn versions.

This branch contains research code and local artifacts. No new commit, push, automation or production promotion has been performed for this cycle.

Suggested commit: `research(f1-live): add verified nonlinear forecasting challenger`.
