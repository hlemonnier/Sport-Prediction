The online residual hypothesis failed its predeclared 2023 screen. The best of
eight candidates reduced event-balanced MAE by only **0.1767%** against the
strong nonlinear baseline; the required threshold to open the later transfer
table was 0.5%. No 2024–26 transfer was run for this candidate.

Four direct residual filters worsened MAE by 1.95–6.37%; the two Huber corrections
worsened it by about 2%. The seven-leaf residual HGB produced the tiny positive
result. These findings do not support a material gain from carrying recent
forecast errors into the next prediction under this particular state design.
They do not prove that all online learning or dynamic factor models are futile.

The replay maintains driver and recent peer residual state. It releases each
residual only when its actual target lap completes, permits a driver's own
current completion, excludes equal-timestamp peer updates, ignores cross-stint
residual updates, and resets driver state across stint generations. It preserves
the production model's original issuance times and next-eligible-lap targets.
This is separate from the checkpoint expert, which uses later information.

The fixed baseline HGB was trained on 2022 for 2023 selection. Meta training used
three expanding 2022 base-model folds with disjoint future prediction blocks.
All eight candidates, cross-fit blocks, source/input locks and predictions are
recorded under `artifacts/research/boundary_20260908/online_residual/`.

The motivation comes from work on [non-stationary online regression](https://jmlr.org/papers/v16/moroshko15a.html)
and [forgetting in dynamic-system prediction](https://arxiv.org/abs/1809.05870).
The tested HGB/Huber residual features are a custom implementation, not those
papers' algorithms, and their theorems do not establish performance here.

Reproduce from the repository root with one numerical thread and `PYTHONPATH=.`:

```sh
.venv-f1/bin/python -m pytest -q research/experiments/boundary_20260908/online_residual/test_run.py
.venv-f1/bin/python research/experiments/boundary_20260908/online_residual/verify.py
```

`run.py discover` refuses to overwrite an existing selection. The verified
artifacts should be preserved if another experiment is designed. The existing
CSV records provide reconstructed eligibility rather than original feed receipt
times, and all later race dates have previously been examined in other research.
No production change, calibrated uncertainty or strategy value is claimed.

Suggested commit: `research(f1-live): test causal online residual assimilation`.
