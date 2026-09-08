# Symmetric and ordered boosting on the original F1 horizon

This fixed experiment compares CatBoost Plain and Ordered boosting with the
saved original-feature HGB. The earlier frontier tested asymmetric HGB,
ExtraTrees and a row-wise neural model; neither CatBoost variant was tested.
The new comparison uses exactly the same 80 observed inputs, naive anchor,
forecast times, targets and complete population. It adds no telemetry or gaps.

Both candidates use 1,000 symmetric depth-six trees, MAE, learning rate0.03,
L2=10, exact leaf estimation, no bootstrap sampling and one CPU thread. All
parameters are explicit in the specification. Each fit uses all18,363 matched
2022 rows, with weights N/(E*n_event), clipped residual targets at±5seconds
and correction bounds at±3seconds. There is no selection eval_set, early
stopping or current-year training. SymmetricTree has no equivalent to HGB's
minimum leaf count80; the learning algorithm and regularization differ.

All20,432 original2023 issuances receive predictions before external labels
are attached. Scores use the same20,007 matched targets and retain425 unmatched
forecasts. Choose the lower event-MAE candidate, with Plain winning exact
ties, and report both. Advancement requires at least1% improvement over the
saved2022-only HGB, negative upper95% event and circular-block3 intervals,
and improvement after every single-event omission. The selected model alone
may then undergo the predeclared10%2024–25 and5%2026 transfer gates against
the current global HGB. Failed selection ends this fixed family.

Ordered boosting is motivated by prediction-shift control in the
[CatBoost paper](https://arxiv.org/abs/1706.09516). That result does not promise
an F1 gain. No categorical variables are added. The training-only permutations
do not expose2023 labels. Selecting between two variants also does not prove
that ordered boosting itself caused any eventual gain. All seasons have been
examined before; chronological execution cannot make them pristine holdouts.

The isolated1.2.10 runtime is pinned by wheel and installed-file hashes and
does not modify the main environment. A full-shape synthetic10-tree check
used under313MB peak RSS. It establishes feasibility, not model quality.
The historical runner allows4GiB peak RSS and retains immutable failure
receipts. Existing original-feature, label and baseline closures are reused;
no provider downloads or raw feature reconstruction are needed.

The independent verifier never imports CatBoost. It reads exported numeric
trees, reproduces float32 input rounding, strict split comparisons, leaf
indices and scale/bias, then checks every saved prediction and all metrics.
Synthetic tests include exact comparisons to actual small CatBoost exports.

Execution order, once the exact source review is complete:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python -m research.experiments.boundary_20260908.catboost_frontier.run freeze --review REVIEW_JSON
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python -m research.experiments.boundary_20260908.catboost_frontier.run select
```

Suggested commit: `research(f1-live): test symmetric and ordered boosting on the original horizon`.
