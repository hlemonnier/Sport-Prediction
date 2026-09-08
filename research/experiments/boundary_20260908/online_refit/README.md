Periodic race-event refitting failed the predeclared validation screen. The
best of five policies was cumulative equal-event-weight HGB, with only
**0.3958%** lower 2023 event-balanced MAE than the fixed 2022-trained baseline.
The required threshold to open the later transfer table was 0.5%, so no
2024–26 evaluation or production change was made.

The experiment refits at each year's first race and every four races thereafter.
Every fit contains only strictly earlier completed events. It keeps the current
80-feature schema, target, prediction time, target clipping and output clipping.
The bounded alternatives used 16/48-event exponential forgetting and 15/31
leaves. The other validation gains were −0.2064%, +0.1000%, +0.1005% and
+0.2857%; all five variants and 30 chronological fit blocks are preserved.

This tests a concrete model-maintenance policy motivated by
[non-stationary online regression](https://jmlr.org/papers/v16/moroshko15a.html)
and [forgetting in dynamic prediction](https://arxiv.org/abs/1809.05870), not a
reproduction of those algorithms. The result does not support a large gain from
this maintenance policy alone, and says nothing about untested richer inputs.

Run from the repository root with `PYTHONPATH=.` and one numerical thread:

```sh
.venv-f1/bin/python -m pytest -q research/experiments/boundary_20260908/online_refit/test_run.py
.venv-f1/bin/python research/experiments/boundary_20260908/online_refit/verify.py
```

The verifier reconstructs all scores, checks every fit/prediction boundary and
raw input hash, and exactly refits the selected six-block policy. `run.py discover`
refuses to overwrite an existing selection. The first test run used exact
floating-point equality for normalized weights; that test assertion was changed
to tolerance1e-14 after observing1.0000000000000002. No fitting code or frozen
design changed.

Suggested commit: `research(f1-live): evaluate causal rolling event refits`.
