# Portable xG test contracts

This suite preserves the 100 experiment-time model tests with one numerical portability correction. Linux CI at `cc995ca` reported three failures at the assertion comparing one-row predictions with the same row in a two-row batch. The largest discrepancy was 1.1102230246251565e-16, consistent with floating-point accumulation changes between BLAS matrix operations of different shapes. All other assertions in those tests passed, including exact JSON replay and exact target-poisoning invariance.

Only that batch-shape comparison uses an absolute tolerance of eight float64 epsilons (approximately 1.78e-15), with relative tolerance zero. This is a numerical equality check, not a statistical tolerance or permission to change predictions using other rows. The assertions that fitted scaling is unchanged, serialization replays exactly and current target fields are unreadable remain byte-for-byte identical.

The original `../execution/test_model.py` remains immutable at SHA256 `89fa364c5f9157a82f1df27b072d23e45ef6e4c301bb179dce8cf2f5dce9f6fe`, preserving the pre-fit design lock and all historical evidence. `test_source_contract.py` verifies both its checksum and that the portable copy differs in exactly the one declared assertion. No runtime model, historical fitted coefficient, prediction, score or selection gate changes.

CI explicitly selects this portable suite and `../execution/test_run.py`. The archived execution suite retains its original platform-dependent bit-equality assertion and is not the portable CI entry point. To run the portable xG checks:

```sh
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python -m pytest -q --import-mode=importlib research/experiments/boundary_20260908/football_xg/ci_contracts research/experiments/boundary_20260908/football_xg/execution/test_run.py
```

Suggested commit: `fix(ci): allow floating-point roundoff across prediction batch sizes`.
