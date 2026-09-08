# Global track-control correction: stopped at discovery selection

The four frozen variants did not establish a useful gain over the strongest verified checkpoint model, C1. The selected context-gated 7-leaf model reduced 2023 event-balanced MAE by **0.1761%**, below the predeclared 0.5% advancement threshold. Its uncertainty and leave-one-event-out checks also failed. No later-year control data were acquired or scored, no final model was fitted, and serving remains unchanged.

## Design and population

The acquisition in the parent directory remains immutable. This model experiment was subsequently authorized and frozen separately. The design uses global status records strictly earlier than `checkpoint_time - 15 seconds`, on the recorded session clock. It retains unknown states, distinguishes missing fields from explicit unknown updates, preserves source order at repeated timestamps, and treats VSC ending as neutralized. Prefix-only durations never assign a known state to an unknown interval. Download receipts do not establish historical message receipt latency. The experiment inherits C1's retrospective recorded eligibility assumptions.

Training reuses 10,401 honest C1 out-of-fold checkpoint predictions from 2022 R12–R22. The underlying HGB and C1 fits each precede their scored events. Selection uses the exact 22,174 checkpoints from 22 races in 2023, with HGB fitted on 2022 and C1 fitted on 2022 expanding HGB forecasts. The 20,007 eligible checkpoints retain the identical HGB/C1 points. All models use the same targets and rows; no per-event choice is allowed.

Four variants combine 7/15 HGB leaves with all known ineligible checkpoints or current/recent flag context. The latter means a current non-clear flag or positive observed non-clear duration in the last 300 seconds. Training support is 1,361/all or 870/context rows; selection support is 2,167/all or 1,094/context. The 222 fixed features comprise original C1 inputs, its point/correction, and 77 global control features. Peer C2 features are excluded. The loss is equal-event weighted absolute residual error, target clipped to ±12 seconds and predicted correction clipped to ±6 seconds; all hyperparameters appear in `specification.json`.

## Results

MAE and deltas are seconds; positive relative gain means lower error. These are retrospective 2023 selection results, not independent transfer evidence.

| Model | Full checkpoint MAE | Gain versus C1 | Target-balanced MAE |
|---|---:|---:|---:|
| Carried HGB | 0.754962141 | — | 0.512080723 |
| Selected C1 reference | 0.713574479 | — | 0.497350872 |
| Control 7 leaves, all | 0.723246376 | −1.3554% | 0.500021853 |
| Control 7 leaves, context (selected) | 0.712317645 | +0.1761% | 0.497089973 |
| Control 15 leaves, all | 0.721205279 | −1.0694% | 0.499701683 |
| Control 15 leaves, context | 0.713711419 | −0.0192% | 0.497484657 |

For the selected model against C1, the full delta is −0.001256833 seconds; its paired event bootstrap 95% interval is [−0.011978281, +0.005836112], and its circular three-event block interval is [−0.011389572, +0.006121052]. Four events improve, eleven worsen, and seven tie. Removing its best event (202313) reverses the aggregate delta to +0.003174581 seconds. The neutralized checkpoint subgroup also worsens by 0.016357342 seconds. This is a concentrated, uncertain gain, not evidence for integration.

Target-balanced gain versus C1 is 0.05246%; its block interval for the delta is [−0.002752235, +0.001615557] seconds. Fixed-model latency sensitivity gives tiny full gains of 0.2268% at lag 0 and 0.1373% at lag 60, with intervals crossing zero. No sensitivity refitting or reselection occurred.

The selected model improves over carried HGB by 5.6486% full and 2.9274% target-balanced, mostly inherited from C1. Neither reaches the intended 10%/5% transfer target, and the explicit C1 advancement gate fails before transfer. The result justifies stopping this family; it does not establish that global control state has no predictive information for other targets.

## Verification and reproduction

Thirteen causal/unit tests pass. The verifier reproduces 32,575 control-feature rows; independently reproduces all 32,575 cached HGB/C1 points from their chronologically appropriate saved models; reproduces 133,044 candidate and latency predictions and every stored metric/interval; checks original features and target pairing; independently recalculates event- and target-balanced means; and verifies prefix truncation plus future poisoning at 80 real checkpoints across two races. All 132 control input hashes and 20 design dependencies are verified.

The source/specification is frozen by `artifacts/research/boundary_20260908/control/model/design_lock.json`. Local raw feeds and earlier C1/C2 caches are required; these commands do not download data. `prepare` and `discover` refuse to overwrite frozen artifacts.

```sh
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python -m pytest -q -p no:cacheprovider research/experiments/boundary_20260908/control/model/test_model.py
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python research/experiments/boundary_20260908/control/model/run_experiment.py prepare
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python research/experiments/boundary_20260908/control/model/run_experiment.py discover
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python research/experiments/boundary_20260908/control/model/verify.py
```

Frozen specification SHA256: `6bb27211223cff1342fe1393c6ad28240f8a30c9a827c7281bc3c74189606d0a`.

Results SHA256: `ce8dfb6034ed33269a5b0bd7f1839f061250238b4c589960bdfbdc6c85bdf859`.

Suggested commit: `research(f1): evaluate causal track-control checkpoint corrections`.
