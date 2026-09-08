The current-peer residual family failed its 2023 advancement screen. All four
variants worsened the selected cycle1 checkpoint expert, so no new 2024–2026
transfer comparisons were run. The deterministic composition retains cycle1
for ineligible checkpoints.

| Peer residual variant | 2023 MAE | Delta versus cycle1 | Relative change |
|---|---:|---:|---:|
| Cycle1 reference | 0.713574s | — | — |
| 7 leaves, ±1s cap | 0.716690s | +0.003116s | 0.437% worse |
| 7 leaves, ±3s cap | 0.719526s | +0.005952s | 0.834% worse |
| 15 leaves, ±1s cap | 0.717028s | +0.003453s | 0.484% worse |
| 15 leaves, ±3s cap | 0.718909s | +0.005334s | 0.748% worse |

The least-bad variant was the 7-leaf, ±1s model. Its paired event delta CI was
[−0.001034, +0.007403]s; the within-year three-event block interval was
[−0.002549, +0.008632]s. Nine of 22 events improved. This does not establish a
benefit from this fitted peer expert; it also does not prove peer information
has no predictive value.

The feature encoder admits only other-driver clean observations strictly after
the old HGB issuance and strictly before the checkpoint, within 180 seconds and
one numbered lap. At least three such peers are required to activate a change.
Otherwise the exact cycle1 point is returned. Features include current field
pace, same-compound cohorts, changes paired within peer drivers, support and
age, and causal own-minus-field offsets. Whole timestamp batches are excluded
from one another's inputs. Old paired records precede the original issuance.
No target values or future stint transitions enter activation or features.

The 2022 training labels use expanding base forecasts inherited from cycle1
plus two additional expanding fits of the cycle1 expert. Every score event is
later than every stage1 expert-fit event. The 2023 comparison uses the
2022-fitted base and the cycle1 discovery expert trained on 2022 only.
There were 761 active training checkpoints. Final refits, preserved before
the rejected advancement decision, used 2,216 active checkpoints from the
2022 cross-fit support and 2023. Those final models were not transfer-scored.
The earlier architecture and historical data have prior research exposure;
this is not prospective evidence.

All recorded eligibility and provider receipt-latency limitations of cycle1
remain. Future-valued pit timestamps are masked for current-state features,
but reconstructed source flags are not an original feed-receipt log. No
production modules, previous artifacts or fitted baseline models were changed.

Evidence is in `artifacts/research/boundary_20260908/checkpoint/cycle2/`:
`results.json` explicitly records `rejected_on_2023_advancement_screen` and
`transfer_evaluated=false`. Its neighboring selection, fit and model files
remain hash-bound. Verification recomputed all 88,696 selection predictions
across the four variants, checked 44 input hashes and independently rebuilt
event MAEs. Two real-event prefix/truncation/future-poisoning cases passed for
all four models. Five semantic regression tests passed, including exact
cycle1 fallback without peer support and unchanged eligible predictions.

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv-f1/bin/python -m pytest -q -p no:cacheprovider research/experiments/boundary_20260908/checkpoint/cycle2/test_peer_experiment.py
```

The discovery and verification commands refuse to overwrite completed evidence.
Transfer was deliberately stopped after the failed selection screen; the saved
rejection result also prevents accidentally running it in this output directory.
Suggested commit: `research(f1-live): record rejected causal peer checkpoint refinement`.
