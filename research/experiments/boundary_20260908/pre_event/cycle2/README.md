# Comparable-lap measurement: no material new qualifying gain

The new measurement signal almost ties the strongest historical Q2 forecast. It improves Q2 on the exposed nine-event 2026 sample but still loses to the stronger 2026 fixed blend Q1. The predefined material-success screen fails; this is retained research evidence with no production changes.

The model compares several accurate, green-flag, non-pit laps on the same compound within 180 seconds of session clock and two laps of tyre age. A robust difference regression estimates team/driver innovations around Q2 and common clock evolution, with observed tyre-age nuisance terms. Unsupported drivers retain the Q2 latent score; a conditional leave-lap-out sensitivity reduces unstable innovations. This sensitivity is not a calibrated uncertainty interval, and unknown fuel is not identified. Only the latest completed FP3 or Sprint Qualifying before GP Qualifying is used.

The four configurations were frozen before scoring. Sixteen eligible 2023 events selected penalty 4 and blend strength 0.5. Their selection MAEs were 3.73125, 3.70625, 3.68750 and 3.70625 in the order recorded in `spec.json`; Q2 scored 3.70000. Only the selected configuration was evaluated on later years. Every scored date was previously exposed, so the chronological transfer results remain retrospective.

| Period | Events / driver rows | Raw rehearsal | Fixed blend Q1 | Prior Q2 | Measurement model |
|---|---:|---:|---:|---:|---:|
| 2024 | 24 / 479 | 3.230044 | 2.980263 | 2.763377 | 2.755044 |
| 2025 | 24 / 480 | 3.366667 | 3.200000 | 3.054167 | 3.058333 |
| Pooled 2024–2025 | 48 / 959 | 3.298355 | 3.090132 | 2.908772 | 2.906689 |
| Exposed 2026 | 9 / 198 | 1.868687 | 1.848485 | 2.010101 | 1.959596 |

MAE is absolute rank-position error, averaged within each complete event roster and then equally across events. Same-row comparator and official-target equality were verified against the immutable preceding experiment.

- Against Q2, pooled 2024–2025 delta is **−0.002083 positions**, with paired-event 95% interval `[-0.016667, +0.012500]` and circular three-event block interval `[-0.016667, +0.010417]`. Five events improve, five worsen and 38 tie. The leave-one-event-out range crosses zero: `[-0.004255, +0.002128]`.
- Against Q2, 2026 delta is **−0.050505**, with event and block intervals both approximately `[-0.111111, -0.010101]`; three events improve and six tie. This is a small retrospective nine-event result, not a general superiority claim.
- Against Q1, the stronger preceding 2026 result, the candidate is **+0.111111 positions worse**. The event interval is `[-0.010101, +0.242424]`; the three-event block interval is `[+0.030303, +0.212121]`. It therefore does not meet the contemporary comparator requirement.

Only 52 of 959 historical driver ranks change from Q2; in 2026, 26 of 198 change. The 2024 round 15 event has no supported driver and exactly preserves Q2. Partial driver support fallbacks preserve latent scores, while other drivers moving can still change unsupported drivers' final rank. All 120 fitted predictions reached the stopping tolerance before 60 iterations (maximum 53); the largest objective-gradient component was `5.35113e-8`. There were no failed variants.

Seven regression tests passed. Independent verification passed 87 implementation hashes, 225 raw-input hashes and three output hashes; 73 pre-target forecast bindings; 340 rank permutations; 1,020 event metric values; reconstruction of 120 fitted predictions, the one whole-event fallback, and 461 unsupported driver-score checks. All 1,477 rows, 2023-only selection, paired bootstrap/block/leave-one-out statistics, and exact reference forecast/target pairing passed. The old reference's five broad-source hash changes remain explicitly disclosed; unchanged forecast, causal feature and target bytes are compared without relabelling the old source closure as current.

Artifacts are in `artifacts/research/boundary_20260908/pre_event/cycle2/`: `results.json`, `predictions.csv`, `selection_lock.json`, `forecasts_before_targets.jsonl`, `verification.json`, `selected_measurement_diagnostics.csv`, `label_blind_inventory.json`, and `execution.log`.

Result SHA256: `f97953fc65cd65fdbf0ad9c57f267d313a583f576d0dcd5454b0d685e35d6af8`

Frozen specification SHA256: `46f9f97006e878a369dfd659695dcd229b81cdbc1c9f3aa8d5874a721c17c905`

```sh
# Use a new output directory to preserve the frozen evidence.
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python research/experiments/boundary_20260908/pre_event/cycle2/run_experiment.py --output artifacts/research/boundary_20260908/pre_event/cycle2_reproduction
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python -m pytest -q -p no:cacheprovider research/experiments/boundary_20260908/pre_event/cycle2/test_experiment.py
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python research/experiments/boundary_20260908/pre_event/cycle2/verify_results.py
```

Suggested commit: `research(f1): test comparable-lap qualifying measurement`
