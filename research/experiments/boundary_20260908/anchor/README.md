# Unclipped live-lap anchors: rejected on 2023 selection

All four frozen anchor candidates were worse than the current HGB baseline on 2023. The predefined advancement screen failed, so **no 2024–2026 candidate was fitted or scored**. Root's separately frozen checkpoint composition therefore retains the existing HGB on the original eligible checkpoints.

The experiment reconstructs unclipped median-of-three and median-of-five anchors directly from raw, causally available clean laps within the current stint. It masks future pit timestamps and propagates missing stint/compound state forwards. A model predicts the residual to each median with the baseline HGB's fixed architecture and all 80 original features plus six unclipped anchor gap/count/MAD features. Both residual targets and candidate point corrections are unclipped. Each anchor has a full-population and a fixed gate policy: use it only with at least three supporting laps and an observed anchor gap of at least one second; otherwise preserve the baseline point exactly.

Two anchor models and the baseline were trained on 18,363 matched 2022 rows. The four policies were selected using 20,007 matched rows from 22 events in 2023. No random train/test split, held-out target-dependent gate or fitted mixture was used. All evaluation dates were already exposed before this research.

| Candidate | Event-balanced MAE, seconds | Change versus HGB | Relative error increase |
|---|---:|---:|---:|
| Existing HGB | 0.512081 | — | — |
| Median 3, full | 0.535808 | +0.023727 | 4.633% |
| Median 3, gated | 0.518406 | +0.006325 | 1.235% |
| Median 5, full | 0.543879 | +0.031799 | 6.210% |
| Median 5, gated | 0.528794 | +0.016713 | 3.264% |

The selected median-3 gate is simply the least harmful candidate. It wins ten of 22 events. Its paired-event 95% delta interval is `[-0.005801, +0.021954]` seconds and its circular three-event block interval is `[-0.004836, +0.019573]`; the maximum leave-one-event-out delta is positive at `+0.009177`. It fails both the required 0.5% improvement and the requirement that every leave-one-event-out delta be negative.

The gate activates on 1,046 of 20,007 selection rows. The difficult, observed anchor-gap-above-three-seconds group contains 155 rows; its row MAE rises from 4.6671 to 5.5813 seconds. Those group values are descriptive, not new selection criteria. They provide no support for simply relaxing a correction cap or assuming a large recent-lap change is a transient outlier. Persistent changes in conditions are one possible explanation, not an identified cause.

Seven tests passed, including unclipped medians, causal prefix invariance, future-pit masking, state resets, exact gate fallback and fixed tie-breaking. Independent verification confirmed:

- All 38,370 discovery rows and all 80 original features remained identical to the baseline data; only the six declared model features were added.
- All 44 raw-input hashes and three implementation/specification hashes match.
- Exact original issuance keys, target laps, target timestamps and target seconds are retained.
- All 100,035 baseline/four-candidate selection points reproduce from saved models, and all four event/bootstrap/block/leave-one-out metrics reproduce.
- Two real-event truncation and future-poison checks pass, with 499 and 455 retained prefix observations.
- No frozen transfer model or candidate transfer predictions exist.

The canonical transfer path in the code is the 61-event full-schema manifest in `artifacts/research/frontier_20260907/live/corrected_input_contract/results.json`; it deliberately excludes the narrow `live_recent_final` source route. Transfer was not executed after discovery failed.

Evidence lives in `artifacts/research/boundary_20260908/anchor/`, including `results.json`, `selection.json`, `selection_predictions.pkl`, `selection_predictions.csv`, `selection_models.pkl`, `discovery_anchors.pkl`, `verification.json`, `selection_failure_diagnostics.json` and `discovery_execution.log`.

- Result SHA256: `e4e6a124d1ddb850822b40b5c5b95aef11a4a1f12ed275f8fd5f6a3ea9ffef00`
- Selection SHA256: `36740bf4328a721ebab358e56df06b5c340c5bc97042a254a5b555d14798ebb1`
- Selection points SHA256: `838a2ca6a9fbeba97a43de16a90664d632efca67e6cae8d65715f722397c780a`
- Frozen specification SHA256: `74b3a622439e960a1de7be2d578d9448d9c64fab53add42a558f3886f60b4367`

Executed commands from the repository root:

```sh
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python research/experiments/boundary_20260908/anchor/run.py discover
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python -m pytest -q -p no:cacheprovider research/experiments/boundary_20260908/anchor/test_run.py
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python research/experiments/boundary_20260908/anchor/verify.py
```

The runner refuses to overwrite the existing discovery. For a repeat, import `run.py`, set its `OUT` to a new directory under the anchor artifact directory, then call `discover()`; the model, feature and selection logic remain unchanged. The transfer command rejects the failed advancement screen.

No production source or Git state was changed. Suggested commit: `research(f1): reject robust live anchors on frozen selection`
