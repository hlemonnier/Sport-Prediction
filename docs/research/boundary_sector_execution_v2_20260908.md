# Sector selection outcome and global integration verification

**The validated `frontier_live_hgb_l15_i150_20260907` is already integrated into the global prediction system.** Fresh independent checks found no missing wiring in the default configuration, canonical predictor, both services, dashboard or container packaging. All **55 integration tests** passed and all **66 original publication file bindings** remain exact. The packaged model SHA256 is `d8bd93f1b04ecf290eb821f02397253b77248cf3c1943b05e1c540d4942e7e36`. [Integration contract](frontier_live_integration_20260907.md).

The corrected sector experiment also completed and was independently verified. Its best candidate improved 2023 target-balanced error by **6.55%** against the strongest same-checkpoint reference, but the fixed uncertainty criterion failed. This candidate remains research only. No sector transfer acquisition, later-year evaluation, production activation or new substantial-gain claim follows from this result.

## Fixed comparison and result

Two residual HGB models fitted **42,084 resolved 2022 checkpoints**. Their 2023 forecasts were closed and hashed before attaching 2023 outcomes. All three candidates and four references used the same **43,974 issued checkpoints across all 22 declared 2023 races**: 43,773 had a later recorded eligible outcome, representing 19,851 event-targets; 201 unmatched issuances remain explicit.

Errors below are in seconds, averaged equally over events. Target-balanced error first averages repeat issuances for the same driver-target within each event.

| Model | Checkpoint MAE | Target-balanced MAE |
| --- | ---: | ---: |
| Last observed remainder template | 0.507520 | 0.332996 |
| Joint median of five templates | 0.582955 | 0.400612 |
| Recency-weighted median of ten | 0.572143 | 0.389599 |
| Pace-scaled median of five | 0.590801 | 0.413329 |
| Gaussian conditional remainder | 0.584597 | 0.405887 |
| Residual HGB, seven leaves | 0.487639 | 0.313240 |
| **Residual HGB, fifteen leaves — selected** | **0.485165** | **0.311187** |

Against the strongest reference, the selected model reduces checkpoint error by **4.40475%** and target-balanced error by **6.54947%**. Twenty of 22 events improve. The target-balanced candidate-minus-reference difference is **−0.0218095 seconds**, with paired three-event 95% interval **[−0.0435470, +0.0067353] seconds**. Its upper bound is positive, so the reference's advancement gate fails. The other three reference gates and complete-event coverage pass. The fixed rule requires every reference gate; the candidate was not replaced or retuned after scoring. [Exact selection result](evidence/boundary_sector_execution_v2_20260908/execution_v2/selection.json).

## Correction and independent verification

The [preserved first attempt](boundary_sector_execution_v1_20260908.md) stopped on an atomic S1 correction arriving with S2. The separate v2 execution keeps the pilot's first prior positive S1 as an attribution prerequisite and uses the latest received value at its actual packet clock. Models, numerical feature definitions and validation gates are unchanged.

The completed 44-event target-free ledger retains **94,828 updates**: 86,384 issued, 6,610 lacking enough history and 1,834 excluded by the pilot. Independent comparison found all **25,360 rows** saved by the original attempt unchanged in order, status, points, 75 features, context and history. Only new S1 provenance was added. Hungary's corrected prefix is exactly **30.486 / 31.233 seconds**, with its actual clock retained; the earlier S1 forecast still uses 30.526 seconds.

Independent selection verification checked **344 file bindings**, manually read **47,997 canonical CSV rows**, and re-resolved all **94,828 retained target associations**. Both fitted HGB models replayed all 43,974 selection forecasts with maximum difference **0 seconds**. Metrics, intervals, leave-one-event-out results, subgroups and selection gates matched within 1e-12. [Verification receipt](evidence/boundary_sector_execution_v2_20260908/execution_v2/verification.json).

The final v2 pre-fit lock records **231 passing tests**. The complete local research CI command passed **454 tests**, including both preserved executions and the existing xG/sector contracts. Those counts overlap and must not be added. The assembly comparator initially rejected JSON nulls used for missing features; its original source and failed receipt are preserved, and its corrected schema check passed without modifying any frozen source or forecast table.

## Scope and reproduction

These S1/S2 forecasts use later information than the global model's completed-lap issuance. They are not measured improvements at that earlier horizon. Errors are conditional on a later recorded eligible target existing, and archived stream clocks are replay proxies rather than measured historical client receipt times. The 2023 selection data are historical research data; these intervals are descriptive, not prospective or adjusted for the broader model search. No calibrated forecast interval, race-order improvement or strategy value is established.

The [publication manifest](evidence/boundary_sector_execution_v2_20260908/publication_manifest.json) binds the exact compact evidence copies, source files and fresh global integration receipt. Raw streams, canonical CSVs, fitted pickles and row-level ledgers remain local with their hashes; a fresh clone alone cannot replay the full historical experiment. The existing CI workflow discovers the new synthetic tests without a workflow change.

The [execution README](../../research/experiments/boundary_20260908/sector_forecast/execution_v2/README.md) contains the freeze/build/select commands. The completed independent verification used:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python -m research.experiments.boundary_20260908.sector_forecast.execution.verification.verify --selection-complete --execution-dir research/experiments/boundary_20260908/sector_forecast/execution_v2 --out artifacts/research/boundary_20260908/sector_forecast/execution_v2 --design-sha256 c716d217b15bd797e219b5f0a10503cb9bce0811b5438545847bcd58c03e993e
```

The verifier preserves completed evidence with exclusive writes; the recorded command will not overwrite the existing verification receipt. Earlier source preparation and pre-fit status text remains an immutable historical snapshot. This report records the subsequent outcome. Repository integration is verified; a hosted deployment is not claimed.

Suggested commit: `docs(research): publish verified sector selection and global integration status`.
