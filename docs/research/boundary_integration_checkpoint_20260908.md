# Global prediction integration and research checkpoint

The validated `frontier_live_hgb_l15_i150_20260907` model is already the default in the global prediction system. A fresh independent audit found no missing integration: global configuration and dispatch, the canonical predictor, both services, container packaging, API transport and the dashboard's **Next clean** forecast use the same implementation. No new production model or default change is justified by the subsequent experiments.

The packaged model SHA256 remains `d8bd93f1b04ecf290eb821f02397253b77248cf3c1943b05e1c540d4942e7e36`. Its target is the next same-driver eligible clean completed lap, which can skip numbered laps. Input-contract failures retain explicit fallback behavior. Point forecasts have no calibrated interval. See the [integration contract](frontier_live_integration_20260907.md) for the exact target and evidence limits.

This checkpoint publishes the work completed after [the preceding publication](boundary_followup_publication_20260908.md):

| Work | Current result | Runtime status |
| --- | --- | --- |
| Delayed historical football xG | Four candidates tested on 760 selection fixtures; best log-loss gain over the prior shot correction is 0.03674%, below the frozen 0.5% threshold, with uncertainty spanning deterioration | Research only; no transfer evaluation or promotion |
| Raw F1 sector pilot | 4,312 updates retained; 4,245 candidate checkpoints, including 14 unmatched outcomes; independent source-time and target checks passed | Verified feasibility, no predictive score |
| Expanded sector discovery inputs | TimingData, SessionStatus and TimingAppData available for all 44 discovery races, plus existing TrackStatus; raw bodies and receipts hashed | Acquisition only |
| Sector forecasting components | Completed-history parser, checkpoint context, four references, Gaussian prototype and residual-model/scoring helpers have unit tests | Unfitted prototype; complete feature/issuance orchestration, final execution protocol and historical evaluation remain pending |

The [xG execution publication](boundary_xg_execution_20260908.md) and [sector publication](boundary_sector_publication_20260908.md) preserve exact results, source fingerprints, independent verification and compact evidence copies. Earlier frozen draft statuses remain historical snapshots. New sector forecasts would be issued after S1/S2 and must beat references issued at those same checkpoints; an improvement there cannot be reported as an improvement at the existing HGB's earlier issuance.

Fresh local checks passed **55 production integration tests** and **203 new research tests**. All 351 file entries across the three preceding publication manifests match their recorded hashes: 66 original integration, 166 boundary publication and 119 follow-up publication entries. These counts include repeated files across manifests. The independent audit also matched the packaged model's bytes and verified both service container definitions include it.

The new `Boundary research contracts` workflow runs the xG and sector tests from a source checkout. It complements the existing production integration CI job. A separate local source-only copy, with no artifact or provider-data directories, passed **201 tests**, with **two optional exact-pilot replay tests skipped**. The synthetic mathematical and causal contracts remain runnable without provider downloads or historical model fitting. Test success is software verification, not evidence of a forecasting gain.

Raw provider bodies, row-level feature/fit tables and large replay ledgers remain local with recorded hashes. Compact published copies preserve the original bytes, but do not make a clean checkout sufficient for full historical replay. This checkpoint changes source and research evidence in the repository; it does not claim a hosted deployment, prospective validation or a new substantial gain.

The [checkpoint manifest](evidence/boundary_integration_checkpoint_20260908.json) records the publication files, verification commands and retained runtime identity.

Suggested commit: `ci: verify xG and sector research contracts in source checkouts`.
