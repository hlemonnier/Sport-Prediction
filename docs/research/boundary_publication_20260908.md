# Boundary publication and global integration check

**The verified lap-time HGB is already integrated in the global prediction system.** This release preserves the latest research and verifies that integration. It introduces no new production model, refit or forecast policy. The shared model was integrated in commit `1f26080`; CI wiring was completed in `8e900a6`.

The default `PredictionConfig.f1_live_next_lap_point_model` is `frontier_hgb`. Global dispatch calls the canonical live predictor. Both the prediction service and platform fallback use `packages/f1/models/live_race/next_lap.py`, and the dashboard displays its separate “Next clean” estimate. The portable model has SHA256 `d8bd93f1b04ecf290eb821f02397253b77248cf3c1943b05e1c540d4942e7e36`. All 66 files in the earlier integration publication still match their recorded hashes. See the [integration contract](frontier_live_integration_20260907.md) for target, fallback, input availability and interval limitations.

The new checkpoint model remains a research candidate. Its historical checkpoint MAE improves by **6.8794%**, and target-balanced MAE by **3.4194%**, below the predeclared **10% / 5%** magnitude criteria. Its 2026 interval includes deterioration. It also predicts at additional, later checkpoints; the original 55,757 eligible forecasts are unchanged. Enabling it would change the issuance contract without a passing promotion result. Football's shot correction improves the strong blend by only **0.2653%**, with an interval crossing zero. These are measured partial findings, not production approvals. The [frozen results snapshot](boundary_research_20260908.md) preserves all completed earlier outcomes and uncertainty.

Since that snapshot, the weather experiment completed: four variants were fitted using 2022 and selected on 22 events / 20,007 original issuances in 2023. All four lose to the strong HGB. The least-bad variant's event MAE is **0.5123239426 versus 0.5120807233 seconds**, a **0.047496% increase**. It worsens all three events where its rain gate changes predictions. Both alternate latency assumptions also lose. The advancement screen failed, so no later-season weather candidate evaluation ran. [Weather report](../../research/experiments/boundary_20260908/weather/model/README.md).

The completed research now comprises **54 predefined candidate configurations across ten fitted families**, plus one deterministic composition that exactly falls back to checkpoint cycle1. No new substantial-gain criterion passed. The existing HGB's previously reported gains are not counted as a new discovery.

The user's integration request paused the remaining lanes before fitting. Global track-control acquisition finished for 44 discovery races, with 569 records and 132 verified raw/CSV/receipt hashes; no control model was designed or fitted. Distributional forecasting has a frozen protocol and an untested implementation draft, with no fitted model or score. Both directories clearly state their incomplete status. They are not evidence of improvement and are not imported by production.

Fresh verification for this publication:

- **55 passed**: packaged HGB, canonical prediction path, platform transport/fallback and prediction-service tests.
- **83 passed**: boundary experiment mathematics, causal input contracts, selection/fallback behavior and parsers. The untested distributional draft is excluded from this coverage.
- All **66** prior integration publication file hashes match, and all **163** recursively recorded path/hash bindings in the earlier boundary journal match their current local bytes.
- All research Python source files compile. Dependency deprecation warnings in time parsing do not indicate failed tests.

The [publication manifest](evidence/boundary_publication_20260908.json) records exact included-file hashes, omitted local artifact hashes, retained model identity and verification commands. Source, locked specifications, structured results, source manifests, verification summaries and failed execution attempts are included. Raw provider feeds, feature caches, fitted pickle objects and large row-level tables remain local. The earlier snapshot and journal are retained byte-for-byte; this document supplies the later weather and paused-lane status.

This is a repository publication, not a hosted-service deployment or prospective validation. All research performance dates were previously exposed; historical data receipt times remain subject to the limitations in each experiment report.

Suggested commit: `docs: publish boundary research and verify global integration`.
