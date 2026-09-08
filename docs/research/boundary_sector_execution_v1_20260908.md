# Preserved first sector execution

The first complete sector experiment implementation passed **215 pre-fit tests**, then failed during target-free historical assembly. It produced no complete data lock, canonical labels, offline HGB fit or selection scores. The failure is an input-availability contract error, not evidence for or against predictive performance. The original sources and failed attempt are preserved byte-for-byte.

The implementation adds the 75-column causal feature encoder, common issuance population, separate canonical target attachment, event-balanced residual fitting, two error metrics, paired uncertainty estimates and fixed selection/transfer criteria. Its three candidates are a regularized Gaussian conditional remainder and two residual HGB configurations. All candidates must beat four references issued at the same S1/S2 checkpoints. These forecasts use later information than the global model's completed-lap issuance.

The 44-race build saved 12 event ledgers, containing 25,360 retained updates, before reporting an exception from Hungary 2022. For driver 47, packet 24,356 contained S1 = 30.526 seconds. Packet 24,357 corrected S1 to 30.486 seconds and supplied S2 = 31.233 seconds. The equality check against the prior S1 rejected this legitimate correction. The current packet was available at archived session clock 5,977,490 milliseconds; its revised value must not be backdated to the prior packet.

The separate v2 amendment retains a prior positive S1 in the same epoch as an attribution prerequisite and uses the latest atomic-packet S1 value with its actual source clock. It preserves the earlier S1 forecast and excludes the current S3 display. This correction was chosen before canonical targets or predictive scores were read. It changes no candidate, numerical feature definition or validation threshold. The original experiment is not rewritten or retried in its existing output directory.

The [publication manifest](evidence/boundary_sector_execution_v1_20260908/publication_manifest.json) binds the frozen source files and 17 byte-identical compact evidence copies: design lock, pre-fit test receipt, failure traceback, two diagnosis records and 12 closed event metadata files. The corresponding 99,105,873 bytes of row-level ledgers and provider inputs remain local at their recorded paths. The baseline and Gaussian points generated during assembly are unscored; claiming that no point forecasts were generated would be incorrect.

The original [mathematical review](../../research/experiments/boundary_20260908/sector_forecast/execution/mathematical_review.md) and [specification](../../research/experiments/boundary_20260908/sector_forecast/execution/specification.json) describe the frozen design. Their pre-execution status text is a historical snapshot superseded by this failure record. Source-only mathematical tests run through the existing boundary research CI workflow. The independent selection verifier is also preserved, but historical selection verification cannot run until a corrected execution completes.

The already integrated `frontier_live_hgb_l15_i150_20260907` remains the production model. This publication adds research code and evidence, with no sector-model activation or hosted deployment.

Suggested commit: `research(f1-live): preserve frozen sector execution and failed assembly evidence`.
