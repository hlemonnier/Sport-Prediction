**The frozen lap-time frontier is integrated into the global F1 prediction system.**

The runtime uses the exact model that reduced retrospective next-eligible-lap event MAE by 10.07% in 2024–2025 and 10.58% in 2026 R1–R13. There was no refit, new selection or modification of the frozen experimental features. [Research evidence](frontier_live_performance_20260907.md).

`run_prediction` → `run_live_race_prediction` now selects `frontier_hgb` by default. It writes the point estimate to the compatible `next_lap_mean` field and identifies its model, target, issuance lap/time, availability and interval status in each row. Despite the legacy field name, this absolute-loss estimate targets a conditional median. The existing SSM continues to estimate its own latent state and order distribution.

The platform preserves original completed-lap observations from accepted FastF1 events, including missing measurements, sectors, speeds, original stints and tyre state. It converts durations explicitly to numeric session-clock seconds. Completed lap events use their endpoint, and untimed transition rows with a known endpoint remain in the history. Snapshot generation indexes lap observations directly, avoiding a scan of every telemetry event.

Both the model service and the platform's local fallback use the same canonical predictor. Their race and next-lap responses expose a separate `lap_time_forecast` object containing seconds, model/hash, original issuance and target. The dashboard displays “Next clean” with a baseline or unavailable label where appropriate. The existing order probabilities retain their own heuristic provenance; these are not calibrated by the point model's accuracy gain. A race session identity is required before activating the frontier through the API.

The target is the **same driver's next eligible clean completed lap, potentially skipping numbered laps**. The model issues after three eligible observations for that driver. It consumes only the observed prefix; strictly earlier peer state is frozen before each equal-timestamp batch. A pending forecast remains associated with its original issuance across ineligible laps. Retired or otherwise known non-running drivers receive no API lap forecast.

Absent input columns, invalid chronology, unknown/future timestamps, missing model files or a model checksum mismatch trigger an explicit fallback. Individual missing measurements present in an otherwise complete source retain the original missing-value encoding. The canonical fallback is the last clean lap, with the existing SSM cold start when no clean lap is available. Snapshot-only API inputs use a clearly labeled last-observed-lap baseline whose clean eligibility is unverified. Current OpenF1-only snapshots do not supply the full required input contract and therefore use that fallback.

The frontier has **no calibrated interval**. Its `next_lap_std` and interval bounds are unavailable; the SSM standard deviation remains separately available as `next_lap_std_ssm`. Likelihood diagnostics are unavailable when the prediction has no variance. Strategy value and prospective performance remain unproven, and maturity metadata records those limits alongside runtime activation.

The portable model contains 150 numeric trees and occupies 217,803 bytes. Production inference requires NumPy and pandas, reads no research paths and never unpickles or retrains. Both service container definitions include the canonical package and model asset.

| Verification | Result |
|---|---|
| Portable model against frozen transfer predictions | 55,757 forecasts, 61 races, maximum difference **0 seconds** |
| Production source conversion and feature encoder | All 61 races match every frozen matched feature exactly |
| Broader F1 and service regression suite | **1,054 passed, 3 skipped** before the final importer refinements |
| Final affected integration and service suite | **181 passed** after importer refinements |
| Original research contract tests | **10 passed** |
| Real Italian GP prefix at session clock 7,800 seconds | 541 observations; both public paths return 22 drivers, with 21 exact frontier forecasts and one history fallback |
| Real HTTP request | 200; approximately 0.63 seconds locally, not a load benchmark |
| Web production build and backend syntax check | Passed |

The [numeric verification](../../artifacts/research/frontier_20260907/integration/runtime_verification.json) and [global/HTTP verification](../../artifacts/research/frontier_20260907/integration/global_prediction_verification.json) record the exact evidence. The repository's CI now uses the current `apps/web` and `apps/api` paths, installs the missing `requests` test dependency and runs the new model/API integration tests.

For rollback, set `PredictionConfig.f1_live_next_lap_point_model="baseline"` in the canonical runner. Set `F1_LIVE_NEXT_LAP_POINT_MODEL=baseline` in the services. The default value is `frontier_hgb`; invalid values are rejected.

Reproduce the portable verification with `PYTHONPATH="$PWD" .venv-f1/bin/python research/experiments/frontier_20260907/verify_runtime.py`. The full cached-source verification needs the local race files and transfer table identified by the committed manifests. The public synthetic contract fixture and bundled numeric model run in CI without those caches. For the real HTTP replay verifier, additionally put `services/f1-platform` and `services/f1-prediction-service` on `PYTHONPATH` and run `verify_global.py` from the same experiment directory.

Research results, selection locks, the original failed transfer, corrections, fitted candidate, source manifests and summaries are committed. Large raw race files, fitting caches and row-level transfer tables remain local. The research report and its earlier verification describe the experiment-time state; this integration is the subsequent user-authorized change. No hosted service deployment or future race observation is claimed by this repository release.

Suggested commit: `feat(f1): integrate frozen lap-time model across prediction runtimes`.
