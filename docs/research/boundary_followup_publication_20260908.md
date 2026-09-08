# Global prediction integration and subsequent research publication

**The validated HGB point model is already active throughout the global F1 prediction system.** This publication verifies that implementation and records the subsequent research. No further production wiring or model substitution is justified by the new results.

The default `PredictionConfig.f1_live_next_lap_point_model="frontier_hgb"` flows through global dispatch into the canonical predictor. The prediction service and platform fallback use the same implementation; the dashboard displays its separate “Next clean” estimate. Model `frontier_live_hgb_l15_i150_20260907` retains SHA256 `d8bd93f1b04ecf290eb821f02397253b77248cf3c1943b05e1c540d4942e7e36`. All 66 earlier integration publication files still match their recorded hashes. [Integration contract](frontier_live_integration_20260907.md).

That model predicts the next eligible clean completed lap for the same driver, which can skip numbered laps. Missing input contracts produce explicit fallback behavior. Its point forecast has no calibrated interval. The earlier verified retrospective point-MAE gains remain 10.07% in 2024–2025 and 10.58% in 2026 rounds 1–13; they are not new discoveries from this follow-up.

The [previous publication](boundary_publication_20260908.md) is an immutable earlier snapshot. The control and distributional lanes subsequently ran in separate execution directories; their earlier paused statuses remain historical records.

| Subsequent experiment | Verified result | Production decision |
| --- | --- | --- |
| Global track-control correction, four settings | Best 2023 checkpoint-MAE gain over C1: 0.1761%; interval crosses zero, advancement gate failed | No transfer evaluation or promotion |
| Conditional quantile HGB, three settings | 2024–2025 WIS improves 5.1645% against conditional empirical quantiles, paired three-event delta interval [-0.0227523, -0.0071836]; 2026 gain 2.6566% | Frozen 10% magnitude and comparative-coverage gates failed; research only |
| Online quantile calibration, four candidate policies and four matched reference policies | Best 2023 WIS gain over static quantiles: 0.78875%, below the frozen 1% advancement requirement | No transfer calibration scores or promotion |
| Practice-informed race ranking, two candidates plus two matched ablations | Selected pairwise model MAE 3.330 positions versus prior Huber 3.235 and matched ablation 3.265 | Selection failed; no later candidate evaluation |
| Public football xG history | 10,260 matches across 27 league-seasons; 10,249 exact joins, 11 date discrepancies excluded | Data acquisition only; no predictive gain established |
| Raw F1 sector feasibility | Raw feed timestamps support a parser pilot; processed sector timestamps can be backdated | Feasibility only; no model fitted |

The distributional result concerns a finite-grid weighted interval score, not lower point error or an exact continuous CRPS. All 55,757 original HGB transfer points remain identical. The comparative-coverage rule fails at the 2025 80% interval: coverage drops 3.138587 percentage points against the conditional reference, exceeding the predeclared allowance of three points. The candidate is closer to nominal coverage than that reference, but the criterion was not changed after scoring. [Full distributional results](../../research/experiments/boundary_20260908/distributional/execution/README.md) and [online calibration](../../research/experiments/boundary_20260908/distributional/online_calibration/README.md).

The race-practice verifier preserves an initial bookkeeping failure: the original runner hashed an open stdout log. The amended verifier proves the exact original 738-byte prefix of the closed log and independently reconstructs all features, 80 fits and 2,800 predictions. No source, model, selection result or gate was altered after scoring. Both the failed attempt and successful amended verification are included. [Race-practice report](../../research/experiments/boundary_20260908/race_practice/README.md).

The xG acquisition verifies 20,520 provider match/team-history value mirrors and records explicit 7/14-day availability assumptions; original publication and revision times remain unknown. Its separate model directory is an **unfinished, unfitted draft**, with no runner or tests and a documented daylight-saving timing discrepancy to resolve before fitting. It is preserved as work in progress, not completed implementation. [Data contract](../../artifacts/research/boundary_20260908/football_xg_data/availability_contract.json) and [draft status](../../research/experiments/boundary_20260908/football_xg/README.md).

Fresh publication checks passed: **55 integration/service tests** and **149 boundary research tests**, with 30 pre-existing dependency deprecation warnings. The unfitted xG model draft has no tests and is excluded from any model-validation claim. An independent source audit found no missing global HGB wiring. Existing publication file hashes and the new artifact bindings are recorded in the [follow-up manifest](evidence/boundary_followup_publication_20260908.json).

Source, specifications, structured outcomes, frozen locks, diagnostic summaries and execution records are published. Raw provider data, large fitted objects and row-level pickle/CSV caches remain local with recorded hashes. Full replay requires those local inputs; this is not a self-contained public-data reconstruction. No new substantial-gain criterion passed, hosted deployment occurred or prospective performance was observed in this publication.

Suggested commit: `docs: verify global prediction integration and publish follow-up evidence`.
