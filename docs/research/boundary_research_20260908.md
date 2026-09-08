# Boundary research — 8 September 2026

**No newly completed experiment has passed its predeclared substantial-gain criteria.** The strongest verified new live partial result is a **6.879%** historical checkpoint-MAE reduction, or **3.419%** when repeated forecasts of the same target are balanced. It uses additional information at later checkpoints; the original eligible forecasts remain exactly unchanged. The objective of finding substantial predictive improvement remains open. Integrating earlier improvements does not satisfy that objective by itself.

This snapshot covers nine fitted families with **50 predefined candidate configurations**, plus one deterministic composition. Reference fallbacks are not counted as new configurations. The [machine-readable journal](evidence/boundary_research_20260908_journal.json) records exact scores, every tried setting, actual artifact SHA256 values, selection decisions and existing verifier results. Weather acquisition/experiments are **in progress**; no weather performance outcome is included. A separate distributional-forecast protocol has been requested but is not part of these completed results.

## Decisions against strong references

Positive loss differences below mean worse predictions. Qualifying uses rank-position MAE; live forecasts use seconds; football uses natural-log multiclass loss. These quantities are not pooled into a cross-sport score.

| Experiment | Relevant result against the strong reference | Decision |
| --- | --- | --- |
| Season-reset qualifying correction, 12 settings | Historical MAE **3.004605 vs Q2 2.908772**; exposed 2026 **2.050505 vs Q1 1.848485** | Both material requirements fail |
| Comparable-lap qualifying measurement, 4 settings | Historical **2.906689 vs Q2 2.908772**; exposed 2026 **1.959596 vs Q1 1.848485** | Tiny historical change; still loses to contemporary Q1 |
| Online live residual assimilation, 8 settings | Best 2023 MAE **0.511176 vs 0.512081 s**, **0.1767%** lower | Below 0.5% advancement screen; no later transfer |
| Periodic live model refitting, 5 settings | Best 2023 **0.510054 vs 0.512081 s**, **0.3958%** lower | Below 0.5% advancement screen; no later transfer |
| Unclipped robust live anchors, 4 settings | Least harmful 2023 **0.518406 vs 0.512081 s**, **1.2352% worse** | All four fail; no later candidate fit or score |
| Later-checkpoint expert, 4 settings | Historical **0.571664 vs carried HGB 0.613896 s**, **6.8794%** lower | Verified partial gain; misses 10% checkpoint and 5% target-balanced thresholds |
| Newly completed peer checkpoint correction, 4 settings | Least harmful 2023 **0.716690 vs checkpoint cycle1 0.713574 s** | All four worsen cycle1; no new transfer |
| Deterministic composition | Same point forecasts as checkpoint cycle1; maximum difference **0 s** | Identity fallback, no additional gain |
| Football shot-strength correction, 6 settings | **0.984127 vs prior blend 0.986745**, **0.2653%** lower | Small gain; strong-reference interval crosses zero and 2% gate fails |
| Football nonlinear residual trees, 3 settings | Selection losses **0.961207 / 0.963488 / 0.965111**, versus incumbent **0.958158** | Incumbent retained; losing trees not transfer-scored |

The original-issuance live selection experiments use the strong current HGB architecture fitted on 2022, scored on the same **20,007 rows from 22 events in 2023**. Their small selection improvements are not independent transfer results. Qualifying Q1/Q2 are the verified preceding research forecasts; their comparisons are not newly measured deployed-model gains. Football's production comparator uses the actual default DC-plus-auto-calibration policy under a controlled common history and cadence, while the prior blend and DC180 provide the stronger research tests.

## What the checkpoint gain does and does not mean

The selected expert is **15 leaves with a ±3-second correction cap**, fixed by 2023 selection. It issues after later observed lap records once the driver has an HGB point, targeting the strictly later next eligible lap. The same earlier HGB point is carried to each new checkpoint for comparison. Thus the expert has later own-car information at ineligible observations. This is a changed forecast-time and information contract, not an improvement to the old forecast made at its original issuance.

Across 61 races in 2024–2026, the new population contains **61,948 record checkpoints**. The **55,757 original eligible predictions and targets** exactly match the prior frozen frontier artifact; maximum point difference is **0 s**. Some checkpoints share a target. The target-balanced statistic averages repeated checkpoints within each driver-target lap before averaging targets within events and events equally. It prevents extra checkpoints from silently dominating the result.

| Period | Events / checkpoints | Carried HGB MAE | Expert MAE | Checkpoint reduction | Target-balanced reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2024 | 24 / 24,309 | 0.637568 s | 0.583174 s | 8.531% | 3.759% |
| 2025 | 24 / 24,002 | 0.590224 s | 0.560153 s | 5.095% | 3.041% |
| 2024–2025 | 48 / 48,311 | 0.613896 s | 0.571664 s | 6.879% | 3.419% |
| Exposed 2026 R1–R13 | 13 / 13,637 | 0.643080 s | 0.612973 s | 4.682% | 1.202% |
| Exposed 2026 R10–R13, nested subset | 4 / 4,172 | 0.499643 s | 0.484528 s | 3.025% | 0.741% |

Historical checkpoint delta is **−0.042232 s**, paired-event 95% interval **[−0.059958, −0.022652] s**, and within-year three-event block interval **[−0.061368, −0.021851] s**. Forty-three of 48 events improve, and all historical leave-one-event-out means remain improving. Historical target-balanced MAE is **0.459089 → 0.443391 s**, with block interval **[−0.020633, −0.011041] s**. Both magnitude thresholds still fail. The 2025 checkpoint interval alone crosses zero.

Exposed 2026 checkpoint delta is **−0.030107 s**, with event interval **[−0.096238, +0.021360] s** and block interval **[−0.097840, +0.028203] s**; seven of 13 events improve. This is not stable contemporary confirmation. Pit transitions account for much of the gain, while neutralized observations worsen. Large gains in that selected subgroup do not replace the full-population criteria.

All four frozen checkpoint variants were transfer-scored and remain in the journal. The later 7.080% point result from the seven-leaf/±3-second alternative did **not** replace the 2023-selected model. The peer refinement subsequently lost on 2023, and the robust anchor also failed. The composition consequently uses unchanged HGB on eligible observations and checkpoint cycle1 elsewhere. Its 61,948 predictions are exactly cycle1 again; composition must not be counted as an independent replication or extra gain. [Checkpoint evidence](../../artifacts/research/boundary_20260908/checkpoint/results.json), [original-issuance parity](../../artifacts/research/boundary_20260908/checkpoint/original_eligible_parity.json), [composition evidence](../../artifacts/research/boundary_20260908/composition/results.json).

## Qualifying and football transfer

Qualifying selection uses 16 eligible 2023 events; six sprint weekends lacked the causal, target-aligned rehearsal and were excluded. Both cycles then cover complete official qualifying rosters for **48 events / 959 driver rows in 2024–2025**, and **9 exposed events / 198 rows in 2026**. MAE is averaged within each event, then equally across events.

The season-reset candidate deteriorates by **+0.095833 positions versus Q2** historically. Its event interval is **[+0.006250, +0.187500]** and three-event block interval **[−0.004167, +0.189583]**. The measurement candidate's historical delta is just **−0.002083**, with event interval **[−0.016667, +0.012500]** and block interval **[−0.016667, +0.010417]**. It changes only 52 of 959 historical driver ranks. In 2026 it improves Q2 by 0.050505 positions but remains **0.111111 worse than Q1**, the stronger preceding contemporary result. Its block interval versus Q1 is **[+0.030303, +0.212121]**. Neither meets the required 0.15-position historical gain plus no contemporary regression. [Cycle1 evidence](../../artifacts/research/boundary_20260908/pre_event/cycle1/results.json), [cycle2 evidence](../../artifacts/research/boundary_20260908/pre_event/cycle2/results.json).

Football selected the 90-day shot-strength affine correction with penalty 0.1 on **760 EPL 2022/23–2023/24 matches**. Earlier prequential histories from all three leagues train its correction; this is chronological league/year assessment, not an unseen-country holdout. Its evaluation covers **2,280 matches**, 760 each in England, Spain and Italy over 2024/25–2025/26.

| Football reference | Reference log loss | Candidate log loss | Relative reduction | Paired 28-day candidate-minus-reference interval |
| --- | ---: | ---: | ---: | --- |
| Controlled production default | 1.0283816775 | 0.9841272141 | 4.3033% | [−0.0558200, −0.0323753] |
| Strong prior DC365/Elo50 | 0.9867449115 | 0.9841272141 | 0.2653% | [−0.0052359, +0.0001834] |
| DC180 | 0.9924257346 | 0.9841272141 | 0.8362% | [−0.0135113, −0.0030657] |

Against the prior blend, league gains are 0.2382% EPL, 0.1634% Spain and 0.3936% Italy; every full-league interval crosses zero, and Spain 2025/26 slightly deteriorates. Brier improves **0.588528 → 0.586917**, while top-label ECE worsens **0.009155 → 0.022848**. A lower proper loss is not an unqualified calibration improvement. Excluding the resumed Fiorentina–Inter completion-day fixture leaves a blend delta of **−0.0026285**, interval **[−0.0052462, +0.0001745]**; the conclusion is unchanged.

The nonlinear tree follow-up improves its training objectives but loses to the existing affine correction on selection at all three complexities. Leaf optimization was verified; this is a generalization failure. Its evaluation artifact repeats the retained incumbent only, with exactly zero improvement over that incumbent. No tree transfer gain or loss was measured. [Football cycle1 evidence](../../artifacts/research/boundary_20260908/football/evidence.json), [cycle2 evidence](../../artifacts/research/boundary_20260908/football/cycle2/evidence.json).

## Verification, failures and remaining limits

The journal freshly hashes the actual canonical result, verification, specification and lock files. All ten artifact-binding checks passed. It incorporates existing detailed verification reports; this synthesis did not rerun models or those verifiers. Recorded tests establish implementation properties, not a predictive edge or production deployment:

- Qualifying verifiers reconstruct reference/target pairing, pre-target forecast bindings, rank permutations and metrics. Cycle1 includes some IRLS fits reaching its 40-iteration cap; it does not claim every coefficient stopping tolerance was reached. Cycle2's 120 fitted predictions reached their tolerance.
- Original-issuance live verifiers reconstruct selected predictions, chronology and future-outcome poisoning checks. The refit verifier exactly refits its selected six-block policy. The anchor verifier preserves all 80 original features and reconstructs 100,035 reference/candidate selection points.
- Checkpoint verification reconstructs all transfer rows for four models, checks 105 input hashes and real-stream prefix/poisoning invariance. Peer-cycle verification reconstructs 88,696 selection predictions and exact unsupported/eligible fallbacks.
- Football verification reproduces 7,600 prior comparator vectors exactly. Cycle1 reconstructs 504 shot models and 108 corrections/scalers. Cycle2 replays 36 ensembles, 2,100 trees and 10,133 leaf gradients, with maximum probability replay error **0.0**. Thirteen combined regressions pass.

Two execution recoveries remain disclosed. Qualifying initially stopped on five old broad-source hashes changed during earlier integration; retries bind unchanged prior prediction and target bytes plus the current implementation, without pretending the old broad source closure is still current. Checkpoint transfer initially stopped before scoring because narrow recent exports omitted required fields; the frozen input-only amendment uses the canonical 61-event full-schema manifest without refitting or changing selection. These are not discarded unfavorable performance runs.

All reported later dates were previously exposed. Paired event, block and football league-season intervals are descriptive historical uncertainty, not posterior probabilities, multiplicity-adjusted prospective evidence, calibrated point intervals, or proof of trading/strategy value. Point forecasts in the F1 families are not probability forecasts.

F1 eligibility/accuracy flags come from reconstructed historical records; original field receipt latency remains unknown. Masking future-valued pit fields and passing prefix tests does not establish original publication timestamps. Comparable practice laps do not identify hidden fuel. Football assumes final results and shot statistics are available by the next local midnight; provider files have no original row-level publication/revision times and are advertised as updated at least twice weekly. The known one-row shots-on-target/total-shots anomaly is retained unchanged. No odds enter these football models.

The next active direction is richer causally available information, including the ongoing observed-weather work, plus separately evaluated probabilistic forecasts. Neither has an outcome in this snapshot. Football's cached corners/cards/fouls are available aggregate proxies; timestamped shot-quality and squad information are absent, and pre-closing odds need their own compatible availability contract. The completed negative results narrow the tested hypotheses; they do not prove that forecasting cannot improve.

Suggested commit: `docs: record boundary research results against strong incumbents`.
