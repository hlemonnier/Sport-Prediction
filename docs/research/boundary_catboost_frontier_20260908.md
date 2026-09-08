# CatBoost frontier: selection failed

The fixed CatBoost comparison did not uncover a substantial gain. Plain boosting reduced 2023 event-balanced MAE by only **0.025615%**; ordered boosting increased error by **0.534506%**. Both candidates failed all four predeclared selection checks. The selected Plain candidate therefore stops at selection: no 2024–2026 transfer, refitting, retuning or global model replacement followed.

This tests a different learning algorithm on the original information horizon. It does not repeat the gap or telemetry feature experiments, and it does not establish that every possible CatBoost configuration must fail. The two declared configurations were retained exactly.

## Complete common population and objective

Each model fitted all **18,363 matched 2022 issuances across 22 races**, with the unchanged 80 observed inputs and weights `N/(E*n_event)`. Both received the same residual target around the original naive anchor, clipped at ±5 seconds, and their forecast correction was clipped at ±3 seconds. The baseline is the saved **2022-only HGB**, whose 2023 forecasts were reused exactly. It has the current global model configuration, but is not the deployed 2022–2023 parameter asset.

Both CatBoost variants use 1,000 symmetric depth-six trees, MAE with exact leaf estimation, learning rate 0.03, L2=10, no bootstrap sampling, identical border settings, training-only permutations and one CPU thread. There was no selection eval_set or early stopping. CatBoost symmetric trees do not implement the HGB minimum leaf count of 80; model capacity and regularization differ. The [CatBoost paper](https://arxiv.org/abs/1706.09516) motivates ordered boosting, while the [official parameters](https://catboost.ai/docs/en/references/training-parameters/common) define the implementation. No categorical variables were added.

All **20,432 original 2023 issuances** received forecasts before external selection labels were attached. Scores use the same **20,007 matched targets in all 22 races**; **425 unmatched forecasts** remain recorded without invented outcomes. Forecast times and target pairings were preserved. These seasons were already exposed by earlier research; the execution order does not create a pristine holdout.

## Fixed results

Lower MAE is better. Event MAE averages race MAEs equally; row MAE averages all matched forecasts. Units are seconds.

| Model | Event MAE | Row MAE | Relative error reduction vs HGB |
|---|---:|---:|---:|
| `base_hgb` | 0.512080723 | 0.532032646 | — |
| `plain` | 0.511949553 | 0.531984091 | +0.025615% |
| `ordered` | 0.514817826 | 0.535366435 | -0.534506% |

Candidate-minus-HGB deltas below are in milliseconds; negative is better. The 95% percentile intervals use 20,000 paired race resamples and 20,000 circular blocks of three consecutive races, seed 20260907.

| Model | Delta ms | Event CI ms | Block3 CI ms | Worst LOO delta ms | Races improved |
|---|---:|---|---|---:|---:|
| `plain` | -0.131171 | [-1.969721, +1.955452] | [-1.848276, +1.804065] | +0.233041 | 13/22 |
| `ordered` | +2.737103 | [-0.652404, +7.336933] | [-0.261303, +6.197695] | +3.368396 | 8/22 |

The minimum 1% improvement, negative event-CI upper bound, negative block-CI upper bound and negative delta under every single-race omission all fail for both variants. Plain wins the predeclared two-model selection but offers only a 0.131ms average reduction, with uncertainty spanning zero. A selected winner is not automatically an advancement or promotion.

## Execution and independent replay

The source lock bound **34 files and 192 inputs** before historical fitting, including the isolated CatBoost 1.2.10 runtime. Every consumed original ledger and label closure was checked against the earlier independent verification, rather than accepting freshly recomputed hashes alone. The global environment was unchanged.

| Fit | Seconds | Peak process RSS, MiB |
|---|---:|---:|
| Plain | 12.220 | 619.38 |
| Ordered | 55.693 | 730.66 |

Peak RSS is cumulative for the same process, not independent per-model allocation. Neither fit exceeded the declared 4GiB limit. The design closed at 08:06:31 UTC, the model exports at 08:08:01, both forecast vectors at 08:08:05, and selection at 08:08:06 on 8 September 2026.

The first independent verification passed without any post-fit source change. It imported no CatBoost and performed no fitting: it evaluated the exported numeric JSON trees directly, including float32 input rounding, strict split thresholds, leaf-bit indices, NaN handling and scale/bias. All **40,864 candidate predictions matched with zero observed error**, and all **20,432 baseline predictions** were reproduced from the unchanged HGB. It checked **241 bindings**, original target pairing, unmatched retention, training matrix/target/event-weight hashes, all metrics, the winner and every gate. It does not retrain the optimizer or repeat the already frozen original-feature extraction.

All **153 pre-fit tests passed**, including real small Plain/Ordered export comparisons. The updated source-only CI job passed **1,346 tests**, with two optional historical tests skipped. Its temporary copy contained no data or artifact directories; the external pinned dependency path supplied CatBoost code only. Source, workflow and dependency hashes remained unchanged.

## Published evidence and decision

The [compact manifest](evidence/boundary_catboost_frontier_20260908/publication_manifest.json) records **14 byte-identical receipts (271,049 bytes)**. Native and JSON model weights, row ledgers, labels, forecasts, wheels and runtime binaries remain local at their bound paths; full replay requires those omitted inputs. The [fixed specification](../../research/experiments/boundary_20260908/catboost_frontier/specification.json) and [runner documentation](../../research/experiments/boundary_20260908/catboost_frontier/README.md) preserve the experiment.

The validated global HGB remains active. This result closes the fixed CatBoost screen and leaves the substantial-gain goal unmet. The next distinct information hypothesis is [football team strength inferred from earlier market forecasts](../../research/experiments/boundary_20260908/architecture_choices_after_gap.md); it has not yet produced a candidate score.

| Artifact | SHA256 |
|---|---|
| `design_lock.json` | `5bca91308a982c84e9343ef99eb0f5340ce53236da3a274db6f93515ce2e222f` |
| `selection.json` | `d7c2a837956b2108537af0810313e949a5bcb52cb7a40a49ef96908acb364b57` |
| `verification/result.json` | `4981fb09a74920564e49974b4a3a15464a17c8c9283ba2f1470d805597d4d0a1` |

Suggested commit: `research(f1-live): record the failed fixed CatBoost selection screen`.
