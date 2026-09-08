# Football market snapshot pooling — 8 September 2026

The frozen experiment **failed its selection gate**. The better of the two candidates, `pool_shared`, scored **0.939232650133** mean natural-log loss on the fixed 760 EPL matches from 2022/23–2023/24. The strongest reference, Bet365 power margin removal, scored **0.936265621783**. The candidate's loss was **0.316900% higher**, and its Brier score was also worse. **No later transfer evaluation or production promotion was performed.** Exact metrics, every candidate/reference comparison and the failed decision are preserved in [selection.json](evidence/boundary_football_information_20260908/selection.json).

This is a retrospective provider-snapshot comparison. Historical quote receipt times are unknown. Its numerical improvement against a carried midnight forecast does **not** establish an improvement at that midnight forecast horizon.

## Fixed comparison and mathematics

The [specification](../../research/experiments/boundary_20260908/football_information/specification.json) fixed two candidates before new historical fits: a shared-coefficient log pool and a class-specific log pool. Both combine the unchanged selected shot-strength forecast with the power-adjusted provider-average market vector. For home/draw/away class \(j\),

\[
q_j\propto\exp\{a_j\log m_j+c_j\log p_j+b_j\},\qquad
0\le a_j,c_j\le4,\quad\sum_j b_j=0.
\]

The shared candidate uses scalar \(a,c\); the class-specific candidate uses three of each. The objective is mean multiclass negative log likelihood plus \(0.01(\|a-1\|^2+\|c\|^2+\|b\|^2)/2\). Orthonormal bias coordinates give four/eight free candidate coordinates. Positive regularization makes the penalized objective strongly convex; data identifiability itself is not assumed. Matched market-only calibrators set \(c=0\) and use the same historical training rows, refits and penalty.

The eleven mandatory references include four internal forecasts, normalized and power margin removal for both provider-average and Bet365 prices, both fitted market-only calibrators, and a fixed half-market/half-internal arithmetic blend. For decimal odds \(o_j\), normalization uses \((1/o_j)/\sum_k(1/o_k)\); power adjustment solves \(\sum_j(1/o_j)^k=1\) for \(k>0\), supporting overrounds and underrounds. Closing-price columns and Pinnacle-only inputs are excluded. All thirteen forecasts were scored on the same 760 matches, with no missing-market fallback needed in that population.

## Complete selection results

Lower is better for both metrics. Brier is the mean **sum over the three classes**, without division by three. Values below are rounded; the linked JSON retains full precision.

| Forecast | Role | NLL | Brier |
|---|---|---:|---:|
| `b365_power` | Bet365 power reference | 0.936265622 | 0.552510257 |
| `avg_power` | Provider-average power reference | 0.936544961 | 0.552839183 |
| `avg_normalized` | Provider-average normalized reference | 0.937525705 | 0.553396028 |
| `b365_normalized` | Bet365 normalized reference | 0.937685373 | 0.553310950 |
| `pool_shared` | Selected candidate | 0.939232650 | 0.554593564 |
| `market_shared` | Matched shared market-only reference | 0.939232651 | 0.554593564 |
| `market_classwise` | Matched class-specific market-only reference | 0.939471511 | 0.554551093 |
| `pool_classwise` | Other candidate | 0.939535889 | 0.554602962 |
| `arithmetic_half` | Fixed equal market/internal blend | 0.944635998 | 0.558190357 |
| `shot_strength_90d_ridge0.1` | Prior selected shot incumbent | 0.958158118 | 0.566955186 |
| `dc365_elo50` | Prior Dixon–Coles/Elo blend | 0.959598905 | 0.568296614 |
| `dc_180` | Recency-weighted Dixon–Coles reference | 0.972201404 | 0.575833636 |
| `production_default_dc_auto` | Frozen controlled production reference | 1.002042055 | 0.598610665 |

Selection required at least **0.5% relative NLL improvement against every reference**, with a strictly negative upper endpoint of the paired 95% 28-day interval against each. It failed. For `pool_shared` minus Bet365 power, the NLL difference was **+0.002967028350**, with interval **[+0.000659624142, +0.005365837424]**. Against provider-average power it was **+0.002687688671**, with interval **[+0.000291667248, +0.005115984136]**. These are historical, season-stratified fixed-block percentile intervals from 4,000 paired resamples; they do not account for all earlier research selection or establish future transfer. The evidence also retains 1-day and 7-day block diagnostics.

Bet365 power beat the selected candidate in both seasons: **0.966417918 versus 0.972122675** in 2022/23, and **0.906113325 versus 0.906342625** in 2023/24. The candidate's pooled advantage over the shot incumbent was 1.975193%, and over the controlled production reference 6.268141%, but those comparisons cannot substitute for the failed strong-market gate or resolve the difference in information timing.

The shared internal coefficient was **exactly zero in all 12 refits**. Its aggregate NLL differs from the shared market-only fit by only approximately \(5.55\times10^{-10}\), consistent with numerical optimizer differences. The nominal interval for that tiny difference must not be interpreted as useful predictive gain. This family found no incremental contribution from the internal vector under its declared constraints and regularization; it does not prove that every possible internal feature is redundant.

## Execution, chronology and limits

The reviewed [implementation and execution contract](../../research/experiments/boundary_20260908/football_information/EXECUTION.md) was frozen after **42 passing pre-fit tests**, including analytic-gradient, constrained-optimization, margin-removal, temporal-filtering, exact-vector-reuse and forecast-before-label tests. The execution then reconstructed **3,800 component forecasts** using the unchanged prior shot configuration, reused the **760 existing selection vectors verbatim**, and closed a target-free table of **4,560 rows**. Reconstruction was necessary to provide historical inputs to the pool; it was not a new shot-model search.

Selection performed **48 new fits**: four fitted families at twelve chronological cutoffs, sequentially with one numerical thread. All reported successful convergence; the largest projected-gradient residual was approximately \(2.96\times10^{-7}\), below the fixed \(10^{-6}\) acceptance threshold. Each cutoff used the same past rows for both candidates and both fitted market-only references. Three historical fixtures lacking a required complete price triple were excluded consistently from these training populations and recorded; no selection fixture was removed.

New pool training required both forecast time and result availability **strictly before** the refit cutoff, with a 1,460-day history limit. Previously resolved selection outcomes could enter later refits. The inherited shot reconstruction retained its original `result_available_at <= cutoff` policy so that the incumbent definition stayed unchanged. All 760 final issued forecast vectors and fit parameters were saved and hash-closed before the final current-row label attachment and scoring pass.

These safeguards constrain outcome use within the retrospective replay. They cannot certify that a historical quote, internal forecast or fitted coefficient was available at the quote's original receipt time: the CSVs contain no such receipt timestamps. `quote_observed_at` remains null and historical receipt certification remains false. The seasons were already exposed during earlier research; they are not a pristine or prospective test. The provider-average and Bet365 columns are the recorded non-closing price sets, not certified opening quotes or a fixed number of minutes before kickoff.

The failed gate stopped the experiment before the proposed 2,280-match, three-league later evaluation. No transfer predictions were newly scored, no parameter settings were changed after selection, and this work changed no production default.

Independent verification passed on its first execution. It checked 172 hash bindings, independently paired 5,700 canonical CSV labels, and reproduced all 3,800 reconstructed component vectors exactly. All 48 serialized pool models replayed 3,040 selected probability vectors within `1.11e-16`; the four frozen incumbent vectors per selection match remained exact. It independently reproduced 39 metric groups, all 66 paired bootstrap comparisons and all eleven reference gates. The [verification receipt](evidence/boundary_football_information_20260908/verification.json) confirms both the failed advancement decision and numerical equivalence to the matched shared market-only fit. The complete expanded local research CI command passed **624 tests**, including the 42 pre-fit tests; those counts overlap.

## Evidence identity and reproduction

The completed selection SHA256 is `edbb800ae4ce5b3e694ce7bcdb2f45274d32663d336ee9e2966ce913440569b2`. Its design-lock SHA256 is `d90a2087dfe5c99c0aa63357ce63fa798108c43d0c7f199757d09219107d0af2`; its data-lock SHA256 is `d5167a2aae09b5e31c9b6bc26e0a0ab446f3bcb91dfbce0424fb5e9fc24e5536`. The reviewed source inventory binds **28 files**, with fingerprint `3d9648117a96975c6f6d6f19fca582c8ab86e3aa3875277e434e199a1c913cd4`, and the design lock binds **40 input files**. The original proposal remains byte-identical.

The execution notes provide exact test and `freeze`/`prepare`/`select` commands. The [publication manifest](evidence/boundary_football_information_20260908/publication_manifest.json) binds the compact evidence and exact source files. Reproduction requires the hash-matched local provider inputs and predecessor feature/evidence files; the compact publication does not replace those inputs. An existing output cannot be overwritten: reruns require a separate output directory and the reviewed source/input bindings.

Suggested commit: `research(football): record market pooling failure against strong odds references`.
