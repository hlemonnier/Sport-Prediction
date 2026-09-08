# Past-market football strengths: failed screen, useful Elo lead

The fixed strength-model family failed its selection gates. Its best candidate, the unchanged 50/50 market-strength/shot mixture, scored **0.960861843 NLL**, **0.2822% worse** than the retained shot research baseline. Only **5/16** required gain/interval checks passed. No seven-day sensitivity, new transfer forecasts, refitting, retuning or production replacement followed.

A separate post hoc diagnostic identified the fixed Elo-Odds reference as a credible lead: **1.0257% lower NLL** than the shot baseline, with a paired 28-day difference interval **[-0.017987870, -0.001487357]**. It was a comparator, not one of this experiment's two selectable candidates. This is not advancement of the failed family and does not yet meet the substantial multi-country-gain objective.

## Fixed experiment

The [predeclared protocol](../../research/experiments/boundary_20260908/football_past_market/README.md) and [specification](../../research/experiments/boundary_20260908/football_past_market/specification.json) fix all settings. The source lock preceded **475 optimizer fits**: two strength estimators at each of **237 forecast clocks**, plus one Elo probability readout. Both new strength models used the same admitted match IDs, weights, centered parameterization, prior and fitting cadence. Their targets differ: power-adjusted previous average odds versus actual results. A matched result/shot blend controls for gains due merely to combining forecasts.

All **760 EPL fixtures from 2022/23 and 2023/24** remain in the scored population. Prior history comes from all three leagues' cached files; this is not a country holdout. The rolling window is 1,095 local calendar days, recency half-life 365 elapsed days, and fixed penalization 0.002 in the equally weighted three-league objective. The primary odds-availability proxy is local midnight D+2, strictly before issuance. The unused sensitivity would use D+8. Original quote receipt/revision times remain unavailable; neither proxy certifies them.

Both teams had prior quote support in **758/760** fixtures. The other two retain the exact shot baseline for every new candidate and control. The candidate family has two forecasts only: the market model and its fixed half-shot mixture. The Elo-Odds comparison uses K=175, home advantage 80 and scale 400, neutral team entry, atomic updates, and a single ordered-logit readout fitted on **3,419 quote-valid 2019–2021 fixtures**. These are declared adaptations of the [published Elo-Odds method](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0198668).

## Primary results

NLL uses natural logarithms; Brier is summed over the three outcome classes. Positive relative reduction means less NLL than the retained shot research baseline. The production-default comparator uses the historical benchmark's controlled 1,095-day history and two-month refit cadence; its score is not an observed deployed artifact's score. The retained research baseline is stronger, so a gain against the production-default comparator alone does not satisfy this experiment's gates.

| Forecast | NLL | Brier | NLL reduction vs shot |
|---|---:|---:|---:|
| Controlled production-default benchmark | 1.002042055 | 0.598610665 | -4.5800% |
| DC180 | 0.972201404 | 0.575833636 | -1.4657% |
| DC365/Elo50 | 0.959598905 | 0.568296614 | -0.1504% |
| Retained shot research baseline | 0.958158118 | 0.566955186 | +0.0000% |
| Existing xG selection safeguard | 0.957806122 | 0.566719223 | +0.0367% |
| New market-strength model | 0.975508940 | 0.578299065 | -1.8109% |
| Selected market/shot blend | 0.960861843 | 0.568548860 | -0.2822% |
| Matched result-strength control | 0.975405928 | 0.576923232 | -1.8001% |
| Matched result/shot blend | 0.961410279 | 0.568677927 | -0.3394% |
| Fixed Elo-Odds reference | 0.948330244 | 0.560474927 | +1.0257% |

The selected blend's NLL difference against shot is **+0.002703725**, with 28-day interval **[-0.006593516, +0.011966103]**. Its difference against the matched result/shot blend is **-0.000548436**, with interval **[-0.006328272, +0.005758451]**. Thus this fixed model does not establish a benefit from its past-market targets beyond the matched control.

The gate required at least 0.5% NLL reduction and a strictly negative 28-day interval upper endpoint against **all eight** references. Four thousand paired resamples use fixed 1-, 7- and 28-day blocks within league-season strata; the shorter blocks are diagnostic. These dates were already exposed in earlier research. The intervals do not correct for the full research search or establish prospective validity.

## Independent verification and next lead

The separate verifier imports neither the fitting model, operational data loader nor runner. It reconstructed CSV metadata and admitted training populations, recency weights and Elo history; checked all **474 strength optima** and the readout's objective/gradient constraints; replayed **3,800 new candidate/control vectors**; checked unchanged references; and recomputed scores, intervals and the rejection decision. Maximum probability replay error was **7.77e-16**, with **zero refits**. All historical fitting and verification completed on their first execution attempts; an earlier pre-fit launch simply encountered a review file that had not yet been written and created no execution lock or fit output.

The complete selection stage took **80.95 seconds**, at **430.0 MB** peak process RSS. The experiment's **250 synthetic tests** passed. Source-only CI passed **1,596 tests with two optional historical skips**, preserving 571 source files and 91 dependency-file hashes. This CI did not use historical assets or perform historical fitting.

The [Elo diagnostic](evidence/boundary_football_past_market_20260908/elo_post_hoc_diagnostic.json) uses only the already closed forecasts. Elo improves pooled NLL and Brier against all five legacy references; pooled 28-day intervals exclude zero. Both EPL seasons improve NLL, but their individual strong-reference intervals cross zero, and accuracy/calibration do not improve uniformly. The next justified step is the [separately specified follow-up](../../research/experiments/boundary_20260908/football_elo_followup/README.md) of the unchanged Elo method, with delayed-availability and three-country transfer tests. It is not yet globally integrated.

The [coefficient diagnostic](../../research/experiments/boundary_20260908/football_past_market_diagnostics.md) identifies substantial probability compression and old training weight, without claiming an ablation-proven remedy. A separate [runtime audit](../../research/experiments/boundary_20260908/football_runtime_gap.md) finds an operational opportunity: existing full-history DC forecasts reduce NLL by 3.432% against the controlled production-default benchmark across the previously closed 2,280-fixture population. That is a post hoc assessment of existing forecasts, not a new candidate transfer result or a passed deployment assessment. The football runtime remains unchanged, and the validated global F1 HGB remains intact.

## Evidence

The [copy manifest](evidence/boundary_football_past_market_20260908/manifest.json) records **11 byte-identical receipts** totaling **359,862 bytes**. Row ledgers, coefficient files, raw data and readout history remain local and hash-bound. Executable sources and tests are committed alongside the protocol.

| Receipt | SHA256 |
|---|---|
| Design lock | `e9a05b1988bb619a59dc815e00fe46b8c6d1f6f9daf30dfb4fe563210f9f6d41` |
| Selection result | `0591f1b13c0e0e12497e18b45d2c56114b0b1cf5e02b2ee018fc3e6cde30555e` |
| Independent verification | `776eed85649217734bc243dc539cb4ccb955187c0c9eac1c9e6b5131ec172eaf` |
| Copy manifest | `871e9cf717761810d8d025a7c00257882960f6494f376840e1ad33a26371ac48` |

Suggested commit: `research(football): record failed strengths and the verified Elo-Odds lead`.
