# Fixed Elo-Odds: selection passed, transfer failed

The fixed Elo-Odds method passed both EPL selection stages, including the matched result-Elo comparison. It did **not** meet the substantial transfer-gain requirement: over 2,280 later fixtures, its natural-log loss improved by **0.4454%** against the retained shot baseline, below the fixed 2% requirement, with a paired 28-day difference interval **[-0.010071995, +0.001254636]**. England and Spain regressed in 2025/26. The second transfer delay was not run, and no model was promoted.

This closes the [predeclared follow-up](../../research/experiments/boundary_20260908/football_elo_followup/README.md), using its unchanged [specification](../../research/experiments/boundary_20260908/football_elo_followup/specification.json) and [executable procedure](../../research/experiments/boundary_20260908/football_elo_followup/execution.md). The previous past-market strength family remains failed. These dates were already exposed in earlier research; this is neither a fresh holdout nor prospective evidence.

## What was actually executed

Only one candidate was evaluated: the existing `elo_odds_reference`, with K=175, home advantage 80, base-10 scale 400 and neutral team entry. Its new matched `elo_result_k14` reference uses actual results with K=14. Both use exactly the same quote-valid, strictly earlier observations, atomic updates and unexpired rating history. Each has its own ordered-logit probability readout. These fixed constants derive from the [Elo-Odds paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0198668); this implementation's entry, timing and data choices are declared adaptations.

Primary market readout bytes and all 760 primary candidate vectors were reused exactly. One result-Elo readout was fitted for primary selection; two readouts were fitted for the longer-delay selection. **Three new optimizer fits total**, all using the same 3,419 admitted calibration identities from season-start years 2019–2021 after the 2017–2018 season-start-year warmup. Transfer used the saved readouts with no refitting. No strength model, blend, alternative parameter setting or new dataset was introduced.

For source/completion day D, the primary availability proxy is D+2 at local midnight; sensitivity is D+8. Availability must be strictly earlier than the original issuance clock. These proxies do not certify when a provider originally supplied or revised the quote. The 1,095-day window controls support only; Elo ratings retain older admitted history. Unsupported fixtures retain the exact shot-incumbent vector: 2/760 primary, 3/760 sensitivity and 8/2,280 transfer.

## Results

NLL uses natural logarithms and Brier sums the three squared class-probability errors. Positive relative gain means lower NLL than the shot baseline. The controlled production-default benchmark uses the inherited historical window and refit cadence; its score is not an observed deployed artifact's measurement.

| Population / availability | Elo-Odds NLL | Matched result-Elo NLL | Shot NLL | Gain vs shot | Elo Brier | Decision |
|---|---:|---:|---:|---:|---:|---|
| EPL 2022/23–2023/24, D+2 | 0.948330244 | 0.961641708 | 0.958158118 | 1.0257% | 0.560474927 | All 16 selection checks passed |
| Same 760 fixtures, D+8 | 0.946781101 | 0.961777409 | 0.958158118 | 1.1874% | 0.559198855 | All 16 selection checks passed |
| Three leagues 2024/25–2025/26, D+2 | 0.979743558 | 0.994460686 | 0.984127214 | 0.4454% | 0.583913510 | 19/30 transfer checks passed; failed |

Selection required at least 0.5% NLL improvement and a negative 28-day interval upper endpoint against all eight fixed references. Transfer required at least 2% against all five available references, a negative interval upper endpoint, improvement in every country, and nonworse NLL/Brier in every country-season. All required checks must pass. Four thousand paired draws use seed 20260908 and fixed calendar blocks within league-season strata; 1- and 7-day block intervals are diagnostics. These intervals do not adjust for the full research search.

The transfer comparisons show why the gain is insufficient:

| Transfer reference | Reference NLL | Elo relative NLL gain | Main limitation |
|---|---:|---:|---|
| Controlled production DC+auto | 1.028381677 | 4.7296% | Passes this weaker-reference comparison |
| DC365/Elo50 | 0.986744912 | 0.7095% | Below 2%; country/season regressions |
| DC180 | 0.992425735 | 1.2779% | Below 2% |
| Shot incumbent | 0.984127214 | 0.4454% | Below 2%; interval crosses zero; country/season regressions |
| Matched result-Elo | 0.994460686 | 1.4799% | Below 2% |

Against matched result-Elo, the primary selection difference is -0.013311464 with 28-day interval [-0.02326622, -0.00359108]. This supports the fixed market/K175 combination relative to result/K14. Because both the observation target and update speed differ, it does not isolate a pure quote-information effect.

Country transfer NLL is 0.999620651 in England, 0.967144916 in Italy and 0.972465109 in Spain. The corresponding shot losses are 1.002080439, 0.979767809 and 0.970533395. Italy supplies most of the pooled improvement. The [diagnostic notes](../../research/experiments/boundary_20260908/football_elo_followup_diagnostics.md) document the England/Spain 2025/26 regressions, class-probability biases, and cases where lower calibration-bin error coexists with worse proper scores. Excluding the resumed Fiorentina–Inter fixture still fails the transfer requirement. These findings do not authorize a post hoc country switch or retuning of this experiment.

## Verification and next operational assessment

All three historical stages and their independent verifications completed on their first attempts. The separate verifier replayed **7,600 candidate/control vectors across the three ledgers**, including the reused 760 primary candidate vectors. Maximum probability discrepancy was **1.05e-15**. It independently reconstructed raw membership, rating states, readout inputs and optima, original vector reuse, metrics, uncertainty intervals and decisions, with zero optimizer reruns. All four distinct readout payloads were checked; transfer reused previously checked coefficients.

The execution source lock binds 43 source files and 323 inherited inputs. The new lane's **176 synthetic tests passed**. The complete source-only research CI passed **1,772 tests with two optional historical skips**, with 583 source files and 91 dependency files unchanged. No historical assets or provider downloads were used by CI. Preparation took 39.20 seconds; the three forecast/scoring stages took 40.04, 51.29 and 71.55 seconds, with maximum process RSS 418.6 MB.

The next concrete opportunity is the [global football fitting policy](../../research/experiments/boundary_20260908/football_runtime_gap.md). Existing full-history equal-weight DC forecasts improve the controlled runtime's 1X2 loss by 3.432%, but that operational gap still requires faithful runtime implementation and assessment. The [scoreline-state audit](../../research/experiments/boundary_20260908/football_runtime_scoreline_diagnostic.md) identifies missing saved runtime parameters: a paired scoreline comparison cannot be recovered from old 1X2 vectors alone when the underlying goal model changes. It also states the exact conditions under which replacing only outcome masses on a common score kernel transfers a 1X2 log-loss difference to scoreline log loss.

The substantial new research-gain objective remains unmet. The validated global F1 model remains intact; this release changes football research and CI, not production behavior.

## Evidence

The [publication manifest](evidence/boundary_football_elo_followup_20260908/manifest.json) records **18 byte-identical receipts**, totaling **662,071 bytes**. Raw CSVs, readouts, rating/support state ledgers and row-level forecasts remain local, with their hashes bound by the published receipts.

| Receipt | SHA256 |
|---|---|
| Design lock | `dc0714b708b07b62649cf6a19c1148a8c438bb96a5f08842ac4ac57ebf6f8939` |
| Primary selection | `82aef83d574127c882a8404825aa81916aad7d44a0c79bb87a13f4f6a64046af` |
| Longer-delay selection | `2251009b5d8d772dc1b949a16d67c50f201a1851915bf8a8bf265a0c65ff2716` |
| Primary transfer | `5140df38f27bab140154748ce6a0ebbcde77479633149dd97c10d69584658856` |
| Copy manifest | `dae68c77826d08c61257f27876ea1ac272e807bff3f3aa3f1aa0be7a8fe278e5` |

Suggested commit: `research(football): publish verified Elo selection and failed transfer`.
