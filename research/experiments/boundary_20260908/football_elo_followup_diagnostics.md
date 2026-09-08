# Fixed Elo follow-up: interpretation of the closed selection and transfer results

These diagnostic notes cover the fixed, already exposed EPL selection and the subsequently closed D+2 three-country transfer population. Both selection stages pass, but the transfer stage fails the advancement gates. The D+8 transfer stage is therefore not permitted. These notes do not change the protocol, choose another model, or establish production eligibility. No new fits or forecasts were produced for this diagnosis.

The primary D+2-local-midnight experiment passes the fixed 0.5% NLL improvement and negative 28-day interval gates against all eight references. Its 760 candidate vectors and market readout bytes are reused exactly from the preceding experiment. The new comparison with matched result-Elo is informative: natural-log loss falls from **0.961641708 to 0.948330244**, a **1.38424%** relative improvement. The paired candidate-minus-reference loss difference is **−0.013311464 nats/match**, with a 95% 28-day block interval **[−0.023266222, −0.003591079]**. The reported 1-day and 7-day intervals also exclude zero for this particular reference.

| EPL season | Market-Elo NLL | Result-Elo NLL | Relative NLL improvement | Market / result class-summed Brier |
| --- | ---: | ---: | ---: | ---: |
| 2022/23 | 0.982472126 | 0.994498813 | 1.20932% | 0.583856 / 0.592214 |
| 2023/24 | 0.914188362 | 0.928784602 | 1.57154% | 0.537093 / 0.547322 |
| Combined | 0.948330244 | 0.961641708 | 1.38424% | 0.560475 / 0.569768 |

The pooled Brier improvement is 1.63102%, but top-choice accuracy is **420/760 versus 421/760**. This is evidence of better probability assignments on these fixtures, not more correct winner/draw classifications. The primary 28-day intervals also exclude zero against shot and xG safeguards, but their 1-day and 7-day intervals cross zero. The frozen 28-day rule passed; robustness to every reasonable block choice has not been established. There are only 10 and 11 nonempty 28-day blocks in the two seasons, and these are conditional historical bootstrap intervals rather than a guarantee of future-season transport.

All four readouts (two mechanisms, two delays) use the **same 3,419 match IDs and observed outcomes** from season-start years 2019–2021: 1,140 English, 1,140 Spanish and 1,139 Italian fixtures. Torino–Fiorentina from season-start 2021 is the single excluded missing-quote fixture. Their outcomes are 1,436 home wins, 868 draws and 1,115 away wins. The common fitting cutoff is `2022-08-04T23:00:00+00:00`. Rating inputs for each calibration fixture were captured at that fixture's original midnight; later resolved outcomes train the readout only before this fixed cutoff.

For `x = (home_rating − away_rating) / 400`, both readouts use `P(A) = sigmoid(t1 − beta*x)`, `P(H) = 1 − sigmoid(t2 − beta*x)`, with the intervening probability assigned to a draw:

| Readout | beta | t1 | t2 | Training NLL | KKT residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| D+2 market | 2.587332148 | −0.806826766 | 0.430102579 | 0.977345330 | 5.33e−8 |
| D+2 result | 2.741018510 | −0.834524511 | 0.378880642 | 0.988879570 | 5.05e−8 |
| D+8 market | 2.565350979 | −0.859486799 | 0.374441080 | 0.978776224 | 2.65e−8 |
| D+8 result | 2.738348564 | −0.838537899 | 0.373987894 | 0.989356135 | 5.26e−8 |

The lower market beta does not imply a weaker strength signal: primary calibration x has standard deviation 0.356788 for market Elo and 0.312843 for result Elo. Consequently `beta * SD(x)` is 0.923128 versus 0.857510. Their x correlation is 0.932078. These are descriptive properties of the stored training arrays; the training NLL difference is not an additional out-of-sample result.

Calibration remains imperfect. Primary market-Elo mean home probability is **5.104 percentage points below** the observed home-win frequency, while draw probability is **2.557 points above** its frequency. Result-Elo has similar signed errors (−5.018 and +2.345 points). The primary market-Elo 2023/24 top-label ECE is **5.747%**, versus **4.489%** for result-Elo and **2.845%** for the retained shot model. Proper scores can improve while a particular binned calibration diagnostic worsens; these quantities measure different aspects of the forecasts. The static readout does not remove the observed seasonal calibration shift.

The prespecified D+8 sensitivity also passes all eight selection references. Market-Elo NLL is **0.946781101**, versus **0.961777409** for result-Elo: **1.55923%** improvement, with 28-day difference interval **[−0.026110753, −0.003692889]**. The gain against the unchanged shot reference is **1.18738%**. All 760 fixtures remain; support declines from 758 to 757 and the unsupported rows retain the exact shot fallback. Its 2023/24 top-label ECE is **7.521%** versus result-Elo **2.643%**, despite better NLL and Brier. A slightly better observed score under the longer delay is not evidence that stale data is intrinsically preferable, and no delay was chosen from these results.

The D+8 result and independent verification are closed and passed. This note checked their result hash binding; the separate verifier receipt records numerical replay.

The substantive comparison is **market-based K=175 ratings with their readout versus result-based K=14 ratings with their readout**, using identical eligible historical update fixtures and timing. It jointly changes the update target, responsiveness and fitted readout. It therefore supports this fixed information-and-update mechanism on the recorded population; it does **not** isolate a causal benefit of quotes at a common K, certify bookmaker receipt times, or imply that the failed convex strength family advanced. The market readout itself uses historical results as training labels, so “quote-only” describes the rating updates, not a completely outcome-free system. Both selection and later outcome dates were previously exposed, and this follow-up was motivated by a post hoc reference lead. The subsequent transfer failure described below prevents a promotion claim.

The primary independent verifier replayed 1,520 candidate/control vectors with maximum absolute error `7.771561172376096e-16`, reconstructed both prequential readout optima, and confirmed the reused bits. This diagnostic checked the closed primary result/decision/issuance/readout bindings and matched training IDs. It does not replace that verifier.

The later D+2 transfer is a **failed advancement screen**, even though the candidate improves the weaker production-default benchmark by 4.72958%. Across all 2,280 original fixtures, the NLL improvement versus the retained shot reference is only **0.445436%**: 0.979743558 versus 0.984127214. Its 28-day paired difference interval **[−0.010071995, +0.001254636]** crosses zero. The gain versus DC365/Elo50 is 0.709540%, and the gain versus matched result-Elo is 1.47991%; all three are below the fixed 2% threshold. The strong-reference failure must remain the conclusion.

| Country, 760 fixtures each | Market-Elo NLL | Shot-reference NLL | Relative improvement | 2024/25 improvement | 2025/26 improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
| England | 0.999620651 | 1.002080439 | +0.24547% | +0.94489% | −0.43204% |
| Italy | 0.967144916 | 0.979767809 | +1.28836% | +0.63344% | +1.92214% |
| Spain | 0.972465109 | 0.970533395 | −0.19904% | +0.48887% | −0.88141% |

Italy supplies most of the net pooled gain. England and Spain both reverse against the shot reference in 2025/26; those two cells also have worse Brier scores. Spain is worse than the shot and DC365/Elo50 references over both years combined. The market mechanism does beat matched result-Elo in every country-season on NLL and Brier, so the historical market-information-plus-update-rate benefit survives that weaker comparison; it is insufficient to beat the strongest existing research references reliably.

The closed probability vectors reveal heterogeneous calibration, rather than uniform underconfidence:

| Country | Market mean home probability minus observed home-win frequency | Market mean top confidence | Market top-choice accuracy | Top-choice disagreements with shot |
| --- | ---: | ---: | ---: | ---: |
| England | +0.156 percentage points | 52.671% | 51.711% | 63/760 |
| Italy | +2.782 percentage points | 52.828% | 53.289% | 73/760 |
| Spain | −5.299 percentage points | 50.776% | 53.684% | 75/760 |

Spain has the clearest directional mismatch: the candidate assigns 41.411% mean home probability versus a 46.711% observed home-win frequency. Its away probabilities are 4.108 points too high. The shot model's corresponding home bias is only −0.767 points. Nevertheless, the market model's Spanish top-label ECE is **3.227%**, below shot's **6.985%**, while its NLL and Brier are worse. Lower binned ECE or lower mean confidence alone cannot diagnose a superior full three-class distribution. England is slightly overconfident in aggregate top choices; Italy is close to aggregate top-choice calibration but overestimates the home frequency. A blanket “increase confidence” explanation is not supported.

The candidate and shot model differ by a mean absolute 3.61–3.82 percentage points per class across the three countries. This is a meaningful forecast disagreement, but the advantage is concentrated in country and year. Different league outcome frequencies and drift under a frozen pooled readout are plausible contributors; these diagnostics do not identify the causal source or establish that a recalibration would improve an unseen period. No corrective parameters have been fitted to these later outcomes.

All 2,280 fixtures remain, with 2,272 supported and eight exact shot fallbacks (two England, three Italy, three Spain). Excluding the resumed Fiorentina–Inter fixture leaves 2,279 rows and a 0.452471% shot-reference gain; its 28-day interval **[−0.010141247, +0.001188390]** still crosses zero. The failure is not explained by that fixture. The independent transfer verifier passed execution checks, replayed 4,560 vectors with maximum error `1.0547118733938987e-15`, and confirmed the scientific gates are false.

The concrete next action is to retain this as a verified failed fixed candidate, keep the strongest reference unchanged, and prohibit the D+8 transfer stage. Country-specific directional calibration and year drift are evidence for a future separately declared research question, not permission to repair the already scored candidate or reinterpret the result as a successful production upgrade. The dates remain retrospectively exposed, and the K=175/K=14 comparison still does not isolate a quote-only causal effect.

Source evidence:

- [Primary result](../../../artifacts/research/boundary_20260908/football_elo_followup/selection_delay1.json), SHA256 `82aef83d574127c882a8404825aa81916aad7d44a0c79bb87a13f4f6a64046af`.
- [Primary verification](../../../artifacts/research/boundary_20260908/football_elo_followup/selection_delay1_verification.json), SHA256 `542cb68ebf4eec367ce833ed789ae39bcddb7b732d6442d3c79f567b18c0afe0`.
- [D+8 result](../../../artifacts/research/boundary_20260908/football_elo_followup/selection_delay7.json), SHA256 `2251009b5d8d772dc1b949a16d67c50f201a1851915bf8a8bf265a0c65ff2716`.
- [D+2 transfer result](../../../artifacts/research/boundary_20260908/football_elo_followup/transfer_delay1.json), SHA256 `5140df38f27bab140154748ce6a0ebbcde77479633149dd97c10d69584658856`.
- [D+2 transfer verification](../../../artifacts/research/boundary_20260908/football_elo_followup/transfer_delay1_verification.json), SHA256 `eb7421f4aa0234d5ab55614e1ac2bbe8d63a23d72f37dbd68103efc153847e96`.
- [Frozen protocol](football_elo_followup/specification.json), SHA256 `bffe2c481257cb99e35564f3f65a02ab2f85de15fcd137bf0f7ba9282c61020a`.

Suggested commit: `docs(football): diagnose fixed Elo transfer failure`.
