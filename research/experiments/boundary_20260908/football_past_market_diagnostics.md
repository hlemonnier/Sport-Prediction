# Why the fixed past-market strength model failed

Post hoc diagnostic, 2026-09-08. This note uses the already closed primary
selection and its admitted training audits. It authorizes no sensitivity,
transfer, retuning or promotion. No optimizer was called and no new candidate
forecasts were generated. Saved training logits were reconstructed only to
decompose the frozen objective and its local curvature.

**The leading explanation is substantial strength compression combined with
stale static pooling. Solver failure and an uninformative quote input are not
supported by the evidence.** Neither explanation has been isolated by an
ablation, so their causal contributions remain unknown.

## What the closed result actually says

All values below are mean natural-log 1X2 loss on the same 760 EPL fixtures.

| Frozen forecast | Selection NLL |
| --- | ---: |
| Past-market strength | 0.975508940 |
| Matched outcome strength | 0.975405928 |
| Past-market/shot 50:50 | 0.960861843 |
| Shot incumbent | 0.958158118 |
| Elo-odds reference | 0.948330244 |

The similar pooled losses of the two strength models do **not** mean identical
forecasts. Their mean total-variation distance is 0.06689; their home/away
log-odds correlation is 0.93225. The soft model loses 0.008607 nats to the
outcome control in season 2022 and gains 0.008401 in season 2023, almost
cancelling in the pooled result. Its mean top-label confidence is 0.50037
against observed accuracy 0.54605, with empirical underconfidence in every
occupied predeclared calibration bin. Sparse extreme bins remain noisy.

Elo-odds improves pooled NLL by 1.0257% against the shot incumbent. Its NLL is
also lower in each season: 0.982472 versus 0.989585 in 2022, and 0.914188 versus
0.926731 in 2023. This remains a discovered reference result on exposed
selection dates, not independently cleared transfer evidence.

## Penalty: correct implementation, materially restrictive scale

The frozen objective is

\[
J=\frac13\sum_{\ell}L_\ell(\theta_\ell)
 +\frac{0.002}{2}\sum_\ell\|\theta_\ell-\theta_{0\ell}\|^2.
\]

The leagues share no fitted coefficients. Multiplying each separable league
objective by three therefore gives the same optimum as
\(L_\ell+0.006\|\theta_\ell-\theta_{0\ell}\|^2/2\).
Thus **0.006 is the effective per-league ridge coefficient**, not an accidental
extra factor in the code. All 237 saved soft fits have maximum absolute
gradient at most 1.614e-7. Their median pooled data cross-entropy is 0.993666
and median ridge term 0.017009. Cross-entropy levels cannot fairly compare soft
and one-hot fits because their target entropies differ.

First, middle and last fit indices were selected deterministically, without
searching for favorable examples:

| EPL diagnostic | 0000: 2022-08-04 | 0118: 2023-08-11 | 0236: 2024-05-18 |
| --- | ---: | ---: | ---: |
| Quote-implied H/A log-odds weighted SD | 1.2840 | 1.2324 | 1.2670 |
| Saved fitted H/A log-odds weighted SD | 0.7966 | 0.7287 | 0.7634 |
| Weighted quote-to-fit KL, nats | 0.02160 | 0.02673 | 0.02471 |
| Effective league ridge term, nats | 0.02074 | 0.01867 | 0.02045 |
| Strength-direction data curvature | 0.01391 | 0.01346 | 0.01334 |
| Weighted mean history age, days | 399.6 | 407.8 | 335.3 |
| Weight older than 365 days | 43.47% | 44.09% | 41.85% |

The quote contrast is exactly \(\log(q_H/q_A)\), whereas the saved model
contrast is \(h+s_H-s_A\). Quote probabilities span 0.01673–0.92651 across
these snapshots; the input has substantial variation. Fitted contrast
dispersion is only 59.1–62.0% of quote dispersion. Along the fitted centered
strength direction, \(\lambda/(\kappa+\lambda)\) is 30.1–31.0%, where
\(\lambda=0.006\) and \(\kappa\) is the exact weighted likelihood Hessian
Rayleigh quotient at the saved optimum. This measures the local penalty's
importance; it is **not** an estimated improvement from removing the penalty
or a globally valid shrinkage factor. Time variation and model misspecification
also contribute to the quote-to-fit gap.

## Static pooling discards changes that Elo can carry

A daily refit is still a static model of the last 1,095 calendar days. Every
admitted match in a fit uses the same current pair of team coefficients. The
365-day half-life leaves 14.2–14.6% of weight more than two years old in the
three snapshots above.

At the final cutoff, 18 teams have at least eight admitted observations in the
last 90 days and 20 older than a year. The median absolute change in their
opponent-adjusted quote strength between those buckets is 0.2689 log-odds
units. This is a descriptive quote statistic, using the saved opponent effects,
not a new fitted rating. For example, Arsenal's recent-versus-old change is
+0.8274 and Manchester United's is -0.7177. Manchester City's recent and old
signals are both far above its saved coefficient, illustrating compression
even without a large recent change. These examples have not been selected by
match outcome or forecast error.

The frozen Elo reference instead updates states after each admitted quote and
has no repeated ridge pull toward neutral ratings. At a neutral expected
result and with the opponent held fixed, its one-update derivative with
respect to the team's previous rating is
\(1-175(\log 10)/400\times0.25=0.74815\). This demonstrates responsiveness;
it is not an exact memory half-life for the coupled nonlinear rating system.
Its ordered-logit scale, fitted only on 2019–2021, is 2.58733. Its saved
selection H/A log-odds SD is 1.25493 versus 0.76180 for soft strengths and
1.20053 for the incumbent. Elo's gains therefore confound temporal updates,
penalty structure, normalized-versus-power de-vig, and the different outcome
readout. They do not isolate any one of these mechanisms.

## Next action

Do not simply lower ridge using these selection errors and call that a clean
improvement. The clearest next test is a separately frozen evaluation of the
**exact existing Elo-odds reference**, without tuning its K, delay or readout,
against the incumbent and an otherwise matched outcome-driven Elo control.
That would test whether sequential market information survives beyond this
exposed selection. The current failed strength family must remain stopped;
any such test requires its own declared protocol. A state-space market model
is a later hypothesis, not evidence produced by this diagnosis.

## Evidence and reproducibility limits

All 237 fit files were checked against the closed fit index. The full admitted
row digests at indices 0000, 0118 and 0236 were reproduced exactly; saved
objectives and analytic gradients were replayed within 1e-12. No transfer
forecast or sensitivity model was read or generated.

Evidence under `artifacts/research/boundary_20260908/football_past_market/`:

| File | SHA256 |
| --- | --- |
| `design_lock.json` | `e9a05b1988bb619a59dc815e00fe46b8c6d1f6f9daf30dfb4fe563210f9f6d41` |
| `selection_delay1.json` | `0591f1b13c0e0e12497e18b45d2c56114b0b1cf5e02b2ee018fc3e6cde30555e` |
| `selection_delay1_fit_index.json` | `2121e833b1bd93ef8e1ed7145b23db86227a2c16ed7da7662574f51d3109ad12` |
| `selection_delay1_issued.json` | `5e5395bb779688089e6c687eab4fc51dd4c1dc61a61dd999480a01be6726530f` |
| `selection_delay1_verification.json` | `776eed85649217734bc243dc539cb4ccb955187c0c9eac1c9e6b5131ec172eaf` |
| `elo_readout_delay1.json` | `b6b32e919462cfbe436acdfe6d27022b1a10dc75c0984e14c53f137a0e4b0dc3` |

Local diagnostic script: `/private/tmp/diagnose-football-past-market-20260908.py`,
SHA256 `58c80304ea67ac6a3adb1e72e384083244b578b729a2518b980fb9ef245e20e3`.
Its strict JSON output is `/private/tmp/football-past-market-diagnostics-20260908.json`,
SHA256 `fd86051051585ea390bd35841cf721420453eb1c0f4c46c211906a9a5c3c5848`.
These temporary diagnostic files are local audit material, not published
experiment assets. The formulas and snapshot selection above define the
reported calculations; the frozen experiment itself is unchanged.

Suggested commit: `research(football): diagnose shrinkage and stale market strengths`.
