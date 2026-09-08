# Same-checkpoint sector reference review

This design review uses only the closed two-race raw-stream pilot, its source and the canonical next-lap target. No sector forecast was fitted or scored. The four fixed references and Gaussian prototype below are implemented in `baselines.py`; the baseline module accepts only already-filtered numerical history and an observed S1 or S1+S2 prefix. It does not consume target labels or final `IsAccurate`.

The task is to predict the **same driver's next recorded eligible clean completed lap strictly after the sector checkpoint**, potentially skipping the current numbered lap. It is a later issuance with a shorter horizon than the canonical HGB's preceding completed-lap issuance. A gain here cannot be called a gain on the original HGB issuance population.

## What the closed pilot establishes

The two Bahrain races contain 4,312 positive S1/S2 ledger rows, including 4,245 recorded-input candidate checkpoints. Of those candidates, 4,231 have a later recorded eligible lap and 14 are unmatched in the terminal recorded archive. Every candidate lacks a receipt-certified `IsAccurate` and current S3 display value; 80 lack `LastLapTime`. These are support and outcome-accounting counts, not predictive scores. The pilot's results SHA256 is `1272a6f004dea9583b31996b0fce8d350a15f39b24a4c2e5e77a6bebfbd6bf7e`.

Normal current S1 packets clear the old S2/S3 display fields. Therefore `received_driver_fields.Sector3` at a checkpoint cannot supply a historical remaining sector. A separate immutable completed-lap history is necessary.

An independent raw-packet count found that all 1,103 positive `LastLapTime` updates in 2022 share a packet with positive S3 and a counter field. In 2023, 1,029 of 1,031 do; the two remaining `LastLapTime` packets lack same-packet S3. These counts concern packet structure, not certified lap identity or clean training support.

The completed-history parser's frozen minimal association is appropriate: snapshot the old epoch before a unit counter advance; require its ordered, unambiguous, earlier-packet S1/S2; combine these with same-packet S3 and `LastLapTime`; check positive finite durations and the fixed 0.003-second sum tolerance. A later S3 may produce another diagnostic record but must never retroactively upgrade the original completion. Preserve raw clock, effective cumulative clock and physical packet sequence for every component. The recorded counter is not certified canonical `LapNumber`.

## Core issuance and history contract

1. Preserve every pilot candidate checkpoint before outcome resolution. All four references and all challengers share an as-of warm-up gate of **at least three valid own-driver completions**. A checkpoint lacking support remains in the ledger as `warmup_unavailable`; it is not removed after examining its target. Report this support coverage separately from target-match coverage.
2. The orchestrator admits only completed records with `observed_valid_completed is True` and `available_ms < checkpoint_ms`. Issue all forecasts at a shared timestamp before admitting any completion at that timestamp. A later packet may not rewrite an earlier feature or issuance. Auxiliary control/tyre facts must obey their strictly-prior stream rule.
3. A historical clean proxy requires observed, unambiguous sector association, known `Started` and track status 1 or 2 throughout the epoch, known outside-pit state throughout, and no observed pit-out, stoppage, retirement or neutralization. This uses observed states, not the final CSV's `IsAccurate`, eventual target eligibility, final stint or future tyre life. It is a raw clean proxy, not proof of physical cleanliness.
4. Keep valid own-driver templates across observed tyre changes; these four minimal references do not invent a compound adjustment or certify the highest received stint index as active. Expose observed compound/stint-candidate changes and template staleness to the separately specified challenger. History itself is never relabeled using final tyre rows.
5. A current S1 forecast receives exactly one observed prefix value; S2 receives its own epoch's prior S1 and current S2. The numerical API rejects a third sector. Do not supply future or stale later-sector display slots. Simultaneous/ambiguous sector packets need the ledger's declared attribution treatment, not target-based repair.
6. Resolve the first later recorded eligible target only after the issuance ledger closes. Keep every issued but unmatched forecast with its terminal-archive or incomplete-archive status. Point MAE is explicitly conditional on an eligible target being observed; no finite lap-duration error exists for an undefined no-completion target. This point study does not silently claim an all-issued joint completion/duration score.

The parser API is `iter_completed(paths, event_key, stats=None)`. Each completion carries `completion_id`, `available_ms`, `packet_sequence`, observed old/new counters, `sectors_seconds`, `full_lap_seconds`, per-field sources, sum residual, quality/ambiguity reasons and tyre support. `predict_baselines(history, prefix_seconds, known_asof_contamination=...)` consumes only the numerical fields after the orchestrator applies that contract. Its output is `{"points": {...}, "diagnostics": {...}}`.

## Four frozen references

For current observed sector count \(k\in\{1,2\}\), let \(x=(x_1,\ldots,x_k)\) and \(P=\sum_{j\le k}x_j\). For historical coherent template \(h\), retain its joint sector vector \(s_h\) and full lap \(L_h\), and define

\[
P_{h,k}=\sum_{j\le k}s_{h,j},\qquad R_{h,k}=L_h-P_{h,k}.
\]

Defining the remainder from full lap minus prefix preserves the small received rounding residual rather than silently substituting a sector-sum target. Every remainder must be positive.

Let \(H_m\) denote the last at most \(m\) valid own-driver templates, newest last. On the as-of uncontaminated/unknown-future branch:

| Reference ID | Fixed point |
|---|---|
| `sector_last_template` | \(P+R_{\mathrm{latest},k}\) |
| `sector_joint_median5` | \(P+\operatorname{median}_{h\in H_5}R_{h,k}\) |
| `sector_weighted_median10` | \(P+\operatorname{wmedian}_{h\in H_{10}}R_{h,k}\), weights \(0.2(0.8)^{\mathrm{age}(h)}\) |
| `sector_pace_scaled_median5` | \(P+\operatorname{median}_{h\in H_5}\{a_hR_{h,k}\}\), \(a_h=\operatorname{clip}(P/P_{h,k},0.97,1.03)\) |

The ordinary median uses the midpoint convention for an even number of samples. The weighted median uses the lower inverse-CDF convention. Both are absolute-loss minimizers for their respective empirical distributions; neither guarantees the conditional median of the future target. The pace-scaled reference assumes modest approximately proportional pace change across sectors and limits that adjustment with a fixed, pre-score ratio range.

These are medians of **joint remainders**, not sums of marginal sector medians. For remaining-sector pairs `(1,101)`, `(101,1)`, `(101,101)`, the median remainder is 102, whereas the sum of marginal medians is 202.

The proposed recency alpha of 0.5 was corrected **before scoring**. With normalized finite-window exponential weights, its newest weight is \(0.5/(1-0.5^n)>0.5\), so the weighted median always equals the latest template. Alpha 0.2 avoids that duplicated reference once there are at least three observations; the actual weighted output can still coincide with another reference on some inputs.

If the current epoch has **already observed** pit/neutralization contamination, the current prefix no longer reliably describes the eventual eligible target. Replace each remainder estimate with its corresponding whole-lap estimate: last \(L\), median last-five \(L\), weighted median last-ten \(L\), and median last-five \(L\), respectively. This branch never consults eventual CSV eligibility. Unknown future contamination remains unknown; the reference can be wrong when a later incident causes the target to skip the current lap. Do not retrospectively select the correct branch.

## Gaussian conditional-remainder prototype

For the last at most ten valid own-driver completions, build joint samples

\[
z_h=(s_{h,1},\ldots,s_{h,k},R_{h,k}),\qquad
\widehat\Sigma=0.5S+0.5\operatorname{diag}(S)+0.01 I,
\]

where \(S\) is sample covariance using denominator \(n-1\). All entries are in seconds squared; the diagonal addition is 0.01 seconds squared, not 0.01 seconds. It guarantees positive definiteness for finite valid moments, even with three templates, rank deficiency or zero sample variance. Let \(X\) denote the prefix coordinates and \(R\) the last coordinate. The point is

\[
\widehat L=P+\bar R+
\widehat\Sigma_{RX}\operatorname{solve}
  (\widehat\Sigma_{XX},\ x-\bar X).
\]

The implementation solves the system without explicitly inverting it. This is the conditional mean, and also median, **under the assumed joint Gaussian model**. The samples are neither guaranteed Gaussian nor stationary or independent; successive laps have tyre/fuel/track trends, and control/compound changes can invalidate that approximation. Covariance shrinkage improves numerical stability but does not remove those model limitations or establish predictive uncertainty.

Condition on `(prefix, remainder)`, not `(prefix, full lap)`: shrinking cross-covariance of the latter incorrectly shrinks the coefficient of the already known prefix. If all historical remaining times equal 65 seconds, the forecast must be current prefix plus 65 regardless of historical prefix variance; the implemented formulation satisfies this invariant.

There is no additional performance clipping. A nonfinite moment/point, linear-solve failure or nonpositive conditional remainder returns **the same checkpoint's pace-scaled reference** with an explicit fallback reason; it never drops an issued row. Known as-of contamination instead uses the last-five whole-lap median, exactly like reference four. This prototype is one declared challenger, not an additional family of tuned covariance settings.

## Comparison and gain interpretation

The four same-checkpoint references are the primary comparators. A carried canonical HGB point is secondary and must retain its original issue time, source contract and missing receipt-certification caveat. Finalized canonical features cannot be inserted into an earlier raw checkpoint. Do not use HGB availability or future target eligibility to make one candidate's scored population easier than another's.

Before any new scores, the execution protocol should freeze a bounded challenger grid, select only on the declared discovery split, and require improvement against **every** fixed reference, which avoids selecting a weak comparator after evaluation. A defensible substantial-transfer criterion is at least 10% lower event-balanced matched-target MAE against the strongest same-checkpoint reference, at least 5% target-balanced improvement, paired whole-event confidence intervals below zero, no reversal when an individual event is left out, and positive separate-year and S1/S2 comparisons. Report known-contamination, current-lap-target and later-lap-target subsets as diagnostics; the latter two are outcome classifications, never input gates. Exact thresholds belong in the new execution protocol before fitting.

For target-balanced scoring, first average errors among all issued checkpoints sharing an `(event, driver, target lap)` and then average targets within each event. For ordinary event-balanced scoring, average all matched issued errors within each event before averaging events. Preserve both because pit/neutralized stretches can cause many checkpoints to share one future target. Whole-event uncertainty must not treat sector checkpoints or drivers as independent races. Two Bahrain pilot events cannot establish a broad transfer claim, irrespective of their thousands of rows.

Keep the original eligible-completed-lap HGB benchmark unchanged and separately reported. A same-checkpoint gain demonstrates the usefulness of newly observed sector information at the later horizon; it is neither an original point-MAE gain nor a production promotion.

Suggested commit: `research(f1): add coherent sector remainder references and Gaussian prototype`
